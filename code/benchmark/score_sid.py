# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Score OneRec models on the benchmark with the two-stage listwise protocol (paper Sec. 3 and 4).

A thin CLI around utils/sid_engine.py, which documents the protocol: greedy stage-1 reasoning with a
structured answer after </think>, and a stage-2 forced extraction only for rows stage 1 left without
a valid list. --render picks the item rendering for history and candidates (sid, text or sid_text,
the paper's SID, Text and Both views); caption renderings need --max-model-len 24576 at k=64.
--lora-dirs evaluates trained adapters by hot-swapping them on one loaded model (Sec. 5).

--model is onerec-1.7b or onerec-8b (loaded at the revision pinned in utils/config.py) or a local
model directory such as saves/onerec-8b/sft-merged. One JSON is written per checkpoint and k:

    results/diagnosis/<model>/<view>/k<k>_n<N>.json         --method base (default), view sid|text|both
    results/posttraining/<method>/<model>/test/k<k>_n<N>.json      --method sft|rft|opd|rlvr-*
    results/posttraining/<method>/<model>/dev/<checkpoint>.json    --slice dev, one per adapter

A non-zero --shuffle adds _s<i> to the cell name, a run over several checkpoints prefixes test
files with the checkpoint name, and --outdir overrides the directory. Unless --skip-base is given,
the loaded model itself is also scored, as checkpoint "base" under --method base and as "init"
(the method's initializer) otherwise.

    python code/benchmark/score_sid.py --model onerec-8b --k-grid 0,4,8,16,32,64,128,256
    python code/benchmark/score_sid.py --model onerec-8b --render text --max-model-len 24576
    python code/benchmark/score_sid.py --model onerec-1.7b --method sft --slice dev --skip-base \\
        --lora-dirs step1035=saves/onerec-1.7b/sft/checkpoint-1035
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ReRankID/code
from utils import config
config.setup_env()
import argparse
import json

from vllm.lora.request import LoRARequest

from utils.data_io import dump_json, load_bench
from utils.metrics import aggregate
from utils.prompts import TOP_K
from utils.sid_engine import (load_llm, parse_lora_dirs, run_listwise,
                              two_stage_stats, vllm_version)

METHODS = ("base", "sft", "rft", "opd", "rlvr-ndcg5", "rlvr-hit1")
ANCHOR = "k64_n30"
VIEWS = {"sid": "sid", "text": "text", "sid_text": "both"}   # --render -> the paper's view name


def resolve_model(model):
    """(name, repo, revision, local_dir) for --model. A MODELS key or its HF id loads the pinned
    revision; a local directory keeps its path and takes its name from a saves/<name>/ parent."""
    for name, (repo, revision) in config.MODELS.items():
        if model.lower() in (name, repo.lower()):
            return name, repo, revision, None
    parts = os.path.abspath(model).split(os.sep)
    name = next((p for p in reversed(parts) if p in config.MODELS), os.path.basename(model.rstrip("/")))
    repo, revision = config.MODELS.get(name, (None, None))
    return name, repo, revision, model


def output_path(args, name, ckpt, k, n_checkpoints):
    """Result file of one (checkpoint, k) under the release layout described in the module doc."""
    cell = f"k{k}_n{args.n_cand}" + (f"_s{args.shuffle}" if args.shuffle else "")
    if args.method == "base":
        root = f"{config.RESULTS}/diagnosis/{name}/{VIEWS[args.render]}"
        if args.slice != "test":
            cell = f"{cell}_{args.slice}"
    else:
        root = f"{config.RESULTS}/posttraining/{args.method}/{name}/{args.slice}"
        if args.render != "sid":
            cell = f"{cell}_{VIEWS[args.render]}"
    if args.slice == "dev" and args.method != "base":
        stem = ckpt if cell == ANCHOR else f"{ckpt}_{cell}"
    else:
        stem = f"{ckpt}_{cell}" if n_checkpoints > 1 else cell
    if args.n:
        stem += f"_first{args.n}"
    return os.path.join(args.outdir or root, f"{stem}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="onerec-1.7b",
                    help="onerec-1.7b, onerec-8b (pinned revisions) or a local model directory")
    ap.add_argument("--method", default="base", choices=METHODS,
                    help="selects the results subtree; base writes under results/diagnosis")
    ap.add_argument("--bench", default=config.BENCH)
    ap.add_argument("--slice", default="test", choices=["test", "dev", "all"],
                    help="test (400 users) for reported numbers, dev (400 users) for checkpoint selection")
    ap.add_argument("--n", type=int, default=None, help="score only the first n users of the slice")
    ap.add_argument("--n-cand", type=int, default=30, help="slate size N (10, 20, 30, 50 or 100)")
    ap.add_argument("--shuffle", type=int, default=0,
                    help="display-order seed; 0 is the reported protocol")
    ap.add_argument("--render", default="sid", choices=["sid", "text", "sid_text"],
                    help="item rendering for history and candidates")
    ap.add_argument("--k", type=int, default=64, help="history length")
    ap.add_argument("--k-grid", default=None,
                    help="comma-separated history lengths scored with one model load")
    ap.add_argument("--top-k", type=int, default=TOP_K, help="length of the requested ranking")
    ap.add_argument("--max-tokens", type=int, default=2048, help="stage-1 reasoning and answer budget")
    ap.add_argument("--extract-tokens", type=int, default=256, help="stage-2 answer budget")
    ap.add_argument("--k-ndcg", type=int, default=5)
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="context length (default 8192); text and sid_text need about 24576 at k=64")
    ap.add_argument("--tp", type=int, default=None,
                    help="tensor-parallel size (default: all visible GPUs)")
    ap.add_argument("--no-eager", action="store_true",
                    help="allow CUDA graphs; every reported result used eager mode")
    ap.add_argument("--lora-dirs", default=None,
                    help="comma-separated LoRA adapters as name=path or path (name = basename)")
    ap.add_argument("--max-lora-rank", type=int, default=32)
    ap.add_argument("--skip-base", action="store_true",
                    help="score only the adapters, not the loaded model itself")
    ap.add_argument("--outdir", default=None, help="override the output directory")
    args = ap.parse_args()

    name, repo, revision, local_dir = resolve_model(args.model)
    ks = [int(x) for x in args.k_grid.split(",")] if args.k_grid else [args.k]
    lora_dirs = parse_lora_dirs(args.lora_dirs)
    loaded = "base" if args.method == "base" else "init"
    checkpoints = ([] if args.skip_base else [(loaded, None, None)]) + [
        (ckpt, path, LoRARequest(lora_name=ckpt, lora_int_id=i + 1, lora_path=path))
        for i, (ckpt, path) in enumerate(lora_dirs)]
    if not checkpoints:
        raise SystemExit("nothing to score: --skip-base without --lora-dirs")
    print(f"model={name} ({local_dir or f'{repo}@{revision}'}) | k-grid={ks} | "
          f"checkpoints={[c[0] for c in checkpoints]}")

    llm, tok = load_llm(local_dir or repo, revision=None if local_dir else revision, tp=args.tp,
                        eager=not args.no_eager, enable_lora=bool(lora_dirs),
                        max_lora_rank=args.max_lora_rank, max_model_len=args.max_model_len)
    rows = load_bench(args.bench, n=args.n, n_cand=args.n_cand, shuffle=args.shuffle,
                      bench_slice=args.slice)
    for ckpt, adapter, lora_request in checkpoints:
        for k in ks:
            res = run_listwise(llm, tok, rows, k, args.top_k, args.max_tokens, args.extract_tokens,
                               lora_request=lora_request, render=args.render)
            m = aggregate(res, k_ndcg=args.k_ndcg, n_out=args.top_k)
            ts = two_stage_stats(res, args.top_k)
            m["two_stage"] = ts
            print(f"\n=== {name} | {ckpt} | k={k} N={args.n_cand} ===")
            print(json.dumps(m, indent=2, ensure_ascii=False))
            print(f"  stage 1 valid {ts['n_stage1_selfserved']}/{len(res)} | stage 2 needed "
                  f"{ts['n_two_stage']}, rescued {ts['n_stage2_rescued']}")
            meta = {"model": name, "base_model": repo, "revision": revision,
                    "model_dir": config.relpath(local_dir) if local_dir else None,
                    "method": args.method, "checkpoint": ckpt,
                    "adapter": config.relpath(adapter) if adapter else None,
                    "bench": config.relpath(args.bench), "slice": args.slice, "k": k,
                    "n_cand": args.n_cand, "shuffle": args.shuffle, "render": args.render,
                    "top_k": args.top_k, "k_ndcg": args.k_ndcg,
                    "max_tokens": args.max_tokens, "extract_tokens": args.extract_tokens,
                    "max_model_len": args.max_model_len, "enforce_eager": not args.no_eager,
                    "tp": args.tp or config.gpu_count(),
                    "arm": "sid", "protocol": "listwise-two-stage",
                    "decoding": "greedy+structured", "vllm": vllm_version()}
            if args.n:
                meta["n"] = args.n
            path = output_path(args, name, ckpt, k, len(checkpoints))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            dump_json(path, {"meta": meta, "metrics": m, "results": res})
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
