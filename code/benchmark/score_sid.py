"""SID arm of the rerank benchmark: OneRec reranks the candidate slate rendered as SIDs. TWO-STAGE,
greedy, structured output — on the selected cu128/cu130 vLLM profile.

The generation engine lives in utils/sid_engine.py — see its docstring for the full protocol: stage 1
(free CoT + reasoning-aware structured output), stage 2 (forced extraction, only for rows stage 1 did
not resolve), the recorded self-serve/fallback split, and the fairness rules (greedy, NO
repetition_penalty; NO repair of repeated/short lists; HONEST parse only after a closed </think>).
This file keeps only the CLI around it.

Scoring: utils.metrics.aggregate (strict: success requires a valid top-K list of k distinct indices).
Rendering / prompt: utils.prompts; bench I/O: utils.data_io. Chat template:
utils/templates/qwen3_soft_switch.jinja2.
Also supports evaluating LoRA checkpoints via vLLM's multi-LoRA hot-swap (--lora-dirs): the base model
is loaded once, adapters are swapped per request, no merge-and-save per checkpoint.

Usage:
  python code/benchmark/score_sid.py --n 4 --k 0                       # smoke
  python code/benchmark/score_sid.py --k-grid 0,4,8,16,32,64,128,256   # full history axis
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenOneRec/OneRec-1.7B-pro")
    ap.add_argument("--bench", default=config.BENCH)
    ap.add_argument("--benches", default=None,
                    help="comma-separated bench paths; overrides --bench, for the rare case a single "
                         "invocation needs to cover more than one bench file")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--slice", default="test", choices=["test", "dev", "all"],
                    help="bench.parquet holds test 400 + dev 400 interleaved; headline runs score "
                         "'test', development runs 'dev'")
    ap.add_argument("--n-cand", type=int, default=30,
                    help="slate size N: render-time prefix of the stored 100-candidate priority "
                         "pool (see utils.data_io.select_slate); 30 is the anchor")
    ap.add_argument("--shuffle", type=int, default=0,
                    help="display-shuffle id; 0 = headline protocol, 1/2 = stability audit only")
    ap.add_argument("--render", default="sid", choices=["sid", "text", "sid_text"],
                    help="item rendering for history AND candidates (matched-model decomposition: "
                         "the same OneRec engine under sid / text / sid_text). Caption renderings "
                         "exceed the 8192 default context at ANY k — pass --max-model-len "
                         "(~24576 at the anchor k=64)")
    ap.add_argument("--corrupt-sid", action="store_true",
                    help="C4 attribution control: deterministically derange the slate's candidate "
                         "SIDs (utils.data_io.corrupt_sids) — semantics destroyed, format intact")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="override the engine's 8192-token context (required for --render text/"
                         "sid_text at any k; ~24576 covers the anchor)")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--k-grid", default=None)                 # e.g. "0,16,32,64,128,256"; one model load
    ap.add_argument("--top-k", type=int, default=TOP_K)       # how many top candidates to output
    ap.add_argument("--max-tokens", type=int, default=2048)   # stage-1 CoT + answer budget
    ap.add_argument("--extract-tokens", type=int, default=256)  # stage-2 forced answer
    ap.add_argument("--k-ndcg", type=int, default=5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tp", type=int, default=None,
                    help="vLLM tensor-parallel size; defaults to the visible GPU count")
    ap.add_argument("--no-profile", action="store_true",
                    help="drop the profile header; rank from history (+candidates) only")
    ap.add_argument("--no-eager", action="store_true",
                    help="enable torch.compile + cudagraphs (faster inference, longer warmup)")
    ap.add_argument("--outdir", default=config.RESULTS)
    ap.add_argument("--lora-dirs", default=None,
                    help="comma-separated LoRA adapter dirs ('name=path' or bare 'path'); base model is "
                         "always included as one of the sweep entries alongside these, tagged 'base'")
    ap.add_argument("--max-lora-rank", type=int, default=32,
                    help="must be >= the actual trained LoRA rank (checked lazily by vLLM on first use)")
    ap.add_argument("--skip-base", action="store_true",
                    help="omit the base-model ('base', None) sweep entry, e.g. when base results for "
                         "this bench/k-grid already exist from a prior run")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    bench_paths = [b.strip() for b in args.benches.split(",")] if args.benches else [args.bench]
    ks = [int(x) for x in args.k_grid.split(",")] if args.k_grid else [args.k]
    short = args.model.split("/")[-1]

    lora_dirs = parse_lora_dirs(args.lora_dirs)
    base_entry = [] if args.skip_base else [("base", None)]
    checkpoints = base_entry + [(name, LoRARequest(lora_name=name, lora_int_id=i + 1, lora_path=path))
                                for i, (name, path) in enumerate(lora_dirs)]
    print(f"benches={bench_paths} | k-grid={ks} | model={args.model} | "
          f"checkpoints={[c[0] for c in checkpoints]}")

    llm, tok = load_llm(args.model, tp=args.tp, eager=not args.no_eager, enable_lora=bool(lora_dirs),
                        max_lora_rank=args.max_lora_rank, max_model_len=args.max_model_len)
    for bench_path in bench_paths:
        rows = load_bench(bench_path, n=args.n, no_profile=args.no_profile,
                          n_cand=args.n_cand, shuffle=args.shuffle, corrupt_sid=args.corrupt_sid,
                          bench_slice=args.slice)
        bench_stem = os.path.splitext(os.path.basename(bench_path))[0]
        print(f"\n### bench={bench_stem} rows={len(rows)}")
        for ckpt_name, lora_request in checkpoints:
            for k in ks:
                res = run_listwise(llm, tok, rows, k, args.top_k, args.max_tokens, args.extract_tokens,
                                   lora_request=lora_request, render=args.render)
                m = aggregate(res, k_ndcg=args.k_ndcg, n_out=args.top_k)
                ts = two_stage_stats(res, args.top_k)
                m["two_stage"] = ts
                print(f"\n=== {bench_stem} | ckpt={ckpt_name} | k={k} ===")
                print(json.dumps(m, indent=2, ensure_ascii=False))
                print(f"  two-stage: stage-1 self-served {ts['n_stage1_selfserved']}/{len(res)} "
                      f"({ts['stage1_valid_rate']:.1%}) | needed stage-2 {ts['n_two_stage']}/{len(res)} "
                      f"({ts['two_stage_rate']:.1%}) | stage-2 rescued {ts['n_stage2_rescued']}/{ts['n_two_stage']}")
                base = args.tag or "sid"
                suf = f"_n{args.n}" if args.n else ""
                ckpt_tag = f"__{ckpt_name}" if lora_dirs else ""
                bench_tag = f"{bench_stem}_" if args.benches else ""
                var = ((f"_{args.slice}" if args.slice != "test" else "")
                       + (f"_r{args.render}" if args.render != "sid" else "")
                       + ("_corrupt" if args.corrupt_sid else ""))
                path = (f"{args.outdir}/{short}{ckpt_tag}__{bench_tag}{base}"
                        f"_k{k}_N{args.n_cand}_s{args.shuffle}{var}{suf}.json")
                dump_json(path, {"meta": {**vars(args), "k": k, "bench": bench_path, "checkpoint": ckpt_name,
                                          "arm": "sid", "protocol": "listwise-two-stage", "tp": args.tp,
                                          "render": args.render,
                                          "corrupt_sid": args.corrupt_sid,
                                          "decoding": "greedy+structured",
                                          "vllm": vllm_version()},
                                 "metrics": m, "results": res})
                print(f"wrote {path}")


if __name__ == "__main__":
    main()
