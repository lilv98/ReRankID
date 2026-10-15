# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Random and Caption reference rankers on the benchmark test slice (paper Sec. 4).

random ranks each slate uniformly at random (seed 7 per user); its expectation is the analytic
floor in data/benchmark/stats.json["bench"]["random_floors"]. caption scores each candidate by the
cosine between its caption embedding and the centroid of the last-k history caption embeddings
(Qwen/Qwen3-Embedding-0.6B through vLLM, one caption per input). The embedder revision is not
pinned by default: meta.emb_revision records the snapshot loaded, and --emb-revision pins one.
Results have the scorers' format, so paired_stats.py reads them unchanged.

Writes results/diagnosis/<baseline>/k<k>_n<N>.json (_s<i> for a non-zero --shuffle).

    python code/benchmark/score_baselines.py --baseline random --n-cand 30
    python code/benchmark/score_baselines.py --baseline caption --k 64 --n-cand 30   # GPU
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ReRankID/code
from utils import config
config.setup_env()

import argparse
import json
import time

import numpy as np

from utils.data_io import dump_json, load_bench
from utils.metrics import aggregate
from utils.prompts import TOP_K, clean_caption

RANDOM_SEED = 7


def resolve_embedder(model, revision):
    """(local path, revision) of the caption embedder; HF ids are fetched at `revision`."""
    if os.path.isdir(model):
        return model, None
    from huggingface_hub import snapshot_download
    path = snapshot_download(model, revision=revision)
    return path, os.path.basename(path.rstrip("/"))


def run_caption(rows, k, model_path, tp):
    """Per-row candidate scores: cosine to the centroid of the last-k history captions. Each unique
    caption is embedded once and on its own."""
    from vllm import LLM
    llm = LLM(model=model_path, runner="pooling", dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=tp or 1, gpu_memory_utilization=0.90,
              max_model_len=2048, enforce_eager=True)

    tid = {}   # unique caption -> embedding row, in first-seen order
    per_row = []
    for row in rows:
        uni = [i for i, (s, c) in enumerate(zip(row["hist_sids_str"], row["hist_caps"])) if s and c]
        keep = uni[-k:] if k and k > 0 else []
        cand_ids = [tid.setdefault(clean_caption(c), len(tid)) for c in row["cand_captions"]]
        hist_ids = [tid.setdefault(clean_caption(row["hist_caps"][i]), len(tid)) for i in keep]
        per_row.append((cand_ids, hist_ids))

    emb = np.array([o.outputs.embedding for o in llm.embed(list(tid))], dtype=np.float32)
    emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
    print(f"  embedded {len(tid):,} unique captions (dim {emb.shape[1]})")

    scores = []
    for cand_ids, hist_ids in per_row:
        if not hist_ids:
            scores.append([0.0] * len(cand_ids))
            continue
        ctr = emb[hist_ids].mean(axis=0)
        nrm = np.linalg.norm(ctr)
        ctr = ctr / nrm if nrm > 0 else ctr
        scores.append([float(emb[ci] @ ctr) for ci in cand_ids])
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", nargs="+", default=["random"], choices=["random", "caption"])
    ap.add_argument("--bench", default=config.BENCH)
    ap.add_argument("--n-cand", type=int, default=30, help="slate size N (10, 20, 30, 50 or 100)")
    ap.add_argument("--shuffle", type=int, default=0, help="display-order seed")
    ap.add_argument("--k", type=int, default=64, help="history length (caption)")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--k-ndcg", type=int, default=5)
    ap.add_argument("--emb-model", default="Qwen/Qwen3-Embedding-0.6B", help="caption encoder")
    ap.add_argument("--emb-revision", default=None, help="HF revision of --emb-model")
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--outdir", default=None, help="override the output directory")
    args = ap.parse_args()

    rows = load_bench(args.bench, n_cand=args.n_cand, shuffle=args.shuffle, bench_slice="test")
    print(f"rows={len(rows)} slice=test N={args.n_cand} s={args.shuffle} k={args.k} "
          f"baselines={args.baseline}")

    for name in args.baseline:
        t0 = time.time()
        meta = {"baseline": name, "bench": config.relpath(args.bench), "slice": "test",
                "n_cand": args.n_cand, "shuffle": args.shuffle, "k": args.k, "top_k": args.top_k,
                "k_ndcg": args.k_ndcg, "arm": "baseline"}
        if name == "caption":
            emb_path, emb_revision = resolve_embedder(args.emb_model, args.emb_revision)
            meta.update(emb_model=args.emb_model, emb_revision=emb_revision)
            all_scores = run_caption(rows, args.k, emb_path, args.tp)
        else:
            meta["seed"] = RANDOM_SEED
            all_scores = [list(np.random.default_rng([RANDOM_SEED, int(row["uid"])])
                               .random(len(row["cand_pids"]))) for row in rows]
        res = []
        for row, sc in zip(rows, all_scores):
            order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))[:args.top_k]
            res.append({"uid": int(row["uid"]), "grades": [int(g) for g in row["cand_grades"]],
                        "order": order, "n_listed": len(order), "n_hist_kept": args.k})
        m = aggregate(res, k_ndcg=args.k_ndcg, n_out=args.top_k)
        print(f"\n=== {name} | {time.time() - t0:.0f}s ===")
        print(json.dumps(m, indent=2, ensure_ascii=False))
        cell = f"k{args.k}_n{args.n_cand}" + (f"_s{args.shuffle}" if args.shuffle else "")
        path = os.path.join(args.outdir or f"{config.RESULTS}/diagnosis/{name}", f"{cell}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dump_json(path, {"meta": meta, "metrics": m, "results": res})
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
