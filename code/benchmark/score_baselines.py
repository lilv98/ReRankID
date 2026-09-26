"""S5 calibration baselines: zero- or low-LLM-cost rankers over the SAME render-time slates, emitting
result JSONs in the scorers' exact shape (meta/metrics/results with per-user order+grades), so
paired_stats.py and stability_audit.py work on them unchanged.

  random       seeded uniform top-10 — the empirical twin of the analytic floor in bench_manifest.
  caption      cosine between each candidate caption's embedding and the centroid of the last-k
               history caption embeddings (Qwen3-Embedding-0.6B via vLLM, each caption embedded
               separately). Bounds how far an instruction-free caption-similarity signal goes on
               the same slates — if a text arm only matches this, it isn't reasoning.

Usage:
  python code/benchmark/score_baselines.py --baseline random --n 50
  python code/benchmark/score_baselines.py --baseline caption          # GPU (vLLM embeddings)
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

RANDOM_SEED = 7   # distinct from SLATE_SEED(1)/build_pool(2)/format-only(3)/corrupt(4)


# ---------- per-baseline candidate scoring (higher = ranked earlier) ----------
def scores_random(row, k, rng):
    return list(rng.random(len(row["cand_pids"])))


BASELINES = {"random": scores_random}


# ---------- caption similarity (GPU, vLLM embeddings) ----------
def run_caption(rows, k, model, tp):
    """Cosine between each candidate caption's embedding and the centroid of the last-k history
    caption embeddings. Captions are embedded individually (unique texts encoded once) — short-text
    encoding matches the embedder's training distribution better than one concatenated history doc."""
    from vllm import LLM
    llm = LLM(model=model, runner="pooling", dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=tp or 1, gpu_memory_utilization=0.90,
              max_model_len=2048, enforce_eager=True)

    texts, tid = [], {}
    def _tid(t):
        if t not in tid:
            tid[t] = len(texts)
            texts.append(t)
        return tid[t]

    per_row = []
    for row in rows:
        uni = [i for i, (s, c) in enumerate(zip(row["hist_sids_str"], row["hist_caps"])) if s and c]
        keep = uni[-k:] if k and k > 0 else []
        cand_ids = [_tid(clean_caption(c)) for c in row["cand_captions"]]
        hist_ids = [_tid(clean_caption(row["hist_caps"][i])) for i in keep]
        per_row.append((cand_ids, hist_ids))

    emb = np.array([o.outputs.embedding for o in llm.embed(texts)], dtype=np.float32)
    emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
    print(f"  embedded {len(texts):,} unique captions (dim {emb.shape[1]})")

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", default=["random"],
                    choices=list(BASELINES) + ["caption"])
    ap.add_argument("--bench", default=config.BENCH)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--n-cand", type=int, default=30)
    ap.add_argument("--shuffle", type=int, default=0)
    ap.add_argument("--k", type=int, default=64,
                    help="history length (caption)")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--k-ndcg", type=int, default=5)
    ap.add_argument("--emb-model", default="Qwen/Qwen3-Embedding-0.6B", help="caption encoder")
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default=config.RESULTS)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # Floors are reporting references, estimated once on the frozen test slice. The dev slice is
    # for model selection (relative comparisons between checkpoints) and needs no floors.
    rows = load_bench(args.bench, n=args.n, no_profile=True, n_cand=args.n_cand,
                      shuffle=args.shuffle, bench_slice="test")
    print(f"rows={len(rows)} slice=test N={args.n_cand} s={args.shuffle} k={args.k} "
          f"baselines={args.baseline}")

    for name in args.baseline:
        t0 = time.time()
        if name == "caption":
            all_scores = run_caption(rows, args.k, args.emb_model, args.tp)
        else:
            fn = BASELINES[name]
            all_scores = []
            for row in rows:
                rng = np.random.default_rng([RANDOM_SEED, int(row["uid"])])
                all_scores.append(fn(row, args.k, rng))
        res = []
        for row, sc in zip(rows, all_scores):
            order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))[:args.top_k]
            res.append({"uid": int(row["uid"]), "grades": [int(g) for g in row["cand_grades"]],
                        "order": order, "n_listed": len(order), "n_hist_kept": min(args.k, 999)})
        m = aggregate(res, k_ndcg=args.k_ndcg, n_out=args.top_k)
        print(f"\n=== {name} | {time.time() - t0:.0f}s ===")
        print(json.dumps(m, indent=2, ensure_ascii=False))
        # caption depends on a scorer model; bake it into the filename so runs with
        # different models never overwrite each other (the CPU baselines are model-free)
        mod = f"_{args.emb_model.split('/')[-1]}" if name == "caption" else ""
        suf = f"_n{args.n}" if args.n else ""
        tag = args.tag or f"baseline_{name}{mod}_k{args.k}_N{args.n_cand}_s{args.shuffle}"
        path = f"{args.outdir}/{tag}{suf}.json"
        meta = {**vars(args), "arm": "baseline", "baseline": name}
        dump_json(path, {"meta": meta, "metrics": m, "results": res})
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
