"""Measure stage-1 prompt length (OpenOneRec tokenizer) across the history grid.

Replicates sid_engine.stage1_prompt exactly (system + user, chat template,
enable_thinking=True) for each rendering, so the k=64 means must reproduce the
anchor-table input-token cells (845 / 16,722 / 17,255).  Feeds the token-growth
panel of the paper scaling figure.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
K_GRID = [0, 4, 8, 16, 32, 64, 128, 256]
RENDERS = ["sid", "text", "sid_text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(ROOT / "downloads/wandb-paper-final/data/benchmark/bench.parquet"))
    ap.add_argument("--tokenizer", default="OpenOneRec/OneRec-1.7B-pro")
    ap.add_argument("--n-cand", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "experiments/prompt_tokens_by_k.json"))
    args = ap.parse_args()

    from utils.config import api_key
    hf_token = api_key("huggingface")
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)

    from transformers import AutoTokenizer

    from utils.data_io import load_bench
    from utils.prompts import (build_context, render_candidates,
                               rerank_instruction, rerank_system)

    def stage1_prompt(tok, row, k, top_k, render):
        # Mirrors sid_engine.stage1_prompt without importing vllm.
        ctx, _ = build_context(row, k, render)
        user = rerank_instruction(ctx, render_candidates(row, render), render, top_k=top_k)
        msgs = [{"role": "system", "content": rerank_system(render)},
                {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True), None

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = load_bench(args.bench, n_cand=args.n_cand, shuffle=0, bench_slice="test")
    assert len(rows) == 400, len(rows)

    out = {"tokenizer": args.tokenizer, "n_users": len(rows), "n_cand": args.n_cand,
           "shuffle": 0, "slice": "test", "k_grid": K_GRID, "mean_prompt_tokens": {}}
    for render in RENDERS:
        means = []
        for k in K_GRID:
            counts = [len(tok(stage1_prompt(tok, r, k, 10, render=render)[0]).input_ids)
                      for r in rows]
            means.append(round(float(np.mean(counts)), 1))
            print(f"{render} k={k}: mean={means[-1]}", flush=True)
        out["mean_prompt_tokens"][render] = means

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
