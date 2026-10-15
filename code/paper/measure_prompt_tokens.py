# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Mean stage-1 prompt length under the OpenOneRec tokenizer across the history grid.

Builds the stage-1 prompt exactly as sid_engine.stage1_prompt does (prompts.stage1_messages under
the chat template with enable_thinking=True, without importing vLLM) for the SID, text and combined
renderings of the 400 test users at N=30. The k=64 means are the token-economy cells of the paper
(845 / 16,722 / 17,255), and the file feeds panel (a) of Figure 2 (plot_scaling.py). CPU only;
downloads the pinned tokenizer.

    python code/paper/measure_prompt_tokens.py [--out results/diagnosis/prompt-tokens.json]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config

config.setup_env()

import numpy as np

from utils.prompts import TOP_K, stage1_messages

K_GRID = [0, 4, 8, 16, 32, 64, 128, 256]
RENDERS = ["sid", "text", "sid_text"]


def stage1_prompt(tok, row, k, render):
    """The stage-1 prompt text of sid_engine.stage1_prompt."""
    msgs, _ = stage1_messages(row, k, render, TOP_K)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)


def main():
    parser = argparse.ArgumentParser(description="Measure stage-1 prompt tokens by history length.")
    parser.add_argument("--bench", default=config.BENCH)
    parser.add_argument("--model", default="onerec-1.7b", choices=sorted(config.MODELS),
                        help="pinned model whose tokenizer is used")
    parser.add_argument("--n-cand", type=int, default=30)
    parser.add_argument("--out", type=Path,
                        default=Path(config.RESULTS) / "diagnosis/prompt-tokens.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from utils.data_io import dump_json, load_bench

    repo, revision = config.MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    rows = load_bench(args.bench, n_cand=args.n_cand, shuffle=0, bench_slice="test")
    print(f"{len(rows)} test users", flush=True)

    out = {"tokenizer": repo, "n_users": len(rows), "n_cand": args.n_cand,
           "shuffle": 0, "slice": "test", "k_grid": K_GRID, "mean_prompt_tokens": {}}
    for render in RENDERS:
        means = []
        for k in K_GRID:
            counts = [len(tok(stage1_prompt(tok, row, k, render)).input_ids) for row in rows]
            means.append(round(float(np.mean(counts)), 1))
            print(f"{render} k={k}: mean={means[-1]}", flush=True)
        out["mean_prompt_tokens"][render] = means

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(args.out, out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
