# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Appendix numbers recomputed from the per-user results.

First, the forced-extraction accounting table (Table B.1) at the anchor, plus the
k=0 SID Base rows quoted in its paragraph: Extraction% is the share of users whose answer came from
the stage-2 extraction call, Stage-1 only rescores with every extraction-salvaged answer set to
zero, and Gain is their difference. Second, the direct RLVR reward contrast (NDCG@5 reward minus
Hit@1 reward) with the user-paired bootstrap CI of paired_stats.py (10,000 resamples, seed 0).
Third, the mean API-reported prompt tokens of the three text references at the anchor (App. D.2).
No model is run.

    python code/paper/make_appendix_tables.py [--results downloads/rerankid/results]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.config import ROOT
from utils.metrics import BOOT, half_up, is_valid, ndcg, paired_bootstrap
from utils.prompts import TOP_K

RESULTS = Path(ROOT) / "downloads/rerankid/results"

# (system, view, result file) in the row order of Table B.1.
EXTRACTION_ROWS = [
    ("Base 1.7B", "SID", "diagnosis/onerec-1.7b/sid/k64_n30.json"),
    ("Base 1.7B", "Text", "diagnosis/onerec-1.7b/text/k64_n30.json"),
    ("Base 1.7B", "Both", "diagnosis/onerec-1.7b/both/k64_n30.json"),
    ("Base 8B", "SID", "diagnosis/onerec-8b/sid/k64_n30.json"),
    ("Base 8B", "Text", "diagnosis/onerec-8b/text/k64_n30.json"),
    ("Base 8B", "Both", "diagnosis/onerec-8b/both/k64_n30.json"),
    ("SFT 1.7B", "SID", "posttraining/sft/onerec-1.7b/test/k64_n30.json"),
    ("SFT 8B", "SID", "posttraining/sft/onerec-8b/test/k64_n30.json"),
    ("RFT 1.7B", "SID", "posttraining/rft/onerec-1.7b/test/k64_n30.json"),
    ("RFT 8B", "SID", "posttraining/rft/onerec-8b/test/k64_n30.json"),
    ("RLVR (NDCG@5) 1.7B", "SID", "posttraining/rlvr-ndcg5/onerec-1.7b/test/k64_n30.json"),
    ("RLVR (NDCG@5) 8B", "SID", "posttraining/rlvr-ndcg5/onerec-8b/test/k64_n30.json"),
    ("RLVR (Hit@1) 1.7B", "SID", "posttraining/rlvr-hit1/onerec-1.7b/test/k64_n30.json"),
    ("RLVR (Hit@1) 8B", "SID", "posttraining/rlvr-hit1/onerec-8b/test/k64_n30.json"),
]
NO_HISTORY_ROWS = [
    ("Base 1.7B (k=0)", "SID", "diagnosis/onerec-1.7b/sid/k0_n30.json"),
    ("Base 8B (k=0)", "SID", "diagnosis/onerec-8b/sid/k0_n30.json"),
]

# (graded reward, binary reward) at each scale, in the order the CIs are drawn.
REWARD_CONTRASTS = [
    ("8B", "posttraining/rlvr-ndcg5/onerec-8b/test/k64_n30.json",
     "posttraining/rlvr-hit1/onerec-8b/test/k64_n30.json"),
    ("1.7B", "posttraining/rlvr-ndcg5/onerec-1.7b/test/k64_n30.json",
     "posttraining/rlvr-hit1/onerec-1.7b/test/k64_n30.json"),
]

TEXT_REFERENCES = [
    ("Gemini-3.6-Flash", "diagnosis/gemini-3.6-flash/k64_n30.json"),
    ("GPT-5.6-Terra", "diagnosis/gpt-5.6-terra/k64_n30.json"),
    ("Claude-Sonnet-5", "diagnosis/claude-sonnet-5/k64_n30.json"),
]


def load_rows(results, relative):
    with (results / relative).open() as handle:
        payload = json.load(handle)
    return sorted(payload["results"], key=lambda row: row["uid"])


def per_user_ndcg5(rows):
    """End-to-end NDCG@5 per user; invalid outputs score zero."""
    return np.array([ndcg(row["order"], row["grades"], 5) if is_valid(row, TOP_K) else 0.0
                     for row in rows])


def full_list(row):
    order = row.get("order")
    return isinstance(order, list) and len(order) == TOP_K and len(set(order)) == TOP_K


def extraction_table(results):
    print("Forced-extraction accounting (anchor k=64, N=30)")
    print(f"{'System':20s} {'View':5s} {'Extr%':>6s} {'Valid%':>7s} {'NDCG@5':>7s} "
          f"{'S1-only':>8s} {'Gain':>7s}")
    for block in (EXTRACTION_ROWS, NO_HISTORY_ROWS):
        for system, view, relative in block:
            rows = load_rows(results, relative)
            scores = per_user_ndcg5(rows)
            extracted = np.array([row.get("stage") == 2 for row in rows])
            valid = np.array([full_list(row) for row in rows])
            stage1_only = scores * ~extracted
            print(f"{system:20s} {view:5s} {half_up(100 * extracted.mean(), 1):>6} "
                  f"{half_up(100 * valid.mean(), 1):>7} {half_up(scores.mean(), 3):>7} "
                  f"{half_up(stage1_only.mean(), 3):>8} "
                  f"{half_up(scores.mean() - stage1_only.mean(), 3):>+7}")
        print()


def reward_contrast(results):
    print(f"RLVR reward contrast, NDCG@5 reward minus Hit@1 reward "
          f"(paired bootstrap, {BOOT:,} resamples, seed 0)")
    for scale, graded, binary in REWARD_CONTRASTS:
        rows_a, rows_b = load_rows(results, graded), load_rows(results, binary)
        if [row["uid"] for row in rows_a] != [row["uid"] for row in rows_b]:
            raise SystemExit(f"{scale}: the two result files cover different users")
        diffs = per_user_ndcg5(rows_a) - per_user_ndcg5(rows_b)
        lo, hi = paired_bootstrap(diffs)
        print(f"  {scale:5s} {half_up(diffs.mean(), 3):+}  "
              f"CI [{half_up(lo, 3):+}, {half_up(hi, 3):+}]")
    print()


def api_prompt_tokens(results):
    print("Mean API-reported prompt tokens at the anchor")
    for name, relative in TEXT_REFERENCES:
        counts = [row["usage"]["prompt_tokens"] for row in load_rows(results, relative)
                  if row.get("usage") and row["usage"].get("prompt_tokens")]
        print(f"  {name:17s} n={len(counts)}  mean={np.mean(counts):,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Recompute appendix tables from result files.")
    parser.add_argument("--results", type=Path, default=RESULTS, help="results root")
    args = parser.parse_args()
    extraction_table(args.results)
    reward_contrast(args.results)
    api_prompt_tokens(args.results)


if __name__ == "__main__":
    main()
