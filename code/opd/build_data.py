# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the label-free OPD parquets from the RLVR parquets (paper Sec. 5).

OPD uses the same 6,000 training prompts and 400 dev monitor prompts as RLVR, with identical prompt
text and order. This script drops the ranking labels and every label-derived field: ground_truth
becomes empty (zero_reward.py ignores it) and extra_info keeps only row identifiers and the
history/slate sizes. The released data/opd/ files are the output of this script; rebuilding is
optional.

Input:  data/rlvr/{train,val}.parquet
Output: data/opd/{train,val}.parquet, data/opd/stats.json

    python code/opd/build_data.py
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from utils import config
from utils.data_io import dump_json

DATA_SOURCE = "rerankid_opd_sid"   # verl reward-routing key, recorded in the released parquets
KEEP_EXTRA_INFO = ("uid", "slate_id", "split", "k_hist", "n_cand", "n_hist_kept")
TRAIN_BATCH_SIZE = 30              # run_opd.sh data.train_batch_size: 6,000 / 30 = 200 steps


def normalize_prompt(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [
        {"role": str(dict(message)["role"]), "content": str(dict(message)["content"])}
        for message in value
    ]


def build_frame(source):
    """One label-free verl row per source row, in source order."""
    rows = []
    for index, row in source.iterrows():
        old_extra = dict(row["extra_info"])
        extra = {"index": int(index), "source_index": int(old_extra["index"])}
        for key in KEEP_EXTRA_INFO:
            value = old_extra.get(key)
            if hasattr(value, "item"):
                value = value.item()
            extra[key] = value
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": normalize_prompt(row["prompt"]),
                "ability": "rerank",
                # Required by verl's reward manager; deliberately empty.
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": extra,
            }
        )
    return pd.DataFrame(rows)


def distribution(frame, key):
    counts = Counter(int(dict(item)[key]) for item in frame["extra_info"])
    return {str(value): count for value, count in sorted(counts.items())}


def main():
    parser = argparse.ArgumentParser(description="Build data/opd from data/rlvr.")
    parser.add_argument("--src", default=config.RLVR_DIR, help="directory with RLVR parquets")
    parser.add_argument("--out", default=config.OPD_DIR, help="output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames = {}
    for split in ("train", "val"):
        source = pd.read_parquet(os.path.join(args.src, f"{split}.parquet"))
        frames[split] = build_frame(source)
        assert [normalize_prompt(p) for p in source["prompt"]] == list(frames[split]["prompt"])
        frames[split].to_parquet(os.path.join(args.out, f"{split}.parquet"),
                                 index=False, engine="pyarrow", compression="zstd")

    n_train = len(frames["train"])
    stats = {
        "train_rows": n_train,
        "val_rows": len(frames["val"]),
        "train_batch_size": TRAIN_BATCH_SIZE,
        "optimizer_steps": n_train // TRAIN_BATCH_SIZE,
        "teacher_query_budget": n_train,
        "train_distribution": {
            "k": distribution(frames["train"], "k_hist"),
            "n_cand": distribution(frames["train"], "n_cand"),
        },
        # The monitor set is the benchmark dev slice at the anchor cell, as for RLVR.
        "val_protocol": {"slice": "dev", "k": 64, "n_cand": 30, "shuffle": 0},
        "label_fields": {
            "reward_ground_truth": "empty",
            "positive_grade": "removed",
            "teacher_hint": "absent",
        },
    }
    dump_json(os.path.join(args.out, "stats.json"), stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
