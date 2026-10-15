# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the RLVR training and monitoring sets (paper Sec. 5, App. C.4).

Train: the deterministic 6,000-prompt subset of the shared post-training pool
(data/posttrain/pool.parquet, built by code/sft/build_pool.py), the same prompts used by RFT and
OPD. Monitor: the 400 benchmark dev users at k=64, N=30, with no policy-gradient updates. Every
prompt is the SID-only scoring prompt plus the /think switch; the reward ground truth holds the
candidate grades. Writes data/rlvr/{train.parquet,val.parquet,stats.json}, the files released
on Hugging Face (rlvr-ndcg5 and rlvr-hit1 share them).
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config  # noqa: E402

config.setup_env()

import pandas as pd  # noqa: E402

from rlvr.recipe import TRAIN_ROWS, VAL_K, VAL_N_CAND, VAL_ROWS  # noqa: E402
from utils.data_io import dump_json, load_bench, load_rows  # noqa: E402
from utils.pools import select_subset  # noqa: E402
from utils.prompts import (TOP_K, build_context, render_candidates,  # noqa: E402
                           rerank_instruction, rerank_system)

DATA_SOURCE = "listwise_rerank_ndcg5"   # verl reward-routing key, shared by both rewards


def validate_task_row(row):
    grades = [int(value) for value in row["cand_grades"]]
    n_cand = int(row["n_cand"])
    if len(grades) != n_cand or sum(value > 0 for value in grades) != 1:
        raise ValueError(f"expected one positive in an N={n_cand} slate for uid={row['uid']}")
    if len(row["cand_sids"]) != n_cand:
        raise ValueError(f"candidate SID count differs from N for uid={row['uid']}")
    if int(row["n_hist_unified"]) < int(row["k"]):
        raise ValueError(f"history is shorter than k for uid={row['uid']}")
    return grades


def build_row(row, index, split, k=None):
    k = int(row["k"] if k is None else k)
    grades = validate_task_row(row)
    context, n_kept = build_context(row, k, "sid")
    user = rerank_instruction(
        context, render_candidates(row, "sid"), "sid", top_k=TOP_K
    ) + "/think"
    ground_truth = json.dumps(
        {"grades": grades, "n_cand": len(grades), "k_hist": k},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": rerank_system("sid")},
            {"role": "user", "content": user},
        ],
        "ability": "rerank",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "uid": int(row["uid"]),
            "slate_id": str(row.get("slate_id", row["uid"])),
            "index": int(index),
            "split": str(split),
            "k_hist": k,
            "n_hist_kept": int(n_kept),
            "n_cand": len(grades),
        },
    }


def load_val_rows(bench):
    """Benchmark dev users at the anchor, with the history length the prompt builder expects."""
    rows = load_bench(bench, n_cand=VAL_N_CAND, shuffle=0, bench_slice="dev")
    for row in rows:
        row["k"] = VAL_K
        row["n_hist_unified"] = sum(
            bool(sid) and bool(caption)
            for sid, caption in zip(row["hist_sids_str"], row["hist_caps"])
        )
    return rows


def distribution(rows, field):
    values = Counter(str(int(row[field])) for row in rows)
    return dict(sorted(values.items(), key=lambda item: int(item[0])))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", default=config.POSTTRAIN_POOL)
    parser.add_argument("--bench", default=config.BENCH)
    parser.add_argument("--outdir", default=config.RLVR_DIR)
    args = parser.parse_args()

    train_rows = select_subset(load_rows(args.pool))
    val_rows = load_val_rows(args.bench)
    test_rows = load_bench(args.bench, n_cand=VAL_N_CAND, shuffle=0, bench_slice="test")
    train_uids = {int(row["uid"]) for row in train_rows}
    val_uids = {int(row["uid"]) for row in val_rows}
    test_uids = {int(row["uid"]) for row in test_rows}
    if len(train_rows) != TRAIN_ROWS or len(val_rows) != VAL_ROWS:
        raise ValueError(f"unexpected train/val size {len(train_rows)}/{len(val_rows)}")
    if len(train_uids) != TRAIN_ROWS or len(val_uids) != VAL_ROWS:
        raise ValueError("RLVR train and val must each contain one row per user")
    if train_uids & (val_uids | test_uids):
        raise ValueError("RLVR train users overlap benchmark dev or test users")

    train = [build_row(row, index, "train") for index, row in enumerate(train_rows)]
    val = [build_row(row, index, "dev", k=VAL_K) for index, row in enumerate(val_rows)]
    if any(record["prompt"][1]["content"].count("/think") != 1 for record in train + val):
        raise ValueError("each RLVR prompt must contain exactly one /think switch")

    os.makedirs(args.outdir, exist_ok=True)
    train_path = os.path.join(args.outdir, "train.parquet")
    val_path = os.path.join(args.outdir, "val.parquet")
    pd.DataFrame(train).to_parquet(train_path, index=False)
    pd.DataFrame(val).to_parquet(val_path, index=False)

    stats = {
        "train_rows": len(train),
        "val_rows": len(val),
        "train_distribution": {
            "k": distribution(train_rows, "k"),
            "n_cand": distribution(train_rows, "n_cand"),
            "pos_grade": distribution(train_rows, "pos_grade"),
        },
        "val_protocol": {"slice": "dev", "k": VAL_K, "n_cand": VAL_N_CAND, "shuffle": 0},
        "val_positive_grade_distribution": distribution(val_rows, "pos_grade"),
        "overlap": {
            "train_vs_benchmark_dev_users": 0,
            "train_vs_benchmark_test_users": 0,
        },
    }
    dump_json(os.path.join(args.outdir, "stats.json"), stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
