"""Build the frozen 6k RLVR train set and 400-row benchmark-dev monitor set.

The train prompts are exactly the deterministic 6k phase of the shared
post-training pool used by RFT.  No train row is withheld for validation.
Validation uses the frozen benchmark dev slice at the primary k=64, N=30
setting.  The formal benchmark evaluator remains the authority for checkpoint
selection because verl validation is single-stage and unconstrained.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config

config.setup_env()

import pandas as pd

from rft.stages import (key_digest, row_key, select_stage_rows,
                        validate_phase_manifest)
from rl.contract import (BENCH_SHA256, DATA_SCHEMA, POSTTRAIN_MANIFEST_SHA256,
                         POSTTRAIN_POOL_SHA256, RECIPE, TRAIN_KEY_SHA256,
                         RLVR_STATS_SHA256, TRAIN_PARQUET_SHA256, TRAIN_ROWS,
                         VAL_K, VAL_N_CAND, VAL_PARQUET_SHA256, VAL_ROWS)
from utils.data_io import dump_json, load_bench, load_rows
from utils.prompts import (TOP_K, build_context, render_candidates,
                           rerank_instruction, rerank_system)

DATA_SOURCE = "listwise_rerank_ndcg5"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path, expected):
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}: {actual} != {expected}")
    return actual


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


def distribution(rows, field):
    values = Counter(str(int(row[field])) for row in rows)
    return dict(sorted(values.items(), key=lambda item: int(item[0])))


def atomic_parquet(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    pd.DataFrame(records).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=config.POSTTRAIN_V1_POOL)
    parser.add_argument("--manifest", default=config.POSTTRAIN_V1_PHASE_MANIFEST)
    parser.add_argument("--bench", default=config.BENCH)
    parser.add_argument("--outdir", default=config.RLVR_V1_DATA_DIR)
    args = parser.parse_args()

    pool_sha = verify_hash(args.pool, POSTTRAIN_POOL_SHA256)
    manifest_sha = verify_hash(args.manifest, POSTTRAIN_MANIFEST_SHA256)
    bench_sha = verify_hash(args.bench, BENCH_SHA256)

    parent = load_rows(args.pool)
    with open(args.manifest) as handle:
        manifest = json.load(handle)
    validate_phase_manifest(
        manifest, parent, pool_sha, manifest["excluded_sft_uid_sha256"]
    )
    train_rows = select_stage_rows(parent, manifest, TRAIN_ROWS)
    train_keys = {row_key(row) for row in train_rows}
    expected_stage = manifest["stages"][str(TRAIN_ROWS)]
    train_key_sha = key_digest(train_keys)
    if train_key_sha != expected_stage["key_sha256"] or train_key_sha != TRAIN_KEY_SHA256:
        raise ValueError("selected RLVR train keys differ from the frozen 6k phase")

    val_rows = load_bench(
        args.bench, no_profile=True, n_cand=VAL_N_CAND, shuffle=0,
        bench_slice="dev",
    )
    test_rows = load_bench(
        args.bench, no_profile=True, n_cand=VAL_N_CAND, shuffle=0,
        bench_slice="test",
    )
    for row in val_rows:
        row["k"] = VAL_K
        row["n_hist_unified"] = sum(
            bool(sid) and bool(caption)
            for sid, caption in zip(row["hist_sids_str"], row["hist_caps"])
        )
    if len(train_rows) != TRAIN_ROWS or len(val_rows) != VAL_ROWS:
        raise ValueError(
            f"unexpected train/val size: {len(train_rows)}/{len(val_rows)} "
            f"!= {TRAIN_ROWS}/{VAL_ROWS}"
        )
    train_uids = {int(row["uid"]) for row in train_rows}
    val_uids = {int(row["uid"]) for row in val_rows}
    test_uids = {int(row["uid"]) for row in test_rows}
    if len(train_uids) != TRAIN_ROWS or len(val_uids) != VAL_ROWS:
        raise ValueError("RLVR train and val must each contain one row per user")
    if train_uids & val_uids:
        raise ValueError("RLVR train users overlap benchmark dev users")
    if train_uids & test_uids:
        raise ValueError("RLVR train users overlap benchmark test users")

    train = [build_row(row, index, "train") for index, row in enumerate(train_rows)]
    val = [build_row(row, index, "dev", k=VAL_K) for index, row in enumerate(val_rows)]
    if any(record["prompt"][1]["content"].count("/think") != 1 for record in train + val):
        raise ValueError("each RLVR prompt must contain exactly one /think switch")

    train_path = os.path.join(args.outdir, "train.parquet")
    val_path = os.path.join(args.outdir, "val.parquet")
    stats_path = os.path.join(args.outdir, "stats.json")
    atomic_parquet(train_path, train)
    atomic_parquet(val_path, val)
    train_sha = file_sha256(train_path)
    val_sha = file_sha256(val_path)
    if train_sha != TRAIN_PARQUET_SHA256 or val_sha != VAL_PARQUET_SHA256:
        raise ValueError(
            "built RLVR parquet bytes differ from the frozen environment output: "
            f"train={train_sha}, val={val_sha}"
        )

    stats = {
        "schema_version": DATA_SCHEMA,
        "recipe": RECIPE,
        "train_rows": len(train),
        "val_rows": len(val),
        "train_key_sha256": train_key_sha,
        "train_distribution": {
            "k": distribution(train_rows, "k"),
            "n_cand": distribution(train_rows, "n_cand"),
            "pos_grade": distribution(train_rows, "pos_grade"),
        },
        "val_protocol": {
            "slice": "dev", "k": VAL_K, "n_cand": VAL_N_CAND,
            "shuffle": 0, "no_profile": True,
        },
        "val_positive_grade_distribution": distribution(val_rows, "pos_grade"),
        "overlap": {
            "train_vs_benchmark_dev_users": 0,
            "train_vs_benchmark_test_users": 0,
        },
        "source": {
            "posttrain_pool_sha256": pool_sha,
            "posttrain_manifest_sha256": manifest_sha,
            "benchmark_sha256": bench_sha,
        },
        "files": {
            "train_sha256": train_sha,
            "val_sha256": val_sha,
        },
    }
    dump_json(stats_path, stats)
    if file_sha256(stats_path) != RLVR_STATS_SHA256:
        raise ValueError("built RLVR stats.json differs from the frozen contract")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote {train_path}")
    print(f"wrote {val_path}")


if __name__ == "__main__":
    main()
