"""Audit the frozen shared post-training pool and one deterministic phase."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
config.setup_env()

import argparse
import hashlib
import json
from collections import Counter

import pandas as pd

from rft.contract import POOL_TARGET_ROWS, RFT_RECIPE
from rft.stages import select_stage_rows, uid_digest, validate_phase_manifest
from sft.build_dataset import validate_pool_row
from utils.data_io import dump_json, load_rows


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(rows, field):
    values = Counter(str(int(row[field])) for row in rows)
    return dict(sorted(values.items(), key=lambda item: int(item[0])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=config.POSTTRAIN_V1_POOL)
    parser.add_argument("--manifest", default=config.POSTTRAIN_V1_PHASE_MANIFEST)
    parser.add_argument("--sft-manifest", default=config.SFT_V3_PHASE_MANIFEST)
    parser.add_argument("--registry", default=os.path.join(config.BENCHMARK_DIR, "pools.parquet"))
    parser.add_argument("--target-rows", type=int, choices=POOL_TARGET_ROWS, default=12000)
    parser.add_argument("--out", default=config.POSTTRAIN_V1_AUDIT)
    args = parser.parse_args()

    pool_sha = file_sha256(args.pool)
    rows = load_rows(args.pool)
    with open(args.manifest) as handle:
        manifest = json.load(handle)
    validate_phase_manifest(
        manifest, rows, pool_sha, manifest["excluded_sft_uid_sha256"]
    )
    selected = select_stage_rows(rows, manifest, args.target_rows)
    for row in selected:
        validate_pool_row(row)
        if int(row["n_hist_unified"]) < int(row["k"]):
            raise ValueError(f"history shorter than requested k for uid={row['uid']}")

    with open(args.sft_manifest) as handle:
        sft_manifest = json.load(handle)
    sft_uids = {int(uid) for uid, _ in sft_manifest.get("full_keys") or []}
    if uid_digest(sft_uids) != manifest["excluded_sft_uid_sha256"]:
        raise ValueError("post-training manifest records the wrong excluded SFT-v3 user set")
    selected_uids = {int(row["uid"]) for row in selected}
    registry = pd.read_parquet(args.registry, columns=["uid", "pool"])
    protected = set(registry.uid[registry.pool.isin(["test", "dev", "rl"])].astype(int))
    overlaps = {
        "sft_v3": len(selected_uids & sft_uids),
        "test_dev_rl_registry": len(selected_uids & protected),
    }
    if any(overlaps.values()):
        raise ValueError(f"post-training user leakage: {overlaps}")

    report = {
        "schema_version": 1,
        "recipe": RFT_RECIPE,
        "target_rows": args.target_rows,
        "rows": len(selected),
        "unique_users": len(selected_uids),
        "single_positive_rows": sum(
            sum(int(value) > 0 for value in row["cand_grades"]) == 1 for row in selected
        ),
        "unique_candidate_pid_rows": sum(
            len(row["cand_pids"]) == len(set(row["cand_pids"])) for row in selected
        ),
        "overlap": overlaps,
        "distribution": {
            "k": distribution(selected, "k"),
            "n_cand": distribution(selected, "n_cand"),
            "pos_grade": distribution(selected, "pos_grade"),
        },
        "hard_negatives": {
            "mean": round(sum(int(row["n_hard_neg"]) for row in selected) / len(selected), 3),
            "rows_with_any": sum(int(row["n_hard_neg"]) > 0 for row in selected),
        },
        "source": {
            "pool_sha256": pool_sha,
            "phase_manifest_sha256": file_sha256(args.manifest),
            "sft_phase_manifest_sha256": file_sha256(args.sft_manifest),
        },
    }
    dump_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
