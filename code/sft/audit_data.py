"""Audit SFT v3 pool composition, disjointness, and item overlap.

Formal v3 paths always validate the full 12,000-row parent pool, then audit either its frozen
balanced 6,000-row pilot or the complete pool. Custom inputs are accepted only as visibly isolated
`/tmp` smoke paths and enforce the same row-level invariants at their smaller size.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
config.setup_env()

import argparse
import hashlib
import json

import numpy as np
import pyarrow.parquet as pq

from sft.v3_contract import (EXPECTED_ROWS, FORMAL_TARGET_ROWS, K_GRID, K_WEIGHTS,
                             N_CAND_GRID, N_CAND_WEIGHTS, PILOT_ROWS, RECIPE,
                             is_smoke_path)
from sft.v3_stages import select_stage_rows, validate_phase_manifest
from utils.data_io import dump_json, load_rows, update_stats

REGISTRY = f"{config.BENCHMARK_DIR}/pools.parquet"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def value_counts(rows, field):
    counts = {}
    for row in rows:
        value = str(int(row[field]))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def validate_rows(rows, formal, target_rows=EXPECTED_ROWS):
    uids = [int(row["uid"]) for row in rows]
    if len(uids) != len(set(uids)):
        raise AssertionError("SFT v3 requires exactly one slate per user")
    if not rows:
        raise AssertionError("SFT pool is empty")

    for row in rows:
        uid = row["uid"]
        if str(row.get("pool")) != "sft":
            raise AssertionError(f"pool is not sft for uid={uid}")
        pos_grade = int(row["pos_grade"])
        if pos_grade not in (2, 3):
            raise AssertionError(f"pos_grade must be 2 or 3 for uid={uid}")
        n = int(row["n_cand"])
        pids = [int(pid) for pid in row["cand_pids"]]
        sids = list(row["cand_sids"])
        captions = list(row["cand_captions"])
        grades = [int(grade) for grade in row["cand_grades"]]
        gold = [int(pid) for pid in row["gold_pids"]]
        if not (len(pids) == len(sids) == len(captions) == len(grades) == n):
            raise AssertionError(f"candidate arrays differ in length for uid={row['uid']}")
        if len(pids) != len(set(pids)):
            raise AssertionError(f"duplicate candidate PID for uid={row['uid']}")
        if len(gold) != 1 or pids.count(gold[0]) != 1:
            raise AssertionError(f"gold PID is not unique for uid={row['uid']}")
        positives = [index for index, grade in enumerate(grades) if grade > 0]
        if len(positives) != 1 or pids[positives[0]] != gold[0]:
            raise AssertionError(f"single-positive grade contract failed for uid={row['uid']}")
        if grades[positives[0]] != pos_grade:
            raise AssertionError(f"pos_grade disagrees with the positive candidate for uid={uid}")
        if int(row.get("n_pos", 1)) != 1:
            raise AssertionError(f"n_pos is not one for uid={row['uid']}")
        if not all(sids) or not all(captions):
            raise AssertionError(f"unrenderable candidate for uid={row['uid']}")
        if row.get("profile_header"):
            raise AssertionError(f"profile header is not empty for uid={row['uid']}")
        hist_pids = list(row["hist_pids"])
        hist_sids = list(row["hist_sids_str"])
        hist_captions = list(row["hist_caps"])
        n_hist = int(row["n_hist"])
        if not (len(hist_pids) == len(hist_sids) == len(hist_captions) == n_hist):
            raise AssertionError(f"history arrays differ in length for uid={uid}")
        n_hist_unified = int(row["n_hist_unified"])
        actual_unified = sum(bool(sid) and bool(caption)
                             for sid, caption in zip(hist_sids, hist_captions))
        if n_hist_unified != actual_unified:
            raise AssertionError(f"n_hist_unified disagrees with history arrays for uid={uid}")
        if n_hist_unified < max(K_GRID):
            raise AssertionError(f"unified history is shorter than 256 for uid={row['uid']}")
        if n_hist_unified < int(row["k"]):
            raise AssertionError(f"sampled k exceeds unified history for uid={row['uid']}")
        if int(row["k"]) not in K_GRID or n not in N_CAND_GRID:
            raise AssertionError(f"unsupported k or N for uid={row['uid']}")

    grade_counts = value_counts(rows, "pos_grade")
    k_counts = value_counts(rows, "k")
    n_counts = value_counts(rows, "n_cand")
    if grade_counts.get("2") != grade_counts.get("3"):
        raise AssertionError("SFT v3 must be balanced between grade 2 and grade 3")
    if formal:
        if target_rows not in FORMAL_TARGET_ROWS:
            raise AssertionError(f"unsupported formal target_rows={target_rows}")
        if len(rows) != target_rows or len(set(uids)) != target_rows:
            raise AssertionError(f"formal v3 stage requires {target_rows} rows and users")
        expected_per_grade = target_rows // 2
        if grade_counts != {"2": expected_per_grade, "3": expected_per_grade}:
            raise AssertionError(
                f"formal v3 stage requires exactly {expected_per_grade} users per positive grade"
            )
        if set(map(int, k_counts)) != set(K_GRID):
            raise AssertionError("formal v3 does not cover the complete k support")
        if set(map(int, n_counts)) != set(N_CAND_GRID):
            raise AssertionError("formal v3 does not cover the complete N support")
    return uids, grade_counts, k_counts, n_counts


def realized_distribution(counts, support, weights):
    total = sum(counts.values())
    return {
        str(value): {
            "rows": counts.get(str(value), 0),
            "share": round(counts.get(str(value), 0) / total, 6),
            "target_share": target,
        }
        for value, target in zip(support, weights)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=config.SFT_V3_POOL)
    ap.add_argument("--target-rows", type=int, choices=FORMAL_TARGET_ROWS,
                    default=EXPECTED_ROWS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-update-stats", action="store_true",
                    help="do not merge the formal audit into benchmark/stats.json")
    args = ap.parse_args()

    formal = os.path.realpath(args.pool) == os.path.realpath(config.SFT_V3_POOL)
    if args.out is None:
        args.out = (config.SFT_V3_PILOT_AUDIT
                    if args.target_rows == PILOT_ROWS else config.SFT_V3_AUDIT)
    if not formal and (not is_smoke_path(args.pool) or not is_smoke_path(args.out)):
        raise SystemExit("custom audits require pool and output under /tmp with 'smoke' in the path")
    if formal:
        expected_out = (config.SFT_V3_PILOT_AUDIT
                        if args.target_rows == PILOT_ROWS else config.SFT_V3_AUDIT)
        if os.path.realpath(args.out) != os.path.realpath(expected_out):
            raise SystemExit(
                f"formal v3 target {args.target_rows} audit must use {expected_out}"
            )
    elif args.target_rows != EXPECTED_ROWS:
        raise SystemExit("smoke audits do not use formal --target-rows staging")

    source_rows = load_rows(args.pool)
    pool_sha256 = file_sha256(args.pool)
    phase_manifest = None
    phase_manifest_sha256 = None
    if formal:
        validate_rows(source_rows, True, EXPECTED_ROWS)
        with open(config.SFT_V3_PHASE_MANIFEST) as handle:
            phase_manifest = json.load(handle)
        validate_phase_manifest(phase_manifest, source_rows, pool_sha256)
        phase_manifest_sha256 = file_sha256(config.SFT_V3_PHASE_MANIFEST)
        rows = select_stage_rows(
            source_rows, args.target_rows, phase_manifest, pool_sha256
        )
    else:
        rows = source_rows
    uids, grade_counts, k_counts, n_counts = validate_rows(
        rows, formal, args.target_rows
    )

    registry = pq.read_table(REGISTRY, columns=["uid", "pool"]).to_pandas()
    forbidden = set(registry.uid[registry.pool.isin(["test", "dev", "rl"])].astype(int))
    overlap = set(uids) & forbidden
    if overlap:
        raise AssertionError(f"SFT users overlap dev/test/RL registry users: {len(overlap)}")

    bench = load_rows(config.BENCH)
    test = [row for row in bench if row["slice"] == "test"]
    sft_positive = [int(row["gold_pids"][0]) for row in rows]

    sft_candidates = {int(pid) for row in rows for pid in row["cand_pids"]}
    sft_positive_set = set(sft_positive)
    test_pool = {int(pid) for row in test for pid in row["pool_pids"]}
    test_anchor = {int(pid) for row in test for pid in row["pool_pids"][:30]}
    test_positive = {int(row["pool_pids"][0]) for row in test}
    item_overlap = {
        "test_pool_items": len(test_pool),
        "test_pool_in_sft_slates": round(len(test_pool & sft_candidates) / len(test_pool), 4),
        "test_anchor_items": len(test_anchor),
        "test_anchor_in_sft_slates": round(len(test_anchor & sft_candidates) / len(test_anchor), 4),
        "test_positives_in_sft_slates": round(
            len(test_positive & sft_candidates) / len(test_positive), 4),
        "test_positives_as_sft_positives": round(
            len(test_positive & sft_positive_set) / len(test_positive), 4),
    }

    audit = {
        "recipe": RECIPE,
        "formal": formal,
        "target_rows": int(args.target_rows) if formal else len(rows),
        "source_pool_rows": len(source_rows),
        "pool_path": os.path.relpath(os.path.realpath(args.pool), config.ROOT)
        if formal else os.path.realpath(args.pool),
        "pool_sha256": pool_sha256,
        "phase_manifest_sha256": phase_manifest_sha256,
        "selected_key_sha256": (phase_manifest["stages"][str(args.target_rows)]["key_sha256"]
                                if formal else None),
        "composition": {
            "rows": len(rows),
            "users": len(set(uids)),
            "rows_per_user": 1.0,
            "positive_grade_distribution": grade_counts,
            "k_distribution": realized_distribution(k_counts, K_GRID, K_WEIGHTS),
            "n_cand_distribution": realized_distribution(
                n_counts, N_CAND_GRID, N_CAND_WEIGHTS),
            "hard_neg_per_slate_mean": round(
                float(np.mean([row["n_hard_neg"] for row in rows])), 2),
            "candidate_pid_duplicates": 0,
            "user_overlap_dev_test_rl": 0,
        },
        "item_overlap": item_overlap,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dump_json(args.out, audit)
    print(f"validated {len(rows):,} rows/users | grades={grade_counts}")
    print(f"k={k_counts}")
    print(f"N={n_counts}")
    print(f"wrote {args.out}")
    if formal and args.target_rows == EXPECTED_ROWS and not args.no_update_stats:
        path = update_stats("sft_v3_audit", audit)
        print(f"wrote [sft_v3_audit] section of {path}")
    else:
        print("canonical benchmark stats left unchanged")


if __name__ == "__main__":
    main()
