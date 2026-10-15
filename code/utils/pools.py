# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Recipe of the two 12k training pools and the shared post-training subset (paper Sec. 5).

sft/build_pool.py draws the SFT pool and the post-training pool with the history and slate grids
below, 6,000 users per positive grade each. RFT, OPD and RLVR all train on the same deterministic
6,000-prompt subset of the post-training pool: within each positive grade, slates are ranked by a
SHA-256 of (recipe, seed, label, grade, uid, slate) and the first 3,000 are kept, so the subset
does not depend on row order. POSTTRAIN_RECIPE, SUBSET_SEED and the "pilot" label are hash inputs
and must stay exactly as written to reproduce the released data.
"""
import hashlib

USERS_PER_GRADE = 6_000                         # users per positive grade (2 and 3) in each pool
K_GRID = [32, 64, 128, 256]                     # history lengths k and their sampling weights
K_WEIGHTS = [0.10, 0.30, 0.30, 0.30]
N_CAND_GRID = [10, 20, 30, 50, 100]             # slate sizes N and their sampling weights
N_CAND_WEIGHTS = [0.10, 0.15, 0.40, 0.20, 0.15]

POSTTRAIN_RECIPE = "posttrain-v1-balanced-longhist"
SUBSET_SEED = 19
PARENT_ROWS = 2 * USERS_PER_GRADE
SUBSET_USERS_PER_GRADE = 3_000


def row_key(row):
    """(uid, slate_id) of a post-training row."""
    slate_id = str(row.get("slate_id", ""))
    if not slate_id:
        raise ValueError("post-training row has no slate_id")
    return int(row["uid"]), slate_id


def _rank_hash(grade, key):
    uid, slate = key
    text = f"{POSTTRAIN_RECIPE}|{SUBSET_SEED}|pilot|{grade}|{uid}|{slate}"
    return hashlib.sha256(text.encode()).hexdigest()


def validate_parent_rows(rows):
    """Check the 12k pool invariants and return its (uid, slate_id) keys."""
    if len(rows) != PARENT_ROWS:
        raise ValueError(f"post-training parent must have {PARENT_ROWS} rows, got {len(rows)}")
    keys = [row_key(row) for row in rows]
    uids = [int(row["uid"]) for row in rows]
    if len(keys) != len(set(keys)) or len(uids) != len(set(uids)):
        raise ValueError("post-training parent must have one unique slate per unique user")
    grades = {2: 0, 3: 0}
    for row in rows:
        if str(row.get("pool")) != "posttrain":
            raise ValueError(f"unexpected pool label for {row_key(row)}")
        grade = int(row["pos_grade"])
        if grade not in grades:
            raise ValueError(f"unsupported positive grade {grade}")
        grades[grade] += 1
        if int(row["k"]) not in K_GRID or int(row["n_cand"]) not in N_CAND_GRID:
            raise ValueError(f"unsupported k/N for {row_key(row)}")
        candidate_grades = [int(x) for x in row["cand_grades"]]
        if len(candidate_grades) != int(row["n_cand"]) or sum(x > 0 for x in candidate_grades) != 1:
            raise ValueError(f"invalid single-positive slate for {row_key(row)}")
    expected = {2: USERS_PER_GRADE, 3: USERS_PER_GRADE}
    if grades != expected:
        raise ValueError(f"post-training parent must be grade-balanced: {grades} != {expected}")
    return keys


def subset_keys(rows):
    """Keys of the 6k subset: the 3,000 lowest rank hashes within each positive grade."""
    validate_parent_rows(rows)
    by_grade = {2: [], 3: []}
    for row in rows:
        by_grade[int(row["pos_grade"])].append(row_key(row))
    subset = set()
    for grade in (2, 3):
        ranked = sorted(by_grade[grade], key=lambda key: (_rank_hash(grade, key), key))
        subset.update(ranked[:SUBSET_USERS_PER_GRADE])
    return subset


def select_subset(rows):
    """The 6k subset rows of the 12k pool, sorted by (uid, slate_id)."""
    keys = subset_keys(rows)
    by_key = {row_key(row): row for row in rows}
    return [by_key[key] for key in sorted(keys)]
