# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""SFT teacher settings and the hash-based train/dev split (paper Sec. 5 and App. C.1).

The SFT pool (utils/pools.py) holds 12,000 single-positive slates from 12,000 users, 6,000 per
positive grade.
Dev membership is a pure function of (uid, slate_id, grade) through SHA-256 ranks, so it never
depends on parquet row order: inside each grade the keys are ranked once and cut into two halves
of 3,000, and 8% of each half (by a second rank) goes to dev. This gives 960 dev slates out of
12,000. The hash labels below ("pilot", "dev-pilot", "dev-extension") and RECIPE are part of the
hash input and must stay exactly as written to reproduce the released split.
"""
import hashlib

RECIPE = "sft-v3-single-longhist"      # hash salt of the dev split; keep the literal

# Teacher settings used for the released annotations (Table C.1).
TEACHER_MODEL = "gemini-3.6-flash"
TEACHER_BACKEND = "vertex"
TEACHER_EFFORT = "medium"
TEACHER_THINKING_LEVEL = "medium"
TEACHER_TEMPERATURE = None             # provider default
MAX_TOKENS = 8192
FORMAT_RETRIES = 1

# Dev-split hashing constants.
SPLIT_SEED = 5
HALF_USERS_PER_GRADE = 3_000
DEV_FRAC = 0.08


def row_key(row):
    """(uid, slate_id) join key shared by pool rows and annotations."""
    return int(row["uid"]), str(row["slate_id"])


def _hash(label, grade, key):
    uid, slate_id = key
    payload = f"{RECIPE}|{SPLIT_SEED}|{label}|{grade}|{uid}|{slate_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ranked(keys, label, grade):
    return sorted(keys, key=lambda key: (_hash(label, grade, key), key))


def dev_keys(rows):
    """Set of dev (uid, slate_id) keys for the SFT pool rows (needs uid, slate_id, pos_grade)."""
    by_grade = {}
    for row in rows:
        by_grade.setdefault(int(row["pos_grade"]), []).append(row_key(row))
    dev = set()
    for grade, keys in by_grade.items():
        ranked = _ranked(keys, "pilot", grade)
        halves = {"pilot": ranked[:HALF_USERS_PER_GRADE],
                  "extension": ranked[HALF_USERS_PER_GRADE:]}
        for name, half in halves.items():
            n_dev = int(round(len(half) * DEV_FRAC))
            dev.update(_ranked(half, f"dev-{name}", grade)[:n_dev])
    return dev
