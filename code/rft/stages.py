"""Order-independent 6k pilot selection inside the frozen 12k post-training pool."""

import hashlib
import json

from rft.contract import (K_GRID, N_CAND_GRID, PARENT_ROWS, POOL_TARGET_ROWS,
                          PHASE_SCHEMA, PHASE_SEED, PHASE_SELECTION_VERSION,
                          PILOT_ROWS, PILOT_USERS_PER_GRADE, POSTTRAIN_RECIPE,
                          USERS_PER_GRADE)


def row_key(row):
    slate_id = str(row.get("slate_id", ""))
    if not slate_id:
        raise ValueError("post-training row has no slate_id")
    return int(row["uid"]), slate_id


def _serialized(keys):
    return [[int(uid), str(slate)] for uid, slate in sorted(keys)]


def key_digest(keys):
    payload = json.dumps(_serialized(keys), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def uid_digest(uids):
    payload = "\n".join(str(int(uid)) for uid in sorted(set(uids))) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def _rank_hash(grade, key):
    uid, slate = key
    text = f"{POSTTRAIN_RECIPE}|{PHASE_SEED}|pilot|{grade}|{uid}|{slate}"
    return hashlib.sha256(text.encode()).hexdigest()


def validate_parent_rows(rows):
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


def build_phase_manifest(rows, pool_sha256, excluded_sft_uid_sha256):
    full = set(validate_parent_rows(rows))
    by_grade = {2: [], 3: []}
    row_by_key = {row_key(row): row for row in rows}
    for key, row in row_by_key.items():
        by_grade[int(row["pos_grade"])].append(key)
    pilot = set()
    for grade in (2, 3):
        ranked = sorted(by_grade[grade], key=lambda key: (_rank_hash(grade, key), key))
        pilot.update(ranked[:PILOT_USERS_PER_GRADE])

    def distribution(keys, field):
        out = {}
        for key in keys:
            value = str(int(row_by_key[key][field]))
            out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items(), key=lambda item: int(item[0])))

    stages = {}
    for target, keys in ((PILOT_ROWS, pilot), (PARENT_ROWS, full)):
        stages[str(target)] = {
            "rows": len(keys),
            "key_sha256": key_digest(keys),
            "positive_grade_distribution": distribution(keys, "pos_grade"),
            "k_distribution": distribution(keys, "k"),
            "n_cand_distribution": distribution(keys, "n_cand"),
        }
    return {
        "schema_version": PHASE_SCHEMA,
        "recipe": POSTTRAIN_RECIPE,
        "pool_sha256": str(pool_sha256),
        "excluded_sft_uid_sha256": str(excluded_sft_uid_sha256),
        "parent_rows": PARENT_ROWS,
        "selection": {
            "version": PHASE_SELECTION_VERSION,
            "seed": PHASE_SEED,
            "pilot_rows": PILOT_ROWS,
            "pilot_users_per_grade": PILOT_USERS_PER_GRADE,
        },
        "full_keys": _serialized(full),
        "pilot_keys": _serialized(pilot),
        "stages": stages,
    }


def manifest_key_set(manifest, target_rows):
    target_rows = int(target_rows)
    if target_rows not in POOL_TARGET_ROWS:
        raise ValueError(f"target_rows must be one of {POOL_TARGET_ROWS}")
    field = "pilot_keys" if target_rows == PILOT_ROWS else "full_keys"
    serialized = manifest.get(field) or []
    keys = {(int(uid), str(slate)) for uid, slate in serialized}
    stage = (manifest.get("stages") or {}).get(str(target_rows)) or {}
    if (len(keys) != len(serialized) or len(keys) != target_rows or
            stage.get("rows") != target_rows or stage.get("key_sha256") != key_digest(keys)):
        raise ValueError(f"invalid {target_rows}-prompt stage in manifest")
    return keys


def validate_phase_manifest(manifest, rows, pool_sha256, excluded_sft_uid_sha256):
    expected = build_phase_manifest(rows, pool_sha256, excluded_sft_uid_sha256)
    if manifest != expected:
        differing = sorted(k for k in set(manifest) | set(expected)
                           if manifest.get(k) != expected.get(k))
        raise ValueError(f"post-training phase manifest differs: {differing}")
    return manifest


def select_stage_rows(rows, manifest, target_rows):
    keys = manifest_key_set(manifest, target_rows)
    by_key = {row_key(row): row for row in rows}
    if not keys <= set(by_key):
        raise ValueError("manifest contains keys outside parent pool")
    return [by_key[key] for key in sorted(keys)]
