"""Deterministic two-stage plan for the frozen SFT-v3 parent pool.

The 6k pilot is a grade-balanced subset of one 12k parent pool. Selection and
train/dev membership depend only on frozen identifiers and versioned hashes,
never parquet order. The resulting manifest is consumed by annotation, dataset
assembly, and audit code so both stages share exactly one experimental plan.
"""

import hashlib
import json

from sft.v3_contract import (DEV_FRAC, EXPECTED_ROWS, FORMAL_TARGET_ROWS, K_GRID,
                             N_CAND_GRID,
                             PHASE_MANIFEST_SCHEMA, PHASE_SEED,
                             PHASE_SELECTION_VERSION, PHASE_SPLIT_VERSION,
                             PILOT_ROWS, PILOT_USERS_PER_GRADE, RECIPE,
                             USERS_PER_GRADE)


def row_key(row):
    """Return the canonical join key for one parent-pool or annotation row."""
    if "slate_id" not in row or row["slate_id"] is None or str(row["slate_id"]) == "":
        raise ValueError("SFT v3 row is missing a nonempty slate_id")
    return int(row["uid"]), str(row["slate_id"])


def _hash(label, grade, key):
    uid, slate_id = key
    payload = f"{RECIPE}|{PHASE_SEED}|{label}|{grade}|{uid}|{slate_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _serialized_keys(keys):
    return [[int(uid), str(slate_id)] for uid, slate_id in sorted(keys)]


def key_digest(keys):
    payload = json.dumps(
        _serialized_keys(keys), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_parent_rows(rows):
    """Fail closed unless rows are the complete balanced 12k mother pool."""
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"SFT v3 parent pool must have {EXPECTED_ROWS} rows, got {len(rows)}")
    keys = [row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("SFT v3 parent pool has duplicate (uid, slate_id) keys")
    uids = [int(row["uid"]) for row in rows]
    if len(uids) != len(set(uids)):
        raise ValueError("SFT v3 parent pool must contain one slate per user")
    grades = {2: 0, 3: 0}
    for row in rows:
        if str(row.get("pool")) != "sft":
            raise ValueError(f"SFT v3 parent row is not in pool=sft: {row_key(row)}")
        grade = int(row["pos_grade"])
        if grade not in grades:
            raise ValueError(f"unsupported positive grade in SFT v3 parent pool: {grade}")
        grades[grade] += 1
        if int(row["k"]) not in K_GRID or int(row["n_cand"]) not in N_CAND_GRID:
            raise ValueError(f"unsupported k or N in SFT v3 parent pool: {row_key(row)}")
    expected = {2: USERS_PER_GRADE, 3: USERS_PER_GRADE}
    if grades != expected:
        raise ValueError(f"SFT v3 parent pool must be grade-balanced: {grades} != {expected}")
    return keys


def _pilot_key_set(rows):
    by_grade = {2: [], 3: []}
    for row in rows:
        grade = int(row["pos_grade"])
        by_grade[grade].append(row_key(row))
    selected = set()
    for grade in (2, 3):
        ranked = sorted(by_grade[grade], key=lambda key: (_hash("pilot", grade, key), key))
        selected.update(ranked[:PILOT_USERS_PER_GRADE])
    return selected


def _dev_key_set(rows, pilot_keys):
    """Freeze exact 8% dev membership in each grade and stage partition.

    Splitting the pilot and its complement separately makes the 6k pilot exactly
    8% dev while also making the full 12k exactly 8% dev. Therefore pilot train
    and dev are strict subsets of their corresponding full-stage splits.
    """
    groups = {(grade, partition): [] for grade in (2, 3) for partition in ("pilot", "extension")}
    for row in rows:
        key = row_key(row)
        grade = int(row["pos_grade"])
        partition = "pilot" if key in pilot_keys else "extension"
        groups[(grade, partition)].append(key)
    dev = set()
    for (grade, partition), keys in groups.items():
        n_dev = int(round(len(keys) * DEV_FRAC))
        ranked = sorted(
            keys, key=lambda key: (_hash(f"dev-{partition}", grade, key), key)
        )
        dev.update(ranked[:n_dev])
    return dev


def build_phase_manifest(rows, pool_sha256):
    """Build a deterministic manifest for the complete parent pool."""
    full_keys = set(validate_parent_rows(rows))
    pilot_keys = _pilot_key_set(rows)
    if len(pilot_keys) != PILOT_ROWS:
        raise AssertionError(f"pilot selection produced {len(pilot_keys)} keys")
    dev_keys = _dev_key_set(rows, pilot_keys)

    row_by_key = {row_key(row): row for row in rows}

    def distribution(keys, field):
        counts = {}
        for key in keys:
            value = str(int(row_by_key[key][field]))
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: int(item[0])))

    stages = {}
    for target, keys in ((PILOT_ROWS, pilot_keys), (EXPECTED_ROWS, full_keys)):
        stage_dev = keys & dev_keys
        stage_train = keys - dev_keys
        stages[str(target)] = {
            "rows": len(keys),
            "key_sha256": key_digest(keys),
            "positive_grade_distribution": distribution(keys, "pos_grade"),
            "k_distribution": distribution(keys, "k"),
            "n_cand_distribution": distribution(keys, "n_cand"),
            "train_rows": len(stage_train),
            "dev_rows": len(stage_dev),
            "train_key_sha256": key_digest(stage_train),
            "dev_key_sha256": key_digest(stage_dev),
            "dev_positive_grade_distribution": distribution(stage_dev, "pos_grade"),
        }
    return {
        "schema_version": PHASE_MANIFEST_SCHEMA,
        "recipe": RECIPE,
        "pool_sha256": str(pool_sha256),
        "parent_rows": EXPECTED_ROWS,
        "selection": {
            "version": PHASE_SELECTION_VERSION,
            "seed": PHASE_SEED,
            "pilot_rows": PILOT_ROWS,
            "pilot_users_per_grade": PILOT_USERS_PER_GRADE,
        },
        "split": {
            "version": PHASE_SPLIT_VERSION,
            "dev_fraction": DEV_FRAC,
            "dev_rows": len(dev_keys),
            "dev_key_sha256": key_digest(dev_keys),
        },
        "full_keys": _serialized_keys(full_keys),
        "pilot_keys": _serialized_keys(pilot_keys),
        "dev_keys": _serialized_keys(dev_keys),
        "stages": stages,
    }


def validate_phase_manifest(manifest, rows, pool_sha256):
    """Return the manifest after exact deterministic validation."""
    expected = build_phase_manifest(rows, pool_sha256)
    if manifest != expected:
        differing = sorted(
            key for key in set(manifest) | set(expected)
            if manifest.get(key) != expected.get(key)
        )
        raise ValueError(f"SFT v3 phase manifest differs from frozen parent plan: {differing}")
    return manifest


def manifest_key_set(manifest, target_rows):
    if int(target_rows) not in FORMAL_TARGET_ROWS:
        raise ValueError(
            f"target_rows must be one of {FORMAL_TARGET_ROWS}, got {target_rows}"
        )
    field = "pilot_keys" if int(target_rows) == PILOT_ROWS else "full_keys"
    serialized = manifest.get(field) or []
    keys = {(int(uid), str(slate_id)) for uid, slate_id in serialized}
    if len(keys) != len(serialized) or len(keys) != int(target_rows):
        raise ValueError(f"phase manifest {field} does not contain {target_rows} unique keys")
    stage = (manifest.get("stages") or {}).get(str(target_rows)) or {}
    if stage.get("rows") != int(target_rows) or stage.get("key_sha256") != key_digest(keys):
        raise ValueError(f"phase manifest stage {target_rows} key digest/cardinality mismatch")
    return keys


def select_stage_rows(rows, target_rows, manifest=None, pool_sha256=None):
    """Select one formal stage in canonical key order, independent of input order."""
    validate_parent_rows(rows)
    if manifest is None:
        if pool_sha256 is None:
            raise ValueError("pool_sha256 is required when constructing the phase plan")
        manifest = build_phase_manifest(rows, pool_sha256)
    else:
        if pool_sha256 is None:
            raise ValueError("pool_sha256 is required when validating the phase manifest")
        validate_phase_manifest(manifest, rows, pool_sha256)
    selected = manifest_key_set(manifest, target_rows)
    by_key = {row_key(row): row for row in rows}
    if not selected <= set(by_key):
        raise ValueError("phase manifest contains keys outside the parent pool")
    return [by_key[key] for key in sorted(selected)]


def manifest_split_sets(manifest, target_rows):
    """Return validated train/dev key sets for one formal stage."""
    stage_keys = manifest_key_set(manifest, target_rows)
    serialized_dev = manifest.get("dev_keys") or []
    all_dev = {(int(uid), str(slate_id)) for uid, slate_id in serialized_dev}
    full_keys = manifest_key_set(manifest, EXPECTED_ROWS)
    if len(all_dev) != len(serialized_dev) or not all_dev <= full_keys:
        raise ValueError("phase manifest contains duplicate or unknown dev keys")
    dev = stage_keys & all_dev
    train = stage_keys - dev
    stage = manifest["stages"][str(target_rows)]
    if (stage.get("train_rows") != len(train) or stage.get("dev_rows") != len(dev) or
            stage.get("train_key_sha256") != key_digest(train) or
            stage.get("dev_key_sha256") != key_digest(dev)):
        raise ValueError(f"phase manifest stage {target_rows} split mismatch")
    return train, dev


def split_for_key(manifest, target_rows, key):
    train, dev = manifest_split_sets(manifest, target_rows)
    normalized = int(key[0]), str(key[1])
    if normalized not in train | dev:
        raise ValueError(f"key is outside phase {target_rows}: {normalized}")
    return "dev" if normalized in dev else "train"
