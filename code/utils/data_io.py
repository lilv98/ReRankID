"""Bench / pool parquet -> python row dicts, and json output. Every array-valued (arrow list) column
is converted to a plain python list automatically, so callers never maintain their own list_cols
tuples (the old per-script `for c in (...): r[c] = list(r[c])` pattern)."""
import json
import os

import numpy as np
import pyarrow.parquet as pq


def load_rows(path, n=None, where=None):
    """Read a parquet -> list of row dicts; first n (matching `where`, if given); numpy-array cells
    -> python lists.

    Batch-wise via pyarrow ON PURPOSE: pd.read_parquet on a multi-row-group file with list columns
    (the 1.3GB training pools) hits Arrow's "Nested data conversions not implemented for chunked
    array outputs" and hard-aborts the process; per-batch conversion is single-chunk so it never
    can, and honoring `n` inside the loop stops reading instead of scanning the whole file."""
    rows = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
        for r in batch.to_pandas().to_dict("records"):
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    r[k] = list(v)
            if where and not where(r):
                continue
            rows.append(r)
            if n and len(rows) >= n:
                return rows
    return rows


SLATE_SEED = 1   # must match benchmark/build_bench.py's SEED


def select_slate(row, n_cand, shuffle_id=0):
    """Render-time slate selection from the stored candidate pool (the N axis analog of last-k
    history truncation).

    The bench stores per user ONE candidate pool of 100 items in PRIORITY order -- [positive] +
    exposure-hard + random fill -- so slates are nested: slate(N) = the first N pool items, and
    slate(10) subset of slate(20) subset of ... subset of slate(100).  Because the positive is always
    pool item 0, the pool order must NEVER be shown to a model; this function applies the
    deterministic display shuffle seeded by (SLATE_SEED, uid, n_cand, shuffle_id) -- identical for
    every system, so all comparisons stay paired -- and emits the standard cand_* keys the scorers
    consume.  shuffle_id=0 is the headline protocol; 1/2 exist for the stability audit.
    """
    total = len(row["pool_pids"])
    if not 1 <= n_cand <= total:
        raise ValueError(f"n_cand={n_cand} outside [1, {total}]")
    perm = np.random.default_rng(
        [SLATE_SEED, int(row["uid"]), int(n_cand), int(shuffle_id)]).permutation(n_cand)
    out = dict(row)
    for src, dst in (("pool_pids", "cand_pids"), ("pool_sids", "cand_sids"),
                     ("pool_captions", "cand_captions"), ("pool_grades", "cand_grades")):
        prefix = list(row[src])[:n_cand]
        out[dst] = [prefix[j] for j in perm]
    out["n_cand"] = int(n_cand)
    out["shuffle_id"] = int(shuffle_id)
    return out


CORRUPT_SEED = 4   # distinct from SLATE_SEED(1), build_pool(2), format-only targets(3)


def corrupt_sids(row):
    """C4 attribution control (corrupted-SID eval): deterministically DERANGE the slate's candidate
    SIDs so every candidate displays another slate member's SID — the SID multiset, the captions, the
    grades, and the history all stay intact, but the SID<->item correspondence is destroyed. A model
    that reads SID semantics collapses toward random (it now ranks the wrong indices); a model that
    ignores SIDs is unaffected. The derangement is one seeded n-cycle (walk the cycle of a seeded
    permutation), so no candidate ever keeps its own SID. Apply AFTER select_slate."""
    n = len(row["cand_sids"])
    perm = np.random.default_rng(
        [CORRUPT_SEED, int(row["uid"]), int(row["n_cand"]), int(row["shuffle_id"])]).permutation(n)
    sids = list(row["cand_sids"])
    out = dict(row)
    out["cand_sids"] = [None] * n
    for j in range(n):
        out["cand_sids"][perm[j]] = sids[perm[(j + 1) % n]]
    out["sid_corrupted"] = True
    return out


def load_bench(path, n=None, no_profile=False, n_cand=30, shuffle=0, corrupt_sid=False,
               bench_slice="test"):
    """load_rows + slice filter + render-time slate selection (select_slate) + optionally blank
    profile_header (the --no-profile / history-only protocol) + optionally the C4 corrupted-SID
    control. n_cand=30 / shuffle=0 is the anchor protocol.

    bench.parquet holds BOTH evaluation slices (test 400 + dev 400) interleaved by uid, so every
    consumer must pick one: bench_slice='test' (the headline default) / 'dev' / 'all'. `n` counts
    rows of the SELECTED slice."""
    if bench_slice not in (None, "all"):
        where = lambda r: r.get("slice", bench_slice) == bench_slice  # noqa: E731
    else:
        where = None
    rows = [select_slate(r, n_cand, shuffle) for r in load_rows(path, n=n, where=where)]
    if corrupt_sid:
        rows = [corrupt_sids(r) for r in rows]
    if no_profile:
        for r in rows:
            r["profile_header"] = ""
    return rows


def rows_by_uid(path):
    """Pool parquet -> {uid: row dict}. ONLY safe for one-row-per-user files (the eval bench). v1
    TRAINING pools carry multiple slates per user (per-engaged expansion) — joining those on uid
    alone silently pairs an annotation with the wrong slate; use rows_by_slate instead."""
    return {int(r["uid"]): r for r in load_rows(path)}


def rows_by_slate(path):
    """Training-pool parquet -> {(uid, slate_id): row dict} — the join key for annotation/rollout
    reassembly on v1 multi-slate pools. slate_id is an OPAQUE STRING ("uid_pospid" on v1 pools;
    rows without a slate_id column get "0"), so both sides of every join must stringify it."""
    return {(int(r["uid"]), str(r.get("slate_id", 0))): r for r in load_rows(path)}


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def update_stats(section, payload, path=None):
    """Merge one builder's stats into THE single data-generation stats file
    (data/benchmark/stats.json). Each builder owns one top-level section ("pools", "bench",
    "sft_v3_pool", "posttrain_v1_pool") and overwrites only that section on rebuild — no per-artifact
    manifests. Returns the stats path."""
    from utils import config
    path = path or config.STATS
    stats = json.load(open(path)) if os.path.exists(path) else {}
    stats[section] = payload
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump_json(path, stats)
    return path
