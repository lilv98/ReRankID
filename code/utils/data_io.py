# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark and pool parquet to Python row dicts, nested-slate selection, JSON and stats output.
Array-valued (Arrow list) columns become plain Python lists (paper Sec. 3)."""
import json
import os

import numpy as np
import pyarrow.parquet as pq


def load_rows(path, n=None, where=None):
    """Read a parquet into a list of row dicts: the first n rows matching `where`, if given.

    Reads batch-wise with pyarrow, because pandas cannot convert chunked nested list columns of the
    large training pools, and stops as soon as n rows are collected."""
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


SLATE_SEED = 1   # equals benchmark/build_bench.py SEED


def select_slate(row, n_cand, shuffle_id=0):
    """Select and shuffle the first n_cand items of a user's 100-item candidate pool.

    The pool is stored in priority order (positive, exposure-based hard negatives, random fill), so
    slates are nested: slate(10) is a subset of slate(20), and so on up to slate(100). Because the
    positive is always pool item 0, the slate is shown in a display order seeded by (SLATE_SEED,
    uid, n_cand, shuffle_id), identical for every system so comparisons stay paired. shuffle_id=0 is
    the reported protocol; other ids serve the display-order audit. Emits the cand_* keys.
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


def load_bench(path, n=None, n_cand=30, shuffle=0, bench_slice="test"):
    """load_rows, slice filter and select_slate. n_cand=30, shuffle=0 is the anchor.

    bench.parquet holds both evaluation slices (400 test, 400 dev): bench_slice is 'test' (default),
    'dev' or 'all'. `n` counts rows of the selected slice."""
    if bench_slice not in (None, "all"):
        where = lambda r: r.get("slice", bench_slice) == bench_slice  # noqa: E731
    else:
        where = None
    return [select_slate(r, n_cand, shuffle) for r in load_rows(path, n=n, where=where)]


def dist(values):
    """Compact distribution summary for stats.json."""
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": round(float(a.mean()), 1),
            "median": float(np.median(a)), "min": int(a.min()),
            "p1": float(np.percentile(a, 1)), "p99": float(np.percentile(a, 99)),
            "max": int(a.max())}


def dump_json(path, obj, newline=True):
    """Write `obj` as indented UTF-8 JSON. Stats and results end with a newline; the LLaMA-Factory
    training files are written with newline=False, as released."""
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        if newline:
            f.write("\n")


def update_stats(section, payload):
    """Write one benchmark builder's section ("pools" or "bench") into data/benchmark/stats.json,
    keeping the other. Returns the stats path."""
    from utils import config
    path = config.BENCH_STATS
    stats = {}
    if os.path.exists(path):
        with open(path) as f:
            stats = json.load(f)
    stats[section] = payload
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump_json(path, stats)
    return path
