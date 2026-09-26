"""Build the frozen rerank benchmark from the user registry.

This implementation is the executable benchmark specification. It stores one
row per test/dev user. Slate size and shuffle are render-time parameters, like
history length, rather than separately stored rows:

  candidate pool   each row stores 100 candidates in PRIORITY order --
                   [the single positive] + clean exposure-hard negatives (raw exposure order,
                   renderability-filtered) + uniform-random fill from the SID-cap intersection pool.
                   slate(N) = the first N pool items, so slates are nested across
                   N in {10, 20, 30, 50, 100} and share the positive + all hard negatives.
  display shuffle  because the positive is always pool item 0, the pool order must never reach a
                   model: utils.data_io.select_slate applies the deterministic display shuffle
                   seeded by (SEED, uid, n_cand, shuffle_id) at load time and emits the standard
                   cand_* keys.  shuffle_id=0 is the headline protocol for every system (paired
                   comparisons); 1/2 exist for the one-model stability audit.
  hardening        ALL of the user's other positive-signal items (sibling engaged, every longview)
                   are excluded from the pool -- never graded 0; profile_header is always "".
  rendering        every pool item is renderable in ALL modalities (SID + non-empty caption): the
                   positive is guaranteed by pools eligibility (asserted), exposure-hard negatives
                   lacking either are dropped, the random pool is the intersection by construction.

Outputs (data/benchmark/):
  bench.parquet          one row per user (test 400 + dev 400); pool_* columns consumed via
                         utils.data_io.load_bench / select_slate
  stats.json ["bench"]   this builder's section of the single data-generation stats file:
                         config, per-slice composition, pool make-up, history-length stats,
                         random floors per N, sha256

CPU-only. Run after build_pools.py:
  python code/benchmark/build_bench.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ReRankID/code
from utils import config
config.setup_env()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import hashlib
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from benchmark.build_pools import dist
from utils.data_io import SLATE_SEED, update_stats
from utils.metrics import random_ndcg, random_recall
from utils.raw_data import MASKS, RAW, build_neg_pool, engaged_mask, resolve_pids

SEED = SLATE_SEED    # one seed constant: pool fill at build time, display shuffle at render time
POOLS = f"{config.BENCHMARK_DIR}/pools.parquet"
DST = config.BENCH

N_POOL = 100
N_TIERS = (10, 20, 30, 50, 100)   # render-time slate sizes; 30 is the anchor


def main():
    t0 = time.time()
    pools = pd.read_parquet(POOLS)
    sel = pools[pools.pool.isin(["test", "dev"])].set_index("uid")
    print(f"selected users: {sel.pool.value_counts().to_dict()}")

    # raw rows for selected uids
    want = set(sel.index.tolist())
    raw = {}
    cols = ["uid", "target_video_pid", "hist_video_pid"] + MASKS
    for b in pq.ParquetFile(RAW).iter_batches(batch_size=20_000, columns=cols):
        d = b.to_pydict()
        for i, u in enumerate(d["uid"]):
            if int(u) in want:
                raw[int(u)] = {c: d[c][i] for c in cols[1:]}
    print(f"loaded raw rows for {len(raw):,} users ({time.time()-t0:.0f}s)")

    neg_pool = build_neg_pool(np.random.default_rng(SEED + 7))
    pool_pids = np.array(sorted(neg_pool.keys()), dtype=np.int64)

    # resolve rendering for all pool/history pids (random negatives come from neg_pool)
    need = set()
    for u, r in raw.items():
        need.update(int(p) for p in r["target_video_pid"])
        need.update(int(p) for p in (r["hist_video_pid"] or []))
    pid2sid, pid2cap = resolve_pids(need, neg_pool)
    print(f"resolved {len(need):,} pool/hist pids ({time.time()-t0:.0f}s)")

    def rendering(p):
        sid = pid2sid.get(p) or (neg_pool[p][0] if p in neg_pool else "")
        cap = pid2cap.get(p) or (neg_pool[p][1] if p in neg_pool else "")
        return sid, cap

    def renderable(p):
        return bool(pid2sid.get(p)) and bool(pid2cap.get(p))

    rows = []
    hard_dropped = 0

    for uid, meta in sel.iterrows():
        r = raw[uid]
        real_pids = [int(p) for p in r["target_video_pid"]]
        eng = engaged_mask(r["target_video_like"], r["target_video_follow"], r["target_video_forward"])
        lv = [bool(x) for x in r["target_video_longview"]]
        grade = int(meta.pos_grade)
        if grade == 3:
            pos_local = eng.index(True)
            # hardening: extra longview items are neither positives nor negatives
            hard = [p for j, p in enumerate(real_pids) if j != pos_local and not lv[j]]
        else:
            pos_local = [j for j, (e, v) in enumerate(zip(eng, lv)) if v and not e][0]
            hard = [p for j, p in enumerate(real_pids) if j != pos_local]
        pos_pid = real_pids[pos_local]
        assert renderable(pos_pid), f"unrenderable positive {pos_pid} for uid {uid} (stale pools?)"
        n_hard_all = len(hard)
        hard = [p for p in hard if renderable(p)]
        hard_dropped += n_hard_all - len(hard)
        hp = [int(p) for p in (r["hist_video_pid"] or [])]

        # priority pool: positive + hard (raw exposure order) + seeded random fill to N_POOL
        vrng = np.random.default_rng([SEED, uid])
        exclude = set(real_pids) | set(hp)
        rand_negs = []
        n_rand = N_POOL - 1 - len(hard)
        for p in vrng.choice(pool_pids, size=min(len(pool_pids), n_rand * 4 + 50),
                             replace=False).tolist():
            if p not in exclude:
                rand_negs.append(int(p)); exclude.add(int(p))
                if len(rand_negs) == n_rand:
                    break
        pool = [pos_pid] + hard + rand_negs
        assert len(pool) == N_POOL, f"short pool for uid {uid}"
        grades = [grade] + [0] * (N_POOL - 1)
        rend = [rendering(p) for p in pool]
        assert all(s and c for s, c in rend), f"unrenderable pool item for uid {uid}"
        hist_rend = [(pid2sid.get(p, ""), pid2cap.get(p, "")) for p in hp]

        rows.append({
            "uid": uid, "slice": meta.pool, "pos_grade": grade,
            "n_pool": N_POOL, "n_pos": 1,
            "n_hard_neg": len(hard), "n_rand_neg": len(rand_negs),
            "n_hist": len(hp), "profile_header": "",
            "pool_pids": pool, "pool_sids": [x[0] for x in rend],
            "pool_captions": [x[1] for x in rend], "pool_grades": grades,
            "gold_pids": [pos_pid],
            "hist_pids": hp,
            "hist_sids_str": [x[0] for x in hist_rend],
            "hist_caps": [x[1] for x in hist_rend],
        })

    out = pd.DataFrame(rows)
    out.to_parquet(DST, index=False)

    # single-positive floors depend only on N (the gain cancels)
    floors = {str(n): {"ndcg@5": round(float(random_ndcg([1] + [0] * (n - 1), 5)), 4),
                       "recall@5": round(float(random_recall([1] + [0] * (n - 1), 5)), 4)}
              for n in N_TIERS}

    uni_hist = [sum(1 for s, c in zip(ss, cc) if s and c)
                for ss, cc in zip(out.hist_sids_str, out.hist_caps)]
    out["_hist_uni"] = uni_hist
    stats = {
        "seed": SEED, "pools": POOLS, "n_pool": N_POOL, "n_tiers": list(N_TIERS),
        "rows": int(len(out)),
        "by_slice": {k: int(v) for k, v in out.slice.value_counts().to_dict().items()},
        "by_slice_grade": {sl: {str(g): int(n) for g, n in
                                out[out.slice == sl].pos_grade.value_counts().to_dict().items()}
                           for sl in ("test", "dev")},
        "pool_composition": {
            "n_pos_per_user": 1,
            "hard_neg_per_pool": dist(out.n_hard_neg),
            "rand_neg_per_pool": dist(out.n_rand_neg),
            "hard_dropped_unrenderable": int(hard_dropped),
        },
        "hist_len": {"raw": dist(out.n_hist), "unified": dist(out._hist_uni)},
        "random_floors": floors,
        "sha256": hashlib.sha256(open(DST, "rb").read()).hexdigest(),
    }
    path = update_stats("bench", stats)
    print(f"\nwrote {DST} rows={len(out)} + [bench] section of {path} ({time.time()-t0:.0f}s)")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
