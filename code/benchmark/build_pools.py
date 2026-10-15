# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Partition all RecIF users into disjoint, seeded pools (paper Sec. 3, App. A.2).

Pools are mutually disjoint. test and dev hold 400 users each (200 with a grade-3 and 200 with a
grade-2 single positive), restricted to users whose positive and at least MIN_HIST_EVAL history
items are renderable both as a SID and as a non-empty caption, so every k up to 256 is available
in all three renderings. sft and rl are two further seeded partitions of 12,000 users each with at
least one engaged target, from which no training set draws. The rest is reserve: sft/build_pool.py
draws the SFT and post-training pools from reserve users with one positive and at least 256 unified
history items.

Input: data/recif (code/download/recif.py). Output: data/benchmark/pools.parquet (uid, pool,
pos_grade, hist_len, hist_len_uni) and the "pools" section of data/benchmark/stats.json (the
eligibility funnel, pool sizes and history-length distributions). CPU only:

    python code/benchmark/build_pools.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ReRankID/code
from utils import config
config.setup_env()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils.data_io import dist, update_stats
from utils.prompts import is_empty
from utils.raw_data import PID2CAP, PID2SID, RAW, engaged_mask

SEED = 1
DST = config.POOLS

N_TEST_PER_GRADE = 200
N_DEV_PER_GRADE = 200
N_SFT = 12_000
N_RL = 12_000
MIN_HIST_EVAL = 256


def funnel(df, grade):
    """Eligibility funnel of one grade, from all users down to the sampled evaluation slices."""
    g = df.pos_grade == grade
    long_hist = g & (df.hist_len >= MIN_HIST_EVAL)
    renderable = long_hist & df.pos_renderable
    return {"exactly_one_positive": int(g.sum()),
            f"+raw_hist>={MIN_HIST_EVAL}": int(long_hist.sum()),
            "+pos_renderable": int(renderable.sum()),
            f"+unified_hist>={MIN_HIST_EVAL} (eligible)":
                int((renderable & (df.hist_len_uni >= MIN_HIST_EVAL)).sum()),
            "sampled_test": N_TEST_PER_GRADE, "sampled_dev": N_DEV_PER_GRADE}


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    t0 = time.time()
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    pf = pq.ParquetFile(RAW)

    uids, hist_lens, pos_grades, n_engaged = [], [], [], []
    cols = ["uid", "hist_video_pid", "target_video_like", "target_video_follow",
            "target_video_forward", "target_video_longview"]
    for b in pf.iter_batches(batch_size=20_000, columns=cols):
        uu = b.column("uid").to_pylist()
        hl = b.column("hist_video_pid").value_lengths().to_pylist()
        lk = b.column("target_video_like").to_pylist()
        fo = b.column("target_video_follow").to_pylist()
        fw = b.column("target_video_forward").to_pylist()
        lv = b.column("target_video_longview").to_pylist()
        for u, h, l, f, w, v in zip(uu, hl, lk, fo, fw, lv):
            uids.append(int(u))
            hist_lens.append(int(h or 0))
            if not l:
                pos_grades.append(0)
                n_engaged.append(-1)  # no masks
                continue
            eng = engaged_mask(l, f, w)
            ne = sum(eng)
            n_engaged.append(ne)
            if ne == 1:
                pos_grades.append(3)
            elif ne == 0 and v and sum(bool(x) for x in v) == 1:
                pos_grades.append(2)
            else:
                pos_grades.append(0)

    df = pd.DataFrame({"uid": uids, "hist_len": hist_lens,
                       "pos_grade": pos_grades, "n_engaged": n_engaged})
    df = df.sort_values("uid").reset_index(drop=True)  # deterministic base order
    print(f"scanned {len(df):,} users ({time.time()-t0:.0f}s); "
          f"single-pos g3={int((df.pos_grade == 3).sum()):,} g2={int((df.pos_grade == 2).sum()):,}")

    # pass 2: renderability of the positive + unified history length, for the
    # eval candidates only (single-positive users with a long enough raw history).
    cand = set(df.uid[(df.pos_grade.isin([3, 2])) & (df.hist_len >= MIN_HIST_EVAL)].tolist())
    pos_pid, hists, need = {}, {}, set()
    cols2 = ["uid", "hist_video_pid", "target_video_pid"] + \
            ["target_video_like", "target_video_follow", "target_video_forward",
             "target_video_longview"]
    for b in pf.iter_batches(batch_size=20_000, columns=cols2):
        d = b.to_pydict()
        for i, u in enumerate(d["uid"]):
            u = int(u)
            if u not in cand:
                continue
            eng = engaged_mask(d["target_video_like"][i], d["target_video_follow"][i],
                               d["target_video_forward"][i])
            lv = [bool(x) for x in d["target_video_longview"][i]]
            j = eng.index(True) if any(eng) else lv.index(True)
            pos_pid[u] = int(d["target_video_pid"][i][j])
            hists[u] = np.asarray(d["hist_video_pid"][i] or [], dtype=np.int64)
            need.add(pos_pid[u])
            need.update(int(p) for p in hists[u])
    print(f"eval candidates: {len(cand):,} users, {len(need):,} distinct pids "
          f"({time.time()-t0:.0f}s)")

    # renderable = has BOTH a SID and a non-empty caption (restricted to `need`)
    sid_ok = set()
    for b in pq.ParquetFile(PID2SID).iter_batches(batch_size=2_000_000, columns=["pid"]):
        sid_ok.update(p for p in (int(x) for x in b.column("pid").to_pylist()) if p in need)
    cap_ok = set()
    for b in pq.ParquetFile(PID2CAP).iter_batches(batch_size=2_000_000,
                                                  columns=["pid", "dense_caption"]):
        for p, c in zip(b.column("pid").to_pylist(), b.column("dense_caption").to_pylist()):
            p = int(p)
            if p in need and not is_empty(c):
                cap_ok.add(p)
    ok = sid_ok & cap_ok
    print(f"renderable pids: {len(ok):,}/{len(need):,} "
          f"(sid {len(sid_ok):,}, caption {len(cap_ok):,}) ({time.time()-t0:.0f}s)")

    df["pos_renderable"] = df.uid.map(lambda u: pos_pid.get(u, -1) in ok)
    df["hist_len_uni"] = df.uid.map(
        lambda u: sum(int(p) in ok for p in hists[u]) if u in hists else 0)
    n_pos_bad = int((df.uid.isin(cand) & ~df.pos_renderable).sum())
    n_hist_bad = int((df.uid.isin(cand) & df.pos_renderable
                      & (df.hist_len_uni < MIN_HIST_EVAL)).sum())
    print(f"dropped from eval eligibility: {n_pos_bad:,} unrenderable positive, "
          f"{n_hist_bad:,} unified hist < {MIN_HIST_EVAL}")

    rng = np.random.default_rng(SEED)
    pool = pd.Series("reserve", index=df.index)

    # 1) test / dev per grade from the eligible eval users.
    for grade in (3, 2):
        elig = df.index[(df.pos_grade == grade) & df.pos_renderable
                        & (df.hist_len_uni >= MIN_HIST_EVAL)
                        & (pool == "reserve")].to_numpy()
        take = rng.choice(elig, size=N_TEST_PER_GRADE + N_DEV_PER_GRADE, replace=False)
        pool[take[:N_TEST_PER_GRADE]] = "test"
        pool[take[N_TEST_PER_GRADE:]] = "dev"

    # 2) sft / rl: any user with >=1 engaged target, not already assigned.
    elig = df.index[(df.n_engaged >= 1) & (pool == "reserve")].to_numpy()
    take = rng.choice(elig, size=N_SFT + N_RL, replace=False)
    pool[take[:N_SFT]] = "sft"
    pool[take[N_SFT:]] = "rl"

    out = df[["uid", "pos_grade", "hist_len", "hist_len_uni"]].copy()
    out["pool"] = pool
    out.to_parquet(DST, index=False)

    counts = {k: int(v) for k, v in out.pool.value_counts().to_dict().items()}

    with_masks = int((df.n_engaged >= 0).sum())
    eng = df.n_engaged
    stats = {
        "seed": SEED, "min_hist_eval": MIN_HIST_EVAL,
        "funnel": {
            "total_users": int(len(df)),
            "with_engagement_masks": with_masks,
            "engaged_count_split": {"0": int((eng == 0).sum()), "1": int((eng == 1).sum()),
                                    ">=2": int((eng >= 2).sum())},
            "g3_exactly_one_engaged": funnel(df, 3),
            "g2_zero_engaged_exactly_one_longview": funnel(df, 2),
            "sft_rl_supply_ge1_engaged": int((eng >= 1).sum()),
        },
        "sampled_counts": counts,
        "test_by_grade": {str(k): int(v) for k, v in
                          out[out.pool == "test"].pos_grade.value_counts().to_dict().items()},
        "dev_by_grade": {str(k): int(v) for k, v in
                         out[out.pool == "dev"].pos_grade.value_counts().to_dict().items()},
        "hist_len": {
            "all_users_raw": dist(df.hist_len),
            "test_raw": dist(out[out.pool == "test"].hist_len),
            "test_unified": dist(out[out.pool == "test"].hist_len_uni),
            "dev_raw": dist(out[out.pool == "dev"].hist_len),
            "dev_unified": dist(out[out.pool == "dev"].hist_len_uni),
            "sft_raw": dist(out[out.pool == "sft"].hist_len),
            "rl_raw": dist(out[out.pool == "rl"].hist_len),
        },
        "renderability": {"eval_candidates": len(cand),
                          "distinct_pids": len(need),
                          "renderable_pids": len(ok),
                          "dropped_pos_unrenderable": n_pos_bad,
                          "dropped_hist_uni_short": n_hist_bad},
    }
    path = update_stats("pools", stats)
    print(f"wrote {DST} + [pools] section of {path} ({time.time()-t0:.0f}s)\n"
          f"{json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
