# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the SFT pool and the shared post-training pool from RecIF (paper Sec. 5, App. C.1).

Each pool has 12,000 users drawn from the reserve partition of data/benchmark/pools.parquet,
6,000 with a grade-3 positive (exactly one engaged target) and 6,000 with a grade-2 positive (no
engaged and exactly one long-view target), all with at least 256 unified history items. Every user
contributes one slate that mirrors the evaluation task: the one positive, the user's renderable
exposure-hard negatives, and uniform random catalog items, nested to a sampled slate size
N in {10, 20, 30, 50, 100}, with a sampled history length k in {32, 64, 128, 256}. Every candidate
has a SID and a caption.

--pool sft writes data/sft/pool.parquet (input of annotate.py and build_dataset.py).
--pool posttrain writes data/posttrain/pool.parquet, the prompt pool shared by RFT, OPD and RLVR;
it also excludes every SFT user, so build the SFT pool first. Both pools are disjoint from the
test, dev and rl partitions. Construction statistics go to data/sft/pool_stats.json or
data/posttrain/stats.json. CPU only.

    python code/sft/build_pool.py --pool sft
    python code/sft/build_pool.py --pool posttrain
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ReRankID/code
from utils import config
config.setup_env()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU-only builder

import argparse
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils.pools import K_GRID, K_WEIGHTS, N_CAND_GRID, N_CAND_WEIGHTS, USERS_PER_GRADE
from utils.data_io import dist, dump_json
from utils.metrics import random_ndcg
from utils.prompts import TOP_K
from utils.raw_data import MASKS, RAW, build_neg_pool, engaged_mask, resolve_pids

OUT = {"sft": config.SFT_POOL, "posttrain": config.POSTTRAIN_POOL}
STATS = {"sft": config.SFT_POOL_STATS, "posttrain": config.POSTTRAIN_STATS}
SEED = {"sft": 2, "posttrain": 17}       # distinct from the benchmark builders' seed 1
RNG_STREAM = {"sft": 0, "posttrain": 2}
MIN_HIST = 256                           # every sampled k is fully populated


def unique_pids(items, exclude=()):
    """Integer PIDs in first-seen order, without repeats and without `exclude`."""
    seen = {int(p) for p in exclude}
    out = []
    for value in items:
        pid = int(value)
        if pid not in seen:
            out.append(pid)
            seen.add(pid)
    return out


def renderable(pid, pid2sid, pid2cap):
    """True when the item has both a SID and a caption."""
    return bool(pid2sid.get(pid)) and bool(pid2cap.get(pid))


def counts(column):
    """Value counts of a pandas column as {str(value): count}, sorted by value."""
    return {str(k): int(v) for k, v in column.value_counts().sort_index().items()}


def validate_candidate_slate(cand_pids, cand_grades, gold_pid):
    """Fail fast unless the slate has unique PIDs and exactly one positive, the gold PID."""
    if len(cand_pids) != len(cand_grades):
        raise AssertionError("candidate PID and grade lengths differ")
    if len(cand_pids) != len(set(cand_pids)):
        raise AssertionError("duplicate candidate PID")
    if list(cand_pids).count(int(gold_pid)) != 1:
        raise AssertionError("gold PID must occur exactly once")
    positives = [i for i, grade in enumerate(cand_grades) if int(grade) > 0]
    if len(positives) != 1 or int(cand_pids[positives[0]]) != int(gold_pid):
        raise AssertionError("candidate grades do not mark exactly the gold PID")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool", choices=["sft", "posttrain"], required=True)
    args = ap.parse_args()
    dst, pool_seed, users_per_grade = OUT[args.pool], SEED[args.pool], USERS_PER_GRADE
    t0 = time.time()
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    pools = pd.read_parquet(config.POOLS)
    forbidden_pools = ["test", "dev", "rl"]
    forbidden_uids = set(pools.uid[pools.pool.isin(forbidden_pools)].tolist())
    rng = np.random.default_rng([pool_seed, RNG_STREAM[args.pool]])
    selected_grade = {}
    excluded_sft_uids = set()
    if args.pool == "posttrain":
        if not os.path.exists(config.SFT_POOL):
            raise SystemExit(f"missing {config.SFT_POOL}; run build_pool.py --pool sft first")
        excluded_sft_uids = set(pd.read_parquet(config.SFT_POOL, columns=["uid"]).uid.astype(int))
        forbidden_pools.append("sft pool")
        forbidden_uids.update(excluded_sft_uids)

    # Candidate users: a 10% margin over the target covers unrenderable positives.
    take = []
    for grade in (3, 2):
        members_g = sorted(pools.uid[(pools.pool == "reserve")
                                     & (pools.pos_grade == grade)
                                     & (pools.hist_len_uni >= MIN_HIST)
                                     & (~pools.uid.isin(excluded_sft_uids))].tolist())
        margin = max(20, (users_per_grade + 9) // 10)
        n_candidate = min(len(members_g), users_per_grade + margin)
        if n_candidate < users_per_grade:
            raise SystemExit(f"not enough reserve g{grade} users: need {users_per_grade}, "
                             f"found {len(members_g)}")
        chosen = rng.choice(members_g, size=n_candidate, replace=False).tolist()
        take.extend(chosen)
        selected_grade.update({int(u): grade for u in chosen})
    members = int(sum((pools.pool == "reserve") & pools.pos_grade.isin([2, 3])
                      & (pools.hist_len_uni >= MIN_HIST)
                      & (~pools.uid.isin(excluded_sft_uids))))
    print(f"pool={args.pool}: {members:,} eligible reserve users; candidate draw={len(take):,} "
          f"for {users_per_grade:,} g3 + {users_per_grade:,} g2")
    want = set(take)

    raw = {}
    cols = ["uid", "target_video_pid", "hist_video_pid"] + MASKS
    for b in pq.ParquetFile(RAW).iter_batches(batch_size=20_000, columns=cols):
        d = b.to_pydict()
        for i, u in enumerate(d["uid"]):
            if int(u) in want:
                raw[int(u)] = {c: d[c][i] for c in cols[1:]}
    print(f"loaded raw rows for {len(raw):,} users ({time.time()-t0:.0f}s)")
    missing_raw = want - set(raw)
    if missing_raw:
        raise RuntimeError(f"raw dataset is missing {len(missing_raw)} selected users")

    neg_pool = build_neg_pool(np.random.default_rng(pool_seed + 7))
    pool_pids = np.array(sorted(neg_pool.keys()), dtype=np.int64)

    need = set()
    for r in raw.values():
        need.update(int(p) for p in r["target_video_pid"])
        need.update(int(p) for p in (r["hist_video_pid"] or []))
    pid2sid, pid2cap = resolve_pids(need, neg_pool)
    print(f"resolved {len(need):,} slate/hist pids ({time.time()-t0:.0f}s)")

    rows, floors = [], []
    n_pos_skipped = 0     # selected positives dropped because unrenderable
    hard_dropped = 0      # exposure-hard negatives dropped because unrenderable
    hard_deduped = 0      # repeated or positive-conflicting exposure rows removed
    produced_grade = {2: 0, 3: 0}
    for uid in take:
        if produced_grade[selected_grade[uid]] >= users_per_grade:
            continue
        r = raw[uid]
        real_pids = [int(p) for p in r["target_video_pid"]]
        eng = engaged_mask(r["target_video_like"], r["target_video_follow"],
                           r["target_video_forward"])
        n_engaged_actual = int(sum(bool(x) for x in eng))
        lv = [bool(x) for x in r["target_video_longview"]]
        hp = [int(p) for p in (r["hist_video_pid"] or [])]

        if selected_grade[uid] == 2:
            pos_grade = 2     # zero engaged and exactly one long-view target
            pos_all = [p for p, e, v in zip(real_pids, eng, lv) if v and not e]
        else:
            pos_grade = 3     # exactly one engaged target
            pos_all = [p for p, e in zip(real_pids, eng) if e]
        pos_all = unique_pids(pos_all)
        if len(pos_all) != 1:
            raise AssertionError(f"g{pos_grade} user {uid} has {len(pos_all)} positives")
        pos_ok = [p for p in pos_all if renderable(p, pid2sid, pid2cap)]
        n_pos_skipped += len(pos_all) - len(pos_ok)

        hard_all = [p for p, e, v in zip(real_pids, eng, lv) if not e and not v]
        hard_renderable = [p for p in hard_all if renderable(p, pid2sid, pid2cap)]
        hard = unique_pids(hard_renderable, exclude=pos_all)
        hard_dropped += len(hard_all) - len(hard_renderable)
        hard_deduped += len(hard_renderable) - len(hard)

        hist_sids = [pid2sid.get(p, "") for p in hp]
        hist_caps = [pid2cap.get(p, "") for p in hp]
        n_uni = sum(1 for s, c in zip(hist_sids, hist_caps) if s and c)

        for pos_pid in pos_ok:
            srng = np.random.default_rng([pool_seed, uid, pos_pid])
            k_row = int(srng.choice(K_GRID, p=K_WEIGHTS))
            n_row = int(srng.choice(N_CAND_GRID, p=N_CAND_WEIGHTS))
            if n_row < TOP_K or n_row > max(N_CAND_GRID):
                raise ValueError(f"n_cand must be in [{TOP_K}, {max(N_CAND_GRID)}], got {n_row}")

            # Full 100-item priority pool (positive, hard negatives, random fill), then a nested
            # prefix of size N and one display shuffle. Sibling targets and history are excluded.
            pool_n = max(N_CAND_GRID)
            exclude = set(real_pids) | set(hp)
            n_slots = pool_n - 1
            use_hard = hard if len(hard) <= n_slots else \
                [hard[j] for j in srng.choice(len(hard), size=n_slots, replace=False)]
            rand_negs, n_rand = [], n_slots - len(use_hard)
            if n_rand > 0:
                for p in srng.choice(pool_pids, size=min(len(pool_pids), n_rand * 4 + 50),
                                     replace=False).tolist():
                    if p not in exclude:
                        rand_negs.append(int(p)); exclude.add(int(p))
                        if len(rand_negs) == n_rand:
                            break
            priority = [pos_pid] + use_hard + rand_negs
            if len(priority) != pool_n:
                raise RuntimeError(f"short candidate pool for uid={uid}: {len(priority)}/{pool_n}")
            cand = priority[:n_row]
            grades = [pos_grade] + [0] * (len(cand) - 1)
            order = srng.permutation(n_row)
            cand = [cand[j] for j in order]
            grades = [grades[j] for j in order]
            validate_candidate_slate(cand, grades, pos_pid)
            n_hard_row = min(len(use_hard), n_row - 1)
            floors.append(random_ndcg(grades, 5))

            rows.append({
                "uid": uid, "slate_id": f"{uid}_{pos_pid}", "pool": args.pool,
                "pos_grade": pos_grade, "k": k_row, "n_cand": len(cand), "n_pos": 1,
                "n_hard_neg": n_hard_row, "n_rand_neg": n_row - 1 - n_hard_row,
                "n_hist": len(hp), "n_hist_unified": n_uni,
                "n_engaged_user": n_engaged_actual,
                "cand_pids": cand,
                "cand_sids": [pid2sid[p] if p in pid2sid else neg_pool[p][0] for p in cand],
                "cand_captions": [pid2cap[p] if p in pid2cap else neg_pool[p][1] for p in cand],
                "cand_grades": grades,
                "gold_pids": [pos_pid],
                "hist_pids": hp,
                "hist_sids_str": hist_sids,
                "hist_caps": hist_caps,
            })
            produced_grade[pos_grade] += 1

    out = pd.DataFrame(rows)
    if produced_grade != {2: users_per_grade, 3: users_per_grade}:
        raise RuntimeError(f"could not fill the balanced {args.pool} pool: {produced_grade}")

    # Invariants: user disjointness, single-positive slates, renderable candidates.
    assert not set(out.uid) & forbidden_uids, f"{args.pool} users overlap {forbidden_pools}"
    assert len(out) == out.uid.nunique(), "pool must have one slate per user"
    assert all(sum(1 for g in gr if g > 0) == 1 for gr in out.cand_grades), "not single-positive"
    assert all(len(ps) == len(set(ps)) for ps in out.cand_pids), "duplicate candidate PID"
    assert all(list(ps).count(int(gold[0])) == 1 for ps, gold in
               zip(out.cand_pids, out.gold_pids)), "gold PID is not unique in candidate slate"
    assert all(len(ps) == len(ss) == len(cs) == len(gs) == int(n) for ps, ss, cs, gs, n in
               zip(out.cand_pids, out.cand_sids, out.cand_captions, out.cand_grades,
                   out.n_cand)), "candidate arrays have inconsistent lengths"
    assert all(all(s for s in ss) for ss in out.cand_sids), "unrenderable candidate SID"
    assert all(all(c for c in cc) for cc in out.cand_captions), "unrenderable candidate caption"
    assert all(int(n) >= int(k) for n, k in zip(out.n_hist_unified, out.k)), \
        "sampled k exceeds unified history"
    out.to_parquet(dst, index=False)

    stats = {
        "seed": pool_seed,
        "n_cand_grid": N_CAND_GRID, "n_cand_weights": N_CAND_WEIGHTS,
        "k_grid": K_GRID, "k_weights": K_WEIGHTS,
        "max_slates_per_user": 1,
        "registry_members": members, "candidate_users": len(want),
        "rows": int(len(out)), "users_with_rows": int(out.uid.nunique()),
        "slates_per_user_mean": round(len(out) / out.uid.nunique(), 2),
        "engaged_per_user_mean": round(float(out.groupby("uid").n_engaged_user.first().mean()), 2),
        "dropped_unrenderable": {"selected_positives": int(n_pos_skipped),
                                 "exposure_hard_negs": int(hard_dropped)},
        "deduplicated_exposure_hard_rows": int(hard_deduped),
        "k_distribution": counts(out.k),
        "n_cand_distribution": counts(out.n_cand),
        "positive_grade_distribution": counts(out.pos_grade),
        "hard_neg_per_slate": dist(out.n_hard_neg),
        "hist_len_raw": dist(out.n_hist),
        "hist_len_unified": dist(out.n_hist_unified),
        "random_floor_ndcg5": round(float(np.mean(floors)), 4),
    }
    dump_json(STATS[args.pool], stats)
    print(f"\nwrote {dst}  rows={len(out)} users={out.uid.nunique():,} ({time.time()-t0:.0f}s)")
    print(f"dropped unrenderable: {n_pos_skipped} positives, {hard_dropped} exposure-hard negs; "
          f"deduplicated exposure-hard rows: {hard_deduped}")
    print(f"k: {stats['k_distribution']}\nN: {stats['n_cand_distribution']}\n"
          f"grade: {stats['positive_grade_distribution']}")
    print(f"hard negs per slate: mean {out.n_hard_neg.mean():.2f}; "
          f"random floor NDCG@5: {stats['random_floor_ndcg5']:.4f}")
    print(f"wrote {STATS[args.pool]}")


if __name__ == "__main__":
    main()
