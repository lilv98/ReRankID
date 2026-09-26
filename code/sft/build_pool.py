"""Build final SFT and shared post-training slates from the frozen registry.

Training mirrors the evaluation task: exactly ONE behavior-verified positive per slate, the user's
clean exposure-hard negatives topped up with uniform-random catalog items, and every candidate
renderable in ALL modalities (SID + non-empty caption).

``--pool sft`` implements the frozen v3 recipe:
  * draw an equal number of frozen-reserve g3 users (exactly one engaged target) and g2 users
    (zero engaged and exactly one longview target), all with >=256 unified history items;
  * each user therefore contributes exactly one single-positive slate, matching the 50/50 g3/g2
    evaluation mixture without a separate per-user weighting mechanism;
  * build the same priority candidate pool as evaluation, then sample one render-time candidate
    count N from {10,20,30,50,100}; sample k from {32,64,128,256}, with most mass on long
    histories. The recipe has 6,000 users per grade and 12,000 unique users total.

``--pool posttrain`` builds a second balanced 12,000-user parent with the same simple long-history
distribution and candidate hygiene. It excludes all SFT-v3 users in addition to dev, test, and the
pre-reserved RL registry. RFT, OPD, and RLVR reuse its deterministic 6,000-user subset.

Both pools draw from the frozen reserve and are mutually disjoint from test/dev by registry
construction, which is asserted again here. ``profile_header`` is always empty because the
benchmark is scored history-only, and a leaked profile would let the teacher rationale cite
context the student never sees.

Output columns cover everything annotate.py / build_dataset.py / rft/sample.py / rl/build_data.py
consume.

CPU-only:
  python code/sft/build_pool.py --pool sft                 # -> data/rerank/sft_pool_v3.parquet
  python code/sft/build_pool.py --pool posttrain           # -> posttrain_pool_v1.parquet + manifest
  python code/sft/build_pool.py --pool sft --users-per-grade 5 --n-cand 30 \
      --out /tmp/rerankid_v3_smoke/sft_pool.parquet
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ReRankID/code
from utils import config
config.setup_env()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU-only builder

import argparse
import hashlib
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from benchmark.build_pools import dist
from utils.data_io import dump_json, update_stats
from utils.metrics import random_ndcg
from utils.prompts import TOP_K
from utils.raw_data import MASKS, RAW, build_neg_pool, engaged_mask, resolve_pids
from sft.v3_contract import (K_GRID as V3_K_GRID, K_WEIGHTS as V3_K_WEIGHTS,
                             N_CAND_GRID as V3_N_CAND_GRID,
                             N_CAND_WEIGHTS as V3_N_CAND_WEIGHTS,
                             PILOT_ROWS as V3_PILOT_ROWS,
                             RECIPE as V3_RECIPE, USERS_PER_GRADE as V3_USERS_PER_GRADE,
                             is_smoke_path)
from sft.v3_stages import build_phase_manifest
from rft.contract import (K_GRID as POSTTRAIN_K_GRID,
                          K_WEIGHTS as POSTTRAIN_K_WEIGHTS,
                          N_CAND_GRID as POSTTRAIN_N_CAND_GRID,
                          N_CAND_WEIGHTS as POSTTRAIN_N_CAND_WEIGHTS,
                          PARENT_ROWS as POSTTRAIN_ROWS,
                          POOL_SEED as POSTTRAIN_SEED,
                          POSTTRAIN_RECIPE,
                          USERS_PER_GRADE as POSTTRAIN_USERS_PER_GRADE)
from rft.stages import (build_phase_manifest as build_posttrain_phase_manifest,
                        uid_digest as posttrain_uid_digest)

POOLS = f"{config.BENCHMARK_DIR}/pools.parquet"

SEED = 2                       # distinct from the benchmark builders' SEED=1
MIN_HIST_SFT = 256             # every sampled k must be fully populated, as on the eval bench
N_CAND_GRID = [10, 20, 30, 50, 100]
N_CAND_WEIGHTS = [0.10, 0.15, 0.40, 0.20, 0.15]  # anchor-heavy, otherwise ordinary size augmentation
assert N_CAND_GRID == V3_N_CAND_GRID and N_CAND_WEIGHTS == V3_N_CAND_WEIGHTS

RECIPES = {
    "v3": {
        "name": V3_RECIPE,
        "users_per_grade": V3_USERS_PER_GRADE,
        "k_grid": V3_K_GRID,
        "k_weights": V3_K_WEIGHTS,
        "out": config.SFT_V3_POOL,
    },
}


def unique_pids(items, exclude=()):
    """Return integer PIDs in first-seen order, excluding a supplied set."""
    seen = {int(p) for p in exclude}
    out = []
    for value in items:
        pid = int(value)
        if pid not in seen:
            out.append(pid)
            seen.add(pid)
    return out


def validate_candidate_slate(cand_pids, cand_grades, gold_pid):
    """Fail fast when PID and grade arrays do not encode one unique positive."""
    if len(cand_pids) != len(cand_grades):
        raise AssertionError("candidate PID and grade lengths differ")
    if len(cand_pids) != len(set(cand_pids)):
        raise AssertionError("duplicate candidate PID")
    if list(cand_pids).count(int(gold_pid)) != 1:
        raise AssertionError("gold PID must occur exactly once")
    positives = [i for i, grade in enumerate(cand_grades) if int(grade) > 0]
    if len(positives) != 1 or int(cand_pids[positives[0]]) != int(gold_pid):
        raise AssertionError("candidate grades do not mark exactly the gold PID")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_sft_output(recipe_name, out, users_per_grade_override, n_cand, update_stats=False):
    """Resolve an SFT output without allowing custom runs to overwrite a formal recipe.

    A formal recipe may write only its own canonical path and may not use smoke-size overrides.
    Every other SFT destination is an isolated /tmp path visibly marked as smoke output.
    """
    canonical = RECIPES[recipe_name]["out"]
    dst = out or canonical
    is_canonical = os.path.realpath(dst) == os.path.realpath(canonical)
    has_override = users_per_grade_override is not None or n_cand is not None
    if is_canonical:
        if has_override:
            raise SystemExit("formal SFT output cannot use --users-per-grade or --n-cand overrides")
    else:
        if not is_smoke_path(dst):
            raise SystemExit("custom SFT output must be under /tmp with 'smoke' in the path")
        if update_stats:
            raise SystemExit("smoke SFT output cannot update canonical benchmark stats")
    return dst


def resolve_posttrain_output(out, users_per_grade_override, n_cand, update_stats=False):
    """Keep the formal shared post-training pool separate from smoke builds."""
    dst = out or config.POSTTRAIN_V1_POOL
    canonical = os.path.realpath(dst) == os.path.realpath(config.POSTTRAIN_V1_POOL)
    has_override = users_per_grade_override is not None or n_cand is not None
    if canonical and has_override:
        raise SystemExit(
            "formal post-training output cannot use --users-per-grade or --n-cand overrides"
        )
    if not canonical:
        if not is_smoke_path(dst):
            raise SystemExit("custom post-training output must be under /tmp with 'smoke' in the path")
        if update_stats:
            raise SystemExit("smoke post-training output cannot update canonical benchmark stats")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["sft", "posttrain"], default="sft")
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="v3",
                    help="frozen SFT recipe; ignored by the shared post-training pool")
    ap.add_argument("--n-cand", type=int, default=None,
                    help="fixed N override (mainly smoke/debug); SFT otherwise samples the N grid")
    ap.add_argument("--users-per-grade", type=int, default=None,
                    help="override balanced users per grade for smoke/debug")
    ap.add_argument("--out", default=None, help="output parquet (default follows --pool)")
    ap.add_argument("--update-stats", action="store_true",
                    help="also update canonical stats.json for a canonical output")
    ap.add_argument("--force-rebuild", action="store_true",
                    help="replace a formal v3/post-training pool and manifest after explicit review")
    args = ap.parse_args()
    recipe = RECIPES[args.recipe]
    if args.pool == "sft":
        users_per_grade = (int(recipe["users_per_grade"]) if args.users_per_grade is None
                           else args.users_per_grade)
        if users_per_grade < 1:
            raise SystemExit("--users-per-grade must be positive")
        dst = resolve_sft_output(args.recipe, args.out, args.users_per_grade,
                                 args.n_cand, args.update_stats)
    elif args.pool == "posttrain":
        users_per_grade = (POSTTRAIN_USERS_PER_GRADE if args.users_per_grade is None
                           else args.users_per_grade)
        if users_per_grade < 1:
            raise SystemExit("--users-per-grade must be positive")
        dst = resolve_posttrain_output(args.out, args.users_per_grade,
                                       args.n_cand, args.update_stats)

    formal_v3 = (args.pool == "sft" and args.recipe == "v3" and
                 os.path.realpath(dst) == os.path.realpath(config.SFT_V3_POOL))
    formal_posttrain = (args.pool == "posttrain" and
                        os.path.realpath(dst) == os.path.realpath(config.POSTTRAIN_V1_POOL))
    if args.force_rebuild and not (formal_v3 or formal_posttrain):
        raise SystemExit("--force-rebuild applies only to a canonical formal pool")
    if formal_v3:
        if os.path.exists(config.SFT_V3_ANNOTATIONS):
            raise SystemExit(
                "refusing to rebuild the frozen v3 parent pool while canonical annotations exist; "
                "archive and deliberately remove the annotations before starting a new experiment"
            )
        existing_plan = [path for path in (config.SFT_V3_POOL, config.SFT_V3_PHASE_MANIFEST)
                         if os.path.exists(path)]
        if existing_plan and not args.force_rebuild:
            raise SystemExit(
                "formal v3 pool/manifest already exists; reuse it or pass --force-rebuild before "
                "annotation starts"
            )
    if formal_posttrain:
        if os.path.exists(config.RFT_V1_SAMPLES) and os.path.getsize(config.RFT_V1_SAMPLES) > 0:
            raise SystemExit(
                "refusing to rebuild the post-training parent pool after formal RFT samples exist"
            )
        existing_plan = [path for path in
                         (config.POSTTRAIN_V1_POOL, config.POSTTRAIN_V1_PHASE_MANIFEST)
                         if os.path.exists(path)]
        if existing_plan and not args.force_rebuild:
            raise SystemExit(
                "formal post-training pool/manifest already exists; reuse it or pass "
                "--force-rebuild before rollout sampling starts"
            )

    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)

    pools = pd.read_parquet(POOLS)
    # The frozen reserve is already disjoint from these registry partitions.
    # Rechecking all three here makes the no-dev/test/RL-user contract explicit
    # for both SFT and the shared post-training parent.
    forbidden_pools = ["test", "dev", "rl"]
    forbidden_uids = set(pools.uid[pools.pool.isin(forbidden_pools)].tolist())
    pool_seed = POSTTRAIN_SEED if args.pool == "posttrain" else SEED
    pool_stream = 2 if args.pool == "posttrain" else 0
    rng = np.random.default_rng([pool_seed, pool_stream])
    selected_grade = {}
    excluded_sft_uids = set()
    excluded_sft_uid_sha256 = None
    if args.pool == "posttrain":
        if not os.path.exists(config.SFT_V3_PHASE_MANIFEST):
            raise SystemExit(
                f"missing frozen SFT v3 phase manifest: {config.SFT_V3_PHASE_MANIFEST}"
            )
        import json
        with open(config.SFT_V3_PHASE_MANIFEST) as handle:
            sft_manifest = json.load(handle)
        full_sft_keys = sft_manifest.get("full_keys") or []
        if len(full_sft_keys) != 12_000:
            raise SystemExit("SFT v3 phase manifest does not contain exactly 12,000 full keys")
        excluded_sft_uids = {int(uid) for uid, _ in full_sft_keys}
        if len(excluded_sft_uids) != len(full_sft_keys):
            raise SystemExit("SFT v3 phase manifest does not contain one key per user")
        excluded_sft_uid_sha256 = posttrain_uid_digest(excluded_sft_uids)
        forbidden_uids.update(excluded_sft_uids)

    # Both retained recipes draw from the frozen reserve. The post-training
    # path additionally excludes every SFT-v3 user loaded above.
    take = []
    for grade in (3, 2):
        members_g = sorted(pools.uid[(pools.pool == "reserve")
                                     & (pools.pos_grade == grade)
                                     & (pools.hist_len_uni >= MIN_HIST_SFT)
                                     & (~pools.uid.isin(excluded_sft_uids))].tolist())
        # A 10% margin covers the observed unrenderable-positive rate (~5% for g3, ~1% for g2)
        # without doubling the expensive raw/history resolution work. Tiny smoke runs retain a
        # 20-user margin so one unlucky draw does not make the builder flaky.
        margin = max(20, (users_per_grade + 9) // 10)
        n_candidate = min(len(members_g), users_per_grade + margin)
        if n_candidate < users_per_grade:
            raise SystemExit(f"not enough reserve g{grade} users: need {users_per_grade}, "
                             f"found {len(members_g)}")
        chosen = rng.choice(members_g, size=n_candidate, replace=False).tolist()
        take.extend(chosen)
        selected_grade.update({int(u): grade for u in chosen})
    members = int(sum((pools.pool == "reserve") & pools.pos_grade.isin([2, 3])
                      & (pools.hist_len_uni >= MIN_HIST_SFT)
                      & (~pools.uid.isin(excluded_sft_uids))))
    label = f"sft-{args.recipe}" if args.pool == "sft" else POSTTRAIN_RECIPE
    print(f"pool={label}: {members:,} eligible reserve exact-one users; "
          f"candidate draw={len(take):,} for target "
          f"{users_per_grade:,} g3 + {users_per_grade:,} g2")
    want = set(take)

    # raw rows for sampled uids
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
        raise RuntimeError(f"raw dataset is missing {len(missing_raw)} selected registry users")

    neg_pool = build_neg_pool(np.random.default_rng(pool_seed + 7))
    pool_pids = np.array(sorted(neg_pool.keys()), dtype=np.int64)

    need = set()
    for r in raw.values():
        need.update(int(p) for p in r["target_video_pid"])
        need.update(int(p) for p in (r["hist_video_pid"] or []))
    pid2sid, pid2cap = resolve_pids(need, neg_pool)
    print(f"resolved {len(need):,} slate/hist pids ({time.time()-t0:.0f}s)")

    def renderable(p):
        return bool(pid2sid.get(p)) and bool(pid2cap.get(p))

    rows, floors = [], []
    n_pos_skipped = 0     # selected g2/g3 positives dropped because unrenderable
    hard_dropped = 0      # exposure-hard negatives dropped because unrenderable
    hard_deduped = 0      # repeated or positive-conflicting exposure rows removed
    produced_grade = {2: 0, 3: 0}
    for uid in take:
        if produced_grade[selected_grade[uid]] >= users_per_grade:
            continue
        r = raw[uid]
        real_pids = [int(p) for p in r["target_video_pid"]]
        eng = engaged_mask(r["target_video_like"], r["target_video_follow"], r["target_video_forward"])
        n_engaged_actual = int(sum(bool(x) for x in eng))
        lv = [bool(x) for x in r["target_video_longview"]]
        hp = [int(p) for p in (r["hist_video_pid"] or [])]

        if selected_grade[uid] == 2:
            # Frozen pos_grade=2 means zero engaged and exactly one longview target.
            pos_grade = 2
            pos_all = [p for p, e, v in zip(real_pids, eng, lv) if v and not e]
        else:
            pos_grade = 3
            pos_all = [p for p, e in zip(real_pids, eng) if e]
        pos_all = unique_pids(pos_all)
        if len(pos_all) != 1:
            raise AssertionError(f"frozen exact-one g{pos_grade} user {uid} has {len(pos_all)} positives")
        pos_ok = [p for p in pos_all if renderable(p)]
        n_pos_skipped += len(pos_all) - len(pos_ok)

        # Both final recipes remove repeated or positive-conflicting PIDs.
        hard_all = [p for p, e, v in zip(real_pids, eng, lv) if not e and not v]
        hard_renderable = [p for p in hard_all if renderable(p)]
        hard = unique_pids(hard_renderable, exclude=pos_all)
        hard_dropped += len(hard_all) - len(hard_renderable)
        hard_deduped += len(hard_renderable) - len(hard)

        hist_sids = [pid2sid.get(p, "") for p in hp]
        hist_caps = [pid2cap.get(p, "") for p in hp]
        n_uni = sum(1 for s, c in zip(hist_sids, hist_caps) if s and c)

        for pos_pid in pos_ok:
            srng = np.random.default_rng([pool_seed, uid, pos_pid])
            if args.pool == "posttrain":
                k_row = int(srng.choice(POSTTRAIN_K_GRID, p=POSTTRAIN_K_WEIGHTS))
            else:
                k_row = int(srng.choice(recipe["k_grid"], p=recipe["k_weights"]))
            if args.n_cand is not None:
                n_row = int(args.n_cand)
            elif args.pool == "posttrain":
                n_row = int(srng.choice(POSTTRAIN_N_CAND_GRID,
                                        p=POSTTRAIN_N_CAND_WEIGHTS))
            else:
                n_row = int(srng.choice(N_CAND_GRID, p=N_CAND_WEIGHTS))
            if n_row < TOP_K or n_row > max(N_CAND_GRID):
                raise ValueError(f"n_cand must be in [{TOP_K}, {max(N_CAND_GRID)}], got {n_row}")

            # Construct a full N=100 priority pool, sample one nested prefix,
            # then apply one ordinary display shuffle.
            pool_n = max(N_CAND_GRID)
            # sibling positives + all longview items are excluded from candidates entirely
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
            # Nested prefix retains all available exposure-hard items before random fill.
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
                "profile_header": "",
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
        raise RuntimeError(
            f"could not fill balanced {args.pool} target: produced={produced_grade}"
        )

    # ---- invariants + report ----
    assert not set(out.uid) & forbidden_uids, f"LEAKAGE: {args.pool} users overlap {forbidden_pools}"
    assert len(out) == out.uid.nunique(), "balanced pool must have one slate per user"
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
    if args.n_cand is not None:
        assert (out.n_cand == args.n_cand).all(), "fixed-N override not honored"
    out.to_parquet(dst, index=False)
    pool_sha256 = file_sha256(dst)
    phase_manifest = None
    if formal_v3:
        manifest_columns = ["uid", "slate_id", "pool", "pos_grade", "k", "n_cand"]
        phase_manifest = build_phase_manifest(
            out[manifest_columns].to_dict("records"), pool_sha256
        )
        dump_json(config.SFT_V3_PHASE_MANIFEST, phase_manifest)
        print(f"wrote frozen phase manifest: {config.SFT_V3_PHASE_MANIFEST}")
    elif formal_posttrain:
        manifest_columns = ["uid", "slate_id", "pool", "pos_grade", "k", "n_cand",
                            "cand_grades"]
        phase_manifest = build_posttrain_phase_manifest(
            out[manifest_columns].to_dict("records"), pool_sha256,
            excluded_sft_uid_sha256,
        )
        dump_json(config.POSTTRAIN_V1_PHASE_MANIFEST, phase_manifest)
        print(f"wrote frozen phase manifest: {config.POSTTRAIN_V1_PHASE_MANIFEST}")
    print(f"\nwrote {dst}  rows={len(out)} from {out.uid.nunique():,} users ({time.time()-t0:.0f}s)")
    row_cap = 1
    print(f"slates/user: mean {len(out)/out.uid.nunique():.2f} (cap {row_cap}); "
          f"engaged/user: mean {out.groupby('uid').n_engaged_user.first().mean():.2f}")
    print(f"dropped unrenderable: {n_pos_skipped} selected positives, "
          f"{hard_dropped} exposure-hard negs")
    print(f"deduplicated or positive-conflicting exposure-hard rows: {hard_deduped}")
    print(f"k distribution:\n{out.k.value_counts().sort_index()}")
    print(f"N distribution:\n{out.n_cand.value_counts().sort_index()}")
    print(f"positive-grade distribution:\n{out.pos_grade.value_counts().sort_index()}")
    print(f"hard negs per slate: mean {out.n_hard_neg.mean():.2f}")
    print(f"random floor NDCG@5: {np.mean(floors):.4f}")

    stats = {
        "recipe": POSTTRAIN_RECIPE if args.pool == "posttrain" else recipe["name"],
        "recipe_version": "v1" if args.pool == "posttrain" else args.recipe,
        "seed": pool_seed, "n_cand": args.n_cand,
        "n_cand_grid": (POSTTRAIN_N_CAND_GRID if args.pool == "posttrain" and args.n_cand is None
                        else N_CAND_GRID if args.pool == "sft" and args.n_cand is None else None),
        "n_cand_weights": (POSTTRAIN_N_CAND_WEIGHTS
                           if args.pool == "posttrain" and args.n_cand is None else
                           N_CAND_WEIGHTS if args.pool == "sft" and args.n_cand is None else None),
        "max_slates_per_user": 1,
        "registry_members": members, "candidate_users": len(want),
        "rows": int(len(out)), "users_with_rows": int(out.uid.nunique()),
        "slates_per_user_mean": round(len(out) / out.uid.nunique(), 2),
        "engaged_per_user_mean": round(float(out.groupby("uid").n_engaged_user.first().mean()), 2),
        "dropped_unrenderable": {"selected_positives": int(n_pos_skipped),
                                 "exposure_hard_negs": int(hard_dropped)},
        "deduplicated_exposure_hard_rows": int(hard_deduped),
        "k_grid": POSTTRAIN_K_GRID if args.pool == "posttrain" else recipe["k_grid"],
        "k_weights": (POSTTRAIN_K_WEIGHTS if args.pool == "posttrain"
                      else recipe["k_weights"]),
        "k_distribution": {str(k): int(v) for k, v in
                           out.k.value_counts().sort_index().to_dict().items()},
        "n_cand_distribution": {str(k): int(v) for k, v in
                                out.n_cand.value_counts().sort_index().to_dict().items()},
        "positive_grade_distribution": {str(k): int(v) for k, v in
                                         out.pos_grade.value_counts().sort_index().to_dict().items()},
        "hard_neg_per_slate": dist(out.n_hard_neg),
        "hist_len_raw": dist(out.n_hist),
        "hist_len_unified": dist(out.n_hist_unified),
        "random_floor_ndcg5": round(float(np.mean(floors)), 4),
        "sha256": pool_sha256,
        "excluded_sft_uid_sha256": excluded_sft_uid_sha256,
        "phase_manifest": ({
            "path": os.path.relpath(
                config.POSTTRAIN_V1_PHASE_MANIFEST if args.pool == "posttrain" else
                config.SFT_V3_PHASE_MANIFEST, config.ROOT),
            "sha256": file_sha256(
                config.POSTTRAIN_V1_PHASE_MANIFEST if args.pool == "posttrain" else
                config.SFT_V3_PHASE_MANIFEST),
            "pilot": phase_manifest["stages"][str(
                POSTTRAIN_ROWS // 2 if args.pool == "posttrain" else V3_PILOT_ROWS)],
            "split": phase_manifest.get("split"),
        } if phase_manifest is not None else None),
    }
    if args.out is None or args.update_stats:
        stats_key = ("posttrain_v1_pool" if args.pool == "posttrain"
                     else f"sft_{args.recipe}_pool")
        path = update_stats(stats_key, stats)
        print(f"wrote [{stats_key}] section of {path}")
    else:
        print("custom --out: canonical stats.json left unchanged (pass --update-stats to override)")


if __name__ == "__main__":
    main()
