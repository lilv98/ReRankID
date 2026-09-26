"""Raw OpenOneRec data paths and candidate/negative-pool assembly primitives, used by the bench
builder (benchmark/build_bench.py) and the SFT pool builder (sft/build_pool.py). Each of those
implements its own per-row candidate-assembly loop independently (they have different
positive-grading designs -- see their own docstrings), so only the low-level primitives below are
actually shared.

Raw inputs live under data/recif/ (the gated OpenOneRec/OpenOneRec-RecIF HF dataset, not tracked in
git -- see .gitignore).
"""
import json

import pyarrow.parquet as pq

from utils.config import RAW_DIR
from utils.prompts import is_empty, sid_str

RAW = f"{RAW_DIR}/onerec_bench_release.parquet"
PID2SID = f"{RAW_DIR}/video_ad_pid2sid.parquet"     # pid -> [a,b,c], video codebook
PID2CAP = f"{RAW_DIR}/pid2caption.parquet"          # pid -> dense_caption
POOL_TARGET = 600_000       # reservoir of pid->sid to then intersect with captions for the neg pool
PROFILE_KEYS = ["性别", "年龄", "最近搜索内容", "视频消费类目", "关注博主类型", "最近观看的直播"]

MASKS = ["target_video_like", "target_video_follow", "target_video_forward",
         "target_video_longview", "target_video_not_interested"]


def engaged_mask(like, follow, forward):
    """grade-3 (engaged) iff like|follow|forward."""
    return [bool(lk or fo or fw) for lk, fo, fw in zip(like, follow, forward)]


def profile_header(raw_profile):
    if not raw_profile or not isinstance(raw_profile, str) or len(raw_profile) < 5:
        return ""
    try:
        d = json.loads(raw_profile)
    except Exception:
        return ""
    if not isinstance(d, dict):
        return ""
    keep = {k: d[k] for k in PROFILE_KEYS if k in d}
    return ("用户画像：" + json.dumps(keep, ensure_ascii=False)) if keep else ""


def build_neg_pool(rng):
    """Global negative pool: pids that have BOTH a SID and a non-empty caption. Reservoir-sample
    ~POOL_TARGET pids from pid2sid, then keep those with a caption. Returns {pid: (sid_str, caption)}."""
    sid_rows = pq.ParquetFile(PID2SID).metadata.num_rows
    frac = min(1.0, POOL_TARGET / sid_rows)
    pool_sid = {}
    for b in pq.ParquetFile(PID2SID).iter_batches(batch_size=2_000_000, columns=["pid", "sid"]):
        ps = b.column("pid").to_pylist()
        ss = b.column("sid").to_pylist()
        take = rng.random(len(ps)) < frac
        for t, p, s in zip(take, ps, ss):
            if t:
                pool_sid[int(p)] = sid_str(s)
    print(f"  neg pool: sampled {len(pool_sid):,} pid->SID (frac={frac:.3f})")
    pool = {}
    for b in pq.ParquetFile(PID2CAP).iter_batches(batch_size=2_000_000, columns=["pid", "dense_caption"]):
        ps = b.column("pid").to_pylist()
        cs = b.column("dense_caption").to_pylist()
        for p, c in zip(ps, cs):
            p = int(p)
            if p in pool_sid and not is_empty(c):
                pool[p] = (pool_sid[p], c)
    print(f"  neg pool: {len(pool):,} pids with BOTH sid+caption")
    return pool


def resolve_pids(need, neg_pool):
    """pid -> (sid_str, caption) for every pid in `need`, seeded from the already-loaded neg_pool and
    backfilled by streaming the two mapping parquets once each. Returns (pid2sid, pid2cap) dicts; a pid
    can be missing from either (caller decides the fallback). Shared by both builders, which previously
    each carried a verbatim copy of this double streaming loop."""
    pid2sid = {p: neg_pool[p][0] for p in need if p in neg_pool}
    miss = need - set(pid2sid)
    for b in pq.ParquetFile(PID2SID).iter_batches(batch_size=2_000_000, columns=["pid", "sid"]):
        if not miss:
            break
        ps = b.column("pid").to_pylist()
        ss = b.column("sid").to_pylist()
        for p, s in zip(ps, ss):
            p = int(p)
            if p in miss:
                pid2sid[p] = sid_str(s)
                miss.discard(p)
    print(f"  pid->SID: {len(pid2sid):,}/{len(need):,}")

    pid2cap = {p: neg_pool[p][1] for p in need if p in neg_pool}
    miss = need - set(pid2cap)
    for b in pq.ParquetFile(PID2CAP).iter_batches(batch_size=2_000_000, columns=["pid", "dense_caption"]):
        if not miss:
            break
        ps = b.column("pid").to_pylist()
        cs = b.column("dense_caption").to_pylist()
        for p, c in zip(ps, cs):
            p = int(p)
            if p in miss and not is_empty(c):
                pid2cap[p] = c
                miss.discard(p)
    print(f"  pid->caption: {len(pid2cap):,}/{len(need):,}")
    return pid2sid, pid2cap
