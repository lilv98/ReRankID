"""Paired significance statistics for two scorer result JSONs.

The frozen protocol uses a per-user paired bootstrap confidence interval and a
Wilcoxon signed-rank test on the same users.

Both scorers already store per-user `order` + `grades` in results, so per-user metric values are
RECOMPUTED here under the exact aggregate() scoring rule (invalid response = 0) rather than trusted
from the summary block — the two files therefore never need to have been produced by the same code
version, only by the same protocol (same bench, same N/s/k). Users are matched on uid; the script
refuses to compare files whose uid sets differ (that would silently break pairing).

Everything is deterministic (seeded bootstrap) and dependency-free (numpy only; the Wilcoxon test is
the normal approximation with tie correction and continuity correction, fine at n=400 — at the tiny
smoke n it prints a warning rather than pretending exactness).

Usage:
  python code/benchmark/paired_stats.py --a results/A.json --b results/B.json
  python code/benchmark/paired_stats.py --a A.json --b B.json --metric ndcg@10 --bench data/benchmark/bench.parquet
    (--bench joins pos_grade back on uid and adds per-stratum g2/g3 deltas)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ReRankID/code
from utils import config
config.setup_env()

import argparse
import json
import math

import numpy as np

from utils.metrics import best_rank, is_valid, ndcg, recall

BOOT = 10_000


def per_user_scores(results, metric, n_out):
    """{uid: score} under the aggregate() rule: invalid -> 0."""
    out = {}
    for r in results:
        uid = int(r["uid"])
        if not is_valid(r, n_out):
            out[uid] = 0.0
        elif metric.startswith("ndcg@"):
            out[uid] = ndcg(r["order"], r["grades"], int(metric.split("@")[1]))
        elif metric.startswith("recall@"):
            out[uid] = recall(r["order"], r["grades"], int(metric.split("@")[1]))
        elif metric == "mrr":
            b = best_rank(r["order"], r["grades"])
            out[uid] = (1.0 / b) if b else 0.0
        else:
            raise ValueError(f"unknown metric {metric!r} (ndcg@K / recall@K / mrr)")
    return out


def paired_bootstrap(diffs, n_boot=BOOT, seed=0):
    """Percentile CI of the mean difference under user-level resampling."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_signed_rank(diffs):
    """Normal-approximation Wilcoxon signed-rank (zeros dropped, average ranks for ties, tie-corrected
    variance, continuity correction). Returns (W_plus, n_effective, z, two-sided p)."""
    d = np.asarray([x for x in diffs if x != 0.0])
    n = len(d)
    if n == 0:
        return 0.0, 0, 0.0, 1.0
    ranks = np.empty(n)
    order = np.argsort(np.abs(d))
    sorted_abs = np.abs(d)[order]
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        i = j + 1
    w_plus = float(ranks[d > 0].sum())
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var_w <= 0:
        return w_plus, n, 0.0, 1.0
    z = (w_plus - mean_w - math.copysign(0.5, w_plus - mean_w)) / math.sqrt(var_w)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return w_plus, n, z, min(1.0, p)


def compare(a_res, b_res, metric, n_out, n_boot=BOOT, seed=0, label=""):
    sa, sb = per_user_scores(a_res, metric, n_out), per_user_scores(b_res, metric, n_out)
    if set(sa) != set(sb):
        raise SystemExit(f"uid sets differ ({len(sa)} vs {len(sb)} users, "
                         f"{len(set(sa) & set(sb))} shared) — refusing to pair")
    uids = sorted(sa)
    diffs = [sa[u] - sb[u] for u in uids]
    lo, hi = paired_bootstrap(diffs, n_boot=n_boot, seed=seed)
    w_plus, n_eff, z, p = wilcoxon_signed_rank(diffs)
    out = {
        "label": label or None, "metric": metric, "n_users": len(uids),
        "mean_a": round(float(np.mean(list(sa.values()))), 4),
        "mean_b": round(float(np.mean(list(sb.values()))), 4),
        "mean_diff_a_minus_b": round(float(np.mean(diffs)), 4),
        "bootstrap_ci95": [round(lo, 4), round(hi, 4)],
        "wilcoxon": {"w_plus": w_plus, "n_nonzero": n_eff, "z": round(z, 3), "p_two_sided": round(p, 6)},
        "n_a_wins": int(sum(d > 0 for d in diffs)), "n_b_wins": int(sum(d < 0 for d in diffs)),
        "n_ties": int(sum(d == 0 for d in diffs)),
    }
    if n_eff and n_eff < 20:
        out["warning"] = f"only {n_eff} nonzero-diff users: the normal approximation is rough at this n"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="result JSON of system A")
    ap.add_argument("--b", required=True, help="result JSON of system B")
    ap.add_argument("--metric", default="ndcg@5", help="ndcg@K / recall@K / mrr (default ndcg@5)")
    ap.add_argument("--n-boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", default=None,
                    help="bench parquet; joins pos_grade on uid for per-stratum g2/g3 deltas")
    ap.add_argument("--out", default=None, help="write the report JSON here (default: print only)")
    args = ap.parse_args()

    A, B = json.load(open(args.a)), json.load(open(args.b))
    n_out = int(A["metrics"].get("n_out", 10))
    for key in ("n_cand", "shuffle", "k"):
        va, vb = A["meta"].get(key), B["meta"].get(key)
        if va != vb:
            print(f"WARNING: meta.{key} differs (A={va} B={vb}) — the comparison is not protocol-matched")

    report = {"a": args.a, "b": args.b,
              "overall": compare(A["results"], B["results"], args.metric, n_out,
                                 n_boot=args.n_boot, seed=args.seed)}

    if args.bench:
        from utils.data_io import load_rows
        grade = {int(r["uid"]): int(r["pos_grade"]) for r in load_rows(args.bench)}
        strata = {}
        for g in sorted(set(grade.values())):
            ar = [r for r in A["results"] if grade.get(int(r["uid"])) == g]
            br = [r for r in B["results"] if grade.get(int(r["uid"])) == g]
            if ar and br:
                strata[f"g{g}"] = compare(ar, br, args.metric, n_out,
                                          n_boot=args.n_boot, seed=args.seed, label=f"pos_grade={g}")
        report["strata"] = strata

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        from utils.data_io import dump_json
        dump_json(args.out, report)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
