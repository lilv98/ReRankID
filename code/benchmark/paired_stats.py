# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""User-paired comparison of two result files: bootstrap CI and Wilcoxon signed-rank test (paper
Sec. 3 statistics; App. D.1 paired contrasts and D.2 engagement strata).

Per-user scores are recomputed from each row's order and grades under the aggregate() rule (an
invalid response scores 0), users are matched on uid, and files with different uid sets are
refused. The CI is a percentile bootstrap over users (10,000 resamples, seed 0); the Wilcoxon test
is the normal approximation with tie and continuity corrections, so a warning is printed when few
users differ. --bench adds the same comparison within the grade-2 and grade-3 strata.

    python code/benchmark/paired_stats.py \\
        --a results/posttraining/rlvr-ndcg5/onerec-8b/test/k64_n30.json \\
        --b results/diagnosis/onerec-8b/sid/k64_n30.json --bench data/benchmark/bench.parquet
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ReRankID/code
import argparse
import json
import math

import numpy as np

from utils import config
from utils.data_io import dump_json, load_rows
from utils.metrics import BOOT, best_rank, is_valid, ndcg, paired_bootstrap, recall


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
                         f"{len(set(sa) & set(sb))} shared); refusing to pair")
    uids = sorted(sa)
    diffs = [sa[u] - sb[u] for u in uids]
    lo, hi = paired_bootstrap(diffs, n_boot=n_boot, seed=seed)
    w_plus, n_eff, z, p = wilcoxon_signed_rank(diffs)
    out = {
        "label": label or None, "metric": metric, "n_users": len(uids),
        "mean_a": round(float(np.mean(list(sa.values()))), 6),
        "mean_b": round(float(np.mean(list(sb.values()))), 6),
        "mean_diff_a_minus_b": round(float(np.mean(diffs)), 6),
        "bootstrap_ci95": [round(lo, 6), round(hi, 6)],   # round once, half up, to report
        "wilcoxon": {"w_plus": w_plus, "n_nonzero": n_eff, "z": round(z, 3), "p_two_sided": round(p, 6)},
        "n_a_wins": int(sum(d > 0 for d in diffs)), "n_b_wins": int(sum(d < 0 for d in diffs)),
        "n_ties": int(sum(d == 0 for d in diffs)),
    }
    if n_eff and n_eff < 20:
        out["warning"] = f"only {n_eff} nonzero-diff users: the normal approximation is rough at this n"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
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
            print(f"WARNING: meta.{key} differs (A={va} B={vb}); the comparison is not protocol-matched")

    report = {"a": config.relpath(args.a), "b": config.relpath(args.b),
              "overall": compare(A["results"], B["results"], args.metric, n_out,
                                 n_boot=args.n_boot, seed=args.seed)}

    if args.bench:
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
        dump_json(args.out, report)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
