"""Paired task-level parity audit for dynamic LoRA and merged-model inference.

The two inputs are ordinary ``score_sid.py`` result JSONs produced on the same
users under the same benchmark protocol.  Users are joined by ``uid`` and all
per-user metrics are recomputed from the stored ranking, so aggregate rounding
in the source files cannot hide a difference.

This tool is intentionally route-agnostic.  It can compare dynamic LoRA with a
merged model, two repeated runs of either route, or an isolated adapter run
with the same adapter evaluated inside a multi-adapter sweep.  It does not load
a model or use a GPU.

Example:
  python code/benchmark/adapter_parity_audit.py \
      --a /tmp/parity/dynamic.json --label-a dynamic \
      --b /tmp/parity/merged.json --label-b merged \
      --out /tmp/parity/dynamic_vs_merged.json

The benchmark has one positive per user.  ``gold_rank`` is its 1-based rank in
the requested top-K output; a miss or invalid output is assigned K+1.  Thus a
negative ``A-B`` gold-rank difference favours A.  All confidence intervals are
deterministic paired user-level percentile bootstrap intervals.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.paired_stats import BOOT, paired_bootstrap
from utils.data_io import dump_json
from utils.metrics import is_identity_collapse, is_valid, ndcg


# Route-specific fields such as model/checkpoint/lora_dirs are deliberately
# excluded.  These fields define the benchmark and decoding protocol and must
# match for a meaningful parity claim.
PROTOCOL_KEYS = (
    "bench", "benches", "n", "slice", "n_cand", "shuffle", "render",
    "corrupt_sid", "max_model_len", "k", "top_k", "max_tokens",
    "extract_tokens", "k_ndcg", "tp", "no_profile", "no_eager", "arm",
    "protocol", "decoding", "vllm",
)


def _index_results(results, label):
    indexed = {}
    for row in results:
        if "uid" not in row:
            raise ValueError(f"{label}: result row has no uid")
        uid = int(row["uid"])
        if uid in indexed:
            raise ValueError(f"{label}: duplicate uid {uid}")
        indexed[uid] = row
    if not indexed:
        raise ValueError(f"{label}: no result rows")
    return indexed


def _n_out(run, label):
    value = run.get("metrics", {}).get("n_out", run.get("meta", {}).get("top_k"))
    if value is None:
        raise ValueError(f"{label}: neither metrics.n_out nor meta.top_k is present")
    return int(value)


def _protocol_mismatches(a_meta, b_meta):
    mismatches = {}
    for key in PROTOCOL_KEYS:
        a_value = a_meta[key] if key in a_meta else "<MISSING>"
        b_value = b_meta[key] if key in b_meta else "<MISSING>"
        if a_value != b_value:
            mismatches[key] = {"a": a_value, "b": b_value}
    return mismatches


def _validate_row_pair(a, b, uid, n_out):
    if a.get("grades") != b.get("grades"):
        raise ValueError(f"uid {uid}: candidate grades differ")
    grades = a.get("grades")
    if not isinstance(grades, list):
        raise ValueError(f"uid {uid}: grades must be a list")
    positives = [i for i, grade in enumerate(grades) if float(grade) > 0]
    if len(positives) != 1:
        raise ValueError(f"uid {uid}: expected exactly one positive, found {len(positives)}")
    for label, row in (("A", a), ("B", b)):
        missing = [key for key in ("stage", "s1_closed_think") if key not in row]
        if missing:
            raise ValueError(f"uid {uid} {label}: missing diagnostics {missing}")
        if row["stage"] not in (1, 2):
            raise ValueError(f"uid {uid} {label}: stage must be 1 or 2")
        order = row.get("order")
        if order is not None and any(not isinstance(i, int) for i in order):
            raise ValueError(f"uid {uid} {label}: order must contain integers")
        if row.get("n_listed", 0) > 0 and order is None:
            raise ValueError(f"uid {uid} {label}: n_listed > 0 but order is null")
        # An order may be short or repeated, but that is an invalid model output,
        # not malformed audit input.  is_valid() handles it under the benchmark rule.
        if row.get("n_listed", 0) < 0 or n_out <= 0:
            raise ValueError(f"uid {uid} {label}: invalid output length metadata")
    return positives[0]


def _gold_rank(row, positive, n_out):
    if not is_valid(row, n_out):
        return n_out + 1
    try:
        rank = row["order"][:n_out].index(positive) + 1
    except ValueError:
        return n_out + 1
    return rank


def _ndcg5(row, n_out):
    return ndcg(row["order"], row["grades"], 5) if is_valid(row, n_out) else 0.0


def _round(value):
    return round(float(value), 6)


def _mean_ci(values, n_boot, seed):
    values = np.asarray(values, dtype=float)
    lo, hi = paired_bootstrap(values, n_boot=n_boot, seed=seed)
    return {"mean": _round(values.mean()), "bootstrap_ci95": [_round(lo), _round(hi)]}


def _paired_summary(a_values, b_values, n_boot, seed, direction):
    a_values = np.asarray(a_values, dtype=float)
    b_values = np.asarray(b_values, dtype=float)
    diffs = a_values - b_values
    lo, hi = paired_bootstrap(diffs, n_boot=n_boot, seed=seed)
    return {
        "mean_a": _round(a_values.mean()),
        "mean_b": _round(b_values.mean()),
        "mean_diff_a_minus_b": _round(diffs.mean()),
        "bootstrap_ci95": [_round(lo), _round(hi)],
        "direction": direction,
        "n_a_higher": int(np.sum(diffs > 0)),
        "n_b_higher": int(np.sum(diffs < 0)),
        "n_ties": int(np.sum(diffs == 0)),
    }


def build_report(a, b, path_a="A.json", path_b="B.json", label_a="dynamic",
                 label_b="merged", n_boot=BOOT, seed=0, allow_meta_mismatch=False):
    """Build a strict UID-paired parity report from two score_sid result dicts."""
    for label, run in ((label_a, a), (label_b, b)):
        if not isinstance(run, dict) or "meta" not in run or "results" not in run:
            raise ValueError(f"{label}: expected score_sid JSON with meta and results")

    n_out_a, n_out_b = _n_out(a, label_a), _n_out(b, label_b)
    if n_out_a != n_out_b:
        raise ValueError(f"n_out differs ({label_a}={n_out_a}, {label_b}={n_out_b})")
    n_out = n_out_a

    mismatches = _protocol_mismatches(a["meta"], b["meta"])
    if mismatches and not allow_meta_mismatch:
        details = ", ".join(f"{key}: {v['a']!r} != {v['b']!r}"
                            for key, v in mismatches.items())
        raise ValueError(f"protocol metadata differs; refusing unpaired comparison ({details})")

    rows_a = _index_results(a["results"], label_a)
    rows_b = _index_results(b["results"], label_b)
    if set(rows_a) != set(rows_b):
        shared = len(set(rows_a) & set(rows_b))
        raise ValueError(f"uid sets differ ({len(rows_a)} vs {len(rows_b)}; {shared} shared)")

    uids = sorted(rows_a)
    order_same, top1_same, both_valid, top1_same_valid = [], [], [], []
    ndcg_a, ndcg_b, rank_a, rank_b = [], [], [], []
    valid_a, valid_b, identity_a, identity_b = [], [], [], []
    loop_a, loop_b, fallback_a, fallback_b = [], [], [], []

    for uid in uids:
        ra, rb = rows_a[uid], rows_b[uid]
        positive = _validate_row_pair(ra, rb, uid, n_out)
        va, vb = is_valid(ra, n_out), is_valid(rb, n_out)
        oa, ob = ra.get("order"), rb.get("order")
        ta = oa[0] if oa else None
        tb = ob[0] if ob else None

        order_same.append(float(oa == ob and ra.get("n_listed", 0) == rb.get("n_listed", 0)))
        top1_same.append(float(ta == tb))
        both_valid.append(float(va and vb))
        if va and vb:
            top1_same_valid.append(float(ta == tb))

        ndcg_a.append(_ndcg5(ra, n_out))
        ndcg_b.append(_ndcg5(rb, n_out))
        rank_a.append(_gold_rank(ra, positive, n_out))
        rank_b.append(_gold_rank(rb, positive, n_out))
        valid_a.append(float(va))
        valid_b.append(float(vb))
        identity_a.append(float(is_identity_collapse(oa, n_out)))
        identity_b.append(float(is_identity_collapse(ob, n_out)))
        loop_a.append(float(not bool(ra["s1_closed_think"])))
        loop_b.append(float(not bool(rb["s1_closed_think"])))
        fallback_a.append(float(ra["stage"] == 2))
        fallback_b.append(float(rb["stage"] == 2))

    changed = [uid for uid, same in zip(uids, order_same) if not same]
    report = {
        "files": {"a": path_a, "b": path_b},
        "labels": {"a": label_a, "b": label_b},
        "source_meta": {"a": a["meta"], "b": b["meta"]},
        "n_users": len(uids),
        "n_out": n_out,
        "gold_rank_missing_value": n_out + 1,
        "bootstrap": {"unit": "paired user", "n_resamples": n_boot, "seed": seed,
                      "interval": "percentile 95%"},
        "protocol_mismatches": mismatches,
        "agreement": {
            "exact_order_and_length": _mean_ci(order_same, n_boot, seed),
            "top1_all_users": _mean_ci(top1_same, n_boot, seed),
            "top1_both_valid": (_mean_ci(top1_same_valid, n_boot, seed)
                                if top1_same_valid else None),
            "both_valid": _mean_ci(both_valid, n_boot, seed),
            "n_changed_orders": len(changed),
            "changed_order_uids_first_20": changed[:20],
        },
        "ranking": {
            "ndcg@5": _paired_summary(ndcg_a, ndcg_b, n_boot, seed,
                                       "positive A-B favours A"),
            "gold_rank": _paired_summary(rank_a, rank_b, n_boot, seed,
                                          "negative A-B favours A; miss/invalid is K+1"),
            "gold_at_top1": _paired_summary([x == 1 for x in rank_a],
                                             [x == 1 for x in rank_b], n_boot, seed,
                                             "positive A-B favours A"),
        },
        "interface": {
            "valid_output": _paired_summary(valid_a, valid_b, n_boot, seed,
                                             "positive A-B favours A"),
            "identity_collapse": _paired_summary(identity_a, identity_b, n_boot, seed,
                                                  "negative A-B favours A"),
            "stage1_loop": _paired_summary(loop_a, loop_b, n_boot, seed,
                                           "negative A-B favours A"),
            "stage2_fallback": _paired_summary(fallback_a, fallback_b, n_boot, seed,
                                               "negative A-B favours A"),
        },
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="score_sid JSON for route A")
    ap.add_argument("--b", required=True, help="score_sid JSON for route B")
    ap.add_argument("--label-a", default="dynamic")
    ap.add_argument("--label-b", default="merged")
    ap.add_argument("--n-boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-meta-mismatch", action="store_true",
                    help="report protocol mismatches instead of refusing; use only after review")
    ap.add_argument("--out", default=None, help="optional report JSON path")
    args = ap.parse_args()
    if args.n_boot <= 0:
        raise SystemExit("--n-boot must be positive")

    with open(args.a, encoding="utf-8") as handle:
        a = json.load(handle)
    with open(args.b, encoding="utf-8") as handle:
        b = json.load(handle)
    try:
        report = build_report(a, b, args.a, args.b, args.label_a, args.label_b,
                              args.n_boot, args.seed, args.allow_meta_mismatch)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        dump_json(args.out, report)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
