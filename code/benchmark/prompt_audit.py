# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-axis audit at one cell: prompt wording and display order, each judged against the decode
floor (paper App. D.3, Display-Order and Prompt-Wording Stability).

The wording axis holds the display order fixed and varies the prompt (score_text.py
--prompt-variant p0 to p3); the order axis holds the wording fixed and varies the display seed.
Repeated decodes of unchanged conditions give a pooled decode floor (per-user SD and mean absolute
delta). For each axis the report gives the mean NDCG@5 and validity per condition, each condition's
user-paired delta against the first (reference) condition with a bootstrap CI and whether it
exceeds the floor, item-ranking agreement with the reference, and the share of variance that the
axis explains beyond decode noise. Validity here is strict: exactly top_k distinct indices.

    D=results/audits/stability/gemini-3.6-flash; A=results/diagnosis/gemini-3.6-flash/k64_n30.json
    python code/benchmark/prompt_audit.py \\
        --variants p0=$A p1=$D/k64_n30_p1.json p2=$D/k64_n30_p2.json p3=$D/k64_n30_p3.json \\
        --orders s0=$A s1=$D/k64_n30_s1.json s2=$D/k64_n30_s2.json s3=$D/k64_n30_s3.json \\
        --repeats s0_repeat1=$D/k64_n30_s0_repeat1.json s1_repeat1=$D/k64_n30_s1_repeat1.json \\
                  s1_repeat2=$D/k64_n30_s1_repeat2.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ReRankID/code
from benchmark.stability_audit import PROTOCOL_KEYS  # noqa: E402
from utils import config  # noqa: E402
from utils.data_io import dump_json, load_rows, select_slate  # noqa: E402
from utils.metrics import c4, kendall_tau_topk, ndcg  # noqa: E402

BOOT = 10000


def is_valid(row, n_out):
    """Strict validity used by this audit: exactly n_out distinct indices, all listed."""
    order = row.get("order")
    return (isinstance(order, list) and len(order) == n_out
            and len(set(order)) == n_out and row.get("n_listed") == n_out)


def paired_bootstrap(diffs, n_boot=BOOT, seed=0):
    """95% percentile CI of the mean paired difference, as [lo, hi] rounded to 4 decimals."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def parse_run(spec):
    """Load one label=path result file with its (shuffle, prompt_variant) cell."""
    if "=" not in spec:
        raise SystemExit(f"expected label=path, got {spec!r}")
    label, path = spec.split("=", 1)
    with open(path) as handle:
        d = json.load(handle)
    meta = d["meta"]
    return {"label": label, "file": path, "path": config.relpath(path), "meta": meta,
            # a file without prompt_variant is the reported wording p0
            "cell": (int(meta["shuffle"]), meta.get("prompt_variant", "p0")),
            "res": {int(r["uid"]): r for r in d["results"]}}


def score_run(run, uids, bench, n_cand, n_out, k_ndcg, top_k):
    """Per-user NDCG@k_ndcg (0 when invalid), top-k item PIDs, and the number of valid answers."""
    vals, tops, valid = {}, {}, 0
    for u in uids:
        row = run["res"][u]
        ok = is_valid(row, n_out)
        valid += ok
        vals[u] = ndcg(row["order"], row["grades"], k_ndcg) if ok else 0.0
        pids = select_slate(bench[u], n_cand, run["cell"][0])["cand_pids"]
        tops[u] = [int(pids[i]) for i in row["order"][:top_k]] if ok else []
    return vals, tops, valid


def agreement(top_a, top_b, uids, top_k):
    """Mean Kendall tau and top-5 overlap between two runs' item rankings."""
    taus, top5 = [], []
    for u in uids:
        a, b = top_a[u], top_b[u]
        if not a or not b:
            continue
        t = kendall_tau_topk(a, b, top_k)
        if t is not None:
            taus.append(t)
        top5.append(len(set(a[:5]) & set(b[:5])) / 5.0)
    return {"n_pairs_scored": len(top5),
            "kendall_tau_mean": round(float(np.mean(taus)), 4) if taus else None,
            "top5_agreement_mean": round(float(np.mean(top5)), 4) if top5 else None}


def axis_report(runs, reference, axis_name, scores, uids, floor, top_k):
    """Conditions, deltas against the reference and variance share of one axis."""
    floor_sd, floor_abs = floor
    ref_vals, ref_tops, _ = scores[reference["label"]]
    conditions, deltas = {}, {}
    for run in runs:
        vals, tops, valid = scores[run["label"]]
        conditions[run["label"]] = {
            "path": run["path"], "shuffle": run["cell"][0], "prompt_variant": run["cell"][1],
            "mean_ndcg5": round(float(np.mean(list(vals.values()))), 4),
            "n_valid": valid, "validity": round(valid / len(uids), 4),
        }
        if run is reference:
            continue
        d = [vals[u] - ref_vals[u] for u in uids]
        mean_d = float(np.mean(d))
        deltas[run["label"]] = {
            "mean_delta_vs_reference": round(mean_d, 4),
            "bootstrap_ci95": paired_bootstrap(d),
            "exceeds_decode_floor": bool(abs(mean_d) > floor_abs),
            "agreement_vs_reference": agreement(ref_tops, tops, uids, top_k),
        }
    per_user_sd = [float(np.std([scores[r["label"]][0][u] for r in runs], ddof=1)) for u in uids]
    sd_axis = float(np.mean(per_user_sd)) / c4(len(runs))
    return {
        "reference": reference["label"],
        "n_conditions": len(runs),
        "conditions": conditions,
        "deltas_vs_reference": deltas,
        "per_user_sd_unbiased": round(sd_axis, 4),
        f"{axis_name}_share_of_variance": (
            round(max(0.0, 1.0 - (floor_sd / sd_axis) ** 2), 4) if sd_axis > 0 else None),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variants", nargs="+", required=True,
                    help="label=path per prompt wording at one display order; the first is the "
                         "reported wording and the reference")
    ap.add_argument("--orders", nargs="+", default=None,
                    help="label=path per display order under the reported wording; the first is "
                         "the reference (normally the same file as the first --variants entry)")
    ap.add_argument("--repeats", nargs="+", required=True,
                    help="label=path, repeated decodes of conditions listed above, matched by "
                         "(shuffle, prompt_variant); they form the pooled decode floor")
    ap.add_argument("--bench", default=config.BENCH)
    ap.add_argument("--k-ndcg", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default=f"{config.RESULTS}/audits/stability/two-axis.json")
    args = ap.parse_args()

    wording = [parse_run(s) for s in args.variants]
    orders = [parse_run(s) for s in args.orders] if args.orders else []
    repeats = [parse_run(s) for s in args.repeats]
    everything = wording + orders + repeats
    if len({r["label"] for r in everything}) != len(everything):
        raise SystemExit("duplicate labels across --variants/--orders/--repeats")

    # only the two axes may vary; each axis must hold the other one fixed
    for key in PROTOCOL_KEYS:
        seen = {r["label"]: r["meta"].get(key) for r in everything}
        if len({json.dumps(v, sort_keys=True) for v in seen.values()}) > 1:
            raise SystemExit(f"the files differ in {key!r}, so they are not one system: "
                             f"{json.dumps(seen, ensure_ascii=False)}")
    if len({r["cell"][0] for r in wording}) > 1:
        raise SystemExit("--variants span several display orders; the wording axis must fix it")
    if orders and len({r["cell"][1] for r in orders}) > 1:
        raise SystemExit("--orders span several wordings; the order axis must fix it")

    ref_w, ref_o = wording[0], (orders[0] if orders else None)
    n_cand = int(ref_w["meta"]["n_cand"])
    n_out = int(ref_w["meta"].get("top_k", args.top_k))
    uids = set(ref_w["res"])
    for r in everything:
        if set(r["res"]) != uids:
            raise SystemExit(f"{r['label']} scores a different uid set")
    uids = sorted(uids)

    bench = {int(r["uid"]): r for r in load_rows(args.bench)}
    scores = {r["label"]: score_run(r, uids, bench, n_cand, n_out, args.k_ndcg, args.top_k)
              for r in everything}

    # pooled decode floor over every condition run more than once; files are deduplicated by path
    # because the reference file normally appears on both axes
    groups, seen_paths = defaultdict(list), {}
    for run in everything:
        key = os.path.realpath(run["file"])
        if key in seen_paths:
            continue
        seen_paths[key] = run["label"]
        groups[run["cell"]].append(run)
    replicated = {cell: runs for cell, runs in groups.items() if len(runs) > 1}
    if not replicated:
        raise SystemExit("no condition was run twice; --repeats matched nothing")

    floor_groups, pooled_sd, pooled_abs = {}, [], []
    for cell, runs in sorted(replicated.items()):
        vals = [scores[r["label"]][0] for r in runs]
        tops = [scores[r["label"]][1] for r in runs]
        sd = [float(np.std([v[u] for v in vals], ddof=1)) for u in uids]
        pooled_sd.append(np.mean(sd) / c4(len(runs)))
        pair_abs, pairs = [], {}
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                d = [vals[i][u] - vals[j][u] for u in uids]
                pair_abs.append(float(np.mean(np.abs(d))))
                pairs[f"{runs[i]['label']}~{runs[j]['label']}"] = {
                    "mean_delta": round(float(np.mean(d)), 4),
                    "bootstrap_ci95": paired_bootstrap(d),
                    "mean_abs_delta": round(float(np.mean(np.abs(d))), 4),
                    "agreement": agreement(tops[i], tops[j], uids, args.top_k),
                }
        pooled_abs.extend(pair_abs)
        floor_groups[f"s{cell[0]}/{cell[1]}"] = {
            "n_runs": len(runs),
            "runs": {r["label"]: round(float(np.mean(list(scores[r["label"]][0].values()))), 4)
                     for r in runs},
            "per_user_sd_mean_unbiased": round(float(np.mean(sd)) / c4(len(runs)), 4),
            "pairs": pairs,
        }
    floor_sd = float(np.mean(pooled_sd))
    floor_abs = float(np.mean(pooled_abs))

    floor = (floor_sd, floor_abs)
    rng = np.random.default_rng(0)
    mc = [kendall_tau_topk(list(rng.permutation(n_cand)[:args.top_k]),
                           list(rng.permutation(n_cand)[:args.top_k]), args.top_k)
          for _ in range(500)]

    payload = {
        "cell": {"model": ref_w["meta"].get("model"), "k": ref_w["meta"].get("k"), "n_cand": n_cand,
                 "slice": ref_w["meta"].get("slice"), "n_users": len(uids),
                 "k_ndcg": args.k_ndcg, "top_k": args.top_k,
                 "wording_axis_at_shuffle": ref_w["cell"][0],
                 "order_axis_at_variant": ref_o["cell"][1] if ref_o else None},
        "decode_floor": {
            "pooled_per_user_sd_unbiased": round(floor_sd, 4),
            "pooled_mean_abs_delta": round(floor_abs, 4),
            "groups": floor_groups,
            "note": ("Repeated runs of an unchanged condition. Both axes are judged against this; a "
                     "delta smaller than it is unresolved, not absent."),
        },
        "wording_axis": axis_report(wording, ref_w, "wording", scores, uids, floor, args.top_k),
        "order_axis": (axis_report(orders, ref_o, "order", scores, uids, floor, args.top_k)
                       if orders else None),
        "variance_note": ("Cross-condition SD carries both the axis effect and decode noise; the "
                          "floor carries only decode noise. Under independence their variances add, "
                          "so the axis-attributable share is 1 - (floor/total)^2."),
        "agreement_floors": {
            "top5_agreement_random": round(5.0 / n_cand, 4),
            "kendall_tau_random": round(float(np.mean([t for t in mc if t is not None])), 4),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dump_json(args.out, payload)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
