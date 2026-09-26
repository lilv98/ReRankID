"""Two-axis robustness audit at one cell: does the result depend on how the prompt is worded, or on
which display order the slate happens to be in?

The two axes are measured symmetrically -- four wordings at a fixed display order, four display
orders under the fixed production wording -- and both are judged against the same baseline: repeated
runs of an UNCHANGED condition. On this cell that baseline is not small. Rerunning the production
prompt at the production order with nothing altered moves NDCG@5 by .014 and leaves only tau=.54
agreement between the two per-user rankings. A wording or order effect below that floor is not
evidence that the model is insensitive to it; it is evidence that the measurement cannot resolve the
question, which is a statement about the protocol and belongs in the report as one.

Reported per axis:

1. Mean NDCG@5 per condition and each condition's user-paired delta against the reference condition,
   with a bootstrap CI. The comparison of interest is delta-vs-floor, not delta-vs-zero.
2. Per-user SD across the axis against the measured floor, and the share of the cross-condition
   variance attributable to the axis rather than to decode noise.
3. Per-user item-ranking agreement against the reference condition, against the agreement between
   two identical runs and against a random floor.
4. Interface health: final validity per condition, which separates "this condition ranks
   differently" from "this condition broke the output format".

The floor is pooled over every repeat group supplied, so it is not one display order's peculiarity.
Every run of a repeated condition is reported; none is dropped in favour of another, because
selecting the run that best matches the other conditions would bias the audit toward the stability
it exists to test.

Decoding for the text arm is provider-default temperature, so the floor here is API sampling rather
than the kernel nondeterminism the SID arms show. The audit does not care which: it measures the
floor on the same model and cell it judges against.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.stability_audit import PROTOCOL_KEYS, c4, kendall_tau_topk
from utils import config

config.setup_env()

from utils.data_io import dump_json, load_rows, select_slate
from utils.metrics import ndcg

BOOT = 10000


def is_valid(row, n_out):
    order = row.get("order")
    return (isinstance(order, list) and len(order) == n_out
            and len(set(order)) == n_out and row.get("n_listed") == n_out)


def paired_bootstrap(diffs, n_boot=BOOT, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True,
                    help="label=path, the WORDING axis: one run per prompt wording at one fixed "
                         "display order. The first is the production wording and is the reference.")
    ap.add_argument("--orders", nargs="+", default=None,
                    help="label=path, the DISPLAY-ORDER axis: one run per display order under the "
                         "production wording. The first is the reference and is normally the same "
                         "file as the first --variants entry.")
    ap.add_argument("--repeats", nargs="+", default=None,
                    help="label=path, additional runs of conditions already listed above. Each is "
                         "matched to its condition by (shuffle, prompt_variant) and contributes to "
                         "the pooled decode floor. Required: without at least one, no delta on "
                         "either axis is interpretable.")
    ap.add_argument("--bench", default=config.BENCH)
    ap.add_argument("--k-ndcg", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--allow-protocol-drift", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.repeats:
        raise SystemExit("--repeats is required: a delta is only interpretable against a measured "
                         "floor, and this cell's floor is larger than the effects being tested")

    def parse(spec):
        if "=" not in spec:
            raise SystemExit(f"expected label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        with open(path) as handle:
            d = json.load(handle)
        meta = d["meta"]
        return {"label": label, "path": path, "meta": meta,
                # Runs predating --prompt-variant carry no such key and are the production wording.
                "cell": (int(meta["shuffle"]), meta.get("prompt_variant", "p0")),
                "res": {int(r["uid"]): r for r in d["results"]}}

    wording = [parse(s) for s in args.variants]
    orders = [parse(s) for s in args.orders] if args.orders else []
    repeats = [parse(s) for s in args.repeats]
    everything = wording + orders + repeats
    if len({r["label"] for r in everything}) != len(everything):
        raise SystemExit("duplicate labels across --variants/--orders/--repeats")

    # Only the two axes may move. Anything in PROTOCOL_KEYS differing means these are not one system.
    for key in PROTOCOL_KEYS:
        seen = {r["label"]: r["meta"].get(key) for r in everything}
        if len({json.dumps(v, sort_keys=True) for v in seen.values()}) > 1:
            msg = f"protocol drift on {key!r}: {json.dumps(seen, ensure_ascii=False)}"
            if not args.allow_protocol_drift:
                raise SystemExit(msg + "\nthese are not one system under several conditions.")
            print("WARNING: " + msg)
    # Each axis must hold the other axis fixed, or a single number would mix both effects.
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

    scored = {}

    def score(run):
        if run["label"] in scored:
            return scored[run["label"]]
        s = run["cell"][0]
        vals, tops, valid = {}, {}, 0
        for u in uids:
            row = run["res"][u]
            ok = is_valid(row, n_out)
            valid += ok
            vals[u] = ndcg(row["order"], row["grades"], args.k_ndcg) if ok else 0.0
            pids = select_slate(bench[u], n_cand, s)["cand_pids"]
            tops[u] = [int(pids[i]) for i in row["order"][:args.top_k]] if ok else []
        scored[run["label"]] = (vals, tops, valid)
        return scored[run["label"]]

    def agreement(top_a, top_b):
        taus, top5 = [], []
        for u in uids:
            a, b = top_a[u], top_b[u]
            if not a or not b:
                continue
            t = kendall_tau_topk(a, b, args.top_k)
            if t is not None:
                taus.append(t)
            top5.append(len(set(a[:5]) & set(b[:5])) / 5.0)
        return {"n_pairs_scored": len(top5),
                "kendall_tau_mean": round(float(np.mean(taus)), 4) if taus else None,
                "top5_agreement_mean": round(float(np.mean(top5)), 4) if top5 else None}

    # --- pooled decode floor: every condition that was run more than once ---
    # Deduplicate by PATH, not by object: the two axes share a corner (the production wording at
    # the production order is both p0 and s0, normally the same file), and counting that file
    # twice would invent a zero-variance replicate and deflate the floor.
    groups, seen_paths = defaultdict(list), {}
    for run in everything:
        key = os.path.realpath(run["path"])
        if key in seen_paths:
            continue
        seen_paths[key] = run["label"]
        groups[run["cell"]].append(run)
    replicated = {cell: runs for cell, runs in groups.items() if len(runs) > 1}
    if not replicated:
        raise SystemExit("no condition was run twice; --repeats matched nothing")

    floor_groups, pooled_sd, pooled_abs = {}, [], []
    for cell, runs in sorted(replicated.items()):
        vals = [score(r)[0] for r in runs]
        tops = [score(r)[1] for r in runs]
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
                    "agreement": agreement(tops[i], tops[j]),
                }
        pooled_abs.extend(pair_abs)
        floor_groups[f"s{cell[0]}/{cell[1]}"] = {
            "n_runs": len(runs),
            "runs": {r["label"]: round(float(np.mean(list(score(r)[0].values()))), 4) for r in runs},
            "per_user_sd_mean_unbiased": round(float(np.mean(sd)) / c4(len(runs)), 4),
            "pairs": pairs,
        }
    floor_sd = float(np.mean(pooled_sd))
    floor_abs = float(np.mean(pooled_abs))

    def axis_report(runs, reference, axis_name):
        if not runs:
            return None
        ref_vals, ref_tops, _ = score(reference)
        conditions, deltas = {}, {}
        for run in runs:
            vals, tops, valid = score(run)
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
                "agreement_vs_reference": agreement(ref_tops, tops),
            }
        per_user_sd = [float(np.std([score(r)[0][u] for r in runs], ddof=1)) for u in uids]
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
        "wording_axis": axis_report(wording, ref_w, "wording"),
        "order_axis": axis_report(orders, ref_o, "order") if orders else None,
        "variance_note": ("Cross-condition SD carries both the axis effect and decode noise; the "
                          "floor carries only decode noise. Under independence their variances add, "
                          "so the axis-attributable share is 1 - (floor/total)^2."),
        "agreement_floors": {
            "top5_agreement_random": round(5.0 / n_cand, 4),
            "kendall_tau_random": round(float(np.mean([t for t in mc if t is not None])), 4),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.out:
        dump_json(args.out, payload)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
