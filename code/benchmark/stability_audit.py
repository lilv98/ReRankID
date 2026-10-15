# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Display-order audit of one model at one cell (paper App. D.3, Display-Order and Prompt-Wording
Stability).

Takes the model's result files for the same cell under several display seeds (score_*.py --shuffle)
plus one repeated decode at the first seed, and reports: (1) the seed-0 mean against the mean over
display orders, with a user-paired bootstrap CI of the difference; (2) the per-user score SD across
orders against the decode floor measured from the repeat, and the share of variance attributable
to order; (3) Kendall tau and top-5 agreement of the induced item rankings across orders, against a
random floor; (4) how often each display position reaches the top 1 and top 10, with a chi-square
test of uniform top-1 positions. Display indices are mapped back to items with
utils.data_io.select_slate on the benchmark row, so only uid, order, grades and n_listed are read
from the result rows. All files must agree on PROTOCOL_KEYS.

    D=results/audits/stability/gemini-3.6-flash; A=results/diagnosis/gemini-3.6-flash/k64_n30.json
    python code/benchmark/stability_audit.py \\
        --results $A $D/k64_n30_s1.json $D/k64_n30_s2.json \\
        --decode-repeat $D/k64_n30_s0_repeat1.json
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ReRankID/code
from utils import config  # noqa: E402
from utils.data_io import dump_json, load_rows, select_slate  # noqa: E402
from utils.metrics import c4, is_valid, kendall_tau_topk, ndcg, paired_bootstrap  # noqa: E402

# Meta keys that define what the model was asked; they must match across all audited files.
PROTOCOL_KEYS = ("n_cand", "k", "model", "checkpoint", "slice", "render",
                 "emit_rationale", "max_tokens", "effort", "backend", "top_k")


def load_run(path):
    """One result file with its display seed and per-user rows."""
    with open(path) as f:
        d = json.load(f)
    return {"path": config.relpath(path), "s": int(d["meta"]["shuffle"]), "meta": d["meta"],
            "res": {int(r["uid"]): r for r in d["results"]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", nargs="+", required=True,
                    help="two or more result files of one model and cell, one per display seed")
    ap.add_argument("--decode-repeat", required=True,
                    help="a second decode at the display seed of the lowest --results seed")
    ap.add_argument("--bench", default=config.BENCH)
    ap.add_argument("--k-ndcg", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default=f"{config.RESULTS}/audits/stability/display-order.json")
    args = ap.parse_args()

    runs = [load_run(p) for p in args.results]
    runs.sort(key=lambda r: r["s"])
    ss = [r["s"] for r in runs]
    if len(set(ss)) != len(ss):
        raise SystemExit(f"duplicate display seeds {ss}; pass one file per seed")
    repeat = load_run(args.decode_repeat)
    if repeat["s"] != ss[0]:
        raise SystemExit(f"--decode-repeat is at seed {repeat['s']} but the first --results file is "
                         f"at seed {ss[0]}; the repeat must keep the display order fixed")
    drift = {}
    for key in PROTOCOL_KEYS:
        checked = runs + [repeat]
        if len({json.dumps(r["meta"].get(key), sort_keys=True) for r in checked}) > 1:
            drift[key] = {(f"s{r['s']}" + ("_repeat" if r is repeat else "")): r["meta"].get(key)
                          for r in checked}
    if drift:
        raise SystemExit("the result files differ in protocol, so they are not one system under "
                         "several display orders:\n" + json.dumps(drift, indent=2, ensure_ascii=False))
    n_cand = int(runs[0]["meta"]["n_cand"])
    n_out = int(runs[0]["meta"].get("top_k", args.top_k))

    uids = set(runs[0]["res"])
    for r in runs[1:]:
        if set(r["res"]) != uids:
            raise SystemExit("uid sets differ across the result files")
    uids = sorted(uids)

    bench = {int(r["uid"]): r for r in load_rows(args.bench)}
    missing = [u for u in uids if u not in bench]
    if missing:
        raise SystemExit(f"{len(missing)} uids not in {args.bench} (e.g. {missing[:3]})")

    # per-user scores per seed (invalid -> 0) and the top-k lists as item ids
    scores = {r["s"]: {} for r in runs}     # s -> uid -> ndcg@k
    top_items = {r["s"]: {} for r in runs}  # s -> uid -> top-k pool pids
    pos_curves = {r["s"]: {"top1": np.zeros(n_cand), "topk": np.zeros(n_cand)} for r in runs}
    for run in runs:
        s = run["s"]
        for u in uids:
            row = run["res"][u]
            ok = is_valid(row, n_out)
            scores[s][u] = ndcg(row["order"], row["grades"], args.k_ndcg) if ok else 0.0
            slate = select_slate(bench[u], n_cand, s)
            if ok:
                top_items[s][u] = [int(slate["cand_pids"][i]) for i in row["order"][:args.top_k]]
                pos_curves[s]["top1"][row["order"][0]] += 1
                for i in row["order"][:args.top_k]:
                    pos_curves[s]["topk"][i] += 1
            else:
                top_items[s][u] = []

    # decode floor: a second decode of the display order runs[0] used
    if set(repeat["res"]) != set(uids):
        raise SystemExit("--decode-repeat scores a different uid set than --results")
    rep_scores, rep_top = {}, {}
    for u in uids:
        row = repeat["res"][u]
        ok = is_valid(row, n_out)
        rep_scores[u] = ndcg(row["order"], row["grades"], args.k_ndcg) if ok else 0.0
        slate = select_slate(bench[u], n_cand, repeat["s"])
        rep_top[u] = [int(slate["cand_pids"][i]) for i in row["order"][:args.top_k]] if ok else []
    base, base_top = scores[ss[0]], top_items[ss[0]]
    both = [u for u in uids if base_top[u] and rep_top[u]]
    sd_raw = float(np.mean([np.std([base[u], rep_scores[u]], ddof=1) for u in uids]))
    taus = [t for t in (kendall_tau_topk(base_top[u], rep_top[u], args.top_k) for u in both)
            if t is not None]
    decode = {
        "file": repeat["path"], "shuffle": repeat["s"], "n_pairs_scored": len(both),
        "mean": round(float(np.mean([rep_scores[u] for u in uids])), 4),
        "per_user_sd_mean": round(sd_raw, 4),
        "per_user_sd_mean_unbiased": round(sd_raw / c4(2), 4),
        "mean_abs_delta": round(float(np.mean([abs(base[u] - rep_scores[u]) for u in uids])), 4),
        "top5_agreement_mean": round(float(np.mean(
            [len(set(base_top[u][:5]) & set(rep_top[u][:5])) / 5.0 for u in both])), 4)
            if both else None,
        "kendall_tau_mean": round(float(np.mean(taus)), 4) if taus else None,
    }

    # 1. seed-0 mean against the mean over display orders
    means = {f"s{s}": round(float(np.mean(list(scores[s].values()))), 4) for s in scores}
    perm_avg_per_user = [float(np.mean([scores[s][u] for s in scores])) for u in uids]
    perm_avg = float(np.mean(perm_avg_per_user))
    s0 = ss[0]
    delta = float(means[f"s{s0}"] - perm_avg)
    delta_ci = paired_bootstrap([scores[s0][u] - a for u, a in zip(uids, perm_avg_per_user)])

    # 2. per-user SD across orders against the decode floor
    per_user_sd = [float(np.std([scores[s][u] for s in scores], ddof=1)) for u in uids] \
        if len(runs) > 1 else [0.0]
    per_user_range = [float(max(scores[s][u] for s in scores) - min(scores[s][u] for s in scores))
                      for u in uids]
    # The cross-order SD carries order and decode variation, the floor only decode variation; under
    # independence the variances add, so the order share is 1 - (floor/total)^2.
    sd_order = float(np.mean(per_user_sd)) / c4(max(len(runs), 2))
    order_share = None
    if sd_order > 0:
        order_share = round(max(0.0, 1.0 - (decode["per_user_sd_mean_unbiased"] / sd_order) ** 2), 4)

    # 3. item-ranking agreement across orders, with random floors
    pair_stats = {}
    for ra, rb in itertools.combinations(ss, 2):
        taus, top5 = [], []
        for u in uids:
            a, b = top_items[ra][u], top_items[rb][u]
            if not a or not b:
                continue
            t = kendall_tau_topk(a, b, args.top_k)
            if t is not None:
                taus.append(t)
            top5.append(len(set(a[:5]) & set(b[:5])) / 5.0)
        pair_stats[f"s{ra}~s{rb}"] = {
            "n_pairs_scored": len(top5),
            "kendall_tau_mean": round(float(np.mean(taus)), 4) if taus else None,
            "top5_agreement_mean": round(float(np.mean(top5)), 4) if top5 else None,
        }
    # With absent items tied at rank k+1, two random top-k lists score a negative tau (each ranks
    # its own items above the other's), so the tau floor is estimated by Monte Carlo.
    rng = np.random.default_rng(0)
    mc = [kendall_tau_topk(list(rng.permutation(n_cand)[:args.top_k]),
                           list(rng.permutation(n_cand)[:args.top_k]), args.top_k)
          for _ in range(500)]
    floors = {"top5_agreement_random": round(5.0 / n_cand, 4),   # E[overlap of two top-5] / 5
              "kendall_tau_random": round(float(np.mean([t for t in mc if t is not None])), 4)}

    # 4. top-1 and top-k rates by display position, with a chi-square test of uniform top-1 positions
    curves = {}
    for s in pos_curves:
        n_ok = sum(1 for u in uids if top_items[s][u])
        expected = n_ok / n_cand
        chi2 = float(((pos_curves[s]["top1"] - expected) ** 2).sum() / expected) if n_ok else None
        curves[f"s{s}"] = {
            "n_valid": n_ok,
            "top1_uniformity_chi2": round(chi2, 2) if chi2 is not None else None,
            "top1_uniformity_df": n_cand - 1,
            "p_top1_by_position": [round(x / n_ok, 4) if n_ok else 0.0
                                   for x in pos_curves[s]["top1"]],
            "p_topk_by_position": [round(x / n_ok, 4) if n_ok else 0.0
                                   for x in pos_curves[s]["topk"]],
        }

    report = {
        "files": {f"s{r['s']}": r["path"] for r in runs},
        "n_users": len(uids), "n_cand": n_cand, "k_ndcg": args.k_ndcg,
        "means": {**means, "permutation_avg": round(perm_avg, 4),
                  "delta_s0_minus_avg": round(delta, 4),
                  "delta_s0_minus_avg_ci95": [round(delta_ci[0], 4), round(delta_ci[1], 4)]},
        "order_variance": {
            "per_user_sd_mean": round(float(np.mean(per_user_sd)), 4),
            "per_user_sd_mean_unbiased": round(sd_order, 4),
            "per_user_sd_p90": round(float(np.percentile(per_user_sd, 90)), 4),
            "per_user_range_mean": round(float(np.mean(per_user_range)), 4),
            "share_users_range_gt_0.1": round(float(np.mean([r > 0.1 for r in per_user_range])), 4),
            "decode_noise_floor": decode["per_user_sd_mean_unbiased"],
            "decode_noise_source": "measured (--decode-repeat, same display order)",
            "order_share_of_variance": order_share,
            "decode_repeat": decode,
        },
        "item_ranking_agreement": {**pair_stats, "floors": floors},
        "position_exposure": curves,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "position_exposure"},
                     indent=2, ensure_ascii=False))
    for name, c in curves.items():
        print(f"{name}: top-1 position chi2({c['top1_uniformity_df']}) = {c['top1_uniformity_chi2']}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dump_json(args.out, report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
