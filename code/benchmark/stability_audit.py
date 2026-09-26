"""A1 permutation-stability audit aggregation (paper app:stability): given the SAME model's result
JSONs for the anchor cell at s in {0,1,2} (score_sid/score_text with --shuffle 0/1/2), report the four
preregistered quantities:

  1. s=0 mean vs the permutation-averaged mean (expectation: within the paired bootstrap CI of s=0);
  2. order-induced per-user score variance, set against the ±0.01 decode-repeat noise floor;
  3. cross-permutation Kendall tau and top-5 agreement on the induced ITEM ranking, against the
     random-permutation floor;
  4. exposure-by-input-position curves (P(display position p reaches the top-10 / top-1)), which
     detect position-copying heuristics no single order can reveal.

The mapping from a result row's `order` (DISPLAY indices, which differ per s) back to item identity
is re-derived deterministically: select_slate(bench_row, N, s) reproduces the exact display order each
run saw, so display index -> pool pid needs only the frozen bench parquet — nothing beyond uid is
trusted from the result files. Escape hatch (preregistered): if the s=0 mean moves by more than the
minimal detectable delta, headline scoring switches to permutation-averaged metrics for all systems —
computable from these same files, no artifact rebuild.

Usage:
  python code/benchmark/stability_audit.py --results r_s0.json r_s1.json r_s2.json \
      --bench data/benchmark/bench.parquet [--out audit.json]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ReRankID/code
from utils import config
config.setup_env()

import argparse
import itertools
import json
import math

import numpy as np

from utils.data_io import dump_json, load_rows, select_slate
from utils.metrics import is_valid, ndcg

DECODE_NOISE = 0.01   # the recorded ±0.01 decode-repeat variation (v0 report), the comparison floor

# Meta keys that define WHAT the model was asked, as opposed to where the output landed. Every one of
# them must be identical across the files: an audit over runs that differ in any of these compares two
# systems, not two display orders. `emit_rationale` and `max_tokens` are the live hazard — the text
# arm's anchor was scored in rationale mode (no responseSchema, 8192-token cap), so a plain re-run at
# s=1 silently swaps the protocol. Keys absent from a scorer's meta read as None for every file and
# therefore never trip the check.
PROTOCOL_KEYS = ("n_cand", "k", "model", "checkpoint", "slice", "render", "no_profile",
                 "emit_rationale", "max_tokens", "effort", "backend", "corrupt_sid", "top_k")


def c4(n):
    """E[s]/sigma for a size-n sample: the mean sample SD is biased LOW, and the bias depends on n
    (0.798 at n=2, 0.886 at n=3). Dividing by c4 makes an SD estimated from 2 decode replicates
    comparable to one estimated from 3 display orders. Strictly this is the Gaussian correction and
    single-positive NDCG@5 is far from Gaussian (it takes six discrete values); it is applied as a
    first-order adjustment for the differing replicate counts, not as a distributional claim."""
    return math.sqrt(2.0 / (n - 1)) * math.gamma(n / 2.0) / math.gamma((n - 1) / 2.0)


def kendall_tau_topk(list_a, list_b, k):
    """Kendall tau-b between two top-k ITEM lists (any hashable ids). Items absent from a list are
    tied at rank k+1, so the comparison runs over the union of the two lists. Returns None when the
    union has no comparable pair (both lists identical singletons etc.)."""
    union = list(dict.fromkeys(list(list_a) + list(list_b)))
    ra = {x: i + 1 for i, x in enumerate(list_a)}
    rb = {x: i + 1 for i, x in enumerate(list_b)}
    conc = disc = ties_a = ties_b = 0
    for x, y in itertools.combinations(union, 2):
        da = ra.get(x, k + 1) - ra.get(y, k + 1)
        db = rb.get(x, k + 1) - rb.get(y, k + 1)
        if da == 0 and db == 0:
            continue
        if da == 0:
            ties_a += 1
        elif db == 0:
            ties_b += 1
        elif (da > 0) == (db > 0):
            conc += 1
        else:
            disc += 1
    denom = np.sqrt((conc + disc + ties_a) * (conc + disc + ties_b))
    return None if denom == 0 else float((conc - disc) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True,
                    help="2+ result JSONs of ONE model/protocol differing only in --shuffle")
    ap.add_argument("--bench", default=config.BENCH)
    ap.add_argument("--k-ndcg", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-protocol-drift", action="store_true",
                    help="downgrade the PROTOCOL_KEYS mismatch check to a warning")
    ap.add_argument("--decode-repeat", default=None,
                    help="result JSON re-scored at the SAME shuffle as the first --results file. Used "
                         "only to MEASURE the decode-noise floor that quantity 2 is judged against; "
                         "without it the report falls back to the unverified DECODE_NOISE constant.")
    args = ap.parse_args()

    def load_run(path):
        d = json.load(open(path))
        return {"path": path, "s": int(d["meta"]["shuffle"]), "meta": d["meta"],
                "res": {int(r["uid"]): r for r in d["results"]}}

    runs = [load_run(p) for p in args.results]
    runs.sort(key=lambda r: r["s"])
    ss = [r["s"] for r in runs]
    if len(set(ss)) != len(ss):
        raise SystemExit(f"duplicate shuffle ids {ss} — pass one file per s")
    # The repeat is deliberately NOT part of --results: it shares a shuffle id with runs[0] and would
    # trip the duplicate check. It is a second decode of the same display order, nothing else.
    repeat = load_run(args.decode_repeat) if args.decode_repeat else None
    if repeat and repeat["s"] != ss[0]:
        raise SystemExit(f"--decode-repeat is at shuffle {repeat['s']}, but the first --results file "
                         f"is at shuffle {ss[0]} — the repeat must hold the display order fixed")
    drift = {}
    for key in PROTOCOL_KEYS:
        checked = runs + ([repeat] if repeat else [])
        if len({json.dumps(r["meta"].get(key), sort_keys=True) for r in checked}) > 1:
            drift[key] = {(f"s{r['s']}" + ("_rep" if r is repeat else "")): r["meta"].get(key)
                          for r in checked}
    if drift:
        msg = ("protocol drift across the result files — these are not one system at several "
               "display orders:\n" + json.dumps(drift, indent=2, ensure_ascii=False))
        if not args.allow_protocol_drift:
            raise SystemExit(msg + "\nRe-run the odd file out, or pass --allow-protocol-drift.")
        print("WARNING: " + msg)
    n_cand = int(runs[0]["meta"]["n_cand"])
    n_out = int(runs[0]["meta"].get("top_k", args.top_k))

    uids = set(runs[0]["res"])
    for r in runs[1:]:
        if set(r["res"]) != uids:
            raise SystemExit("uid sets differ across the result files — not the same run protocol")
    uids = sorted(uids)

    bench = {int(r["uid"]): r for r in load_rows(args.bench)}
    missing = [u for u in uids if u not in bench]
    if missing:
        raise SystemExit(f"{len(missing)} uids not in {args.bench} (e.g. {missing[:3]})")

    # --- per-user scores per shuffle (aggregate() rule: invalid -> 0) + item-level top-k lists ---
    scores = {r["s"]: {} for r in runs}     # s -> uid -> ndcg@k
    top_items = {r["s"]: {} for r in runs}  # s -> uid -> [pid...] (model's top-k as ITEM ids)
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

    # --- decode-noise floor: a SECOND decode of the display order runs[0] already used. The order is
    # held fixed, so every per-user movement here is the model's own nondeterminism — the only honest
    # baseline for the cross-order movement in quantity 2. Without it the floor is a constant nobody
    # measured on this model.
    decode = None
    if repeat:
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

    # --- 1. means: s=0 vs permutation average ---
    means = {f"s{s}": round(float(np.mean(list(scores[s].values()))), 4) for s in scores}
    perm_avg_per_user = [float(np.mean([scores[s][u] for s in scores])) for u in uids]
    perm_avg = float(np.mean(perm_avg_per_user))
    s0 = ss[0]
    delta = float(means[f"s{s0}"] - perm_avg)

    # --- 2. per-user order-induced variance vs the decode-noise floor ---
    per_user_sd = [float(np.std([scores[s][u] for s in scores], ddof=1)) for u in uids] \
        if len(runs) > 1 else [0.0]
    per_user_range = [float(max(scores[s][u] for s in scores) - min(scores[s][u] for s in scores))
                      for u in uids]
    # Cross-order SD carries BOTH order and decode variation; the decode floor carries only decode.
    # Under independence their variances add, so the order-attributable share is 1 - (floor/total)^2.
    sd_order = float(np.mean(per_user_sd)) / c4(max(len(runs), 2))
    order_share = None
    if decode and sd_order > 0:
        order_share = round(max(0.0, 1.0 - (decode["per_user_sd_mean_unbiased"] / sd_order) ** 2), 4)

    # --- 3. cross-permutation item-ranking agreement + random floor ---
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
    # Random floors under the absent-items-tied-at-k+1 convention. NOTE the tau floor is NEGATIVE,
    # not 0: two independent random top-k lists mostly disagree about which items belong in the top
    # at all (each ranks its own picks above the other's), which this convention scores as
    # discordance — so it is Monte-Carlo'd, not assumed.
    rng = np.random.default_rng(0)
    mc = [kendall_tau_topk(list(rng.permutation(n_cand)[:args.top_k]),
                           list(rng.permutation(n_cand)[:args.top_k]), args.top_k)
          for _ in range(500)]
    floors = {"top5_agreement_random": round(5.0 / n_cand, 4),   # E|A∩B|/5 = 25/N / 5
              "kendall_tau_random": round(float(np.mean([t for t in mc if t is not None])), 4)}

    # --- 4. exposure-by-input-position curves ---
    curves = {}
    for s in pos_curves:
        n_ok = sum(1 for u in uids if top_items[s][u])
        curves[f"s{s}"] = {
            "n_valid": n_ok,
            "p_top1_by_position": [round(x / n_ok, 4) if n_ok else 0.0
                                   for x in pos_curves[s]["top1"]],
            "p_topk_by_position": [round(x / n_ok, 4) if n_ok else 0.0
                                   for x in pos_curves[s]["topk"]],
        }

    report = {
        "files": {f"s{r['s']}": r["path"] for r in runs},
        "n_users": len(uids), "n_cand": n_cand, "k_ndcg": args.k_ndcg,
        "means": {**means, "permutation_avg": round(perm_avg, 4),
                  "delta_s0_minus_avg": round(delta, 4)},
        "order_variance": {
            "per_user_sd_mean": round(float(np.mean(per_user_sd)), 4),
            "per_user_sd_mean_unbiased": round(sd_order, 4),
            "per_user_sd_p90": round(float(np.percentile(per_user_sd, 90)), 4),
            "per_user_range_mean": round(float(np.mean(per_user_range)), 4),
            "share_users_range_gt_0.1": round(float(np.mean([r > 0.1 for r in per_user_range])), 4),
            "decode_noise_floor": decode["per_user_sd_mean_unbiased"] if decode else DECODE_NOISE,
            "decode_noise_source": ("measured (--decode-repeat, same display order)" if decode
                                    else "UNVERIFIED hardcoded DECODE_NOISE constant"),
            "order_share_of_variance": order_share,
            "decode_repeat": decode,
        },
        "item_ranking_agreement": {**pair_stats, "floors": floors},
        "position_exposure": curves,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "position_exposure"},
                     indent=2, ensure_ascii=False))
    flat = curves[f"s{s0}"]["p_top1_by_position"]
    print(f"\ns{s0} top-1 display-position spread: min={min(flat):.3f} max={max(flat):.3f} "
          f"(uniform would be {1.0 / n_cand:.3f})")
    if args.out:
        dump_json(args.out, report)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
