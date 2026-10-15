# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Tables 1, 2, D.1, D.2 and D.4 recomputed from the per-user results (paper Sec. 4 and 5, App. D).

Every row reads one anchor file (k=64, N=30, display seed 0, 400 test users). Table 1 and Table D.1
print the stored metrics; Identity% and Loop% are the interface diagnostics, and input tokens come
from diagnosis/prompt-tokens.json (OpenOneRec tokenizer). Table 2 adds each method's NDCG@5 change
against Base and against its declared initializer, with the user-paired bootstrap CI of
benchmark/paired_stats.py (10,000 resamples, seed 0) against the initializer. Table D.2 lists the
paper's paired NDCG@5 contrasts, and Table D.4 splits the SFT-minus-Base contrast by positive grade,
read from --bench. No model is run.

    python code/paper/make_main_tables.py [--results downloads/rerankid/results]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from benchmark.paired_stats import per_user_scores
from utils import config
from utils.data_io import load_rows
from utils.metrics import half_up, paired_bootstrap
from utils.prompts import TOP_K

RESULTS = Path(config.ROOT) / "downloads/rerankid/results"
ANCHOR = "k64_n30.json"
SCALES = (("1.7B", "onerec-1.7b"), ("8B", "onerec-8b"))
TEXT_REFERENCES = (("Claude-Sonnet-5", "claude-sonnet-5"), ("Gemini-3.6-Flash", "gemini-3.6-flash"),
                   ("GPT-5.6-Terra", "gpt-5.6-terra"))
VIEWS = (("SID", "sid", "sid"), ("Text", "text", "text"), ("Both", "both", "sid_text"))
# (row label, method directory, supervision, initializer) in the row order of Tables 2 and D.1.
METHOD_ROWS = (("SFT", "sft", "Teacher traces", "base"),
        ("OPD", "opd", "Teacher distributions", "base"),
        ("RFT", "rft", "Filtered rollouts", "sft"),
        ("RLVR", "rlvr-ndcg5", "NDCG@5 reward", "sft"),
        ("RLVR", "rlvr-hit1", "Hit@1 reward", "sft"))
# Finish reasons of an API reply that ran to its token cap (the Loop% of a text reference).
TOKEN_CAP = ("MAX_TOKENS", "length", "max_tokens")


def base_file(model, view="sid"):
    return f"diagnosis/{model}/{view}/{ANCHOR}"


def method_file(method, model):
    return f"posttraining/{method}/{model}/test/{ANCHOR}"


def text_file(model):
    return f"diagnosis/{model}/{ANCHOR}"


# Table D.2: (A, B) pairs by name; names resolve through system_file().
PAIRS = (("sft 1.7B", "base 1.7B"), ("sft 8B", "base 8B"),
         ("gpt-5.6-terra", "gemini-3.6-flash"), ("gpt-5.6-terra", "sft 1.7B"),
         ("sft 1.7B", "gemini-3.6-flash"), ("sft 8B", "gemini-3.6-flash"), ("sft 8B", "sft 1.7B"),
         ("opd 1.7B", "base 1.7B"), ("opd 1.7B", "sft 1.7B"),
         ("opd 8B", "base 8B"), ("opd 8B", "sft 8B"),
         ("rft 1.7B", "sft 1.7B"), ("rft 8B", "sft 8B"),
         ("rlvr-ndcg5 1.7B", "sft 1.7B"), ("rlvr-hit1 1.7B", "sft 1.7B"),
         ("rlvr-hit1 1.7B", "rlvr-ndcg5 1.7B"),
         ("rlvr-ndcg5 8B", "sft 8B"), ("rlvr-hit1 8B", "sft 8B"), ("rlvr-hit1 8B", "rlvr-ndcg5 8B"),
         ("rlvr-ndcg5 1.7B", "base 1.7B"), ("rlvr-ndcg5 8B", "base 8B"),
         ("rlvr-ndcg5 8B", "gemini-3.6-flash"), ("rlvr-ndcg5 8B", "gpt-5.6-terra"))


def system_file(name):
    """Anchor file of a system named '<method> <scale>' or by a text reference's model name."""
    if " " not in name:
        return text_file(name)
    method, scale = name.split()
    model = dict(SCALES)[scale]
    return base_file(model) if method == "base" else method_file(method, model)


def load(results, relative):
    with (results / relative).open() as handle:
        return json.load(handle)


def contrast(rows_a, rows_b):
    """Mean A, mean B, mean A-B and its 95% CI for user-paired NDCG@5 (as paired_stats.py)."""
    sa = per_user_scores(rows_a, "ndcg@5", TOP_K)
    sb = per_user_scores(rows_b, "ndcg@5", TOP_K)
    if set(sa) != set(sb):
        raise SystemExit("the two result files cover different users")
    diffs = [sa[uid] - sb[uid] for uid in sorted(sa)]
    lo, hi = paired_bootstrap(diffs)
    return np.mean(list(sa.values())), np.mean(list(sb.values())), np.mean(diffs), lo, hi


def num(value):
    """Paper style: three decimals without the leading zero."""
    return f"{half_up(value, 3)}".replace("0.", ".", 1)


def signed(value):
    return ("+" if value >= 0 else "-") + num(abs(value))


def percent(rate):
    return f"{half_up(100 * rate, 1)}"


def ci(lo, hi):
    return f"[{signed(lo)},{signed(hi)}]"


def interface(payload):
    """Identity% and Loop% of one result file, as printed strings."""
    metrics = payload["metrics"]
    if "two_stage" in metrics:
        loop = metrics["two_stage"]["stage1_loop_rate"]
    else:
        finish = payload.get("finish_reasons", {})
        loop = sum(finish.get(key, 0) for key in TOKEN_CAP) / metrics["n_users"]
    return percent(metrics["identity_collapse_rate"]), percent(loop)


def ranking_metrics(metrics):
    return " ".join(num(metrics[key]) for key in ("ndcg@5", "ndcg@10", "recall@5", "recall@10",
                                                  "mrr_top"))


def table1(results):
    tokens = load(results, "diagnosis/prompt-tokens.json")
    at_anchor = {render: f"{values[tokens['k_grid'].index(64)]:,.0f}"
                 for render, values in tokens["mean_prompt_tokens"].items()}
    print("Table 1: anchor results before post-training")
    print(f"{'System':20s} {'Input':5s} {'N@5':>4s} {'N@10':>4s} {'R@5':>4s} {'R@10':>4s} "
          f"{'MRR':>4s} {'Ident%':>6s} {'Loop%':>5s} {'Tokens':>7s}")
    for name, view in (("Random", "--"), ("Caption", "Text")):
        payload = load(results, f"diagnosis/{name.lower()}/{ANCHOR}")
        print(f"{name:20s} {view:5s} {ranking_metrics(payload['metrics'])} "
              f"{'--':>6s} {'--':>5s} {'--':>7s}")
    for name, model in TEXT_REFERENCES:
        payload = load(results, text_file(model))
        identity, loop = interface(payload)
        print(f"{name:20s} {'Text':5s} {ranking_metrics(payload['metrics'])} "
              f"{identity:>6s} {loop:>5s} {at_anchor['text']:>7s}")
    for scale, model in SCALES:
        for label, view, render in VIEWS:
            payload = load(results, base_file(model, view))
            identity, loop = interface(payload)
            print(f"{'OpenOneRec-' + scale:20s} {label:5s} {ranking_metrics(payload['metrics'])} "
                  f"{identity:>6s} {loop:>5s} {at_anchor[render]:>7s}")
    print()


def table2(results):
    print("Table 2: SID-only post-training at the anchor (95% CI vs. the initializer)")
    print(f"{'Method':6s} {'Init':5s} {'Supervision':22s} {'N@5':>4s} {'dBase':>5s} {'dInit':>5s} "
          f"{'CI vs. Init':>15s} {'Ident%':>6s} {'Loop%':>5s}")
    for scale, model in SCALES:
        print(f"OpenOneRec-{scale}")
        base = load(results, base_file(model))
        identity, loop = interface(base)
        print(f"{'Base':6s} {'--':5s} {'--':22s} {num(base['metrics']['ndcg@5']):>4s} {'--':>5s} "
              f"{'--':>5s} {'--':>15s} {identity:>6s} {loop:>5s}")
        sft = load(results, method_file("sft", model))
        for label, method, supervision, init in METHOD_ROWS:
            payload = load(results, method_file(method, model))
            init_rows = (base if init == "base" else sft)["results"]
            mean, _, d_base, _, _ = contrast(payload["results"], base["results"])
            _, _, d_init, lo, hi = contrast(payload["results"], init_rows)
            identity, loop = interface(payload)
            print(f"{label:6s} {init.upper() if init == 'sft' else 'Base':5s} {supervision:22s} "
                  f"{num(mean):>4s} {signed(d_base):>5s} {signed(d_init):>5s} {ci(lo, hi):>15s} "
                  f"{identity:>6s} {loop:>5s}")
    print()


def table_d1(results):
    print("Table D.1: ranking metrics of the post-training methods")
    print(f"{'Method':6s} {'Scale':5s} {'Supervision':22s} {'N@5':>4s} {'N@10':>4s} {'R@5':>4s} "
          f"{'R@10':>4s} {'MRR':>4s} {'Valid':>7s}")
    for scale, model in SCALES:
        for label, method, supervision, _ in METHOD_ROWS:
            metrics = load(results, method_file(method, model))["metrics"]
            valid = f"{metrics['parsed']}/{metrics['n_users']}"
            print(f"{label:6s} {scale:5s} {supervision:22s} {ranking_metrics(metrics)} {valid:>7s}")
    print()


def table_d2(results):
    print("Table D.2: user-paired NDCG@5 comparisons")
    print(f"{'A':18s} {'B':18s} {'Mean A':>6s} {'Mean B':>6s} {'A-B':>5s} {'95% paired CI':>15s}")
    for a, b in PAIRS:
        rows_a = load(results, system_file(a))["results"]
        rows_b = load(results, system_file(b))["results"]
        mean_a, mean_b, diff, lo, hi = contrast(rows_a, rows_b)
        print(f"{a:18s} {b:18s} {num(mean_a):>6s} {num(mean_b):>6s} {signed(diff):>5s} "
              f"{ci(lo, hi):>15s}")
    print()


def table_d4(results, bench):
    if not os.path.exists(bench):
        print(f"Table D.4 skipped: {bench} not found (python code/download/release.py --install)")
        return
    grade = {int(row["uid"]): int(row["pos_grade"]) for row in load_rows(bench)}
    print("Table D.4: anchor NDCG@5 by positive-engagement stratum (SFT minus Base)")
    print(f"{'Scale / stratum':16s} {'Base':>5s} {'SFT':>5s} {'Delta':>6s} {'95% CI':>15s}")
    for scale, model in SCALES:
        base = load(results, base_file(model))["results"]
        sft = load(results, method_file("sft", model))["results"]
        for g in (2, 3):
            rows_b = [row for row in base if grade[int(row["uid"])] == g]
            rows_a = [row for row in sft if grade[int(row["uid"])] == g]
            mean_sft, mean_base, diff, lo, hi = contrast(rows_a, rows_b)
            print(f"{scale + ' / grade ' + str(g):16s} {num(mean_base):>5s} {num(mean_sft):>5s} "
                  f"{signed(diff):>6s} {ci(lo, hi):>15s}")


def main():
    parser = argparse.ArgumentParser(description="Recompute the main tables from result files.")
    parser.add_argument("--results", type=Path, default=RESULTS, help="results root")
    parser.add_argument("--bench", default=config.BENCH, help="bench.parquet, for Table D.4")
    args = parser.parse_args()
    table1(args.results)
    table2(args.results)
    table_d1(args.results)
    table_d2(args.results)
    table_d4(args.results, args.bench)


if __name__ == "__main__":
    main()
