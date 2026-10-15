# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Corruption controls for the pairwise judge (App. D.4, judge validity).

Takes users the main run already judged, damages one arm's rationale in a known way and judges the
pair again in both orders. Each corruption targets some aspects and should leave others alone:
other_user_rationale substitutes the same arm's rationale for a different user (item and user
claims become unsupported); false_item_swap exchanges two candidate numbers throughout the text
(item claims become false, coverage is unchanged); truncate_to_top1 keeps only the sentence that
justifies the first pick (coverage collapses, the surviving claims stay true).

Paper run (three users, the Gemini rationale corrupted, judge settings read from the main run):

    python code/judge/pairwise_controls.py

The arms are loaded by label from --results (default downloads/rerankid/results); --arm LABEL=PATH
overrides one. The output keeps each corrupted rationale so the control can be inspected. Writes
results/audits/rationale-judge/controls.json.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge.judge_arms import (
    ANCHOR,
    JUDGED_ARMS,
    RELEASE_RESULTS,
    ROOT,
    frozen_output,
    load_arm,
    mention_order,
    parse_arm,
)
from judge.pairwise_judge import ASPECTS, judge_pair, to_arm
from utils import config

config.setup_env()

from utils.data_io import dump_json, load_bench


DEFAULT_RUN = RELEASE_RESULTS / "audits/rationale-judge/pairwise.json"
DEFAULT_OUT = ROOT / "results/audits/rationale-judge/controls.json"

# variant -> (aspects that should turn against the corrupted arm,
#             aspects that should NOT turn against it)
EXPECTATIONS = {
    "other_user_rationale": (("item_claim_accuracy", "user_model_accuracy"), ()),
    "false_item_swap": (("item_claim_accuracy",), ("coverage",)),
    "truncate_to_top1": (("coverage",), ("item_claim_accuracy",)),
}


def false_item_swap(rationale, n_candidates):
    """Exchange the first two candidate numbers the rationale names, everywhere they appear.

    Returns None when the rationale names fewer than two candidates.
    """
    mentioned = mention_order(rationale, n_candidates)
    if len(mentioned) < 2:
        return None
    first, second = mentioned[0], mentioned[1]
    placeholder = "\x00SWAP\x00"
    out = re.sub(rf"候选\s*{first}(?!\d)", f"候选{placeholder}", rationale)
    out = re.sub(rf"候选\s*{second}(?!\d)", f"候选{first}", out)
    return out.replace(placeholder, str(second))


def truncate_to_top1(rationale, n_candidates):
    """Keep only through the end of the sentence that first names a candidate."""
    match = re.search(r"候选\s*\d{1,3}", rationale)
    if not match:
        return None
    tail = rationale[match.end():]
    stop = re.search(r"[。！？\n]", tail)
    return rationale[: match.end() + (stop.end() if stop else len(tail))].strip()


def arm_paths(labels, results, overrides):
    """Result file for each arm label of the run: an --arm override, else the released layout."""
    paths = dict(parse_arm(spec) for spec in overrides or [])
    for label in labels:
        if label not in paths:
            if label not in JUDGED_ARMS:
                raise SystemExit(f"no result file for arm {label!r}; pass --arm {label}=PATH")
            paths[label] = results / JUDGED_ARMS[label]
    return paths


def collapse(pair, labels):
    """Per-aspect winner of a pair judged in both orders, or "inconsistent" when they disagree."""
    out = {}
    for aspect in ASPECTS:
        forward = to_arm(pair[0]["verdict"][aspect], *labels)
        backward = to_arm(pair[1]["verdict"][aspect], *labels[::-1])
        out[aspect] = forward if forward == backward else "inconsistent"
    return out


def moved(before, after, target):
    """Whether the corrupted arm stopped winning an aspect; None when it was not winning it."""
    if before != target:
        return None
    return after != target


def pass_rate(checks, field):
    """Share of applicable (non-None) checks in `field` that passed."""
    values = [v for c in checks for v in c[field].values() if v is not None]
    return {"n": len(values), "passed": sum(values),
            "rate": round(sum(values) / len(values), 4) if values else None}


def judge_task(task, judge):
    """Judge one corrupted pair in one presentation order."""
    judged = judge_pair(task["row"], task["first"], task["second"], **judge)
    print(f"judged uid={task['uid']} {task['variant']} order={task['order']}", flush=True)
    return {"uid": task["uid"], "variant": task["variant"], "order": task["order"],
            "shown_first": task["shown"][0], "corrupted_rationale": task["text"],
            "verdict": judged["verdict"], "usage": judged["usage"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN,
                        help="the pairwise run whose verdicts are the uncorrupted baseline")
    parser.add_argument("--results", type=Path, default=RELEASE_RESULTS,
                        help="results root holding the anchor results of the run's arms")
    parser.add_argument("--arm", action="append", metavar="LABEL=PATH",
                        help="result file for an arm label of the run (repeatable)")
    parser.add_argument("--bench", default=config.BENCH)
    parser.add_argument("--corrupt", default=None,
                        help="which arm to damage (default: the first arm of the run)")
    parser.add_argument("--n", type=int, default=3, help="users to corrupt, per variant")
    parser.add_argument("--judge-model", default=None, help="default: the run's judge model")
    parser.add_argument("--judge-effort", default=None, help="default: the run's judge effort")
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--backend", default=None, choices=("openrouter", "vertex"),
                        help="default: the run's judge backend")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    with args.run.open() as handle:
        run = json.load(handle)
    meta = run["meta"]
    labels = list(meta["arms"])
    target = args.corrupt or labels[0]
    if target not in labels:
        raise SystemExit(f"--corrupt {target!r} is not one of {labels}")
    other = next(label for label in labels if label != target)
    model = args.judge_model or meta["judge_model"]
    effort = args.judge_effort or meta["judge_effort"]
    backend = args.backend or meta.get("judge_backend")

    bench = {int(row["uid"]): row for row in load_bench(
        args.bench, n_cand=ANCHOR["n_cand"], shuffle=ANCHOR["shuffle"], bench_slice=ANCHOR["slice"])}
    paths = arm_paths(labels, args.results, args.arm)
    arms = {label: load_arm(paths[label], list(bench.values())) for label in labels}

    # Baseline verdicts, collapsed as in the main run: a verdict counts only when both orders agree.
    baseline = {}
    orders = {}
    for record in run["records"]:
        orders.setdefault(record["uid"], {})[record["order"]] = record
    for uid, pair in orders.items():
        if len(pair) != 2:
            continue
        baseline[uid] = collapse(pair, labels)

    uids = sorted(baseline)[: args.n]
    outputs = {uid: {label: frozen_output(arms[label]["rows"][uid]) for label in labels}
               for uid in uids}

    tasks = []
    for uid in uids:
        n_cand = len(bench[uid]["cand_grades"])
        original = outputs[uid][target]["rationale"]
        variants = {
            "other_user_rationale": next(
                (outputs[u][target]["rationale"] for u in uids if u != uid), None),
            "false_item_swap": false_item_swap(original, n_cand),
            "truncate_to_top1": truncate_to_top1(original, n_cand),
        }
        for variant, text in variants.items():
            if not text or text == original:
                print(f"skip uid={uid} {variant}: corruption undefined or a no-op", flush=True)
                continue
            pair = {target: dict(outputs[uid][target], rationale=text),
                    other: outputs[uid][other]}
            for order, shown in enumerate((labels, labels[::-1])):
                tasks.append({"uid": uid, "variant": variant, "order": order, "text": text,
                              "row": bench[uid], "shown": shown,
                              "first": pair[shown[0]], "second": pair[shown[1]]})

    judge = {"model": model, "effort": effort, "backend": backend,
             "max_tokens": args.judge_max_tokens}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(partial(judge_task, judge=judge), tasks))

    checks = []
    by_case = {}
    for record in records:
        by_case.setdefault((record["uid"], record["variant"]), {})[record["order"]] = record
    for (uid, variant), pair in sorted(by_case.items()):
        if len(pair) != 2:
            continue
        after = collapse(pair, labels)
        targeted, protected = EXPECTATIONS[variant]
        move = {a: moved(baseline[uid][a], after[a], target) for a in targeted + protected}
        checks.append({
            "uid": uid,
            "variant": variant,
            "corrupted_arm": target,
            "baseline": {a: baseline[uid][a] for a in ASPECTS},
            "after": after,
            "targeted_moved": {a: move[a] for a in targeted},
            "protected_held": {a: (None if move[a] is None else not move[a]) for a in protected},
        })

    payload = {
        "meta": {
            "baseline_run": config.relpath(args.run),
            "corrupted_arm": target,
            "n_users": len(uids),
            "uids": uids,
            "variants": list(EXPECTATIONS),
            "expectations": {k: {"should_move": list(v[0]), "should_hold": list(v[1])}
                             for k, v in EXPECTATIONS.items()},
            "judge_model": model,
            "judge_effort": effort,
            "judge_backend": backend,
            "aspects": ASPECTS,
            "note": ("A verdict counts only when both presentation orders agree, exactly as in the "
                     "main run. Aspects the corrupted arm was not already winning are reported as "
                     "not-applicable, not as passes."),
        },
        "summary": {
            "targeted_aspects_moved": pass_rate(checks, "targeted_moved"),
            "protected_aspects_held": pass_rate(checks, "protected_held"),
            "by_variant": {
                variant: Counter(
                    "moved" if v else "unmoved"
                    for c in checks if c["variant"] == variant
                    for v in c["targeted_moved"].values() if v is not None
                )
                for variant in EXPECTATIONS
            },
        },
        "checks": checks,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(str(args.out), payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, default=str))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
