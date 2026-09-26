"""Validate the pairwise judge by degrading one side of a real pair and checking what moves.

The main run measures the judge's RELIABILITY -- both presentation orders, so a win rate cannot be
a position preference, and the two orders' disagreement rate bounds its self-consistency. Neither
shows the verdicts track anything real. A judge that answered from writing style alone would be
perfectly reliable and perfectly uninformative.

So take pairs the judge has already ruled on, damage one side in a known way, and ask again. Each
corruption targets specific aspects and is designed to leave the others alone:

* other_user_rationale -- substitute the same arm's rationale from a DIFFERENT user. Every claim
  about this user and these candidates becomes unsupported, so item_claim_accuracy and
  user_model_accuracy should turn against the corrupted side.
* false_item_swap -- exchange two candidate numbers throughout the text. Length, style, structure
  and the set of candidates discussed are all preserved, so coverage should NOT move; only the
  claims themselves became false.
* truncate_to_top1 -- keep only the sentence justifying the first pick. Coverage should collapse;
  the surviving claims are still true, so item_claim_accuracy should NOT turn against it.

A corruption that moves everything is a judge reading vibes; one that moves nothing is a judge not
reading at all. The interesting result is targeted movement.

One aspect already has an external check that needs no corruption at all: coverage is computable
from the text, and the main run agreed with the computed answer on 98/100 pairs with 2 genuine ties.

Unlike the main run, this output DOES contain rationale text: each record keeps the corrupted
variant, because a control nobody can inspect is not a control. The originals it derives from are
already published verbatim in the anchor result files.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge_arms import ANCHOR, ROOT, frozen_output, load_arm, mention_order
from pairwise_judge import ASPECTS, judge_pair, to_arm
from utils import config

config.setup_env()

from utils.data_io import dump_json, load_bench


DEFAULT_RUN = ROOT / "results/rationale-judge/pairwise_n100_full.json"
DEFAULT_OUT = ROOT / "results/rationale-judge/pairwise_controls.json"

GENERIC = ("根据用户历史可以看出其具有稳定的兴趣偏好，因此优先推荐与整体偏好较为一致的候选，"
           "并将其余候选依次排列。")

# variant -> (aspects that should turn against the corrupted arm,
#             aspects that should NOT turn against it)
EXPECTATIONS = {
    "other_user_rationale": (("item_claim_accuracy", "user_model_accuracy"), ()),
    "false_item_swap": (("item_claim_accuracy",), ("coverage",)),
    "truncate_to_top1": (("coverage",), ("item_claim_accuracy",)),
}


def false_item_swap(rationale: str, n_candidates: int) -> str | None:
    """Exchange the first two candidate numbers the rationale names, everywhere they appear.

    Every claim attached to either number now describes the other candidate, while length, style,
    the discussed-candidate set and the argument structure are untouched. Returns None when the
    rationale names fewer than two candidates and the corruption is not defined.
    """
    mentioned = mention_order(rationale, n_candidates)
    if len(mentioned) < 2:
        return None
    first, second = mentioned[0], mentioned[1]
    placeholder = "\x00SWAP\x00"
    out = re.sub(rf"候选\s*{first}(?!\d)", f"候选{placeholder}", rationale)
    out = re.sub(rf"候选\s*{second}(?!\d)", f"候选{first}", out)
    return out.replace(placeholder, str(second))


def truncate_to_top1(rationale: str, n_candidates: int) -> str | None:
    """Keep only through the end of the sentence that first names a candidate."""
    match = re.search(r"候选\s*\d{1,3}", rationale)
    if not match:
        return None
    tail = rationale[match.end():]
    stop = re.search(r"[。！？\n]", tail)
    return rationale[: match.end() + (stop.end() if stop else len(tail))].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN,
                        help="the pairwise run whose verdicts are the uncorrupted baseline")
    parser.add_argument("--bench", default=config.BENCH)
    parser.add_argument("--corrupt", default=None,
                        help="which arm to damage (default: the first arm of the run)")
    parser.add_argument("--n", type=int, default=3, help="users to corrupt, per variant")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-effort", default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--confirm-external-data", action="store_true")
    args = parser.parse_args()
    if not args.confirm_external_data:
        raise SystemExit(
            "control validation sends real-user histories and candidate captions to an external "
            "provider; rerun with --confirm-external-data only after explicit authorization"
        )

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
    arms = {label: load_arm(Path(meta["arms"][label]["result_file"]), list(bench.values()))
            for label in labels}

    # Baseline verdicts, collapsed the same way the main run does: a verdict only counts when both
    # presentation orders agree, so the control is measured against a stable starting point.
    baseline = {}
    orders = {}
    for record in run["records"]:
        orders.setdefault(record["uid"], {})[record["order"]] = record
    for uid, pair in orders.items():
        if len(pair) != 2:
            continue
        collapsed = {}
        for aspect in ASPECTS:
            forward = to_arm(pair[0]["verdict"][aspect], *labels)
            backward = to_arm(pair[1]["verdict"][aspect], *labels[::-1])
            collapsed[aspect] = forward if forward == backward else "inconsistent"
        baseline[uid] = collapsed

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
            for order in (0, 1):
                tasks.append((uid, variant, text, order))

    progress = [0]

    def run_one(task):
        uid, variant, text, order = task
        damaged = dict(outputs[uid][target], rationale=text)
        pair = {target: damaged, other: outputs[uid][other]}
        first_label, second_label = (labels if order == 0 else labels[::-1])
        judged = judge_pair(bench[uid], pair[first_label], pair[second_label], model, effort,
                            backend=backend, max_tokens=args.judge_max_tokens)
        progress[0] += 1
        print(f"{progress[0]}/{len(tasks)} judged (uid={uid}, {variant}, order={order})", flush=True)
        return {"uid": uid, "variant": variant, "order": order,
                "shown_first": first_label, "corrupted_rationale": text,
                "verdict": judged["verdict"], "usage": judged["usage"]}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(run_one, tasks))

    checks = []
    by_case = {}
    for record in records:
        by_case.setdefault((record["uid"], record["variant"]), {})[record["order"]] = record
    for (uid, variant), pair in sorted(by_case.items()):
        if len(pair) != 2:
            continue
        after = {}
        for aspect in ASPECTS:
            forward = to_arm(pair[0]["verdict"][aspect], *labels)
            backward = to_arm(pair[1]["verdict"][aspect], *labels[::-1])
            after[aspect] = forward if forward == backward else "inconsistent"
        targeted, protected = EXPECTATIONS[variant]
        # "Turned against" means the corrupted arm stopped being the winner on that aspect. Starting
        # from a baseline where it was not winning anyway, the aspect carries no information here
        # and is reported as not-applicable rather than as a pass.
        def moved(aspect):
            if baseline[uid][aspect] != target:
                return None
            return after[aspect] != target
        checks.append({
            "uid": uid,
            "variant": variant,
            "corrupted_arm": target,
            "baseline": {a: baseline[uid][a] for a in ASPECTS},
            "after": after,
            "targeted_moved": {a: moved(a) for a in targeted},
            "protected_held": {a: (None if moved(a) is None else not moved(a)) for a in protected},
        })

    def rate(field):
        values = [v for c in checks for v in c[field].values() if v is not None]
        return {"n": len(values), "passed": sum(values),
                "rate": round(sum(values) / len(values), 4) if values else None}

    payload = {
        "meta": {
            "baseline_run": str(args.run),
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
        "cost_usd": round(sum(r["usage"]["cost_usd"] for r in records), 6),
        "summary": {
            "targeted_aspects_moved": rate("targeted_moved"),
            "protected_aspects_held": rate("protected_held"),
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
    print(f"cost=${payload['cost_usd']:.4f}; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
