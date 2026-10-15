# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Pick the dev-best checkpoint from score_sid dev results (SFT, RFT, OPD, RLVR; paper Sec. 5).

Reads every result JSON in a directory (400 dev users at k64_n30), recomputes each checkpoint's
mean NDCG@5 from the per-user rows (invalid outputs score 0, no rounding), and selects the highest
mean, taking the earliest checkpoint on ties. Initializer entries (base, sft, init) are reported
but never selected. A checkpoint's step comes from the release file name (epoch<e>, step<s>), else
from the trailing number of its adapter path in meta.adapter (checkpoint-<n>, step-<n>), else from
the trailing number of its label in meta.checkpoint.

    python code/utils/select_checkpoint.py results/posttraining/rlvr-ndcg5/onerec-8b/dev
    eval "$(python code/utils/select_checkpoint.py <dev-dir> --shell)"   # BEST_NAME, BEST_STEP, ...
"""
import argparse
import glob
import json
import os
import re
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import is_valid, ndcg  # noqa: E402

INIT_LABELS = ("base", "sft", "init")
RELEASE_NAME = re.compile(r"^(?:epoch|step)(\d+(?:\.\d+)?)$")
TRAILING_NUMBER = re.compile(r"(\d+(?:\.\d+)?)/?$")


def mean_ndcg5(payload):
    """Unrounded mean NDCG@5 over all users; invalid outputs count as 0."""
    n_out = int(payload.get("meta", {}).get("top_k") or 10)
    rows = payload["results"]
    scores = [ndcg(r["order"], r["grades"], 5) if is_valid(r, n_out) else 0.0 for r in rows]
    return sum(scores) / len(scores)


def checkpoint_step(path, meta):
    """Numeric training position of one result file, or None if it cannot be determined."""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = RELEASE_NAME.match(stem)
    if match:
        return float(match.group(1))
    for candidate in (str(meta.get("adapter") or ""), str(meta.get("checkpoint") or "")):
        match = TRAILING_NUMBER.search(candidate)
        if match:
            return float(match.group(1))
    return None


def collect(dev_dir):
    """Score every result file in dev_dir. Returns (candidates, initializers)."""
    candidates, initializers = [], []
    for path in sorted(glob.glob(os.path.join(dev_dir, "*.json"))):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        meta = payload.get("meta", {})
        stem = os.path.splitext(os.path.basename(path))[0]
        name = stem if RELEASE_NAME.match(stem) else str(meta.get("checkpoint") or stem)
        entry = {"name": name, "file": path, "ndcg@5": mean_ndcg5(payload)}
        if name in INIT_LABELS or stem in INIT_LABELS:
            initializers.append(entry)
            continue
        step = checkpoint_step(path, meta)
        if step is None:
            raise SystemExit(f"cannot determine the checkpoint step of {path}")
        entry["step"] = int(step) if step.is_integer() else step
        candidates.append(entry)
    if not candidates:
        raise SystemExit(f"no checkpoint results in {dev_dir}")
    candidates.sort(key=lambda e: e["step"])
    return candidates, initializers


def choose(candidates):
    """Highest mean NDCG@5; the earliest step wins exact ties."""
    return min(candidates, key=lambda e: (-e["ndcg@5"], e["step"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dev_dir", help="directory of score_sid dev result JSONs")
    parser.add_argument("--shell", action="store_true",
                        help="print BEST_NAME/BEST_STEP/BEST_NDCG5/BEST_FILE exports")
    args = parser.parse_args()
    candidates, initializers = collect(args.dev_dir)
    best = choose(candidates)
    if args.shell:
        for key, value in (("BEST_NAME", best["name"]), ("BEST_STEP", best["step"]),
                           ("BEST_NDCG5", best["ndcg@5"]), ("BEST_FILE", best["file"])):
            print(f"export {key}={shlex.quote(str(value))}")
        return
    print(json.dumps({"best": best, "candidates": candidates, "initializers": initializers},
                     indent=2))


if __name__ == "__main__":
    main()
