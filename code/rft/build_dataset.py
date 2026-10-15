# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""RFT continuation dataset from best-of-8 rollouts (paper Sec. 5 and App. C.2, Table C.3).

For each prompt of the 6k subset, keeps the first rollout that is a valid, completed top-10 answer
with a nonempty rationale, puts the single positive at rank one, and is not the identity order.
At most one trace per prompt; no teacher traces, regex or length filters. Reads
data/rft/<model>/rollouts.jsonl (rft/sample.py) and writes the LLaMA-Factory files train.json,
dataset_info.json (train plus the unchanged SFT dev set) and stats.json to data/rft/<model>/.

    python code/rft/build_dataset.py --model onerec-1.7b
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config  # noqa: E402

config.setup_env()

import argparse  # noqa: E402
import json  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

from rft.recipe import MODELS  # noqa: E402
from utils.data_io import dump_json, load_rows  # noqa: E402
from utils.metrics import is_identity_collapse  # noqa: E402
from utils.pools import row_key, select_subset  # noqa: E402
from utils.prompts import TOP_K  # noqa: E402

SELECTION = "first valid nonidentity rank1 rollout"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def valid_order(order, n_cand):
    return (
        isinstance(order, list)
        and len(order) == TOP_K
        and len(set(order)) == TOP_K
        and all(isinstance(index, int) and not isinstance(index, bool)
                and 0 <= index < n_cand for index in order)
    )


def first_survivor(record, pool_row, rejection_counts):
    """The first rollout passing the gate, as ({cot_text, order}, rollout); counts rejections."""
    grades = [int(value) for value in pool_row["cand_grades"]]
    gold = [index for index, value in enumerate(grades) if value > 0]
    if len(gold) != 1:
        raise ValueError(f"pool row does not have one positive: {row_key(pool_row)}")
    if [int(value) for value in record.get("grades") or []] != grades:
        raise ValueError(f"sample grades differ from pool: {row_key(pool_row)}")
    if [int(value) for value in record.get("gold_idx") or []] != gold:
        raise ValueError(f"sample gold index differs from pool: {row_key(pool_row)}")
    for rollout in record.get("rollouts") or []:
        order = rollout.get("order")
        if not valid_order(order, len(grades)) or rollout.get("n_listed") != TOP_K:
            rejection_counts["format_or_unclosed"] += 1
            continue
        if rollout.get("finish_reason") == "length":
            rejection_counts["length_truncated"] += 1
            continue
        cot = (rollout.get("cot_text") or "").strip()
        if not cot:
            rejection_counts["empty_rationale"] += 1
            continue
        if order[0] != gold[0]:
            rejection_counts["not_rank1"] += 1
            continue
        if is_identity_collapse(order, TOP_K):
            rejection_counts["identity"] += 1
            continue
        return {"cot_text": cot, "order": order}, rollout
    return None, None


def distribution(rows, field):
    values = Counter(str(row[field]) for row in rows)
    return dict(sorted(values.items(), key=lambda item: int(item[0])))


def coverage_table(source_rows, retained_keys, field):
    grouped = defaultdict(lambda: [0, 0])
    for row in source_rows:
        key = str(row[field])
        grouped[key][0] += 1
        grouped[key][1] += row_key(row) in retained_keys
    return {
        key: {"prompts": total, "retained": kept, "coverage": round(kept / total, 4)}
        for key, (total, kept) in sorted(grouped.items(), key=lambda item: int(item[0]))
    }


def coverage_combo(source_rows, retained_keys):
    grouped = defaultdict(lambda: [0, 0])
    for row in source_rows:
        key = int(row["k"]), int(row["n_cand"])
        grouped[key][0] += 1
        grouped[key][1] += row_key(row) in retained_keys
    return {
        f"k{k}_N{n}": {"prompts": total, "retained": kept, "coverage": round(kept / total, 4)}
        for (k, n), (total, kept) in sorted(grouped.items())
    }


def summary(values):
    return {"min": min(values), "mean": round(sum(values) / len(values), 1), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--pool", default=config.POSTTRAIN_POOL)
    parser.add_argument("--rollouts", default=None, help="default: data/rft/<model>/rollouts.jsonl")
    parser.add_argument("--outdir", default=None, help="default: data/rft/<model>")
    args = parser.parse_args()
    outdir = args.outdir or config.rft_dir(args.model)
    rollouts_path = args.rollouts or os.path.join(config.rft_dir(args.model), "rollouts.jsonl")

    subset = select_subset(load_rows(args.pool))
    records = {}
    for record in load_jsonl(rollouts_path):
        key = row_key(record)
        if key in records:
            raise ValueError(f"duplicate rollout record for {key}")
        records[key] = record
    subset_keys = {row_key(row) for row in subset}
    if not set(records) <= subset_keys:
        raise ValueError("rollouts contain prompts outside the 6k subset")
    source_rows = [row for row in subset if row_key(row) in records]
    if len(source_rows) < len(subset):
        print(f"note: rollouts cover {len(source_rows):,} of {len(subset):,} subset prompts")

    from sft.build_dataset import assemble_example
    rejections = Counter()
    examples, kept_rows, kept_rollouts = [], [], []
    for row in source_rows:
        annotation, rollout = first_survivor(records[row_key(row)], row, rejections)
        if annotation is None:
            continue
        examples.append(assemble_example(row, annotation))
        kept_rows.append(row)
        kept_rollouts.append(rollout)
    if not examples:
        raise ValueError("no rollout passed the RFT gate")

    retained_keys = {row_key(row) for row in kept_rows}
    os.makedirs(outdir, exist_ok=True)
    train_file = os.path.join(outdir, "train.json")
    info_file = os.path.join(outdir, "dataset_info.json")
    dump_json(train_file, examples, newline=False)
    columns = {"prompt": "instruction", "response": "output", "system": "system"}
    dump_json(info_file, {
        "rft_train": {"file_name": "train.json", "columns": columns},
        "sft_dev": {"file_name": os.path.relpath(os.path.join(config.SFT_DIR, "dev.json"), outdir),
                    "columns": columns},
    })
    stats = {
        "model": args.model,
        "target_prompts": len(source_rows),
        "retained_examples": len(examples),
        "prompt_coverage": round(len(examples) / len(source_rows), 4),
        "one_trace_per_prompt": True,
        "selection": SELECTION,
        "rejected_rollouts_before_first_survivor": dict(sorted(rejections.items())),
        "source_distribution": {f: distribution(source_rows, f) for f in ("k", "n_cand", "pos_grade")},
        "retained_distribution": {f: distribution(kept_rows, f) for f in ("k", "n_cand", "pos_grade")},
        "coverage_by": {
            "k": coverage_table(source_rows, retained_keys, "k"),
            "n_cand": coverage_table(source_rows, retained_keys, "n_cand"),
            "pos_grade": coverage_table(source_rows, retained_keys, "pos_grade"),
            "k_x_n": coverage_combo(source_rows, retained_keys),
        },
        "retained_rationale_chars": summary([len(r["cot_text"]) for r in kept_rollouts]),
        "retained_output_tokens": summary([int(r["output_tokens"]) for r in kept_rollouts]),
    }
    dump_json(os.path.join(outdir, "stats.json"), stats)
    print(json.dumps({k: v for k, v in stats.items() if k != "coverage_by"}, indent=2))
    print(f"wrote {len(examples):,} examples to {train_file}")


if __name__ == "__main__":
    main()
