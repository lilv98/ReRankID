# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter teacher annotations and write the SID-only LLaMA-Factory SFT dataset (paper Sec. 5).

Each annotation is matched to its pool slate (same grades and positive index) and kept only if it
passes four quality gates (Table C.1): exact ten-index format, the positive at rank one,
explicit grounding to that candidate (the Chinese "candidate X" form), and a rationale of at most
600 characters. Answer-reveal audit hits are counted, not filtered. A kept example pairs the
SID-only prompt (ending in /think) with the teacher response: the rationale inside <think> tags,
then the JSON ranking. Train/dev membership comes from the hash-based split in sft/recipe.py, so
the dev users are fixed by the pool.

Inputs: data/sft/pool.parquet (build_pool.py --pool sft) and data/sft/annotations.jsonl
(annotate.py, or the released file). Outputs in data/sft/: train.json and dev.json (11,031 / 960
examples in the paper), dataset_info.json (sft_train, sft_dev) and stats.json.

    python code/sft/build_dataset.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
config.setup_env()

import argparse
import json
import re

import numpy as np

from sft import recipe
from sft.recipe import row_key
from utils.data_io import dump_json, load_rows
from utils.metrics import ndcg
from utils.prompts import TOP_K, build_context, render_candidates, rerank_instruction, rerank_system

MAX_COT_CHARS = 600


def dataset_info(train_file="train.json", dev_file="dev.json"):
    """LLaMA-Factory registry entries sft_train and sft_dev."""
    columns = {"prompt": "instruction", "response": "output", "system": "system"}
    return {
        "sft_train": {"file_name": train_file, "columns": columns},
        "sft_dev": {"file_name": dev_file, "columns": columns},
    }


def load_annotations(path):
    """Read JSONL, skipping only a final line cut short by an interrupted run."""
    rows, bad_lines = [], []
    with open(path) as f:
        lines = f.readlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad_lines.append(index)
    if bad_lines and bad_lines != [len(lines) - 1]:
        raise ValueError(f"annotation file has corrupt lines: {bad_lines[:5]}")
    return rows


def mentions_candidate(cot_text, idx):
    """Whether a rationale explicitly names candidate `idx` in the Chinese "candidate X" form."""
    pattern = rf"候选\s*[\[【(（]?\s*{int(idx)}(?!\d)\s*[\]】)）]?"
    return re.search(pattern, cot_text or "") is not None


def passes_filters(ann):
    """(ok, rejection reason): format, correctness, grounding, length, checked in that order."""
    order = ann.get("order")
    n_candidates = len(ann.get("grades") or [])
    strict_order = (
        ann.get("valid_annotation") is True
        and isinstance(order, list)
        and len(order) == TOP_K
        and len(set(order)) == TOP_K
        and all(isinstance(index, int) and not isinstance(index, bool)
                and 0 <= index < n_candidates for index in order)
    )
    if ann.get("n_listed") != TOP_K or not strict_order or not ann.get("cot_text"):
        return False, "format"
    if ndcg(order, ann["grades"], TOP_K) < 0.999:
        return False, "correctness"
    gold = [int(i) for i in ann.get("gold_idx", [])]
    if len(gold) != 1 or not mentions_candidate(ann["cot_text"], gold[0]):
        return False, "grounding"
    if len(ann["cot_text"]) > MAX_COT_CHARS:
        return False, "length"
    return True, None


def validate_pool_row(row):
    """Return the one positive index after validating a complete candidate slate."""
    n = int(row["n_cand"])
    pids = [int(pid) for pid in row["cand_pids"]]
    sids = list(row["cand_sids"])
    captions = list(row["cand_captions"])
    grades = [int(grade) for grade in row["cand_grades"]]
    if not (len(pids) == len(sids) == len(captions) == len(grades) == n):
        raise ValueError(f"candidate arrays differ in length for uid={row['uid']}")
    if len(pids) != len(set(pids)):
        raise ValueError(f"duplicate candidate PID for uid={row['uid']}")
    if not all(sids) or not all(captions):
        raise ValueError(f"unrenderable candidate for uid={row['uid']}")
    positive_indices = [i for i, grade in enumerate(grades) if grade > 0]
    if len(positive_indices) != 1:
        raise ValueError(f"expected one positive for uid={row['uid']}")
    gold_pids = [int(pid) for pid in row.get("gold_pids", [])]
    if len(gold_pids) != 1 or pids.count(gold_pids[0]) != 1:
        raise ValueError(f"gold PID must occur exactly once for uid={row['uid']}")
    if pids[positive_indices[0]] != gold_pids[0]:
        raise ValueError(f"positive grade and gold PID disagree for uid={row['uid']}")
    if int(row.get("n_pos", 1)) != 1:
        raise ValueError(f"n_pos is not one for uid={row['uid']}")
    return positive_indices


def validate_inputs(pool_by_key, annotations):
    """Annotation keys are unique pool keys whose grades and positive index match the pool."""
    if not annotations:
        raise ValueError("annotation file is empty")
    keys = [row_key(row) for row in annotations]
    if len(keys) != len(set(keys)):
        raise ValueError("annotation file has duplicate (uid, slate_id) keys")
    unknown = set(keys) - set(pool_by_key)
    if unknown:
        raise ValueError(f"{len(unknown)} annotations have no slate in the pool")
    for key, ann in zip(keys, annotations):
        pool_row = pool_by_key[key]
        if [int(g) for g in ann.get("grades", [])] != [int(g) for g in pool_row["cand_grades"]]:
            raise ValueError(f"annotation grades differ from pool for key={key}")
        if [int(i) for i in ann.get("gold_idx", [])] != validate_pool_row(pool_row):
            raise ValueError(f"annotation gold_idx differs from pool for key={key}")


def assemble_example(pool_row, ann):
    """Build one final SID-only rationale-plus-ranking example."""
    render = "sid"
    ctx, _ = build_context(pool_row, int(pool_row["k"]), render)
    instruction = (rerank_instruction(
        ctx, render_candidates(pool_row, render), render, top_k=TOP_K) + "/think")
    output = (
        "<think>\n"
        + ann["cot_text"].strip()
        + "\n</think>\n\n"
        + json.dumps(ann["order"])
    )
    return {"instruction": instruction, "output": output, "system": rerank_system(render)}


def count_field(rows, field):
    counts = {}
    for row in rows:
        value = str(int(row[field]))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def distribution(rows):
    return {name: count_field(rows, field)
            for name, field in (("positive_grade", "pos_grade"), ("k", "k"), ("n_cand", "n_cand"))}


def describe_lengths(values):
    if not values:
        return {"n": 0}
    arr = np.asarray(values)
    return {
        "n": len(values),
        "min": int(arr.min()),
        "mean": round(float(arr.mean()), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "max": int(arr.max()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool", default=config.SFT_POOL)
    ap.add_argument("--annotations", default=config.SFT_ANNOTATIONS)
    ap.add_argument("--outdir", default=config.SFT_DIR)
    args = ap.parse_args()

    pool_rows = load_rows(args.pool)
    pool_by_key = {row_key(row): row for row in pool_rows}
    if len(pool_by_key) != len(pool_rows):
        raise ValueError("pool has duplicate (uid, slate_id) keys")
    annotations = load_annotations(args.annotations)
    validate_inputs(pool_by_key, annotations)
    annotation_by_key = {row_key(row): row for row in annotations}
    keys = sorted(annotation_by_key)
    if len(keys) < len(pool_rows):
        print(f"note: {len(pool_rows) - len(keys)} pool slates have no annotation")
    dev_keys = recipe.dev_keys(pool_rows)

    rejects = {}
    train, dev, accepted = [], [], []
    for key in keys:
        ann = annotation_by_key[key]
        ok, reason = passes_filters(ann)
        if not ok:
            rejects[reason] = rejects.get(reason, 0) + 1
            continue
        accepted.append(key)
        (dev if key in dev_keys else train).append(assemble_example(pool_by_key[key], ann))
    train_uids = {uid for uid, slate in accepted if (uid, slate) not in dev_keys}
    dev_uids = {uid for uid, slate in accepted if (uid, slate) in dev_keys}
    if train_uids & dev_uids:
        raise AssertionError("train/dev user overlap")

    os.makedirs(args.outdir, exist_ok=True)
    dump_json(os.path.join(args.outdir, "train.json"), train, newline=False)
    dump_json(os.path.join(args.outdir, "dev.json"), dev, newline=False)
    dump_json(os.path.join(args.outdir, "dataset_info.json"), dataset_info(), newline=False)

    n_in = len(keys)
    stats = {
        "render": "sid",
        "pool_rows": len(pool_rows),
        "raw_annotations": n_in,
        "passed": len(accepted),
        "rejected": n_in - len(accepted),
        "pass_rate": round(len(accepted) / n_in, 6),
        "rejections": rejects,
        "answer_reveal_audit_hits": sum(bool(annotation_by_key[k].get("answer_reveal_audit_hit"))
                                        for k in keys),
        "answer_reveal_audit_hits_accepted": sum(
            bool(annotation_by_key[k].get("answer_reveal_audit_hit")) for k in accepted),
        "rationale_chars_all": describe_lengths(
            [len(annotation_by_key[k]["cot_text"]) for k in keys
             if annotation_by_key[k].get("cot_text")]),
        "rationale_chars_accepted": describe_lengths(
            [len(annotation_by_key[k]["cot_text"]) for k in accepted]),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_users": len(train_uids),
        "dev_users": len(dev_uids),
        "train_dev_user_overlap": 0,
        "pool_distribution": distribution([pool_by_key[k] for k in keys]),
        "accepted_distribution": distribution([pool_by_key[k] for k in accepted]),
        "teacher": {
            "teacher_model": recipe.TEACHER_MODEL,
            "backend": recipe.TEACHER_BACKEND,
            "effort": recipe.TEACHER_EFFORT,
            "temperature": recipe.TEACHER_TEMPERATURE,
            "thinking_level": recipe.TEACHER_THINKING_LEVEL,
            "max_tokens": recipe.MAX_TOKENS,
            "format_retries": recipe.FORMAT_RETRIES,
            "attempts": sum(int(annotation_by_key[k].get("attempts", 1) or 1) for k in keys),
            "format_retries_used": sum(int(annotation_by_key[k].get("format_retries_used", 0) or 0)
                                       for k in keys),
        },
    }
    dump_json(os.path.join(args.outdir, "stats.json"), stats)
    print(f"passed {len(accepted)}/{n_in}; rejections {rejects}; "
          f"answer-reveal audit hits {stats['answer_reveal_audit_hits']}")
    print(f"wrote {args.outdir}/train.json n={len(train)}, dev.json n={len(dev)}, "
          f"dataset_info.json, stats.json")
    if train:
        print(f"\n=== example ===\ninstruction[:200]: {train[0]['instruction'][:200]}")
        print(f"output: {train[0]['output']}")


if __name__ == "__main__":
    main()
