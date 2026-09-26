"""Label-free rollout diagnostics for an OPD smoke or formal run.

This script never reads ranking labels and does not participate in training.
It measures whether the student reaches the benchmark interface while OPD
progresses: a closed reasoning block followed by ten distinct, in-range
candidate indices.
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from opd.contract import MAX_RESPONSE_LENGTH, STUDENT, TOP_K, rooted
from utils import config

NEAR_CAP_TOKENS = MAX_RESPONSE_LENGTH - 12


def candidate_count(prompt):
    matches = re.findall(r"取值\s*0\s*到\s*(\d+)", prompt or "")
    if not matches:
        raise ValueError("could not recover candidate range from OPD prompt")
    return int(matches[-1]) + 1


def strict_answer(output, n_cand):
    if "</think>" not in output:
        return False
    answer = output.rsplit("</think>", 1)[-1].strip()
    try:
        values = json.loads(answer)
    except (json.JSONDecodeError, TypeError):
        return False
    return (
        isinstance(values, list)
        and len(values) == TOP_K
        and all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value < n_cand
            for value in values
        )
        and len(set(values)) == TOP_K
    )


def summarize(rows, tokenizer):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["step"])].append(row)

    result = {}
    for step, current in sorted(grouped.items()):
        outputs = [str(row.get("output", "")) for row in current]
        # The rollout dump stores decoded text rather than original token IDs.
        # Re-encoding can differ by a token, so use a documented near-cap band.
        token_lengths = [
            len(
                tokenizer(
                    output,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
            )
            for output in outputs
        ]
        closed = ["</think>" in output for output in outputs]
        strict = [
            strict_answer(output, candidate_count(row["input"]))
            for row, output in zip(current, outputs)
        ]
        sid_after_think = [
            "<|sid_begin|>"
            in (
                output.rsplit("</think>", 1)[-1]
                if "</think>" in output
                else output
            )
            for output in outputs
        ]
        count = len(current)
        result[str(step)] = {
            "rows": count,
            "unique_inputs": len({row["input"] for row in current}),
            "closed_think_rate": round(sum(closed) / count, 4),
            "strict_answer_rate": round(sum(strict) / count, 4),
            "near_response_cap_rate": round(
                sum(length >= NEAR_CAP_TOKENS for length in token_lengths)
                / count,
                4,
            ),
            "near_cap_threshold_tokens": NEAR_CAP_TOKENS,
            "sid_answer_rate": round(sum(sid_after_think) / count, 4),
            "response_tokens": {
                "min": min(token_lengths),
                "mean": round(sum(token_lengths) / count, 2),
                "max": max(token_lengths),
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        help="rollout JSONL files or directories containing JSONL files",
    )
    args = parser.parse_args()

    paths = []
    for value in args.paths:
        if os.path.isdir(value):
            paths.extend(glob.glob(os.path.join(value, "*.jsonl")))
        else:
            paths.extend(glob.glob(value))
    paths = sorted(set(os.path.abspath(path) for path in paths))
    if not paths:
        raise FileNotFoundError("no rollout JSONL files found")

    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    tokenizer = AutoTokenizer.from_pretrained(
        rooted(config.ROOT, STUDENT["base_rel"]),
        trust_remote_code=True,
        local_files_only=True,
    )
    report = {
        "files": paths,
        "rows": len(rows),
        "all_unique_inputs": len({row["input"] for row in rows}),
        "by_step": summarize(rows, tokenizer),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
