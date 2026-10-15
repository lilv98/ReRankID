# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Anchor result files as judgeable arms: rationales, rankings and their benchmark bindings.

Shared by pairwise_judge.py and pairwise_controls.py (App. D.4). Nothing here calls a
model and nothing is regenerated: every arm is read from a result JSON that the benchmark scorers
wrote at the anchor cell (test slice, k=64, N=30, display seed 0), and the rationale is the exact
<think> block that produced the ranking the paper reports.

Run as a script, it counts the completed rationales per arm (the "Systems compared" paragraph):

    python code/judge/judge_arms.py --results downloads/rerankid/results
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config

config.setup_env()

from utils.metrics import kendall_tau_topk
from utils.prompts import TOP_K


ROOT = Path(config.ROOT)
RELEASE_RESULTS = ROOT / "downloads/rerankid/results"

# The two arms of the paper's rationale audit, by the labels stored in the released judge outputs.
JUDGED_ARMS = {
    "gemini-3.6-flash": "diagnosis/gemini-3.6-flash/k64_n30.json",
    "rlvr-ndcg5-8b": "posttraining/rlvr-ndcg5/onerec-8b/test/k64_n30.json",
}

# Arms counted by the CLI: the two base models, the RLVR (NDCG@5) models and the text reference.
COUNTED_ARMS = {
    "base-1.7b": "diagnosis/onerec-1.7b/sid/k64_n30.json",
    "base-8b": "diagnosis/onerec-8b/sid/k64_n30.json",
    "rlvr-ndcg5-1.7b": "posttraining/rlvr-ndcg5/onerec-1.7b/test/k64_n30.json",
    "rlvr-ndcg5-8b": JUDGED_ARMS["rlvr-ndcg5-8b"],
    "gemini-3.6-flash": JUDGED_ARMS["gemini-3.6-flash"],
}

# Every arm must have been scored in this cell on all test users (load_arm checks the user set).
# Render, checkpoint and model differ by design.
ANCHOR = {"slice": "test", "n_cand": 30, "shuffle": 0, "k": 64, "top_k": TOP_K}

# score_sid.py keeps the stage-1 text in raw1; score_text.py stores the whole response in raw.
RATIONALE_FIELDS = ("raw1", "raw")

# prompts.rerank_instruction asks every arm to refer to items by the candidate marker matched here.
CANDIDATE_MENTION = re.compile(r"候选\s*(\d{1,3})")

# A rationale ends with the ranking array or with a closing tag of any name (</think>, </text>, ...).
_TRAILING_TAG = re.compile(r"(?:\s*</\s*[A-Za-z_][\w\-]*\s*>)+\s*$")
_FINAL_ARRAY = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*$")


def json_object(text):
    """Extract one JSON object from a model response."""
    if not text:
        raise ValueError("empty response")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    return json.loads(cleaned[start : end + 1])


def mention_order(rationale, n_candidates):
    """Candidate indices in the order the rationale first names them, without out-of-slate ones."""
    seen = []
    for match in CANDIDATE_MENTION.findall(rationale or ""):
        index = int(match)
        if 0 <= index < n_candidates and index not in seen:
            seen.append(index)
    return seen


def rationale_ranking_stats(rationale, ranking, n_candidates):
    """Coverage and mention-order fidelity, computed from the text rather than judged.

    Coverage is the share of the ranked candidates the rationale names. Fidelity is Kendall tau
    between mention order and ranking, restricted to candidates that are both named and ranked, so
    that coverage does not leak into it.
    """
    mentioned = mention_order(rationale, n_candidates)
    in_ranking = [i for i in mentioned if i in ranking]
    kept = set(in_ranking)
    ranked_subset = [i for i in ranking if i in kept]
    return {
        "n_mentioned": len(mentioned),
        "n_mentioned_in_ranking": len(in_ranking),
        "coverage_of_ranking": round(len(in_ranking) / len(ranking), 4) if ranking else None,
        "mention_vs_rank_tau": (
            round(kendall_tau_topk(in_ranking, ranked_subset, len(ranked_subset)), 4)
            if len(in_ranking) >= 2 else None
        ),
    }


def recorded_ranking(row, n_candidates, top_k=TOP_K):
    """The ranking a scorer recorded, plus whether it is a complete top_k list.

    A short list is a legitimate failure of the arm on that user (the parser dropped an out-of-range
    or duplicate index). A duplicate, an out-of-slate index or a length that disagrees with n_listed
    means the file does not describe the ranking its metrics were computed from, and raises.
    """
    value = row.get("order")
    if not isinstance(value, list):
        raise ValueError("ranking is not a list")
    ranking = [int(x) for x in value]
    if len(set(ranking)) != len(ranking):
        raise ValueError("ranking contains a duplicate index")
    if any(x < 0 or x >= n_candidates for x in ranking):
        raise ValueError(f"ranking contains an index outside 0..{n_candidates - 1}")
    if len(ranking) > top_k:
        raise ValueError(f"ranking is longer than top_k={top_k}")
    if row.get("n_listed") != len(ranking):
        raise ValueError("n_listed disagrees with the recorded ranking")
    return ranking, len(ranking) == top_k


def parse_rationale(raw):
    """Split a stage-1 output into its visible reasoning and whether that reasoning finished.

    Reads from the last opening <think>, then drops the trailing ranking array and any closing tags.
    The rationale is complete only if it was terminated by a tag or by the array. Base models often
    spend the whole stage-1 budget transcribing SID tokens and are cut off; such text is incomplete
    and is not judged. A closed but empty block is complete and empty.
    """
    if not raw or "<think>" not in raw:
        return {"text": "", "complete": False, "terminator": None}
    body = raw.rsplit("<think>", 1)[1]
    terminator = None
    array = _FINAL_ARRAY.search(body)
    if array:
        body, terminator = body[: array.start()], "array"
    if _TRAILING_TAG.search(body):
        body, terminator = _TRAILING_TAG.sub("", body), "tag"
    return {"text": body.strip(), "complete": terminator is not None, "terminator": terminator}


def completed_rationale(row):
    """Whether stage 1 wrote a nonempty, terminated rationale (the "Systems compared" count).

    SID result rows keep only the first 400 characters of stage 1, so termination comes from the
    recorded s1_closed_think, and a block counts as empty only when its closing tag is visible
    with nothing before it. Text-reference rows keep the full response; any text besides the
    tags and the final ranking counts.
    """
    if "s1_closed_think" in row:
        head = row["raw1"].split("</think>")[0] if "</think>" in row["raw1"] else None
        return row["s1_closed_think"] and (head is None or head.replace("<think>", "").strip() != "")
    text = re.sub(r"<[^>]*>", "", row.get("raw") or "").strip()
    return _FINAL_ARRAY.sub("", text).strip() != ""


def usage(record):
    raw = record.get("usage") or {}
    return {
        "prompt_tokens": int(raw.get("promptTokenCount", 0) or 0),
        "output_tokens": int(raw.get("candidatesTokenCount", 0) or 0),
        "reasoning_tokens": int(raw.get("thoughtsTokenCount", 0) or 0),
    }


def add_usage(total, value):
    for key in ("prompt_tokens", "output_tokens", "reasoning_tokens"):
        total[key] += int(value.get(key, 0) or 0)


def load_arm(path, benchmark_rows):
    """Load one anchor result and bind every row to the benchmark.

    Each row must carry the benchmark grades and slate id, the k=64 history and a well-formed
    ranking; the user set must equal the benchmark's.
    """
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("meta"), dict):
        raise ValueError(f"result file has no metadata object: {path}")
    meta = payload["meta"]
    for field, expected in ANCHOR.items():
        if meta.get(field) != expected:
            raise ValueError(f"{path.name}: {field}={meta.get(field)!r} is not the anchor cell "
                             f"(expected {expected!r})")

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError(f"result file has no result list: {path}")
    benchmark = {}
    for row in benchmark_rows:
        uid = int(row["uid"])
        if uid in benchmark:
            raise ValueError(f"benchmark contains duplicate uid {uid}")
        benchmark[uid] = row

    indexed = {}
    for row in raw_results:
        if not isinstance(row, dict) or "uid" not in row:
            raise ValueError("result row is not a UID-keyed object")
        uid = int(row["uid"])
        if uid in indexed:
            raise ValueError(f"result contains duplicate uid {uid}")
        if uid not in benchmark:
            raise ValueError(f"result uid {uid} is absent from the benchmark")
        source = benchmark[uid]
        if row.get("grades") != [int(value) for value in source["cand_grades"]]:
            raise ValueError(f"result uid {uid} grades differ from the benchmark")
        if "slate_id" in row and str(row["slate_id"]) != str(source["slate_id"]):
            raise ValueError(f"result uid {uid} slate_id differs from the benchmark")
        if row.get("n_hist_kept") != ANCHOR["k"]:
            raise ValueError(f"result uid {uid} does not use the k={ANCHOR['k']} history")
        try:
            ranking, complete = recorded_ranking(row, len(source["cand_grades"]))
        except ValueError as exc:
            raise ValueError(f"result uid {uid}: {exc}") from exc
        row = dict(row)
        row["order"] = ranking
        row["ranking_complete"] = complete
        indexed[uid] = row
    if set(indexed) != set(benchmark):
        missing = sorted(set(benchmark) - set(indexed))
        raise ValueError(f"{path.name}: UID set differs from the benchmark; missing={missing[:8]}")
    incomplete = sorted(uid for uid, row in indexed.items() if not row["ranking_complete"])
    return {
        "path": path,
        "meta": meta,
        "rows": indexed,
        "incomplete_ranking_uids": incomplete,
        "source": {
            "result_file": config.relpath(path),
            "model": meta.get("model"),
            "checkpoint": meta.get("checkpoint"),
            "render": meta.get("render"),
            "arm": meta.get("arm"),
            "emit_rationale": meta.get("emit_rationale"),
            "n_users": len(indexed),
            "n_incomplete_ranking": len(incomplete),
        },
    }


def frozen_output(result_row):
    """The rationale and ranking this arm produced, verbatim."""
    raw = next((result_row[field] for field in RATIONALE_FIELDS if result_row.get(field)), "")
    parsed = parse_rationale(raw)
    detail = {"finish": result_row.get("finish") or "frozen_result",
              "rationale_terminator": parsed["terminator"]}
    for field in ("stage", "s1_closed_think", "s1_valid"):
        if field in result_row:
            detail[field] = result_row[field]
    return {
        "rationale": parsed["text"],
        "rationale_available": bool(parsed["text"]) and parsed["complete"],
        "ranking": [int(x) for x in result_row["order"]],
        "detail": detail,
    }


def select_rows(rows, arms, n, seed):
    """Sample n users that every arm ranked completely, and report the users the filter removed.

    The comparison is paired, so a user with a short ranking in any arm is dropped from all arms.
    """
    dropped = {
        arm["source"]["result_file"]: arm["incomplete_ranking_uids"]
        for arm in arms
        if arm["incomplete_ranking_uids"]
    }
    excluded = {uid for uids in dropped.values() for uid in uids}
    eligible = [
        row for row in rows
        if int(row["uid"]) not in excluded
        and all(int(row["uid"]) in arm["rows"] for arm in arms)
    ]
    if n > len(eligible):
        raise ValueError(f"requested {n} users from only {len(eligible)} eligible")
    import numpy as np

    rng = np.random.default_rng(seed)
    chosen = set(int(x) for x in rng.choice(len(eligible), size=n, replace=False))
    return [row for i, row in enumerate(eligible) if i in chosen], {
        "n_eligible": len(eligible),
        "n_excluded_incomplete_ranking": len(excluded),
        "excluded_uids_by_arm": dropped,
    }


def parse_arm(spec):
    """Parse a LABEL=PATH command-line argument."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--arm expects LABEL=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"--arm has an empty label: {spec!r}")
    return label, Path(path.strip()).expanduser()


def resolve_arms(specs, results, default):
    """Explicit LABEL=PATH arms, or the default arms under a results root."""
    if specs:
        return [parse_arm(spec) for spec in specs]
    return [(label, results / relative) for label, relative in default.items()]


def main():
    parser = argparse.ArgumentParser(description="Count completed anchor rationales per arm.")
    parser.add_argument("--results", type=Path, default=RELEASE_RESULTS,
                        help="results root holding the default arms")
    parser.add_argument("--bench", default=config.BENCH)
    parser.add_argument("--arm", action="append", metavar="LABEL=PATH",
                        help="anchor result JSON to count (repeatable; default: COUNTED_ARMS "
                             "under --results)")
    args = parser.parse_args()

    from utils.data_io import load_bench

    bench = load_bench(args.bench, n_cand=ANCHOR["n_cand"], shuffle=ANCHOR["shuffle"],
                       bench_slice=ANCHOR["slice"])
    for label, path in resolve_arms(args.arm, args.results, COUNTED_ARMS):
        arm = load_arm(path, bench)
        done = sum(completed_rationale(row) for row in arm["rows"].values())
        print(f"{label:20s} {done:4d} / {len(arm['rows'])} complete rationales")


if __name__ == "__main__":
    main()
