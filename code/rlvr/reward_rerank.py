# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Verifiable rewards for GRPO (paper Sec. 5, App. C.4).

A valid response closes its reasoning block and then emits only one JSON array of exactly ten
distinct, in-range candidate indices. compute_score (rlvr-ndcg5) returns the benchmark's
single-positive NDCG@5, 1/log2(1+r) for the positive at rank r <= 5. compute_score_hit1
(rlvr-hit1, the reward ablation) returns 1 only at rank 1. Every other response scores 0.
The remaining returned fields (true NDCG@5, NDCG@10, Hit@1, rank, identity, format flags) are
logged diagnostics and never change the optimized score.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import is_identity_collapse, parse_topk_diag  # noqa: E402
from utils.prompts import TOP_K  # noqa: E402

REWARD_CUTOFF = 5


def _grades_from_gt(ground_truth):
    gt = ground_truth if isinstance(ground_truth, dict) else json.loads(ground_truth)
    grades = [int(value) for value in gt["grades"]]
    n_cand = int(gt.get("n_cand", len(grades)))
    if len(grades) != n_cand or sum(value > 0 for value in grades) != 1:
        raise ValueError("RLVR ground truth must contain exactly one positive")
    return grades, n_cand


def _ndcg5(rank):
    return 1.0 / math.log2(1 + rank) if rank and rank <= REWARD_CUTOFF else 0.0


def _result(score=0.0, rank=0, parse_ok=0.0, closed_think=0.0,
            identity=0.0, has_array=0.0, wrong_length=0.0,
            out_of_range=0.0, duplicate=0.0):
    ndcg10 = 0.0
    hit1 = 0.0
    if rank:
        ndcg10 = 1.0 / math.log2(1 + rank) if rank <= TOP_K else 0.0
        hit1 = float(rank == 1)
    return {
        "score": float(score),
        "ndcg5": float(_ndcg5(rank)),
        "ndcg10": float(ndcg10),
        "hit1": hit1,
        "rank_pos": float(rank),
        "parse_ok": float(parse_ok),
        "closed_think": float(closed_think),
        "identity": float(identity),
        "has_array": float(has_array),
        "wrong_length": float(wrong_length),
        "out_of_range": float(out_of_range),
        "duplicate": float(duplicate),
    }


def _verify(solution_str, ground_truth):
    """Check the output contract. Returns (order, rank, fields): order is the valid ten-index list
    or None, rank the 1-based position of the positive in it (0 if absent), and fields the
    format diagnostics for an invalid response."""
    text = solution_str or ""
    # A malformed ground truth raises instead of silently scoring a whole group 0.
    grades, n_cand = _grades_from_gt(ground_truth)
    if "</think>" not in text:
        return None, 0, {}
    answer = text.rsplit("</think>", 1)[-1]
    parsed_order, n_listed, diag = parse_topk_diag(answer, n_cand, TOP_K)
    try:
        raw_answer = json.loads(answer.strip())
    except (json.JSONDecodeError, TypeError):
        raw_answer = None
    answer_only_array = (
        isinstance(raw_answer, list)
        and all(isinstance(value, int) and not isinstance(value, bool)
                for value in raw_answer)
    )
    strict_valid = (
        answer_only_array
        and parsed_order is not None
        and n_listed == TOP_K
        and diag["arr_len"] == TOP_K
        and diag["n_oob"] == 0
        and diag["n_dup"] == 0
    )
    if not strict_valid:
        return None, 0, {
            "closed_think": 1.0,
            "has_array": float(diag["has_array"]),
            "wrong_length": float(diag["has_array"] and diag["arr_len"] != TOP_K),
            "out_of_range": float(diag["n_oob"] > 0),
            "duplicate": float(diag["n_dup"] > 0),
        }
    order = raw_answer
    positive = next(index for index, grade in enumerate(grades) if grade > 0)
    rank = order.index(positive) + 1 if positive in order else 0
    return order, rank, {}


def _score(solution_str, ground_truth, reward):
    order, rank, fields = _verify(solution_str, ground_truth)
    if order is None:
        return _result(**fields)
    return _result(
        score=reward(rank),
        rank=rank,
        parse_ok=1.0,
        closed_think=1.0,
        identity=float(is_identity_collapse(order, TOP_K)),
        has_array=1.0,
    )


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """rlvr-ndcg5: single-positive NDCG@5 of a valid response, else 0."""
    del data_source, extra_info
    return _score(solution_str, ground_truth, _ndcg5)


def compute_score_hit1(data_source, solution_str, ground_truth, extra_info=None):
    """rlvr-hit1: 1 if a valid response ranks the positive first, else 0."""
    del data_source, extra_info
    return _score(solution_str, ground_truth, lambda rank: 1.0 if rank == 1 else 0.0)


if __name__ == "__main__":
    gt = json.dumps({"grades": [0, 0, 0, 3] + [0] * 26, "n_cand": 30})
    for answer in (
        "<think>r</think>\n[3,0,1,2,4,5,6,7,8,9]",
        "<think>r</think>\n[0,1,2,4,3,5,6,7,8,9]",
        "<think>r</think>\n[0,1,2,4,5,3,6,7,8,9]",
        "no closed reasoning [3,0,1,2,4,5,6,7,8,9]",
    ):
        ndcg5 = compute_score("listwise_rerank_ndcg5", answer, gt)["score"]
        hit1 = compute_score_hit1("listwise_rerank_ndcg5", answer, gt)["score"]
        print(f"ndcg5={ndcg5:.4f} hit1={hit1:.0f}  {answer!r}")
