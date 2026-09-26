"""Strict single-positive NDCG@5 verifier for GRPO RLVR.

The response must close its reasoning block and then emit exactly ten distinct,
in-range candidate indices in one JSON array.  A valid response receives the
benchmark's exact single-positive NDCG@5.  NDCG@10, Hit@1, identity, and format
fields are diagnostics only and never alter the optimized score.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.contract import REWARD_CUTOFF, TOP_K
from utils.metrics import is_identity_collapse, parse_topk_diag

_DBG = os.environ.get("REWARD_DEBUG")
_DBG_N = [0]


def _grades_from_gt(ground_truth):
    gt = ground_truth if isinstance(ground_truth, dict) else json.loads(ground_truth)
    grades = [int(value) for value in gt["grades"]]
    n_cand = int(gt.get("n_cand", len(grades)))
    if len(grades) != n_cand or sum(value > 0 for value in grades) != 1:
        raise ValueError("RLVR ground truth must contain exactly one positive")
    return grades, n_cand


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
        "ndcg5": float(score),
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


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    del data_source, extra_info
    text = solution_str or ""
    if _DBG and _DBG_N[0] < 6:
        _DBG_N[0] += 1
        with open(_DBG, "a") as handle:
            handle.write(
                f"\n===== sample {_DBG_N[0]} (len={len(text)}) =====\n{text!r}\n"
            )
    # Ground truth is a frozen, trusted input.  Fail the run on corruption
    # instead of silently turning a bad training row into an all-zero group.
    grades, n_cand = _grades_from_gt(ground_truth)

    if "</think>" not in text:
        return _result()
    answer = text.rsplit("</think>", 1)[-1]
    parsed_order, n_listed, diag = parse_topk_diag(answer, n_cand, TOP_K)
    has_array = float(diag["has_array"])
    wrong_length = float(diag["has_array"] and diag["arr_len"] != TOP_K)
    out_of_range = float(diag["n_oob"] > 0)
    duplicate = float(diag["n_dup"] > 0)
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
        return _result(
            closed_think=1.0, has_array=has_array,
            wrong_length=wrong_length, out_of_range=out_of_range,
            duplicate=duplicate,
        )

    order = raw_answer
    positive = next(index for index, grade in enumerate(grades) if grade > 0)
    rank = order.index(positive) + 1 if positive in order else 0
    score = (
        1.0 / math.log2(1 + rank)
        if rank and rank <= REWARD_CUTOFF
        else 0.0
    )
    return _result(
        score=score,
        rank=rank,
        parse_ok=1.0,
        closed_think=1.0,
        identity=float(is_identity_collapse(order, TOP_K)),
        has_array=1.0,
    )


if __name__ == "__main__":
    gt = json.dumps({"grades": [0, 0, 0, 3] + [0] * 26, "n_cand": 30})
    for answer in (
        "<think>r</think>\n[3,0,1,2,4,5,6,7,8,9]",
        "<think>r</think>\n[0,1,2,4,3,5,6,7,8,9]",
        "<think>r</think>\n[0,1,2,4,5,3,6,7,8,9]",
        "no closed reasoning [3,0,1,2,4,5,6,7,8,9]",
    ):
        print(compute_score("listwise_rerank_ndcg5", answer, gt))
