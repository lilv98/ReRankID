"""Reading frozen anchor results as judgeable arms: rationales, rankings, and their bindings.

Shared by the pairwise judge and its controls.  Nothing here calls a model.

Nothing is re-generated either.  Every arm is read from a result JSON that the ordinary benchmark
scorers produced at the anchor cell (test / k=64 / N=30 / shuffle 0), and the rationale exposed is
the exact ``<think>`` block that arm emitted when it produced the ranking the paper reports.
Regenerating a rationale with a bespoke prompt would score a system the paper never reports; reusing
the frozen output means the judged text and the measured NDCG came from one call.

Two things this module is careful about, because both were wrong once:

* Whether an arm FINISHED its reasoning, not merely whether it emitted text -- see parse_rationale.
* Whether a short recorded ranking is an arm failing or a file being corrupt -- see recorded_ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config

config.setup_env()

from utils.prompts import TOP_K


ROOT = Path(__file__).resolve().parents[2]

# The one cell every arm must have been scored in. An arm evaluated anywhere else is a different
# system under this comparison, not the same system with a different rationale. Arm-specific meta
# (render, checkpoint, model, backend) is expected to differ -- that is what is being compared.
ANCHOR = {"slice": "test", "n": None, "n_cand": 30, "shuffle": 0, "k": 64, "top_k": TOP_K}

# Where each scorer parks the stage-1 text. score_sid.py runs two stages and keeps them apart;
# score_text.py makes one call and stores the whole response. Both wrap reasoning in <think>.
RATIONALE_FIELDS = ("raw1", "raw")

def _json_object(text: str) -> dict:
    """Extract one JSON object from a model response and reject trailing prose."""
    if not text:
        raise ValueError("empty response")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    return json.loads(cleaned[start : end + 1])


def _validate_ranking(value, n_candidates: int, top_k: int = TOP_K) -> list[int]:
    if not isinstance(value, list):
        raise ValueError("ranking is not a list")
    ranking = [int(x) for x in value]
    if (
        len(ranking) != top_k
        or len(set(ranking)) != top_k
        or any(x < 0 or x >= n_candidates for x in ranking)
    ):
        raise ValueError("ranking must contain ten distinct in-range indices")
    return ranking


# Every arm is instructed by prompts.rerank_instruction to refer to items as 「候选N」, so this reads
# back the arm's own referencing convention rather than imposing one. Verified to fire on 94% of
# gemini rationales and 100% of RLVR-8B ones at the anchor.
CANDIDATE_MENTION = re.compile(r"候选\s*(\d{1,3})")


def mention_order(rationale: str, n_candidates: int) -> list[int]:
    """Candidate indices in the order the rationale first names them.

    Out-of-slate indices are dropped: those are the arm miscounting, not a reference to a candidate.
    """
    seen = []
    for match in CANDIDATE_MENTION.findall(rationale or ""):
        index = int(match)
        if 0 <= index < n_candidates and index not in seen:
            seen.append(index)
    return seen


def rationale_ranking_stats(rationale: str, ranking: list[int], n_candidates: int) -> dict:
    """Coverage and mention-order fidelity, computed rather than judged.

    An LLM judge asked to compare two rationales on ranking fidelity agreed with itself only 35% of
    the time across presentation orders -- barely above chance for a three-way choice -- so this
    aspect is measured directly. Both quantities are observable: the mention order comes from the
    rationale text and the ranking is recorded in the result file.

    The tau is restricted to candidates the rationale actually discusses. Including unmentioned ones
    (tied at the bottom) would fold coverage back into the fidelity number, and separating those two
    is the whole point: a terse rationale that argues three candidates in exactly the right order is
    perfectly faithful over what it claims, and separately covers little.
    """
    from benchmark.stability_audit import kendall_tau_topk

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


def recorded_ranking(row: dict, n_candidates: int, top_k: int = TOP_K) -> tuple[list[int], bool]:
    """The ranking a scorer recorded, plus whether it is a complete top_k list.

    A scorer legitimately records fewer than top_k indices when the arm emitted an out-of-range or
    duplicate index and the parser dropped it (score_text keeps the evidence in diag.n_oob/n_dup).
    That is a failure of that arm on that user, not a corrupt result file, so it loads with the flag
    cleared. Anything else -- a surviving duplicate, an index outside the slate, a length that
    disagrees with the scorer's own n_listed -- means the file no longer describes the ranking its
    published metrics were computed from, and raises.
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


# A rationale is delimited by an opening <think> and, at the other end, whatever the arm chose to put
# there. Requiring a literal </think> silently discarded 34 of gemini's 400 anchor rows, which closed
# the block with </text>, </script> or nothing at all -- complete rationales at normal token counts,
# not truncations. Those rows were not a random 8.5%: their NDCG@5 averaged 0.322 against 0.240 for
# the rest, so dropping them removed the text arm's best-performing users from the comparison.
_TRAILING_TAG = re.compile(r"(?:\s*</\s*[A-Za-z_][\w\-]*\s*>)+\s*$")
_FINAL_ARRAY = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*$")


def parse_rationale(raw: str) -> dict:
    """Split an arm's stage-1 output into its visible reasoning and whether that reasoning finished.

    Reads from the LAST opening <think> -- one gemini row prefixes an empty <script></script> before
    the real block -- then drops the trailing ranking array and any closing tags the arm appended,
    whatever it named them.

    `complete` is the load-bearing field, not `text`. Two very different things both leave text in a
    think block:

    * An arm that reasoned and stopped. It terminates the block, with a closing tag of some name or
      by emitting the ranking array.
    * An arm that never got there. The untrained OneRec bases spend the whole 2048-token stage-1
      budget re-transcribing the opaque SID tokens back at themselves and are cut off mid-token
      (39--58% land exactly on the cap) -- the documented degenerate mode in prompts.py. They close
      nothing and reach no array.

    Judging the second kind as a rationale would credit the base arms for four hundred truncated
    token dumps. A closed but EMPTY block -- the literal ``<think>\\n</think>[...]`` the bases also
    produce -- is complete and empty, which is the finding rather than a gap in the measurement.
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


def extract_rationale(raw: str) -> str:
    """The visible reasoning text alone. Callers deciding whether to judge want parse_rationale."""
    return parse_rationale(raw)["text"]


def _usage(record: dict) -> dict:
    raw = record.get("usage") or {}
    return {
        "prompt_tokens": int(raw.get("promptTokenCount", 0) or 0),
        "output_tokens": int(raw.get("candidatesTokenCount", 0) or 0),
        "reasoning_tokens": int(raw.get("thoughtsTokenCount", 0) or 0),
        "cost_usd": float(record.get("cost_usd") or 0.0),
    }


def _add_usage(total: dict, value: dict) -> None:
    for key in ("prompt_tokens", "output_tokens", "reasoning_tokens"):
        total[key] += int(value.get(key, 0) or 0)
    total["cost_usd"] += float(value.get("cost_usd", 0.0) or 0.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arm(path: Path, benchmark_rows: list[dict]) -> dict:
    """Load one frozen anchor result and bind every row back to the benchmark.

    The per-row checks are the integrity contract: identical grades and slate id prove the arm was
    scored on this exact frozen slate, and a valid ten-index ranking proves the recorded order is the
    one the metrics were computed from. They are arm-agnostic on purpose -- only the ANCHOR cell is
    pinned, because render/checkpoint/model are exactly what differs between the systems compared.
    """
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("meta"), dict):
        raise ValueError(f"result file has no metadata object: {path}")
    meta = payload["meta"]
    for field, expected in ANCHOR.items():
        if meta.get(field) != expected:
            raise ValueError(
                f"{path.name}: {field}={meta.get(field)!r} is not the anchor cell "
                f"(expected {expected!r})"
            )

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
            raise ValueError(
                f"result uid {uid} grades differ from the frozen benchmark"
            )
        if "slate_id" in row and str(row["slate_id"]) != str(source["slate_id"]):
            raise ValueError(
                f"result uid {uid} slate_id differs from the frozen benchmark"
            )
        if row.get("n_hist_kept") != ANCHOR["k"]:
            raise ValueError(
                f"result uid {uid} does not use the k={ANCHOR['k']} history"
            )
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
        raise ValueError(
            f"{path.name}: UID set differs from the frozen benchmark; missing={missing[:8]}"
        )
    incomplete = sorted(uid for uid, row in indexed.items() if not row["ranking_complete"])
    return {
        "path": path,
        "meta": meta,
        "rows": indexed,
        "incomplete_ranking_uids": incomplete,
        "provenance": {
            "result_file": str(path),
            "sha256": _sha256(path),
            "model": meta.get("model"),
            "checkpoint": meta.get("checkpoint"),
            "render": meta.get("render"),
            "arm": meta.get("arm"),
            "emit_rationale": meta.get("emit_rationale"),
            "system_saw_profile": not bool(meta.get("no_profile")),
            "n_users": len(indexed),
            "n_incomplete_ranking": len(incomplete),
        },
    }


def frozen_output(result_row: dict) -> dict:
    """The rationale and ranking this arm actually produced, verbatim."""
    raw = next(
        (result_row[field] for field in RATIONALE_FIELDS if result_row.get(field)), ""
    )
    parsed = parse_rationale(raw)
    detail = {"finish": result_row.get("finish") or "frozen_result",
              "rationale_terminator": parsed["terminator"]}
    for field in ("stage", "s1_closed_think", "s1_valid"):
        if field in result_row:
            detail[field] = result_row[field]
    return {
        "rationale": parsed["text"],
        # Both conditions matter: empty means the arm skipped reasoning, incomplete means it never
        # finished. Neither is a rationale a judge can be asked to assess.
        "rationale_available": bool(parsed["text"]) and parsed["complete"],
        "ranking": [int(x) for x in result_row["order"]],
        "detail": detail,
    }


def select_rows(rows: list[dict], arms: list[dict], n: int, seed: int) -> tuple[list[dict], dict]:
    """Sample n users that every arm ranked completely, and report what the filter removed.

    A user whose ranking is short in ANY arm is dropped from ALL of them, because the comparison is
    paired: keeping the user for the arms that succeeded would let one arm be scored on an easier
    subset than the other. The dropped uids are returned so the exclusion appears in the output
    rather than only in the arithmetic.
    """
    dropped = {
        arm["provenance"]["result_file"]: arm["incomplete_ranking_uids"]
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


def _parse_arm(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--arm expects LABEL=PATH, got {spec!r}"
        )
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"--arm has an empty label: {spec!r}")
    return label, Path(path.strip()).expanduser()
