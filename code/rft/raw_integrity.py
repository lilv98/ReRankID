"""Integrity checks for raw RFT rollout records.

The raw JSONL is an audit source, not a trusted annotation file.  Every field
used by the rejection gate is therefore recomputed from the model's verbatim
text before a continuation dataset can be assembled.
"""

import hashlib
import json

from rft.contract import N_SAMPLES, RFT_RECIPE, TOP_K
from utils.metrics import is_identity_collapse, ndcg, parse_topk_diag


RAW_SCHEMA_VERSION = 1
_NO_ANSWER_DIAG = {"has_array": False, "arr_len": 0, "n_oob": 0, "n_dup": 0}


def extract_cot(text):
    """Return the completed reasoning segment using the sampler's exact rule."""
    if not text or "</think>" not in text:
        return None
    before = text.split("</think>", 1)[0]
    if "<think>" in before:
        before = before.split("<think>", 1)[-1]
    cot = before.strip()
    return cot or None


def parse_rollout_text(text, n_cand):
    """Parse only an answer emitted after a closed reasoning segment."""
    if not isinstance(text, str) or "</think>" not in text:
        return None, 0, dict(_NO_ANSWER_DIAG)
    return parse_topk_diag(text.split("</think>")[-1], n_cand, TOP_K)


def prompt_seed(uid, slate_id, seed):
    payload = f"{RFT_RECIPE}|{seed}|{int(uid)}|{str(slate_id)}"
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16) % (2 ** 31)


def record_sha256(record):
    """Digest every raw-record field except the digest itself."""
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def seal_record(record):
    """Attach a schema marker and content digest to a newly sampled record."""
    if "record_sha256" in record:
        raise ValueError("raw RFT record is already sealed")
    record["raw_schema_version"] = RAW_SCHEMA_VERSION
    record["record_sha256"] = record_sha256(record)
    return record


def verify_record_seal(record):
    if record.get("raw_schema_version") != RAW_SCHEMA_VERSION:
        raise ValueError("raw RFT record has an unsupported or missing schema version")
    claimed = record.get("record_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError("raw RFT record is missing its SHA256 seal")
    actual = record_sha256(record)
    if claimed != actual:
        raise ValueError("raw RFT record SHA256 seal does not match its contents")
    return actual


def _require_plain_int(value, label, minimum=None, maximum=None):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return value


def _validate_record_metadata(record, pool_row, provenance):
    expected_scalars = {
        "uid": int(pool_row["uid"]),
        "slate_id": str(pool_row["slate_id"]),
        "k": int(pool_row["k"]),
        "n_cand": int(pool_row["n_cand"]),
        "pos_grade": int(pool_row["pos_grade"]),
        "n_hard_neg": int(pool_row["n_hard_neg"]),
    }
    for field, expected in expected_scalars.items():
        if record.get(field) != expected:
            raise ValueError(f"raw RFT record {field} differs from the frozen pool row")

    grades = [int(value) for value in pool_row["cand_grades"]]
    gold = [index for index, grade in enumerate(grades) if grade > 0]
    if len(grades) != expected_scalars["n_cand"] or len(gold) != 1:
        raise ValueError("frozen RFT pool row has an invalid candidate slate")
    if record.get("grades") != grades:
        raise ValueError("raw RFT record grades differ from the frozen pool row")
    if record.get("gold_idx") != gold:
        raise ValueError("raw RFT record gold index differs from the frozen pool row")

    prompt_tokens = _require_plain_int(
        record.get("prompt_tokens"), "prompt_tokens", minimum=1
    )
    if not prompt_tokens:
        raise ValueError("raw RFT prompt token count is empty")
    sampling = provenance["sampling"]
    expected_seed = prompt_seed(
        expected_scalars["uid"],
        expected_scalars["slate_id"],
        _require_plain_int(sampling.get("seed"), "sampling.seed", minimum=0),
    )
    if record.get("prompt_seed") != expected_seed:
        raise ValueError("raw RFT record prompt seed differs from the frozen recipe")


def _validate_rollout(rollout, sample_index, grades, gold, max_tokens):
    if not isinstance(rollout, dict):
        raise ValueError(f"RFT sample {sample_index} is not an object")
    if rollout.get("sample_index") != sample_index:
        raise ValueError(
            f"RFT sample_index sequence must be exact; expected {sample_index}"
        )
    if "rollout_index" in rollout:
        raise ValueError("legacy rollout_index is forbidden; use sample_index")

    text = rollout.get("text")
    if not isinstance(text, str):
        raise ValueError(f"RFT sample {sample_index} text must be a string")
    order, n_listed, diag = parse_rollout_text(text, len(grades))
    cot = extract_cot(text)
    score = ndcg(order, grades, TOP_K) if order is not None else None
    rank = order.index(gold[0]) + 1 if order is not None and gold[0] in order else None
    expected = {
        "cot_text": cot,
        "order": order,
        "n_listed": n_listed,
        "diag": diag,
        "ndcg10": round(float(score), 8) if score is not None else None,
        "gold_rank": rank,
        "valid": order is not None and n_listed == TOP_K,
        "rank1": rank == 1,
        "identity": bool(
            order is not None and is_identity_collapse(order, TOP_K)
        ),
    }
    for field, value in expected.items():
        if rollout.get(field) != value:
            raise ValueError(
                f"RFT sample {sample_index} {field} disagrees with reparsed text"
            )
    for field in ("valid", "rank1", "identity"):
        if not isinstance(rollout.get(field), bool):
            raise ValueError(f"RFT sample {sample_index} {field} must be boolean")

    finish_reason = rollout.get("finish_reason")
    if finish_reason not in {"stop", "length"}:
        raise ValueError(
            f"RFT sample {sample_index} has unsupported finish_reason "
            f"{finish_reason!r}"
        )
    output_tokens = _require_plain_int(
        rollout.get("output_tokens"),
        f"RFT sample {sample_index} output_tokens",
        minimum=1,
        maximum=max_tokens,
    )
    if finish_reason == "length" and output_tokens != max_tokens:
        raise ValueError(
            f"RFT sample {sample_index} length finish must consume max_tokens"
        )
    if rollout.get("stop_reason") is not None:
        raise ValueError(
            f"RFT sample {sample_index} has an unexpected custom stop reason"
        )


def validate_rollout_record(record, pool_row, provenance, expected_samples):
    """Verify one complete raw prompt record and all derived rollout fields."""
    verify_record_seal(record)
    if record.get("provenance") != provenance:
        raise ValueError("raw RFT record provenance differs from the file contract")
    _validate_record_metadata(record, pool_row, provenance)

    expected_samples = _require_plain_int(
        expected_samples, "expected RFT samples", minimum=1
    )
    rollouts = record.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != expected_samples:
        raise ValueError(
            f"each RFT prompt must contain exactly {expected_samples} samples"
        )
    grades = [int(value) for value in pool_row["cand_grades"]]
    gold = [index for index, grade in enumerate(grades) if grade > 0]
    max_tokens = _require_plain_int(
        provenance["sampling"].get("max_tokens"),
        "sampling.max_tokens",
        minimum=1,
    )
    for index, rollout in enumerate(rollouts):
        _validate_rollout(rollout, index, grades, gold, max_tokens)
    return record


def validate_raw_records(records, pool_by_key, provenance, formal):
    """Validate an entire raw JSONL before any rejection selection occurs."""
    sampling = provenance.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("raw RFT provenance is missing sampling settings")
    requested = _require_plain_int(sampling.get("n"), "sampling.n", minimum=1)
    expected_samples = N_SAMPLES if formal else requested
    if formal and requested != N_SAMPLES:
        raise ValueError(f"formal RFT requires exactly {N_SAMPLES} samples per prompt")

    for record in records:
        key = int(record.get("uid")), str(record.get("slate_id"))
        if key not in pool_by_key:
            raise ValueError(f"raw RFT record key is outside the frozen pool: {key}")
        validate_rollout_record(
            record,
            pool_by_key[key],
            provenance,
            expected_samples,
        )
    return expected_samples
