"""Fail-closed dev selection and test-once gate for formal OPD.

This module is deliberately CPU-only.  It validates compact adapters and
``score_sid.py`` JSON outputs, freezes the dev-selected adapter, and emits
explicit scorer command lines.  It never imports or launches vLLM.

State transitions:

``dev-command -> select -> test-command -> accept-test -> validate-test``

The test transition is irreversible: ``test_started.json`` binds the exact
selection and the twelve preregistered cells.  Subsequent calls may plan only
missing cells and may never overwrite an accepted result.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import sys
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from opd import contract
from opd.provenance import (
    ManifestError,
    ensure_immutable_json,
    load_manifest,
    payload_sha256,
    validate_weight_artifacts,
)
from utils import config
from utils.data_io import select_slate
from utils.metrics import aggregate, is_valid, ndcg, random_mrr, random_ndcg


EXPECTED_STEPS = tuple(range(25, 201, 25))
DEV_TAG = "opd_v3_dev_select"
TEST_TAG = "opd_v3_test"
SELECTION_NAME = "selection.json"
FROZEN_ADAPTER_NAME = "selected_adapter"
TEST_STARTED_NAME = "test_started.json"
TEST_RECEIPT_DIR_NAME = "test_receipts"
LEGACY_TEST_VALIDATION_NAME = "test_validation.json"
SELECTION_SCHEMA_VERSION = 2
TEST_STARTED_SCHEMA_VERSION = 2
TEST_RECEIPT_SCHEMA_VERSION = 1
LEGACY_SCHEMA_VERSION = 1
DEFAULT_POLICY = contract.DEFAULT_POLICY
SELECTION_SCORE_KEY = "ndcg@5_unrounded"

HISTORY_KS = (0, 4, 8, 16, 32, 64, 128, 256)
CANDIDATE_NS = (10, 20, 30, 50, 100)
TEST_CELLS = tuple(
    [(k, 30) for k in HISTORY_KS]
    + [(64, n_cand) for n_cand in CANDIDATE_NS if n_cand != 30]
)

DEV_K = 64
DEV_N_CAND = 30
TOP_K = 10
K_NDCG = 5
MAX_TOKENS = 2048
EXTRACT_TOKENS = 256
MAX_MODEL_LEN = 8192

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_STEP_DIR = re.compile(r"^step-(\d{6})$")

_EVALUATION_SOURCE_FILES = (
    "code/benchmark/score_sid.py",
    "code/opd/eval_protocol.py",
    "code/utils/config.py",
    "code/utils/data_io.py",
    "code/utils/metrics.py",
    "code/utils/prompts.py",
    "code/utils/sid_engine.py",
    "code/utils/templates/qwen3_soft_switch.jinja2",
)
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)
_MODEL_RUNTIME_FILES = ("config.json", "generation_config.json")


class ProtocolError(RuntimeError):
    """Raised whenever formal evaluation state is incomplete or ambiguous."""


def _policy_spec(policy):
    if policy is None:
        raise ProtocolError(
            "OPD evaluation policy must be explicit; use 1.7b-pro or 8b-pro"
        )
    try:
        policy = contract.normalize_policy(policy)
        spec = contract.policy_spec(policy)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(str(exc)) from exc
    student = spec["student"]
    if policy == DEFAULT_POLICY:
        # Keep these exact keys and values for previously generated 1.7B state.
        model_identity = {
            "revision": student["revision"],
            "weight_sha256": student["weight_sha256"],
        }
        scale = "1.7b"
    else:
        model_identity = {
            "name": student["name"],
            "revision": student["revision"],
            "weight_sha256": student["weight_sha256"],
            "weight_index_sha256": student["weight_index_sha256"],
        }
        scale = "8b-pro"
    return {
        "name": policy,
        **spec,
        "selection_kind": f"opd-{scale}-dev-selection",
        "test_kind": f"opd-{scale}-test-once",
        "model_identity": model_identity,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_path(path):
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def _read_json_bytes(path, label):
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"{label} must be a JSON object: {path}")
    return raw, payload


def _require_hex(value, label):
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ProtocolError(f"{label} is not a lowercase SHA-256 digest")


def _require_equal(actual, expected, label):
    if actual != expected:
        raise ProtocolError(
            f"{label}={actual!r}, expected {expected!r}"
        )


def _require_exact_keys(payload, expected, label):
    if not isinstance(payload, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    actual = set(payload)
    expected = set(expected)
    if actual != expected:
        raise ProtocolError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _atomic_create_bytes(path, raw):
    directory = os.path.dirname(_canonical_path(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".opd-eval.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@contextmanager
def _state_lock(state_dir):
    os.makedirs(state_dir, exist_ok=True)
    lock_path = os.path.join(state_dir, ".eval_protocol.lock")
    with open(lock_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def _selection_path(state_dir):
    return os.path.join(_canonical_path(state_dir), SELECTION_NAME)


def _frozen_adapter_path(state_dir):
    return os.path.join(_canonical_path(state_dir), FROZEN_ADAPTER_NAME)


def _test_started_path(state_dir):
    return os.path.join(_canonical_path(state_dir), TEST_STARTED_NAME)


def _test_receipt_dir(state_dir):
    return os.path.join(_canonical_path(state_dir), TEST_RECEIPT_DIR_NAME)


def _legacy_test_validation_path(state_dir):
    return os.path.join(
        _canonical_path(state_dir), LEGACY_TEST_VALIDATION_NAME
    )


def _evaluation_stack(model, policy=DEFAULT_POLICY):
    """Hash every local input that can change scorer behavior.

    The dev result tag binds these hashes to the actual scorer invocation.
    Selection stores the same payload, and every later test transition
    recomputes it before accepting or planning any test work.
    """

    policy_spec = _policy_spec(policy)
    root = _canonical_path(config.ROOT)
    sources = {}
    for relative in _EVALUATION_SOURCE_FILES:
        path = os.path.join(root, relative)
        if not os.path.isfile(path):
            raise ProtocolError(f"missing evaluation source file: {path}")
        sources[relative] = sha256_file(path)

    tokenizer_hashes = {}
    for filename in _TOKENIZER_FILES:
        path = os.path.join(model, filename)
        if not os.path.isfile(path):
            raise ProtocolError(f"missing evaluation tokenizer file: {path}")
        tokenizer_hashes[filename] = sha256_file(path)
    _require_equal(
        tokenizer_hashes["tokenizer.json"],
        policy_spec["student"]["tokenizer_file_sha256"],
        "evaluation tokenizer.json hash",
    )
    model_runtime_hashes = {}
    for filename in _MODEL_RUNTIME_FILES:
        path = os.path.join(model, filename)
        if not os.path.isfile(path):
            raise ProtocolError(f"missing model runtime file: {path}")
        model_runtime_hashes[filename] = sha256_file(path)
    return {
        "schema_version": 1,
        "source_sha256": sources,
        "tokenizer_sha256": tokenizer_hashes,
        "model_runtime_sha256": model_runtime_hashes,
        "logical_vocab_sha256": policy_spec["student"]["vocab_sha256"],
    }


def _evaluation_stack_sha256(stack):
    digest = payload_sha256(stack)
    _require_hex(digest, "evaluation stack hash")
    return digest


def _dev_result_tag(stack):
    return f"{DEV_TAG}__{_evaluation_stack_sha256(stack)}"


def _test_result_tag(nonce):
    _require_hex(nonce, "test nonce")
    return f"{TEST_TAG}__{nonce}"


def _validate_model_path(model, policy=DEFAULT_POLICY):
    policy_spec = _policy_spec(policy)
    model = _canonical_path(model)
    expected = _canonical_path(
        os.path.join(config.ROOT, policy_spec["student"]["base_rel"])
    )
    if model != expected:
        raise ProtocolError(
            "formal OPD evaluation model must be the pinned original "
            f"{policy} Base path {expected!r}"
        )
    if not os.path.isdir(model):
        raise ProtocolError(f"formal OPD base model is missing: {model}")
    try:
        validate_weight_artifacts(
            model,
            policy_spec["student"],
            legacy_filename="model-00000-of-00001.safetensors",
        )
    except ManifestError as exc:
        raise ProtocolError(
            f"formal OPD base model failed weight verification: {exc}"
        ) from exc
    return model


def _validate_manifest(manifest_path, policy=DEFAULT_POLICY):
    _policy_spec(policy)
    manifest_path = _canonical_path(manifest_path)
    if not os.path.isfile(manifest_path):
        raise ProtocolError(f"missing formal run manifest: {manifest_path}")
    try:
        manifest = load_manifest(manifest_path, policy=policy)
    except ManifestError as exc:
        raise ProtocolError(str(exc)) from exc
    _require_equal(manifest.get("run_mode"), "formal", "manifest.run_mode")
    repository = manifest.get("audit", {}).get("repository", {})
    training_commit = repository.get("head")
    if not isinstance(training_commit, str) or not training_commit:
        raise ProtocolError("formal run manifest has no training commit")
    return {
        "path": manifest_path,
        "sha256": sha256_file(manifest_path),
        "payload_sha256": payload_sha256(manifest),
        "training_commit": training_commit,
        "payload": manifest,
    }


def _adapter_paths(adapter_dir):
    return {
        "weights": os.path.join(adapter_dir, "adapter_model.safetensors"),
        "config": os.path.join(adapter_dir, "adapter_config.json"),
        "provenance": os.path.join(adapter_dir, "provenance.json"),
    }


def _validate_one_adapter(
    adapter_dir, step, manifest_info, policy=DEFAULT_POLICY
):
    policy_spec = _policy_spec(policy)
    adapter_dir = _canonical_path(adapter_dir)
    if not os.path.isdir(adapter_dir):
        raise ProtocolError(f"missing compact adapter for step {step}: {adapter_dir}")
    paths = _adapter_paths(adapter_dir)
    for label, path in paths.items():
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            raise ProtocolError(
                f"step {step} has no nonempty adapter {label}: {path}"
            )
    _, config_payload = _read_json_bytes(paths["config"], "adapter config")
    expected_config = {
        "r": contract.LORA_RANK,
        "lora_alpha": contract.LORA_ALPHA,
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
    }
    for key, value in expected_config.items():
        _require_equal(
            config_payload.get(key), value, f"step {step} adapter_config.{key}"
        )

    _, provenance = _read_json_bytes(paths["provenance"], "adapter provenance")
    expected_provenance = {
        "schema_version": 2,
        "recipe": contract.RECIPE,
        "data_recipe": contract.DATA_RECIPE,
        "run_mode": "formal",
        "step": step,
        "student_base": policy_spec["student"]["base_rel"],
        "student_revision": policy_spec["student"]["revision"],
        "teacher": policy_spec["teacher"]["merged_rel"],
        "training_commit": manifest_info["training_commit"],
        "run_manifest_sha256": manifest_info["sha256"],
        "manifest_payload_sha256": manifest_info["payload_sha256"],
    }
    for key, value in expected_provenance.items():
        _require_equal(
            provenance.get(key), value, f"step {step} provenance.{key}"
        )

    source_actor = provenance.get("source_actor")
    if not isinstance(source_actor, str):
        raise ProtocolError(f"step {step} provenance has no source_actor")
    source_parts = Path(os.path.normpath(source_actor)).parts
    if len(source_parts) < 2 or source_parts[-2:] != (
        f"global_step_{step}",
        "actor",
    ):
        raise ProtocolError(
            f"step {step} provenance.source_actor={source_actor!r} "
            "does not identify the matching checkpoint"
        )

    weights_sha = sha256_file(paths["weights"])
    config_sha = sha256_file(paths["config"])
    provenance_sha = sha256_file(paths["provenance"])
    _require_equal(
        provenance.get("adapter_model_sha256"),
        weights_sha,
        f"step {step} provenance.adapter_model_sha256",
    )
    _require_equal(
        provenance.get("adapter_config_sha256"),
        config_sha,
        f"step {step} provenance.adapter_config_sha256",
    )
    for key in (
        "adapter_model_sha256",
        "adapter_config_sha256",
        "source_model_shard_sha256",
    ):
        _require_hex(provenance.get(key), f"step {step} provenance.{key}")
    metadata_hashes = provenance.get("checkpoint_metadata_sha256")
    if not isinstance(metadata_hashes, dict):
        raise ProtocolError(
            f"step {step} provenance has no checkpoint metadata hashes"
        )
    for key in ("data", "fsdp", "lora"):
        _require_hex(
            metadata_hashes.get(key),
            f"step {step} checkpoint_metadata_sha256.{key}",
        )
    tensor_count = provenance.get("adapter_tensor_count")
    module_count = provenance.get("adapter_module_count")
    if (
        not isinstance(tensor_count, int)
        or isinstance(tensor_count, bool)
        or tensor_count <= 0
        or tensor_count % 2
        or not isinstance(module_count, int)
        or isinstance(module_count, bool)
        or module_count <= 0
        or tensor_count != 2 * module_count
    ):
        raise ProtocolError(
            f"step {step} provenance has inconsistent LoRA tensor counts"
        )
    return {
        "step": step,
        "label": f"step{step}",
        "path": adapter_dir,
        "adapter_model_sha256": weights_sha,
        "adapter_config_sha256": config_sha,
        "provenance_sha256": provenance_sha,
    }


def validate_adapters(
    adapter_root, run_manifest, policy=DEFAULT_POLICY
):
    """Validate exactly the eight formal compact adapters and their provenance."""

    adapter_root = _canonical_path(adapter_root)
    if not os.path.isdir(adapter_root):
        raise ProtocolError(f"missing compact-adapter root: {adapter_root}")
    manifest_info = _validate_manifest(run_manifest, policy)
    expected_names = {f"step-{step:06d}" for step in EXPECTED_STEPS}
    actual_names = set(os.listdir(adapter_root))
    if actual_names != expected_names:
        raise ProtocolError(
            "compact-adapter root must contain exactly the eight formal "
            f"exports; missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    records = []
    for step in EXPECTED_STEPS:
        records.append(
            _validate_one_adapter(
                os.path.join(adapter_root, f"step-{step:06d}"),
                step,
                manifest_info,
                policy,
            )
        )
    return records, manifest_info


def _benchmark_info(benchmark):
    benchmark = _canonical_path(benchmark)
    if not os.path.isfile(benchmark):
        raise ProtocolError(f"missing frozen benchmark: {benchmark}")
    try:
        import pyarrow.parquet as pq

        split_uids = {"dev": [], "test": []}
        grade_pools = {"dev": {}, "test": {}}
        parquet = pq.ParquetFile(benchmark)
        required_columns = {"uid", "slice", "pool_grades"}
        missing_columns = required_columns - set(parquet.schema_arrow.names)
        if missing_columns:
            raise ProtocolError(
                "benchmark is missing frozen evaluation columns: "
                f"{sorted(missing_columns)}"
            )
        for batch in parquet.iter_batches(
            batch_size=2048, columns=["uid", "slice", "pool_grades"]
        ):
            values = batch.to_pydict()
            for uid, split, grades in zip(
                values["uid"], values["slice"], values["pool_grades"]
            ):
                if split not in split_uids:
                    raise ProtocolError(
                        f"benchmark contains unexpected split {split!r}"
                    )
                if not isinstance(uid, int) or isinstance(uid, bool):
                    raise ProtocolError(f"benchmark contains invalid uid {uid!r}")
                if (
                    not isinstance(grades, list)
                    or len(grades) != 100
                    or any(
                        not isinstance(grade, int) or isinstance(grade, bool)
                        for grade in grades
                    )
                ):
                    raise ProtocolError(
                        f"benchmark uid {uid} has an invalid 100-item grade pool"
                    )
                if uid in grade_pools[split]:
                    raise ProtocolError(
                        f"benchmark {split} contains duplicate uid {uid}"
                    )
                split_uids[split].append(uid)
                grade_pools[split][uid] = tuple(grades)
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError(f"cannot inspect benchmark {benchmark}: {exc}") from exc

    result = {
        "path": benchmark,
        "sha256": sha256_file(benchmark),
        "grade_pools": grade_pools,
        "_grade_cache": {},
    }
    for split in ("dev", "test"):
        values = split_uids[split]
        if len(values) != 400 or len(set(values)) != 400:
            raise ProtocolError(
                f"benchmark {split} split must contain 400 unique users"
            )
        result[f"{split}_uids"] = tuple(sorted(values))
    if set(result["dev_uids"]) & set(result["test_uids"]):
        raise ProtocolError("benchmark dev and test UID sets overlap")
    return result


def _frozen_benchmark_info(benchmark):
    """Validate the canonical benchmark frozen for schema-2 evaluation."""

    expected_path = _canonical_path(
        os.path.join(config.ROOT, contract.BENCHMARK_REL)
    )
    actual_path = _canonical_path(benchmark)
    if actual_path != expected_path:
        raise ProtocolError(
            "schema-2 benchmark must use the pinned canonical realpath; "
            f"got {actual_path!r}, expected {expected_path!r}"
        )
    benchmark_info = _benchmark_info(actual_path)
    _require_equal(
        benchmark_info["sha256"],
        contract.BENCHMARK_SHA256,
        "schema-2 benchmark.sha256",
    )
    return benchmark_info


def _expected_grades(benchmark_info, split, n_cand):
    cache_key = (split, n_cand)
    cache = benchmark_info["_grade_cache"]
    if cache_key in cache:
        return cache[cache_key]
    expected = {}
    for uid, grades in benchmark_info["grade_pools"][split].items():
        # select_slate owns the benchmark's deterministic display permutation.
        # Supply harmless stand-ins for fields that are permuted in parallel.
        pool = list(range(len(grades)))
        selected = select_slate(
            {
                "uid": uid,
                "pool_pids": pool,
                "pool_sids": pool,
                "pool_captions": pool,
                "pool_grades": list(grades),
            },
            n_cand,
            shuffle_id=0,
        )
        expected[uid] = tuple(int(value) for value in selected["cand_grades"])
    cache[cache_key] = expected
    return expected


def _parse_lora_dirs(value, label):
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{label} has no --lora-dirs specification")
    result = []
    names = set()
    paths = set()
    for raw in value.split(","):
        if raw.count("=") != 1:
            raise ProtocolError(f"{label} has malformed LoRA entry {raw!r}")
        name, path = raw.split("=", 1)
        if not name or not path or name in names:
            raise ProtocolError(f"{label} has duplicate or empty LoRA name")
        canonical = _canonical_path(path)
        if canonical in paths:
            raise ProtocolError(f"{label} reuses one LoRA path")
        names.add(name)
        paths.add(canonical)
        result.append((name, canonical))
    return result


def lora_spec(adapter_records):
    return ",".join(
        f"{record['label']}={record['path']}" for record in adapter_records
    )


def _expected_common_meta(model, benchmark, tag, split, outdir):
    return {
        "tag": tag,
        "slice": split,
        "model": model,
        "bench": benchmark,
        "outdir": _canonical_path(outdir),
        "benches": None,
        "n": None,
        "shuffle": 0,
        "render": "sid",
        "corrupt_sid": False,
        "no_profile": True,
        "top_k": TOP_K,
        "max_tokens": MAX_TOKENS,
        "extract_tokens": EXTRACT_TOKENS,
        "k_ndcg": K_NDCG,
        "max_model_len": MAX_MODEL_LEN,
        "max_lora_rank": 32,
        "skip_base": True,
        "arm": "sid",
        "protocol": "listwise-two-stage",
        "decoding": "greedy+structured",
    }


_INFERENCE_META_KEYS = (
    "tp",
    "no_eager",
    "max_model_len",
    "max_tokens",
    "extract_tokens",
    "top_k",
    "k_ndcg",
    "max_lora_rank",
    "render",
    "corrupt_sid",
    "no_profile",
    "shuffle",
    "protocol",
    "decoding",
    "vllm",
)


def _nonnegative_int(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _validate_result_schema(result, *, label, uid, k, n_cand, grades):
    actual_grades = result.get("grades")
    if (
        not isinstance(actual_grades, list)
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in actual_grades
        )
    ):
        raise ProtocolError(f"{label} uid {uid} has invalid grades")
    if tuple(actual_grades) != grades:
        raise ProtocolError(
            f"{label} uid {uid} grades do not match the frozen benchmark"
        )

    n_listed = result.get("n_listed")
    if not _nonnegative_int(n_listed) or n_listed > TOP_K:
        raise ProtocolError(f"{label} uid {uid} has invalid n_listed")
    order = result.get("order")
    if order is None:
        if n_listed != 0:
            raise ProtocolError(
                f"{label} uid {uid} has no order but nonzero n_listed"
            )
    elif not isinstance(order, list) or len(order) != n_listed:
        raise ProtocolError(f"{label} uid {uid} has an invalid ranked order")
    elif any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value < n_cand
            for value in order
    ) or len(set(order)) != len(order):
        raise ProtocolError(f"{label} uid {uid} has an invalid ranked order")

    stage = result.get("stage")
    s1_valid = result.get("s1_valid")
    s1_closed = result.get("s1_closed_think")
    if stage not in (1, 2):
        raise ProtocolError(f"{label} uid {uid} has invalid stage")
    if not isinstance(s1_valid, bool) or not isinstance(s1_closed, bool):
        raise ProtocolError(f"{label} uid {uid} has invalid stage-1 flags")
    if (stage == 1) != s1_valid:
        raise ProtocolError(
            f"{label} uid {uid} has inconsistent stage and s1_valid"
        )
    if s1_valid and (not s1_closed or n_listed != TOP_K):
        raise ProtocolError(
            f"{label} uid {uid} has inconsistent stage-1 success"
        )

    for field in (
        "s1_out_tokens",
        "prompt_tokens",
        "s2_out_tokens",
        "n_hist_kept",
    ):
        if not _nonnegative_int(result.get(field)):
            raise ProtocolError(f"{label} uid {uid} has invalid {field}")
    if result["n_hist_kept"] > k:
        raise ProtocolError(
            f"{label} uid {uid} kept more than the requested history"
        )
    if stage == 1 and result["s2_out_tokens"] != 0:
        raise ProtocolError(
            f"{label} uid {uid} records stage-2 tokens without fallback"
        )
    for field in ("raw1", "raw2"):
        if not isinstance(result.get(field), str):
            raise ProtocolError(f"{label} uid {uid} has invalid {field}")

    diag = result.get("diag")
    if not isinstance(diag, dict) or not isinstance(
        diag.get("has_array"), bool
    ):
        raise ProtocolError(f"{label} uid {uid} has invalid parse diagnostics")
    for field in ("arr_len", "n_oob", "n_dup"):
        if not _nonnegative_int(diag.get(field)):
            raise ProtocolError(
                f"{label} uid {uid} has invalid diagnostic {field}"
            )


@lru_cache(maxsize=None)
def _cached_random_ndcg(grades, k):
    return random_ndcg(list(grades), k=k, seed=0)


@lru_cache(maxsize=None)
def _cached_random_mrr(grades, k):
    return random_mrr(list(grades), k=k, seed=0)


def _two_stage_stats(results):
    count = len(results)
    s1_valid = sum(1 for result in results if result["s1_valid"])
    used_stage2 = [result for result in results if result["stage"] == 2]
    rescued = sum(
        1
        for result in used_stage2
        if result["n_listed"] >= TOP_K
    )
    looped = sum(
        1 for result in results if not result["s1_closed_think"]
    )
    return {
        "stage1_valid_rate": round(s1_valid / count, 4),
        "two_stage_rate": round(len(used_stage2) / count, 4),
        "stage2_rescue_rate": (
            round(rescued / len(used_stage2), 4)
            if used_stage2
            else None
        ),
        "stage1_loop_rate": round(looped / count, 4),
        "n_stage1_selfserved": s1_valid,
        "n_two_stage": len(used_stage2),
        "n_stage2_rescued": rescued,
        "mean_prompt_tokens": round(
            sum(result["prompt_tokens"] for result in results) / count, 1
        ),
        "mean_s1_out_tokens": round(
            sum(result["s1_out_tokens"] for result in results) / count, 1
        ),
        "mean_s2_out_tokens": (
            round(
                sum(
                    result["s2_out_tokens"] for result in used_stage2
                )
                / len(used_stage2),
                1,
            )
            if used_stage2
            else 0.0
        ),
    }


def _recompute_metrics(results):
    recomputed = aggregate(
        results, k_ndcg=K_NDCG, n_out=TOP_K, with_random=False
    )
    count = len(results)
    recomputed["random_ndcg@5"] = round(
        sum(
            _cached_random_ndcg(tuple(result["grades"]), K_NDCG)
            for result in results
        )
        / count,
        4,
    )
    recomputed["random_ndcg@10"] = round(
        sum(
            _cached_random_ndcg(tuple(result["grades"]), TOP_K)
            for result in results
        )
        / count,
        4,
    )
    recomputed["random_mrr_top"] = round(
        sum(
            _cached_random_mrr(tuple(result["grades"]), TOP_K)
            for result in results
        )
        / count,
        4,
    )
    recomputed["two_stage"] = _two_stage_stats(results)
    return recomputed


def _unrounded_ndcg_at_5(results):
    if not results:
        raise ProtocolError("cannot select a checkpoint from zero users")
    values = (
        ndcg(result["order"], result["grades"], K_NDCG)
        if is_valid(result, TOP_K)
        else 0.0
        for result in results
    )
    score = math.fsum(values) / len(results)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ProtocolError(f"invalid unrounded NDCG@5={score!r}")
    return score


def _validate_recomputed_metrics(metrics, results, label):
    try:
        recomputed = _recompute_metrics(results)
    except Exception as exc:
        raise ProtocolError(
            f"{label} per-user results cannot be rescored: {exc}"
        ) from exc
    if set(metrics) != set(recomputed):
        raise ProtocolError(
            f"{label} metrics keys differ from recomputed scorer schema"
        )
    for key, expected in recomputed.items():
        if metrics.get(key) != expected:
            raise ProtocolError(
                f"{label} metrics.{key}={metrics.get(key)!r} differs from "
                f"recomputed value {expected!r}"
            )
    exact_score = _unrounded_ndcg_at_5(results)
    return {
        "ndcg@5": float(recomputed["ndcg@5"]),
        SELECTION_SCORE_KEY: exact_score,
    }


def _validate_score_payload(
    payload,
    *,
    label,
    model,
    benchmark_info,
    split,
    tag,
    checkpoint,
    expected_loras,
    k,
    n_cand,
    outdir,
):
    meta = payload.get("meta")
    metrics = payload.get("metrics")
    results = payload.get("results")
    if not isinstance(meta, dict) or not isinstance(metrics, dict):
        raise ProtocolError(f"{label} has no meta/metrics objects")
    if not isinstance(results, list):
        raise ProtocolError(f"{label} has no per-user results list")
    expected = _expected_common_meta(
        model, benchmark_info["path"], tag, split, outdir
    )
    expected.update({"checkpoint": checkpoint, "k": k, "n_cand": n_cand})
    for key, value in expected.items():
        _require_equal(meta.get(key), value, f"{label} meta.{key}")
    tp = meta.get("tp")
    if not isinstance(tp, int) or isinstance(tp, bool) or tp <= 0:
        raise ProtocolError(f"{label} must record an explicit positive meta.tp")
    if not isinstance(meta.get("no_eager"), bool):
        raise ProtocolError(f"{label} has invalid meta.no_eager")
    if not isinstance(meta.get("vllm"), str) or not meta["vllm"]:
        raise ProtocolError(f"{label} has no vLLM version")
    parsed_loras = _parse_lora_dirs(meta.get("lora_dirs"), label)
    if parsed_loras != expected_loras:
        raise ProtocolError(
            f"{label} used noncanonical LoRA adapters: {parsed_loras!r}"
        )
    if len(results) != 400 or metrics.get("n_users") != 400:
        raise ProtocolError(f"{label} must contain all 400 {split} users")
    expected_uids = benchmark_info[f"{split}_uids"]
    expected_grade_map = _expected_grades(
        benchmark_info, split, n_cand
    )
    uids = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ProtocolError(f"{label} result {index} is not an object")
        uid = result.get("uid")
        if not isinstance(uid, int) or isinstance(uid, bool):
            raise ProtocolError(f"{label} result {index} has invalid uid")
        if uid not in expected_grade_map:
            raise ProtocolError(
                f"{label} result {index} has a non-{split} uid {uid}"
            )
        _validate_result_schema(
            result,
            label=label,
            uid=uid,
            k=k,
            n_cand=n_cand,
            grades=expected_grade_map[uid],
        )
        uids.append(uid)
    if len(set(uids)) != 400 or set(uids) != set(expected_uids):
        raise ProtocolError(
            f"{label} UID set is not the frozen 400-user {split} split"
        )
    scores = _validate_recomputed_metrics(metrics, results, label)
    signature = {key: meta.get(key) for key in _INFERENCE_META_KEYS}
    return scores, signature


def collect_dev_results(
    dev_dir, adapter_records, benchmark_info, model, dev_tag
):
    """Validate exactly eight complete dev-anchor JSONs."""

    dev_dir = _canonical_path(dev_dir)
    if not os.path.isdir(dev_dir):
        raise ProtocolError(f"missing official dev result directory: {dev_dir}")
    entries = sorted(os.listdir(dev_dir))
    if len(entries) != len(EXPECTED_STEPS) or any(
        not entry.endswith(".json")
        or not os.path.isfile(os.path.join(dev_dir, entry))
        for entry in entries
    ):
        raise ProtocolError(
            "dev result directory must contain exactly eight JSON files"
        )
    expected_loras = [
        (record["label"], record["path"]) for record in adapter_records
    ]
    by_label = {record["label"]: record for record in adapter_records}
    found = {}
    common_signature = None
    for filename in entries:
        path = os.path.join(dev_dir, filename)
        raw, payload = _read_json_bytes(path, "dev result")
        checkpoint = payload.get("meta", {}).get("checkpoint")
        if checkpoint not in by_label:
            raise ProtocolError(
                f"unexpected dev checkpoint label {checkpoint!r} in {filename}"
            )
        if checkpoint in found:
            raise ProtocolError(f"duplicate dev result for {checkpoint}")
        step = by_label[checkpoint]["step"]
        scores, signature = _validate_score_payload(
            payload,
            label=f"dev step {step}",
            model=model,
            benchmark_info=benchmark_info,
            split="dev",
            tag=dev_tag,
            checkpoint=checkpoint,
            expected_loras=expected_loras,
            k=DEV_K,
            n_cand=DEV_N_CAND,
            outdir=dev_dir,
        )
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise ProtocolError(
                f"dev step {step} used a different inference stack"
            )
        found[checkpoint] = {
            "step": step,
            **scores,
            "result_path": _canonical_path(path),
            "result_sha256": hashlib.sha256(raw).hexdigest(),
        }
    missing = sorted(set(by_label) - set(found))
    if missing:
        raise ProtocolError(f"missing dev checkpoints: {missing}")
    return {
        record["step"]: found[record["label"]] for record in adapter_records
    }, common_signature


def choose_step(dev_records, *, score_key=SELECTION_SCORE_KEY):
    """Choose only by NDCG@5; exact ties resolve to the earlier step."""

    if set(dev_records) != set(EXPECTED_STEPS):
        raise ProtocolError("checkpoint selection requires all eight steps")
    scores = {}
    for step in EXPECTED_STEPS:
        score = dev_records[step].get(score_key)
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ProtocolError(
                f"step {step} has invalid dev selection score {score_key}"
            )
        scores[step] = float(score)
    return min(EXPECTED_STEPS, key=lambda step: (-scores[step], step))


def _copy_frozen_adapter(source, destination):
    if os.path.lexists(destination):
        raise ProtocolError(
            f"refusing to overwrite frozen selected adapter: {destination}"
        )
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix=".selected_adapter.", dir=parent)
    os.rmdir(temporary)
    try:
        shutil.copytree(source, temporary, symlinks=False)
        if os.path.lexists(destination):
            raise ProtocolError(
                f"refusing to overwrite frozen selected adapter: {destination}"
            )
        os.rename(temporary, destination)
    finally:
        if os.path.isdir(temporary):
            shutil.rmtree(temporary)


def _frozen_hashes(adapter_dir):
    paths = _adapter_paths(adapter_dir)
    for label, path in paths.items():
        if not os.path.isfile(path):
            raise ProtocolError(f"frozen adapter is missing {label}: {path}")
    return {
        "adapter_model_sha256": sha256_file(paths["weights"]),
        "adapter_config_sha256": sha256_file(paths["config"]),
        "provenance_sha256": sha256_file(paths["provenance"]),
    }


def _selection_payload(
    *,
    policy,
    adapter_records,
    manifest_info,
    benchmark_info,
    model,
    evaluation_stack,
    dev_records,
    inference_signature,
    best_step,
    frozen_adapter,
):
    policy_spec = _policy_spec(policy)
    selected_source = next(
        record for record in adapter_records if record["step"] == best_step
    )
    frozen_hashes = _frozen_hashes(frozen_adapter)
    expected_hashes = {
        key: selected_source[key]
        for key in (
            "adapter_model_sha256",
            "adapter_config_sha256",
            "provenance_sha256",
        )
    }
    if frozen_hashes != expected_hashes:
        raise ProtocolError("frozen selected adapter differs from its source")
    candidates = []
    for record in adapter_records:
        dev = dev_records[record["step"]]
        candidates.append(
            {
                "step": record["step"],
                "adapter_path": record["path"],
                "adapter_model_sha256": record["adapter_model_sha256"],
                "adapter_config_sha256": record["adapter_config_sha256"],
                "provenance_sha256": record["provenance_sha256"],
                "dev_result_path": dev["result_path"],
                "dev_result_sha256": dev["result_sha256"],
                "ndcg@5": dev["ndcg@5"],
                SELECTION_SCORE_KEY: dev[SELECTION_SCORE_KEY],
            }
        )
    payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "kind": policy_spec["selection_kind"],
        "recipe": contract.RECIPE,
        "data_recipe": contract.DATA_RECIPE,
        "expected_steps": list(EXPECTED_STEPS),
        "selection_rule": {
            "primary_metric": SELECTION_SCORE_KEY,
            "reported_metric": "ndcg@5",
            "tie_break": "earliest-step-on-exact-unrounded-tie",
            "other_metrics_used": False,
        },
        "training": {
            "run_manifest": manifest_info["path"],
            "run_manifest_sha256": manifest_info["sha256"],
            "manifest_payload_sha256": manifest_info["payload_sha256"],
            "training_commit": manifest_info["training_commit"],
        },
        "benchmark": {
            "path": benchmark_info["path"],
            "sha256": benchmark_info["sha256"],
            "dev_users": 400,
            "test_users": 400,
        },
        "model": {
            "path": model,
            **policy_spec["model_identity"],
        },
        "evaluation_stack": evaluation_stack,
        "evaluation_stack_sha256": _evaluation_stack_sha256(
            evaluation_stack
        ),
        "dev_protocol": {
            "tag": _dev_result_tag(evaluation_stack),
            "slice": "dev",
            "k": DEV_K,
            "n_cand": DEV_N_CAND,
            "n": None,
            "skip_base": True,
            "inference": inference_signature,
        },
        "candidates": candidates,
        "selected": {
            "step": best_step,
            "ndcg@5": dev_records[best_step]["ndcg@5"],
            SELECTION_SCORE_KEY: dev_records[best_step][
                SELECTION_SCORE_KEY
            ],
            "source_adapter": selected_source["path"],
            "frozen_adapter": _canonical_path(frozen_adapter),
            **frozen_hashes,
        },
    }
    if policy != DEFAULT_POLICY:
        payload["policy"] = policy
    return payload


def seal_dev_selection(
    *,
    state_dir,
    dev_dir,
    adapter_root,
    run_manifest,
    benchmark,
    model,
    policy=DEFAULT_POLICY,
):
    """Validate dev, copy the selected adapter, and create selection.json once."""

    state_dir = _canonical_path(state_dir)
    _policy_spec(policy)
    model = _validate_model_path(model, policy)
    with _state_lock(state_dir):
        if os.path.exists(_test_started_path(state_dir)):
            raise ProtocolError(
                "test has started; dev selection can never be run again"
            )
        adapter_records, manifest_info = validate_adapters(
            adapter_root, run_manifest, policy
        )
        benchmark_info = _frozen_benchmark_info(benchmark)
        evaluation_stack = _evaluation_stack(model, policy)
        dev_records, inference_signature = collect_dev_results(
            dev_dir,
            adapter_records,
            benchmark_info,
            model,
            _dev_result_tag(evaluation_stack),
        )
        best_step = choose_step(dev_records)
        selection_path = _selection_path(state_dir)
        frozen_adapter = _frozen_adapter_path(state_dir)
        if not os.path.exists(selection_path):
            source = next(
                record["path"]
                for record in adapter_records
                if record["step"] == best_step
            )
            _copy_frozen_adapter(source, frozen_adapter)
        elif not os.path.isdir(frozen_adapter):
            raise ProtocolError(
                "selection exists without its frozen selected adapter"
            )
        payload = _selection_payload(
            policy=policy,
            adapter_records=adapter_records,
            manifest_info=manifest_info,
            benchmark_info=benchmark_info,
            model=model,
            evaluation_stack=evaluation_stack,
            dev_records=dev_records,
            inference_signature=inference_signature,
            best_step=best_step,
            frozen_adapter=frozen_adapter,
        )
        try:
            status = ensure_immutable_json(
                selection_path, payload, label="OPD dev selection"
            )
        except ManifestError as exc:
            raise ProtocolError(str(exc)) from exc
        return {"status": status, "selection": payload}


def _load_legacy_selection_for_validation(
    state_dir, *, policy=DEFAULT_POLICY
):
    """Load the completed schema-1 1.7B selection without reopening it.

    Schema 1 predates the nonce and receipt ledger. It is accepted only by the
    read-only complete-test validator, and only after the original adapters,
    eight 400-user dev files, benchmark, model, and selected frozen adapter all
    pass the current fail-closed checks.
    """

    if policy != DEFAULT_POLICY:
        raise ProtocolError(
            "schema-1 evaluation state is valid only for 1.7b-pro"
        )
    policy_spec = _policy_spec(policy)
    state_dir = _canonical_path(state_dir)
    selection_path = _selection_path(state_dir)
    raw, selection = _read_json_bytes(selection_path, "legacy OPD dev selection")
    if (
        not isinstance(selection.get("schema_version"), int)
        or isinstance(selection.get("schema_version"), bool)
    ):
        raise ProtocolError("legacy selection has an invalid schema version")
    _require_exact_keys(
        selection,
        {
            "schema_version",
            "kind",
            "recipe",
            "data_recipe",
            "expected_steps",
            "selection_rule",
            "training",
            "benchmark",
            "model",
            "dev_protocol",
            "candidates",
            "selected",
        },
        "legacy selection",
    )
    expected_top = {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "kind": "opd-1.7b-dev-selection",
        "recipe": contract.RECIPE,
        "data_recipe": contract.DATA_RECIPE,
        "expected_steps": list(EXPECTED_STEPS),
        "selection_rule": {
            "primary_metric": "ndcg@5",
            "tie_break": "earliest-step-on-exact-tie",
            "other_metrics_used": False,
        },
    }
    for key, value in expected_top.items():
        _require_equal(selection.get(key), value, f"legacy selection.{key}")

    training = selection["training"]
    _require_exact_keys(
        training,
        {
            "run_manifest",
            "run_manifest_sha256",
            "manifest_payload_sha256",
            "training_commit",
        },
        "legacy selection.training",
    )
    if not isinstance(training["run_manifest"], str):
        raise ProtocolError("legacy selection has no run manifest path")
    manifest_info = _validate_manifest(training["run_manifest"], policy)
    expected_training = {
        "run_manifest_sha256": manifest_info["sha256"],
        "manifest_payload_sha256": manifest_info["payload_sha256"],
        "training_commit": manifest_info["training_commit"],
    }
    for key, value in expected_training.items():
        _require_equal(
            training.get(key), value, f"legacy selection.training.{key}"
        )

    benchmark = selection["benchmark"]
    _require_exact_keys(
        benchmark,
        {"path", "sha256", "dev_users", "test_users"},
        "legacy selection.benchmark",
    )
    if not isinstance(benchmark["path"], str):
        raise ProtocolError("legacy selection has no benchmark path")
    benchmark_info = _benchmark_info(benchmark["path"])
    _require_equal(
        benchmark["sha256"],
        benchmark_info["sha256"],
        "legacy selection.benchmark.sha256",
    )
    _require_equal(
        benchmark["dev_users"], 400, "legacy selection.benchmark.dev_users"
    )
    _require_equal(
        benchmark["test_users"], 400, "legacy selection.benchmark.test_users"
    )

    model = selection["model"]
    _require_exact_keys(
        model,
        {"path", "revision", "weight_sha256"},
        "legacy selection.model",
    )
    if not isinstance(model["path"], str):
        raise ProtocolError("legacy selection has no model path")
    model_path = _validate_model_path(model["path"], policy)
    for key in ("revision", "weight_sha256"):
        _require_equal(
            model.get(key),
            policy_spec["model_identity"][key],
            f"legacy selection.model.{key}",
        )

    candidates = selection["candidates"]
    if not isinstance(candidates, list) or len(candidates) != len(EXPECTED_STEPS):
        raise ProtocolError(
            "legacy selection does not contain exactly eight candidates"
        )
    adapter_records = []
    for expected_step, candidate in zip(EXPECTED_STEPS, candidates):
        _require_exact_keys(
            candidate,
            {
                "step",
                "adapter_path",
                "adapter_model_sha256",
                "adapter_config_sha256",
                "provenance_sha256",
                "dev_result_path",
                "dev_result_sha256",
                "ndcg@5",
            },
            f"legacy selection candidate step {expected_step}",
        )
        _require_equal(
            candidate["step"],
            expected_step,
            f"legacy selection candidate step {expected_step}",
        )
        if not isinstance(candidate["adapter_path"], str):
            raise ProtocolError(
                f"legacy selection step {expected_step} has no adapter path"
            )
        record = _validate_one_adapter(
            candidate["adapter_path"],
            expected_step,
            manifest_info,
            policy,
        )
        _require_equal(
            _canonical_path(candidate["adapter_path"]),
            record["path"],
            f"legacy selection step {expected_step} adapter path",
        )
        for key in (
            "adapter_model_sha256",
            "adapter_config_sha256",
            "provenance_sha256",
        ):
            _require_equal(
                candidate.get(key),
                record[key],
                f"legacy selection step {expected_step} {key}",
            )
        _require_hex(
            candidate.get("dev_result_sha256"),
            f"legacy selection step {expected_step} dev result hash",
        )
        if not isinstance(candidate.get("dev_result_path"), str):
            raise ProtocolError(
                f"legacy selection step {expected_step} has no dev result path"
            )
        adapter_records.append(record)

    dev_parents = {
        os.path.dirname(_canonical_path(candidate["dev_result_path"]))
        for candidate in candidates
    }
    if len(dev_parents) != 1:
        raise ProtocolError(
            "legacy selection dev results do not share one canonical directory"
        )
    dev_records, inference_signature = collect_dev_results(
        dev_parents.pop(),
        adapter_records,
        benchmark_info,
        model_path,
        DEV_TAG,
    )
    for candidate in candidates:
        step = candidate["step"]
        dev = dev_records[step]
        _require_equal(
            _canonical_path(candidate["dev_result_path"]),
            dev["result_path"],
            f"legacy selection step {step} dev result path",
        )
        _require_equal(
            candidate["dev_result_sha256"],
            dev["result_sha256"],
            f"legacy selection step {step} dev result hash",
        )
        _require_equal(
            candidate["ndcg@5"],
            dev["ndcg@5"],
            f"legacy selection step {step} dev NDCG@5",
        )

    dev_protocol = selection["dev_protocol"]
    _require_exact_keys(
        dev_protocol,
        {"tag", "slice", "k", "n_cand", "n", "skip_base", "inference"},
        "legacy selection.dev_protocol",
    )
    expected_dev_protocol = {
        "tag": DEV_TAG,
        "slice": "dev",
        "k": DEV_K,
        "n_cand": DEV_N_CAND,
        "n": None,
        "skip_base": True,
        "inference": inference_signature,
    }
    _require_equal(
        dev_protocol,
        expected_dev_protocol,
        "legacy selection.dev_protocol",
    )

    best_step = choose_step(dev_records, score_key="ndcg@5")
    selected = selection["selected"]
    _require_exact_keys(
        selected,
        {
            "step",
            "ndcg@5",
            "source_adapter",
            "frozen_adapter",
            "adapter_model_sha256",
            "adapter_config_sha256",
            "provenance_sha256",
        },
        "legacy selection.selected",
    )
    _require_equal(
        selected["step"], best_step, "legacy selection.selected.step"
    )
    chosen = next(
        candidate for candidate in candidates if candidate["step"] == best_step
    )
    _require_equal(
        selected["ndcg@5"],
        chosen["ndcg@5"],
        "legacy selection.selected.ndcg@5",
    )
    _require_equal(
        _canonical_path(selected["source_adapter"]),
        _canonical_path(chosen["adapter_path"]),
        "legacy selection.selected.source_adapter",
    )
    frozen_adapter = _canonical_path(selected["frozen_adapter"])
    _require_equal(
        frozen_adapter,
        _frozen_adapter_path(state_dir),
        "legacy selection.selected.frozen_adapter",
    )
    frozen_hashes = _frozen_hashes(frozen_adapter)
    for key, value in frozen_hashes.items():
        _require_equal(
            selected.get(key), value, f"legacy selection.selected.{key}"
        )
        _require_equal(
            chosen.get(key), value, f"legacy selected candidate.{key}"
        )

    return {
        "path": selection_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload": selection,
        "benchmark": benchmark_info,
        "model": model_path,
        "frozen_adapter": frozen_adapter,
        "policy": policy,
        "legacy_schema": True,
    }


def _load_selection(
    state_dir,
    *,
    verify_dev=True,
    verify_current_stack=True,
    policy=DEFAULT_POLICY,
):
    policy_spec = _policy_spec(policy)
    state_dir = _canonical_path(state_dir)
    selection_path = _selection_path(state_dir)
    raw, selection = _read_json_bytes(selection_path, "OPD dev selection")
    if selection.get("schema_version") == LEGACY_SCHEMA_VERSION:
        if policy != DEFAULT_POLICY:
            raise ProtocolError(
                "schema-1 evaluation state is valid only for 1.7b-pro"
            )
        raise ProtocolError(
            "legacy 1.7b-pro schema-1 state is read-only; only validate-test "
            "may inspect its already sealed complete sweep"
        )
    expected_top = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "kind": policy_spec["selection_kind"],
        "recipe": contract.RECIPE,
        "data_recipe": contract.DATA_RECIPE,
        "expected_steps": list(EXPECTED_STEPS),
        "selection_rule": {
            "primary_metric": SELECTION_SCORE_KEY,
            "reported_metric": "ndcg@5",
            "tie_break": "earliest-step-on-exact-unrounded-tie",
            "other_metrics_used": False,
        },
    }
    for key, value in expected_top.items():
        _require_equal(selection.get(key), value, f"selection.{key}")
    if policy != DEFAULT_POLICY:
        _require_equal(selection.get("policy"), policy, "selection.policy")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or [
        item.get("step") for item in candidates if isinstance(item, dict)
    ] != list(EXPECTED_STEPS):
        raise ProtocolError("selection does not contain the eight ordered steps")
    training = selection.get("training", {})
    manifest_info = _validate_manifest(
        training.get("run_manifest", ""), policy
    )
    expected_training = {
        "run_manifest_sha256": manifest_info["sha256"],
        "manifest_payload_sha256": manifest_info["payload_sha256"],
        "training_commit": manifest_info["training_commit"],
    }
    for key, value in expected_training.items():
        _require_equal(
            training.get(key), value, f"selection.training.{key}"
        )
    benchmark = selection.get("benchmark", {})
    benchmark_info = _frozen_benchmark_info(benchmark.get("path", ""))
    _require_equal(
        benchmark.get("sha256"),
        benchmark_info["sha256"],
        "selection.benchmark.sha256",
    )
    _require_equal(
        benchmark.get("dev_users"), 400, "selection.benchmark.dev_users"
    )
    _require_equal(
        benchmark.get("test_users"), 400, "selection.benchmark.test_users"
    )
    model = selection.get("model", {})
    model_path = _validate_model_path(model.get("path", ""), policy)
    for key, value in policy_spec["model_identity"].items():
        _require_equal(model.get(key), value, f"selection.model.{key}")
    evaluation_stack = selection.get("evaluation_stack")
    if not isinstance(evaluation_stack, dict):
        raise ProtocolError("selection has no frozen evaluation stack")
    stack_sha256 = _evaluation_stack_sha256(evaluation_stack)
    _require_equal(
        selection.get("evaluation_stack_sha256"),
        stack_sha256,
        "selection.evaluation_stack_sha256",
    )
    if verify_current_stack:
        current_stack = _evaluation_stack(model_path, policy)
        _require_equal(
            current_stack, evaluation_stack, "selection.evaluation_stack"
        )
    dev_protocol = selection.get("dev_protocol")
    if not isinstance(dev_protocol, dict):
        raise ProtocolError("selection has no dev protocol")
    expected_dev_protocol = {
        "tag": _dev_result_tag(evaluation_stack),
        "slice": "dev",
        "k": DEV_K,
        "n_cand": DEV_N_CAND,
        "n": None,
        "skip_base": True,
    }
    for key, value in expected_dev_protocol.items():
        _require_equal(
            dev_protocol.get(key), value, f"selection.dev_protocol.{key}"
        )
    if not isinstance(dev_protocol.get("inference"), dict):
        raise ProtocolError("selection has no dev inference signature")
    selected = selection.get("selected", {})
    if selected.get("step") not in EXPECTED_STEPS:
        raise ProtocolError("selection has an invalid selected step")
    frozen_adapter = _canonical_path(selected.get("frozen_adapter", ""))
    _require_equal(
        frozen_adapter,
        _frozen_adapter_path(state_dir),
        "selection.selected.frozen_adapter",
    )
    hashes = _frozen_hashes(frozen_adapter)
    for key, value in hashes.items():
        _require_equal(
            selected.get(key), value, f"selection.selected.{key}"
        )
    chosen_candidates = [
        item for item in candidates if item["step"] == selected["step"]
    ]
    if len(chosen_candidates) != 1:
        raise ProtocolError("selection has no unique selected candidate")
    chosen = chosen_candidates[0]
    for key in (
        "adapter_model_sha256",
        "adapter_config_sha256",
        "provenance_sha256",
    ):
        _require_equal(
            chosen.get(key), selected.get(key), f"selected candidate {key}"
        )
    scores = {
        item["step"]: {
            SELECTION_SCORE_KEY: item.get(SELECTION_SCORE_KEY)
        }
        for item in candidates
    }
    _require_equal(
        choose_step(scores), selected["step"], "selection.selected.step"
    )
    for key in ("ndcg@5", SELECTION_SCORE_KEY):
        _require_equal(
            chosen.get(key), selected.get(key), f"selected candidate {key}"
        )
    if verify_dev:
        for item in candidates:
            path = item.get("dev_result_path")
            if not isinstance(path, str) or not os.path.isfile(path):
                raise ProtocolError(
                    f"selection dev result is missing for step {item['step']}"
                )
            _require_equal(
                sha256_file(path),
                item.get("dev_result_sha256"),
                f"selection step {item['step']} dev result hash",
            )
    return {
        "path": selection_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload": selection,
        "benchmark": benchmark_info,
        "model": model_path,
        "frozen_adapter": frozen_adapter,
        "policy": policy,
    }


def _cell_record(k, n_cand):
    return {
        "k": k,
        "n_cand": n_cand,
        "filename": f"cell_k{k:03d}_N{n_cand:03d}.json",
    }


def _receipt_filename(k, n_cand):
    return f"cell_k{k:03d}_N{n_cand:03d}.receipt.json"


def _test_started_payload(
    selection_info, test_dir, staging_dir, nonce
):
    selection = selection_info["payload"]
    policy = selection_info["policy"]
    policy_spec = _policy_spec(policy)
    _require_hex(nonce, "test nonce")
    payload = {
        "schema_version": TEST_STARTED_SCHEMA_VERSION,
        "kind": policy_spec["test_kind"],
        "recipe": contract.RECIPE,
        "selection_path": selection_info["path"],
        "selection_sha256": selection_info["sha256"],
        "selected_step": selection["selected"]["step"],
        "selected_adapter_sha256": selection["selected"][
            "adapter_model_sha256"
        ],
        "benchmark_sha256": selection["benchmark"]["sha256"],
        "test_dir": _canonical_path(test_dir),
        "staging_dir": _canonical_path(staging_dir),
        "receipt_dir": _test_receipt_dir(
            os.path.dirname(selection_info["path"])
        ),
        "nonce": nonce,
        "test_protocol": {
            "tag": _test_result_tag(nonce),
            "slice": "test",
            "n": None,
            "skip_base": True,
            "cells": [_cell_record(*cell) for cell in TEST_CELLS],
        },
    }
    if policy != DEFAULT_POLICY:
        payload["policy"] = policy
    return payload


def _load_test_started(
    state_dir, selection_info, test_dir, staging_dir=None
):
    marker_path = _test_started_path(state_dir)
    raw, marker = _read_json_bytes(marker_path, "OPD test-start ledger")
    nonce = marker.get("nonce")
    _require_hex(nonce, "test-start nonce")
    bound_staging = marker.get("staging_dir")
    if not isinstance(bound_staging, str):
        raise ProtocolError("test-start ledger has no staging directory")
    prospective = _test_started_payload(
        selection_info, test_dir, bound_staging, nonce
    )
    _require_equal(marker, prospective, "test-start ledger")
    if staging_dir is not None:
        _require_equal(
            _canonical_path(staging_dir),
            bound_staging,
            "test-start staging directory",
        )
    return {
        "path": marker_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload": marker,
        "status": "matched",
    }


def _legacy_test_started_payload(selection_info, test_dir):
    selection = selection_info["payload"]
    return {
        "schema_version": LEGACY_SCHEMA_VERSION,
        "kind": "opd-1.7b-test-once",
        "recipe": contract.RECIPE,
        "selection_path": selection_info["path"],
        "selection_sha256": selection_info["sha256"],
        "selected_step": selection["selected"]["step"],
        "selected_adapter_sha256": selection["selected"][
            "adapter_model_sha256"
        ],
        "benchmark_sha256": selection["benchmark"]["sha256"],
        "test_dir": _canonical_path(test_dir),
        "test_protocol": {
            "tag": TEST_TAG,
            "slice": "test",
            "n": None,
            "skip_base": True,
            "cells": [_cell_record(*cell) for cell in TEST_CELLS],
        },
    }


def _load_legacy_test_started(state_dir, selection_info, test_dir):
    marker_path = _test_started_path(state_dir)
    raw, marker = _read_json_bytes(
        marker_path, "legacy OPD test-start ledger"
    )
    _require_equal(
        marker,
        _legacy_test_started_payload(selection_info, test_dir),
        "legacy test-start ledger",
    )
    return {
        "path": marker_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "payload": marker,
    }


def _validate_legacy_test_payload(
    payload, label, selection_info, k, n_cand, staging_dir
):
    inference = selection_info["payload"]["dev_protocol"]["inference"]
    scores, signature = _validate_score_payload(
        payload,
        label=label,
        model=selection_info["model"],
        benchmark_info=selection_info["benchmark"],
        split="test",
        tag=TEST_TAG,
        checkpoint="best",
        expected_loras=[("best", selection_info["frozen_adapter"])],
        k=k,
        n_cand=n_cand,
        outdir=staging_dir,
    )
    if signature != inference:
        raise ProtocolError(f"{label} does not reuse the dev inference stack")
    return scores


def _scan_legacy_complete_test_tree(test_dir, selection_info):
    test_dir = _canonical_path(test_dir)
    if not os.path.isdir(test_dir):
        raise ProtocolError(
            f"legacy canonical test tree is missing: {test_dir}"
        )
    expected_names = {
        _cell_record(k, n_cand)["filename"]: (k, n_cand)
        for k, n_cand in TEST_CELLS
    }
    legacy_staging_dir = os.path.join(
        os.path.dirname(test_dir), ".test_staging"
    )
    entries = sorted(os.listdir(test_dir))
    if set(entries) != set(expected_names):
        raise ProtocolError(
            "legacy test tree must contain exactly the twelve sealed cells; "
            f"missing={sorted(set(expected_names) - set(entries))}, "
            f"extra={sorted(set(entries) - set(expected_names))}"
        )
    found = {}
    for filename, cell in expected_names.items():
        path = os.path.join(test_dir, filename)
        if not os.path.isfile(path):
            raise ProtocolError(
                f"legacy test cell is not a regular file: {path}"
            )
        _, payload = _read_json_bytes(path, "legacy canonical test result")
        _validate_legacy_test_payload(
            payload,
            f"legacy test k={cell[0]} N={cell[1]}",
            selection_info,
            *cell,
            legacy_staging_dir,
        )
        found[cell] = {
            "path": _canonical_path(path),
            "sha256": sha256_file(path),
        }
    return found


def _validate_legacy_completion_ledger(
    state_dir, selection_info, found
):
    path = _legacy_test_validation_path(state_dir)
    _, ledger = _read_json_bytes(path, "legacy OPD test validation ledger")
    expected = {
        "selection_sha256": selection_info["sha256"],
        "selected_step": selection_info["payload"]["selected"]["step"],
        "cells": [
            {
                **_cell_record(*cell),
                "sha256": found[cell]["sha256"],
            }
            for cell in TEST_CELLS
        ],
    }
    _require_equal(ledger, expected, "legacy test validation ledger")
    return {"path": path, "sha256": sha256_file(path), "payload": ledger}


def _create_test_started(
    state_dir, selection_info, test_dir, staging_dir
):
    marker_path = _test_started_path(state_dir)
    prospective = _test_started_payload(
        selection_info,
        test_dir,
        staging_dir,
        secrets.token_hex(32),
    )
    try:
        status = ensure_immutable_json(
            marker_path, prospective, label="OPD test-start ledger"
        )
    except ManifestError as exc:
        raise ProtocolError(str(exc)) from exc
    if status != "created":
        raise ProtocolError("test-start ledger was created concurrently")
    info = _load_test_started(
        state_dir, selection_info, test_dir, staging_dir
    )
    info["status"] = status
    return info


def _validate_test_payload(
    payload, label, selection_info, marker_info, k, n_cand
):
    selection = selection_info["payload"]
    inference = selection["dev_protocol"]["inference"]
    expected_loras = [("best", selection_info["frozen_adapter"])]
    scores, signature = _validate_score_payload(
        payload,
        label=label,
        model=selection_info["model"],
        benchmark_info=selection_info["benchmark"],
        split="test",
        tag=marker_info["payload"]["test_protocol"]["tag"],
        checkpoint="best",
        expected_loras=expected_loras,
        k=k,
        n_cand=n_cand,
        outdir=marker_info["payload"]["staging_dir"],
    )
    if signature != inference:
        raise ProtocolError(f"{label} does not reuse the dev inference stack")
    return scores


def _receipt_payload(
    selection_info, marker_info, k, n_cand, result_path, result_sha256
):
    return {
        "schema_version": TEST_RECEIPT_SCHEMA_VERSION,
        "kind": "opd-test-cell-receipt",
        "selection_sha256": selection_info["sha256"],
        "test_started_sha256": marker_info["sha256"],
        "nonce": marker_info["payload"]["nonce"],
        "cell": _cell_record(k, n_cand),
        "result_path": _canonical_path(result_path),
        "result_sha256": result_sha256,
    }


def _scan_test_tree(
    state_dir, test_dir, selection_info, marker_info
):
    test_dir = _canonical_path(test_dir)
    receipt_dir = _test_receipt_dir(state_dir)
    _require_equal(
        marker_info["payload"]["receipt_dir"],
        receipt_dir,
        "test-start receipt directory",
    )
    if os.path.islink(test_dir):
        raise ProtocolError(f"test result path is a symlink: {test_dir}")
    if os.path.exists(test_dir) and not os.path.isdir(test_dir):
        raise ProtocolError(f"test result path is not a directory: {test_dir}")
    if os.path.islink(receipt_dir):
        raise ProtocolError(
            f"test receipt path is a symlink: {receipt_dir}"
        )
    if os.path.exists(receipt_dir) and not os.path.isdir(receipt_dir):
        raise ProtocolError(
            f"test receipt path is not a directory: {receipt_dir}"
        )
    expected_names = {
        _cell_record(k, n_cand)["filename"]: (k, n_cand)
        for k, n_cand in TEST_CELLS
    }
    expected_receipts = {
        _receipt_filename(k, n_cand): (k, n_cand)
        for k, n_cand in TEST_CELLS
    }
    result_entries = sorted(os.listdir(test_dir)) if os.path.isdir(test_dir) else []
    receipt_entries = (
        sorted(os.listdir(receipt_dir)) if os.path.isdir(receipt_dir) else []
    )
    unexpected_results = set(result_entries) - set(expected_names)
    unexpected_receipts = set(receipt_entries) - set(expected_receipts)
    if unexpected_results:
        raise ProtocolError(
            "unexpected canonical test result: "
            f"{sorted(unexpected_results)[0]}"
        )
    if unexpected_receipts:
        raise ProtocolError(
            "unexpected immutable test receipt: "
            f"{sorted(unexpected_receipts)[0]}"
        )

    found = {}
    result_names = set(result_entries)
    receipt_names = set(receipt_entries)
    for cell in TEST_CELLS:
        filename = _cell_record(*cell)["filename"]
        receipt_name = _receipt_filename(*cell)
        has_result = filename in result_names
        has_receipt = receipt_name in receipt_names
        if has_result != has_receipt:
            if has_receipt:
                raise ProtocolError(
                    f"accepted test cell {cell} lost its canonical result"
                )
            raise ProtocolError(
                f"accepted test cell {cell} lost its immutable receipt"
            )
        if not has_result:
            continue
        path = os.path.join(test_dir, filename)
        receipt_path = os.path.join(receipt_dir, receipt_name)
        if (
            os.path.islink(path)
            or os.path.islink(receipt_path)
            or not os.path.isfile(path)
            or not os.path.isfile(receipt_path)
        ):
            raise ProtocolError(
                f"accepted test cell {cell} is not stored as regular files"
            )
        _, payload = _read_json_bytes(path, "canonical test result")
        _validate_test_payload(
            payload,
            f"test k={cell[0]} N={cell[1]}",
            selection_info,
            marker_info,
            *cell,
        )
        result_sha256 = sha256_file(path)
        _, receipt = _read_json_bytes(
            receipt_path, "immutable test receipt"
        )
        expected_receipt = _receipt_payload(
            selection_info,
            marker_info,
            *cell,
            path,
            result_sha256,
        )
        _require_equal(
            receipt, expected_receipt, f"test receipt for cell {cell}"
        )
        found[cell] = {
            "path": _canonical_path(path),
            "sha256": result_sha256,
            "receipt_path": _canonical_path(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
        }
    return found


def _score_command_prefix(
    selection_info, *, split, tag, outdir, checkpoint_spec
):
    selection = selection_info["payload"]
    inference = selection["dev_protocol"]["inference"]
    command = [
        sys.executable,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "benchmark",
            "score_sid.py",
        ),
        "--model",
        selection_info["model"],
        "--bench",
        selection_info["benchmark"]["path"],
        "--slice",
        split,
        "--shuffle",
        "0",
        "--render",
        "sid",
        "--no-profile",
        "--top-k",
        str(TOP_K),
        "--max-tokens",
        str(MAX_TOKENS),
        "--extract-tokens",
        str(EXTRACT_TOKENS),
        "--k-ndcg",
        str(K_NDCG),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-lora-rank",
        "32",
        "--tp",
        str(inference["tp"]),
        "--tag",
        tag,
        "--lora-dirs",
        checkpoint_spec,
        "--skip-base",
        "--outdir",
        _canonical_path(outdir),
    ]
    if inference["no_eager"]:
        command.append("--no-eager")
    return command


def build_dev_command(
    *,
    state_dir,
    dev_dir,
    adapter_root,
    run_manifest,
    benchmark,
    model,
    tp,
    no_eager=False,
    policy=DEFAULT_POLICY,
):
    """Return the one explicit dev-only scorer command after preflight checks."""

    if not isinstance(tp, int) or isinstance(tp, bool) or tp <= 0:
        raise ProtocolError("dev command requires an explicit positive --tp")
    state_dir = _canonical_path(state_dir)
    dev_dir = _canonical_path(dev_dir)
    if os.path.exists(_selection_path(state_dir)) or os.path.exists(
        _test_started_path(state_dir)
    ):
        raise ProtocolError("formal dev selection is already sealed")
    if os.path.exists(dev_dir) and os.listdir(dev_dir):
        raise ProtocolError(
            f"refusing to overwrite nonempty dev result directory: {dev_dir}"
        )
    _policy_spec(policy)
    adapter_records, _ = validate_adapters(
        adapter_root, run_manifest, policy
    )
    benchmark_info = _frozen_benchmark_info(benchmark)
    model = _validate_model_path(model, policy)
    evaluation_stack = _evaluation_stack(model, policy)
    selection_stub = {
        "payload": {
            "dev_protocol": {
                "inference": {"tp": tp, "no_eager": bool(no_eager)}
            }
        },
        "model": model,
        "benchmark": benchmark_info,
    }
    command = _score_command_prefix(
        selection_stub,
        split="dev",
        tag=_dev_result_tag(evaluation_stack),
        outdir=dev_dir,
        checkpoint_spec=lora_spec(adapter_records),
    )
    command.extend(["--k", str(DEV_K), "--n-cand", str(DEV_N_CAND)])
    return command


def arm_and_plan_test(
    *,
    state_dir,
    test_dir,
    staging_dir,
    confirm_test_once=False,
    policy=DEFAULT_POLICY,
):
    """Irreversibly arm test and return commands for only missing cells."""

    if not confirm_test_once:
        raise ProtocolError(
            "test remains closed without explicit --confirm-test-once"
        )
    state_dir = _canonical_path(state_dir)
    test_dir = _canonical_path(test_dir)
    staging_dir = _canonical_path(staging_dir)
    for protected, label in (
        (state_dir, "evaluation state"),
        (test_dir, "canonical test tree"),
    ):
        try:
            common = os.path.commonpath([staging_dir, protected])
        except ValueError:
            common = None
        if common in {staging_dir, protected}:
            raise ProtocolError(
                f"staging directory must not overlap the {label}"
            )
    with _state_lock(state_dir):
        selection_info = _load_selection(
            state_dir, verify_dev=True, policy=policy
        )
        marker_path = _test_started_path(state_dir)
        if not os.path.exists(marker_path):
            for path, label in (
                (test_dir, "canonical test tree"),
                (_test_receipt_dir(state_dir), "immutable receipt tree"),
                (staging_dir, "staging directory"),
            ):
                if os.path.exists(path) and not os.path.isdir(path):
                    raise ProtocolError(f"{label} is not a directory: {path}")
                if os.path.isdir(path) and os.listdir(path):
                    if path == test_dir:
                        raise ProtocolError(
                            "test results exist without a test-start ledger; "
                            "refusing to adopt them"
                        )
                    raise ProtocolError(
                        f"{label} must be empty before test starts: {path}"
                    )
            # Directory creation is reversible. The random nonce and immutable
            # marker are written only after every preflight check succeeds.
            os.makedirs(staging_dir, exist_ok=True)
            marker_info = _create_test_started(
                state_dir,
                selection_info,
                test_dir,
                staging_dir,
            )
        else:
            marker_info = _load_test_started(
                state_dir,
                selection_info,
                test_dir,
                staging_dir,
            )
        found = _scan_test_tree(
            state_dir, test_dir, selection_info, marker_info
        )
        missing = [cell for cell in TEST_CELLS if cell not in found]
        if os.path.exists(staging_dir) and not os.path.isdir(staging_dir):
            raise ProtocolError(
                f"staging path is not a directory: {staging_dir}"
            )
        if os.path.isdir(staging_dir) and os.listdir(staging_dir):
            raise ProtocolError(
                f"staging directory must be empty: {staging_dir}"
            )
        os.makedirs(staging_dir, exist_ok=True)
        selected_spec = f"best={selection_info['frozen_adapter']}"
        result_tag = marker_info["payload"]["test_protocol"]["tag"]
        commands = []
        missing_history = [k for k in HISTORY_KS if (k, 30) in missing]
        if missing_history:
            command = _score_command_prefix(
                selection_info,
                split="test",
                tag=result_tag,
                outdir=staging_dir,
                checkpoint_spec=selected_spec,
            )
            command.extend(
                [
                    "--k-grid",
                    ",".join(map(str, missing_history)),
                    "--n-cand",
                    "30",
                ]
            )
            commands.append(command)
        for n_cand in CANDIDATE_NS:
            if n_cand == 30 or (64, n_cand) not in missing:
                continue
            command = _score_command_prefix(
                selection_info,
                split="test",
                tag=result_tag,
                outdir=staging_dir,
                checkpoint_spec=selected_spec,
            )
            command.extend(["--k", "64", "--n-cand", str(n_cand)])
            commands.append(command)
        return {
            "marker_status": marker_info["status"],
            "test_started": marker_info["payload"],
            "completed": [_cell_record(*cell) for cell in found],
            "missing": [_cell_record(*cell) for cell in missing],
            "commands": commands,
        }


def accept_test_result(
    *, state_dir, test_dir, result, policy=DEFAULT_POLICY
):
    """Validate and consume one staged JSON into its canonical cell once."""

    state_dir = _canonical_path(state_dir)
    test_dir = _canonical_path(test_dir)
    result = _canonical_path(result)
    for protected_root, label in (
        (state_dir, "evaluation state"),
        (test_dir, "canonical test tree"),
    ):
        try:
            inside = os.path.commonpath([result, protected_root]) == protected_root
        except ValueError:
            inside = False
        if inside:
            raise ProtocolError(
                f"staged result must not be inside the {label}: {result}"
            )
    with _state_lock(state_dir):
        selection_info = _load_selection(
            state_dir, verify_dev=True, policy=policy
        )
        marker_path = _test_started_path(state_dir)
        if not os.path.isfile(marker_path):
            raise ProtocolError("test is not armed; missing test_started.json")
        marker_info = _load_test_started(
            state_dir, selection_info, test_dir
        )
        staging_dir = marker_info["payload"]["staging_dir"]
        _require_equal(
            os.path.dirname(result),
            staging_dir,
            "staged result parent directory",
        )
        _scan_test_tree(
            state_dir, test_dir, selection_info, marker_info
        )
        raw, payload = _read_json_bytes(result, "staged test result")
        meta = payload.get("meta", {})
        k, n_cand = meta.get("k"), meta.get("n_cand")
        if (k, n_cand) not in TEST_CELLS:
            raise ProtocolError(
                f"staged result cell {(k, n_cand)!r} was not preregistered"
            )
        _validate_test_payload(
            payload,
            f"staged test k={k} N={n_cand}",
            selection_info,
            marker_info,
            k,
            n_cand,
        )
        os.makedirs(test_dir, exist_ok=True)
        receipt_dir = _test_receipt_dir(state_dir)
        os.makedirs(receipt_dir, exist_ok=True)
        target = os.path.join(
            test_dir, _cell_record(k, n_cand)["filename"]
        )
        receipt_path = os.path.join(
            receipt_dir, _receipt_filename(k, n_cand)
        )
        if os.path.exists(target) or os.path.exists(receipt_path):
            raise ProtocolError(
                f"refusing to overwrite accepted test cell {(k, n_cand)}"
            )
        result_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            _atomic_create_bytes(target, raw)
        except FileExistsError as exc:
            raise ProtocolError(
                f"refusing to overwrite accepted test cell {(k, n_cand)}"
            ) from exc
        _require_equal(
            sha256_file(target),
            result_sha256,
            f"accepted test cell {(k, n_cand)} hash",
        )
        receipt = _receipt_payload(
            selection_info,
            marker_info,
            k,
            n_cand,
            target,
            result_sha256,
        )
        receipt_raw = (
            json.dumps(
                receipt, ensure_ascii=False, sort_keys=True, indent=2
            )
            + "\n"
        ).encode("utf-8")
        try:
            _atomic_create_bytes(receipt_path, receipt_raw)
        except Exception:
            # This invocation created target, so rolling it back is safe.
            # A process death between these calls is detected fail-closed by
            # the result/receipt parity check on the next transition.
            try:
                os.unlink(target)
            except FileNotFoundError:
                pass
            raise
        try:
            os.unlink(result)
        except OSError as exc:
            raise ProtocolError(
                "test cell was accepted but its staged source could not be "
                f"consumed: {result}: {exc}"
            ) from exc
        return {
            "cell": _cell_record(k, n_cand),
            "path": _canonical_path(target),
            "sha256": result_sha256,
            "receipt_path": _canonical_path(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
        }


def validate_complete_test(
    *,
    state_dir,
    test_dir,
    policy=DEFAULT_POLICY,
    verify_current_stack=True,
):
    """Require the immutable selected adapter on all twelve test cells.

    ``verify_current_stack=False`` is reserved for publishing an already
    sealed historical release. The selection's frozen evaluator identity and
    every bound dev result, test result, and receipt are still validated.
    """

    state_dir = _canonical_path(state_dir)
    test_dir = _canonical_path(test_dir)
    with _state_lock(state_dir):
        _, selection_header = _read_json_bytes(
            _selection_path(state_dir), "OPD dev selection"
        )
        schema_version = selection_header.get("schema_version")
        if (
            isinstance(schema_version, int)
            and not isinstance(schema_version, bool)
            and schema_version == LEGACY_SCHEMA_VERSION
        ):
            selection_info = _load_legacy_selection_for_validation(
                state_dir, policy=policy
            )
            if not os.path.isfile(_test_started_path(state_dir)):
                raise ProtocolError(
                    "legacy test is not sealed; missing test_started.json"
                )
            marker_info = _load_legacy_test_started(
                state_dir, selection_info, test_dir
            )
            found = _scan_legacy_complete_test_tree(
                test_dir, selection_info
            )
            ledger_info = _validate_legacy_completion_ledger(
                state_dir, selection_info, found
            )
            return {
                "legacy_read_only": True,
                "schema_version": LEGACY_SCHEMA_VERSION,
                "selection_sha256": marker_info["payload"][
                    "selection_sha256"
                ],
                "selected_step": marker_info["payload"]["selected_step"],
                "legacy_completion_ledger_sha256": ledger_info["sha256"],
                "cells": [
                    {
                        **_cell_record(*cell),
                        "sha256": found[cell]["sha256"],
                    }
                    for cell in TEST_CELLS
                ],
            }
        selection_info = _load_selection(
            state_dir,
            verify_dev=True,
            verify_current_stack=verify_current_stack,
            policy=policy,
        )
        if not os.path.isfile(_test_started_path(state_dir)):
            raise ProtocolError("test is not armed")
        marker_info = _load_test_started(
            state_dir, selection_info, test_dir
        )
        found = _scan_test_tree(
            state_dir, test_dir, selection_info, marker_info
        )
        missing = [cell for cell in TEST_CELLS if cell not in found]
        if missing:
            raise ProtocolError(f"formal test sweep is missing cells: {missing}")
        return {
            "selection_sha256": marker_info["payload"][
                "selection_sha256"
            ],
            "selected_step": marker_info["payload"]["selected_step"],
            "cells": [
                {
                    **_cell_record(*cell),
                    "sha256": found[cell]["sha256"],
                    "receipt_sha256": found[cell]["receipt_sha256"],
                }
                for cell in TEST_CELLS
            ],
        }


def _add_selection_inputs(parser):
    parser.add_argument(
        "--policy",
        choices=contract.SUPPORTED_POLICIES,
        default=DEFAULT_POLICY,
        help=f"policy scale (default: {DEFAULT_POLICY})",
    )
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--dev-dir", required=True)
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--model", required=True)


def _print_json(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fail-closed formal OPD evaluation gate"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dev = subparsers.add_parser(
        "dev-command", help="validate inputs and print the explicit dev command"
    )
    _add_selection_inputs(dev)
    dev.add_argument("--tp", type=int, required=True)
    dev.add_argument("--no-eager", action="store_true")

    select = subparsers.add_parser(
        "select", help="validate dev and freeze the unique best adapter"
    )
    _add_selection_inputs(select)

    test = subparsers.add_parser(
        "test-command",
        help="arm test once and print commands for missing preregistered cells",
    )
    test.add_argument(
        "--policy",
        choices=contract.SUPPORTED_POLICIES,
        default=DEFAULT_POLICY,
        help=f"policy scale (default: {DEFAULT_POLICY})",
    )
    test.add_argument("--state-dir", required=True)
    test.add_argument("--test-dir", required=True)
    test.add_argument("--staging-dir", required=True)
    test.add_argument("--confirm-test-once", action="store_true")

    accept = subparsers.add_parser(
        "accept-test", help="validate and exclusively accept one staged test JSON"
    )
    accept.add_argument(
        "--policy",
        choices=contract.SUPPORTED_POLICIES,
        default=DEFAULT_POLICY,
        help=f"policy scale (default: {DEFAULT_POLICY})",
    )
    accept.add_argument("--state-dir", required=True)
    accept.add_argument("--test-dir", required=True)
    accept.add_argument("--result", required=True)

    validate = subparsers.add_parser(
        "validate-test", help="validate the complete twelve-cell test sweep"
    )
    validate.add_argument(
        "--policy",
        choices=contract.SUPPORTED_POLICIES,
        default=DEFAULT_POLICY,
        help=f"policy scale (default: {DEFAULT_POLICY})",
    )
    validate.add_argument("--state-dir", required=True)
    validate.add_argument("--test-dir", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "dev-command":
            command = build_dev_command(
                state_dir=args.state_dir,
                dev_dir=args.dev_dir,
                adapter_root=args.adapter_root,
                run_manifest=args.run_manifest,
                benchmark=args.bench,
                model=args.model,
                tp=args.tp,
                no_eager=args.no_eager,
                policy=args.policy,
            )
            _print_json({"command": command})
        elif args.command == "select":
            _print_json(
                seal_dev_selection(
                    state_dir=args.state_dir,
                    dev_dir=args.dev_dir,
                    adapter_root=args.adapter_root,
                    run_manifest=args.run_manifest,
                    benchmark=args.bench,
                    model=args.model,
                    policy=args.policy,
                )
            )
        elif args.command == "test-command":
            _print_json(
                arm_and_plan_test(
                    state_dir=args.state_dir,
                    test_dir=args.test_dir,
                    staging_dir=args.staging_dir,
                    confirm_test_once=args.confirm_test_once,
                    policy=args.policy,
                )
            )
        elif args.command == "accept-test":
            _print_json(
                accept_test_result(
                    state_dir=args.state_dir,
                    test_dir=args.test_dir,
                    result=args.result,
                    policy=args.policy,
                )
            )
        elif args.command == "validate-test":
            _print_json(
                validate_complete_test(
                    state_dir=args.state_dir,
                    test_dir=args.test_dir,
                    policy=args.policy,
                )
            )
        else:  # pragma: no cover - argparse requires a known subcommand.
            raise ProtocolError("no evaluation action was selected")
    except (ProtocolError, ManifestError) as exc:
        raise SystemExit(f"OPD evaluation gate closed: {exc}") from exc


if __name__ == "__main__":
    main()
