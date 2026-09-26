"""Create and verify immutable provenance for merged RLVR SFT initializers.

The formal SFT-initialized RLVR cells train on a standalone model produced by
merging one frozen SFT-v3 LoRA into its matching Pro base.  A directory name is
not sufficient evidence for that operation.  This tool binds the exact base
revision, source adapter hash, merge recipe, merged file bundle, and the
20-user dynamic-LoRA versus merged-model parity report in one create-once JSON
record.

Run ``--write`` once after the merge and parity audit.  Formal launchers call
the same verifier before accepting the merged model.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd.provenance import (
    ManifestError as WeightManifestError,
    validate_weight_artifacts,
)
from rl.contract import POLICIES, policy_contract, rooted
from utils import config


SCHEMA_VERSION = 3
PROVENANCE_NAME = "initializer_provenance.json"
MERGE_SPECS = {
    "1.7b-pro": "code/rl/merge_sft_v3_1.7b_pro.yaml",
    "8b-pro": "code/rl/merge_sft_v3_8b_pro.yaml",
}


class InitializerError(RuntimeError):
    """Raised when a merged initializer cannot be tied to frozen SFT-v3."""


_CONFIG_SERIALIZATION_FIELDS = {
    "_name_or_path",
    "auto_map",
    "dtype",
    "pad_token_id",
    "rope_parameters",
    "rope_scaling",
    "torch_dtype",
    "transformers_version",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _load_json(path, label):
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise InitializerError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InitializerError(f"{label} must be a JSON object: {path}")
    return payload


def _verify_config(path, expected_sha256, label):
    if (
        not isinstance(expected_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
    ):
        raise InitializerError(f"{label} has no frozen config SHA-256")
    payload = _load_json(path, label)
    if not os.path.getsize(path):
        raise InitializerError(f"{label} is empty: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise InitializerError(
            f"{label} hash {actual} != {expected_sha256}"
        )
    return payload, actual


def _scientific_config(payload):
    """Normalize model-affecting fields across HF serialization versions."""

    identity = {
        key: value
        for key, value in payload.items()
        if key not in _CONFIG_SERIALIZATION_FIELDS
    }
    identity["dtype"] = payload.get("dtype", payload.get("torch_dtype"))
    identity["pad_token_id"] = payload.get("pad_token_id")
    theta = payload.get("rope_theta")
    identity["rope"] = {
        "rope_theta": theta,
        "rope_type": "default",
    }
    return identity


def _simple_yaml(path):
    """Read the scalar subset used by the two frozen merge specifications."""

    values = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InitializerError(f"cannot read merge specification {path}: {exc}") from exc
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise InitializerError(
                f"merge specification has a non-scalar line {number}: {raw!r}"
            )
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if not key or not value or key in values:
            raise InitializerError(
                f"invalid merge specification line {number}: {raw!r}"
            )
        values[key] = value.strip("\"'")
    return values


def _check_merge_spec(policy_name, root):
    policy = policy_contract(policy_name)
    rel = MERGE_SPECS[policy_name]
    path = rooted(root, rel)
    values = _simple_yaml(path)
    expected = {
        "model_name_or_path": policy["model"],
        "model_revision": policy["model_revision"],
        "adapter_name_or_path": policy["adapter_rel"],
        "template": "qwen3",
        "trust_remote_code": "true",
        "finetuning_type": "lora",
        "export_dir": policy["merged_rel"],
        "export_legacy_format": "false",
    }
    drift = {
        key: {"actual": values.get(key), "expected": value}
        for key, value in expected.items()
        if values.get(key) != value
    }
    if drift:
        raise InitializerError(
            f"{policy_name} merge specification drifted: "
            f"{json.dumps(drift, sort_keys=True)}"
        )
    return {
        "path": rel,
        "sha256": sha256_file(path),
    }


def _check_adapter(policy_name, root):
    policy = policy_contract(policy_name)
    path = rooted(
        root, os.path.join(policy["adapter_rel"], "adapter_model.safetensors")
    )
    if not os.path.isfile(path) or not os.path.getsize(path):
        raise InitializerError(f"missing frozen SFT adapter weights: {path}")
    actual = sha256_file(path)
    if actual != policy["adapter_sha256"]:
        raise InitializerError(
            f"{policy_name} SFT adapter hash {actual} != "
            f"{policy['adapter_sha256']}"
        )
    adapter_config = rooted(
        root, os.path.join(policy["adapter_rel"], "adapter_config.json")
    )
    _, adapter_config_sha256 = _verify_config(
        adapter_config,
        policy.get("adapter_config_sha256"),
        f"{policy_name} frozen SFT adapter config",
    )
    return {
        "artifact": policy["adapter_artifact"],
        "checkpoint": policy["adapter_rel"],
        "adapter_model_sha256": actual,
        "adapter_config_sha256": adapter_config_sha256,
    }


def check_base_snapshot(policy_name, root):
    """Verify the exact Pro base revision, config, and all weight bytes."""

    policy = policy_contract(policy_name)
    path = rooted(root, policy["base_rel"])
    if os.path.basename(os.path.normpath(path)) != policy["model_revision"]:
        raise InitializerError(
            f"{policy_name} base path does not end in the frozen revision"
        )
    config_path = os.path.join(path, "config.json")
    config_payload, config_sha256 = _verify_config(
        config_path,
        policy.get("config_sha256"),
        f"{policy_name} frozen Pro base config",
    )
    try:
        weight_artifacts = validate_weight_artifacts(
            path,
            policy,
            legacy_filename=policy.get(
                "weight_filename", "model-00000-of-00001.safetensors"
            ),
        )
    except WeightManifestError as exc:
        raise InitializerError(
            f"{policy_name} frozen Pro base weights failed verification: {exc}"
        ) from exc
    return {
        "model": policy["model"],
        "revision": policy["model_revision"],
        "snapshot": policy["base_rel"],
        "config_sha256": config_sha256,
        "scientific_config": _scientific_config(config_payload),
        "scientific_config_sha256": canonical_sha256(
            _scientific_config(config_payload)
        ),
        "weight_artifacts": weight_artifacts,
    }


def _merged_bundle(policy_name, root):
    policy = policy_contract(policy_name)
    directory = rooted(root, policy["merged_rel"])
    if not os.path.isdir(directory):
        raise InitializerError(f"missing merged initializer: {directory}")
    config_path = os.path.join(directory, "config.json")
    payload, config_sha256 = _verify_config(
        config_path,
        policy.get("merged_config_sha256"),
        f"{policy_name} merged model config",
    )
    base_config_path = rooted(
        root, os.path.join(policy["base_rel"], "config.json")
    )
    base_payload, base_config_sha256 = _verify_config(
        base_config_path,
        policy.get("config_sha256"),
        f"{policy_name} frozen Pro base config",
    )
    nested = payload.get("rope_parameters")
    if (
        payload.get("rope_theta") != policy["rope_theta"]
        or not isinstance(nested, dict)
        or nested.get("rope_theta") != policy["rope_theta"]
        or nested.get("rope_type") != "default"
    ):
        raise InitializerError(
            f"{policy_name} merged model has not passed fix_merged_config.py"
        )
    scientific_config = _scientific_config(payload)
    base_scientific_config = _scientific_config(base_payload)
    if scientific_config != base_scientific_config:
        raise InitializerError(
            f"{policy_name} merged model scientific config differs from "
            "the frozen Pro base"
        )

    files = {}
    for entry in sorted(Path(directory).rglob("*"), key=lambda item: str(item)):
        relative = entry.relative_to(directory).as_posix()
        if relative == PROVENANCE_NAME:
            continue
        if entry.is_symlink():
            raise InitializerError(
                f"merged initializer contains a symlink: {entry}"
            )
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise InitializerError(f"merged initializer contains a special file: {entry}")
        if not entry.stat().st_size:
            raise InitializerError(f"merged initializer contains an empty file: {entry}")
        files[relative] = {
            "bytes": entry.stat().st_size,
            "sha256": sha256_file(entry),
        }
    weights = [
        name
        for name in files
        if name.startswith("model") and name.endswith(".safetensors")
    ]
    if not weights:
        raise InitializerError(
            f"{policy_name} merged initializer has no safetensors weights"
        )
    return {
        "directory": policy["merged_rel"],
        "config_sha256": config_sha256,
        "source_base_config_sha256": base_config_sha256,
        "scientific_config": scientific_config,
        "scientific_config_sha256": canonical_sha256(scientific_config),
        "files": files,
        "bundle_sha256": canonical_sha256(files),
    }


def _meta_path_matches(value, expected):
    if not isinstance(value, str) or not value:
        return False
    normalized = os.path.normpath(value)
    suffix = os.path.normpath(expected)
    return normalized == suffix or normalized.endswith(os.sep + suffix)


def _parity_evidence(policy_name, root, report_path):
    policy = policy_contract(policy_name)
    report = _load_json(report_path, "adapter parity report")
    if report.get("labels") != {"a": "dynamic", "b": "merged"}:
        raise InitializerError(
            "parity labels must be exactly dynamic (A) and merged (B)"
        )
    if report.get("protocol_mismatches") != {}:
        raise InitializerError("parity report contains protocol mismatches")
    if report.get("n_users", 0) < 20:
        raise InitializerError("initializer parity requires at least 20 paired users")
    source_meta = report.get("source_meta", {})
    dynamic = source_meta.get("a", {})
    merged = source_meta.get("b", {})
    if not _meta_path_matches(dynamic.get("model"), policy["base_rel"]):
        raise InitializerError("dynamic parity route does not use the frozen Pro base")
    lora_dirs = dynamic.get("lora_dirs")
    if not isinstance(lora_dirs, str) or policy["adapter_rel"] not in lora_dirs:
        raise InitializerError("dynamic parity route does not use the frozen SFT adapter")
    if not _meta_path_matches(merged.get("model"), policy["merged_rel"]):
        raise InitializerError("merged parity route does not use the frozen export")
    if merged.get("lora_dirs") not in (None, ""):
        raise InitializerError("merged parity route must not apply another adapter")

    try:
        ndcg = report["ranking"]["ndcg@5"]
        valid = report["interface"]["valid_output"]
        identity = report["interface"]["identity_collapse"]
        loop = report["interface"]["stage1_loop"]
        fallback = report["interface"]["stage2_fallback"]
        degradations = {
            "ndcg@5": float(ndcg["mean_a"]) - float(ndcg["mean_b"]),
            "valid_output": float(valid["mean_a"]) - float(valid["mean_b"]),
            "identity_collapse": float(identity["mean_b"]) - float(identity["mean_a"]),
            "stage1_loop": float(loop["mean_b"]) - float(loop["mean_a"]),
            "stage2_fallback": float(fallback["mean_b"]) - float(fallback["mean_a"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise InitializerError("parity report is missing required metrics") from exc
    limits = {
        "ndcg@5": 0.10,
        "valid_output": 0.05,
        "identity_collapse": 0.05,
        "stage1_loop": 0.05,
        "stage2_fallback": 0.05,
    }
    failed = {
        metric: {"degradation": value, "limit": limits[metric]}
        for metric, value in degradations.items()
        if value > limits[metric] + 1e-12
    }
    if failed:
        raise InitializerError(
            "merged initializer failed the catastrophic parity guard: "
            f"{json.dumps(failed, sort_keys=True)}"
        )
    return {
        "report_sha256": sha256_file(report_path),
        "n_users": report["n_users"],
        "routes": {
            "dynamic_base": policy["base_rel"],
            "dynamic_adapter": policy["adapter_rel"],
            "merged_model": policy["merged_rel"],
        },
        "limits": limits,
        "observed_degradation": degradations,
    }


def _build_source_provenance(policy_name, root):
    if policy_name not in POLICIES:
        raise InitializerError(f"unknown RLVR policy: {policy_name}")
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": policy_name,
        "base": check_base_snapshot(policy_name, root),
        "sft_adapter": _check_adapter(policy_name, root),
        "merge_spec": _check_merge_spec(policy_name, root),
        "merged": _merged_bundle(policy_name, root),
    }


def build_provenance(policy_name, root=config.ROOT, parity_report=None):
    if not parity_report:
        raise InitializerError("a dynamic-versus-merged parity report is required")
    return {
        **_build_source_provenance(policy_name, root),
        "parity": _parity_evidence(policy_name, root, parity_report),
    }


def _validate_stored_parity(policy_name, parity):
    if not isinstance(parity, dict):
        raise InitializerError("initializer provenance has no parity evidence")
    report_sha = parity.get("report_sha256")
    if (
        not isinstance(report_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", report_sha)
    ):
        raise InitializerError("initializer parity report SHA-256 is invalid")
    if parity.get("n_users", 0) < 20:
        raise InitializerError("initializer parity evidence has fewer than 20 users")
    expected_limits = {
        "ndcg@5": 0.10,
        "valid_output": 0.05,
        "identity_collapse": 0.05,
        "stage1_loop": 0.05,
        "stage2_fallback": 0.05,
    }
    if parity.get("limits") != expected_limits:
        raise InitializerError("initializer parity acceptance limits drifted")
    observed = parity.get("observed_degradation")
    if not isinstance(observed, dict) or set(observed) != set(expected_limits):
        raise InitializerError("initializer parity evidence is incomplete")
    for metric, limit in expected_limits.items():
        value = observed[metric]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value > limit + 1e-12
        ):
            raise InitializerError(
                f"initializer parity evidence fails {metric}: {value!r}"
            )
    policy = policy_contract(policy_name)
    routes = parity.get("routes", {})
    if routes != {
        "dynamic_base": policy["base_rel"],
        "dynamic_adapter": policy["adapter_rel"],
        "merged_model": policy["merged_rel"],
    }:
        raise InitializerError("initializer parity route evidence drifted")
    return parity


def _first_difference(left, right, path="$"):
    if type(left) is not type(right):
        return f"{path}: {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                return f"{path}.{key}: missing from existing provenance"
            if key not in right:
                return f"{path}.{key}: missing from current evidence"
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: list lengths differ"
        for index, values in enumerate(zip(left, right)):
            difference = _first_difference(*values, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def _atomic_create(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=".initializer_provenance.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_provenance(policy_name, root=config.ROOT, parity_report=None):
    path = rooted(
        root, os.path.join(policy_contract(policy_name)["merged_rel"], PROVENANCE_NAME)
    )
    if os.path.exists(path):
        raise InitializerError(
            f"refusing to overwrite initializer provenance: {path}"
        )
    payload = build_provenance(policy_name, root, parity_report)
    try:
        _atomic_create(path, payload)
    except FileExistsError as exc:
        raise InitializerError(
            f"refusing to overwrite initializer provenance: {path}"
        ) from exc
    return payload


def verify_provenance(policy_name, root=config.ROOT):
    policy = policy_contract(policy_name)
    path = rooted(root, os.path.join(policy["merged_rel"], PROVENANCE_NAME))
    existing = _load_json(path, "initializer provenance")
    parity = _validate_stored_parity(policy_name, existing.get("parity"))
    prospective = {
        **_build_source_provenance(policy_name, root),
        "parity": parity,
    }
    difference = _first_difference(existing, prospective)
    if difference:
        raise InitializerError(
            f"initializer provenance no longer matches its inputs at {difference}"
        )
    return existing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=sorted(POLICIES), required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--parity-report",
        help="adapter_parity_audit.py output; required with --write",
    )
    args = parser.parse_args()
    try:
        if args.write:
            payload = write_provenance(
                args.policy, parity_report=args.parity_report
            )
            status = "created"
        else:
            if args.parity_report:
                raise InitializerError("--parity-report is only valid with --write")
            payload = verify_provenance(args.policy)
            status = "verified"
    except InitializerError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(
        json.dumps(
            {
                "status": status,
                "policy": args.policy,
                "provenance_sha256": canonical_sha256(payload),
                "merged_bundle_sha256": payload["merged"]["bundle_sha256"],
                "source_adapter_sha256": payload["sft_adapter"][
                    "adapter_model_sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
