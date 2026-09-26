"""Fail-closed verification for the frozen Pro base models used by RFT.

The public Hugging Face repository name is useful metadata, but it is not an
immutable runtime input.  Formal RFT rollout therefore resolves only the local
snapshot named by the frozen commit and verifies every model-weight byte before
vLLM is constructed.  The returned identity is machine-independent and is
embedded in every raw rollout record.
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd.provenance import (
    ManifestError as WeightManifestError,
    validate_weight_artifacts,
    weight_bundle_sha256,
)
from rft.contract import POLICIES, policy_contract
from utils import config


class BaseIdentityError(RuntimeError):
    """Raised when a base model cannot be tied to the frozen RFT contract."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_revision(value):
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def contract_weight_artifacts(policy):
    """Return the exact weight-artifact record implied by one policy contract."""

    bundle = policy.get("weight_sha256")
    if not _valid_sha256(bundle):
        raise BaseIdentityError("base weight SHA-256 is invalid")

    files = policy.get("weight_files_sha256")
    if files is None:
        filename = policy.get("weight_filename")
        if (
            not isinstance(filename, str)
            or not filename
            or os.path.basename(filename) != filename
            or not filename.startswith("model")
            or not filename.endswith(".safetensors")
        ):
            raise BaseIdentityError("single-file base weight name is invalid")
        return {
            "format": "single",
            "files": {filename: bundle},
        }

    if not isinstance(files, dict) or not files:
        raise BaseIdentityError("sharded base has no frozen shard digest map")
    if any(
        not isinstance(name, str)
        or not name
        or os.path.basename(name) != name
        or not _valid_sha256(digest)
        for name, digest in files.items()
    ):
        raise BaseIdentityError("sharded base has an invalid shard contract")
    index_sha = policy.get("weight_index_sha256")
    total_size = policy.get("weight_total_size")
    if not _valid_sha256(index_sha):
        raise BaseIdentityError("sharded base index SHA-256 is invalid")
    if (
        not isinstance(total_size, int)
        or isinstance(total_size, bool)
        or total_size <= 0
    ):
        raise BaseIdentityError("sharded base total tensor size is invalid")
    if weight_bundle_sha256(index_sha, files) != bundle:
        raise BaseIdentityError("sharded base bundle SHA-256 is inconsistent")
    return {
        "format": "sharded",
        "index_sha256": index_sha,
        "files": dict(files),
        "total_size": total_size,
        "bundle_sha256": bundle,
    }


def expected_base_identity(policy):
    """Build the portable identity expected in one rollout provenance record."""

    if not _valid_revision(policy.get("model_revision")):
        raise BaseIdentityError("base revision must be a full lowercase commit SHA")
    config_sha256 = policy.get("config_sha256")
    if not _valid_sha256(config_sha256):
        raise BaseIdentityError("base config SHA-256 is invalid")
    model = policy.get("model")
    snapshot = policy.get("base_rel")
    if not isinstance(model, str) or not model:
        raise BaseIdentityError("base model ID is missing")
    if (
        not isinstance(snapshot, str)
        or not snapshot
        or os.path.isabs(snapshot)
        or os.path.normpath(snapshot).startswith(".." + os.sep)
    ):
        raise BaseIdentityError("base snapshot path must be repository-relative")
    if os.path.basename(os.path.normpath(snapshot)) != policy["model_revision"]:
        raise BaseIdentityError("base snapshot path does not end in its frozen revision")
    return {
        "model_id": model,
        "revision": policy["model_revision"],
        "snapshot_rel": snapshot,
        "config_sha256": config_sha256,
        "weight_sha256": policy["weight_sha256"],
        "weight_artifacts": contract_weight_artifacts(policy),
    }


def validate_recorded_base_identity(recorded, policy):
    """Reject a rollout identity that omits or drifts from the frozen contract."""

    if not isinstance(recorded, dict):
        raise BaseIdentityError("rollout provenance has no base-model identity")
    expected_keys = {
        "model_id",
        "revision",
        "snapshot_rel",
        "config_sha256",
        "weight_sha256",
        "weight_artifacts",
    }
    if set(recorded) != expected_keys:
        raise BaseIdentityError(
            "rollout base-model identity fields differ from the frozen schema"
        )
    expected = expected_base_identity(policy)
    if recorded != expected:
        raise BaseIdentityError(
            "rollout base-model identity differs from the frozen contract"
        )
    return recorded


def verify_base_snapshot(policy, root):
    """Verify the exact snapshot path, config, and complete model-weight set."""

    # Validate the static contract before consulting the filesystem.
    expected_artifacts = contract_weight_artifacts(policy)
    revision = policy.get("model_revision")
    snapshot = policy.get("base_rel")
    if not _valid_revision(revision):
        raise BaseIdentityError("base revision must be a full lowercase commit SHA")
    if (
        not isinstance(snapshot, str)
        or not snapshot
        or os.path.isabs(snapshot)
        or os.path.normpath(snapshot).startswith(".." + os.sep)
    ):
        raise BaseIdentityError("base snapshot path must be repository-relative")

    root_real = os.path.realpath(root)
    base_path = os.path.realpath(os.path.join(root_real, snapshot))
    if os.path.basename(os.path.normpath(base_path)) != revision:
        raise BaseIdentityError("resolved base snapshot is not the frozen revision")
    if not os.path.isdir(base_path):
        raise BaseIdentityError(f"missing frozen base snapshot: {base_path}")

    config_path = os.path.join(base_path, "config.json")
    if not os.path.isfile(config_path) or not os.path.getsize(config_path):
        raise BaseIdentityError(f"missing frozen base config: {config_path}")
    actual_config_sha256 = sha256_file(config_path)
    if actual_config_sha256 != policy.get("config_sha256"):
        raise BaseIdentityError(
            "frozen base config failed verification: "
            f"{actual_config_sha256} != {policy.get('config_sha256')}"
        )
    try:
        actual_artifacts = validate_weight_artifacts(
            base_path,
            policy,
            legacy_filename=policy.get(
                "weight_filename", "model-00000-of-00001.safetensors"
            ),
        )
    except WeightManifestError as exc:
        raise BaseIdentityError(
            f"frozen base weights failed verification: {exc}"
        ) from exc

    # The shared validator already enforces this for sharded checkpoints.  Do
    # it here as well for the legacy single-file 1.7B checkpoint.
    on_disk = {
        name
        for name in os.listdir(base_path)
        if name.startswith("model") and name.endswith(".safetensors")
    }
    if on_disk != set(expected_artifacts["files"]):
        raise BaseIdentityError(
            "base weight files differ from the frozen set: "
            f"expected={sorted(expected_artifacts['files'])}, "
            f"found={sorted(on_disk)}"
        )
    if actual_artifacts != expected_artifacts:
        raise BaseIdentityError("verified base artifacts differ from the contract")

    identity = expected_base_identity(policy)
    return base_path, identity


def verify_policy_base(policy_name, root=None):
    return verify_base_snapshot(
        policy_contract(policy_name),
        root or config.ROOT,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=sorted(POLICIES), required=True)
    args = parser.parse_args()
    try:
        path, identity = verify_policy_base(args.policy)
    except BaseIdentityError as exc:
        raise SystemExit(f"base verification failed: {exc}") from exc
    print(json.dumps(
        {"resolved_path": path, "identity": identity},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
