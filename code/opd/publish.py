"""Validate and optionally publish one sealed formal OPD release to W&B.

Only the exact training manifest, dev-selected adapter, eight formal dev
files, twelve accepted test cells, and immutable evaluation state are
included. Smoke, probe, raw checkpoints, and rollout dumps are never uploaded.
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd import contract, eval_protocol
from opd.export_lora_checkpoint import ExportError, verify_compact_adapter
from opd.provenance import ManifestError, validate_policy_model_configs
from utils import config
from utils.artifact_release import (
    git_snapshot,
    log_immutable_artifact,
    require_clean_pushed_release,
    require_exact_regular_files,
    require_regular_file,
)

config.setup_env()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_files(directory):
    if not os.path.isdir(directory) or os.path.islink(directory):
        raise ValueError(f"missing formal OPD directory: {directory}")
    entries = sorted(os.listdir(directory))
    invalid = [
        name
        for name in entries
        if not name.endswith(".json")
        or os.path.islink(os.path.join(directory, name))
        or not os.path.isfile(os.path.join(directory, name))
    ]
    if invalid:
        raise ValueError(
            f"formal OPD directory contains unexpected entries: {invalid}"
        )
    return [os.path.join(directory, name) for name in entries]


def _validate_selected_adapter(
    selection, selected_adapter, step, adapter_files, root, policy
):
    """Re-bind the frozen adapter to its immutable training manifest."""

    training = selection.get("training")
    if not isinstance(training, dict):
        raise ValueError("OPD selection has no training identity")
    run_manifest = training.get("run_manifest")
    if not isinstance(run_manifest, str) or not run_manifest:
        raise ValueError("OPD selection has no run manifest path")
    require_regular_file(run_manifest, "OPD run manifest")
    run_manifest_sha256 = sha256(run_manifest)
    if run_manifest_sha256 != training.get("run_manifest_sha256"):
        raise ValueError("OPD run manifest differs from the dev selection")

    verified = verify_compact_adapter(selected_adapter, step, run_manifest)
    provenance = verified["provenance"]
    expected_training = {
        "training_commit": provenance["training_commit"],
        "run_manifest_sha256": provenance["run_manifest_sha256"],
        "manifest_payload_sha256": provenance["manifest_payload_sha256"],
    }
    for key, value in expected_training.items():
        if training.get(key) != value:
            raise ValueError(
                f"OPD selection training identity {key} differs from "
                "the verified selected adapter"
            )
    expected_adapter = {
        "adapter_model_sha256": verified["adapter_model_sha256"],
        "adapter_config_sha256": verified["adapter_config_sha256"],
        "provenance_sha256": verified["provenance_sha256"],
    }
    selected = selection["selected"]
    for key, value in expected_adapter.items():
        if selected.get(key) != value:
            raise ValueError(
                f"OPD selected adapter {key} differs from its verified "
                "training provenance"
            )
    for name, key in (
        ("adapter_model.safetensors", "adapter_model_sha256"),
        ("adapter_config.json", "adapter_config_sha256"),
        ("provenance.json", "provenance_sha256"),
    ):
        if sha256(adapter_files[name]) != expected_adapter[key]:
            raise ValueError(
                f"OPD selected adapter {name} changed after verification"
            )
    if sha256(run_manifest) != expected_training["run_manifest_sha256"]:
        raise ValueError("OPD run manifest changed after adapter verification")
    model_configs = validate_policy_model_configs(root, policy)

    source_checkpoint = {
        "source_actor": provenance["source_actor"],
        "source_model_shard_sha256": provenance["source_model_shard_sha256"],
        "checkpoint_metadata_sha256": provenance[
            "checkpoint_metadata_sha256"
        ],
        "checkpoint_shard_bytes": provenance["checkpoint_shard_bytes"],
    }
    return {
        "run_manifest": run_manifest,
        **expected_training,
        **expected_adapter,
        "model_configs": model_configs,
        "source_checkpoint": source_checkpoint,
    }


def validate_release(policy, root=config.ROOT):
    spec = contract.policy_spec(policy)
    result_root = os.path.join(root, spec["storage"]["results_rel"])
    state_dir = os.path.join(result_root, "eval_state")
    dev_dir = os.path.join(result_root, "formal_dev")
    test_dir = os.path.join(result_root, "test_once")
    selection_path = os.path.join(state_dir, eval_protocol.SELECTION_NAME)
    state_files = [
        os.path.join(state_dir, eval_protocol.TEST_STARTED_NAME),
    ]
    if os.path.islink(state_dir) or not os.path.isdir(state_dir):
        raise ValueError("OPD evaluation state is not a regular directory")
    require_regular_file(selection_path, "OPD selection")
    for path in state_files:
        require_regular_file(path, "OPD test-once state")
    with open(selection_path, encoding="utf-8") as handle:
        selection = json.load(handle)
    if selection.get("schema_version") == eval_protocol.LEGACY_SCHEMA_VERSION:
        state_files.append(
            os.path.join(
                state_dir, eval_protocol.LEGACY_TEST_VALIDATION_NAME
            )
        )
        require_regular_file(
            state_files[-1], "OPD legacy completion ledger"
        )
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("OPD selection has no selected adapter")
    step = selected.get("step")
    if step not in eval_protocol.EXPECTED_STEPS:
        raise ValueError(f"OPD selected step is invalid: {step!r}")

    selected_adapter = os.path.join(
        state_dir, eval_protocol.FROZEN_ADAPTER_NAME
    )
    adapter_files = require_exact_regular_files(
        selected_adapter,
        (
            "adapter_model.safetensors",
            "adapter_config.json",
            "provenance.json",
        ),
        "OPD frozen selected adapter",
    )
    training_identity = _validate_selected_adapter(
        selection,
        selected_adapter,
        step,
        adapter_files,
        root,
        policy,
    )

    dev_files = _json_files(dev_dir)
    test_files = _json_files(test_dir)
    if len(dev_files) != len(eval_protocol.EXPECTED_STEPS):
        raise ValueError(
            f"expected eight formal dev files, found {len(dev_files)}"
        )
    if len(test_files) != len(eval_protocol.TEST_CELLS):
        raise ValueError(
            f"expected twelve formal test files, found {len(test_files)}"
        )
    validation = eval_protocol.validate_complete_test(
        state_dir=state_dir,
        test_dir=test_dir,
        policy=policy,
        verify_current_stack=False,
    )
    receipt_dir = os.path.join(
        state_dir, eval_protocol.TEST_RECEIPT_DIR_NAME
    )
    receipt_files = _json_files(receipt_dir) if os.path.isdir(receipt_dir) else []
    return {
        "policy": policy,
        "recipe": contract.RECIPE,
        "data_recipe": contract.DATA_RECIPE,
        "selected_step": step,
        "dev_ndcg@5": selected.get("ndcg@5"),
        "adapter_model_sha256": training_identity["adapter_model_sha256"],
        "adapter_config_sha256": training_identity["adapter_config_sha256"],
        "adapter_provenance_sha256": training_identity["provenance_sha256"],
        "training_commit": training_identity["training_commit"],
        "run_manifest_sha256": training_identity["run_manifest_sha256"],
        "manifest_payload_sha256": training_identity[
            "manifest_payload_sha256"
        ],
        "model_configs": training_identity["model_configs"],
        "source_checkpoint": training_identity["source_checkpoint"],
        "selection_sha256": sha256(selection_path),
        "benchmark_sha256": contract.BENCHMARK_SHA256,
        "dev_files": dev_files,
        "test_files": test_files,
        "state_dir": state_dir,
        "selection_path": selection_path,
        "state_files": state_files,
        "receipt_files": receipt_files,
        "adapter_files": list(adapter_files.values()),
        "selected_adapter": selected_adapter,
        "run_manifest": training_identity["run_manifest"],
        "test_validation": validation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", choices=contract.SUPPORTED_POLICIES, required=True
    )
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    try:
        release = validate_release(args.policy)
    except (
        OSError,
        ValueError,
        eval_protocol.ProtocolError,
        ManifestError,
        ExportError,
    ) as error:
        raise SystemExit(str(error)) from error
    git = git_snapshot(config.ROOT)
    commit = git["commit"]
    dirty = git["dirty"]
    metadata = {
        key: value
        for key, value in release.items()
        if key
        not in {
            "dev_files",
            "test_files",
            "state_dir",
            "selection_path",
            "state_files",
            "receipt_files",
            "adapter_files",
            "selected_adapter",
            "run_manifest",
            "test_validation",
        }
    }
    metadata.update({"git_commit": commit, "git_dirty": dirty})
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if not args.upload:
        print("dry-run only; pass --upload after reviewing the sealed release")
        return
    if dirty:
        raise SystemExit("formal OPD publication requires a clean worktree")
    require_clean_pushed_release(config.ROOT, expected_commit=commit)

    import wandb

    artifact_name = f"opd-v3-{args.policy}-release"
    run = wandb.init(
        project=config.WANDB_PROJECT,
        job_type="publish-formal-opd",
        name=f"publish_{artifact_name}_step{release['selected_step']}",
    )
    artifact = wandb.Artifact(
        artifact_name,
        type="model-evaluation",
        metadata=metadata,
    )
    for path in release["adapter_files"]:
        artifact.add_file(
            path, name=f"selected_adapter/{os.path.basename(path)}"
        )
    artifact.add_file(
        release["run_manifest"], name="release/run_manifest.json"
    )
    for path in release["dev_files"]:
        artifact.add_file(path, name=f"formal_dev/{os.path.basename(path)}")
    for path in release["test_files"]:
        artifact.add_file(path, name=f"test_once/{os.path.basename(path)}")
    artifact.add_file(
        release["selection_path"], name="eval_state/selection.json"
    )
    for path in release["state_files"]:
        artifact.add_file(
            path, name=f"eval_state/{os.path.basename(path)}"
        )
    for path in release["receipt_files"]:
        artifact.add_file(
            path,
            name=f"eval_state/test_receipts/{os.path.basename(path)}",
        )
    try:
        identity = log_immutable_artifact(run, artifact)
    finally:
        run.finish()
    print(json.dumps({"uploaded_artifact": identity}, indent=2))


if __name__ == "__main__":
    main()
