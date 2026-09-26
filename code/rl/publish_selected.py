"""Publish one dev-selected formal RLVR LoRA adapter to W&B."""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
from utils.artifact_release import (git_snapshot, log_immutable_artifact,
                                    require_clean_pushed_release,
                                    require_exact_regular_files)

config.setup_env()

from rl.contract import (BENCH_SHA256, POLICIES, POSTTRAIN_MANIFEST_SHA256,
                         POSTTRAIN_POOL_SHA256, REWARD_FILES, SAVE_FREQ,
                         rlvr_contract, rooted)
from rl import run_contract
from rl.select_checkpoint import selection


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_selected_adapter(
    policy_name,
    initializer,
    reward,
    step,
    *,
    root=config.ROOT,
):
    """Bind one compact adapter to its exact completed training contract."""
    try:
        policy = rlvr_contract(policy_name, initializer, reward)
    except ValueError as error:
        raise ValueError(str(error)) from error
    adapter_dir = os.path.join(
        rooted(root, policy["export_rel"]), f"step-{step:06d}"
    )
    adapter_files = require_exact_regular_files(
        adapter_dir,
        (
            "adapter_model.safetensors",
            "adapter_config.json",
            "provenance.json",
        ),
        "RLVR compact adapter",
    )
    required = [
        adapter_files["adapter_model.safetensors"],
        adapter_files["adapter_config.json"],
        adapter_files["provenance.json"],
    ]
    try:
        with open(required[2], encoding="utf-8") as handle:
            provenance = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("compact adapter provenance is unreadable") from error
    if (
        provenance.get("policy") != policy_name
        or provenance.get("variant") != policy["variant"]
        or provenance.get("step") != step
    ):
        raise ValueError(
            "compact adapter provenance differs from requested policy/step"
        )
    if provenance.get("model_revision") != policy["model_revision"]:
        raise ValueError(
            "compact adapter provenance has the wrong model revision"
        )
    if provenance.get("initializer_model") != policy["model_rel"]:
        raise ValueError(
            "compact adapter provenance has the wrong initializer model"
        )
    try:
        contract_info = run_contract.load_verified_run_contract(
            policy_name,
            initializer,
            reward,
            root=root,
        )
        checkpoint = run_contract.validate_export_checkpoint(
            contract_info["run_dir"], step
        )
    except run_contract.RunContractError as error:
        raise ValueError(str(error)) from error
    expected_provenance = {
        "schema_version": 3,
        "run_contract_sha256": contract_info["sha256"],
        "run_contract_canonical_sha256": contract_info["canonical_sha256"],
        "training_commit": contract_info["payload"]["repository"][
            "initial_commit"
        ],
        "source_actor_tree_sha256": checkpoint["tree_sha256"],
        "source_actor_files": checkpoint["files"],
        "source_actor_bytes": checkpoint["bytes"],
    }
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            raise ValueError(
                f"compact adapter provenance has wrong {key}: "
                f"{provenance.get(key)!r} != {value!r}"
            )
    actual_adapter_sha = sha256(required[0])
    actual_config_sha = sha256(required[1])
    if provenance.get("adapter_model_sha256") != actual_adapter_sha:
        raise ValueError("compact adapter weights differ from provenance")
    if provenance.get("adapter_config_sha256") != actual_config_sha:
        raise ValueError("compact adapter config differs from provenance")
    return {
        "policy_contract": policy,
        "adapter_dir": adapter_dir,
        "files": required,
        "provenance": provenance,
        "provenance_path": required[2],
        "adapter_model_sha256": actual_adapter_sha,
        "adapter_config_sha256": actual_config_sha,
        "contract_info": contract_info,
        "checkpoint": checkpoint,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=sorted(POLICIES), required=True)
    parser.add_argument("--init", choices=("sft",), default="sft")
    parser.add_argument("--reward", choices=sorted(REWARD_FILES), default="ndcg5")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if args.step not in range(SAVE_FREQ, 101, SAVE_FREQ):
        raise SystemExit("--step must be one of 20,40,60,80,100")
    try:
        adapter = validate_selected_adapter(
            args.policy,
            args.init,
            args.reward,
            args.step,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    policy = adapter["policy_contract"]
    required = adapter["files"]
    actual_adapter_sha = adapter["adapter_model_sha256"]
    actual_config_sha = adapter["adapter_config_sha256"]
    contract_info = adapter["contract_info"]
    checkpoint = adapter["checkpoint"]
    try:
        selected = selection(
            args.policy, initializer=args.init, reward=args.reward
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.step != selected["best_step"]:
        raise SystemExit(
            f"step {args.step} is not the official dev-selected step "
            f"{selected['best_step']}"
        )
    git = git_snapshot(config.ROOT)
    commit = git["commit"]
    dirty = git["dirty"]
    metadata = {
        "recipe": policy["recipe"],
        "policy": args.policy,
        "variant": policy["variant"],
        "initializer": args.init,
        "reward": args.reward,
        "step": args.step,
        "model": policy["model"],
        "model_revision": policy["model_revision"],
        "initializer_model": policy["model_rel"],
        "sft_adapter_sha256": (
            policy["adapter_sha256"] if args.init == "sft" else None
        ),
        "rl_adapter_sha256": actual_adapter_sha,
        "rl_adapter_config_sha256": actual_config_sha,
        "run_contract_sha256": contract_info["sha256"],
        "run_contract_canonical_sha256": contract_info[
            "canonical_sha256"
        ],
        "training_commit": contract_info["payload"]["repository"][
            "initial_commit"
        ],
        "source_actor_tree_sha256": checkpoint["tree_sha256"],
        "dev_ndcg@5": selected["best_ndcg@5"],
        "dev_delta_vs_init": selected["delta_vs_init"],
        "posttrain_pool_sha256": POSTTRAIN_POOL_SHA256,
        "posttrain_manifest_sha256": POSTTRAIN_MANIFEST_SHA256,
        "benchmark_sha256": BENCH_SHA256,
        "git_commit": commit,
    }
    print(json.dumps(metadata, indent=2))
    if not args.upload:
        print("dry-run only; pass --upload after verifying the dev-selected step")
        return
    if dirty:
        raise SystemExit("formal selected-adapter publication requires a clean Git worktree")
    require_clean_pushed_release(config.ROOT, expected_commit=commit)
    import wandb

    artifact_name = f"rlvr-{policy['variant']}-{args.policy}-adapter"
    run = wandb.init(
        project=config.WANDB_PROJECT,
        job_type="publish-rlvr-selected-adapter",
        name=f"publish_{artifact_name}_step{args.step}",
    )
    artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata)
    for path in required:
        artifact.add_file(path, name=os.path.basename(path))
    artifact.add_file(
        contract_info["path"], name="training/run_contract.json"
    )
    try:
        identity = log_immutable_artifact(run, artifact)
    finally:
        run.finish()
    print(json.dumps({"uploaded_artifact": identity}, indent=2))


if __name__ == "__main__":
    main()
