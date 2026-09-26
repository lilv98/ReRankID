"""Validate and publish one exact RLVR arm as an independent W&B artifact."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
from utils.artifact_release import (git_snapshot, log_immutable_artifact,
                                    require_clean_pushed_release)

config.setup_env()

from rl.contract import (BENCH_SHA256, POLICIES, REWARD_FILES, rlvr_contract)
from rl.select_checkpoint import (DEFAULT_MANIFEST, EXPECTED_STEPS,
                                  selection, validate_result_tree)
from rl.publish_selected import validate_selected_adapter

DEV_STEPS = EXPECTED_STEPS


def require_canonical_upload_scope(root, manifest_path):
    """Prevent an uploaded tree from being attributed to an unrelated checkout."""

    if os.path.realpath(root) != os.path.realpath(config.ROOT):
        raise ValueError(
            "formal upload requires the canonical repository root"
        )
    if os.path.realpath(manifest_path) != os.path.realpath(DEFAULT_MANIFEST):
        raise ValueError(
            "formal upload requires experiments/final_manifest.json "
            "from the canonical repository"
        )


def validate_rlvr_tree(
    policy_name=None,
    root=config.ROOT,
    initializer="sft",
    reward="ndcg5",
    profile="canonical",
    arm=None,
    manifest_path=DEFAULT_MANIFEST,
):
    """Backward-compatible entrypoint returning the 18 strictly validated files."""

    report = validate_result_tree(
        policy_name,
        root=root,
        initializer=initializer,
        reward=reward,
        profile=profile,
        arm=arm,
        manifest_path=manifest_path,
        require_test=True,
    )
    return report["files"]


def _parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("canonical", "executed"), default="canonical")
    parser.add_argument("--arm")
    parser.add_argument("--policy", choices=sorted(POLICIES))
    parser.add_argument("--init", choices=("sft",))
    parser.add_argument("--reward", choices=sorted(REWARD_FILES))
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--root", default=config.ROOT)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if args.profile == "canonical":
        if not args.policy:
            parser.error("--profile canonical requires one --policy")
        if args.arm:
            parser.error("--arm is valid only with --profile executed")
        args.init = args.init or "sft"
        args.reward = args.reward or "ndcg5"
    else:
        if not args.arm:
            parser.error("--profile executed requires explicit --arm")
        if args.policy or args.init or args.reward:
            parser.error("executed profile accepts only --arm, not policy/init/reward")
        args.init = "sft"
        args.reward = "ndcg5"
    return args


def main():
    args = _parse_cli()
    try:
        if args.upload:
            require_canonical_upload_scope(args.root, args.manifest)
        tree = validate_result_tree(
            args.policy,
            root=args.root,
            initializer=args.init,
            reward=args.reward,
            profile=args.profile,
            arm=args.arm,
            manifest_path=args.manifest,
            require_test=True,
        )
        chosen = selection(
            args.policy,
            root=args.root,
            initializer=args.init,
            reward=args.reward,
            profile=args.profile,
            arm=args.arm,
            manifest_path=args.manifest,
            require_adapter=False,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    spec = tree["spec"]
    contract = rlvr_contract(spec["policy"], spec["initializer"], spec["reward"])
    adapter = None
    if args.profile == "canonical":
        try:
            adapter = validate_selected_adapter(
                spec["policy"],
                spec["initializer"],
                spec["reward"],
                chosen["best_step"],
                root=args.root,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
    snapshot = git_snapshot(config.ROOT)
    commit = snapshot["commit"]
    dirty = snapshot["dirty"]
    report = {
        "recipe": contract["recipe"],
        "profile": args.profile,
        "arm": spec["arm"],
        "policy": spec["policy"],
        "variant": spec["variant"],
        "initializer": spec["initializer"],
        "reward": spec["reward"],
        "git_commit": commit,
        "git_dirty": dirty,
        "benchmark_sha256": BENCH_SHA256,
        "result_artifact_name": spec["result_artifact_name"],
        "result_root": os.path.relpath(tree["result_root"], args.root),
        "result_files": [
            os.path.relpath(path, tree["result_root"]) for path in tree["files"]
        ],
        "selected_step": chosen["best_step"],
        "selected_dev_ndcg@5": chosen["best_ndcg@5"],
        "test_anchor": os.path.relpath(tree["anchor_path"], tree["result_root"]),
        "evidence_route": (
            "contract-bound-adapter"
            if adapter is not None
            else "manifest-pinned-executed-tree"
        ),
    }
    if adapter is not None:
        report.update(
            {
                "adapter_model_sha256": adapter[
                    "adapter_model_sha256"
                ],
                "adapter_config_sha256": adapter[
                    "adapter_config_sha256"
                ],
                "run_contract_sha256": adapter["contract_info"]["sha256"],
                "run_contract_canonical_sha256": adapter[
                    "contract_info"
                ]["canonical_sha256"],
                "training_commit": adapter["contract_info"]["payload"][
                    "repository"
                ]["initial_commit"],
                "source_actor_tree_sha256": adapter["checkpoint"][
                    "tree_sha256"
                ],
            }
        )
    if args.profile == "executed":
        report["executed_tree_sha256"] = spec["tree_sha256"]
    print(json.dumps(report, indent=2))
    if not args.upload:
        print("dry-run only; pass --upload after checking this exact arm")
        return
    release = require_clean_pushed_release(
        config.ROOT, expected_commit=commit
    )

    import wandb

    run = wandb.init(
        project=config.WANDB_PROJECT,
        job_type="publish-formal-rlvr-results",
        name=f"publish_{spec['arm']}_results",
    )
    artifact = wandb.Artifact(
        spec["result_artifact_name"],
        type="results",
        metadata={
            "recipe": contract["recipe"],
            "profile": args.profile,
            "arm": spec["arm"],
            "policy": spec["policy"],
            "variant": spec["variant"],
            "initializer": spec["initializer"],
            "reward": spec["reward"],
            "git_commit": release["commit"],
            "benchmark_sha256": BENCH_SHA256,
            "selected_step": chosen["best_step"],
            "selected_dev_ndcg@5": chosen["best_ndcg@5"],
            "result_files": len(tree["files"]),
            "evidence_route": report["evidence_route"],
            "adapter_model_sha256": report.get("adapter_model_sha256"),
            "adapter_config_sha256": report.get("adapter_config_sha256"),
            "run_contract_sha256": report.get("run_contract_sha256"),
            "run_contract_canonical_sha256": report.get(
                "run_contract_canonical_sha256"
            ),
            "training_commit": report.get("training_commit"),
            "source_actor_tree_sha256": report.get(
                "source_actor_tree_sha256"
            ),
            "executed_tree_sha256": (
                spec["tree_sha256"] if args.profile == "executed" else None
            ),
        },
    )
    for path in tree["files"]:
        artifact.add_file(
            path,
            name=os.path.relpath(path, tree["result_root"]),
        )
    if adapter is not None:
        artifact.add_file(
            adapter["provenance_path"],
            name="release/selected_adapter_provenance.json",
        )
        artifact.add_file(
            adapter["contract_info"]["path"],
            name="release/run_contract.json",
        )
    try:
        identity = log_immutable_artifact(run, artifact)
    finally:
        run.finish()
    print(json.dumps(identity, indent=2))
    print("record this immutable version and digest in experiments/final_manifest.json")


if __name__ == "__main__":
    main()
