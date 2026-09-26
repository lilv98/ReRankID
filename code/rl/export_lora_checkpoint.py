"""Convert one verl FSDP checkpoint into a compact PEFT LoRA adapter.

verl's merger temporarily materializes a complete Hugging Face model even for
LoRA checkpoints.  This wrapper retains only ``lora_adapter`` and deletes only
the private temporary directory it created, avoiding five duplicate 8B bases.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
from utils.artifact_release import require_exact_regular_files

from rl.contract import POLICIES, REWARD_FILES, SAVE_FREQ, rlvr_contract, rooted
from rl import run_contract


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=sorted(POLICIES), required=True)
    parser.add_argument("--init", choices=("sft",), default="sft")
    parser.add_argument("--reward", choices=sorted(REWARD_FILES), default="ndcg5")
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    if args.step not in range(SAVE_FREQ, 101, SAVE_FREQ):
        raise SystemExit("--step must be one of 20,40,60,80,100")

    try:
        policy = rlvr_contract(args.policy, args.init, args.reward)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        contract_info = run_contract.load_verified_run_contract(
            args.policy,
            args.init,
            args.reward,
            root=config.ROOT,
        )
        checkpoint = run_contract.validate_export_checkpoint(
            contract_info["run_dir"], args.step
        )
    except run_contract.RunContractError as error:
        raise SystemExit(str(error)) from error
    actor_dir = checkpoint["actor_dir"]
    export_root = rooted(config.ROOT, policy["export_rel"])
    final_dir = os.path.join(export_root, f"step-{args.step:06d}")
    if os.path.exists(final_dir):
        raise SystemExit(f"refusing to overwrite existing export: {final_dir}")
    os.makedirs(export_root, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix=f".merge_step_{args.step}_", dir=export_root)
    try:
        subprocess.run(
            [
                sys.executable, "-m", "verl.model_merger", "merge",
                "--backend", "fsdp", "--local_dir", actor_dir,
                "--target_dir", temp_dir, "--trust-remote-code",
            ],
            check=True,
        )
        adapter_dir = os.path.join(temp_dir, "lora_adapter")
        weights = os.path.join(adapter_dir, "adapter_model.safetensors")
        adapter_config = os.path.join(adapter_dir, "adapter_config.json")
        if not os.path.isfile(weights) or not os.path.isfile(adapter_config):
            raise RuntimeError("verl merger did not produce a PEFT lora_adapter")
        provenance = {
            "schema_version": 3,
            "policy": args.policy,
            "variant": policy["variant"],
            "initializer": policy["initializer"],
            "reward": policy["reward"],
            "step": args.step,
            "source_actor": os.path.relpath(actor_dir, config.ROOT),
            "source_actor_tree_sha256": checkpoint["tree_sha256"],
            "source_actor_files": checkpoint["files"],
            "source_actor_bytes": checkpoint["bytes"],
            "run_contract_sha256": contract_info["sha256"],
            "run_contract_canonical_sha256": contract_info[
                "canonical_sha256"
            ],
            "training_commit": contract_info["payload"]["repository"][
                "initial_commit"
            ],
            "initializer_model": policy["model_rel"],
            "model_revision": policy["model_revision"],
            "adapter_model_sha256": sha256(weights),
            "adapter_config_sha256": sha256(adapter_config),
        }
        with open(os.path.join(adapter_dir, "provenance.json"), "w") as handle:
            json.dump(provenance, handle, indent=2)
        require_exact_regular_files(
            adapter_dir,
            (
                "adapter_model.safetensors",
                "adapter_config.json",
                "provenance.json",
            ),
            "exported RLVR compact adapter",
        )
        os.replace(adapter_dir, final_dir)
        print(json.dumps(provenance, indent=2))
        print(f"wrote compact adapter: {final_dir}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
