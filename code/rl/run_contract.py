"""Create-once RLVR run contracts and fail-closed resume validation."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl import contract
from rl.prepare_initializers import (
    InitializerError,
    canonical_sha256,
    sha256_file,
    verify_provenance,
)
from utils import config
from utils.artifact_release import require_regular_file


SCHEMA_VERSION = 2
RUN_CONTRACT_NAME = "run_contract.json"
TOTAL_STEPS = contract.TRAIN_ROWS // contract.TRAIN_BATCH_SIZE


class RunContractError(RuntimeError):
    """Raised when a fresh run or resume is not attributable to one contract."""


def git_commit(root):
    process = subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    value = process.stdout.strip()
    if process.returncode or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise RunContractError("cannot resolve a full initial Git commit")
    return value


def _verify_file(root, rel, expected=None):
    path = os.path.join(root, rel)
    if not os.path.isfile(path) or not os.path.getsize(path):
        raise RunContractError(f"missing run-contract input: {path}")
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise RunContractError(f"{rel} hash {actual} != {expected}")
    return actual


def _initializer_evidence(policy_name, initializer, root):
    if initializer != "sft":
        raise RunContractError("RLVR accepts only the frozen SFT-v3 initializer")
    policy = contract.policy_contract(policy_name)
    try:
        provenance = verify_provenance(policy_name, root)
    except InitializerError as exc:
        raise RunContractError(str(exc)) from exc
    base = provenance["base"]
    evidence = {
        "model": policy["model"],
        "model_path": policy["merged_rel"],
        "base_snapshot": policy["base_rel"],
        "base_revision": policy["model_revision"],
        "base_config_sha256": base["config_sha256"],
        "base_scientific_config_sha256": base[
            "scientific_config_sha256"
        ],
        "base_weight_artifacts": base["weight_artifacts"],
        "sft_adapter_artifact": policy["adapter_artifact"],
        "sft_adapter_path": policy["adapter_rel"],
        "sft_adapter_sha256": policy["adapter_sha256"],
        "sft_adapter_config_sha256": provenance["sft_adapter"][
            "adapter_config_sha256"
        ],
        "initializer_provenance_sha256": canonical_sha256(provenance),
        "merged_config_sha256": provenance["merged"]["config_sha256"],
        "merged_scientific_config_sha256": provenance["merged"][
            "scientific_config_sha256"
        ],
        "merged_bundle_sha256": provenance["merged"]["bundle_sha256"],
    }
    return evidence


def build_run_contract(
    policy_name,
    initializer,
    reward,
    *,
    root=config.ROOT,
    mode="formal",
    gpu_count=4,
    machine_profile=None,
    vllm_memory_utilization=0.5,
    ppo_max_tokens_per_gpu=16384,
    initial_commit=None,
):
    try:
        resolved = contract.rlvr_contract(policy_name, initializer, reward)
    except ValueError as exc:
        raise RunContractError(str(exc)) from exc
    if mode not in {"smoke", "formal"}:
        raise RunContractError(f"unsupported RLVR run mode: {mode}")
    if not isinstance(gpu_count, int) or isinstance(gpu_count, bool) or gpu_count < 1:
        raise RunContractError("visible GPU count must be a positive integer")
    if TOTAL_STEPS * contract.TRAIN_BATCH_SIZE != contract.TRAIN_ROWS:
        raise RunContractError("frozen RLVR row count is not divisible by batch size")

    reward_sha = _verify_file(root, resolved["reward_rel"])
    data = {
        "train": {
            "path": "data/rlvr_v1_6k/train.parquet",
            "sha256": _verify_file(
                root,
                "data/rlvr_v1_6k/train.parquet",
                contract.TRAIN_PARQUET_SHA256,
            ),
        },
        "val": {
            "path": "data/rlvr_v1_6k/val.parquet",
            "sha256": _verify_file(
                root,
                "data/rlvr_v1_6k/val.parquet",
                contract.VAL_PARQUET_SHA256,
            ),
        },
        "stats": {
            "path": "data/rlvr_v1_6k/stats.json",
            "sha256": _verify_file(
                root,
                "data/rlvr_v1_6k/stats.json",
                contract.RLVR_STATS_SHA256,
            ),
        },
    }
    commit = initial_commit or git_commit(root)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RunContractError("initial Git commit must be a full lowercase SHA")
    return {
        "schema_version": SCHEMA_VERSION,
        "recipe": resolved["recipe"],
        "policy": policy_name,
        "variant": resolved["variant"],
        "initializer": initializer,
        "reward": {
            "name": reward,
            "path": resolved["reward_rel"],
            "sha256": reward_sha,
        },
        "data": data,
        "model": _initializer_evidence(policy_name, initializer, root),
        "optimization": {
            "advantage_estimator": "grpo",
            "normalize_grpo_advantage_by_std": True,
            "use_kl_in_reward": False,
            "train_rows": contract.TRAIN_ROWS,
            "val_rows": contract.VAL_ROWS,
            "train_batch_size": contract.TRAIN_BATCH_SIZE,
            "ppo_mini_batch_size": contract.PPO_MINI_BATCH_SIZE,
            "total_epochs": contract.TOTAL_EPOCHS,
            "total_steps": TOTAL_STEPS,
            "rollouts_per_prompt": contract.N_SAMPLES,
            "temperature": contract.TEMPERATURE,
            "top_p": contract.TOP_P,
            "learning_rate": contract.LEARNING_RATE,
            "warmup_ratio": contract.WARMUP_RATIO,
            "kl_coefficient": contract.KL_COEF,
            "kl_loss_type": "low_var_kl",
            "use_kl_loss": True,
            "entropy_coefficient": 0,
            "ppo_epochs": contract.PPO_EPOCHS,
            "lora_rank": contract.LORA_RANK,
            "lora_alpha": contract.LORA_ALPHA,
            "lora_target_modules": "all-linear",
            "max_prompt_length": contract.MAX_PROMPT_LENGTH,
            "max_response_length": contract.MAX_RESPONSE_LENGTH,
            "max_model_length": contract.MAX_MODEL_LENGTH,
            "data_seed": contract.DATA_SEED,
            "data_shuffle": True,
            "validation_shuffle": False,
            "rollout_engine": "vllm",
            "rollout_load_format": "safetensors",
            "rollout_layered_summon": True,
            "rollout_tensor_parallel_size": 1,
            "save_frequency": contract.SAVE_FREQ,
            "test_frequency": contract.TEST_FREQ,
        },
        "execution_overrides": (
            {
                "train_max_samples": 120,
                "val_max_samples": 20,
                "total_training_steps": 2,
                "save_frequency": -1,
                "test_frequency": 1,
                "logger": ["console"],
                "resume_mode": "disable",
            }
            if mode == "smoke"
            else {"resume_mode": "auto-or-disable-from-explicit-cli"}
        ),
        "runtime": {
            "mode": mode,
            "machine_profile": machine_profile,
            "visible_gpus": gpu_count,
            "vllm_memory_utilization": float(vllm_memory_utilization),
            "ppo_max_tokens_per_gpu": int(ppo_max_tokens_per_gpu),
        },
        "repository": {"initial_commit": commit},
    }


def _first_difference(left, right, path="$"):
    if type(left) is not type(right):
        return f"{path}: {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                return f"{path}.{key}: missing from existing contract"
            if key not in right:
                return f"{path}.{key}: missing from prospective contract"
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
    directory = os.path.dirname(path)
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=".run_contract.", suffix=".tmp", dir=directory
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


def _checkpoint_entries(outdir):
    return sorted(
        name
        for name in os.listdir(outdir)
        if name.startswith("global_step_")
        or name == "latest_checkpointed_iteration.txt"
    )


def load_verified_run_contract(
    policy_name,
    initializer,
    reward,
    *,
    root=config.ROOT,
):
    """Rebuild and byte-compare the scientific contract for one formal run."""
    try:
        resolved = contract.rlvr_contract(policy_name, initializer, reward)
    except ValueError as exc:
        raise RunContractError(str(exc)) from exc
    run_dir = contract.rooted(root, resolved["run_rel"])
    path = os.path.join(run_dir, RUN_CONTRACT_NAME)
    try:
        require_regular_file(path, "RLVR run contract")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RunContractError(f"cannot load exact run contract: {path}") from exc
    runtime = payload.get("runtime")
    repository = payload.get("repository")
    if not isinstance(runtime, dict) or not isinstance(repository, dict):
        raise RunContractError("run contract lacks runtime or repository identity")
    if runtime.get("mode") != "formal":
        raise RunContractError("only a formal RLVR run may export an adapter")
    try:
        expected = build_run_contract(
            policy_name,
            initializer,
            reward,
            root=root,
            mode="formal",
            gpu_count=runtime["visible_gpus"],
            machine_profile=runtime["machine_profile"],
            vllm_memory_utilization=runtime["vllm_memory_utilization"],
            ppo_max_tokens_per_gpu=runtime["ppo_max_tokens_per_gpu"],
            initial_commit=repository["initial_commit"],
        )
    except (KeyError, TypeError) as exc:
        raise RunContractError(
            "run contract has incomplete runtime or repository identity"
        ) from exc
    difference = _first_difference(payload, expected)
    if difference:
        raise RunContractError(
            f"stored run contract differs from the frozen recipe at {difference}"
        )
    return {
        "path": path,
        "run_dir": run_dir,
        "payload": payload,
        "sha256": sha256_file(path),
        "canonical_sha256": canonical_sha256(payload),
    }


def _tree_identity(directory):
    if os.path.islink(directory) or not os.path.isdir(directory):
        raise RunContractError(
            f"actor checkpoint is not a regular directory: {directory}"
        )
    records = []
    total_size = 0
    for current, dirs, names in os.walk(directory):
        dirs.sort()
        for name in dirs:
            path = os.path.join(current, name)
            if os.path.islink(path):
                raise RunContractError(
                    f"actor checkpoint contains a directory symlink: {path}"
                )
        for name in sorted(names):
            path = os.path.join(current, name)
            try:
                require_regular_file(path, "actor checkpoint file")
            except ValueError as exc:
                raise RunContractError(str(exc)) from exc
            rel = os.path.relpath(path, directory)
            size = os.path.getsize(path)
            digest = sha256_file(path)
            records.append((rel, size, digest))
            total_size += size
    if not records:
        raise RunContractError(f"actor checkpoint is empty: {directory}")
    digest = hashlib.sha256()
    for rel, size, file_sha in records:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
    return {
        "tree_sha256": digest.hexdigest(),
        "files": len(records),
        "bytes": total_size,
    }


def validate_export_checkpoint(run_dir, step):
    """Bind one scheduled actor tree to a completed formal run."""
    if step not in range(contract.SAVE_FREQ, TOTAL_STEPS + 1, contract.SAVE_FREQ):
        raise RunContractError(f"invalid export step: {step}")
    marker = os.path.join(run_dir, "latest_checkpointed_iteration.txt")
    try:
        require_regular_file(marker, "RLVR completion marker")
        with open(marker, encoding="utf-8") as handle:
            marker_step = int(handle.read().strip())
    except (OSError, ValueError) as exc:
        raise RunContractError("cannot read RLVR completion marker") from exc
    if marker_step != TOTAL_STEPS:
        raise RunContractError(
            f"adapter export requires a completed step-{TOTAL_STEPS} run; "
            f"marker={marker_step}"
        )
    actor_dir = os.path.join(run_dir, f"global_step_{step}", "actor")
    return {
        "step": step,
        "marker_step": marker_step,
        "actor_dir": actor_dir,
        **_tree_identity(actor_dir),
    }


def validate_resume_state(outdir):
    marker = os.path.join(outdir, "latest_checkpointed_iteration.txt")
    try:
        require_regular_file(marker, "RLVR resume marker")
        with open(marker, encoding="utf-8") as handle:
            step = int(handle.read().strip())
    except (OSError, ValueError) as exc:
        raise RunContractError(
            "resume requires latest_checkpointed_iteration.txt"
        ) from exc
    if step == TOTAL_STEPS:
        raise RunContractError(f"RLVR run is already complete at step {step}")
    if step <= 0 or step >= TOTAL_STEPS or step % contract.SAVE_FREQ:
        raise RunContractError(
            f"invalid resume step {step}; expected a saved step below {TOTAL_STEPS}"
        )
    seen = []
    for name in os.listdir(outdir):
        if not name.startswith("global_step_"):
            continue
        raw = name.removeprefix("global_step_")
        if not raw.isdigit():
            raise RunContractError(f"invalid checkpoint directory: {name}")
        checkpoint = os.path.join(outdir, name)
        if os.path.islink(checkpoint) or not os.path.isdir(checkpoint):
            raise RunContractError(
                f"checkpoint is not a regular directory: {name}"
            )
        candidate = int(raw)
        if (
            candidate <= 0
            or candidate > TOTAL_STEPS
            or candidate % contract.SAVE_FREQ
        ):
            raise RunContractError(f"off-schedule checkpoint directory: {name}")
        seen.append(candidate)
    if any(candidate > step for candidate in seen):
        raise RunContractError(
            f"checkpoint directory is newer than committed marker {step}: {seen}"
        )
    actor = os.path.join(outdir, f"global_step_{step}", "actor")
    try:
        _tree_identity(actor)
    except RunContractError as exc:
        raise RunContractError(
            f"resume checkpoint global_step_{step}/actor is invalid: {exc}"
        ) from exc
    return step


def prepare_output(outdir, payload, *, resume=False):
    outdir = os.path.abspath(outdir)
    path = os.path.join(outdir, RUN_CONTRACT_NAME)
    if resume:
        if os.path.islink(outdir) or not os.path.isdir(outdir):
            raise RunContractError(f"resume output directory does not exist: {outdir}")
        try:
            require_regular_file(path, "RLVR run contract")
            with open(path, encoding="utf-8") as handle:
                existing = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RunContractError(
                "refusing to adopt output without a readable run_contract.json"
            ) from exc
        difference = _first_difference(existing, payload)
        if difference:
            raise RunContractError(
                f"resume run contract differs at {difference}; "
                f"existing={canonical_sha256(existing)} "
                f"prospective={canonical_sha256(payload)}"
            )
        step = validate_resume_state(outdir)
        return {"status": "resume", "path": path, "resume_step": step}

    if os.path.exists(outdir):
        if os.path.islink(outdir) or not os.path.isdir(outdir):
            raise RunContractError(f"output path is not a directory: {outdir}")
        entries = sorted(os.listdir(outdir))
        if entries:
            raise RunContractError(
                "fresh RLVR refuses a non-empty output directory; "
                f"preserve it and choose a new contract path: {entries[:8]}"
            )
    else:
        os.makedirs(outdir)
    try:
        _atomic_create(path, payload)
    except FileExistsError as exc:
        raise RunContractError(
            f"refusing to overwrite existing run contract: {path}"
        ) from exc
    return {"status": "created", "path": path, "resume_step": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=sorted(contract.POLICIES), required=True)
    parser.add_argument("--init", choices=("sft",), default="sft")
    parser.add_argument("--reward", choices=sorted(contract.REWARD_FILES), default="ndcg5")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--machine-profile")
    parser.add_argument("--vllm-memory-utilization", type=float, required=True)
    parser.add_argument("--ppo-max-tokens-per-gpu", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "smoke" and args.resume:
        raise SystemExit("ERROR: smoke runs cannot resume")
    try:
        payload = build_run_contract(
            args.policy,
            args.init,
            args.reward,
            mode=args.mode,
            gpu_count=args.gpus,
            machine_profile=args.machine_profile,
            vllm_memory_utilization=args.vllm_memory_utilization,
            ppo_max_tokens_per_gpu=args.ppo_max_tokens_per_gpu,
        )
        report = prepare_output(args.outdir, payload, resume=args.resume)
    except RunContractError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(
        json.dumps(
            {
                **report,
                "contract_sha256": canonical_sha256(payload),
                "policy": args.policy,
                "variant": payload["variant"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
