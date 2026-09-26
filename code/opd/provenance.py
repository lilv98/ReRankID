"""Immutable run provenance for the frozen OPD recipe."""

import hashlib
import json
import os
import tempfile

from opd import contract
from opd.verl_topk_patch import (
    EXPECTED_STOCK_KL_SHA256,
    PATCHED_KL_SHA256,
    PATCH_NAME,
)


MANIFEST_SCHEMA_VERSION = 2
LEGACY_MANIFEST_SCHEMA_VERSION = 1


class ManifestError(RuntimeError):
    """Raised when an OPD run cannot be attributed to one frozen recipe."""


def canonical_json(payload):
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def payload_sha256(payload):
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_bundle_sha256(index_sha256, files_sha256):
    """Hash one sharded checkpoint's index and complete shard digest map."""

    return payload_sha256(
        {
            "index_sha256": index_sha256,
            "files": files_sha256,
        }
    )


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_model_config(model_path, expected_sha256, *, role):
    """Verify one complete ``config.json`` against its frozen byte identity."""

    if role not in {"student", "teacher"}:
        raise ManifestError(f"invalid OPD model role: {role!r}")
    if not _valid_sha256(expected_sha256):
        raise ManifestError(f"{role} model config SHA-256 is invalid")
    path = os.path.join(model_path, "config.json")
    if not os.path.isfile(path) or not os.path.getsize(path):
        raise ManifestError(f"missing {role} model config: {path}")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise ManifestError(
            f"{role} model config {path} hash {actual} != {expected_sha256}"
        )
    return {
        "path": "config.json",
        "sha256": actual,
    }


def validate_policy_model_configs(root, policy=None):
    """Verify both on-disk configs for one frozen matched-scale policy."""

    policy = contract.normalize_policy(policy)
    spec = contract.policy_spec(policy)
    expected = contract.model_config_identity(policy)
    paths = {
        "student": os.path.join(root, spec["student"]["base_rel"]),
        "teacher": os.path.join(root, spec["teacher"]["merged_rel"]),
    }
    return {
        role: validate_model_config(
            paths[role],
            expected[role],
            role=role,
        )
        for role in ("student", "teacher")
    }


def validate_weight_artifacts(model_path, model, *, legacy_filename):
    """Verify all model weights, failing closed for sharded safetensors.

    Legacy 1.7B contracts intentionally retain their original single-file
    ``weight_sha256`` field.  New sharded contracts bind the index, the exact
    set of referenced shard names, every shard digest, and the logical tensor
    byte count reported by the index.
    """

    expected_files = model.get("weight_files_sha256")
    if expected_files is None:
        expected = model.get("weight_sha256")
        if not _valid_sha256(expected):
            raise ManifestError("legacy model weight SHA-256 is invalid")
        path = os.path.join(model_path, legacy_filename)
        if not os.path.isfile(path) or not os.path.getsize(path):
            raise ManifestError(f"missing model weight file: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ManifestError(f"{path} hash {actual} != {expected}")
        return {
            "format": "single",
            "files": {legacy_filename: actual},
        }

    if not isinstance(expected_files, dict) or not expected_files:
        raise ManifestError("sharded model has no frozen shard digest map")
    for name, digest in expected_files.items():
        if (
            not isinstance(name, str)
            or not name
            or os.path.basename(name) != name
            or not name.startswith("model-")
            or not name.endswith(".safetensors")
        ):
            raise ManifestError(f"invalid frozen safetensors shard name: {name!r}")
        if not _valid_sha256(digest):
            raise ManifestError(f"invalid frozen shard SHA-256 for {name}")

    expected_index = model.get("weight_index_sha256")
    expected_bundle = model.get("weight_sha256")
    expected_total_size = model.get("weight_total_size")
    if not _valid_sha256(expected_index):
        raise ManifestError("sharded model index SHA-256 is invalid")
    if not _valid_sha256(expected_bundle):
        raise ManifestError("sharded model bundle SHA-256 is invalid")
    if (
        not isinstance(expected_total_size, int)
        or isinstance(expected_total_size, bool)
        or expected_total_size <= 0
    ):
        raise ManifestError("sharded model total tensor size is invalid")
    if weight_bundle_sha256(expected_index, expected_files) != expected_bundle:
        raise ManifestError("sharded model contract has an inconsistent bundle hash")

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path) or not os.path.getsize(index_path):
        raise ManifestError(f"missing sharded model index: {index_path}")
    actual_index = _sha256_file(index_path)
    if actual_index != expected_index:
        raise ManifestError(
            f"{index_path} hash {actual_index} != {expected_index}"
        )
    try:
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read sharded model index: {index_path}") from exc
    if not isinstance(index, dict) or set(index) != {"metadata", "weight_map"}:
        raise ManifestError("sharded model index has an unexpected top-level schema")
    metadata = index["metadata"]
    weight_map = index["weight_map"]
    if not isinstance(metadata, dict) or metadata.get(
        "total_size"
    ) != expected_total_size:
        raise ManifestError("sharded model index total_size drifted")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ManifestError("sharded model index has no weight map")
    if any(not isinstance(name, str) or not name for name in weight_map):
        raise ManifestError("sharded model index has an invalid tensor name")
    referenced = list(weight_map.values())
    if any(
        not isinstance(name, str)
        or os.path.basename(name) != name
        or name not in expected_files
        for name in referenced
    ):
        raise ManifestError("sharded model index references an unfrozen shard")
    if set(referenced) != set(expected_files):
        raise ManifestError("sharded model index does not reference every frozen shard")

    on_disk = {
        name
        for name in os.listdir(model_path)
        if name.startswith("model") and name.endswith(".safetensors")
    }
    if on_disk != set(expected_files):
        raise ManifestError(
            "sharded model files differ from the frozen set: "
            f"expected={sorted(expected_files)}, found={sorted(on_disk)}"
        )
    actual_files = {}
    for name, expected in expected_files.items():
        path = os.path.join(model_path, name)
        if not os.path.isfile(path) or not os.path.getsize(path):
            raise ManifestError(f"missing model weight shard: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ManifestError(f"{path} hash {actual} != {expected}")
        actual_files[name] = actual
    if weight_bundle_sha256(actual_index, actual_files) != expected_bundle:
        raise ManifestError("actual sharded model bundle hash drifted")
    return {
        "format": "sharded",
        "index_sha256": actual_index,
        "files": actual_files,
        "total_size": expected_total_size,
        "bundle_sha256": expected_bundle,
    }


def formal_wandb_run_id(training_commit, policy=None):
    """Return the stable W&B identity for one formal scientific run."""

    if (
        not isinstance(training_commit, str)
        or len(training_commit) != 40
        or any(character not in "0123456789abcdef" for character in training_commit)
    ):
        raise ManifestError(
            "formal OPD W&B identity requires a full lowercase Git commit"
        )
    spec = contract.policy_spec(policy)
    student = spec["student"]
    teacher = spec["teacher"]
    identity = {
        "recipe": contract.RECIPE,
        "student": {
            "name": student["name"],
            "revision": student["revision"],
            "weight_sha256": student["weight_sha256"],
        },
        "teacher": {
            "name": teacher["name"],
            "weight_sha256": teacher["weight_sha256"],
        },
        "training_commit": training_commit,
    }
    return f"opdv3_{payload_sha256(identity)[:20]}"


def frozen_contract(policy=None):
    """Return every scientific choice that must remain fixed across resume."""

    spec = contract.policy_spec(policy)
    return {
        "recipe": contract.RECIPE,
        "data_recipe": contract.DATA_RECIPE,
        "student": spec["student"],
        "teacher": spec["teacher"],
        "data": {
            "train_rows": contract.TRAIN_ROWS,
            "val_rows": contract.VAL_ROWS,
            "train_parquet_sha256": contract.TRAIN_PARQUET_SHA256,
            "val_parquet_sha256": contract.VAL_PARQUET_SHA256,
            "stats_sha256": contract.STATS_SHA256,
            "seed": contract.DATA_SEED,
        },
        "optimization": {
            "train_batch_size": contract.TRAIN_BATCH_SIZE,
            "ppo_mini_batch_size": contract.PPO_MINI_BATCH_SIZE,
            "total_training_steps": contract.TOTAL_TRAINING_STEPS,
            "n_samples": contract.N_SAMPLES,
            "agent_loop_workers": contract.AGENT_LOOP_WORKERS,
            "temperature": contract.TEMPERATURE,
            "top_p": contract.TOP_P,
            "top_k_sampling": contract.TOP_K_SAMPLING,
            "learning_rate": contract.LEARNING_RATE,
            "warmup_ratio": contract.WARMUP_RATIO,
            "ppo_epochs": contract.PPO_EPOCHS,
            "loss_agg_mode": contract.LOSS_AGG_MODE,
            "lora_rank": contract.LORA_RANK,
            "lora_alpha": contract.LORA_ALPHA,
            "max_prompt_length": contract.MAX_PROMPT_LENGTH,
            "max_response_length": contract.MAX_RESPONSE_LENGTH,
            "max_model_length": contract.MAX_MODEL_LENGTH,
            "save_freq": contract.SAVE_FREQ,
            "test_freq": contract.TEST_FREQ,
        },
        "objective": {
            "loss_mode": contract.DISTILLATION_LOSS,
            "topk": contract.DISTILLATION_TOPK,
            "use_policy_gradient": (
                contract.DISTILLATION_USE_POLICY_GRADIENT
            ),
            "use_task_rewards": contract.DISTILLATION_USE_TASK_REWARDS,
            "loss_max_clamp": contract.DISTILLATION_LOSS_MAX_CLAMP,
            "log_prob_min_clamp": (
                contract.DISTILLATION_LOG_PROB_MIN_CLAMP
            ),
            "reference_kl": False,
        },
    }


def make_run_manifest(
    *,
    audit,
    run_mode,
    training_overrides,
    runtime,
    policy=None,
):
    if run_mode not in {"smoke", "probe", "formal"}:
        raise ManifestError(f"unknown OPD run mode: {run_mode!r}")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "recipe": contract.RECIPE,
        "data_recipe": contract.DATA_RECIPE,
        "run_mode": run_mode,
        "frozen_contract": frozen_contract(policy),
        "training_overrides": list(training_overrides),
        "runtime": dict(runtime),
        "audit": audit,
    }


def training_override_map(payload):
    overrides = payload.get("training_overrides")
    if not isinstance(overrides, list) or not overrides:
        raise ManifestError("OPD run manifest has no training overrides")
    result = {}
    for raw in overrides:
        if not isinstance(raw, str) or "=" not in raw:
            raise ManifestError(f"invalid OPD training override: {raw!r}")
        key, value = raw.split("=", 1)
        key = key.lstrip("+")
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        if key in result:
            raise ManifestError(f"duplicate OPD training override: {key}")
        result[key] = value
    return result


def _validate_training_overrides(payload, policy):
    values = training_override_map(payload)
    spec = contract.policy_spec(policy)
    runtime = payload.get("runtime", {})
    expected = {
        "algorithm.use_kl_in_reward": "False",
        "data.filter_overlong_prompts": "False",
        "data.max_prompt_length": str(contract.MAX_PROMPT_LENGTH),
        "data.max_response_length": str(contract.MAX_RESPONSE_LENGTH),
        "data.seed": str(contract.DATA_SEED),
        "data.shuffle": "True",
        "data.truncation": "error",
        "actor_rollout_ref.model.lora_rank": str(contract.LORA_RANK),
        "actor_rollout_ref.model.lora_alpha": str(contract.LORA_ALPHA),
        "actor_rollout_ref.model.target_modules": "all-linear",
        "actor_rollout_ref.actor.optim.lr": "5e-5",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": str(
            contract.WARMUP_RATIO
        ),
        "actor_rollout_ref.actor.ppo_epochs": str(contract.PPO_EPOCHS),
        "actor_rollout_ref.actor.loss_agg_mode": contract.LOSS_AGG_MODE,
        "actor_rollout_ref.actor.use_kl_loss": "False",
        "actor_rollout_ref.actor.entropy_coeff": "0",
        "actor_rollout_ref.rollout.temperature": str(contract.TEMPERATURE),
        "actor_rollout_ref.rollout.top_p": str(contract.TOP_P),
        "actor_rollout_ref.rollout.top_k": str(contract.TOP_K_SAMPLING),
        "actor_rollout_ref.rollout.n": str(contract.N_SAMPLES),
        "actor_rollout_ref.rollout.max_model_len": str(
            contract.MAX_MODEL_LENGTH
        ),
        "actor_rollout_ref.rollout.agent.num_workers": str(
            contract.AGENT_LOOP_WORKERS
        ),
        "model_engine": "dp",
        "distillation.enabled": "True",
        "distillation.nnodes": "1",
        "distillation.n_gpus_per_node": "1",
        "distillation.teacher_models.teacher_model.key": (
            "rerankid_opd_sid"
        ),
        "distillation.teacher_models.teacher_model.num_replicas": "1",
        "distillation.teacher_models.teacher_model.inference.temperature": (
            str(contract.TEMPERATURE)
        ),
        "distillation.distillation_loss.loss_mode": (
            contract.DISTILLATION_LOSS
        ),
        "distillation.distillation_loss.topk": str(
            contract.DISTILLATION_TOPK
        ),
        "distillation.distillation_loss.use_task_rewards": "False",
        "distillation.distillation_loss.use_policy_gradient": "False",
        "distillation.distillation_loss.loss_max_clamp": "null",
        "distillation.distillation_loss.log_prob_min_clamp": "null",
        "trainer.n_gpus_per_node": "1",
        "trainer.nnodes": "1",
        "trainer.test_freq": str(contract.TEST_FREQ),
        "ray_kwargs.ray_init.runtime_env.env_vars."
        "RERANKID_OPD_NORMALIZE_TOPK": "1",
        "ray_kwargs.ray_init.runtime_env.worker_process_setup_hook": (
            "opd.verl_topk_patch.ray_worker_setup"
        ),
    }
    if policy != contract.DEFAULT_POLICY:
        expected.update(
            {
                "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "4096",
                "actor_rollout_ref.actor.fsdp_config.param_offload": "True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload": "False",
                "actor_rollout_ref.rollout.gpu_memory_utilization": "0.30",
                "actor_rollout_ref.rollout.enforce_eager": "True",
                "distillation.teacher_models.teacher_model.inference."
                "gpu_memory_utilization": "0.60",
                "ray_kwargs.ray_init.runtime_env.env_vars."
                "RERANKID_OPD_CPU_LAYERED_LORA": "1",
            }
        )
    run_mode = payload.get("run_mode")
    if run_mode == "formal":
        expected.update(
            {
                "data.train_batch_size": str(contract.TRAIN_BATCH_SIZE),
                "actor_rollout_ref.actor.ppo_mini_batch_size": str(
                    contract.PPO_MINI_BATCH_SIZE
                ),
                "trainer.total_training_steps": str(
                    contract.TOTAL_TRAINING_STEPS
                ),
                "trainer.save_freq": str(contract.SAVE_FREQ),
                "trainer.resume_mode": "auto",
            }
        )
        if policy != contract.DEFAULT_POLICY:
            from opd.checkpoint_archive import (
                ARCHIVE_ENV,
                EXPORT_DIR_ENV,
                MERGER_TMP_ENV,
                RAW_KEEP_ENV,
            )

            archive_runtime = {
                ARCHIVE_ENV: "1",
                MERGER_TMP_ENV: spec["storage"]["merge_work_physical"],
                RAW_KEEP_ENV: "2",
                contract.POLICY_ENV: policy,
            }
            for key, expected_value in archive_runtime.items():
                if runtime.get(key) != expected_value:
                    raise ManifestError(
                        f"8B formal OPD runtime {key}={runtime.get(key)!r}, "
                        f"expected {expected_value!r}"
                    )
            export_dir = runtime.get(EXPORT_DIR_ENV)
            if (
                not isinstance(export_dir, str)
                or not os.path.isabs(export_dir)
                or not os.path.normpath(export_dir).endswith(
                    os.path.normpath(spec["storage"]["compact_rel"])
                )
            ):
                raise ManifestError(
                    "8B formal OPD compact export directory is not the "
                    "frozen absolute policy path"
                )
            expected.update(
                {
                    "trainer.max_actor_ckpt_to_keep": "2",
                    "trainer.del_local_ckpt_after_load": "False",
                    "actor_rollout_ref.actor.checkpoint.async_save": "False",
                    "ray_kwargs.ray_init.runtime_env.env_vars."
                    f"{ARCHIVE_ENV}": "1",
                    "ray_kwargs.ray_init.runtime_env.env_vars."
                    f"{EXPORT_DIR_ENV}": export_dir,
                    "ray_kwargs.ray_init.runtime_env.env_vars."
                    f"{MERGER_TMP_ENV}": spec["storage"][
                        "merge_work_physical"
                    ],
                    "ray_kwargs.ray_init.runtime_env.env_vars."
                    f"{RAW_KEEP_ENV}": "2",
                    "ray_kwargs.ray_init.runtime_env.env_vars."
                    f"{contract.POLICY_ENV}": policy,
                }
            )
    elif run_mode == "probe":
        expected.update(
            {
                "data.train_batch_size": str(contract.TRAIN_BATCH_SIZE),
                "data.train_max_samples": str(contract.PROBE_ROWS),
                "data.val_max_samples": "20",
                "actor_rollout_ref.actor.ppo_mini_batch_size": str(
                    contract.PPO_MINI_BATCH_SIZE
                ),
                "actor_rollout_ref.actor.optim.lr_warmup_steps": str(
                    contract.PROBE_WARMUP_STEPS
                ),
                "trainer.total_training_steps": str(contract.PROBE_STEPS),
                "trainer.save_freq": str(contract.PROBE_SAVE_FREQ),
                "trainer.resume_mode": "disable",
            }
        )
    elif run_mode == "smoke":
        expected.update(
            {
                "data.train_batch_size": "50",
                "data.train_max_samples": "100",
                "data.val_max_samples": "20",
                "actor_rollout_ref.actor.ppo_mini_batch_size": "50",
                "trainer.total_training_steps": "2",
                "trainer.save_freq": (
                    "2"
                    if policy != contract.DEFAULT_POLICY
                    else "-1"
                ),
                "trainer.resume_mode": "disable",
            }
        )
    for key, expected_value in expected.items():
        actual = values.get(key)
        if actual != expected_value:
            raise ManifestError(
                f"OPD training override {key}={actual!r}, "
                f"expected {expected_value!r}"
            )
    expected_paths = {
        "actor_rollout_ref.model.path": spec["student"]["base_rel"],
        "distillation.teacher_models.teacher_model.model_path": (
            spec["teacher"]["merged_rel"]
        ),
        "data.train_files": os.path.join(contract.DATA_REL, "train.parquet"),
        "data.val_files": os.path.join(contract.DATA_REL, "val.parquet"),
        "custom_reward_function.path": "code/opd/zero_reward.py",
    }
    for key, expected_suffix in expected_paths.items():
        actual = values.get(key)
        if not actual or not os.path.normpath(actual).endswith(
            os.path.normpath(expected_suffix)
        ):
            raise ManifestError(
                f"OPD training path {key}={actual!r} does not end with "
                f"{expected_suffix!r}"
            )
    runtime_pythonpath = payload.get("runtime", {}).get("PYTHONPATH")
    if values.get(
        "ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH"
    ) != runtime_pythonpath:
        raise ManifestError(
            "Ray worker PYTHONPATH does not match the audited driver PYTHONPATH"
        )


def manifest_policy(payload):
    """Return the exact policy identified by a schema-1 manifest."""

    try:
        return contract.policy_from_frozen_models(
            payload.get("frozen_contract")
        )
    except (AttributeError, ValueError) as exc:
        raise ManifestError(
            "OPD run manifest does not identify a supported policy"
        ) from exc


def validate_run_manifest(payload):
    schema_version = payload.get("schema_version")
    if schema_version not in {
        LEGACY_MANIFEST_SCHEMA_VERSION,
        MANIFEST_SCHEMA_VERSION,
    }:
        raise ManifestError("unsupported OPD run manifest schema")
    if payload.get("recipe") != contract.RECIPE:
        raise ManifestError(
            "OPD run manifest recipe does not match the current contract"
        )
    if payload.get("data_recipe") != contract.DATA_RECIPE:
        raise ManifestError(
            "OPD run manifest data recipe does not match the current contract"
        )
    policy = manifest_policy(payload)
    if payload.get("frozen_contract") != frozen_contract(policy):
        raise ManifestError(
            "OPD run manifest frozen contract does not match contract.py"
        )
    if payload.get("run_mode") not in {"smoke", "probe", "formal"}:
        raise ManifestError("OPD run manifest has an invalid run mode")
    runtime = payload.get("runtime", {})
    if runtime.get("RERANKID_OPD_NORMALIZE_TOPK") != "1":
        raise ManifestError("OPD run manifest is missing the patch runtime flag")
    if (
        policy != contract.DEFAULT_POLICY
        and runtime.get("RERANKID_OPD_CPU_LAYERED_LORA") != "1"
    ):
        raise ManifestError(
            "8B OPD manifest is missing the CPU layered LoRA runtime flag"
        )
    if (
        policy == contract.DEFAULT_POLICY
        and runtime.get("RERANKID_OPD_CPU_LAYERED_LORA") is not None
    ):
        raise ManifestError(
            "1.7B OPD manifest unexpectedly enables the 8B LoRA sync patch"
        )
    if not runtime.get("PYTHONPATH"):
        raise ManifestError("OPD run manifest is missing its runtime PYTHONPATH")
    runtime_policy = runtime.get(contract.POLICY_ENV)
    if policy == contract.DEFAULT_POLICY:
        if runtime_policy not in (None, contract.DEFAULT_POLICY):
            raise ManifestError(
                "1.7B OPD manifest carries a mismatched runtime policy"
            )
    elif runtime_policy != policy:
        raise ManifestError(
            f"{policy} OPD manifest must freeze runtime policy {policy!r}"
        )
    archive_env_names = (
        "RERANKID_OPD_CHECKPOINT_ARCHIVE",
        "RERANKID_OPD_EXPORT_DIR",
        "RERANKID_OPD_MERGER_TMPDIR",
        "RERANKID_OPD_RAW_KEEP",
    )
    if payload.get("run_mode") != "formal" and any(
        runtime.get(name) is not None for name in archive_env_names
    ):
        raise ManifestError(
            "non-formal OPD manifest must not carry checkpoint archive state"
        )
    _validate_training_overrides(payload, policy)
    audit = payload.get("audit")
    if not isinstance(audit, dict):
        raise ManifestError("OPD run manifest has no audit payload")
    if audit.get("recipe") != contract.RECIPE:
        raise ManifestError("OPD audit recipe does not match the current contract")
    if audit.get("data_recipe") != contract.DATA_RECIPE:
        raise ManifestError(
            "OPD audit data recipe does not match the current contract"
        )
    objective = audit.get("objective", {})
    expected_objective = {
        "loss_mode": contract.DISTILLATION_LOSS,
        "topk": contract.DISTILLATION_TOPK,
        "use_policy_gradient": contract.DISTILLATION_USE_POLICY_GRADIENT,
        "task_reward": contract.DISTILLATION_USE_TASK_REWARDS,
        "reference_kl": False,
        "loss_max_clamp": contract.DISTILLATION_LOSS_MAX_CLAMP,
        "log_prob_min_clamp": contract.DISTILLATION_LOG_PROB_MIN_CLAMP,
        "loss_agg_mode": contract.LOSS_AGG_MODE,
    }
    for key, expected in expected_objective.items():
        if objective.get(key) != expected:
            raise ManifestError(
                f"OPD audit objective {key} does not match contract.py"
            )
    patch = objective.get("verl_patch", {})
    expected_patch = {
        "name": PATCH_NAME,
        "stock_kl_sha256": EXPECTED_STOCK_KL_SHA256,
        "patched_function_sha256": PATCHED_KL_SHA256,
        "target_module": "opd.verl_topk_patch",
        "target_qualname": "normalized_teacher_topk_kl",
        "target_identity_verified": True,
    }
    for key, expected in expected_patch.items():
        if patch.get(key) != expected:
            raise ManifestError(
                f"OPD runtime patch provenance {key} is invalid"
            )
    layered_lora_patch = objective.get("layered_lora_sync_patch")
    if policy != contract.DEFAULT_POLICY:
        from opd.verl_lora_sync_patch import (
            EXPECTED_STOCK_METHOD_SHA256,
            PATCHED_METHOD_SHA256,
            PATCH_NAME as LORA_SYNC_PATCH_NAME,
            TARGET_CLASS as LORA_SYNC_TARGET_CLASS,
            TARGET_METHOD as LORA_SYNC_TARGET_METHOD,
            TARGET_MODULE as LORA_SYNC_TARGET_MODULE,
        )

        expected_lora_sync_patch = {
            "name": LORA_SYNC_PATCH_NAME,
            "stock_method_sha256": EXPECTED_STOCK_METHOD_SHA256,
            "patched_method_sha256": PATCHED_METHOD_SHA256,
            "target_module": LORA_SYNC_TARGET_MODULE,
            "target_class": LORA_SYNC_TARGET_CLASS,
            "target_method": LORA_SYNC_TARGET_METHOD,
            "target_identity_verified": True,
            "dispatch_metadata_verified": True,
        }
        if layered_lora_patch != expected_lora_sync_patch:
            raise ManifestError(
                "8B OPD layered LoRA sync patch provenance is invalid"
            )
    elif layered_lora_patch is not None:
        raise ManifestError(
            "layered LoRA sync patch is only valid for 8B OPD"
        )
    checkpoint_patch = objective.get("checkpoint_archive_patch")
    if policy != contract.DEFAULT_POLICY and payload.get("run_mode") == "formal":
        from opd.verl_checkpoint_hook import (
            EXPECTED_LOAD_SHA256,
            EXPECTED_SAVE_SHA256,
            PATCHED_LOAD_SHA256,
            PATCHED_SAVE_SHA256,
            PATCH_NAME as CHECKPOINT_PATCH_NAME,
            TARGET_MODULE as CHECKPOINT_TARGET_MODULE,
        )

        expected_checkpoint_patch = {
            "name": CHECKPOINT_PATCH_NAME,
            "stock_save_sha256": EXPECTED_SAVE_SHA256,
            "stock_load_sha256": EXPECTED_LOAD_SHA256,
            "patched_save_sha256": PATCHED_SAVE_SHA256,
            "patched_load_sha256": PATCHED_LOAD_SHA256,
            "target_module": CHECKPOINT_TARGET_MODULE,
            "target_class": "RayPPOTrainer",
            "target_identity_verified": True,
        }
        if checkpoint_patch != expected_checkpoint_patch:
            raise ManifestError(
                "8B formal OPD checkpoint archive patch provenance is invalid"
            )
    elif checkpoint_patch is not None:
        raise ManifestError(
            "checkpoint archive patch provenance is only valid for 8B formal OPD"
        )
    data = audit.get("data", {})
    expected_data = {
        "train": (contract.TRAIN_ROWS, contract.TRAIN_PARQUET_SHA256),
        "val": (contract.VAL_ROWS, contract.VAL_PARQUET_SHA256),
    }
    for split, (rows, digest) in expected_data.items():
        split_report = data.get(split, {})
        if (
            split_report.get("rows") != rows
            or split_report.get("sha256") != digest
        ):
            raise ManifestError(
                f"OPD audit {split} data does not match contract.py"
            )
    spec = contract.policy_spec(policy)
    expected_configs = {
        role: {
            "path": "config.json",
            "sha256": digest,
        }
        for role, digest in contract.model_config_identity(policy).items()
    }
    audited_configs = audit.get("model_configs")
    if schema_version == MANIFEST_SCHEMA_VERSION and (
        audited_configs != expected_configs
    ):
        raise ManifestError(
            "OPD audit model config provenance does not match contract.py"
        )
    if (
        schema_version == LEGACY_MANIFEST_SCHEMA_VERSION
        and audited_configs is not None
        and audited_configs != expected_configs
    ):
        raise ManifestError(
            "legacy OPD audit carries mismatched model config provenance"
        )
    if policy != contract.DEFAULT_POLICY:
        weight_artifacts = audit.get("weight_artifacts")
        if not isinstance(weight_artifacts, dict):
            raise ManifestError("sharded OPD audit has no weight provenance")
        for role in ("student", "teacher"):
            artifact = weight_artifacts.get(role)
            model = spec[role]
            expected_artifact = {
                "format": "sharded",
                "index_sha256": model["weight_index_sha256"],
                "files": model["weight_files_sha256"],
                "total_size": model["weight_total_size"],
                "bundle_sha256": model["weight_sha256"],
            }
            if artifact != expected_artifact:
                raise ManifestError(
                    f"OPD audit {role} sharded weight provenance drifted"
                )
    repository = audit.get("repository", {})
    if not repository.get("head"):
        raise ManifestError("OPD audit is missing its training commit")
    if payload.get("run_mode") == "formal" and not repository.get(
        "opd_scoped_clean"
    ):
        raise ManifestError("formal OPD manifest was created from dirty code")
    runtime = payload.get("runtime", {})
    if payload.get("run_mode") == "formal":
        expected_wandb_id = formal_wandb_run_id(repository["head"], policy)
        if runtime.get("WANDB_RUN_ID") != expected_wandb_id:
            raise ManifestError(
                "formal OPD W&B run ID does not match recipe/model/commit"
            )
        if policy != contract.DEFAULT_POLICY:
            from opd.wandb_resume import (
                RESUME_POLICY,
                RESUME_POLICY_ENV,
                WANDB_ENTITY,
                WANDB_PROJECT,
            )

            expected_runtime = {
                "WANDB_ENTITY": WANDB_ENTITY,
                "WANDB_PROJECT": WANDB_PROJECT,
                RESUME_POLICY_ENV: RESUME_POLICY,
                "WANDB_RESUME": None,
                "WANDB_RESUME_FROM": None,
            }
            for key, expected in expected_runtime.items():
                if runtime.get(key) != expected:
                    raise ManifestError(
                        f"8B formal OPD W&B runtime {key}="
                        f"{runtime.get(key)!r}, expected {expected!r}"
                    )
        else:
            if runtime.get("WANDB_RESUME") != "allow":
                raise ManifestError(
                    "formal OPD W&B resume mode must be allow"
                )
            if runtime.get("WANDB_RESUME_FROM") is not None:
                raise ManifestError(
                    "1.7B formal OPD must not carry W&B resume_from"
                )
    elif runtime.get("WANDB_RUN_ID") is not None or runtime.get(
        "WANDB_RESUME"
    ) is not None or runtime.get("WANDB_RESUME_FROM") is not None:
        raise ManifestError(
            "smoke/probe OPD manifests must not carry a W&B run identity"
        )
    audited_paths = {
        "student": spec["student"]["base_rel"],
        "teacher": spec["teacher"]["merged_rel"],
    }
    for key, expected_suffix in audited_paths.items():
        actual = audit.get(key)
        if not actual or not os.path.normpath(actual).endswith(
            os.path.normpath(expected_suffix)
        ):
            raise ManifestError(
                f"OPD audit {key} path does not match contract.py"
            )
    teacher_source_hash = audit.get("native_teacher_source_sha256")
    if (
        not isinstance(teacher_source_hash, str)
        or len(teacher_source_hash) != 64
    ):
        raise ManifestError("OPD audit has no valid native teacher source hash")
    environment = audit.get("environment", {})
    if not environment.get("python_executable"):
        raise ManifestError("OPD audit is missing environment.python_executable")
    expected_versions = {
        "ray": "2.55.1",
        "torch": "2.10.0+cu128",
        "transformers": "4.57.6",
        "verl": "0.8.0",
        "vllm": "0.19.1",
    }
    if policy != contract.DEFAULT_POLICY and payload.get("run_mode") == "formal":
        expected_versions["wandb"] = "0.28.0"
    for name, expected in expected_versions.items():
        if environment.get(name) != expected:
            raise ManifestError(
                f"OPD audit environment.{name} does not match the "
                "frozen runtime"
            )
    sources = audit.get("runtime_sources", {})
    required_sources = {
        "verl.trainer.distillation.fsdp.losses",
        "verl.trainer.distillation.losses",
        "verl.workers.config.distillation",
        "verl.experimental.agent_loop.agent_loop",
        "code/utils/config.py",
        "code/opd/audit.py",
        "code/opd/contract.py",
        "code/opd/provenance.py",
        "code/opd/run_opd.sh",
        "code/opd/verl_topk_patch.py",
        "code/opd/verl_overlay/sitecustomize.py",
    }
    if policy != contract.DEFAULT_POLICY and payload.get("run_mode") == "formal":
        required_sources.update(
            {
                "code/opd/checkpoint_archive.py",
                "code/opd/export_lora_checkpoint.py",
                "code/opd/verl_checkpoint_hook.py",
                "code/opd/wandb_resume.py",
            }
        )
    if policy != contract.DEFAULT_POLICY:
        required_sources.add("code/opd/verl_lora_sync_patch.py")
    missing_sources = required_sources - set(sources)
    if missing_sources:
        raise ManifestError(
            "OPD audit is missing runtime source hashes: "
            f"{sorted(missing_sources)}"
        )
    for name, item in sources.items():
        digest = item.get("sha256") if isinstance(item, dict) else None
        if not isinstance(digest, str) or len(digest) != 64:
            raise ManifestError(f"invalid OPD runtime source hash for {name}")
    return payload


def load_manifest(path, policy=contract.DEFAULT_POLICY):
    """Load a manifest and require the requested policy.

    The default is intentionally 1.7B Pro for compatibility with all existing
    callers.  Policy-aware callers must pass ``policy="8b-pro"`` for the large
    arm.  Passing ``None`` explicitly accepts whichever supported policy the
    frozen student and teacher identify.
    """

    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read OPD run manifest {path}: {exc}") from exc
    payload = validate_run_manifest(payload)
    actual_policy = manifest_policy(payload)
    if policy is not None:
        try:
            expected_policy = contract.normalize_policy(policy)
        except ValueError as exc:
            raise ManifestError(f"invalid requested OPD policy: {policy!r}") from exc
        if actual_policy != expected_policy:
            raise ManifestError(
                f"OPD run manifest policy {actual_policy!r} does not match "
                f"requested policy {expected_policy!r}"
            )
    return payload


def _checkpoint_evidence(run_dir):
    if not os.path.isdir(run_dir):
        return []
    evidence = []
    for name in os.listdir(run_dir):
        if name.startswith("global_step_") or name in {
            "latest_checkpointed_iteration.txt",
            "checkpoint_tracker.json",
        }:
            evidence.append(name)
    return sorted(evidence)


def validate_formal_resume_state(run_dir, manifest):
    """Validate the exact checkpoint that verl auto-resume is allowed to see."""

    evidence = _checkpoint_evidence(run_dir)
    if not evidence:
        return None
    marker_path = os.path.join(
        run_dir, "latest_checkpointed_iteration.txt"
    )
    try:
        with open(marker_path, encoding="utf-8") as handle:
            latest_step = int(handle.read().strip())
    except (OSError, ValueError) as exc:
        raise ManifestError(
            "formal OPD checkpoint evidence has no valid latest-step marker"
        ) from exc
    overrides = training_override_map(manifest)
    total_steps = int(overrides["trainer.total_training_steps"])
    save_freq = int(overrides["trainer.save_freq"])
    if (
        latest_step <= 0
        or latest_step >= total_steps
        or latest_step % save_freq
    ):
        if latest_step == total_steps:
            raise ManifestError(
                f"formal OPD run is already complete at step {latest_step}"
            )
        raise ManifestError(
            f"invalid formal OPD resume step {latest_step} for "
            f"save_freq={save_freq}, total={total_steps}"
        )
    checkpoint_steps = []
    for name in os.listdir(run_dir):
        if not name.startswith("global_step_"):
            continue
        raw_step = name.removeprefix("global_step_")
        if not raw_step.isdigit():
            raise ManifestError(
                f"invalid formal OPD checkpoint directory: {name}"
            )
        step = int(raw_step)
        if step <= 0 or step > total_steps or step % save_freq:
            raise ManifestError(
                f"off-schedule formal OPD checkpoint directory: {name}"
            )
        checkpoint_steps.append(step)
    newer = sorted(step for step in checkpoint_steps if step > latest_step)
    if newer:
        raise ManifestError(
            "formal OPD run contains checkpoint directories newer than its "
            f"committed marker {latest_step}: {newer}"
        )
    checkpoint_dir = os.path.join(run_dir, f"global_step_{latest_step}")
    actor_dir = os.path.join(checkpoint_dir, "actor")
    required = (
        os.path.join(checkpoint_dir, "data.pt"),
        os.path.join(actor_dir, "fsdp_config.json"),
        os.path.join(actor_dir, "lora_train_meta.json"),
        os.path.join(actor_dir, "model_world_size_1_rank_0.pt"),
        os.path.join(actor_dir, "optim_world_size_1_rank_0.pt"),
        os.path.join(actor_dir, "extra_state_world_size_1_rank_0.pt"),
    )
    missing = [
        path
        for path in required
        if not os.path.isfile(path) or not os.path.getsize(path)
    ]
    if missing:
        raise ManifestError(
            f"formal OPD resume checkpoint is incomplete: {missing}"
        )
    try:
        with open(
            os.path.join(actor_dir, "lora_train_meta.json"),
            encoding="utf-8",
        ) as handle:
            lora_meta = json.load(handle)
        with open(
            os.path.join(actor_dir, "fsdp_config.json"),
            encoding="utf-8",
        ) as handle:
            fsdp_meta = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(
            "formal OPD resume checkpoint metadata is unreadable"
        ) from exc
    frozen = manifest["frozen_contract"]["optimization"]
    expected_lora = {
        "r": frozen["lora_rank"],
        "lora_alpha": frozen["lora_alpha"],
        "task_type": "CAUSAL_LM",
    }
    for key, expected in expected_lora.items():
        if lora_meta.get(key) != expected:
            raise ManifestError(
                f"formal OPD resume LoRA metadata {key} drifted"
            )
    if fsdp_meta.get("world_size") != 1:
        raise ManifestError("formal OPD resume FSDP world size drifted")
    return latest_step


def _first_difference(left, right, path="$"):
    if type(left) is not type(right):
        return f"{path}: {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left:
                return f"{child}: missing from existing manifest"
            if key not in right:
                return f"{child}: missing from prospective manifest"
            difference = _first_difference(left[key], right[key], child)
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: list length {len(left)} != {len(right)}"
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            difference = _first_difference(
                left_value, right_value, f"{path}[{index}]"
            )
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
        prefix=".run_manifest.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
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


def ensure_immutable_json(path, payload, *, label):
    """Create an immutable JSON artifact, accepting only an exact rerun."""

    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                existing = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(f"cannot read {label} {path}: {exc}") from exc
        difference = _first_difference(existing, payload)
        if difference:
            raise ManifestError(
                f"refusing to overwrite immutable {label}: {difference}"
            )
        return "matched"
    try:
        _atomic_create(path, payload)
    except FileExistsError:
        return ensure_immutable_json(path, payload, label=label)
    return "created"


def ensure_immutable_manifest(path, prospective, *, allow_resume):
    """Create once, or require an exact match before a permitted resume."""

    validate_run_manifest(prospective)
    policy = manifest_policy(prospective)
    run_dir = os.path.dirname(os.path.abspath(path))
    if os.path.exists(path):
        existing = load_manifest(path, policy=policy)
        if not allow_resume:
            raise ManifestError(
                f"resume is disabled and manifest already exists: {path}"
            )
        difference = _first_difference(existing, prospective)
        if difference:
            raise ManifestError(
                "refusing OPD resume because immutable provenance differs at "
                f"{difference}; existing={payload_sha256(existing)}, "
                f"prospective={payload_sha256(prospective)}"
            )
        if existing["run_mode"] == "formal":
            validate_formal_resume_state(run_dir, existing)
        return "matched"

    evidence = _checkpoint_evidence(run_dir)
    if evidence:
        raise ManifestError(
            "refusing to adopt checkpoint output without an immutable OPD "
            f"manifest: {evidence}"
        )
    try:
        _atomic_create(path, prospective)
    except FileExistsError:
        if not allow_resume:
            raise ManifestError(
                f"resume is disabled and manifest already exists: {path}"
            )
        existing = load_manifest(path, policy=policy)
        difference = _first_difference(existing, prospective)
        if difference:
            raise ManifestError(
                "concurrent OPD manifest creation disagreed at "
                f"{difference}"
            )
        if existing["run_mode"] == "formal":
            validate_formal_resume_state(run_dir, existing)
        return "matched"
    return "created"
