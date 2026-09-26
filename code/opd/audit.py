"""Fail-closed preflight for matched-scale Pro SID-only OPD."""

import argparse
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import config

config.setup_env(hf_offline=True)

import numpy as np
import pandas as pd
import ray
import torch
import transformers
import verl
import vllm
from transformers import AutoConfig, AutoTokenizer

from opd.contract import (
    AGENT_LOOP_WORKERS,
    DATA_RECIPE,
    DATA_REL,
    DISTILLATION_LOSS,
    DISTILLATION_LOG_PROB_MIN_CLAMP,
    DISTILLATION_LOSS_MAX_CLAMP,
    DISTILLATION_TOPK,
    DISTILLATION_USE_POLICY_GRADIENT,
    DISTILLATION_USE_TASK_REWARDS,
    MAX_PROMPT_LENGTH,
    MAX_RESPONSE_LENGTH,
    LOSS_AGG_MODE,
    PPO_MINI_BATCH_SIZE,
    RECIPE,
    SOURCE_TRAIN_PARQUET_SHA256,
    SOURCE_VAL_PARQUET_SHA256,
    DEFAULT_POLICY,
    POLICY_ENV,
    SUPPORTED_POLICIES,
    TOKENIZER_SPECIAL_IDS,
    TOTAL_TRAINING_STEPS,
    TRAIN_BATCH_SIZE,
    TRAIN_PARQUET_SHA256,
    TRAIN_ROWS,
    VAL_PARQUET_SHA256,
    VAL_ROWS,
    STATS_SHA256,
    normalize_policy,
    policy_spec,
    rooted,
)
from opd.provenance import (
    ensure_immutable_json,
    ensure_immutable_manifest,
    make_run_manifest,
    validate_policy_model_configs,
    validate_weight_artifacts,
)
from opd.verl_topk_patch import report as topk_patch_report
from opd.verl_lora_sync_patch import report as lora_sync_patch_report
from opd.wandb_resume import RESUME_POLICY_ENV


EXPECTED_RUNTIME_SOURCE_SHA256 = {
    "verl.trainer.distillation.fsdp.losses": (
        "e9594e0adfba3146a08123d1f301c5659ab4be1b33d18af4ec3af846db3ce7c6"
    ),
    "verl.trainer.distillation.losses": (
        "8a7a0748d18573083997baef41a5453c256ceb4e1df4b8b2e94f5d02d4a3e7c9"
    ),
    "verl.workers.config.distillation": (
        "6cf8dd490a3c30f58f81a54a0e21a572806911ea3c874040421665acd06836de"
    ),
    "verl.experimental.agent_loop.agent_loop": (
        "9bcbc04a190ac809baf5225aaf2c98107cc91b6e3996cf46d99387d7050dd195"
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [dict(message) for message in value]


def vocab_digest(tokenizer):
    payload = json.dumps(
        sorted(tokenizer.get_vocab().items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_report(root, policy=None):
    policy = normalize_policy(policy)
    spec = policy_spec(policy)
    student = spec["student"]
    teacher = spec["teacher"]
    student_path = rooted(root, student["base_rel"])
    teacher_path = rooted(root, teacher["merged_rel"])
    student_tokenizer_file = os.path.join(student_path, "tokenizer.json")
    teacher_tokenizer_file = os.path.join(teacher_path, "tokenizer.json")

    weight_artifacts = {
        "student": validate_weight_artifacts(
            student_path,
            student,
            legacy_filename="model-00000-of-00001.safetensors",
        ),
        "teacher": validate_weight_artifacts(
            teacher_path,
            teacher,
            legacy_filename="model.safetensors",
        ),
    }
    model_configs = validate_policy_model_configs(root, policy)
    expected_hashes = (
        (student_tokenizer_file, student["tokenizer_file_sha256"]),
        (teacher_tokenizer_file, teacher["tokenizer_file_sha256"]),
    )
    for path, expected in expected_hashes:
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{path} hash {actual} != {expected}")

    configs = {
        "student": AutoConfig.from_pretrained(
            student_path, trust_remote_code=True, local_files_only=True
        ),
        "teacher": AutoConfig.from_pretrained(
            teacher_path, trust_remote_code=True, local_files_only=True
        ),
    }
    tokenizers = {
        "student": AutoTokenizer.from_pretrained(
            student_path, trust_remote_code=True, local_files_only=True
        ),
        "teacher": AutoTokenizer.from_pretrained(
            teacher_path, trust_remote_code=True, local_files_only=True
        ),
    }
    for name, model_config in configs.items():
        if model_config.architectures != ["Qwen3ForCausalLM"]:
            raise ValueError(f"{name} architecture drifted")
        if int(model_config.vocab_size) != 176_384:
            raise ValueError(f"{name} model vocabulary drifted")
        if int(model_config.max_position_embeddings) != 40_960:
            raise ValueError(f"{name} context length drifted")
        if int(model_config.rope_theta) != 1_000_000:
            raise ValueError(f"{name} rope_theta drifted")

    for name, tokenizer in tokenizers.items():
        expected = student if name == "student" else teacher
        if vocab_digest(tokenizer) != expected["vocab_sha256"]:
            raise ValueError(f"{name} logical vocabulary drifted")
        template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        if template_hash != expected["chat_template_sha256"]:
            raise ValueError(f"{name} chat template drifted")
    if tokenizers["student"].get_vocab() != tokenizers["teacher"].get_vocab():
        raise ValueError("student and teacher token-id maps differ")
    if tokenizers["student"].chat_template != tokenizers["teacher"].chat_template:
        raise ValueError("student and teacher chat templates differ")
    for name, tokenizer in tokenizers.items():
        for field, expected in TOKENIZER_SPECIAL_IDS.items():
            actual = getattr(tokenizer, field)
            if actual != expected:
                raise ValueError(
                    f"{name} {field} drifted: expected {expected}, found {actual}"
                )
    architecture = spec["architecture"]
    for name, model_config in configs.items():
        for field, expected in architecture.items():
            if int(getattr(model_config, field)) != expected:
                raise ValueError(
                    f"{name} {field} drifted: expected {expected}, "
                    f"found {getattr(model_config, field)}"
                )

    return (
        student_path,
        teacher_path,
        tokenizers["student"],
        weight_artifacts,
        model_configs,
    )


def native_teacher_report():
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    source = inspect.getsource(AgentLoopWorker._compute_teacher_logprobs)
    compact = " ".join(source.split())
    required = "sequence_ids=prompt_ids + response_ids"
    if required not in compact:
        raise ValueError("sid-verl no longer scores the same prompt/response prefix")
    forbidden = ("ground_truth", "teacher_prompt", "privileged", "hint_text")
    found = [term for term in forbidden if term in source]
    if found:
        raise ValueError(f"teacher path contains privileged fields: {found}")
    return hashlib.sha256(source.encode()).hexdigest()


def data_report(root, tokenizer):
    data_dir = rooted(root, DATA_REL)
    expected = {
        "train": (TRAIN_ROWS, TRAIN_PARQUET_SHA256),
        "val": (VAL_ROWS, VAL_PARQUET_SHA256),
    }
    stats_path = os.path.join(data_dir, "stats.json")
    if None in (TRAIN_PARQUET_SHA256, VAL_PARQUET_SHA256, STATS_SHA256):
        raise ValueError("OPD output hashes are not frozen in contract.py")
    if sha256(stats_path) != STATS_SHA256:
        raise ValueError("OPD stats hash drifted")
    with open(stats_path, encoding="utf-8") as handle:
        stats = json.load(handle)
    if stats["source"]["train_sha256"] != SOURCE_TRAIN_PARQUET_SHA256:
        raise ValueError("OPD train source provenance drifted")
    if stats["source"]["val_sha256"] != SOURCE_VAL_PARQUET_SHA256:
        raise ValueError("OPD val source provenance drifted")

    report = {}
    for split, (expected_rows, expected_hash) in expected.items():
        path = os.path.join(data_dir, f"{split}.parquet")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"{split} parquet hash drifted")
        frame = pd.read_parquet(path)
        if len(frame) != expected_rows:
            raise ValueError(f"{split} has {len(frame)} rows")
        if list(frame.columns) != [
            "data_source",
            "prompt",
            "ability",
            "reward_model",
            "extra_info",
        ]:
            raise ValueError(f"{split} schema drifted")
        if set(frame["data_source"]) != {"rerankid_opd_sid"}:
            raise ValueError(f"{split} teacher routing key drifted")
        if any(dict(value).get("ground_truth") not in ("", None) for value in frame["reward_model"]):
            raise ValueError(f"{split} contains a nonempty ground truth")
        forbidden_extra = {
            key
            for value in frame["extra_info"]
            for key in dict(value)
            if key in {"grades", "positive", "gold", "gold_index", "teacher_hint"}
        }
        if forbidden_extra:
            raise ValueError(
                f"{split} extra_info contains label fields: {sorted(forbidden_extra)}"
            )

        lengths = []
        think_counts = Counter()
        for value in frame["prompt"]:
            messages = normalize_prompt(value)
            rendered = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            think_counts[rendered.count("/think")] += 1
            token_ids = tokenizer(
                rendered, add_special_tokens=False, return_attention_mask=False
            )["input_ids"]
            lengths.append(len(token_ids))
        if think_counts != Counter({1: expected_rows}):
            raise ValueError(f"{split} /think multiplicity is {dict(think_counts)}")
        if max(lengths) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"{split} prompt max {max(lengths)} exceeds {MAX_PROMPT_LENGTH}"
            )
        report[split] = {
            "rows": expected_rows,
            "sha256": actual_hash,
            "prompt_tokens": {
                "min": min(lengths),
                "mean": round(float(np.mean(lengths)), 2),
                "p95": round(float(np.percentile(lengths, 95)), 2),
                "max": max(lengths),
            },
        }

    if TRAIN_ROWS % TRAIN_BATCH_SIZE:
        raise ValueError("formal DataLoader would drop training prompts")
    if TRAIN_BATCH_SIZE % AGENT_LOOP_WORKERS:
        raise ValueError("formal rollout batch cannot be split across agent workers")
    if TRAIN_ROWS // TRAIN_BATCH_SIZE != TOTAL_TRAINING_STEPS:
        raise ValueError("formal optimizer step count is inconsistent")
    if PPO_MINI_BATCH_SIZE != TRAIN_BATCH_SIZE:
        raise ValueError("OPD expects one optimizer mini-batch per rollout batch")
    return report


def repository_report(root):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", root, *args],
            text=True,
        ).strip()

    scoped_status = git(
        "status",
        "--porcelain=v1",
        "--",
        "code/opd",
        "code/utils/config.py",
        "code/utils/model_config.py",
        "tests/test_opd.py",
        "tests/test_opd_checkpoint_archive.py",
        "tests/test_opd_eval_protocol.py",
        "tests/test_opd_wandb_resume.py",
    )
    return {
        "head": git("rev-parse", "HEAD"),
        "opd_scoped_clean": not bool(scoped_status),
        "opd_scoped_status": scoped_status.splitlines(),
    }


def source_report(
    root,
    *,
    include_checkpoint_archive=False,
    include_lora_sync=False,
):
    module_names = (
        "verl.trainer.distillation.fsdp.losses",
        "verl.trainer.distillation.losses",
        "verl.workers.config.distillation",
        "verl.experimental.agent_loop.agent_loop",
    )
    report = {}
    for name in module_names:
        module = importlib.import_module(name)
        path = inspect.getsourcefile(module)
        if not path:
            raise ValueError(f"cannot locate audited runtime module {name}")
        actual_hash = sha256(path)
        expected_hash = EXPECTED_RUNTIME_SOURCE_SHA256[name]
        if actual_hash != expected_hash:
            raise ValueError(
                f"runtime module {name} hash {actual_hash} != "
                f"{expected_hash}"
            )
        report[name] = {
            "path": os.path.realpath(path),
            "sha256": actual_hash,
        }
    local_sources = [
        "code/utils/config.py",
        "code/utils/model_config.py",
        "code/opd/audit.py",
        "code/opd/contract.py",
        "code/opd/fix_merged_config.py",
        "code/opd/provenance.py",
        "code/opd/run_opd.sh",
        "code/opd/verl_topk_patch.py",
        "code/opd/verl_overlay/sitecustomize.py",
    ]
    if include_lora_sync:
        local_sources.append("code/opd/verl_lora_sync_patch.py")
    if include_checkpoint_archive:
        local_sources.extend(
            (
                "code/opd/checkpoint_archive.py",
                "code/opd/export_lora_checkpoint.py",
                "code/opd/verl_checkpoint_hook.py",
                "code/opd/wandb_resume.py",
            )
        )
    for rel in local_sources:
        path = os.path.join(root, rel)
        report[rel] = {
            "path": os.path.realpath(path),
            "sha256": sha256(path),
        }
    return report


def build_audit(policy=None, *, include_checkpoint_archive=False):
    policy = normalize_policy(policy)
    (
        student_path,
        teacher_path,
        tokenizer,
        weight_artifacts,
        model_configs,
    ) = model_report(config.ROOT, policy)
    report = {
        "recipe": RECIPE,
        "data_recipe": DATA_RECIPE,
        "environment": {
            "python": sys.version,
            "python_executable": os.path.realpath(sys.executable),
            "ray": ray.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "verl": verl.__version__,
            "vllm": vllm.__version__,
        },
        "student": student_path,
        "teacher": teacher_path,
        "teacher_input": "same SID-only prompt_ids + student response_ids",
        "model_configs": model_configs,
        "native_teacher_source_sha256": native_teacher_report(),
        "runtime_sources": source_report(
            config.ROOT,
            include_checkpoint_archive=include_checkpoint_archive,
            include_lora_sync=policy != DEFAULT_POLICY,
        ),
        "repository": repository_report(config.ROOT),
        "objective": {
            "loss_mode": DISTILLATION_LOSS,
            "topk": DISTILLATION_TOPK,
            "use_policy_gradient": DISTILLATION_USE_POLICY_GRADIENT,
            "task_reward": DISTILLATION_USE_TASK_REWARDS,
            "reference_kl": False,
            "loss_max_clamp": DISTILLATION_LOSS_MAX_CLAMP,
            "log_prob_min_clamp": DISTILLATION_LOG_PROB_MIN_CLAMP,
            "response_budget": MAX_RESPONSE_LENGTH,
            "agent_loop_workers": AGENT_LOOP_WORKERS,
            "loss_agg_mode": LOSS_AGG_MODE,
            "verl_patch": topk_patch_report(),
        },
        "data": data_report(config.ROOT, tokenizer),
    }
    if policy != DEFAULT_POLICY:
        report["weight_artifacts"] = weight_artifacts
        report["objective"]["layered_lora_sync_patch"] = (
            lora_sync_patch_report()
        )
    if include_checkpoint_archive:
        if policy == DEFAULT_POLICY:
            raise ValueError("checkpoint archive is only defined for 8B OPD")
        import wandb

        if wandb.__version__ != "0.28.0":
            raise ValueError(
                f"expected wandb 0.28.0, found {wandb.__version__}"
            )
        report["environment"]["wandb"] = wandb.__version__
        from opd.verl_checkpoint_hook import report as checkpoint_hook_report

        report["objective"]["checkpoint_archive_patch"] = (
            checkpoint_hook_report()
        )
    if verl.__version__ != "0.8.0":
        raise ValueError(f"expected verl 0.8.0, found {verl.__version__}")
    if vllm.__version__ != "0.19.1":
        raise ValueError(f"expected vLLM 0.19.1, found {vllm.__version__}")
    if ray.__version__ != "2.55.1":
        raise ValueError(f"expected Ray 2.55.1, found {ray.__version__}")
    if torch.__version__ != "2.10.0+cu128":
        raise ValueError(f"expected torch 2.10.0+cu128, found {torch.__version__}")
    if transformers.__version__ != "4.57.6":
        raise ValueError(
            f"expected transformers 4.57.6, found {transformers.__version__}"
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=SUPPORTED_POLICIES,
        help="matched Pro student/teacher scale (default: environment or 1.7b-pro)",
    )
    parser.add_argument(
        "--output",
        help="create or exactly match this immutable audit JSON",
    )
    parser.add_argument(
        "--manifest",
        help="create or validate this immutable run manifest",
    )
    parser.add_argument(
        "--run-mode",
        choices=("smoke", "probe", "formal"),
    )
    parser.add_argument(
        "--resume-policy",
        choices=("disable", "auto"),
        default="disable",
    )
    parser.add_argument(
        "--training-arg",
        action="append",
        default=[],
        help="one exact Hydra training override, in invocation order",
    )
    args = parser.parse_args()

    if args.manifest and (not args.run_mode or not args.training_arg):
        parser.error("--manifest requires --run-mode and --training-arg")
    policy = normalize_policy(args.policy)
    include_checkpoint_archive = (
        policy != DEFAULT_POLICY and args.run_mode == "formal"
    )
    report = build_audit(
        policy,
        include_checkpoint_archive=include_checkpoint_archive,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    output_path = os.path.abspath(args.output) if args.output else None
    if output_path and os.path.exists(output_path):
        ensure_immutable_json(
            output_path, report, label="OPD preflight audit"
        )
    if args.manifest:
        runtime_keys = [
            "CUDA_VISIBLE_DEVICES",
            "HF_HUB_OFFLINE",
            "PYTHONPATH",
            "RERANKID_OPD_NORMALIZE_TOPK",
            "TOKENIZERS_PARALLELISM",
            "TRANSFORMERS_OFFLINE",
            "WANDB_ENTITY",
            "WANDB_PROJECT",
            "WANDB_RESUME",
            "WANDB_RESUME_FROM",
            "WANDB_RUN_ID",
            "WANDB_RUN_GROUP",
            "WANDB_TAGS",
        ]
        if policy != DEFAULT_POLICY:
            runtime_keys.append("RERANKID_OPD_CPU_LAYERED_LORA")
        if include_checkpoint_archive:
            runtime_keys.extend(
                (
                    "RERANKID_OPD_CHECKPOINT_ARCHIVE",
                    "RERANKID_OPD_EXPORT_DIR",
                    "RERANKID_OPD_MERGER_TMPDIR",
                    "RERANKID_OPD_RAW_KEEP",
                    RESUME_POLICY_ENV,
                )
            )
        runtime = {
            key: os.environ.get(key)
            for key in runtime_keys
        }
        if policy != DEFAULT_POLICY:
            runtime[POLICY_ENV] = policy
        prospective = make_run_manifest(
            audit=report,
            run_mode=args.run_mode,
            training_overrides=args.training_arg,
            runtime=runtime,
            policy=policy,
        )
        status = ensure_immutable_manifest(
            os.path.abspath(args.manifest),
            prospective,
            allow_resume=args.resume_policy == "auto",
        )
        print(
            f"OPD immutable run manifest {status}: "
            f"{os.path.abspath(args.manifest)}",
            file=sys.stderr,
        )
    if output_path and not os.path.exists(output_path):
        ensure_immutable_json(
            output_path, report, label="OPD preflight audit"
        )


if __name__ == "__main__":
    main()
