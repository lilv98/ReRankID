"""Lightweight verification for prepared OPD data and matched model pairs."""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd import contract
from opd.provenance import (
    validate_policy_model_configs,
    validate_weight_artifacts,
)
from utils import config


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path, expected):
    if not os.path.isfile(path):
        raise ValueError(f"missing prepared input: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{path} hash {actual} != {expected}")
    return actual


def verify_data(root=config.ROOT):
    data_dir = os.path.join(root, contract.DATA_REL)
    files = {
        "train": (
            os.path.join(data_dir, "train.parquet"),
            contract.TRAIN_PARQUET_SHA256,
        ),
        "val": (
            os.path.join(data_dir, "val.parquet"),
            contract.VAL_PARQUET_SHA256,
        ),
        "stats": (
            os.path.join(data_dir, "stats.json"),
            contract.STATS_SHA256,
        ),
    }
    return {
        name: _require_hash(path, expected)
        for name, (path, expected) in files.items()
    }


def _verify_rope(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    nested = payload.get("rope_parameters")
    if (
        payload.get("rope_theta") != 1_000_000
        or not isinstance(nested, dict)
        or nested.get("rope_theta") != 1_000_000
    ):
        raise ValueError(f"merged teacher has an incompatible RoPE schema: {path}")


def verify_policy(policy, root=config.ROOT):
    spec = contract.policy_spec(policy)
    student = spec["student"]
    teacher = spec["teacher"]
    student_path = os.path.join(root, student["base_rel"])
    teacher_path = os.path.join(root, teacher["merged_rel"])
    model_configs = validate_policy_model_configs(root, policy)
    artifacts = {
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
    for role in ("student", "teacher"):
        artifacts[role]["config"] = model_configs[role]
    for role, model, path in (
        ("student", student, student_path),
        ("teacher", teacher, teacher_path),
    ):
        _require_hash(
            os.path.join(path, "tokenizer.json"),
            model["tokenizer_file_sha256"],
        )
        artifacts[role]["tokenizer_file_sha256"] = (
            model["tokenizer_file_sha256"]
        )
    _verify_rope(os.path.join(teacher_path, "config.json"))
    return artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=("all", *contract.SUPPORTED_POLICIES),
        default="all",
    )
    args = parser.parse_args()
    policies = (
        contract.SUPPORTED_POLICIES
        if args.policy == "all"
        else (args.policy,)
    )
    report = {
        "data": verify_data(),
        "policies": {
            policy: verify_policy(policy)
            for policy in policies
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
