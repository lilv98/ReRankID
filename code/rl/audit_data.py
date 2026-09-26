"""Audit frozen RLVR data with the exact selected Pro initializer tokenizer."""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config

config.setup_env(hf_offline=True)

import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoTokenizer

from rl.contract import (MAX_PROMPT_LENGTH, POLICIES, TRAIN_PARQUET_SHA256,
                         RLVR_STATS_SHA256, TRAIN_ROWS, VAL_PARQUET_SHA256,
                         VAL_ROWS, rlvr_contract, rooted)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_messages(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [dict(message) for message in value]


def length_summary(values):
    array = np.asarray(values, dtype=np.int64)
    return {
        "min": int(array.min()),
        "mean": round(float(array.mean()), 2),
        "p95": round(float(np.percentile(array, 95)), 2),
        "max": int(array.max()),
    }


def audit_rope_config(model_path, expected_theta, require_dual_schema):
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    nested = raw.get("rope_parameters")
    if raw.get("rope_theta") != expected_theta:
        raise ValueError(
            f"{model_path} legacy rope_theta {raw.get('rope_theta')!r} != "
            f"{expected_theta}; run code/rl/fix_merged_config.py after export"
        )
    if isinstance(nested, dict) and nested.get("rope_theta") != expected_theta:
        raise ValueError(
            f"{model_path} nested rope theta {nested.get('rope_theta')!r} != "
            f"{expected_theta}"
        )
    if require_dual_schema and not isinstance(nested, dict):
        raise ValueError(
            f"{model_path} merged export is missing rope_parameters; "
            "run code/rl/fix_merged_config.py after export"
        )

    # Validate with the actual training/inference environment.  In Transformers
    # 4, vLLM's compatibility patch copies the top-level field into the nested
    # schema; in Transformers 5 the nested schema is native.
    from vllm.transformers_utils.config import patch_rope_parameters

    model_config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    patch_rope_parameters(model_config)
    effective = (getattr(model_config, "rope_parameters", None) or {}).get("rope_theta")
    if effective != expected_theta:
        raise ValueError(
            f"{model_path} effective vLLM rope theta {effective!r} != {expected_theta}"
        )
    return effective


def audit_policy(name, frames, initializer="sft"):
    policy = rlvr_contract(name, initializer, "ndcg5")
    model_path = rooted(config.ROOT, policy["model_rel"])
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise ValueError(f"missing {initializer} initializer for {name}: {model_path}")
    effective_theta = audit_rope_config(
        model_path, policy["rope_theta"], require_dual_schema=initializer == "sft"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    report = {
        "initializer": initializer,
        "model": policy["model_rel"],
        "effective_rope_theta": effective_theta,
    }
    for split, frame in frames.items():
        lengths = []
        for value in frame["prompt"]:
            messages = prompt_messages(value)
            if messages[-1]["content"].count("/think") != 1:
                raise ValueError(f"{name} {split} prompt does not contain one explicit /think")
            rendered = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            if rendered.count("/think") != 1:
                raise ValueError(
                    f"{name} tokenizer changes /think multiplicity to {rendered.count('/think')}"
                )
            token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            lengths.append(len(token_ids))
        report[split] = length_summary(lengths)
        if report[split]["max"] > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"{name} {split} max prompt {report[split]['max']} exceeds "
                f"{MAX_PROMPT_LENGTH}"
            )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["all", *sorted(POLICIES)], default="all")
    parser.add_argument("--init", choices=("sft",), default="sft")
    args = parser.parse_args()
    if sha256(config.RLVR_V1_STATS) != RLVR_STATS_SHA256:
        raise ValueError("RLVR stats.json differs from the frozen contract")
    with open(config.RLVR_V1_STATS) as handle:
        stats = json.load(handle)
    expected = {
        "train": (config.RLVR_V1_TRAIN, TRAIN_ROWS, TRAIN_PARQUET_SHA256),
        "val": (config.RLVR_V1_VAL, VAL_ROWS, VAL_PARQUET_SHA256),
    }
    frames = {}
    for split, (path, rows, frozen_sha) in expected.items():
        actual_sha = sha256(path)
        if actual_sha != stats["files"][f"{split}_sha256"] or actual_sha != frozen_sha:
            raise ValueError(f"{split} parquet differs from the frozen contract")
        frame = pd.read_parquet(path)
        if len(frame) != rows:
            raise ValueError(f"{split} rows {len(frame)} != {rows}")
        frames[split] = frame
    names = sorted(POLICIES) if args.policy == "all" else [args.policy]
    report = {
        "train_sha256": stats["files"]["train_sha256"],
        "val_sha256": stats["files"]["val_sha256"],
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "policies": {
            name: audit_policy(name, frames, initializer=args.init) for name in names
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
