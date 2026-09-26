"""Fetch and hash-check formal RLVR inputs from W&B.

The dataset artifact contains only frozen formal data. SFT-v3 adapters remain
in their existing model artifacts. Every download resolves an exact W&B
version and is checked again by SHA-256 before training starts.
"""

import argparse
import hashlib
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config

config.setup_env()

import wandb

from rl.contract import (
    ADAPTER_ENTRY_PREFIXES,
    BENCH_SHA256,
    INPUT_ARTIFACT,
    POLICIES,
    POSTTRAIN_AUDIT_SHA256,
    POSTTRAIN_MANIFEST_SHA256,
    RLVR_STATS_SHA256,
    TRAIN_PARQUET_SHA256,
    VAL_PARQUET_SHA256,
    policy_contract,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"hash mismatch for {path}: {actual} != {expected}")
    print(f"verified {os.path.relpath(path, config.ROOT)}  sha256={actual}")


def download_entry(artifact, entry, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rerankid-wandb-") as temporary:
        downloaded = artifact.get_entry(entry).download(root=temporary)
        shutil.copy2(downloaded, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy", choices=["all", *sorted(POLICIES)], default="all",
        help="which SFT-v3 adapter to fetch; formal data are always fetched",
    )
    args = parser.parse_args()
    api = wandb.Api()

    artifact = api.artifact(INPUT_ARTIFACT)
    entries = {
        "rlvr/train.parquet": (config.RLVR_V1_TRAIN, TRAIN_PARQUET_SHA256),
        "rlvr/val.parquet": (config.RLVR_V1_VAL, VAL_PARQUET_SHA256),
        "rlvr/stats.json": (config.RLVR_V1_STATS, RLVR_STATS_SHA256),
        "shared-posttrain/phase_manifest.json": (
            config.POSTTRAIN_V1_PHASE_MANIFEST, POSTTRAIN_MANIFEST_SHA256
        ),
        "shared-posttrain/audit.json": (
            config.POSTTRAIN_V1_AUDIT, POSTTRAIN_AUDIT_SHA256
        ),
        "benchmark/bench.parquet": (config.BENCH, BENCH_SHA256),
    }
    for entry, (path, expected) in entries.items():
        download_entry(artifact, entry, path)
        verify(path, expected)

    names = sorted(POLICIES) if args.policy == "all" else [args.policy]
    for name in names:
        policy = policy_contract(name)
        adapter_dir = os.path.join(config.ROOT, policy["adapter_rel"])
        checkpoint = api.artifact(policy["adapter_artifact"])
        prefix = ADAPTER_ENTRY_PREFIXES[name]
        for filename in ("adapter_model.safetensors", "adapter_config.json"):
            download_entry(
                checkpoint,
                f"{prefix}/{filename}",
                os.path.join(adapter_dir, filename),
            )
        verify(
            os.path.join(adapter_dir, "adapter_model.safetensors"),
            policy["adapter_sha256"],
        )


if __name__ == "__main__":
    main()
