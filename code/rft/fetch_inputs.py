"""Fetch and hash-check the three frozen W&B inputs needed by local RFT."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
config.setup_env()

import argparse
import hashlib
import shutil
import tempfile

import wandb

from rft.contract import (
    ADAPTER_ENTRY_PREFIXES,
    DEFAULT_POLICY,
    POLICIES,
    SFT_DATA_ARTIFACT,
    SFT_DEV_SHA256,
    SFT_PHASE_MANIFEST_SHA256,
    SFT_RERANK_ARTIFACT,
    policy_contract,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, expected):
    actual = file_sha256(path)
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
    parser.add_argument("--policy", choices=sorted(POLICIES), default=DEFAULT_POLICY)
    parser.add_argument("--skip-adapter", action="store_true")
    args = parser.parse_args()
    policy = policy_contract(args.policy)
    api = wandb.Api()

    os.makedirs(config.RERANK_DIR, exist_ok=True)
    rerank = api.artifact(SFT_RERANK_ARTIFACT)
    download_entry(
        rerank,
        "sft/annotation_phase_manifest.json",
        config.SFT_V3_PHASE_MANIFEST,
    )
    verify(config.SFT_V3_PHASE_MANIFEST, SFT_PHASE_MANIFEST_SHA256)

    os.makedirs(config.SFT_V3_DATA_DIR, exist_ok=True)
    sft = api.artifact(SFT_DATA_ARTIFACT)
    dev_path = os.path.join(config.SFT_V3_DATA_DIR, "dev.json")
    download_entry(sft, "sft/dev.json", dev_path)
    verify(dev_path, SFT_DEV_SHA256)

    adapter_dir = os.path.join(config.ROOT, policy["adapter_rel"])
    if not args.skip_adapter:
        checkpoint = api.artifact(policy["adapter_artifact"])
        prefix = ADAPTER_ENTRY_PREFIXES[args.policy]
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            download_entry(
                checkpoint,
                f"{prefix}/{name}",
                os.path.join(adapter_dir, name),
            )
    verify(os.path.join(adapter_dir, "adapter_model.safetensors"),
           policy["adapter_sha256"])


if __name__ == "__main__":
    main()
