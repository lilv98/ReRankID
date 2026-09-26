"""Validate and optionally publish the frozen formal RLVR input artifact.

Dry-run is the default.  ``--upload`` refuses a dirty worktree, and no smoke or
generated rollout data are included.
"""

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
from utils.artifact_release import (git_snapshot, log_immutable_artifact,
                                    require_clean_pushed_release)

config.setup_env()

from rl.contract import (BENCH_SHA256, POSTTRAIN_AUDIT_SHA256,
                         POSTTRAIN_MANIFEST_SHA256, POSTTRAIN_POOL_SHA256,
                         RECIPE, RLVR_STATS_SHA256, TRAIN_PARQUET_SHA256,
                         VAL_PARQUET_SHA256)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


FILES = (
    (config.RLVR_V1_TRAIN, "rlvr_v1_6k/train.parquet", TRAIN_PARQUET_SHA256),
    (config.RLVR_V1_VAL, "rlvr_v1_6k/val.parquet", VAL_PARQUET_SHA256),
    (config.RLVR_V1_STATS, "rlvr_v1_6k/stats.json", RLVR_STATS_SHA256),
    (config.POSTTRAIN_V1_PHASE_MANIFEST, "rerank/posttrain_v1_phase_manifest.json",
     POSTTRAIN_MANIFEST_SHA256),
    (config.POSTTRAIN_V1_AUDIT, "rerank/posttrain_v1_audit.json", POSTTRAIN_AUDIT_SHA256),
    (config.BENCH, "benchmark/bench.parquet", BENCH_SHA256),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    manifest = []
    source_pool_sha = sha256(config.POSTTRAIN_V1_POOL)
    if source_pool_sha != POSTTRAIN_POOL_SHA256:
        raise SystemExit(
            f"source pool hash mismatch: {source_pool_sha} != {POSTTRAIN_POOL_SHA256}"
        )
    for path, artifact_path, expected in FILES:
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"hash mismatch for {path}: {actual} != {expected}")
        manifest.append({"path": artifact_path, "sha256": actual, "bytes": os.path.getsize(path)})
    git = git_snapshot(config.ROOT)
    commit = git["commit"]
    dirty = git["dirty"]
    report = {
        "recipe": RECIPE,
        "git_commit": commit,
        "git_dirty": dirty,
        "source_posttrain_pool_sha256": source_pool_sha,
        "files": manifest,
    }
    print(json.dumps(report, indent=2))
    if not args.upload:
        print("dry-run only; pass --upload after committing and pushing this workflow")
        return
    if dirty:
        raise SystemExit("formal RLVR input publication requires a clean Git worktree")
    require_clean_pushed_release(config.ROOT, expected_commit=commit)

    import wandb

    run = wandb.init(project=config.WANDB_PROJECT, job_type="publish-rlvr-v1-inputs")
    artifact = wandb.Artifact(
        "rlvr-v1-inputs", type="dataset",
        metadata={
            "recipe": RECIPE,
            "git_commit": commit,
            "source_posttrain_pool_sha256": source_pool_sha,
            "files": manifest,
        },
    )
    for path, artifact_path, _ in FILES:
        artifact.add_file(path, name=artifact_path)
    try:
        identity = log_immutable_artifact(run, artifact)
    finally:
        run.finish()
    print(json.dumps({"uploaded_artifact": identity}, indent=2))


if __name__ == "__main__":
    main()
