# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Download the released ReRankID data and results from Hugging Face, verify them, install the data.

The release is the gated Hugging Face dataset lilvjosephtang/ReRankID. Accept its terms on the Hub
and set HF_TOKEN (or the "huggingface" entry of API_KEYS_FILE). It has two bundles, each with a
MANIFEST.json listing the size and SHA-256 of every file:

  data/     benchmark, SFT annotations and datasets, RFT, OPD and RLVR training sets
  results/  per-user results behind every table and figure of the paper

    python code/download/release.py              # download and verify into downloads/rerankid
    python code/download/release.py --install    # also copy data/ into the repository's data/

--install copies each data file to the same relative path under data/ (for example
data/rft/onerec-8b/train.json), where the scorers and training launchers read it. Figure and table
scripts in code/paper read downloads/rerankid/results directly.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
config.setup_env()

import argparse
import hashlib
import json
import shutil

REPO_ID = "lilvjosephtang/ReRankID"
REVISION = "f2224a2a13bb673e7833f183c4e2d2d9b0281aff"   # the release commit on the Hub
DEFAULT_DIR = os.path.join(config.ROOT, "downloads", "rerankid")
BUNDLES = ("data", "results")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_files(bundle_dir):
    """{path relative to the bundle: record} from the bundle's MANIFEST.json."""
    bundle = os.path.basename(os.path.normpath(bundle_dir))
    with open(os.path.join(bundle_dir, "MANIFEST.json"), encoding="utf-8") as handle:
        records = json.load(handle)["files"]
    out = {}
    for record in records:
        rel = record["path"]
        if rel.startswith(bundle + "/"):
            rel = rel[len(bundle) + 1:]
        out[rel] = record
    return out


def verify_bundle(bundle_dir):
    records = manifest_files(bundle_dir)
    for rel, record in sorted(records.items()):
        path = os.path.join(bundle_dir, rel)
        if not os.path.isfile(path):
            raise SystemExit(f"missing {path}")
        if os.path.getsize(path) != record["bytes"] or sha256(path) != record["sha256"]:
            raise SystemExit(f"size or hash mismatch for {path}")
    print(f"verified {len(records)} files in {bundle_dir}")
    return records


def install(data_dir, records, force=False):
    """Copy verified data files to <repo>/data/<same path>."""
    for rel, record in sorted(records.items()):
        target = os.path.join(config.DATA, rel)
        if os.path.isfile(target) and sha256(target) == record["sha256"]:
            continue
        if os.path.exists(target) and not force:
            raise SystemExit(f"{target} exists and differs from the release; use --force "
                             "to overwrite it")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(os.path.join(data_dir, rel), target)
        print(f"installed data/{rel}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", default=DEFAULT_DIR, help="download directory")
    parser.add_argument("--revision", default=REVISION,
                        help="dataset branch, tag or commit hash (default: %(default)s)")
    parser.add_argument("--bundle", choices=BUNDLES, action="append",
                        help="download only this bundle (repeatable); default both")
    parser.add_argument("--skip-download", action="store_true",
                        help="verify an existing download instead of fetching it")
    parser.add_argument("--install", action="store_true", help="copy the data bundle into data/")
    parser.add_argument("--force", action="store_true",
                        help="with --install, overwrite local files that differ")
    args = parser.parse_args()
    bundles = args.bundle or list(BUNDLES)
    if args.install and "data" not in bundles:
        parser.error("--install needs the data bundle")

    if not args.skip_download:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO_ID, repo_type="dataset", revision=args.revision,
                          local_dir=args.dir, allow_patterns=[f"{b}/*" for b in bundles])
    verified = {b: verify_bundle(os.path.join(args.dir, b)) for b in bundles}
    if args.install:
        install(os.path.join(args.dir, "data"), verified["data"], force=args.force)


if __name__ == "__main__":
    main()
