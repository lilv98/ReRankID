# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Download and verify the raw OpenOneRec-RecIF files the benchmark is built from (paper Sec. 3).

Needed only to rebuild the benchmark or the training pools; the release (download/release.py)
already holds every derived file. The source is the gated Hugging Face dataset
OpenOneRec/OpenOneRec-RecIF: accept its terms on the Hub and set HF_TOKEN. The revision is pinned
because negatives are sampled in parquet batch order, so other revisions give a different benchmark.
With these files, benchmark/build_pools.py and benchmark/build_bench.py reproduce pools.parquet and
bench.parquet row for row (file bytes differ by writer metadata, so compare contents).

    python code/download/recif.py            # download into data/recif/ and verify
    python code/download/recif.py --verify   # verify files already in data/recif/
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
config.setup_env()

import argparse
import hashlib

REPO_ID = "OpenOneRec/OpenOneRec-RecIF"
REVISION = "8f7cf2ee0b949e955a87a708d02024687be232c8"
FILES = {
    "onerec_bench_release.parquet": (
        "4800e435990859ac7800fbe358e633caa2ffa6c5a350b4dc865126785f18ccb5", 1721493116),
    "video_ad_pid2sid.parquet": (
        "b92eda31971a771eb7267079cf9eb2f3c492af67389af49457f5abe076200a06", 160633770),
    "pid2caption.parquet": (
        "30fdd0602c8c50996215374329dd45da75b8c53ec23b8179bc562fc2c8a873a9", 5863423098),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(directory=config.RAW_DIR):
    for name, (expected, size) in FILES.items():
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            raise SystemExit(f"missing {path}; run python code/download/recif.py")
        if os.path.getsize(path) != size:
            raise SystemExit(f"size mismatch for {path}: {os.path.getsize(path)} != {size}")
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"hash mismatch for {path}: {actual} != {expected}")
        print(f"verified {name}  sha256={actual}")


def download(directory=config.RAW_DIR):
    from huggingface_hub import hf_hub_download
    os.makedirs(directory, exist_ok=True)
    for name in FILES:
        hf_hub_download(REPO_ID, name, repo_type="dataset", revision=REVISION,
                        local_dir=directory)
        print(f"downloaded {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--verify", action="store_true",
                        help="only verify files already present")
    parser.add_argument("--dir", default=config.RAW_DIR)
    args = parser.parse_args()
    if not args.verify:
        download(args.dir)
    verify(args.dir)


if __name__ == "__main__":
    main()
