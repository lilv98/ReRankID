# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Repository paths, pinned base models, API keys and process defaults.

Every input, intermediate and output lives under the repository root, which is found from this
file's location. API keys come from environment variables (HF_TOKEN, OPENROUTER_API_KEY) or from
API_KEYS_FILE (default <repo>/api_keys.json, never committed), a JSON object such as
{"huggingface": "hf_...", "openrouter": "sk-..."}. Shell launchers use

    eval "$(python code/utils/config.py --shell-exports)"     # HF_HOME and HF_TOKEN
    python code/utils/config.py --model-path onerec-8b        # local snapshot of a pinned base
"""
import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])

# Base models, pinned to the Hugging Face revisions used in the paper.
MODELS = {
    "onerec-1.7b": ("OpenOneRec/OneRec-1.7B-pro", "5dc1b097ab8194f48f14730e5400a276a22f4ca1"),
    "onerec-8b": ("OpenOneRec/OneRec-8B-Pro", "17b594d1e6b322752d939a769a07860f7e4a0261"),
}

DATA = f"{ROOT}/data"
RAW_DIR = f"{DATA}/recif"                          # raw OpenOneRec-RecIF files
BENCHMARK_DIR = f"{DATA}/benchmark"
BENCH = f"{BENCHMARK_DIR}/bench.parquet"           # 400 dev + 400 test users
POOLS = f"{BENCHMARK_DIR}/pools.parquet"           # user partition (test, dev, sft, rl, reserve)
BENCH_STATS = f"{BENCHMARK_DIR}/stats.json"        # "pools" and "bench" construction statistics
SFT_DIR = f"{DATA}/sft"                            # train.json, dev.json, dataset_info.json, ...
SFT_POOL = f"{SFT_DIR}/pool.parquet"               # built locally by sft/build_pool.py
SFT_POOL_STATS = f"{SFT_DIR}/pool_stats.json"
SFT_ANNOTATIONS = f"{SFT_DIR}/annotations.jsonl"   # teacher annotations
POSTTRAIN_DIR = f"{DATA}/posttrain"
POSTTRAIN_POOL = f"{POSTTRAIN_DIR}/pool.parquet"   # shared RFT/OPD/RLVR prompt pool, built locally
POSTTRAIN_STATS = f"{POSTTRAIN_DIR}/stats.json"
OPD_DIR = f"{DATA}/opd"                            # train.parquet, val.parquet, stats.json
RLVR_DIR = f"{DATA}/rlvr"                          # train.parquet, val.parquet, stats.json

RESULTS = f"{ROOT}/results"
SAVES = f"{ROOT}/saves"
MODELS_DIR = os.environ.get("RERANKID_MODELS", f"{ROOT}/models")   # model cache, the HF_HOME default

API_KEYS_FILE = os.environ.get("API_KEYS_FILE", f"{ROOT}/api_keys.json")


def rft_dir(model):
    """data/rft/<model>: LLaMA-Factory train.json, dataset_info.json, stats.json, rollouts.jsonl."""
    return f"{DATA}/rft/{model}"


def save_dir(model, method):
    """saves/<model>/<method>: LoRA adapters or verl checkpoints of one training run."""
    return f"{SAVES}/{model}/{method}"


def merged_sft(model):
    """saves/<model>/sft-merged: merged SFT model (RLVR initializer and OPD teacher)."""
    return f"{SAVES}/{model}/sft-merged"


def relpath(path):
    """`path` relative to the repository root when it lies inside it, else unchanged. Used for every
    path written into stats and result metadata."""
    full = os.path.abspath(path)
    if full == ROOT or full.startswith(ROOT + os.sep):
        return os.path.relpath(full, ROOT)
    return str(path)


def api_key(name, default=None):
    """One key by name ('huggingface', 'openrouter', ...). <NAME>_API_KEY wins over API_KEYS_FILE,
    whose entries are {"api_key": "..."} dicts or bare strings."""
    env = os.environ.get(f"{name.upper()}_API_KEY")
    if env:
        return env
    try:
        with open(API_KEYS_FILE) as f:
            value = json.load(f).get(name)
    except (OSError, json.JSONDecodeError):
        return default
    if isinstance(value, dict):
        return value.get("api_key") or value.get("token") or value.get("key") or default
    return value if isinstance(value, str) and value else default


def setup_env(hf_offline=False):
    """Process defaults. Call at the top of every entry script, before importing vllm,
    transformers or datasets (they read HF_HOME at import time)."""
    os.environ.setdefault("HF_HOME", MODELS_DIR)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if hf_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not os.environ.get("HF_TOKEN"):
        token = api_key("huggingface")
        if token:
            os.environ["HF_TOKEN"] = token


def model_path(name):
    """Local snapshot directory of a pinned base model (downloaded on first use). Any other value,
    such as a local merged model directory, is returned unchanged."""
    if name not in MODELS:
        return name
    from huggingface_hub import snapshot_download
    repo, revision = MODELS[name]
    return snapshot_download(repo, revision=revision)


def gpu_count():
    """Number of GPUs visible to this process (respects CUDA_VISIBLE_DEVICES)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        devices = [x.strip() for x in cvd.split(",") if x.strip()]
        return 0 if not devices or devices == ["-1"] else len(devices)
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return 0
    if out.returncode != 0:
        return 0
    return sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))


def _shell_exports():
    """Export lines for launchers. The output may contain HF_TOKEN, so do not run it under xtrace."""
    pairs = [("HF_HOME", os.environ.get("HF_HOME", MODELS_DIR))]
    token = os.environ.get("HF_TOKEN") or api_key("huggingface")
    if token:
        pairs.append(("HF_TOKEN", token))
    return "\n".join(f"export {k}={shlex.quote(str(v))}" for k, v in pairs)


def main():
    parser = argparse.ArgumentParser(description="Print launcher settings.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shell-exports", action="store_true", help="print HF_HOME/HF_TOKEN exports")
    group.add_argument("--model-path", metavar="MODEL",
                       help=f"print the local snapshot path of one of {sorted(MODELS)}")
    args = parser.parse_args()
    if args.shell_exports:
        print(_shell_exports())
    else:
        setup_env()
        print(model_path(args.model_path))


if __name__ == "__main__":
    main()
