"""Frozen contracts for the shared post-training pool and ordinary RFT baseline."""

import os

POSTTRAIN_RECIPE = "posttrain-v1-balanced-longhist"
RFT_RECIPE = "rft-v1-rank1-bestof8"

PARENT_ROWS = 12_000
USERS_PER_GRADE = 6_000
PILOT_ROWS = 6_000
PILOT_USERS_PER_GRADE = 3_000
# The pool manifest preserves both the auditable 12k parent and its
# deterministic 6k subset. RFT rollout and continuation use only the subset.
POOL_TARGET_ROWS = (PILOT_ROWS, PARENT_ROWS)
RFT_TARGET_ROWS = PILOT_ROWS

POOL_SEED = 17
PHASE_SEED = 19
PHASE_SCHEMA = 1
PHASE_SELECTION_VERSION = "sha256-grade-balanced-v1"

K_GRID = [32, 64, 128, 256]
K_WEIGHTS = [0.10, 0.30, 0.30, 0.30]
N_CAND_GRID = [10, 20, 30, 50, 100]
N_CAND_WEIGHTS = [0.10, 0.15, 0.40, 0.20, 0.15]

DEFAULT_POLICY = "1.7b"
_WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "anonymous")
DATA_ARTIFACT = f"{_WANDB_ENTITY}/ReRankID/data:v0"
CHECKPOINT_ARTIFACT = f"{_WANDB_ENTITY}/ReRankID/checkpoints:v0"
ADAPTER_ENTRY_PREFIXES = {
    "1.7b": "sft/1.7b-pro",
    "8b-pro": "sft/8b-pro",
}
POLICIES = {
    "1.7b": {
        "model": "OpenOneRec/OneRec-1.7B-pro",
        "model_revision": "5dc1b097ab8194f48f14730e5400a276a22f4ca1",
        "base_rel": (
            "models/hub/models--OpenOneRec--OneRec-1.7B-pro/snapshots/"
            "5dc1b097ab8194f48f14730e5400a276a22f4ca1"
        ),
        "config_sha256": (
            "43b8f3bd16d42450a441ba040bb20c0acf5e889f793d87ea4d3213d62572b94f"
        ),
        "weight_filename": "model-00000-of-00001.safetensors",
        "weight_sha256": (
            "5e1711ee090fb25c11abb6b0467f5c95de8170992b4b21b72b2091fa4e8a0e9d"
        ),
        "adapter_artifact": CHECKPOINT_ARTIFACT,
        "adapter_rel": (
            "saves/onerec-1.7b-pro/lora/sft_v3_12k_lr1e4/checkpoint-1035"
        ),
        "adapter_sha256": (
            "b248d6f9bfefad88de9f0cef0531f86392e9f178661e4ebf716d596a9b5a242c"
        ),
        "lora_name": "sft-v3-1.7b",
        "samples_attr": "RFT_V1_SAMPLES",
        "stats_attr": "RFT_V1_SAMPLE_STATS",
        "data_attr": "RFT_V1_DATA_DIR",
    },
    "8b-pro": {
        "model": "OpenOneRec/OneRec-8B-Pro",
        "model_revision": "17b594d1e6b322752d939a769a07860f7e4a0261",
        "base_rel": (
            "models/hub/models--OpenOneRec--OneRec-8B-Pro/snapshots/"
            "17b594d1e6b322752d939a769a07860f7e4a0261"
        ),
        "config_sha256": (
            "7cfca63823e48a9a61cb457bf40bf30263704f7e0dfa4472a626008a38728a30"
        ),
        # Canonical SHA-256 of the index digest and sorted shard digests.
        "weight_sha256": (
            "0249cd2ec9af8b70441a9ffbfa7b5ec157a59dd55f950bf0d0d3d1009e007352"
        ),
        "weight_index_sha256": (
            "c463df452d9333ec9f5c954bd6b315ff9ebd46f817c610d09bdb458afe6b531f"
        ),
        "weight_total_size": 16_782_026_752,
        "weight_files_sha256": {
            "model-00000-of-00003.safetensors": (
                "461c2ea0c6dbddb7e9d08a839f7a1c7a49d977f733033d72f8679baeb9f4cdcf"
            ),
            "model-00001-of-00003.safetensors": (
                "cb2d73d72c630b3235c84fa5b8d55e7ea42e3f5ec26228126b4750f97f4979ae"
            ),
            "model-00002-of-00003.safetensors": (
                "7d43e981515c9812e8d253bb826f77e7d04da8896e3af50a023f497cb0303d2b"
            ),
        },
        "adapter_artifact": CHECKPOINT_ARTIFACT,
        "adapter_rel": (
            "saves/onerec-8b-pro/lora/hp_e128_lr5e5/checkpoint-88"
        ),
        "adapter_sha256": (
            "d8a40d9f09d2c6a4246ef0a0709de465316145cdd6c4afd1136608cc669b63b9"
        ),
        "lora_name": "sft-v3-8b-e128-lr5e5",
        "samples_attr": "RFT_V1_8B_PRO_SAMPLES",
        "stats_attr": "RFT_V1_8B_PRO_SAMPLE_STATS",
        "data_attr": "RFT_V1_8B_PRO_DATA_DIR",
    },
}


def policy_contract(name):
    if name not in POLICIES:
        raise ValueError(f"unknown RFT policy {name!r}; choose from {sorted(POLICIES)}")
    return POLICIES[name]


# Backward-compatible aliases for the already frozen 1.7B formal run.
MODEL = POLICIES[DEFAULT_POLICY]["model"]
SFT_ADAPTER_ARTIFACT = POLICIES[DEFAULT_POLICY]["adapter_artifact"]
SFT_RERANK_ARTIFACT = DATA_ARTIFACT
SFT_DATA_ARTIFACT = DATA_ARTIFACT
SFT_ADAPTER_REL = POLICIES[DEFAULT_POLICY]["adapter_rel"]
SFT_ADAPTER_SHA256 = POLICIES[DEFAULT_POLICY]["adapter_sha256"]
SFT_PHASE_MANIFEST_SHA256 = "6c61237304f3c3a27deeb680a95f2c8a76bb75693bc369f92d7d93aac131fbb0"
SFT_DEV_SHA256 = "6249e6e7d5222174fc0f4407c8214633db436134fc274f6a1903d9f8ca283b33"

N_SAMPLES = 8
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 2048
SEED = 0
TOP_K = 10


def is_smoke_path(path):
    real = os.path.realpath(path)
    return real.startswith(os.path.realpath("/tmp") + os.sep) and "smoke" in real.lower()
