"""Frozen contracts for the RLVR main runs and reward ablation.

Both model sizes use Pro backbones, the same SID-only prompts, 6,000 training
users, rollout budget, optimization settings, and dev-selected SFT-v3
initializer. The main arm optimizes NDCG@5. The diagnostic arm changes exactly
one factor:

* ``sft + hit1`` changes only the verifier to binary top-1 reward.
"""

import argparse
import json
import os
import shlex

RECIPE = "rlvr-v1-grpo-ndcg5"
ABLATION_RECIPE = "rlvr-v2-reward-ablation"
DATA_SCHEMA = 1
TRAIN_ROWS = 6_000
VAL_ROWS = 400
VAL_K = 64
VAL_N_CAND = 30
REWARD_CUTOFF = 5
TOP_K = 10

TRAIN_BATCH_SIZE = 60
PPO_MINI_BATCH_SIZE = 30
TOTAL_EPOCHS = 1
N_SAMPLES = 8
TEMPERATURE = 1.0
TOP_P = 0.95
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.03
KL_COEF = 0.001
PPO_EPOCHS = 1
LORA_RANK = 16
LORA_ALPHA = 32
MAX_PROMPT_LENGTH = 6_144
MAX_RESPONSE_LENGTH = 2_048
MAX_MODEL_LENGTH = 8_192
ROPE_THETA = 1_000_000
DATA_SEED = 17
SAVE_FREQ = 20
TEST_FREQ = 20

POSTTRAIN_POOL_SHA256 = "1bad03f836f0a6db2eaf234d77eb3445156544ca5539a206e8e7734b718c2e65"
POSTTRAIN_MANIFEST_SHA256 = "b2fc728f0e76bc72953bcb315de2828cd3bffe108a031e74e3ee006f51425c69"
POSTTRAIN_AUDIT_SHA256 = "58878247a788401ff5b778e0de97abb539e1f267ed58509899ddae1179c37b47"
BENCH_SHA256 = "f0beaef9bca6b6512b8f854e801e44e50ccd39cf14685eee4fb7974b87c8ca89"
TRAIN_KEY_SHA256 = "3e730fd69556509ca7cd9d775239f3581c571916ea9677cb6a5fbdcdce30013e"
TRAIN_PARQUET_SHA256 = "8c6c6c58b079f765b400b7efa890a76f52cb33114a6bff1b0bd014846e9b38ad"
VAL_PARQUET_SHA256 = "8e1a781c05edf9b898ef777cebb0b4a1c2713b0fb28f94dbcb0bfd1996cfd96d"
RLVR_STATS_SHA256 = "a998f4d63c55f55e0976a7ba636350c8dffad09b85d8492e9e64d21d1f1eef2f"

_WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "anonymous")
INPUT_ARTIFACT = f"{_WANDB_ENTITY}/ReRankID/data:v0"
CHECKPOINT_ARTIFACT = f"{_WANDB_ENTITY}/ReRankID/checkpoints:v0"
ADAPTER_ENTRY_PREFIXES = {
    "1.7b-pro": "sft/1.7b-pro",
    "8b-pro": "sft/8b-pro",
}

REWARD_FILES = {
    "ndcg5": "code/rl/reward_rerank.py",
    "hit1": "code/rl/reward_rerank_top1.py",
}
VALID_VARIANTS = {
    ("sft", "ndcg5"),
    ("sft", "hit1"),
}


def _runs(
    prefix,
    scale_label,
    main_version,
    main_experiment,
    main_results,
    main_results_artifact,
):
    """Return canonical output contracts for the two frozen RLVR cells."""

    return {
        "sft_ndcg5": {
            "run_rel": f"saves/{prefix}/{main_version}",
            "export_rel": f"saves/{prefix}/{main_version}_adapters",
            "results_rel": main_results,
            "results_artifact": main_results_artifact,
            "experiment": main_experiment,
            "dev_tag": "rlvr_sft_ndcg5_dev_select",
            "test_tag": "rlvr_sft_ndcg5_test",
        },
        "sft_hit1": {
            "run_rel": f"saves/{prefix}/rlvr_v2_sft_top1",
            "export_rel": f"saves/{prefix}/rlvr_v2_sft_top1_adapters",
            "results_rel": f"results/rlvr-v2-{scale_label}-sft-top1",
            "results_artifact": f"rlvr-v2-{scale_label}-sft-top1-results",
            "experiment": f"rlvr_v2_{prefix.removeprefix('onerec-').replace('-', '_')}_sft_top1",
            "dev_tag": "rlvr_sft_hit1_dev_select",
            "test_tag": "rlvr_sft_hit1_test",
        },
    }


POLICIES = {
    "1.7b-pro": {
        "model": "OpenOneRec/OneRec-1.7B-pro",
        "model_revision": "5dc1b097ab8194f48f14730e5400a276a22f4ca1",
        "base_rel": (
            "models/hub/models--OpenOneRec--OneRec-1.7B-pro/snapshots/"
            "5dc1b097ab8194f48f14730e5400a276a22f4ca1"
        ),
        "weight_filename": "model-00000-of-00001.safetensors",
        "weight_sha256": (
            "5e1711ee090fb25c11abb6b0467f5c95de8170992b4b21b72b2091fa4e8a0e9d"
        ),
        "config_sha256": (
            "43b8f3bd16d42450a441ba040bb20c0acf5e889f793d87ea4d3213d62572b94f"
        ),
        "rope_theta": ROPE_THETA,
        "adapter_artifact": CHECKPOINT_ARTIFACT,
        "adapter_rel": (
            "saves/onerec-1.7b-pro/lora/sft_v3_12k_lr1e4/checkpoint-1035"
        ),
        "adapter_sha256": (
            "b248d6f9bfefad88de9f0cef0531f86392e9f178661e4ebf716d596a9b5a242c"
        ),
        "adapter_config_sha256": (
            "ed4f86f7eeb8545319e672987b713204d6f62ca2a7c1abb02e10fa6e7a25fb0d"
        ),
        "merged_rel": "saves/onerec-1.7b-pro/merged_sft_v3_12k_lr1e4",
        "merged_config_sha256": (
            "f158bd9b40b95bb9f01172c11b22f8225722af66b5d035a81ddcc1e8309b7c1e"
        ),
        "runs": _runs(
            "onerec-1.7b-pro",
            "1.7b",
            "rlvr_v1_6k",
            "rlvr_v1_1.7b_pro_6k",
            "results/rlvr-1.7b-pro-v1-6k",
            "rlvr-v1-1.7b-pro-6k-results",
        ),
    },
    "8b-pro": {
        "model": "OpenOneRec/OneRec-8B-Pro",
        "model_revision": "17b594d1e6b322752d939a769a07860f7e4a0261",
        "base_rel": (
            "models/hub/models--OpenOneRec--OneRec-8B-Pro/snapshots/"
            "17b594d1e6b322752d939a769a07860f7e4a0261"
        ),
        # Canonical SHA-256 of the index digest and sorted per-shard digests.
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
        "config_sha256": (
            "7cfca63823e48a9a61cb457bf40bf30263704f7e0dfa4472a626008a38728a30"
        ),
        "rope_theta": ROPE_THETA,
        "adapter_artifact": CHECKPOINT_ARTIFACT,
        "adapter_rel": (
            "saves/onerec-8b-pro/lora/hp_e128_lr5e5/checkpoint-88"
        ),
        "adapter_sha256": (
            "d8a40d9f09d2c6a4246ef0a0709de465316145cdd6c4afd1136608cc669b63b9"
        ),
        "adapter_config_sha256": (
            "e6c009bca7b573a0cf6701e6f9cbbd2d68baa397db0c2455c8bb2de855cb0d9d"
        ),
        "merged_rel": "saves/onerec-8b-pro/merged_sft_v3_e128_lr5e5",
        "merged_config_sha256": (
            "a66d3a1ee015a3fd72fc06489b39c90cc2ef39bec6355ec005a28ad006366b33"
        ),
        "runs": _runs(
            "onerec-8b-pro",
            "8b",
            "rlvr_v2_e128_lr5e5",
            "rlvr_v2_8b_pro_e128_lr5e5",
            "results/rlvr-8b-pro-v2-e128-lr5e5-formal",
            "rlvr-v2-8b-e128-lr5e5-formal-results",
        ),
    },
}


def policy_contract(name):
    """Return the scale-specific model and SFT initializer contract."""

    if name not in POLICIES:
        raise ValueError(f"unknown RLVR policy {name!r}; choose from {sorted(POLICIES)}")
    return POLICIES[name]


def variant_key(initializer="sft", reward="ndcg5"):
    pair = (initializer, reward)
    if pair not in VALID_VARIANTS:
        allowed = ", ".join(f"{init}+{rew}" for init, rew in sorted(VALID_VARIANTS))
        raise ValueError(
            f"unsupported RLVR variant {initializer}+{reward}; choose from {allowed}"
        )
    return f"{initializer}_{reward}"


def arm_key(name, initializer="sft", reward="ndcg5"):
    """Return the manifest arm key for one registered policy and variant."""

    policy_contract(name)
    return f"{name}_{variant_key(initializer, reward)}"


def rlvr_contract(name, initializer="sft", reward="ndcg5"):
    """Return one fully resolved model, verifier, and output contract."""

    policy = policy_contract(name)
    key = variant_key(initializer, reward)
    resolved = dict(policy)
    resolved.update(policy["runs"][key])
    resolved.update(
        {
            "policy": name,
            "arm": arm_key(name, initializer, reward),
            "variant": key,
            "initializer": initializer,
            "reward": reward,
            "model_rel": policy["merged_rel"],
            "reward_rel": REWARD_FILES[reward],
            "recipe": RECIPE if key == "sft_ndcg5" else ABLATION_RECIPE,
        }
    )
    return resolved


def rooted(root, rel):
    return os.path.join(root, rel)


def main():
    parser = argparse.ArgumentParser(description="Print one frozen RLVR contract.")
    parser.add_argument("--policy", choices=sorted(POLICIES), required=True)
    parser.add_argument("--init", choices=("sft",), default="sft")
    parser.add_argument("--reward", choices=sorted(REWARD_FILES), default="ndcg5")
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    try:
        contract = rlvr_contract(args.policy, args.init, args.reward)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.shell:
        values = {
            "MODEL_REL": contract["model_rel"],
            "REWARD_REL": contract["reward_rel"],
            "OUT_REL": contract["run_rel"],
            "EXPORT_REL": contract["export_rel"],
            "RESULTS_REL": contract["results_rel"],
            "RESULTS_ARTIFACT": contract["results_artifact"],
            "EXPERIMENT": contract["experiment"],
            "DEV_TAG": contract["dev_tag"],
            "TEST_TAG": contract["test_tag"],
            "VARIANT": contract["variant"],
            "ARM": contract["arm"],
        }
        for key, value in values.items():
            print(f"export {key}={shlex.quote(str(value))}")
    else:
        print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
