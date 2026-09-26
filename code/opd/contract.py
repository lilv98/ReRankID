"""Frozen contracts for matched-scale, SID-only on-policy distillation.

Each arm starts a fresh LoRA from the released Pro Base checkpoint.  A frozen,
merged, scale-matched Pro SFT-v3 policy scores the student's sampled response
tokens under the exact same SID-only prompt and prefix.  The update is pure
top-k forward KL with direct backpropagation: no task reward, label hint,
caption view, or additional reference KL participates.

The default remains the original 1.7B Pro arm.  Its public module constants and
schema-1 frozen contract are intentionally unchanged so already-created run
manifests remain loadable.
"""

import os

DEFAULT_POLICY = "1.7b-pro"
POLICY_ENV = "RERANKID_OPD_POLICY"
STORAGE_ROOT_ENV = "OPD_STORAGE_ROOT"
MERGE_ROOT_ENV = "OPD_MERGE_ROOT"
SMOKE_ROOT_ENV = "OPD_SMOKE_ROOT"
DEFAULT_STORAGE_ROOT = "/datadrive2/rerankid-opd"
DEFAULT_MERGE_ROOT = "/datadrive2/rerankid_opd_merge_e128_lr5e5"
DEFAULT_SMOKE_ROOT = "/datadrive2/rerankid-opd-smoke"

RECIPE = "opd-v3-sid-matched-forward-kl-top32-renorm-seqmean"
DATA_RECIPE = "opd-v1-sid-matched-k1"
DATA_SCHEMA = 1

TRAIN_ROWS = 6_000
VAL_ROWS = 400
VAL_K = 64
VAL_N_CAND = 30
TOP_K = 10

# sid-verl 0.8 hard-codes ``drop_last=True`` for the training DataLoader.
# Thirty divides 6,000 exactly, so every frozen prompt is queried once without
# padding, repetition, or a custom DataLoader.
TRAIN_BATCH_SIZE = 30
PPO_MINI_BATCH_SIZE = 30
TOTAL_TRAINING_STEPS = 200
N_SAMPLES = 1
AGENT_LOOP_WORKERS = 2
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K_SAMPLING = -1
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.03
PPO_EPOCHS = 1
LOSS_AGG_MODE = "seq-mean-token-mean"
LORA_RANK = 16
LORA_ALPHA = 32
MAX_PROMPT_LENGTH = 3_072
MAX_RESPONSE_LENGTH = 512
MAX_MODEL_LENGTH = 4_096
DATA_SEED = 17
SAVE_FREQ = 25
TEST_FREQ = -1

PROBE_ROWS = 1_500
PROBE_STEPS = 50
PROBE_SAVE_FREQ = 25
PROBE_WARMUP_STEPS = 6

DISTILLATION_LOSS = "forward_kl_topk"
DISTILLATION_TOPK = 32
DISTILLATION_USE_POLICY_GRADIENT = False
DISTILLATION_USE_TASK_REWARDS = False
DISTILLATION_LOSS_MAX_CLAMP = None
DISTILLATION_LOG_PROB_MIN_CLAMP = None

TRAIN_KEY_SHA256 = (
    "3e730fd69556509ca7cd9d775239f3581c571916ea9677cb6a5fbdcdce30013e"
)
SOURCE_TRAIN_PARQUET_SHA256 = (
    "8c6c6c58b079f765b400b7efa890a76f52cb33114a6bff1b0bd014846e9b38ad"
)
SOURCE_VAL_PARQUET_SHA256 = (
    "8e1a781c05edf9b898ef777cebb0b4a1c2713b0fb28f94dbcb0bfd1996cfd96d"
)
TRAIN_PARQUET_SHA256 = (
    "1afdcc53cd17890a7340a80f7c3a5be1aab3c55f3bb8b573e8434d17c18cb32a"
)
VAL_PARQUET_SHA256 = (
    "6b091a44b8f74c46157f8984729431f0647d801b35944b8e452990132a0cb011"
)
STATS_SHA256 = (
    "7e1ffb34e378e80281d873706a1d496480082b0ab3c968210c7e9d2a7d59ba6c"
)

BENCHMARK_REL = "data/benchmark/bench.parquet"
BENCHMARK_SHA256 = (
    "f0beaef9bca6b6512b8f854e801e44e50ccd39cf14685eee4fb7974b87c8ca89"
)

_STUDENT_1_7B_PRO = {
    "name": "OpenOneRec/OneRec-1.7B-pro",
    "revision": "5dc1b097ab8194f48f14730e5400a276a22f4ca1",
    "base_rel": (
        "models/hub/models--OpenOneRec--OneRec-1.7B-pro/snapshots/"
        "5dc1b097ab8194f48f14730e5400a276a22f4ca1"
    ),
    "weight_sha256": (
        "5e1711ee090fb25c11abb6b0467f5c95de8170992b4b21b72b2091fa4e8a0e9d"
    ),
    "tokenizer_file_sha256": (
        "0ccf4231869d9f8755b55133c211b48521a298a86cddf8b59ba9fe0481ecd9ba"
    ),
    "vocab_sha256": (
        "8f2cf58b677594443ef2d2783b13b0db465f37cdff55b5b09856358d0b6dec5a"
    ),
    "chat_template_sha256": (
        "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
    ),
}

_TEACHER_1_7B_PRO = {
    "name": "OpenOneRec/OneRec-1.7B-pro SFT-v3",
    "merged_rel": "saves/onerec-1.7b-pro/merged_sft_v3_12k_lr1e4",
    "weight_sha256": (
        "7ba62af989237e57ea8d488356783769143dc578ed9d2db3024d64f51633c02b"
    ),
    "tokenizer_file_sha256": (
        "1d3ff28c05506b5714b89ffc468baf46f46f128c68d6ab35a0dfb1265c944169"
    ),
    "vocab_sha256": _STUDENT_1_7B_PRO["vocab_sha256"],
    "chat_template_sha256": _STUDENT_1_7B_PRO["chat_template_sha256"],
}

_STUDENT_8B_PRO = {
    "name": "OpenOneRec/OneRec-8B-Pro",
    "revision": "17b594d1e6b322752d939a769a07860f7e4a0261",
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
    "tokenizer_file_sha256": (
        "0ccf4231869d9f8755b55133c211b48521a298a86cddf8b59ba9fe0481ecd9ba"
    ),
    "vocab_sha256": (
        "8f2cf58b677594443ef2d2783b13b0db465f37cdff55b5b09856358d0b6dec5a"
    ),
    "chat_template_sha256": (
        "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
    ),
}

_TEACHER_8B_PRO = {
    "name": "OpenOneRec/OneRec-8B-Pro SFT-v3",
    "merged_rel": "saves/onerec-8b-pro/merged_sft_v3_e128_lr5e5",
    # Canonical SHA-256 of the index digest and sorted per-shard digests.
    "weight_sha256": (
        "6f51dd4cf06aa2c22f8da2a37ec317dc10d22cd112ff93f0eb484ae8b3ece591"
    ),
    "weight_index_sha256": (
        "a73b3be9d1a7a74e4dfaf3dce45428872f3afa59aafcb657eaa80081aa4a7c1f"
    ),
    "weight_total_size": 16_782_026_752,
    "weight_files_sha256": {
        "model-00001-of-00004.safetensors": (
            "1a1fc48aa3c19f39a83b10fcf0dd44a750a30d1afbe1a2c018818bc525622dc3"
        ),
        "model-00002-of-00004.safetensors": (
            "7bada9cc97dcaa13c79c1244370ca7d61d3e03e06edd9b33b44f6b1636bd428d"
        ),
        "model-00003-of-00004.safetensors": (
            "388f7a27bea807e03f19e0cf2d918fe75190ce82c4648a01eb4d60725c47743f"
        ),
        "model-00004-of-00004.safetensors": (
            "e7c1f7afe6f38b50a8ef29dd05053e84a6959ac6b300af6186f2ecd869517905"
        ),
    },
    "tokenizer_file_sha256": (
        "1d3ff28c05506b5714b89ffc468baf46f46f128c68d6ab35a0dfb1265c944169"
    ),
    "vocab_sha256": _STUDENT_8B_PRO["vocab_sha256"],
    "chat_template_sha256": _STUDENT_8B_PRO["chat_template_sha256"],
}

POLICY_SPECS = {
    "1.7b-pro": {
        "student": _STUDENT_1_7B_PRO,
        "teacher": _TEACHER_1_7B_PRO,
        "architecture": {
            "hidden_size": 2_048,
            "intermediate_size": 6_144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        },
        "storage": {
            "run_rel": "saves/onerec-1.7b-pro/opd_v3_6k",
            "raw_run_physical": None,
            "compact_rel": "saves/onerec-1.7b-pro/opd_v3_6k_adapters",
            "merge_work_physical": None,
            "results_rel": "results/opd-1.7b-pro-v3-6k",
        },
        "experiment": "opd_v3_1.7b_pro_6k",
    },
    "8b-pro": {
        "student": _STUDENT_8B_PRO,
        "teacher": _TEACHER_8B_PRO,
        "architecture": {
            "hidden_size": 4_096,
            "intermediate_size": 12_288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        },
        "storage": {
            "run_rel": "saves/onerec-8b-pro/opd_v3_6k_e128_lr5e5",
            "raw_run_suffix": (
                "onerec-8b-pro/opd_v3_6k_e128_lr5e5"
            ),
            "compact_rel": "saves/onerec-8b-pro/opd_v3_6k_e128_lr5e5_adapters",
            "results_rel": "results/opd-8b-pro-v3-6k-e128-lr5e5",
        },
        "experiment": "opd_v3_8b_pro_e128_lr5e5_6k",
    },
}
SUPPORTED_POLICIES = tuple(POLICY_SPECS)

# Exact serialized model configurations are frozen separately from
# ``POLICY_SPECS``.  Keeping this map separate preserves the schema-1
# student/teacher payload used by completed OPD run manifests while making the
# full configuration bytes part of every new preflight and release audit.
MODEL_CONFIG_SHA256 = {
    "1.7b-pro": {
        "student": (
            "43b8f3bd16d42450a441ba040bb20c0acf5e889f793d87ea4d3213d62572b94f"
        ),
        "teacher": (
            "f158bd9b40b95bb9f01172c11b22f8225722af66b5d035a81ddcc1e8309b7c1e"
        ),
    },
    "8b-pro": {
        "student": (
            "7cfca63823e48a9a61cb457bf40bf30263704f7e0dfa4472a626008a38728a30"
        ),
        "teacher": (
            "a66d3a1ee015a3fd72fc06489b39c90cc2ef39bec6355ec005a28ad006366b33"
        ),
    },
}


def normalize_policy(policy=None):
    """Return one exact supported policy name, rejecting aliases."""

    if policy is None:
        policy = os.environ.get(POLICY_ENV, DEFAULT_POLICY)
    if not isinstance(policy, str) or policy not in POLICY_SPECS:
        raise ValueError(
            f"unsupported OPD policy {policy!r}; expected one of "
            f"{SUPPORTED_POLICIES}"
        )
    return policy


def model_config_identity(policy=None):
    """Return the frozen full-file config hashes for one matched model pair."""

    policy = normalize_policy(policy)
    identity = MODEL_CONFIG_SHA256.get(policy)
    if (
        not isinstance(identity, dict)
        or set(identity) != {"student", "teacher"}
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in identity.values()
        )
    ):
        raise ValueError(f"{policy} OPD model config identity is invalid")
    return dict(identity)


ACTIVE_POLICY = normalize_policy()


def _absolute_env_path(environ, name, default):
    value = environ.get(name, default)
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise ValueError(f"{name} must be a nonempty absolute path")
    return os.path.normpath(value)


def storage_spec(policy=None, environ=None):
    """Resolve operational storage without changing the scientific contract.

    Only the 8B Pro arm needs an external raw-checkpoint filesystem.  The
    defaults reproduce the executed run, while environment overrides make the
    launcher portable to another host.  Resolved paths are recorded in each
    immutable run manifest.
    """

    policy = normalize_policy(policy)
    environ = os.environ if environ is None else environ
    template = POLICY_SPECS[policy]["storage"]
    storage = {
        "run_rel": template["run_rel"],
        "compact_rel": template["compact_rel"],
        "results_rel": template["results_rel"],
        "raw_run_physical": None,
        "merge_work_physical": None,
        "smoke_root_physical": None,
    }
    if policy == "8b-pro":
        storage_root = _absolute_env_path(
            environ, STORAGE_ROOT_ENV, DEFAULT_STORAGE_ROOT
        )
        storage["raw_run_physical"] = os.path.join(
            storage_root, template["raw_run_suffix"]
        )
        storage["merge_work_physical"] = _absolute_env_path(
            environ, MERGE_ROOT_ENV, DEFAULT_MERGE_ROOT
        )
        storage["smoke_root_physical"] = _absolute_env_path(
            environ, SMOKE_ROOT_ENV, DEFAULT_SMOKE_ROOT
        )
    return storage


def policy_spec(policy=None, environ=None):
    """Return the matched student, teacher, architecture, and storage contract."""

    if policy is None:
        policy = ACTIVE_POLICY
    policy = normalize_policy(policy)
    spec = dict(POLICY_SPECS[policy])
    spec["storage"] = storage_spec(policy, environ=environ)
    return spec


def policy_from_frozen_models(frozen):
    """Infer a policy from exact frozen student and teacher mappings."""

    if not isinstance(frozen, dict):
        raise ValueError("OPD frozen contract is not a mapping")
    matches = [
        policy
        for policy, spec in POLICY_SPECS.items()
        if frozen.get("student") == spec["student"]
        and frozen.get("teacher") == spec["teacher"]
    ]
    if len(matches) != 1:
        raise ValueError("OPD frozen student/teacher do not identify one policy")
    return matches[0]


_ACTIVE_SPEC = policy_spec(ACTIVE_POLICY)
STUDENT = _ACTIVE_SPEC["student"]
TEACHER = _ACTIVE_SPEC["teacher"]

TOKENIZER_SPECIAL_IDS = {
    "bos_token_id": None,
    "eos_token_id": 151_645,
    "pad_token_id": 151_643,
}

DATA_REL = "data/opd_v1_6k"
RUN_REL = _ACTIVE_SPEC["storage"]["run_rel"]
RAW_RUN_PHYSICAL = _ACTIVE_SPEC["storage"]["raw_run_physical"]
COMPACT_REL = _ACTIVE_SPEC["storage"]["compact_rel"]
MERGE_WORK_PHYSICAL = _ACTIVE_SPEC["storage"]["merge_work_physical"]
RESULTS_REL = _ACTIVE_SPEC["storage"]["results_rel"]
EXPERIMENT = _ACTIVE_SPEC["experiment"]


def rooted(root, rel):
    return os.path.join(root, rel)
