"""Central config: repo-rooted paths, API keys, machine defaults, experiment tracking.

The repo root is auto-detected from this file's location, so the whole tree is machine-portable and
SELF-CONTAINED: every input (raw data, HF model cache), intermediate (pools, SFT datasets) and output
(results, LoRA checkpoints) lives inside the repo root. Env vars override every knob.

API keys come from a local json file (default <repo root>/api_keys.json, override via
API_KEYS_FILE) with per-key env-var overrides. Keys are never committed; shell export output is
available only through the explicit --shell-exports mode used by the launchers.

Also usable as a shell env emitter for tools launched outside python (llamafactory-cli, verl):
    eval "$(python3 code/utils/config.py --shell-exports)"
This launcher-only mode exports HF/W&B credentials, not the unrelated OpenRouter key.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])          # .../ReRankID

# ---- everything below lives inside ROOT (self-contained) ----
DATA = f"{ROOT}/data"
RAW_DIR = f"{DATA}/recif"                                # gated OpenOneRec/OpenOneRec-RecIF dumps
RERANK_DIR = f"{DATA}/rerank"                            # training pools / annotations / rollout files
BENCHMARK_DIR = f"{DATA}/benchmark"                      # FINAL v1 benchmark artifacts (pools/bench)
BENCH = f"{BENCHMARK_DIR}/bench.parquet"          # the frozen benchmark (built by benchmark/build_bench.py)
STATS = f"{BENCHMARK_DIR}/stats.json"             # THE single data-generation stats file: every
                                                  # builder merges its section (funnel, history
                                                  # stats, composition, floors, sha256) via
                                                  # utils.data_io.update_stats — no per-artifact
                                                  # manifests
# Final SFT-v3 artifacts. Explicit paths prevent a formal run from resuming
# from or overwriting another stage.
SFT_V3_POOL = f"{RERANK_DIR}/sft_pool_v3.parquet"
SFT_V3_ANNOTATIONS = f"{RERANK_DIR}/sft_annotations_v3.jsonl"
SFT_V3_ANNOTATION_COST = f"{RERANK_DIR}/sft_annotations_v3_cost.json"
SFT_V3_PHASE_MANIFEST = f"{RERANK_DIR}/sft_v3_phase_manifest.json"
SFT_V3_AUDIT = f"{RERANK_DIR}/sft_v3_audit.json"
SFT_V3_PILOT_AUDIT = f"{RERANK_DIR}/sft_v3_6k_audit.json"
SFT_V3_PILOT_DATA_DIR = f"{DATA}/sft_v3_6k"
SFT_V3_DATA_DIR = f"{DATA}/sft_v3"

# Shared post-training prompts and the final rejection-fine-tuning recipe.
# RFT, OPD, and RLVR consume the same balanced one-row-per-user parent pool,
# while each method keeps its own trajectories and training dataset.
POSTTRAIN_V1_POOL = f"{RERANK_DIR}/posttrain_pool_v1.parquet"
POSTTRAIN_V1_PHASE_MANIFEST = f"{RERANK_DIR}/posttrain_v1_phase_manifest.json"
POSTTRAIN_V1_AUDIT = f"{RERANK_DIR}/posttrain_v1_audit.json"
RFT_V1_SAMPLES = f"{RERANK_DIR}/rft_v1_samples.jsonl"
RFT_V1_SAMPLE_STATS = f"{RERANK_DIR}/rft_v1_samples_stats.json"
RFT_V1_DATA_DIR = f"{DATA}/rft_v1_6k"
RFT_V1_8B_PRO_SAMPLES = f"{RERANK_DIR}/rft_v1_8b_pro_e128_lr5e5_samples.jsonl"
RFT_V1_8B_PRO_SAMPLE_STATS = (
    f"{RERANK_DIR}/rft_v1_8b_pro_e128_lr5e5_samples_stats.json"
)
RFT_V1_8B_PRO_DATA_DIR = f"{DATA}/rft_v1_8b_pro_e128_lr5e5_6k"
RLVR_V1_DATA_DIR = f"{DATA}/rlvr_v1_6k"
RLVR_V1_TRAIN = f"{RLVR_V1_DATA_DIR}/train.parquet"
RLVR_V1_VAL = f"{RLVR_V1_DATA_DIR}/val.parquet"
RLVR_V1_STATS = f"{RLVR_V1_DATA_DIR}/stats.json"

RESULTS = f"{ROOT}/results"                              # benchmark result JSONs live here directly
LOG_DIR = f"{RESULTS}/_logs"
SAVES = f"{ROOT}/saves"                                  # LoRA adapters / merged models
MODELS_DIR = os.environ.get("RERANKID_MODELS", f"{ROOT}/models")   # HF cache (HF_HOME)

WANDB_PROJECT = "ReRankID"

# Local cu128/OpenRouter host default. The cu130/Vertex host uses ADC for LLM calls; it may still
# point API_KEYS_FILE at a private file for HF/W&B, but no key file is committed to this repository.
API_KEYS_FILE = os.environ.get("API_KEYS_FILE", f"{ROOT}/api_keys.json")


def _api_entry(name):
    try:
        with open(API_KEYS_FILE) as f:
            return json.load(f).get(name)
    except (OSError, json.JSONDecodeError):
        return None


def api_key(name, default=None):
    """One key from the local api_keys file ('openrouter' / 'huggingface' / 'wandb' / ...). Entries
    are dicts like {"api_key": "..."} (pypi uses "token"); a bare string value also works. An env var
    of the form <NAME>_API_KEY wins over the file."""
    env = os.environ.get(f"{name.upper()}_API_KEY")
    if env:
        return env
    v = _api_entry(name)
    if isinstance(v, dict):
        return v.get("api_key") or v.get("token") or v.get("key") or default
    return v if isinstance(v, str) and v else default


def wandb_entity(default=None):
    """The W&B entity (team/user) recorded next to the wandb api key, if any."""
    env = os.environ.get("WANDB_ENTITY")
    if env:
        return env
    v = _api_entry("wandb")
    return v.get("entity", default) if isinstance(v, dict) else default


# Python entry points may need all three; training launchers receive only the first two.
_KEY_EXPORTS = (("HF_TOKEN", "huggingface"), ("WANDB_API_KEY", "wandb"),
                ("OPENROUTER_API_KEY", "openrouter"))
_SHELL_KEY_EXPORTS = _KEY_EXPORTS[:2]  # launchers train locally; they never call OpenRouter


def setup_env(hf_offline=False):
    """Process-level env defaults. Call at the very top of every entry script, BEFORE importing
    anything heavy (vllm / transformers / datasets resolve HF_HOME at import time)."""
    os.environ.setdefault("HF_HOME", MODELS_DIR)
    if hf_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    ent = wandb_entity()
    if ent:
        os.environ.setdefault("WANDB_ENTITY", ent)
    for env_name, key_name in _KEY_EXPORTS:
        if not os.environ.get(env_name):
            k = api_key(key_name)
            if k:
                os.environ[env_name] = k


def gpu_count():
    """#GPUs visible to this process (CUDA_VISIBLE_DEVICES-aware) — the vLLM tensor-parallel default."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        devices = [x.strip() for x in cvd.split(",") if x.strip()]
        return 0 if not devices or devices == ["-1"] else len(devices)
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return 0
        return len([ln for ln in out.stdout.splitlines() if ln.startswith("GPU ")])
    except Exception:
        return 0


def _emit_shell_exports():
    """Print exports for a launcher's command substitution. Call only via --shell-exports and never
    with shell xtrace enabled, because the output contains configured credentials."""
    import shlex
    pairs = [("HF_HOME", MODELS_DIR), ("WANDB_PROJECT", WANDB_PROJECT)]
    ent = wandb_entity()
    if ent:
        pairs.append(("WANDB_ENTITY", ent))
    for env_name, key_name in _SHELL_KEY_EXPORTS:
        k = os.environ.get(env_name) or api_key(key_name)
        if k:
            pairs.append((env_name, k))
    for k, v in pairs:
        print(f"export {k}={shlex.quote(str(v))}")


if __name__ == "__main__":
    if sys.argv[1:] != ["--shell-exports"]:
        raise SystemExit("usage: python code/utils/config.py --shell-exports")
    _emit_shell_exports()
