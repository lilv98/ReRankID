#!/usr/bin/env bash
# Build the two host-local ReRankID environments without touching the NVIDIA driver.
#
# Machine profiles (CUDA names describe the PyTorch wheel runtime, not the host nvcc):
#   cu128: this 2xA100 / driver-535 machine; OpenRouter; Python 3.11
#   cu130: 4xA100 / driver>=580 host; Vertex AI ADC; Python 3.12
#
# Both hosts use the same environment names by default:
#   sid-lf:   LLaMA-Factory SFT/RFT training and model export
#   sid-verl: vLLM inference + RFT sampling + verl/GRPO
#
# RERANKID_LF_ENV_NAME and RERANKID_VERL_ENV_NAME may select isolated names
# when one host must retain two validated profiles, as in exact OPD
# reproduction on the training host. The defaults still create no duplicate
# environment.
#
# vLLM is intentionally installed inside sid-verl before verl.  verl's optional
# vLLM dependency range does not describe our validated vLLM 0.19/0.24 stacks,
# so verl itself is installed --no-deps and its remaining runtime deps are layered
# afterward.  There is deliberately no third, duplicate inference environment.
#
# Usage:
#   bash code/envs/setup_envs.sh [--profile auto|cu128|cu130] [--dry-run] [lf|verl|all]
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROFILE=auto
DRY_RUN=0
TARGET=all

while (($#)); do
  case "$1" in
    --profile) PROFILE=${2:?missing profile}; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    lf|verl|all) TARGET=$1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ $PROFILE == auto ]]; then
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
  DRIVER_MAJOR=${DRIVER%%.*}
  if ! [[ $DRIVER_MAJOR =~ ^[0-9]+$ ]]; then
    echo "cannot detect the NVIDIA driver; pass --profile cu128 or --profile cu130" >&2
    exit 2
  fi
  if (( DRIVER_MAJOR >= 580 )); then
    PROFILE=cu130
  else
    PROFILE=cu128
  fi
fi
[[ $PROFILE == cu128 || $PROFILE == cu130 ]] || {
  echo "profile must be auto, cu128, or cu130" >&2; exit 2;
}

PYVER=3.12
BACKEND=vertex
[[ $PROFILE == cu128 ]] && { PYVER=3.11; BACKEND=openrouter; }
LF_ENV=${RERANKID_LF_ENV_NAME:-sid-lf}
VERL_ENV=${RERANKID_VERL_ENV_NAME:-sid-verl}
for ENV_NAME in "$LF_ENV" "$VERL_ENV"; do
  [[ $ENV_NAME =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "environment names may contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 2
  }
done
SPEC=$HERE/$PROFILE
LOCAL_CU128_WHEEL_SHA256=af19d198f12a0a22d72ca6ab0874b64e53a5245289937b037ae8520ee98375a1

# Convenient default location; any cu128 installation may instead provide
# its own ABI-matching wheel explicitly via FLASH_ATTN_WHEEL.
if [[ $PROFILE == cu128 && ( $TARGET == verl || $TARGET == all ) ]]; then
  LOCAL_CU128_WHEEL=$HOME/wheels/flash_attn-2.8.3-cp311-cp311-linux_x86_64.whl
  if [[ -z ${FLASH_ATTN_WHEEL:-} && -f $LOCAL_CU128_WHEEL ]]; then
    FLASH_ATTN_WHEEL=$LOCAL_CU128_WHEEL
  fi
  if [[ ${FLASH_ATTN_WHEEL:-} == "$LOCAL_CU128_WHEEL" &&
        -z ${FLASH_ATTN_WHEEL_SHA256:-} ]]; then
    FLASH_ATTN_WHEEL_SHA256=$LOCAL_CU128_WHEEL_SHA256
  fi
  if [[ -z ${FLASH_ATTN_WHEEL:-} || ! -f $FLASH_ATTN_WHEEL ]]; then
    echo "ERROR: cu128 setup requires an existing FLASH_ATTN_WHEEL" >&2
    echo "       matching Python 3.11 and torch cu128." >&2
    exit 2
  fi
fi

if [[ -n ${FLASH_ATTN_WHEEL:-} ]]; then
  [[ -f $FLASH_ATTN_WHEEL ]] || {
    echo "ERROR: FLASH_ATTN_WHEEL is not a regular file: $FLASH_ATTN_WHEEL" >&2
    exit 2
  }
  [[ ${FLASH_ATTN_WHEEL_SHA256:-} =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: set FLASH_ATTN_WHEEL_SHA256 for the supplied wheel" >&2
    exit 2
  }
  ACTUAL_FLASH_ATTN_SHA256=$(sha256sum "$FLASH_ATTN_WHEEL" | awk '{print $1}')
  [[ $ACTUAL_FLASH_ATTN_SHA256 == "$FLASH_ATTN_WHEEL_SHA256" ]] || {
    echo "ERROR: FlashAttention wheel SHA256 mismatch" >&2
    exit 2
  }
fi

run() {
  if (( DRY_RUN )); then printf '+ '; printf '%q ' "$@"; printf '\n'; else "$@"; fi
}

env_prefix() {
  conda env list | awk -v name="$1" '$1 == name {print $NF; exit}'
}

configure() {
  local env=$1
  local vars=(
    "RERANKID_MACHINE_PROFILE=$PROFILE"
    "RERANKID_LLM_BACKEND=$BACKEND"
    "RERANKID_LF_ENV_NAME=$LF_ENV"
    "RERANKID_VERL_ENV_NAME=$VERL_ENV"
  )
  if [[ $PROFILE == cu128 ]]; then
    vars+=("API_KEYS_FILE=${API_KEYS_FILE:-$(cd "$HERE/../.." && pwd)/api_keys.json}")
  else
    vars+=("VERTEX_PROJECT=${VERTEX_PROJECT:-}")
    vars+=("VERTEX_LOCATION=${VERTEX_LOCATION:-global}")
  fi
  run conda env config vars set -n "$env" "${vars[@]}"
}

create() {
  local name=$1
  local prefix
  prefix=$(env_prefix "$name")
  if [[ -n $prefix ]]; then
    local installed_profile state_file
    state_file=$prefix/conda-meta/state
    installed_profile=$(python3 -c \
      'import json, sys; print(json.load(open(sys.argv[1])).get("env_vars", {}).get("RERANKID_MACHINE_PROFILE", ""))' \
      "$state_file" 2>/dev/null || true)
    if [[ -z $installed_profile ]]; then
      echo "ERROR: existing env $name has no RERANKID_MACHINE_PROFILE marker" >&2
      echo "Remove or rename that incomplete/legacy environment, then rerun setup." >&2
      exit 2
    fi
    if [[ $installed_profile != "$PROFILE" ]]; then
      echo "ERROR: $name belongs to profile $installed_profile, not $PROFILE" >&2
      echo "Use separate hosts as intended, or remove the incompatible environment explicitly." >&2
      exit 2
    fi
    echo "== env $name already exists"
  else
    run conda create -y -n "$name" "python=$PYVER"
    # Mark before large installs, so an interrupted partial environment cannot
    # later be reused by the other machine profile.
    configure "$name"
  fi
}

install_req() {
  local env=$1 req=$2
  run conda run -n "$env" python -m pip install -r "$req"
}

echo "== profile=$PROFILE backend=$BACKEND target=$TARGET python=$PYVER"

if [[ $TARGET == lf || $TARGET == all ]]; then
  create "$LF_ENV"
  install_req "$LF_ENV" "$SPEC/sid-lf.txt"
  configure "$LF_ENV"
fi

if [[ $TARGET == verl || $TARGET == all ]]; then
  create "$VERL_ENV"
  install_req "$VERL_ENV" "$SPEC/sid-verl.txt"
  run conda run -n "$VERL_ENV" python -m pip install --no-deps verl==0.8.0
  install_req "$VERL_ENV" "$HERE/sid-verl-extras.txt"
  # cu128 must use the validated wheel unless a matching CUDA 12.8 toolkit is
  # deliberately configured.  Its host nvcc 12.0 cannot reliably build against
  # torch cu128.  cu130 may use a supplied wheel or its matching local toolkit.
  if [[ -n ${FLASH_ATTN_WHEEL:-} ]]; then
    run conda run -n "$VERL_ENV" python -m pip install "$FLASH_ATTN_WHEEL"
  else
    run conda run -n "$VERL_ENV" python -m pip install \
      --no-deps --no-build-isolation --require-hashes \
      -r "$HERE/flash-attn-source.txt"
  fi
  configure "$VERL_ENV"
fi

echo "done: profile=$PROFILE backend=$BACKEND target=$TARGET"
echo "activate: $LF_ENV (SFT/RFT/export), $VERL_ENV (vLLM inference/RFT sampling/GRPO)"
echo "verify:  python code/envs/verify_env.py --role <lf|verl> --profile $PROFILE"
