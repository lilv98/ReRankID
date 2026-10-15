#!/usr/bin/env bash
# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

# Build the two ReRankID conda environments for one CUDA profile.
#
#   sid-lf    LLaMA-Factory SFT/RFT training and model export
#   sid-verl  vLLM scoring, RFT sampling, OPD and RLVR (verl 0.8.0)
#
# Profiles name the PyTorch wheel runtime: cu130 (Python 3.12, vLLM 0.24, default) and cu128
# (Python 3.11, vLLM 0.19.1, used for OPD). sid-verl installs the profile's vLLM first, then
# verl --no-deps, then sid-verl-extras.txt, then flash-attn 2.8.3: from FLASH_ATTN_WHEEL if set,
# else built from the PyPI source release (--no-deps keeps the pinned torch stack intact).
# RERANKID_LF_ENV_NAME and RERANKID_VERL_ENV_NAME override the environment names.
#
# Usage: bash code/envs/setup_envs.sh --profile cu128|cu130 [--dry-run] [lf|verl|all]
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROFILE=
DRY_RUN=0
TARGET=all

while (($#)); do
  case "$1" in
    --profile) PROFILE=${2:?missing profile}; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    lf|verl|all) TARGET=$1; shift ;;
    -h|--help) echo "usage: setup_envs.sh --profile cu128|cu130 [--dry-run] [lf|verl|all]"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ $PROFILE == cu128 || $PROFILE == cu130 ]] || {
  echo "usage: setup_envs.sh --profile cu128|cu130 [--dry-run] [lf|verl|all]" >&2; exit 2;
}

PYVER=3.12
[[ $PROFILE == cu128 ]] && PYVER=3.11
LF_ENV=${RERANKID_LF_ENV_NAME:-sid-lf}
VERL_ENV=${RERANKID_VERL_ENV_NAME:-sid-verl}
SPEC=$HERE/$PROFILE

if [[ -n ${FLASH_ATTN_WHEEL:-} && ! -f $FLASH_ATTN_WHEEL ]]; then
  echo "FLASH_ATTN_WHEEL not found: $FLASH_ATTN_WHEEL" >&2; exit 2
fi

run() {
  if (( DRY_RUN )); then printf '+ '; printf '%q ' "$@"; printf '\n'; else "$@"; fi
}

create() {
  if conda env list | awk '{print $1}' | grep -qx "$1"; then
    local have
    have=$(conda run -n "$1" python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    if [[ $have != "$PYVER" ]]; then
      echo "env $1 exists with Python $have, but profile $PROFILE needs $PYVER;" \
           "set RERANKID_LF_ENV_NAME / RERANKID_VERL_ENV_NAME to other names" >&2
      exit 2
    fi
    echo "== env $1 already exists (Python $have)"
  else
    run conda create -y -n "$1" "python=$PYVER"
  fi
}

pip_in() {
  local env=$1; shift
  run conda run -n "$env" python -m pip install "$@"
}

echo "== profile=$PROFILE python=$PYVER target=$TARGET"

if [[ $TARGET == lf || $TARGET == all ]]; then
  create "$LF_ENV"
  pip_in "$LF_ENV" -r "$SPEC/sid-lf.txt"
fi

if [[ $TARGET == verl || $TARGET == all ]]; then
  create "$VERL_ENV"
  pip_in "$VERL_ENV" -r "$SPEC/sid-verl.txt"
  pip_in "$VERL_ENV" --no-deps verl==0.8.0
  pip_in "$VERL_ENV" -r "$HERE/sid-verl-extras.txt"
  if [[ -n ${FLASH_ATTN_WHEEL:-} ]]; then
    pip_in "$VERL_ENV" "$FLASH_ATTN_WHEEL"
  else
    pip_in "$VERL_ENV" --no-deps --no-build-isolation flash-attn==2.8.3
  fi
fi

echo "done. verify: python code/envs/verify_env.py --role lf|verl --profile $PROFILE --imports"
