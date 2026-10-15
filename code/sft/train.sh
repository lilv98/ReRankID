#!/usr/bin/env bash
# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

# LLaMA-Factory LoRA launcher for the SFT and RFT configs (paper Sec. 5).
# Keeps the paper's effective batch on any GPU count: accumulation = EFF_BATCH / (PER_DEV x GPUs).
# Defaults per config: SFT 1.7B 32/2, SFT 8B 128/4, RFT 32/2 (EFF_BATCH/PER_DEV, both overridable).
#   bash code/sft/train.sh code/sft/configs/onerec-1.7b.yaml
#   bash code/sft/train.sh code/rft/configs/onerec-8b.yaml
# Extra key=value arguments are passed to llamafactory-cli unchanged.
set -euo pipefail
USAGE="usage: bash code/sft/train.sh <config.yaml> [key=value ...]"
[[ ${1:-} == -h || ${1:-} == --help ]] && { echo "$USAGE"; exit 0; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CFG=$(realpath "${1:?$USAGE}")
cd "$ROOT"
eval "$(python3 code/utils/config.py --shell-exports)"   # HF_HOME and HF_TOKEN

case "$CFG" in
  */sft/configs/onerec-1.7b.yaml) EFF_DEFAULT=32;  PER_DEV_DEFAULT=2 ;;
  */sft/configs/onerec-8b.yaml)   EFF_DEFAULT=128; PER_DEV_DEFAULT=4 ;;
  */rft/configs/onerec-*.yaml)    EFF_DEFAULT=32;  PER_DEV_DEFAULT=2 ;;
  *) EFF_DEFAULT=""; PER_DEV_DEFAULT="" ;;
esac
EFF=${EFF_BATCH:-$EFF_DEFAULT}
PER_DEV=${PER_DEV:-$PER_DEV_DEFAULT}
if [[ -z $EFF || -z $PER_DEV ]]; then
  echo "set EFF_BATCH and PER_DEV for $CFG" >&2; exit 2
fi
NGPUS=$(PYTHONPATH=code python3 -c 'from utils.config import gpu_count; print(gpu_count())')
DENOM=$(( PER_DEV * NGPUS ))
if (( DENOM == 0 || EFF % DENOM != 0 )); then
  echo "EFF_BATCH=$EFF is not a multiple of PER_DEV=$PER_DEV x $NGPUS GPUs" >&2; exit 2
fi
ACC=$(( EFF / DENOM ))
echo "$NGPUS GPUs x per-device $PER_DEV x accumulation $ACC = effective batch $EFF"
FORCE_TORCHRUN=1 llamafactory-cli train "$CFG" \
  per_device_train_batch_size="$PER_DEV" gradient_accumulation_steps="$ACC" "${@:2}"
