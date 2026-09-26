#!/usr/bin/env bash
# Fail-closed LLaMA-Factory launcher for the four retained SFT/RFT recipes.
# It enforces EFFECTIVE BATCH = per-device batch x accumulation x visible GPUs.
# Usage (from the repo root in conda env sid-lf):
#   EFF_BATCH=32 bash code/sft/train.sh code/sft/configs/lora_sft_1.7b_v3.yaml
#   EFF_BATCH=128 PER_DEV=4 bash code/sft/train.sh code/sft/configs/lora_sft_8b_v3.yaml
set -euo pipefail
ACTIVE_ENV=${CONDA_DEFAULT_ENV:-}
if [[ ${ACTIVE_ENV##*/} != sid-lf ]]; then
  echo "ERROR: activate conda environment sid-lf before running this launcher." >&2
  exit 2
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
set +xv  # keep credentials private even if the launcher was invoked with bash -x or bash -v
eval "$(python3 code/utils/config.py --shell-exports)"  # explicit credential handoff; keep xtrace off
CFG_INPUT=${1:?usage: train.sh <final-config.yaml>}; shift || true
if (($#)); then
  echo "ERROR: formal SFT/RFT training does not accept CLI config overrides." >&2
  exit 2
fi
CFG=$(realpath -m "$CFG_INPUT")
case "$CFG" in
  "$ROOT/code/sft/configs/lora_sft_1.7b_v3.yaml")
    EXPECTED_EFF=32; EXPECTED_PER_DEV=2 ;;
  "$ROOT/code/sft/configs/lora_sft_8b_v3.yaml")
    EXPECTED_EFF=128; EXPECTED_PER_DEV=4 ;;
  "$ROOT/code/rft/configs/lora_sft_1.7b_rft_v1_6k.yaml" | \
  "$ROOT/code/rft/configs/lora_sft_8b_pro_rft_v1_6k.yaml")
    EXPECTED_EFF=32; EXPECTED_PER_DEV=2 ;;
  *)
    echo "ERROR: config is not one of the four retained formal recipes: $CFG" >&2
    exit 2 ;;
esac
if [[ ${RERANKID_MACHINE_PROFILE:-} != cu130 ]]; then
  echo "ERROR: formal SFT/RFT training requires the cu130 training-host profile." >&2
  exit 2
fi
NGPUS=$(PYTHONPATH="$ROOT/code" python3 -c 'from utils.config import gpu_count; print(gpu_count())')
if (( NGPUS != 4 )); then
  echo "ERROR: formal SFT/RFT training requires exactly four visible GPUs." >&2
  exit 2
fi
PER_DEV=${PER_DEV:-$EXPECTED_PER_DEV}
EFF=${EFF_BATCH:-$EXPECTED_EFF}
if (( PER_DEV != EXPECTED_PER_DEV || EFF != EXPECTED_EFF )); then
  echo "ERROR: frozen recipe requires EFF_BATCH=$EXPECTED_EFF and PER_DEV=$EXPECTED_PER_DEV." >&2
  exit 2
fi
DENOM=$(( PER_DEV * NGPUS ))
if (( EFF < DENOM || EFF % DENOM != 0 )); then
  echo "ERROR: EFF_BATCH=$EFF must be a positive multiple of per_device($PER_DEV) x GPUs($NGPUS)=$DENOM." >&2
  exit 2
fi
ACC=$(( EFF / DENOM ))
echo "== $NGPUS GPUs x per_device $PER_DEV x accum $ACC = effective batch $(( NGPUS * PER_DEV * ACC ))"
FORCE_TORCHRUN=1 llamafactory-cli train "$CFG" \
  per_device_train_batch_size=$PER_DEV \
  gradient_accumulation_steps=$ACC
