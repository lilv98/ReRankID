#!/usr/bin/env bash
# Prepare every immutable input needed by matched-scale SID-only OPD.
#
# This script performs networked artifact/model preparation only. Training
# remains in run_opd.sh and always starts a fresh LoRA from the matched Pro
# Base model.
set -euo pipefail

POLICY=all
while (($#)); do
    case $1 in
        --policy)
            if (($# < 2)); then
                echo "ERROR: --policy requires a value." >&2
                exit 2
            fi
            POLICY=$2
            shift 2
            ;;
        *)
            echo "ERROR: unknown argument $1" >&2
            exit 2
            ;;
    esac
done
if [[ $POLICY != all && $POLICY != 1.7b-pro && $POLICY != 8b-pro ]]; then
    echo "usage: bash code/opd/prepare.sh [--policy {all|1.7b-pro|8b-pro}]" >&2
    exit 2
fi

RERANKID=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SID_LF_PYTHON=${SID_LF_PYTHON:-/datadrive2/conda_envs/sid-lf/bin/python}
SID_VERL_PYTHON=${SID_VERL_PYTHON:-/datadrive2/conda_envs/sid-verl/bin/python}
LLAMAFACTORY_CLI=${LLAMAFACTORY_CLI:-$(dirname "$SID_LF_PYTHON")/llamafactory-cli}
for executable in "$SID_LF_PYTHON" "$SID_VERL_PYTHON" "$LLAMAFACTORY_CLI"; do
    if [[ ! -x $executable ]]; then
        echo "ERROR: required executable not found: $executable" >&2
        exit 2
    fi
done

cd "$RERANKID"
set +xv
eval "$("$SID_LF_PYTHON" code/utils/config.py --shell-exports)"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

# The shared post-training artifact supplies the frozen 6k prompt pool,
# benchmark, and the exact final SFT-v3 adapter at the requested scale.
"$SID_LF_PYTHON" code/rl/fetch_inputs.py --policy "$POLICY"

# OPD removes every usable ranking label while preserving prompt bytes and
# row order. The builder is deterministic and freezes output hashes.
"$SID_VERL_PYTHON" code/opd/build_data.py

if [[ $POLICY == all ]]; then
    POLICIES=(1.7b-pro 8b-pro)
else
    POLICIES=("$POLICY")
fi

for current in "${POLICIES[@]}"; do
    if [[ $current == 1.7b-pro ]]; then
        MERGE_CONFIG=code/rl/merge_sft_v3_1.7b_pro.yaml
        TEACHER=saves/onerec-1.7b-pro/merged_sft_v3_12k_lr1e4
    else
        MERGE_CONFIG=code/rl/merge_sft_v3_8b_pro.yaml
        TEACHER=saves/onerec-8b-pro/merged_sft_v3_e128_lr5e5
    fi

    if [[ ! -s $TEACHER/config.json ]]; then
        "$LLAMAFACTORY_CLI" export "$MERGE_CONFIG"
    else
        echo "reusing existing merged teacher: $TEACHER"
    fi
    "$SID_VERL_PYTHON" code/opd/fix_merged_config.py --policy "$current"
done

"$SID_VERL_PYTHON" code/opd/verify_inputs.py --policy "$POLICY"
