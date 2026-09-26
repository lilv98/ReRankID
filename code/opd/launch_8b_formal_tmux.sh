#!/usr/bin/env bash
# Launch the long 8B Pro formal OPD job in a detached tmux session.  The
# worker delegates every scientific and recovery check to run_opd.sh.  This
# wrapper only prevents a client or PTY lifetime from terminating training.
set -euo pipefail

RERANKID=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export OPD_STORAGE_ROOT=${OPD_STORAGE_ROOT:-/datadrive2/rerankid-opd}
export OPD_MERGE_ROOT=${OPD_MERGE_ROOT:-/datadrive2/rerankid_opd_merge_e128_lr5e5}
export OPD_SMOKE_ROOT=${OPD_SMOKE_ROOT:-/datadrive2/rerankid-opd-smoke}
for variable in OPD_STORAGE_ROOT OPD_MERGE_ROOT OPD_SMOKE_ROOT; do
    if [[ ${!variable} != /* ]]; then
        echo "ERROR: $variable must be an absolute path." >&2
        exit 2
    fi
done
OUTDIR=$OPD_STORAGE_ROOT/onerec-8b-pro/opd_v3_6k_e128_lr5e5
SESSION=rerankid-opd-8b-pro-e128-lr5e5-formal
STATUS=$OUTDIR/launcher.status
LOG=$OUTDIR/train.log

write_status() {
    local state=$1
    local exit_code=${2:-}
    local temporary
    temporary="${STATUS}.tmp.$$"
    {
        printf 'state=%s\n' "$state"
        printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'session=%s\n' "$SESSION"
        printf 'pid=%s\n' "$$"
        if [[ -n $exit_code ]]; then
            printf 'exit_code=%s\n' "$exit_code"
        fi
    } > "$temporary"
    mv -f -- "$temporary" "$STATUS"
}

if [[ ${1:-} == --worker ]]; then
    if (($# != 1)); then
        echo "usage: bash code/opd/launch_8b_formal_tmux.sh [--worker]" >&2
        exit 2
    fi
    mkdir -p "$OUTDIR"
    export SID_VERL_PYTHON=${SID_VERL_PYTHON:-/datadrive2/conda_envs/sid-verl/bin/python}
    export WANDB_ENTITY=${WANDB_ENTITY:-anonymous}
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
    write_status running
    set +e
    set -o pipefail
    bash "$RERANKID/code/opd/run_opd.sh" --formal --policy 8b-pro \
        2>&1 | tee -a "$LOG"
    return_code=${PIPESTATUS[0]}
    set -e
    if ((return_code == 0)); then
        write_status completed "$return_code"
    else
        write_status failed "$return_code"
    fi
    exit "$return_code"
fi

if (($# != 0)); then
    echo "usage: bash code/opd/launch_8b_formal_tmux.sh" >&2
    exit 2
fi
if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is required for the detached 8B Pro formal run." >&2
    exit 2
fi
if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session $SESSION already exists." >&2
    exit 2
fi

mkdir -p "$OUTDIR"
printf -v WORKER_COMMAND \
    'exec env OPD_STORAGE_ROOT=%q OPD_MERGE_ROOT=%q OPD_SMOKE_ROOT=%q SID_VERL_PYTHON=%q API_KEYS_FILE=%q WANDB_ENTITY=%q CUDA_VISIBLE_DEVICES=%q bash code/opd/launch_8b_formal_tmux.sh --worker' \
    "$OPD_STORAGE_ROOT" \
    "$OPD_MERGE_ROOT" \
    "$OPD_SMOKE_ROOT" \
    "${SID_VERL_PYTHON:-/datadrive2/conda_envs/sid-verl/bin/python}" \
    "${API_KEYS_FILE:-$RERANKID/api_keys.json}" \
    "${WANDB_ENTITY:-anonymous}" \
    "${CUDA_VISIBLE_DEVICES:-0,1}"
tmux new-session -d -s "$SESSION" -c "$RERANKID" \
    "$WORKER_COMMAND"

echo "launched_session=$SESSION"
echo "status=$STATUS"
echo "log=$LOG"
echo "inspect=tmux capture-pane -p -t =$SESSION -S -80"
