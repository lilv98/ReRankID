#!/usr/bin/env bash
# Frozen GRPO RLVR launcher. The main run is SFT-init + NDCG@5, and the
# registered diagnostic changes only the reward to Hit@1.
# --smoke is isolated under /tmp, runs two steps, and never contacts W&B.
# Formal runs are create-once. Resume is disabled unless --resume is explicit.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: bash code/rl/run_grpo.sh {1.7b-pro|8b-pro} [--init sft] [--reward ndcg5|hit1] [--smoke|--resume]" >&2
    exit 2
fi
POLICY=$1
shift
INITIALIZER=sft
REWARD=ndcg5
SMOKE=0
RESUME=0
while (( $# )); do
    case "$1" in
        --init)
            [[ $# -ge 2 ]] || { echo "ERROR: --init requires sft" >&2; exit 2; }
            INITIALIZER=$2
            shift 2
            ;;
        --reward)
            [[ $# -ge 2 ]] || { echo "ERROR: --reward requires ndcg5 or hit1" >&2; exit 2; }
            REWARD=$2
            shift 2
            ;;
        --smoke)
            SMOKE=1
            shift
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        *)
            echo "ERROR: unknown argument $1; arbitrary Hydra overrides are disabled." >&2
            echo "Use RLVR_VLLM_MEMORY_UTILIZATION or RLVR_PPO_MAX_TOKENS_PER_GPU only for memory tuning." >&2
            exit 2
            ;;
    esac
done
if (( SMOKE && RESUME )); then
    echo "ERROR: smoke runs cannot resume." >&2
    exit 2
fi

ACTIVE_ENV=${CONDA_DEFAULT_ENV:-}
if [[ ${ACTIVE_ENV##*/} != sid-verl ]]; then
    echo "ERROR: activate conda environment sid-verl before running this launcher." >&2
    exit 2
fi

RERANKID=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$RERANKID/code${PYTHONPATH:+:$PYTHONPATH}"
if ! CONTRACT=$(
    python3 "$RERANKID/code/rl/contract.py" \
        --policy "$POLICY" --init "$INITIALIZER" --reward "$REWARD" --shell
); then
    exit 2
fi
eval "$CONTRACT"
MODEL=$RERANKID/$MODEL_REL
REWARD_PATH=$RERANKID/$REWARD_REL
OUTDIR=$RERANKID/$OUT_REL
EXP=$EXPERIMENT

for path in "$MODEL/config.json" "$RERANKID/data/rlvr_v1_6k/train.parquet" \
            "$RERANKID/data/rlvr_v1_6k/val.parquet" \
            "$RERANKID/data/rlvr_v1_6k/stats.json" "$REWARD_PATH"; do
    if [[ ! -s $path ]]; then
        echo "ERROR: missing required input $path" >&2
        exit 2
    fi
done

DETECTED_GPUS=$(PYTHONPATH="$RERANKID/code" python3 -c 'from utils.config import gpu_count; print(gpu_count())')
NGPUS=${NGPUS:-$DETECTED_GPUS}
if (( NGPUS < 1 )); then
    echo "ERROR: no GPU is visible; check CUDA_VISIBLE_DEVICES." >&2
    exit 2
fi

set +xv
eval "$(python3 "$RERANKID/code/utils/config.py" --shell-exports)"
RLVR_VLLM_MEMORY_UTILIZATION=${RLVR_VLLM_MEMORY_UTILIZATION:-0.5}
RLVR_PPO_MAX_TOKENS_PER_GPU=${RLVR_PPO_MAX_TOKENS_PER_GPU:-16384}

EXTRA=()
if (( SMOKE )); then
    OUTDIR=$(mktemp -d "/tmp/rerankid_rlvr_${POLICY}_${VARIANT}_smoke.XXXXXX")
    EXP="${EXP}_smoke"
    EXTRA+=(
        "data.train_max_samples=120"
        "data.val_max_samples=20"
        "trainer.total_training_steps=2"
        "trainer.save_freq=-1"
        "trainer.test_freq=1"
        'trainer.logger=["console"]'
        "trainer.resume_mode=disable"
    )
else
    if [[ ${RERANKID_MACHINE_PROFILE:-} != cu130 ]]; then
        echo "ERROR: formal RLVR requires RERANKID_MACHINE_PROFILE=cu130." >&2
        exit 2
    fi
    if (( DETECTED_GPUS != 4 || NGPUS != 4 )); then
        echo "ERROR: formal RLVR requires exactly four visible GPUs; detected=$DETECTED_GPUS NGPUS=$NGPUS." >&2
        exit 2
    fi
    python3 "$RERANKID/code/envs/verify_env.py" \
        --role verl --profile cu130 --imports --skip-llm-auth
    if [[ -n $(git -C "$RERANKID" status --porcelain --untracked-files=all) ]]; then
        echo "ERROR: formal RLVR requires a clean Git worktree." >&2
        exit 2
    fi
    if ! BRANCH=$(git -C "$RERANKID" symbolic-ref --quiet --short HEAD); then
        echo "ERROR: formal RLVR requires a named release branch, not detached HEAD." >&2
        exit 2
    fi
    if ! UPSTREAM=$(git -C "$RERANKID" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'); then
        echo "ERROR: formal RLVR branch $BRANCH has no remote upstream." >&2
        exit 2
    fi
    UPSTREAM_FULL=$(git -C "$RERANKID" rev-parse --symbolic-full-name '@{upstream}')
    if [[ $UPSTREAM_FULL != refs/remotes/* ]]; then
        echo "ERROR: formal RLVR upstream must be a remote-tracking ref, got $UPSTREAM_FULL." >&2
        exit 2
    fi
    read -r AHEAD BEHIND < <(
        git -C "$RERANKID" rev-list --left-right --count "HEAD...$UPSTREAM"
    )
    if (( AHEAD != 0 || BEHIND != 0 )); then
        echo "ERROR: formal RLVR requires HEAD and $UPSTREAM to match; ahead=$AHEAD behind=$BEHIND." >&2
        exit 2
    fi
    if (( RESUME )); then
        EXTRA+=("trainer.resume_mode=auto")
    else
        EXTRA+=("trainer.resume_mode=disable")
    fi
fi

python3 "$RERANKID/code/rl/audit_data.py" \
    --policy "$POLICY" --init "$INITIALIZER"

RUN_MODE=formal
(( SMOKE )) && RUN_MODE=smoke
RUN_CONTRACT_ARGS=(
    --policy "$POLICY"
    --init "$INITIALIZER"
    --reward "$REWARD"
    --outdir "$OUTDIR"
    --mode "$RUN_MODE"
    --gpus "$NGPUS"
    --vllm-memory-utilization "$RLVR_VLLM_MEMORY_UTILIZATION"
    --ppo-max-tokens-per-gpu "$RLVR_PPO_MAX_TOKENS_PER_GPU"
)
if [[ -n ${RERANKID_MACHINE_PROFILE:-} ]]; then
    RUN_CONTRACT_ARGS+=(--machine-profile "$RERANKID_MACHINE_PROFILE")
fi
(( RESUME )) && RUN_CONTRACT_ARGS+=(--resume)
python3 "$RERANKID/code/rl/run_contract.py" "${RUN_CONTRACT_ARGS[@]}"

echo "policy=$POLICY init=$INITIALIZER reward=$REWARD variant=$VARIANT env=sid-verl GPUs=$NGPUS smoke=$SMOKE resume=$RESUME experiment=$EXP"
echo "model=$MODEL"
echo "reward_file=$REWARD_PATH"
echo "outdir=$OUTDIR"
echo "vllm_memory_utilization=$RLVR_VLLM_MEMORY_UTILIZATION ppo_max_tokens_per_gpu=$RLVR_PPO_MAX_TOKENS_PER_GPU"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="$RERANKID/data/rlvr_v1_6k/train.parquet" \
    data.val_files="$RERANKID/data/rlvr_v1_6k/val.parquet" \
    data.train_batch_size=60 \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.shuffle=True \
    data.seed=17 \
    data.validation_shuffle=False \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.ppo_mini_batch_size=30 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$RLVR_PPO_MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$RLVR_VLLM_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    custom_reward_function.path="$REWARD_PATH" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=ReRankID \
    trainer.experiment_name="$EXP" \
    trainer.default_local_dir="$OUTDIR" \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 \
    trainer.max_actor_ckpt_to_keep=6 \
    "${EXTRA[@]}"
