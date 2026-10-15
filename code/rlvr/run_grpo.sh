#!/usr/bin/env bash
# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

# GRPO RLVR in verl (paper Sec. 5, App. C.4, Table C.6): a fresh LoRA on the merged SFT
# initializer, trained on data/rlvr with the NDCG@5 reward (rlvr-ndcg5) or the Hit@1 ablation
# (rlvr-hit1). Both rewards and both scales share every setting below: 8 rollouts per prompt at
# temperature 1.0 and top-p 0.95, prompt batch 60, PPO mini-batch 30, one PPO epoch, lr 1e-5 with
# 3% warmup, low-variance KL loss 0.001 against the initializer, LoRA rank 16 and alpha 32, one
# epoch (100 steps), a checkpoint every 20 steps. Paths come from code/rlvr/recipe.py.
#
#   bash code/rlvr/run_grpo.sh onerec-8b --reward ndcg5
#
# Running the same command again resumes from the latest checkpoint. Optional environment: NGPUS
# (default: all visible GPUs), WANDB_PROJECT (also log to Weights & Biases, with wandb installed),
# RLVR_VLLM_MEMORY_UTILIZATION (0.5) and RLVR_PPO_MAX_TOKENS_PER_GPU (16384), for memory only.
set -euo pipefail

usage="usage: bash code/rlvr/run_grpo.sh {onerec-1.7b|onerec-8b} [--reward ndcg5|hit1]"
[[ ${1:-} == -h || ${1:-} == --help ]] && { echo "$usage"; exit 0; }
[[ $# -ge 1 ]] || { echo "$usage" >&2; exit 2; }
MODEL_NAME=$1
shift
REWARD=ndcg5
while (( $# )); do
    case "$1" in
        --reward) REWARD=${2:?--reward needs ndcg5 or hit1}; shift 2 ;;
        *) echo "$usage" >&2; exit 2 ;;
    esac
done

RERANKID=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$RERANKID/code${PYTHONPATH:+:$PYTHONPATH}"
eval "$(python "$RERANKID/code/rlvr/recipe.py" --model "$MODEL_NAME" --reward "$REWARD" --shell)"
eval "$(python "$RERANKID/code/utils/config.py" --shell-exports)"

for path in "$MODEL/config.json" "$DATA_DIR/train.parquet" "$DATA_DIR/val.parquet"; do
    [[ -s $path ]] || { echo "ERROR: missing $path (see code/rlvr/README.md)" >&2; exit 2; }
done
python - "$MODEL/config.json" <<'EOF'
import json, sys
from utils.model_config import ROPE_THETA
if json.load(open(sys.argv[1])).get("rope_theta") != ROPE_THETA:
    sys.exit("ERROR: top-level rope_theta missing; run code/utils/model_config.py on the merged model")
EOF

NGPUS=${NGPUS:-$(python -c 'from utils.config import gpu_count; print(gpu_count())')}
(( NGPUS >= 1 )) || { echo "ERROR: no visible GPU" >&2; exit 2; }
LOGGER='["console"]'
if [[ -n ${WANDB_PROJECT:-} ]]; then
    LOGGER='["console","wandb"]'
fi
VLLM_MEMORY=${RLVR_VLLM_MEMORY_UTILIZATION:-0.5}
PPO_MAX_TOKENS=${RLVR_PPO_MAX_TOKENS_PER_GPU:-16384}
echo "model=$MODEL reward=$REWARD_FN gpus=$NGPUS outdir=$OUTDIR"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val.parquet" \
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKENS" \
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
    actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_MEMORY" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    custom_reward_function.path="$RERANKID/code/rlvr/reward_rerank.py" \
    custom_reward_function.name="$REWARD_FN" \
    trainer.critic_warmup=0 \
    trainer.logger="$LOGGER" \
    trainer.project_name="${WANDB_PROJECT:-ReRankID}" \
    trainer.experiment_name="$EXPERIMENT" \
    trainer.default_local_dir="$OUTDIR" \
    trainer.resume_mode=auto \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node="$NGPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 \
    trainer.max_actor_ckpt_to_keep=6
