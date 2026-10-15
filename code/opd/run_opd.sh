#!/usr/bin/env bash
# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

# On-policy distillation (paper Sec. 5). The OneRec base model (student, LoRA) samples one response
# per SID-only prompt on GPU 0; the frozen merged SFT model of the same scale (teacher) scores the
# same prompt and response prefix on GPU 1. The loss is the forward KL from the teacher's
# renormalized top-32 distribution, with no task reward and no reference KL. 6,000 prompts, batch 30,
# 200 steps, a checkpoint every 25 steps under saves/<model>/opd.
#
# Usage: bash code/opd/run_opd.sh onerec-1.7b|onerec-8b
# Inputs: data/opd/{train,val}.parquet and saves/<model>/sft-merged (see code/opd/README.md).
# Running the same command again resumes from the latest checkpoint. Optional environment:
# WANDB_PROJECT (also log to Weights & Biases, with wandb installed) and, for memory only,
# OPD_STUDENT_VLLM_MEMORY, OPD_TEACHER_VLLM_MEMORY and OPD_PPO_MAX_TOKENS_PER_GPU.
set -euo pipefail

USAGE="usage: bash code/opd/run_opd.sh onerec-1.7b|onerec-8b"
MODEL=${1:-}
if [[ $MODEL == -h || $MODEL == --help ]]; then
    echo "$USAGE"
    exit 0
fi
if [[ $# -ne 1 || ($MODEL != onerec-1.7b && $MODEL != onerec-8b) ]]; then
    echo "$USAGE" >&2
    exit 2
fi

RERANKID=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TEACHER=$RERANKID/saves/$MODEL/sft-merged
TRAIN=$RERANKID/data/opd/train.parquet
VAL=$RERANKID/data/opd/val.parquet
OUTDIR=$RERANKID/saves/$MODEL/opd

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
if [[ $(tr ',' '\n' <<< "$CUDA_VISIBLE_DEVICES" | grep -c .) != 2 ]]; then
    echo "ERROR: OPD needs exactly two visible GPUs (student and teacher)." >&2
    exit 2
fi

eval "$(python "$RERANKID/code/utils/config.py" --shell-exports)"
STUDENT=$(python "$RERANKID/code/utils/config.py" --model-path "$MODEL")
for path in "$STUDENT/config.json" "$TEACHER/config.json" "$TRAIN" "$VAL"; do
    if [[ ! -s $path ]]; then
        echo "ERROR: missing OPD input $path" >&2
        exit 2
    fi
done

# Per-scale memory settings. 8B also offloads actor parameters and syncs LoRA weights from CPU
# (verl_lora_sync_patch.py); neither changes data, objective, sampling, batch or learning rate.
if [[ $MODEL == onerec-8b ]]; then
    STUDENT_VLLM_MEMORY=${OPD_STUDENT_VLLM_MEMORY:-0.30}
    PPO_MAX_TOKENS=${OPD_PPO_MAX_TOKENS_PER_GPU:-4096}
    PARAM_OFFLOAD=True
    STUDENT_ENFORCE_EAGER=True
    export RERANKID_OPD_CPU_LAYERED_LORA=1
else
    STUDENT_VLLM_MEMORY=${OPD_STUDENT_VLLM_MEMORY:-0.45}
    PPO_MAX_TOKENS=${OPD_PPO_MAX_TOKENS_PER_GPU:-8192}
    PARAM_OFFLOAD=False
    STUDENT_ENFORCE_EAGER=False
    unset RERANKID_OPD_CPU_LAYERED_LORA
fi
TEACHER_VLLM_MEMORY=${OPD_TEACHER_VLLM_MEMORY:-0.60}

export RERANKID_OPD_NORMALIZE_TOPK=1
export PYTHONPATH="$RERANKID/code/opd/verl_overlay:$RERANKID/code"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

LOGGER='["console"]'
if [[ -n ${WANDB_PROJECT:-} ]]; then
    LOGGER='["console","wandb"]'
fi

RAY_ENV=++ray_kwargs.ray_init.runtime_env.env_vars
EXTRA=()
if [[ $MODEL == onerec-8b ]]; then
    EXTRA+=("$RAY_ENV.RERANKID_OPD_CPU_LAYERED_LORA=\"1\"")
fi

mkdir -p "$OUTDIR"
echo "model=$MODEL student=$STUDENT teacher=$TEACHER outdir=$OUTDIR gpus=$CUDA_VISIBLE_DEVICES"

exec python -m verl.trainer.main_ppo \
    "$RAY_ENV.RERANKID_OPD_NORMALIZE_TOPK=\"1\"" \
    "$RAY_ENV.PYTHONPATH=$PYTHONPATH" \
    "$RAY_ENV.HF_HUB_OFFLINE=\"1\"" \
    "$RAY_ENV.TRANSFORMERS_OFFLINE=\"1\"" \
    "$RAY_ENV.TOKENIZERS_PARALLELISM=\"false\"" \
    "++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=opd.verl_topk_patch.ray_worker_setup" \
    "${EXTRA[@]}" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.train_batch_size=30 \
    data.max_prompt_length=3072 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.shuffle=True \
    data.seed=17 \
    data.validation_shuffle=False \
    actor_rollout_ref.model.path="$STUDENT" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=5e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=30 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKENS" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload="$PARAM_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$STUDENT_VLLM_MEMORY" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.enforce_eager="$STUDENT_ENFORCE_EAGER" \
    model_engine=dp \
    distillation.enabled=True \
    distillation.nnodes=1 \
    distillation.n_gpus_per_node=1 \
    ++distillation.teacher_models.teacher_model.key=rerankid_opd_sid \
    ++distillation.teacher_models.teacher_model.model_path="$TEACHER" \
    ++distillation.teacher_models.teacher_model.num_replicas=1 \
    ++distillation.teacher_models.teacher_model.inference.name=vllm \
    ++distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1 \
    ++distillation.teacher_models.teacher_model.inference.max_num_seqs=64 \
    ++distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="$TEACHER_VLLM_MEMORY" \
    ++distillation.teacher_models.teacher_model.inference.max_model_len=4096 \
    ++distillation.teacher_models.teacher_model.inference.temperature=1.0 \
    distillation.distillation_loss.loss_mode=forward_kl_topk \
    distillation.distillation_loss.topk=32 \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=False \
    distillation.distillation_loss.loss_max_clamp=null \
    distillation.distillation_loss.log_prob_min_clamp=null \
    custom_reward_function.path="$RERANKID/code/opd/zero_reward.py" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger="$LOGGER" \
    trainer.project_name="${WANDB_PROJECT:-ReRankID}" \
    trainer.experiment_name="opd-$MODEL" \
    trainer.default_local_dir="$OUTDIR" \
    trainer.rollout_data_dir="$OUTDIR/rollouts" \
    trainer.val_before_train=False \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=200 \
    trainer.max_actor_ckpt_to_keep=8 \
    trainer.resume_mode=auto
