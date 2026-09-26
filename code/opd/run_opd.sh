#!/usr/bin/env bash
# Matched-scale, SID-only OPD.  A Pro Base student samples trajectories on
# GPU 0; the frozen merged, scale-matched Pro SFT-v3 teacher supplies a top-32
# distribution on the same prefixes on GPU 1.  The default policy remains
# 1.7B Pro.  The 8B Pro formal arm exports every saved LoRA synchronously and
# retains only the newest two full checkpoints on the configured storage
# filesystem.
set -euo pipefail

MODE=
POLICY=1.7b-pro
while (($#)); do
    case $1 in
        --smoke|--probe|--formal)
            if [[ -n $MODE ]]; then
                echo "ERROR: specify exactly one run mode." >&2
                exit 2
            fi
            MODE=$1
            shift
            ;;
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
if [[ -z $MODE ||
      ($POLICY != 1.7b-pro && $POLICY != 8b-pro) ]]; then
    echo "usage: bash code/opd/run_opd.sh {--smoke|--probe|--formal} [--policy {1.7b-pro|8b-pro}]" >&2
    exit 2
fi

RERANKID=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SID_VERL_PYTHON=${SID_VERL_PYTHON:-/datadrive2/conda_envs/sid-verl/bin/python}
if [[ ! -x $SID_VERL_PYTHON ]]; then
    echo "ERROR: sid-verl Python not found at $SID_VERL_PYTHON" >&2
    exit 2
fi

if [[ $MODE == --formal ]]; then
    OPD_STATUS=$(git -C "$RERANKID" status --porcelain=v1 -- \
        code/opd code/utils/config.py code/utils/model_config.py \
        tests/test_opd.py \
        tests/test_opd_checkpoint_archive.py \
        tests/test_opd_eval_protocol.py \
        tests/test_opd_wandb_resume.py)
    if [[ -n $OPD_STATUS ]]; then
        echo "ERROR: formal OPD code must be committed and unchanged:" >&2
        echo "$OPD_STATUS" >&2
        exit 2
    fi
fi

export OPD_STORAGE_ROOT=${OPD_STORAGE_ROOT:-/datadrive2/rerankid-opd}
export OPD_MERGE_ROOT=${OPD_MERGE_ROOT:-/datadrive2/rerankid_opd_merge_e128_lr5e5}
export OPD_SMOKE_ROOT=${OPD_SMOKE_ROOT:-/datadrive2/rerankid-opd-smoke}
for variable in OPD_STORAGE_ROOT OPD_MERGE_ROOT OPD_SMOKE_ROOT; do
    value=${!variable}
    if [[ $value != /* ]]; then
        echo "ERROR: $variable must be an absolute path." >&2
        exit 2
    fi
done

if [[ $POLICY == 1.7b-pro ]]; then
    STUDENT=$RERANKID/models/hub/models--OpenOneRec--OneRec-1.7B-pro/snapshots/5dc1b097ab8194f48f14730e5400a276a22f4ca1
    TEACHER=$RERANKID/saves/onerec-1.7b-pro/merged_sft_v3_12k_lr1e4
    EXPERIMENT=opd_v3_1.7b_pro_6k
    FORMAL_OUT=$RERANKID/saves/onerec-1.7b-pro/opd_v3_6k
    PROBE_OUT=$RERANKID/saves/onerec-1.7b-pro/opd_v3_probe50
    MAX_ACTOR_CKPTS=9
else
    STUDENT=$RERANKID/models/hub/models--OpenOneRec--OneRec-8B-Pro/snapshots/17b594d1e6b322752d939a769a07860f7e4a0261
    TEACHER=$RERANKID/saves/onerec-8b-pro/merged_sft_v3_e128_lr5e5
    EXPERIMENT=opd_v3_8b_pro_e128_lr5e5_6k
    FORMAL_OUT=$RERANKID/saves/onerec-8b-pro/opd_v3_6k_e128_lr5e5
    FORMAL_OUT_PHYSICAL=$OPD_STORAGE_ROOT/onerec-8b-pro/opd_v3_6k_e128_lr5e5
    PROBE_OUT=$RERANKID/saves/onerec-8b-pro/opd_v3_probe50_e128_lr5e5
    PROBE_OUT_PHYSICAL=$OPD_STORAGE_ROOT/onerec-8b-pro/opd_v3_probe50_e128_lr5e5
    MAX_ACTOR_CKPTS=2
fi
TRAIN=$RERANKID/data/opd_v1_6k/train.parquet
VAL=$RERANKID/data/opd_v1_6k/val.parquet
for path in "$STUDENT/config.json" "$TEACHER/config.json" "$TRAIN" "$VAL"; do
    if [[ ! -s $path ]]; then
        echo "ERROR: missing OPD input $path" >&2
        exit 2
    fi
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
IFS=',' read -r CUDA_DEVICE_0 CUDA_DEVICE_1 CUDA_DEVICE_EXTRA \
    <<< "$CUDA_VISIBLE_DEVICES"
if [[ -z $CUDA_DEVICE_0 || -z $CUDA_DEVICE_1 ||
      -n $CUDA_DEVICE_EXTRA || $CUDA_VISIBLE_DEVICES == *,*,* ||
      $CUDA_DEVICE_0 == "$CUDA_DEVICE_1" ||
      $CUDA_VISIBLE_DEVICES == *[[:space:]]* ]]; then
    echo "ERROR: OPD requires exactly two distinct visible GPU IDs." >&2
    exit 2
fi

set +xv
eval "$("$SID_VERL_PYTHON" "$RERANKID/code/utils/config.py" --shell-exports)"
export RERANKID_OPD_POLICY=$POLICY
export RERANKID_OPD_NORMALIZE_TOPK=1
if [[ $POLICY == 8b-pro ]]; then
    export RERANKID_OPD_CPU_LAYERED_LORA=1
else
    unset RERANKID_OPD_CPU_LAYERED_LORA
fi
unset RERANKID_OPD_CHECKPOINT_ARCHIVE RERANKID_OPD_EXPORT_DIR
unset RERANKID_OPD_MERGER_TMPDIR RERANKID_OPD_RAW_KEEP
export PYTHONPATH="$RERANKID/code/opd/verl_overlay:$RERANKID/code"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=ReRankID
EXPECTED_WANDB_ENTITY=${EXPECTED_WANDB_ENTITY:-${WANDB_ENTITY:-anonymous}}
if [[ ${WANDB_ENTITY:-$EXPECTED_WANDB_ENTITY} != "$EXPECTED_WANDB_ENTITY" ]]; then
    echo "ERROR: formal OPD W&B entity must be $EXPECTED_WANDB_ENTITY" >&2
    exit 2
fi
export WANDB_ENTITY=$EXPECTED_WANDB_ENTITY
export WANDB_RUN_GROUP=opd_v3_sid_matched
export WANDB_TAGS=opd,v3,"$POLICY",sid-only,forward-kl,top32,renormalized,seqmean
if [[ $MODE == --formal ]]; then
    TRAINING_COMMIT=$(git -C "$RERANKID" rev-parse HEAD)
    WANDB_RUN_ID=$(
        "$SID_VERL_PYTHON" -c \
            'from opd.provenance import formal_wandb_run_id; import sys; print(formal_wandb_run_id(sys.argv[1], sys.argv[2]))' \
            "$TRAINING_COMMIT" "$POLICY"
    )
    export WANDB_RUN_ID
    if [[ $POLICY == 8b-pro ]]; then
        export RERANKID_OPD_WANDB_RESUME_POLICY=checkpoint-rewind-v1
        unset WANDB_RESUME WANDB_RESUME_FROM
    else
        unset RERANKID_OPD_WANDB_RESUME_POLICY WANDB_RESUME_FROM
        export WANDB_RESUME=allow
    fi
else
    unset WANDB_RUN_ID WANDB_RESUME WANDB_RESUME_FROM
    unset RERANKID_OPD_WANDB_RESUME_POLICY
fi

if [[ $POLICY == 8b-pro ]]; then
    DEFAULT_STUDENT_VLLM_MEMORY=0.30
    DEFAULT_PPO_MAX_TOKENS=4096
    DEFAULT_PARAM_OFFLOAD=True
    DEFAULT_STUDENT_ENFORCE_EAGER=True
else
    DEFAULT_STUDENT_VLLM_MEMORY=0.45
    DEFAULT_PPO_MAX_TOKENS=8192
    DEFAULT_PARAM_OFFLOAD=False
    DEFAULT_STUDENT_ENFORCE_EAGER=False
fi
STUDENT_VLLM_MEMORY=${OPD_STUDENT_VLLM_MEMORY:-$DEFAULT_STUDENT_VLLM_MEMORY}
TEACHER_VLLM_MEMORY=${OPD_TEACHER_VLLM_MEMORY:-0.60}
PPO_MAX_TOKENS=${OPD_PPO_MAX_TOKENS_PER_GPU:-$DEFAULT_PPO_MAX_TOKENS}
PARAM_OFFLOAD=${OPD_PARAM_OFFLOAD:-$DEFAULT_PARAM_OFFLOAD}
STUDENT_ENFORCE_EAGER=${OPD_STUDENT_ENFORCE_EAGER:-$DEFAULT_STUDENT_ENFORCE_EAGER}
if [[ $PARAM_OFFLOAD != True && $PARAM_OFFLOAD != False ]]; then
    echo "ERROR: OPD_PARAM_OFFLOAD must be True or False." >&2
    exit 2
fi
if [[ $STUDENT_ENFORCE_EAGER != True &&
      $STUDENT_ENFORCE_EAGER != False ]]; then
    echo "ERROR: OPD_STUDENT_ENFORCE_EAGER must be True or False." >&2
    exit 2
fi

prepare_backed_outdir() {
    local logical=$1
    local physical=$2
    mkdir -p "$(dirname "$logical")" "$(dirname "$physical")" "$physical"
    if [[ -L $logical ]]; then
        if [[ $(readlink -f "$logical") != $(readlink -f "$physical") ]]; then
            echo "ERROR: $logical points outside the frozen backing directory." >&2
            exit 2
        fi
    elif [[ -e $logical ]]; then
        echo "ERROR: refusing non-symlink 8B output path $logical" >&2
        exit 2
    else
        ln -s "$physical" "$logical"
    fi
}

check_8b_formal_storage() {
    local merger_root=$1
    local backing_dir=$2
    local minimum_free=$((120 * 1024 * 1024 * 1024))
    mkdir -p "$merger_root"
    if [[ ! -w $backing_dir || ! -w $merger_root ]]; then
        echo "ERROR: 8B formal backing and merger directories must be writable." >&2
        exit 2
    fi
    local available
    available=$(df -B1 --output=avail "$backing_dir" | tail -n 1)
    if [[ ! $available =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ||
          ${available//[[:space:]]/} -lt $minimum_free ]]; then
        echo "ERROR: 8B formal training requires at least 120 GiB free on the backing filesystem." >&2
        exit 2
    fi
    local stale_merge
    stale_merge=$(find "$merger_root" -mindepth 1 -maxdepth 1 \
        -type d -name '.merge_step_*' -print -quit)
    if [[ -n $stale_merge ]]; then
        echo "ERROR: stale merger workspace must be inspected before training: $stale_merge" >&2
        exit 2
    fi
}

EXTRA=()
ARCHIVE_ARGS=()
if [[ $MODE == --smoke ]]; then
    if [[ $POLICY == 8b-pro ]]; then
        mkdir -p "$OPD_SMOKE_ROOT"
        OUTDIR=$(mktemp -d \
            "$OPD_SMOKE_ROOT/rerankid_opd_v3_${POLICY}_smoke.XXXXXX")
        # The real step-2 raw is an operational gate for the 8B merger and
        # compact-adapter loader before the much longer formal run.
        SAVE_FREQ=2
    else
        OUTDIR=$(mktemp -d \
            "/tmp/rerankid_opd_v3_${POLICY}_smoke.XXXXXX")
        SAVE_FREQ=-1
    fi
    EXPERIMENT=${EXPERIMENT}_smoke
    TRAIN_BATCH=50
    MINI_BATCH=50
    TOTAL_STEPS=2
    LOGGER='["console"]'
    RESUME_POLICY=disable
    EXTRA+=(
        "data.train_max_samples=100"
        "data.val_max_samples=20"
        "trainer.resume_mode=disable"
    )
elif [[ $MODE == --probe ]]; then
    OUTDIR=$PROBE_OUT
    if [[ -e $OUTDIR || -L $OUTDIR ]]; then
        echo "ERROR: refusing to overwrite probe output $OUTDIR" >&2
        exit 2
    fi
    if [[ $POLICY == 8b-pro ]]; then
        if [[ -d $PROBE_OUT_PHYSICAL &&
              -n $(find "$PROBE_OUT_PHYSICAL" -mindepth 1 -print -quit) ]]; then
            echo "ERROR: refusing nonempty 8B probe backing directory $PROBE_OUT_PHYSICAL" >&2
            exit 2
        fi
        prepare_backed_outdir "$OUTDIR" "$PROBE_OUT_PHYSICAL"
    fi
    EXPERIMENT=${EXPERIMENT}_probe50
    TRAIN_BATCH=30
    MINI_BATCH=30
    TOTAL_STEPS=50
    SAVE_FREQ=25
    LOGGER='["console"]'
    RESUME_POLICY=disable
    EXTRA+=(
        "data.train_max_samples=1500"
        "data.val_max_samples=20"
        "actor_rollout_ref.actor.optim.lr_warmup_steps=6"
        "trainer.resume_mode=disable"
    )
else
    OUTDIR=$FORMAL_OUT
    if [[ $POLICY == 8b-pro ]]; then
        prepare_backed_outdir "$OUTDIR" "$FORMAL_OUT_PHYSICAL"
        export RERANKID_OPD_CHECKPOINT_ARCHIVE=1
        export RERANKID_OPD_EXPORT_DIR=$RERANKID/saves/onerec-8b-pro/opd_v3_6k_e128_lr5e5_adapters
        export RERANKID_OPD_MERGER_TMPDIR=$OPD_MERGE_ROOT
        export RERANKID_OPD_RAW_KEEP=2
        check_8b_formal_storage \
            "$RERANKID_OPD_MERGER_TMPDIR" "$FORMAL_OUT_PHYSICAL"
        ARCHIVE_ARGS+=(
            "trainer.del_local_ckpt_after_load=False"
            "actor_rollout_ref.actor.checkpoint.async_save=False"
            "++ray_kwargs.ray_init.runtime_env.env_vars.RERANKID_OPD_CHECKPOINT_ARCHIVE=\"1\""
            "++ray_kwargs.ray_init.runtime_env.env_vars.RERANKID_OPD_EXPORT_DIR=$RERANKID_OPD_EXPORT_DIR"
            "++ray_kwargs.ray_init.runtime_env.env_vars.RERANKID_OPD_MERGER_TMPDIR=$RERANKID_OPD_MERGER_TMPDIR"
            "++ray_kwargs.ray_init.runtime_env.env_vars.RERANKID_OPD_RAW_KEEP=\"2\""
            "++ray_kwargs.ray_init.runtime_env.env_vars.RERANKID_OPD_POLICY=$POLICY"
        )
    fi
    TRAIN_BATCH=30
    MINI_BATCH=30
    TOTAL_STEPS=200
    SAVE_FREQ=25
    LOGGER='["console","wandb"]'
    RESUME_POLICY=auto
    EXTRA+=("trainer.resume_mode=auto")
fi

mkdir -p "$OUTDIR"
if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required to protect the OPD run directory." >&2
    exit 2
fi
exec {OPD_LOCK_FD}>"$OUTDIR/.run.lock"
if ! flock -n "$OPD_LOCK_FD"; then
    echo "ERROR: another OPD process owns $OUTDIR." >&2
    exit 2
fi

if [[ $MODE == --formal && $POLICY == 8b-pro &&
      -s $OUTDIR/latest_checkpointed_iteration.txt ]]; then
    EXISTING_MARKER=$(<"$OUTDIR/latest_checkpointed_iteration.txt")
    if [[ $EXISTING_MARKER == 200 ]]; then
        OPD_FINALIZE_RUN_DIR="$OUTDIR" \
        OPD_FINALIZE_TRAINING_COMMIT="$TRAINING_COMMIT" \
            "$SID_VERL_PYTHON" -c \
            'from opd.checkpoint_archive import CheckpointArchive
from opd.provenance import load_manifest
import os
run_dir = os.environ["OPD_FINALIZE_RUN_DIR"]
manifest = load_manifest(
    os.path.join(run_dir, "run_manifest.json"), policy="8b-pro"
)
expected = manifest["audit"]["repository"]["head"]
actual = os.environ["OPD_FINALIZE_TRAINING_COMMIT"]
if actual != expected:
    raise RuntimeError(
        f"finalizer commit {actual} != training commit {expected}"
    )
archive = CheckpointArchive(
    run_dir,
    os.environ["RERANKID_OPD_EXPORT_DIR"],
    merger_temp_root=os.environ["RERANKID_OPD_MERGER_TMPDIR"],
    raw_keep=int(os.environ["RERANKID_OPD_RAW_KEEP"]),
)
state = archive.reconcile_resume()
print(
    "8B OPD checkpoint archive finalized through step "
    + str(state.get("latest_marker"))
)'
        "$SID_VERL_PYTHON" "$RERANKID/code/opd/wandb_resume.py" \
            verify-final \
            --run-id "$WANDB_RUN_ID" \
            --entity "$WANDB_ENTITY" \
            --project "$WANDB_PROJECT" \
            --total-steps 200
        echo "8B OPD formal training was already committed at step 200; compact archive is complete."
        exit 0
    fi
fi

LORA_SYNC_ARGS=()
if [[ $POLICY == 8b-pro ]]; then
    LORA_SYNC_ARGS+=(
        "++ray_kwargs.ray_init.runtime_env.env_vars.RERANKID_OPD_CPU_LAYERED_LORA=\"1\""
    )
fi

TRAIN_ARGS=(
    "++ray_kwargs.ray_init.runtime_env.env_vars.RERANKID_OPD_NORMALIZE_TOPK=\"1\"" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$PYTHONPATH" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE=\"1\"" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE=\"1\"" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.TOKENIZERS_PARALLELISM=\"false\"" \
    "++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=opd.verl_topk_patch.ray_worker_setup" \
    "${LORA_SYNC_ARGS[@]}" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.train_batch_size="$TRAIN_BATCH" \
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
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH" \
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
    trainer.project_name=ReRankID \
    trainer.experiment_name="$EXPERIMENT" \
    trainer.default_local_dir="$OUTDIR" \
    trainer.rollout_data_dir="$OUTDIR/rollouts" \
    trainer.val_before_train=False \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPTS" \
    "${ARCHIVE_ARGS[@]}" \
    "${EXTRA[@]}"
)

AUDIT_TRAINING_ARGS=()
for argument in "${TRAIN_ARGS[@]}"; do
    AUDIT_TRAINING_ARGS+=(--training-arg "$argument")
done

"$SID_VERL_PYTHON" "$RERANKID/code/opd/audit.py" \
    --policy "$POLICY" \
    --output "$OUTDIR/preflight_audit.json" \
    --manifest "$OUTDIR/run_manifest.json" \
    --run-mode "${MODE#--}" \
    --resume-policy "$RESUME_POLICY" \
    "${AUDIT_TRAINING_ARGS[@]}"

if [[ $MODE == --formal && $POLICY == 8b-pro ]]; then
    WANDB_EXPORTS=$(
        "$SID_VERL_PYTHON" "$RERANKID/code/opd/wandb_resume.py" \
            prepare \
            --run-dir "$OUTDIR" \
            --run-id "$WANDB_RUN_ID" \
            --entity "$WANDB_ENTITY" \
            --project "$WANDB_PROJECT" \
            --total-steps "$TOTAL_STEPS" \
            --save-freq "$SAVE_FREQ"
    )
    eval "$WANDB_EXPORTS"
fi

echo "mode=$MODE env=sid-verl visible_gpus=$CUDA_VISIBLE_DEVICES"
echo "policy=$POLICY"
echo "student=$STUDENT"
echo "teacher=$TEACHER"
echo "teacher_input=the exact same SID-only prompt and student response prefix"
echo "objective=direct renormalized top-32 forward-KL; task_reward=off; reference_KL=off"
echo "optimization=lr=5e-5; loss_aggregation=seq-mean-token-mean"
echo "memory=student_vllm=$STUDENT_VLLM_MEMORY; ppo_max_tokens=$PPO_MAX_TOKENS; param_offload=$PARAM_OFFLOAD; student_enforce_eager=$STUDENT_ENFORCE_EAGER"
echo "batch=$TRAIN_BATCH steps=$TOTAL_STEPS outdir=$OUTDIR"

"$SID_VERL_PYTHON" -m verl.trainer.main_ppo "${TRAIN_ARGS[@]}"

if [[ $MODE == --formal && $POLICY == 8b-pro ]]; then
    "$SID_VERL_PYTHON" "$RERANKID/code/opd/wandb_resume.py" \
        verify-final \
        --run-id "$WANDB_RUN_ID" \
        --entity "$WANDB_ENTITY" \
        --project "$WANDB_PROJECT" \
        --total-steps "$TOTAL_STEPS"
fi
