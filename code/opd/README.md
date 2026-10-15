# OPD: on-policy distillation

On-policy distillation (paper Section 5, OPD rows of Table 2, App. C.3). The student is the
OneRec base model with a fresh LoRA. It samples one response per SID-only prompt, and the frozen
merged SFT model of the same scale scores the same prompt and response prefix. The update is the
forward KL from the teacher's top-32 distribution, renormalized over those 32 tokens, to the
student. No ranking label, task reward or reference KL enters training: `build_data.py` empties
the ground truth and `zero_reward.py` returns 0.

| Setting | Value (both scales) |
|---|---|
| student / teacher | `onerec-<scale>` base + LoRA / `saves/<model>/sft-merged` |
| data | 6,000 label-free prompts (`data/opd/train.parquet`), each used once |
| steps, batch | 200 steps of 30 prompts, checkpoint every 25 steps |
| rollout | n=1, temperature 1.0, top-p 1.0, 3,072 prompt / 512 response tokens |
| loss | renormalized teacher top-32 forward KL, `seq-mean-token-mean` |
| LoRA, optimizer | rank 16, alpha 32, all linear layers; lr 5e-5 constant, 3% warmup |

`run_opd.sh` holds the full verl configuration. `verl_topk_patch.py` adds the renormalization to
verl 0.8.0. For 8B, `verl_lora_sync_patch.py` and parameter offload keep the actor and the
rollout within one GPU; they change memory use only.

## Data

OPD runs on two GPUs (student and teacher) in a `cu128` verl environment (vLLM 0.19.1, verl
0.8.0). Build it under its own name, next to the default `cu130` environments:

```bash
RERANKID_VERL_ENV_NAME=sid-verl-cu128 bash code/envs/setup_envs.sh --profile cu128 verl
conda activate sid-verl-cu128 && python code/envs/verify_env.py --role verl --profile cu128 --imports
python code/download/release.py --install          # data/opd, data/benchmark, data/sft
```

`python code/opd/build_data.py` rebuilds `data/opd` from `data/rlvr` (optional).

The teacher is the dev-selected SFT model (`code/sft`; the paper used checkpoint-1035 for 1.7B and
checkpoint-88 for 8B). Point `adapter_name_or_path` in the merge config at your checkpoint, then
merge in `sid-lf` and fix the RoPE field:

```bash
conda activate sid-lf
MODEL=onerec-8b                                    # or onerec-1.7b
eval "$(python code/utils/config.py --shell-exports)"
llamafactory-cli export code/sft/configs/merge-$MODEL.yaml    # writes saves/$MODEL/sft-merged
python code/utils/model_config.py saves/$MODEL/sft-merged
```

## Train

```bash
conda activate sid-verl-cu128
mkdir -p saves/$MODEL/opd
nohup bash code/opd/run_opd.sh $MODEL > saves/$MODEL/opd/train.log 2>&1 &
```

Checkpoints go to `saves/$MODEL/opd/global_step_<s>`; running the same command again resumes
from the latest one. Set `WANDB_PROJECT` to also log to Weights & Biases. Each 8B checkpoint
stores the full FP32 actor (over 30 GB), so keep about 300 GB free or export and delete steps as
they finish.

## Select and evaluate

In `sid-verl-cu128`, export the eight checkpoints as LoRA adapters (`--cpu` avoids GPU memory
pressure for 8B), score them on the 400 dev users at k=64, N=30, and pick the highest mean NDCG@5
(earliest step on ties):

```bash
STEPS="25 50 75 100 125 150 175 200"
python code/utils/export_lora.py --run-dir saves/$MODEL/opd --out saves/$MODEL/opd/adapters \
    --step $STEPS --cpu
LORAS=$(for s in $STEPS; do printf 'step%d=saves/%s/opd/adapters/step-%06d,' $s $MODEL $s; done)
python code/benchmark/score_sid.py --model $MODEL --method opd --slice dev --tp 1 --skip-base \
    --lora-dirs "${LORAS%,}"
eval "$(python code/utils/select_checkpoint.py results/posttraining/opd/$MODEL/dev --shell)"
python code/benchmark/score_sid.py --model $MODEL --method opd --tp 1 --skip-base \
    --lora-dirs "$BEST_NAME=saves/$MODEL/opd/adapters/step-$(printf %06d $BEST_STEP)"
```

Scores go to `results/posttraining/opd/$MODEL/dev/step<s>.json` and
`results/posttraining/opd/$MODEL/test/k64_n30.json` (`score_sid.py` defaults to `--k 64
--n-cand 30`). The adapter is always loaded on the base model, never on the merged teacher. The
same selection runs on the released dev results:
`python code/utils/select_checkpoint.py downloads/rerankid/results/posttraining/opd/$MODEL/dev`.

## Paper results

| Model | Selected checkpoint | Dev NDCG@5 | Test NDCG@5 |
|---|---:|---:|---:|
| `onerec-1.7b` | `step75` | .211 | .195 |
| `onerec-8b` | `step50` | .266 | .210 |

Released files: `results/posttraining/opd/<model>/{dev/step<s>,test/k64_n30}.json`.
