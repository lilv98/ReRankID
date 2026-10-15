# RLVR: reinforcement learning with verifiable rewards

Paper Section 5 (RLVR), Appendix C.4, Table C.6 for the settings and Table C.7 for the dev curves.
Starting from the merged, dev-selected SFT model, GRPO trains a fresh LoRA adapter on the 6,000
shared post-training prompts. The model sees the SID-only scoring prompt and is rewarded only for
a valid response: a closed reasoning block (the text up to the last `</think>`) followed by a
single JSON array of ten distinct, in-range candidate indices.

| Method | Reward (`reward_rerank.py`) | Selected step, 1.7B / 8B |
|---|---|---|
| `rlvr-ndcg5` | `compute_score`: 1/log2(1+r) for the positive at rank r <= 5, else 0 | 60 / 100 |
| `rlvr-hit1` (ablation) | `compute_score_hit1`: 1 if the positive is ranked first, else 0 | 60 / 80 |

Both methods share every other setting (`run_grpo.sh`): 8 rollouts per prompt at temperature 1.0
and top-p 0.95, prompt batch 60, PPO mini-batch 30, one PPO epoch, learning rate 1e-5 with 3%
warmup, KL loss 0.001 (low_var_kl) against the merged SFT model, LoRA rank 16 and alpha 32 on all
linear layers, 6,144 / 2,048 / 8,192 prompt / response / model tokens, one epoch (100 optimizer
steps), checkpoints every 20 steps. The 400 benchmark dev users at k=64, N=30 are only monitored.

| File | Role |
|---|---|
| `recipe.py` | data sizes and the paths of one run (`--shell` prints them as exports) |
| `build_data.py` | training and monitoring sets |
| `reward_rerank.py` | both rewards |
| `run_grpo.sh` | verl launcher with every training setting |

Adapter export and checkpoint selection use `code/utils/export_lora.py` and
`code/utils/select_checkpoint.py`.

## Data

`python code/download/release.py --install` places the released `data/rlvr/{train,val}.parquet`
and `stats.json`. To rebuild them instead, build `data/posttrain/pool.parquet` as described in
`code/sft/README.md`, then run `python code/rlvr/build_data.py`.

The initializer is the dev-selected SFT checkpoint (`code/sft`), merged into its base model with
the top-level `rope_theta` that the verl and vLLM stack reads (`run_grpo.sh` checks it). Point
`adapter_name_or_path` in the merge config at that checkpoint, then in `sid-lf`:

```bash
eval "$(python code/utils/config.py --shell-exports)"
llamafactory-cli export code/sft/configs/merge-onerec-8b.yaml    # saves/onerec-8b/sft-merged
python code/utils/model_config.py saves/onerec-8b/sft-merged
```

## Train

In `sid-verl`:

```bash
bash code/rlvr/run_grpo.sh onerec-8b --reward ndcg5      # likewise onerec-1.7b, and --reward hit1
```

Checkpoints go to `saves/<model>/rlvr-<reward>/global_step_{20,...,100}`; running the same command
again resumes from the latest one. `NGPUS`, `WANDB_PROJECT` and two memory knobs are described at
the top of `run_grpo.sh`.

## Select and evaluate

Export the five checkpoints as LoRA adapters, score them and the initializer on the 400 dev users,
keep the best mean NDCG@5 (the earlier step on ties), and score only that adapter on test:

```bash
eval "$(python code/rlvr/recipe.py --model onerec-8b --reward ndcg5 --shell)"   # MODEL, EXPORT, ...
python code/utils/export_lora.py --run-dir "$OUTDIR" --step 20 40 60 80 100 --out "$EXPORT"
LORAS=$(for s in 20 40 60 80 100; do printf 'step%d=%s/step-%06d,' "$s" "$EXPORT" "$s"; done)
python code/benchmark/score_sid.py --model "$MODEL" --method rlvr-ndcg5 --slice dev \
    --lora-dirs "${LORAS%,}"
eval "$(python code/utils/select_checkpoint.py "$RESULTS/dev" --shell)"   # BEST_NAME, BEST_STEP
python code/benchmark/score_sid.py --model "$MODEL" --method rlvr-ndcg5 --skip-base \
    --lora-dirs "$BEST_NAME=$EXPORT/step-$(printf %06d "$BEST_STEP")"
```

`score_sid.py` defaults to the anchor cell (`--k 64 --n-cand 30`). The dev run writes
`$RESULTS/dev/step<s>.json` and `init.json` for the initializer, and the test run writes
`$RESULTS/test/k64_n30.json`. Use `--method rlvr-hit1` for the Hit@1 reward. For Figure 2, the
selected 8B `rlvr-ndcg5` adapter is also scored with `--k-grid 0,4,8,16,32,64,128,256` at
`--n-cand 30` and with `--k 64` at `--n-cand` 10, 20, 50 and 100.

## Paper results

| Method | Model | Selected checkpoint | Dev NDCG@5 (init) | Test NDCG@5 |
|---|---|---:|---:|---:|
| `rlvr-ndcg5` | `onerec-1.7b` | `step60` | .233 (.218) | .231 |
| `rlvr-ndcg5` | `onerec-8b` | `step100` | .301 (.230) | .277 |
| `rlvr-hit1` | `onerec-1.7b` | `step60` | .218 (.211) | .232 |
| `rlvr-hit1` | `onerec-8b` | `step80` | .259 (.214) | .232 |

Released files: `results/posttraining/rlvr-{ndcg5,hit1}/<model>/{dev/step<s>,dev/init,test/k64_n30}.json`.
Each method's initializer is rescored in its own dev run, and greedy vLLM outputs vary between runs,
so the two `init.json` files of one model differ slightly.
