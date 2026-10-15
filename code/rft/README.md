# RFT: rejection-sampling fine-tuning on the SFT model's own rollouts

Paper Section 5 (RFT) and Appendix C.2. The dev-selected SFT model samples eight SID-only
rollouts per prompt without seeing labels. A prompt contributes at most one training trace: its
first rollout that is a valid, completed top-10 ranking with a nonempty rationale, places the
single positive at rank one, and is not the identity order. The SFT LoRA is then trained further
on these traces. Unlike SFT, RFT applies no rationale length gate or answer-reveal audit, and no
reward term enters the loss.

| Setting | Value |
|---|---|
| prompts | the 6,000-prompt post-training subset (3,000 per positive grade), shared with OPD and RLVR |
| history length k | 32, 64, 128, 256 with probabilities .10, .30, .30, .30 |
| slate size N | 10, 20, 30, 50, 100 with probabilities .10, .15, .40, .20, .15 |
| sampling | 8 rollouts, temperature 1.0, top-p 0.95, at most 2,048 tokens, per-prompt seeds |
| initializer | SFT LoRA checkpoint-1035 (onerec-1.7b) or checkpoint-88 (onerec-8b) |
| training | LoRA rank 16, alpha 32, all linear layers; lr 5e-5, cosine, warmup .03, 2 epochs, effective batch 32 |
| selection | highest mean dev NDCG@5 at k64_n30 (400 dev users) |

| File | Role |
|---|---|
| `recipe.py` | sampling settings and the SFT initializer per model |
| `sample.py` | best-of-8 rollouts into `data/rft/<model>/rollouts.jsonl` |
| `build_dataset.py` | rank-1 gate, LLaMA-Factory `train.json`, `dataset_info.json`, `stats.json` |
| `configs/onerec-{1.7b,8b}.yaml` | LLaMA-Factory continuation configs |

The 6,000-prompt subset is selected in `code/utils/pools.py`.

## Data

The released training sets install to `data/rft/onerec-1.7b/` and `data/rft/onerec-8b/`, next to
the SFT dev set they use for loss monitoring:

```bash
python code/download/release.py --install
```

To regenerate them, build the post-training pool and sample from your own SFT adapter (trained
with `code/sft`) in `sid-verl`:

```bash
python code/sft/build_pool.py --pool posttrain       # data/posttrain/pool.parquet
python code/rft/sample.py --model onerec-1.7b --adapter saves/onerec-1.7b/sft/checkpoint-1035
python code/rft/build_dataset.py --model onerec-1.7b
```

`sample.py` is resumable (run it again to continue), and `--n 10` samples only the first ten
prompts. Without `--adapter` it loads the paper's checkpoint path. A retrained SFT adapter will not
reproduce the released traces exactly.

## Train

In `sid-lf`, after pointing `adapter_name_or_path` in the config at your SFT checkpoint:

```bash
bash code/sft/train.sh code/rft/configs/onerec-1.7b.yaml   # saves/onerec-1.7b/rft/checkpoint-*
bash code/sft/train.sh code/rft/configs/onerec-8b.yaml     # saves/onerec-8b/rft/checkpoint-*
```

## Select and evaluate

In `sid-verl`, score every checkpoint on the dev anchor, keep the best, and score it once on test:

```bash
MODEL=onerec-1.7b
LORAS=$(for d in $(ls -d saves/$MODEL/rft/checkpoint-* | sort -V); do
    printf 'step%s=%s,' "${d##*-}" "$d"; done)
python code/benchmark/score_sid.py --model $MODEL --method rft --slice dev --skip-base \
    --lora-dirs "${LORAS%,}"
eval "$(python code/utils/select_checkpoint.py results/posttraining/rft/$MODEL/dev --shell)"
python code/benchmark/score_sid.py --model $MODEL --method rft --skip-base \
    --lora-dirs "$BEST_NAME=saves/$MODEL/rft/checkpoint-$BEST_STEP"
```

Results go to `results/posttraining/rft/$MODEL/dev/step<n>.json` and
`results/posttraining/rft/$MODEL/test/k64_n30.json`, the layout of the released results. The
paper scored the RFT checkpoints in the `cu128` environment (vLLM 0.19.1; see
`code/envs/README.md`).

## Paper results

| Model | Retained prompts | Coverage | Selected checkpoint | Dev NDCG@5 | Test NDCG@5 |
|---|---:|---:|---:|---:|---:|
| `onerec-1.7b` | 1,940 | 32.3% | `step62` | .217 | .223 |
| `onerec-8b` | 1,912 | 31.9% | `step90` | .254 | .211 |

Released files: `results/posttraining/rft/<model>/{dev/step<n>,dev/init,test/k64_n30}.json`;
`dev/init.json` scores the SFT initializer.
