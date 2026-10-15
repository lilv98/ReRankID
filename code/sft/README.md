# SFT: supervised fine-tuning on hindsight teacher traces

Paper Section 5 (SFT) and Appendix C.1. Gemini-3.6-Flash reads the caption rendering of a slate
together with its one observed positive and writes a short rationale and a top-10 ranking with
that positive first. The student, OneRec-1.7B-pro or OneRec-8B-Pro with LoRA, imitates the same
response from the item-matched SID-only prompt by next-token likelihood. The pool has 12,000 users
disjoint from the benchmark (6,000 per positive grade, one slate each, history k in
{32, 64, 128, 256}, slate size N in {10, 20, 30, 50, 100}). Quality gates keep 11,991 annotations,
split by user hash into 11,031 training and 960 development examples. The selected SFT checkpoint
also initializes RFT and RLVR and serves as the OPD teacher.

| File | Role |
|---|---|
| `recipe.py` | teacher settings and the hash-based train/dev split |
| `build_pool.py` | SFT pool and the shared post-training pool (`--pool sft`, `--pool posttrain`) |
| `annotate.py` | teacher annotation, resumable |
| `build_dataset.py` | quality gates and the LLaMA-Factory dataset |
| `train.sh` | LLaMA-Factory LoRA launcher for the SFT and RFT configs |
| `configs/onerec-{1.7b,8b}.yaml` | training recipes |
| `configs/merge-onerec-{1.7b,8b}.yaml` | merge the selected adapter into its base |

The pool grids and the post-training subset are in `code/utils/pools.py`.

## Data

The released dataset is all that training needs:

```bash
python code/download/release.py --install   # data/sft/{train,dev,dataset_info,stats}.json and more
```

Rebuilding it is optional. It needs the raw RecIF files in `data/recif/` (`code/download/recif.py`)
and `data/benchmark/pools.parquet`, and runs on CPU except for the teacher calls. With the released
`annotations.jsonl` in place, `annotate.py` finds every slate done and makes no calls, and
`build_dataset.py` reproduces the released `train.json` and `dev.json`.

```bash
python code/sft/build_pool.py --pool sft    # data/sft/pool.parquet
python code/sft/annotate.py                 # data/sft/annotations.jsonl (teacher calls)
python code/sft/build_dataset.py            # data/sft/{train,dev,dataset_info,stats}.json
```

`python code/sft/build_pool.py --pool posttrain` builds `data/posttrain/pool.parquet`, the prompt
pool of RFT, OPD and RLVR, which also excludes every SFT user. `data/sft/stats.json` holds the
numbers of Table C.1: pool and accepted distributions, rejections by gate (grounding 4, length 4,
correctness 1 in the paper), answer-reveal audit hits, rationale lengths, train/dev sizes and the
teacher settings.

## Train

Run in the `sid-lf` environment from the repository root. `train.sh` keeps the paper's effective
batch (32 for 1.7B, 128 for 8B) on any GPU count that divides it (for example 1, 2, 4 or 8) by
setting gradient accumulation.

```bash
bash code/sft/train.sh code/sft/configs/onerec-1.7b.yaml   # saves/onerec-1.7b/sft/checkpoint-*
bash code/sft/train.sh code/sft/configs/onerec-8b.yaml     # saves/onerec-8b/sft/checkpoint-*
```

## Select and evaluate

In `sid-verl`, score every checkpoint on the 400 dev users at the anchor (k=64, N=30), keep the
highest mean NDCG@5, and score that adapter once on the 400 test users:

```bash
MODEL=onerec-8b
LORAS=$(for d in $(ls -d saves/$MODEL/sft/checkpoint-* | sort -V); do
    printf 'step%s=%s,' "${d##*-}" "$d"; done)
python code/benchmark/score_sid.py --model $MODEL --method sft --slice dev --skip-base \
    --lora-dirs "${LORAS%,}"
eval "$(python code/utils/select_checkpoint.py results/posttraining/sft/$MODEL/dev --shell)"
python code/benchmark/score_sid.py --model $MODEL --method sft --skip-base \
    --lora-dirs "$BEST_NAME=saves/$MODEL/sft/checkpoint-$BEST_STEP"
```

`score_sid.py` defaults to the anchor cell (`--k 64 --n-cand 30`) and writes
`results/posttraining/sft/$MODEL/dev/step<n>.json` per checkpoint, then
`results/posttraining/sft/$MODEL/test/k64_n30.json`. The release labels SFT dev files by epoch, as
Table C.2 does: checkpoint-1035 of 1.7B is `dev/epoch3.0.json` and checkpoint-88 of 8B is
`dev/epoch1.01.json`.

## Merge for RLVR and OPD

Set `adapter_name_or_path` in the merge config to your selected checkpoint, then export the merged
model in `sid-lf` and restore its top-level `rope_theta`:

```bash
eval "$(python code/utils/config.py --shell-exports)"
llamafactory-cli export code/sft/configs/merge-onerec-8b.yaml   # saves/onerec-8b/sft-merged
python code/utils/model_config.py saves/onerec-8b/sft-merged
```

## Paper results

| Model | Selected checkpoint | Dev NDCG@5 | Test NDCG@5 |
|---|---:|---:|---:|
| `onerec-1.7b` | checkpoint-1035 (`epoch3.0`) | .220 | .225 |
| `onerec-8b` | checkpoint-88 (`epoch1.01`) | .238 | .237 |

Released files: `results/posttraining/sft/<model>/{dev/epoch<e>,test/k64_n30}.json`.
