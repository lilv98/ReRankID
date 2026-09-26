# RFT v1 final runbook

This runbook reproduces the final rejection fine-tuning comparison for 1.7B Pro and 8B Pro. Both
arms use the same deterministic 6,000-prompt subset, the same best-of-8 rejection rule, and the
same continuation hyperparameters. There is no 12k RFT training arm.

RFT v1 is ordinary offline self-training. A frozen SFT policy samples eight responses for each
unlabeled SID-only prompt. The builder retains at most one response per prompt: the first completed,
nonidentity response with a valid top-10 ranking and the single positive item at rank one. The model
does not receive the gold label during generation.

## 1. Frozen symmetric contract

| Field | Final value |
|---|---|
| recipe | `rft-v1-rank1-bestof8` |
| prompt source | deterministic 6k subset of the shared 12k post-training pool |
| positive grades | 3,000 grade-3 and 3,000 grade-2 prompts |
| history lengths | `32,64,128,256` with probabilities `.10,.30,.30,.30` |
| candidate counts | `10,20,30,50,100` with probabilities `.10,.15,.40,.20,.15` |
| rollouts per prompt | 8 |
| sampling | temperature `1.0`, top-p `0.95`, max 2048 tokens |
| retention | first valid, completed, nonidentity, rank-1 response |
| examples per prompt | at most one |
| continuation | LoRA rank 16, alpha 32, all linear layers |
| optimization | lr `5e-5`, cosine, warmup `.03`, 2 epochs, effective batch 32 |
| dev selection | maximum mean NDCG@5 on 400 users at `k=64`, `N=30` |
| test | selected checkpoint only, 400-user `k` and `N` sweeps |

There is no leakage regex, rationale-length heuristic, teacher replay, or task reward in the
training loss. An output is rejected only for invalid or unclosed format, length truncation, empty
rationale, the positive item below rank one, or identity collapse.

The two model contracts differ only in scale and their matched SFT initializer:

| Arm | Base | Revision | Frozen SFT initializer | Adapter SHA256 |
|---|---|---|---|---|
| 1.7B Pro | `OpenOneRec/OneRec-1.7B-pro` | `5dc1b097ab8194f48f14730e5400a276a22f4ca1` | `checkpoints:v0/sft/1.7b-pro`, checkpoint 1035 | `b248d6f9bfefad88de9f0cef0531f86392e9f178661e4ebf716d596a9b5a242c` |
| 8B Pro | `OpenOneRec/OneRec-8B-Pro` | `17b594d1e6b322752d939a769a07860f7e4a0261` | `checkpoints:v0/sft/8b-pro`, checkpoint 88 | `d8a40d9f09d2c6a4246ef0a0709de465316145cdd6c4afd1136608cc669b63b9` |

The canonical base-weight identities are
`5e1711ee090fb25c11abb6b0467f5c95de8170992b4b21b72b2091fa4e8a0e9d` for
1.7B Pro and `0249cd2ec9af8b70441a9ffbfa7b5ec157a59dd55f950bf0d0d3d1009e007352`
for the complete three-shard 8B Pro bundle.

Do not substitute the non-Pro 8B checkpoint.

## 2. Environment and immutable inputs

Run pool construction, rollout, audit, and scoring in `sid-verl`. Run LLaMA-Factory continuation
in `sid-lf`.

```bash
cd /path/to/ReRankID
source /path/to/conda/etc/profile.d/conda.sh
conda activate sid-verl
export API_KEYS_FILE=/path/to/api_keys.json
eval "$(python code/utils/config.py --shell-exports)"
```

Fetch and hash-check the shared SFT manifest/dev set plus both exact initializers:

```bash
python code/rft/fetch_inputs.py --policy 1.7b
python code/rft/fetch_inputs.py --policy 8b-pro

hf download OpenOneRec/OneRec-1.7B-pro \
  --revision 5dc1b097ab8194f48f14730e5400a276a22f4ca1 \
  --cache-dir models/hub
hf download OpenOneRec/OneRec-8B-Pro \
  --revision 17b594d1e6b322752d939a769a07860f7e4a0261 \
  --cache-dir models/hub

python code/rft/base_provenance.py --policy 1.7b
python code/rft/base_provenance.py --policy 8b-pro
```

The artifact versions, base revisions, local snapshot paths, and SHA256 values are frozen in
`code/rft/contract.py`. The verifier checks `config.json`, the exact weight-file set, every weight
digest, the 8B index, and the 8B logical tensor size. Formal sampling loads only the verified local
snapshot and records that complete identity in every trajectory. It also requires a clean Git
checkout so each trajectory records immutable code provenance.

## 3. Build and audit the shared prompt subset

Build the 12k parent once, then audit the parent and the only RFT training subset:

```bash
python -u code/sft/build_pool.py --pool posttrain
python code/rft/audit_data.py --target-rows 12000
python code/rft/audit_data.py --target-rows 6000 \
  --out data/rerank/posttrain_v1_6k_audit.json
```

The parent makes user exclusion and deterministic subset construction auditable. RFT consumes only
the nested 6k subset. Require 6,000 unique users, one positive per slate, 3,000 prompts per positive
grade, full `k` and `N` support, unique renderable candidates, and zero overlap with SFT, dev, test,
or the reserved RL registry. The 12k stage remains available only to the pool auditor;
`sample.py` and `build_dataset.py` reject every RFT target other than 6,000.

Commit the exact implementation before formal rollout:

```bash
test -z "$(git status --porcelain)"
```

## 4. Ten-prompt smoke for each policy

Smoke outputs remain under `/tmp` and are never uploaded:

```bash
for policy in 1.7b 8b-pro; do
  smoke=$(mktemp -d "/tmp/rerankid_rft_${policy}_smoke.XXXXXX")
  python -u code/rft/sample.py \
    --policy "$policy" --target-rows 6000 --n 10 \
    --out "$smoke/samples.jsonl" \
    --stats-out "$smoke/stats.json"
  jq '{completed_prompts,rollouts,valid_rate,rank1_rate_among_valid,prompt_coverage}' \
    "$smoke/stats.json"
done
```

Read several complete responses. Ten prompts validate execution and format only; they are not
large enough to estimate retention or ranking quality.

## 5. Formal best-of-8 sampling and dataset assembly

Run each matched policy on the same 6,000 prompt keys:

```bash
test -z "$(git status --porcelain)"

python -u code/rft/sample.py --policy 1.7b --target-rows 6000
python code/rft/build_dataset.py --policy 1.7b --target-rows 6000

python -u code/rft/sample.py --policy 8b-pro --target-rows 6000
python code/rft/build_dataset.py --policy 8b-pro --target-rows 6000
```

`sample.py` appends one record per completed prompt, flushes every batch, and resumes only when the
prompt keys and complete provenance match. Rerun the identical command after an interruption.
Require each stats file to report 6,000 completed prompts and 48,000 rollouts.

`build_dataset.py` accepts only the sealed raw-record schema emitted by this version of
`sample.py`. For every prompt it requires exactly eight samples with `sample_index` equal to
`0,...,7`, verifies the record digest, and reparses the verbatim rollout text to check the stored
rationale, ranking, diagnostics, score, and selection flags. It also checks that completion status
and output-token counts are consistent with the frozen 2,048-token limit. A truncated JSONL or an
older or edited raw-record schema is rejected rather than repaired during dataset assembly.

The final observed assembly checks were:

| Arm | Valid rollout rate | Retained prompts | Prompt coverage |
|---|---:|---:|---:|
| 1.7B Pro | 0.9820 | 1,940 | 0.3233 |
| 8B Pro | 0.9839 | 1,912 | 0.3187 |

Do not relax the rank-1 gate to equalize retained counts. Audit coverage by history length,
candidate count, positive grade, and their cross-product before training.

## 6. Train the two matched 6k arms

```bash
conda activate sid-lf

EFF_BATCH=32 bash code/sft/train.sh \
  code/rft/configs/lora_sft_1.7b_rft_v1_6k.yaml

EFF_BATCH=32 bash code/sft/train.sh \
  code/rft/configs/lora_sft_8b_pro_rft_v1_6k.yaml
```

Both runs continue their matched SFT LoRA with a fresh optimizer. They do not continue one another
and do not replay SFT teacher examples. The unchanged SFT dev set is registered for loss
diagnostics, but checkpoint selection uses benchmark dev NDCG@5.

Expected checkpoint directories are:

```text
saves/onerec-1.7b-pro/lora/rft_v1_6k/checkpoint-{31,62,93,122}
saves/onerec-8b-pro/lora/rft_v1_8b_pro_e128_lr5e5_6k/checkpoint-{30,60,90,120}
```

## 7. Dev selection

Score every continuation checkpoint on the same 400-user dev anchor:

```bash
conda activate sid-verl

BASE_17=models/hub/models--OpenOneRec--OneRec-1.7B-pro/snapshots/5dc1b097ab8194f48f14730e5400a276a22f4ca1
BASE_8=models/hub/models--OpenOneRec--OneRec-8B-Pro/snapshots/17b594d1e6b322752d939a769a07860f7e4a0261
python code/rft/base_provenance.py --policy 1.7b
python code/rft/base_provenance.py --policy 8b-pro

OUT=results/rft-1.7b-pro-v1-6k
SAVE=saves/onerec-1.7b-pro/lora/rft_v1_6k
mkdir -p "$OUT/dev_select"
LORA=$(for checkpoint in $(find "$SAVE" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V); do
  printf '%s=%s,' "$(basename "$checkpoint")" "$checkpoint"
done | sed 's/,$//')
python code/benchmark/score_sid.py \
  --model "$BASE_17" --slice dev --no-profile \
  --k 64 --n-cand 30 --lora-dirs "$LORA" --skip-base \
  --tag rft_v1_dev_select --outdir "$OUT/dev_select"

OUT=results/rft-8b-pro-v1-6k-e128-lr5e5
SAVE=saves/onerec-8b-pro/lora/rft_v1_8b_pro_e128_lr5e5_6k
mkdir -p "$OUT/dev_select"
LORA=$(for checkpoint in $(find "$SAVE" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V); do
  printf '%s=%s,' "$(basename "$checkpoint")" "$checkpoint"
done | sed 's/,$//')
python code/benchmark/score_sid.py \
  --model "$BASE_8" --slice dev --no-profile \
  --k 64 --n-cand 30 --lora-dirs "$LORA" --skip-base \
  --tag rft_v1_8b_pro_e128_lr5e5_dev_select \
  --outdir "$OUT/dev_select"
```

The frozen winners are checkpoint 62 for 1.7B Pro and checkpoint 90 for 8B Pro. Selection uses the
maximum unrounded per-user mean dev NDCG@5; an exact tie goes to the earlier step.

## 8. One test sweep per selected checkpoint

```bash
test_sweep() {
  local model=$1
  local lora=$2
  local out=$3
  local tag=$4
  mkdir -p "$out"
  python code/benchmark/score_sid.py --model "$model" --slice test --no-profile \
    --k-grid 0,4,8,16,32,64,128,256 --n-cand 30 \
    --lora-dirs "rft=$lora" --skip-base --tag "$tag" --outdir "$out"
  for n in 10 20 50 100; do
    python code/benchmark/score_sid.py --model "$model" --slice test --no-profile \
      --k 64 --n-cand "$n" --lora-dirs "rft=$lora" --skip-base \
      --tag "$tag" --outdir "$out"
  done
}

BASE_17=models/hub/models--OpenOneRec--OneRec-1.7B-pro/snapshots/5dc1b097ab8194f48f14730e5400a276a22f4ca1
BASE_8=models/hub/models--OpenOneRec--OneRec-8B-Pro/snapshots/17b594d1e6b322752d939a769a07860f7e4a0261
python code/rft/base_provenance.py --policy 1.7b
python code/rft/base_provenance.py --policy 8b-pro

test_sweep "$BASE_17" \
  saves/onerec-1.7b-pro/lora/rft_v1_6k/checkpoint-62 \
  results/rft-1.7b-pro-v1-6k/test_sweep \
  rft_v1_test_sweep
test_sweep "$BASE_8" \
  saves/onerec-8b-pro/lora/rft_v1_8b_pro_e128_lr5e5_6k/checkpoint-90 \
  results/rft-8b-pro-v1-6k-e128-lr5e5/test_sweep \
  rft_v1_8b_pro_e128_lr5e5_test_sweep
```

Do not run test on nonselected checkpoints. Keep smoke, dev-selection, and test outputs in separate
directories, and publish only complete formal artifacts.
