# SFT v3 final runbook

This runbook reproduces the final supervised fine-tuning pipeline. SFT v3 uses one frozen
12,000-user, single-positive dataset and ordinary LoRA training. The only public SFT models are the
final 12k 1.7B Pro and 8B Pro runs.

Teacher annotation is resumable in two API stages. The deterministic 6k subset is an operational
quality gate before spending the remaining teacher budget. It is not a separate training arm,
model, or published dataset. The 12k model always starts from the corresponding base model.

## 1. Environments and source checkout

Run data generation and evaluation in `sid-verl`, and LoRA training in `sid-lf`.

```bash
cd /path/to/ReRankID
source /path/to/conda/etc/profile.d/conda.sh
conda activate sid-verl
export API_KEYS_FILE=/path/to/api_keys.json
eval "$(python code/utils/config.py --shell-exports)"
```

Vertex annotation additionally requires valid application-default credentials. W&B access is
needed only to publish or retrieve frozen artifacts. Never print credentials in logs.

Formal annotation records the Git commit and refuses a dirty checkout. Before the first teacher
call:

```bash
git status --short
test -z "$(git status --porcelain)"
```

Keep that exact commit and the same pool, manifest, prompt, and teacher settings throughout the
6k-to-12k annotation sequence.

Fetch the two base models at their frozen revisions. The shared verifier hashes `config.json`, the
exact weight-file set, every model-weight file, and the 8B shard index before evaluation:

```bash
hf download OpenOneRec/OneRec-1.7B-pro \
  --revision 5dc1b097ab8194f48f14730e5400a276a22f4ca1 \
  --cache-dir models/hub
hf download OpenOneRec/OneRec-8B-Pro \
  --revision 17b594d1e6b322752d939a769a07860f7e4a0261 \
  --cache-dir models/hub
python code/rft/base_provenance.py --policy 1.7b
python code/rft/base_provenance.py --policy 8b-pro
```

## 2. Frozen data and teacher contract

| Field | Final value |
|---|---|
| recipe | `sft-v3-single-longhist` |
| prompt | `sft-v3-single-longhist-20260721-v5` |
| parent pool | 12,000 users, one slate per user |
| positive grades | 6,000 grade-3 and 6,000 grade-2 |
| positives per slate | exactly one |
| history lengths | `32,64,128,256` with probabilities `.10,.30,.30,.30` |
| candidate counts | `10,20,30,50,100` with probabilities `.10,.15,.40,.20,.15` |
| teacher | `gemini-3.6-flash` through Vertex |
| teacher reasoning | effort/thinking `medium`; no explicit temperature |
| generation | `max_tokens=8192`, one format retry |
| teacher input | caption-rendered history and candidates |
| teacher supervision | the single behavior-verified positive index is visible |
| student input | SID-only history and candidates |
| target | teacher rationale followed by one top-10 index array |

The executable constants are in `code/sft/v3_contract.py`. Canonical files are:

```text
data/rerank/sft_pool_v3.parquet
data/rerank/sft_v3_phase_manifest.json
data/rerank/sft_v3_audit.json
data/rerank/sft_v3_6k_audit.json
data/rerank/sft_annotations_v3.jsonl
data/rerank/sft_annotations_v3_cost.json
data/sft_v3/train.json
data/sft_v3/dev.json
data/sft_v3/dataset_info.json
data/sft_v3/stats.json
```

## 3. Build and audit the frozen parent pool

```bash
conda activate sid-verl
python code/sft/build_pool.py --pool sft --recipe v3
python code/sft/audit_data.py --target-rows 12000
python code/sft/audit_data.py --target-rows 6000
```

Require 12,000 unique users, one positive per slate, a 50/50 positive-grade split, full `k` and `N`
support, unique renderable candidates, and no user overlap with benchmark or the reserved RL
registry.
The phase manifest deterministically freezes a balanced 6k subset inside the 12k parent.

`build_pool.py` refuses to overwrite an existing formal pool unless `--force-rebuild` is supplied.
Do not rebuild after annotation begins.

## 4. Ten-row Vertex smoke

Keep smoke files under a temporary path containing `smoke`; they are never published.

```bash
SMOKE=$(mktemp -d /tmp/rerankid_sft_v3_vertex_smoke.XXXXXX)
cp data/rerank/sft_pool_v3.parquet "$SMOKE/sft_pool_v3.parquet"
python code/sft/annotate.py \
  --backend vertex --model gemini-3.6-flash --effort medium \
  --thinking-level medium --max-tokens 8192 --format-retries 1 \
  --workers 2 --n 10 \
  --pool "$SMOKE/sft_pool_v3.parquet" \
  --out "$SMOKE/sft_annotations_v3.jsonl"
jq '{rows,valid_annotation,errors}' "$SMOKE/sft_annotations_v3_cost.json"
```

Inspect several complete rationales and require 10 valid annotations and zero errors before the
formal calls.

## 5. Resumable 6k-to-12k annotation

The first command annotates the deterministic 6k subset. Rerunning the identical command is safe:
valid matching rows are retained, a crash-truncated final line is repaired, and missing or invalid
rows are retried.

```bash
conda activate sid-verl
for attempt in 1 2 3 4 5; do
  python code/sft/annotate.py \
    --backend vertex --model gemini-3.6-flash --effort medium \
    --thinking-level medium --max-tokens 8192 --format-retries 1 \
    --workers 16 --target-rows 6000 \
    --pool data/rerank/sft_pool_v3.parquet \
    --out data/rerank/sft_annotations_v3.jsonl
  valid=$(jq -r .valid_annotation data/rerank/sft_annotations_v3_cost.json)
  errors=$(jq -r .errors data/rerank/sft_annotations_v3_cost.json)
  [ "$valid" = 6000 ] && [ "$errors" = 0 ] && break
done
```

At the 6k gate, require exactly 6,000 JSONL rows, zero errors, one shared provenance record, no
answer-reveal audit hits, and healthy rationale/format samples. Do not train or publish a 6k model.
Rebuild the deterministic quality-gate dataset once to exercise the same filters and split logic
used by the final assembly:

```bash
python code/sft/build_dataset.py \
  --target-rows 6000 \
  --pool data/rerank/sft_pool_v3.parquet \
  --annotations data/rerank/sft_annotations_v3.jsonl \
  --outdir data/sft_v3_6k
```

Inspect `data/sft_v3_6k/stats.json`, then continue annotation. This directory is a reproducible
quality check only. It has no training configuration and is not published.

Expand the same file under the same frozen provenance:

```bash
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = \
  "$(jq -r .provenance.git_commit data/rerank/sft_annotations_v3_cost.json)"

for attempt in 1 2 3 4 5; do
  python code/sft/annotate.py \
    --backend vertex --model gemini-3.6-flash --effort medium \
    --thinking-level medium --max-tokens 8192 --format-retries 1 \
    --workers 16 --target-rows 12000 \
    --pool data/rerank/sft_pool_v3.parquet \
    --out data/rerank/sft_annotations_v3.jsonl
  valid=$(jq -r .valid_annotation data/rerank/sft_annotations_v3_cost.json)
  errors=$(jq -r .errors data/rerank/sft_annotations_v3_cost.json)
  [ "$valid" = 12000 ] && [ "$errors" = 0 ] && break
done
```

This adds only the missing 6,000 manifest keys. It does not regenerate the completed first stage.

## 6. Assemble the only public training dataset

```bash
python code/sft/build_dataset.py \
  --target-rows 12000 \
  --pool data/rerank/sft_pool_v3.parquet \
  --annotations data/rerank/sft_annotations_v3.jsonl \
  --outdir data/sft_v3
```

Require `scope=full`, `source_pool_rows=12000`, `target_rows=12000`, disjoint train/dev users, the
fixed rationale-plus-ranking target, SID-only rendering, and exactly the two `sft_train` and
`sft_dev` registry entries. The builder exposes no alternative target or rendering modes.
Quality filtering may remove a small number of otherwise API-valid annotations; use `stats.json` as
the exact assembled count source.

## 7. Final LoRA recipes

Both models use LoRA rank 16, alpha 32, all linear layers, cosine scheduling, warmup ratio 0.08,
zero weight decay, bf16, and the complete `data/sft_v3` dataset. Their optimization settings differ
because the 8B Pro recipe is the dev-selected HPO winner.

| Field | 1.7B Pro | 8B Pro |
|---|---|---|
| base | `OpenOneRec/OneRec-1.7B-pro` | `OpenOneRec/OneRec-8B-Pro` |
| base revision | `5dc1b097ab8194f48f14730e5400a276a22f4ca1` | `17b594d1e6b322752d939a769a07860f7e4a0261` |
| cutoff | 8192 | 4096 |
| learning rate | `1e-4` | `5e-5` |
| effective batch | 32 | 128 |
| per-device batch in executed 4-GPU run | 2 | 4 |
| gradient accumulation in executed 4-GPU run | 4 | 8 |
| configured horizon | 3 epochs | 3 epochs |
| frozen dev-selected checkpoint | `checkpoint-1035` (epoch 3) | `checkpoint-88` (epoch 1.01) |
| output directory | `sft_v3_12k_lr1e4` | `hp_e128_lr5e5` |

Launch from the repository root:

```bash
conda activate sid-lf
EFF_BATCH=32 bash code/sft/train.sh \
  code/sft/configs/lora_sft_1.7b_v3.yaml

EFF_BATCH=128 PER_DEV=4 bash code/sft/train.sh \
  code/sft/configs/lora_sft_8b_v3.yaml
```

`train.sh` recomputes accumulation from the visible GPU count. For the executed 8B run,
`4 per device x 4 GPUs x 8 accumulation = 128`.

The 8B YAML deliberately keeps `num_train_epochs: 3.0`. Checkpoint 88 was selected from the
configured three-epoch trajectory using dev NDCG@5. Describing or reconfiguring this run as
one-epoch training would be inaccurate.

## 8. Dev selection and one final test sweep

Select checkpoints only on the 400-user dev anchor at `k=64`, `N=30`:

```bash
conda activate sid-verl
BASE_17=models/hub/models--OpenOneRec--OneRec-1.7B-pro/snapshots/5dc1b097ab8194f48f14730e5400a276a22f4ca1
BASE_8=models/hub/models--OpenOneRec--OneRec-8B-Pro/snapshots/17b594d1e6b322752d939a769a07860f7e4a0261
python code/rft/base_provenance.py --policy 1.7b
python code/rft/base_provenance.py --policy 8b-pro

select_dev() {
  local model=$1
  local save=$2
  local out=$3
  local loras
  mkdir -p "$out"
  loras=$(for checkpoint in \
    $(find "$save" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V); do
      printf '%s=%s,' "$(basename "$checkpoint")" "$checkpoint"
    done | sed 's/,$//')
  python code/benchmark/score_sid.py \
    --model "$model" --slice dev --no-profile --k 64 --n-cand 30 \
    --lora-dirs "$loras" --skip-base --tag sft_v3_dev_select --outdir "$out"
}

select_dev "$BASE_17" \
  saves/onerec-1.7b-pro/lora/sft_v3_12k_lr1e4 \
  results/sft-1.7b-pro-v3-12k/dev_select
select_dev "$BASE_8" \
  saves/onerec-8b-pro/lora/hp_e128_lr5e5 \
  results/sft-8b-pro-v3-12k/dev_select
```

The frozen winners are checkpoint 1035 for 1.7B Pro and checkpoint 88 for 8B Pro. Never use test
metrics to select a checkpoint.

Run the selected checkpoint once over the shared test sweep:

```bash
test_sweep() {
  local model=$1
  local lora=$2
  local out=$3
  mkdir -p "$out"
  python code/benchmark/score_sid.py --model "$model" --slice test --no-profile \
    --k-grid 0,4,8,16,32,64,128,256 --n-cand 30 \
    --lora-dirs "best=$lora" --skip-base --tag sft_v3_test --outdir "$out"
  for n in 10 20 50 100; do
    python code/benchmark/score_sid.py --model "$model" --slice test --no-profile \
      --k 64 --n-cand "$n" --lora-dirs "best=$lora" --skip-base \
      --tag sft_v3_test --outdir "$out"
  done
}

test_sweep "$BASE_17" \
  saves/onerec-1.7b-pro/lora/sft_v3_12k_lr1e4/checkpoint-1035 \
  results/sft-1.7b-pro-v3-12k
test_sweep "$BASE_8" \
  saves/onerec-8b-pro/lora/hp_e128_lr5e5/checkpoint-88 \
  results/sft-8b-pro-v3-12k
```

## 9. Verify frozen outputs

The final paper archive is already frozen as `data:v0`, `checkpoints:v0`, and
`results:v0`. A reproduction should validate its local files without recreating
the historical per-stage artifacts:

```bash
conda activate sid-verl
python code/validate_final_manifest.py
```

Do not overwrite any canonical `v0` artifact. If a later paper revision adopts
new outputs, stage and validate a complete replacement bundle before publishing
a new immutable version.

The frozen downstream initializer identities are:

| Model | Adapter artifact | Adapter SHA256 |
|---|---|---|
| 1.7B Pro | `checkpoints:v0/sft/1.7b-pro` | `b248d6f9bfefad88de9f0cef0531f86392e9f178661e4ebf716d596a9b5a242c` |
| 8B Pro | `checkpoints:v0/sft/8b-pro` | `d8a40d9f09d2c6a4246ef0a0709de465316145cdd6c4afd1136608cc669b63b9` |

Verify the base ID, checkpoint, and adapter hash before starting RFT, OPD, or RLVR. Do not resolve
an unversioned `latest` alias when reproducing a frozen experiment.
