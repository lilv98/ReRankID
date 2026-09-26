# RLVR runbook

This runbook reproduces the final GRPO experiments for both Pro model sizes.
It is the authoritative operational guide for `code/rl/`. The scientific
contract lives in `code/rl/contract.py`.

## 1. Experimental cells

All cells use the same SID-only prompt, 6,000 training users, 400-user
benchmark dev slice, optimizer, rollout budget, and checkpoint schedule.

| Initializer | Training reward | Role | Launcher flags |
|---|---|---|---|
| SFT v3 | NDCG@5 | primary RLVR | `--init sft --reward ndcg5` |
| SFT v3 | Hit@1 | reward ablation | `--init sft --reward hit1` |

Both cells start from the same dev-selected SFT-v3 initializer. The only
ablation changes the training reward.

The two sizes use these frozen initializers:

| Policy | Base revision | Base config SHA256 | SFT checkpoint | Merged config SHA256 |
|---|---|---|---|---|
| 1.7B Pro | `5dc1b097ab8194f48f14730e5400a276a22f4ca1` | `43b8f3bd...72b94f` | `checkpoint-1035` | `f158bd9b...9b7c1e` |
| 8B Pro | `17b594d1e6b322752d939a769a07860f7e4a0261` | `7cfca638...28a30` | `checkpoint-88` from the batch-128, lr-5e-5 HPO run | `a66d3a1e...66b33` |

The complete hashes, including the SFT adapter and adapter-config identities,
live in `code/rl/contract.py`; the shortened display above is for readability.

## 2. Shared training contract

| Setting | Value |
|---|---:|
| train prompts | 6,000 |
| in-loop validation | benchmark dev, k=64, N=30, 400 users |
| epochs / optimizer steps | 1 / 100 |
| prompt batch / PPO mini-batch | 60 / 30 |
| rollouts per prompt | 8 |
| temperature / top-p | 1.0 / 0.95 |
| learning rate / warmup | `1e-5` / 0.03 |
| LoRA | rank 16, alpha 32, all-linear |
| KL loss | 0.001, `low_var_kl`, against the initializer |
| PPO epochs | 1 |
| prompt / response / model length | 6,144 / 2,048 / 8,192 |
| checkpoints | 20, 40, 60, 80, 100 |
| selection metric | benchmark-dev NDCG@5 |

The NDCG@5 verifier awards `1/log2(1+rank)` for the single positive at ranks
1 through 5 and zero otherwise. The Hit@1 verifier awards one only at rank 1.
Both require a closed reasoning block and exactly ten distinct in-range
indices. Both log NDCG@5, NDCG@10, Hit@1, rank, identity, and format
diagnostics.

Checkpoint selection always uses benchmark-dev NDCG@5, including for the
Hit@1 training-reward arm. The 400-user test set is evaluated once, after the
checkpoint has been selected.

## 3. Training-host setup

The recorded training host uses:

```text
repo:        /mnt/data/workspace/ReRankID
credentials: /mnt/data/api_keys.json
environment: sid-verl for data, GRPO, export, and evaluation
             sid-lf for merging the SFT adapters
GPU host:    4 x A100-80GB
```

Start a shell with:

```bash
cd /mnt/data/workspace/ReRankID
source ~/miniconda3/etc/profile.d/conda.sh
export API_KEYS_FILE=/mnt/data/api_keys.json
export WANDB_ENTITY=<your-wandb-entity>
export RERANKID_MACHINE_PROFILE=cu130
set +xv
```

Before a formal run:

```bash
git fetch -q --all --prune
test -z "$(git status --porcelain)"
BRANCH=$(git symbolic-ref --quiet --short HEAD)
UPSTREAM=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')
case "$(git rev-parse --symbolic-full-name '@{upstream}')" in
  refs/remotes/*) ;;
  *) false ;;
esac
test "$(git rev-list --left-right --count "HEAD...$UPSTREAM")" = $'0\t0'
test "$(git ls-files -s paper | awk '{print $1}')" = 160000

conda activate sid-verl
python code/envs/verify_env.py \
  --role verl --profile cu130 --imports --skip-llm-auth
```

The branch may be `main` or a named release branch. It must be clean, have a
remote upstream, and match that upstream exactly. Detached HEAD and unpushed
commits are rejected. Smoke outputs must stay under `/tmp` and must not be
uploaded. RLVR does not require the private Overleaf submodule contents, so
the runbook verifies the committed paper gitlink without initializing it.
It also skips only external-LLM credential readiness because GRPO uses local
OneRec models and W&B, not Vertex or OpenRouter.

## 4. Fetch and verify frozen inputs

```bash
conda activate sid-verl
eval "$(python code/utils/config.py --shell-exports)"
python code/rl/fetch_inputs.py --policy all
```

The fetcher checks every file against `code/rl/contract.py`. Required hashes:

```text
train parquet  8c6c6c58b079f765b400b7efa890a76f52cb33114a6bff1b0bd014846e9b38ad
val parquet    8e1a781c05edf9b898ef777cebb0b4a1c2713b0fb28f94dbcb0bfd1996cfd96d
benchmark      f0beaef9bca6b6512b8f854e801e44e50ccd39cf14685eee4fb7974b87c8ca89
```

The initializer provenance gate requires exact local Hugging Face snapshots:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

models = {
    "OpenOneRec/OneRec-1.7B-pro":
        "5dc1b097ab8194f48f14730e5400a276a22f4ca1",
    "OpenOneRec/OneRec-8B-Pro":
        "17b594d1e6b322752d939a769a07860f7e4a0261",
}
for model, revision in models.items():
    print(snapshot_download(model, revision=revision))
PY
```

`HF_HOME` must point to the repository `models/` directory, as configured by
`utils.config`.

## 5. Merge the SFT initializers

The SFT-init cells train a new LoRA on a fully merged SFT model. This is
required because VERL uses the initializer with the new LoRA disabled as the
KL reference.

```bash
conda activate sid-lf
llamafactory-cli export code/rl/merge_sft_v3_1.7b_pro.yaml
llamafactory-cli export code/rl/merge_sft_v3_8b_pro.yaml

conda activate sid-verl
python code/rl/fix_merged_config.py --policy all
python code/rl/audit_data.py --policy all --init sft
```

The normalizer is mandatory. It keeps the modern and legacy RoPE fields equal
at `1,000,000`, then `audit_data.py` verifies the value through the actual vLLM
configuration path.

Expected merged directories:

```text
saves/onerec-1.7b-pro/merged_sft_v3_12k_lr1e4
saves/onerec-8b-pro/merged_sft_v3_e128_lr5e5
```

### Create-once initializer provenance

The parity check is an input to a create-once provenance record, not a visual
spot check. For each policy, generate the dynamic and merged result JSONs with
the same 20 dev users. The dynamic route must use the exact local base snapshot
plus the frozen SFT adapter. The merged route must use the standalone export.
The following loop performs the complete gate for both Pro scales:

```bash
for POLICY in 1.7b-pro 8b-pro; do
  case "$POLICY" in
    1.7b-pro)
      BASE=models/hub/models--OpenOneRec--OneRec-1.7B-pro/snapshots/5dc1b097ab8194f48f14730e5400a276a22f4ca1
      ADAPTER=saves/onerec-1.7b-pro/lora/sft_v3_12k_lr1e4/checkpoint-1035
      MERGED=saves/onerec-1.7b-pro/merged_sft_v3_12k_lr1e4
      ;;
    8b-pro)
      BASE=models/hub/models--OpenOneRec--OneRec-8B-Pro/snapshots/17b594d1e6b322752d939a769a07860f7e4a0261
      ADAPTER=saves/onerec-8b-pro/lora/hp_e128_lr5e5/checkpoint-88
      MERGED=saves/onerec-8b-pro/merged_sft_v3_e128_lr5e5
      ;;
  esac
  P=$(mktemp -d "/tmp/rerankid_rlvr_${POLICY}_initializer.XXXXXX")

  python code/benchmark/score_sid.py \
    --model "$BASE" --slice dev --n 20 --no-profile --k 64 --n-cand 30 \
    --lora-dirs "sft=$ADAPTER" --skip-base --tag initializer_dynamic \
    --outdir "$P/dynamic"
  python code/benchmark/score_sid.py \
    --model "$MERGED" --slice dev --n 20 --no-profile --k 64 --n-cand 30 \
    --tag initializer_merged --outdir "$P/merged"

  DYNAMIC=$(find "$P/dynamic" -maxdepth 1 -type f -name '*.json' -print -quit)
  MERGED_RESULT=$(find "$P/merged" -maxdepth 1 -type f -name '*.json' -print -quit)
  test -n "$DYNAMIC"
  test -n "$MERGED_RESULT"
  python code/benchmark/adapter_parity_audit.py \
    --a "$DYNAMIC" --label-a dynamic \
    --b "$MERGED_RESULT" --label-b merged \
    --out "$P/parity.json"
  python code/rl/prepare_initializers.py \
    --policy "$POLICY" --write --parity-report "$P/parity.json"
  python code/rl/prepare_initializers.py --policy "$POLICY" --verify
done
```

Do not require byte-identical generations. Require no material degradation in
paired NDCG@5, valid output, identity, or fallback diagnostics. The writer
refuses to overwrite an existing record. It verifies the merge YAML, exact Pro
base revision, exact Base config, and every base-weight byte, source adapter
and adapter-config hashes, exact merged config, normalized model-affecting
fields, every merged file hash, and the paired parity route. The 8B record
therefore machine-checks that the
standalone initializer came from checkpoint 88, not an older 8B SFT run.

The parity report may remain under `/tmp`: its SHA256 and the accepted route
and metric evidence are embedded in `initializer_provenance.json`. The formal
launcher rehashes the adapter and complete merged bundle against that record.
Never hand-edit or replace the provenance file. If a merge must be redone,
preserve the old merged directory and export to a new, explicitly reviewed
contract path.

## 6. Smoke tests

Run the main cell first:

```bash
conda activate sid-verl
bash code/rl/run_grpo.sh 1.7b-pro --init sft --reward ndcg5 --smoke
bash code/rl/run_grpo.sh 8b-pro   --init sft --reward ndcg5 --smoke
```

Then smoke the reward ablation:

```bash
bash code/rl/run_grpo.sh 1.7b-pro --init sft  --reward hit1  --smoke
bash code/rl/run_grpo.sh 8b-pro   --init sft  --reward hit1  --smoke
```

Each smoke uses 120 prompts, 20 validation users, two optimizer steps, console
logging only, and a unique `/tmp/rerankid_rlvr_*_smoke.*` directory. Accept it
only if losses, KL, gradients, and rewards are finite and no prompt is silently
filtered or truncated.

Operational memory controls may change only:

```bash
RLVR_VLLM_MEMORY_UTILIZATION=0.45 \
RLVR_PPO_MAX_TOKENS_PER_GPU=12288 \
  bash code/rl/run_grpo.sh 8b-pro --init sft --reward ndcg5 --smoke
```

Do not change scientific batch size, rollout count, or PPO mini-batch to hide
an OOM.

## 7. Formal training

The launcher verifies a clean named branch synchronized with its remote
upstream before starting.
It additionally requires `RERANKID_MACHINE_PROFILE=cu130`, exactly four
visible GPUs, and a successful
`python code/envs/verify_env.py --role verl --profile cu130 --imports`.
Use a persistent terminal and keep the logs.

Every invocation creates `run_contract.json` before VERL starts. The record
freezes the initial Git commit, Pro base revision, complete verified base
weight identity, SFT adapter SHA256 when applicable, merged-initializer
provenance, verifier SHA256, all three RLVR data SHA256 values, the variant,
scientific hyperparameters, and the two allowed memory controls. For 8B, the
base identity covers the safetensors index and all three shards. A fresh
invocation refuses any non-empty output directory and never deletes, empties,
or adopts it.

Primary cell:

```bash
export RERANKID_MACHINE_PROFILE=cu130
mkdir -p logs
set -o pipefail

bash code/rl/run_grpo.sh 1.7b-pro --init sft --reward ndcg5 \
  2>&1 | tee logs/rlvr_1.7b_pro_sft_ndcg5.log

bash code/rl/run_grpo.sh 8b-pro --init sft --reward ndcg5 \
  2>&1 | tee logs/rlvr_8b_pro_sft_ndcg5.log
```

Reward ablation:

```bash
bash code/rl/run_grpo.sh 1.7b-pro --init sft --reward hit1 \
  2>&1 | tee logs/rlvr_1.7b_pro_sft_hit1.log

bash code/rl/run_grpo.sh 8b-pro --init sft --reward hit1 \
  2>&1 | tee logs/rlvr_8b_pro_sft_hit1.log
```

Each run must reach global step 100 and contain checkpoints 20, 40, 60, 80,
and 100. Record the W&B run ID in `experiments/final_manifest.json`. W&B state
alone is not enough to reject a run if the process exited during cleanup.
Require the five checkpoints, complete logs, and successful downstream
evaluation.

Automatic resume is disabled. Resume only after checking the interrupted run,
using the same flags plus `--resume`, for example:

```bash
bash code/rl/run_grpo.sh 8b-pro --init sft --reward ndcg5 --resume \
  2>&1 | tee -a logs/rlvr_8b_pro_sft_ndcg5.log
```

Resume requires an exact byte-semantic match to the existing
`run_contract.json`, including the original Git commit and memory controls. It
also requires a valid saved-step marker and a non-empty actor checkpoint at a
scheduled step below 100. A completed run, an orphan checkpoint, a changed
verifier, changed data, changed initializer, or changed hyperparameter fails
closed. Never remove an old directory to bypass this guard.

## 8. Export compact adapters

For one cell, set its initializer and reward once:

```bash
POLICY=1.7b-pro
INIT=sft
REWARD=ndcg5

for STEP in 20 40 60 80 100; do
  python code/rl/export_lora_checkpoint.py \
    --policy "$POLICY" --init "$INIT" --reward "$REWARD" --step "$STEP"
done
```

Repeat for each policy and registered cell. The exporter writes a
`provenance.json` beside every compact adapter and refuses to overwrite an
existing export. It first rebuilds and verifies the exact formal
`run_contract.json`, requires the completed step-100 marker, hashes the full
source actor tree, and binds those identities plus the original training
commit into the compact adapter. The selected-adapter publisher repeats these
checks before upload and includes the verified run contract beside the three
adapter files in the artifact.

## 9. Evaluate all checkpoints on dev

Resolve the exact contract:

```bash
POLICY=1.7b-pro
INIT=sft
REWARD=ndcg5
eval "$(python code/rl/contract.py \
  --policy "$POLICY" --init "$INIT" --reward "$REWARD" --shell)"

LORAS=$(python - <<PY
print(",".join(
    f"step{step}=${EXPORT_REL}/step-{step:06d}"
    for step in (20, 40, 60, 80, 100)
))
PY
)

python code/benchmark/score_sid.py \
  --model "$MODEL_REL" --slice dev --no-profile --k 64 --n-cand 30 \
  --lora-dirs "$LORAS" --tag "$DEV_TAG" \
  --outdir "$RESULTS_REL/dev_select"
```

Select by NDCG@5:

```bash
eval "$(python code/rl/select_checkpoint.py \
  --policy "$POLICY" --init "$INIT" --reward "$REWARD" --shell)"
echo "selected $POLICY $VARIANT step $BEST_STEP at $BEST"
```

The unadapted initializer is reported for comparison but cannot win
checkpoint selection. Ties between RL checkpoints select the earlier step.

## 10. One-time test sweep

Only the dev-selected adapter may touch test:

```bash
POLICY=1.7b-pro
INIT=sft
REWARD=ndcg5
eval "$(python code/rl/select_checkpoint.py \
  --profile canonical --policy "$POLICY" \
  --init "$INIT" --reward "$REWARD" --shell)"

python code/benchmark/score_sid.py \
  --model "$MODEL" --slice test --no-profile \
  --k-grid 0,4,8,16,32,64,128,256 --n-cand 30 \
  --lora-dirs "best=$BEST" --skip-base --tag "$TEST_TAG" --outdir "$OUT"

for N in 10 20 50 100; do
  python code/benchmark/score_sid.py \
    --model "$MODEL" --slice test --no-profile --k 64 --n-cand "$N" \
    --lora-dirs "best=$BEST" --skip-base --tag "$TEST_TAG" --outdir "$OUT"
done
```

Do not evaluate another checkpoint on test after seeing these results.

## 11. Validate and publish

Start this section by resolving the exact cell again. It does not depend on
shell variables left by Sections 9 or 10:

```bash
POLICY=1.7b-pro
INIT=sft
REWARD=ndcg5
eval "$(python code/rl/select_checkpoint.py \
  --profile canonical --policy "$POLICY" \
  --init "$INIT" --reward "$REWARD" --shell)"
```

Dry-run the selected adapter and its contract-bound result artifact:

```bash
python code/rl/publish_selected.py \
  --policy "$POLICY" --init "$INIT" --reward "$REWARD" \
  --step "$BEST_STEP"

python code/rl/publish_results.py \
  --profile canonical --policy "$POLICY" \
  --init "$INIT" --reward "$REWARD"
```

After reviewing the metadata from a clean, pushed release checkout:

```bash
python code/rl/publish_selected.py \
  --policy "$POLICY" --init "$INIT" --reward "$REWARD" \
  --step "$BEST_STEP" --upload

python code/rl/publish_results.py \
  --profile canonical --policy "$POLICY" \
  --init "$INIT" --reward "$REWARD" --upload
```

`publish_results.py` publishes only this arm. It does not download, overlay, or
republish a shared `results` bundle, and it supplies no mutable artifact alias.
The canonical artifact contains the exact 18 result files plus the selected
adapter provenance and immutable training run contract. The publisher
revalidates the adapter bytes and source actor tree before admitting the
result evidence.
After upload it prints the assigned immutable W&B qualified name, version, and
digest. Record all three in `experiments/final_manifest.json`.

The result publisher rejects subset runs, smoke paths, non-SID views, corrupted
SIDs, the wrong benchmark hash or UID set, missing dev checkpoints,
non-selected test adapters, symlinked result paths, and any file without
exactly 400 distinct users. For every evaluation cell it reconstructs each
user's display-order grades from the frozen benchmark, validates the parsed
order and diagnostics, and reruns deterministic aggregation. Checkpoint
selection uses the recomputed NDCG@5 rather than trusting stored summary
metrics.

To validate one already completed historical result tree without requiring its
adapter directory, use its exact manifest-pinned executed profile:

```bash
ARM=8b-pro_sft_hit1
python code/rl/select_checkpoint.py \
  --profile executed --arm "$ARM"
python code/rl/publish_results.py \
  --profile executed --arm "$ARM"
```

Only completed arms with an `executed_profile` in the manifest accept this
route. All four registered Pro-scale reward cells are now complete and have
manifest-pinned executed profiles. Executed validation accepts no alternate
tag, path, checkpoint label, or step-directory alias. It checks the exact
18-file tree SHA256, the selected dev and anchor-test evidence SHA256 values,
revalidates all per-user evidence, and recomputes the dev argmax.

## 12. Completion record

For every `(policy, reward)` cell, retain:

- Git commit and environment profile.
- W&B run ID and terminal status.
- Initializer model revision and, for SFT init, adapter artifact version,
  digest, and SHA256.
- Reward file SHA256.
- Data artifact immutable version, digest, and file hashes.
- Steps 20 through 100.
- Dev NDCG@5 for each checkpoint and the selected step.
- Selected adapter artifact version and digest.
- Twelve test cells in total, including the k64/N30 anchor.
- Final result artifact immutable version and digest.

Write immutable identities to `experiments/final_manifest.json`. Do not use a
`latest` alias as the final paper provenance.
