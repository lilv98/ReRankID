# OPD: matched-scale SID-only distillation

This is the authoritative reproduction guide for the final OPD arm. The
student is a released OneRec Pro Base model. Its frozen teacher is the
dev-selected SFT-v3 12k model at the same scale. Teacher and student receive
the exact same SID-only prompt and the same student-generated response prefix.

OPD is an independent post-training route from Base. The OPD adapter must
never be stacked on an SFT adapter or evaluated on the merged SFT teacher.

## 1. Frozen recipe

| Field | 1.7B Pro | 8B Pro |
|---|---|---|
| student | OneRec-1.7B-pro Base | OneRec-8B-Pro Base |
| frozen teacher | SFT-v3 12k, checkpoint 1035 | SFT-v3 12k, checkpoint 88 |
| input to both models | SID-only prompt and student prefix | same |
| training prompts | 6,000 label-free prompts | same |
| optimizer steps | 200 | same |
| checkpoints | every 25 steps | same |
| objective | renormalized teacher top-32 forward KL | same |
| loss aggregation | `seq-mean-token-mean` | same |
| rollout | `n=1`, temperature 1.0, top-p 1.0 | same |
| prompt / response budget | 3,072 / 512 tokens | same |
| LoRA | rank 16, alpha 32, all linear modules | same |
| learning rate | `5e-5`, constant, 3% warmup | same |
| logical batch | 30 | same |
| task reward | disabled | disabled |
| separate reference KL | disabled | disabled |

The implementation is intentionally label-free. `build_data.py` removes
positive labels and label-derived metadata. verl still requires a reward
callback, so `zero_reward.py` returns exactly zero and the task-reward path is
disabled in the training configuration. No caption, gold candidate, task
reward, or privileged teacher prompt enters the update.

The final 1.7B and 8B arms differ only in matched model scale and operational
memory controls. The 8B arm uses parameter offload, eager rollout, a smaller
vLLM allocation, and online compact-checkpoint export. These controls do not
change its data, objective, sampling, logical batch, or learning rate.

## 2. Exact environment and host paths

The executed OPD recipe is pinned to the two-GPU `cu128` environment in
`code/envs/cu128/`, including verl 0.8.0, vLLM 0.19.1, Ray 2.55.1,
Transformers 4.57.6, and torch 2.10.0+cu128. The runtime patches fail closed
if their audited verl sources differ.

Under the current machine policy, any new reproduction runs on the training
host, not the writing host. A newer NVIDIA driver can run this cu128 wheel
stack. From a clean checkout of the exact experiment commit, keep the training
host's cu130 environments intact by installing the OPD stack under isolated
names:

```bash
cd /absolute/path/to/ReRankID
export RERANKID_LF_ENV_NAME=sid-lf-opd-cu128
export RERANKID_VERL_ENV_NAME=sid-verl-opd-cu128
export FLASH_ATTN_WHEEL=/absolute/path/to/flash_attn-2.8.3-cp311-cp311-linux_x86_64.whl
export FLASH_ATTN_WHEEL_SHA256=af19d198f12a0a22d72ca6ab0874b64e53a5245289937b037ae8520ee98375a1
bash code/envs/setup_envs.sh --profile cu128 all

export SID_LF_PYTHON=$(
  conda run -n "$RERANKID_LF_ENV_NAME" python -c \
    'import sys; print(sys.executable)'
)
export SID_VERL_PYTHON=$(
  conda run -n "$RERANKID_VERL_ENV_NAME" python -c \
    'import sys; print(sys.executable)'
)
```

The FlashAttention wheel must match Python 3.11 and the pinned torch cu128
runtime. Then set the host-local credentials, storage, and visible devices:

```bash
export API_KEYS_FILE=/absolute/private/path/api_keys.json
export WANDB_ENTITY=<your-wandb-entity>
export CUDA_VISIBLE_DEVICES=0,1
```

The 1.7B run lives under the repository. The 8B raw checkpoints and merger
workspace are large, so they may be redirected to any writable filesystem:

```bash
export OPD_STORAGE_ROOT=/absolute/large/path/rerankid-opd
export OPD_MERGE_ROOT=/absolute/large/path/rerankid-opd-merge
export OPD_SMOKE_ROOT=/absolute/large/path/rerankid-opd-smoke
```

All three values must be absolute. If omitted, the launcher reproduces the
executed host defaults:

```text
OPD_STORAGE_ROOT=/datadrive2/rerankid-opd
OPD_MERGE_ROOT=/datadrive2/rerankid_opd_merge_e128_lr5e5
OPD_SMOKE_ROOT=/datadrive2/rerankid-opd-smoke
```

Storage paths are operational only. Their resolved values are written into
the immutable formal run manifest. The 8B backing filesystem needs at least
120 GiB free at launch.

Never share either visible GPU with another training or vLLM process.

## 3. Prepare a fresh clone

Verify the isolated exact-stack environments first:

```bash
conda run -n "$RERANKID_LF_ENV_NAME" python code/envs/verify_env.py \
  --role lf --profile cu128 --imports --skip-llm-auth
conda run -n "$RERANKID_VERL_ENV_NAME" python code/envs/verify_env.py \
  --role verl --profile cu128 --imports --skip-llm-auth
```

`--skip-llm-auth` is valid here because OPD never calls an external LLM
provider. It does not skip package, CUDA, GPU-count, driver, or import checks.
`conda run` is required so the environment-scoped machine and backend markers
are active during verification. The exact Python paths remain available to
the launchers after this gate.

Then prepare both OPD scales:

```bash
bash code/opd/prepare.sh --policy all
```

The preparation script performs the complete immutable input chain:

1. fetch the hash-checked 6k post-training prompt artifact and exact SFT-v3
   adapters from W&B;
2. download the pinned Pro Base revisions through the merge configurations;
3. merge checkpoint 1035 for 1.7B Pro and checkpoint 88 for 8B Pro;
4. normalize both merged teachers to the RoPE schema expected by sid-verl;
5. construct label-free `data/opd_v1_6k` parquets;
6. verify data, tokenizer, Base-weight, and teacher-weight hashes;
7. verify the complete Base and merged-teacher `config.json` bytes against
   their scale-specific frozen SHA256 values.

The exact merge configurations are:

```text
code/rl/merge_sft_v3_1.7b_pro.yaml
code/rl/merge_sft_v3_8b_pro.yaml
```

Preparation is safe to rerun. An existing merged teacher is reused, then
hash-verified. A partial or different teacher closes the verifier rather than
being silently adopted.

To prepare only one scale:

```bash
bash code/opd/prepare.sh --policy 1.7b-pro
bash code/opd/prepare.sh --policy 8b-pro
```

Frozen OPD data hashes:

```text
train.parquet  1afdcc53cd17890a7340a80f7c3a5be1aab3c55f3bb8b573e8434d17c18cb32a
val.parquet    6b091a44b8f74c46157f8984729431f0647d801b35944b8e452990132a0cb011
stats.json     7e1ffb34e378e80281d873706a1d496480082b0ab3c968210c7e9d2a7d59ba6c
```

## 4. Code and numerical gates

When the repository's test suite is present (full release), run all OPD
unit tests before using a GPU:

```bash
PYTHONPATH=code "$SID_VERL_PYTHON" \
  -m unittest discover -s tests -p 'test_opd*.py' -v
```

The formal launcher requires all OPD-scoped code and tests to be committed and
unchanged. It also validates exact package versions, source hashes, model
weights, complete model-config identities, tokenizer identity, data hashes,
prompt lengths, and the full Hydra override list. New run manifests use
schema 2 and embed both config hashes. Completed schema-1 manifests remain
readable, but publication independently rechecks both on-disk configs against
the frozen contract.

Run one smoke per scale:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash code/opd/run_opd.sh --smoke --policy 1.7b-pro

CUDA_VISIBLE_DEVICES=0,1 \
  bash code/opd/run_opd.sh --smoke --policy 8b-pro
```

Each smoke uses 100 prompts and two updates. It must have finite loss and
gradient norm, nonzero gradient, exactly zero task reward, no truncation, and
no tokenizer drift. Inspect the printed smoke directory with:

```bash
"$SID_VERL_PYTHON" code/opd/audit_rollouts.py /absolute/smoke/path/rollouts
```

For 8B, the smoke also saves a real step-2 raw checkpoint. Export and load it
once to test the memory-sensitive merger path:

```bash
SMOKE=/absolute/path/printed/by/the/8b/smoke
mkdir -p "$OPD_MERGE_ROOT/smoke"

"$SID_VERL_PYTHON" code/opd/export_lora_checkpoint.py \
  --step 2 \
  --run-dir "$SMOKE" \
  --export-dir "$SMOKE/compact_adapters" \
  --merger-temp-root "$OPD_MERGE_ROOT/smoke"

CUDA_VISIBLE_DEVICES=0 "$SID_VERL_PYTHON" code/benchmark/score_sid.py \
  --model models/hub/models--OpenOneRec--OneRec-8B-Pro/snapshots/17b594d1e6b322752d939a769a07860f7e4a0261 \
  --slice dev --n 20 --no-profile --k 64 --n-cand 30 \
  --shuffle 0 --render sid --top-k 10 --k-ndcg 5 \
  --max-model-len 8192 --max-tokens 2048 --extract-tokens 256 --tp 1 \
  --lora-dirs smoke="$SMOKE/compact_adapters/step-000002" \
  --max-lora-rank 32 --skip-base --tag opd_8b_checkpoint_smoke \
  --outdir "$SMOKE/adapter_load"
```

This 20-user call is only a loader gate. It is not a reported result.

## 5. Formal training

Run 1.7B Pro:

```bash
mkdir -p saves/onerec-1.7b-pro/opd_v3_6k
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 \
  bash code/opd/run_opd.sh --formal --policy 1.7b-pro 2>&1 | \
  tee -a saves/onerec-1.7b-pro/opd_v3_6k/train.log
```

Run 8B Pro in a detached tmux session:

```bash
bash code/opd/launch_8b_formal_tmux.sh
tmux capture-pane -p -t =rerankid-opd-8b-pro-e128-lr5e5-formal -S -80
cat "$OPD_STORAGE_ROOT/onerec-8b-pro/opd_v3_6k_e128_lr5e5/launcher.status"
```

Only formal training creates a W&B run. Smoke outputs stay local. Each formal
run consumes all 6,000 prompts exactly once and saves steps:

```text
25 50 75 100 125 150 175 200
```

The 8B launcher synchronously exports each compact LoRA before pruning raw
checkpoints. It retains the newest two recoverable raw checkpoints. A failed
export stops the run before pruning.

To resume after interruption, rerun the exact formal command from the same
commit with the same environment and storage variables. Never edit
`run_manifest.json`, adopt a partial checkpoint, or start a second writer in
the same output directory. The launcher locks the directory and validates
checkpoint, compact-adapter, and W&B history alignment before continuing.

## 6. Export formal checkpoints

The 8B formal launcher exports all eight checkpoints online. For 1.7B, export
them after training:

```bash
for step in 25 50 75 100 125 150 175 200; do
  "$SID_VERL_PYTHON" code/opd/export_lora_checkpoint.py \
    --step "$step" \
    --run-dir saves/onerec-1.7b-pro/opd_v3_6k \
    --export-dir saves/onerec-1.7b-pro/opd_v3_6k_adapters \
    --if-exists verify
done
```

Every compact adapter is bound to its raw checkpoint and immutable run
manifest. Evaluate it on the corresponding Pro Base.

## 7. Dev selection

Set one scale:

```bash
POLICY=1.7b-pro
MODEL=models/hub/models--OpenOneRec--OneRec-1.7B-pro/snapshots/5dc1b097ab8194f48f14730e5400a276a22f4ca1
RUN=saves/onerec-1.7b-pro/opd_v3_6k
ADAPTERS=saves/onerec-1.7b-pro/opd_v3_6k_adapters
RESULTS=results/opd-1.7b-pro-v3-6k
```

or:

```bash
POLICY=8b-pro
MODEL=models/hub/models--OpenOneRec--OneRec-8B-Pro/snapshots/17b594d1e6b322752d939a769a07860f7e4a0261
RUN=saves/onerec-8b-pro/opd_v3_6k_e128_lr5e5
ADAPTERS=saves/onerec-8b-pro/opd_v3_6k_e128_lr5e5_adapters
RESULTS=results/opd-8b-pro-v3-6k-e128-lr5e5
```

Then define common paths and evaluate all eight checkpoints on the complete
400-user dev anchor:

```bash
BENCH=data/benchmark/bench.parquet
MANIFEST="$RUN/run_manifest.json"
STATE="$RESULTS/eval_state"
DEV="$RESULTS/formal_dev"
TEST="$RESULTS/test_once"
STAGING="$RESULTS/.test_staging"
mkdir -p "$STATE"

"$SID_VERL_PYTHON" code/opd/eval_protocol.py dev-command \
  --policy "$POLICY" \
  --state-dir "$STATE" --dev-dir "$DEV" \
  --adapter-root "$ADAPTERS" --run-manifest "$MANIFEST" \
  --bench "$BENCH" --model "$MODEL" --tp 1 \
  > "$STATE/dev_plan.json"

"$SID_VERL_PYTHON" -c \
  'import json,subprocess,sys; p=json.load(open(sys.argv[1])); subprocess.run(p["command"], check=True)' \
  "$STATE/dev_plan.json"

"$SID_VERL_PYTHON" code/opd/eval_protocol.py select \
  --policy "$POLICY" \
  --state-dir "$STATE" --dev-dir "$DEV" \
  --adapter-root "$ADAPTERS" --run-manifest "$MANIFEST" \
  --bench "$BENCH" --model "$MODEL" \
  > "$STATE/selection_report.json"
```

Selection uses only the unrounded mean NDCG@5 over 400 dev users. An exact tie
goes to the earlier step. Test results, format diagnostics, identity rate,
loop rate, and W&B training loss are not selection criteria.

## 8. Test once

The sealed test sweep contains 12 unique cells:

```text
history axis at N=30: k = 0, 4, 8, 16, 32, 64, 128, 256
candidate axis at k=64: N = 10, 20, 30, 50, 100
```

Arm the irreversible test transition only after checking `selection.json`:

```bash
"$SID_VERL_PYTHON" code/opd/eval_protocol.py test-command \
  --policy "$POLICY" \
  --state-dir "$STATE" --test-dir "$TEST" --staging-dir "$STAGING" \
  --confirm-test-once > "$STATE/test_plan.json"

"$SID_VERL_PYTHON" -c \
  'import json,subprocess,sys; p=json.load(open(sys.argv[1])); [subprocess.run(c, check=True) for c in p["commands"]]' \
  "$STATE/test_plan.json"

while IFS= read -r result; do
  "$SID_VERL_PYTHON" code/opd/eval_protocol.py accept-test \
    --policy "$POLICY" \
    --state-dir "$STATE" --test-dir "$TEST" --result "$result"
done < <(find "$STAGING" -maxdepth 1 -type f -name '*.json' -print | sort)

"$SID_VERL_PYTHON" code/opd/eval_protocol.py validate-test \
  --policy "$POLICY" \
  --state-dir "$STATE" --test-dir "$TEST" \
  > "$STATE/test_validation.json"
```

The gate validates exact UIDs, benchmark-derived grades, scorer metrics,
model and adapter identities, source-stack hashes, and one create-once receipt
per accepted cell. Never delete or replace an accepted result or receipt.

## 9. Publish the sealed release

The publisher revalidates the selected compact adapter with the exporter
verifier against the exact `run_manifest.json` named by `selection.json`.
The release contains that manifest at `release/run_manifest.json`, the
selected adapter, eight formal dev files, twelve accepted test files, and
their immutable evaluation state. Release metadata records the training
commit, the byte-level and canonical manifest hashes, and the selected raw
checkpoint identity carried by the adapter provenance. It excludes smoke
outputs, rollout dumps, raw checkpoints, and any preliminary run.

Publication verifies the evaluator identity frozen inside the sealed
selection and every result and receipt bound to it. It does not require a
later release checkout to have byte-identical evaluator source files.
Training, selection, test arming, and result acceptance keep the default
strict current-source check; only the final publisher uses this historical
release mode.

Dry-run validation:

```bash
"$SID_VERL_PYTHON" code/opd/publish.py --policy "$POLICY"
```

After reviewing the report from a clean commit:

```bash
"$SID_VERL_PYTHON" code/opd/publish.py --policy "$POLICY" --upload
```

The W&B artifact is committed without aliases. Record the immutable
`collection:vN` and digest printed by the publisher in the final manifest.

## 10. Recorded final selections

These values identify the completed runs and are not launch-time defaults:

| Scale | Dev-selected step | Dev NDCG@5 | Test anchor NDCG@5 |
|---|---:|---:|---:|
| 1.7B Pro | 75 | .2109 | .1954 |
| 8B Pro | 50 | .2662 | .2100 |

The complete 12-cell outputs, hashes, W&B run identities, and selected adapter
digests belong in the repository-level final experiment manifest. The test
anchor values above must never be used to revise the recipe or checkpoint.
