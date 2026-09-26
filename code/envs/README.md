# ReRankID environments

ReRankID uses two Conda environment roles. The names are the same on both
machines, while the selected host profile determines their CUDA and package
versions.

| Environment | Workloads |
|---|---|
| `sid-lf` | LLaMA-Factory SFT/RFT training and model export |
| `sid-verl` | benchmark inference, RFT sampling, OPD, RLVR, and evaluation |

| Profile | Validated host | `sid-lf` | `sid-verl` |
|---|---|---|---|
| `cu128` | current 2xA100-80GB host, driver 535 | torch 2.9.1 cu128, LLaMA-Factory 0.9.5 | torch 2.10 cu128, vLLM 0.19.1, verl 0.8.0 |
| `cu130` | training 4xA100-80GB host, driver 580 or newer | torch 2.11 cu130, LLaMA-Factory 0.9.5 | torch 2.11 cu130, vLLM 0.24, verl 0.8.0 |

`cu128` and `cu130` identify the PyTorch wheel runtime, not necessarily the
host `nvcc` version.

## Machine policy

The current 2xA100 host is reserved for writing, result analysis, plotting,
paper compilation, and OpenRouter calls. Future GPU data generation, training,
checkpoint selection, and formal model evaluation run on the training host.
The local profile remains documented so completed runs can be inspected and
the validated environment can be reconstructed when necessary.

Exact OPD reproduction is the one exception at the environment level, not at
the machine level: it still runs on the training host, but uses the executed
two-GPU cu128 stack. Install it under isolated environment names so the
training host's cu130 `sid-lf` and `sid-verl` environments remain intact. The
OPD runbook gives the exact commands.

Only the OneRec Pro models are in scope:

```text
OpenOneRec/OneRec-1.7B-pro
OpenOneRec/OneRec-8B-Pro
```

Before updating the training host, preserve its uncommitted work and create a
clean worktree at the exact release commit.

## Why two environments?

`sid-verl` installs the profile-specific vLLM stack before installing
`verl==0.8.0 --no-deps`, then adds the explicitly pinned extras. This supports
standalone SID inference and verl rollouts in one environment.

`sid-lf` remains separate because LLaMA-Factory and the vLLM/verl stack require
different torch and dependency combinations. It is used for ordinary LoRA
training and export.

## Build the validated profiles

Current host:

```bash
export FLASH_ATTN_WHEEL=/path/to/wheels/flash_attn-2.8.3-cp311-cp311-linux_x86_64.whl
export FLASH_ATTN_WHEEL_SHA256=af19d198f12a0a22d72ca6ab0874b64e53a5245289937b037ae8520ee98375a1
bash code/envs/setup_envs.sh --profile cu128 all
```

The explicit wheel is required because this host's CUDA toolkit is older than
the selected torch runtime. Do not attempt an unreviewed source build and do
not reuse this wheel on the training host.

Training host:

```bash
unset FLASH_ATTN_WHEEL FLASH_ATTN_WHEEL_SHA256
nvidia-smi
nvcc --version
bash code/envs/setup_envs.sh --profile cu130 all
```

For the separately named cu128 OPD environments on the training host, follow
[`code/opd/RUNBOOK.md`](../opd/RUNBOOK.md). `setup_envs.sh` accepts
`RERANKID_LF_ENV_NAME` and `RERANKID_VERL_ENV_NAME`; their defaults remain
`sid-lf` and `sid-verl`.

The training host may use a matching prebuilt FlashAttention wheel by setting
`FLASH_ATTN_WHEEL` and its lowercase hexadecimal
`FLASH_ATTN_WHEEL_SHA256` explicitly. Otherwise its toolkit must be compatible
with the selected cu130 build. Source installation uses the hash-pinned PyPI
release in `flash-attn-source.txt`.

`--profile auto` is available for diagnostics, but formal reproduction should
always pass the profile explicitly:

```bash
bash code/envs/setup_envs.sh --dry-run --profile cu130 all
```

## Credentials

Credentials are host-local and must never be committed.

- `HF_TOKEN` grants access to gated OpenOneRec data and checkpoints.
- `WANDB_API_KEY` permits artifact transfer and run logging.
- `OPENROUTER_API_KEY` is used only for approved API calls on the current host.
- Vertex calls on the training host use its approved ADC configuration.
- `API_KEYS_FILE` may point to a private JSON file outside the repository.

The setup script records only non-secret environment settings. Verification
reports boolean credential readiness and never prints credential values.

## Verify

Current host:

```bash
conda activate sid-lf
python code/envs/verify_env.py --role lf --profile cu128 --imports

conda activate sid-verl
python code/envs/verify_env.py --role verl --profile cu128 --imports
```

Training host:

```bash
conda activate sid-lf
python code/envs/verify_env.py --role lf --profile cu130 --imports

conda activate sid-verl
python code/envs/verify_env.py --role verl --profile cu130 --imports
```


`verify_env.py` checks the selected profile, expected imports, package
versions, GPU visibility, and credential readiness. Workloads that provably
never call an external LLM, such as OPD, may pass `--skip-llm-auth`; this does
not weaken runtime or GPU validation. Custom environment names are read from
`RERANKID_LF_ENV_NAME` and `RERANKID_VERL_ENV_NAME`.

## Data and artifacts

The repository does not distribute generated data, model weights, or results.
The following paths are ignored by Git:

```text
data/  models/  results/  saves/
```

Restore formal inputs using the exact W&B versions and hashes in
[`experiments/final_manifest.json`](../../experiments/final_manifest.json).
Do not use `latest` as final provenance.

The gated raw files remain outside project artifacts:

```text
data/recif/onerec_bench_release.parquet
data/recif/video_ad_pid2sid.parquet
data/recif/pid2caption.parquet
```

Their transfer must follow the data owner's access rules. Never upload them to
the ReRankID W&B project.

After environment verification, follow the stage-specific runbook:

- [SFT v3 12k](../sft/V3_RUNBOOK.md)
- [RFT 6k](../rft/RUNBOOK.md)
- [OPD 6k](../opd/RUNBOOK.md)
- [RLVR and its Hit@1 reward ablation](../rl/RUNBOOK.md)
