# Environments

ReRankID uses two conda environments with pinned packages.

| Environment | Used for |
|---|---|
| `sid-lf` | LLaMA-Factory SFT and RFT training, model export and merge, figures and tables |
| `sid-verl` | vLLM scoring, RFT sampling, OPD and RLVR (verl 0.8.0) |

Two profiles name the PyTorch CUDA runtime of the wheels (not the system `nvcc`).

| Profile | Python | `sid-lf` | `sid-verl` | Used for |
|---|---|---|---|---|
| `cu130` | 3.12 | torch 2.11, LLaMA-Factory 0.9.5 | torch 2.11, vLLM 0.24.0 | default |
| `cu128` | 3.11 | torch 2.9.1, LLaMA-Factory 0.9.5 | torch 2.10, vLLM 0.19.1 | OPD |

OPD training and the scoring of the OPD and RFT checkpoints used the `cu128` stack (vLLM 0.19.1);
all other results used `cu130`. The vLLM version of every score is in its `meta.vllm`. Both
profiles need an NVIDIA driver that supports their CUDA runtime.

## Build

```bash
bash code/envs/setup_envs.sh --profile cu130 all            # sid-lf and sid-verl
bash code/envs/setup_envs.sh --profile cu130 --dry-run all  # print the commands only
```

`sid-verl` is built in layers: the profile's `sid-verl.txt` (vLLM and its torch), then
`verl==0.8.0 --no-deps`, then `sid-verl-extras.txt`, then flash-attn 2.8.3. Set `FLASH_ATTN_WHEEL`
to install a prebuilt flash-attn wheel matching your Python and torch; otherwise flash-attn is
built from its PyPI source release, which needs a CUDA toolkit compatible with the profile.

To keep both profiles on one machine, give one of them other names, for example

```bash
RERANKID_VERL_ENV_NAME=sid-verl-cu128 bash code/envs/setup_envs.sh --profile cu128 verl
```

`setup_envs.sh` refuses to install into an existing environment whose Python version belongs to
the other profile.

## Verify

```bash
conda activate sid-lf   && python code/envs/verify_env.py --role lf --profile cu130 --imports
conda activate sid-verl && python code/envs/verify_env.py --role verl --profile cu130 --imports
```

## Credentials

| Variable | Needed for |
|---|---|
| `HF_TOKEN` | gated RecIF data, OneRec models and the ReRankID release |
| `OPENROUTER_API_KEY` | text-reading LLM references, teacher annotation and the rationale judge via OpenRouter |
| `VERTEX_PROJECT`, `VERTEX_LOCATION` | the same calls via Vertex AI (Application Default Credentials; `RERANKID_LLM_BACKEND=vertex`) |

Instead of the first two variables, a JSON file named by `API_KEYS_FILE` (default `api_keys.json`
in the repository root, ignored by git) can hold `{"huggingface": "hf_...", "openrouter": "..."}`.
The Vertex AI settings are read from the environment only. `RERANKID_MODELS` sets the model cache
(`HF_HOME`, default `models/` in the repository root).

The paper's API calls used zero-data-retention, no-training settings. Enable them in your
OpenRouter privacy settings or Vertex AI project before sending benchmark text to a provider.

Experiment tracking is off by default. The OPD and RLVR launchers log to Weights & Biases when
`WANDB_PROJECT` is set and `wandb` is installed; for SFT and RFT, set `report_to` in the config.

Next, fetch the released data with `python code/download/release.py --install`.
