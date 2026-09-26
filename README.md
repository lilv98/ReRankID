# ReRankID

ReRankID is a controlled benchmark for reasoning reranking when Semantic IDs
(SIDs) are the sole item representation. A model receives a user history and a
fixed candidate slate, both rendered only as SIDs, then produces a rationale
and an ordered list of candidate indices.

The benchmark separates four questions:

1. Can a SID-pretrained model access item information through its learned SID
   vocabulary?
2. Does it use longer histories effectively?
3. How robust is it as the candidate slate grows?
4. Can common post-training strategies elicit better SID-only reasoning
   reranking?

All OneRec experiments use only the Pro checkpoints:
`OpenOneRec/OneRec-1.7B-pro` and `OpenOneRec/OneRec-8B-Pro`.

## Final experimental scope

The repository retains one final recipe for each experimental arm:

| Arm | Frozen setting |
|---|---|
| Benchmark | One positive per user, 400-user dev and 400-user test splits, nested candidate slates, controlled history lengths |
| SFT | SFT v3, a 12,000-row annotation target, caption-view hindsight teacher, SID-only student input, ordinary LoRA |
| RFT | 6,000 prompts, best-of-8 self-sampling, ordinary LoRA |
| OPD | 6,000 on-policy trajectories, matched-scale frozen SFT teacher, identical SID-only teacher and student input |
| RLVR | SFT-initialized NDCG@5 primary arm, plus an SFT-initialized Hit@1 reward ablation |

The two model scales use matched data and evaluation protocols. Checkpoints are
selected on the 400-user dev split. The selected checkpoint is then evaluated
once on the frozen 400-user test sweep.

Exact identities for available frozen artifacts, selected checkpoints, W&B
run identifiers, and per-result evidence hashes are recorded in
[`experiments/final_manifest.json`](experiments/final_manifest.json). The
manifest, rather than an artifact alias such as `latest`, is the source of
truth for reproduction.

## Repository map

```text
code/
  benchmark/   benchmark construction, SID/text scoring, statistics, plots
  sft/         final 12k SFT data generation, training, selection, publishing
  rft/         final 6k rejection-sampled fine-tuning pipeline
  opd/         matched-scale SID-only on-policy distillation
  rl/          RLVR data, rewards, launchers, selection, publishing
  judge/       rationale-quality judging and control validation
  envs/        validated Conda environment profiles
  utils/       shared prompts, metrics, data access, inference utilities
experiments/   frozen final-scope manifest and prompt-token measurements
```

Generated data, model weights, results, and caches are intentionally excluded
from Git:

```text
data/  models/  results/  saves/
```

## Archive scope and anonymization

This archive contains the code only. Frozen benchmark builds, trained
adapters, and per-user result files are distributed as exact versioned
artifacts whose identities and evidence hashes are recorded in
[`experiments/final_manifest.json`](experiments/final_manifest.json); the raw
interaction data remains governed by the OpenOneRec access terms and is not
redistributed.

For double-blind review, account-specific values are anonymized: the W&B
entity defaults to `anonymous` (override with `WANDB_ENTITY`), the Vertex
project must be supplied via `VERTEX_PROJECT`, and all machine-local paths
are environment-variable defaults. No functional code differs from the
version that produced the reported results.

## Reproduction entry points

Use the stage runbooks rather than reconstructing commands from experiment
logs:

- [Environment setup](code/envs/README.md)
- [SFT v3 12k](code/sft/V3_RUNBOOK.md)
- [RFT 6k](code/rft/RUNBOOK.md)
- [OPD 6k](code/opd/RUNBOOK.md)
- [RLVR and its reward ablation](code/rl/RUNBOOK.md)

The benchmark builders and scorers are:

```bash
python code/benchmark/build_pools.py
python code/benchmark/build_bench.py

python code/benchmark/score_sid.py --help
python code/benchmark/score_text.py --help
python code/benchmark/score_baselines.py --help
python code/benchmark/paired_stats.py --help
python code/benchmark/stability_audit.py --help
```

The rationale-quality judge and its display-order controls are:

```bash
python code/judge/judge_arms.py --help
python code/judge/pairwise_judge.py --help
python code/judge/pairwise_controls.py --help
python code/judge/plot_rationale_judge.py --help
```

All launchers should be run from the repository root. Paths are resolved
relative to that root, and every formal run should use the exact inputs and
model revisions listed in the final manifest.

## Environments

The repository defines two roles on each host:

- `sid-lf`: LLaMA-Factory SFT/RFT training and model export.
- `sid-verl`: benchmark inference, RFT sampling, OPD, and RLVR.

The validated host profiles and verification commands are documented in
[`code/envs/README.md`](code/envs/README.md).

## Data and credentials

The raw recommendation data comes from the gated
`OpenOneRec/OpenOneRec-RecIF` release. It contains real user histories and is
not redistributed by this repository. Formal derived data and published
results are transferred only through exact W&B artifact versions in the final
manifest. Completed runs without a published result artifact remain fixed by
their run identifier and evidence SHA-256 in that manifest. Smoke data must
never be uploaded to the formal artifact collections.

Credentials remain outside Git. `code/utils/config.py` accepts host-local
environment variables or a private `API_KEYS_FILE`. Never commit API keys,
W&B credentials, model tokens, raw user data, or generated rationale records.
The OpenRouter rationale-audit tools additionally require
`--confirm-external-data` before sending selected histories and captions.

## License

The code in this archive is released under the MIT License; see
[`LICENSE`](LICENSE).
