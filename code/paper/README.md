# Paper figures and tables

These scripts rebuild the paper's tables and figures from the released per-user results. They run
on CPU and call no model; run them in `sid-lf`, which provides numpy, pandas, pyarrow, matplotlib
and scipy. Download the release first:

```bash
python code/download/release.py --install   # downloads/rerankid/{data,results}, copies data/ into place
```

Scripts that read results use `downloads/rerankid/results` by default; pass `--results results` to
use your own outputs, which follow the same layout. The figure scripts render text with LaTeX and
newtx; pass `--no-tex` if LaTeX is not installed.

| Paper | Command | Reads |
| --- | --- | --- |
| Tables 1, 2, D.1, D.2, D.4 | `python code/paper/make_main_tables.py` | anchor files under `diagnosis/` and `posttraining/*/*/test/`, `diagnosis/prompt-tokens.json`, `data/benchmark/bench.parquet` |
| Fig. 2 | `python code/paper/plot_scaling.py` | `diagnosis/prompt-tokens.json`, `diagnosis/{onerec-8b/sid,gemini-3.6-flash,random}/`, `posttraining/rlvr-ndcg5/onerec-8b/test/` |
| Fig. 3 | `python code/paper/plot_rationale_judge.py` | `audits/rationale-judge/pairwise.json` |
| Table B.1, RLVR reward contrast, API prompt tokens | `python code/paper/make_appendix_tables.py` | anchor files under `diagnosis/` and `posttraining/*/*/test/` |
| Tables C.2 and C.7 | `python code/utils/select_checkpoint.py <results>/posttraining/<method>/<model>/dev` | dev files of one method |
| Table D.3, Fig. 2a input | `python code/paper/measure_prompt_tokens.py` | `data/benchmark/bench.parquet` and the pinned OneRec-1.7B-pro tokenizer |
| App. D.3 | `code/benchmark/README.md` (Stability audits) | `audits/stability/` |
| App. D.4 | `code/judge/README.md` | `audits/rationale-judge/` |

Figures are written to `figures/scaling.pdf` and `figures/rationale_judge.pdf` (change with
`--out`). `measure_prompt_tokens.py` writes `results/diagnosis/prompt-tokens.json`; pass
`--tokens results/diagnosis/prompt-tokens.json` to `plot_scaling.py` to plot your own measurement.
