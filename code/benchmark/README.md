# Benchmark and diagnosis (Sections 3 and 4)

This directory builds the benchmark, scores models on it, and computes the paired statistics and
stability audits reported in the paper. Every scorer writes one JSON per setting with the per-user
rankings, so all paper numbers can be recomputed from the files alone. Settings are named by grid
cell `k<history>_n<slate>`; the anchor is `k64_n30`.

## Data

The released benchmark is installed with `python code/download/release.py --install`, which places
`bench.parquet`, `pools.parquet` and `stats.json` in `data/benchmark/`. To rebuild them from raw
RecIF instead (CPU only; the pinned RecIF revision reproduces them row for row):

```bash
python code/download/recif.py
python code/benchmark/build_pools.py    # user partition: test, dev and three training-side pools
python code/benchmark/build_bench.py    # 800 users with nested 100-item candidate pools
```

## Scoring (Section 4, Table 1, Figure 2)

| Script | Systems | Output |
| --- | --- | --- |
| `score_sid.py` | OneRec 1.7B and 8B under the SID, Text and Both views; trained adapters | `results/diagnosis/<model>/<view>/`, `results/posttraining/<method>/<model>/{test,dev}/` |
| `score_text.py` | Gemini-3.6-Flash, Claude-Sonnet-5, GPT-5.6-Terra | `results/diagnosis/<model>/` |
| `score_baselines.py` | Random (seed 7), Caption (Qwen3-Embedding-0.6B) | `results/diagnosis/{random,caption}/` |

```bash
# OneRec 8B: history sweep, slate sweep, and the three renderings at the anchor
python code/benchmark/score_sid.py --model onerec-8b --k-grid 0,4,8,16,32,64,128,256
for n in 10 20 50 100; do python code/benchmark/score_sid.py --model onerec-8b --n-cand $n; done
python code/benchmark/score_sid.py --model onerec-8b --render text --max-model-len 24576
python code/benchmark/score_sid.py --model onerec-8b --render sid_text --max-model-len 24576

# OneRec 1.7B: k=0 and the anchor, and the three renderings at the anchor
python code/benchmark/score_sid.py --model onerec-1.7b --k-grid 0,64
python code/benchmark/score_sid.py --model onerec-1.7b --render text --max-model-len 24576
python code/benchmark/score_sid.py --model onerec-1.7b --render sid_text --max-model-len 24576

# text-reading references (API credentials: see code/envs/README.md)
G="--backend vertex --model gemini-3.6-flash --emit-rationale"
python code/benchmark/score_text.py $G
for k in 0 4 8 16 32 128 256; do python code/benchmark/score_text.py $G --k $k; done
for n in 10 20 50 100; do python code/benchmark/score_text.py $G --n-cand $n; done
python code/benchmark/score_text.py --backend vertex --model claude-sonnet-5 --workers 16
python code/benchmark/score_text.py --backend openrouter --model gpt-5.6-terra

# reference rankers
for n in 10 20 30 50 100; do
    python code/benchmark/score_baselines.py --baseline random --n-cand $n
done
python code/benchmark/score_baselines.py --baseline caption
```

`score_sid.py` loads `onerec-1.7b` and `onerec-8b` at the revisions pinned in `code/utils/config.py`
and runs in eager mode, as for every reported result. Greedy vLLM decoding is not invariant to the
vLLM version, the tensor-parallel size or the batch composition: the same initializer scored in two
runs differs by up to about .02 NDCG@5 on dev (Table C.7, Init column), which is why each method's
initializer is rescored in its own run. Compare rescored files with `paired_stats.py` rather than
expecting the released numbers exactly. Every result records `meta.vllm`, and new runs also record
`meta.tp`. Trained models are scored with `--method <sft|rft|opd|rlvr-ndcg5|rlvr-hit1>` and
`--lora-dirs`; the stage READMEs give the exact dev-selection and test commands.

## Paired statistics (Section 3, Appendix D.1 and D.2)

```bash
python code/benchmark/paired_stats.py \
  --a results/posttraining/rlvr-ndcg5/onerec-8b/test/k64_n30.json \
  --b results/diagnosis/onerec-8b/sid/k64_n30.json --bench data/benchmark/bench.parquet
```

The CI is a user-level percentile bootstrap (10,000 resamples, seed 0) with a Wilcoxon signed-rank
test; `--metric` selects `ndcg@K`, `recall@K` or `mrr`, and `--bench` adds the grade-2 and grade-3
strata. `code/paper/make_main_tables.py` runs every paired contrast of Tables 2, D.2 and D.4.

## Stability audits (Appendix D.3)

The Gemini anchor is rescored under three more display orders, three prompt paraphrases and three
repeated decodes; the two audits then compare each change with the decode floor.

```bash
D=results/audits/stability/gemini-3.6-flash
A=results/diagnosis/gemini-3.6-flash/k64_n30.json
G="--backend vertex --model gemini-3.6-flash --emit-rationale --outdir $D"
for s in 1 2 3; do python code/benchmark/score_text.py $G --shuffle $s; done
for p in p1 p2 p3; do python code/benchmark/score_text.py $G --prompt-variant $p; done
python code/benchmark/score_text.py $G --repeat 1
python code/benchmark/score_text.py $G --shuffle 1 --repeat 1
python code/benchmark/score_text.py $G --shuffle 1 --repeat 2

python code/benchmark/stability_audit.py --results $A $D/k64_n30_s1.json $D/k64_n30_s2.json \
  --decode-repeat $D/k64_n30_s0_repeat1.json                 # results/audits/stability/display-order.json
python code/benchmark/prompt_audit.py \
  --variants p0=$A p1=$D/k64_n30_p1.json p2=$D/k64_n30_p2.json p3=$D/k64_n30_p3.json \
  --orders s0=$A s1=$D/k64_n30_s1.json s2=$D/k64_n30_s2.json s3=$D/k64_n30_s3.json \
  --repeats s0_repeat1=$D/k64_n30_s0_repeat1.json s1_repeat1=$D/k64_n30_s1_repeat1.json \
            s1_repeat2=$D/k64_n30_s1_repeat2.json            # results/audits/stability/two-axis.json
```

Both audits also run on the released files: point `A` and `D` at `downloads/rerankid/results/...`.
