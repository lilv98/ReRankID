# Rationale audit

Paper Section 5 (rationale quality), Figure 3 and Appendix D.4. A blind pairwise LLM judge compares
the rationales of two systems at the anchor (k=64, N=30): the Gemini-3.6-Flash text reference and
the 8B `rlvr-ndcg5` model. Rationales are read verbatim from the released result files; nothing
is regenerated.

For each of 100 test users (sampled with seed 56 among users that both systems ranked completely),
Claude-Opus-5 sees the caption-rendered history and slate, both anonymized rationales and both
rankings, never the observed positive, the grades or the system names. It returns A, B or tie for
item-claim accuracy, user-model accuracy, discriminative justification and coverage, plus an
overall verdict. Every pair is judged in both presentation orders, and a verdict counts only when
the two orders agree. Coverage and mention-order fidelity are also computed from the text, without
the judge. Corruption controls then damage the Gemini rationale of three judged users in three
known ways and check that the targeted aspects move and the others hold.

| File | Role |
|---|---|
| `judge_arms.py` | loads anchor results as judgeable arms; as a script, counts complete rationales per system |
| `pairwise_judge.py` | the pairwise audit (200 judge calls) |
| `pairwise_controls.py` | the corruption controls, with the main run's judge settings |

The judge is called through `code/utils/llm.py` (Vertex AI in the paper; see
`code/envs/README.md` for credentials). By default all three scripts read the released results
under `downloads/rerankid/results`; pass `--results results` for your own, or `--arm LABEL=PATH`.

```bash
python code/judge/judge_arms.py                       # complete rationales per system
python code/judge/pairwise_judge.py --backend vertex  # results/audits/rationale-judge/pairwise.json
python code/judge/pairwise_controls.py --run results/audits/rationale-judge/pairwise.json
                                                      # results/audits/rationale-judge/controls.json
python code/paper/plot_rationale_judge.py --results results    # Figure 3
```

The released `audits/rationale-judge/{pairwise,controls}.json` hold the paper's verdicts, token
usage and computed statistics; `plot_rationale_judge.py` renders Figure 3 from them directly.
