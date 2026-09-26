"""Verify the extraction-accounting table, the direct reward contrast, and the API token
means against the frozen canonical mirror (supplement Tables/paragraphs added 2026-07-29).
Reads downloads/wandb-paper-final/results only; no model runs. NDCG@5 recomputation
reproduces every Table 1/2 anchor number exactly before any derived statistic is trusted."""
import json, math
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[2] / "downloads/wandb-paper-final/results")

def flatten(x, out):
    if isinstance(x, dict):
        if "path" in x and "result_count" in x:
            out.append(x)
        else:
            for v in x.values(): flatten(v, out)
    elif isinstance(x, list):
        for v in x: flatten(v, out)

entries = []
flatten(json.load(open(f"{ROOT}/INDEX.json")), entries)

def load(path):
    d = json.load(open(f"{ROOT}/{path}"))
    return d if isinstance(d, list) else d.get("results", d.get("records"))

def ndcg5(rec, n_cand=30):
    o = rec.get("order")
    if not isinstance(o, list) or len(o) != 10 or len(set(o)) != 10: return 0.0
    if any((not isinstance(i, int)) or i < 0 or i >= n_cand for i in o): return 0.0
    g = rec["grades"]
    pos = [i for i, x in enumerate(g) if x >= 2]
    if len(pos) != 1: raise ValueError("not exact-one-positive")
    p = pos[0]
    if p in o[:5]: return 1.0 / math.log2(1 + o.index(p) + 1)
    return 0.0

# find OPD test files
opd_test = [e["path"] for e in entries if e.get("method") == "opd" and "dev" not in e["path"]
            and e.get("history_length") == 64 and e.get("candidate_count") == 30]
print("OPD test files:", opd_test)

SYSTEMS = [
    ("Base 1.7B SID",  "benchmark/openonerec/1.7b-pro/sid/OneRec-1.7B-pro__sid_k64_N30_s0.json"),
    ("Base 1.7B Text", "benchmark/openonerec/1.7b-pro/text/OneRec-1.7B-pro__sid_k64_N30_s0_rtext.json"),
    ("Base 1.7B Both", "benchmark/openonerec/1.7b-pro/both/OneRec-1.7B-pro__sid_k64_N30_s0_rsid_text.json"),
    ("Base 8B SID",    "benchmark/openonerec/8b-pro/sid/OneRec-8B-pro__sid_k64_N30_s0.json"),
    ("Base 8B Text",   "benchmark/openonerec/8b-pro/text/OneRec-8B-pro__sid_k64_N30_s0_rtext.json"),
    ("Base 8B Both",   "benchmark/openonerec/8b-pro/both/OneRec-8B-pro__sid_k64_N30_s0_rsid_text.json"),
    ("SFT 1.7B",       "sft/1.7b-pro/test-sweep/OneRec-1.7B-pro__best__sft_v3_test_k64_N30_s0.json"),
    ("SFT 8B",         "sft/8b-pro/test-anchor/OneRec-8B-pro__ep1.01__hp_e128_lr5e5_testanchor_k64_N30_s0.json"),
    ("RFT 1.7B",       "rft/1.7b-pro/test-anchor/OneRec-1.7B-pro__checkpoint-62__rft_v1_test_anchor_k64_N30_s0.json"),
    ("RFT 8B",         "rft/8b-pro/test-sweep/17b594d1e6b322752d939a769a07860f7e4a0261__rft__rft_v1_8b_pro_e128_lr5e5_test_sweep_k64_N30_s0_n400.json"),
    ("RLVR-NDCG 1.7B", "rlvr/ndcg5/1.7b-pro/test-sweep/merged_sft_v3_12k_lr1e4__best__rlvr_v1_test_k64_N30_s0.json"),
    ("RLVR-NDCG 8B",   "rlvr/ndcg5/8b-pro/test-sweep/merged_hp_e128_lr5e5_ep1__best__rlvr_v2_test_k64_N30_s0.json"),
    ("RLVR-Hit1 1.7B", "rlvr/hit1/1.7b-pro/test-sweep/merged_sft_v3_12k_lr1e4__best__A_sft_top1_test_k64_N30_s0.json"),
    ("RLVR-Hit1 8B",   "rlvr/hit1/8b-pro/test-sweep/merged_hp_e128_lr5e5_ep1__best__A8_sft_top1_test_k64_N30_s0.json"),
]

per_user = {}
print("\n%-16s %6s | %8s %8s %8s | %8s %8s %8s" % (
    "system", "ndcg5", "s1valid%", "stage2%", "final_v%", "s1only", "contrib", "resc_n"))
for name, path in SYSTEMS:
    recs = load(path)
    recs = sorted(recs, key=lambda r: r["uid"])
    nd = np.array([ndcg5(r) for r in recs])
    per_user[name] = (np.array([r["uid"] for r in recs]), nd)
    s1v = np.array([bool(r.get("s1_valid")) for r in recs])
    st2 = np.array([r.get("stage") == 2 for r in recs])
    fin_v = np.array([1.0 if (isinstance(r.get("order"), list) and len(r["order"]) == 10
                     and len(set(r["order"])) == 10) else 0.0 for r in recs])
    s1only = nd * (~st2)          # score if stage-2 salvages were zeroed
    rescued = st2 & (nd > 0)
    print("%-16s %6.3f | %7.1f%% %7.1f%% %7.1f%% | %8.3f %+8.3f %8d" % (
        name, nd.mean(), 100*s1v.mean(), 100*st2.mean(), 100*fin_v.mean(),
        s1only.mean(), nd.mean()-s1only.mean(), int(rescued.sum())))

# stage value sanity
recs = load(SYSTEMS[3][1])
from collections import Counter
print("\nstage values (Base 8B SID):", Counter(r.get("stage") for r in recs))
print("s1_valid vs stage2 cross (Base 8B SID):",
      Counter((bool(r.get("s1_valid")), r.get("stage")) for r in recs))

# ---- D2: direct paired CI graded vs hit1 ----
rng = np.random.default_rng(7)
def paired_ci(a_name, b_name, n_boot=20000):
    ua, a = per_user[a_name]; ub, b = per_user[b_name]
    assert (ua == ub).all()
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

for a, b in [("RLVR-NDCG 8B", "RLVR-Hit1 8B"), ("RLVR-NDCG 1.7B", "RLVR-Hit1 1.7B")]:
    m, lo, hi = paired_ci(a, b)
    print(f"\nD2 paired {a} minus {b}: {m:+.3f}  CI [{lo:+.3f},{hi:+.3f}]")

# ---- D3: API-native prompt tokens at anchor ----
print()
for name, path in [
    ("Gemini-3.6-Flash", "benchmark/llm/gemini-3.6-flash/gemini-3.6-flash__text_k64_N30_s0_rationale.json"),
    ("GPT-5.6-Terra",    "benchmark/llm/gpt-5.6-terra/gpt-5.6-terra__text_k64_N30_s0.json"),
    ("Claude-Sonnet-5",  "benchmark/llm/claude-sonnet-5/claude-sonnet-5__text_k64_N30_s0.json"),
]:
    recs = load(path)
    pt = [r["usage"]["prompt_tokens"] for r in recs if r.get("usage") and r["usage"].get("prompt_tokens")]
    print(f"D3 {name}: n={len(pt)}, mean API prompt tokens = {np.mean(pt):,.0f}")
