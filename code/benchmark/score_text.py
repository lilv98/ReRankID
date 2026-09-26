"""Text arm of the rerank benchmark: a frontier LLM reranks the candidate slate under the SAME
listwise protocol as the SID arm, scored with the same NDCG suite. The default protocol renders items
as captions with thinking ON; --render {text,sid,sid_text} swaps the item rendering (R3: frontier
under the GR2-style SID+caption view, or reading raw SID strings).

All LLM calls go through utils/llm.py — two interchangeable backends, selected by --backend
(default from RERANKID_LLM_BACKEND: "openrouter" in the cu128 profile, "vertex" in cu130):
  * "openrouter" — EVERY model is prompt+parse: the answer FORMAT is enforced by the prompt
    (rerank_instruction already asks for the array) and read back with utils.metrics.parse_topk
    (same last-JSON-array parse as the SID arm) — no schema forcing on this backend.
  * "vertex" — the original Vertex AI paths preserved: Gemini gets the responseSchema that forces the
    exact top-k integer array (dynamic thinking ON — see utils/llm.py for the thinkingBudget quirks);
    Claude (any --model containing "claude", e.g. claude-sonnet-5) uses AnthropicVertex adaptive
    thinking (--effort {low,medium,high}), with NO temperature and no forced JSON schema.
A repeat/short list is a real failure on every backend, exactly as in the SID arm — there is no
stage-2 rescue in the text arm.

Context = optional profile header + last-k watch-history items rendered as captions (the same unified
items the SID arm renders as SIDs). Scoring comes from utils/metrics.py; rendering / prompt helpers
from utils/prompts.py; bench I/O from utils/data_io.py.

Auth: the openrouter key comes from api_keys.json (utils.config); the vertex backend uses Application
Default Credentials (cloud-platform scope) for BOTH Gemini and Claude — see utils/llm.py.

Usage:
  python code/benchmark/score_text.py --n 5 --k 32                    # smoke; backend follows profile
  python code/benchmark/score_text.py --k 128 --workers 16            # full (gemini)
  python code/benchmark/score_text.py --model claude-sonnet-5 --n 5 --k 32   # smoke (claude, adaptive thinking)
  python code/benchmark/score_text.py --model claude-sonnet-5 --k 128 --effort medium
  python code/benchmark/score_text.py --backend vertex --n 5 --k 32   # original Vertex/ADC path
  python code/benchmark/score_text.py --render sid_text --n 5 --k 32  # R3: GR2-style SID+caption view
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ReRankID/code
from utils import config
config.setup_env()
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.data_io import dump_json, load_bench
from utils.llm import DEFAULT_BACKEND, OPENROUTER_IDS, PRICING, chat, is_claude, top_k_schema
from utils.metrics import aggregate, parse_topk_diag
from utils.prompts import (PROMPT_VARIANTS, TOP_K, build_context, rationale_suffix,
                           render_candidates, rerank_instruction, rerank_system)

DEFAULT_MODEL = "gemini-3.6-flash"

# ---------- per-user scoring ----------
def score_one(t):
    (model, row, k, max_tokens, top_k, effort, backend, render, emit_rationale, variant) = t
    ctx, n_kept = build_context(row, k, render)
    user = rerank_instruction(ctx, render_candidates(row, render), render, top_k=top_k,
                              variant=variant)
    if emit_rationale:
        # case-study mode: force a VISIBLE <think> rationale, DROP the schema, and align sampling to the
        # SFT-v3 teacher (temperature=None, thinking_level='medium', no thinkingBudget). See
        # prompts.rationale_suffix for why nothing short of a format instruction recovers the rationale.
        user += rationale_suffix(top_k)
        out = chat(model, rerank_system(render, variant), user, max_tokens, backend=backend, effort=effort,
                   schema=None, thinking_budget=None, thinking_level="medium", temperature=None)
        raw_cap = 4000  # keep the full rationale for case studies (headline path stays at 400)
    else:
        # schema forcing only on vertex+gemini (the original responseSchema path); openrouter and Claude
        # are prompt+parse — matching the SID arm's format handling.
        schema = top_k_schema(top_k) if (backend == "vertex" and not is_claude(model)) else None
        out = chat(model, rerank_system(render, variant), user, max_tokens, backend=backend, effort=effort,
                   schema=schema)
        raw_cap = 400
    grades = [int(g) for g in row["cand_grades"]]
    perm, n_listed, diag = parse_topk_diag(out.get("text"), len(grades), top_k)
    u = out.get("usage", {}) or {}
    usage = {"prompt_tokens": u.get("promptTokenCount", 0),
             "output_tokens": u.get("candidatesTokenCount", 0),
             "thinking_tokens": u.get("thoughtsTokenCount", 0)}
    # cost comes straight from chat(): openrouter reports the authoritative per-call cost itself,
    # the vertex paths compute it from the pricing table.
    return {"uid": int(row["uid"]), "grades": grades, "order": perm, "n_listed": n_listed,
            "n_hist_kept": n_kept, "finish": out.get("finish"), "diag": diag,
            "usage": usage, "cost_usd": out.get("cost_usd"), "raw": (out.get("text") or "")[:raw_cap]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--backend", default=DEFAULT_BACKEND, choices=["openrouter", "vertex"],
                    help="LLM backend (see utils/llm.py); default from RERANKID_LLM_BACKEND")
    ap.add_argument("--bench", default=config.BENCH)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--slice", default="test", choices=["test", "dev", "all"],
                    help="bench.parquet holds test 400 + dev 400 interleaved; headline runs score "
                         "'test', development runs 'dev'")
    ap.add_argument("--n-cand", type=int, default=30,
                    help="slate size N: render-time prefix of the stored 100-candidate priority "
                         "pool (see utils.data_io.select_slate); 30 is the anchor")
    ap.add_argument("--shuffle", type=int, default=0,
                    help="display-shuffle id; 0 = headline protocol, 1/2 = stability audit only")
    ap.add_argument("--render", default="text", choices=["text", "sid", "sid_text"],
                    help="item rendering for history AND candidates (R3: frontier model under the "
                         "GR2-style sid_text rendering; sid = frontier reading raw SID strings)")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=TOP_K)       # how many top candidates to output
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="output cap shared by thinking + answer, unified across every text-arm "
                         "system for protocol fairness (a cap, not a target; finish reasons "
                         "record truncation)")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"],
                    help="reasoning effort: Claude adaptive thinking (either backend) and OpenRouter "
                         "reasoning generally; ignored by vertex-Gemini")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--k-ndcg", type=int, default=5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-profile", action="store_true",
                    help="drop the profile header; rank from history (+candidates) only")
    ap.add_argument("--prompt-variant", default="p0", choices=PROMPT_VARIANTS,
                    help="p0 is the production wording; p1/p2/p3 are paraphrases that keep the "
                         "same task and the same literal output contract (prompt-sensitivity audit)")
    ap.add_argument("--emit-rationale", action="store_true",
                    help="CASE-STUDY mode: drop the responseSchema, append a <think> format instruction, "
                         "and align sampling to the SFT-v3 teacher (temperature=None, thinking_level=medium, "
                         "no thinkingBudget, max_tokens=8192) so gemini/claude emit a VISIBLE rationale "
                         "comparable to the student. Does NOT change the frozen headline protocol.")
    ap.add_argument("--outdir", default=config.RESULTS)
    args = ap.parse_args()

    # case-study default: align the output cap to the SFT-v3 teacher (8192) unless the user overrode it
    if args.emit_rationale and args.max_tokens == 32768:
        args.max_tokens = 8192

    os.makedirs(args.outdir, exist_ok=True)
    rows = load_bench(args.bench, n=args.n, no_profile=args.no_profile,
                      n_cand=args.n_cand, shuffle=args.shuffle, bench_slice=args.slice)
    print(f"scored rows={len(rows)} model={args.model} backend={args.backend} "
          f"N={args.n_cand} s={args.shuffle} k={args.k} "
          f"max_tokens={args.max_tokens} workers={args.workers}")

    t0 = time.time()
    tasks = [(args.model, r, args.k, args.max_tokens, args.top_k, args.effort, args.backend,
              args.render, args.emit_rationale, args.prompt_variant)
             for r in rows]
    res = [None] * len(tasks)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(score_one, t): i for i, t in enumerate(tasks)}
        for f in as_completed(futs):
            res[futs[f]] = f.result()
            done += 1
            if done % 25 == 0 or done == len(tasks):
                ok = sum(x["order"] is not None for x in res if x)
                print(f"  {done}/{len(tasks)} done | parsed={ok} | {time.time()-t0:.0f}s")

    m = aggregate(res, k_ndcg=args.k_ndcg, n_out=args.top_k)
    print("\n=== metrics (n=%d) ===" % m["n_users"])
    print(json.dumps(m, indent=2, ensure_ascii=False))

    fin = {}
    for x in res:
        fin[x["finish"]] = fin.get(x["finish"], 0) + 1
    t_in = sum(x["usage"]["prompt_tokens"] for x in res)
    t_out = sum(x["usage"]["output_tokens"] for x in res)
    t_think = sum(x["usage"]["thinking_tokens"] for x in res)
    total_cost = sum(x["cost_usd"] for x in res if x["cost_usd"] is not None)
    mlen = len(res)
    cost = {"model": args.model, "pricing_per_1M": PRICING.get(args.model),
            "input_tokens": t_in, "output_tokens": t_out, "thinking_tokens": t_think,
            "total_cost_usd": round(total_cost, 4),
            "cost_per_user_usd": round(total_cost / mlen, 5) if mlen else None,
            "mean_prompt_tokens": round(t_in / mlen, 1) if mlen else None,
            "elapsed_sec": round(time.time() - t0, 1)}
    mean_kept = round(sum(x["n_hist_kept"] for x in res) / mlen, 1) if mlen else 0
    print("\nfinish reasons:", fin)
    print(f"mean history items shown: {mean_kept}")
    print("\n=== cost ===")
    print(json.dumps(cost, indent=2, ensure_ascii=False))

    var = ((f"_{args.slice}" if args.slice != "test" else "")
           + (f"_r{args.render}" if args.render != "text" else "")
           + ("_rationale" if args.emit_rationale else "")
           + (f"_{args.prompt_variant}" if args.prompt_variant != "p0" else ""))
    tag = args.tag or f"{args.model}__text_k{args.k}_N{args.n_cand}_s{args.shuffle}{var}"
    suf = f"_n{args.n}" if args.n else ""
    path = f"{args.outdir}/{tag}{suf}.json"
    model_id = ((args.model if "/" in args.model else OPENROUTER_IDS.get(args.model, args.model))
                if args.backend == "openrouter" else args.model)
    dump_json(path, {"meta": {**vars(args), "arm": "text", "render": args.render,
                              "backend": args.backend, "model_id": model_id},
                     "metrics": m, "cost": cost, "finish_reasons": fin, "results": res})
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
