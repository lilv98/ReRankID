# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Score text-reading LLMs on the benchmark under the same listwise prompt and metrics as the SID
models (paper Sec. 4, the text-reading references; App. D.3, the stability audits).

Items are rendered as captions by default (--render sid or sid_text swaps the rendering). Calls go
through utils/llm.py (--backend vertex or openrouter); every model is prompted and its reply parsed,
with no structured output. A repeated or short list is a failure, as for the SID models, and there
is no extraction stage. --emit-rationale is the protocol of the Gemini reference in the paper: a
visible <think> rationale and the SFT teacher sampling settings. Credentials: Application Default
Credentials and VERTEX_PROJECT for vertex, OPENROUTER_API_KEY (or api_keys.json) for openrouter.

Writes results/diagnosis/<model>/k<k>_n<N>.json; non-default settings add suffixes in this order:
_dev, _<render>, _s<i> (display seed), _p<i> (prompt wording), _repeat<i> (a second decode of the
same condition). --outdir overrides the directory, e.g. results/audits/stability/gemini-3.6-flash.

    python code/benchmark/score_text.py --backend vertex --model gemini-3.6-flash --emit-rationale
    python code/benchmark/score_text.py --backend vertex --model claude-sonnet-5 --workers 16
    python code/benchmark/score_text.py --backend openrouter --model gpt-5.6-terra
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
from utils.llm import DEFAULT_BACKEND, OPENROUTER_IDS, chat
from utils.metrics import aggregate, parse_topk_diag
from utils.prompts import (PROMPT_VARIANTS, TOP_K, build_context, rationale_suffix,
                           render_candidates, rerank_instruction, rerank_system)

DEFAULT_MODEL = "gemini-3.6-flash"


def score_one(t):
    """Score one user; returns the per-user result row."""
    (model, row, k, max_tokens, top_k, effort, backend, render, emit_rationale, variant) = t
    ctx, n_kept = build_context(row, k, render)
    user = rerank_instruction(ctx, render_candidates(row, render), render, top_k=top_k,
                              variant=variant)
    system = rerank_system(render, variant)
    if emit_rationale:
        # visible <think> rationale with the SFT teacher sampling (provider-default temperature,
        # thinking_level medium, no thinkingBudget); see prompts.rationale_suffix
        user += rationale_suffix(top_k)
        out = chat(model, system, user, max_tokens, backend=backend, effort=effort,
                   thinking_budget=None, thinking_level="medium", temperature=None)
        raw_cap = 4000  # keep the full rationale
    else:
        out = chat(model, system, user, max_tokens, backend=backend, effort=effort)
        raw_cap = 400
    grades = [int(g) for g in row["cand_grades"]]
    perm, n_listed, diag = parse_topk_diag(out.get("text"), len(grades), top_k)
    u = out.get("usage", {}) or {}
    usage = {"prompt_tokens": u.get("promptTokenCount", 0),
             "output_tokens": u.get("candidatesTokenCount", 0),
             "thinking_tokens": u.get("thoughtsTokenCount", 0)}
    return {"uid": int(row["uid"]), "grades": grades, "order": perm, "n_listed": n_listed,
            "n_hist_kept": n_kept, "finish": out.get("finish"), "diag": diag,
            "usage": usage, "raw": (out.get("text") or "")[:raw_cap]}


def output_path(args):
    """Result file under the release layout described in the module docstring."""
    name = args.model.split("/")[-1]
    stem = (f"k{args.k}_n{args.n_cand}"
            + (f"_{args.slice}" if args.slice != "test" else "")
            + (f"_{args.render}" if args.render != "text" else "")
            + (f"_s{args.shuffle}" if args.shuffle or args.repeat else "")
            + (f"_{args.prompt_variant}" if args.prompt_variant != "p0" else "")
            + (f"_repeat{args.repeat}" if args.repeat else "")
            + (f"_first{args.n}" if args.n else ""))
    return os.path.join(args.outdir or f"{config.RESULTS}/diagnosis/{name}", f"{stem}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="gemini-3.6-flash, claude-sonnet-5, gpt-5.6-terra or an OpenRouter id")
    ap.add_argument("--backend", default=DEFAULT_BACKEND, choices=["openrouter", "vertex"],
                    help="default: RERANKID_LLM_BACKEND or openrouter")
    ap.add_argument("--bench", default=config.BENCH)
    ap.add_argument("--slice", default="test", choices=["test", "dev", "all"])
    ap.add_argument("--n", type=int, default=None, help="score only the first n users of the slice")
    ap.add_argument("--n-cand", type=int, default=30, help="slate size N (10, 20, 30, 50 or 100)")
    ap.add_argument("--shuffle", type=int, default=0,
                    help="display-order seed; 0 is the reported protocol")
    ap.add_argument("--render", default="text", choices=["text", "sid", "sid_text"],
                    help="item rendering for history and candidates")
    ap.add_argument("--k", type=int, default=64, help="history length")
    ap.add_argument("--top-k", type=int, default=TOP_K, help="length of the requested ranking")
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="output cap shared by thinking and answer (8192 with --emit-rationale)")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"],
                    help="reasoning effort for Claude and OpenRouter models; ignored by vertex Gemini")
    ap.add_argument("--workers", type=int, default=8, help="concurrent API requests")
    ap.add_argument("--k-ndcg", type=int, default=5)
    ap.add_argument("--prompt-variant", default="p0", choices=PROMPT_VARIANTS,
                    help="p0 is the reported wording; p1 to p3 are the paraphrases of App. D.3")
    ap.add_argument("--emit-rationale", action="store_true",
                    help="Gemini reference protocol used in the paper: visible rationale, "
                         "SFT teacher sampling, max_tokens 8192 unless set")
    ap.add_argument("--repeat", type=int, default=0,
                    help="index of a repeated decode of an unchanged condition (decode floor)")
    ap.add_argument("--outdir", default=None, help="override the output directory")
    args = ap.parse_args()

    if args.emit_rationale and args.max_tokens == 32768:
        args.max_tokens = 8192

    rows = load_bench(args.bench, n=args.n, n_cand=args.n_cand, shuffle=args.shuffle,
                      bench_slice=args.slice)
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
    mlen = len(res)
    usage = {"input_tokens": t_in,
             "output_tokens": sum(x["usage"]["output_tokens"] for x in res),
             "thinking_tokens": sum(x["usage"]["thinking_tokens"] for x in res),
             "mean_prompt_tokens": round(t_in / mlen, 1) if mlen else None,
             "elapsed_sec": round(time.time() - t0, 1)}
    print("\nfinish reasons:", fin)
    print("usage:", json.dumps(usage))

    model_id = ((args.model if "/" in args.model else OPENROUTER_IDS.get(args.model, args.model))
                if args.backend == "openrouter" else args.model)
    meta = {"model": args.model.split("/")[-1], "model_id": model_id, "backend": args.backend,
            "bench": config.relpath(args.bench), "slice": args.slice, "n_cand": args.n_cand,
            "shuffle": args.shuffle, "render": args.render, "k": args.k, "top_k": args.top_k,
            "max_tokens": args.max_tokens, "effort": args.effort, "k_ndcg": args.k_ndcg,
            "prompt_variant": args.prompt_variant,
            "emit_rationale": args.emit_rationale, "arm": "text"}
    if args.repeat:
        meta["repeat"] = args.repeat
    if args.n:
        meta["n"] = args.n
    path = output_path(args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump_json(path, {"meta": meta, "metrics": m, "usage": usage, "finish_reasons": fin,
                     "results": res})
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
