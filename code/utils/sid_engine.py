"""The vLLM two-stage listwise generation engine for OneRec SID models. The eval scorer uses both
stages, while the RFT sampler shares its stage-1 prompt, grammar, and honest parser but retains only
completed single-stage policy rollouts. Supported project profiles use vLLM 0.19 (cu128) or 0.24
(cu130).

  stage 1 — FREE CoT + reasoning-aware STRUCTURED OUTPUT (reasoning_parser='qwen3'): the model thinks
            (<think>...</think>) then emits its TOP-K picks as a JSON array whose FORMAT is
            grammar-constrained (exactly k ints in [0,n-1]). If it self-terminates with a valid top-K
            list (k distinct indices), that is the answer — no second call.
  stage 2 — FORCED extraction (only for rows where stage 1 did NOT yield a valid top-K list): feed the
            stage-1 reasoning back, thinking OFF, same JSON-array grammar. Rescues the loops (CoT never
            closed </think>) and the non-distinct answers by making the model commit.

We RECORD the split: what fraction self-served in stage 1 vs. needed the stage-2 fallback, and how
often stage 2 rescued a valid permutation.

Fairness rules (unchanged):
  * GREEDY, NO repetition_penalty — a CoT loop is a genuine capability failure, not something to mask.
  * The grammar can't enforce distinctness; a repeated index is a real failure. NO repair — an
    incomplete/duplicate ranking (even after stage 2) is scored 0 per the strict rule.
  * HONEST parse: stage-1 answer is read ONLY after a closed </think> (no </think> => stage-1 failure ->
    stage 2). We never scan the reasoning (it echoes the [0..n-1] candidate labels and would fake a
    complete ranking).

Also supports evaluating LoRA checkpoints via vLLM's multi-LoRA hot-swap: the base model is loaded
once, adapters are swapped per request, no merge-and-save per checkpoint.

Import this module only from GPU entry scripts (it imports vllm at module load), and call
config.setup_env() BEFORE importing it so HF_HOME resolves inside the repo.
"""
import os
import sys
from importlib.metadata import version

import numpy as np
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from utils import config
from utils.metrics import parse_topk_diag
from utils.prompts import (build_context, extract_instruction, extract_system,
                           render_candidates, rerank_instruction, rerank_system)

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "templates", "qwen3_soft_switch.jinja2")

# vLLM serving knobs for OneRec on A100-80GB (tensor-parallel size defaults to the visible GPU count).
# 8192 covers every SID-render condition with ~2x headroom (prompt peaks at ~1.8k tokens at k=256,
# + 2048 output); caption renderings exceed it at ANY k and must pass --max-model-len (~24576 at the
# anchor k=64, matching the C3 text-student training cutoff).
MAX_MODEL_LEN = 8192
GPU_MEM_UTIL = 0.90


def vllm_version():
    """Installed version, recorded in every result for cross-profile auditability."""
    return version("vllm")


def answer_schema(n, k):
    """JSON array of exactly k integers each in [0, n-1] (enum -> grammar-native, no numeric ranges) —
    the model's top-k picks. Distinctness is NOT expressible in the grammar; a repeat (n_listed < k) is
    caught downstream as a failure."""
    return {"type": "array", "items": {"enum": list(range(n))}, "minItems": k, "maxItems": k}


def load_llm(model, tp=None, eager=True, enable_lora=False, max_lora_rank=32, max_loras=1,
             max_cpu_loras=2, max_model_len=None):
    # reasoning_parser='qwen3' -> structured output applies only to the answer AFTER </think>, leaving
    # the CoT free. eager=True is the reliable default; eager=False enables torch.compile+cudagraphs
    # (faster, needs ninja on PATH — handled below) at the cost of a longer warmup.
    # enable_lora lets the SAME loaded base model be evaluated against multiple LoRA checkpoints via a
    # per-request LoRARequest — no merge-and-save per checkpoint, no reloading the base weights. No-op
    # (default) for all normal invocations; confirmed no conflict with TP/structured_outputs/
    # reasoning_parser/enforce_eager in vLLM 0.24's source.
    # put this env's bin (ninja, needed by the V1 compile/grammar warmup) on PATH
    os.environ["PATH"] = os.path.dirname(sys.executable) + ":" + os.environ.get("PATH", "")
    lora_kwargs = {}
    if enable_lora:
        lora_kwargs = dict(enable_lora=True, max_loras=max_loras, max_lora_rank=max_lora_rank,
                           max_cpu_loras=max_cpu_loras)
    llm = LLM(model=model, dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=tp or config.gpu_count(), gpu_memory_utilization=GPU_MEM_UTIL,
              max_model_len=max_model_len or MAX_MODEL_LEN, enforce_eager=eager,
              reasoning_parser="qwen3", **lora_kwargs)
    tok = llm.get_tokenizer()
    with open(TEMPLATE) as f:
        tok.chat_template = f.read()
    return llm, tok


def stage1_prompt(tok, row, k, top_k, render="sid"):
    """Listwise top-K prompt: free CoT then the grammar-constrained answer array. `render` picks the
    item modality (sid / text / sid_text) for BOTH the history and the candidates — the matched-model
    decomposition runs the same engine under all three."""
    ctx, n_kept = build_context(row, k, render)
    user = rerank_instruction(ctx, render_candidates(row, render), render, top_k=top_k)
    msgs = [{"role": "system", "content": rerank_system(render)},
            {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=True)
    return prompt, n_kept


def stage2_prompt(tok, row, stage1_text, top_k):
    """Forced-extraction prompt, thinking OFF: feed stage-1 reasoning back, demand only the top-K list.
    (Verified: no_think + StructuredOutputsParams still enforces the grammar under reasoning_parser.)"""
    reasoning = stage1_text.split("</think>")[0] if "</think>" in (stage1_text or "") else (stage1_text or "")
    msgs = [{"role": "system", "content": extract_system()},
            {"role": "user", "content": extract_instruction(reasoning, len(row["cand_sids"]), top_k=top_k)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


_NO_ANSWER_DIAG = {"has_array": False, "arr_len": 0, "n_oob": 0, "n_dup": 0}


def parse_stage1(txt, n, k):
    """HONEST parse: require a closed </think>, then parse ONLY the answer segment after it — no
    </think> means the CoT never produced an answer -> stage-1 failure (None, 0). Returns
    (picks, n_listed, diag) with diag from metrics.parse_topk_diag (the CIR validity breakdown)."""
    if txt is None or "</think>" not in txt:
        return None, 0, dict(_NO_ANSWER_DIAG)
    return parse_topk_diag(txt.split("</think>")[-1], n, k)


def run_listwise(llm, tok, rows, k, top_k, max_tokens, extract_tokens, lora_request=None,
                 render="sid"):
    """Stage 1 on all rows; stage 2 (forced extraction) only on the rows stage 1 didn't resolve.
    `lora_request` (vllm.lora.request.LoRARequest or None) is threaded into BOTH generate() calls so a
    LoRA-checkpoint sweep exercises the exact same two-stage logic as the base-model runs. `render`
    (see stage1_prompt) selects the matched-model rendering; each row's final-answer parse diagnostics
    are recorded under "diag" (aggregated by metrics.aggregate into the CIR validity block)."""
    if not rows:
        return []
    # ---- stage 1 ----
    p1, kepts = [], []
    for r in rows:
        p, nk = stage1_prompt(tok, r, k, top_k, render=render)
        p1.append(p)
        kepts.append(nk)
    n_cand = len(rows[0]["cand_grades"]) if rows else 30
    so = StructuredOutputsParams(json=answer_schema(n_cand, top_k))
    o1 = llm.generate(p1, SamplingParams(temperature=0.0, max_tokens=max_tokens, structured_outputs=so),
                      lora_request=lora_request)

    stage1 = []          # per-row dict from stage 1
    todo = []            # indices needing stage 2
    for i, (r, o) in enumerate(zip(rows, o1)):
        t1 = o.outputs[0].text
        n = len(r["cand_grades"])
        perm1, nl1, diag1 = parse_stage1(t1, n, top_k)
        closed = "</think>" in (t1 or "")
        s1_valid = (perm1 is not None and nl1 >= top_k)
        stage1.append({"text": t1, "perm": perm1, "n_listed": nl1, "closed": closed,
                       "valid": s1_valid, "diag": diag1, "out_tokens": len(o.outputs[0].token_ids),
                       "prompt_tokens": len(o.prompt_token_ids)})
        if not s1_valid:
            todo.append(i)

    # ---- stage 2 (only the unresolved rows) ----
    stage2 = {i: None for i in todo}
    if todo:
        p2 = [stage2_prompt(tok, rows[i], stage1[i]["text"], top_k) for i in todo]
        o2 = llm.generate(p2, SamplingParams(temperature=0.0, max_tokens=extract_tokens,
                                             structured_outputs=so), lora_request=lora_request)
        for i, o in zip(todo, o2):
            t2 = o.outputs[0].text
            perm2, nl2, diag2 = parse_topk_diag(t2, len(rows[i]["cand_grades"]), top_k)
            stage2[i] = {"text": t2, "perm": perm2, "n_listed": nl2, "diag": diag2,
                         "valid": (perm2 is not None and nl2 >= top_k),
                         "out_tokens": len(o.outputs[0].token_ids)}

    # ---- assemble final results (stage-1 answer if it self-served, else stage-2) ----
    res = []
    for i, (r, nk) in enumerate(zip(rows, kepts)):
        s1 = stage1[i]
        grades = [int(g) for g in r["cand_grades"]]
        if s1["valid"]:
            order, n_listed, stage, diag = s1["perm"], s1["n_listed"], 1, s1["diag"]
        else:
            s2 = stage2[i]
            order, n_listed, stage, diag = s2["perm"], s2["n_listed"], 2, s2["diag"]
        res.append({
            "uid": int(r["uid"]), "grades": grades, "order": order, "n_listed": n_listed,
            "stage": stage,                                  # 1 = self-served, 2 = forced extraction
            "s1_closed_think": s1["closed"], "s1_valid": s1["valid"],
            "s1_out_tokens": s1["out_tokens"], "prompt_tokens": s1["prompt_tokens"],
            "s2_out_tokens": (stage2[i]["out_tokens"] if stage == 2 else 0),
            "n_hist_kept": nk, "diag": diag,
            "raw1": (s1["text"] or "")[:400],
            "raw2": ((stage2[i]["text"] or "")[:200] if stage == 2 else ""),
        })
    return res


def two_stage_stats(res, top_k):
    """The recorded split: stage-1 self-serve vs. stage-2 fallback, and rescue rate."""
    m = len(res)
    if not m:
        return {"stage1_valid_rate": 0.0, "two_stage_rate": 0.0,
                "stage2_rescue_rate": None, "stage1_loop_rate": 0.0,
                "n_stage1_selfserved": 0, "n_two_stage": 0, "n_stage2_rescued": 0,
                "mean_prompt_tokens": 0.0, "mean_s1_out_tokens": 0.0, "mean_s2_out_tokens": 0.0}
    s1_valid = sum(1 for r in res if r["s1_valid"])
    used_s2 = [r for r in res if r["stage"] == 2]
    s2_rescued = sum(1 for r in res if r["stage"] == 2 and r["n_listed"] >= top_k)
    s1_looped = sum(1 for r in res if not r["s1_closed_think"])       # CoT never closed </think>
    return {
        "stage1_valid_rate": round(s1_valid / m, 4),                  # self-served, no 2nd call
        "two_stage_rate": round(len(used_s2) / m, 4),                 # fraction that needed stage 2
        "stage2_rescue_rate": round(s2_rescued / len(used_s2), 4) if used_s2 else None,
        "stage1_loop_rate": round(s1_looped / m, 4),                  # </think> never closed in stage 1
        "n_stage1_selfserved": s1_valid, "n_two_stage": len(used_s2), "n_stage2_rescued": s2_rescued,
        "mean_prompt_tokens": round(float(np.mean([r.get("prompt_tokens", 0) for r in res])), 1),
        "mean_s1_out_tokens": round(float(np.mean([r["s1_out_tokens"] for r in res])), 1),
        "mean_s2_out_tokens": round(float(np.mean([r["s2_out_tokens"] for r in res if r["stage"] == 2])), 1)
        if used_s2 else 0.0,
    }


def parse_lora_dirs(spec):
    """--lora-dirs "name1=path1,name2=path2" or bare "path1,path2" (name = basename). Returns
    [(name, path), ...], empty list if spec is falsy."""
    if not spec:
        return []
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            name, path = os.path.basename(item.rstrip("/")), item
        out.append((name, path))
    return out
