# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM two-stage listwise engine for OneRec SID models (paper Sec. 3 scoring protocol).

Stage 1: free reasoning in <think>, then the top-K array under a JSON grammar applied only after
</think> (reasoning_parser='qwen3'). A closed </think> followed by k distinct indices is the answer.
Stage 2: only for rows where stage 1 gave no valid list; the stage-1 reasoning is fed back with
thinking off and the same grammar, forcing a final answer.

Decoding is greedy without repetition penalty. The grammar cannot enforce distinct indices, so a
repeat is a failure; nothing is repaired, and the reasoning text is never scanned for an answer.
LoRA checkpoints are evaluated by hot-swapping adapters on one loaded base model. RFT sampling
reuses the stage-1 prompt, grammar and parser. Tested with vLLM 0.19 and 0.24.

Imports vllm at module load; call config.setup_env() first so HF_HOME resolves inside the repo.
"""
import os
import sys
from importlib.metadata import version

import numpy as np
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from utils import config
from utils.metrics import parse_topk_diag
from utils.prompts import extract_instruction, extract_system, stage1_messages

# vLLM settings (tensor parallel size defaults to the visible GPU count). 8192 tokens cover every SID
# rendering (prompts peak near 1.8k tokens at k=256, plus 2048 output tokens); caption renderings
# need --max-model-len (about 24576 at k=64).
MAX_MODEL_LEN = 8192
GPU_MEM_UTIL = 0.90


def vllm_version():
    """Installed vLLM version, recorded in every result."""
    return version("vllm")


def answer_schema(n, k):
    """JSON schema for exactly k integers in [0, n-1] (an enum, so the grammar needs no numeric
    ranges). Distinctness is not expressible; a repeat is scored as a failure downstream."""
    return {"type": "array", "items": {"enum": list(range(n))}, "minItems": k, "maxItems": k}


def load_llm(model, tp=None, eager=True, enable_lora=False, max_lora_rank=32, max_loras=1,
             max_cpu_loras=2, max_model_len=None, revision=None):
    """Load `model` (HF id or local dir, at `revision` if given) with its own chat template.
    eager=True (enforce_eager) is the setting of every reported result; enable_lora allows
    per-request LoRA adapters on the same base."""
    # ninja from this environment is needed by the grammar and compile warmup
    os.environ["PATH"] = os.path.dirname(sys.executable) + ":" + os.environ.get("PATH", "")
    extra = {}
    if enable_lora:
        extra = dict(enable_lora=True, max_loras=max_loras, max_lora_rank=max_lora_rank,
                     max_cpu_loras=max_cpu_loras)
    if revision:
        extra.update(revision=revision, tokenizer_revision=revision)
    llm = LLM(model=model, dtype="bfloat16", trust_remote_code=True,
              tensor_parallel_size=tp or config.gpu_count(), gpu_memory_utilization=GPU_MEM_UTIL,
              max_model_len=max_model_len or MAX_MODEL_LEN, enforce_eager=eager,
              reasoning_parser="qwen3", **extra)
    return llm, llm.get_tokenizer()


def stage1_prompt(tok, row, k, top_k, render="sid"):
    """Stage-1 chat prompt with thinking on (prompts.stage1_messages); returns (prompt, n_kept)."""
    msgs, n_kept = stage1_messages(row, k, render, top_k)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=True)
    return prompt, n_kept


def stage2_prompt(tok, row, stage1_text, top_k):
    """Stage-2 forced-extraction prompt with thinking off: the stage-1 reasoning plus a request for
    the top-K list only. The grammar still applies."""
    reasoning = stage1_text.split("</think>")[0] if "</think>" in (stage1_text or "") else (stage1_text or "")
    msgs = [{"role": "system", "content": extract_system()},
            {"role": "user", "content": extract_instruction(reasoning, len(row["cand_sids"]), top_k=top_k)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


_NO_ANSWER_DIAG = {"has_array": False, "arr_len": 0, "n_oob": 0, "n_dup": 0}


def parse_stage1(txt, n, k):
    """Parse only the text after a closed </think>; without one, stage 1 failed. Returns
    (picks, n_listed, diag) as metrics.parse_topk_diag."""
    if txt is None or "</think>" not in txt:
        return None, 0, dict(_NO_ANSWER_DIAG)
    return parse_topk_diag(txt.split("</think>")[-1], n, k)


def run_listwise(llm, tok, rows, k, top_k, max_tokens, extract_tokens, lora_request=None,
                 render="sid"):
    """Run stage 1 on all rows and stage 2 on the rows stage 1 did not resolve. `lora_request`
    (a LoRARequest or None) applies to both stages. Returns one result dict per row, including the
    final answer's parse diagnostics under "diag"."""
    if not rows:
        return []
    # Stage 1
    p1, kepts = [], []
    for r in rows:
        p, nk = stage1_prompt(tok, r, k, top_k, render=render)
        p1.append(p)
        kepts.append(nk)
    n_cand = len(rows[0]["cand_grades"])
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

    # Stage 2 (only the unresolved rows)
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

    # Assemble final results (stage-1 answer if it self-served, else stage-2)
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
    """Share of rows answered in stage 1, sent to stage 2 and rescued there, plus token means."""
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
    """Parse --lora-dirs "name1=path1,name2=path2" or "path1,path2" (name = basename) into
    [(name, path), ...]."""
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
