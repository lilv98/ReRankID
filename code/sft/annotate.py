# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Teacher annotation of the SFT pool with Gemini-3.6-Flash (paper Sec. 5, App. C.1).

The teacher reads the caption rendering of each slate together with the index of its one observed
positive, and writes a short rationale and a top-10 ranking that puts the positive first (the
prompt is reproduced in App. C.1). The student later imitates the same response from the
item-matched SID-only rendering. The prompt asks the teacher to refer to items only by candidate
index. Phrases that reveal the supplied answer are prohibited by the prompt and counted
(answer_reveal_audit_hit), not filtered; build_dataset.py applies the quality gates.

Input is data/sft/pool.parquet (build_pool.py --pool sft). Output is data/sft/annotations.jsonl,
one record per slate, plus <out>_summary.json with counts and token usage. Running it again
resumes: valid records are kept and failed ones are retried. The vertex backend uses Application
Default Credentials and VERTEX_PROJECT; --backend openrouter uses OPENROUTER_API_KEY.

    python code/sft/annotate.py
    python code/sft/annotate.py --backend openrouter --n 10 --out /tmp/annotations.jsonl
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
config.setup_env()

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from sft.recipe import (FORMAT_RETRIES, MAX_TOKENS, TEACHER_BACKEND, TEACHER_EFFORT, TEACHER_MODEL,
                        TEACHER_TEMPERATURE, TEACHER_THINKING_LEVEL, row_key)
from utils.data_io import dump_json, load_rows
from utils.llm import chat
from utils.metrics import parse_topk
from utils.prompts import TOP_K, build_context, clean_caption

# Informational only. This targets explicit answer disclosure and deliberately does not match broad
# Chinese words for "label" or "hint", which occur naturally in item reasoning.
ANSWER_REVEAL_AUDIT_RE = re.compile(
    r"(?:正确答案|参考答案|内部参考|被告知(?:了)?(?:答案|结果)?|"
    r"ground[\s_-]*truth|answer[\s_-]*key)",
    re.IGNORECASE,
)

_SYSTEM = (
    "你是一个模拟推荐重排序模型内部推理过程的助手。你会看到用户上下文、若干个带编号的候选，"
    "候选用文字描述呈现，以及仅供内部使用的一个真实相关候选编号。请根据用户历史写出读起来像独立"
    "判断的简明分析，并给出 top-10 编号列表。不要提及内部参考、正确答案、参考答案、被告知、"
    "ground truth 或 answer key。分析时只用「候选X」指代物品，不要复述候选描述，也不要逐项长篇罗列。"
)

_USER_TMPL = (
    "{ctx}\n\n候选物品（共 {n} 个，编号 0 到 {n_minus_1}）：\n{lines}\n\n"
    "供内部构造监督使用的相关候选编号：{gold}。其余候选没有观察到正反馈，但这不代表它们语义上"
    "完全不相关；仍需根据用户历史区分其相关程度。\n\n"
    "写作要求：\n"
    "1. 用简洁的 3-5 句话概括有历史证据支持的用户兴趣，并说明首选候选的匹配依据，不要写得过长。"
    "措辞应与历史证据的强度相称，避免使用「极其」「精准」「高度」「强烈」「完美」「必然」等夸张或"
    "绝对化判断；证据较少时使用「可能」或「一定程度」等克制措辞。\n"
    "2. 同时考虑近期兴趣和更长历史中的兴趣，但只在历史证据支持时区分近期突出兴趣、长历史中反复出现的"
    "稳定兴趣和偶然主题。不要为了覆盖近期与长期而强行编造偏好，也不要把少量出现的主题描述成稳定兴趣。"
    "历史记录本身没有候选编号，不要编造「历史X」之类的历史编号。\n"
    "3. 用「候选X」明确提到首选候选，并简要比较一到两个备选候选，说明首选为什么更合适。若比较一个，"
    "它必须是最终数组中的第二名；若比较两个，它们必须恰好是最终数组中的第二名和第三名。分析中的判断"
    "必须与最终排序一致。不要逐个候选罗列，也不要在分析中列出或复述完整 top-10。\n"
    "4. 最后按与用户完整历史的语义匹配程度对所有候选进行判断。把相关候选排在第一位；除首选外，"
    "其余候选也应分别按照它们与完整用户历史的匹配程度排序，不要仅按照它们与首选候选的相似程度排序。"
    "排序前检查历史中有多条记录支持的主要兴趣：若候选中有相应物品，应按证据纳入合适位置，不要因为它"
    "与首选主题不同而忽略；也不要为不同兴趣机械分配名额。\n"
    "5. 回复必须以 `<think>` 开头。先输出 `<think>` 和 `</think>` 包围的非空分析，再另起一行输出 JSON 数组。"
    "数组必须恰好包含 10 个互不相同且范围在 0 到 {n_minus_1} 的编号，且完整排序只能在 `</think>` 后出现一次。"
    "只输出数组或把数组放在分析前均无效。"
)

_FORMAT_RETRY_SUFFIX = (
    "\n\n输出校正：请重新独立完成上述任务。回复必须严格以 `<think>` 开始，以 `</think>` 结束非空分析，"
    "然后只输出一次包含 10 个互不相同且范围合法的候选编号的 JSON 数组，不要添加其他标签、标题或数组。"
)


def build_prompt(row):
    ctx, n_kept = build_context(row, int(row["k"]), "text")
    rendered = [clean_caption(c) for c in row["cand_captions"]]
    n = len(rendered)
    lines = "\n".join(f"[{i}] {caption}" for i, caption in enumerate(rendered))
    grades = list(row["cand_grades"])
    gold = [i for i, grade in enumerate(grades) if int(grade) > 0]
    if len(gold) != 1:
        raise AssertionError(f"SFT requires exactly one positive, found {len(gold)}")
    user = _USER_TMPL.format(ctx=ctx, n=n, n_minus_1=n - 1, lines=lines, gold=gold)
    return _SYSTEM, user, gold, n_kept


_FENCE_RE = re.compile(r"```(?:json)?", re.IGNORECASE)
_STRICT_RESPONSE_RE = re.compile(
    r"^\s*<think>\s*(?P<cot>.*?)\s*</think>\s*(?P<array>\[[^\[\]]*\])\s*$",
    re.DOTALL,
)


def parse_response(text, n):
    """Return rationale, parsed order, and number of valid listed candidate indices."""
    if not text:
        return None, None, 0
    order, n_listed = parse_topk(text, n, TOP_K)
    if order is None:
        return None, None, 0
    last_start = None
    for match in re.finditer(r"\[[^\[\]]*\]", text, flags=re.S):
        last_start = match.start()
    cot = text[:last_start] if last_start is not None else text
    cot = cot.replace("<think>", "").replace("</think>", "")
    cot = _FENCE_RE.sub("", cot).strip()
    return cot, order, n_listed


def valid_annotation(cot, order, n_listed, n_candidates, raw_text=None):
    """Whether one teacher response is safe to retain as a complete training annotation.

    Retention requires a nonempty rationale, ten distinct in-range integer candidates, and, when raw
    text is available, the requested single ``<think>...</think>`` block followed by exactly one
    JSON array.
    """
    basic = (
        isinstance(cot, str)
        and bool(cot.strip())
        and n_listed == TOP_K
        and isinstance(order, list)
        and len(order) == TOP_K
        and len(set(order)) == TOP_K
        and all(isinstance(index, int) and not isinstance(index, bool)
                and 0 <= index < n_candidates for index in order)
    )
    if not basic or raw_text is None:
        return basic
    if len(re.findall(r"\[[^\[\]]*\]", raw_text, flags=re.DOTALL)) != 1:
        return False
    match = _STRICT_RESPONSE_RE.fullmatch(raw_text)
    if not match or not match.group("cot").strip():
        return False
    try:
        raw_order = json.loads(match.group("array"))
    except json.JSONDecodeError:
        return False
    return (
        isinstance(raw_order, list)
        and len(raw_order) == TOP_K
        and len(set(raw_order)) == TOP_K
        and all(isinstance(index, int) and not isinstance(index, bool)
                and 0 <= index < n_candidates for index in raw_order)
        and raw_order == order
    )


def stored_annotation_is_valid(row):
    """Resume gate for a previously written record without re-parsing truncated raw diagnostics."""
    if row.get("error"):
        return False
    if "valid_annotation" in row:
        return row["valid_annotation"] is True
    order = row.get("order")
    return valid_annotation(
        row.get("cot_text"), order, row.get("n_listed"), len(row.get("grades") or []))


def normalized_usage(out):
    usage = out.get("usage", {}) or {}
    return {
        "prompt_tokens": int(usage.get("promptTokenCount", 0) or 0),
        "output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
        "thinking_tokens": int(usage.get("thoughtsTokenCount", 0) or 0),
    }


def annotate_one(row, args):
    """Query the teacher for one slate, with up to --format-retries corrective retries."""
    system, user, gold, n_kept = build_prompt(row)
    n_candidates = len(row["cand_grades"])
    usage = {"prompt_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
    any_answer_reveal_hit = False
    out, text, cot, order, n_listed, err = {}, None, None, None, 0, None
    is_valid = False
    attempts = 0
    for attempt_index in range(args.format_retries + 1):
        attempt_user = user if attempt_index == 0 else user + _FORMAT_RETRY_SUFFIX
        attempts += 1
        err = None
        try:
            out = chat(args.model, system, attempt_user, args.max_tokens, backend=args.backend,
                       effort=args.effort, thinking_budget=None,
                       thinking_level=args.thinking_level, temperature=args.temperature)
        except Exception as exc:
            out, err = {}, f"{type(exc).__name__}: {exc}"
        text = out.get("text")
        if not text and not err:
            err = str(out.get("err") or f"empty response (finish={out.get('finish')})")
        cot, order, n_listed = parse_response(text, n_candidates)
        is_valid = not err and valid_annotation(cot, order, n_listed, n_candidates, text)
        any_answer_reveal_hit = any_answer_reveal_hit or bool(
            cot and ANSWER_REVEAL_AUDIT_RE.search(cot))
        for name, value in normalized_usage(out).items():
            usage[name] += value
        if is_valid:
            break

    rec = {
        "uid": int(row["uid"]),
        "slate_id": str(row["slate_id"]),
        "k": int(row["k"]),
        "grades": [int(g) for g in row["cand_grades"]],
        "gold_idx": gold,
        "n_hist_kept": n_kept,
        "cot_text": cot,
        "order": order,
        "n_listed": n_listed,
        "valid_annotation": is_valid,
        "answer_reveal_audit_hit": bool(cot and ANSWER_REVEAL_AUDIT_RE.search(cot)),
        "answer_reveal_audit_hit_any_attempt": any_answer_reveal_hit,
        "finish": out.get("finish"),
        "usage": usage,
        "attempts": attempts,
        "format_retries_used": attempts - 1,
    }
    if not is_valid:
        failure = err or (f"invalid annotation after {attempts} attempt(s): "
                          f"nonempty_rationale={bool(cot and cot.strip())}, n_listed={n_listed}")
        rec["error"] = failure[:300]
        rec["raw_text"] = (text or "")[:1000]
    return rec


def load_previous(path):
    """Records of an earlier run; a line cut short by an interrupted run is skipped."""
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def summarize(rows, args):
    tokens = {"prompt_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
    for row in rows:
        for name in tokens:
            tokens[name] += int((row.get("usage") or {}).get(name, 0) or 0)
    return {
        "teacher": {"model": args.model, "backend": args.backend, "effort": args.effort,
                    "thinking_level": args.thinking_level, "temperature": args.temperature,
                    "max_tokens": args.max_tokens, "format_retries": args.format_retries},
        "rows": len(rows),
        "valid_annotation": sum(stored_annotation_is_valid(row) for row in rows),
        "errors": sum(bool(row.get("error")) for row in rows),
        "teacher_attempts": sum(int(row.get("attempts", 1) or 1) for row in rows),
        "format_retries_used": sum(int(row.get("format_retries_used", 0) or 0) for row in rows),
        "answer_reveal_audit_hits": sum(bool(row.get("answer_reveal_audit_hit")) for row in rows),
        "tokens": tokens,
    }


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool", default=config.SFT_POOL)
    ap.add_argument("--out", default=config.SFT_ANNOTATIONS)
    ap.add_argument("--n", type=int, default=None, help="annotate only the first n slates")
    ap.add_argument("--model", default=TEACHER_MODEL)
    ap.add_argument("--backend", choices=["vertex", "openrouter"], default=TEACHER_BACKEND)
    ap.add_argument("--effort", default=TEACHER_EFFORT, help="reasoning effort (OpenRouter)")
    ap.add_argument("--thinking-level", choices=("minimal", "low", "medium", "high"),
                    default=TEACHER_THINKING_LEVEL, help="Gemini thinking level (Vertex)")
    ap.add_argument("--temperature", type=float, default=TEACHER_TEMPERATURE,
                    help="sampling temperature (default: provider default)")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--format-retries", type=int, choices=(0, 1), default=FORMAT_RETRIES,
                    help="retries with a format-correction suffix after an invalid response")
    ap.add_argument("--workers", type=int, default=16)
    return ap


def main():
    args = build_arg_parser().parse_args()
    rows = sorted(load_rows(args.pool), key=row_key)
    if args.n is not None:
        rows = rows[:args.n]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    previous = load_previous(args.out)
    done = {row_key(row): row for row in previous if stored_annotation_is_valid(row)}
    if len(done) != len(previous):
        with open(args.out, "w") as f:
            for row in done.values():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"resume: kept {len(done)} valid records, retrying {len(previous) - len(done)}")
    todo = [row for row in rows if row_key(row) not in done]
    print(f"annotating {len(todo)}/{len(rows)} slates with {args.model} via {args.backend} "
          f"(workers={args.workers})")

    t0 = time.time()
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor, open(args.out, "a") as fout:
        futures = [executor.submit(annotate_one, row, args) for row in todo]
        for future in as_completed(futures):
            rec = future.result()
            completed.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if len(completed) % 25 == 0 or len(completed) == len(todo):
                valid = sum(stored_annotation_is_valid(item) for item in completed)
                print(f"  {len(completed)}/{len(todo)} done | valid={valid} | "
                      f"{time.time()-t0:.0f}s")

    records = list(done.values()) + completed
    summary = summarize(records, args)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    dump_json(summary_path, summary)
    n = max(1, summary["rows"])
    print(f"\nwrote {args.out}  rows={summary['rows']}")
    print(f"valid_annotation={summary['valid_annotation']} ({summary['valid_annotation']/n:.1%}), "
          f"errors={summary['errors']}, format retries used={summary['format_retries_used']}")
    print(f"answer_reveal_audit_hits={summary['answer_reveal_audit_hits']} (informational)")
    print(f"tokens: {summary['tokens']}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
