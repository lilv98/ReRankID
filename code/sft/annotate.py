"""Generate hindsight teacher annotations for the single-positive SFT v3 pool.

The teacher reads caption-rendered history and candidates plus the one behavior-verified positive.
It writes a concise rationale and a top-10 ranking. The student later receives the same rationale
with SID-rendered inputs. The prompt therefore asks the teacher to refer to items only by candidate
index and to rank the remaining candidates by semantic fit after placing the positive first.

This is an ordinary single-positive SFT recipe. Phrases that explicitly reveal the supplied answer
are counted as an audit diagnostic, but no regex automatically rejects training examples. Dataset
assembly applies only format, correctness, grounding, and extreme-length quality gates.

The JSONL is resumable. Resume is allowed only when the pool hash, prompt, teacher settings, and Git
state exactly match the existing provenance. Completed valid rows are retained, while API-error or
invalid rows are replaced safely. A deterministic cost summary is rewritten after every run.

Formal Vertex pilot, followed later by forward-only expansion:
  python code/sft/annotate.py --backend vertex --workers 16 --target-rows 6000
  python code/sft/annotate.py --backend vertex --workers 16 --target-rows 12000

Ten-row OpenRouter smoke, with no W&B upload:
  python code/sft/annotate.py --backend openrouter --workers 2 --n 10 \
    --pool /tmp/rerankid_sft_v3_smoke/sft_pool_v3.parquet \
    --out /tmp/rerankid_sft_v3_smoke/sft_annotations_v3.jsonl
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
config.setup_env()

import argparse
import hashlib
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from sft.v3_contract import (EXPECTED_ROWS, FORMAL_TARGET_ROWS, FORMAT_RETRIES, MAX_TOKENS,
                             PROMPT_VERSION, RECIPE, TEACHER_BACKEND, TEACHER_EFFORT, TEACHER_MODEL,
                             TEACHER_TEMPERATURE, TEACHER_THINKING_LEVEL, THINKING_BUDGET,
                             is_smoke_path)
from sft.v3_stages import row_key, select_stage_rows, validate_phase_manifest
from utils.data_io import load_rows
from utils.llm import chat
from utils.metrics import parse_topk
from utils.prompts import TOP_K, build_context, clean_caption

POOL = config.SFT_V3_POOL
OUT = config.SFT_V3_ANNOTATIONS
# Informational only. This deliberately targets explicit answer disclosure and does not match broad
# words such as "标签" or "提示", which occur naturally in item reasoning.
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
        raise AssertionError(f"SFT v3 requires exactly one positive, found {len(gold)}")
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

    The lenient parser remains useful for diagnostics, but retention requires a nonempty rationale,
    ten distinct in-range integer candidates, and, when raw text is available, the requested single
    ``<think>...</think>`` block followed by exactly one JSON array.
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


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=config.ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=config.ROOT, text=True)
    return commit, bool(status.strip())


def prompt_sha256():
    protocol = _SYSTEM + "\n" + _USER_TMPL + "\n" + _FORMAT_RETRY_SUFFIX
    return hashlib.sha256(protocol.encode("utf-8")).hexdigest()


def annotation_provenance(model, backend, effort, temperature, thinking_level, pool_sha256,
                          max_tokens, thinking_budget, format_retries, git_commit, git_dirty):
    return {
        "recipe": RECIPE,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "pool_sha256": pool_sha256,
        "teacher_model": model,
        "backend": backend,
        "effort": effort,
        "temperature": None if temperature is None else float(temperature),
        "thinking_level": thinking_level,
        "max_tokens": int(max_tokens),
        "thinking_budget": None if thinking_budget is None else int(thinking_budget),
        "format_retries": int(format_retries),
        "git_commit": git_commit,
        "git_dirty": bool(git_dirty),
    }


def annotate_one(task):
    (model, row, backend, effort, temperature, thinking_level, pool_sha256, max_tokens,
     thinking_budget, format_retries, git_commit, git_dirty) = task
    if int(format_retries) not in (0, 1):
        raise ValueError("format_retries must be 0 or 1")
    system, user, gold, n_kept = build_prompt(row)
    n_candidates = len(row["cand_grades"])
    usage = {"prompt_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
    total_cost = 0.0
    cost_attempts_reported = 0
    attempt_log = []
    retry_reasons = []
    any_answer_reveal_hit = False
    out, text, cot, order, n_listed, err = {}, None, None, None, 0, None
    is_valid = False

    for attempt_index in range(int(format_retries) + 1):
        attempt_user = user if attempt_index == 0 else user + _FORMAT_RETRY_SUFFIX
        err = None
        try:
            out = chat(model, system, attempt_user, max_tokens, backend=backend, effort=effort,
                       thinking_budget=thinking_budget, thinking_level=thinking_level,
                       temperature=temperature)
        except Exception as exc:
            out, err = {}, f"{type(exc).__name__}: {exc}"
        text = out.get("text")
        if not text and not err:
            err = str(out.get("err") or f"empty response (finish={out.get('finish')})")
        cot, order, n_listed = parse_response(text, n_candidates)
        is_valid = not err and valid_annotation(cot, order, n_listed, n_candidates, text)
        any_answer_reveal_hit = any_answer_reveal_hit or bool(
            cot and ANSWER_REVEAL_AUDIT_RE.search(cot))

        attempt_usage = normalized_usage(out)
        for name, value in attempt_usage.items():
            usage[name] += value
        attempt_cost = out.get("cost_usd")
        if attempt_cost is not None:
            total_cost += float(attempt_cost)
            cost_attempts_reported += 1
        attempt_log.append({
            "attempt": attempt_index + 1,
            "finish": out.get("finish"),
            "error": err,
            "valid_annotation": is_valid,
            "usage": attempt_usage,
            "cost_usd": attempt_cost,
        })
        if is_valid:
            break
        if attempt_index < int(format_retries):
            retry_reasons.append("api_error" if err else "invalid_annotation")

    answer_reveal_hit = bool(cot and ANSWER_REVEAL_AUDIT_RE.search(cot))
    rec = {
        "uid": int(row["uid"]),
        "slate_id": str(row.get("slate_id", 0)),
        "k": int(row["k"]),
        "grades": [int(g) for g in row["cand_grades"]],
        "gold_idx": gold,
        "n_hist_kept": n_kept,
        "cot_text": cot,
        "order": order,
        "n_listed": n_listed,
        "valid_annotation": is_valid,
        "answer_reveal_audit_hit": answer_reveal_hit,
        "answer_reveal_audit_hit_any_attempt": any_answer_reveal_hit,
        "finish": out.get("finish"),
        "usage": usage,
        "cost_usd": round(total_cost, 12) if cost_attempts_reported else None,
        "cost_attempts_reported": cost_attempts_reported,
        "attempts": len(attempt_log),
        "format_retries_used": max(0, len(attempt_log) - 1),
        "retry_reasons": retry_reasons,
        "attempt_log": attempt_log,
        "raw_text": (text or "")[:1000],
        "provenance": annotation_provenance(
            model, backend, effort, temperature, thinking_level, pool_sha256, max_tokens,
            thinking_budget, format_retries, git_commit, git_dirty),
    }
    if not is_valid:
        failure = err or (
            f"invalid annotation after {len(attempt_log)} attempt(s): "
            f"nonempty_rationale={bool(cot and cot.strip())}, n_listed={n_listed}"
        )
        rec["error"] = failure[:300]
    return rec


def cost_summary(rows, provenance):
    by_k = {}
    token_totals = {"prompt_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
    for row in rows:
        key = str(int(row["k"]))
        bucket = by_k.setdefault(key, {"rows": 0, "cost_usd": 0.0,
                                      **{name: 0 for name in token_totals}})
        bucket["rows"] += 1
        cost = row.get("cost_usd")
        if cost is not None:
            bucket["cost_usd"] += float(cost)
        usage = row.get("usage") or {}
        for name in token_totals:
            value = int(usage.get(name, 0) or 0)
            token_totals[name] += value
            bucket[name] += value
    for bucket in by_k.values():
        bucket["cost_usd"] = round(bucket["cost_usd"], 6)
    return {
        "recipe": RECIPE,
        "provenance": provenance,
        "rows": len(rows),
        "valid_format": sum(row.get("n_listed") == TOP_K for row in rows),
        "valid_annotation": sum(stored_annotation_is_valid(row) for row in rows),
        "errors": sum(bool(row.get("error")) for row in rows),
        "teacher_attempts": sum(int(row.get("attempts", 1) or 1) for row in rows),
        "format_retries_used": sum(int(row.get("format_retries_used", 0) or 0)
                                   for row in rows),
        "answer_reveal_audit_hits": sum(bool(row.get("answer_reveal_audit_hit")) for row in rows),
        "cost_rows_reported": sum(row.get("cost_usd") is not None for row in rows),
        "cost_attempts_reported": sum(int(row.get("cost_attempts_reported", 0) or 0)
                                      for row in rows),
        "total_cost_usd": round(sum(float(row["cost_usd"]) for row in rows
                                    if row.get("cost_usd") is not None), 6),
        "tokens": token_totals,
        "by_k": dict(sorted(by_k.items(), key=lambda item: int(item[0]))),
    }


def write_json_atomic(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def write_jsonl_atomic(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def default_cost_path(annotation_path):
    if os.path.realpath(annotation_path) == os.path.realpath(OUT):
        return config.SFT_V3_ANNOTATION_COST
    stem = annotation_path[:-6] if annotation_path.endswith(".jsonl") else annotation_path
    return stem + "_cost.json"


def validate_run_scope(args, cost_out, n_rows, git_dirty):
    """Separate the frozen formal run from disposable local smoke runs before any teacher call."""
    paths = (args.pool, args.out, cost_out)
    canonical = (POOL, OUT, config.SFT_V3_ANNOTATION_COST)
    path_matches = tuple(
        os.path.realpath(path) == os.path.realpath(expected)
        for path, expected in zip(paths, canonical)
    )
    if all(path_matches):
        frozen = {
            "model": (args.model, TEACHER_MODEL),
            "backend": (args.backend, TEACHER_BACKEND),
            "effort": (args.effort, TEACHER_EFFORT),
            "temperature": (args.temperature, TEACHER_TEMPERATURE),
            "thinking_level": (args.thinking_level, TEACHER_THINKING_LEVEL),
            "max_tokens": (args.max_tokens, MAX_TOKENS),
            "thinking_budget": (args.thinking_budget, THINKING_BUDGET),
            "format_retries": (args.format_retries, FORMAT_RETRIES),
        }
        changed = [name for name, (actual, expected) in frozen.items() if actual != expected]
        if changed:
            raise SystemExit(f"formal annotation settings differ from frozen v3 contract: {changed}")
        if args.n is not None:
            raise SystemExit("formal annotation cannot use --n")
        if args.target_rows not in FORMAL_TARGET_ROWS:
            raise SystemExit(
                f"formal annotation requires --target-rows in {FORMAL_TARGET_ROWS}"
            )
        if n_rows != EXPECTED_ROWS:
            raise SystemExit(f"formal annotation requires exactly {EXPECTED_ROWS} pool rows, got {n_rows}")
        if git_dirty:
            raise SystemExit("formal annotation requires a clean Git worktree")
        return "formal"
    if any(path_matches):
        raise SystemExit(
            "refusing to mix canonical and custom annotation paths; a formal run must use all three "
            "canonical paths"
        )
    if not all(is_smoke_path(path) for path in paths):
        raise SystemExit(
            "custom annotation requires --pool, --out, and --cost-out under /tmp with 'smoke' "
            "in every path"
        )
    if args.target_rows is not None:
        raise SystemExit("smoke annotation uses --n, not --target-rows")
    return "smoke"


def plan_resume(selected_rows, previous_rows, expected_provenance):
    """Validate a resume and return retained rows, invalid rows, and missing pool rows.

    A 6k previous file is a valid subset of the 12k target. The reverse direction
    is rejected because canonical annotations are forward-only and never shrink.
    """
    selected_keys = [row_key(row) for row in selected_rows]
    if len(selected_keys) != len(set(selected_keys)):
        raise SystemExit("input pool contains duplicate (uid, slate_id) keys")
    previous_keys = [row_key(row) for row in previous_rows]
    if len(previous_keys) != len(set(previous_keys)):
        raise SystemExit("existing annotation file contains duplicate (uid, slate_id) keys")
    mismatched = [row for row in previous_rows
                  if row.get("provenance") != expected_provenance]
    if mismatched:
        raise SystemExit(
            f"refusing mixed resume: {len(mismatched)}/{len(previous_rows)} rows have different "
            "provenance; use a fresh output path or deliberately delete the stale annotation file"
        )
    unknown = set(previous_keys) - set(selected_keys)
    if unknown:
        raise SystemExit(
            f"existing annotation file has {len(unknown)} keys outside the selected stage; "
            "refusing a 12k-to-6k downgrade"
        )
    invalid = [row for row in previous_rows if not stored_annotation_is_valid(row)]
    retained = [row for row in previous_rows if stored_annotation_is_valid(row)]
    seen = {row_key(row) for row in retained}
    todo = [row for row in selected_rows if row_key(row) not in seen]
    return retained, invalid, todo


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=TEACHER_MODEL)
    ap.add_argument("--backend", choices=["openrouter", "vertex"], default=TEACHER_BACKEND)
    ap.add_argument("--effort", default=TEACHER_EFFORT,
                    help="reasoning effort for OpenRouter and Claude")
    ap.add_argument("--temperature", type=float, default=TEACHER_TEMPERATURE,
                    help="sampling temperature; omit for the frozen Gemini 3 model-default policy")
    ap.add_argument("--thinking-level", choices=("minimal", "low", "medium", "high"),
                    default=TEACHER_THINKING_LEVEL,
                    help="Gemini 3 thinking level; medium matches OpenRouter effort=medium")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--thinking-budget", type=int, default=THINKING_BUDGET)
    ap.add_argument("--format-retries", type=int, choices=(0, 1), default=FORMAT_RETRIES,
                    help="bounded outer retries after an API error or invalid annotation")
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--target-rows", type=int, choices=FORMAL_TARGET_ROWS, default=None,
                    help="formal stage size; 6000 pilot or 12000 forward-only expansion")
    ap.add_argument("--n", type=int, default=None,
                    help="smoke-only row limit; requires isolated /tmp smoke output")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--cost-out", default=None)
    return ap


def main():
    args = build_arg_parser().parse_args()
    cost_out = args.cost_out or default_cost_path(args.out)

    if args.n is not None and args.n < 1:
        raise SystemExit("--n must be positive")
    if args.workers < 1 or args.max_tokens < 1:
        raise SystemExit("--workers and --max-tokens must be positive")
    if args.thinking_level is not None and args.thinking_budget is not None:
        raise SystemExit("use --thinking-level or --thinking-budget, not both")

    pool_sha256 = file_sha256(args.pool)
    git_commit, git_dirty = git_state()
    source_rows = load_rows(args.pool, n=args.n)
    run_scope = validate_run_scope(args, cost_out, len(source_rows), git_dirty)
    phase_manifest = None
    phase_manifest_sha256 = None
    if run_scope == "formal":
        with open(config.SFT_V3_PHASE_MANIFEST) as handle:
            phase_manifest = json.load(handle)
        validate_phase_manifest(phase_manifest, source_rows, pool_sha256)
        phase_manifest_sha256 = file_sha256(config.SFT_V3_PHASE_MANIFEST)
        rows = select_stage_rows(
            source_rows, args.target_rows, phase_manifest, pool_sha256
        )
    else:
        rows = source_rows
    expected_provenance = annotation_provenance(
        args.model, args.backend, args.effort, args.temperature, args.thinking_level,
        pool_sha256, args.max_tokens, args.thinking_budget, args.format_retries,
        git_commit, git_dirty)
    row_keys = [row_key(row) for row in rows]
    if len(row_keys) != len(set(row_keys)):
        raise SystemExit("input pool contains duplicate (uid, slate_id) keys")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    prev = []
    bad = 0
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    prev.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1
        print(f"resume: {len(prev)} rows already in {args.out}"
              + (f" ({bad} corrupt line(s) dropped)" if bad else ""))

    prev, invalid_prev, todo = plan_resume(rows, prev, expected_provenance)
    if bad or invalid_prev:
        write_jsonl_atomic(args.out, prev)
        print(f"resume cleanup: retained {len(prev)} valid rows and queued "
              f"{len(invalid_prev)} invalid/error row(s) for a safe retry")

    print(f"annotating {len(todo)}/{len(rows)} rows with {args.model} via {args.backend} "
          f"(scope={run_scope}, workers={args.workers}, max_tokens={args.max_tokens}, "
          f"temperature={args.temperature}, thinking_level={args.thinking_level}, "
          f"thinking_budget={args.thinking_budget}, format_retries={args.format_retries})")

    t0 = time.time()
    tasks = [(args.model, row, args.backend, args.effort, args.temperature,
              args.thinking_level, pool_sha256, args.max_tokens, args.thinking_budget,
              args.format_retries, git_commit, git_dirty) for row in todo]
    completed = []
    if tasks:
        with ThreadPoolExecutor(max_workers=args.workers) as executor, open(args.out, "a") as fout:
            futures = [executor.submit(annotate_one, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                completed.append(row)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                if len(completed) % 25 == 0 or len(completed) == len(tasks):
                    valid = sum(stored_annotation_is_valid(item) for item in completed)
                    audit_hits = sum(item["answer_reveal_audit_hit"] for item in completed)
                    errors = sum(bool(item.get("error")) for item in completed)
                    print(f"  {len(completed)}/{len(tasks)} done | valid_annotation={valid} | "
                          f"answer_reveal_audit_hits={audit_hits} | errors={errors} | "
                          f"{time.time()-t0:.0f}s")

    all_rows = prev + completed
    summary = cost_summary(all_rows, expected_provenance)
    if run_scope == "formal":
        summary.update({
            "source_pool_rows": len(source_rows),
            "target_rows": int(args.target_rows),
            "phase_manifest_sha256": phase_manifest_sha256,
            "selected_key_sha256": phase_manifest["stages"][str(args.target_rows)][
                "key_sha256"
            ],
        })
    write_json_atomic(cost_out, summary)
    n_rows = len(all_rows)
    if n_rows == 0:
        print("nothing annotated")
        return
    mean_cot_chars = sum(len(row.get("cot_text") or "") for row in all_rows) / n_rows
    print(f"\nwrote {args.out}  n={n_rows}")
    print(f"valid_format={summary['valid_format']}/{n_rows} ({summary['valid_format']/n_rows:.1%})")
    print(f"valid_annotation={summary['valid_annotation']}/{n_rows} "
          f"({summary['valid_annotation']/n_rows:.1%})")
    print(f"errors={summary['errors']}/{n_rows} ({summary['errors']/n_rows:.1%})")
    print(f"answer_reveal_audit_hits={summary['answer_reveal_audit_hits']}/{n_rows} "
          f"({summary['answer_reveal_audit_hits']/n_rows:.1%}); informational only")
    print(f"mean cot length (chars): {mean_cot_chars:.0f}")
    print(f"total cost: ${summary['total_cost_usd']:.2f}")
    print(f"wrote cost/provenance summary: {cost_out}")


if __name__ == "__main__":
    main()
