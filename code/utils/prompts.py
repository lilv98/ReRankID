# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Item rendering and the exact listwise prompts (paper Sec. 3 and the prompt appendix).

Shared by scoring, SFT/RFT dataset assembly and OPD/RLVR data building, so a rendering never
differs between arms. `modality` is "sid" (Semantic IDs), "text" (captions) or "sid_text" (SID
followed by its caption); every helper accepts all three. Prompt strings must stay unchanged:
released results depend on them byte for byte.
"""

MODALITIES = ("sid", "text", "sid_text")

# The model returns an ordered list of its top-K candidates, not a full permutation.
TOP_K = 10

CAP_PLACEHOLDER = "（该视频暂无文字描述）"

_RERANK_SYS = ("你是一个推荐重排序引擎。给定用户的已知信息（可能为空）和一组用{rep}表示的候选物品，"
               "请根据用户兴趣把候选按相关性从高到低排序。")

# Paraphrases for the prompt-wording audit. Each states the same task and the same output contract
# (a JSON array of exactly top_k distinct indices in 0..n-1, most relevant first) in different words.
# "p0" is the reported prompt (_RERANK_SYS and the tail built in rerank_instruction).
PROMPT_VARIANTS = ("p0", "p1", "p2", "p3")

_RERANK_SYS_PARAPHRASE = {
    "p1": ("你的任务是给推荐结果重新排序。下面会给出这个用户的资料（有可能没有）以及若干条用{rep}"
           "呈现的候选内容，请结合用户的兴趣，把这些候选从最相关排到最不相关。"),
    "p2": ("作为排序模块，你需要处理一批候选物品。用户信息可能提供也可能缺失，候选以{rep}的形式给出。"
           "请判断每个候选与该用户兴趣的契合程度，并据此由高到低排列。"),
    "p3": ("请扮演推荐系统里负责重排的组件。已知信息包括用户画像（可能为空）和一组{rep}形式的候选。"
           "目标是按照用户可能的兴趣程度对候选排序，最感兴趣的放前面。"),
}


# Item rendering
def sid_str(code3):
    """Render a (a, b, c) codebook triple as the OneRec SID token string."""
    a, b, c = code3
    return f"<|sid_begin|><s_a_{int(a)}><s_b_{int(b)}><s_c_{int(c)}><|sid_end|>"


def is_empty(x):
    """True for None / blank / the literal strings none|nan|null (case-insensitive)."""
    s = "" if x is None else str(x).strip()
    return s == "" or s.lower() in ("none", "nan", "null")


def clean_caption(c, placeholder=CAP_PLACEHOLDER):
    return placeholder if is_empty(c) else str(c).strip()


def render_last_k_from_arrays(hist_pids, hist_sids, hist_caps, k, modality):
    """Keep the last k unified history items (both a SID and a caption available) in order and render
    each as its SID, its bracketed caption, or both, so every rendering at the same k sees the same
    items. k <= 0 gives an empty history. Returns (text, n_kept).
    """
    uni = [i for i, (s, c) in enumerate(zip(hist_sids, hist_caps)) if s and c]
    keep = uni[-k:] if k and k > 0 else []
    if modality == "sid":
        return "".join(hist_sids[i] for i in keep), len(keep)
    if modality == "sid_text":
        parts = ["%s「%s」" % (hist_sids[i], hist_caps[i]) for i in keep]
    else:
        parts = ["「%s」" % hist_caps[i] for i in keep]
    return "；".join(parts), len(keep)


def render_candidates(row, modality):
    """Render the slate's candidates: 'sid' gives the SID token string, 'text' the cleaned caption
    (placeholder if empty, which never happens on benchmark slates), 'sid_text' the SID followed by
    the bracketed caption."""
    if modality == "sid":
        return list(row["cand_sids"])
    if modality == "text":
        return [clean_caption(c) for c in row["cand_captions"]]
    if modality == "sid_text":
        return ["%s「%s」" % (s, clean_caption(c))
                for s, c in zip(row["cand_sids"], row["cand_captions"])]
    raise ValueError(f"unknown modality: {modality!r} (expected one of {MODALITIES})")


# Listwise prompt
def rerank_system(modality, variant="p0"):
    """The system prompt; the only wording that differs between arms is the rendering noun.

    `variant` selects a paraphrase for the prompt-sensitivity audit and defaults to p0, the reported
    wording.
    """
    rep = {"sid": "语义ID", "text": "文字描述", "sid_text": "语义ID及其文字描述"}[modality]
    template = _RERANK_SYS if variant == "p0" else _RERANK_SYS_PARAPHRASE[variant]
    return template.format(rep=rep)


def build_context(row, k, modality):
    """The last-k history rendered for `modality`. Returns (text, n_kept)."""
    hist, n_kept = render_last_k_from_arrays(row["hist_pids"], row["hist_sids_str"],
                                             row["hist_caps"], k, modality)
    if not hist:
        return "（暂无用户历史信息）", n_kept
    return "用户最近观看的视频（按时间顺序）：" + hist, n_kept


_INSTRUCTION_PARAPHRASE = {
    "p1": lambda n, top_k: (
        f"请先简要说明这位用户的兴趣，以及每个候选跟这些兴趣对得上多少；提到具体物品时用它的编号，"
        f"比如「候选7」。分析写完之后，从这 {n} 个候选里选出最贴合的 {top_k} 个，越贴合的排越前面，"
        f"并以 JSON 数组的形式给出结果：数组里正好 {top_k} 个候选编号，彼此不重复，都落在 0 到 {n - 1} "
        f"之间，形如 [<最相关的编号>, <次相关的编号>, …]。"),
    "p2": lambda n, top_k: (
        f"先用简短的篇幅分析一下用户的偏好和各候选的匹配情况，行文中请以编号（例如「候选7」）称呼物品。"
        f"分析结束后，在这 {n} 个候选中确定相关性最高的 {top_k} 个并按相关性降序排好，"
        f"最后输出一个 JSON 数组，其中恰好有 {top_k} 个互不相同、取值在 0 到 {n - 1} 区间内的候选编号，"
        f"格式为 [<最相关的编号>, <次相关的编号>, …]。"),
    "p3": lambda n, top_k: (
        f"请简明地判断用户的兴趣所在，并说明候选与之匹配的程度，说到某个物品时用「候选7」这样的编号来指代。"
        f"思考完成后，把这 {n} 个候选中你认为最相关的 {top_k} 个挑出来，按从最相关到较不相关的顺序排列，"
        f"用一个 JSON 数组输出：恰好 {top_k} 个不重复的候选编号，取值范围 0 到 {n - 1}，"
        f"形如 [<最相关的编号>, <次相关的编号>, …]。"),
}


def rerank_instruction(context, rendered_cands, modality, top_k=TOP_K, variant="p0"):
    """Listwise top-K instruction: context, numbered candidates, then the task. The model reasons
    in <think>, refers to items by candidate number, and ends with a JSON array of top_k distinct
    indices. The format example is schematic (no literal digits) so it cannot be copied as an answer.
    Decoding constrains the array format separately."""
    n = len(rendered_cands)
    lines = "\n".join(f"[{i}] {c}" for i, c in enumerate(rendered_cands))
    head = f"{context}\n\n候选物品（共 {n} 个，编号 0 到 {n - 1}）：\n{lines}\n\n"
    if variant != "p0":
        return head + _INSTRUCTION_PARAPHRASE[variant](n, top_k)
    tail = (f"从这 {n} 个候选中挑出你认为"
            f"最相关的 {top_k} 个，按相关性从高到低排列（最相关在最前）。"
            f"输出一个 JSON 数组，恰好包含 {top_k} 个互不相同的候选编号（取值 0 到 {n - 1}），"
            f"形如 [<最相关的编号>, <次相关的编号>, …]。")
    return (head + "请简明扼要地分析用户兴趣与各候选的匹配程度，分析时请用候选编号（如「候选7」）指代物品，"
            "然后在思考结束后，" + tail)


def stage1_messages(row, k, render="sid", top_k=TOP_K):
    """System and user turns of the stage-1 listwise prompt, plus the number of history items kept.
    `render` applies to both the history and the candidates."""
    ctx, n_kept = build_context(row, k, render)
    user = rerank_instruction(ctx, render_candidates(row, render), render, top_k=top_k)
    return [{"role": "system", "content": rerank_system(render)},
            {"role": "user", "content": user}], n_kept


def rationale_suffix(top_k=TOP_K):
    """Suffix for text-reading LLMs in rationale mode (score_text --emit-rationale). These models
    keep their reasoning hidden, so the suffix asks for a visible <think> analysis before the array,
    matching what OneRec emits. It constrains only the output format and reveals no answer."""
    return (
        "\n\n输出格式：回复必须以 `<think>` 开头。先输出 `<think>` 和 `</think>` 包围的非空分析"
        "（分析中用「候选X」指代物品），再另起一行只输出一次 JSON 数组，"
        f"数组恰好包含 {top_k} 个互不相同、取值合法的候选编号；完整排序只能在 `</think>` 之后出现一次。"
    )


# Stage-2 extraction prompt (two-stage fallback)
_EXTRACT_SYS = "你是一个排序结果抽取器：读取一段排序分析，输出最终的编号排列。"


def extract_system():
    return _EXTRACT_SYS


def extract_instruction(reasoning, n, top_k=TOP_K, cap=1200):
    """Stage-2 (forced-answer) prompt: feed the stage-1 reasoning back (capped) and demand only the
    final top-K list. Used when stage 1 never committed to a valid top-K ranking."""
    reasoning = (reasoning or "")[:cap]
    return (f"任务：从 {n} 个候选（编号 0 到 {n - 1}）中选出最相关的 {top_k} 个，并从高到低排序。\n"
            f"以下是之前的分析过程：\n{reasoning}\n\n"
            f"请只输出一个 JSON 数组，恰好包含 {top_k} 个互不相同的编号（取值 0 到 {n - 1}），最相关在前。")
