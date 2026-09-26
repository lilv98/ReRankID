"""Unified frontier-LLM chat client for the text arm and the SFT teacher annotation.

Two interchangeable backends, selected per call or via the RERANKID_LLM_BACKEND env var. The
environment setup records openrouter for the local cu128 host and vertex for the cu130 host:

  * "openrouter" (the cu128 default) — OpenRouter chat/completions, key from
    api_keys.json. Reasoning is requested via OpenRouter's unified `reasoning` param; the answer
    FORMAT is enforced by the prompt and parsed downstream (metrics.parse_topk) for EVERY model —
    no schema forcing on this backend, exactly like the SID arm and the Claude/Vertex arm.
  * "vertex" — the original Vertex AI paths, preserved verbatim: Gemini via the generateContent REST
    endpoint (ADC auth, optional responseSchema forcing, thinkingBudget quirk handling — see
    call_gemini_vertex) and Claude via AnthropicVertex (adaptive thinking + effort).

Every path returns the same shape, so scorers stay backend-agnostic:
    {"text", "finish", "usage": {promptTokenCount, candidatesTokenCount, thoughtsTokenCount},
     "cost_usd", ...}          (usage keys keep Gemini's names as the canonical vocabulary)
Retry policy is shared: exponential backoff on transient HTTP errors AND on empty visible text —
the documented Gemini "silent empty response" quirk (finish=STOP, zero text) otherwise silently
drops a large fraction of calls; see call_gemini_vertex's inline comment.

google-auth / anthropic are imported lazily inside the vertex functions, so the cu128 OpenRouter
path (and unit tests) need neither installed.
"""
import os
import threading
import time

import requests

from utils.config import api_key

_PROFILE_BACKEND = {"cu128": "openrouter", "cu130": "vertex"}
DEFAULT_BACKEND = os.environ.get(
    "RERANKID_LLM_BACKEND",
    _PROFILE_BACKEND.get(os.environ.get("RERANKID_MACHINE_PROFILE"), "openrouter"),
)

# Short model names are the cross-backend vocabulary (CLI args, result filenames, pricing); each
# backend resolves them to its own id. A name already containing "/" is passed straight to OpenRouter.
OPENROUTER_IDS = {
    "gemini-3.6-flash": "google/gemini-3.6-flash",
    "gemini-3.5-flash": "google/gemini-3.5-flash",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "claude-sonnet-5": "anthropic/claude-sonnet-5",
    "claude-opus-5": "anthropic/claude-opus-5",
    "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "gpt-5.6-terra": "openai/gpt-5.6-terra",
}

# USD per 1M tokens (Vertex AI global endpoint, checked 2026-07). "out" covers BOTH response and
# reasoning/thinking tokens (single rate). Used for the vertex backend and as an OpenRouter fallback —
# OpenRouter reports the authoritative per-call cost itself (usage.cost) which takes precedence.
PRICING = {
    "gemini-3.6-flash":    {"in": 1.50, "out": 7.50},
    "gemini-3.5-flash":    {"in": 1.50, "out": 9.00},
    "gemini-2.5-flash":    {"in": 0.30, "out": 2.50},
    "gemini-2.5-pro":      {"in": 1.25, "out": 10.00},
    "claude-sonnet-5":     {"in": 3.00, "out": 15.00},
    # Opus 5 is the rationale judge only, never a scored arm. Global endpoint, so no regional premium;
    # its newer tokenizer emits ~30% more tokens for the same text than the Gemini/Sonnet ones, which
    # shows up as a larger promptTokenCount on an identical prompt rather than as a rate difference.
    "claude-opus-5":       {"in": 5.00, "out": 25.00},
    "gpt-5.6-sol":         {"in": 5.00, "out": 30.00},
    "gpt-5.6-terra":       {"in": 2.50, "out": 15.00},
}

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 529}


def cost_usd(model, usage):
    """Table-based cost of one call from Gemini-style usage keys. thoughts billed at the output rate."""
    p = PRICING.get(model)
    if not p or not usage:
        return None
    tin = usage.get("promptTokenCount", 0)
    tout = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    return tin / 1e6 * p["in"] + tout / 1e6 * p["out"]


def is_claude(model):
    return "claude" in model.lower()


# =====================================================================================
# OpenRouter backend
# =====================================================================================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter(model, system, user, max_tokens, effort="medium", retries=5, temperature=0.0):
    """OpenRouter chat completion -> the shared return shape. `reasoning.effort` turns thinking on for
    both Gemini (mapped to its supported thinking control) and Claude (adaptive-thinking effort) — the protocol
    always reasons. Claude 5 has no temperature, so it is omitted for claude models (same rule as the
    Vertex path). usage.include asks OpenRouter to report token counts AND the authoritative cost on
    the response itself."""
    key = api_key("openrouter")
    if not key:
        raise RuntimeError("no openrouter key found (api_keys.json / OPENROUTER_API_KEY)")
    model_id = model if "/" in model else OPENROUTER_IDS.get(model, model)
    body = {
        "model": model_id,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "reasoning": {"effort": effort},
        "usage": {"include": True},
    }
    if not is_claude(model_id) and temperature is not None:
        body["temperature"] = temperature
    backoff = 2.0
    for attempt in range(retries):
        try:
            r = requests.post(OPENROUTER_URL, json=body, timeout=300,
                              headers={"Authorization": f"Bearer {key}",
                                       "Content-Type": "application/json"})
            if r.status_code != 200:
                if r.status_code in RETRY_STATUS and attempt < retries - 1:
                    time.sleep(backoff); backoff *= 2; continue
                return {"text": None, "finish": f"HTTP_{r.status_code}", "usage": {},
                        "cost_usd": None, "err": r.text[:300]}
            d = r.json()
            if "error" in d:      # OpenRouter can also surface provider errors inside a 200
                if attempt < retries - 1:
                    time.sleep(backoff); backoff *= 2; continue
                return {"text": None, "finish": "PROVIDER_ERR", "usage": {},
                        "cost_usd": None, "err": str(d["error"])[:300]}
            choice = (d.get("choices") or [{}])[0]
            txt = (choice.get("message") or {}).get("content") or ""
            if not txt and attempt < retries - 1:   # same silent-empty guard as the Vertex path
                time.sleep(backoff); backoff *= 2; continue
            u = d.get("usage") or {}
            think = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            usage = {"promptTokenCount": u.get("prompt_tokens", 0),
                     "candidatesTokenCount": max(u.get("completion_tokens", 0) - think, 0),
                     "thoughtsTokenCount": think}
            cost = u.get("cost")
            return {"text": txt, "finish": choice.get("finish_reason"), "usage": usage,
                    "cost_usd": float(cost) if cost is not None else cost_usd(model, usage)}
        except Exception as e:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(backoff); backoff *= 2; continue
            return {"text": None, "finish": "EXC", "usage": {}, "cost_usd": None, "err": str(e)[:300]}


# =====================================================================================
# Vertex AI backend (preserved from the original implementation)
# =====================================================================================
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")

_creds = None
_creds_lock = threading.Lock()


def _vertex_token():
    """ADC bearer token (cloud-platform scope), thread-safe refresh. Lazy google-auth import."""
    global _creds
    import google.auth
    from google.auth.transport.requests import Request
    with _creds_lock:
        if _creds is None:
            _creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not _creds.valid:
            _creds.refresh(Request())
        return _creds.token


def _vertex_endpoint(model):
    host = ("aiplatform.googleapis.com" if VERTEX_LOCATION == "global"
            else f"{VERTEX_LOCATION}-aiplatform.googleapis.com")
    return (f"https://{host}/v1/projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}"
            f"/publishers/google/models/{model}:generateContent")


def top_k_schema(top_k):
    """Vertex controlled-generation schema: a JSON array of exactly top_k integers (the model's picks).
    Distinctness isn't expressible; a repeat is caught downstream as a failure (same rule as the SID
    grammar). Thinking still runs internally (thoughtsTokenCount); only the response is constrained.
    VERTEX-GEMINI ONLY — the OpenRouter backend is prompt+parse for every model."""
    return {"type": "ARRAY", "items": {"type": "INTEGER"}, "minItems": top_k, "maxItems": top_k}


def call_gemini_vertex(model, system, user, max_tokens, schema=None, retries=5,
                       thinking_budget=-1, *, thinking_level=None, temperature=0.0):
    # Gemini 3.x should use thinkingLevel and the model-default sampling parameters. The optional
    # arguments let the SFT-v3 teacher do that while preserving the historical temperature=0 and
    # thinkingBudget=-1 defaults used by already-frozen benchmark calls and older models.
    # thinkingBudget=-1 => dynamic CoT (model decides how much to think). REQUIRED alongside a
    # responseSchema in the historical protocol: schema mode otherwise suppressed thinking.
    # CAVEAT (hit by sft/annotate.py): dynamic (-1) invisible thinking shares maxOutputTokens with
    # the VISIBLE response — with a small max_tokens the model can spend the whole budget on invisible
    # thoughtsTokenCount and return empty text (finish=MAX_TOKENS). Callers that want their own PROMPTED
    # visible <think>...</think> reasoning (not Gemini's separate invisible thinking channel) and no
    # thinking_level should pass thinking_budget=None to omit thinkingConfig. This makes no claim that a
    # Gemini 3 model then performs no internal thinking. Explicitly passing thinking_budget=0 was tried
    # and is BUGGY on this endpoint: it
    # reproducibly returns a response with thoughtsTokenCount>0 but zero visible text and no finishReason
    # at all, even with maxOutputTokens raised generously (verified via a raw API smoke test). Omitting
    # the key entirely does not have this problem.
    if thinking_level is not None and thinking_budget is not None:
        raise ValueError("set only one of thinking_level and thinking_budget")
    gen = {"maxOutputTokens": max_tokens}
    if temperature is not None:
        gen["temperature"] = temperature
    if thinking_level is not None:
        gen["thinkingConfig"] = {"thinkingLevel": str(thinking_level).upper()}
    elif thinking_budget is not None:
        gen["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    if schema is not None:
        gen["responseMimeType"] = "application/json"
        gen["responseSchema"] = schema
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen,
    }
    url = _vertex_endpoint(model)
    backoff = 2.0
    for attempt in range(retries):
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {_vertex_token()}",
                                            "Content-Type": "application/json"},
                              json=body, timeout=120)
            if r.status_code == 200:
                d = r.json()
                cand = (d.get("candidates") or [{}])[0]
                parts = cand.get("content", {}).get("parts", []) or []
                txt = "".join(p.get("text", "") for p in parts)
                finish = cand.get("finishReason")
                # Observed (rare, ~a few % of calls, reproducible per-input but not universal): a 200
                # response with NO finishReason and NO visible text at all -- content.parts is simply
                # absent, even with maxOutputTokens raised generously and thoughtsTokenCount far under
                # budget. No promptFeedback/blockReason present either, so not a detectable safety block
                # -- just an anomalous incomplete response. Retry it like a transient failure rather than
                # accepting an empty annotation. Retry ANY empty visible response, not just finish=None:
                # gemini-3.5-flash also returns empty text with finish=STOP (the same silent-empty
                # quirk) — those must be retried too, else a large fraction of calls silently yield
                # nothing.
                if not txt and attempt < retries - 1:
                    time.sleep(backoff); backoff *= 2; continue
                usage = d.get("usageMetadata", {})
                return {"text": txt, "finish": finish, "usage": usage,
                        "cost_usd": cost_usd(model, usage)}
            if r.status_code in (429, 500, 503) and attempt < retries - 1:
                time.sleep(backoff); backoff *= 2; continue
            return {"text": None, "finish": f"HTTP_{r.status_code}", "usage": {},
                    "cost_usd": None, "err": r.text[:300]}
        except Exception as e:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(backoff); backoff *= 2; continue
            return {"text": None, "finish": "EXC", "usage": {}, "cost_usd": None, "err": str(e)[:300]}


_claude = None
_claude_lock = threading.Lock()


def _claude_client():
    global _claude
    with _claude_lock:
        if _claude is None:
            from anthropic import AnthropicVertex
            region = os.environ.get("CLOUD_ML_REGION", VERTEX_LOCATION)  # "global" on this project
            _claude = AnthropicVertex(project_id=VERTEX_PROJECT, region=region)
        return _claude


def call_claude_vertex(model, system, user, max_tokens, effort="medium", retries=5):
    """Claude 5 on Vertex, mirroring call_gemini_vertex's return shape (usage in Gemini's key names so
    downstream stays backend-agnostic). NO temperature (deprecated for Claude 5), NO responseSchema —
    format is carried by the prompt and parsed downstream. Adaptive thinking is billed but its content
    is not returned; `output_tokens` already INCLUDES the thinking tokens, so visible = output - thinking."""
    client = _claude_client()
    think_kwargs = {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}
    backoff = 2.0
    for attempt in range(retries):
        try:
            # Stream: the Anthropic SDK refuses a non-streaming create() whose max_tokens could take
            # >10 min (the text arm's unified 32768 budget trips this). get_final_message() reassembles
            # the same Message (content/usage/stop_reason) so the return shape is unchanged.
            with client.messages.stream(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}], **think_kwargs,
            ) as stream:
                m = stream.get_final_message()
            txt = "".join(b.text for b in m.content if b.type == "text")
            u = m.usage
            think = getattr(getattr(u, "output_tokens_details", None), "thinking_tokens", 0) or 0
            usage = {"promptTokenCount": u.input_tokens,
                     "candidatesTokenCount": max(u.output_tokens - think, 0),
                     "thoughtsTokenCount": think}
            return {"text": txt, "finish": m.stop_reason, "usage": usage,
                    "cost_usd": cost_usd(model, usage)}
        except Exception as e:  # noqa: BLE001
            status = getattr(e, "status_code", None)
            retryable = status in RETRY_STATUS or status is None
            if retryable and attempt < retries - 1:
                time.sleep(backoff); backoff *= 2; continue
            return {"text": None, "finish": f"ERR_{status}" if status else "EXC", "usage": {},
                    "cost_usd": None, "err": str(e)[:300]}


# =====================================================================================
# Dispatcher
# =====================================================================================
def chat(model, system, user, max_tokens, backend=None, effort="medium", schema=None,
         thinking_budget=-1, retries=5, *, thinking_level=None, temperature=0.0):
    """One chat call through the selected backend -> the shared return shape. `schema` and
    Gemini thinking controls apply to vertex+gemini; `effort` applies to Claude on either backend
    and to OpenRouter reasoning generally. Passing ``temperature=None`` omits that sampling field."""
    backend = backend or DEFAULT_BACKEND
    if backend == "openrouter":
        return call_openrouter(model, system, user, max_tokens, effort=effort, retries=retries,
                               temperature=temperature)
    if backend == "vertex":
        if is_claude(model):
            return call_claude_vertex(model, system, user, max_tokens, effort=effort,
                                      retries=retries)
        return call_gemini_vertex(
            model, system, user, max_tokens, schema=schema, retries=retries,
            thinking_budget=thinking_budget, thinking_level=thinking_level,
            temperature=temperature)
    raise ValueError(f"unknown backend: {backend!r} (expected 'openrouter' or 'vertex')")
