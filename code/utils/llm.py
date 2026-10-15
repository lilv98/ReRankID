# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Chat client for the text-reading LLM references, the SFT teacher and the rationale judge.

Two backends, chosen per call or by RERANKID_LLM_BACKEND (default openrouter):
"openrouter" sends every model through OpenRouter chat/completions with its unified `reasoning`
parameter (key: OPENROUTER_API_KEY or api_keys.json); "vertex" calls Gemini through the
generateContent REST endpoint and Claude through AnthropicVertex, using Application Default
Credentials and VERTEX_PROJECT / VERTEX_LOCATION. The output format is set by the prompt and parsed
downstream; no call uses structured output.

Every call returns {"text", "finish", "usage"} with usage in Gemini key names (promptTokenCount,
candidatesTokenCount, thoughtsTokenCount), plus "err" on failure. Transient HTTP errors and empty
responses are retried with exponential backoff. google-auth and anthropic are imported only by the
Vertex paths.
"""
import os
import threading
import time

import requests

from utils.config import api_key

DEFAULT_BACKEND = os.environ.get("RERANKID_LLM_BACKEND", "openrouter")

# Short model names used on the command line and in result files; a name containing "/" is passed to
# OpenRouter unchanged.
OPENROUTER_IDS = {
    "gemini-3.6-flash": "google/gemini-3.6-flash",
    "claude-sonnet-5": "anthropic/claude-sonnet-5",
    "claude-opus-5": "anthropic/claude-opus-5",
    "gpt-5.6-terra": "openai/gpt-5.6-terra",
}

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 529}


def is_claude(model):
    return "claude" in model.lower()


def backoff(attempt):
    """Sleep before retry `attempt` + 1: 2, 4, 8, ... seconds."""
    time.sleep(2.0 * 2 ** attempt)


# =====================================================================================
# OpenRouter backend
# =====================================================================================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter(model, system, user, max_tokens, effort="medium", retries=5, temperature=0.0):
    """One OpenRouter chat completion. `reasoning.effort` turns on thinking for every model; Claude
    takes no temperature, so it is omitted for Claude models."""
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
    for attempt in range(retries):
        last = attempt == retries - 1
        try:
            r = requests.post(OPENROUTER_URL, json=body, timeout=300,
                              headers={"Authorization": f"Bearer {key}",
                                       "Content-Type": "application/json"})
            if r.status_code != 200:
                if r.status_code in RETRY_STATUS and not last:
                    backoff(attempt)
                    continue
                return {"text": None, "finish": f"HTTP_{r.status_code}", "usage": {},
                        "err": r.text[:300]}
            d = r.json()
            if "error" in d:      # provider errors can arrive inside a 200
                if not last:
                    backoff(attempt)
                    continue
                return {"text": None, "finish": "PROVIDER_ERR", "usage": {},
                        "err": str(d["error"])[:300]}
            choice = (d.get("choices") or [{}])[0]
            txt = (choice.get("message") or {}).get("content") or ""
            if not txt and not last:   # retry empty responses
                backoff(attempt)
                continue
            u = d.get("usage") or {}
            think = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            usage = {"promptTokenCount": u.get("prompt_tokens", 0),
                     "candidatesTokenCount": max(u.get("completion_tokens", 0) - think, 0),
                     "thoughtsTokenCount": think}
            return {"text": txt, "finish": choice.get("finish_reason"), "usage": usage}
        except Exception as e:  # noqa: BLE001
            if not last:
                backoff(attempt)
                continue
            return {"text": None, "finish": "EXC", "usage": {}, "err": str(e)[:300]}


# =====================================================================================
# Vertex AI backend
# =====================================================================================
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")

_creds = None
_creds_lock = threading.Lock()


def _vertex_token():
    """ADC bearer token (cloud-platform scope) with thread-safe refresh."""
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


def call_gemini_vertex(model, system, user, max_tokens, retries=5, thinking_budget=-1, *,
                       thinking_level=None, temperature=0.0):
    """Gemini through generateContent. The default is temperature 0 with dynamic thinking
    (thinking_budget=-1). Callers that prompt for a visible <think> rationale pass
    thinking_budget=None with a thinking_level and temperature=None (model-default sampling);
    thinking_budget=0 is not supported, since it returns empty text."""
    if thinking_level is not None and thinking_budget is not None:
        raise ValueError("set only one of thinking_level and thinking_budget")
    gen = {"maxOutputTokens": max_tokens}
    if temperature is not None:
        gen["temperature"] = temperature
    if thinking_level is not None:
        gen["thinkingConfig"] = {"thinkingLevel": str(thinking_level).upper()}
    elif thinking_budget is not None:
        gen["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen,
    }
    url = _vertex_endpoint(model)
    for attempt in range(retries):
        last = attempt == retries - 1
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
                # Retry an empty visible response (a rare 200 with no text) instead of accepting it.
                if not txt and not last:
                    backoff(attempt)
                    continue
                usage = d.get("usageMetadata", {})
                return {"text": txt, "finish": finish, "usage": usage}
            if r.status_code in RETRY_STATUS and not last:
                backoff(attempt)
                continue
            return {"text": None, "finish": f"HTTP_{r.status_code}", "usage": {},
                    "err": r.text[:300]}
        except Exception as e:  # noqa: BLE001
            if not last:
                backoff(attempt)
                continue
            return {"text": None, "finish": "EXC", "usage": {}, "err": str(e)[:300]}


_claude = None
_claude_lock = threading.Lock()


def _claude_client():
    global _claude
    with _claude_lock:
        if _claude is None:
            from anthropic import AnthropicVertex
            region = os.environ.get("CLOUD_ML_REGION", VERTEX_LOCATION)
            _claude = AnthropicVertex(project_id=VERTEX_PROJECT, region=region)
        return _claude


def call_claude_vertex(model, system, user, max_tokens, effort="medium", retries=5):
    """Claude on Vertex with adaptive thinking at `effort` and no temperature. Thinking
    content is not returned, and output_tokens includes thinking tokens, so visible tokens are
    output minus thinking."""
    client = _claude_client()
    think_kwargs = {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}
    for attempt in range(retries):
        try:
            # Streaming is required for large max_tokens; get_final_message() returns the Message.
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
            return {"text": txt, "finish": m.stop_reason, "usage": usage}
        except Exception as e:  # noqa: BLE001
            status = getattr(e, "status_code", None)
            if (status in RETRY_STATUS or status is None) and attempt < retries - 1:
                backoff(attempt)
                continue
            return {"text": None, "finish": f"ERR_{status}" if status else "EXC", "usage": {},
                    "err": str(e)[:300]}


# =====================================================================================
# Dispatcher
# =====================================================================================
def chat(model, system, user, max_tokens, backend=None, effort="medium", thinking_budget=-1,
         retries=5, *, thinking_level=None, temperature=0.0):
    """One chat call through `backend` (default DEFAULT_BACKEND). The Gemini thinking controls apply
    to Gemini on Vertex; `effort` applies to Claude and to OpenRouter reasoning. temperature=None
    omits the field."""
    backend = backend or DEFAULT_BACKEND
    if backend == "openrouter":
        return call_openrouter(model, system, user, max_tokens, effort=effort, retries=retries,
                               temperature=temperature)
    if backend == "vertex":
        if is_claude(model):
            return call_claude_vertex(model, system, user, max_tokens, effort=effort,
                                      retries=retries)
        return call_gemini_vertex(
            model, system, user, max_tokens, retries=retries, thinking_budget=thinking_budget,
            thinking_level=thinking_level, temperature=temperature)
    raise ValueError(f"unknown backend: {backend!r} (expected 'openrouter' or 'vertex')")
