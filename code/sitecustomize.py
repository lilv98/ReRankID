# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for transformers 5.6 flash attention, needed by RLVR in the cu130 sid-verl
environment. Python imports it at startup because rlvr/run_grpo.sh puts code/ on PYTHONPATH.

transformers 5.6 calls s_aux.to(...) unconditionally in flash_attention_forward, which fails for
models without an attention sink (Qwen3, hence OneRec) that pass s_aux=None. This module wraps that
function so a missing s_aux behaves as None. The patch is applied when transformers first imports
its flash-attention module, so no CUDA context is created in Ray's CPU-only actors.
"""
from utils.import_hook import install

TARGET = "transformers.integrations.flash_attention"
_original = None


class NoneTo:
    """Stand-in for a missing s_aux whose .to(...) is None, so the lower-level
    _flash_attention_forward takes its normal no-sink path."""

    __slots__ = ()

    def to(self, *args, **kwargs):
        return None


_NONE_TO = NoneTo()


def flash_attention_forward(*args, **kwargs):
    """transformers' flash_attention_forward with s_aux=None replaced by a NoneTo."""
    if kwargs.get("s_aux") is None:
        kwargs["s_aux"] = _NONE_TO
    return _original(*args, **kwargs)


def activate(module):
    """Replace module.flash_attention_forward and its attention-registry entries (idempotent)."""
    global _original
    original = getattr(module, "flash_attention_forward", None)
    if original is None or original is flash_attention_forward:
        return
    _original = original
    module.flash_attention_forward = flash_attention_forward
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        return
    for key in list(ALL_ATTENTION_FUNCTIONS.keys()):
        if ALL_ATTENTION_FUNCTIONS[key] is original:
            ALL_ATTENTION_FUNCTIONS[key] = flash_attention_forward


install(TARGET, activate)
