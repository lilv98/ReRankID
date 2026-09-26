"""Auto-imported at interpreter startup (via PYTHONPATH=code) in the RLVR driver
and every verl/ray worker.  Self-contained, LAZY environment compatibility shim
for the sid-verl stack; keeps the vendored verl engine and installed transformers
UNMODIFIED and the fix tracked in ReRankID.

CRITICAL: this module must not import torch/transformers (or anything that
initializes a CUDA context) at interpreter startup.  Ray schedules its
orchestrator actor with no GPU (CUDA_VISIBLE_DEVICES=""), and eagerly importing
transformers there caches an empty device view -- later crashing verl's
`get_device_capability()` with "Invalid device id".  We therefore install a
name-only meta-path finder that applies the patch lazily, the first time the
real model code imports `transformers.integrations.flash_attention` (long after
Ray has configured devices for the GPU workers).

Shim -- transformers>=5.6 flash-attention `s_aux` regression.
`flash_attention_forward` calls `s_aux.to(query.dtype)` unconditionally, but
backbones without a learnable attention sink (Qwen3, hence every OneRec Pro
model) pass `s_aux=None`, raising `AttributeError: 'NoneType' object has no
attribute 'to'` inside the actor's FSDP training forward.  verl needs
flash-attention for the `use_remove_padding=True` varlen path, so switching to
sdpa is not an option.  We wrap the entry point so a missing/None `s_aux`
becomes a sentinel whose `.to()` returns None -- exactly the value corrected
upstream code forwards -- and re-point every attention-registry entry still
bound to the original.
"""

import importlib.util
import sys

_TARGET = "transformers.integrations.flash_attention"
_done = [False]


def _apply(module):
    if _done[0]:
        return
    original = getattr(module, "flash_attention_forward", None)
    if original is None or getattr(original, "_rerankid_s_aux_safe", False):
        _done[0] = True
        return

    class _NoneTo:
        # s_aux is None for non-sink models; `.to(dtype)` must yield None so the
        # lower-level _flash_attention_forward takes its normal no-sink path.
        __slots__ = ()

        def to(self, *args, **kwargs):
            return None

    _sentinel = _NoneTo()

    def flash_attention_forward(*args, **kwargs):
        if kwargs.get("s_aux", None) is None:
            kwargs["s_aux"] = _sentinel
        return original(*args, **kwargs)

    flash_attention_forward._rerankid_s_aux_safe = True
    module.flash_attention_forward = flash_attention_forward

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        for key in list(ALL_ATTENTION_FUNCTIONS.keys()):
            if ALL_ATTENTION_FUNCTIONS[key] is original:
                ALL_ATTENTION_FUNCTIONS[key] = flash_attention_forward
    except Exception:
        pass

    _done[0] = True


class _LazyPatchFinder:
    """Name-only finder: no heavy imports until the target is actually loaded."""

    def find_spec(self, name, path=None, target=None):
        if name != _TARGET or _done[0]:
            return None
        # Resolve the real spec without recursing into ourselves.
        try:
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None

        loader = spec.loader
        original_exec = loader.exec_module

        def exec_module(module):
            original_exec(module)
            _apply(module)

        loader.exec_module = exec_module
        return spec


if not any(isinstance(f, _LazyPatchFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _LazyPatchFinder())
