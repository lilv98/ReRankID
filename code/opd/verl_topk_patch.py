"""Minimal runtime correction for verl's truncated top-k forward KL.

verl 0.8 gathers the teacher's top-k probabilities from the full vocabulary
but does not renormalize them before calling its KL helper.  The resulting
truncated surrogate can be negative and is then clamped to zero.  We replace
only that helper so the teacher distribution is normalized over its retained
support.  Student probabilities remain full-vocabulary probabilities, so the
loss also encourages the student to move probability mass onto that support.
"""

import hashlib
import importlib.abc
import importlib.machinery
import inspect
import os
import sys


EXPECTED_STOCK_KL_SHA256 = (
    "b52393b6c1849ba4819e07da704adfeedca3acd1ad876e69a259b7281352e3bc"
)
PATCH_NAME = "rerankid-renormalized-teacher-topk-v1"
PATCH_ENV = "RERANKID_OPD_NORMALIZE_TOPK"


def normalized_teacher_topk_kl(log_q, log_p):
    """KL from a renormalized teacher top-k distribution to full student q."""

    log_p = log_p.float()
    log_q = log_q.float()
    normalized_log_p = log_p - log_p.logsumexp(dim=-1, keepdim=True)
    p = normalized_log_p.exp()
    return (
        p * (normalized_log_p - log_q)
    ).sum(dim=-1)


def _function_sha256(function):
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            f"cannot inspect OPD KL target {function!r}"
        ) from exc
    return hashlib.sha256(source.encode()).hexdigest()


PATCHED_KL_SHA256 = _function_sha256(normalized_teacher_topk_kl)


def _assert_active(losses):
    marker = getattr(losses, "_rerankid_topk_patch", None)
    stock_hash = getattr(losses, "_rerankid_stock_kl_sha256", None)
    target = losses.kl_divergence
    target_hash = _function_sha256(target)
    if marker != PATCH_NAME:
        raise RuntimeError(
            "ReRankID normalized top-k patch marker is not active: "
            f"{marker!r}"
        )
    if stock_hash != EXPECTED_STOCK_KL_SHA256:
        raise RuntimeError(
            "ReRankID normalized top-k stock provenance drifted: "
            f"{stock_hash!r} != {EXPECTED_STOCK_KL_SHA256}"
        )
    if target is not normalized_teacher_topk_kl:
        raise RuntimeError(
            "ReRankID normalized top-k marker is stale or forged: "
            "verl KL target does not have the patched function identity"
        )
    if target_hash != PATCHED_KL_SHA256:
        raise RuntimeError(
            "ReRankID normalized top-k target source drifted: "
            f"{target_hash} != {PATCHED_KL_SHA256}"
        )
    return {
        "name": PATCH_NAME,
        "stock_kl_sha256": stock_hash,
        "patched_function_sha256": target_hash,
        "target_module": target.__module__,
        "target_qualname": target.__qualname__,
        "target_identity_verified": True,
    }


def activate(losses=None):
    """Patch the exact audited verl helper, failing closed on source drift."""

    if losses is None:
        from verl.trainer.distillation.fsdp import losses

    marker = getattr(losses, "_rerankid_topk_patch", None)
    target = losses.kl_divergence
    if target is normalized_teacher_topk_kl:
        return _assert_active(losses)

    actual = _function_sha256(target)
    if actual != EXPECTED_STOCK_KL_SHA256:
        marker_note = f"; existing marker={marker!r}" if marker else ""
        raise RuntimeError(
            "refusing to patch an unknown verl KL helper: "
            f"{actual} != {EXPECTED_STOCK_KL_SHA256}{marker_note}"
        )
    losses.kl_divergence = normalized_teacher_topk_kl
    losses._rerankid_topk_patch = PATCH_NAME
    losses._rerankid_stock_kl_sha256 = actual
    return _assert_active(losses)


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        if hasattr(self.original, "create_module"):
            return self.original.create_module(spec)
        return None

    def exec_module(self, module):
        self.original.exec_module(module)
        activate(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    target = "verl.trainer.distillation.fsdp.losses"

    def find_spec(self, fullname, path, target=None):
        if fullname != self.target:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not find {fullname} for OPD patch")
        spec.loader = _PatchLoader(spec.loader)
        return spec


def install_import_hook():
    """Install a lazy hook without importing torch or initializing CUDA."""

    if not any(isinstance(finder, _PatchFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _PatchFinder())
    module = sys.modules.get(_PatchFinder.target)
    if module is not None:
        activate(module)


def ray_worker_setup():
    """Fail-closed Ray worker bootstrap that preserves lazy torch import."""

    if os.environ.get(PATCH_ENV) != "1":
        raise RuntimeError(
            f"Ray worker is missing required {PATCH_ENV}=1"
        )
    install_import_hook()
    if os.environ.get("RERANKID_OPD_CPU_LAYERED_LORA") == "1":
        from opd.verl_lora_sync_patch import (
            install_import_hook as install_lora_sync_hook,
        )

        install_lora_sync_hook()


def report():
    """Return fail-closed provenance for the active runtime patch."""

    from verl.trainer.distillation.fsdp import losses

    return _assert_active(losses)
