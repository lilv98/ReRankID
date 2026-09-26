"""Lazy, fail-closed hook for synchronous OPD checkpoint archiving.

This targets the audited verl 0.8.0 ``RayPPOTrainer`` used by
``python -m verl.trainer.main_ppo``.  The wrapper calls the stock save first,
then archives only after the stock method has written its latest-step marker.
"""

import hashlib
import importlib.abc
import importlib.machinery
import inspect
import os
import sys

from opd.checkpoint_archive import ARCHIVE_ENV, CheckpointArchive


TARGET_MODULE = "verl.trainer.ppo.ray_trainer"
PATCH_NAME = "rerankid-opd-online-checkpoint-archive-v1"
EXPECTED_SAVE_SHA256 = (
    "1f2ec4c96769f4a993e0a8ca8880d82e46132a2436c2ab0010384a23cdfd5b16"
)
EXPECTED_LOAD_SHA256 = (
    "23626f03573e151c9bb4fdd52c867e5bfed93568ff5a9dc29bb38d6e1fb52757"
)


def _function_sha256(function):
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            f"cannot inspect OPD checkpoint hook target {function!r}"
        ) from exc
    return hashlib.sha256(source.encode()).hexdigest()


def archived_save_checkpoint(self):
    """Run the stock save, then synchronously export and prune."""

    archive = CheckpointArchive.from_trainer(self)
    archive.before_save(int(self.global_steps))
    original = getattr(
        type(self), "_rerankid_original_save_checkpoint", None
    )
    if original is None:
        raise RuntimeError("OPD checkpoint hook lost the stock save method")
    result = original(self)
    archive.after_save(int(self.global_steps))
    return result


def reconciled_load_checkpoint(self):
    """Validate archive state before verl loads a resumed raw checkpoint."""

    archive = CheckpointArchive.from_trainer(self)
    archive.reconcile_resume()
    original = getattr(
        type(self), "_rerankid_original_load_checkpoint", None
    )
    if original is None:
        raise RuntimeError("OPD checkpoint hook lost the stock load method")
    return original(self)


PATCHED_SAVE_SHA256 = _function_sha256(archived_save_checkpoint)
PATCHED_LOAD_SHA256 = _function_sha256(reconciled_load_checkpoint)


def _assert_active(module):
    trainer = module.RayPPOTrainer
    marker = getattr(module, "_rerankid_checkpoint_archive_patch", None)
    stock_save = getattr(
        trainer, "_rerankid_original_save_checkpoint", None
    )
    stock_load = getattr(
        trainer, "_rerankid_original_load_checkpoint", None
    )
    if marker != PATCH_NAME:
        raise RuntimeError(
            f"ReRankID checkpoint archive marker is not active: {marker!r}"
        )
    if trainer._save_checkpoint is not archived_save_checkpoint:
        raise RuntimeError(
            "ReRankID checkpoint archive save wrapper identity changed"
        )
    if trainer._load_checkpoint is not reconciled_load_checkpoint:
        raise RuntimeError(
            "ReRankID checkpoint archive load wrapper identity changed"
        )
    hashes = {
        "stock_save_sha256": _function_sha256(stock_save),
        "stock_load_sha256": _function_sha256(stock_load),
        "patched_save_sha256": _function_sha256(
            trainer._save_checkpoint
        ),
        "patched_load_sha256": _function_sha256(
            trainer._load_checkpoint
        ),
    }
    expected = {
        "stock_save_sha256": EXPECTED_SAVE_SHA256,
        "stock_load_sha256": EXPECTED_LOAD_SHA256,
        "patched_save_sha256": PATCHED_SAVE_SHA256,
        "patched_load_sha256": PATCHED_LOAD_SHA256,
    }
    if hashes != expected:
        raise RuntimeError(
            "ReRankID checkpoint archive source identity drifted: "
            f"{hashes!r} != {expected!r}"
        )
    return {
        "name": PATCH_NAME,
        **hashes,
        "target_module": TARGET_MODULE,
        "target_class": "RayPPOTrainer",
        "target_identity_verified": True,
    }


def activate(module=None):
    """Patch only the exact audited verl 0.8.0 trainer methods."""

    if os.environ.get(ARCHIVE_ENV) != "1":
        raise RuntimeError(f"missing required {ARCHIVE_ENV}=1")
    if module is None:
        from verl.trainer.ppo import ray_trainer as module

    trainer = module.RayPPOTrainer
    if (
        trainer._save_checkpoint is archived_save_checkpoint
        and trainer._load_checkpoint is reconciled_load_checkpoint
    ):
        return _assert_active(module)
    actual = {
        "save": _function_sha256(trainer._save_checkpoint),
        "load": _function_sha256(trainer._load_checkpoint),
    }
    expected = {
        "save": EXPECTED_SAVE_SHA256,
        "load": EXPECTED_LOAD_SHA256,
    }
    if actual != expected:
        raise RuntimeError(
            "refusing to patch unknown verl checkpoint methods: "
            f"{actual!r} != {expected!r}"
        )
    trainer._rerankid_original_save_checkpoint = (
        trainer._save_checkpoint
    )
    trainer._rerankid_original_load_checkpoint = (
        trainer._load_checkpoint
    )
    trainer._save_checkpoint = archived_save_checkpoint
    trainer._load_checkpoint = reconciled_load_checkpoint
    module._rerankid_checkpoint_archive_patch = PATCH_NAME
    return _assert_active(module)


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
    target = TARGET_MODULE

    def find_spec(self, fullname, path, target=None):
        if fullname != self.target:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"could not find {fullname} for OPD checkpoint hook"
            )
        spec.loader = _PatchLoader(spec.loader)
        return spec


def install_import_hook():
    """Install without importing verl, torch, or initializing CUDA."""

    if os.environ.get(ARCHIVE_ENV) != "1":
        raise RuntimeError(f"missing required {ARCHIVE_ENV}=1")
    if not any(
        isinstance(finder, _PatchFinder) for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _PatchFinder())
    module = sys.modules.get(TARGET_MODULE)
    if module is not None:
        activate(module)


def report():
    """Return fail-closed source identities for the active hook."""

    from verl.trainer.ppo import ray_trainer

    return _assert_active(ray_trainer)
