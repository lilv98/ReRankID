# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a patch function on a module right after its first import, without importing it now.

Used by code/sitecustomize.py (transformers flash attention) and by the OPD verl patches, so that
heavy packages such as torch are imported only when the target module itself is.
"""
import importlib.abc
import importlib.machinery
import sys


class PatchLoader(importlib.abc.Loader):
    """Execute the module with its own loader, then call activate(module)."""

    def __init__(self, original, activate):
        self.original = original
        self.activate = activate

    def create_module(self, spec):
        if hasattr(self.original, "create_module"):
            return self.original.create_module(spec)
        return None

    def exec_module(self, module):
        self.original.exec_module(module)
        self.activate(module)


class PatchFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that wraps the loader of one module name."""

    def __init__(self, target, activate):
        self.target = target
        self.activate = activate

    def find_spec(self, fullname, path, target=None):
        if fullname != self.target:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not find {fullname} to patch")
        spec.loader = PatchLoader(spec.loader, self.activate)
        return spec


def install(target, activate):
    """Call activate(module) when `target` is first imported, or now if it already is."""
    if not any(isinstance(f, PatchFinder) and f.target == target for f in sys.meta_path):
        sys.meta_path.insert(0, PatchFinder(target, activate))
    module = sys.modules.get(target)
    if module is not None:
        activate(module)
