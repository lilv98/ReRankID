# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0
#
# Portions adapted from verl 0.8.0 (verl/workers/engine_workers.py),
# Copyright 2024 Bytedance Ltd. and/or its affiliates, licensed under the Apache License 2.0.
# Modified to collect offloaded LoRA weights before resuming the vLLM weights.

"""Collect offloaded LoRA weights before waking the colocated vLLM rollout (8B OPD only).

verl 0.8 wakes the vLLM weights first and only then asks the FSDP engine for updated weights. With
param_offload=True that request moves the full FP32 actor back to the GPU, and the 8B actor plus
the awake BF16 rollout do not fit on one A100. This patch swaps the two steps: while the
rollout is still asleep, the stock FSDP engine collects the LoRA tensors layer by layer to CPU and
offloads itself; then vLLM wakes and receives only those tensors. Every other worker configuration
calls the stock method unchanged. Enabled by RERANKID_OPD_CPU_LAYERED_LORA=1 (run_opd.sh, 8B).
"""

import importlib
import inspect
import sys
from importlib.metadata import version

from utils.import_hook import install

TARGET_MODULE = "verl.workers.engine_workers"
TARGET_CLASS = "ActorRolloutRefWorker"
TARGET_METHOD = "update_weights"
# The replacement reproduces the steps of verl 0.8.0's method in a different order; other versions
# are refused.
VERL_VERSION = "0.8.0"

_stock_update_weights = None
_patched_update_weights = None


def _uses_offloaded_lora_path(self, mode):
    """True only for the colocated, param-offloaded, layered LoRA sync used by 8B OPD."""
    if mode == "auto":
        mode = self.config.rollout.checkpoint_engine.backend
    engine = self.actor.engine
    return (
        mode == "naive"
        and bool(engine.is_param_offload_enabled)
        and bool(getattr(engine, "_is_lora", False))
        and bool(self.config.rollout.free_cache_engine)
        and bool(self.layered_summon)
        and bool(self.base_sync_done)
        and not bool(self.peft_merge)
    )


async def offloaded_lora_update_weights(self, global_steps=None, mode="auto"):
    """Run the stock adapter extraction before resuming vLLM weights."""
    if not _uses_offloaded_lora_path(self, mode):
        return await _stock_update_weights(self, global_steps=global_steps, mode=mode)

    module = sys.modules[TARGET_MODULE]
    log = module.log_gpu_memory_usage
    module.set_expandable_segments(False)
    try:
        log("Before offloaded LoRA collection", logger=module.logger)
        # vLLM weights are still asleep here. The stock FSDP method may therefore bring the actor
        # to GPU, summon one layer at a time, copy the adapter to CPU, and offload the actor.
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=True, base_sync_done=True)
        params = list(per_tensor_param)
        if peft_config is None or not params or not all(
            "lora_" in name and tensor.device.type == "cpu" for name, tensor in params
        ):
            raise RuntimeError("offloaded LoRA sync expected a nonempty set of CPU LoRA tensors")
        if any(p.device.type != "cpu" for p in self.actor.engine.module.parameters()):
            raise RuntimeError("offloaded LoRA sync: actor did not return to CPU before vLLM wakes")
        log("After offloaded LoRA collection", logger=module.logger)

        await self.rollout.resume(tags=["weights"])
        log("After resume weights", logger=module.logger)
        self.rollout.sleep_level = 1
        await self.rollout.update_weights(params, peft_config=peft_config, base_sync_done=True,
                                          global_steps=global_steps)
        log("After update_weights", logger=module.logger)

        # Same postcondition as the stock method, although the engine already offloaded itself.
        if self.actor.engine.is_param_offload_enabled:
            self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        module.aggressive_empty_cache(force_sync=True)

        await self.rollout.resume(tags=["kv_cache"])
        log("After resume kv_cache", logger=module.logger)
        self.base_sync_done = True
    finally:
        module.set_expandable_segments(True)


def activate(module=None):
    """Replace ActorRolloutRefWorker.update_weights with the reordered version (idempotent)."""
    global _stock_update_weights, _patched_update_weights
    if module is None:
        module = importlib.import_module(TARGET_MODULE)
    owner = getattr(module, TARGET_CLASS)
    target = getattr(owner, TARGET_METHOD)
    if _patched_update_weights is not None and target is _patched_update_weights:
        return
    if version("verl") != VERL_VERSION:
        raise RuntimeError(f"OPD requires verl {VERL_VERSION}, found {version('verl')}")
    # The installed method carries verl's dispatch wrappers. The replacement is registered the same
    # way, so delegate to the unwrapped stock body to avoid applying them twice.
    _stock_update_weights = inspect.unwrap(target)
    _patched_update_weights = module.register(
        dispatch_mode=module.Dispatch.ONE_TO_ALL, blocking=False)(offloaded_lora_update_weights)
    setattr(owner, TARGET_METHOD, _patched_update_weights)


def install_import_hook():
    """Patch the module when it is first imported, without importing torch now."""
    install(TARGET_MODULE, activate)
