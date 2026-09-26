"""Collect offloaded 8B LoRA weights before waking the colocated rollout.

verl 0.8 normally wakes the vLLM weight allocation and only then asks the
FSDP engine for updated weights.  With ``param_offload=True``, that request
temporarily moves the full FP32 actor back to its compute device.  The 8B
actor and the awake BF16 rollout do not fit together on one A100.

For the one audited OPD path, reorder those two stock operations.  While the
rollout weights remain asleep, let the unmodified FSDP engine load the actor,
collect the layer-wise LoRA state, and offload the actor again.  After strict
adapter-only and placement checks, wake vLLM and send only those CPU LoRA
tensors.  Every other worker path delegates to the exact stock method.
"""

import hashlib
import importlib.abc
import importlib.machinery
import inspect
import sys


EXPECTED_STOCK_METHOD_SHA256 = (
    "f81c0a9993f052bec1252de615c1e3bb72b7ec05167881742564b1d3029dc149"
)
PATCH_ENV = "RERANKID_OPD_CPU_LAYERED_LORA"
PATCH_NAME = "rerankid-offloaded-lora-sync-order-v2"
TARGET_MODULE = "verl.workers.engine_workers"
TARGET_CLASS = "ActorRolloutRefWorker"
TARGET_METHOD = "update_weights"

_stock_update_weights = None
_patched_update_weights = None


def _function_sha256(function):
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            f"cannot inspect OPD LoRA sync target {function!r}"
        ) from exc
    return hashlib.sha256(source.encode()).hexdigest()


def _effective_mode(self, mode):
    if mode != "auto":
        return mode
    return self.config.rollout.checkpoint_engine.backend


def _uses_audited_offloaded_lora_path(self, mode):
    """Return whether the worker is exactly the colocated 8B OPD path."""

    engine = self.actor.engine
    return (
        _effective_mode(self, mode) == "naive"
        and bool(engine.is_param_offload_enabled)
        and bool(getattr(engine, "_is_lora", False))
        and bool(self.config.rollout.free_cache_engine)
        and bool(self.layered_summon)
        and bool(self.base_sync_done)
        and not bool(self.peft_merge)
    )


def _materialize_and_validate_adapter(per_tensor_param, peft_config):
    """Fail closed unless the engine returned a nonempty CPU LoRA update."""

    if peft_config is None:
        raise RuntimeError(
            "OPD offloaded LoRA sync received no PEFT configuration"
        )
    params = list(per_tensor_param)
    if not params:
        raise RuntimeError("OPD offloaded LoRA sync collected no tensors")

    invalid = []
    non_cpu = []
    duplicate = []
    seen = set()
    for item in params:
        if not isinstance(item, tuple) or len(item) != 2:
            raise RuntimeError(
                "OPD offloaded LoRA sync received a malformed tensor item"
            )
        name, tensor = item
        if not isinstance(name, str) or "lora_" not in name:
            invalid.append(name)
        if name in seen:
            duplicate.append(name)
        seen.add(name)
        device = getattr(tensor, "device", None)
        if device is None or device.type != "cpu":
            non_cpu.append(name)
    if invalid:
        raise RuntimeError(
            "OPD offloaded LoRA sync received non-adapter tensors: "
            f"{invalid[:3]}"
        )
    if duplicate:
        raise RuntimeError(
            "OPD offloaded LoRA sync received duplicate tensors: "
            f"{duplicate[:3]}"
        )
    if non_cpu:
        raise RuntimeError(
            "OPD offloaded LoRA sync retained non-CPU tensors: "
            f"{non_cpu[:3]}"
        )
    return params


def _assert_actor_is_offloaded(engine):
    """Verify stock collection returned the complete actor to CPU."""

    non_cpu = [
        name
        for name, parameter in engine.module.named_parameters()
        if parameter.device.type != "cpu"
    ]
    if non_cpu:
        raise RuntimeError(
            "OPD offloaded LoRA sync would wake vLLM before the actor "
            f"returned to CPU: {non_cpu[:3]}"
        )


async def offloaded_lora_update_weights(
    self,
    global_steps: int = None,
    mode: str = "auto",
):
    """Run the stock adapter extraction before resuming vLLM weights."""

    if not _uses_audited_offloaded_lora_path(self, mode):
        if _stock_update_weights is None:
            raise RuntimeError("OPD LoRA sync stock method is unavailable")
        return await _stock_update_weights(
            self,
            global_steps=global_steps,
            mode=mode,
        )

    module = sys.modules[TARGET_MODULE]
    module.set_expandable_segments(False)
    try:
        module.log_gpu_memory_usage(
            "Before offloaded LoRA collection",
            logger=module.logger,
        )

        # vLLM weights are still asleep here.  The exact stock FSDP method may
        # therefore bring the actor to GPU, summon one layer at a time, copy
        # the adapter to CPU, and offload the actor without co-residency.
        per_tensor_param, peft_config = (
            self.actor.engine.get_per_tensor_param(
                layered_summon=True,
                base_sync_done=True,
            )
        )
        params = _materialize_and_validate_adapter(
            per_tensor_param,
            peft_config,
        )
        _assert_actor_is_offloaded(self.actor.engine)
        module.log_gpu_memory_usage(
            "After offloaded LoRA collection",
            logger=module.logger,
        )

        await self.rollout.resume(tags=["weights"])
        module.log_gpu_memory_usage(
            "After resume weights",
            logger=module.logger,
        )

        self.rollout.sleep_level = 1
        await self.rollout.update_weights(
            params,
            peft_config=peft_config,
            base_sync_done=True,
            global_steps=global_steps,
        )
        module.log_gpu_memory_usage(
            "After update_weights",
            logger=module.logger,
        )

        # Preserve the stock postcondition even though the engine already
        # offloaded itself after collection.
        if self.actor.engine.is_param_offload_enabled:
            self.actor.engine.to(
                "cpu",
                model=True,
                optimizer=False,
                grad=False,
            )
        module.aggressive_empty_cache(force_sync=True)

        await self.rollout.resume(tags=["kv_cache"])
        module.log_gpu_memory_usage(
            "After resume kv_cache",
            logger=module.logger,
        )
        self.base_sync_done = True
    finally:
        module.set_expandable_segments(True)


PATCHED_METHOD_SHA256 = _function_sha256(offloaded_lora_update_weights)


def _make_registered_patch(module):
    return module.register(
        dispatch_mode=module.Dispatch.ONE_TO_ALL,
        blocking=False,
    )(offloaded_lora_update_weights)


def _assert_active(module):
    target = getattr(getattr(module, TARGET_CLASS), TARGET_METHOD)
    marker = getattr(module, "_rerankid_lora_sync_patch", None)
    stock_hash = getattr(module, "_rerankid_lora_sync_stock_sha256", None)
    target_hash = _function_sha256(target)
    if marker != PATCH_NAME:
        raise RuntimeError(
            f"ReRankID LoRA sync patch marker is not active: {marker!r}"
        )
    if stock_hash != EXPECTED_STOCK_METHOD_SHA256:
        raise RuntimeError(
            "ReRankID LoRA sync stock provenance drifted: "
            f"{stock_hash!r} != {EXPECTED_STOCK_METHOD_SHA256}"
        )
    if target is not _patched_update_weights:
        raise RuntimeError(
            "ReRankID LoRA sync marker is stale or forged"
        )
    if target_hash != PATCHED_METHOD_SHA256:
        raise RuntimeError(
            "ReRankID LoRA sync target source drifted: "
            f"{target_hash} != {PATCHED_METHOD_SHA256}"
        )
    dispatch_attrs = [
        value
        for name, value in vars(target).items()
        if name.startswith("attrs_")
    ]
    if len(dispatch_attrs) != 1:
        raise RuntimeError(
            "ReRankID LoRA sync patch lost its register decorator metadata"
        )
    attrs = dispatch_attrs[0]
    if (
        attrs.get("dispatch_mode") is not module.Dispatch.ONE_TO_ALL
        or attrs.get("blocking") is not False
    ):
        raise RuntimeError(
            "ReRankID LoRA sync patch has incorrect dispatch metadata"
        )
    return {
        "name": PATCH_NAME,
        "stock_method_sha256": stock_hash,
        "patched_method_sha256": target_hash,
        "target_module": TARGET_MODULE,
        "target_class": TARGET_CLASS,
        "target_method": TARGET_METHOD,
        "target_identity_verified": True,
        "dispatch_metadata_verified": True,
    }


def activate(module=None):
    """Patch the exact audited worker method, failing closed on drift."""

    global _stock_update_weights
    global _patched_update_weights
    if module is None:
        module = importlib.import_module(TARGET_MODULE)
    owner = getattr(module, TARGET_CLASS)
    target = getattr(owner, TARGET_METHOD)
    if target is _patched_update_weights and target is not None:
        return _assert_active(module)

    actual = _function_sha256(target)
    if actual != EXPECTED_STOCK_METHOD_SHA256:
        marker = getattr(module, "_rerankid_lora_sync_patch", None)
        raise RuntimeError(
            "refusing to patch an unknown verl LoRA sync method: "
            f"{actual} != {EXPECTED_STOCK_METHOD_SHA256}; "
            f"existing marker={marker!r}"
        )
    # The installed target has two verl dispatch/transfer wrappers.  Our
    # replacement is registered in the same way, so delegate to the exact
    # unwrapped stock body to avoid applying those wrappers twice.
    _stock_update_weights = inspect.unwrap(target)
    _patched_update_weights = _make_registered_patch(module)
    setattr(owner, TARGET_METHOD, _patched_update_weights)
    module._rerankid_lora_sync_patch = PATCH_NAME
    module._rerankid_lora_sync_stock_sha256 = actual
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
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"could not find {fullname} for OPD LoRA sync patch"
            )
        spec.loader = _PatchLoader(spec.loader)
        return spec


def install_import_hook():
    """Install a lazy hook without importing torch or initializing CUDA."""

    if not any(isinstance(finder, _PatchFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _PatchFinder())
    module = sys.modules.get(TARGET_MODULE)
    if module is not None:
        activate(module)


def report():
    """Return fail-closed provenance for the active runtime patch."""

    module = importlib.import_module(TARGET_MODULE)
    return _assert_active(module)
