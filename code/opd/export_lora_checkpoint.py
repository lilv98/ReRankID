"""Export one verl OPD actor checkpoint as a compact PEFT LoRA adapter.

The verl merger temporarily materializes a complete Hugging Face model even
for LoRA training.  This wrapper keeps only the generated ``lora_adapter``.
The merger workspace may live on a different filesystem from the final
adapter root.  Publication therefore always copies into a private directory
on the destination filesystem, re-hashes that copy, and atomically renames it
into place.

The public provenance payload remains schema version 2 so adapters exported
by the original 1.7B workflow and adapters produced online are interchangeable.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config

from opd import contract
from opd.provenance import (
    load_manifest,
    manifest_policy,
    payload_sha256,
    training_override_map,
)


PROVENANCE_SCHEMA_VERSION = 2
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "provenance.json",
)
SHARD_PREFIXES = (
    "model_world_size_1_rank_0",
    "optim_world_size_1_rank_0",
    "extra_state_world_size_1_rank_0",
)


class ExportError(RuntimeError):
    """A fail-closed compact-adapter export or validation error."""


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExportError(f"JSON file is not an object: {path}")
    return payload


def _require_hex(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExportError(f"{label} is not a lowercase SHA-256 digest")


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(config.ROOT, path)


def _manifest_context(run_dir):
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.isfile(manifest_path):
        raise ExportError(f"missing immutable run manifest: {manifest_path}")
    try:
        manifest = load_manifest(manifest_path, policy=None)
        overrides = training_override_map(manifest)
    except RuntimeError as exc:
        raise ExportError(str(exc)) from exc
    try:
        max_step = int(overrides["trainer.total_training_steps"])
        save_freq = int(overrides["trainer.save_freq"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExportError(
            "run manifest has invalid checkpoint schedule overrides"
        ) from exc
    return {
        "path": manifest_path,
        "payload": manifest,
        "sha256": sha256(manifest_path),
        "payload_sha256": payload_sha256(manifest),
        "max_step": max_step,
        "save_freq": save_freq,
    }


def _validate_scheduled_step(step, manifest_context):
    max_step = manifest_context["max_step"]
    save_freq = manifest_context["save_freq"]
    if save_freq <= 0 or step > max_step or step % save_freq:
        raise ExportError(
            f"step {step} is not a frozen checkpoint step "
            f"(save_freq={save_freq}, total={max_step})"
        )


def read_checkpoint_marker(run_dir, required=True):
    marker_path = os.path.join(run_dir, "latest_checkpointed_iteration.txt")
    if not os.path.exists(marker_path) and not required:
        return None
    try:
        with open(marker_path, encoding="utf-8") as handle:
            step = int(handle.read().strip())
    except (OSError, ValueError) as exc:
        raise ExportError(f"invalid checkpoint marker: {marker_path}") from exc
    if step <= 0:
        raise ExportError(f"invalid checkpoint marker: {marker_path}")
    return step


def inspect_source_checkpoint(run_dir, step, hash_model=True):
    """Validate one complete single-rank verl checkpoint.

    ``hash_model=False`` is intended for inexpensive retention scans.  The
    exporter itself always hashes the source model shard for provenance.
    """

    checkpoint_dir = os.path.join(run_dir, f"global_step_{step}")
    actor_dir = os.path.join(checkpoint_dir, "actor")
    if not os.path.isdir(actor_dir):
        raise ExportError(f"missing actor checkpoint: {actor_dir}")
    metadata = {
        "data": os.path.join(checkpoint_dir, "data.pt"),
        "fsdp": os.path.join(actor_dir, "fsdp_config.json"),
        "lora": os.path.join(actor_dir, "lora_train_meta.json"),
    }
    for name, path in metadata.items():
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            raise ExportError(f"missing {name} checkpoint metadata: {path}")

    fsdp_meta = load_json(metadata["fsdp"])
    if fsdp_meta.get("world_size") != 1:
        raise ExportError("OPD checkpoint FSDP world size is not one")

    actor_files = os.listdir(actor_dir)
    source_shards = {}
    source_model_shard = None
    for prefix in SHARD_PREFIXES:
        matches = [name for name in actor_files if name.startswith(prefix)]
        if len(matches) != 1:
            raise ExportError(
                f"expected one {prefix} checkpoint shard, found {matches}"
            )
        path = os.path.join(actor_dir, matches[0])
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            raise ExportError(f"empty checkpoint shard: {path}")
        source_shards[matches[0]] = os.path.getsize(path)
        if prefix.startswith("model_"):
            source_model_shard = path
    if source_model_shard is None:
        raise ExportError("missing source model checkpoint shard")
    return {
        "checkpoint_dir": checkpoint_dir,
        "actor_dir": actor_dir,
        "metadata": metadata,
        "source_shards": source_shards,
        "source_model_shard": source_model_shard,
        "source_model_shard_sha256": (
            sha256(source_model_shard) if hash_model else None
        ),
    }


def _expected_lora(manifest):
    try:
        optimization = manifest["frozen_contract"]["optimization"]
        return {
            "r": int(optimization["lora_rank"]),
            "lora_alpha": int(optimization["lora_alpha"]),
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ExportError(
            "run manifest has invalid frozen LoRA configuration"
        ) from exc


def _qwen3_all_linear_modules(num_hidden_layers):
    if (
        not isinstance(num_hidden_layers, int)
        or isinstance(num_hidden_layers, bool)
        or num_hidden_layers <= 0
    ):
        raise ExportError("invalid Qwen3 hidden-layer count")
    result = set()
    for layer in range(num_hidden_layers):
        prefix = f"base_model.model.model.layers.{layer}"
        result.update(
            f"{prefix}.self_attn.{name}_proj"
            for name in ("q", "k", "v", "o")
        )
        result.update(
            f"{prefix}.mlp.{name}_proj"
            for name in ("gate", "up", "down")
        )
    return result


def _expected_lora_modules(manifest):
    try:
        policy = manifest_policy(manifest)
        architecture = contract.policy_spec(policy)["architecture"]
        layers = architecture["num_hidden_layers"]
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ExportError(
            "cannot derive all-linear LoRA modules from manifest policy"
        ) from exc
    return _qwen3_all_linear_modules(layers)


def _qwen3_all_linear_tensor_shapes(architecture, lora_rank):
    """Return the exact PEFT tensor shapes for Qwen3 all-linear LoRA."""

    required = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    )
    try:
        values = {key: architecture[key] for key in required}
    except (KeyError, TypeError) as exc:
        raise ExportError(
            "incomplete Qwen3 architecture for LoRA tensor validation"
        ) from exc
    if (
        any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in values.values()
        )
        or not isinstance(lora_rank, int)
        or isinstance(lora_rank, bool)
        or lora_rank <= 0
    ):
        raise ExportError(
            "invalid Qwen3 architecture or LoRA rank for tensor validation"
        )
    hidden = values["hidden_size"]
    attention_heads = values["num_attention_heads"]
    key_value_heads = values["num_key_value_heads"]
    if hidden % attention_heads or key_value_heads > attention_heads:
        raise ExportError(
            "inconsistent Qwen3 attention dimensions for LoRA validation"
        )
    key_value_size = hidden // attention_heads * key_value_heads
    projections = {
        "self_attn.q_proj": (hidden, hidden),
        "self_attn.k_proj": (hidden, key_value_size),
        "self_attn.v_proj": (hidden, key_value_size),
        "self_attn.o_proj": (hidden, hidden),
        "mlp.gate_proj": (hidden, values["intermediate_size"]),
        "mlp.up_proj": (hidden, values["intermediate_size"]),
        "mlp.down_proj": (values["intermediate_size"], hidden),
    }
    result = {}
    for layer in range(values["num_hidden_layers"]):
        prefix = f"base_model.model.model.layers.{layer}"
        for suffix, (in_features, out_features) in projections.items():
            module = f"{prefix}.{suffix}"
            result[f"{module}.lora_A.weight"] = (
                lora_rank,
                in_features,
            )
            result[f"{module}.lora_B.weight"] = (
                out_features,
                lora_rank,
            )
    return result


def _expected_lora_tensor_shapes(manifest):
    try:
        policy = manifest_policy(manifest)
        architecture = contract.policy_spec(policy)["architecture"]
        lora_rank = _expected_lora(manifest)["r"]
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ExportError(
            "cannot derive all-linear LoRA tensor shapes from manifest policy"
        ) from exc
    return _qwen3_all_linear_tensor_shapes(architecture, lora_rank)


def _validate_adapter_payload(
    adapter_dir,
    expected_lora,
    expected_modules,
    expected_tensor_shapes,
):
    paths = {
        "weights": os.path.join(adapter_dir, "adapter_model.safetensors"),
        "config": os.path.join(adapter_dir, "adapter_config.json"),
    }
    for label, path in paths.items():
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            raise ExportError(f"missing adapter {label}: {path}")

    adapter_config = load_json(paths["config"])
    for key, value in expected_lora.items():
        if adapter_config.get(key) != value:
            raise ExportError(
                f"adapter_config.{key}={adapter_config.get(key)!r}, "
                f"expected {value!r}"
            )
    if not isinstance(expected_tensor_shapes, dict):
        raise ExportError("expected LoRA tensor shapes are not a mapping")
    try:
        with safe_open(paths["weights"], framework="pt") as handle:
            tensor_keys = list(handle.keys())
            if not tensor_keys or len(tensor_keys) % 2:
                raise ExportError(
                    f"unexpected LoRA tensor count: {len(tensor_keys)}"
                )
            allowed_suffixes = (".lora_A.weight", ".lora_B.weight")
            if any(
                not key.endswith(allowed_suffixes) or "base_layer" in key
                for key in tensor_keys
            ):
                raise ExportError(
                    "compact adapter contains a non-LoRA or base-layer tensor"
                )
            a_modules = {
                key.removesuffix(".lora_A.weight")
                for key in tensor_keys
                if key.endswith(".lora_A.weight")
            }
            b_modules = {
                key.removesuffix(".lora_B.weight")
                for key in tensor_keys
                if key.endswith(".lora_B.weight")
            }
            if a_modules != b_modules:
                raise ExportError("LoRA A/B module sets do not match")
            if a_modules != expected_modules:
                missing = sorted(expected_modules - a_modules)
                extra = sorted(a_modules - expected_modules)
                raise ExportError(
                    "compact adapter does not cover the frozen all-linear "
                    "module set: "
                    f"expected_modules={len(expected_modules)}, "
                    f"actual_modules={len(a_modules)}, "
                    f"missing={missing[:8]}, extra={extra[:8]}"
                )
            expected_keys = set(expected_tensor_shapes)
            actual_keys = set(tensor_keys)
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)
                extra = sorted(actual_keys - expected_keys)
                raise ExportError(
                    "compact adapter tensor keys do not match the frozen "
                    f"shape contract: missing={missing[:8]}, "
                    f"extra={extra[:8]}"
                )
            for key in tensor_keys:
                tensor_slice = handle.get_slice(key)
                actual_shape = tuple(tensor_slice.get_shape())
                expected_shape = tuple(expected_tensor_shapes[key])
                if actual_shape != expected_shape:
                    raise ExportError(
                        f"compact adapter tensor {key} has shape "
                        f"{actual_shape}, expected {expected_shape}"
                    )
                actual_dtype = tensor_slice.get_dtype()
                if actual_dtype != "BF16":
                    raise ExportError(
                        f"compact adapter tensor {key} has dtype "
                        f"{actual_dtype}, expected BF16"
                    )
                tensor = handle.get_tensor(key)
                if not bool(tensor.isfinite().all().item()):
                    raise ExportError(
                        f"compact adapter tensor {key} contains a "
                        "non-finite value"
                    )
    except ExportError:
        raise
    except Exception as exc:
        raise ExportError(
            f"cannot inspect compact adapter weights: {paths['weights']}"
        ) from exc
    return {
        "paths": paths,
        "tensor_keys": tensor_keys,
        "module_names": a_modules,
    }


def _validate_compact_layout(adapter_dir):
    if os.path.islink(adapter_dir) or not os.path.isdir(adapter_dir):
        raise ExportError(
            f"compact adapter is not a real directory: {adapter_dir}"
        )
    entries = set(os.listdir(adapter_dir))
    expected = set(ADAPTER_FILES)
    if entries != expected:
        raise ExportError(
            "compact adapter must contain exactly "
            f"{sorted(expected)}; found {sorted(entries)}"
        )
    for name in ADAPTER_FILES:
        path = os.path.join(adapter_dir, name)
        if os.path.islink(path) or not os.path.isfile(path):
            raise ExportError(
                f"compact adapter entry is not a regular file: {path}"
            )


def verify_compact_adapter(adapter_dir, step, manifest_path):
    """Verify a schema-v2 compact adapter without requiring its raw checkpoint."""

    adapter_dir = os.path.abspath(adapter_dir)
    run_dir = os.path.dirname(os.path.abspath(manifest_path))
    context = _manifest_context(run_dir)
    if os.path.abspath(context["path"]) != os.path.abspath(manifest_path):
        raise ExportError("manifest path does not belong to the adapter run")
    _validate_scheduled_step(step, context)
    if not os.path.isdir(adapter_dir):
        raise ExportError(f"missing compact adapter: {adapter_dir}")
    _validate_compact_layout(adapter_dir)
    payload_info = _validate_adapter_payload(
        adapter_dir,
        _expected_lora(context["payload"]),
        _expected_lora_modules(context["payload"]),
        _expected_lora_tensor_shapes(context["payload"]),
    )
    provenance_path = os.path.join(adapter_dir, "provenance.json")
    if not os.path.isfile(provenance_path) or os.path.getsize(
        provenance_path
    ) <= 0:
        raise ExportError(f"missing adapter provenance: {provenance_path}")
    provenance = load_json(provenance_path)
    manifest = context["payload"]
    frozen = manifest["frozen_contract"]
    audit = manifest["audit"]
    expected = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "recipe": manifest["recipe"],
        "data_recipe": manifest["data_recipe"],
        "run_mode": manifest["run_mode"],
        "step": step,
        "student_base": frozen["student"]["base_rel"],
        "student_revision": frozen["student"]["revision"],
        "teacher": frozen["teacher"]["merged_rel"],
        "training_commit": audit["repository"]["head"],
        "run_manifest_sha256": context["sha256"],
        "manifest_payload_sha256": context["payload_sha256"],
        "adapter_model_sha256": sha256(payload_info["paths"]["weights"]),
        "adapter_config_sha256": sha256(payload_info["paths"]["config"]),
        "adapter_tensor_count": len(payload_info["tensor_keys"]),
        "adapter_module_count": len(payload_info["module_names"]),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ExportError(
                f"step {step} provenance.{key}={provenance.get(key)!r}, "
                f"expected {value!r}"
            )

    source_actor = provenance.get("source_actor")
    source_parts = (
        Path(os.path.normpath(source_actor)).parts
        if isinstance(source_actor, str)
        else ()
    )
    if len(source_parts) < 2 or source_parts[-2:] != (
        f"global_step_{step}",
        "actor",
    ):
        raise ExportError(
            f"step {step} provenance.source_actor={source_actor!r} "
            "does not identify the matching checkpoint"
        )
    for key in (
        "adapter_model_sha256",
        "adapter_config_sha256",
        "source_model_shard_sha256",
    ):
        _require_hex(provenance.get(key), f"provenance.{key}")
    metadata_hashes = provenance.get("checkpoint_metadata_sha256")
    if not isinstance(metadata_hashes, dict):
        raise ExportError("provenance has no checkpoint metadata hashes")
    for key in ("data", "fsdp", "lora"):
        _require_hex(
            metadata_hashes.get(key),
            f"provenance.checkpoint_metadata_sha256.{key}",
        )
    shard_bytes = provenance.get("checkpoint_shard_bytes")
    if (
        not isinstance(shard_bytes, dict)
        or not shard_bytes
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in shard_bytes.values()
        )
    ):
        raise ExportError("provenance has invalid checkpoint shard sizes")
    return {
        "step": step,
        "path": adapter_dir,
        "provenance": provenance,
        "adapter_model_sha256": expected["adapter_model_sha256"],
        "adapter_config_sha256": expected["adapter_config_sha256"],
        "provenance_sha256": sha256(provenance_path),
    }


def _tree_hashes(root):
    result = {}
    for current, directories, files in os.walk(root):
        directories.sort()
        files.sort()
        for name in directories + files:
            path = os.path.join(current, name)
            if os.path.islink(path):
                raise ExportError(
                    f"refusing to publish adapter symlink: {path}"
                )
        for name in files:
            path = os.path.join(current, name)
            relative = os.path.relpath(path, root)
            result[relative] = sha256(path)
    return result


def _fsync_tree(root):
    for current, directories, files in os.walk(root, topdown=False):
        for name in files:
            path = os.path.join(current, name)
            with open(path, "rb") as handle:
                os.fsync(handle.fileno())
        for name in directories:
            descriptor = os.open(
                os.path.join(current, name),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    descriptor = os.open(
        root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path):
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_adapter(
    source_dir,
    final_dir,
    step,
    manifest_path,
    if_exists="error",
):
    """Copy, re-hash, and atomically publish an adapter on the target FS."""

    if if_exists not in {"error", "verify"}:
        raise ValueError(f"unsupported if_exists policy: {if_exists}")
    export_root = os.path.dirname(final_dir)
    os.makedirs(export_root, exist_ok=True)
    if os.path.exists(final_dir):
        if if_exists == "verify":
            return verify_compact_adapter(final_dir, step, manifest_path)
        raise ExportError(
            f"refusing to overwrite existing export: {final_dir}"
        )

    stage_parent = tempfile.mkdtemp(
        prefix=f".publish_step_{step}_", dir=export_root
    )
    stage_dir = os.path.join(stage_parent, "adapter")
    try:
        _validate_compact_layout(source_dir)
        source_hashes = _tree_hashes(source_dir)
        shutil.copytree(source_dir, stage_dir, copy_function=shutil.copy2)
        copied_hashes = _tree_hashes(stage_dir)
        if copied_hashes != source_hashes:
            raise ExportError(
                "destination-filesystem adapter copy failed hash verification"
            )
        record = verify_compact_adapter(stage_dir, step, manifest_path)
        _fsync_tree(stage_dir)
        if os.path.exists(final_dir):
            raise ExportError(
                "compact export appeared concurrently during publication: "
                f"{final_dir}"
            )
        os.replace(stage_dir, final_dir)
        _fsync_directory(export_root)
        return {
            **record,
            "path": os.path.abspath(final_dir),
            "provenance_sha256": sha256(
                os.path.join(final_dir, "provenance.json")
            ),
        }
    finally:
        shutil.rmtree(stage_parent, ignore_errors=True)


def export_checkpoint(
    step,
    run_dir,
    export_root,
    merger_temp_root=None,
    if_exists="error",
):
    """Synchronously export and atomically publish one checkpoint."""

    if step <= 0:
        raise ExportError("--step must be positive")
    run_dir = os.path.abspath(run_dir)
    export_root = os.path.abspath(export_root)
    context = _manifest_context(run_dir)
    _validate_scheduled_step(step, context)
    final_dir = os.path.join(export_root, f"step-{step:06d}")
    if os.path.exists(final_dir):
        if if_exists == "verify":
            return verify_compact_adapter(final_dir, step, context["path"])
        raise ExportError(
            f"refusing to overwrite existing export: {final_dir}"
        )

    latest_step = read_checkpoint_marker(run_dir)
    _validate_scheduled_step(latest_step, context)
    if step > latest_step:
        raise ExportError(
            f"checkpoint step {step} is newer than marker {latest_step}"
        )
    source = inspect_source_checkpoint(run_dir, step, hash_model=True)
    manifest = context["payload"]
    lora_meta = load_json(source["metadata"]["lora"])
    expected_lora = _expected_lora(manifest)
    for key in ("r", "lora_alpha", "task_type"):
        if lora_meta.get(key) != expected_lora[key]:
            raise ExportError(
                f"checkpoint lora metadata {key}={lora_meta.get(key)!r}, "
                f"expected {expected_lora[key]!r}"
            )

    os.makedirs(export_root, exist_ok=True)
    if merger_temp_root is None:
        merger_temp_root = export_root
    merger_temp_root = os.path.abspath(merger_temp_root)
    os.makedirs(merger_temp_root, exist_ok=True)
    temp_dir = tempfile.mkdtemp(
        prefix=f".merge_step_{step}_", dir=merger_temp_root
    )
    try:
        merger_environment = os.environ.copy()
        merger_environment["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(
            [
                sys.executable,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                source["actor_dir"],
                "--target_dir",
                temp_dir,
                "--trust-remote-code",
                "--use_cpu_initialization",
            ],
            check=True,
            env=merger_environment,
        )
        adapter_dir = os.path.join(temp_dir, "lora_adapter")
        payload_info = _validate_adapter_payload(
            adapter_dir,
            expected_lora,
            _expected_lora_modules(manifest),
            _expected_lora_tensor_shapes(manifest),
        )
        if sha256(context["path"]) != context["sha256"]:
            raise ExportError("immutable run manifest changed during export")
        frozen = manifest["frozen_contract"]
        audit = manifest["audit"]
        provenance = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "recipe": manifest["recipe"],
            "data_recipe": manifest["data_recipe"],
            "run_mode": manifest["run_mode"],
            "step": step,
            "source_actor": os.path.relpath(
                source["actor_dir"], config.ROOT
            ),
            "student_base": frozen["student"]["base_rel"],
            "student_revision": frozen["student"]["revision"],
            "teacher": frozen["teacher"]["merged_rel"],
            "training_commit": audit["repository"]["head"],
            "run_manifest_sha256": context["sha256"],
            "manifest_payload_sha256": context["payload_sha256"],
            "checkpoint_metadata_sha256": {
                name: sha256(path)
                for name, path in source["metadata"].items()
            },
            "checkpoint_shard_bytes": source["source_shards"],
            "source_model_shard_sha256": source[
                "source_model_shard_sha256"
            ],
            "adapter_model_sha256": sha256(
                payload_info["paths"]["weights"]
            ),
            "adapter_config_sha256": sha256(
                payload_info["paths"]["config"]
            ),
            "adapter_tensor_count": len(payload_info["tensor_keys"]),
            "adapter_module_count": len(payload_info["module_names"]),
        }
        provenance_path = os.path.join(adapter_dir, "provenance.json")
        with open(provenance_path, "w", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        record = publish_adapter(
            adapter_dir,
            final_dir,
            step,
            context["path"],
            if_exists=if_exists,
        )
        return record
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument(
        "--run-dir",
        default=contract.RUN_REL,
        help="OPD run directory, relative to the repository unless absolute",
    )
    parser.add_argument(
        "--export-dir",
        default=f"{contract.RUN_REL}_adapters",
        help="compact-adapter root, relative to the repository unless absolute",
    )
    parser.add_argument(
        "--merger-temp-root",
        help=(
            "filesystem for verl's temporary full-model merge; defaults to "
            "the compact-adapter root"
        ),
    )
    parser.add_argument(
        "--if-exists",
        choices=("error", "verify"),
        default="error",
        help="verify an existing immutable export or fail (legacy default)",
    )
    args = parser.parse_args()

    try:
        record = export_checkpoint(
            args.step,
            _resolve(args.run_dir),
            _resolve(args.export_dir),
            (
                _resolve(args.merger_temp_root)
                if args.merger_temp_root
                else None
            ),
            if_exists=args.if_exists,
        )
    except (ExportError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(record["provenance"], indent=2))
    print(f"wrote compact adapter: {record['path']}")


if __name__ == "__main__":
    main()
