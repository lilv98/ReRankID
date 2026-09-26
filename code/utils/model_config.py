"""Small, dependency-free model-config normalization helpers."""

import json
import os
import tempfile


def normalize_rope_config(path, expected_theta):
    """Keep modern and legacy Qwen RoPE fields equal in an exported config.

    LLaMA-Factory may export only ``rope_parameters.rope_theta`` under
    Transformers 5, while the Transformers 4 + vLLM training stack also reads
    top-level ``rope_theta``.  The nested value is treated as authoritative;
    conflicting values fail rather than being silently overwritten.
    """

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    parameters = payload.get("rope_parameters")
    nested = parameters.get("rope_theta") if isinstance(parameters, dict) else None
    legacy = payload.get("rope_theta")
    if nested != expected_theta:
        raise ValueError(
            f"{path}: nested rope theta {nested!r} != expected {expected_theta!r}"
        )
    if legacy not in (None, expected_theta):
        raise ValueError(
            f"{path}: legacy rope theta {legacy!r} != expected {expected_theta!r}"
        )

    changed = legacy is None
    if changed:
        payload["rope_theta"] = expected_theta
        directory = os.path.dirname(path)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".config.", suffix=".json", dir=directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.replace(temporary, path)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
    return {"path": path, "rope_theta": expected_theta, "changed": changed}
