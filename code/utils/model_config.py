# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the top-level rope_theta into a merged model's config.json (paper Sec. 5 initializers).

LLaMA-Factory under Transformers 5 may export only rope_parameters.rope_theta, while the verl and
vLLM stacks used by OPD and RLVR read the top-level rope_theta and would otherwise fall back to a
wrong default. Run once after every `llamafactory-cli export`:

    python code/utils/model_config.py saves/onerec-8b/sft-merged
"""
import argparse
import json
import os

ROPE_THETA = 1_000_000   # OneRec-1.7B-pro and OneRec-8B-Pro


def normalize_rope_config(path, expected_theta=ROPE_THETA):
    """Make top-level rope_theta equal the nested rope_parameters.rope_theta. Raises ValueError if
    either value differs from expected_theta. Returns {"path", "rope_theta", "changed"}."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    parameters = payload.get("rope_parameters")
    nested = parameters.get("rope_theta") if isinstance(parameters, dict) else None
    top_level = payload.get("rope_theta")
    if nested != expected_theta:
        raise ValueError(f"{path}: nested rope theta {nested!r} != expected {expected_theta!r}")
    if top_level not in (None, expected_theta):
        raise ValueError(f"{path}: top-level rope theta {top_level!r} != expected {expected_theta!r}")
    changed = top_level is None
    if changed:
        payload["rope_theta"] = expected_theta
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return {"path": path, "rope_theta": expected_theta, "changed": changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="merged model directory or its config.json")
    parser.add_argument("--rope-theta", type=int, default=ROPE_THETA)
    args = parser.parse_args()
    path = args.model
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    print(json.dumps(normalize_rope_config(path, args.rope_theta)))


if __name__ == "__main__":
    main()
