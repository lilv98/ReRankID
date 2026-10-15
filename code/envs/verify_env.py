# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Check an installed ReRankID environment against its pinned profile (read-only).

Prints package versions, the torch CUDA runtime and the visible GPUs, and exits non-zero if the
Python, torch, transformers or vLLM versions differ from the profile pins or a role package is
missing. --imports also imports the role's critical modules.

    python code/envs/verify_env.py --role verl --profile cu130 --imports
"""
import argparse
import importlib.metadata as md
import json
import sys

PACKAGES = ("torch", "transformers", "vllm", "llamafactory", "verl", "ray", "tensordict",
            "flash_attn", "peft", "numpy", "pandas", "pyarrow")

EXPECTED = {
    "cu128": {"python": "3.11", "torch_cuda": "12.8", "vllm": "0.19.1",
              "torch": {"lf": "2.9.1+cu128", "verl": "2.10.0"},
              "transformers": {"lf": "5.6.0", "verl": "4.57.6"}},
    "cu130": {"python": "3.12", "torch_cuda": "13.0", "vllm": "0.24.0",
              "torch": {"lf": "2.11.0", "verl": "2.11.0"},
              "transformers": {"lf": "5.6.0", "verl": "5.6.0"}},
}
ROLE_PACKAGES = {"lf": ("llamafactory",), "verl": ("vllm", "verl", "ray", "tensordict", "flash_attn")}


def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--role", choices=["lf", "verl"], required=True)
    ap.add_argument("--profile", choices=sorted(EXPECTED), required=True)
    ap.add_argument("--imports", action="store_true", help="also import the role's critical modules")
    args = ap.parse_args()
    exp, role = EXPECTED[args.profile], args.role
    result = {"python": sys.version.split()[0], **{name: version(name) for name in PACKAGES}}
    errors = []

    try:
        import torch
        result["torch_cuda"] = torch.version.cuda
        result["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:  # noqa: BLE001
        errors.append(f"torch import failed: {type(exc).__name__}: {exc}")

    if not result["python"].startswith(exp["python"] + "."):
        errors.append(f"expected Python {exp['python']}.x")
    if result.get("torch_cuda") != exp["torch_cuda"]:
        errors.append(f"expected torch CUDA {exp['torch_cuda']}")
    for name in ("torch", "transformers"):
        if result[name] != exp[name][role]:
            errors.append(f"expected {name} {exp[name][role]}")
    if role == "verl" and result["vllm"] != exp["vllm"]:
        errors.append(f"expected vLLM {exp['vllm']}")
    errors += [f"missing {name}" for name in ROLE_PACKAGES[role] if not result[name]]

    if args.imports:
        try:
            if role == "lf":
                import llamafactory  # noqa: F401
            else:
                from vllm import LLM, SamplingParams  # noqa: F401
                from vllm.lora.request import LoRARequest  # noqa: F401
                from vllm.sampling_params import StructuredOutputsParams  # noqa: F401
                import verl.trainer.main_ppo  # noqa: F401
            result["imports"] = "ok"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"critical imports failed: {type(exc).__name__}: {exc}")

    result["status"] = "error" if errors else "ok"
    if errors:
        result["errors"] = errors
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
