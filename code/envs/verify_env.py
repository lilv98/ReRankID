"""Read-only ReRankID environment/GPU check; never downloads or prints secrets."""
import argparse
import importlib.metadata as md
import json
import os
import re
import subprocess
import sys
from pathlib import Path


PACKAGES = (
    "torch", "transformers", "vllm", "llamafactory", "verl", "ray",
    "tensordict", "flash_attn", "numpy", "pandas", "pyarrow", "wandb",
)

EXPECTED = {
    "cu128": {
        "python_prefix": "3.11", "torch_cuda": "12.8", "vllm": "0.19.1",
        "torch": {"lf": "2.9.1+cu128", "verl": "2.10.0"},
        "transformers": {"lf": "5.6.0", "verl": "4.57.6"},
        "driver_min": 535, "gpu_count": 2, "backend": "openrouter",
    },
    "cu130": {
        "python_prefix": "3.12", "torch_cuda": "13.0", "vllm": "0.24.0",
        "torch": {"lf": "2.11.0", "verl": "2.11.0"},
        "transformers": {"lf": "5.6.0", "verl": "5.6.0"},
        "driver_min": 580, "gpu_count": 4, "backend": "vertex",
    },
}

ROLE_ENV_VARS = {
    "lf": "RERANKID_LF_ENV_NAME",
    "verl": "RERANKID_VERL_ENV_NAME",
}


def expected_env_name(role, environ=os.environ):
    return environ.get(ROLE_ENV_VARS[role], f"sid-{role}")


def infer_role(env_label, environ=os.environ):
    matches = [
        role
        for role in ROLE_ENV_VARS
        if env_label == expected_env_name(role, environ)
    ]
    return matches[0] if len(matches) == 1 else None


def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def driver_version():
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if p.returncode != 0:
            return None
        value = next((line.strip() for line in p.stdout.splitlines() if line.strip()), None)
        return value if value and re.fullmatch(r"\d+(?:\.\d+)+", value) else None
    except Exception:
        return None


def has_openrouter_auth():
    """Check presence only. Never return or print a key value."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return True
    path = Path(os.environ.get(
        "API_KEYS_FILE", str(Path(__file__).resolve().parents[2] / "api_keys.json")))
    try:
        data = json.loads(path.read_text())
        value = data.get("openrouter")
        if isinstance(value, str):
            return bool(value)
        if isinstance(value, dict):
            return bool(value.get("api_key") or value.get("token") or value.get("key"))
    except (OSError, json.JSONDecodeError):
        pass
    return False


def has_vertex_adc():
    """Check that ADC can be resolved locally without refreshing or printing credentials."""
    try:
        import google.auth
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return credentials is not None
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["lf", "verl"], default=None,
                    help="expected environment role; inferred from CONDA_DEFAULT_ENV when omitted")
    ap.add_argument("--profile", choices=["cu128", "cu130"], default=None,
                    help="expected machine profile; defaults to RERANKID_MACHINE_PROFILE")
    ap.add_argument("--imports", action="store_true",
                    help="also import the role's critical modules (slower but still read-only)")
    ap.add_argument(
        "--skip-llm-auth",
        action="store_true",
        help=(
            "skip provider-credential readiness only for workloads, such as "
            "OPD, that never call an external LLM"
        ),
    )
    args = ap.parse_args()

    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    # `conda activate` normally exposes the short name, while `conda run` may
    # expose the full prefix when environment-scoped variables are configured.
    env_label = Path(env_name).name if env_name else None
    role = args.role
    if role is None:
        role = infer_role(env_label)
    profile = args.profile or os.environ.get("RERANKID_MACHINE_PROFILE")
    backend = os.environ.get("RERANKID_LLM_BACKEND")
    errors = []

    if not env_label:
        errors.append("no active Conda environment")
    if role is None:
        errors.append(
            "cannot infer role from the configured environment names; pass --role"
        )
    if profile is None:
        errors.append("cannot infer machine profile; run setup or pass --profile")
    elif profile not in EXPECTED:
        errors.append(f"unknown machine profile marker: {profile}")
    if backend not in {"openrouter", "vertex"}:
        errors.append("RERANKID_LLM_BACKEND must be openrouter or vertex")

    result = {
        "conda_env": env_name,
        "conda_env_name": env_label,
        "role": role,
        "machine_profile": profile,
        "llm_backend": backend,
        "llm_auth_check_skipped": args.skip_llm_auth,
        "python": sys.version.split()[0],
        "driver": driver_version(),
    }
    for name in PACKAGES:
        result[name] = version(name)

    try:
        import torch
        result.update({
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
        })
        if result["cuda_available"]:
            result["gpus"] = [torch.cuda.get_device_name(i) for i in range(result["gpu_count"])]
        else:
            errors.append("torch is installed but CUDA is unavailable")
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
        errors.append("torch import failed")

    if env_label and role:
        expected_name = expected_env_name(role)
        if env_label != expected_name:
            errors.append(f"expected conda env {expected_name}, got {env_name}")
    if role == "lf" and not result["llamafactory"]:
        errors.append("sid-lf is missing llamafactory")
    if role == "verl":
        for package in ("vllm", "verl", "ray", "tensordict", "flash_attn"):
            if not result[package]:
                errors.append(f"sid-verl is missing {package}")

    if profile in EXPECTED:
        exp = EXPECTED[profile]
        if not result["python"].startswith(exp["python_prefix"]):
            errors.append(f"{profile} expects Python {exp['python_prefix']}.x")
        if result.get("torch_cuda") != exp["torch_cuda"]:
            errors.append(f"{profile} expects torch CUDA {exp['torch_cuda']}")
        if role and result["torch"] != exp["torch"][role]:
            errors.append(f"{profile} sid-{role} expects torch {exp['torch'][role]}")
        if role and result["transformers"] != exp["transformers"][role]:
            errors.append(
                f"{profile} sid-{role} expects transformers {exp['transformers'][role]}"
            )
        if role == "verl" and result["vllm"] != exp["vllm"]:
            errors.append(f"{profile} sid-verl expects vLLM {exp['vllm']}")
        if backend and backend != exp["backend"]:
            errors.append(f"{profile} expects LLM backend {exp['backend']}, got {backend}")
        driver = result.get("driver")
        if not driver:
            errors.append("cannot read NVIDIA driver version")
        elif int(driver.split(".", 1)[0]) < exp["driver_min"]:
            errors.append(f"{profile} requires NVIDIA driver >= {exp['driver_min']}")
        if result.get("gpu_count") != exp["gpu_count"]:
            errors.append(f"{profile} expects {exp['gpu_count']} visible GPUs")
        marker = os.environ.get("RERANKID_MACHINE_PROFILE")
        if not marker:
            errors.append("environment has no RERANKID_MACHINE_PROFILE marker")
        elif marker != profile:
            errors.append(f"environment marker is {marker}, requested profile is {profile}")

    # LLM API clients are a sid-verl/text-scoring concern. sid-lf only trains
    # from prepared datasets and should not duplicate provider SDKs.
    if role != "lf" and backend == "openrouter" and not args.skip_llm_auth:
        result["openrouter_auth_configured"] = has_openrouter_auth()
        if not result["openrouter_auth_configured"]:
            errors.append("OpenRouter backend selected but no key is configured")
    elif role != "lf" and backend == "vertex" and not args.skip_llm_auth:
        result["vertex_clients_installed"] = bool(version("google-auth") and version("anthropic"))
        if not result["vertex_clients_installed"]:
            errors.append("Vertex backend selected but google-auth/anthropic are missing")
        result["vertex_adc_configured"] = has_vertex_adc()
        if not result["vertex_adc_configured"]:
            errors.append("Vertex backend selected but Application Default Credentials are unavailable")

    if args.imports:
        try:
            if role == "lf":
                import llamafactory  # noqa: F401
            elif role == "verl":
                from vllm import LLM, SamplingParams  # noqa: F401
                from vllm.lora.request import LoRARequest  # noqa: F401
                from vllm.sampling_params import StructuredOutputsParams  # noqa: F401
                import verl.trainer.main_ppo  # noqa: F401
            result["critical_imports"] = "ok"
        except Exception as exc:
            result["critical_imports"] = f"{type(exc).__name__}: {exc}"
            errors.append("critical role imports failed")

    result["status"] = "ok" if not errors else "error"
    if errors:
        result["errors"] = errors
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
