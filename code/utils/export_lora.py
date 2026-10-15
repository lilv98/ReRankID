# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Export verl FSDP actor checkpoints as PEFT LoRA adapters (OPD and RLVR, paper Sec. 5).

Runs `python -m verl.model_merger merge` on <run-dir>/global_step_<step>/actor and keeps only the
resulting lora_adapter as <out>/step-<step:06d>/ (adapter_config.json, adapter_model.safetensors).
The merger materializes a full model in a temporary directory, which is deleted afterwards. --cpu
merges with CPU initialization and no visible GPU, which avoids running out of GPU memory for 8B.

    python code/utils/export_lora.py --run-dir saves/onerec-8b/opd --step 25 50 75 100 \\
        --out saves/onerec-8b/opd/adapters --cpu
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def export_step(run_dir, step, out, cpu=False):
    """Export one checkpoint step; returns the adapter directory."""
    actor_dir = os.path.join(run_dir, f"global_step_{step}", "actor")
    if not os.path.isdir(actor_dir):
        raise SystemExit(f"missing checkpoint: {actor_dir}")
    final_dir = os.path.join(out, f"step-{step:06d}")
    if os.path.exists(final_dir):
        raise SystemExit(f"refusing to overwrite existing export: {final_dir}")
    os.makedirs(out, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix=f".merge_step_{step}_", dir=out)
    command = [sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
               "--local_dir", actor_dir, "--target_dir", temp_dir, "--trust-remote-code"]
    env = os.environ.copy()
    if cpu:
        command.append("--use_cpu_initialization")
        env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        subprocess.run(command, check=True, env=env)
        adapter_dir = os.path.join(temp_dir, "lora_adapter")
        missing = [f for f in ADAPTER_FILES if not os.path.isfile(os.path.join(adapter_dir, f))]
        if missing:
            raise SystemExit(f"verl merger produced no PEFT adapter (missing {missing})")
        os.replace(adapter_dir, final_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"wrote {final_dir}")
    return final_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, help="verl trainer.default_local_dir")
    parser.add_argument("--step", type=int, nargs="+", required=True)
    parser.add_argument("--out", required=True, help="adapter root; writes <out>/step-XXXXXX")
    parser.add_argument("--cpu", action="store_true",
                        help="merge with CPU initialization and CUDA_VISIBLE_DEVICES=''")
    args = parser.parse_args()
    for step in args.step:
        export_step(args.run_dir, step, args.out, cpu=args.cpu)


if __name__ == "__main__":
    main()
