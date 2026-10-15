# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Renormalized teacher top-k forward KL for verl 0.8 (paper Sec. 5, OPD objective).

verl gathers the teacher's top-k probabilities from the full vocabulary but does not renormalize
them before its KL helper, so the truncated surrogate can go negative and is clamped to zero. This
module replaces only that helper: the teacher distribution is renormalized over its top-k support,
while the student keeps full-vocabulary probabilities, so the loss also moves student mass onto
that support. It is installed by verl_overlay/sitecustomize.py in the driver and by
ray_worker_setup in every Ray worker when RERANKID_OPD_NORMALIZE_TOPK=1.
"""

import importlib
import os
from importlib.metadata import version

from utils.import_hook import install

TARGET_MODULE = "verl.trainer.distillation.fsdp.losses"
# The replacement assumes the exact inputs of verl 0.8.0's helper; other versions are refused.
VERL_VERSION = "0.8.0"


def normalized_teacher_topk_kl(log_q, log_p):
    """KL from a renormalized teacher top-k distribution to full student q."""
    log_p = log_p.float()
    log_q = log_q.float()
    normalized_log_p = log_p - log_p.logsumexp(dim=-1, keepdim=True)
    p = normalized_log_p.exp()
    return (p * (normalized_log_p - log_q)).sum(dim=-1)


def activate(losses=None):
    """Replace losses.kl_divergence with normalized_teacher_topk_kl (idempotent)."""
    if losses is None:
        losses = importlib.import_module(TARGET_MODULE)
    if losses.kl_divergence is normalized_teacher_topk_kl:
        return
    if version("verl") != VERL_VERSION:
        raise RuntimeError(f"OPD requires verl {VERL_VERSION}, found {version('verl')}")
    losses.kl_divergence = normalized_teacher_topk_kl


def install_import_hook():
    """Patch the module when it is first imported, without importing torch now."""
    install(TARGET_MODULE, activate)


def ray_worker_setup():
    """Ray worker_process_setup_hook: install the top-k patch and, for 8B, the LoRA sync patch."""
    if os.environ.get("RERANKID_OPD_NORMALIZE_TOPK") != "1":
        raise RuntimeError("Ray worker is missing RERANKID_OPD_NORMALIZE_TOPK=1")
    install_import_hook()
    if os.environ.get("RERANKID_OPD_CPU_LAYERED_LORA") == "1":
        from opd.verl_lora_sync_patch import install_import_hook as install_lora_sync_hook

        install_lora_sync_hook()
