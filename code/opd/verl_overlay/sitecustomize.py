"""Activate the scoped ReRankID OPD loss correction at Python startup."""

import os


if os.environ.get("RERANKID_OPD_NORMALIZE_TOPK") == "1":
    from opd.verl_topk_patch import install_import_hook

    install_import_hook()

if os.environ.get("RERANKID_OPD_CPU_LAYERED_LORA") == "1":
    from opd.verl_lora_sync_patch import install_import_hook

    install_import_hook()

if os.environ.get("RERANKID_OPD_CHECKPOINT_ARCHIVE") == "1":
    from opd.verl_checkpoint_hook import install_import_hook

    install_import_hook()
