# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Install the OPD verl patches at interpreter start (run_opd.sh puts this directory on
PYTHONPATH)."""

import os

if os.environ.get("RERANKID_OPD_NORMALIZE_TOPK") == "1":
    from opd.verl_topk_patch import install_import_hook

    install_import_hook()

if os.environ.get("RERANKID_OPD_CPU_LAYERED_LORA") == "1":
    from opd.verl_lora_sync_patch import install_import_hook

    install_import_hook()
