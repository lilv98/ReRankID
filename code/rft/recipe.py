# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""RFT sampling settings and the SFT initializer per model (paper Sec. 5, App. C.2).

RFT samples the 6,000-prompt post-training subset of utils/pools.py. RFT_RECIPE is a hash input of
the per-prompt sampling seed and must stay exactly as written to reproduce the released traces.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config  # noqa: E402

RFT_RECIPE = "rft-v1-rank1-bestof8"

# Best-of-8 rollouts from the SFT model.
N_SAMPLES = 8
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 2048
SEED = 0

# The paper's dev-selected SFT checkpoints: the default adapter of sample.py (--adapter overrides
# it) and the adapter_name_or_path of the RFT configs.
SFT_CHECKPOINT = {"onerec-1.7b": "checkpoint-1035", "onerec-8b": "checkpoint-88"}
MODELS = sorted(SFT_CHECKPOINT)


def sft_adapter(model):
    """Local SFT LoRA adapter that initializes RFT for `model`."""
    return os.path.join(config.save_dir(model, "sft"), SFT_CHECKPOINT[model])
