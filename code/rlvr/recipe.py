# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""RLVR data sizes and paths (paper Sec. 5, App. C.4).

Both scales and both rewards share the data and every training setting (run_grpo.sh); only the
reward function differs between rlvr-ndcg5 and rlvr-hit1. The paths are printed as shell exports:

    eval "$(python code/rlvr/recipe.py --model onerec-8b --reward ndcg5 --shell)"

MODEL (merged SFT initializer), DATA_DIR, OUTDIR (verl checkpoints), EXPORT (LoRA adapters),
RESULTS (dev and test score files), REWARD_FN and EXPERIMENT; README.md and run_grpo.sh use them.
"""
import argparse
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config  # noqa: E402

TRAIN_ROWS = 6_000      # shared post-training prompts
VAL_ROWS = 400          # benchmark dev users, monitoring only
VAL_K = 64
VAL_N_CAND = 30
REWARDS = {"ndcg5": "compute_score", "hit1": "compute_score_hit1"}   # in reward_rerank.py


def paths(model, reward):
    method = f"rlvr-{reward}"
    outdir = config.save_dir(model, method)
    return {
        "MODEL": config.merged_sft(model),
        "DATA_DIR": config.RLVR_DIR,
        "OUTDIR": outdir,
        "EXPORT": f"{outdir}/adapters",
        "RESULTS": f"{config.RESULTS}/posttraining/{method}/{model}",
        "REWARD_FN": REWARDS[reward],
        "EXPERIMENT": f"{method}_{model}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=sorted(config.MODELS), required=True)
    parser.add_argument("--reward", choices=sorted(REWARDS), default="ndcg5")
    parser.add_argument("--shell", action="store_true", help="print shell export lines")
    args = parser.parse_args()
    for key, value in paths(args.model, args.reward).items():
        print(f"export {key}={shlex.quote(str(value))}" if args.shell else f"{key}={value}")


if __name__ == "__main__":
    main()
