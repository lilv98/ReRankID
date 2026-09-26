"""Normalize the RoPE schema of a merged OPD teacher.

LLaMA-Factory exports with Transformers 5 record Qwen3's RoPE base under
``rope_parameters``. The Transformers 4 and vLLM stack used by OPD also
consults the legacy top-level ``rope_theta`` field. Keep both representations
equal so the frozen teacher is interpreted identically during training and
evaluation.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd.contract import SUPPORTED_POLICIES, policy_spec
from utils import config
from utils.model_config import normalize_rope_config


EXPECTED_THETA = 1_000_000


def normalize_config(path, expected_theta=EXPECTED_THETA):
    """Backward-compatible OPD entrypoint for the shared normalizer."""

    return normalize_rope_config(path, expected_theta)


def normalize_policy(policy, root=config.ROOT):
    relative = policy_spec(policy)["teacher"]["merged_rel"]
    path = os.path.join(root, relative, "config.json")
    if not os.path.isfile(path):
        raise ValueError(f"missing merged teacher for {policy}: {path}")
    return normalize_config(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=SUPPORTED_POLICIES, required=True)
    args = parser.parse_args()
    print(json.dumps(normalize_policy(args.policy), indent=2))


if __name__ == "__main__":
    main()
