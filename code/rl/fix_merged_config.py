"""Make LLaMA-Factory exports safe across Transformers 4/5 RoPE schemas.

Transformers 5 serializes Qwen3's RoPE base under ``rope_parameters`` and can
drop the legacy top-level ``rope_theta`` field.  Transformers 4 reconstructs a
default top-level value, and vLLM's compatibility patch then gives that value
precedence over the correct nested value.  Keep both representations in every
merged model so the actor, reference, and benchmark use the same RoPE base.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.contract import POLICIES, policy_contract, rooted
from utils import config
from utils.model_config import normalize_rope_config


def normalize_config(path, expected_theta):
    """Backward-compatible stage entrypoint for the shared normalizer."""

    return normalize_rope_config(path, expected_theta)


def normalize_policy(name, root=config.ROOT):
    policy = policy_contract(name)
    path = rooted(root, os.path.join(policy["merged_rel"], "config.json"))
    if not os.path.isfile(path):
        raise ValueError(f"missing merged model for {name}: {path}")
    return normalize_config(path, policy["rope_theta"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["all", *sorted(POLICIES)], default="all")
    args = parser.parse_args()
    names = sorted(POLICIES) if args.policy == "all" else [args.policy]
    report = {name: normalize_policy(name) for name in names}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
