#!/usr/bin/env python3
"""Validate the immutable identities and final experiment scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "final_manifest.json"
REQUIRED_RECIPES = {"sft", "rft", "opd", "rlvr"}
REQUIRED_MODELS = {"1.7b-pro", "8b-pro"}
VALID_STATUS = {"complete", "pending"}


def iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)


def validate_manifest(payload, root=ROOT):
    errors = []

    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    scope = payload.get("scope", {})
    if set(scope.get("models", [])) != REQUIRED_MODELS:
        errors.append("scope.models must contain exactly the two Pro scales")
    if scope.get("only_pro_models") is not True:
        errors.append("scope.only_pro_models must be true")
    if scope.get("sid_only_post_training") is not True:
        errors.append("scope.sid_only_post_training must be true")

    paper_commit = payload.get("repository", {}).get("paper_submodule_commit", "")
    if len(paper_commit) != 40:
        errors.append("repository.paper_submodule_commit must be a full Git SHA")

    base_models = payload.get("base_models", {})
    if set(base_models) != REQUIRED_MODELS:
        errors.append("base_models must contain exactly the two Pro scales")
    for policy, spec in base_models.items():
        if "pro" not in str(spec.get("model_id", "")).lower():
            errors.append(f"{policy} does not resolve to a Pro model")
        for field in ("revision", "config_sha256", "weight_sha256"):
            value = spec.get(field, "")
            expected_length = 40 if field == "revision" else 64
            if len(value) != expected_length:
                errors.append(f"{policy}.{field} has the wrong length")

    recipes = payload.get("recipes", {})
    if set(recipes) != REQUIRED_RECIPES:
        errors.append(f"recipes must be exactly {sorted(REQUIRED_RECIPES)}")
    for recipe_name, recipe in recipes.items():
        runbook = root / str(recipe.get("runbook", ""))
        if not runbook.is_file():
            errors.append(f"{recipe_name} runbook does not exist: {runbook}")
        for arm_name, arm in recipe.get("arms", {}).items():
            status = arm.get("status")
            if status not in VALID_STATUS:
                errors.append(f"{recipe_name}.{arm_name} has invalid status {status!r}")
            if "pro" not in arm_name:
                errors.append(f"{recipe_name}.{arm_name} is not an explicit Pro arm")
            config = arm.get("config")
            if config and not (root / config).is_file():
                errors.append(f"{recipe_name}.{arm_name} config is missing: {config}")
            if status == "complete":
                for field in (
                    "dev_anchor_ndcg5",
                    "test_anchor_ndcg5",
                    "test_evidence_sha256",
                ):
                    if field not in arm:
                        errors.append(f"{recipe_name}.{arm_name} lacks {field}")
                evidence = arm.get("test_evidence_sha256", "")
                if len(evidence) != 64:
                    errors.append(
                        f"{recipe_name}.{arm_name} test evidence is not SHA-256"
                    )
            elif "test_anchor_ndcg5" in arm:
                errors.append(f"{recipe_name}.{arm_name} freezes a pending test result")

    sft = recipes.get("sft", {})
    if sft.get("annotation_target_rows") != 12_000:
        errors.append("SFT annotation target must be 12,000 rows")
    if sft.get("student_prompt_view") != "sid":
        errors.append("SFT student input must be SID-only")
    if sft.get("teacher_annotation_view") != "caption":
        errors.append("SFT teacher annotation view must be explicit")
    if sft.get("teacher_verified_positive_visible") is not True:
        errors.append("SFT teacher supervision must record the verified positive")

    expected_budgets = {
        "rft": ("rollout_prompt_budget", 6_000),
        "opd": ("trajectory_prompt_budget", 6_000),
        "rlvr": ("training_prompt_budget", 6_000),
    }
    for recipe_name, (field, expected) in expected_budgets.items():
        if recipes.get(recipe_name, {}).get(field) != expected:
            errors.append(f"{recipe_name}.{field} must be {expected}")

    artifacts = payload.get("wandb", {}).get("artifacts", {})
    for key, artifact in artifacts.items():
        qualified = artifact.get("qualified_name", "")
        if not qualified or ":" not in qualified:
            errors.append(f"artifact {key} lacks an exact qualified version")
        if qualified.endswith(":latest"):
            errors.append(f"artifact {key} uses mutable alias latest")
        if len(artifact.get("digest", "")) != 32:
            errors.append(f"artifact {key} has an invalid digest")

    forbidden = (
        "OneRec-8B\"",
        "OneRec-8B/",
    )
    joined = "\n".join(iter_strings(payload))
    for needle in forbidden:
        if needle.lower() in joined.lower():
            errors.append(f"manifest contains forbidden stale identity {needle!r}")

    rlvr_arms = recipes.get("rlvr", {}).get("arms", {})
    expected_rlvr = {
        f"{policy}_{variant}"
        for policy in REQUIRED_MODELS
        for variant in ("sft_ndcg5", "sft_hit1")
    }
    if set(rlvr_arms) != expected_rlvr:
        errors.append("RLVR manifest must contain the two registered variants per scale")

    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate_manifest(payload)
    if errors:
        raise SystemExit("\n".join(f"- {error}" for error in errors))
    print(f"OK: {args.manifest}")


if __name__ == "__main__":
    main()
