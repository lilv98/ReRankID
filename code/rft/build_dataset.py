"""Build the ordinary RFT continuation dataset from frozen raw rollouts.

For each prompt, keep the first rollout that completed a valid top-10 answer,
put the single positive at rank one, and did not reproduce the identity order.
No teacher traces are replayed, no RFT-specific dev split is created, and no
regex or task-specific rationale-length filter is applied.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
config.setup_env()

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict

from rft.base_provenance import (
    BaseIdentityError,
    validate_recorded_base_identity,
)
from rft.contract import (DEFAULT_POLICY, MAX_TOKENS, N_SAMPLES, POLICIES,
                          RFT_RECIPE, RFT_TARGET_ROWS, SEED, TEMPERATURE, TOP_K,
                          TOP_P, is_smoke_path, policy_contract)
from rft.raw_integrity import validate_raw_records
from rft.stages import (manifest_key_set, row_key, select_stage_rows,
                        validate_phase_manifest)
from sft.build_dataset import assemble_example
from utils.data_io import dump_json, load_rows
from utils.metrics import is_identity_collapse


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_dirty():
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=config.ROOT, text=True,
    ).strip()
    return bool(status)


def load_jsonl(path):
    records = []
    with open(path, "rb") as raw_handle:
        payload = raw_handle.read()
    if not payload:
        return records
    if not payload.endswith(b"\n"):
        raise ValueError("raw RFT JSONL is truncated or missing its final newline")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"raw RFT JSONL is not valid UTF-8: {error}") from error
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise ValueError(f"blank raw RFT JSONL line at {line_number}")
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
    return records


def validate_scope(args, outdir, policy):
    expected_outdir = getattr(config, policy["data_attr"])
    canonical_inputs = all((
        os.path.realpath(args.pool) == os.path.realpath(config.POSTTRAIN_V1_POOL),
        os.path.realpath(args.manifest) == os.path.realpath(config.POSTTRAIN_V1_PHASE_MANIFEST),
        os.path.realpath(args.raw) == os.path.realpath(
            getattr(config, policy["samples_attr"])
        ),
    ))
    formal = canonical_inputs and os.path.realpath(outdir) == os.path.realpath(expected_outdir)
    if formal:
        if args.target_rows != RFT_TARGET_ROWS:
            raise SystemExit(
                f"formal --target-rows must be exactly {RFT_TARGET_ROWS}"
            )
        if git_dirty():
            raise SystemExit("formal RFT dataset assembly requires a clean git worktree")
        return "formal"
    if not is_smoke_path(outdir):
        raise SystemExit("nonformal RFT dataset output must be under /tmp with 'smoke' in the path")
    return "smoke"


def validate_sample_provenance(records, pool_sha, manifest_sha, scope, policy_name, policy):
    if not records:
        raise ValueError("RFT sample file is empty")
    variants = {
        json.dumps(record.get("provenance"), sort_keys=True, ensure_ascii=False)
        for record in records
    }
    digests = {record.get("provenance_sha256") for record in records}
    if len(variants) != 1 or len(digests) != 1:
        raise ValueError("RFT sample file contains multiple provenance variants")
    provenance = records[0].get("provenance") or {}
    expected = {
        "schema_version": 2,
        "recipe": RFT_RECIPE,
        "pool_sha256": pool_sha,
        "phase_manifest_sha256": manifest_sha,
        "policy": policy_name,
        "model": policy["model"],
        "sft_adapter_artifact": policy["adapter_artifact"],
        "sft_adapter_model_sha256": policy["adapter_sha256"],
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(f"sample provenance {field} differs from the frozen contract")
    try:
        validate_recorded_base_identity(provenance.get("base_model"), policy)
    except BaseIdentityError as exc:
        raise ValueError(f"sample provenance base model is invalid: {exc}") from exc
    payload = json.dumps(provenance, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    calculated = hashlib.sha256(payload.encode()).hexdigest()
    if digests != {calculated}:
        raise ValueError("sample provenance digest is invalid")
    if scope == "formal" and (
            provenance.get("scope") != "formal" or provenance.get("git_dirty") is not False):
        raise ValueError("formal RFT dataset requires clean formal rollout provenance")
    sampling = provenance.get("sampling")
    if scope == "formal" and not isinstance(sampling, dict):
        raise ValueError("sample provenance is missing sampling settings")
    if isinstance(sampling, dict) and (
        sampling.get("top_k") != TOP_K
        or sampling.get("stage") != "single-stage"
    ):
        raise ValueError("sample provenance parsing contract is invalid")
    if scope == "formal":
        expected_sampling = {
            "n": N_SAMPLES,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "seed": SEED,
            "top_k": TOP_K,
            "stage": "single-stage",
        }
        if sampling != expected_sampling:
            raise ValueError("formal sample provenance differs from frozen sampling settings")
    return provenance, calculated


def valid_order(order, n_cand):
    return (
        isinstance(order, list)
        and len(order) == TOP_K
        and len(set(order)) == TOP_K
        and all(isinstance(index, int) and not isinstance(index, bool)
                and 0 <= index < n_cand for index in order)
    )


def first_survivor(record, pool_row, rejection_counts):
    grades = [int(value) for value in pool_row["cand_grades"]]
    gold = [index for index, value in enumerate(grades) if value > 0]
    if len(gold) != 1:
        raise ValueError(f"pool row does not have one positive: {row_key(pool_row)}")
    if [int(value) for value in record.get("grades") or []] != grades:
        raise ValueError(f"sample grades differ from pool: {row_key(pool_row)}")
    if [int(value) for value in record.get("gold_idx") or []] != gold:
        raise ValueError(f"sample gold index differs from pool: {row_key(pool_row)}")
    for rollout in record.get("rollouts") or []:
        order = rollout.get("order")
        if not valid_order(order, len(grades)) or rollout.get("n_listed") != TOP_K:
            rejection_counts["format_or_unclosed"] += 1
            continue
        if rollout.get("finish_reason") == "length":
            rejection_counts["length_truncated"] += 1
            continue
        cot = (rollout.get("cot_text") or "").strip()
        if not cot:
            rejection_counts["empty_rationale"] += 1
            continue
        if order[0] != gold[0]:
            rejection_counts["not_rank1"] += 1
            continue
        if is_identity_collapse(order, TOP_K):
            rejection_counts["identity"] += 1
            continue
        return {"cot_text": cot, "order": order}, rollout
    return None, None


def distribution(rows, field):
    values = Counter(str(row[field]) for row in rows)
    return dict(sorted(values.items(), key=lambda item: int(item[0])))


def coverage_table(source_rows, retained_keys, field):
    grouped = defaultdict(lambda: [0, 0])
    for row in source_rows:
        key = str(row[field])
        grouped[key][0] += 1
        grouped[key][1] += row_key(row) in retained_keys
    return {
        key: {"prompts": total, "retained": kept,
              "coverage": round(kept / total, 4)}
        for key, (total, kept) in sorted(grouped.items(), key=lambda item: int(item[0]))
    }


def coverage_combo(source_rows, retained_keys):
    grouped = defaultdict(lambda: [0, 0])
    for row in source_rows:
        key = int(row["k"]), int(row["n_cand"])
        grouped[key][0] += 1
        grouped[key][1] += row_key(row) in retained_keys
    return {
        f"k{k}_N{n}": {"prompts": total, "retained": kept,
                        "coverage": round(kept / total, 4)}
        for (k, n), (total, kept) in sorted(grouped.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=sorted(POLICIES), default=DEFAULT_POLICY)
    parser.add_argument("--pool", default=config.POSTTRAIN_V1_POOL)
    parser.add_argument("--manifest", default=config.POSTTRAIN_V1_PHASE_MANIFEST)
    parser.add_argument("--raw", default=None)
    parser.add_argument("--target-rows", type=int, default=RFT_TARGET_ROWS)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()
    policy = policy_contract(args.policy)
    args.raw = args.raw or getattr(config, policy["samples_attr"])

    if args.target_rows != RFT_TARGET_ROWS:
        raise SystemExit(
            f"RFT dataset assembly supports only --target-rows {RFT_TARGET_ROWS}"
        )
    outdir = args.outdir or getattr(config, policy["data_attr"])
    scope = validate_scope(args, outdir, policy)
    pool_sha = file_sha256(args.pool)
    manifest_sha = file_sha256(args.manifest)
    with open(args.manifest) as handle:
        manifest = json.load(handle)
    full_rows = load_rows(args.pool)
    validate_phase_manifest(
        manifest, full_rows, pool_sha, manifest["excluded_sft_uid_sha256"]
    )
    selected_rows = select_stage_rows(full_rows, manifest, args.target_rows)
    selected_by_key = {row_key(row): row for row in selected_rows}

    records = load_jsonl(args.raw)
    provenance, provenance_sha = validate_sample_provenance(
        records, pool_sha, manifest_sha, scope, args.policy, policy
    )
    record_keys = [row_key(record) for record in records]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("sample file contains duplicate prompt records")
    record_key_set = set(record_keys)
    formal_keys = manifest_key_set(manifest, RFT_TARGET_ROWS)
    selected_keys = set(selected_by_key)
    if scope == "formal":
        if record_key_set != formal_keys:
            raise ValueError("formal sample keys must be exactly the completed 6k RFT subset")
    else:
        if not record_key_set <= selected_keys:
            raise ValueError("smoke sample keys fall outside the selected formal stage")
        selected_rows = [row for row in selected_rows if row_key(row) in record_key_set]
        selected_by_key = {row_key(row): row for row in selected_rows}
        selected_keys = set(selected_by_key)
    record_by_key = {row_key(record): record for record in records}
    samples_per_prompt = validate_raw_records(
        records,
        selected_by_key,
        provenance,
        formal=scope == "formal",
    )

    rejections = Counter()
    examples = []
    kept_rows = []
    kept_rollouts = []
    for row in selected_rows:
        annotation, rollout = first_survivor(record_by_key[row_key(row)], row, rejections)
        if annotation is None:
            continue
        examples.append(assemble_example(row, annotation))
        kept_rows.append(row)
        kept_rollouts.append(rollout)

    if not examples:
        raise ValueError("no RFT examples survived the frozen gate")
    retained_keys = {row_key(row) for row in kept_rows}
    os.makedirs(outdir, exist_ok=True)
    train_file = os.path.join(outdir, "train.json")
    info_file = os.path.join(outdir, "dataset_info.json")
    stats_file = os.path.join(outdir, "stats.json")
    dump_json(train_file, examples)
    columns = {"prompt": "instruction", "response": "output", "system": "system"}
    dump_json(info_file, {
        "rft_train": {"file_name": "train.json", "columns": columns},
        "sft_dev": {"file_name": "../sft_v3/dev.json", "columns": columns},
    })
    rationale_lengths = [len(rollout["cot_text"]) for rollout in kept_rollouts]
    output_tokens = [int(rollout["output_tokens"]) for rollout in kept_rollouts]
    stats = {
        "schema_version": 1,
        "recipe": RFT_RECIPE,
        "policy": args.policy,
        "scope": scope,
        "target_prompts": len(selected_rows),
        "retained_examples": len(examples),
        "prompt_coverage": round(len(examples) / len(selected_rows), 4),
        "one_trace_per_prompt": True,
        "raw_integrity": {
            "schema_version": 1,
            "verified_prompt_records": len(records),
            "samples_per_prompt": samples_per_prompt,
            "fields_reparsed_from_text": [
                "cot_text", "order", "n_listed", "diag", "ndcg10",
                "gold_rank", "valid", "rank1", "identity",
            ],
        },
        "selection": "first valid nonidentity rank1 rollout",
        "rejected_rollouts_before_first_survivor": dict(sorted(rejections.items())),
        "source_distribution": {
            "k": distribution(selected_rows, "k"),
            "n_cand": distribution(selected_rows, "n_cand"),
            "pos_grade": distribution(selected_rows, "pos_grade"),
        },
        "retained_distribution": {
            "k": distribution(kept_rows, "k"),
            "n_cand": distribution(kept_rows, "n_cand"),
            "pos_grade": distribution(kept_rows, "pos_grade"),
        },
        "coverage_by": {
            "k": coverage_table(selected_rows, retained_keys, "k"),
            "n_cand": coverage_table(selected_rows, retained_keys, "n_cand"),
            "pos_grade": coverage_table(selected_rows, retained_keys, "pos_grade"),
            "k_x_n": coverage_combo(selected_rows, retained_keys),
        },
        "retained_rationale_chars": {
            "min": min(rationale_lengths),
            "mean": round(sum(rationale_lengths) / len(rationale_lengths), 1),
            "max": max(rationale_lengths),
        },
        "retained_output_tokens": {
            "min": min(output_tokens),
            "mean": round(sum(output_tokens) / len(output_tokens), 1),
            "max": max(output_tokens),
        },
        "source": {
            "pool_sha256": pool_sha,
            "phase_manifest_sha256": manifest_sha,
            "raw_samples_sha256": file_sha256(args.raw),
            "sample_provenance_sha256": provenance_sha,
            "sample_git_commit": provenance["git_commit"],
            "base_model": provenance["base_model"],
        },
        "files": {
            "train_sha256": file_sha256(train_file),
            "dataset_info_sha256": file_sha256(info_file),
        },
    }
    dump_json(stats_file, stats)
    print(json.dumps(stats, indent=2))
    print(f"wrote {len(examples):,} examples to {train_file}")


if __name__ == "__main__":
    main()
