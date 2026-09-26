"""Filter SFT v3 annotations and build a self-contained LLaMA-Factory dataset.

The main output is ``data/sft_v3/{train,dev,dataset_info,stats}.json``. Quality gates are limited to
valid top-10 format, gold-at-rank-one correctness, explicit candidate-index grounding, and an
extreme rationale-length ceiling. Explicit answer-reveal phrases are reported for inspection but
are not rejected by a regex.

The requested stage and annotation key sets must match exactly. A pilot rebuild may also consume a
later complete 12k annotation file, which is fully validated before filtering back to pilot keys.
Candidate uniqueness, single-positive labels, annotation labels, pool hash, prompt version, and
provenance are checked before any example is written. This prevents arbitrary partial or stale
annotation runs from becoming formal training data.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
config.setup_env()

import argparse
import hashlib
import json
import re

import numpy as np

from sft.v3_contract import (EXPECTED_ROWS, FORMAL_TARGET_ROWS, PILOT_ROWS,
                             PROMPT_VERSION, RECIPE, is_smoke_path)
from sft.v3_stages import (manifest_split_sets, row_key, select_stage_rows,
                           validate_phase_manifest)
from utils.data_io import dump_json, load_rows
from utils.metrics import ndcg
from utils.prompts import TOP_K, build_context, render_candidates, rerank_instruction, rerank_system

POOL = config.SFT_V3_POOL
ANNOT = config.SFT_V3_ANNOTATIONS
OUTDIR = config.SFT_V3_DATA_DIR
MAX_COT_CHARS = 600
DEV_FRAC = 0.08
SEED = 0


def _same_path(left, right):
    return os.path.realpath(left) == os.path.realpath(right)


def validate_dataset_paths(pool, annotations, outdir, target_rows=None):
    """Keep canonical v3 builds and smoke builds in disjoint namespaces.

    The two canonical output directories are reserved for the fixed SID-only,
    rationale-plus-ranking recipe. Any other output must be an unmistakable
    smoke path; likewise, noncanonical smoke input must stay under that path.
    """
    full = _same_path(outdir, OUTDIR)
    pilot = _same_path(outdir, config.SFT_V3_PILOT_DATA_DIR)
    canonical_pool = _same_path(pool, POOL)
    canonical_annotations = _same_path(annotations, ANNOT)
    if full or pilot:
        if not (canonical_pool and canonical_annotations):
            raise SystemExit(
                "formal v3 output requires the canonical pool and annotations"
            )
        expected_target = PILOT_ROWS if pilot else EXPECTED_ROWS
        if target_rows != expected_target:
            raise SystemExit(
                f"{outdir} requires --target-rows {expected_target}, got {target_rows}"
            )
        return "pilot" if pilot else "full"
    else:
        if not is_smoke_path(outdir):
            raise SystemExit("custom dataset output must be under /tmp with 'smoke' in the path")
        for label, path, canonical in (
                ("pool", pool, canonical_pool),
                ("annotations", annotations, canonical_annotations)):
            if not canonical and not is_smoke_path(path):
                raise SystemExit(
                    f"custom {label} must be under /tmp with 'smoke' in the path"
                )
        if target_rows is not None:
            raise SystemExit("smoke dataset builds do not use --target-rows")
    return "smoke"


def canonical_dataset_info(train_file="train.json", dev_file="dev.json"):
    """Return the only SFT-v3 LLaMA-Factory registry entries."""
    columns = {"prompt": "instruction", "response": "output", "system": "system"}
    return {
        "sft_train": {"file_name": train_file, "columns": columns},
        "sft_dev": {"file_name": dev_file, "columns": columns},
    }


def validate_formal_input_sizes(n_pool_rows, n_annotations, scope):
    """Validate the full source pool and allowed forward-only annotation sizes."""
    if scope == "smoke":
        return
    if n_pool_rows != EXPECTED_ROWS:
        raise ValueError(
            f"formal v3 requires exactly {EXPECTED_ROWS} source pool rows, got {n_pool_rows}"
        )
    allowed = {PILOT_ROWS, EXPECTED_ROWS} if scope == "pilot" else {EXPECTED_ROWS}
    if n_annotations not in allowed:
        raise ValueError(
            f"formal {scope} dataset requires annotation rows in {sorted(allowed)}, "
            f"got {n_annotations}"
        )


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_annotations(path):
    """Read JSONL while tolerating only a crash-truncated final line."""
    rows, bad_lines = [], []
    with open(path) as f:
        lines = f.readlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad_lines.append(index)
    if bad_lines and bad_lines != [len(lines) - 1]:
        raise ValueError(f"annotation file has corrupt non-final lines: {bad_lines[:5]}")
    if bad_lines:
        print(f"WARNING: skipped one crash-truncated final line in {path}")
    return rows


def select_stage_annotations(annotations, full_pool_by_key, selected_pool_by_key, scope,
                             pool_sha256):
    """Validate the annotation-file key set and return rows for one dataset stage.

    A pilot build accepts either the partial 6k JSONL or a later complete 12k
    JSONL, which makes the pilot dataset reproducible after forward expansion.
    No other partial key set is accepted.
    """
    annotation_keys = [row_key(row) for row in annotations]
    if len(annotation_keys) != len(set(annotation_keys)):
        raise ValueError("annotation file has duplicate (uid, slate_id) keys")
    key_set = set(annotation_keys)
    full_keys = set(full_pool_by_key)
    selected_keys = set(selected_pool_by_key)
    if scope == "pilot":
        allowed = (selected_keys, full_keys)
        if key_set not in allowed:
            missing = len(selected_keys - key_set)
            extra = len(key_set - selected_keys)
            raise ValueError(
                f"pilot annotation keys must be exactly pilot-6k or full-12k: "
                f"pilot_missing={missing}, pilot_extra={extra}, rows={len(key_set)}"
            )
    elif key_set != selected_keys:
        missing = len(selected_keys - key_set)
        extra = len(key_set - selected_keys)
        raise ValueError(f"pool/annotation key mismatch: missing={missing}, extra={extra}")

    # Validate every row present in the canonical JSONL before filtering a
    # completed 12k file back to the pilot subset.
    validation_pool = full_pool_by_key if key_set == full_keys else selected_pool_by_key
    provenance = validate_inputs(validation_pool, annotations, pool_sha256)
    selected = [row for row in annotations if row_key(row) in selected_keys]
    if {row_key(row) for row in selected} != selected_keys:
        raise ValueError("selected annotation rows do not cover the requested stage")
    return selected, provenance


def mentions_candidate(cot_text, idx):
    """Whether a rationale explicitly refers to candidate ``idx`` as ``候选X``."""
    pattern = rf"候选\s*[\[【(（]?\s*{int(idx)}(?!\d)\s*[\]】)）]?"
    return re.search(pattern, cot_text or "") is not None


def passes_filters(ann):
    order = ann.get("order")
    n_candidates = len(ann.get("grades") or [])
    strict_order = (
        ann.get("valid_annotation") is True
        and isinstance(order, list)
        and len(order) == TOP_K
        and len(set(order)) == TOP_K
        and all(isinstance(index, int) and not isinstance(index, bool)
                and 0 <= index < n_candidates for index in order)
    )
    if ann.get("n_listed") != TOP_K or not strict_order or not ann.get("cot_text"):
        return False, "format"
    if ndcg(order, ann["grades"], TOP_K) < 0.999:
        return False, "correctness"
    gold = [int(i) for i in ann.get("gold_idx", [])]
    if len(gold) != 1 or not mentions_candidate(ann["cot_text"], gold[0]):
        return False, "grounding"
    if len(ann["cot_text"]) > MAX_COT_CHARS:
        return False, "length"
    return True, None


def validate_pool_row(row):
    """Return the one positive index after validating a complete candidate slate."""
    n = int(row["n_cand"])
    pids = [int(pid) for pid in row["cand_pids"]]
    sids = list(row["cand_sids"])
    captions = list(row["cand_captions"])
    grades = [int(grade) for grade in row["cand_grades"]]
    if not (len(pids) == len(sids) == len(captions) == len(grades) == n):
        raise ValueError(f"candidate arrays differ in length for uid={row['uid']}")
    if len(pids) != len(set(pids)):
        raise ValueError(f"duplicate candidate PID for uid={row['uid']}")
    if not all(sids) or not all(captions):
        raise ValueError(f"unrenderable candidate for uid={row['uid']}")
    positive_indices = [i for i, grade in enumerate(grades) if grade > 0]
    if len(positive_indices) != 1:
        raise ValueError(f"expected one positive for uid={row['uid']}")
    gold_pids = [int(pid) for pid in row.get("gold_pids", [])]
    if len(gold_pids) != 1 or pids.count(gold_pids[0]) != 1:
        raise ValueError(f"gold PID must occur exactly once for uid={row['uid']}")
    if pids[positive_indices[0]] != gold_pids[0]:
        raise ValueError(f"positive grade and gold PID disagree for uid={row['uid']}")
    if int(row.get("n_pos", 1)) != 1:
        raise ValueError(f"n_pos is not one for uid={row['uid']}")
    return positive_indices


def validate_inputs(pool_by_slate, annotations, pool_sha256):
    if not annotations:
        raise ValueError("annotation file is empty")
    pool_keys = set(pool_by_slate)
    annotation_keys = [(int(row["uid"]), str(row.get("slate_id", 0))) for row in annotations]
    if len(annotation_keys) != len(set(annotation_keys)):
        raise ValueError("annotation file has duplicate (uid, slate_id) keys")
    annotation_key_set = set(annotation_keys)
    annotation_by_key = dict(zip(annotation_keys, annotations))
    if annotation_key_set != pool_keys:
        missing = len(pool_keys - annotation_key_set)
        extra = len(annotation_key_set - pool_keys)
        raise ValueError(f"pool/annotation key mismatch: missing={missing}, extra={extra}")

    provenance_by_json = {
        json.dumps(row.get("provenance"), sort_keys=True, ensure_ascii=False)
        for row in annotations
    }
    if len(provenance_by_json) != 1:
        raise ValueError(f"annotation file has {len(provenance_by_json)} provenance variants")
    provenance = annotations[0].get("provenance") if annotations else None
    if not provenance:
        raise ValueError("annotations have no provenance")
    expected = {"recipe": RECIPE, "prompt_version": PROMPT_VERSION,
                "pool_sha256": pool_sha256}
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"annotation provenance {key} does not match v3 contract")

    for key, pool_row in pool_by_slate.items():
        expected_gold = validate_pool_row(pool_row)
        ann = annotation_by_key[key]
        if [int(g) for g in ann.get("grades", [])] != [int(g) for g in pool_row["cand_grades"]]:
            raise ValueError(f"annotation grades differ from pool for key={key}")
        if [int(i) for i in ann.get("gold_idx", [])] != expected_gold:
            raise ValueError(f"annotation gold_idx differs from pool for key={key}")
    return provenance


def assemble_example(pool_row, ann):
    """Build one final SID-only rationale-plus-ranking example."""
    render = "sid"
    ctx, _ = build_context(pool_row, int(pool_row["k"]), render)
    instruction = (rerank_instruction(
        ctx, render_candidates(pool_row, render), render, top_k=TOP_K) + "/think")
    output = (
        "<think>\n"
        + ann["cot_text"].strip()
        + "\n</think>\n\n"
        + json.dumps(ann["order"])
    )
    return {"instruction": instruction, "output": output, "system": rerank_system(render)}


def count_field(rows, field):
    counts = {}
    for row in rows:
        value = str(int(row[field]))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def describe_lengths(values):
    if not values:
        return {"n": 0}
    arr = np.asarray(values)
    return {
        "n": len(values),
        "min": int(arr.min()),
        "mean": round(float(arr.mean()), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "max": int(arr.max()),
    }


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--annotations", default=ANNOT)
    ap.add_argument("--target-rows", type=int, choices=FORMAL_TARGET_ROWS, default=None,
                    help="formal stage size; selects keys through the frozen phase manifest")
    ap.add_argument("--outdir", default=None)
    return ap


def main():
    args = build_arg_parser().parse_args()
    if args.outdir is None:
        args.outdir = (config.SFT_V3_PILOT_DATA_DIR
                       if args.target_rows == PILOT_ROWS else OUTDIR)

    scope = validate_dataset_paths(
        args.pool, args.annotations, args.outdir, args.target_rows
    )
    formal = scope != "smoke"

    pool_rows = load_rows(args.pool)
    pool_keys = [row_key(row) for row in pool_rows]
    if len(pool_keys) != len(set(pool_keys)):
        raise ValueError("pool has duplicate (uid, slate_id) keys")
    full_pool_by_slate = dict(zip(pool_keys, pool_rows))
    annotations = load_annotations(args.annotations)
    print(f"loaded {len(annotations)} annotation-file rows for {len(pool_rows)} source pool rows")
    validate_formal_input_sizes(len(pool_rows), len(annotations), scope)
    pool_sha256 = file_sha256(args.pool)
    phase_manifest = None
    phase_manifest_sha256 = None
    if formal:
        with open(config.SFT_V3_PHASE_MANIFEST) as handle:
            phase_manifest = json.load(handle)
        validate_phase_manifest(phase_manifest, pool_rows, pool_sha256)
        phase_manifest_sha256 = file_sha256(config.SFT_V3_PHASE_MANIFEST)
        selected_pool_rows = select_stage_rows(
            pool_rows, args.target_rows, phase_manifest, pool_sha256
        )
    else:
        selected_pool_rows = pool_rows
    pool_by_slate = {row_key(row): row for row in selected_pool_rows}
    selected_annotations, provenance = select_stage_annotations(
        annotations, full_pool_by_slate, pool_by_slate, scope, pool_sha256
    )

    rejects = {}
    examples, uids, example_keys, accepted_pool_rows = [], [], [], []
    annotation_by_key = {
        row_key(row): row for row in selected_annotations
    }
    for key, pool_row in pool_by_slate.items():
        ann = annotation_by_key[key]
        ok, reason = passes_filters(ann)
        if not ok:
            rejects[reason] = rejects.get(reason, 0) + 1
            continue
        if pool_row.get("profile_header"):
            rejects["profile_leak"] = rejects.get("profile_leak", 0) + 1
            continue
        examples.append(assemble_example(pool_row, ann))
        uids.append(int(ann["uid"]))
        example_keys.append(key)
        accepted_pool_rows.append(pool_row)

    n_in, n_out = len(selected_annotations), len(examples)
    pass_rate = n_out / n_in if n_in else 0.0
    print(f"\npass rate: {n_out}/{n_in} ({pass_rate:.1%})")
    print(f"rejections by reason: {rejects}")
    cot_lengths_all = [len(row["cot_text"]) for row in selected_annotations if row.get("cot_text")]
    accepted_keys = {row_key(row) for row in accepted_pool_rows}
    cot_lengths_accepted = [len(row["cot_text"]) for row in selected_annotations
                            if row_key(row) in accepted_keys]
    print(f"cot length stats: {describe_lengths(cot_lengths_all)}")

    os.makedirs(args.outdir, exist_ok=True)
    if formal:
        _, stage_dev_keys = manifest_split_sets(phase_manifest, args.target_rows)
        dev_keys = stage_dev_keys & set(example_keys)
        train_examples = [example for example, key in zip(examples, example_keys)
                          if key not in dev_keys]
        dev_examples = [example for example, key in zip(examples, example_keys)
                        if key in dev_keys]
        train_uids = {uid for uid, key in zip(uids, example_keys) if key not in dev_keys}
        dev_uid_set = {uid for uid, key in zip(uids, example_keys) if key in dev_keys}
    else:
        rng = np.random.default_rng(SEED)
        n_dev_target = max(1, int(len(examples) * DEV_FRAC)) if len(examples) >= 20 else 0
        unique_uids = sorted(set(uids))
        per_uid = {}
        for uid in uids:
            per_uid[uid] = per_uid.get(uid, 0) + 1
        dev_uids, assigned = set(), 0
        for index in rng.permutation(len(unique_uids)):
            if assigned >= n_dev_target:
                break
            uid = unique_uids[index]
            dev_uids.add(uid)
            assigned += per_uid[uid]
        train_examples = [example for example, uid in zip(examples, uids) if uid not in dev_uids]
        dev_examples = [example for example, uid in zip(examples, uids) if uid in dev_uids]
        train_uids = {uid for uid in uids if uid not in dev_uids}
        dev_uid_set = set(uids) & dev_uids
    if train_uids & dev_uid_set:
        raise AssertionError("train/dev user overlap")

    train_file = "train.json"
    dev_file = "dev.json"
    train_path = os.path.join(args.outdir, train_file)
    dev_path = os.path.join(args.outdir, dev_file)
    dump_json(train_path, train_examples)
    dump_json(dev_path, dev_examples)
    print(f"\nwrote {train_path}  n={len(train_examples)}")
    print(f"wrote {dev_path}  n={len(dev_examples)}")

    info_path = os.path.join(args.outdir, "dataset_info.json")
    info = canonical_dataset_info(train_file, dev_file)
    dump_json(info_path, info)

    final_distribution = {
        "positive_grade": count_field(accepted_pool_rows, "pos_grade"),
        "k": count_field(accepted_pool_rows, "k"),
        "n_cand": count_field(accepted_pool_rows, "n_cand"),
    }
    stats = {
        "recipe": RECIPE,
        "prompt_version": PROMPT_VERSION,
        "variant": "full",
        "render": "sid",
        "dataset_name": "sft",
        "scope": scope,
        "source_pool_rows": len(pool_rows),
        "target_rows": int(args.target_rows) if formal else len(pool_rows),
        "annotation_file_rows": len(annotations),
        "selected_annotations": len(selected_annotations),
        "pool_rows": len(pool_by_slate),
        "raw_annotations": n_in,
        "passed": n_out,
        "rejected": n_in - n_out,
        "pass_rate": round(pass_rate, 6),
        "rejections": rejects,
        "answer_reveal_audit_hits": sum(bool(row.get("answer_reveal_audit_hit"))
                                         for row in selected_annotations),
        "answer_reveal_audit_hits_accepted": sum(
            bool(annotation_by_key[key].get("answer_reveal_audit_hit"))
            for key in accepted_keys),
        "cot_chars_all": describe_lengths(cot_lengths_all),
        "cot_chars_accepted": describe_lengths(cot_lengths_accepted),
        "train_rows": len(train_examples),
        "dev_rows": len(dev_examples),
        "train_users": len(train_uids),
        "dev_users": len(dev_uid_set),
        "train_dev_user_overlap": 0,
        "pool_distribution": {
            "positive_grade": count_field(list(pool_by_slate.values()), "pos_grade"),
            "k": count_field(list(pool_by_slate.values()), "k"),
            "n_cand": count_field(list(pool_by_slate.values()), "n_cand"),
        },
        "accepted_distribution": final_distribution,
        "provenance_variants": [provenance],
        "sha256": {
            "pool": pool_sha256,
            **({"phase_manifest": phase_manifest_sha256} if formal else {}),
            "annotations": file_sha256(args.annotations),
            "train": file_sha256(train_path),
            "dev": file_sha256(dev_path),
            "dataset_info": file_sha256(info_path),
        },
    }
    stats_file = "stats.json"
    stats_path = os.path.join(args.outdir, stats_file)
    dump_json(stats_path, stats)
    print(f"registered sft_train/sft_dev in {info_path}")
    print(f"wrote data audit: {stats_path}")

    if examples:
        example = examples[0]
        print(f"\n=== example ===\ninstruction[:200]: {example['instruction'][:200]}")
        print(f"output: {example['output']}")


if __name__ == "__main__":
    main()
