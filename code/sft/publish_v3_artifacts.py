"""Publish only the formal SFT-v3 transfer artifacts to W&B.

The source list is a closed allowlist. The script never walks a directory and
does not accept source paths from the command line. It also refuses a dirty Git
worktree, symlinks, temporary paths, smoke files, and missing files.

Before any upload, it verifies the full frozen teacher provenance, the formal
12,000-row audit, the canonical full SID dataset, nonempty train/dev splits,
and the exact two-entry LLaMA-Factory registry.

The rerank and sft artifacts form one bundle. They are always validated and
published together with the same bundle digest, so a receiver can reject a
partially updated pair.

Run the manifest-only check after committing the exact code used to create the
artifacts, then opt in to the paired upload:

  python code/sft/publish_v3_artifacts.py
  python code/sft/publish_v3_artifacts.py --upload
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
from utils.artifact_release import (git_snapshot, log_immutable_artifact,
                                    require_clean_pushed_release)

config.setup_env()

import argparse
import hashlib
import json

from sft.annotate import cost_summary, prompt_sha256
from sft.v3_contract import (EXPECTED_ROWS, FORMAT_RETRIES, MAX_TOKENS,
                             PROMPT_VERSION, RECIPE, TEACHER_BACKEND,
                             TEACHER_EFFORT, TEACHER_MODEL, TEACHER_TEMPERATURE,
                             TEACHER_THINKING_LEVEL, THINKING_BUDGET, USERS_PER_GRADE)
from sft.v3_stages import validate_phase_manifest

GROUPS = {
    "rerank": (
        (os.path.join(config.RERANK_DIR, "sft_pool_v3.parquet"),
         "sft_pool_v3.parquet"),
        (config.SFT_V3_PHASE_MANIFEST, "sft_v3_phase_manifest.json"),
        (os.path.join(config.RERANK_DIR, "sft_annotations_v3.jsonl"),
         "sft_annotations_v3.jsonl"),
        (os.path.join(config.RERANK_DIR, "sft_annotations_v3_cost.json"),
         "sft_annotations_v3_cost.json"),
        (os.path.join(config.RERANK_DIR, "sft_v3_audit.json"),
         "sft_v3_audit.json"),
    ),
    "sft": (
        (os.path.join(config.DATA, "sft_v3", "train.json"), "train.json"),
        (os.path.join(config.DATA, "sft_v3", "dev.json"), "dev.json"),
        (os.path.join(config.DATA, "sft_v3", "dataset_info.json"),
         "dataset_info.json"),
        (os.path.join(config.DATA, "sft_v3", "stats.json"), "stats.json"),
    ),
}


def sha256(path):
    """Return the SHA-256 digest for one regular file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state():
    """Return the current commit and refuse any tracked or untracked change."""
    snapshot = git_snapshot(config.ROOT)
    if snapshot["dirty"]:
        raise SystemExit("refusing to publish from a dirty Git worktree")
    return snapshot["commit"]


def _is_within(path, directory):
    try:
        return os.path.commonpath((path, directory)) == directory
    except ValueError:
        return False


def _reject_symlink_components(path, root):
    """Reject a source whose file or repository-relative parent is a symlink."""
    relative = os.path.relpath(path, root)
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        raise SystemExit(f"refusing non-repository input: {path}")
    current = root
    for component in relative.split(os.sep):
        current = os.path.join(current, component)
        if os.path.islink(current):
            raise SystemExit(f"refusing symlink input: {path}")


def build_manifest(groups):
    """Validate the closed allowlist and return hashes and sizes."""
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise SystemExit(f"refusing unlisted artifact group: {sorted(unknown)}")

    root = os.path.abspath(config.ROOT)
    root_real = os.path.realpath(root)
    manifest = {}
    for group in groups:
        files = []
        seen_names = set()
        for source, artifact_name in GROUPS[group]:
            lexical = os.path.abspath(source)
            _reject_symlink_components(lexical, root)
            real = os.path.realpath(lexical)

            if not _is_within(real, root_real):
                raise SystemExit(f"refusing non-repository input: {source}")
            if _is_within(real, os.path.realpath("/tmp")):
                raise SystemExit(f"refusing temporary input: {source}")
            if "smoke" in real.lower() or "smoke" in artifact_name.lower():
                raise SystemExit(f"refusing smoke input: {source}")
            if artifact_name in seen_names:
                raise SystemExit(
                    f"refusing duplicate artifact name in {group}: {artifact_name}"
                )
            if not os.path.isfile(real):
                raise SystemExit(f"missing formal v3 file: {source}")

            seen_names.add(artifact_name)
            files.append({
                "source": os.path.relpath(real, root_real),
                "artifact_name": artifact_name,
                "bytes": os.path.getsize(real),
                "sha256": sha256(real),
            })
        manifest[group] = files
    return manifest


def _entry_by_name(manifest, group, name):
    return next(entry for entry in manifest[group] if entry["artifact_name"] == name)


def _load_json(path):
    with open(path) as handle:
        return json.load(handle)


def load_pool_keys(path):
    """Read only the formal join columns from a parquet pool."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["uid", "slate_id"])
    values = table.to_pydict()
    return [
        (int(uid), str(slate_id))
        for uid, slate_id in zip(values["uid"], values["slate_id"])
    ]


def load_pool_stage_rows(path):
    """Read only columns needed to recompute the frozen phase manifest."""
    import pyarrow.parquet as pq

    columns = ["uid", "slate_id", "pool", "pos_grade", "k", "n_cand"]
    values = pq.read_table(path, columns=columns).to_pydict()
    return [
        {column: values[column][index] for column in columns}
        for index in range(len(values["uid"]))
    ]


def expected_teacher_provenance(pool_sha256, commit):
    """Exact teacher contract required for every formal annotation row."""
    return {
        "recipe": RECIPE,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "pool_sha256": pool_sha256,
        "teacher_model": TEACHER_MODEL,
        "backend": TEACHER_BACKEND,
        "effort": TEACHER_EFFORT,
        "temperature": TEACHER_TEMPERATURE,
        "thinking_level": TEACHER_THINKING_LEVEL,
        "max_tokens": MAX_TOKENS,
        "thinking_budget": THINKING_BUDGET,
        "format_retries": FORMAT_RETRIES,
        "git_commit": commit,
        "git_dirty": False,
    }


def validate_semantics(manifest, commit):
    """Reject incomplete or inconsistent data even when formal filenames are used."""
    pool_entry = _entry_by_name(manifest, "rerank", "sft_pool_v3.parquet")
    phase_entry = _entry_by_name(manifest, "rerank", "sft_v3_phase_manifest.json")
    annotation_entry = _entry_by_name(manifest, "rerank", "sft_annotations_v3.jsonl")
    audit = _load_json(config.SFT_V3_AUDIT)
    cost = _load_json(config.SFT_V3_ANNOTATION_COST)
    stats = _load_json(os.path.join(config.SFT_V3_DATA_DIR, "stats.json"))
    phase_manifest = _load_json(config.SFT_V3_PHASE_MANIFEST)
    validate_phase_manifest(
        phase_manifest,
        load_pool_stage_rows(os.path.join(config.RERANK_DIR, "sft_pool_v3.parquet")),
        pool_entry["sha256"],
    )
    full_stage = (phase_manifest.get("stages") or {}).get(str(EXPECTED_ROWS)) or {}
    if phase_entry["sha256"] != sha256(config.SFT_V3_PHASE_MANIFEST):
        raise SystemExit("phase manifest hash differs from the closed allowlist")

    if audit.get("recipe") != RECIPE or audit.get("formal") is not True:
        raise SystemExit("formal audit does not declare the frozen v3 recipe")
    if audit.get("pool_sha256") != pool_entry["sha256"]:
        raise SystemExit("formal audit pool hash does not match the allowlisted pool")
    if (audit.get("target_rows") != EXPECTED_ROWS or
            audit.get("source_pool_rows") != EXPECTED_ROWS or
            audit.get("phase_manifest_sha256") != phase_entry["sha256"] or
            audit.get("selected_key_sha256") != full_stage.get("key_sha256")):
        raise SystemExit("formal audit does not match the frozen full-12k phase")
    composition = audit.get("composition") or {}
    if composition.get("rows") != EXPECTED_ROWS or composition.get("users") != EXPECTED_ROWS:
        raise SystemExit("formal audit does not contain 12,000 rows and users")
    if composition.get("positive_grade_distribution") != {
            "2": USERS_PER_GRADE, "3": USERS_PER_GRADE}:
        raise SystemExit("formal audit does not contain 6,000 users per grade")
    if composition.get("candidate_pid_duplicates") != 0:
        raise SystemExit("formal audit reports duplicate candidate PIDs")
    if composition.get("user_overlap_dev_test_rl") != 0:
        raise SystemExit("formal audit reports forbidden user overlap")

    if cost.get("recipe") != RECIPE or cost.get("rows") != EXPECTED_ROWS:
        raise SystemExit("annotation cost summary is not a complete formal v3 run")
    if cost.get("valid_annotation") != EXPECTED_ROWS or cost.get("errors") != 0:
        raise SystemExit("formal v3 annotations contain API or format failures")
    cost_provenance = cost.get("provenance") or {}
    expected_provenance = expected_teacher_provenance(pool_entry["sha256"], commit)
    if cost_provenance != expected_provenance:
        differing = sorted(
            key for key in set(cost_provenance) | set(expected_provenance)
            if cost_provenance.get(key) != expected_provenance.get(key)
        )
        raise SystemExit(
            f"annotation cost provenance does not match frozen teacher contract: {differing}"
        )

    annotation_path = os.path.join(config.RERANK_DIR, "sft_annotations_v3.jsonl")
    annotation_keys = set()
    annotation_rows = []
    with open(annotation_path) as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid annotation JSON at line {line_number}: {exc}") from exc
            key = (int(row["uid"]), str(row.get("slate_id", 0)))
            if key in annotation_keys:
                raise SystemExit(f"duplicate annotation key: {key}")
            annotation_keys.add(key)
            annotation_rows.append(row)
            if row.get("provenance") != cost_provenance:
                raise SystemExit(f"annotation provenance mismatch at line {line_number}")
    if len(annotation_rows) != EXPECTED_ROWS:
        raise SystemExit(
            f"formal annotation file has {len(annotation_rows)}, expected {EXPECTED_ROWS}"
        )

    pool_keys = load_pool_keys(os.path.join(config.RERANK_DIR, "sft_pool_v3.parquet"))
    if len(pool_keys) != EXPECTED_ROWS:
        raise SystemExit(f"formal pool has {len(pool_keys)} keys, expected {EXPECTED_ROWS}")
    if len(pool_keys) != len(set(pool_keys)):
        raise SystemExit("formal pool has duplicate (uid, slate_id) keys")
    pool_key_set = set(pool_keys)
    if pool_key_set != annotation_keys:
        missing = len(pool_key_set - annotation_keys)
        extra = len(annotation_keys - pool_key_set)
        raise SystemExit(
            f"pool/annotation key mismatch: missing={missing}, extra={extra}"
        )

    expected_cost = cost_summary(annotation_rows, cost_provenance)
    expected_cost.update({
        "source_pool_rows": EXPECTED_ROWS,
        "target_rows": EXPECTED_ROWS,
        "phase_manifest_sha256": phase_entry["sha256"],
        "selected_key_sha256": full_stage.get("key_sha256"),
    })
    if cost != expected_cost:
        differing = sorted(
            key for key in set(cost) | set(expected_cost)
            if cost.get(key) != expected_cost.get(key)
        )
        raise SystemExit(
            f"annotation cost summary does not match JSONL recomputation: {differing}"
        )

    if stats.get("recipe") != RECIPE or stats.get("prompt_version") != PROMPT_VERSION:
        raise SystemExit("dataset stats do not declare the frozen v3 contract")
    if (stats.get("variant") != "full" or stats.get("render") != "sid" or
            stats.get("dataset_name") != "sft"):
        raise SystemExit("dataset stats are not the canonical full SID SFT dataset")
    if (stats.get("scope") != "full" or stats.get("source_pool_rows") != EXPECTED_ROWS or
            stats.get("target_rows") != EXPECTED_ROWS or
            stats.get("annotation_file_rows") != EXPECTED_ROWS or
            stats.get("selected_annotations") != EXPECTED_ROWS):
        raise SystemExit("dataset stats do not declare the complete full-12k stage")
    if stats.get("pool_rows") != EXPECTED_ROWS or stats.get("raw_annotations") != EXPECTED_ROWS:
        raise SystemExit("dataset stats do not cover the complete pool and annotation file")
    provenance_variants = stats.get("provenance_variants") or []
    if provenance_variants != [cost_provenance]:
        raise SystemExit("dataset stats provenance differs from annotation provenance")
    hashes = stats.get("sha256") or {}
    expected_hashes = {
        "pool": pool_entry["sha256"],
        "phase_manifest": phase_entry["sha256"],
        "annotations": annotation_entry["sha256"],
        "train": _entry_by_name(manifest, "sft", "train.json")["sha256"],
        "dev": _entry_by_name(manifest, "sft", "dev.json")["sha256"],
        "dataset_info": _entry_by_name(manifest, "sft", "dataset_info.json")["sha256"],
    }
    if hashes != expected_hashes:
        raise SystemExit("dataset stats hashes do not match the closed allowlist")

    train = _load_json(os.path.join(config.SFT_V3_DATA_DIR, "train.json"))
    dev = _load_json(os.path.join(config.SFT_V3_DATA_DIR, "dev.json"))
    if not isinstance(train, list) or not isinstance(dev, list) or not train or not dev:
        raise SystemExit("canonical train.json and dev.json must be non-empty JSON lists")
    if len(train) != stats.get("train_rows") or len(dev) != stats.get("dev_rows"):
        raise SystemExit("dataset row counts differ from stats.json")
    if len(train) + len(dev) != stats.get("passed"):
        raise SystemExit("train and dev rows do not sum to the accepted count")
    if stats.get("train_dev_user_overlap") != 0:
        raise SystemExit("dataset stats report train/dev user overlap")
    info = _load_json(os.path.join(config.SFT_V3_DATA_DIR, "dataset_info.json"))
    columns = {"prompt": "instruction", "response": "output", "system": "system"}
    expected_info = {
        "sft_train": {"file_name": "train.json", "columns": columns},
        "sft_dev": {"file_name": "dev.json", "columns": columns},
    }
    if info != expected_info:
        raise SystemExit("dataset_info.json must contain exactly canonical sft_train and sft_dev")
    return expected_provenance


def bundle_descriptor(manifest, commit, teacher_provenance):
    """Canonical, path-independent description shared by both W&B artifacts."""
    files = {
        group: [
            {key: entry[key] for key in ("artifact_name", "bytes", "sha256")}
            for entry in sorted(manifest[group], key=lambda item: item["artifact_name"])
        ]
        for group in sorted(GROUPS)
    }
    return {
        "schema_version": 1,
        "recipe": RECIPE,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": teacher_provenance["prompt_sha256"],
        "teacher_provenance": teacher_provenance,
        "git_commit": commit,
        "files": files,
    }


def bundle_sha256(descriptor):
    payload = json.dumps(
        descriptor, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_metadata(manifest, commit, teacher_provenance):
    """Return per-group metadata carrying one shared bundle identity."""
    descriptor = bundle_descriptor(manifest, commit, teacher_provenance)
    digest = bundle_sha256(descriptor)
    common = {
        "bundle_schema_version": descriptor["schema_version"],
        "bundle_sha256": digest,
        "bundle_groups": sorted(GROUPS),
        "bundle_files": descriptor["files"],
        "recipe": RECIPE,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": teacher_provenance["prompt_sha256"],
        "teacher_provenance": teacher_provenance,
        "git_commit": commit,
    }
    return digest, {
        group: {**common, "artifact_group": group, "files": manifest[group]}
        for group in GROUPS
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upload",
        action="store_true",
        help="create paired rerank and sft versions; the default only prints a manifest",
    )
    args = parser.parse_args()

    commit = git_state()
    manifest = build_manifest(list(GROUPS))
    teacher_provenance = validate_semantics(manifest, commit)
    digest, metadata = artifact_metadata(manifest, commit, teacher_provenance)
    print(json.dumps({
        "bundle_sha256": digest,
        "teacher_provenance": teacher_provenance,
        "artifacts": metadata,
    }, indent=2, ensure_ascii=False))
    if not args.upload:
        print("dry run only; pass --upload after checking this exact paired bundle")
        return
    require_clean_pushed_release(config.ROOT, expected_commit=commit)

    import wandb

    run = wandb.init(
        entity=config.wandb_entity(),
        project=config.WANDB_PROJECT,
        job_type="artifact-transfer",
        name="publish-sft-v3",
    )
    uploaded = {}
    try:
        for group in GROUPS:
            artifact = wandb.Artifact(
                group,
                type="dataset",
                metadata=metadata[group],
            )
            for entry in manifest[group]:
                artifact.add_file(
                    os.path.join(config.ROOT, entry["source"]),
                    name=entry["artifact_name"],
                )
            uploaded[group] = log_immutable_artifact(run, artifact)
    finally:
        run.finish()
    print(json.dumps({
        "bundle_sha256": digest,
        "uploaded_artifacts": uploaded,
    }, indent=2))


if __name__ == "__main__":
    main()
