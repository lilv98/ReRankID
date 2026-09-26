"""Build label-free OPD parquets from the frozen 6k SID prompt set.

The prompt bytes and ordering are inherited from the already audited RLVR
parquets.  This stage removes the positive labels and all label-derived
metadata so the OPD runtime contains only the common SID prompt plus harmless
row identifiers.  The frozen teacher receives exactly the same prompt as the
student through VERL's native single-teacher path.
"""

import hashlib
import json
import os
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from opd.contract import (
    DATA_REL,
    DATA_RECIPE,
    DATA_SCHEMA,
    SOURCE_TRAIN_PARQUET_SHA256,
    SOURCE_VAL_PARQUET_SHA256,
    TRAIN_BATCH_SIZE,
    TRAIN_KEY_SHA256,
    TRAIN_ROWS,
    VAL_K,
    VAL_N_CAND,
    VAL_ROWS,
)
from utils import config


SOURCE_DIR = os.path.join(config.ROOT, "data", "rlvr_v1_6k")
OUTPUT_DIR = os.path.join(config.ROOT, DATA_REL)
KEEP_EXTRA_INFO = (
    "uid",
    "slate_id",
    "split",
    "k_hist",
    "n_cand",
    "n_hist_kept",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [
        {"role": str(dict(message)["role"]), "content": str(dict(message)["content"])}
        for message in value
    ]


def prompt_digest(prompts):
    digest = hashlib.sha256()
    for prompt in prompts:
        payload = json.dumps(
            normalize_prompt(prompt),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def build_frame(source):
    rows = []
    for index, row in source.iterrows():
        old_extra = dict(row["extra_info"])
        extra = {"index": int(index), "source_index": int(old_extra["index"])}
        for key in KEEP_EXTRA_INFO:
            value = old_extra.get(key)
            if hasattr(value, "item"):
                value = value.item()
            extra[key] = value
        rows.append(
            {
                "data_source": "rerankid_opd_sid",
                "prompt": normalize_prompt(row["prompt"]),
                "ability": "rerank",
                # Required by VERL's reward manager.  It is deliberately empty,
                # and zero_reward.py ignores it.
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": extra,
            }
        )
    return pd.DataFrame(rows)


def atomic_parquet(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=".opd.", suffix=".parquet"
    )
    os.close(fd)
    try:
        frame.to_parquet(
            temporary,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=".opd.", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def distribution(frame, key):
    return {
        str(value): int(count)
        for value, count in sorted(
            Counter(int(dict(item)[key]) for item in frame["extra_info"]).items()
        )
    }


def main():
    source_paths = {
        "train": os.path.join(SOURCE_DIR, "train.parquet"),
        "val": os.path.join(SOURCE_DIR, "val.parquet"),
    }
    expected_source_hashes = {
        "train": SOURCE_TRAIN_PARQUET_SHA256,
        "val": SOURCE_VAL_PARQUET_SHA256,
    }
    source_frames = {}
    for split, path in source_paths.items():
        actual = sha256(path)
        if actual != expected_source_hashes[split]:
            raise ValueError(
                f"{split} source hash {actual} != {expected_source_hashes[split]}"
            )
        source_frames[split] = pd.read_parquet(path)

    if len(source_frames["train"]) != TRAIN_ROWS:
        raise ValueError("unexpected source train row count")
    if len(source_frames["val"]) != VAL_ROWS:
        raise ValueError("unexpected source val row count")

    output_frames = {
        split: build_frame(frame) for split, frame in source_frames.items()
    }
    output_paths = {
        split: os.path.join(OUTPUT_DIR, f"{split}.parquet")
        for split in ("train", "val")
    }
    for split in ("train", "val"):
        if prompt_digest(source_frames[split]["prompt"]) != prompt_digest(
            output_frames[split]["prompt"]
        ):
            raise AssertionError(f"{split} prompt changed while removing labels")
        atomic_parquet(output_frames[split], output_paths[split])

    stats = {
        "schema_version": DATA_SCHEMA,
        "recipe": DATA_RECIPE,
        "train_rows": TRAIN_ROWS,
        "val_rows": VAL_ROWS,
        "train_key_sha256": TRAIN_KEY_SHA256,
        "teacher_query_budget": TRAIN_ROWS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "optimizer_steps": TRAIN_ROWS // TRAIN_BATCH_SIZE,
        "train_distribution": {
            "k": distribution(output_frames["train"], "k_hist"),
            "n_cand": distribution(output_frames["train"], "n_cand"),
        },
        "val_protocol": {
            "slice": "dev",
            "k": VAL_K,
            "n_cand": VAL_N_CAND,
            "shuffle": 0,
            "no_profile": True,
        },
        "label_fields": {
            "reward_ground_truth": "empty",
            "positive_grade": "removed",
            "teacher_hint": "absent",
        },
        "source": {
            "train_sha256": expected_source_hashes["train"],
            "val_sha256": expected_source_hashes["val"],
        },
        "prompt_sha256": {
            split: prompt_digest(output_frames[split]["prompt"])
            for split in ("train", "val")
        },
        "files": {
            split + "_sha256": sha256(path)
            for split, path in output_paths.items()
        },
    }
    stats_path = os.path.join(OUTPUT_DIR, "stats.json")
    atomic_json(stats, stats_path)
    print(json.dumps({**stats, "stats_sha256": sha256(stats_path)}, indent=2))


if __name__ == "__main__":
    main()
