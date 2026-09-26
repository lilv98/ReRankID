"""Select canonical RLVR checkpoints and verify frozen executed result profiles."""

import argparse
import hashlib
import json
import os
import shlex
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
from utils.artifact_release import require_regular_file
from utils.data_io import SLATE_SEED
from utils.metrics import aggregate

from rl.contract import (BENCH_SHA256, POLICIES, REWARD_FILES, SAVE_FREQ,
                         arm_key, rlvr_contract, rooted)

EXPECTED_STEPS = tuple(range(SAVE_FREQ, 101, SAVE_FREQ))
TEST_SETTINGS = {(k, 30) for k in (0, 4, 8, 16, 32, 64, 128, 256)} | {
    (64, n) for n in (10, 20, 50, 100)
}
DEFAULT_MANIFEST = os.path.join(config.ROOT, "experiments", "final_manifest.json")

COMMON_META = {
    "arm": "sid",
    "render": "sid",
    "corrupt_sid": False,
    "max_model_len": None,
    "top_k": 10,
    "max_tokens": 2048,
    "extract_tokens": 256,
    "k_ndcg": 5,
    "no_profile": True,
    "no_eager": False,
    "benches": None,
    "protocol": "listwise-two-stage",
    "decoding": "greedy+structured",
    "n": None,
    "shuffle": 0,
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root, files):
    """Hash exact relative names and file hashes in deterministic order."""

    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: os.path.relpath(value, root)):
        relative = os.path.relpath(path, root).replace(os.sep, "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def _load_manifest(path):
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read final manifest {path}: {error}") from error
    return payload


def _format_step(template, step):
    if template == "step{step}":
        return f"step{step}"
    if template == "rl_step{step}":
        return f"rl_step{step}"
    if template == "step-{step}":
        return f"step-{step}"
    if template == "step-{step:06d}":
        return f"step-{step:06d}"
    raise ValueError(f"unsupported exact executed-profile step template {template!r}")


def _canonical_spec(policy_name, initializer, reward, root):
    contract = rlvr_contract(policy_name, initializer, reward)
    benchmark = rooted(root, "data/benchmark/bench.parquet")
    return {
        "profile": "canonical",
        "arm": contract["arm"],
        "policy": policy_name,
        "variant": contract["variant"],
        "initializer": initializer,
        "reward": reward,
        "results_rel": contract["results_rel"],
        "dev_subdir": "dev_select",
        "test_subdir": "",
        "dev_tag": contract["dev_tag"],
        "test_tag": contract["test_tag"],
        "model_meta": contract["model_rel"],
        "benchmark_meta": benchmark,
        "checkpoint_label_template": "step{step}",
        "adapter_root_meta": contract["export_rel"],
        "adapter_step_template": "step-{step:06d}",
        "result_artifact_name": contract["results_artifact"],
        "manifest_arm": None,
    }


def _executed_spec(arm, manifest_path):
    manifest = _load_manifest(manifest_path)
    arms = manifest.get("recipes", {}).get("rlvr", {}).get("arms", {})
    if arm not in arms:
        raise ValueError(f"unknown executed RLVR arm {arm!r}")
    manifest_arm = arms[arm]
    if manifest_arm.get("status") != "complete":
        raise ValueError(f"RLVR arm {arm} is pending and has no executed profile")
    profile = manifest_arm.get("executed_profile")
    if not isinstance(profile, dict):
        raise ValueError(f"completed RLVR arm {arm} has no executed_profile")
    required = {
        "schema_version", "policy", "initializer", "reward", "results_rel",
        "dev_subdir", "test_subdir", "dev_tag", "test_tag", "model_meta",
        "benchmark_meta", "checkpoint_label_template", "adapter_root_meta",
        "adapter_step_template", "result_artifact_name", "tree_sha256",
    }
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"executed profile {arm} lacks fields {missing}")
    if profile["schema_version"] != 1:
        raise ValueError(f"executed profile {arm} has unknown schema")
    policy = profile["policy"]
    initializer = profile["initializer"]
    reward = profile["reward"]
    expected_arm = arm_key(policy, initializer, reward)
    if arm != expected_arm:
        raise ValueError(f"executed profile key {arm} != resolved arm {expected_arm}")
    if (
        manifest_arm.get("initializer") != initializer
        or manifest_arm.get("reward") != reward
    ):
        raise ValueError(f"executed profile {arm} disagrees with its manifest arm")
    contract = rlvr_contract(policy, initializer, reward)
    if profile["result_artifact_name"] != contract["results_artifact"]:
        raise ValueError(
            f"executed profile {arm} artifact name differs from canonical contract"
        )
    for field in ("tree_sha256",):
        if len(str(profile.get(field, ""))) != 64:
            raise ValueError(f"executed profile {arm} has invalid {field}")
    for field in ("dev_evidence_sha256", "test_evidence_sha256"):
        if len(str(manifest_arm.get(field, ""))) != 64:
            raise ValueError(f"executed profile {arm} has invalid {field}")
    _format_step(profile["checkpoint_label_template"], EXPECTED_STEPS[0])
    _format_step(profile["adapter_step_template"], EXPECTED_STEPS[0])
    return {
        "profile": "executed",
        "arm": arm,
        "policy": policy,
        "variant": contract["variant"],
        "initializer": initializer,
        "reward": reward,
        **profile,
        "manifest_arm": manifest_arm,
    }


def resolve_spec(
    policy_name=None,
    initializer="sft",
    reward="ndcg5",
    root=config.ROOT,
    profile="canonical",
    arm=None,
    manifest_path=DEFAULT_MANIFEST,
):
    if profile == "canonical":
        if arm is not None:
            raise ValueError("--arm is valid only with --profile executed")
        if policy_name is None:
            raise ValueError("canonical profile requires --policy")
        return _canonical_spec(policy_name, initializer, reward, root)
    if profile == "executed":
        if not arm:
            raise ValueError("executed profile requires explicit --arm")
        return _executed_spec(arm, manifest_path)
    raise ValueError(f"unknown profile {profile!r}")


def _load_benchmark_contract(root):
    """Load the frozen per-user candidate-grade pools used to rescore evidence."""

    benchmark = rooted(root, "data/benchmark/bench.parquet")
    require_regular_file(benchmark, "frozen benchmark")
    actual = sha256(benchmark)
    if actual != BENCH_SHA256:
        raise ValueError(f"benchmark SHA256 {actual} != frozen {BENCH_SHA256}")
    frame = pd.read_parquet(
        benchmark, columns=["uid", "slice", "pool_grades"]
    )
    expected = {"dev": {}, "test": {}}
    for record in frame.itertuples(index=False):
        split = record.slice
        if split not in expected:
            raise ValueError(f"benchmark has unexpected split {split!r}")
        uid = int(record.uid)
        if uid in expected[split]:
            raise ValueError(f"benchmark {split} contains duplicate uid {uid}")
        raw_grades = list(record.pool_grades)
        if (
            len(raw_grades) != 100
            or any(
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                for value in raw_grades
            )
        ):
            raise ValueError(
                f"benchmark {split} uid {uid} has an invalid grade pool"
            )
        grades = [int(value) for value in raw_grades]
        if (
            sum(value > 0 for value in grades) != 1
            or any(value not in (0, 2, 3) for value in grades)
        ):
            raise ValueError(
                f"benchmark {split} uid {uid} has an invalid grade pool"
            )
        expected[split][uid] = grades
    for split in ("dev", "test"):
        if len(expected[split]) != 400:
            raise ValueError(f"benchmark {split} must contain 400 unique users")
    if set(expected["dev"]) & set(expected["test"]):
        raise ValueError("benchmark dev and test users overlap")
    return benchmark, expected


def _expected_grades(
    benchmark_rows, split, k, n_cand, shuffle, cache
):
    """Reconstruct display-order grades for one exact evaluation cell."""

    if (
        split not in benchmark_rows
        or type(k) is not int
        or k < 0
        or type(n_cand) is not int
        or n_cand < 1
        or type(shuffle) is not int
        or shuffle < 0
    ):
        raise ValueError(
            f"invalid benchmark cell split={split!r}, k={k!r}, "
            f"N={n_cand!r}, shuffle={shuffle!r}"
        )
    key = (split, k, n_cand, shuffle)
    if key not in cache:
        rebuilt = {}
        for uid, pool_grades in benchmark_rows[split].items():
            if n_cand > len(pool_grades):
                raise ValueError(
                    f"benchmark uid {uid} cannot construct N={n_cand}"
                )
            permutation = np.random.default_rng(
                [SLATE_SEED, uid, n_cand, shuffle]
            ).permutation(n_cand)
            prefix = pool_grades[:n_cand]
            rebuilt[uid] = [prefix[index] for index in permutation]
        cache[key] = rebuilt
    return cache[key]


def _load_result(path):
    require_regular_file(path, "RLVR result")
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read result {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"invalid result payload: {path}")
    if not isinstance(payload.get("meta"), dict):
        raise ValueError(f"result has no metadata: {path}")
    if not isinstance(payload.get("metrics"), dict):
        raise ValueError(f"result has no metrics: {path}")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"result has no per-user results: {path}")
    return payload


def _validate_result_rows(
    path,
    payload,
    benchmark_rows,
    split,
    k,
    n_cand,
    shuffle,
    grades_cache,
):
    """Validate every row and recompute all deterministic aggregate metrics."""

    results = payload["results"]
    if len(results) != 400:
        raise ValueError(f"{path} must contain exactly 400 result rows")
    uids = []
    for row in results:
        if not isinstance(row, dict) or type(row.get("uid")) is not int:
            raise ValueError(f"{path} has a result row without an integer uid")
        uids.append(row["uid"])
    if len(set(uids)) != len(uids):
        raise ValueError(f"{path} contains duplicate user ids")
    expected = _expected_grades(
        benchmark_rows, split, k, n_cand, shuffle, grades_cache
    )
    if set(uids) != set(expected):
        raise ValueError(f"{path} user ids differ from frozen benchmark {split}")
    diag_keys = {"has_array", "arr_len", "n_oob", "n_dup"}
    for row in results:
        uid = row["uid"]
        grades = row.get("grades")
        if (
            not isinstance(grades, list)
            or any(type(value) is not int for value in grades)
            or grades != expected[uid]
        ):
            raise ValueError(
                f"{path} uid {uid} grades differ from the frozen benchmark cell"
            )

        n_listed = row.get("n_listed")
        if type(n_listed) is not int or not 0 <= n_listed <= 10:
            raise ValueError(f"{path} uid {uid} has invalid n_listed")
        order = row.get("order")
        if order is None:
            if n_listed != 0:
                raise ValueError(
                    f"{path} uid {uid} has null order but nonzero n_listed"
                )
        elif (
            not isinstance(order, list)
            or any(type(index) is not int for index in order)
            or len(order) != n_listed
            or len(set(order)) != len(order)
            or any(not 0 <= index < n_cand for index in order)
        ):
            raise ValueError(f"{path} uid {uid} has invalid order/n_listed")

        diag = row.get("diag")
        if not isinstance(diag, dict) or set(diag) != diag_keys:
            raise ValueError(f"{path} uid {uid} has invalid diag schema")
        if type(diag["has_array"]) is not bool or any(
            type(diag[key]) is not int
            for key in ("arr_len", "n_oob", "n_dup")
        ):
            raise ValueError(f"{path} uid {uid} has invalid diag types")
        if (
            min(diag["arr_len"], diag["n_oob"], diag["n_dup"]) < 0
            or diag["n_oob"] + diag["n_dup"] + n_listed
            != diag["arr_len"]
            or (not diag["has_array"] and diag["arr_len"] != 0)
            or (not diag["has_array"] and order is not None)
            or (order is not None and not diag["has_array"])
        ):
            raise ValueError(f"{path} uid {uid} has inconsistent diag values")

    recalculated = aggregate(
        results, k_ndcg=5, n_out=10, with_random=False
    )
    stored = payload["metrics"]
    for key, expected_value in recalculated.items():
        if key not in stored or stored[key] != expected_value:
            raise ValueError(
                f"{path} stored metric {key!r}={stored.get(key)!r} "
                f"!= recomputed {expected_value!r}"
            )
    return recalculated


def _validate_common(path, payload, spec, benchmark):
    meta = payload["meta"]
    for key, value in COMMON_META.items():
        if meta.get(key) != value:
            raise ValueError(
                f"{path} has meta.{key}={meta.get(key)!r}, expected {value!r}"
            )
    if meta.get("model") != spec["model_meta"]:
        raise ValueError(
            f"{path} has meta.model={meta.get('model')!r}, "
            f"expected {spec['model_meta']!r}"
        )
    if spec["profile"] == "executed":
        if meta.get("bench") != spec["benchmark_meta"]:
            raise ValueError(f"{path} has the wrong executed benchmark metadata")
    elif os.path.realpath(str(meta.get("bench", ""))) != os.path.realpath(benchmark):
        raise ValueError(f"{path} does not reference the frozen benchmark")


def _list_tree(result_root):
    if os.path.islink(result_root) or not os.path.isdir(result_root):
        raise ValueError(
            f"RLVR result root is not a regular directory: {result_root}"
        )
    files = []
    for current, dirs, names in os.walk(result_root):
        dirs.sort()
        linked_dirs = [
            os.path.join(current, name)
            for name in dirs
            if os.path.islink(os.path.join(current, name))
        ]
        if linked_dirs:
            raise ValueError(
                f"unexpected symlink directory in RLVR results: {linked_dirs[0]}"
            )
        for name in sorted(names):
            path = os.path.join(current, name)
            if not name.endswith(".json"):
                raise ValueError(f"unexpected non-JSON RLVR result: {path}")
            require_regular_file(path, "RLVR result")
            files.append(path)
    return files


def _expected_lora(spec):
    return ",".join(
        f"{_format_step(spec['checkpoint_label_template'], step)}="
        f"{spec['adapter_root_meta']}/"
        f"{_format_step(spec['adapter_step_template'], step)}"
        for step in EXPECTED_STEPS
    )


def _validate_dev(
    spec,
    root,
    benchmark,
    benchmark_rows,
    grades_cache,
):
    result_root = rooted(root, spec["results_rel"])
    dev_dir = os.path.join(result_root, spec["dev_subdir"])
    if os.path.islink(result_root) or not os.path.isdir(result_root):
        raise ValueError(
            f"RLVR result root is not a regular directory: {result_root}"
        )
    if os.path.islink(dev_dir) or not os.path.isdir(dev_dir):
        raise ValueError(
            f"official dev result is not a regular non-symlink directory: "
            f"{dev_dir}"
        )
    names = sorted(os.listdir(dev_dir))
    if len(names) != 6 or any(not name.endswith(".json") for name in names):
        raise ValueError(f"{dev_dir} must contain exactly six dev JSON files")
    expected_lora = _expected_lora(spec)
    label_to_step = {
        _format_step(spec["checkpoint_label_template"], step): step
        for step in EXPECTED_STEPS
    }
    by_step = {}
    base_scores = []
    files = []
    for filename in names:
        path = os.path.join(dev_dir, filename)
        payload = _load_result(path)
        _validate_common(path, payload, spec, benchmark)
        meta = payload["meta"]
        expected = {
            "tag": spec["dev_tag"],
            "slice": "dev",
            "k": 64,
            "n_cand": 30,
            "skip_base": False,
            "lora_dirs": expected_lora,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(
                    f"{path} has meta.{key}={meta.get(key)!r}, expected {value!r}"
                )
        recalculated = _validate_result_rows(
            path,
            payload,
            benchmark_rows,
            "dev",
            64,
            30,
            0,
            grades_cache,
        )
        checkpoint = meta.get("checkpoint")
        score = recalculated["ndcg@5"]
        if checkpoint == "base":
            base_scores.append(float(score))
        elif checkpoint in label_to_step:
            step = label_to_step[checkpoint]
            if step in by_step:
                raise ValueError(f"duplicate official dev result for step {step}")
            by_step[step] = {
                "path": path,
                "ndcg@5": float(score),
            }
        else:
            raise ValueError(f"unexpected exact dev checkpoint label {checkpoint!r}")
        files.append(path)
    if set(by_step) != set(EXPECTED_STEPS):
        raise ValueError("dev tree does not contain the five exact checkpoint steps")
    if len(base_scores) != 1:
        raise ValueError(f"expected one initializer result, found {len(base_scores)}")
    return result_root, files, by_step, base_scores[0]


def choose_step(scores):
    if not scores:
        raise ValueError("no checkpoint scores to select")
    return min(scores, key=lambda step: (-scores[step]["ndcg@5"], step))


def validate_result_tree(
    policy_name=None,
    root=config.ROOT,
    initializer="sft",
    reward="ndcg5",
    profile="canonical",
    arm=None,
    manifest_path=DEFAULT_MANIFEST,
    require_test=True,
):
    """Validate an exact canonical or manifest-pinned executed result profile."""

    spec = resolve_spec(
        policy_name, initializer, reward, root, profile, arm, manifest_path
    )
    benchmark, benchmark_rows = _load_benchmark_contract(root)
    grades_cache = {}
    result_root, dev_files, scores, base_score = _validate_dev(
        spec,
        root,
        benchmark,
        benchmark_rows,
        grades_cache,
    )
    best_step = choose_step(scores)
    if profile == "executed":
        manifest_step = int(spec["manifest_arm"].get("selected_step", -1))
        if best_step != manifest_step:
            raise ValueError(
                f"executed dev argmax {best_step} != manifest selected step "
                f"{manifest_step}"
            )
    if not require_test:
        return {
            "spec": spec,
            "result_root": result_root,
            "files": dev_files,
            "scores": scores,
            "base_score": base_score,
            "best_step": best_step,
        }

    all_files = _list_tree(result_root)
    if len(all_files) != 18:
        raise ValueError(f"{spec['arm']} must have exactly 18 JSONs, found {len(all_files)}")
    dev_set = set(dev_files)
    test_files = [path for path in all_files if path not in dev_set]
    expected_test_dir = os.path.normpath(os.path.join(result_root, spec["test_subdir"]))
    test_settings = set()
    anchor_path = None
    expected_best = (
        f"best={spec['adapter_root_meta']}/"
        f"{_format_step(spec['adapter_step_template'], best_step)}"
    )
    for path in test_files:
        if os.path.normpath(os.path.dirname(path)) != expected_test_dir:
            raise ValueError(f"{path} is outside the exact test result directory")
        payload = _load_result(path)
        _validate_common(path, payload, spec, benchmark)
        meta = payload["meta"]
        expected = {
            "tag": spec["test_tag"],
            "slice": "test",
            "checkpoint": "best",
            "skip_base": True,
            "lora_dirs": expected_best,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(
                    f"{path} has meta.{key}={meta.get(key)!r}, expected {value!r}"
                )
        k = meta.get("k")
        n_cand = meta.get("n_cand")
        if type(k) is not int or type(n_cand) is not int:
            raise ValueError(f"{path} has non-integer k/N metadata")
        setting = (k, n_cand)
        _validate_result_rows(
            path,
            payload,
            benchmark_rows,
            "test",
            k,
            n_cand,
            0,
            grades_cache,
        )
        if setting in test_settings:
            raise ValueError(f"duplicate selected test setting {setting}")
        test_settings.add(setting)
        if setting == (64, 30):
            anchor_path = path
    if test_settings != TEST_SETTINGS:
        raise ValueError(
            f"test settings differ: missing={sorted(TEST_SETTINGS-test_settings)}, "
            f"extra={sorted(test_settings-TEST_SETTINGS)}"
        )
    if anchor_path is None:
        raise ValueError("test tree has no k64/N30 anchor")

    if profile == "executed":
        manifest_arm = spec["manifest_arm"]
        selected_dev = scores[best_step]["path"]
        if sha256(selected_dev) != manifest_arm["dev_evidence_sha256"]:
            raise ValueError("selected dev result differs from manifest evidence SHA256")
        if sha256(anchor_path) != manifest_arm["test_evidence_sha256"]:
            raise ValueError("test anchor differs from manifest evidence SHA256")
        if tree_sha256(result_root, all_files) != spec["tree_sha256"]:
            raise ValueError("executed result tree differs from manifest tree SHA256")

    return {
        "spec": spec,
        "result_root": result_root,
        "files": all_files,
        "scores": scores,
        "base_score": base_score,
        "best_step": best_step,
        "anchor_path": anchor_path,
    }


def collect_dev_scores(
    policy_name,
    root=config.ROOT,
    initializer="sft",
    reward="ndcg5",
    profile="canonical",
    arm=None,
    manifest_path=DEFAULT_MANIFEST,
):
    report = validate_result_tree(
        policy_name,
        root,
        initializer,
        reward,
        profile,
        arm,
        manifest_path,
        require_test=profile == "executed",
    )
    return report["scores"], report["base_score"]


def selection(
    policy_name=None,
    root=config.ROOT,
    initializer="sft",
    reward="ndcg5",
    profile="canonical",
    arm=None,
    manifest_path=DEFAULT_MANIFEST,
    require_adapter=None,
):
    report = validate_result_tree(
        policy_name,
        root,
        initializer,
        reward,
        profile,
        arm,
        manifest_path,
        require_test=profile == "executed",
    )
    spec = report["spec"]
    best_step = report["best_step"]
    best_meta = (
        f"{spec['adapter_root_meta']}/"
        f"{_format_step(spec['adapter_step_template'], best_step)}"
    )
    if require_adapter is None:
        require_adapter = profile == "canonical"
    if require_adapter:
        best_path = best_meta if os.path.isabs(best_meta) else rooted(root, best_meta)
        if not os.path.isdir(best_path):
            raise ValueError(f"selected compact adapter is missing: {best_path}")
    scores = report["scores"]
    return {
        "profile": profile,
        "arm": spec["arm"],
        "policy": spec["policy"],
        "variant": spec["variant"],
        "initializer": spec["initializer"],
        "reward": spec["reward"],
        "best_step": best_step,
        "best_ndcg@5": scores[best_step]["ndcg@5"],
        "init_ndcg@5": report["base_score"],
        "delta_vs_init": scores[best_step]["ndcg@5"] - report["base_score"],
        "best": best_meta,
        "model": spec["model_meta"],
        "out": spec["results_rel"],
        "dev_tag": spec["dev_tag"],
        "test_tag": spec["test_tag"],
        "result_artifact_name": spec["result_artifact_name"],
        "scores": {str(step): scores[step]["ndcg@5"] for step in EXPECTED_STEPS},
    }


def _parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("canonical", "executed"), default="canonical")
    parser.add_argument("--arm")
    parser.add_argument("--policy", choices=sorted(POLICIES))
    parser.add_argument("--init", choices=("sft",))
    parser.add_argument("--reward", choices=sorted(REWARD_FILES))
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--root", default=config.ROOT)
    parser.add_argument("--no-require-adapter", action="store_true")
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    if args.profile == "canonical":
        if not args.policy:
            parser.error("--profile canonical requires --policy")
        if args.arm:
            parser.error("--arm is valid only with --profile executed")
        args.init = args.init or "sft"
        args.reward = args.reward or "ndcg5"
    else:
        if not args.arm:
            parser.error("--profile executed requires --arm")
        if args.policy or args.init or args.reward:
            parser.error("executed profile accepts only --arm, not policy/init/reward")
        args.init = "sft"
        args.reward = "ndcg5"
    return args


def main():
    args = _parse_cli()
    try:
        report = selection(
            args.policy,
            root=args.root,
            initializer=args.init,
            reward=args.reward,
            profile=args.profile,
            arm=args.arm,
            manifest_path=args.manifest,
            require_adapter=(
                False if args.profile == "executed" or args.no_require_adapter else None
            ),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.shell:
        values = {
            "PROFILE": report["profile"],
            "ARM": report["arm"],
            "POLICY": report["policy"],
            "INITIALIZER": report["initializer"],
            "REWARD": report["reward"],
            "VARIANT": report["variant"],
            "BEST_STEP": report["best_step"],
            "BEST": report["best"],
            "MODEL": report["model"],
            "OUT": report["out"],
            "DEV_TAG": report["dev_tag"],
            "TEST_TAG": report["test_tag"],
            "RESULT_ARTIFACT_NAME": report["result_artifact_name"],
        }
        for key, value in values.items():
            print(f"export {key}={shlex.quote(str(value))}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
