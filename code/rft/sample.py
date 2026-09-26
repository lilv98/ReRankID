"""Sample resumable best-of-8 RFT rollouts from the frozen SFT-v3 student.

The formal run uses the deterministic 6k subset of one shared 12k
post-training prompt pool. Each prompt receives eight independent single-stage
rollouts.  Raw generations are retained locally for audit; filtering and
LLaMA-Factory assembly happen offline in ``rft/build_dataset.py``.

This is deliberately an ordinary rejection-fine-tuning baseline.  The model
never sees labels while generating, so there is no answer-leakage regex.  The
only later quality gate is valid completed output, gold at rank one, and the
existing identity-collapse diagnostic.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config
config.setup_env(hf_offline=True)

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter

from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams

from rft.base_provenance import verify_policy_base
from rft.contract import (DEFAULT_POLICY, MAX_TOKENS, N_SAMPLES, POLICIES,
                          RFT_RECIPE, RFT_TARGET_ROWS, SEED, TEMPERATURE, TOP_K,
                          TOP_P, is_smoke_path, policy_contract)
from rft.raw_integrity import (extract_cot, prompt_seed, seal_record,
                               validate_rollout_record)
from rft.stages import (row_key, select_stage_rows, validate_phase_manifest)
from utils.data_io import load_rows
from utils.metrics import is_identity_collapse, ndcg
from utils.sid_engine import (answer_schema, load_llm, parse_stage1,
                              stage1_prompt, vllm_version)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=config.ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=config.ROOT, text=True,
    ).strip()
    return commit, bool(dirty)


def atomic_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_existing(path):
    """Load complete records and repair only a crash-truncated final line."""
    if not os.path.exists(path):
        return []
    records = []
    last_good = 0
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                records.append(json.loads(line))
                last_good = handle.tell()
            except (json.JSONDecodeError, UnicodeDecodeError):
                if handle.tell() != size:
                    raise ValueError(f"corrupt non-final JSONL line at byte {line_start}")
                print(f"repairing crash-truncated final JSONL line at byte {line_start}")
                break
    if last_good != size:
        with open(path, "r+b") as handle:
            handle.truncate(last_good)
            handle.flush()
            os.fsync(handle.fileno())
    return records


def validate_scope(args, git_dirty, policy):
    canonical_inputs = (
        os.path.realpath(args.pool) == os.path.realpath(config.POSTTRAIN_V1_POOL)
        and os.path.realpath(args.manifest) == os.path.realpath(config.POSTTRAIN_V1_PHASE_MANIFEST)
    )
    canonical_output = os.path.realpath(args.out) == os.path.realpath(
        getattr(config, policy["samples_attr"])
    )
    formal = canonical_inputs and canonical_output and args.n is None
    if formal:
        frozen = {
            "n_samples": N_SAMPLES, "temperature": TEMPERATURE, "top_p": TOP_P,
            "max_tokens": MAX_TOKENS, "seed": SEED,
        }
        actual = {
            "n_samples": args.n_samples, "temperature": args.temperature,
            "top_p": args.top_p, "max_tokens": args.max_tokens, "seed": args.seed,
        }
        if actual != frozen:
            raise SystemExit(f"formal RFT sampling parameters are frozen: {frozen}")
        if args.target_rows != RFT_TARGET_ROWS:
            raise SystemExit(
                f"formal --target-rows must be exactly {RFT_TARGET_ROWS}"
            )
        if git_dirty:
            raise SystemExit("formal RFT sampling requires a clean git worktree")
        return "formal"
    if not is_smoke_path(args.out):
        raise SystemExit("nonformal RFT output must be under /tmp with 'smoke' in the path")
    if args.n is None or args.n < 1:
        raise SystemExit("smoke sampling requires a positive --n")
    return "smoke"


def provenance(
    args,
    pool_sha,
    manifest_sha,
    adapter_path,
    commit,
    dirty,
    scope,
    policy,
    base_identity,
):
    return {
        "schema_version": 2,
        "recipe": RFT_RECIPE,
        "scope": scope,
        "git_commit": commit,
        "git_dirty": dirty,
        "pool_sha256": pool_sha,
        "phase_manifest_sha256": manifest_sha,
        "policy": args.policy,
        "model": policy["model"],
        "base_model": base_identity,
        "sft_adapter_artifact": policy["adapter_artifact"],
        "sft_adapter_path": os.path.relpath(adapter_path, config.ROOT),
        "sft_adapter_model_sha256": policy["adapter_sha256"],
        "vllm_version": vllm_version(),
        "sampling": {
            "n": args.n_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "top_k": TOP_K,
            "stage": "single-stage",
        },
    }


def provenance_digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def record_for(row, output, prov, prov_sha, seed=SEED):
    n_cand = int(row["n_cand"])
    grades = [int(value) for value in row["cand_grades"]]
    positive = [index for index, grade in enumerate(grades) if grade > 0]
    if len(positive) != 1:
        raise ValueError(f"expected one positive for {row_key(row)}")
    rollouts = []
    for index, sample in enumerate(output.outputs):
        text = sample.text or ""
        order, n_listed, diag = parse_stage1(text, n_cand, TOP_K)
        score = ndcg(order, grades, TOP_K) if order is not None else None
        rank = order.index(positive[0]) + 1 if order is not None and positive[0] in order else None
        cot = extract_cot(text)
        rollouts.append({
            "sample_index": index,
            "text": text,
            "cot_text": cot,
            "order": order,
            "n_listed": n_listed,
            "diag": diag,
            "ndcg10": round(float(score), 8) if score is not None else None,
            "gold_rank": rank,
            "valid": order is not None and n_listed == TOP_K,
            "rank1": rank == 1,
            "identity": bool(order is not None and is_identity_collapse(order, TOP_K)),
            "finish_reason": getattr(sample, "finish_reason", None),
            "stop_reason": getattr(sample, "stop_reason", None),
            "output_tokens": len(sample.token_ids),
        })
    return seal_record({
        "uid": int(row["uid"]),
        "slate_id": str(row["slate_id"]),
        "k": int(row["k"]),
        "n_cand": n_cand,
        "pos_grade": int(row["pos_grade"]),
        "n_hard_neg": int(row["n_hard_neg"]),
        "grades": grades,
        "gold_idx": positive,
        "prompt_tokens": len(output.prompt_token_ids),
        "prompt_seed": prompt_seed(row["uid"], row["slate_id"], seed),
        "provenance_sha256": prov_sha,
        "provenance": prov,
        "rollouts": rollouts,
    })


def summarize(records, target_rows, prov_sha):
    prompts = len(records)
    rollouts = [sample for record in records for sample in record["rollouts"]]
    base_identity = (
        records[0].get("provenance", {}).get("base_model")
        if records else None
    )
    covered = [record for record in records if any(
        sample["valid"] and sample["rank1"] and not sample["identity"]
        and sample.get("cot_text") for sample in record["rollouts"]
    )]

    def dist(field, source=records):
        return dict(sorted(Counter(str(row[field]) for row in source).items(),
                           key=lambda item: int(item[0])))

    def coverage(field):
        totals = Counter(str(row[field]) for row in records)
        kept = Counter(str(row[field]) for row in covered)
        return {key: {"prompts": totals[key], "covered": kept[key],
                      "coverage": round(kept[key] / totals[key], 4)}
                for key in sorted(totals, key=int)}

    def combo_coverage():
        totals = Counter((int(row["k"]), int(row["n_cand"])) for row in records)
        kept = Counter((int(row["k"]), int(row["n_cand"])) for row in covered)
        return {
            f"k{k}_N{n}": {"prompts": totals[(k, n)], "covered": kept[(k, n)],
                            "coverage": round(kept[(k, n)] / totals[(k, n)], 4)}
            for k, n in sorted(totals)
        }

    parsed = [sample for sample in rollouts if sample["valid"]]
    rank1_hits = [sum(
        sample["valid"] and sample["rank1"] and not sample["identity"]
        and bool(sample.get("cot_text")) for sample in record["rollouts"]
    ) for record in records]
    random_rollout_rate = (
        sum(1 / int(record["n_cand"]) for record in records) / prompts if prompts else 0.0
    )
    random_prompt_coverage = (
        sum(1 - (1 - 1 / int(record["n_cand"])) ** len(record["rollouts"])
            for record in records) / prompts if prompts else 0.0
    )
    observed_rank1_rate = (
        sum(sample["rank1"] for sample in parsed) / len(parsed) if parsed else 0.0
    )
    observed_coverage = len(covered) / prompts if prompts else 0.0
    return {
        "schema_version": 2,
        "recipe": RFT_RECIPE,
        "base_model": base_identity,
        "target_rows": target_rows,
        "completed_prompts": prompts,
        "complete": prompts == target_rows,
        "provenance_sha256": prov_sha,
        "rollouts": len(rollouts),
        "valid_rollouts": len(parsed),
        "rank1_rollouts": sum(sample["rank1"] for sample in parsed),
        "identity_rollouts": sum(sample["identity"] for sample in parsed),
        "valid_rate": round(len(parsed) / len(rollouts), 4) if rollouts else 0.0,
        "rank1_rate_among_valid": round(observed_rank1_rate, 4),
        "covered_prompts": len(covered),
        "prompt_coverage": round(observed_coverage, 4),
        "rank1_hits_per_prompt": dict(sorted(Counter(str(value) for value in rank1_hits).items(),
                                              key=lambda item: int(item[0]))),
        "uniform_random_reference": {
            "rank1_rate": round(random_rollout_rate, 4),
            "best_of_k_prompt_coverage": round(random_prompt_coverage, 4),
            "observed_to_random_rank1_ratio": round(
                observed_rank1_rate / random_rollout_rate, 3
            ) if random_rollout_rate else None,
            "observed_to_random_coverage_ratio": round(
                observed_coverage / random_prompt_coverage, 3
            ) if random_prompt_coverage else None,
        },
        "prompt_distribution": {
            "k": dist("k"), "n_cand": dist("n_cand"), "pos_grade": dist("pos_grade")
        },
        "coverage_by": {
            "k": coverage("k"), "n_cand": coverage("n_cand"),
            "pos_grade": coverage("pos_grade"), "k_x_n": combo_coverage(),
        },
        "updated_unix": int(time.time()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=sorted(POLICIES), default=DEFAULT_POLICY)
    parser.add_argument("--pool", default=config.POSTTRAIN_V1_POOL)
    parser.add_argument("--manifest", default=config.POSTTRAIN_V1_PHASE_MANIFEST)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--target-rows", type=int, default=RFT_TARGET_ROWS)
    parser.add_argument("--n", type=int, default=None, help="first N selected prompts, smoke only")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-prompts", type=int, default=128)
    parser.add_argument("--out", default=None)
    parser.add_argument("--stats-out", default=None)
    parser.add_argument("--no-eager", action="store_true")
    args = parser.parse_args()
    policy = policy_contract(args.policy)
    args.adapter = args.adapter or os.path.join(config.ROOT, policy["adapter_rel"])
    args.out = args.out or getattr(config, policy["samples_attr"])

    if args.batch_prompts < 1 or args.n_samples < 1:
        raise SystemExit("--batch-prompts and --n-samples must be positive")
    if args.target_rows != RFT_TARGET_ROWS:
        raise SystemExit(
            f"RFT rollout supports only --target-rows {RFT_TARGET_ROWS}"
        )
    commit, dirty = git_state()
    scope = validate_scope(args, dirty, policy)
    stats_out = args.stats_out or (
        getattr(config, policy["stats_attr"])
        if scope == "formal" else args.out + ".stats.json"
    )
    if scope == "smoke" and not is_smoke_path(stats_out):
        raise SystemExit("smoke stats output must remain under /tmp with 'smoke' in the path")

    adapter_model = os.path.join(args.adapter, "adapter_model.safetensors")
    if not os.path.isfile(adapter_model):
        raise SystemExit(f"missing SFT v3 adapter weights: {adapter_model}")
    actual_adapter_sha = file_sha256(adapter_model)
    if actual_adapter_sha != policy["adapter_sha256"]:
        raise SystemExit(
            f"adapter hash mismatch: {actual_adapter_sha} != {policy['adapter_sha256']}"
        )
    base_path, base_identity = verify_policy_base(args.policy, root=config.ROOT)

    pool_sha = file_sha256(args.pool)
    manifest_sha = file_sha256(args.manifest)
    with open(args.manifest) as handle:
        manifest = json.load(handle)
    all_rows = load_rows(args.pool)
    validate_phase_manifest(
        manifest, all_rows, pool_sha, manifest["excluded_sft_uid_sha256"]
    )
    rows = select_stage_rows(all_rows, manifest, args.target_rows)
    if args.n is not None:
        rows = rows[:args.n]

    prov = provenance(
        args,
        pool_sha,
        manifest_sha,
        args.adapter,
        commit,
        dirty,
        scope,
        policy,
        base_identity,
    )
    prov_sha = provenance_digest(prov)
    existing = load_existing(args.out)
    target_by_key = {row_key(row): row for row in rows}
    existing_by_key = {}
    for record in existing:
        key = row_key(record)
        if key in existing_by_key:
            raise ValueError(f"duplicate existing prompt record: {key}")
        if record.get("provenance_sha256") != prov_sha or record.get("provenance") != prov:
            raise ValueError("existing sample provenance differs from this run")
        if key not in target_by_key:
            raise ValueError("sample file contains prompts outside the requested stage")
        validate_rollout_record(
            record, target_by_key[key], prov, expected_samples=args.n_samples
        )
        existing_by_key[key] = record
    target_keys = {row_key(row) for row in rows}
    if not set(existing_by_key) <= target_keys:
        raise ValueError("sample file contains prompts outside the requested stage")
    pending = [row for row in rows if row_key(row) not in existing_by_key]
    print(
        f"scope={scope} prompts={len(rows):,} existing={len(existing):,} "
        f"pending={len(pending):,} K={args.n_samples} temp={args.temperature} top_p={args.top_p}"
    )
    if not pending:
        summary = summarize(existing, len(rows), prov_sha)
        atomic_json(stats_out, summary)
        print(json.dumps(summary, indent=2))
        return

    llm, tokenizer = load_llm(
        base_path, eager=not args.no_eager, enable_lora=True, max_lora_rank=16
    )
    lora = LoRARequest(
        lora_name=policy["lora_name"], lora_int_id=1, lora_path=args.adapter
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    records = list(existing)
    started = time.time()
    for start in range(0, len(pending), args.batch_prompts):
        chunk = pending[start:start + args.batch_prompts]
        generated = {}
        by_n = {}
        for row in chunk:
            by_n.setdefault(int(row["n_cand"]), []).append(row)
        for n_cand, group in sorted(by_n.items()):
            prompts = [stage1_prompt(tokenizer, row, int(row["k"]), TOP_K)[0]
                       for row in group]
            structured = StructuredOutputsParams(json=answer_schema(n_cand, TOP_K))
            params = [SamplingParams(
                n=args.n_samples, temperature=args.temperature, top_p=args.top_p,
                max_tokens=args.max_tokens, structured_outputs=structured,
                seed=prompt_seed(row["uid"], row["slate_id"], args.seed),
            ) for row in group]
            outputs = llm.generate(prompts, params, lora_request=lora)
            generated.update({row_key(row): output for row, output in zip(group, outputs)})
        new_records = [record_for(row, generated[row_key(row)], prov, prov_sha, args.seed)
                       for row in chunk]
        with open(args.out, "a") as handle:
            for record in new_records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        records.extend(new_records)
        summary = summarize(records, len(rows), prov_sha)
        atomic_json(stats_out, summary)
        elapsed = time.time() - started
        print(
            f"completed {len(records):,}/{len(rows):,} prompts | "
            f"coverage={summary['prompt_coverage']:.1%} | elapsed={elapsed/60:.1f}m"
        )

    print(json.dumps(summarize(records, len(rows), prov_sha), indent=2))


if __name__ == "__main__":
    main()
