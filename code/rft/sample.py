# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Best-of-8 RFT rollouts from the SFT model (paper Sec. 5, RFT).

Loads the pinned base model with the dev-selected SFT LoRA in vLLM and draws eight grammar-
constrained single-stage rollouts for each prompt of the 6k post-training subset, using SID-only
prompts and no labels. Each prompt gets its own sampling seed derived from its key. Records are
appended to data/rft/<model>/rollouts.jsonl after every batch (progress and the valid-rollout rate
go to rollouts_stats.json), and running the same command again skips prompts that are already done.
rft/build_dataset.py applies the rank-1 gate afterwards. --adapter selects your own SFT checkpoint;
the default is the paper's (rft/recipe.py).

    python code/rft/sample.py --model onerec-1.7b --adapter saves/onerec-1.7b/sft/checkpoint-1035
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config  # noqa: E402

config.setup_env()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

from rft.recipe import (MAX_TOKENS, MODELS, N_SAMPLES, RFT_RECIPE, SEED, TEMPERATURE,  # noqa: E402
                        TOP_P, sft_adapter)
from utils.data_io import dump_json, load_rows  # noqa: E402
from utils.metrics import is_identity_collapse, ndcg  # noqa: E402
from utils.pools import row_key, select_subset  # noqa: E402
from utils.prompts import TOP_K  # noqa: E402


def extract_cot(text):
    """The completed reasoning segment between <think> and </think>, or None."""
    if not text or "</think>" not in text:
        return None
    before = text.split("</think>", 1)[0]
    if "<think>" in before:
        before = before.split("<think>", 1)[-1]
    cot = before.strip()
    return cot or None


def prompt_seed(uid, slate_id, seed):
    """Per-prompt vLLM seed derived from the recipe string, the run seed and the prompt key."""
    payload = f"{RFT_RECIPE}|{seed}|{int(uid)}|{str(slate_id)}"
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16) % (2 ** 31)


def load_records(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def record_for(row, output, seed):
    from utils.sid_engine import parse_stage1
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
        rollouts.append({
            "sample_index": index,
            "text": text,
            "cot_text": extract_cot(text),
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
    return {
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
        "rollouts": rollouts,
    }


def summarize(records, target_rows, model, sampling):
    """Progress and the valid-rollout rate; rft/build_dataset.py reports coverage after its gate."""
    rollouts = [sample for record in records for sample in record["rollouts"]]
    valid = sum(sample["valid"] for sample in rollouts)
    return {
        "model": model,
        "sampling": sampling,
        "target_rows": target_rows,
        "completed_prompts": len(records),
        "complete": len(records) == target_rows,
        "rollouts": len(rollouts),
        "valid_rollouts": valid,
        "valid_rate": round(valid / len(rollouts), 4) if rollouts else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--adapter", default=None,
                        help="SFT LoRA adapter (default: saves/<model>/sft/<selected checkpoint>)")
    parser.add_argument("--pool", default=config.POSTTRAIN_POOL)
    parser.add_argument("--out", default=None, help="default: data/rft/<model>/rollouts.jsonl")
    parser.add_argument("--n", type=int, default=None, help="only the first N subset prompts")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-prompts", type=int, default=128)
    parser.add_argument("--no-eager", action="store_true")
    args = parser.parse_args()
    adapter = args.adapter or sft_adapter(args.model)
    out = args.out or os.path.join(config.rft_dir(args.model), "rollouts.jsonl")
    stats_out = os.path.splitext(out)[0] + "_stats.json"
    if not os.path.isfile(os.path.join(adapter, "adapter_model.safetensors")):
        raise SystemExit(f"missing SFT adapter: {adapter} (train it with code/sft first)")

    rows = select_subset(load_rows(args.pool))
    if args.n is not None:
        rows = rows[:args.n]
    wanted = {row_key(row) for row in rows}
    records = [record for record in load_records(out) if row_key(record) in wanted]
    done = {row_key(record) for record in records}
    pending = [row for row in rows if row_key(row) not in done]
    sampling = {"n": args.n_samples, "temperature": args.temperature, "top_p": args.top_p,
                "max_tokens": args.max_tokens, "seed": args.seed, "top_k": TOP_K}
    print(f"prompts={len(rows):,} done={len(done):,} pending={len(pending):,} {sampling}")

    if pending:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        from vllm.sampling_params import StructuredOutputsParams

        from utils.sid_engine import answer_schema, load_llm, stage1_prompt
        llm, tokenizer = load_llm(config.model_path(args.model), eager=not args.no_eager,
                                  enable_lora=True, max_lora_rank=16)
        lora = LoRARequest(lora_name="sft", lora_int_id=1, lora_path=adapter)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        started = time.time()
        for start in range(0, len(pending), args.batch_prompts):
            chunk = pending[start:start + args.batch_prompts]
            generated = {}
            by_n = {}
            for row in chunk:
                by_n.setdefault(int(row["n_cand"]), []).append(row)
            for n_cand, group in sorted(by_n.items()):
                prompts = [stage1_prompt(tokenizer, row, int(row["k"]), TOP_K)[0] for row in group]
                structured = StructuredOutputsParams(json=answer_schema(n_cand, TOP_K))
                params = [SamplingParams(
                    n=args.n_samples, temperature=args.temperature, top_p=args.top_p,
                    max_tokens=args.max_tokens, structured_outputs=structured,
                    seed=prompt_seed(row["uid"], row["slate_id"], args.seed),
                ) for row in group]
                outputs = llm.generate(prompts, params, lora_request=lora)
                generated.update({row_key(row): output for row, output in zip(group, outputs)})
            new_records = [record_for(row, generated[row_key(row)], args.seed) for row in chunk]
            with open(out, "a") as f:
                for record in new_records:
                    f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            records.extend(new_records)
            summary = summarize(records, len(rows), args.model, sampling)
            dump_json(stats_out, summary)
            print(f"completed {len(records):,}/{len(rows):,} prompts | "
                  f"valid={summary['valid_rate']:.1%} | "
                  f"elapsed={(time.time() - started) / 60:.1f}m")

    summary = summarize(records, len(rows), args.model, sampling)
    dump_json(stats_out, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
