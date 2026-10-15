# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Output parsing, ranking metrics, strict aggregation and paired statistics (paper Sec. 3, 4).

Every slate has exactly one positive: grade 3 (like, follow or forward) or grade 2 (long view);
the grade only labels strata, all other candidates are grade 0. A model returns its top-K
candidate indices, best first. Gains are 2**g - 1, but with a single positive the normalization
cancels the gain, so NDCG@k depends only on the positive's rank.

Pure Python and NumPy. Used by the scorers, the dataset builders, the RLVR reward and the audits.
"""
import itertools
import json
import math
import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np


# Parse a model's ranking output
def parse_topk_diag(text, n, k):
    """parse_topk plus a diagnostic dict describing how the raw array failed. Returns
    (picks, n_listed, diag) with diag = {"has_array", "arr_len" (raw array length), "n_oob" (indices
    outside [0, n), i.e. hallucinated ids), "n_dup" (repeated in-range indices)}."""
    diag = {"has_array": False, "arr_len": 0, "n_oob": 0, "n_dup": 0}
    if not text:
        return None, 0, diag
    if "</think>" in text:
        text = text.split("</think>")[-1]
    arr = None
    for m in re.findall(r"\[[^\[\]]*\]", text, flags=re.S):   # flat int arrays; keep the last one
        try:
            cand = json.loads(m)
        except Exception:
            continue
        if isinstance(cand, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in cand):
            arr = cand
    if arr is None:
        return None, 0, diag
    diag["has_array"] = True
    diag["arr_len"] = len(arr)
    seen, picks = set(), []
    for v in arr:
        if not 0 <= v < n:
            diag["n_oob"] += 1
        elif v in seen:
            diag["n_dup"] += 1
        else:
            seen.add(v)
            picks.append(v)
    if not picks:
        return None, 0, diag
    return picks, len(picks), diag


def parse_topk(text, n, k):
    """Parse a top-K ranking from the last flat JSON integer array in `text` (after any </think>),
    keeping distinct in-range indices in listed order. Returns (picks, n_listed), or (None, 0) if no
    array is found. Only the array is read, never digits in the reasoning prose.
    """
    picks, n_listed, _ = parse_topk_diag(text, n, k)
    return picks, n_listed


# NDCG over graded gains
def gain(g):
    return (2.0 ** g) - 1.0


def dcg(order, grades, k=None):
    """order: candidate indices, best first. grades[i]: grade of candidate i."""
    if k is None:
        k = len(order)
    s = 0.0
    for p, idx in enumerate(order[:k], start=1):
        s += gain(grades[idx]) / math.log2(p + 1)
    return s


def idcg(grades, k=None):
    return dcg(sorted(range(len(grades)), key=lambda i: grades[i], reverse=True), grades, k)


def ndcg(order, grades, k=None):
    ideal = idcg(grades, k)
    return 1.0 if ideal == 0 else dcg(order, grades, k) / ideal


def random_ndcg(grades, k=None, trials=2000, seed=0):
    """Monte-Carlo mean NDCG@k of a uniformly random permutation (the no-signal baseline)."""
    ideal = idcg(grades, k)
    if ideal == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    n = len(grades)
    tot = 0.0
    for _ in range(trials):
        order = list(rng.permutation(n))
        tot += dcg(order, grades, k) / ideal
    return tot / trials


def best_rank(order, grades):
    """1-based rank of the first max-grade item under `order` (for MRR and Hit@1)."""
    mx = max(grades)
    for p, idx in enumerate(order, start=1):
        if grades[idx] == mx:
            return p
    return None


def recall(order, grades, k=None):
    """Recall@k: fraction of the user's positives (grade > 0) in order[:k]; 1.0 without positives,
    matching ndcg() for an empty ideal."""
    if k is None:
        k = len(order)
    n_pos = sum(1 for g in grades if g > 0)
    if n_pos == 0:
        return 1.0
    hits = sum(1 for idx in order[:k] if grades[idx] > 0)
    return hits / n_pos


def random_recall(grades, k):
    """Expected Recall@k of a uniform random permutation, k/n (independent of the positive count)."""
    n_pos = sum(1 for g in grades if g > 0)
    if n_pos == 0:
        return 1.0
    return min(1.0, k / len(grades))


def random_mrr(grades, trials=2000, seed=0, k=None):
    """Monte Carlo MRR of a random permutation. With `k` set, a first max-grade item beyond rank k
    contributes 0, as in the top-K task."""
    mx = max(grades)
    rng = np.random.default_rng(seed)
    n = len(grades)
    tot = 0.0
    for _ in range(trials):
        for p, idx in enumerate(rng.permutation(n), start=1):
            if grades[idx] == mx:
                if k is None or p <= k:
                    tot += 1.0 / p
                break
    return tot / trials


# Aggregate metrics over per-user results
def is_identity_collapse(order, n_out):
    """True iff `order` is the identity list [0, 1, ..., n_out-1], i.e. the model echoed the first
    candidates in display order. This is a valid answer, so is_valid cannot detect it."""
    return order == list(range(n_out))


def identity_collapse_rate(results, n_out):
    if not results:
        return 0.0
    return sum(1 for r in results if is_identity_collapse(r.get("order"), n_out)) / len(results)


def is_valid(r, n_out):
    """Valid iff the model listed at least n_out distinct in-range indices. A short list, a repeated
    index or no parseable array is a failure and scores 0."""
    order = r.get("order")
    if order is None:
        return False
    return r.get("n_listed", 0) >= n_out


def user_score(metric, r, k, n_out):
    """metric(order, grades, k) for a valid response, else 0."""
    return metric(r["order"], r["grades"], k) if is_valid(r, n_out) else 0.0


def aggregate(results, k_ndcg=5, n_out=10, rnd_seed=0, with_random=True):
    """Aggregate per-user results (dicts with `order`, `n_listed`, `grades`) into metrics.

    Every metric averages over all users and an invalid response (see is_valid) contributes 0.
    Reports NDCG and Recall at k_ndcg and n_out, MRR, the valid rate (`parse_rate`), NDCG among valid
    responses, failure counts, identity-collapse rate, interface diagnostics when rows carry "diag",
    and random baselines (with_random=False skips the slow Monte Carlo NDCG and MRR baselines).
    """
    m = len(results)
    if m == 0:
        return {"n_users": 0, "parsed": 0, "parse_rate": 0.0}
    valid = [r for r in results if is_valid(r, n_out)]
    p = len(valid)
    n_incomplete = sum(1 for r in results
                       if r.get("order") is not None and r.get("n_listed", 0) < n_out)
    n_unparsed = sum(1 for r in results if r.get("order") is None)
    ndcg_k = [user_score(ndcg, r, k_ndcg, n_out) for r in results]
    ndcg_out = [user_score(ndcg, r, n_out, n_out) for r in results]
    recall_k = [user_score(recall, r, k_ndcg, n_out) for r in results]
    recall_out = [user_score(recall, r, n_out, n_out) for r in results]
    branks = [best_rank(r["order"], r["grades"]) if is_valid(r, n_out) else None
              for r in results]
    mrr_top = [(1.0 / b) if b else 0.0 for b in branks]

    parsed_ndcg_k = [ndcg(r["order"], r["grades"], k_ndcg) for r in valid]
    out = {
        "n_users": m, "parsed": p, "parse_rate": round(p / m, 4),
        "incomplete": n_incomplete, "unparsed": n_unparsed,
        "n_out": n_out, "k_ndcg": k_ndcg,
        f"ndcg@{k_ndcg}": round(float(np.mean(ndcg_k)), 4),
        f"ndcg@{n_out}": round(float(np.mean(ndcg_out)), 4),
        f"recall@{k_ndcg}": round(float(np.mean(recall_k)), 4),
        f"recall@{n_out}": round(float(np.mean(recall_out)), 4),
        "mrr_top": round(float(np.mean(mrr_top)), 4),
        f"ndcg@{k_ndcg}_parsed": round(float(np.mean(parsed_ndcg_k)), 4) if valid else None,
        "ideal_ndcg": 1.0,
        "mean_n_cand": round(float(np.mean([len(r["grades"]) for r in results])), 2),
        "identity_collapse_rate": round(identity_collapse_rate(results, n_out), 4),
    }
    # cir (candidate-index reliability): diagnostics from parse_topk_diag on the final answer, as
    # rates over all users. hallucinated = any out-of-range index, repeated = any duplicate index,
    # wrong_length = array length != n_out, no_array = no parseable array.
    diags = [r["diag"] for r in results if isinstance(r.get("diag"), dict)]
    if diags:
        out["cir"] = {
            "hallucinated_index_rate": round(sum(d["n_oob"] > 0 for d in diags) / len(diags), 4),
            "repeated_index_rate": round(sum(d["n_dup"] > 0 for d in diags) / len(diags), 4),
            "wrong_length_rate": round(sum(d["has_array"] and d["arr_len"] != n_out
                                           for d in diags) / len(diags), 4),
            "no_array_rate": round(sum(not d["has_array"] for d in diags) / len(diags), 4),
            "mean_oob_per_response": round(float(np.mean([d["n_oob"] for d in diags])), 3),
            "mean_dup_per_response": round(float(np.mean([d["n_dup"] for d in diags])), 3),
        }
    out[f"random_recall@{k_ndcg}"] = round(float(np.mean(
        [random_recall(r["grades"], k_ndcg) for r in results])), 4)
    out[f"random_recall@{n_out}"] = round(float(np.mean(
        [random_recall(r["grades"], n_out) for r in results])), 4)
    if with_random:
        rnd_k = [random_ndcg(r["grades"], k_ndcg, seed=rnd_seed) for r in results]
        rnd_out = [random_ndcg(r["grades"], n_out, seed=rnd_seed) for r in results]
        out.update({
            f"random_ndcg@{k_ndcg}": round(float(np.mean(rnd_k)), 4),
            f"random_ndcg@{n_out}": round(float(np.mean(rnd_out)), 4),
            "random_mrr_top": round(float(np.mean([random_mrr(r["grades"], seed=rnd_seed, k=n_out)
                                                   for r in results])), 4),
        })
    return out


# Paired statistics
def half_up(value, places):
    """Round half up, as the paper does (Python's format rounds half to even)."""
    return Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


BOOT = 10_000


def paired_bootstrap(diffs, n_boot=BOOT, seed=0):
    """95% percentile CI of the mean of per-user paired differences, resampling users."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def c4(n):
    """Bias factor E[s]/sigma of the sample SD for a size-n Gaussian sample, used to compare SDs
    estimated from different numbers of replicates."""
    return math.sqrt(2.0 / (n - 1)) * math.gamma(n / 2.0) / math.gamma((n - 1) / 2.0)


def kendall_tau_topk(list_a, list_b, k):
    """Kendall tau-b between two top-k item lists. Items missing from a list tie at rank k+1, and
    pairs run over the union of both lists. None when no pair is comparable."""
    union = list(dict.fromkeys(list(list_a) + list(list_b)))
    ra = {x: i + 1 for i, x in enumerate(list_a)}
    rb = {x: i + 1 for i, x in enumerate(list_b)}
    conc = disc = ties_a = ties_b = 0
    for x, y in itertools.combinations(union, 2):
        da = ra.get(x, k + 1) - ra.get(y, k + 1)
        db = rb.get(x, k + 1) - rb.get(y, k + 1)
        if da == 0 and db == 0:
            continue
        if da == 0:
            ties_a += 1
        elif db == 0:
            ties_b += 1
        elif (da > 0) == (db > 0):
            conc += 1
        else:
            disc += 1
    denom = np.sqrt((conc + disc + ties_a) * (conc + disc + ties_b))
    return None if denom == 0 else float((conc - disc) / denom)
