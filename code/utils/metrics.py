"""Ranking metrics, parsing, and strict aggregation for the rerank benchmark.

All final slates carry exactly one positive per row: grade 3 (engaged
like/follow/forward) or grade 2 (longview) -- strata labels for reporting, everything else grade 0.
A scorer asks the model for its TOP-K candidate indices (best->worst) and scores the list with NDCG.
The functions below use gains gain(g)=2**g-1, but with a single positive the NDCG normalization
cancels the gain exactly, so NDCG@k depends only on the positive's rank -- the paper therefore just
says "NDCG@5". The graded implementation remains general, while the frozen
benchmark contract is single-positive.

Pure-python / numpy, no GPU. Imported by both scorers, the bench builder, the SFT/RFT quality gates
and the RL reward.
"""
import json
import math
import re

import numpy as np


# ---------- parse a model's ranking output ----------
def parse_topk_diag(text, n, k):
    """parse_topk plus a per-response DIAGNOSTIC dict (the CIR-style validity breakdown): how the raw
    emitted array failed, not just whether the cleaned picks are usable. Returns (picks, n_listed, diag)
    with diag = {"has_array": bool, "arr_len": raw array length, "n_oob": indices outside [0, n)
    ("hallucinated" candidate ids), "n_dup": in-range indices repeated}. picks/n_listed are identical
    to parse_topk's."""
    diag = {"has_array": False, "arr_len": 0, "n_oob": 0, "n_dup": 0}
    if not text:
        return None, 0, diag
    if "</think>" in text:
        text = text.split("</think>")[-1]
    arr = None
    for m in re.findall(r"\[[^\[\]]*\]", text, flags=re.S):   # flat int arrays only; keep the LAST valid
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
    """Parse a TOP-K ranking by extracting the LAST JSON array in `text` (after any </think>), keeping
    distinct in-range indices in listed order. Returns (picks, n_listed): picks = the distinct in-range
    indices the model listed (order preserved, may be < or == k), n_listed = len(picks); (None, 0) if no
    array is found.

    Array-only ON PURPOSE (not a global \\d+ scan of the whole text): a digit scan picks up markdown
    enumeration ("1. 2. 3." interest points) and [N] citations from the CoT prose and pastes them into
    the ranks, which both fakes completeness and corrupts NDCG. Reading only the emitted array avoids that.
    """
    picks, n_listed, _ = parse_topk_diag(text, n, k)
    return picks, n_listed


# ---------- NDCG over graded gains ----------
def gain(g):
    return (2.0 ** g) - 1.0


def dcg(order, grades, k=None):
    """order = list of candidate indices best->worst. grades[i] = grade of candidate i."""
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
    """1-based rank of the first max-grade item under `order` (for MRR_top / Hit@1)."""
    mx = max(grades)
    for p, idx in enumerate(order, start=1):
        if grades[idx] == mx:
            return p
    return None


def random_hit1(grades):
    """Analytic P(rank-1 item is a max-grade item) under a uniform random permutation = c/n."""
    mx = max(grades)
    c = sum(1 for g in grades if g == mx)
    return c / len(grades)


def recall(order, grades, k=None):
    """Recall@k over binary-ish relevance (grade > 0 = positive): fraction of this user's positives
    that appear anywhere in order[:k]. len(grades) (not order) is irrelevant to the denominator — only
    the positive count matters. No positives -> trivially 1.0 (nothing to miss), matching ndcg()'s
    convention for an empty ideal."""
    if k is None:
        k = len(order)
    n_pos = sum(1 for g in grades if g > 0)
    if n_pos == 0:
        return 1.0
    hits = sum(1 for idx in order[:k] if grades[idx] > 0)
    return hits / n_pos


def random_recall(grades, k):
    """Analytic E[Recall@k] under a uniform random permutation: E[hits] = k * n_pos / n (hypergeometric
    mean), so E[recall] = k/n exactly — independent of n_pos, no Monte Carlo needed."""
    n_pos = sum(1 for g in grades if g > 0)
    if n_pos == 0:
        return 1.0
    return min(1.0, k / len(grades))


def random_mrr(grades, trials=2000, seed=0, k=None):
    """Monte-Carlo mean MRR_top (1/rank of first max-grade item) under a random permutation. If `k` is
    set (top-K task), a first max-grade item beyond rank k contributes 0 (it's not in the returned list)."""
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


# ---------- aggregate metrics over per-user results ----------
def is_identity_collapse(order, n_out):
    """True iff `order` is the literal identity permutation [0,1,...,n_out-1] — the model skipped real
    reasoning and emitted the candidates in their (per-user-shuffled) input order. Since candidates are
    shuffled per-user at bench-build time, this is statistically a random top-n_out (doesn't bias
    NDCG/Recall), but it means the model didn't engage with the task — `is_valid` alone can't tell this
    apart from genuine reasoning, since the identity list is syntactically a perfectly valid answer."""
    return order == list(range(n_out))


def identity_collapse_rate(results, n_out):
    if not results:
        return 0.0
    return sum(1 for r in results if is_identity_collapse(r.get("order"), n_out)) / len(results)


def is_valid(r, n_out):
    """A response is a SUCCESS only if the model committed to a full TOP-K list: it listed at least
    `n_out` distinct in-range indices (n_listed >= n_out). Fewer than n_out (a repeat or a short list)
    is a FAILURE, as is nothing parseable (order is None). With grammar/responseSchema forcing exactly
    n_out items, n_listed < n_out means the model repeated an index — a genuine failure."""
    order = r.get("order")
    if order is None:
        return False
    return r.get("n_listed", 0) >= n_out


def aggregate(results, k_ndcg=5, n_out=10, rnd_seed=0, with_random=True):
    """results: list of dicts each with `order` (the model's top-K picks, or None if unparsed),
    `n_listed` (# distinct valid indices listed) and `grades`.

    TOP-K task: the model returns an ordered list of its `n_out` best candidates from the 30-slate;
    ranks beyond n_out are not requested. All quality metrics are over the FULL user set — a failure
    contributes 0. **A response is valid ONLY if n_listed >= n_out** (see `is_valid`); NDCG@{k_ndcg} /
    Recall@{k_ndcg} (headline) and NDCG@{n_out} / Recall@{n_out} are scored over the picks. The random
    baseline is a random ranker's top-K, a fixed benchmark constant. `parse_rate` (= valid-top-K rate)
    and `ndcg@{k}_parsed` (conditional on valid) are reported for diagnosis; `incomplete`/`unparsed`
    break down failures. `with_random=False` skips the (slow) Monte-Carlo NDCG/MRR baselines (Recall's
    random baseline is a cheap closed form, always included).
    """
    m = len(results)
    if m == 0:
        return {"n_users": 0, "parsed": 0, "parse_rate": 0.0}
    valid = [r for r in results if is_valid(r, n_out)]
    p = len(valid)
    n_incomplete = sum(1 for r in results
                       if r.get("order") is not None and r.get("n_listed", 0) < n_out)
    n_unparsed = sum(1 for r in results if r.get("order") is None)

    def score(r, k):
        return ndcg(r["order"], r["grades"], k) if is_valid(r, n_out) else 0.0

    def recall_score(r, k):
        return recall(r["order"], r["grades"], k) if is_valid(r, n_out) else 0.0

    ndcg_k = [score(r, k_ndcg) for r in results]
    ndcg_out = [score(r, n_out) for r in results]
    recall_k = [recall_score(r, k_ndcg) for r in results]
    recall_out = [recall_score(r, n_out) for r in results]
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
    # CIR-style validity breakdown (only when the scorer recorded per-row parse diagnostics — the
    # "diag" dict from parse_topk_diag on the FINAL answer text used for scoring). Rates are over all
    # users: hallucinated = emitted >=1 out-of-range index; repeated = emitted >=1 duplicate in-range
    # index; wrong_length = emitted an array of length != n_out; no_array = no parseable index array.
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
