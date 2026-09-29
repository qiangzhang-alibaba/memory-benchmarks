"""
Metrics Computation
===================

Shared metrics helpers for all benchmarks:
- Overall accuracy and average score
- Per-group breakdown (category, question_type, etc.)
- Multi-cutoff evaluation
- Kendall tau-b for event ordering (BEAM)
"""

from __future__ import annotations

import re
import statistics
import string
from collections import Counter, defaultdict
from typing import Any

from .schema import CutoffMetrics, GroupMetrics, Metrics


# ===============================================================================
# TOKEN-LEVEL F1 (LoCoMo official methodology, stdlib-only reimplementation)
# ===============================================================================
# Mirrors datasets/locomo/repo/task_eval/evaluation.py (normalize_answer +
# Porter-stemmed token F1) without its heavy dependencies (nltk / regex /
# numpy), so the benchmark pipeline stays installable from requirements.txt.


def _normalize_answer(s: str) -> str:
    """Lowercase, drop commas/punctuation/articles and collapse whitespace.

    Same normalization as the official LoCoMo ``normalize_answer`` (which also
    strips "and" as an article-like token).
    """
    s = s.replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s.lower())
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    return " ".join(s.split())


def _cv_pattern(w: str) -> str:
    """Consonant/vowel pattern string, with Porter's special 'y' handling.

    'y' is a consonant at word start or after a vowel, and a vowel after a
    consonant (so "cycling" -> C V C C V C G..., letting -ing strip to "cycl").
    """
    out = []
    for i, ch in enumerate(w):
        if ch in "aeiou":
            out.append("V")
        elif ch == "y":
            prev_vowel = i > 0 and w[i - 1] in "aeiou"
            out.append("C" if (i == 0 or prev_vowel) else "V")
        else:
            out.append("C")
    return "".join(out)


def _measure(stem: str) -> int:
    """Porter stemmer m(): number of VC sequences in [C](VC)^m[V]."""
    cv = _cv_pattern(stem)
    cv = re.sub(r"C+", "C", cv)
    cv = re.sub(r"V+", "V", cv)
    return cv.count("VC")


def _has_vowel(stem: str) -> bool:
    return "V" in _cv_pattern(stem)


def _ends_double_consonant(stem: str) -> bool:
    return len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou"


def porter_stem(word: str) -> str:
    """Compact pure-Python Porter stemmer (algorithm from Porter 1980).

    Produces stems equivalent to nltk.stem.PorterStemmer used by the official
    LoCoMo evaluation for common English words.
    """
    w = word.lower()
    if len(w) <= 2:
        return w

    # Step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif not w.endswith("ss") and w.endswith("s"):
        w = w[:-1]

    # Step 1b
    step1b_extra = False
    if w.endswith("eed"):
        stem = w[:-3]
        if _measure(stem) > 0:
            w = w[:-1]
    elif w.endswith("ed"):
        stem = w[:-2]
        if _has_vowel(stem):
            w = stem
            step1b_extra = True
    elif w.endswith("ing"):
        stem = w[:-3]
        if _has_vowel(stem):
            w = stem
            step1b_extra = True

    if step1b_extra:
        if w.endswith("at") or w.endswith("bl") or w.endswith("iz"):
            w += "e"
        elif _ends_double_consonant(w) and w[-1] not in "lsz":
            w = w[:-1]
        elif _measure(w) == 1 and _ends_cvc_simple(w):
            w += "e"

    # Step 1c
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"

    # Step 2
    step2 = {
        "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
        "izer": "ize", "abli": "able", "alli": "al", "entli": "ent",
        "eli": "e", "ousli": "ous", "ization": "ize", "ation": "ate",
        "ator": "ate", "alism": "al", "iveness": "ive", "fulness": "ful",
        "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble",
    }
    for suffix, repl in step2.items():
        if w.endswith(suffix):
            stem = w[: -len(suffix)]
            if _measure(stem) > 0:
                w = stem + repl
            break

    # Step 3
    step3 = {
        "icate": "ic", "ative": "", "alize": "al",
        "iciti": "ic", "ical": "ic", "ful": "", "ness": "",
    }
    for suffix, repl in step3.items():
        if w.endswith(suffix):
            stem = w[: -len(suffix)]
            if _measure(stem) > 0:
                w = stem + repl
            break

    # Step 4: (m>1 on the STEM) SUFFIX -> "", except -ion where the stem must
    # end in s/t. Classic Porter measures m on the remaining stem, not the
    # full word (so "disable"->"dis" has m=1 and is NOT stripped to "dis").
    step4_suffixes = (
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
        "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
        "ous", "ive", "ize",
    )
    for suffix in step4_suffixes:
        if w.endswith(suffix):
            stem = w[: -len(suffix)]
            if _measure(stem) > 1:
                if suffix == "ion":
                    if stem and stem[-1] in "st":
                        w = stem
                else:
                    w = stem
            break

    # Step 5a
    if w.endswith("e"):
        stem = w[:-1]
        if _measure(stem) > 1 or (_measure(stem) == 1 and not _ends_cvc_simple(stem)):
            w = stem
    # Step 5b
    if _measure(w) > 1 and _ends_double_consonant(w) and w[-1] == "l":
        w = w[:-1]

    return w


def _ends_cvc_simple(w: str) -> bool:
    """*o condition: ends C-V-C where the final consonant is not w, x or y.

    Vowels are aeiou; y counts as a consonant everywhere except when it is the
    final letter, which is excluded via the w/x/y check (classic Porter *o).
    """
    if len(w) < 3:
        return False
    c1, v, c2 = w[-3], w[-2], w[-1]
    if c2 in "aeiouwxy" or v not in "aeiou" or c1 in "aeiou":
        return False
    return True


def token_f1(prediction: str, ground_truth: str) -> float:
    """Porter-stemmed token-level F1 between one prediction and one answer.

    Same computation as official LoCoMo ``f1_score``: normalize both sides,
    stem tokens, precision/recall over the multiset intersection.
    """
    pred_tokens = [porter_stem(t) for t in _normalize_answer(prediction).split()]
    gt_tokens = [porter_stem(t) for t in _normalize_answer(ground_truth).split()]
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def locomo_f1(prediction: str, ground_truth: str, category: int) -> float:
    """Token F1 following the official LoCoMo per-category convention.

    - category 1 (multi-hop): comma-split both sides into sub-answers and take
      the mean over ground-truth parts of the best prediction match (official
      ``f1``), giving partial credit for multi-part answers.
    - categories 2/3/4: plain stemmed token F1 (official ``f1_score``).
    """
    if category == 1:
        preds = [p.strip() for p in prediction.split(",") if p.strip()]
        gts = [g.strip() for g in ground_truth.split(",") if g.strip()]
        if not preds or not gts:
            return token_f1(prediction, ground_truth)
        return statistics.mean(
            max(token_f1(p, gt) for p in preds) for gt in gts
        )
    return token_f1(prediction, ground_truth)


def compute_group_metrics(
    evaluations: list[dict[str, Any]],
    group_key: str,
    cutoff_label: str | None = None,
    pass_threshold: float = 0.5,
) -> dict[str, GroupMetrics]:
    """Compute metrics broken down by a group key.

    Args:
        evaluations: List of evaluation result dicts.
        group_key: Key to group by (e.g., "category_name", "question_type").
        cutoff_label: If set, read score from cutoff_results[label].
        pass_threshold: Score threshold for "correct" classification.

    Returns:
        Dict mapping group name to GroupMetrics.
    """
    groups: dict[str, list[float]] = defaultdict(list)

    for e in evaluations:
        group = e.get(group_key, "unknown")
        if cutoff_label:
            cr = e.get("cutoff_results", {}).get(cutoff_label, {})
            score = cr.get("score", 0.0)
        else:
            score = e.get("score", 0.0)
        groups[group].append(score)

    result = {}
    for name in sorted(groups):
        scores = groups[name]
        correct = sum(1 for s in scores if s >= pass_threshold)
        result[name] = GroupMetrics(
            group_name=name,
            total=len(scores),
            correct=correct,
            accuracy=correct / len(scores) * 100 if scores else 0.0,
            avg_score=statistics.mean(scores) * 100 if scores else 0.0,
        )
    return result


def compute_overall_metrics(
    evaluations: list[dict[str, Any]],
    group_key: str,
    cutoffs: list[str] | None = None,
    pass_threshold: float = 0.5,
) -> Metrics:
    """Compute full metrics suite including per-group and multi-cutoff breakdowns.

    Args:
        evaluations: List of evaluation result dicts.
        group_key: Key to group by.
        cutoffs: List of cutoff label strings (e.g., ["top_10", "top_50"]).
        pass_threshold: Score threshold for "correct".

    Returns:
        Metrics object.
    """
    if not evaluations:
        return Metrics()

    # Primary cutoff (the largest one, or first if no cutoffs)
    primary_cutoff = cutoffs[-1] if cutoffs else None

    # Overall scores from primary cutoff
    all_scores: list[float] = []
    error_count = 0
    for e in evaluations:
        if primary_cutoff:
            cr = e.get("cutoff_results", {}).get(primary_cutoff, {})
            all_scores.append(cr.get("score", 0.0))
            if cr.get("judgment") == "ERROR" or cr.get("error"):
                error_count += 1
        else:
            all_scores.append(e.get("score", 0.0))
            if e.get("judgment") == "ERROR":
                error_count += 1

    correct = sum(1 for s in all_scores if s >= pass_threshold)
    total = len(all_scores)

    metrics = Metrics(
        overall_accuracy=correct / total * 100 if total else 0.0,
        overall_avg_score=statistics.mean(all_scores) * 100 if all_scores else 0.0,
        total=total,
        correct=correct,
        errors=error_count,
    )

    # By-group at primary cutoff
    if primary_cutoff:
        metrics.by_group = compute_group_metrics(evaluations, group_key, primary_cutoff, pass_threshold)
    else:
        metrics.by_group = compute_group_metrics(evaluations, group_key, None, pass_threshold)

    # By-cutoff
    if cutoffs:
        for label in cutoffs:
            group_metrics = compute_group_metrics(evaluations, group_key, label, pass_threshold)

            cutoff_scores = []
            cutoff_errors = 0
            for e in evaluations:
                cr = e.get("cutoff_results", {}).get(label, {})
                cutoff_scores.append(cr.get("score", 0.0))
                if cr.get("judgment") == "ERROR" or cr.get("error"):
                    cutoff_errors += 1

            cutoff_correct = sum(1 for s in cutoff_scores if s >= pass_threshold)
            metrics.by_cutoff[label] = CutoffMetrics(
                cutoff=label,
                overall={
                    "total": len(cutoff_scores),
                    "correct": cutoff_correct,
                    "errors": cutoff_errors,
                    "accuracy": cutoff_correct / len(cutoff_scores) * 100 if cutoff_scores else 0.0,
                    "avg_score": statistics.mean(cutoff_scores) * 100 if cutoff_scores else 0.0,
                },
                by_group=group_metrics,
            )

    return metrics


def compute_latency_summary(latency_seconds: list[float]) -> dict[str, Any]:
    """Compute latency summary statistics (all values in seconds).

    Args:
        latency_seconds: List of latency measurements in seconds.

    Returns:
        Dict with count, avg_s, min_s, p50_s, p95_s, max_s.
    """
    if not latency_seconds:
        return {
            "count": 0,
            "avg_s": 0.0,
            "min_s": 0.0,
            "p50_s": 0.0,
            "p95_s": 0.0,
            "max_s": 0.0,
        }

    vals = sorted(latency_seconds)

    def pct(p: float) -> float:
        # Linear interpolation between closest ranks (numpy-style)
        k = (len(vals) - 1) * p
        lo = int(k)
        hi = min(lo + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)

    return {
        "count": len(vals),
        "avg_s": round(statistics.mean(vals), 3),
        "min_s": round(vals[0], 3),
        "p50_s": round(pct(0.50), 3),
        "p95_s": round(pct(0.95), 3),
        "max_s": round(vals[-1], 3),
    }


def compute_latency_by_cutoff(
    evaluations: list[dict[str, Any]],
    cutoff_labels: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute latency summaries separately for each top-k cutoff.

    For every cutoff label (e.g. "top_10", "top_20", "top_50", "top_200") this
    aggregates the per-question latencies recorded in ``cutoff_results[label]``:

    - ``search``: retrieval latency measured by an independent search executed
      at that cutoff's own top_k (the primary metric of interest). Entries
      recorded as ``None`` (e.g. evaluate-only runs, where no Mem0 search
      happens and the latency is explicitly not measured rather than reused
      from the primary top_k search) are skipped, so a cutoff may report
      ``count == 0``.
    - ``generation``: answer-generation latency at that cutoff.
    - ``total``: search + generation (judge excluded); skipped when the
      per-cutoff search latency was not measured.

    Args:
        evaluations: List of evaluation result dicts.
        cutoff_labels: Cutoff label strings to report, in display order.

    Returns:
        Dict mapping cutoff label -> {"search", "generation", "total"} summaries.
    """
    result: dict[str, dict[str, Any]] = {}
    for label in cutoff_labels:
        total_ms: list[float] = []
        gen_ms: list[float] = []
        search_ms: list[float] = []
        for e in evaluations:
            cr = e.get("cutoff_results", {}).get(label, {})
            if not isinstance(cr, dict):
                continue
            tl = cr.get("latency_ms", 0) or 0
            gl = cr.get("generation_latency_ms", 0) or 0
            sl = cr.get("search_latency_ms", 0) or 0
            if tl > 0:
                total_ms.append(tl)
            if gl > 0:
                gen_ms.append(gl)
            if sl > 0:
                search_ms.append(sl)
        result[label] = {
            "search": compute_latency_summary([v / 1000 for v in search_ms]),
            "generation": compute_latency_summary([v / 1000 for v in gen_ms]),
            "total": compute_latency_summary([v / 1000 for v in total_ms]),
        }
    return result


def print_latency_by_cutoff(
    latency_by_cutoff: dict[str, dict[str, Any]],
    cutoff_labels: list[str],
) -> None:
    """Print per-cutoff SEARCH latency summaries (seconds) to stdout."""
    print(
        "\nSearch latency by cutoff (seconds; each cutoff searched independently at its own top_k):"
    )
    for label in cutoff_labels:
        entry = latency_by_cutoff.get(label)
        if not entry:
            continue
        s = entry["search"]
        if s["count"] == 0:
            print(f"  {label}: no search latency samples recorded")
            continue
        print(
            f"  {label}: search p50={s['p50_s']:.3f}s p95={s['p95_s']:.3f}s "
            f"avg={s['avg_s']:.3f}s min={s['min_s']:.3f}s max={s['max_s']:.3f}s "
            f"({s['count']} queries)"
        )


def compute_kendall_tau_b(predicted_order: list[int], reference_order: list[int]) -> float:
    """Compute Kendall tau-b rank correlation coefficient.

    Used by BEAM for event_ordering questions to measure how well
    the predicted ordering matches the reference ordering.

    Args:
        predicted_order: List of indices in predicted order.
        reference_order: List of indices in reference order.

    Returns:
        Tau-b coefficient in [-1, 1]. 1 = perfect agreement.
    """
    if len(predicted_order) < 2 or len(reference_order) < 2:
        return 0.0

    # Build rank maps
    n = max(len(predicted_order), len(reference_order))
    pred_rank = {v: i for i, v in enumerate(predicted_order)}
    ref_rank = {v: i for i, v in enumerate(reference_order)}

    # Only consider items in both lists
    common = set(predicted_order) & set(reference_order)
    items = sorted(common)

    if len(items) < 2:
        return 0.0

    concordant = 0
    discordant = 0
    tied_pred = 0
    tied_ref = 0

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            pred_diff = pred_rank[a] - pred_rank[b]
            ref_diff = ref_rank[a] - ref_rank[b]

            if pred_diff == 0 and ref_diff == 0:
                tied_pred += 1
                tied_ref += 1
            elif pred_diff == 0:
                tied_pred += 1
            elif ref_diff == 0:
                tied_ref += 1
            elif (pred_diff > 0 and ref_diff > 0) or (pred_diff < 0 and ref_diff < 0):
                concordant += 1
            else:
                discordant += 1

    n_pairs = len(items) * (len(items) - 1) / 2
    n1 = concordant + discordant + tied_pred
    n2 = concordant + discordant + tied_ref

    if n1 == 0 or n2 == 0:
        return 0.0

    tau_b = (concordant - discordant) / ((n1 * n2) ** 0.5)
    return tau_b
