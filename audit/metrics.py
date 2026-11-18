from __future__ import annotations

import math
import random
from statistics import mean, median
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


FPR_TARGETS_DEFAULT: Tuple[float, ...] = (0.001, 0.005, 0.01, 0.05)


def _to_set(values: Optional[Iterable[str]]) -> Set[str]:
    if values is None:
        return set()
    return {str(v) for v in values}


def set_jaccard(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    true_set, pred_set = _to_set(y_true), _to_set(y_pred)
    union = true_set | pred_set
    if not union:
        return 1.0
    return len(true_set & pred_set) / len(union)


def set_precision(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    true_set, pred_set = _to_set(y_true), _to_set(y_pred)
    if not pred_set:
        return 0.0
    return len(true_set & pred_set) / len(pred_set)


def set_recall(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    true_set, pred_set = _to_set(y_true), _to_set(y_pred)
    if not true_set:
        return 1.0 if not pred_set else 0.0
    return len(true_set & pred_set) / len(true_set)


def set_f1(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    p = set_precision(y_true, y_pred)
    r = set_recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def set_binary_mse(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    true_set, pred_set = _to_set(y_true), _to_set(y_pred)
    universe = true_set | pred_set
    if not universe:
        return 0.0
    # Equivalent to the mean squared error on multi-hot vectors over the union support.
    return len(true_set ^ pred_set) / len(universe)


def build_idf_weights_from_records(
    records: Sequence[Mapping[str, object]],
    true_key: str = "true_codes",
) -> Dict[str, float]:
    """Build visit-level IDF weights from observed true code sets.

    Weights are PHI-safe aggregated statistics only (no identifiers).
    """
    doc_freq: Dict[str, int] = {}
    n_docs = 0
    for rec in records:
        if not bool(rec.get("is_masked", True)):
            continue
        codes = _to_set(rec.get(true_key))
        if not codes:
            continue
        n_docs += 1
        for code in codes:
            doc_freq[code] = doc_freq.get(code, 0) + 1
    if n_docs == 0:
        return {}
    weights: Dict[str, float] = {}
    for code, df in doc_freq.items():
        # Smooth to keep finite weights and avoid exploding values on tiny data.
        weights[code] = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0
    return weights


def weighted_set_f1(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    code_weights: Mapping[str, float],
    default_weight: float = 1.0,
) -> float:
    true_set, pred_set = _to_set(y_true), _to_set(y_pred)
    if not true_set and not pred_set:
        return 1.0

    def w(code: str) -> float:
        try:
            val = float(code_weights.get(code, default_weight))
        except (TypeError, ValueError):
            return float(default_weight)
        return val if val > 0 else float(default_weight)

    inter = true_set & pred_set
    w_inter = sum(w(c) for c in inter)
    w_true = sum(w(c) for c in true_set)
    w_pred = sum(w(c) for c in pred_set)

    precision = 0.0 if w_pred <= 0 else (w_inter / w_pred)
    recall = 1.0 if w_true <= 0 and w_pred <= 0 else (0.0 if w_true <= 0 else (w_inter / w_true))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def aggregate_patient_scores(scores: Sequence[float], agg: str) -> float:
    if not scores:
        return 0.0
    if agg == "min":
        return min(scores)
    if agg == "mean":
        return float(mean(scores))
    if agg == "median":
        return float(median(scores))
    raise ValueError(f"Unknown aggregation: {agg}")


def roc_auc_binary(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(labels) == 0 or len(scores) == 0 or len(labels) != len(scores):
        return float("nan")
    y = [int(v) for v in labels]
    s = [float(v) for v in scores]
    pos = sum(1 for v in y if v == 1)
    neg = sum(1 for v in y if v == 0)
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(s)), key=lambda i: s[i])
    sorted_scores = [s[i] for i in order]
    ranks = [0.0] * len(sorted_scores)
    n = len(sorted_scores)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    full_ranks = [0.0] * n
    for pos_idx, original_idx in enumerate(order):
        full_ranks[original_idx] = ranks[pos_idx]
    sum_ranks_pos = sum(full_ranks[i] for i, cls in enumerate(y) if cls == 1)
    auc = (sum_ranks_pos - (pos * (pos + 1) / 2.0)) / (pos * neg)
    return float(auc)


def bootstrap_auc_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    if len(labels) == 0 or len(scores) == 0 or len(labels) != len(scores):
        return {
            "estimate": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "confidence": float(confidence),
            "n_bootstrap": 0,
        }
    pos_scores = [float(score) for label, score in zip(labels, scores) if int(label) == 1]
    neg_scores = [float(score) for label, score in zip(labels, scores) if int(label) == 0]
    estimate = float(roc_auc_binary(labels, scores))
    if not pos_scores or not neg_scores:
        return {
            "estimate": estimate,
            "lower": float("nan"),
            "upper": float("nan"),
            "confidence": float(confidence),
            "n_bootstrap": 0,
        }

    n_boot = max(int(n_bootstrap), 0)
    conf = min(max(float(confidence), 1e-6), 1.0 - 1e-6)
    rng = random.Random(int(seed))
    aucs: List[float] = []
    for _ in range(n_boot):
        boot_pos = [pos_scores[rng.randrange(len(pos_scores))] for _ in range(len(pos_scores))]
        boot_neg = [neg_scores[rng.randrange(len(neg_scores))] for _ in range(len(neg_scores))]
        auc = roc_auc_binary(
            [1] * len(boot_pos) + [0] * len(boot_neg),
            boot_pos + boot_neg,
        )
        if not math.isnan(auc):
            aucs.append(float(auc))
    if not aucs:
        return {
            "estimate": estimate,
            "lower": float("nan"),
            "upper": float("nan"),
            "confidence": conf,
            "n_bootstrap": 0,
        }

    alpha = 0.5 * (1.0 - conf)
    return {
        "estimate": estimate,
        "lower": float(_quantile(aucs, alpha)),
        "upper": float(_quantile(aucs, 1.0 - alpha)),
        "confidence": conf,
        "n_bootstrap": len(aucs),
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    q = min(max(float(q), 0.0), 1.0)
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    pos = q * (len(arr) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    w = pos - lo
    return arr[lo] * (1.0 - w) + arr[hi] * w


def tpr_at_fpr_from_neg_quantiles(
    member_scores: Sequence[float],
    nonmember_scores: Sequence[float],
    fpr_targets: Sequence[float] = FPR_TARGETS_DEFAULT,
) -> List[Dict[str, float]]:
    members = [float(x) for x in member_scores]
    nonmembers = [float(x) for x in nonmember_scores]
    if len(members) == 0 or len(nonmembers) == 0:
        return []
    rows: List[Dict[str, float]] = []
    for fpr in fpr_targets:
        fpr = float(fpr)
        q = min(max(1.0 - fpr, 0.0), 1.0)
        threshold = _quantile(nonmembers, q)
        empirical_fpr = sum(1 for x in nonmembers if x >= threshold) / len(nonmembers)
        tpr = sum(1 for x in members if x >= threshold) / len(members)
        rows.append(
            {
                "target_fpr": fpr,
                "threshold": threshold,
                "tpr": tpr,
                "empirical_fpr": empirical_fpr,
            }
        )
    return rows


def decode_with_threshold_topk(
    code_to_prob: Mapping[str, float],
    threshold: float,
    topk: Optional[int] = None,
    temperature: float = 1.0,
    ensure_non_empty: bool = True,
) -> Set[str]:
    if not code_to_prob:
        return set()
    temp = float(temperature) if float(temperature) > 0 else 1.0
    transformed: List[Tuple[str, float]] = []
    for code, prob in code_to_prob.items():
        p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
        if temp != 1.0:
            logit = math.log(p / (1.0 - p))
            p = 1.0 / (1.0 + math.exp(-(logit / temp)))
        transformed.append((str(code), p))
    ranked = sorted(transformed, key=lambda kv: kv[1], reverse=True)
    selected = [code for code, prob in ranked if float(prob) >= threshold]
    if topk is not None and topk > 0:
        selected = selected[:topk]
    if ensure_non_empty and not selected:
        selected = [ranked[0][0]]
    return set(selected)


def summarize_threshold_sweep(
    records: Sequence[Mapping[str, object]],
    thresholds: Sequence[float],
    topk: Optional[int] = None,
    diagnostic_k: Optional[int] = None,
    temperature: float = 1.0,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        pred_sizes: List[int] = []
        jaccards: List[float] = []
        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []
        jaccards_k: List[float] = []
        f1s_k: List[float] = []
        for rec in records:
            if not bool(rec.get("is_masked", True)):
                continue
            y_true = _to_set(rec.get("true_codes"))
            probs = rec.get("pred_probs")
            if isinstance(probs, Mapping):
                pred_set = decode_with_threshold_topk(
                    probs,
                    float(threshold),
                    topk=topk,
                    temperature=temperature,
                )
            else:
                pred_set = _to_set(rec.get("pred_codes"))
                if topk is not None and topk > 0 and len(pred_set) > topk:
                    # deterministic truncation if only hard predictions are available
                    pred_set = set(sorted(pred_set)[:topk])
            pred_sizes.append(len(pred_set))
            jaccards.append(set_jaccard(y_true, pred_set))
            precisions.append(set_precision(y_true, pred_set))
            recalls.append(set_recall(y_true, pred_set))
            f1s.append(set_f1(y_true, pred_set))
            if diagnostic_k is not None and diagnostic_k > 0:
                if isinstance(probs, Mapping):
                    pred_set_k = decode_with_threshold_topk(
                        probs,
                        float(threshold),
                        topk=diagnostic_k,
                        temperature=temperature,
                    )
                else:
                    pred_set_k = set(sorted(pred_set)[:diagnostic_k])
                jaccards_k.append(set_jaccard(y_true, pred_set_k))
                f1s_k.append(set_f1(y_true, pred_set_k))
        n_visits = len(pred_sizes)
        if n_visits == 0:
            rows.append(
                {
                    "threshold": float(threshold),
                    "n_visits": 0,
                    "avg_pred_size": 0.0,
                    "pred_size_mean": 0.0,
                    "pred_size_p50": 0.0,
                    "pred_size_p95": 0.0,
                    "jaccard": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "jaccard_at_k": 0.0,
                    "f1_at_k": 0.0,
                }
            )
            continue
        size_mean = float(mean(pred_sizes))
        size_p50 = float(_quantile(pred_sizes, 0.50))
        size_p95 = float(_quantile(pred_sizes, 0.95))
        rows.append(
            {
                "threshold": float(threshold),
                "n_visits": n_visits,
                "avg_pred_size": size_mean,
                "pred_size_mean": size_mean,
                "pred_size_p50": size_p50,
                "pred_size_p95": size_p95,
                "jaccard": float(mean(jaccards)),
                "precision": float(mean(precisions)),
                "recall": float(mean(recalls)),
                "f1": float(mean(f1s)),
                "jaccard_at_k": float(mean(jaccards_k)) if jaccards_k else 0.0,
                "f1_at_k": float(mean(f1s_k)) if f1s_k else 0.0,
            }
        )
    return rows


def calibrate_decoding(
    records: Sequence[Mapping[str, object]],
    thresholds: Sequence[float],
    topk_values: Sequence[int],
    temperature_values: Sequence[float],
    objective: str,
    diagnostic_k: Optional[int] = None,
) -> Dict[str, object]:
    metric_key_map = {
        "f1": "f1",
        "jaccard": "jaccard",
        "f1_at_k": "f1_at_k",
        "jaccard_at_k": "jaccard_at_k",
    }
    if objective not in metric_key_map:
        raise ValueError(f"Unsupported calibration objective: {objective}")
    metric_key = metric_key_map[objective]

    best = None
    for temp in temperature_values:
        for threshold in thresholds:
            for topk in topk_values:
                rows = summarize_threshold_sweep(
                    records=records,
                    thresholds=[float(threshold)],
                    topk=(int(topk) if int(topk) > 0 else None),
                    diagnostic_k=diagnostic_k,
                    temperature=float(temp),
                )
                row = rows[0] if rows else {}
                metric = float(row.get(metric_key, 0.0))
                pred_size = float(row.get("pred_size_mean", 0.0))
                candidate = {
                    "temperature": float(temp),
                    "threshold": float(threshold),
                    "topk_cap": int(topk) if int(topk) > 0 else None,
                    "objective": objective,
                    "objective_value": metric,
                    "pred_size_mean": pred_size,
                    "diagnostics": row,
                }
                if best is None:
                    best = candidate
                    continue
                best_metric = float(best["objective_value"])
                better_metric = metric > best_metric
                tie_break = (math.isclose(metric, best_metric) and pred_size < float(best["pred_size_mean"]))
                if better_metric or tie_break:
                    best = candidate

    return best if best is not None else {
        "temperature": 1.0,
        "threshold": 0.5,
        "topk_cap": None,
        "objective": objective,
        "objective_value": 0.0,
        "pred_size_mean": 0.0,
        "diagnostics": {},
    }
