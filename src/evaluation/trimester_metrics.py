"""Trimester-specific evaluation metrics for pregnancy drug-safety signals.

Provides per-trimester AUROC/AUPRC, ΔAUROC between trimesters, calibration
error (ECE) per trimester, and Brier scores. These metrics explicitly measure
whether the model actually learns trimester differences or collapses to a
single-stage predictor.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Compute AUROC, returning None if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return None
    # Rank-based AUROC (Mann-Whitney U statistic)
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # Handle ties by averaging ranks
    unique_scores, inverse = np.unique(y_score, return_inverse=True)
    for i in range(len(unique_scores)):
        mask = inverse == i
        ranks[mask] = ranks[mask].mean()
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    auroc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auroc)


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Compute AUPRC using trapezoidal integration of precision-recall curve."""
    if len(np.unique(y_true)) < 2:
        return None
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(tp[-1], 1)
    # Prepend (recall=0, precision=1)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    # Trapezoidal integration (np.trapezoid for NumPy 2.0+, np.trapz for older)
    try:
        auprc = float(np.trapezoid(precision, recall))
    except AttributeError:
        auprc = float(np.trapz(precision, recall))
    return auprc


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10
) -> float | None:
    """Expected Calibration Error (ECE) with uniform binning."""
    if len(np.unique(y_true)) < 2 or len(y_true) == 0:
        return None
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = len(y_true)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if upper == 1.0:
            in_bin = (y_prob >= lower) & (y_prob <= upper)
        else:
            in_bin = (y_prob >= lower) & (y_prob < upper)
        if not in_bin.any():
            continue
        bin_weight = in_bin.mean()
        bin_acc = np.abs(y_true[in_bin].mean() - y_prob[in_bin].mean())
        ece += bin_weight * bin_acc
    return float(ece)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Brier score: mean squared error between probabilities and outcomes."""
    if len(y_true) == 0:
        return None
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def precision_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k: int = 10
) -> float | None:
    """Precision at top-k ranked by score."""
    if len(y_true) == 0 or k <= 0:
        return None
    k = min(k, len(y_true))
    order = np.argsort(-y_score)[:k]
    return float(y_true[order].mean())


def evaluate_trimester_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    trimesters: np.ndarray,
    trimester_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate model performance per trimester.

    Returns per-trimester AUROC, AUPRC, ECE, Brier, Precision@k, and
    ΔAUROC between trimesters to detect whether the model learns
    trimester-specific patterns or collapses to a single-stage predictor.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_score: Predicted probabilities in [0, 1].
        trimesters: Array of trimester strings (e.g., "first_trimester").
        trimester_labels: Optional display labels for trimesters.

    Returns:
        Dict with per-trimester metrics and cross-trimester comparisons.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    trimesters = np.asarray(trimesters)

    if trimester_labels is None:
        trimester_labels = {
            "first_trimester": "First Trimester (T1)",
            "second_trimester": "Second Trimester (T2)",
            "third_trimester": "Third Trimester (T3)",
            "entire_pregnancy": "Entire Pregnancy",
        }

    unique_trimesters = np.unique(trimesters)
    per_trimester: dict[str, Any] = {}

    for t in unique_trimesters:
        mask = trimesters == t
        t_y = y_true[mask]
        t_s = y_score[mask]
        per_trimester[str(t)] = {
            "label": trimester_labels.get(str(t), str(t)),
            "n": int(len(t_y)),
            "positives": int(t_y.sum()),
            "auroc": _safe_auroc(t_y, t_s),
            "auprc": _safe_auprc(t_y, t_s),
            "ece": expected_calibration_error(t_y, t_s),
            "brier": brier_score(t_y, t_s),
            "precision_at_10": precision_at_k(t_y, t_s, k=10),
        }

    # ΔAUROC between trimesters (max - min) to detect trimester discrimination
    aurocs = [
        v["auroc"] for v in per_trimester.values() if v["auroc"] is not None
    ]
    if len(aurocs) >= 2:
        delta_auroc = max(aurocs) - min(aurocs)
    else:
        delta_auroc = None

    # Overall metrics
    overall = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "auroc": _safe_auroc(y_true, y_score),
        "auprc": _safe_auprc(y_true, y_score),
        "ece": expected_calibration_error(y_true, y_score),
        "brier": brier_score(y_true, y_score),
        "precision_at_10": precision_at_k(y_true, y_score, k=10),
    }

    return {
        "per_trimester": per_trimester,
        "overall": overall,
        "delta_auroc": delta_auroc,
        "trimester_count": int(len(unique_trimesters)),
        "interpretation": (
            "ΔAUROC > 0 indicates the model learns trimester-specific patterns. "
            "ΔAUROC ≈ 0 suggests the model collapses to a single-stage predictor "
            "and does not distinguish gestational-stage-dependent risk."
        ),
    }


def evaluate_confidence_vs_evidence(
    confidence: np.ndarray,
    evidence_quality: np.ndarray,
    y_true: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate whether confidence and evidence quality are distinct signals.

    Tests that prediction confidence (model's certainty) and evidence
    reliability (data completeness/quality) are separable concepts -- a
    requirement for evidence-aware prediction.

    Returns correlation between confidence and evidence quality, and
    (if labels provided) calibration of each separately.
    """
    confidence = np.asarray(confidence, dtype=float)
    evidence_quality = np.asarray(evidence_quality, dtype=float)

    # Pearson correlation between confidence and evidence quality
    if len(confidence) > 1 and np.std(confidence) > 0 and np.std(evidence_quality) > 0:
        corr = float(np.corrcoef(confidence, evidence_quality)[0, 1])
    else:
        corr = None

    result: dict[str, Any] = {
        "n": int(len(confidence)),
        "confidence_mean": float(np.mean(confidence)),
        "confidence_std": float(np.std(confidence)),
        "evidence_quality_mean": float(np.mean(evidence_quality)),
        "evidence_quality_std": float(np.std(evidence_quality)),
        "confidence_evidence_correlation": corr,
        "distinct_signals": (
            corr is None or abs(corr) < 0.7
        ),  # Distinct if not strongly correlated
        "interpretation": (
            "Confidence and evidence quality should be distinct signals. "
            "High correlation suggests the model is simply more confident "
            "when more data is available, rather than being well-calibrated."
        ),
    }

    if y_true is not None:
        y_true = np.asarray(y_true, dtype=int)
        result["confidence_ece"] = expected_calibration_error(y_true, confidence)
        result["evidence_quality_ece"] = expected_calibration_error(
            y_true, evidence_quality
        )
        result["confidence_brier"] = brier_score(y_true, confidence)
        result["evidence_quality_brier"] = brier_score(y_true, evidence_quality)

    return result


def detect_evidence_conflict(
    modality_scores: dict[str, np.ndarray],
    conflict_threshold: float = 0.3,
) -> dict[str, Any]:
    """Detect conflicts between evidence modalities.

    If molecular evidence and literature evidence disagree strongly
    (e.g., one says high risk, other says low risk), the model should
    lower confidence rather than averaging the signals.

    Args:
        modality_scores: Dict mapping modality name to array of scores [0,1].
        conflict_threshold: Absolute difference threshold for conflict.

    Returns:
        Dict with per-sample conflict flags and aggregate conflict rate.
    """
    modalities = list(modality_scores.keys())
    n_samples = len(next(iter(modality_scores.values())))

    conflict_flags = np.zeros(n_samples, dtype=bool)
    max_disagreement = np.zeros(n_samples, dtype=float)

    for i in range(len(modalities)):
        for j in range(i + 1, len(modalities)):
            diff = np.abs(
                modality_scores[modalities[i]] - modality_scores[modalities[j]]
            )
            conflict_flags = conflict_flags | (diff > conflict_threshold)
            max_disagreement = np.maximum(max_disagreement, diff)

    return {
        "n_samples": int(n_samples),
        "conflict_count": int(conflict_flags.sum()),
        "conflict_rate": float(conflict_flags.mean()),
        "max_disagreement_mean": float(max_disagreement.mean()),
        "max_disagreement_max": float(max_disagreement.max()),
        "conflict_threshold": conflict_threshold,
        "interpretation": (
            "High conflict rate indicates contradictory evidence across "
            "modalities. Predictions on conflicting samples should be "
            "treated with lower confidence."
        ),
    }
