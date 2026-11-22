"""Conflict-aware evidence detection for multimodal fusion.

Detects when different evidence modalities disagree (e.g., molecular evidence
suggests high risk while literature suggests low risk). On conflicting samples,
the model should lower confidence rather than averaging contradictory signals.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def detect_modality_conflict(
    modality_scores: dict[str, np.ndarray],
    conflict_threshold: float = 0.30,
    high_conflict_threshold: float = 0.50,
) -> dict[str, Any]:
    """Detect conflicts between evidence modalities.

    Computes pairwise disagreement between modalities and flags samples where
    any pair exceeds the conflict threshold. Also computes an evidence-conflict
    score that can be used to down-weight predictions.

    Args:
        modality_scores: Dict mapping modality name to array of scores in [0, 1].
            Expected modalities: "molecular", "clinical", "literature", "adr".
        conflict_threshold: Absolute difference above which two modalities
            are considered conflicting (default 0.30).
        high_conflict_threshold: Difference above which conflict is severe
            (default 0.50).

    Returns:
        Dict with per-sample conflict flags, conflict scores, and aggregate
        statistics. The "confidence_penalty" array can be multiplied by the
        model's raw confidence to reduce confidence on conflicting samples.
    """
    modalities = list(modality_scores.keys())
    n_samples = len(next(iter(modality_scores.values())))

    if n_samples == 0:
        return {
            "n_samples": 0,
            "conflict_count": 0,
            "conflict_rate": 0.0,
            "high_conflict_count": 0,
            "high_conflict_rate": 0.0,
            "mean_disagreement": 0.0,
            "max_disagreement": 0.0,
            "confidence_penalty": np.array([]),
            "pairwise_disagreement": {},
            "interpretation": "No samples to evaluate.",
        }

    conflict_flags = np.zeros(n_samples, dtype=bool)
    high_conflict_flags = np.zeros(n_samples, dtype=bool)
    max_disagreement = np.zeros(n_samples, dtype=float)
    pairwise: dict[str, float] = {}

    for i in range(len(modalities)):
        for j in range(i + 1, len(modalities)):
            m1, m2 = modalities[i], modalities[j]
            diff = np.abs(modality_scores[m1] - modality_scores[m2])
            pair_key = f"{m1}_vs_{m2}"
            pairwise[pair_key] = {
                "mean_disagreement": float(diff.mean()),
                "max_disagreement": float(diff.max()),
                "conflict_count": int((diff > conflict_threshold).sum()),
                "conflict_rate": float((diff > conflict_threshold).mean()),
            }
            conflict_flags = conflict_flags | (diff > conflict_threshold)
            high_conflict_flags = high_conflict_flags | (diff > high_conflict_threshold)
            max_disagreement = np.maximum(max_disagreement, diff)

    # Confidence penalty: scale from 1.0 (no conflict) to 0.3 (severe conflict)
    # Penalty = 1 - (max_disagreement / max_possible_disagreement) * penalty_strength
    penalty_strength = 0.7
    confidence_penalty = np.clip(
        1.0 - (max_disagreement * penalty_strength), 0.3, 1.0
    )

    return {
        "n_samples": int(n_samples),
        "conflict_count": int(conflict_flags.sum()),
        "conflict_rate": float(conflict_flags.mean()),
        "high_conflict_count": int(high_conflict_flags.sum()),
        "high_conflict_rate": float(high_conflict_flags.mean()),
        "mean_disagreement": float(max_disagreement.mean()),
        "max_disagreement": float(max_disagreement.max()),
        "confidence_penalty": confidence_penalty.tolist(),
        "pairwise_disagreement": pairwise,
        "conflict_threshold": conflict_threshold,
        "high_conflict_threshold": high_conflict_threshold,
        "interpretation": (
            "Conflict rate indicates how often evidence modalities disagree. "
            "High conflict suggests contradictory evidence -- predictions on "
            "those samples should be treated with lower confidence. "
            "Confidence penalty can be applied as: adjusted_confidence = "
            "raw_confidence * confidence_penalty."
        ),
    }


def fuse_with_conflict_awareness(
    modality_scores: dict[str, float],
    modality_weights: dict[str, float] | None = None,
    conflict_threshold: float = 0.30,
) -> dict[str, Any]:
    """Fuse modality scores with conflict awareness for a single sample.

    Instead of simple averaging, this detects when modalities disagree and
    applies a conflict-aware weighting that reduces the influence of
    outlier modalities.

    Args:
        modality_scores: Dict mapping modality name to score in [0, 1].
        modality_weights: Optional weights for each modality (default equal).
        conflict_threshold: Disagreement threshold for conflict detection.

    Returns:
        Dict with fused score, conflict flag, conflict score, and per-modality
        adjusted weights.
    """
    modalities = list(modality_scores.keys())
    scores = np.array([modality_scores[m] for m in modalities])

    if modality_weights is None:
        weights = np.ones(len(modalities)) / len(modalities)
    else:
        weights = np.array([modality_weights.get(m, 1.0) for m in modalities])
        weights = weights / weights.sum()

    # Compute pairwise disagreements
    disagreements = []
    for i in range(len(modalities)):
        for j in range(i + 1, len(modalities)):
            disagreements.append(abs(scores[i] - scores[j]))

    max_disagreement = max(disagreements) if disagreements else 0.0
    is_conflict = max_disagreement > conflict_threshold

    # Conflict-aware weighting: reduce weight of outlier modalities
    # A modality is an outlier if it disagrees with the median by > threshold
    median_score = float(np.median(scores))
    adjusted_weights = weights.copy()
    for i, m in enumerate(modalities):
        deviation = abs(scores[i] - median_score)
        if deviation > conflict_threshold:
            # Reduce weight of outlier modality
            adjusted_weights[i] *= 0.5

    # Renormalize
    if adjusted_weights.sum() > 0:
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
    else:
        adjusted_weights = np.ones(len(modalities)) / len(modalities)

    # Weighted fusion
    fused_score = float(np.sum(scores * adjusted_weights))

    # Conflict-aware confidence: lower when modalities disagree
    conflict_penalty = max(0.3, 1.0 - max_disagreement * 0.7)
    adjusted_confidence = fused_score * conflict_penalty

    return {
        "fused_score": round(fused_score, 4),
        "adjusted_confidence": round(adjusted_confidence, 4),
        "is_conflict": bool(is_conflict),
        "max_disagreement": round(max_disagreement, 4),
        "conflict_penalty": round(conflict_penalty, 4),
        "modality_scores": {m: round(s, 4) for m, s in modality_scores.items()},
        "adjusted_weights": {
            m: round(w, 4) for m, w in zip(modalities, adjusted_weights)
        },
        "interpretation": (
            "If is_conflict is True, evidence modalities disagree and the "
            "adjusted_confidence is penalized. The fused score still reflects "
            "the weighted average, but confidence is reduced."
        ),
    }
