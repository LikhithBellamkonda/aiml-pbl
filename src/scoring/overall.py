"""Overall pregnancy safety prioritization.

Combines maternal and fetal evidence-priority indices into an overall
trimester-specific index using the transparent convention:

    Overall(T) = 0.5 * Maternal(T) + 0.5 * Fetal(T)

If either component is unavailable (None), the overall score is also
unavailable. Missing values are never replaced with zero.
"""

from __future__ import annotations

from typing import Any


def combine(
    maternal: float | None,
    fetal: float | None,
    weight_maternal: float = 0.5,
    weight_fetal: float = 0.5,
) -> float | None:
    """Combine maternal and fetal indices into an overall index.

    Args:
        maternal: Maternal evidence-priority index (0-100) or None.
        fetal: Fetal evidence-priority index (0-100) or None.
        weight_maternal: Weight for maternal component (default 0.5).
        weight_fetal: Weight for fetal component (default 0.5).

    Returns:
        Combined index (0-100) or None when either component is unavailable.
    """
    if maternal is None or fetal is None:
        return None
    total_weight = weight_maternal + weight_fetal
    if total_weight <= 0:
        return None
    combined = (weight_maternal * maternal + weight_fetal * fetal) / total_weight
    return round(min(100.0, max(0.0, combined)), 1)


def overall_for_trimester(
    maternal_result: dict[str, Any],
    fetal_result: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an overall prioritization result for one trimester.

    Args:
        maternal_result: Output of ``MaternalEvidenceEngine.score_trimester``.
        fetal_result: Output of ``FetalEvidenceEngine.score_trimester``.
        config: Optional dict with ``overall_weights``.

    Returns:
        Dict with the overall score, availability, and component scores.
    """
    weights = {"maternal": 0.5, "fetal": 0.5}
    if config and isinstance(config, dict):
        weights.update(config.get("overall_weights", {}))

    maternal_score = maternal_result.get("score")
    fetal_score = fetal_result.get("score")
    overall = combine(maternal_score, fetal_score, weights["maternal"], weights["fetal"])

    return {
        "trimester": maternal_result.get("trimester", fetal_result.get("trimester")),
        "trimester_label": maternal_result.get("trimester_label")
        or fetal_result.get("trimester_label"),
        "score": overall,
        "score_available": overall is not None,
        "maternal_score": maternal_score,
        "maternal_available": maternal_result.get("score_available", False),
        "fetal_score": fetal_score,
        "fetal_available": fetal_result.get("score_available", False),
        "eligible_record_count": (
            maternal_result.get("eligible_record_count", 0)
            + fetal_result.get("eligible_record_count", 0)
        ),
        "total_human_records": (
            maternal_result.get("total_human_records", 0)
            + fetal_result.get("total_human_records", 0)
        ),
        "total_nonclinical_records": fetal_result.get("total_nonclinical_records", 0),
        "has_nonclinical_evidence": fetal_result.get("has_nonclinical_evidence", False),
        "method": "evidence_weighted_prioritization",
        "formula": "Overall(T) = 0.5 * Maternal(T) + 0.5 * Fetal(T)",
        "note": "If either component is unavailable, the overall score is unavailable.",
    }
