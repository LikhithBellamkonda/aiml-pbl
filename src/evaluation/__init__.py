"""Evaluation module for trimester-specific metrics, evidence quality,
and conflict-aware fusion."""

from .trimester_metrics import (
    evaluate_confidence_vs_evidence,
    evaluate_trimester_metrics,
    expected_calibration_error,
    detect_evidence_conflict,
    brier_score,
    precision_at_k,
)
from .evidence_quality import (
    compute_evidence_quality_score,
    compute_batch_evidence_quality,
    load_source_records_from_data_dir,
)
from .conflict_detection import (
    detect_modality_conflict,
    fuse_with_conflict_awareness,
)

__all__ = [
    "evaluate_trimester_metrics",
    "evaluate_confidence_vs_evidence",
    "expected_calibration_error",
    "detect_evidence_conflict",
    "brier_score",
    "precision_at_k",
    "compute_evidence_quality_score",
    "compute_batch_evidence_quality",
    "load_source_records_from_data_dir",
    "detect_modality_conflict",
    "fuse_with_conflict_awareness",
]
