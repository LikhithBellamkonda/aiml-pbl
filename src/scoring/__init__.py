"""MEPRank evidence-based trimester-wise prioritization scoring modules.

This package implements an evidence-priority index (0-100) for maternal and
fetal pregnancy safety, computed per trimester from actual available evidence.
No machine-learning predictions, pseudo-labels, or synthetic scores are used.
"""

from .config import ScoringConfig, load_config
from .endpoint_mapping import (
    TRIMESTER_WINDOWS,
    FETUS_OUTCOMES_BY_TRIMESTER,
    MATERNAL_OUTCOMES,
    NONCLINICAL_SOURCES,
    classify_timing,
    classify_outcome_group,
    endpoint_relevance,
    source_reliability,
    timing_relevance,
    bounded_volume_factor,
)
from .maternal_engine import MaternalEvidenceEngine
from .fetal_engine import FetalEvidenceEngine
from .overall import combine, overall_for_trimester
from .prioritization_engine import PrioritizationEngine, PrioritizationResult

__all__ = [
    "ScoringConfig",
    "load_config",
    "TRIMESTER_WINDOWS",
    "FETUS_OUTCOMES_BY_TRIMESTER",
    "MATERNAL_OUTCOMES",
    "NONCLINICAL_SOURCES",
    "classify_timing",
    "classify_outcome_group",
    "endpoint_relevance",
    "source_reliability",
    "timing_relevance",
    "bounded_volume_factor",
    "combine",
    "overall_for_trimester",
    "MaternalEvidenceEngine",
    "FetalEvidenceEngine",
    "PrioritizationEngine",
    "PrioritizationResult",
]
