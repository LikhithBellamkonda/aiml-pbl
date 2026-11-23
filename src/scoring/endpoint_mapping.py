"""Trimester evidence classification and endpoint mapping.

This module defines which outcomes are relevant to which trimester, and how to
classify a record's timing from its raw source fields. Timing attribution is
strict: a record is assigned to T1/T2/T3 only when the source explicitly
provides gestational weeks or a trimester label. Untimed records are classified
as ``unknown`` and are never assigned to a trimester by assumption.

Trimester windows (gestational weeks):
- T1: 0-13 (organogenesis - congenital anomalies, organogenesis-related
  developmental toxicity, miscarriage)
- T2: 14-27 (fetal growth abnormalities, developmental abnormalities,
  pregnancy loss)
- T3: 28+ (fetal growth abnormalities, preterm birth, stillbirth, late fetal
  developmental effects, neonatal adaptation effects)

These categories are configurable via ``scoring_config.json`` and are supported
by the endpoint-mapping data below.
"""

from __future__ import annotations

from typing import Any

from .config import ScoringConfig, load_config


#: Gestational week windows for each trimester (inclusive lower, exclusive upper
#: except T3 which is open-ended).
TRIMESTER_WINDOWS: dict[str, tuple[float | None, float | None]] = {
    "first_trimester": (0.0, 13.0),
    "second_trimester": (14.0, 27.0),
    "third_trimester": (28.0, None),
    "entire_pregnancy": (None, None),
    "unknown": (None, None),
}

#: Display labels for each trimester key.
TRIMESTER_LABELS: dict[str, str] = {
    "first_trimester": "First Trimester (T1)",
    "second_trimester": "Second Trimester (T2)",
    "third_trimester": "Third Trimester (T3)",
    "entire_pregnancy": "Entire Pregnancy",
    "unknown": "Timing Not Recorded",
}

#: Fetal outcomes relevant to each trimester. An outcome may be relevant to
#: multiple trimesters; relevance weights live in the scoring config.
FETUS_OUTCOMES_BY_TRIMESTER: dict[str, list[str]] = {
    "first_trimester": [
        "congenital_anomaly",
        "fetal_death",
        "miscarriage",
        "other_pregnancy_outcome",
    ],
    "second_trimester": [
        "fetal_growth_restriction",
        "fetal_growth_outcome",
        "fetal_death",
        "preterm_birth",
        "stillbirth",
        "other_pregnancy_outcome",
    ],
    "third_trimester": [
        "fetal_growth_restriction",
        "fetal_growth_outcome",
        "preterm_birth",
        "stillbirth",
        "fetal_death",
        "other_pregnancy_outcome",
    ],
}

#: Maternal outcomes (not trimester-specific).
MATERNAL_OUTCOMES: list[str] = ["maternal_adverse_reaction"]

#: All fetal outcomes across trimesters.
ALL_FETAL_OUTCOMES: list[str] = sorted(
    {outcome for outcomes in FETUS_OUTCOMES_BY_TRIMESTER.values() for outcome in outcomes}
)

#: Nonclinical evidence sources (kept in a separate tier).
NONCLINICAL_SOURCES: list[str] = ["reprotox_kg", "epa_toxrefdb"]


def load_endpoint_mapping(config: ScoringConfig | None = None) -> dict[str, Any]:
    """Return the endpoint-mapping structure as a plain dict.

    The structure groups outcomes by trimester and lists maternal outcomes.
    Relevance weights are read from the scoring config at scoring time.
    """
    if config is None:
        config = load_config()
    return {
        "trimester_windows": TRIMESTER_WINDOWS,
        "trimester_labels": TRIMESTER_LABELS,
        "fetus_outcomes_by_trimester": FETUS_OUTCOMES_BY_TRIMESTER,
        "all_fetal_outcomes": ALL_FETAL_OUTCOMES,
        "maternal_outcomes": MATERNAL_OUTCOMES,
        "nonclinical_sources": NONCLINICAL_SOURCES,
        "endpoint_relevance": config.endpoint_relevance,
        "source_reliability": config.source_reliability,
        "timing_relevance": config.timing_relevance,
    }


def classify_timing(
    gestational_weeks: Any = None,
    trimester_label: Any = None,
) -> tuple[str, str]:
    """Classify a record's timing into T1/T2/T3/entire_pregnancy/unknown.

    Returns a tuple of ``(trimester, timing_class)`` where ``timing_class`` is
    one of:
    - ``gestational_weeks``: timing derived from numeric gestational weeks
    - ``trimester_label``: timing derived from a source trimester label
    - ``entire_pregnancy``: source explicitly states "entire pregnancy"
    - ``unknown``: no timing information available

    Rules (applied in order):
    1. If ``gestational_weeks`` is a valid number, map to T1 (0-13), T2 (14-27),
       or T3 (28+).
    2. Else if ``trimester_label`` is a recognizable trimester name, use it.
    3. Else return ``unknown`` (never infer a trimester from an untimed record).

    Args:
        gestational_weeks: Numeric gestational weeks, or None/NaN.
        trimester_label: Free-text trimester label from the source, or None.

    Returns:
        Tuple of (trimester, timing_class). Trimester is one of:
        ``first_trimester``, ``second_trimester``, ``third_trimester``,
        ``entire_pregnancy``, ``unknown``.
    """
    # 1. Exact gestational weeks (highest precision).
    if gestational_weeks is not None:
        try:
            weeks = float(gestational_weeks)
        except (TypeError, ValueError):
            weeks = None
        if weeks is not None and weeks >= 0:
            if weeks <= 13.0:
                return "first_trimester", "gestational_weeks"
            if weeks <= 27.0:
                return "second_trimester", "gestational_weeks"
            return "third_trimester", "gestational_weeks"

    # 2. Trimester label from the source.
    if trimester_label is not None:
        text = str(trimester_label).strip().lower().replace("-", "_").replace(" ", "_")
        if text in ("first_trimester", "t1"):
            return "first_trimester", "trimester_label"
        if text in ("second_trimester", "t2"):
            return "second_trimester", "trimester_label"
        if text in ("third_trimester", "t3"):
            return "third_trimester", "trimester_label"
        if text in ("entire_pregnancy", "entire pregnancy", "all trimesters", "all"):
            return "entire_pregnancy", "entire_pregnancy"
        if text in ("unknown", "not recorded", "not specified", "none", ""):
            return "unknown", "unknown"

    return "unknown", "unknown"


def classify_outcome_group(outcome: str) -> str:
    """Classify an outcome as ``fetal``, ``maternal``, or ``other``.

    Args:
        outcome: Outcome category string from the source.

    Returns:
        ``fetal``, ``maternal``, or ``other``.
    """
    from .config import load_config

    config = load_config()
    fetal = set(config.outcome_groups.get("fetal", []))
    maternal = set(config.outcome_groups.get("maternal", []))
    if outcome in maternal:
        return "maternal"
    if outcome in fetal:
        return "fetal"
    return "other"


def endpoint_relevance(outcome: str, trimester: str, config: ScoringConfig | None = None) -> float:
    """Return the relevance of an outcome to a trimester (0.0-1.0).

    Reads from the scoring config. Outcomes not listed for a trimester have
    relevance 0.0.
    """
    if config is None:
        config = load_config()
    return config.endpoint_relevance_score(outcome, trimester)


def source_reliability(source: str, config: ScoringConfig | None = None) -> float:
    """Return the reliability weight of an evidence source (0.0-1.0)."""
    if config is None:
        config = load_config()
    return config.source_reliability_score(source)


def timing_relevance(timing_class: str, config: ScoringConfig | None = None) -> float:
    """Return the timing relevance weight (0.0-1.0)."""
    if config is None:
        config = load_config()
    return config.timing_relevance_score(timing_class)


def bounded_volume_factor(record_count: int, observed_total: float | None, config: ScoringConfig | None = None) -> float:
    """Return a bounded, log-scaled evidence-volume factor (0.0-1.0)."""
    if config is None:
        config = load_config()
    return config.volume_factor(record_count, observed_total)
