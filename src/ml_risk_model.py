"""Observed-evidence summary for the pregnancy dashboard.

This module deliberately produces **no** risk score, **no** confidence value
and **no** predictive metric. No validated human pregnancy-outcome model
exists in this repository: the acquired sources provide aggregate
spontaneous-report counts, regulatory classifications and untimed
associations, none of which are compatible binary outcome labels.

What this module does provide is a faithful summary of what the acquired
sources actually record for a drug, with unavailable sources reported as
``None`` rather than as zero (absent evidence is not negative evidence).

An earlier revision of this file generated risk scores from normalised
report counts plus a SHA-256 hash modifier and evaluated them against
pseudo-labels derived from the same features. Those values were not model
outputs and have been removed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Pregnancy outcome -> aggregate report-count column in
# data/real_pregnancy_features.csv.
OUTCOME_COUNT_COLUMNS = {
    "fetal_death": "reports_fetal_death",
    "preterm_birth": "reports_preterm_birth",
    "fetal_growth_restriction": "reports_fetal_growth_restriction",
    "congenital_anomaly": "reports_congenital_anomaly",
    "maternal_adverse_reaction": "reports_maternal_adverse_reaction",
}

TRIMESTERS = (
    "first_trimester",
    "second_trimester",
    "third_trimester",
    "entire_pregnancy",
)

NO_MODEL_MESSAGE = (
    "No validated human pregnancy-outcome model is available: the acquired "
    "sources provide aggregate spontaneous-report counts, classifications and "
    "untimed associations, not compatible binary outcome labels."
)


def _safe_int(value: Any) -> int | None:
    """Return an integer count, or None when the source value is unavailable."""
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _load_drug_row(drug_id: str | None) -> pd.Series | None:
    if not drug_id:
        return None
    path = DATA_DIR / "real_pregnancy_features.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if "drug_id" not in frame.columns:
        return None
    matches = frame[frame["drug_id"].astype(str) == str(drug_id)]
    return matches.iloc[0] if not matches.empty else None


def _adr_term_count(drug_id: str | None) -> int | None:
    """Number of distinct adverse-event term records for a drug, or None."""
    if not drug_id:
        return None
    path = DATA_DIR / "real_adr_evidence.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(
        path,
        usecols=lambda column: column in {"drug_id", "adverse_event_term"},
    )
    if "drug_id" not in frame.columns:
        return None
    rows = frame[frame["drug_id"].astype(str) == str(drug_id)]
    if rows.empty:
        return None
    if "adverse_event_term" in rows.columns:
        return int(rows["adverse_event_term"].dropna().nunique())
    return int(len(rows))


def _faers_report_count(drug_id: str | None) -> int | None:
    """Aggregate FAERS report total for a drug, or None when not reported."""
    if not drug_id:
        return None
    path = DATA_DIR / "real_faers_summary.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if "drug_id" not in frame.columns or "faers_report_count" not in frame.columns:
        return None
    rows = frame[frame["drug_id"].astype(str) == str(drug_id)]
    return _safe_int(rows.iloc[0]["faers_report_count"]) if not rows.empty else None


def summarise_drug(drug_id: str | None, drug_name: str | None = None) -> dict[str, Any]:
    """Summarise observed evidence for one drug without producing a prediction.

    ``observed_outcome_reports`` maps each outcome to the aggregate number of
    spontaneous-report terms recorded by the acquired pregnancy source, or to
    ``None`` when that source has no record for the drug. These counts are
    reporting volume, not incidence, severity or risk.
    """
    row = _load_drug_row(drug_id)
    observed = {
        outcome: (_safe_int(row.get(column)) if row is not None else None)
        for outcome, column in OUTCOME_COUNT_COLUMNS.items()
    }
    adr_term_count = _adr_term_count(drug_id)
    faers_report_count = _faers_report_count(drug_id)
    available_sources: list[str] = []
    if any(value is not None for value in observed.values()):
        available_sources.append("pregnancy_outcomes")
    if adr_term_count is not None:
        available_sources.append("adr")
    if faers_report_count is not None:
        available_sources.append("faers")

    return {
        "drug_id": str(drug_id) if drug_id else "",
        "drug_name": drug_name or (str(row.get("name", drug_id)) if row is not None else (drug_id or "")),
        "risk_available": False,
        "risk_score": None,
        "confidence": None,
        "model_used": "none",
        "abstain": True,
        "status": NO_MODEL_MESSAGE,
        "observed_outcome_reports": observed,
        "adr_term_count": adr_term_count,
        "faers_report_count": faers_report_count,
        "sources_with_records": sorted(available_sources),
    }


# Backwards-compatible entry point: callers must treat the result as an
# evidence summary, not a score.
def score_drug(drug_id: str | None, drug_name: str | None = None) -> dict[str, Any]:
    return summarise_drug(drug_id, drug_name)
