"""Evidence-summary engine for pregnancy drug-safety queries.

This module reports what the acquired sources actually record for a drug and
how much of the tracked evidence space is covered. It produces **no risk
score, no confidence and no model prediction**: the locally available sources
contain aggregate spontaneous-report counts, regulatory classifications and
untimed associations, not compatible binary human pregnancy-outcome labels.

Earlier revisions of this module derived a "risk" number from a SHA-256 hash of
the drug identifier plus hard-coded trimester constants, and scored those same
numbers against pseudo-labels built with more hash noise. That was neither a
model nor an evaluation, so it has been removed. Every public function below
now abstains explicitly and reports only observed evidence and coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation import (
    compute_evidence_quality_score,
    load_source_records_from_data_dir,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "src" / "models"
METRICS_PATH = DATA_DIR / "ml_risk_metrics.json"
MODEL_METADATA_PATH = MODEL_DIR / "ml_risk_model_metadata.json"

TRIMESTERS = (
    "first_trimester",
    "second_trimester",
    "third_trimester",
    "entire_pregnancy",
)
TRIMESTER_LABELS = {
    "first_trimester": "First Trimester (T1)",
    "second_trimester": "Second Trimester (T2)",
    "third_trimester": "Third Trimester (T3)",
    "entire_pregnancy": "Entire Pregnancy",
}
FETUS_OUTCOMES = (
    "fetal_death",
    "congenital_anomaly",
    "preterm_birth",
    "fetal_growth_restriction",
)
MATERNAL_OUTCOMES = ("maternal_adverse_reaction",)
ALL_OUTCOMES = FETUS_OUTCOMES + MATERNAL_OUTCOMES


NO_MODEL_MESSAGE = (
    "No validated human pregnancy-outcome model is available: the acquired "
    "sources provide aggregate spontaneous-report counts, classifications and "
    "untimed associations, not compatible binary outcome labels."
)

# Pregnancy outcome -> aggregate report-count column in
# data/real_pregnancy_features.csv (used by the observed-evidence summary).
OUTCOME_COUNT_COLUMNS = {
    "fetal_death": "reports_fetal_death",
    "preterm_birth": "reports_preterm_birth",
    "fetal_growth_restriction": "reports_fetal_growth_restriction",
    "congenital_anomaly": "reports_congenital_anomaly",
    "maternal_adverse_reaction": "reports_maternal_adverse_reaction",
}


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
        path, usecols=lambda column: column in {"drug_id", "adverse_event_term"}
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
        "drug_name": drug_name
        or (str(row.get("name", drug_id)) if row is not None else (drug_id or "")),
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


class PregnancyEvidenceEngine:
    """Evidence-quality and availability engine; it never scores risk.

    Every method returns ``risk_available=False`` together with the recorded
    evidence counts, the availability mask and the coverage score. This lets a
    caller state *why* no prediction is possible instead of inventing one.
    """

    def __init__(self) -> None:
        self._source_records: dict[str, pd.DataFrame] | None = None

    # ------------------------------------------------------------------ #
    # Evidence quality / availability
    # ------------------------------------------------------------------ #
    def _sources(self) -> dict[str, pd.DataFrame]:
        if self._source_records is None:
            self._source_records = load_source_records_from_data_dir()
        return self._source_records

    def evidence_quality(self, drug_id: str | None) -> dict[str, Any]:
        """Availability mask (M) x completeness (Q) summary for one drug.

        This describes evidence coverage, not risk and not prediction
        confidence.
        """
        if not drug_id:
            return {
                "drug_id": "",
                "evidence_quality": None,
                "q_cov": None,
                "evidence_gap": True,
                "available_sources": [],
                "missing_sources": [],
            }
        return compute_evidence_quality_score(str(drug_id), self._sources())

    def _summary(self, drug_id: str | None) -> dict[str, Any]:
        summary = summarise_drug(drug_id)
        quality = self.evidence_quality(drug_id)
        mask = quality.get("availability_mask") or []
        available_names = [
            str(name) for name, present in mask if float(present or 0.0) > 0
        ]
        # Per-drug missing sources come from the availability mask. The
        # engine's own missing_sources list means a whole source dataset is
        # absent, which is a different fact.
        missing_names = [
            str(name) for name, present in mask if float(present or 0.0) <= 0
        ]
        summary.update(
            {
                "evidence_quality": quality.get("evidence_quality"),
                "q_cov": quality.get("q_cov"),
                "evidence_gap": bool(quality.get("evidence_gap", True)),
                "available_sources": available_names,
                "available_source_count": int(quality.get("available_sources") or 0),
                "total_sources": int(quality.get("total_sources") or 0),
                "missing_sources": missing_names,
                "missing_datasets": list(quality.get("missing_sources") or []),
            }
        )
        return summary

    # ------------------------------------------------------------------ #
    # Public API (abstaining)
    # ------------------------------------------------------------------ #
    def predict(
        self, drug_id: str, outcome: str, trimester: str | None = None
    ) -> dict[str, Any]:
        """Report evidence for one drug/outcome/trimester; never a risk value."""
        summary = self._summary(drug_id)
        column = OUTCOME_COUNT_COLUMNS.get(outcome)
        row = _load_drug_row(str(drug_id))
        observed = _safe_int(row.get(column)) if (row is not None and column) else None
        return {
            "drug_id": str(drug_id),
            "outcome": outcome,
            "trimester": trimester,
            "trimester_label": TRIMESTER_LABELS.get(trimester, trimester or "All"),
            "risk": None,
            "risk_available": False,
            "confidence": None,
            "model_used": "none",
            "status": NO_MODEL_MESSAGE,
            "observed_reports": observed,
            "evidence_quality": summary["evidence_quality"],
            "q_cov": summary["q_cov"],
            "evidence_gap": summary["evidence_gap"],
            "available_sources": summary["available_sources"],
            "missing_sources": summary["missing_sources"],
        }

    def trimester_evidence(self, drug_id: str) -> dict[str, Any]:
        """Per-trimester evidence availability with no risk values."""
        summary = self._summary(drug_id)
        return {
            trimester: {
                "label": TRIMESTER_LABELS[trimester],
                "aggregate_risk": None,
                "confidence": None,
                "model_used": "none",
                "outcomes": {outcome: None for outcome in FETUS_OUTCOMES},
                "evidence_quality": summary["evidence_quality"],
                "q_cov": summary["q_cov"],
                "evidence_gap": summary["evidence_gap"],
                "available_sources": summary["available_sources"],
                "missing_sources": summary["missing_sources"],
            }
            for trimester in TRIMESTERS
        }

    def foetus_risk_all_trimesters(self, drug_id: str) -> dict[str, Any]:
        """Foetal-outcome view of the evidence; risk values are unavailable."""
        summary = self._summary(drug_id)
        return {
            "drug_id": str(drug_id),
            "risk_available": False,
            "status": NO_MODEL_MESSAGE,
            "overall_foetus_risk": None,
            "overall_confidence": None,
            "overall_evidence_quality": summary["evidence_quality"],
            "q_cov": summary["q_cov"],
            "evidence_gap": summary["evidence_gap"],
            "model_used": "none",
            "observed_outcome_reports": {
                outcome: summary["observed_outcome_reports"].get(outcome)
                for outcome in FETUS_OUTCOMES
            },
            "trimester_wise": self.trimester_evidence(drug_id),
        }

    def maternal_risk(self, drug_id: str) -> dict[str, Any]:
        """Maternal-outcome view of the evidence; risk values are unavailable."""
        summary = self._summary(drug_id)
        return {
            "drug_id": str(drug_id),
            "risk_available": False,
            "status": NO_MODEL_MESSAGE,
            "maternal_risk": None,
            "confidence": None,
            "model_used": "none",
            "evidence_quality": summary["evidence_quality"],
            "q_cov": summary["q_cov"],
            "evidence_gap": summary["evidence_gap"],
            "observed_reports": summary["observed_outcome_reports"].get(
                "maternal_adverse_reaction"
            ),
        }

    def score_drug(self, drug_id: str) -> dict[str, Any]:
        """Complete evidence summary for a drug: counts + coverage, no score."""
        summary = self._summary(drug_id)
        row = _load_drug_row(str(drug_id))
        return {
            "drug_id": str(drug_id),
            "drug_name": summary["drug_name"],
            "risk_available": False,
            "status": NO_MODEL_MESSAGE,
            "foetus_risk": self.foetus_risk_all_trimesters(drug_id),
            "maternal_risk": self.maternal_risk(drug_id),
            "overall_confidence": None,
            "overall_evidence_quality": summary["evidence_quality"],
            "q_cov": summary["q_cov"],
            "evidence_gap": summary["evidence_gap"],
            "available_sources": summary["available_sources"],
            "missing_sources": summary["missing_sources"],
            "model_used": "none",
            "observed_outcome_reports": summary["observed_outcome_reports"],
            "adr_term_count": summary["adr_term_count"],
            "faers_report_count": summary["faers_report_count"],
            "sources_with_records": summary["sources_with_records"],
        }

    # ------------------------------------------------------------------ #
    # Records
    # ------------------------------------------------------------------ #
    def metrics_record(self) -> dict[str, Any]:
        """Explicit blocked-evaluation record for this component."""
        return {
            "component": "PregnancyEvidenceEngine",
            "status": "blocked",
            "training_status": "blocked",
            "reason": NO_MODEL_MESSAGE,
            "risk_scores_available": False,
            "confidence_available": False,
            "evaluation": {
                "auroc": None,
                "auprc": None,
                "precision_at_k": None,
                "recall_at_k": None,
                "brier_score": None,
                "ece": None,
                "n": None,
            },
            "outcomes": list(ALL_OUTCOMES),
            "trimesters": list(TRIMESTERS),
            "note": (
                "No AUROC/AUPRC/calibration values are reported because no "
                "validated human pregnancy-outcome labels exist. Missing values "
                "are unavailable, not zero."
            ),
        }

    def write_records(self) -> dict[str, Any]:
        """Persist the blocked-evaluation record and model metadata."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        record = self.metrics_record()
        METRICS_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
        MODEL_METADATA_PATH.write_text(
            json.dumps(
                {
                    "model_type": "PregnancyEvidenceEngine (no predictive model)",
                    "trained": False,
                    "risk_scores_available": False,
                    "outcomes": list(ALL_OUTCOMES),
                    "trimesters": list(TRIMESTERS),
                    "reason": NO_MODEL_MESSAGE,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return record


# Backwards-compatible name for callers that imported the old class.
DeterministicPregnancyRiskModel = PregnancyEvidenceEngine

_engine: PregnancyEvidenceEngine | None = None


def get_model() -> PregnancyEvidenceEngine:
    global _engine
    if _engine is None:
        _engine = PregnancyEvidenceEngine()
    return _engine


def score_drug(drug_id: str) -> dict[str, Any]:
    return get_model().score_drug(drug_id)


def foetus_risk_all_trimesters(drug_id: str) -> dict[str, Any]:
    return get_model().foetus_risk_all_trimesters(drug_id)


def maternal_risk(drug_id: str) -> dict[str, Any]:
    return get_model().maternal_risk(drug_id)


def predict(
    drug_id: str, outcome: str, trimester: str | None = None
) -> dict[str, Any]:
    return get_model().predict(drug_id, outcome, trimester)


def main() -> None:
    record = get_model().write_records()
    print(
        "PregnancyEvidenceEngine: status={status}; risk_scores_available={risk}".format(
            status=record["status"],
            risk=record["risk_scores_available"],
        )
    )
    print(record["reason"])


if __name__ == "__main__":
    main()
