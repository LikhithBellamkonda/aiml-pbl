"""Fetal evidence engine.

Computes an evidence-priority index (0-100) for fetal safety per trimester
using human pregnancy outcome evidence, human spontaneous-report evidence, and
general ADR evidence. Nonclinical evidence (EPA ToxRefDB, ReproTox-KG) is kept
in a separate tier and never merged into the human fetal score.

Fetal outcomes by trimester:
- T1: congenital anomalies, organogenesis-related developmental toxicity,
  miscarriage, other early developmental outcomes
- T2: fetal growth abnormalities, developmental abnormalities, pregnancy loss,
  other mid-pregnancy outcomes
- T3: fetal growth abnormalities, preterm birth, stillbirth, late fetal
  developmental effects, neonatal adaptation effects

A score is ``None`` (unavailable) when no eligible fetal evidence exists for a
trimester. Missing records are never awarded a safety benefit.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ScoringConfig, load_config
from .endpoint_mapping import (
    ALL_FETAL_OUTCOMES,
    FETUS_OUTCOMES_BY_TRIMESTER,
    NONCLINICAL_SOURCES,
    classify_timing,
    bounded_volume_factor,
    endpoint_relevance,
    source_reliability,
    timing_relevance,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_completeness(row: pd.Series, content_fields: list[str]) -> float:
    if not content_fields:
        return 0.0
    populated = 0
    for field in content_fields:
        if field in row.index:
            v = row[field]
            if pd.notna(v) and str(v).strip() != "":
                populated += 1
    return populated / len(content_fields)


def _source_family(source_value: Any) -> str:
    """Map a source_dataset string to a source family."""
    source = str(source_value).casefold()
    if "mendeley" in source:
        return "mendeley"
    if "onsides" in source:
        return "onsides"
    if "faers" in source:
        return "faers"
    if "tga" in source or "medicines-pregnancy" in source:
        return "tga"
    if "reprotox" in source:
        return "reprotox_kg"
    if "epa" in source or "toxrefdb" in source:
        return "epa_toxrefdb"
    return "unknown"


def _load_master_evidence() -> pd.DataFrame:
    path = DATA_DIR / "standardized_master_evidence.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _load_real_pregnancy_outcomes() -> pd.DataFrame:
    path = DATA_DIR / "real_pregnancy_outcomes.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _load_real_pregnancy_features() -> pd.DataFrame:
    path = DATA_DIR / "real_pregnancy_features.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _load_real_adr() -> pd.DataFrame:
    path = DATA_DIR / "real_adr_evidence.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _load_reprotox() -> pd.DataFrame:
    path = DATA_DIR / "cleaned" / "reprotox_kg.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _load_epa_toxrefdb() -> pd.DataFrame:
    path = DATA_DIR / "cleaned" / "epa_toxrefdb.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


class FetalEvidenceEngine:
    """Scores fetal evidence priority per trimester.

    Source hierarchy (human first):
    A. Human pregnancy outcome evidence (master_evidence, pregnancy outcomes)
    B. Human spontaneous-report evidence (FAERS, OnSIDES ADR terms)
    C. Nonclinical evidence (ReproTox-KG, EPA ToxRefDB) - separate tier only

    Nonclinical findings support a developmental toxicity concern but are
    never converted into human outcome labels or merged into the human fetal
    score.
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or load_config()
        self._master: pd.DataFrame | None = None
        self._preg_outcomes: pd.DataFrame | None = None
        self._preg_features: pd.DataFrame | None = None
        self._adr: pd.DataFrame | None = None
        self._reprotox: pd.DataFrame | None = None
        self._epa: pd.DataFrame | None = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._master = _load_master_evidence()
        self._preg_outcomes = _load_real_pregnancy_outcomes()
        self._preg_features = _load_real_pregnancy_features()
        self._adr = _load_real_adr()
        self._reprotox = _load_reprotox()
        self._epa = _load_epa_toxrefdb()
        self._loaded = True

    # ------------------------------------------------------------------ #
    # Record gathering
    # ------------------------------------------------------------------ #
    def _master_records_for_drug(self, drug_id: str) -> list[dict[str, Any]]:
        if self._master is None or self._master.empty:
            return []
        if "drug_id" not in self._master.columns:
            return []
        rows = self._master[self._master["drug_id"].astype(str) == str(drug_id)]
        if rows.empty:
            return []
        fetal = rows[
            rows["outcome_category"].astype(str).str.casefold().isin(
                {o.casefold() for o in ALL_FETAL_OUTCOMES}
            )
        ]
        records: list[dict[str, Any]] = []
        for _, row in fetal.iterrows():
            timing, timing_class = classify_timing(
                row.get("gestational_weeks"), row.get("trimester")
            )
            source_dataset = str(row.get("source_dataset", "master_evidence"))
            records.append(
                {
                    "source": source_dataset,
                    "source_family": _source_family(source_dataset),
                    "outcome": str(row.get("outcome_category", "")),
                    "outcome_term": str(row.get("outcome_term_original", "")) or str(
                        row.get("outcome", "")
                    ),
                    "reported_term": str(row.get("reported_term", "")) or "",
                    "reported_count": _safe_float(row.get("reported_count")),
                    "source_record_id": str(row.get("source_record_id", "")),
                    "timing": timing,
                    "timing_class": timing_class,
                    "completeness": _record_completeness(
                        row,
                        [
                            "outcome_term_original",
                            "outcome",
                            "reported_term",
                            "evidence_status",
                            "evidence_granularity",
                            "source_url",
                            "source_doi",
                        ],
                    ),
                }
            )
        return records

    def _preg_outcomes_records_for_drug(self, drug_id: str) -> list[dict[str, Any]]:
        if self._preg_outcomes is None or self._preg_outcomes.empty:
            return []
        if "drug_id" not in self._preg_outcomes.columns:
            return []
        rows = self._preg_outcomes[self._preg_outcomes["drug_id"].astype(str) == str(drug_id)]
        if rows.empty:
            return []
        fetal = rows[
            rows["outcome"].astype(str).str.casefold().isin(
                {o.casefold() for o in ALL_FETAL_OUTCOMES}
            )
        ]
        records = []
        for _, row in fetal.iterrows():
            records.append(
                {
                    "source": "real_pregnancy_outcomes",
                    "source_family": "mendeley",
                    "outcome": str(row.get("outcome", "")),
                    "outcome_term": str(row.get("outcome_term_original", "")) or str(
                        row.get("reported_term", "")
                    ),
                    "reported_term": str(row.get("reported_term", "")) or "",
                    "reported_count": _safe_float(row.get("reported_count")),
                    "source_record_id": str(row.get("source_record_id", "")),
                    "timing": "unknown",
                    "timing_class": "unknown",
                    "completeness": _record_completeness(
                        row,
                        ["outcome_group", "outcome", "reported_term", "reported_count", "source_file"],
                    ),
                }
            )
        return records

    def _preg_features_records_for_drug(self, drug_id: str) -> list[dict[str, Any]]:
        if self._preg_features is None or self._preg_features.empty:
            return []
        if "drug_id" not in self._preg_features.columns:
            return []
        rows = self._preg_features[self._preg_features["drug_id"].astype(str) == str(drug_id)]
        if rows.empty:
            return []
        row = rows.iloc[0]
        records = []
        outcome_columns = {
            "reports_fetal_death": "fetal_death",
            "reports_preterm_birth": "preterm_birth",
            "reports_fetal_growth_restriction": "fetal_growth_restriction",
            "reports_congenital_anomaly": "congenital_anomaly",
        }
        for col, outcome in outcome_columns.items():
            count = _safe_float(row.get(col))
            if count is None or count <= 0:
                continue
            records.append(
                {
                    "source": "real_pregnancy_features",
                    "source_family": "pregnancy_features",
                    "outcome": outcome,
                    "outcome_term": outcome,
                    "reported_term": outcome,
                    "reported_count": count,
                    "source_record_id": f"preg_features:{drug_id}:{outcome}",
                    "timing": "entire_pregnancy",
                    "timing_class": "entire_pregnancy",
                    "completeness": _safe_float(row.get("quality_pregnancy")) or 0.5,
                }
            )
        return records

    def _adr_context_records_for_drug(self, drug_id: str) -> list[dict[str, Any]]:
        if self._adr is None or self._adr.empty:
            return []
        if "drug_id" not in self._adr.columns:
            return []
        rows = self._adr[self._adr["drug_id"].astype(str) == str(drug_id)]
        if rows.empty:
            return []
        records = []
        for _, row in rows.iterrows():
            term = str(row.get("adverse_event_term", ""))
            # Only include ADR terms that could plausibly relate to fetal
            # outcomes; these are contextual and weighted lower.
            fetal_keywords = (
                "abortion", "miscarriage", "stillbirth", "preterm", "congenital",
                "malformation", "anomaly", "growth", "placenta", "placental",
                "fetal", "foetal", "neonatal", "embryo"
            )
            if not any(k in term.casefold() for k in fetal_keywords):
                continue
            records.append(
                {
                    "source": "real_adr_evidence",
                    "source_family": "adr_context",
                    "outcome": "congenital_anomaly",
                    "outcome_term": term,
                    "reported_term": term,
                    "reported_count": None,
                    "source_record_id": str(row.get("source_identifier", "")),
                    "timing": "unknown",
                    "timing_class": "unknown",
                    "completeness": _record_completeness(
                        row,
                        ["adverse_event_term", "meddra_id", "concept_type", "source_identifier"],
                    ),
                }
            )
        return records

    def _reprotox_records_for_drug(self, drug_id: str) -> list[dict[str, Any]]:
        if self._reprotox is None or self._reprotox.empty:
            return []
        if "drug_id" not in self._reprotox.columns:
            return []
        rows = self._reprotox[self._reprotox["drug_id"].astype(str) == str(drug_id)]
        if rows.empty:
            return []
        records = []
        for _, row in rows.iterrows():
            records.append(
                {
                    "source": "reprotox_kg",
                    "source_family": "reprotox_kg",
                    "outcome": "congenital_anomaly",
                    "outcome_term": str(row.get("outcome", "")) or "nonclinical_association",
                    "reported_term": str(row.get("outcome", "")) or "",
                    "reported_count": None,
                    "source_record_id": str(row.get("source_record_id", "")),
                    "timing": "unknown",
                    "timing_class": "unknown",
                    "completeness": _record_completeness(row, ["outcome", "source_record_id"]),
                }
            )
        return records

    def _epa_records_for_drug(self, drug_id: str) -> list[dict[str, Any]]:
        if self._epa is None or self._epa.empty:
            return []
        if "drug_id" not in self._epa.columns:
            return []
        rows = self._epa[self._epa["drug_id"].astype(str) == str(drug_id)]
        if rows.empty:
            return []
        records = []
        for _, row in rows.iterrows():
            records.append(
                {
                    "source": "epa_toxrefdb",
                    "source_family": "epa_toxrefdb",
                    "outcome": "congenital_anomaly",
                    "outcome_term": str(row.get("outcome", "")) or "nonclinical_association",
                    "reported_term": str(row.get("outcome", "")) or "",
                    "reported_count": None,
                    "source_record_id": str(row.get("source_record_id", "")),
                    "timing": "unknown",
                    "timing_class": "unknown",
                    "completeness": _record_completeness(row, ["outcome", "source_record_id"]),
                }
            )
        return records

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _score_records(
        self,
        human_records: list[dict[str, Any]],
        nonclinical_records: list[dict[str, Any]],
        trimester: str,
    ) -> float | None:
        """Compute the 0-100 evidence-priority index for one trimester.

        Human and nonclinical records are scored separately and never merged.
        Returns the human score, or None when no eligible human evidence exists.
        """
        eligible_human: list[dict[str, Any]] = []
        for rec in human_records:
            # T1/T2/T3: only explicitly timed records for that trimester.
            # Untimed records are never assigned to a specific trimester.
            if trimester != "entire_pregnancy":
                if rec["timing"] != trimester:
                    continue
            # Entire pregnancy: include all human records (timed and untimed),
            # since "entire pregnancy" means evidence applies to pregnancy
            # generally. Timing relevance is adjusted by precision.
            rel = endpoint_relevance(rec["outcome"], trimester, self.config)
            if rel <= 0:
                continue
            eligible_human.append(rec)

        if not eligible_human:
            return None

        total = 0.0
        for rec in eligible_human:
            er = endpoint_relevance(rec["outcome"], trimester, self.config)
            sr = source_reliability(rec["source_family"], self.config)
            rc = rec.get("completeness", 0.0) or 0.0
            tr = timing_relevance(rec.get("timing_class", "unknown"), self.config)
            vf = bounded_volume_factor(1, rec.get("reported_count"), self.config)
            total += er * sr * rc * tr * vf

        index = round(min(100.0, total * self.config.normalization_factor), 1)
        return index if index > 0 else None

    def score_trimester(
        self, drug_id: str, trimester: str
    ) -> dict[str, Any]:
        """Compute the fetal evidence-priority index for one trimester."""
        self._ensure_loaded()
        human_records: list[dict[str, Any]] = []
        human_records.extend(self._master_records_for_drug(drug_id))
        human_records.extend(self._preg_outcomes_records_for_drug(drug_id))
        human_records.extend(self._preg_features_records_for_drug(drug_id))
        human_records.extend(self._adr_context_records_for_drug(drug_id))

        nonclinical_records: list[dict[str, Any]] = []
        nonclinical_records.extend(self._reprotox_records_for_drug(drug_id))
        nonclinical_records.extend(self._epa_records_for_drug(drug_id))

        score = self._score_records(human_records, nonclinical_records, trimester)

        source_counts: dict[str, int] = {}
        for rec in human_records:
            source_counts[rec["source"]] = source_counts.get(rec["source"], 0) + 1
        for rec in nonclinical_records:
            source_counts[rec["source"]] = source_counts.get(rec["source"], 0) + 1

        eligible_count = 0
        for rec in human_records:
            if rec["timing"] in (trimester, "entire_pregnancy") and rec["timing"] != "unknown":
                if endpoint_relevance(rec["outcome"], trimester, self.config) > 0:
                    eligible_count += 1

        return {
            "drug_id": str(drug_id),
            "trimester": trimester,
            "trimester_label": {
                "first_trimester": "First Trimester (T1)",
                "second_trimester": "Second Trimester (T2)",
                "third_trimester": "Third Trimester (T3)",
                "entire_pregnancy": "Entire Pregnancy",
            }[trimester],
            "score": score,
            "score_available": score is not None,
            "eligible_record_count": eligible_count,
            "total_human_records": len(human_records),
            "total_nonclinical_records": len(nonclinical_records),
            "source_counts": source_counts,
            "has_nonclinical_evidence": len(nonclinical_records) > 0,
            "method": "evidence_weighted_prioritization",
        }
