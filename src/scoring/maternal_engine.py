"""Maternal evidence engine.

Computes an evidence-priority index (0-100) for maternal safety per trimester
using only actual available human evidence. General ADR records are used as
contextual maternal evidence and are never silently reclassified as
pregnancy-specific maternal outcomes. Nonclinical evidence is excluded.

A score is ``None`` (unavailable) when no eligible maternal evidence exists for
a trimester. Missing records are never awarded a safety benefit.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ScoringConfig, load_config
from .endpoint_mapping import (
    MATERNAL_OUTCOMES,
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


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _record_completeness(row: pd.Series, content_fields: list[str]) -> float:
    """Fraction of populated content fields for one record (0.0-1.0)."""
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


def _load_real_adr() -> pd.DataFrame:
    path = DATA_DIR / "real_adr_evidence.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _load_real_pregnancy_features() -> pd.DataFrame:
    path = DATA_DIR / "real_pregnancy_features.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _load_faers_summary() -> pd.DataFrame:
    path = DATA_DIR / "real_faers_summary.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _load_pregnancy_classifications() -> pd.DataFrame:
    path = DATA_DIR / "pregnancy_classifications.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


class MaternalEvidenceEngine:
    """Scores maternal evidence priority per trimester.

    Evidence sources (human only):
    - standardized_master_evidence.csv: maternal_adverse_reaction records
    - real_adr_evidence.csv: general ADR terms (contextual, lower weight)
    - real_pregnancy_features.csv: reports_maternal_adverse_reaction counts
    - real_faers_summary.csv: aggregate FAERS report counts
    - pregnancy_classifications.csv: TGA classifications (contextual)

    Nonclinical sources (reprotox_kg, epa_toxrefdb) are excluded.
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or load_config()
        self._master: pd.DataFrame | None = None
        self._adr: pd.DataFrame | None = None
        self._preg_features: pd.DataFrame | None = None
        self._faers: pd.DataFrame | None = None
        self._classifications: pd.DataFrame | None = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._master = _load_master_evidence()
        self._adr = _load_real_adr()
        self._preg_features = _load_real_pregnancy_features()
        self._faers = _load_faers_summary()
        self._classifications = _load_pregnancy_classifications()
        self._loaded = True

    # ------------------------------------------------------------------ #
    # Record gathering
    # ------------------------------------------------------------------ #
    def _master_records_for_drug(self, drug_id: str) -> list[dict[str, Any]]:
        """Maternal outcome records from the master evidence table."""
        if self._master is None or self._master.empty:
            return []
        if "drug_id" not in self._master.columns:
            return []
        rows = self._master[self._master["drug_id"].astype(str) == str(drug_id)]
        if rows.empty:
            return []
        maternal = rows[
            rows["outcome_category"].astype(str).str.casefold().eq("maternal_adverse_reaction")
        ]
        records: list[dict[str, Any]] = []
        for _, row in maternal.iterrows():
            timing, timing_class = classify_timing(
                row.get("gestational_weeks"), row.get("trimester")
            )
            source_dataset = str(row.get("source_dataset", "master_evidence"))
            records.append(
                {
                    "source": source_dataset,
                    "source_family": _source_family(source_dataset),
                    "outcome": str(row.get("outcome_category", "maternal_adverse_reaction")),
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

    def _adr_context_records_for_drug(self, drug_id: str) -> list[dict[str, Any]]:
        """General ADR terms as contextual maternal evidence (lower weight)."""
        if self._adr is None or self._adr.empty:
            return []
        if "drug_id" not in self._adr.columns:
            return []
        rows = self._adr[self._adr["drug_id"].astype(str) == str(drug_id)]
        if rows.empty:
            return []
        records = []
        for _, row in rows.iterrows():
            records.append(
                {
                    "source": "real_adr_evidence",
                    "source_family": "adr_context",
                    "outcome": "maternal_adverse_reaction",
                    "outcome_term": str(row.get("adverse_event_term", "")),
                    "reported_term": str(row.get("adverse_event_term", "")),
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

    def _preg_features_context_for_drug(self, drug_id: str) -> list[dict[str, Any]]:
        """Aggregate maternal adverse reaction counts from pregnancy features."""
        if self._preg_features is None or self._preg_features.empty:
            return []
        if "drug_id" not in self._preg_features.columns:
            return []
        rows = self._preg_features[self._preg_features["drug_id"].astype(str) == str(drug_id)]
        if rows.empty:
            return []
        row = rows.iloc[0]
        count = _safe_float(row.get("reports_maternal_adverse_reaction"))
        if count is None or count <= 0:
            return []
        return [
            {
                "source": "real_pregnancy_features",
                "source_family": "pregnancy_features",
                "outcome": "maternal_adverse_reaction",
                "outcome_term": "maternal_adverse_reaction",
                "reported_term": "maternal_adverse_reaction",
                "reported_count": count,
                "source_record_id": f"preg_features:{drug_id}",
                "timing": "entire_pregnancy",
                "timing_class": "entire_pregnancy",
                "completeness": _safe_float(row.get("quality_pregnancy")) or 0.5,
            }
        ]

    def _faers_context_for_drug(self, drug_id: str) -> list[dict[str, Any]]:
        """Aggregate FAERS report counts as contextual maternal evidence."""
        if self._faers is None or self._faers.empty:
            return []
        if "drug_id" not in self._faers.columns:
            return []
        rows = self._faers[self._faers["drug_id"].astype(str) == str(drug_id)]
        if rows.empty:
            return []
        row = rows.iloc[0]
        count = _safe_float(row.get("faers_report_count"))
        if count is None or count <= 0:
            return []
        return [
            {
                "source": "real_faers_summary",
                "source_family": "faers_context",
                "outcome": "maternal_adverse_reaction",
                "outcome_term": "faers_report_count",
                "reported_term": "faers_report_count",
                "reported_count": count,
                "source_record_id": f"faers:{drug_id}",
                "timing": "entire_pregnancy",
                "timing_class": "entire_pregnancy",
                "completeness": _safe_float(row.get("quality_faers")) or 0.5,
            }
        ]

    def _classification_context_for_drug(self, drug_id: str) -> list[dict[str, Any]]:
        """TGA pregnancy classifications as contextual maternal evidence."""
        if self._classifications is None or self._classifications.empty:
            return []
        if "drug_id" not in self._classifications.columns:
            return []
        rows = self._classifications[self._classifications["drug_id"].astype(str) == str(drug_id)]
        if rows.empty:
            return []
        row = rows.iloc[0]
        category = str(row.get("Category", "")) or ""
        safety = str(row.get("Safety statement", "")) or ""
        if not category and not safety:
            return []
        return [
            {
                "source": "pregnancy_classifications",
                "source_family": "tga_classification",
                "outcome": "maternal_adverse_reaction",
                "outcome_term": f"tga_category:{category}",
                "reported_term": safety or category,
                "reported_count": None,
                "source_record_id": f"tga:{drug_id}",
                "timing": "entire_pregnancy",
                "timing_class": "entire_pregnancy",
                "completeness": 0.6 if (category or safety) else 0.0,
            }
        ]

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _score_records(
        self,
        records: list[dict[str, Any]],
        trimester: str,
        is_nonclinical: bool = False,
    ) -> float | None:
        """Compute the 0-100 evidence-priority index for a list of records.

        Returns ``None`` when no records are eligible for the trimester.
        """
        if not records:
            return None

        eligible: list[dict[str, Any]] = []
        for rec in records:
            if is_nonclinical:
                # Nonclinical records are scored only for the entire_pregnancy
                # bucket and never mixed into human trimester scores.
                if trimester != "entire_pregnancy":
                    continue
            else:
                # T1/T2/T3: only explicitly timed records for that trimester.
                # Untimed records are never assigned to a specific trimester.
                if trimester != "entire_pregnancy":
                    if rec["timing"] != trimester:
                        continue
                # Entire pregnancy: include all human records (timed and
                # untimed), since "entire pregnancy" means evidence applies
                # to pregnancy generally.
            er = endpoint_relevance(rec["outcome"], trimester, self.config)
            if er <= 0:
                continue
            sr = source_reliability(rec["source"], self.config)
            if sr <= 0:
                continue
            eligible.append(rec)

        if not eligible:
            return None

        total = 0.0
        for rec in eligible:
            er = endpoint_relevance(rec["outcome"], trimester, self.config)
            sr = source_reliability(rec["source_family"], self.config)
            rc = rec.get("completeness", 0.0) or 0.0
            tr = timing_relevance(rec.get("timing_class", "unknown"), self.config)
            vf = bounded_volume_factor(
                1, rec.get("reported_count"), self.config
            )
            contribution = er * sr * rc * tr * vf
            total += contribution

        index = round(min(100.0, total * self.config.normalization_factor), 1)
        return index if index > 0 else None

    def score_trimester(
        self, drug_id: str, trimester: str
    ) -> dict[str, Any]:
        """Compute the maternal evidence-priority index for one trimester.

        Returns a dict with the score (0-100 or None), record counts,
        source breakdown, and status.
        """
        self._ensure_loaded()
        records: list[dict[str, Any]] = []
        records.extend(self._master_records_for_drug(drug_id))
        records.extend(self._adr_context_records_for_drug(drug_id))
        records.extend(self._preg_features_context_for_drug(drug_id))
        records.extend(self._faers_context_for_drug(drug_id))
        records.extend(self._classification_context_for_drug(drug_id))

        human_records = [
            r for r in records if r["source_family"] not in NONCLINICAL_SOURCES
        ]
        nonclinical_records = [
            r for r in records if r["source_family"] in NONCLINICAL_SOURCES
        ]

        score = self._score_records(human_records, trimester, is_nonclinical=False)

        # Record counts by source for transparency
        source_counts: dict[str, int] = {}
        for rec in human_records:
            source_counts[rec["source"]] = source_counts.get(rec["source"], 0) + 1

        # Count records that are actually eligible for this trimester
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
            "source_counts": source_counts,
            "has_nonclinical": len(nonclinical_records) > 0,
            "method": "evidence_weighted_prioritization",
        }
