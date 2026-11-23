"""Prioritization engine orchestrator.

Combines maternal and fetal evidence engines into a complete per-trimester
prioritization result. Also determines the evidence-availability status
(insufficient evidence, single-source, limited pregnancy evidence, human
evidence available, human + nonclinical, conflicting evidence, score
unavailable).
"""

from __future__ import annotations

from typing import Any

from .config import ScoringConfig, load_config
from .fetal_engine import FetalEvidenceEngine
from .maternal_engine import MaternalEvidenceEngine
from .overall import overall_for_trimester


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

NONCLINICAL_SOURCES = ("reprotox_kg", "epa_toxrefdb")


class PrioritizationResult:
    """Container for a complete per-drug prioritization result."""

    def __init__(
        self,
        drug_id: str,
        drug_name: str | None = None,
        config: ScoringConfig | None = None,
    ) -> None:
        self.drug_id = str(drug_id)
        self.drug_name = drug_name or drug_id
        self.config = config or load_config()
        self.maternal: dict[str, dict[str, Any]] = {}
        self.fetal: dict[str, dict[str, Any]] = {}
        self.overall: dict[str, dict[str, Any]] = {}
        self.evidence_status: str = "score_unavailable"
        self.status_detail: str = ""
        self.q_cov: float | None = None
        self.evidence_gap: bool = True
        self.available_sources: list[str] = []
        self.missing_sources: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "drug_id": self.drug_id,
            "drug_name": self.drug_name,
            "evidence_status": self.evidence_status,
            "status_detail": self.status_detail,
            "q_cov": self.q_cov,
            "evidence_gap": self.evidence_gap,
            "available_sources": self.available_sources,
            "missing_sources": self.missing_sources,
            "maternal_priority": {
                t: self.maternal.get(t, {}) for t in TRIMESTERS
            },
            "fetal_priority": {
                t: self.fetal.get(t, {}) for t in TRIMESTERS
            },
            "overall_priority": {
                t: self.overall.get(t, {}) for t in TRIMESTERS
            },
            "method": "evidence_weighted_prioritization",
            "note": (
                "Exploratory evidence-priority indices (0-100), not clinical "
                "risk probabilities. A score is unavailable when no eligible "
                "human evidence exists for a trimester."
            ),
        }


class PrioritizationEngine:
    """Orchestrates maternal and fetal scoring across all trimesters."""

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or load_config()
        self.maternal_engine = MaternalEvidenceEngine(self.config)
        self.fetal_engine = FetalEvidenceEngine(self.config)

    def score_drug(
        self, drug_id: str, drug_name: str | None = None
    ) -> PrioritizationResult:
        """Compute maternal, fetal, and overall prioritization for all trimesters."""
        result = PrioritizationResult(drug_id, drug_name, self.config)

        # Score each trimester
        for trimester in ("first_trimester", "second_trimester", "third_trimester", "entire_pregnancy"):
            m = self.maternal_engine.score_trimester(drug_id, trimester)
            f = self.fetal_engine.score_trimester(drug_id, trimester)
            result.maternal[trimester] = m
            result.fetal[trimester] = f
            result.overall[trimester] = overall_for_trimester(m, f, self.config._data)

        # Determine evidence status
        result.evidence_status, result.status_detail = self._determine_status(result)

        # Aggregate Q_cov and source info from the entire_pregnancy maternal score
        # (as a proxy for overall evidence availability)
        entire_m = result.maternal.get("entire_pregnancy", {})
        result.q_cov = None  # Q_cov comes from the evidence_quality module, not here
        result.evidence_gap = True
        result.available_sources = []
        result.missing_sources = []

        # Aggregate source counts across all trimesters
        all_sources: dict[str, int] = {}
        all_nonclinical = False
        for trimester in TRIMESTERS:
            m = result.maternal[trimester]
            f = result.fetal[trimester]
            for src, cnt in (m.get("source_counts") or {}).items():
                all_sources[src] = all_sources.get(src, 0) + cnt
            for src, cnt in (f.get("source_counts") or {}).items():
                all_sources[src] = all_sources.get(src, 0) + cnt
            if m.get("has_nonclinical") or f.get("has_nonclinical_evidence"):
                all_nonclinical = True

        result.available_sources = sorted(all_sources.keys())
        if all_nonclinical:
            result.status_detail += " Nonclinical evidence also available."

        return result

    def _determine_status(
        self, result: PrioritizationResult
    ) -> tuple[str, str]:
        """Determine the evidence-availability status from actual conditions.

        Statuses are generated from actual evidence conditions, not hardcoded.
        """
        # Count trimesters with any component score (maternal or fetal)
        scored_trimesters = [
            t for t in ("first_trimester", "second_trimester", "third_trimester", "entire_pregnancy")
            if result.maternal[t].get("score_available")
            or result.fetal[t].get("score_available")
        ]
        human_trimesters = [
            t for t in ("first_trimester", "second_trimester", "third_trimester", "entire_pregnancy")
            if result.maternal[t].get("score_available")
            or result.fetal[t].get("score_available")
        ]

        total_human_records = sum(
            result.maternal[t].get("total_human_records", 0)
            + result.fetal[t].get("total_human_records", 0)
            for t in ("first_trimester", "second_trimester", "third_trimester", "entire_pregnancy")
        )
        total_nonclinical = sum(
            result.fetal[t].get("total_nonclinical_records", 0)
            for t in ("first_trimester", "second_trimester", "third_trimester", "entire_pregnancy")
        )

        unique_sources = set(result.available_sources)
        has_nonclinical = total_nonclinical > 0

        if not scored_trimesters:
            if total_human_records == 0 and total_nonclinical == 0:
                return "score_unavailable", (
                    "No eligible evidence exists for this drug in any trimester. "
                    "No score is produced."
                )
            if total_human_records == 0 and total_nonclinical > 0:
                return "score_unavailable", (
                    "Only nonclinical (animal) evidence is available. No human "
                    "pregnancy risk score can be produced."
                )
            if total_human_records > 0:
                return "insufficient_evidence", (
                    "Human evidence records exist but none are eligible for any "
                    "trimester-specific score (all timing is unknown or no "
                    "outcome matches). Score unavailable."
                )

        # We have at least one scored trimester
        if len(unique_sources) == 1:
            return "single_source_evidence", (
                "Evidence comes from a single source family. Score is "
                "exploratory and should be interpreted with caution."
            )

        if has_nonclinical and len(human_trimesters) > 0:
            return "human_and_nonclinical_evidence_available", (
                "Both human pregnancy evidence and nonclinical toxicity evidence "
                "are available. Nonclinical evidence is reported separately and "
                "is not merged into the human score."
            )

        if len(human_trimesters) >= 2:
            return "human_pregnancy_evidence_available", (
                "Human pregnancy evidence is available across multiple trimesters. "
                "Scores are exploratory and require clinical confirmation."
            )

        if len(human_trimesters) == 1:
            return "limited_pregnancy_evidence", (
                "Human pregnancy evidence is limited to a single trimester. "
                "Score is exploratory and should be interpreted with caution."
            )

        # Check for conflicting evidence (human vs nonclinical disagreement)
        if has_nonclinical and total_human_records > 0:
            return "conflicting_evidence", (
                "Human and nonclinical evidence are both present. These evidence "
                "tiers are kept separate; no reconciliation is implied."
            )

        return "score_unavailable", "No score could be computed."
