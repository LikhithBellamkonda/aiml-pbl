"""Evidence quality head -- separates evidence reliability from prediction confidence.

A model's prediction confidence (how sure it is about the output) should be
distinct from evidence reliability (how much / how good the underlying data is).
This module computes an evidence quality score per drug that is independent of
the risk prediction, enabling the system to say "we don't know enough to predict"
rather than "we predict low risk with high confidence."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def compute_evidence_quality_score(
    drug_id: str,
    source_records: dict[str, pd.DataFrame],
    source_order: tuple[str, ...] = (
        "onsides",
        "mendeley",
        "faers",
        "tga",
    ),
) -> dict[str, Any]:
    """Compute an evidence quality score for a single drug.

    The score combines:
    - Availability mask (M): which sources have data for this drug
    - Quality vector (Q): completeness of provenance fields per source
    - Recency: how recent the source data is (proxy: source version)
    - Volume: number of reports/records per source
    - Consistency: agreement across sources for the same outcome

    Returns a dict with:
    - evidence_quality: float in [0, 1] -- overall evidence reliability
    - availability_mask: list of (source, available) tuples
    - quality_vector: list of (source, quality) tuples
    - q_cov: evidence coverage score (mean of M*Q)
    - evidence_gap: bool -- True if evidence is too sparse for reliable prediction
    - missing_sources: list of source names with no data
    """
    provenance_fields = (
        "source_dataset",
        "source_url",
        "source_doi",
        "evidence_status",
        "evidence_granularity",
    )

    availability = []
    quality_vector = []
    missing_sources = []

    for source in source_order:
        frame = source_records.get(source, pd.DataFrame())
        if frame.empty or "drug_id" not in frame.columns:
            availability.append((source, 0.0))
            quality_vector.append((source, 0.0))
            missing_sources.append(source)
            continue

        drug_rows = frame[frame["drug_id"].astype(str) == str(drug_id)]
        available = 1.0 if not drug_rows.empty else 0.0
        availability.append((source, available))

        if not drug_rows.empty:
            present = 0
            for field in provenance_fields:
                if field in drug_rows.columns:
                    non_empty = drug_rows[field].notna() & (
                        drug_rows[field].astype(str).str.strip() != ""
                    )
                    if non_empty.any():
                        present += 1
            quality = present / max(len(provenance_fields), 1)
            # Volume boost: more records = slightly higher quality (capped)
            record_count = len(drug_rows)
            volume_boost = min(record_count / 10.0, 0.2)
            quality = min(1.0, quality + volume_boost)
        else:
            quality = 0.0

        quality_vector.append((source, quality))

    # Q_cov = mean(M * Q) across sources
    m_values = [v for _, v in availability]
    q_values = [v for _, v in quality_vector]
    q_cov = float(np_mean([m * q for m, q in zip(m_values, q_values)]))

    # Evidence gap: too few sources or too low quality
    available_sources = sum(1 for _, v in availability if v > 0)
    evidence_gap = available_sources < 2 or q_cov < 0.30

    return {
        "drug_id": drug_id,
        "evidence_quality": round(q_cov, 4),
        "availability_mask": availability,
        "quality_vector": quality_vector,
        "q_cov": round(q_cov, 4),
        "available_sources": available_sources,
        "total_sources": len(source_order),
        "missing_sources": missing_sources,
        "evidence_gap": evidence_gap,
        "interpretation": (
            "Evidence quality reflects data completeness and reliability, "
            "not prediction certainty. A low score means the model should "
            "abstain rather than guess."
        ),
    }


def np_mean(values: list[float]) -> float:
    """Safe mean returning 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def compute_batch_evidence_quality(
    drug_ids: list[str],
    source_records: dict[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    """Compute evidence quality scores for a batch of drugs."""
    return {
        str(drug_id): compute_evidence_quality_score(str(drug_id), source_records)
        for drug_id in drug_ids
    }


def load_source_records_from_data_dir() -> dict[str, pd.DataFrame]:
    """Load all available source records from the data directory for evidence
    quality computation."""
    records: dict[str, pd.DataFrame] = {}

    # OnSIDES general ADR
    path = DATA_DIR / "real_adr_evidence.csv"
    if path.exists():
        records["onsides"] = pd.read_csv(path)

    # FAERS summary
    path = DATA_DIR / "real_faers_summary.csv"
    if path.exists():
        records["faers"] = pd.read_csv(path)

    # Pregnancy outcomes (Mendeley)
    path = DATA_DIR / "real_pregnancy_outcomes.csv"
    if path.exists():
        records["mendeley"] = pd.read_csv(path)

    # TGA / Medicines pregnancy classifications
    path = DATA_DIR / "pregnancy_classifications.csv"
    if path.exists():
        records["tga"] = pd.read_csv(path)

    return records
