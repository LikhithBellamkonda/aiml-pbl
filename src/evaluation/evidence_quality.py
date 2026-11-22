"""Evidence quality -- measured per-drug evidence completeness, not confidence.

A model's prediction confidence (how sure it is about the output) must be
distinct from evidence reliability (how much / how good the underlying data
is). This module computes an evidence quality score per drug that is
independent of any risk prediction.

The score is computed from the values actually present in each acquired source
table for that drug. Earlier revisions checked for provenance column names that
do not exist in these tables, so every drug collapsed onto the same
``records / 10`` value and repeated values such as 2.5% or 12.5% appeared for
unrelated drugs. The current implementation measures:

* completeness -- fraction of the source's own content fields that are
  populated for this drug (measured, per source);
* volume -- log-scaled amount of evidence for this drug, relative to a
  documented reference volume per source (measured, per source).

Each source's quality is the mean of those two measured components. Q_cov is
the mean over the tracked evidence source families, so a drug with records in
one family cannot look as well supported as a drug with records in all four.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

SOURCE_ORDER: tuple[str, ...] = ("onsides", "mendeley", "faers", "tga")

SOURCE_LABELS = {
    "onsides": "OnSIDES general ADR terms",
    "mendeley": "Human pregnancy outcomes (aggregate)",
    "faers": "FAERS reporting summary (aggregate)",
    "tga": "Pregnancy risk classifications",
}

# Per-source measurement definition. ``content_fields`` are the columns whose
# populated fraction is the completeness component; ``volume_column`` is the
# column whose sum (or, when absent, the row count) is the volume component.
# ``volume_reference`` is the count at which the volume component reaches 1.0.
SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "onsides": {
        "content_fields": (
            "adverse_event_term",
            "meddra_id",
            "concept_type",
            "source_identifier",
        ),
        "volume_column": None,
        "volume_reference": 500.0,
        "granularity": "record-level labelled ADR terms",
    },
    "mendeley": {
        "content_fields": (
            "reported_term",
            "reported_count",
            "evidence_type",
            "source_table",
            "source_file",
        ),
        "volume_column": "reported_count",
        "volume_reference": 5000.0,
        "granularity": "aggregate spontaneous-report summary",
    },
    "faers": {
        "content_fields": (
            "faers_report_count",
            "faers_outcome_count",
            "faers_source_name",
        ),
        "volume_column": "faers_report_count",
        "volume_reference": 5000.0,
        "granularity": "aggregate reporting summary",
    },
    "tga": {
        "content_fields": (
            "Category",
            "Safety statement",
            "Classification 1",
            "Classification 2",
            "Classification 3",
        ),
        "volume_column": None,
        "volume_reference": None,
        "granularity": "regulatory classification record",
    },
}

QUALITY_METHOD = (
    "Per source: quality = 0.5 x completeness + 0.5 x volume, where "
    "completeness is the populated fraction of that source's content fields "
    "for this drug and volume is log1p(observed count) / "
    "log1p(source reference volume), capped at 1. Q_cov = mean of quality "
    "across the four tracked source families, with 0 for a family that has no "
    "record for the drug."
)


def _normalise_trimester(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return text


def _source_rows(
    frame: pd.DataFrame, drug_id: str, trimester: str | None
) -> pd.DataFrame:
    """Rows for this drug, restricted to a trimester when one is requested."""
    if frame.empty or "drug_id" not in frame.columns:
        return frame.iloc[0:0]
    rows = frame[frame["drug_id"].astype(str) == str(drug_id)]
    if rows.empty or not trimester:
        return rows
    # Trimester attribution is only possible where the source records timing.
    # Untimed records are never assigned to a trimester.
    if "trimester" not in rows.columns:
        return rows.iloc[0:0]
    wanted = _normalise_trimester(trimester)
    timing = rows["trimester"].map(_normalise_trimester)
    return rows[timing == wanted]


def _completeness(frame: pd.DataFrame, fields: tuple[str, ...]) -> float:
    present = [field for field in fields if field in frame.columns]
    if not present:
        return 0.0
    scores = []
    for field in present:
        values = frame[field]
        populated = values.notna() & (values.astype(str).str.strip() != "")
        scores.append(float(populated.mean()) if len(values) else 0.0)
    return sum(scores) / len(scores)


def _volume(row_count: int, observed_total: float, reference: float) -> float:
    if reference is None or reference <= 0:
        # Sources without a report volume contribute completeness only.
        return 1.0 if row_count else 0.0
    value = max(float(observed_total), float(row_count), 0.0)
    return min(1.0, math.log1p(value) / math.log1p(reference))


def compute_evidence_quality_score(
    drug_id: str,
    source_records: dict[str, pd.DataFrame],
    source_order: tuple[str, ...] = SOURCE_ORDER,
    trimester: str | None = None,
) -> dict[str, Any]:
    """Compute measured evidence quality for one drug.

    Returns a dict with the overall ``evidence_quality`` (mean Q_cov, or None
    when no source family has a record for the drug), the per-source
    ``source_breakdown`` used to derive it, and the availability mask.
    """
    availability: list[tuple[str, float]] = []
    quality_vector: list[tuple[str, float]] = []
    breakdown: list[dict[str, Any]] = []
    missing_sources: list[str] = []

    for source in source_order:
        frame = source_records.get(source, pd.DataFrame())
        spec = SOURCE_SPECS.get(source, {})
        rows = _source_rows(frame, drug_id, trimester)
        available = 1.0 if not rows.empty else 0.0
        availability.append((source, available))
        if frame.empty:
            missing_sources.append(source)

        if rows.empty:
            quality = 0.0
            breakdown.append(
                {
                    "source": source,
                    "label": SOURCE_LABELS.get(source, source),
                    "available": False,
                    "records": 0,
                    "observed_total": None,
                    "completeness": None,
                    "volume_score": None,
                    "quality": 0.0,
                    "granularity": spec.get("granularity"),
                }
            )
            quality_vector.append((source, 0.0))
            continue

        completeness = _completeness(rows, tuple(spec.get("content_fields", ())))
        volume_column = spec.get("volume_column")
        observed_total: float | None = None
        if volume_column and volume_column in rows.columns:
            values = pd.to_numeric(rows[volume_column], errors="coerce")
            if values.notna().any():
                observed_total = float(values.fillna(0.0).sum())
        volume = _volume(
            len(rows),
            observed_total if observed_total is not None else len(rows),
            spec.get("volume_reference"),
        )
        quality = 0.5 * completeness + 0.5 * volume
        quality_vector.append((source, quality))
        breakdown.append(
            {
                "source": source,
                "label": SOURCE_LABELS.get(source, source),
                "available": True,
                "records": int(len(rows)),
                "observed_total": observed_total,
                "completeness": round(completeness, 4),
                "volume_score": round(volume, 4),
                "quality": round(quality, 4),
                "granularity": spec.get("granularity"),
            }
        )

    available_sources = sum(1 for _, value in availability if value > 0)
    if not available_sources:
        q_cov: float | None = None
    else:
        q_cov = round(
            sum(quality for _, quality in quality_vector) / len(source_order), 4
        )

    evidence_gap = q_cov is None or available_sources < 2 or q_cov < 0.30

    return {
        "drug_id": drug_id,
        "trimester": trimester,
        "evidence_quality": q_cov,
        "availability_mask": availability,
        "quality_vector": quality_vector,
        "source_breakdown": breakdown,
        "q_cov": q_cov,
        "available_sources": available_sources,
        "total_sources": len(source_order),
        "missing_sources": missing_sources,
        "evidence_gap": evidence_gap,
        "quality_method": QUALITY_METHOD,
        "interpretation": (
            "Evidence quality is the measured completeness and volume of the "
            "records held for this drug. It is not a risk score and not "
            "prediction confidence; None means no recorded evidence, not zero "
            "risk."
        ),
    }


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
    """Load all available source records from the data directory."""
    records: dict[str, pd.DataFrame] = {}

    path = DATA_DIR / "real_adr_evidence.csv"
    if path.exists():
        records["onsides"] = pd.read_csv(path)

    path = DATA_DIR / "real_faers_summary.csv"
    if path.exists():
        records["faers"] = pd.read_csv(path)

    path = DATA_DIR / "real_pregnancy_outcomes.csv"
    if path.exists():
        records["mendeley"] = pd.read_csv(path)

    path = DATA_DIR / "pregnancy_classifications.csv"
    if path.exists():
        records["tga"] = pd.read_csv(path)

    return records
