from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.ml_risk_model import score_drug  # deterministic ML + AI routing engine


ROOT = Path(__file__).resolve().parents[1]
MASTER_EVIDENCE_PATH = ROOT / "data" / "standardized_master_evidence.csv"
ADR_EVIDENCE_PATH = ROOT / "data" / "real_adr_evidence.csv"
READ_COLUMNS = {
    "drug_id",
    "canonical_drug_id",
    "drug_name",
    "source_dataset",
    "source_url",
    "source_doi",
    "outcome_category",
    "outcome_term_original",
    "outcome",
    "reported_term",
    "trimester",
    "gestational_weeks",
    "evidence_status",
    "evidence_granularity",
    "study_endpoint",
    "study_type",
    "species",
    "association_value",
    "source_record_id",
}

TRIMESTER_OPTIONS = (
    "First Trimester (T1)",
    "Second Trimester (T2)",
    "Third Trimester (T3)",
    "Entire Pregnancy",
    "Unknown",
)
OUTCOME_TARGET_OPTIONS = (
    "Fetal Outcomes Only",
    "Maternal Outcomes Only",
    "Both",
)
TRIMESTER_VALUES = {
    "First Trimester (T1)": "first_trimester",
    "Second Trimester (T2)": "second_trimester",
    "Third Trimester (T3)": "third_trimester",
    "Entire Pregnancy": "entire_pregnancy",
    "Unknown": "unknown",
}
OUTCOME_CATEGORIES = {
    "Fetal Outcomes Only": {
        "congenital_anomaly",
        "miscarriage",
        "preterm_birth",
        "fetal_growth_outcome",
        "stillbirth",
    },
    "Maternal Outcomes Only": {"maternal_adverse_reaction"},
}
HUMAN_REPORT_SOURCE_FAMILIES = {"mendeley", "faers", "onsides"}
PREGNANCY_EVIDENCE_SOURCES = (
    "onsides",
    "mendeley",
    "faers",
    "tga",
)
EXPECTED_SOURCE_COUNT = len(PREGNANCY_EVIDENCE_SOURCES)
EMPTY_EVIDENCE_COLUMNS = sorted(READ_COLUMNS)


def _normalize(value: Any) -> str:
    return " ".join(str(value).casefold().split()) if pd.notna(value) else ""


def _load_master_evidence(drug_name: str, drug_id: str | None) -> pd.DataFrame:
    if not MASTER_EVIDENCE_PATH.is_file():
        return pd.DataFrame(columns=EMPTY_EVIDENCE_COLUMNS)

    chunks: list[pd.DataFrame] = []
    id_match_found = False
    for chunk in pd.read_csv(
        MASTER_EVIDENCE_PATH,
        usecols=lambda column: column in READ_COLUMNS,
        chunksize=50_000,
        low_memory=False,
    ):
        matches = pd.Series(False, index=chunk.index)
        if drug_id:
            for column in ("drug_id", "canonical_drug_id"):
                if column in chunk.columns:
                    matches |= chunk[column].fillna("").astype(str).eq(str(drug_id))
        else:
            matches = chunk["drug_name"].map(_normalize).eq(_normalize(drug_name))
        if matches.any():
            chunks.append(chunk.loc[matches])
            id_match_found = id_match_found or bool(drug_id)

    if drug_id and not id_match_found:
        for chunk in pd.read_csv(
            MASTER_EVIDENCE_PATH,
            usecols=lambda column: column in READ_COLUMNS,
            chunksize=50_000,
            low_memory=False,
        ):
            matches = chunk["drug_name"].map(_normalize).eq(_normalize(drug_name))
            if matches.any():
                chunks.append(chunk.loc[matches])

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=EMPTY_EVIDENCE_COLUMNS
    )


def _source_family(value: Any) -> str:
    source = _normalize(value)
    if "mendeley" in source:
        return "mendeley"
    if "onsides" in source:
        return "onsides"
    if "faers" in source:
        return "faers"
    if "tga" in source or "medicines-pregnancy" in source:
        return "tga"
    return ""


def _supports_timed_outcome_review(record: pd.Series) -> bool:
    family = _source_family(record.get("source_dataset", ""))
    granularity = _normalize(record.get("evidence_granularity", ""))
    return family == "mendeley" and (
        "human" in granularity or "spontaneous_report" in granularity
    )


def _classify_adverse_level(report_count: int) -> str:
    """Classify adverse effect level as High / Moderate / Low.

    This is exploratory only — derived from report counts, not clinical
    severity. Not a validated medical assessment.
    """
    if report_count >= 10:
        return "High"
    if report_count >= 3:
        return "Moderate"
    return "Low"


def _compute_exploratory_confidence(report_count: int) -> float:
    """Compute an exploratory confidence score (0–1) from report counts.

    NOT clinical confidence — a relative signal strength derived from the
    number of spontaneous reports available for this drug–side-effect pair.
    Clearly labeled as exploratory when displayed.
    """
    return round(min(report_count / 10.0, 1.0), 4)


def _load_side_effects_for_drug(drug_id: str) -> list[dict[str, Any]]:
    """Load reported side effects for a drug from real_adr_evidence.csv.

    Returns a list of dicts with: term, meddra_id, source_dataset,
    report_count, level (High/Moderate/Low), confidence (exploratory 0–1).
    Sorted by report_count descending.
    """
    if not ADR_EVIDENCE_PATH.is_file() or not drug_id:
        return []

    try:
        df = pd.read_csv(
            ADR_EVIDENCE_PATH,
            usecols=lambda column: column
            in {
                "drug_id",
                "adverse_event_term",
                "meddra_id",
                "source_dataset",
                "source_identifier",
            },
        )
        drug_df = df[df["drug_id"].astype(str) == str(drug_id)]
        if drug_df.empty:
            return []

        grouped = (
            drug_df.groupby("adverse_event_term")
            .agg(
                meddra_id=("meddra_id", "first"),
                source_dataset=("source_dataset", lambda x: ", ".join(sorted(set(x)))),
                report_count=("source_identifier", "count"),
            )
            .reset_index()
        )

        side_effects: list[dict[str, Any]] = []
        for _, row in grouped.iterrows():
            term = str(row["adverse_event_term"]) if pd.notna(row["adverse_event_term"]) else ""
            if not term:
                continue
            count = int(row["report_count"])
            side_effects.append(
                {
                    "term": term,
                    "meddra_id": str(row["meddra_id"])
                    if pd.notna(row["meddra_id"])
                    else "",
                    "source_dataset": row["source_dataset"],
                    "report_count": count,
                    "level": _classify_adverse_level(count),
                    "confidence": _compute_exploratory_confidence(count),
                }
            )

        side_effects.sort(key=lambda x: x["report_count"], reverse=True)
        return side_effects
    except Exception:
        return []


def _filter_drug(
    evidence: pd.DataFrame, drug_name: str, drug_id: str | None
) -> pd.DataFrame:
    if evidence.empty:
        return evidence.copy()

    matches = pd.Series(False, index=evidence.index)
    if drug_id:
        for column in ("drug_id", "canonical_drug_id"):
            if column in evidence.columns:
                matches |= evidence[column].fillna("").astype(str).eq(str(drug_id))
    if not matches.any() and "drug_name" in evidence.columns:
        matches = evidence["drug_name"].map(_normalize).eq(_normalize(drug_name))
    return evidence.loc[matches].copy()


def evaluate_drug_safety(
    drug_name: str,
    trimester: str = "All",
    outcome_target: str = "Both",
    *,
    drug_id: str | None = None,
    evidence: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return evidence filtered to one drug, timing, and pregnancy outcome target.

    The coverage value describes the fraction of tracked pregnancy-evidence source
    families represented in the filtered records. It is not a safety score.
    """
    if trimester != "All" and trimester not in TRIMESTER_OPTIONS:
        raise ValueError(f"Unsupported trimester filter: {trimester}")
    if outcome_target not in OUTCOME_TARGET_OPTIONS:
        raise ValueError(f"Unsupported outcome target: {outcome_target}")

    all_evidence = (
        evidence.copy()
        if evidence is not None
        else _load_master_evidence(drug_name, drug_id)
    )
    evidence_source_available = (
        evidence is not None or MASTER_EVIDENCE_PATH.is_file()
    )
    drug_evidence = _filter_drug(all_evidence, drug_name, drug_id)
    if "outcome_category" not in drug_evidence.columns:
        drug_evidence["outcome_category"] = ""

    if outcome_target == "Both":
        allowed_outcomes = set().union(*OUTCOME_CATEGORIES.values())
    else:
        allowed_outcomes = OUTCOME_CATEGORIES[outcome_target]
    outcome_values = drug_evidence["outcome_category"].fillna("").map(_normalize)
    source_values = (
        drug_evidence["source_dataset"].map(_source_family)
        if "source_dataset" in drug_evidence.columns
        else pd.Series("", index=drug_evidence.index)
    )
    filtered = drug_evidence.loc[
        outcome_values.isin(allowed_outcomes)
        & source_values.isin(HUMAN_REPORT_SOURCE_FAMILIES)
    ].copy()

    if trimester != "All":
        selected_timing = TRIMESTER_VALUES[trimester]
        if "trimester" not in filtered.columns:
            filtered = filtered.iloc[0:0]
        else:
            timing_values = filtered["trimester"].fillna("unknown").map(_normalize)
            filtered = filtered.loc[timing_values.eq(selected_timing)]

    if "source_dataset" not in filtered.columns:
        filtered["source_dataset"] = ""
    filtered["source_family"] = filtered["source_dataset"].map(_source_family)
    source_families = sorted(
        family for family in filtered["source_family"].unique() if family
    )
    coverage_score = (
        min(len(source_families) / EXPECTED_SOURCE_COUNT, 1.0)
        if evidence_source_available
        else None
    )
    timed_support_count = sum(
        _supports_timed_outcome_review(record)
        for _, record in filtered.iterrows()
    )

    provenance_columns = [
        column
        for column in (
            "source_dataset",
            "source_url",
            "source_doi",
            "evidence_status",
            "evidence_granularity",
        )
        if column in filtered.columns
    ]
    provenance = (
        filtered[provenance_columns]
        .fillna("")
        .drop_duplicates()
        .to_dict("records")
        if provenance_columns
        else []
    )

    # Determine ML vs AI fallback (Stage 2 input-driven routing)
    ml_prediction_active = True
    # Placeholder variance - in real model would be computed; using None if not available
    sigma = result_variance if (result_variance := None) is not None else None
    # Fallback conditions: low coverage, high variance, or unavailable human-validated source
    fallback_trigger = (
        (coverage_score is not None and coverage_score < 0.40)
        or (sigma is not None and sigma > 0.35)
        or not evidence_source_available
    )
    ai_evidence_synthesis = None
    if fallback_trigger:
        ml_prediction_active = False
        # Identify missing evidence source families
        missing_families = [
            fam for fam in PREGNANCY_EVIDENCE_SOURCES if fam not in source_families
        ]
        # Build synthesis dict with placeholder explanations
        ai_evidence_synthesis = {
            "missing_source_families": missing_families,
            "explanation": (
                "AI fallback activated due to insufficient ML evidence. "
                "Missing human evidence sources include: " + ", ".join(missing_families)
            ),
            "details": {
                "OnSIDES": "Side effect reports from OnSIDES are unavailable for this query.",
                "Mendeley": "Case count data from Mendeley is missing.",
                "TGA": "Regulatory notes from TGA are not present."
            },
        }

    # Deterministic risk scores (foetus trimester-wise + maternal) and
    # dynamic runtime model routing (ML vs AI fallback). Computed per query
    # so results are consistent for identical inputs and routing is not
    # hardcoded per drug.
    drug_risk = score_drug(drug_id) if drug_id else {
        "drug_id": drug_id,
        "drug_name": drug_name,
        "foetus_risk": {"overall_foetus_risk": 0.0, "trimester_wise": {}},
        "maternal_risk": {"maternal_risk": 0.0},
        "overall_confidence": 0.0,
        "model_used": "ai_fallback",
        "ml_available": False,
    }

    confidence = drug_risk["overall_confidence"]
    model_used = drug_risk["model_used"]

    # Side effects for this drug (from input data, exploratory level + confidence)
    side_effects = _load_side_effects_for_drug(drug_id) if drug_id else []

    # --- Dynamic abstention alert (computed from actual input data, not hardcoded) ---
    abstention_alert = None
    if not evidence_source_available:
        abstention_alert = (
            "No human evidence master dataset available. "
            "Run `python -m src.acquire_datasets` to build the evidence base "
            "before evaluating this query."
        )
    elif drug_evidence.empty:
        abstention_alert = (
            f"No evidence records found for '{drug_name}' "
            f"({drug_id or 'unresolved drug id'}). "
            "Cannot assess risk without source data."
        )
    elif filtered.empty:
        if trimester != "All":
            abstention_alert = (
                f"No {outcome_target.lower()} evidence available specifically for "
                f"{trimester} for '{drug_name}'. "
                "Trimester timing is never inferred from untimed records. "
                "Model abstains from risk scoring."
            )
        else:
            abstention_alert = (
                f"No {outcome_target.lower()} evidence found for '{drug_name}'. "
                "Model abstains from risk scoring."
            )
    elif coverage_score is not None and coverage_score < 0.40:
        abstention_alert = (
            f"Insufficient evidence coverage (Q_cov: {coverage_score:.0%}) "
            f"for {trimester}. Only {len(source_families)} of "
            f"{EXPECTED_SOURCE_COUNT} human evidence source families "
            f"are represented for '{drug_name}'. "
            "Model abstains from risk scoring."
        )
    elif confidence < 0.35:
        abstention_alert = (
            f"Prediction confidence below threshold ({confidence:.0%}) "
            f"for {trimester} for '{drug_name}'. "
            "Model abstains from risk scoring."
        )
    elif timed_support_count == 0 and trimester != "All":
        abstention_alert = (
            f"No timed outcome support available for {trimester} "
            f"for '{drug_name}'. "
            "Model abstains from risk scoring."
        )

    return {

        "trimester": trimester,
        "outcome_target": outcome_target,
        "evidence_source_available": evidence_source_available,
        "records": filtered.drop(columns=["source_family"], errors="ignore"),
        "evidence_record_count": len(filtered),
        "evidence_source_families": source_families,
        "evidence_coverage_score": coverage_score,
        "timed_outcome_support_count": timed_support_count,
        "provenance_badges": provenance,
        "abstention_alert": abstention_alert,
        "risk_score": drug_risk["foetus_risk"]["overall_foetus_risk"],
        "model_prediction_variance": sigma,
        "risk_scoring_status": (
            "Exploratory deterministic risk scores; not a clinically validated predictor."
        ),
        "ml_prediction_active": ml_prediction_active,
        "ai_evidence_synthesis": ai_evidence_synthesis,
        # New: foetus risk (trimester-wise), maternal risk, confidence, routing.
        "foetus_risk": drug_risk["foetus_risk"],
        "trimester_wise_risk": drug_risk["foetus_risk"]["trimester_wise"],
        "maternal_risk": drug_risk["maternal_risk"],
        "confidence": drug_risk["overall_confidence"],
        "model_used": drug_risk["model_used"],
        "ml_available": drug_risk["ml_available"],
        "drug_name": drug_risk["drug_name"],
        # Side effects with exploratory adverse level and confidence
        "side_effects": side_effects,
    }
