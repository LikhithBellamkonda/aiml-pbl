"""Compile local ADR, animal and pregnancy evidence without estimating risk.

Run ``python -m src.services.pregnancy_evidence_compiler <drug query>`` to
print a JSON evidence report. Veterinary ADRs and animal study records are
kept separate from human pregnancy outcomes; none is treated as a measured
placental-transfer result or a clinical risk estimate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.services.animal_adr_service import animal_evidence
from src.services.drug_service import DATA_DIR, build_drug_profile, normalise

TRANSFER_MEASUREMENT_FIELDS = {
    "placental_transfer",
    "placental_transfer_measured",
    "maternal_fetal_ratio",
    "fetal_maternal_ratio",
    "cord_blood_concentration",
    "umbilical_cord_concentration",
    "fetal_plasma_concentration",
    "fetal_tissue_concentration",
    "placenta_concentration",
}

TRANSFER_TABLES = (
    "real_adr_evidence.csv",
    "real_toxicity_evidence.csv",
    "real_pregnancy_outcomes.csv",
    "cleaned/animal_adr_mapped.csv",
)

_NEGATED_TRANSFER = re.compile(
    r"\b(?:does\s+not|do\s+not|doesn't|don't|not)\s+"
    r"(?:(?:appear|expected|likely)\s+to\s+)?(?:readily\s+)?"
    r"cross(?:es)?\s+(?:the\s+)?placent(?:a|al barrier)\b"
)
_POSSIBLE_TRANSFER = re.compile(
    r"\b(?:may|might|could|possibly)\s+cross(?:es)?\s+"
    r"(?:the\s+)?placent(?:a|al barrier)\b"
)
_AFFIRMATIVE_TRANSFER = re.compile(
    r"\b(?:crosses|cross|crossing)\s+(?:the\s+)?placent(?:a|al barrier)\b"
)


def classify_narrative_transfer_statement(statement: Any) -> str:
    """Classify literal wording only; this does not verify a transfer claim."""
    if not isinstance(statement, str) or not statement.strip():
        return "no_explicit_transfer_statement"
    text = " ".join(statement.casefold().split())
    if _NEGATED_TRANSFER.search(text):
        return "narrative_says_no_crossing"
    if _POSSIBLE_TRANSFER.search(text):
        return "narrative_suggests_possible_crossing"
    if _AFFIRMATIVE_TRANSFER.search(text):
        return "narrative_says_crossing"
    return "no_explicit_transfer_statement"


def _transfer_measurement_fields(data_dir: Path) -> list[str]:
    fields: list[str] = []
    for relative_path in TRANSFER_TABLES:
        path = data_dir / relative_path
        if not path.is_file():
            continue
        columns = pd.read_csv(path, nrows=0).columns
        for column in columns:
            normalized = re.sub(r"[^a-z0-9]+", "_", str(column).casefold()).strip("_")
            if normalized in TRANSFER_MEASUREMENT_FIELDS:
                values = pd.read_csv(path, usecols=[column], low_memory=False)[column]
                populated = values.notna() & values.astype(str).str.strip().ne("")
                if populated.any():
                    fields.append(f"{relative_path}:{column}")
    return fields


def _narrative_transfer_claims(
    profile: dict[str, Any], data_dir: Path
) -> list[dict[str, Any]]:
    path = data_dir / "pregnancy_classifications.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path, low_memory=False)
    if "Safety statement" not in frame.columns:
        return []

    drug_id = str(profile.get("drug_id") or "")
    rows = frame.iloc[0:0]
    if drug_id and "drug_id" in frame.columns:
        rows = frame[frame["drug_id"].fillna("").astype(str).eq(drug_id)]
    if rows.empty and profile.get("name_key") and "name_key" in frame.columns:
        key = normalise(profile["name_key"])
        rows = frame[frame["name_key"].map(normalise).eq(key)]

    claims: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        statement = row.get("Safety statement")
        status = classify_narrative_transfer_statement(statement)
        if status == "no_explicit_transfer_statement":
            continue
        claims.append(
            {
                "status": status,
                "statement": str(statement),
                "source": "data/pregnancy_classifications.csv",
                "source_citation_available": False,
            }
        )
    return claims


def _not_estimable(reason: str) -> dict[str, Any]:
    return {
        "status": "not_estimable",
        "value": None,
        "confidence": None,
        "reason": reason,
    }


def compile_pregnancy_evidence(
    query: str, data_dir: Path | str = DATA_DIR
) -> dict[str, Any]:
    """Compile source summaries and explicit limits for a resolved drug."""
    data_dir = Path(data_dir)
    profile = build_drug_profile(query, data_dir=data_dir)
    risk_reason = (
        "No validated human pregnancy outcome model, exposure denominator, "
        "or compatible endpoint-specific positive and negative cohort labels "
        "are available locally. Confidence is therefore not estimable."
    )
    if not profile.get("matched") or profile.get("ambiguous"):
        return {
            "query": query,
            "status": "unresolved",
            "drug_id": profile.get("drug_id"),
            "reason": "Resolve one drug identity before compiling linked evidence.",
            "fetal_risk": _not_estimable(risk_reason),
            "maternal_risk": _not_estimable(risk_reason),
            "transfer_assessment": {
                "direct_measurement_status": "not_assessed",
                "measured_fields": [],
                "narrative_claims": [],
            },
        }

    drug_id = str(profile["drug_id"])
    drug_name = str(profile.get("drug_name") or query)
    pregnancy = profile.get("pregnancy") or {}
    toxicity = profile.get("reproductive_toxicity") or {}
    adr = profile.get("adr") or {}
    animal = animal_evidence(drug_id, drug_name, data_dir=data_dir)

    measured_fields = _transfer_measurement_fields(data_dir)
    narrative_claims = _narrative_transfer_claims(profile, data_dir)
    categories = pregnancy.get("categories") or {}
    direct_transfer_available = bool(measured_fields)

    return {
        "query": query,
        "status": "evidence_compiled_risk_unavailable",
        "drug_id": drug_id,
        "drug_name": drug_name,
        "evidence": {
            "human_general_adr": {
                "available": bool(adr.get("available")),
                "distinct_terms": adr.get("term_count", 0),
                "source_datasets": adr.get("source_datasets", []),
                "note": "General ADR terms are not pregnancy-specific.",
            },
            "animal_spontaneous_adr": {
                "available": bool(animal.get("available")),
                "report_count": animal.get("report_count"),
                "species_count": animal.get("species_count"),
                "top_species": animal.get("top_species"),
                "serious_report_count": animal.get("serious_report_count"),
                "note": (
                    "Veterinary spontaneous reports do not establish human "
                    "maternal effects, fetal exposure, or placental transfer."
                ),
            },
            "animal_experimental_toxicity": {
                "available": bool(toxicity.get("available")),
                "record_count": toxicity.get("record_count", 0),
                "species": toxicity.get("species", []),
                "study_types": toxicity.get("study_types", []),
                "note": (
                    "Study records are nonclinical evidence; the local summary "
                    "does not provide placental transfer measurements or a "
                    "validated human risk estimate."
                ),
            },
            "human_pregnancy_outcomes": {
                "available": bool(pregnancy.get("available")),
                "record_count": pregnancy.get("record_count", 0),
                "fetal_records": categories.get("fetal", []),
                "maternal_records": categories.get("maternal", []),
                "sources": pregnancy.get("sources", []),
                "timing": pregnancy.get("trimesters", []),
                "note": (
                    "Aggregate reported outcomes are not incidence and do not "
                    "measure placental transfer."
                ),
            },
        },
        "transfer_assessment": {
            "direct_measurement_status": (
                "measured_transfer_fields_found"
                if direct_transfer_available
                else "no_direct_measurement_field_found_in_checked_tables"
            ),
            "measured_fields": measured_fields,
            "narrative_claims": narrative_claims,
            "interpretation": (
                "Narrative statements are reported verbatim and are not "
                "independently verified. No narrative or animal ADR count is "
                "treated as a measured fetal-transfer result."
            ),
        },
        "fetal_risk": _not_estimable(risk_reason),
        "maternal_risk": _not_estimable(risk_reason),
        "research_priority": profile.get("research_priority"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drug_query", help="Drug name or supported identifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing the processed evidence CSV files",
    )
    args = parser.parse_args()
    report = compile_pregnancy_evidence(args.drug_query, data_dir=args.data_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
