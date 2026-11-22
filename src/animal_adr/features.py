"""Per-drug Animal ADR feature extraction.

Every feature is a count, a period, a completeness measure or a mapping
quality flag taken from reported records.  No incidence, probability or
toxicity score is computed: spontaneous reporting has no exposure
denominator here, and absence of reports is recorded as *unavailable*
evidence rather than a negative label.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MAPPED_PATH = DATA_DIR / "cleaned" / "animal_adr_mapped.csv"
FEATURES_PATH = DATA_DIR / "animal_adr_features.csv"
SUMMARY_PATH = DATA_DIR / "animal_adr_feature_summary.json"

SOURCE_URL = "https://open.fda.gov/apis/animalandveterinary/event/"

SPECIES_FEATURES = (
    "Dog",
    "Cat",
    "Horse",
    "Cattle",
    "Pig",
    "Poultry",
    "Other animal",
)

# Most frequent VeDDRA System Organ Classes in the acquired ADAE snapshot,
# plus one explicit bucket for the long tail.
CATEGORY_FEATURES = (
    "Systemic disorders",
    "Investigations",
    "Digestive tract disorders",
    "Neurological disorders",
    "Skin and appendages disorders",
    "Medication errors and intentional misuse",
    "Reproductive system disorders",
    "Other categories",
)
LACK_OF_EFFICACY_CATEGORY = "Lack of efficacy"
MEDICATION_ERROR_CATEGORY = "Medication errors and intentional misuse"

# Ordered numeric vector consumed by the Animal ADR encoder / model.
NUMERIC_FEATURES = (
    "log_reports",
    "log_reactions",
    "log_species",
    "log_serious_reports",
    "log_death_or_euthanasia_reports",
    "log_lack_of_efficacy_records",
    "log_medication_error_records",
    "log_category_count",
    "record_completeness",
    "mapping_confidence",
    "is_combination_product",
    "log_other_species_reports",
    *[f"share_{name.split(' ')[0].lower()}" for name in CATEGORY_FEATURES],
)

SERIOUS_OUTCOMES = ("died", "euthanized")


def _log_count(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(math.log1p(max(number, 0.0)), 4)


def _share(part: float, total: float) -> float:
    return round(float(part) / float(total), 4) if total else 0.0


def _category_group(category: str) -> str:
    return category if category in CATEGORY_FEATURES else "Other categories"


def _species_group(species: str) -> str:
    return species if species in SPECIES_FEATURES else "Other animal"


def build_features(mapped: pd.DataFrame | None = None) -> dict[str, Any]:
    """Aggregate mapped animal ADR rows into one feature row per drug."""
    if mapped is None:
        if not MAPPED_PATH.exists():
            return {"status": "no_mapped_data"}
        mapped = pd.read_csv(MAPPED_PATH, low_memory=False)

    animal = mapped.loc[
        mapped["standardized_drug_id"].astype(str).str.len().gt(0)
        & mapped["standardized_drug_id"].notna()
    ].copy()
    # Veterinary records flagged as human exposure are not animal evidence.
    animal = animal.loc[~animal["is_human_exposure"].fillna(False)]

    if animal.empty:
        frame = pd.DataFrame(columns=["drug_id", "animal_adr_available"])
        frame.to_csv(FEATURES_PATH, index=False, encoding="utf-8")
        return {"status": "no_mapped_animal_records", "drugs": 0}

    animal["category_group"] = animal["reaction_category"].map(_category_group)
    animal["species_group"] = animal["species"].map(_species_group)
    animal["report_date"] = pd.to_datetime(animal["report_date"], errors="coerce")
    animal["is_serious"] = animal["serious_ae"].astype(str).str.casefold().eq("true")
    outcomes = animal["outcomes"].astype(str).str.casefold()
    animal["is_death"] = outcomes.apply(
        lambda value: any(token in value for token in SERIOUS_OUTCOMES)
    )
    animal["is_lack_of_efficacy"] = animal["reaction_category"].eq(
        LACK_OF_EFFICACY_CATEGORY
    )
    animal["is_medication_error"] = animal["reaction_category"].eq(
        MEDICATION_ERROR_CATEGORY
    )

    rows: list[dict[str, Any]] = []
    for drug_id, group in animal.groupby("standardized_drug_id", sort=False):
        reports = group.drop_duplicates("report_id")
        species_counts = group.groupby("species_group")["report_id"].nunique()
        category_counts = group.groupby("category_group")["report_id"].nunique()
        total_reports = int(reports["report_id"].nunique())
        top_category = (
            category_counts.sort_values(ascending=False).index[0]
            if len(category_counts)
            else ""
        )
        top_species = (
            species_counts.sort_values(ascending=False).index[0] if len(species_counts) else ""
        )
        dates = group["report_date"].dropna()
        row: dict[str, Any] = {
            "drug_id": str(drug_id),
            "animal_adr_available": 1,
            "animal_adr_report_count": total_reports,
            "animal_adr_reaction_count": int(group["reaction_normalized"].nunique()),
            "animal_adr_reaction_records": int(len(group)),
            "animal_adr_species_count": int(group["species"].nunique()),
            "serious_report_count": int(reports.loc[reports["is_serious"], "report_id"].nunique()),
            "death_or_euthanasia_report_count": int(
                reports.loc[reports["is_death"], "report_id"].nunique()
            ),
            "lack_of_efficacy_records": int(group["is_lack_of_efficacy"].sum()),
            "medication_error_records": int(group["is_medication_error"].sum()),
            "reaction_category_count": int(group["reaction_category"].nunique()),
            "top_reaction_category": top_category,
            "top_species": top_species,
            "report_date_min": dates.min().date().isoformat() if len(dates) else "",
            "report_date_max": dates.max().date().isoformat() if len(dates) else "",
            "record_completeness": round(float(group["record_completeness"].mean()), 4),
            "mapping_confidence": round(float(group["mapping_confidence"].max()), 4),
            "mapping_status": "; ".join(sorted(set(group["mapping_status"].dropna().astype(str)))),
            "is_combination_product": int(bool(group["attribution"].astype(str).eq("product_level_only").all())),
            "sources": "FDA CVM ADAE",
            "source_url": SOURCE_URL,
            "evidence_type": "animal_spontaneous_adr",
        }
        for name in SPECIES_FEATURES:
            row[f"species_{name.split(' ')[0].lower()}_reports"] = int(
                species_counts.get(name, 0)
            )
        for name in CATEGORY_FEATURES:
            row[f"category_{name.split(' ')[0].lower()}_reports"] = int(
                category_counts.get(name, 0)
            )
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(
        "animal_adr_report_count", ascending=False
    ).reset_index(drop=True)

    frame["log_reports"] = frame["animal_adr_report_count"].map(_log_count)
    frame["log_reactions"] = frame["animal_adr_reaction_count"].map(_log_count)
    frame["log_species"] = frame["animal_adr_species_count"].map(_log_count)
    frame["log_serious_reports"] = frame["serious_report_count"].map(_log_count)
    frame["log_death_or_euthanasia_reports"] = frame[
        "death_or_euthanasia_report_count"
    ].map(_log_count)
    frame["log_lack_of_efficacy_records"] = frame["lack_of_efficacy_records"].map(_log_count)
    frame["log_medication_error_records"] = frame["medication_error_records"].map(_log_count)
    frame["log_category_count"] = frame["reaction_category_count"].map(_log_count)
    frame["log_other_species_reports"] = frame[
        "species_other_reports"
    ].map(_log_count)
    for name in CATEGORY_FEATURES:
        column = f"category_{name.split(' ')[0].lower()}_reports"
        frame[f"share_{name.split(' ')[0].lower()}"] = [
            _share(value, total)
            for value, total in zip(frame[column], frame["animal_adr_reaction_records"])
        ]

    frame.to_csv(FEATURES_PATH, index=False, encoding="utf-8")

    summary = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "drugs_with_animal_adr_evidence": int(len(frame)),
        "total_reports": int(frame["animal_adr_report_count"].sum()),
        "total_reaction_records": int(frame["animal_adr_reaction_records"].sum()),
        "species": sorted(
            {
                species
                for species in animal["species"].unique()
                if isinstance(species, str) and species
            }
        ),
        "categories": sorted(animal["reaction_category"].unique().tolist()),
        "serious_reports": int(frame["serious_report_count"].sum()),
        "death_or_euthanasia_reports": int(
            frame["death_or_euthanasia_report_count"].sum()
        ),
        "numeric_features": list(NUMERIC_FEATURES),
        "evidence_type": "animal_spontaneous_adr",
        "not_incidence": True,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_features(path: Path | None = None) -> pd.DataFrame:
    """Load the animal ADR feature table indexed by ``drug_id``."""
    path = Path(path or FEATURES_PATH)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, low_memory=False)
    if "drug_id" not in frame.columns:
        return pd.DataFrame()
    frame["drug_id"] = frame["drug_id"].astype(str)
    return frame.set_index("drug_id")


def numeric_vector(row: pd.Series) -> list[float]:
    """Return the fixed-order numeric vector for one drug's animal evidence."""
    return [float(row.get(name, 0.0) or 0.0) for name in NUMERIC_FEATURES]


def species_index(row: pd.Series) -> int:
    """Index of the dominant reported species (0 = unknown/other)."""
    species = str(row.get("top_species", "") or "")
    return SPECIES_FEATURES.index(species) + 1 if species in SPECIES_FEATURES else 0


def category_index(row: pd.Series) -> int:
    """Index of the dominant reported reaction category (0 = unknown)."""
    category = str(row.get("top_reaction_category", "") or "")
    return CATEGORY_FEATURES.index(category) + 1 if category in CATEGORY_FEATURES else 0


def evidence_for_drug(drug_id: str | None, features: pd.DataFrame | None = None) -> dict[str, Any]:
    """Return the recorded animal ADR evidence for one drug, or availability info."""
    features = features if features is not None else load_features()
    if not drug_id or features.empty or str(drug_id) not in features.index:
        return {
            "available": False,
            "drug_id": str(drug_id) if drug_id else "",
            "message": (
                "Animal ADR evidence unavailable. This does not indicate that "
                "the drug is safe."
            ),
            "evidence_type": "animal_spontaneous_adr",
        }
    row = features.loc[str(drug_id)]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return {
        "available": True,
        "drug_id": str(drug_id),
        "evidence_type": "animal_spontaneous_adr",
        "report_count": int(row.get("animal_adr_report_count", 0)),
        "reaction_count": int(row.get("animal_adr_reaction_count", 0)),
        "species_count": int(row.get("animal_adr_species_count", 0)),
        "top_species": str(row.get("top_species", "") or ""),
        "top_reaction_category": str(row.get("top_reaction_category", "") or ""),
        "serious_report_count": int(row.get("serious_report_count", 0)),
        "death_or_euthanasia_report_count": int(
            row.get("death_or_euthanasia_report_count", 0)
        ),
        "record_completeness": float(row.get("record_completeness", 0.0) or 0.0),
        "mapping_confidence": float(row.get("mapping_confidence", 0.0) or 0.0),
        "mapping_status": str(row.get("mapping_status", "") or ""),
        "report_date_min": str(row.get("report_date_min", "") or ""),
        "report_date_max": str(row.get("report_date_max", "") or ""),
        "sources": str(row.get("sources", "") or ""),
        "source_url": str(row.get("source_url", "") or ""),
        "is_combination_product": bool(row.get("is_combination_product", 0)),
        "message": (
            "Reported animal adverse-event evidence. It is not a human "
            "pregnancy risk conclusion, not incidence, and implies no causality."
        ),
    }
