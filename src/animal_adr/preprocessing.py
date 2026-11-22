"""Preprocess FDA CVM animal adverse-event reports into analysis tables.

The output is a long table with one row per (report, product, reaction).  The
original reported reaction and the normalised reaction/SOC category are kept
side by side.  Optional fields are never required: a report missing a reaction
description, species or dates is retained and marked, not silently dropped.

Spontaneous reports are *not* controlled toxicity experiments and counts are
never converted into incidence, probability or a safety label.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.animal_adr import acquisition

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CLEANED_DIR = DATA_DIR / "cleaned"

LOGGER = logging.getLogger("meprank.animal_adr.preprocessing")

SOURCE_NAME = "FDA CVM ADAE"

# Species vocabulary: spelling/presentation variants mapped to one category.
# A value is only grouped when the wording is unambiguous.
SPECIES_SYNONYMS = {
    "dog": "Dog",
    "dogs": "Dog",
    "canine": "Dog",
    "canines": "Dog",
    "puppy": "Dog",
    "puppies": "Dog",
    "cat": "Cat",
    "cats": "Cat",
    "feline": "Cat",
    "kitten": "Cat",
    "kittens": "Cat",
    "horse": "Horse",
    "horses": "Horse",
    "equine": "Horse",
    "foal": "Horse",
    "pony": "Horse",
    "cattle": "Cattle",
    "cow": "Cattle",
    "cows": "Cattle",
    "bovine": "Cattle",
    "calf": "Cattle",
    "calves": "Cattle",
    "pig": "Pig",
    "pigs": "Pig",
    "swine": "Pig",
    "porcine": "Pig",
    "hog": "Pig",
    "chicken": "Poultry",
    "chickens": "Poultry",
    "poultry": "Poultry",
    "hen": "Poultry",
    "rooster": "Poultry",
    "turkey": "Poultry",
    "ferret": "Ferret",
    "ferrets": "Ferret",
    "guinea pig": "Guinea Pig",
    "guinea pigs": "Guinea Pig",
    "rat": "Rodent",
    "rats": "Rodent",
    "mouse": "Rodent",
    "mice": "Rodent",
    "rodent": "Rodent",
    "rabbit": "Rabbit",
    "rabbits": "Rabbit",
    "goat": "Goat",
    "goats": "Goat",
    "sheep": "Sheep",
    "lamb": "Sheep",
    "donkey": "Equine (other)",
    "mule": "Equine (other)",
    "human": "Human",
    "humans": "Human",
    "other": "Other",
}

# Reaction categories used when the VeDDRA SOC cannot be joined (e.g. a term
# text that was edited by the reporter). Keyword rules only; they never
# override a real VeDDRA SOC.
REACTION_KEYWORD_CATEGORIES = (
    ("Lack of efficacy", ("lack of efficacy", "no effect", "ineffective", "efficacy")),
    (
        "Medication errors and intentional misuse",
        ("medication error", "wrong dose", "overdose", "misuse", "off label"),
    ),
    (
        "Digestive tract disorders",
        ("vomit", "diarrh", "nausea", "anorexia", "salivat", "gastro", "abdomin", "stool"),
    ),
    (
        "Neurological disorders",
        ("letharg", "tremor", "seizure", "ataxia", "convuls", "neuro", "paresis", "stupor"),
    ),
    (
        "Skin and appendages disorders",
        ("prurit", "alopecia", "dermat", "rash", "erythema", "urticaria", "skin", "hair"),
    ),
    (
        "Respiratory tract disorders",
        ("dyspn", "respirat", "cough", "tachypn", "pneumon"),
    ),
    (
        "Cardio-vascular system disorders",
        ("cardiac", "heart", "tachycard", "bradycard", "circulat"),
    ),
    (
        "Hepato-biliary disorders",
        ("hepat", "liver", "jaundice", "bile"),
    ),
    (
        "Renal and urinary disorders",
        ("renal", "kidney", "urin", "polyur", "azot"),
    ),
    (
        "Reproductive system disorders",
        ("reproduct", "abortion", "stillbirth", "miscarr", "mammary", "oestr", "semen"),
    ),
    (
        "Musculoskeletal disorders",
        ("lameness", "muscle", "arthr", "bone", "myopath"),
    ),
    (
        "Blood and lymphatic system disorders",
        ("anaem", "anem", "thrombocyt", "coagul", "haemat", "hemat"),
    ),
    (
        "Systemic disorders",
        ("death", "died", "euthan", "collapse", "fever", "pyrex", "pain", "oedema", "edema"),
    ),
    ("Injury and poisoning", ("injury", "poison", "toxic", "ingest", "trauma")),
)

OUTPUT_COLUMNS = (
    "source",
    "report_id",
    "report_date",
    "onset_date",
    "domestic_or_foreign",
    "primary_reporter",
    "report_type",
    "serious_ae",
    "outcomes",
    "species_original",
    "species",
    "is_human_exposure",
    "breed",
    "gender",
    "age",
    "reproductive_status",
    "number_of_animals_affected",
    "number_of_animals_treated",
    "product_name",
    "active_ingredient_original",
    "is_combination_product",
    "manufacturer",
    "route",
    "atc_vet_code",
    "dose",
    "first_exposure_date",
    "last_exposure_date",
    "used_according_to_label",
    "reaction_term_original",
    "reaction_normalized",
    "reaction_code",
    "reaction_category",
    "reaction_animals_affected",
    "reaction_accuracy",
    "missing_reaction_description",
    "record_completeness",
)


def _as_text(value: Any) -> str:
    """Render scalar, list or nested value as readable, non-fabricated text."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(text for text in (_as_text(item) for item in value) if text)
    if isinstance(value, dict):
        return "; ".join(
            f"{key}={_as_text(item)}" for key, item in value.items() if _as_text(item)
        )
    text = str(value).strip()
    return "" if text.casefold() in {"none", "nan", "unknown"} else text


def _iso_date(value: Any) -> str:
    """Convert FDA ``YYYYMMDD`` dates to ISO; keep unparseable values blank."""
    text = _as_text(value)
    digits = re.sub(r"\D", "", text)
    if len(digits) != 8:
        return ""
    try:
        return datetime.strptime(digits, "%Y%m%d").date().isoformat()
    except ValueError:
        return ""


def normalise_species(value: Any) -> str:
    """Map a reported species string onto the controlled category vocabulary."""
    text = _as_text(value).casefold().strip()
    if not text:
        return "Unknown"
    if text in SPECIES_SYNONYMS:
        return SPECIES_SYNONYMS[text]
    for token, category in SPECIES_SYNONYMS.items():
        if token in text:
            return category
    return text.title()


def normalise_reaction(value: Any) -> str:
    """Lowercase and strip reporter commentary, keeping the term itself."""
    text = _as_text(value).casefold()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _veddra_lookup(veddra: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if veddra.empty or "soc" not in veddra.columns:
        return lookup
    for _, row in veddra.iterrows():
        soc = str(row.get("soc") or "").strip()
        if not soc:
            continue
        for column in ("pt", "llt"):
            term = normalise_reaction(row.get(column, ""))
            if term:
                lookup.setdefault(term, soc)
    return lookup


def reaction_category(term: Any, lookup: dict[str, str] | None = None) -> str:
    """Return the VeDDRA SOC, else a transparent keyword category."""
    lookup = lookup or {}
    keys = (normalise_reaction(term), normalise_reaction(_as_text(term).split("(")[0]))
    for key in keys:
        if key and key in lookup:
            return lookup[key]
    folded = _as_text(term).casefold()
    for category, tokens in REACTION_KEYWORD_CATEGORIES:
        if any(token in folded for token in tokens):
            return category
    return "Unclassified"


def _completeness(values: list[Any]) -> float:
    return round(sum(1 for value in values if _as_text(value)) / max(len(values), 1), 4)


def flatten_reports(
    records: list[dict[str, Any]],
    veddra: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Flatten raw ADAE reports to (report, product, reaction) rows."""
    lookup = _veddra_lookup(veddra if veddra is not None else pd.DataFrame())
    rows: list[dict[str, Any]] = []

    for record in records:
        animal = record.get("animal") or {}
        reactions = record.get("reaction") or []
        drugs = record.get("drug") or []
        if not reactions:
            reactions = [{}]
        if not drugs:
            drugs = [{}]

        base = {
            "source": SOURCE_NAME,
            "report_id": _as_text(record.get("unique_aer_id_number")),
            "report_date": _iso_date(record.get("original_receive_date")),
            "onset_date": _iso_date(record.get("onset_date")),
            "domestic_or_foreign": _as_text(record.get("foreign_or_domestic")),
            "primary_reporter": _as_text(record.get("primary_reporter")),
            "report_type": _as_text(record.get("type_of_information")),
            "serious_ae": _as_text(record.get("serious_ae")),
            "outcomes": _as_text(
                [item.get("medical_status") for item in (record.get("outcome") or [])]
            ),
            "species_original": _as_text(animal.get("species")),
            "species": normalise_species(animal.get("species")),
            "is_human_exposure": normalise_species(animal.get("species")) == "Human",
            "breed": _as_text(animal.get("breed")),
            "gender": _as_text(animal.get("gender")),
            "age": _as_text(animal.get("age")),
            "reproductive_status": _as_text(animal.get("reproductive_status")),
            "number_of_animals_affected": _as_text(record.get("number_of_animals_affected")),
            "number_of_animals_treated": _as_text(record.get("number_of_animals_treated")),
        }

        for drug in drugs:
            ingredient = _as_text(
                [item.get("name") for item in (drug.get("active_ingredients") or [])]
            )
            product_name = _as_text(drug.get("brand_name"))
            drug_row = {
                **base,
                "product_name": product_name,
                "active_ingredient_original": ingredient,
                "is_combination_product": bool(
                    re.search(r"[;,/]|\band\b", ingredient, flags=re.IGNORECASE)
                ),
                "manufacturer": _as_text((drug.get("manufacturer") or {}).get("name")),
                "route": _as_text(drug.get("route")),
                "atc_vet_code": _as_text(drug.get("atc_vet_code")),
                "dose": _as_text(drug.get("dose")),
                "first_exposure_date": _iso_date(drug.get("first_exposure_date")),
                "last_exposure_date": _iso_date(drug.get("last_exposure_date")),
                "used_according_to_label": _as_text(drug.get("used_according_to_label")),
            }
            for reaction in reactions:
                term = _as_text(reaction.get("veddra_term_name"))
                row = {
                    **drug_row,
                    "reaction_term_original": term,
                    "reaction_normalized": normalise_reaction(term),
                    "reaction_code": _as_text(reaction.get("veddra_term_code")),
                    "reaction_category": reaction_category(term, lookup),
                    "reaction_animals_affected": _as_text(
                        reaction.get("number_of_animals_affected")
                    ),
                    "reaction_accuracy": _as_text(reaction.get("accuracy")),
                    "missing_reaction_description": not bool(term),
                }
                row["record_completeness"] = _completeness(
                    [
                        row["report_date"],
                        row["species_original"],
                        row["product_name"],
                        row["active_ingredient_original"],
                        row["route"],
                        row["outcomes"],
                        row["serious_ae"],
                        row["reaction_term_original"],
                    ]
                )
                rows.append(row)

    frame = pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
    return frame


def _deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    key = [
        "report_id",
        "product_name",
        "active_ingredient_original",
        "reaction_code",
        "reaction_normalized",
    ]
    before = len(frame)
    deduped = frame.drop_duplicates(subset=key, keep="first")
    return deduped, {
        "rows_before_dedup": before,
        "duplicate_rows_removed": before - len(deduped),
        "duplicate_report_ids": int(frame["report_id"].duplicated().sum()),
    }


def preprocess(
    records: list[dict[str, Any]] | None = None,
    veddra: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Clean the animal ADR source and write every derived table."""
    if records is None:
        records = acquisition.read_fda_reports()
    if veddra is None:
        veddra = acquisition.read_veddra()
    if not records:
        summary = {
            "status": "no_records",
            "reason": "No FDA ADAE records were available in datasets/.",
        }
        (DATA_DIR / "animal_adr_dataset_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    flattened = flatten_reports(records, veddra)
    cleaned, dedup = _deduplicate(flattened)

    # Report-level integrity only: a row must belong to an identified report.
    # Optional clinical fields are never required.
    valid = cleaned["report_id"].astype(str).str.len() > 0
    invalid_rows = int((~valid).sum())
    cleaned = cleaned.loc[valid].reset_index(drop=True)

    for column in ("report_date", "onset_date", "first_exposure_date", "last_exposure_date"):
        cleaned[column] = cleaned[column].fillna("")

    cleaned.to_csv(CLEANED_DIR / "animal_adr_cleaned.csv", index=False, encoding="utf-8")

    animal_rows = cleaned.loc[~cleaned["is_human_exposure"]]
    species_summary = (
        animal_rows.groupby("species")
        .agg(
            reports=("report_id", "nunique"),
            reactions=("reaction_normalized", "nunique"),
            reaction_records=("reaction_normalized", "size"),
        )
        .reset_index()
        .sort_values("reports", ascending=False)
    )
    species_summary.to_csv(
        CLEANED_DIR / "animal_adr_species_summary.csv", index=False, encoding="utf-8"
    )

    reaction_summary = (
        cleaned.groupby(["reaction_category", "reaction_normalized"])
        .agg(
            reports=("report_id", "nunique"),
            animals_affected=("reaction_animals_affected", "count"),
        )
        .reset_index()
        .sort_values("reports", ascending=False)
    )
    reaction_summary.to_csv(
        CLEANED_DIR / "animal_adr_reaction_summary.csv", index=False, encoding="utf-8"
    )

    optional_fields = (
        "report_date",
        "onset_date",
        "species_original",
        "breed",
        "gender",
        "age",
        "reproductive_status",
        "product_name",
        "active_ingredient_original",
        "manufacturer",
        "route",
        "atc_vet_code",
        "outcomes",
        "serious_ae",
        "reaction_term_original",
    )
    missing_rates = {
        field: round(float(cleaned[field].astype(str).str.len().eq(0).mean()), 4)
        for field in optional_fields
    }
    dates = pd.to_datetime(cleaned["report_date"], errors="coerce").dropna()
    quality = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": SOURCE_NAME,
        "raw_reports": len(records),
        "flattened_rows": int(len(flattened)),
        "cleaned_rows": int(len(cleaned)),
        **dedup,
        "invalid_rows_removed": invalid_rows,
        "unique_reports": int(cleaned["report_id"].nunique()),
        "unique_products": int(cleaned["product_name"].nunique()),
        "unique_active_ingredients": int(cleaned["active_ingredient_original"].nunique()),
        "unique_reaction_terms": int(cleaned["reaction_normalized"].nunique()),
        "missing_reaction_description_rows": int(
            cleaned["missing_reaction_description"].sum()
        ),
        "unclassified_reaction_rows": int(
            cleaned["reaction_category"].eq("Unclassified").sum()
        ),
        "human_exposure_reports": int(
            cleaned.loc[cleaned["is_human_exposure"], "report_id"].nunique()
        ),
        "combination_product_rows": int(cleaned["is_combination_product"].sum()),
        "report_date_min": dates.min().date().isoformat() if len(dates) else None,
        "report_date_max": dates.max().date().isoformat() if len(dates) else None,
        "mean_record_completeness": round(float(cleaned["record_completeness"].mean()), 4),
        "missing_value_rates": missing_rates,
        "veddra_terms_loaded": int(len(veddra)),
        "notes": [
            "Spontaneous reports only; counts are not incidence and imply no causality.",
            "Human-flagged exposures are retained but excluded from animal species counts.",
            "Missing reaction descriptions are retained and flagged, never imputed.",
        ],
    }
    (DATA_DIR / "animal_adr_data_quality_report.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )

    summary = {
        "generated_at": quality["generated_at"],
        "source": SOURCE_NAME,
        "raw_reports": quality["raw_reports"],
        "cleaned_rows": quality["cleaned_rows"],
        "unique_reports": quality["unique_reports"],
        "unique_species": int(
            animal_rows["species"].nunique() + (1 if cleaned["is_human_exposure"].any() else 0)
        ),
        "animal_species": sorted(animal_rows["species"].unique().tolist()),
        "unique_reaction_terms": quality["unique_reaction_terms"],
        "unique_reaction_categories": int(cleaned["reaction_category"].nunique()),
        "serious_report_count": int(
            cleaned.loc[cleaned["serious_ae"].str.casefold() == "true", "report_id"].nunique()
        ),
        "report_date_range": [quality["report_date_min"], quality["report_date_max"]],
        "evidence_type": "animal_spontaneous_adr",
    }
    (DATA_DIR / "animal_adr_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
