"""Map animal ADR product/ingredient names onto existing project drug IDs.

Resolution uses only authoritative local artefacts already committed to the
project (``data/standardized_drug_master.csv`` and ``data/drug_mapping.csv``),
so a run is reproducible offline and no identifier is invented.

Combination products are preserved as products.  Their ingredients are only
recorded as *matched ingredients* when the string resolves to an existing
drug, and a reaction is never re-attributed as a confirmed causal association
with every ingredient.  Unresolved products are retained in a dedicated
unmatched table rather than dropped.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CLEANED_DIR = DATA_DIR / "cleaned"

MAPPING_PATH = CLEANED_DIR / "animal_adr_drug_mapping.csv"
MAPPED_PATH = CLEANED_DIR / "animal_adr_mapped.csv"
UNMATCHED_PATH = CLEANED_DIR / "animal_adr_unmatched.csv"
OVERLAP_PATH = DATA_DIR / "animal_adr_overlap.csv"

# Presentation words that are not part of an ingredient identity.
FORM_TOKENS = (
    "chewable", "chew", "tablet", "tablets", "injection", "inj", "injectable",
    "suspension", "susp", "solution", "sol", "powder", "oral", "topical",
    "spot on", "spot-on", "drops", "drop", "gel", "paste", "shampoo", "spray",
    "premix", "granules", "bolus", "implants", "implant", "collar", "shampoos",
    "for dogs", "for cats", "for", "weeks", "week", "month", "months",
)

SPLIT_PATTERN = re.compile(r"\s*[;,/+]\s*|\s+\band\b\s+", flags=re.IGNORECASE)


def _normalise(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return unicodedata.normalize("NFKC", str(value)).casefold().strip()


def name_key(value: Any) -> str:
    """Punctuation-insensitive key matching the project's name-key convention."""
    return re.sub(r"[^a-z0-9]+", " ", _normalise(value)).strip()


def clean_ingredient(value: Any) -> str:
    """Drop dosage strengths and presentation words from an ingredient string."""
    text = _normalise(value)
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\d+(\.\d+)?\s*%", " ", text)
    text = re.sub(r"\d+(\.\d+)?\s*(mg|ml|g|kg|iu|mcg)\b", " ", text)
    for token in FORM_TOKENS:
        text = re.sub(rf"\b{re.escape(token)}\b", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_ingredients(value: Any) -> list[str]:
    parts = [clean_ingredient(part) for part in SPLIT_PATTERN.split(_normalise(value))]
    unique: list[str] = []
    for part in parts:
        if part and part not in unique:
            unique.append(part)
    return unique


def _animal_adr_drug_id(product: str, ingredient: str) -> str:
    digest = hashlib.sha1(f"{name_key(product)}|{name_key(ingredient)}".encode()).hexdigest()
    return f"ANIMALADR:{digest[:12]}"


def _alias_index() -> dict[str, list[dict[str, str]]]:
    """Build alias -> candidates from the project's standardised drug tables."""
    index: dict[str, list[dict[str, str]]] = {}

    def add(alias: Any, drug_id: str, source: str) -> None:
        key = name_key(alias)
        if not key or not drug_id:
            return
        entry = {"drug_id": str(drug_id), "mapping_source": source}
        bucket = index.setdefault(key, [])
        if entry not in bucket:
            bucket.append(entry)

    master_path = DATA_DIR / "standardized_drug_master.csv"
    if master_path.exists():
        master = pd.read_csv(master_path, low_memory=False)
        for column in ("drug_name_key", "preferred_name"):
            if column in master.columns:
                for row in master[["drug_id", column]].dropna().to_dict("records"):
                    add(row[column], row["drug_id"], master_path.name)

    mapping_path = DATA_DIR / "drug_mapping.csv"
    if mapping_path.exists():
        mapping = pd.read_csv(mapping_path, low_memory=False)
        if {"drug_id", "original_name"}.issubset(mapping.columns):
            for row in mapping[["drug_id", "original_name"]].dropna().to_dict("records"):
                add(row["original_name"], row["drug_id"], mapping_path.name)
        if {"drug_id", "name_key"}.issubset(mapping.columns):
            for row in mapping[["drug_id", "name_key"]].dropna().to_dict("records"):
                add(row["name_key"], row["drug_id"], mapping_path.name)
    return index


def _match_ingredient(
    ingredient: str, index: dict[str, list[dict[str, str]]]
) -> dict[str, Any]:
    key = name_key(ingredient)
    candidates = index.get(key)
    if candidates and len(candidates) == 1:
        return {
            "standardized_drug_id": candidates[0]["drug_id"],
            "mapping_method": "exact_name",
            "mapping_confidence": 0.95,
            "mapping_status": "matched",
            "mapping_source": candidates[0]["mapping_source"],
            "mapping_candidates": "",
        }
    if candidates and len(candidates) > 1:
        return {
            "standardized_drug_id": "",
            "mapping_method": "exact_name",
            "mapping_confidence": 0.0,
            "mapping_status": "ambiguous",
            "mapping_source": "; ".join(sorted({c["mapping_source"] for c in candidates})),
            "mapping_candidates": "; ".join(sorted({c["drug_id"] for c in candidates})),
        }

    if len(key) >= 5:
        partial = [
            (alias, items)
            for alias, items in index.items()
            if key in alias.split() or alias in key.split() or key == alias
        ]
        distinct_ids = {item["drug_id"] for _, items in partial for item in items}
        if len(distinct_ids) == 1:
            item = partial[0][1][0]
            return {
                "standardized_drug_id": item["drug_id"],
                "mapping_method": "token_match",
                "mapping_confidence": 0.7,
                "mapping_status": "matched",
                "mapping_source": item["mapping_source"],
                "mapping_candidates": "",
            }
        if len(distinct_ids) > 1:
            return {
                "standardized_drug_id": "",
                "mapping_method": "token_match",
                "mapping_confidence": 0.0,
                "mapping_status": "ambiguous",
                "mapping_source": "; ".join(
                    sorted({item["mapping_source"] for _, items in partial for item in items})
                ),
                "mapping_candidates": "; ".join(sorted(distinct_ids)),
            }

    return {
        "standardized_drug_id": "",
        "mapping_method": "none",
        "mapping_confidence": 0.0,
        "mapping_status": "unmatched",
        "mapping_source": "",
        "mapping_candidates": "",
    }


def map_products(
    cleaned: pd.DataFrame,
    index: dict[str, list[dict[str, str]]] | None = None,
) -> pd.DataFrame:
    """Return one mapping row per distinct (product, active ingredient) pair."""
    index = index if index is not None else _alias_index()
    if cleaned.empty:
        return pd.DataFrame(columns=["animal_adr_drug_id", "standardized_drug_id"])

    pairs = (
        cleaned[["product_name", "active_ingredient_original", "is_combination_product"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for record in pairs.to_dict("records"):
        product = str(record.get("product_name") or "")
        ingredient_string = str(record.get("active_ingredient_original") or "")
        ingredients = split_ingredients(ingredient_string)
        is_combination = bool(record.get("is_combination_product")) or len(ingredients) > 1

        if not ingredients and product:
            ingredients = split_ingredients(product)

        matches = [_match_ingredient(item, index) for item in ingredients]
        matched = [m for m in matches if m["mapping_status"] == "matched"]

        if is_combination and matched:
            status = (
                "combination_all_ingredients_matched"
                if len(matched) == len(matches)
                else "combination_partial"
            )
            standardized = ""
            method = "combination_ingredient_match"
            confidence = round(
                min(m["mapping_confidence"] for m in matched) * (len(matched) / len(matches)),
                3,
            )
            candidates = "; ".join(
                m["standardized_drug_id"] for m in matched if m["standardized_drug_id"]
            )
        elif len(matches) == 1 and matches[0]["mapping_status"] == "matched":
            status = "matched"
            standardized = matched[0]["standardized_drug_id"]
            method = matched[0]["mapping_method"]
            confidence = matched[0]["mapping_confidence"]
            candidates = ""
        elif any(m["mapping_status"] == "ambiguous" for m in matches):
            status = "ambiguous"
            standardized = ""
            method = next(
                m["mapping_method"] for m in matches if m["mapping_status"] == "ambiguous"
            )
            confidence = 0.0
            candidates = "; ".join(
                m["mapping_candidates"] for m in matches if m["mapping_candidates"]
            )
        else:
            status = "unmatched"
            standardized = ""
            method = "none"
            confidence = 0.0
            candidates = ""

        sources = sorted({m["mapping_source"] for m in matches if m["mapping_source"]})
        rows.append(
            {
                "join_product_name": product,
                "join_ingredient": ingredient_string,
                "animal_adr_drug_id": _animal_adr_drug_id(product, ingredient_string),
                "standardized_drug_id": standardized,
                "mapping_method": method,
                "mapping_confidence": confidence,
                "mapping_status": status,
                "original_product_name": product,
                "matched_ingredient": "; ".join(
                    item
                    for item, match in zip(ingredients, matches)
                    if match["mapping_status"] == "matched"
                ),
                "mapping_source": "; ".join(sources),
                "mapping_candidates": candidates,
                "is_combination_product": is_combination,
                # A product-level report is never a confirmed causal
                # association for every ingredient it contains.
                "attribution": (
                    "product_level_only" if is_combination else "ingredient_level"
                ),
            }
        )
    return pd.DataFrame(rows)


def _evidence_families() -> dict[str, set[str]]:
    families: dict[str, set[str]] = {}

    def ids(path: Path, column: str = "drug_id") -> set[str]:
        if not path.exists():
            return set()
        frame = pd.read_csv(path, usecols=lambda name: name == column, low_memory=False)
        if column not in frame.columns:
            return set()
        return set(frame[column].dropna().astype(str))

    families["general_adr"] = ids(DATA_DIR / "real_adr_evidence.csv")
    families["pregnancy"] = ids(DATA_DIR / "real_pregnancy_features.csv")
    families["toxicity"] = ids(DATA_DIR / "real_toxicity_evidence.csv")
    families["faers"] = ids(DATA_DIR / "real_faers_summary.csv")
    return families


def write_overlap(mapped: pd.DataFrame) -> pd.DataFrame:
    """Write the drug overlap report across evidence families."""
    animal_ids = set(
        mapped.loc[mapped["standardized_drug_id"].astype(str).str.len() > 0, "standardized_drug_id"]
        .astype(str)
        .unique()
    )
    families = _evidence_families()
    rows = [
        {"metric": "animal_adr_drugs", "value": len(animal_ids)},
        {"metric": "animal_adr_unmatched_products", "value": int(mapped["mapping_status"].eq("unmatched").sum())},
        {"metric": "animal_adr_ambiguous_products", "value": int(mapped["mapping_status"].eq("ambiguous").sum())},
        {"metric": "animal_adr_combination_products", "value": int(mapped["is_combination_product"].sum())},
    ]
    for name, drug_ids in families.items():
        rows.append({"metric": f"{name}_drugs", "value": len(drug_ids)})
        rows.append(
            {"metric": f"animal_adr_shared_with_{name}", "value": len(animal_ids & drug_ids)}
        )
    shared_all = animal_ids & set.intersection(*families.values()) if families else set()
    rows.append({"metric": "animal_adr_shared_with_all_human_families", "value": len(shared_all)})
    overlap = pd.DataFrame(rows)
    overlap.to_csv(OVERLAP_PATH, index=False, encoding="utf-8")
    return overlap


def map_dataset(cleaned: pd.DataFrame | None = None) -> dict[str, Any]:
    """Map the cleaned animal ADR table and write mapping artefacts."""
    if cleaned is None:
        path = CLEANED_DIR / "animal_adr_cleaned.csv"
        if not path.exists():
            return {"status": "no_cleaned_data"}
        cleaned = pd.read_csv(path, low_memory=False)

    mapping = map_products(cleaned)
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(MAPPING_PATH, index=False, encoding="utf-8")

    join_columns = [
        "join_product_name",
        "join_ingredient",
        "animal_adr_drug_id",
        "standardized_drug_id",
        "mapping_method",
        "mapping_confidence",
        "mapping_status",
        "matched_ingredient",
        "mapping_source",
        "mapping_candidates",
        "attribution",
    ]
    merged = cleaned.merge(
        mapping[join_columns],
        left_on=["product_name", "active_ingredient_original"],
        right_on=["join_product_name", "join_ingredient"],
        how="left",
        suffixes=("", "_mapping"),
    ).drop(columns=["join_product_name", "join_ingredient"])
    merged["mapping_status"] = merged["mapping_status"].fillna("unmatched")

    merged.to_csv(MAPPED_PATH, index=False, encoding="utf-8")

    unmatched = merged[
        merged["mapping_status"].isin(["unmatched", "ambiguous"])
        | merged["mapping_status"].isna()
    ]
    unmatched.to_csv(UNMATCHED_PATH, index=False, encoding="utf-8")
    overlap = write_overlap(mapping)

    summary = {
        "mapping_rows": int(len(mapping)),
        "matched_products": int(mapping["mapping_status"].eq("matched").sum()),
        "combination_products": int(mapping["is_combination_product"].sum()),
        "combination_all_ingredients_matched": int(
            mapping["mapping_status"].eq("combination_all_ingredients_matched").sum()
        ),
        "combination_partial": int(mapping["mapping_status"].eq("combination_partial").sum()),
        "ambiguous_products": int(mapping["mapping_status"].eq("ambiguous").sum()),
        "unmatched_products": int(mapping["mapping_status"].eq("unmatched").sum()),
        "unique_standardized_drugs": int(
            mapping.loc[mapping["standardized_drug_id"].astype(str).str.len() > 0, "standardized_drug_id"].nunique()
        ),
        "overlap": dict(zip(overlap["metric"], overlap["value"])),
    }
    (DATA_DIR / "animal_adr_mapping_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
