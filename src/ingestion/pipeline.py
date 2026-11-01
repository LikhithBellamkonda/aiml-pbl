from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion import dryad, mendeley, onsides, pubchem, reprotox
from src.ingestion.common import (
    IngestedSource,
    drug_name_key,
    is_combination_name,
    normalize_drug_name,
    source_drug_id,
    write_records,
)


LOGGER = logging.getLogger("pregnancy_dataset_ingestion.pipeline")
PREGNANCY_OUTCOMES = (
    "congenital_anomaly",
    "miscarriage",
    "preterm_birth",
    "fetal_growth_outcome",
    "stillbirth",
    "maternal_adverse_reaction",
)


def _read_existing_mapping(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "drug_name" not in frame:
        if "original_name" in frame:
            frame = frame.rename(columns={"original_name": "drug_name"})
        else:
            return pd.DataFrame()
    if "drug_name_key" not in frame:
        key_column = "name_key" if "name_key" in frame else None
        frame["drug_name_key"] = (
            frame[key_column].fillna("").astype(str)
            if key_column
            else frame["drug_name"].map(drug_name_key)
        )
    return frame


def _valid_cid(value: Any) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace("CID:", "").replace("CID", "")
    try:
        cid = int(float(text))
    except (TypeError, ValueError):
        return None
    return cid if cid > 0 else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_entity_names(frames: list[pd.DataFrame]) -> list[str]:
    names: set[str] = set()
    for frame in frames:
        for column in ("drug_name", "generic_ingredient_name"):
            if column not in frame:
                continue
            names.update(
                name
                for name in pd.Series(frame[column].dropna().astype(str).unique()).str.strip()
                if name
            )
        if "generic_ingredient_names" in frame:
            for value in frame["generic_ingredient_names"].dropna().astype(str).unique():
                try:
                    ingredients = json.loads(value)
                except json.JSONDecodeError:
                    continue
                if isinstance(ingredients, list):
                    names.update(
                        str(name).strip()
                        for name in ingredients
                        if str(name).strip()
                    )
    return sorted(names, key=str.casefold)


def _merge_existing_chemical_records(
    chemical_frame: pd.DataFrame, existing_mapping: pd.DataFrame
) -> pd.DataFrame:
    if chemical_frame.empty or existing_mapping.empty:
        return chemical_frame
    existing_by_key = {
        drug_name_key(record.get("drug_name", "")): record
        for record in existing_mapping.to_dict("records")
        if drug_name_key(record.get("drug_name", ""))
    }
    result = chemical_frame.copy()
    for index, row in result.iterrows():
        key = drug_name_key(row.get("drug_name", ""))
        cached = existing_by_key.get(key, {})
        cid = _valid_cid(row.get("pubchem_cid")) or _valid_cid(
            cached.get("pubchem_cid")
        )
        if not cid:
            continue
        result.at[index, "pubchem_cid"] = cid
        result.at[index, "drug_id"] = f"CID{cid:09d}"
        result.at[index, "resolution_status"] = (
            row.get("resolution_status")
            if _valid_cid(row.get("pubchem_cid"))
            else cached.get("resolution_status", "resolved_existing_mapping")
        )
        result.at[index, "resolution_source"] = (
            row.get("resolution_source")
            if _valid_cid(row.get("pubchem_cid"))
            else cached.get("resolution_source", "existing project mapping")
        )
        for target, source in (
            ("inchikey", "inchikey"),
            ("smiles", "canonical_smiles"),
            ("molecular_formula", "molecular_formula"),
            ("pubchem_molecular_weight", "molecular_weight"),
        ):
            if pd.isna(row.get(target)) or not str(row.get(target, "")).strip():
                result.at[index, target] = cached.get(source, row.get(target))
    return result


def _make_identifier_map(
    frames_by_source: dict[str, pd.DataFrame],
    existing_mapping: pd.DataFrame,
    pubchem_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    names = _source_entity_names(list(frames_by_source.values()))
    names.extend(
        existing_mapping["drug_name"].dropna().astype(str).tolist()
        if "drug_name" in existing_mapping
        else []
    )
    names = sorted({name.strip() for name in names if name.strip()}, key=str.casefold)

    existing_by_key: dict[str, dict[str, Any]] = {}
    if not existing_mapping.empty:
        for record in existing_mapping.to_dict("records"):
            key = drug_name_key(record.get("drug_name", ""))
            if key and key not in existing_by_key:
                existing_by_key[key] = record

    pubchem_by_key: dict[str, dict[str, Any]] = {}
    for record in pubchem_frame.to_dict("records"):
        key = drug_name_key(record.get("drug_name", ""))
        if key:
            pubchem_by_key[key] = record

    cid_by_key: dict[str, int] = {}
    for key, record in existing_by_key.items():
        cid = _valid_cid(record.get("pubchem_cid"))
        if cid:
            cid_by_key[key] = cid
    for frame in frames_by_source.values():
        if "drug_name" not in frame or "pubchem_cid" not in frame:
            continue
        for record in frame[["drug_name", "pubchem_cid"]].to_dict("records"):
            key = drug_name_key(record.get("drug_name", ""))
            cid = _valid_cid(record.get("pubchem_cid"))
            if key and cid:
                cid_by_key.setdefault(key, cid)
    for key, record in pubchem_by_key.items():
        cid = _valid_cid(record.get("pubchem_cid"))
        if cid:
            cid_by_key.setdefault(key, cid)

    generic_match_keys: set[str] = set()
    for frame in frames_by_source.values():
        if not {"drug_name", "generic_ingredient_name"}.issubset(frame.columns):
            continue
        generic_records = frame[
            ["drug_name", "generic_ingredient_name"]
        ].drop_duplicates()
        if "generic_ingredient_mapping_allowed" in frame:
            allowed = frame[
                [
                    "drug_name",
                    "generic_ingredient_name",
                    "generic_ingredient_mapping_allowed",
                ]
            ].drop_duplicates()
            generic_records = generic_records.merge(
                allowed,
                on=["drug_name", "generic_ingredient_name"],
                how="inner",
            )
        for record in generic_records.to_dict("records"):
            allowed = record.get("generic_ingredient_mapping_allowed", True)
            if pd.isna(allowed) or not bool(allowed):
                continue
            drug_key = drug_name_key(record.get("drug_name", ""))
            ingredient_key = drug_name_key(record.get("generic_ingredient_name", ""))
            cid = cid_by_key.get(ingredient_key)
            if drug_key and ingredient_key and cid and drug_key not in cid_by_key:
                cid_by_key[drug_key] = cid
                generic_match_keys.add(drug_key)

    canonical_by_key = {
        key: f"CID{cid:09d}" if cid else source_drug_id(key)
        for key in {drug_name_key(name) for name in names}
        if key
        for cid in [cid_by_key.get(key)]
    }

    mapping_rows: list[dict[str, Any]] = []
    for source, frame in frames_by_source.items():
        if frame.empty or "drug_name" not in frame:
            continue
        columns = [
            column
            for column in (
                "drug_name",
                "source_identifier",
                "generic_ingredient_name",
                "generic_ingredient_ids",
                "salt_form_review_required",
                "generic_ingredient_mapping_allowed",
            )
            if column in frame
        ]
        for record in frame[columns].drop_duplicates().to_dict("records"):
            name = str(record.get("drug_name", "") or "").strip()
            key = drug_name_key(name)
            if not key:
                continue
            cid = cid_by_key.get(key)
            mapping_rows.append(
                {
                    "source_dataset": source,
                    "source_record_id": record.get("source_record_id", ""),
                    "source_identifier": record.get("source_identifier", ""),
                    "source_drug_name": name,
                    "generic_ingredient_name": record.get(
                        "generic_ingredient_name", ""
                    ),
                    "generic_ingredient_ids": record.get(
                        "generic_ingredient_ids", ""
                    ),
                    "generic_ingredient_mapping_allowed": record.get(
                        "generic_ingredient_mapping_allowed", False
                    ),
                    "source_product_identifier": "",
                    "mapping_role": "drug_record",
                    "salt_form_review_required": record.get(
                        "salt_form_review_required", False
                    ),
                    "normalized_drug_name": normalize_drug_name(name),
                    "drug_name_key": key,
                    "canonical_drug_id": canonical_by_key[key],
                    "pubchem_cid": cid,
                    "match_method": (
                        "rxnorm_single_ingredient_to_pubchem"
                        if key in generic_match_keys
                        else "pubchem_cid"
                        if cid
                        else "exact_normalized_name_unresolved_to_pubchem"
                    ),
                    "is_combination": is_combination_name(name),
                }
            )
        if {"source_identifier", "generic_ingredient_ids", "generic_ingredient_names"}.issubset(frame.columns):
            ingredient_records = frame[
                [
                    "source_identifier",
                    "generic_ingredient_ids",
                    "generic_ingredient_names",
                ]
            ].drop_duplicates()
            for record in ingredient_records.to_dict("records"):
                try:
                    ingredient_ids = json.loads(record["generic_ingredient_ids"])
                    ingredient_names = json.loads(record["generic_ingredient_names"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(ingredient_ids, list) or not isinstance(ingredient_names, list):
                    continue
                for ingredient_id, ingredient_name in zip(ingredient_ids, ingredient_names):
                    key = drug_name_key(ingredient_name)
                    if not key:
                        continue
                    cid = cid_by_key.get(key)
                    mapping_rows.append(
                        {
                            "source_dataset": source,
                            "source_record_id": "",
                            "source_identifier": f"RxNormIngredient:{ingredient_id}",
                            "source_drug_name": ingredient_name,
                            "generic_ingredient_name": ingredient_name,
                            "generic_ingredient_ids": json.dumps(
                                [ingredient_id], ensure_ascii=False
                            ),
                            "generic_ingredient_mapping_allowed": True,
                            "source_product_identifier": record["source_identifier"],
                            "mapping_role": "rxnorm_ingredient_relation",
                            "normalized_drug_name": normalize_drug_name(
                                ingredient_name
                            ),
                            "drug_name_key": key,
                            "canonical_drug_id": canonical_by_key[key],
                            "pubchem_cid": cid,
                            "match_method": (
                                "rxnorm_ingredient_to_pubchem"
                                if cid
                                else "exact_normalized_name_unresolved_to_pubchem"
                            ),
                            "is_combination": False,
                        }
                    )

    for record in existing_mapping.to_dict("records"):
        name = str(record.get("drug_name", "") or "").strip()
        key = drug_name_key(name)
        if not key:
            continue
        cid = cid_by_key.get(key)
        mapping_rows.append(
            {
                "source_dataset": "existing_project_mapping",
                "source_record_id": "",
                "source_identifier": record.get("drug_id", ""),
                "source_drug_name": name,
                "generic_ingredient_name": "",
                "generic_ingredient_ids": "",
                "generic_ingredient_mapping_allowed": False,
                "source_product_identifier": "",
                "mapping_role": "existing_project_mapping",
                "normalized_drug_name": normalize_drug_name(name),
                "drug_name_key": key,
                "canonical_drug_id": canonical_by_key[key],
                "pubchem_cid": cid,
                "match_method": "existing_name_mapping" if cid else "existing_name_unresolved",
                "is_combination": is_combination_name(name),
            }
        )

    identifier_map = pd.DataFrame(mapping_rows).drop_duplicates(
        [
            "source_dataset",
            "source_record_id",
            "source_identifier",
            "source_product_identifier",
            "drug_name_key",
        ]
    )
    preferred_name_by_cid: dict[int, str] = {}
    for record in pubchem_frame.to_dict("records"):
        cid = _valid_cid(record.get("pubchem_cid"))
        preferred_name = record.get("pubchem_title") or record.get("drug_name", "")
        if cid and preferred_name and cid not in preferred_name_by_cid:
            preferred_name_by_cid[cid] = str(preferred_name)

    drug_rows = []
    for key, canonical_id in canonical_by_key.items():
        record = existing_by_key.get(key, {})
        pubchem_record = pubchem_by_key.get(key, {})
        cid = cid_by_key.get(key)
        drug_rows.append(
            {
                "drug_id": canonical_id,
                "drug_name_key": key,
                "preferred_name": preferred_name_by_cid.get(cid)
                or record.get("pubchem_title")
                or pubchem_record.get("pubchem_title")
                or pubchem_record.get("drug_name")
                or record.get("drug_name", ""),
                "pubchem_cid": cid,
                "inchikey": record.get("inchikey") or pubchem_record.get("inchikey", ""),
                "smiles": record.get("canonical_smiles")
                or pubchem_record.get("smiles", ""),
                "resolution_status": (
                    "resolved_pubchem"
                    if cid
                    else pubchem_record.get("resolution_status", "unresolved")
                ),
                "is_combination": (
                    False
                    if cid
                    else is_combination_name(
                        record.get("drug_name")
                        or pubchem_record.get("drug_name")
                        or key
                    )
                ),
            }
        )
    drug_mapping = pd.DataFrame(drug_rows).drop_duplicates("drug_id")
    return identifier_map, drug_mapping, canonical_by_key


def _canonicalize_records(
    frame: pd.DataFrame, canonical_by_key: dict[str, str]
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    if "drug_name" not in result:
        result["drug_name"] = ""
    result["drug_name"] = result["drug_name"].fillna("").astype(str)
    result["drug_name_key"] = result["drug_name"].map(drug_name_key)
    result["canonical_drug_id"] = result["drug_name_key"].map(canonical_by_key)
    if "pubchem_cid" in result:
        normalized_cids = result["pubchem_cid"].map(_valid_cid)
        result["pubchem_cid"] = pd.array(normalized_cids, dtype="Int64")
        for index, value in result["pubchem_cid"].items():
            cid = _valid_cid(value)
            if cid and pd.isna(result.at[index, "canonical_drug_id"]):
                result.at[index, "canonical_drug_id"] = f"CID{cid:09d}"
    result["canonical_drug_id"] = result["canonical_drug_id"].fillna("")
    result["drug_id"] = result["canonical_drug_id"]
    if "is_combination" not in result:
        result["is_combination"] = result["drug_name"].map(is_combination_name)
    else:
        missing = result["is_combination"].isna()
        if missing.any():
            result.loc[missing, "is_combination"] = result.loc[
                missing, "drug_name"
            ].map(is_combination_name)
    return result


def _source_summary(
    sources: dict[str, IngestedSource], canonical_frames: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for source_name, source in sources.items():
        frame = canonical_frames.get(source_name, pd.DataFrame())
        if not frame.empty and source_name == "pubchem" and "pubchem_cid" in frame:
            matched_frame = frame.loc[frame["pubchem_cid"].notna()]
        else:
            matched_frame = frame
        drug_ids = (
            set(matched_frame.loc[matched_frame["drug_id"] != "", "drug_id"])
            if not matched_frame.empty
            else set()
        )
        pubchem_matched_frame = (
            matched_frame if source_name == "pubchem" else frame
        )
        source_record_count = (
            pd.to_numeric(frame.get("source_record_count"), errors="coerce").sum()
            if "source_record_count" in frame
            else len(frame)
        )
        rows.append(
            {
                "source_dataset": source_name,
                "status": source.status,
                "available": source.available,
                "cleaned_record_count": (
                    len(matched_frame) if source_name == "pubchem" else len(frame)
                ),
                "represented_source_record_count": int(
                    len(matched_frame) if source_name == "pubchem" else source_record_count
                ),
                "unique_drug_count": len(drug_ids),
                "pubchem_matched_drug_count": int(
                    pubchem_matched_frame.loc[
                        pubchem_matched_frame["drug_id"]
                        .astype(str)
                        .str.startswith("CID"),
                        "drug_id",
                    ].nunique()
                ) if "drug_id" in frame else 0,
                "source_version": source.source_version,
                "source_url": source.source_url,
                "error": source.error,
            }
        )
    return pd.DataFrame(rows)


def _overlap_summary(canonical_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    source_drugs = {
        name: set(frame.loc[frame["drug_id"] != "", "drug_id"])
        if not frame.empty and "drug_id" in frame
        else set()
        for name, frame in canonical_frames.items()
    }
    rows = []
    for left_index, left in enumerate(source_drugs):
        for right in list(source_drugs)[left_index + 1 :]:
            rows.append(
                {
                    "source_a": left,
                    "source_b": right,
                    "unique_drugs_a": len(source_drugs[left]),
                    "unique_drugs_b": len(source_drugs[right]),
                    "overlapping_drug_count": len(
                        source_drugs[left] & source_drugs[right]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _availability_matrices(
    sources: dict[str, IngestedSource],
    canonical_frames: dict[str, pd.DataFrame],
    drug_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    drugs = (
        drug_mapping[["drug_id", "preferred_name"]]
        .dropna(subset=["drug_id"])
        .drop_duplicates("drug_id")
    )
    source_counts: dict[str, dict[str, int]] = {}
    source_resolution: dict[str, dict[str, str]] = {}
    source_outcomes: dict[str, dict[tuple[str, str], int]] = {}
    for source_name, frame in canonical_frames.items():
        if frame.empty or "drug_id" not in frame:
            source_counts[source_name] = {}
            source_resolution[source_name] = {}
            source_outcomes[source_name] = {}
            continue
        relevant = frame.loc[frame["drug_id"] != ""]
        if source_name == "pubchem" and "resolution_status" in relevant:
            resolution_rows = relevant
            matched_rows = (
                relevant.loc[relevant["pubchem_cid"].notna()]
                if "pubchem_cid" in relevant
                else relevant.iloc[0:0]
            )
            source_resolution[source_name] = (
                resolution_rows.drop_duplicates("drug_id")
                .set_index("drug_id")["resolution_status"]
                .astype(str)
                .to_dict()
            )
            source_counts[source_name] = (
                matched_rows.groupby("drug_id", sort=False)
                .size()
                .astype(int)
                .to_dict()
            )
        else:
            source_resolution[source_name] = {}
            source_counts[source_name] = (
                relevant.groupby("drug_id", sort=False).size().astype(int).to_dict()
            )
        if "outcome_category" in relevant and "evidence_status" in relevant:
            positive = relevant.loc[
                relevant["evidence_status"] == "positive_safety_concern_evidence"
            ]
            source_outcomes[source_name] = (
                positive.groupby(["drug_id", "outcome_category"], sort=False)
                .size()
                .astype(int)
                .to_dict()
            )
        else:
            source_outcomes[source_name] = {}

    availability_rows = []
    outcome_rows = []
    pregnancy_sources = ("mendeley_pregnancy_v2", "dryad_antiepileptic_pregnancy")

    for drug in drugs.to_dict("records"):
        drug_id = drug["drug_id"]
        for source_name, source in sources.items():
            record_count = source_counts.get(source_name, {}).get(drug_id, 0)
            resolution_status = source_resolution.get(source_name, {}).get(drug_id, "")
            status = (
                "source_unavailable"
                if not source.available
                and source.status != "not_queried"
                else "source_not_queried"
                if source.status == "not_queried" or resolution_status == "not_queried"
                else "source_parse_error"
                if source.status == "parse_error"
                else "source_query_error"
                if resolution_status == "lookup_error"
                else "source_identifier_ambiguous"
                if resolution_status == "ambiguous"
                else "identifier_unresolved_combination"
                if resolution_status.startswith("combination")
                else "available"
                if record_count
                else "source_available_no_relevant_record"
            )
            availability_rows.append(
                {
                    "drug_id": drug_id,
                    "drug_name": drug.get("preferred_name", ""),
                    "source_dataset": source_name,
                    "source_status": source.status,
                    "evidence_availability": status,
                    "cleaned_record_count": record_count,
                    "source_available": source.available,
                    "identifier_resolution_status": resolution_status,
                }
            )

        for source_name in pregnancy_sources:
            source = sources[source_name]
            record_count = source_counts.get(source_name, {}).get(drug_id, 0)
            if not source.available or source.status == "parse_error":
                for outcome in PREGNANCY_OUTCOMES:
                    outcome_rows.append(
                        {
                            "drug_id": drug_id,
                            "drug_name": drug.get("preferred_name", ""),
                            "source_dataset": source_name,
                            "outcome": outcome,
                            "evidence_status": "source_unavailable",
                            "observed_positive_concern_records": 0,
                            "observed_negative_records": 0,
                            "validated_binary_label_records": 0,
                        }
                    )
                continue
            for outcome in PREGNANCY_OUTCOMES:
                positive_count = source_outcomes.get(source_name, {}).get(
                    (drug_id, outcome), 0
                )
                evidence_status = (
                    "positive_safety_concern_evidence"
                    if positive_count
                    else "outcome_not_reported"
                    if record_count
                    else "source_available_no_relevant_record"
                )
                outcome_rows.append(
                    {
                        "drug_id": drug_id,
                        "drug_name": drug.get("preferred_name", ""),
                        "source_dataset": source_name,
                        "outcome": outcome,
                        "evidence_status": evidence_status,
                        "observed_positive_concern_records": positive_count,
                        "observed_negative_records": 0,
                        "validated_binary_label_records": 0,
                    }
                )

    return pd.DataFrame(availability_rows), pd.DataFrame(outcome_rows)


def _write_analysis(
    output_dir: Path,
    sources: dict[str, IngestedSource],
    summary: pd.DataFrame,
    overlaps: pd.DataFrame,
    master: pd.DataFrame,
    identifier_map: pd.DataFrame,
    outcome_matrix: pd.DataFrame,
) -> dict[str, Any]:
    data_dir = output_dir / "data"
    report_dir = output_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    outcome_rows = master.loc[
        master.get("outcome_category", pd.Series(dtype=str)).isin(PREGNANCY_OUTCOMES)
    ].copy()
    pregnancy_source_names = (
        "mendeley_pregnancy_v2",
        "dryad_antiepileptic_pregnancy",
    )
    outcome_index = pd.MultiIndex.from_product(
        [pregnancy_source_names, PREGNANCY_OUTCOMES],
        names=["source_dataset", "outcome_category"],
    )
    outcome_summary = pd.DataFrame(index=outcome_index).reset_index()
    if not outcome_rows.empty:
        observed_summary = (
            outcome_rows.groupby(["source_dataset", "outcome_category"], dropna=False)
            .agg(
                evidence_record_count=("source_record_id", "count"),
                positive_concern_record_count=(
                    "evidence_status",
                    lambda values: int((values == "positive_safety_concern_evidence").sum()),
                ),
            )
            .reset_index()
        )
        outcome_summary = outcome_summary.merge(
            observed_summary,
            on=["source_dataset", "outcome_category"],
            how="left",
        )
    else:
        outcome_summary["evidence_record_count"] = 0
        outcome_summary["positive_concern_record_count"] = 0
    outcome_summary["evidence_record_count"] = outcome_summary[
        "evidence_record_count"
    ].fillna(0).astype(int)
    outcome_summary["positive_concern_record_count"] = outcome_summary[
        "positive_concern_record_count"
    ].fillna(0).astype(int)
    outcome_summary["observed_negative_record_count"] = 0
    outcome_summary["validated_positive_label_count"] = 0
    outcome_summary["validated_negative_label_count"] = 0
    outcome_summary["trainable_binary_label_count"] = 0

    trimester_summary = (
        master.groupby(
            ["source_dataset", "trimester"], dropna=False
        ).size().rename("record_count").reset_index()
        if "trimester" in master
        else pd.DataFrame(columns=["source_dataset", "trimester", "record_count"])
    )
    if master.empty:
        missingness = pd.DataFrame(columns=["field", "missing_count", "missing_percent"])
    else:
        missingness = pd.DataFrame(
            [
                {
                    "field": column,
                    "missing_count": int(
                        (master[column].isna() | master[column].eq("").fillna(False)).sum()
                    ),
                    "missing_percent": round(
                        100
                        * (master[column].isna() | master[column].eq("").fillna(False)).sum()
                        / len(master),
                        4,
                    ),
                }
                for column in master.columns
            ]
        ).sort_values("missing_percent", ascending=False)

    matching_summary = identifier_map.groupby("source_dataset", dropna=False).agg(
        source_identifier_rows=("source_record_id", "size"),
        unique_source_drugs=("drug_name_key", "nunique"),
    ).reset_index()
    matched = (
        identifier_map.loc[identifier_map["pubchem_cid"].notna()]
        .groupby("source_dataset")["drug_name_key"]
        .nunique()
        .rename("pubchem_matched_drugs")
    )
    matching_summary = matching_summary.merge(
        matched, on="source_dataset", how="left"
    )
    matching_summary["pubchem_matched_drugs"] = matching_summary[
        "pubchem_matched_drugs"
    ].fillna(0).astype(int)
    matching_summary["pubchem_match_percent"] = (
        100
        * matching_summary["pubchem_matched_drugs"]
        / matching_summary["unique_source_drugs"].replace(0, pd.NA)
    ).fillna(0).round(2)
    source_registry = pd.DataFrame(
        [
            {
                "source_dataset": "onsides_general_adr",
                "citation_or_doi": "OnSIDES release v3.1.1",
                "version": "v3.1.1",
                "license_or_access": "Review upstream data and release terms before redistribution.",
                "granularity": "regulatory product-label adverse-effect associations",
                "limitation": "No pregnancy outcome labels; source prediction scores are not confidence.",
            },
            {
                "source_dataset": "mendeley_pregnancy_v2",
                "citation_or_doi": mendeley.DOI,
                "version": "Version 2",
                "license_or_access": "CC BY 4.0 (verified against official metadata)",
                "granularity": "drug-level aggregate spontaneous-report counts",
                "limitation": "No patient-level outcomes, exposed-pregnancy denominator, or validated negative labels.",
            },
            {
                "source_dataset": "dryad_antiepileptic_pregnancy",
                "citation_or_doi": dryad.DOI,
                "version": dryad.DOI,
                "license_or_access": "CC0 1.0 (verified against Dryad metadata)",
                "granularity": "study-level aggregate supplementary tables",
                "limitation": "No patient-level observations; ASD/ADHD analyses are not the requested pregnancy endpoints.",
            },
            {
                "source_dataset": "pubchem",
                "citation_or_doi": "PubChem PUG REST",
                "version": "retrieval time",
                "license_or_access": "PubChem public chemical reference service",
                "granularity": "chemical identifiers and structures",
                "limitation": "Identity/features only; not outcome evidence.",
            },
            {
                "source_dataset": "reprotox_kg",
                "citation_or_doi": "ReproTox-KG graph serialization",
                "version": "version field retained from graph when available",
                "license_or_access": "License is not stated in the downloaded graph; review upstream terms.",
                "granularity": "curated/computational knowledge-graph associations",
                "limitation": "Associations are not observed patient outcomes or incidence estimates.",
            },
        ]
    )
    source_registry = source_registry.merge(
        summary[["source_dataset", "status", "cleaned_record_count", "unique_drug_count"]],
        on="source_dataset",
        how="left",
    )

    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No records."
        lines = [
            "| " + " | ".join(str(column) for column in frame.columns) + " |",
            "| " + " | ".join("---" for _ in frame.columns) + " |",
        ]
        lines.extend(
            "| "
            + " | ".join(
                "" if pd.isna(value) else str(value).replace("|", "\\|")
                for value in row
            )
            + " |"
            for row in frame.itertuples(index=False, name=None)
        )
        return "\n".join(lines)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_count": len(sources),
        "source_statuses": {
            name: {
                "status": source.status,
                "available": source.available,
                "error": source.error,
                "version": source.source_version,
                "url": source.source_url,
            }
            for name, source in sources.items()
        },
        "master_record_count": len(master),
        "unique_mapped_drug_count": int(master.loc[master["drug_id"] != "", "drug_id"].nunique())
        if not master.empty
        else 0,
        "outcome_matrix_label_counts": {
            "positive": int(outcome_matrix["observed_positive_concern_records"].sum())
            if not outcome_matrix.empty
            else 0,
            "negative": 0,
            "validated_binary": 0,
        },
        "target_decisions": [
            {
                "outcome": outcome,
                "decision": "not_trainable",
                "positive_labels": 0,
                "negative_labels": 0,
                "reason": (
                    "Sources provide aggregate spontaneous-report counts, report/label associations, "
                    "or study-level evidence, but no validated human drug-level binary observations "
                    "with compatible positive and negative outcomes. Missing reports remain unknown."
                ),
            }
            for outcome in PREGNANCY_OUTCOMES
        ],
    }

    summary.to_csv(data_dir / "dataset_summary.csv", index=False)
    source_registry.to_csv(data_dir / "source_registry.csv", index=False)
    overlaps.to_csv(data_dir / "drug_overlap.csv", index=False)
    outcome_summary.to_csv(data_dir / "outcome_summary.csv", index=False)
    trimester_summary.to_csv(data_dir / "trimester_summary.csv", index=False)
    missingness.to_csv(data_dir / "missingness_summary.csv", index=False)
    matching_summary.to_csv(data_dir / "identifier_matching_summary.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        ax = summary.set_index("source_dataset")["unique_drug_count"].plot(
            kind="bar", figsize=(10, 5), color="#4178a4"
        )
        ax.set_ylabel("Unique mapped drugs")
        ax.set_title("Drug coverage by source")
        ax.figure.tight_layout()
        ax.figure.savefig(figures_dir / "drug_coverage.png", dpi=150)
        plt.close(ax.figure)
    if not trimester_summary.empty:
        trimester_pivot = trimester_summary.pivot_table(
            index="source_dataset", columns="trimester", values="record_count", aggfunc="sum", fill_value=0
        )
        ax = trimester_pivot.plot(kind="bar", stacked=True, figsize=(10, 5))
        ax.set_ylabel("Cleaned record count")
        ax.set_title("Recorded exposure timing (unknown is not inferred)")
        ax.figure.tight_layout()
        ax.figure.savefig(figures_dir / "trimester_coverage.png", dpi=150)
        plt.close(ax.figure)
    if not outcome_summary.empty:
        ax = outcome_summary.pivot_table(
            index="outcome_category",
            columns="source_dataset",
            values="evidence_record_count",
            aggfunc="sum",
            fill_value=0,
        ).plot(kind="bar", figsize=(10, 5))
        ax.set_ylabel("Aggregate evidence records (not labels)")
        ax.set_title("Pregnancy outcome evidence coverage")
        ax.figure.tight_layout()
        ax.figure.savefig(figures_dir / "pregnancy_outcome_coverage.png", dpi=150)
        plt.close(ax.figure)
    if not matching_summary.empty:
        ax = matching_summary.set_index("source_dataset")[
            "pubchem_match_percent"
        ].plot(kind="bar", figsize=(10, 5), color="#4d9b78", ylim=(0, 100))
        ax.set_ylabel("Unique drug names matched to PubChem CID (%)")
        ax.set_title("Chemical identifier matching")
        ax.figure.tight_layout()
        ax.figure.savefig(figures_dir / "identifier_matching.png", dpi=150)
        plt.close(ax.figure)
    if not missingness.empty:
        top_missing = missingness.head(15).sort_values("missing_percent")
        ax = top_missing.set_index("field")["missing_percent"].plot(
            kind="barh", figsize=(10, 6), color="#b7794b", xlim=(0, 100)
        )
        ax.set_xlabel("Missing or blank values (%)")
        ax.set_title("Master evidence missingness (top 15 fields)")
        ax.figure.tight_layout()
        ax.figure.savefig(figures_dir / "master_missingness.png", dpi=150)
        plt.close(ax.figure)

    decisions = "\n".join(
        f"- **{item['outcome']} — {item['decision']}:** {item['reason']}"
        for item in report["target_decisions"]
    )
    sources_table = markdown_table(
        summary[
            [
                "source_dataset",
                "status",
                "cleaned_record_count",
                "unique_drug_count",
                "pubchem_matched_drug_count",
            ]
        ]
    )
    overlap_table = markdown_table(overlaps)
    outcome_table = markdown_table(
        outcome_summary[
            [
                "source_dataset",
                "outcome_category",
                "evidence_record_count",
                "positive_concern_record_count",
                "observed_negative_record_count",
                "validated_positive_label_count",
                "validated_negative_label_count",
            ]
        ]
    )
    matching_table = markdown_table(matching_summary)
    trimester_table = markdown_table(trimester_summary)
    missingness_table = markdown_table(missingness.head(15))
    registry_table = markdown_table(
        source_registry[
            [
                "source_dataset",
                "citation_or_doi",
                "version",
                "license_or_access",
                "granularity",
                "limitation",
            ]
        ]
    )
    report_text = f"""# Dataset integration and feasibility report

Generated: {report['generated_at_utc']}

## Result

The source-specific ingestion pipeline completed without creating training labels. It preserves source granularity and distinguishes unknown outcomes from explicitly reported evidence. No pregnancy endpoint is currently trainable.

## Source coverage

{sources_table}

## Provenance and access

{registry_table}

## Pairwise drug overlap

{overlap_table}

Name matching uses normalized exact names and existing/PubChem CID mappings. An OnSIDES product maps to PubChem through an RxNorm ingredient only when exactly one ingredient is documented and no differing salt form is detected; multiple-ingredient products remain intact with their ingredient list and are not split into outcome records. Salt names are not stripped. Unresolved names retain stable `NAME:` identifiers. The source mapping table records original identifiers, ingredient relations, and match methods.

## Pregnancy prediction targets

{decisions}

### Outcome evidence and label distribution

{outcome_table}

No negative labels are inferred from absent reports, omitted endpoints, or missing files. The Dryad supplement is an aggregate DOCX (tables of cohort characteristics, ICD codes, and study-level ASD/ADHD hazard-ratio analyses), not individual participant outcomes. Its ASD/ADHD endpoints are distinct from the six pregnancy endpoints and are not converted into labels.

The Mendeley source contains drug-level aggregate spontaneous-report counts without exposed-pregnancy denominators or patient-level non-events. These are reporting signals only, not incidence or validated binary labels. OnSIDES contributes regulatory product-label adverse-effect associations and source model scores; those scores are not confidence estimates or pregnancy outcome labels. ReproTox-KG birth-defect edges are curated/computational knowledge-graph associations, not observed pregnancy events. PubChem contributes chemical identity and descriptors only.

## Potential leakage

The version, DOI, data-use note, evidence granularity, and limitations for each source are recorded in `../data/source_registry.csv`. Exact canonical-drug overlaps are reported in `../data/drug_overlap.csv`; this measures entity overlap, not shared patients. No individual-level cohort data are available here to test patient overlap or deduplicate outcomes across studies. Drug-level entity overlap is expected across chemical and safety sources, so any future model evaluation must separate canonical drug IDs (and consider scaffold grouping) across train/test sets. Do not use Mendeley report counts as features when evaluating labels derived from the same reports or cohort.

## Identifier matching

{matching_table}

## Trimester and missingness

### Recorded trimester availability

{trimester_table}

### Master missingness (top 15 fields)

{missingness_table}

Trimester values are used only when explicitly documented as a trimester or gestational weeks; otherwise they remain `unknown`. See `../data/trimester_summary.csv` and `../data/missingness_summary.csv`. The evidence availability matrices distinguish `source_unavailable`, `source_available_no_relevant_record`, `outcome_not_reported`, and positive concern evidence. No source currently provides validated negative human outcome observations.

## Additional data required

For each target endpoint, obtain a documented human pregnancy cohort or controlled study with:

1. Patient- or pregnancy-level exposure and follow-up records, or validated drug-level event and non-event counts with denominators.
2. Endpoint-specific definitions and observed positive and negative outcomes; do not merge unrelated endpoints.
3. Drug exposure timing, gestational weeks/trimester when documented, comparator definition, and relevant confounders.
4. Linkable drug identifiers and sufficient independent drugs/cohorts for leakage-resistant drug/scaffold-separated evaluation.
5. Provenance, validation procedures, license/permissions, and a clear missing-data mechanism.

Until then, keep the existing pregnancy-model training safeguard enabled. Do not report accuracy, recall, precision, calibration, or confidence for a pregnancy predictor.
"""
    (report_dir / "dataset_integration_feasibility.md").write_text(
        report_text, encoding="utf-8"
    )
    (data_dir / "dataset_integration_feasibility.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def run_pipeline(
    *,
    root: Path,
    download_missing: bool = False,
    max_pubchem_queries: int = 0,
) -> dict[str, Any]:
    root = Path(root).resolve()
    dataset_dir = root / "datasets"
    raw_dir = dataset_dir / "raw"
    output_dir = root
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "onsides_general_adr": raw_dir / "onsides-v3.1.1.zip",
        "mendeley_pregnancy_v2": dataset_dir / "FAERS_JAAD_Mendeley Supplement.docx",
        "dryad_antiepileptic_pregnancy": dataset_dir / dryad.EXPECTED_FILENAME,
        "reprotox_kg": raw_dir / "reprotox_serialization.valid.json",
    }
    acquire_functions = {
        "onsides_general_adr": onsides.acquire,
        "mendeley_pregnancy_v2": mendeley.acquire,
        "dryad_antiepileptic_pregnancy": dryad.acquire,
        "reprotox_kg": reprotox.acquire,
    }
    acquisition_errors: dict[str, str] = {}
    if download_missing:
        for source_name, acquire in acquire_functions.items():
            if paths[source_name].is_file():
                continue
            try:
                acquire(paths[source_name])
            except Exception as error:
                LOGGER.exception("Acquisition failed for %s", source_name)
                acquisition_errors[source_name] = str(error)

    sources: dict[str, IngestedSource] = {
        "onsides_general_adr": onsides.ingest(paths["onsides_general_adr"]),
    }
    mendeley_source, mendeley_totals = mendeley.ingest(
        paths["mendeley_pregnancy_v2"]
    )
    if not mendeley_totals.empty:
        mendeley_totals = mendeley_totals.copy()
        mendeley_totals["source_dataset"] = "Mendeley Data"
        mendeley_totals["source_record_id"] = mendeley_totals.apply(
            lambda row: f"total:{row.get('source_table', '')}:{row.get('drug_name', '')}",
            axis=1,
        )
        mendeley_totals["source_identifier"] = ""
        mendeley_totals["drug_name_key"] = mendeley_totals["drug_name"].map(drug_name_key)
        mendeley_totals["outcome_category"] = ""
        mendeley_totals["outcome_term_original"] = ""
        mendeley_totals["outcome_value_original"] = mendeley_totals["reports"]
        mendeley_totals["evidence_status"] = "aggregate_report_total_context"
        mendeley_totals["evidence_granularity"] = "drug_level_aggregate_spontaneous_reports"
        mendeley_totals["trimester"] = "unknown"
        mendeley_totals["gestational_weeks"] = pd.NA
        mendeley_totals["is_combination"] = mendeley_totals["drug_name"].map(
            is_combination_name
        )
    sources["mendeley_pregnancy_v2"] = mendeley_source
    sources["dryad_antiepileptic_pregnancy"] = dryad.ingest(
        paths["dryad_antiepileptic_pregnancy"], dryad.DOI
    )
    sources["reprotox_kg"] = reprotox.ingest(paths["reprotox_kg"])
    for source_name, acquisition_error in acquisition_errors.items():
        existing_error = sources[source_name].error
        sources[source_name].error = (
            f"Acquisition failed: {acquisition_error}"
            + (f"; ingestion status: {existing_error}" if existing_error else "")
        )

    frames_by_source = {
        source_name: source.records.copy()
        for source_name, source in sources.items()
    }
    if not mendeley_totals.empty:
        frames_by_source["mendeley_pregnancy_v2"] = pd.concat(
            [frames_by_source["mendeley_pregnancy_v2"], mendeley_totals],
            ignore_index=True,
            sort=False,
        )
    existing_mapping = _read_existing_mapping(root / "data" / "drug_mapping.csv")
    names = _source_entity_names(list(frames_by_source.values()))
    names.extend(
        existing_mapping["drug_name"].dropna().astype(str).tolist()
        if "drug_name" in existing_mapping
        else []
    )
    pubchem_source = pubchem.ingest_names(
        sorted({name for name in names if name.strip()}, key=str.casefold),
        offline=max_pubchem_queries <= 0,
        max_queries=max_pubchem_queries,
    )
    pubchem_source.records = _merge_existing_chemical_records(
        pubchem_source.records, existing_mapping
    )
    resolved_pubchem_count = (
        int(pubchem_source.records["pubchem_cid"].notna().sum())
        if not pubchem_source.records.empty
        and "pubchem_cid" in pubchem_source.records
        else 0
    )
    pubchem_source.available = bool(
        resolved_pubchem_count or max_pubchem_queries > 0
    )
    pubchem_source.status = (
        "available_cached"
        if resolved_pubchem_count and max_pubchem_queries <= 0
        else "available"
        if resolved_pubchem_count or max_pubchem_queries > 0
        else "not_queried"
    )
    sources["pubchem"] = pubchem_source
    frames_by_source["pubchem"] = pubchem_source.records.copy()

    identifier_map, drug_mapping, canonical_by_key = _make_identifier_map(
        frames_by_source, existing_mapping, pubchem_source.records
    )
    canonical_frames = {
        name: _canonicalize_records(frame, canonical_by_key)
        for name, frame in frames_by_source.items()
    }

    data_dir = root / "data"
    clean_dir = data_dir / "cleaned"
    clean_dir.mkdir(parents=True, exist_ok=True)
    for source_name, frame in canonical_frames.items():
        write_records(frame, clean_dir / f"{source_name}.csv")
    write_records(
        mendeley_source.records, clean_dir / "mendeley_outcome_records.csv"
    )
    write_records(mendeley_totals, clean_dir / "mendeley_report_totals.csv")
    write_records(canonical_frames["pubchem"], data_dir / "pubchem_chemicals.csv")
    write_records(identifier_map, data_dir / "drug_identifier_mapping.csv")
    write_records(drug_mapping, data_dir / "standardized_drug_master.csv")

    master_sources = (
        "onsides_general_adr",
        "mendeley_pregnancy_v2",
        "dryad_antiepileptic_pregnancy",
        "reprotox_kg",
    )
    master = pd.concat(
        [canonical_frames[name] for name in master_sources if not canonical_frames[name].empty],
        ignore_index=True,
        sort=False,
    )
    write_records(master, data_dir / "standardized_master_evidence.csv")
    availability, outcome_matrix = _availability_matrices(
        sources, canonical_frames, drug_mapping
    )
    write_records(availability, data_dir / "evidence_availability_matrix.csv")
    write_records(outcome_matrix, data_dir / "outcome_evidence_matrix.csv")

    summary = _source_summary(sources, canonical_frames)
    overlaps = _overlap_summary(canonical_frames)
    report = _write_analysis(
        root,
        sources,
        summary,
        overlaps,
        master,
        identifier_map,
        outcome_matrix,
    )

    manifest_sources = {}
    for source_name, source in sources.items():
        source_files = []
        for file_name in source.source_files:
            path = Path(file_name)
            if path.is_file():
                source_files.append(
                    {
                        "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": _file_sha256(path),
                    }
                )
        manifest_sources[source_name] = {
            "status": source.status,
            "available": source.available,
            "source_url": source.source_url,
            "source_version": source.source_version,
            "error": source.error,
            "acquisition_error": acquisition_errors.get(source_name, ""),
            "source_files": source_files,
        }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "acquisition_requested": download_missing,
        "max_pubchem_queries": max_pubchem_queries,
        "sources": manifest_sources,
    }
    (data_dir / "dataset_acquisition_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
