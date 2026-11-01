from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from zipfile import ZipFile
from xml.etree import ElementTree

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET_DIR = ROOT / "datasets"
PUG_PROPERTIES = (
    "Title,ConnectivitySMILES,InChIKey,MolecularFormula,MolecularWeight"
)
OUTCOME_NAMES = (
    "fetal_death",
    "preterm_birth",
    "fetal_growth_restriction",
    "congenital_anomaly",
)
LOGGER = logging.getLogger("process_real_datasets")
WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def normalize_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"\s+", " ", text)


def _name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_name(value)).strip()


def _internal_name_id(name_key: str) -> str:
    return "NAME:" + urllib.parse.quote(name_key, safe="")


def drug_id_for_resolution(name_key: str, record: dict[str, Any]) -> str:
    cid = record.get("cid")
    return _cid_id(cid) if cid else _internal_name_id(name_key)


def _cid_id(cid: Any) -> str:
    number = int(cid)
    if number <= 0:
        raise ValueError(f"Invalid PubChem CID: {cid}")
    return f"CID{number:09d}"


def _is_combination(name: str) -> bool:
    return bool(re.search(r"\s/\s|\s\+\s|\s+\band\b\s+", normalize_name(name)))


def _pubchem_record_from_local_csv(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    records: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        try:
            cid = int(row["CID"])
        except (KeyError, TypeError, ValueError):
            continue
        name = "aspirin" if cid == 2244 else str(row.get("Title", ""))
        records[_name_key(name)] = {
            "cid": cid,
            "title": name,
            "smiles": row.get("ConnectivitySMILES") or row.get("SMILES"),
            "inchikey": "",
            "molecular_formula": row.get("MolecularFormula", ""),
            "molecular_weight": row.get("MolecularWeight", np.nan),
            "resolution_status": "resolved_local_source",
            "resolution_source": path.name,
        }
    return records


def _query_pubchem(name: str, retries: int = 3) -> dict[str, Any]:
    encoded_name = urllib.parse.quote(name, safe="")
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded_name}/property/{PUG_PROPERTIES}/JSON"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PregnancyDrugSafetyResearchPrototype/1.0"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            properties = payload.get("PropertyTable", {}).get("Properties", [])
            unique = {int(item["CID"]): item for item in properties if item.get("CID")}
            if not unique:
                return {"resolution_status": "unresolved", "resolution_source": "PubChem PUG REST"}
            if len(unique) != 1:
                return {
                    "resolution_status": "ambiguous",
                    "resolution_source": "PubChem PUG REST",
                }
            item = next(iter(unique.values()))
            cid = int(item["CID"])
            if cid <= 0:
                return {"resolution_status": "invalid_identifier", "resolution_source": "PubChem PUG REST"}
            return {
                "cid": cid,
                "title": item.get("Title", ""),
                "smiles": item.get("ConnectivitySMILES", ""),
                "inchikey": item.get("InChIKey", ""),
                "molecular_formula": item.get("MolecularFormula", ""),
                "molecular_weight": item.get("MolecularWeight", np.nan),
                "resolution_status": "resolved",
                "resolution_source": "PubChem PUG REST",
            }
        except urllib.error.HTTPError as error:
            if error.code in (400, 404):
                return {"resolution_status": "unresolved", "resolution_source": "PubChem PUG REST"}
            if attempt + 1 == retries:
                return {
                    "resolution_status": "lookup_error",
                    "resolution_source": "PubChem PUG REST",
                    "error": f"HTTP {error.code}",
                }
            time.sleep(min(2**attempt, 8))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt + 1 == retries:
                return {
                    "resolution_status": "lookup_error",
                    "resolution_source": "PubChem PUG REST",
                    "error": str(error),
                }
            time.sleep(min(2**attempt, 8))
    return {"resolution_status": "lookup_error", "resolution_source": "PubChem PUG REST"}


def resolve_names(
    names: list[str],
    offline: bool = False,
    max_queries: int | None = None,
    cache_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    cache_path = cache_path or DATA_DIR / "pubchem_cache.json"
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            LOGGER.warning("Ignoring invalid PubChem cache %s: %s", cache_path, error)

    local_records = _pubchem_record_from_local_csv(DATASET_DIR / "CSV.csv")
    keys = sorted({_name_key(name) for name in names if _name_key(name)})
    new_queries = 0
    results: dict[str, dict[str, Any]] = {}
    queries: list[tuple[str, str]] = []
    for key in keys:
        candidates = [name for name in names if _name_key(name) == key]
        name = candidates[0]
        if key in cache and cache[key].get("resolution_status") != "lookup_error":
            results[key] = cache[key]
            continue
        if key in cache:
            del cache[key]
        if _is_combination(name):
            record = {
                "resolution_status": "combination_not_resolved",
                "resolution_source": "not queried: combination product",
            }
        elif key in local_records and offline:
            record = local_records[key]
        elif offline or (max_queries is not None and new_queries >= max_queries):
            record = {
                "resolution_status": "not_queried",
                "resolution_source": "offline or lookup limit",
            }
        else:
            new_queries += 1
            queries.append((key, name))
            continue
        results[key] = record

    next_query_time = [0.0]
    query_lock = threading.Lock()

    def rate_limited_query(name: str) -> dict[str, Any]:
        with query_lock:
            delay = 0.22 - (time.monotonic() - next_query_time[0])
            if delay > 0:
                time.sleep(delay)
            next_query_time[0] = time.monotonic()
        return _query_pubchem(name)

    def save_cache() -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    completed_queries = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_key = {
            executor.submit(rate_limited_query, name): key for key, name in queries
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            record = future.result()
            results[key] = record
            cache[key] = record
            completed_queries += 1
            if completed_queries % 50 == 0:
                save_cache()

    save_cache()
    LOGGER.info("PubChem requests performed: %s", new_queries)
    return results


def _extract_faers_outcomes(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    with ZipFile(path) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))

    outcome_rows: list[dict[str, Any]] = []
    total_rows: list[dict[str, Any]] = []
    tables = document.findall(".//w:tbl", WORD_NS)
    for table_number, table in enumerate(tables):
        rows: list[list[str]] = []
        for row in table.findall("./w:tr", WORD_NS):
            cells = []
            for cell in row.findall("./w:tc", WORD_NS):
                text = " ".join(
                    "".join(node.text or "" for node in paragraph.findall(".//w:t", WORD_NS))
                    for paragraph in cell.findall("./w:p", WORD_NS)
                ).strip()
                cells.append(text)
            rows.append(cells)

        current_drug: str | None = None
        reports: int | None = None
        outcomes_total: int | None = None
        for cells in rows:
            if not cells:
                continue
            heading = cells[0].strip()
            match = re.match(
                r"(.+?)\s+\(N\s*=\s*([\d,]+)(?:,\s*n\s*=\s*([\d,]+))?\)",
                heading,
                flags=re.IGNORECASE,
            )
            if match is None:
                match = re.match(r"(.+?)\s+\(N\s*=\s*n\s*=\s*([\d,]+)\)", heading, re.I)
                if match:
                    current_drug = match.group(1).strip()
                    reports = outcomes_total = int(match.group(2).replace(",", ""))
            elif match:
                current_drug = match.group(1).strip()
                reports = int(match.group(2).replace(",", ""))
                outcomes_total = (
                    int(match.group(3).replace(",", "")) if match.group(3) else None
                )
            if current_drug and reports is not None and heading.startswith(current_drug + " "):
                total_rows.append(
                    {
                        "drug_name": current_drug,
                        "reports": reports,
                        "outcomes": outcomes_total,
                        "source_table": table_number,
                    }
                )
                continue
            if not current_drug or len(cells) < 2:
                continue
            group = heading
            outcome_key = _classify_outcome(group)
            details = cells[1]
            for item in details.split(";"):
                item = item.strip()
                term_match = re.match(r"(.+?)\s*\(([\d,]+)\)\s*$", item)
                if not item:
                    continue
                outcome_rows.append(
                    {
                        "drug_name": current_drug,
                        "outcome_group": group,
                        "outcome": outcome_key,
                        "reported_term": term_match.group(1).strip() if term_match else item,
                        "reported_count": (
                            int(term_match.group(2).replace(",", "")) if term_match else np.nan
                        ),
                        "source_table": table_number,
                        "source_file": path.name,
                        "evidence_type": "aggregate spontaneous-report summary",
                    }
                )

    outcomes = pd.DataFrame(outcome_rows).drop_duplicates()
    totals = pd.DataFrame(total_rows).drop_duplicates(subset=["drug_name", "source_table"])
    return outcomes, totals


def _classify_outcome(group: str) -> str:
    value = normalize_name(group)
    if "fetal death" in value or "foetal death" in value:
        return "fetal_death"
    if "prematur" in value or "preterm" in value:
        return "preterm_birth"
    if "growth restriction" in value or "low birth weight" in value:
        return "fetal_growth_restriction"
    if any(token in value for token in ("congenital", "malformation", "defect", "anomal")):
        return "congenital_anomaly"
    return ""


def _process_adr(resolved_by_drug_id: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = DATASET_DIR / "meddra_all_se.tsv.gz"
    columns = [
        "stitch_stereo",
        "stitch_flat",
        "umls_id",
        "concept_type",
        "meddra_id",
        "side_effect_name",
    ]
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    sider = pd.read_csv(path, sep="\t", names=columns, dtype=str)
    sider["pubchem_id"] = sider["stitch_flat"].map(_sider_cid)
    cid_to_drug_ids: dict[int, list[str]] = {}
    for drug_id, record in resolved_by_drug_id.items():
        if record.get("cid"):
            cid_to_drug_ids.setdefault(int(record["cid"]), []).append(drug_id)

    evidence = []
    for row in sider.itertuples(index=False):
        if pd.isna(row.pubchem_id):
            continue
        for drug_id in cid_to_drug_ids.get(int(row.pubchem_id), []):
            evidence.append(
                {
                    "drug_id": drug_id,
                    "pubchem_cid": int(row.pubchem_id),
                    "adverse_event_term": row.side_effect_name,
                    "meddra_id": row.meddra_id,
                    "concept_type": row.concept_type,
                    "source_dataset": "SIDER",
                    "source_identifier": row.stitch_flat,
                }
            )
    evidence_df = pd.DataFrame(evidence).drop_duplicates()
    top_terms = (
        evidence_df["adverse_event_term"].value_counts().head(20).index.tolist()
        if not evidence_df.empty
        else []
    )
    drug_ids = list(resolved_by_drug_id)
    features = []
    for drug_id in drug_ids:
        terms = (
            set(evidence_df.loc[evidence_df["drug_id"] == drug_id, "adverse_event_term"])
            if not evidence_df.empty
            else set()
        )
        row = {
            "drug_id": drug_id,
            "has_adr_data": int(bool(terms)),
            "adr_data_status": "available" if terms else "not_matched_in_SIDER",
            "quality_adr": float(bool(terms)),
        }
        row.update({f"ADR_{index}": float(term in terms) for index, term in enumerate(top_terms)})
        for index in range(len(top_terms), 20):
            row[f"ADR_{index}"] = 0.0
        features.append(row)
    return pd.DataFrame(features), evidence_df


def _sider_cid(value: Any) -> int | None:
    text = str(value).strip()
    if not text.startswith("CID"):
        return None
    digits = text[4:] if text.startswith("CID1") else text[3:]
    try:
        cid = int(digits)
    except ValueError:
        return None
    return cid if cid > 0 else None


def process_datasets(offline: bool = False, max_pubchem_queries: int | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pregnancy_path = DATASET_DIR / "medicines-pregnancy-current-database-2026-10-21.csv"
    if not pregnancy_path.exists():
        raise FileNotFoundError(f"Required local dataset not found: {pregnancy_path}")
    pregnancy = pd.read_csv(pregnancy_path, on_bad_lines="skip")
    required = {"Name", "Category"}
    if not required.issubset(pregnancy.columns):
        raise ValueError(f"Pregnancy classification file is missing columns: {required - set(pregnancy.columns)}")

    pregnancy["original_name"] = pregnancy["Name"].fillna("").astype(str).str.strip()
    pregnancy["normalized_name"] = pregnancy["original_name"].map(normalize_name)
    pregnancy["name_key"] = pregnancy["original_name"].map(_name_key)
    names = pregnancy.loc[pregnancy["name_key"] != "", "original_name"].drop_duplicates().tolist()
    pubchem = resolve_names(names, offline=offline, max_queries=max_pubchem_queries)

    map_rows = []
    for key, group in pregnancy[pregnancy["name_key"] != ""].groupby("name_key", sort=True):
        record = pubchem.get(key, {"resolution_status": "not_queried"})
        cid = record.get("cid")
        drug_id = drug_id_for_resolution(key, record)
        for original_name in group["original_name"].drop_duplicates():
            map_rows.append(
                {
                    "original_name": original_name,
                    "normalized_name": normalize_name(original_name),
                    "name_key": key,
                    "drug_id": drug_id,
                    "pubchem_cid": int(cid) if cid else np.nan,
                    "pubchem_title": record.get("title", ""),
                    "inchikey": record.get("inchikey", ""),
                    "canonical_smiles": record.get("smiles", ""),
                    "resolution_status": record.get("resolution_status", "unresolved"),
                    "resolution_source": record.get("resolution_source", ""),
                }
            )
    mapping = pd.DataFrame(map_rows)
    mapping.to_csv(DATA_DIR / "drug_mapping.csv", index=False)
    mapping[["original_name", "drug_id"]].rename(
        columns={"original_name": "drug_name"}
    ).drop_duplicates().to_csv(DATA_DIR / "drug_names.csv", index=False)
    mapping[mapping["pubchem_cid"].isna()].to_csv(
        DATA_DIR / "unmatched_drugs.csv", index=False
    )

    pubchem_rows = []
    for drug_id, group in mapping.groupby("drug_id", sort=False):
        row = group.iloc[0]
        pubchem_rows.append(
            {
                "drug_id": drug_id,
                "pubchem_cid": row["pubchem_cid"],
                "inchikey": row["inchikey"],
                "smiles": row["canonical_smiles"],
                "title": row["pubchem_title"],
                "resolution_status": row["resolution_status"],
                "molecular_formula": pubchem.get(row["name_key"], {}).get("molecular_formula", ""),
                "molecular_weight": pubchem.get(row["name_key"], {}).get("molecular_weight", np.nan),
                "quality_descriptors": _metadata_completeness(
                    [
                        row["pubchem_cid"],
                        row["inchikey"],
                        row["canonical_smiles"],
                        pubchem.get(row["name_key"], {}).get("molecular_formula"),
                        pubchem.get(row["name_key"], {}).get("molecular_weight"),
                    ]
                ),
                "quality_graph": float(
                    pd.notna(row["pubchem_cid"])
                    and isinstance(row["canonical_smiles"], str)
                    and bool(row["canonical_smiles"])
                    and bool(row["inchikey"])
                ),
            }
        )
    pubchem_df = pd.DataFrame(pubchem_rows)
    pubchem_df.to_csv(DATA_DIR / "real_pubchem.csv", index=False)

    resolved_by_drug_id = {}
    for row in mapping.to_dict("records"):
        record = pubchem.get(row["name_key"], {})
        if row["drug_id"].startswith("CID") and record.get("cid"):
            resolved_by_drug_id[row["drug_id"]] = record

    adr_df, adr_evidence = _process_adr(resolved_by_drug_id)
    if adr_df.empty:
        adr_df = pd.DataFrame(columns=["drug_id", "has_adr_data", "adr_data_status"])
    adr_df.to_csv(DATA_DIR / "real_faers_adr.csv", index=False)
    adr_evidence.to_csv(DATA_DIR / "real_adr_evidence.csv", index=False)

    tox_source_path = DATASET_DIR / "toxrefdb_3_0_study_chem_summary.xlsx"
    if tox_source_path.exists():
        tox_source = pd.read_excel(tox_source_path)
    else:
        tox_source = pd.DataFrame()
    tox_features, tox_evidence = _process_toxref(tox_source, mapping)
    tox_features.to_csv(DATA_DIR / "real_toxicity.csv", index=False)
    tox_evidence.to_csv(DATA_DIR / "real_toxicity_evidence.csv", index=False)

    outcomes, totals = _extract_faers_outcomes(
        DATASET_DIR / "FAERS_JAAD_Mendeley Supplement.docx"
    )
    if not outcomes.empty:
        outcomes["name_key"] = outcomes["drug_name"].map(_name_key)
        outcomes = outcomes.merge(
            mapping[["name_key", "drug_id"]].drop_duplicates(),
            on="name_key",
            how="left",
            validate="many_to_one",
        )
        outcomes.to_csv(DATA_DIR / "real_pregnancy_outcomes.csv", index=False)
    else:
        outcomes = pd.DataFrame(
            columns=[
                "drug_id",
                "drug_name",
                "outcome_group",
                "outcome",
                "reported_term",
                "reported_count",
            ]
        )
        outcomes.to_csv(DATA_DIR / "real_pregnancy_outcomes.csv", index=False)
    if not totals.empty:
        totals["name_key"] = totals["drug_name"].map(_name_key)
        totals = totals.merge(
            mapping[["name_key", "drug_id"]].drop_duplicates(),
            on="name_key",
            how="left",
            validate="many_to_one",
        )
        totals = totals.groupby("drug_id", as_index=False).agg(
            faers_report_count=("reports", "max"),
            faers_outcome_count=("outcomes", "max"),
            faers_source_name=("drug_name", "first"),
        )
        totals["has_faers_data"] = 1
        totals["quality_faers"] = 1.0
    else:
        totals = pd.DataFrame(
            columns=[
                "drug_id",
                "faers_report_count",
                "faers_outcome_count",
                "faers_source_name",
                "has_faers_data",
                "quality_faers",
            ]
        )
    totals.to_csv(DATA_DIR / "real_faers_summary.csv", index=False)

    classifications = pregnancy.copy()
    classifications["name_key"] = classifications["Name"].map(_name_key)
    classifications = classifications.merge(
        mapping[["name_key", "drug_id"]].drop_duplicates(),
        on="name_key",
        how="left",
        validate="many_to_one",
    )
    classifications.to_csv(DATA_DIR / "pregnancy_classifications.csv", index=False)
    pregnancy_features = _pregnancy_features(outcomes)
    pregnancy_features.to_csv(DATA_DIR / "real_pregnancy_features.csv", index=False)
    _write_source_coverage(
        mapping, adr_df, pubchem_df, tox_features, pregnancy_features, totals
    )

    LOGGER.info("Drug-name records: %d", len(mapping))
    LOGGER.info(
        "PubChem resolution: %d resolved, %d unresolved/other",
        int(mapping["pubchem_cid"].notna().sum()),
        int(mapping["pubchem_cid"].isna().sum()),
    )
    LOGGER.info(
        "SIDER matched drug rows: %d; pregnancy FAERS summary drugs: %d; ToxRef exact-name-linked study rows: %d",
        int(adr_df["has_adr_data"].sum()) if "has_adr_data" in adr_df else 0,
        len(totals),
        len(tox_evidence),
    )


def _process_toxref(source: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = ["drug_id", "has_tox_data", "quality_tox"] + [
        f"Study_{study_type}" for study_type in ("DEV", "CHR", "SUB", "MGR", "SAC")
    ]
    if source.empty or "preferred_name" not in source.columns:
        return pd.DataFrame(columns=feature_columns), pd.DataFrame()
    source = source.copy()
    source["name_key"] = source["preferred_name"].map(_name_key)
    mapping_by_key = mapping[["name_key", "drug_id"]].drop_duplicates()
    evidence = source.merge(mapping_by_key, on="name_key", how="inner", validate="many_to_one")
    evidence["evidence_type"] = "chemical toxicity study record; no effect outcome in summary"
    evidence["source_dataset"] = "EPA ToxRefDB v3.0"
    requested = [
        "drug_id",
        "chemical_id",
        "dsstox_substance_id",
        "casrn",
        "preferred_name",
        "study_id",
        "study_citation",
        "study_year",
        "study_type",
        "species",
        "strain",
        "admin_route",
        "admin_method",
        "dose_start",
        "dose_start_unit",
        "dose_end",
        "dose_end_unit",
        "evidence_type",
        "source_dataset",
    ]
    evidence = evidence[[column for column in requested if column in evidence.columns]]
    rows = []
    for drug_id in mapping["drug_id"].drop_duplicates():
        matched = evidence[evidence["drug_id"] == drug_id]
        row = {"drug_id": drug_id, "has_tox_data": int(not matched.empty)}
        metadata_columns = [
            column
            for column in ("study_citation", "species", "admin_route", "dose_start")
            if column in matched.columns
        ]
        row["quality_tox"] = (
            float(matched[metadata_columns].notna().to_numpy().mean())
            if not matched.empty and metadata_columns
            else 0.0
        )
        for study_type in ("DEV", "CHR", "SUB", "MGR", "SAC"):
            row[f"Study_{study_type}"] = int(
                matched["study_type"].eq(study_type).sum()
            ) if "study_type" in matched else 0
        rows.append(row)
    return pd.DataFrame(rows, columns=feature_columns), evidence


def _pregnancy_features(outcomes: pd.DataFrame) -> pd.DataFrame:
    columns = ["drug_id", "has_pregnancy_outcome_data", "quality_pregnancy"] + [
        f"reports_{name}" for name in OUTCOME_NAMES
    ]
    if outcomes.empty or "drug_id" not in outcomes:
        return pd.DataFrame(columns=columns)
    supported = outcomes[outcomes["outcome"].isin(OUTCOME_NAMES)].copy()
    rows = []
    for drug_id in outcomes["drug_id"].dropna().drop_duplicates():
        matching = supported[supported["drug_id"] == drug_id]
        row = {
            "drug_id": drug_id,
            "has_pregnancy_outcome_data": int(not matching.empty),
            "quality_pregnancy": (
                _metadata_completeness(
                    matching[["outcome_group", "reported_term", "reported_count", "source_file"]]
                    .notna()
                    .sum()
                    .tolist()
                )
                if not matching.empty
                else 0.0
            ),
        }
        for outcome in OUTCOME_NAMES:
            values = matching.loc[matching["outcome"] == outcome, "reported_count"]
            row[f"reports_{outcome}"] = float(values.sum(min_count=1)) if not values.empty else 0.0
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _metadata_completeness(values: list[Any]) -> float:
    if not values:
        return 0.0
    available = sum(value is not None and not pd.isna(value) and str(value).strip() != "" for value in values)
    return float(available / len(values))


def _write_source_coverage(
    mapping: pd.DataFrame,
    adr: pd.DataFrame,
    pubchem: pd.DataFrame,
    tox: pd.DataFrame,
    pregnancy: pd.DataFrame,
    faers: pd.DataFrame,
) -> None:
    sources = pd.DataFrame({"drug_id": mapping["drug_id"].drop_duplicates()})
    for source_name, frame, flag, status in (
        ("adr", adr, "has_adr_data", "available"),
        ("molecular", pubchem, "resolution_status", "resolved"),
        ("toxicity", tox, "has_tox_data", "available"),
        ("pregnancy_outcomes", pregnancy, "has_pregnancy_outcome_data", "available"),
        ("faers", faers, "drug_id", "available"),
    ):
        if frame.empty:
            sources[f"{source_name}_status"] = "unavailable"
            continue
        if source_name == "molecular":
            available_ids = set(frame.loc[frame["resolution_status"].isin(
                ["resolved", "resolved_local_source"]
            ), "drug_id"])
        elif flag in frame.columns:
            available_ids = set(frame.loc[frame[flag].astype(bool), "drug_id"])
        else:
            available_ids = set(frame["drug_id"].dropna())
        sources[f"{source_name}_status"] = sources["drug_id"].map(
            lambda drug_id: status if drug_id in available_ids else "unavailable_or_unmatched"
        )
        quality_column = {
            "adr": "quality_adr",
            "molecular": "quality_descriptors",
            "toxicity": "quality_tox",
            "pregnancy_outcomes": "quality_pregnancy",
            "faers": "quality_faers",
        }[source_name]
        if quality_column in frame.columns:
            quality_by_id = frame.drop_duplicates("drug_id").set_index("drug_id")[quality_column]
            sources[f"{source_name}_metadata_completeness"] = sources["drug_id"].map(
                quality_by_id
            ).fillna(0.0)
    if "quality_graph" in pubchem.columns:
        graph_quality = pubchem.drop_duplicates("drug_id").set_index("drug_id")["quality_graph"]
        sources["graph_metadata_completeness"] = sources["drug_id"].map(graph_quality).fillna(0.0)
    sources.to_csv(DATA_DIR / "source_availability.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process the real local project datasets.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached/local PubChem records only; do not make network requests.",
    )
    parser.add_argument(
        "--max-pubchem-queries",
        type=int,
        default=None,
        help="Optional limit for new PubChem lookups.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    process_datasets(offline=args.offline, max_pubchem_queries=args.max_pubchem_queries)


if __name__ == "__main__":
    main()
