from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

from src.ingestion.common import (
    IngestedSource,
    classify_pregnancy_outcome,
    download_file,
    drug_name_key,
    is_combination_name,
    normalize_trimester,
    source_drug_id,
)


LOGGER = logging.getLogger("pregnancy_dataset_ingestion.mendeley")
DATASET_ID = "fnf4t4x2dz"
VERSION = 2
DOI = "10.17632/fnf4t4x2dz.2"
METADATA_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
EXPECTED_FILENAME = "FAERS_JAAD_Mendeley Supplement.docx"
EXPECTED_SHA256 = "309f40c7d7d90fbd15711da85e87dd96cdf4b1127cec44a768d961ab1d7d0da5"


def acquire(destination: Path) -> Path:
    request = Request(METADATA_URL, headers={"User-Agent": "PregnancySafetyDatasetPipeline/1.0"})
    with urlopen(request, timeout=30) as response:
        metadata: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    if metadata.get("version") != VERSION or metadata.get("doi", {}).get("id") != DOI:
        raise ValueError(f"Mendeley metadata did not resolve to the required version {DOI}.")
    licence = metadata.get("data_licence", {}).get("short_name", "")
    if licence != "CC BY 4.0":
        raise ValueError(f"Unexpected Mendeley dataset licence: {licence!r}")
    files = [item for item in metadata.get("files", []) if item.get("filename") == EXPECTED_FILENAME]
    if len(files) != 1:
        raise ValueError(f"Expected one {EXPECTED_FILENAME!r} in Mendeley {DOI}.")
    details = files[0]["content_details"]
    if details.get("sha256_hash") != EXPECTED_SHA256:
        raise ValueError("Mendeley file checksum differs from the pinned Version 2 record.")
    return download_file(details["download_url"], destination, expected_sha256=EXPECTED_SHA256)


def _extract_docx(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.process_real_datasets import _extract_faers_outcomes

    outcomes, totals = _extract_faers_outcomes(Path(path))
    if outcomes.empty:
        return outcomes, totals
    outcomes = outcomes.copy()
    outcomes["source_record_id"] = [
        f"table:{table}:row:{index}"
        for index, table in zip(outcomes.index, outcomes["source_table"])
    ]
    outcomes["source_dataset"] = f"Mendeley Data {DOI}"
    outcomes["source_doi"] = DOI
    outcomes["drug_id"] = outcomes["drug_name"].map(source_drug_id)
    outcomes["drug_name_key"] = outcomes["drug_name"].map(drug_name_key)
    outcomes["is_combination"] = outcomes["drug_name"].map(is_combination_name)
    outcomes["outcome_category"] = outcomes.apply(
        lambda row: classify_pregnancy_outcome(row.get("outcome_group"), row.get("reported_term")),
        axis=1,
    )
    outcomes["evidence_status"] = outcomes["reported_count"].map(
        lambda count: (
            "positive_safety_concern_evidence"
            if pd.notna(count) and count > 0
            else "outcome_not_reported"
        )
    )
    outcomes["evidence_granularity"] = "drug_level_aggregate_spontaneous_reports"
    trimester_values = outcomes.apply(lambda _: normalize_trimester(), axis=1)
    outcomes["trimester"] = [value[0] for value in trimester_values]
    outcomes["gestational_weeks"] = [value[1] for value in trimester_values]
    return outcomes, totals


def ingest(path: Path) -> tuple[IngestedSource, pd.DataFrame]:
    path = Path(path)
    if not path.is_file():
        return IngestedSource("mendeley_pregnancy_v2", False, status="source_unavailable", source_url=f"https://doi.org/{DOI}", source_version="2", source_files=[str(path)]), pd.DataFrame()
    try:
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash.casefold() != EXPECTED_SHA256:
            raise ValueError("Local Mendeley file checksum differs from the pinned Version 2 file.")
        outcomes, totals = _extract_docx(path)
        outcomes.attrs["source_totals"] = totals
        return IngestedSource(
            "mendeley_pregnancy_v2",
            True,
            outcomes,
            status="available",
            source_url=f"https://doi.org/{DOI}",
            source_version="2",
            source_files=[str(path)],
        ), totals
    except Exception as error:
        LOGGER.exception("Failed parsing Mendeley %s", path)
        return IngestedSource("mendeley_pregnancy_v2", True, status="parse_error", source_url=f"https://doi.org/{DOI}", source_version="2", error=str(error), source_files=[str(path)]), pd.DataFrame()
