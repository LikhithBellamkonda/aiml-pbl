from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pandas as pd

from src.ingestion.common import IngestedSource, json_safe_record


LOGGER = logging.getLogger("pregnancy_dataset_ingestion.dryad")
API_ROOT = "https://datadryad.org/api/v2"
DOI = "10.5061/dryad.2z34tmpk0"
PUBLICATION_DOI = "10.1212/WNL.0000000000010993"
EXPECTED_FILENAME = "Supplemental_tables.docx"
EXPECTED_SHA256 = "cc3209d25877e536845c52abe2e56667883b5b570306017e95132fc2a8f7f53b"
WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PregnancySafetyDatasetPipeline/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def acquire(destination: Path) -> Path:
    dataset_url = f"{API_ROOT}/datasets/{urllib.parse.quote(f'doi:{DOI}', safe='')}"
    metadata = _request_json(dataset_url)
    if metadata.get("identifier", "").casefold() != f"doi:{DOI}".casefold():
        raise ValueError(f"Dryad metadata did not resolve to the required DOI {DOI}.")
    if metadata.get("license") != "https://spdx.org/licenses/CC0-1.0.html":
        raise ValueError(f"Unexpected Dryad dataset license: {metadata.get('license')!r}")

    version_id = metadata.get("_links", {}).get("stash:version", {}).get("href", "")
    if not version_id:
        raise ValueError("Dryad metadata did not identify a dataset version.")
    version = _request_json(urllib.parse.urljoin("https://datadryad.org", version_id))
    file_list_url = version.get("_links", {}).get("stash:files", {}).get("href", "")
    if not file_list_url:
        raise ValueError("Dryad version metadata did not expose a file list.")
    file_list = _request_json(urllib.parse.urljoin("https://datadryad.org", file_list_url))
    files = file_list.get("_embedded", {}).get("stash:files", [])
    matches = [
        item
        for item in files
        if item.get("path") == EXPECTED_FILENAME
        and item.get("digest", "").casefold() == EXPECTED_SHA256
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Dryad DOI {DOI} did not contain exactly one checksum-verified {EXPECTED_FILENAME}."
        )
    download_href = matches[0].get("_links", {}).get("stash:download", {}).get("href", "")
    if not download_href:
        raise ValueError("Dryad file metadata did not expose a download link.")
    download_url = urllib.parse.urljoin("https://datadryad.org", download_href)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        download_url,
        headers={"User-Agent": "PregnancySafetyDatasetPipeline/1.0"},
    )
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
    except (OSError, urllib.error.URLError, TimeoutError) as error:
        temp.unlink(missing_ok=True)
        LOGGER.exception("Dryad file download failed for DOI %s", DOI)
        raise RuntimeError(
            f"Unable to download Dryad file for DOI {DOI}: {error}. "
            "The public API may require an authenticated download."
        ) from error
    if digest.hexdigest().casefold() != EXPECTED_SHA256:
        temp.unlink(missing_ok=True)
        raise ValueError("Downloaded Dryad file does not match the published SHA-256 digest.")
    temp.replace(destination)
    return destination


def _word_cell_text(cell: ElementTree.Element) -> str:
    return " ".join(
        "".join(node.text or "" for node in paragraph.findall(".//w:t", WORD_NS))
        for paragraph in cell.findall("./w:p", WORD_NS)
    ).strip()


def _parse_docx(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    rows: list[dict[str, Any]] = []
    for table_number, table in enumerate(document.findall(".//w:tbl", WORD_NS), start=1):
        table_rows: list[list[str]] = []
        current_endpoint = ""
        for row_number, row in enumerate(table.findall("./w:tr", WORD_NS), start=1):
            cells = [_word_cell_text(cell) for cell in row.findall("./w:tc", WORD_NS)]
            table_rows.append(cells)
            if not any(cells):
                continue
            caption = next((value for value in cells if value), "")
            first_cell = cells[0] if cells else ""
            lower_caption = " ".join(cells).casefold()
            if "autism spectrum disorder" in lower_caption:
                current_endpoint = "ASD"
            elif "adhd" in lower_caption or "attention-deficit" in lower_caption:
                current_endpoint = "ADHD"
            drug_name = ""
            for candidate in ("Valproic Acid", "Lamotrigine", "Carbamazepine"):
                if candidate.casefold() in first_cell.casefold():
                    drug_name = candidate
                    break
            rows.append(
                {
                    "source_dataset": f"Dryad {DOI}",
                    "source_doi": DOI,
                    "publication_doi": PUBLICATION_DOI,
                    "source_file": path.name,
                    "source_table_id": f"table_{table_number}",
                    "source_record_id": f"table_{table_number}:row_{row_number}",
                    "table_caption_or_row_text": caption,
                    "source_row_number": row_number,
                    "drug_name": drug_name,
                    "drug_id": "",
                    "study_endpoint": current_endpoint,
                    "outcome_category": "",
                    "outcome_term_original": first_cell,
                    "outcome_value_original": json.dumps(cells, ensure_ascii=False),
                    "evidence_status": "aggregate_association_estimate",
                    "evidence_granularity": "study_level_aggregate_table",
                    "trimester": "unknown",
                    "gestational_weeks": None,
                    "raw_record_json": json_safe_record(
                        {
                            "table_number": table_number,
                            "row_number": row_number,
                            "cells": cells,
                        }
                    ),
                    "outcome_semantics_verified": False,
                    "individual_outcome_observation": False,
                    "is_requested_pregnancy_endpoint": False,
                }
            )
    return pd.DataFrame(rows)


def ingest(path: Path | None, doi: str | None = DOI) -> IngestedSource:
    if doi != DOI:
        return IngestedSource(
            "dryad_antiepileptic_pregnancy",
            False,
            status="identity_unverified",
            source_url=f"https://doi.org/{doi}" if doi else "",
            error=f"Configured DOI must match the verified Dryad record {DOI}.",
        )
    if path is None or not Path(path).is_file():
        return IngestedSource(
            "dryad_antiepileptic_pregnancy",
            False,
            status="source_unavailable",
            source_url=f"https://doi.org/{DOI}",
            source_version=DOI,
            error="The verified Dryad supplement is not present locally.",
            source_files=[str(path)] if path else [],
        )

    path = Path(path)
    try:
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash.casefold() != EXPECTED_SHA256:
            raise ValueError("Local Dryad supplement checksum differs from the published file.")
        if path.suffix.casefold() != ".docx":
            raise ValueError(
                "This Dryad deposit contains a DOCX supplement; unsupported files are not treated as patient records."
            )
        records = _parse_docx(path)
        return IngestedSource(
            "dryad_antiepileptic_pregnancy",
            True,
            records,
            status="available_aggregate_tables_only",
            source_url=f"https://doi.org/{DOI}",
            source_version=DOI,
            source_files=[str(path)],
        )
    except Exception as error:
        LOGGER.exception("Failed parsing Dryad source %s", path)
        return IngestedSource(
            "dryad_antiepileptic_pregnancy",
            True,
            status="parse_error",
            source_url=f"https://doi.org/{DOI}",
            source_version=DOI,
            error=str(error),
            source_files=[str(path)],
        )
