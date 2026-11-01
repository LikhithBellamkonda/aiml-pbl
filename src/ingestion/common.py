from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd


LOGGER = logging.getLogger("pregnancy_dataset_ingestion")
TRIMESTERS = (
    "first_trimester",
    "second_trimester",
    "third_trimester",
    "entire_pregnancy",
    "unknown",
)


@dataclass
class IngestedSource:
    source: str
    available: bool
    records: pd.DataFrame = field(default_factory=pd.DataFrame)
    status: str = "not_found"
    source_url: str = ""
    source_version: str = ""
    error: str = ""
    source_files: list[str] = field(default_factory=list)


def normalize_drug_name(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"\s+", " ", text)


def drug_name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_drug_name(value)).strip()


def source_drug_id(value: Any) -> str:
    from urllib.parse import quote

    return "NAME:" + quote(drug_name_key(value), safe="")


def is_combination_name(value: Any) -> bool:
    name = normalize_drug_name(value)
    return bool(re.search(r"\s(?:/|\+|&)\s|\s+\b(?:and|with)\b\s+", name))


def normalize_trimester(
    explicit_value: Any = None,
    gestational_weeks: Any = None,
    exposure_period: Any = None,
) -> tuple[str, float | None]:
    explicit = normalize_drug_name(explicit_value)
    period = normalize_drug_name(exposure_period)
    combined = f"{explicit} {period}".strip()
    weeks = pd.to_numeric(pd.Series([gestational_weeks]), errors="coerce").iloc[0]
    if pd.isna(weeks) and isinstance(gestational_weeks, str):
        match = re.match(
            r"\s*(\d{1,2})(?:\s*\+\s*(\d))?\s*(?:weeks?|wk|w)?\s*$",
            gestational_weeks,
            flags=re.I,
        )
        if match and (match.group(2) is None or int(match.group(2)) < 7):
            weeks = float(match.group(1))
            if match.group(2) is not None:
                weeks += int(match.group(2)) / 7
    valid_weeks = (
        float(weeks)
        if pd.notna(weeks) and 0 <= float(weeks) <= 45
        else None
    )
    if any(token in combined for token in ("entire pregnancy", "whole pregnancy", "throughout pregnancy")):
        return "entire_pregnancy", valid_weeks
    if any(token in combined for token in ("first trimester", "1st trimester", "trimester 1")):
        return "first_trimester", valid_weeks
    if any(token in combined for token in ("second trimester", "2nd trimester", "trimester 2")):
        return "second_trimester", valid_weeks
    if any(token in combined for token in ("third trimester", "3rd trimester", "trimester 3")):
        return "third_trimester", valid_weeks

    if valid_weeks is not None:
        value = valid_weeks
        trimester = (
            "first_trimester"
            if value < 14
            else "second_trimester"
            if value < 28
            else "third_trimester"
        )
        return trimester, value
    return "unknown", None


def classify_pregnancy_outcome(*values: Any) -> str:
    text = normalize_drug_name(" ".join(str(value) for value in values if value is not None))
    if any(token in text for token in ("miscarriage", "spontaneous abortion", "spontaneous miscarriage")):
        return "miscarriage"
    if any(token in text for token in ("stillbirth", "still born", "intrauterine death")):
        return "stillbirth"
    if any(token in text for token in ("congenital", "malformation", "birth defect", "anomal")):
        return "congenital_anomaly"
    if any(token in text for token in ("prematur", "preterm", "pre term")):
        return "preterm_birth"
    if any(token in text for token in ("growth restriction", "low birth weight", "small for gestational age")):
        return "fetal_growth_outcome"
    if any(token in text for token in ("maternal", "mother", "puerper", "pregnancy disorder")):
        return "maternal_adverse_reaction"
    if any(token in text for token in ("pregnan", "gestation", "fetal", "foetal", "birth")):
        return "other_pregnancy_outcome"
    return ""


def download_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    timeout: int = 120,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PregnancySafetyDatasetPipeline/1.0"},
    )
    digest = hashlib.sha256()
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temp_path.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
    except (OSError, urllib.error.URLError, TimeoutError) as error:
        temp_path.unlink(missing_ok=True)
        LOGGER.exception("Download failed: %s", url)
        raise RuntimeError(f"Unable to download {url}: {error}") from error
    actual_hash = digest.hexdigest()
    if expected_sha256 and actual_hash.lower() != expected_sha256.lower():
        temp_path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual_hash}"
        )
    temp_path.replace(destination)
    LOGGER.info("Downloaded %s (%d bytes; sha256=%s)", destination, destination.stat().st_size, actual_hash)
    return destination


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffixes = "".join(path.suffixes).casefold()
    if suffixes.endswith(".csv.gz") or suffixes.endswith(".tsv.gz"):
        separator = "\t" if ".tsv" in suffixes else ","
        return pd.read_csv(path, sep=separator, compression="gzip", low_memory=False)
    if path.suffix.casefold() in {".csv", ".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t" if path.suffix.casefold() in {".tsv", ".txt"} else ",", low_memory=False)
    if path.suffix.casefold() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported tabular file type: {path}")


def write_records(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def json_safe_record(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, default=str, sort_keys=True)


def download_url_for_doi(doi: str) -> str:
    return "https://doi.org/" + quote(doi.strip(), safe="/")
