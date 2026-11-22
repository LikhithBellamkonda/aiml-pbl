"""Acquisition for animal spontaneous adverse-event sources.

The primary source is the FDA CVM Animal & Veterinary Adverse Event (ADAE)
bulk export from openFDA.  The repository already carries one full export
(``datasets/animalandveterinary-event-0001-of-0001.json.zip``); acquisition
therefore works offline by default and only contacts the network when
``download_missing`` is requested.

EMA EudraVigilance Veterinary and the UK VMD dataset are documented but do
not expose a stable, scriptable structured bulk download.  They are recorded
in the manifest as ``manual_access_required`` and have explicit ingestion
interfaces so structured files can be placed locally later without changing
the pipeline.  Nothing is scraped and no access restriction is bypassed.

Raw downloads are stored byte-for-byte and every artefact is recorded with
its source URL, retrieval timestamp, SHA-256, size and record count.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "datasets"
RAW_DIR = ROOT / "data" / "raw" / "animal_adr"
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "animal_adr_acquisition_manifest.json"

LOGGER = logging.getLogger("meprank.animal_adr.acquisition")

OPENFDA_DOWNLOAD_INDEX = "https://api.fda.gov/download.json"
OPENFDA_EVENT_API = "https://api.fda.gov/animalandveterinary/event.json"
OPENFDA_LANDING = "https://open.fda.gov/data/downloads/"

SOURCES: dict[str, dict[str, Any]] = {
    "fda_adae": {
        "label": "FDA Animal & Veterinary Adverse Event Reports (ADAE)",
        "publisher": "U.S. Food and Drug Administration, Center for Veterinary Medicine",
        "download_portal": OPENFDA_LANDING,
        "api": OPENFDA_EVENT_API,
        "structured": True,
        "local_patterns": ["animalandveterinary-event-*.json.zip"],
    },
    "veddra": {
        "label": "VeDDRA terminology (combined animal/human veterinary list)",
        "publisher": "European Medicines Agency",
        "download_portal": (
            "https://www.ema.europa.eu/en/veterinary-regulatory-overview/"
            "post-authorisation-veterinary-medicines/pharmacovigilance-veterinary-medicines/"
            "eudravigilance-veterinary"
        ),
        "structured": True,
        "local_patterns": ["combined-veddra-list*.xlsx"],
    },
    "ema_evvet": {
        "label": "EMA EudraVigilance Veterinary (EVVet) line listings",
        "publisher": "European Medicines Agency",
        "download_portal": "https://www.adrreports.eu/vet/en/index.html",
        "structured": False,
        "local_patterns": ["evvet*.csv", "evvet*.xlsx"],
        "limitation": (
            "adrreports.eu serves an interactive query interface without a "
            "scriptable structured bulk export; automated acquisition would "
            "require bypassing that interface, which this project does not do. "
            "Place a structured EVVet export in datasets/raw/animal_adr/ and "
            "ingest_evvet() will process it."
        ),
    },
    "uk_vmd": {
        "label": "UK VMD suspected adverse event data",
        "publisher": "UK Veterinary Medicines Directorate (data.gov.uk)",
        "download_portal": (
            "https://www.data.gov.uk/dataset/ab3eea9d-1dd6-4c61-8b38-7e36e5518d99/"
            "suspected-adverse-event-data-submitted-to-the-vmd"
        ),
        "structured": False,
        "local_patterns": ["vmd*.csv", "vmd*.xlsx"],
        "limitation": (
            "The published VMD release does not provide a stable machine-readable "
            "endpoint in this repository's pinned snapshot. Place an official "
            "spreadsheet export in datasets/raw/animal_adr/ and ingest_vmd() will "
            "process it without changing the pipeline."
        ),
    },
}


class SourceUnavailable(RuntimeError):
    """Raised when a source has no accessible structured download."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _find_local(patterns: list[str]) -> list[Path]:
    """Locate each source file once, preferring the committed datasets copy.

    Acquisition copies raw files into ``data/raw/animal_adr/``; without the
    de-duplication below the same archive would be parsed twice and its
    records counted twice.
    """
    found: list[Path] = []
    seen: set[str] = set()
    for directory in (DATASET_DIR, DATASET_DIR / "raw", RAW_DIR):
        for pattern in patterns:
            for path in sorted(Path(directory).glob(pattern)):
                if path.is_file() and path.name not in seen:
                    seen.add(path.name)
                    found.append(path)
    return found


def _download(url: str, destination: Path, *, retries: int = 3, timeout: int = 120) -> Path:
    """Download a file with explicit retries, timeouts and a .part file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "MEPRank-AnimalADRAcquisition/1.0"}
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            temp = destination.with_suffix(destination.suffix + ".part")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if getattr(response, "status", 200) != 200:
                    raise RuntimeError(f"HTTP {response.status} for {url}")
                with temp.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            temp.replace(destination)
            return destination
        except (OSError, urllib.error.URLError, TimeoutError, RuntimeError) as error:
            last_error = error
            LOGGER.warning("Download attempt %d/%d failed for %s: %s", attempt, retries, url, error)
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Unable to download {url}: {last_error}")


def fetch_openfda_partitions(*, timeout: int = 120) -> list[str]:
    """Return the FDA ADAE bulk partition URLs from openFDA's download index."""
    request = urllib.request.Request(
        OPENFDA_DOWNLOAD_INDEX,
        headers={"User-Agent": "MEPRank-AnimalADRAcquisition/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    partitions = (
        payload.get("results", {})
        .get("animalandveterinary", {})
        .get("event", {})
        .get("partitions", [])
    )
    urls = [item.get("file") for item in partitions if item.get("file")]
    if not urls:
        raise SourceUnavailable("openFDA download index returned no ADAE partitions.")
    return urls


def download_openfda(*, destination_dir: Path | None = None) -> list[Path]:
    """Download every FDA ADAE partition into the raw cache."""
    destination_dir = Path(destination_dir or RAW_DIR)
    downloaded: list[Path] = []
    for url in fetch_openfda_partitions():
        name = url.rsplit("/", 1)[-1] or "animalandveterinary-event.json.zip"
        downloaded.append(_download(url, destination_dir / name))
    return downloaded


def validate_fda_archive(path: Path) -> dict[str, Any]:
    """Validate that an ADAE archive contains parseable event records."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".json")]
        if not members:
            raise ValueError(f"{path} contains no JSON member")
        with archive.open(members[0]) as handle:
            payload = json.load(handle)
    records = payload.get("results")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} contains no event records")
    meta = payload.get("meta", {})
    return {
        "member": members[0],
        "records": len(records),
        "last_updated": meta.get("last_updated"),
        "api_total": (meta.get("results") or {}).get("total"),
    }


def read_fda_reports(path: Path | None = None) -> list[dict[str, Any]]:
    """Read raw FDA ADAE event records without modifying their content."""
    paths = [Path(path)] if path else _find_local(SOURCES["fda_adae"]["local_patterns"])
    if not paths:
        return []
    records: list[dict[str, Any]] = []
    for archive_path in paths:
        with zipfile.ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".json")]
            for member in members:
                with archive.open(member) as handle:
                    payload = json.load(handle)
                records.extend(payload.get("results") or [])
    return records


def read_veddra(path: Path | None = None):
    """Load the VeDDRA terminology sheet, lowercase-normalised for joining."""
    import pandas as pd

    paths = [Path(path)] if path else _find_local(SOURCES["veddra"]["local_patterns"])
    if not paths:
        return pd.DataFrame(columns=["soc", "hlt", "pt", "llt"])
    frame = pd.read_excel(paths[0], sheet_name=0)
    columns = {
        "Current System Organ Class (SOC) Term": "soc",
        "Current High Level Term (HLT)": "hlt",
        "Current Preferred Term (PT)": "pt",
        "Current Low Level Term (LLT)": "llt",
        "Current PT Key": "pt_key",
    }
    frame = frame.rename(columns={key: value for key, value in columns.items() if key in frame.columns})
    keep = [name for name in ("soc", "hlt", "pt", "llt", "pt_key") if name in frame.columns]
    frame = frame[keep].dropna(how="all")
    for column in keep:
        frame[column] = frame[column].astype(str).str.strip()
    return frame.reset_index(drop=True)


def ingest_evvet(path: Path) -> list[dict[str, Any]]:
    """Optional structured EVVet ingestion interface.

    A structured export is not distributed for automated download, so this
    function only accepts a locally supplied file and never pretends to have
    acquired one otherwise.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise SourceUnavailable("No structured EVVet export was supplied.")
    return pd.read_csv(path).to_dict("records")


def ingest_vmd(path: Path) -> list[dict[str, Any]]:
    """Optional structured UK VMD ingestion interface (local file required)."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise SourceUnavailable("No structured UK VMD export was supplied.")
    if path.suffix.casefold() in {".xlsx", ".xls"}:
        return pd.read_excel(path).to_dict("records")
    return pd.read_csv(path).to_dict("records")


def acquire(*, download_missing: bool = False, copy_raw: bool = True) -> dict[str, Any]:
    """Acquire every accessible animal ADR source and write the manifest."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "generated_at": _now(),
        "purpose": (
            "Animal spontaneous adverse-event evidence. Not animal experimental "
            "toxicity and not human pregnancy outcome evidence."
        ),
        "sources": {},
    }

    fda_paths = _find_local(SOURCES["fda_adae"]["local_patterns"])
    if not fda_paths and download_missing:
        LOGGER.info("No local ADAE archive; downloading from openFDA.")
        try:
            fda_paths = download_openfda()
        except (RuntimeError, SourceUnavailable, urllib.error.URLError) as error:
            fda_paths = []
            manifest["sources"]["fda_adae"] = {
                **SOURCES["fda_adae"],
                "status": "download_failed",
                "error": f"{type(error).__name__}: {error}",
            }
    if fda_paths and "fda_adae" not in manifest["sources"]:
        entries = []
        for path in fda_paths:
            try:
                validation = validate_fda_archive(path)
                status = "acquired"
            except (ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
                validation = {"error": f"{type(error).__name__}: {error}"}
                status = "invalid"
            stored = path
            if copy_raw and status == "acquired":
                stored = RAW_DIR / path.name
                if not stored.exists():
                    shutil.copy2(path, stored)
            entries.append(
                {
                    "local_path": str(path.relative_to(ROOT)),
                    "raw_copy": str(stored.relative_to(ROOT)) if stored.exists() else None,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "retrieved_at": _now(),
                    "status": status,
                    **validation,
                }
            )
        manifest["sources"]["fda_adae"] = {
            **SOURCES["fda_adae"],
            "status": "acquired" if any(e["status"] == "acquired" for e in entries) else "invalid",
            "files": entries,
        }

    veddra_paths = _find_local(SOURCES["veddra"]["local_patterns"])
    if veddra_paths:
        path = veddra_paths[0]
        stored = path
        if copy_raw:
            stored = RAW_DIR / path.name
            if not stored.exists():
                shutil.copy2(path, stored)
        manifest["sources"]["veddra"] = {
            **SOURCES["veddra"],
            "status": "acquired",
            "files": [
                {
                    "local_path": str(path.relative_to(ROOT)),
                    "raw_copy": str(stored.relative_to(ROOT)) if stored.exists() else None,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "retrieved_at": _now(),
                }
            ],
        }
    else:
        manifest["sources"]["veddra"] = {**SOURCES["veddra"], "status": "unavailable"}

    for key in ("ema_evvet", "uk_vmd"):
        local = _find_local(SOURCES[key]["local_patterns"])
        manifest["sources"][key] = {
            **SOURCES[key],
            "status": "local_file_available" if local else "manual_access_required",
            "files": [
                {"local_path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                for path in local
            ],
        }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
