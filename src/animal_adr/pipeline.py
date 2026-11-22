"""Animal ADR evidence pipeline: acquire -> preprocess -> map -> features.

The stage is independent: it can be run on its own (``python -m
src.animal_adr.pipeline``) and it never raises for a missing source.  When no
FDA ADAE records are available it records that state and leaves all other
pipeline stages untouched.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.animal_adr import acquisition, features, mapping, preprocessing

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RECORD_PATH = DATA_DIR / "animal_adr_pipeline.json"

LOGGER = logging.getLogger("meprank.animal_adr.pipeline")


def run(*, download_missing: bool = False) -> dict[str, Any]:
    """Run every animal ADR stage and record the outcome."""
    record: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "stages": {},
    }

    manifest = acquisition.acquire(download_missing=download_missing)
    fda = manifest.get("sources", {}).get("fda_adae", {})
    record["stages"]["acquire"] = {
        "status": fda.get("status", "unavailable"),
        "files": len(fda.get("files", [])),
        "records": sum(int(entry.get("records") or 0) for entry in fda.get("files", [])),
        "sources": {
            name: details.get("status")
            for name, details in manifest.get("sources", {}).items()
        },
    }

    records = acquisition.read_fda_reports()
    if not records:
        record.update(
            {
                "status": "skipped",
                "reason": "No FDA ADAE records available; other stages unaffected.",
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        RECORD_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
        LOGGER.warning("Animal ADR stage skipped: no FDA ADAE records.")
        return record

    veddra = acquisition.read_veddra()
    summary = preprocessing.preprocess(records, veddra)
    record["stages"]["preprocess"] = {
        "status": "ok",
        "cleaned_rows": summary.get("cleaned_rows"),
        "unique_reports": summary.get("unique_reports"),
        "unique_species": summary.get("unique_species"),
    }

    mapping_summary = mapping.map_dataset()
    record["stages"]["map"] = {
        "status": "ok" if mapping_summary.get("mapping_rows") else "empty",
        **{
            key: value
            for key, value in mapping_summary.items()
            if key != "overlap" and isinstance(value, (int, float, str))
        },
    }

    feature_summary = features.build_features()
    record["stages"]["features"] = {
        "status": "ok" if feature_summary.get("drugs_with_animal_adr_evidence") else "empty",
        "drugs": feature_summary.get("drugs_with_animal_adr_evidence"),
        "reports": feature_summary.get("total_reports"),
    }

    record["status"] = "ok"
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    RECORD_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Animal ADR evidence pipeline.")
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download the openFDA ADAE bulk export when no local archive exists.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    record = run(download_missing=args.download_missing)
    for stage, details in record["stages"].items():
        print(f"{stage}: {details}")
    print(f"status: {record['status']}")
    raise SystemExit(0 if record["status"] in {"ok", "skipped"} else 1)


if __name__ == "__main__":
    main()
