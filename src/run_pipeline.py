"""End-to-end, reproducible project pipeline.

Runs every stage that the locally available data permits, validates each
stage's output, and records what happened in ``data/pipeline_run.json``.
A failed stage is recorded with its exception; independent later stages still
run on whatever artefacts exist. No stage fabricates data to satisfy a check.
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.data_preprocessing import PregnancyDataset
from src.dataset_feasibility import write_report
from src.ingestion.pipeline import run_pipeline as run_ingestion
from src.process_real_datasets import process_datasets

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUN_RECORD = DATA_DIR / "pipeline_run.json"
LOGGER = logging.getLogger("meprank.pipeline")

REQUIRED_OUTPUTS = (
    "drug_mapping.csv",
    "real_pubchem.csv",
    "real_adr_evidence.csv",
    "real_pregnancy_outcomes.csv",
    "real_toxicity_evidence.csv",
    "real_faers_summary.csv",
    "real_pregnancy_features.csv",
    "real_faers_adr.csv",
    "source_availability.csv",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_outputs() -> dict[str, Any]:
    present = {name: (DATA_DIR / name).is_file() for name in REQUIRED_OUTPUTS}
    missing = [name for name, exists in present.items() if not exists]
    empty = [
        name
        for name in REQUIRED_OUTPUTS
        if present[name] and (DATA_DIR / name).stat().st_size == 0
    ]
    return {"present": present, "missing": missing, "empty": empty}


def _stage_ingest(download_missing: bool, max_pubchem_queries: int) -> dict[str, Any]:
    report = run_ingestion(
        root=ROOT,
        download_missing=download_missing,
        max_pubchem_queries=max_pubchem_queries,
    )
    trainable = sum(
        item.get("decision") == "trainable" for item in report.get("target_decisions", [])
    )
    return {
        "evidence_records": report.get("master_record_count"),
        "mapped_drugs": report.get("unique_mapped_drug_count"),
        "trainable_pregnancy_endpoints": trainable,
    }


def _stage_process(offline: bool, max_pubchem_queries: int | None) -> dict[str, Any]:
    process_datasets(offline=offline, max_pubchem_queries=max_pubchem_queries)
    outputs = _validate_outputs()
    if outputs["missing"]:
        raise RuntimeError(
            "Dataset processing did not produce: " + ", ".join(outputs["missing"])
        )
    return {"outputs_validated": len(REQUIRED_OUTPUTS), "empty": outputs["empty"]}


def _stage_features() -> dict[str, Any]:
    """Build and validate the model-ready feature tables."""
    animal_features = DATA_DIR / "animal_adr_features.csv"
    dataset = PregnancyDataset(
        DATA_DIR / "real_faers_adr.csv",
        DATA_DIR / "real_pubchem.csv",
        pregnancy_path=DATA_DIR / "real_pregnancy_features.csv",
        labels_path=DATA_DIR / "pregnancy_labels.csv",
        faers_path=DATA_DIR / "real_faers_summary.csv",
        animal_adr_path=animal_features if animal_features.exists() else None,
    )
    if len(dataset) == 0:
        raise RuntimeError("Model-ready feature table is empty.")
    summary = {
        "drugs": int(len(dataset)),
        "adr_features": len(dataset.adr_cols),
        "pregnancy_features": len(dataset.pregnancy_cols),
        "label_columns_present": bool(dataset.labels_df.shape[1]),
        "validated_labels": 0,
        "modality_availability": {},
    }
    for index in range(len(dataset)):
        sample = dataset[index]
        for position, name in enumerate(
            ("general_adr", "chemical_descriptors", "molecular_graph",
             "pregnancy_outcomes", "faers_reporting", "animal_adr")
        ):
            if position >= len(sample.availability_mask):
                continue
            if float(sample.availability_mask[position]) > 0:
                summary["modality_availability"][name] = (
                    summary["modality_availability"].get(name, 0) + 1
                )
    if not dataset.labels_df.empty:
        mask_columns = [
            column for column in dataset.mask_cols if column in dataset.labels_df
        ]
        if mask_columns:
            summary["validated_labels"] = int(
                dataset.labels_df[mask_columns].fillna(0).astype(bool).sum().sum()
            )
    (DATA_DIR / "model_ready_features.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _stage_animal_adr() -> dict[str, Any]:
    """Animal spontaneous ADR evidence; optional and independent of other stages."""
    from src.animal_adr.pipeline import run as run_animal_adr

    report = run_animal_adr(download_missing=False)
    if report.get("status") == "skipped":
        return {"status": "skipped", "reason": report.get("reason")}
    stages = report.get("stages", {})
    return {
        "status": report.get("status"),
        "raw_records": stages.get("acquire", {}).get("records"),
        "cleaned_rows": stages.get("preprocess", {}).get("cleaned_rows"),
        "mapped_drugs": stages.get("map", {}).get("unique_standardized_drugs"),
        "feature_drugs": stages.get("features", {}).get("drugs"),
    }


def _stage_feasibility() -> dict[str, Any]:
    report = write_report(ROOT)
    decisions = report.get("target_decisions", [])
    return {
        "trainable_endpoints": sum(
            item.get("decision") == "trainable" for item in decisions
        ),
        "endpoints_assessed": len(decisions),
    }


def _stage_trimester_evaluation() -> dict[str, Any]:
    """Trimester metric report; blocked (not zero) without labels."""
    from src.trimester_evaluation import write_report as write_trimester_report

    report = write_trimester_report()
    return {"status": report.get("status"), "reason": report.get("reason")}


def _stage_evidence_inventory() -> dict[str, Any]:
    """Evidence-tiered coverage inventory for the literature control drugs."""
    from src.evidence_inventory import write_inventory

    summary = write_inventory()
    return {"controls": summary.get("controls"), "matched": summary.get("matched")}


STAGES: tuple[tuple[str, Callable[..., dict[str, Any]]], ...] = (
    ("ingest", _stage_ingest),
    ("process", _stage_process),
    ("animal_adr", _stage_animal_adr),
    ("features", _stage_features),
    ("feasibility", _stage_feasibility),
    ("trimester_evaluation", _stage_trimester_evaluation),
    ("evidence_inventory", _stage_evidence_inventory),
)


def run(
    download_missing: bool = False,
    offline: bool = True,
    max_pubchem_queries: int = 0,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "started_at": _now(),
        "finished_at": None,
        "status": "running",
        "stages": [],
        "options": {
            "download_missing": download_missing,
            "offline": offline,
            "max_pubchem_queries": max_pubchem_queries,
        },
    }
    for name, stage in STAGES:
        LOGGER.info("Running stage: %s", name)
        started = _now()
        entry: dict[str, Any] = {"stage": name, "started_at": started}
        try:
            if name == "ingest":
                details = stage(download_missing, max_pubchem_queries)
            elif name == "process":
                details = stage(offline, max_pubchem_queries or None)
            else:
                details = stage()
            entry.update({"status": "ok", "details": details})
        except Exception as error:  # record, then let independent stages continue
            LOGGER.error("Stage %s failed: %s", name, error)
            entry.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(limit=5),
                }
            )
        entry["finished_at"] = _now()
        record["stages"].append(entry)

    outputs = _validate_outputs()
    record["outputs"] = outputs
    failed = [item["stage"] for item in record["stages"] if item["status"] != "ok"]
    record["status"] = "ok" if not failed else "partial_failure"
    record["finished_at"] = _now()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUN_RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _print_summary(record: dict[str, Any]) -> None:
    for item in record["stages"]:
        line = f"{item['stage']}: {item['status']}"
        if item["status"] == "ok":
            line += f" ({json.dumps(item.get('details', {}))})"
        else:
            line += f" — {item.get('error')}"
        print(line)
    if record["outputs"]["missing"]:
        print("Missing outputs: " + ", ".join(record["outputs"]["missing"]))
    print(f"Run record: {RUN_RECORD}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full MEPRank data pipeline.")
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Attempt to acquire missing pinned sources from their official APIs.",
    )
    parser.add_argument(
        "--online-pubchem",
        action="store_true",
        help="Allow new PubChem queries during dataset processing.",
    )
    parser.add_argument(
        "--max-pubchem-queries",
        type=int,
        default=0,
        help="Maximum new PubChem lookups (default 0 = cached/offline).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    record = run(
        download_missing=args.download_missing,
        offline=not args.online_pubchem,
        max_pubchem_queries=args.max_pubchem_queries,
    )
    _print_summary(record)
    raise SystemExit(0 if record["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
