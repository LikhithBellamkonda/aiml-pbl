"""System status: what data, models and dependencies are actually present."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.services.drug_service import DATA_DIR, SOURCE_FILES
from src.services.inference import model_status


ROOT = Path(__file__).resolve().parents[2]

DATASET_FILES = {
    "Drug identifier mapping": "drug_mapping.csv",
    "PubChem structures": "real_pubchem.csv",
    "General ADR terms": "real_adr_evidence.csv",
    "Human pregnancy outcomes": "real_pregnancy_outcomes.csv",
    "Nonclinical toxicity studies": "real_toxicity_evidence.csv",
    "FAERS reporting summary": "real_faers_summary.csv",
    "Source availability matrix": "source_availability.csv",
    "Standardized evidence master": "standardized_master_evidence.csv",
    "Pregnancy labels (required for training)": "pregnancy_labels.csv",
}


def _last_run(data_dir: Path) -> dict[str, Any] | None:
    path = data_dir / "pipeline_run.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _metrics_summary(data_dir: Path) -> dict[str, Any] | None:
    path = data_dir / "reproductive_tox_training_metrics.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    endpoints = payload.get("endpoints") or {}
    return {
        "generated_at": payload.get("generated_at"),
        "status": payload.get("status"),
        "clinical_pregnancy_model_trained": payload.get(
            "clinical_pregnancy_model_trained"
        ),
        "endpoints": {
            name: {
                model: {
                    key: details["models"][model].get(key)
                    for key in ("auroc", "auprc", "f1", "accuracy")
                }
                for model in (details.get("models") or {})
            }
            for name, details in endpoints.items()
        },
    }


def system_status(
    data_dir: Path | str = DATA_DIR,
    model_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Summarise dataset, model and training readiness without guessing."""
    data_dir = Path(data_dir)
    datasets = []
    for label, file_name in DATASET_FILES.items():
        path = data_dir / file_name
        present = path.is_file()
        rows = None
        if present:
            try:
                rows = int(len(pd.read_csv(path, usecols=[0])))
            except (pd.errors.ParserError, ValueError, OSError):
                rows = None
        datasets.append(
            {
                "name": label,
                "file": file_name,
                "present": present,
                "rows": rows,
            }
        )
    missing_datasets = [item["name"] for item in datasets if not item["present"]]
    status = model_status() if model_dir is None else model_status(model_dir)

    tox_available = [
        endpoint
        for endpoint, details in status["nonclinical_toxicity"].items()
        if details["available"]
    ]
    return {
        "datasets": datasets,
        "missing_datasets": missing_datasets,
        "models": status,
        "trained_models": {
            "human_pregnancy": [],
            "nonclinical_toxicity": tox_available,
        },
        "training_readiness": {
            "human_pregnancy": {
                "status": "blocked",
                "reason": status["human_pregnancy"]["reason"],
            },
            "nonclinical_toxicity": {
                "status": "available" if tox_available else "blocked",
                "reason": (
                    None
                    if tox_available
                    else "datasets/Supplementary_S3_Raw data.xlsx is not present, "
                    "so the nonclinical endpoints cannot be trained."
                ),
            },
        },
        "metrics": _metrics_summary(data_dir),
        "last_pipeline_run": _last_run(data_dir),
        "source_tables": {
            name: (data_dir / file_name).is_file()
            for name, file_name in SOURCE_FILES.items()
        },
    }
