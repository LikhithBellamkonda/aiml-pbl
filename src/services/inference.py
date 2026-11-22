"""Central inference service.

Every prediction request flows through this module. It validates the drug,
checks whether a trained model exists for the requested target, applies the
training-time preprocessing to the same features, and returns either a real
model output or an explicit abstention.

There is deliberately no fallback that synthesises a score. A pregnancy
outcome request has no trained model (no provenance-backed human labels exist
in this repository), so it always returns ``status="abstained"`` with the
reason. Nonclinical reproductive-toxicity endpoints return a probability only
when their trained checkpoint is present on disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np

from src.services.drug_service import build_drug_profile, resolve_drug

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TOX_MODEL_DIR = ROOT / "models" / "reproductive_tox"
TOX_METRICS_PATH = DATA_DIR / "reproductive_tox_training_metrics.json"
PREGNANCY_LABELS_PATH = DATA_DIR / "pregnancy_labels.csv"
PREGNANCY_CHECKPOINT = ROOT / "src" / "models" / "pregnancy_adr_model.pt"

LOGGER = logging.getLogger(__name__)

PREGNANCY_TARGETS = (
    "fetal_death",
    "preterm_birth",
    "fetal_growth_restriction",
    "congenital_anomaly",
)
TOX_ENDPOINTS = ("TG414", "TG416", "TG421t", "TG422t", "TG443t")
TOX_ENDPOINT_DESCRIPTIONS = {
    "TG414": "OECD TG 414 developmental toxicity (prenatal developmental toxicity study)",
    "TG416": "OECD TG 416 two-generation reproductive toxicity",
    "TG421t": "OECD TG 421t reproduction/developmental toxicity screening",
    "TG422t": "OECD TG 422t combined repeated-dose/reproductive screening",
    "TG443t": "OECD TG 443t extended one-generation reproductive toxicity",
}

PREGNANCY_ABSTENTION_REASON = (
    "No validated human pregnancy-outcome model is available. The acquired "
    "sources provide aggregate spontaneous-report counts, regulatory "
    "classifications and untimed associations, none of which are compatible "
    "binary outcome labels. Add provenance-backed data/pregnancy_labels.csv "
    "before any pregnancy prediction can be produced."
)
NONCLINICAL_WARNING = (
    "This is a nonclinical animal-toxicity endpoint model. It is not a human "
    "pregnancy risk prediction and must not be presented as clinical evidence."
)


def _tox_model_path(endpoint: str, model_name: str, model_dir: Path) -> Path:
    return Path(model_dir) / f"{endpoint}_{model_name}.joblib"


def _tox_metrics(endpoint: str, metrics_path: Path) -> dict[str, Any] | None:
    path = Path(metrics_path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("Could not read toxicity metrics %s: %s", path, error)
        return None
    return (payload.get("endpoints") or {}).get(endpoint)


def model_status(
    model_dir: Path | str = TOX_MODEL_DIR,
    metrics_path: Path | str = TOX_METRICS_PATH,
) -> dict[str, Any]:
    """Report which models actually exist on disk, without loading them."""
    model_dir = Path(model_dir)
    endpoints: dict[str, Any] = {}
    for endpoint in TOX_ENDPOINTS:
        available_models = [
            name
            for name in ("logistic_regression", "random_forest", "xgboost")
            if _tox_model_path(endpoint, name, model_dir).is_file()
        ]
        endpoints[endpoint] = {
            "endpoint": endpoint,
            "description": TOX_ENDPOINT_DESCRIPTIONS[endpoint],
            "task": "nonclinical_reproductive_toxicity",
            "available": bool(available_models),
            "models": available_models,
            "metrics_available": _tox_metrics(endpoint, metrics_path) is not None,
            "reason": (
                None
                if available_models
                else "No trained checkpoint found. The source workbook "
                "datasets/Supplementary_S3_Raw data.xlsx is not present, so this "
                "endpoint cannot be trained."
            ),
        }
    return {
        "human_pregnancy": {
            "task": "human_pregnancy_outcome",
            "targets": list(PREGNANCY_TARGETS),
            "available": False,
            "labels_present": Path(PREGNANCY_LABELS_PATH).is_file(),
            "checkpoint_present": Path(PREGNANCY_CHECKPOINT).is_file(),
            "reason": PREGNANCY_ABSTENTION_REASON,
        },
        "nonclinical_toxicity": endpoints,
    }


def list_targets(
    model_dir: Path | str = TOX_MODEL_DIR,
    metrics_path: Path | str = TOX_METRICS_PATH,
) -> list[dict[str, Any]]:
    """Flat list of every target the application can be asked about."""
    status = model_status(model_dir, metrics_path)
    targets: list[dict[str, Any]] = []
    for outcome in PREGNANCY_TARGETS:
        targets.append(
            {
                "target": outcome,
                "task": "human_pregnancy_outcome",
                "available": False,
                "reason": PREGNANCY_ABSTENTION_REASON,
            }
        )
    for endpoint, details in status["nonclinical_toxicity"].items():
        targets.append(
            {
                "target": endpoint,
                "task": "nonclinical_reproductive_toxicity",
                "available": details["available"],
                "reason": details["reason"],
                "description": details["description"],
            }
        )
    return targets


def _response(
    target: str,
    task: str,
    *,
    status: str,
    reasons: list[str],
    prediction: float | None = None,
    uncertainty: float | None = None,
    model: dict[str, Any] | None = None,
    availability: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
    drug_id: str | None = None,
) -> dict[str, Any]:
    return {
        "target": target,
        "task": task,
        "drug_id": drug_id,
        "status": status,
        "prediction": prediction,
        "uncertainty": uncertainty,
        "model": model or {"name": None, "available": False},
        "modalities_available": availability or [],
        "provenance": provenance or {},
        "reasons": reasons,
    }


def _pregnancy_prediction(drug_id: str | None, target: str) -> dict[str, Any]:
    profile = build_drug_profile(drug_id) if drug_id else {}
    if (
        drug_id
        and not profile.get("matched")
        and str(drug_id).startswith("NAME:")
    ):
        # Name-keyed evidence drugs are only in the evidence index, not the
        # identifier mapping; re-resolve through that index to report sources.
        profile = build_drug_profile(unquote(str(drug_id)[5:]))
    pregnancy = profile.get("pregnancy") or {}
    availability = [
        name
        for name, present in (
            ("pregnancy_outcomes", pregnancy.get("available")),
            ("faers_reporting", (profile.get("faers") or {}).get("available")),
            ("general_adr", (profile.get("adr") or {}).get("available")),
            ("molecular", (profile.get("molecular") or {}).get("available")),
        )
        if present
    ]
    reasons = [PREGNANCY_ABSTENTION_REASON]
    if not pregnancy.get("available"):
        reasons.append(
            "This drug also has no human pregnancy-outcome evidence records, so "
            "the absent label is not treated as a negative outcome."
        )
    return _response(
        target,
        "human_pregnancy_outcome",
        status="abstained",
        reasons=reasons,
        availability=availability,
        provenance={
            "evidence_source": "data/real_pregnancy_outcomes.csv",
            "sources": pregnancy.get("sources", []),
            "trimesters_reported": pregnancy.get("trimesters", []),
        },
        drug_id=drug_id,
    )


def _toxicity_prediction(
    drug_id: str,
    target: str,
    *,
    smiles: str | None,
    model_dir: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    if target not in TOX_ENDPOINTS:
        return _response(
            target,
            "nonclinical_reproductive_toxicity",
            status="unsupported",
            reasons=[f"Unsupported nonclinical endpoint: {target}"],
            drug_id=drug_id,
        )

    available_models = [
        name
        for name in ("random_forest", "xgboost", "logistic_regression")
        if _tox_model_path(target, name, model_dir).is_file()
    ]
    if not available_models:
        return _response(
            target,
            "nonclinical_reproductive_toxicity",
            status="abstained",
            reasons=[
                f"No trained model checkpoint exists for {target}.",
                "The source workbook datasets/Supplementary_S3_Raw data.xlsx is "
                "absent, so labels are unavailable and training cannot run.",
            ],
            drug_id=drug_id,
        )

    if not isinstance(smiles, str) or not smiles.strip():
        return _response(
            target,
            "nonclinical_reproductive_toxicity",
            status="abstained",
            reasons=["This drug has no SMILES, so the structure-based model cannot run."],
            model={"name": available_models[0], "available": True},
            drug_id=drug_id,
        )

    try:
        import joblib
        from rdkit import Chem, RDLogger
        from rdkit.Chem import DataStructs, rdFingerprintGenerator
    except ImportError as error:
        return _response(
            target,
            "nonclinical_reproductive_toxicity",
            status="abstained",
            reasons=[f"Required dependency is missing: {error}"],
            drug_id=drug_id,
        )

    RDLogger.DisableLog("rdApp.*")
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        return _response(
            target,
            "nonclinical_reproductive_toxicity",
            status="abstained",
            reasons=["The recorded SMILES could not be parsed by RDKit."],
            drug_id=drug_id,
        )

    fingerprint = np.zeros((1, 1024), dtype=np.uint8)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    DataStructs.ConvertToNumpyArray(
        generator.GetFingerprint(molecule), fingerprint[0]
    )

    model_name = available_models[0]
    try:
        bundle = joblib.load(_tox_model_path(target, model_name, model_dir))
        classifier = bundle["model"] if isinstance(bundle, dict) else bundle
        probability = float(classifier.predict_proba(fingerprint)[0, 1])
    except Exception as error:  # a corrupted checkpoint must not crash the app
        LOGGER.exception("Failed to run %s model for %s", target, target)
        return _response(
            target,
            "nonclinical_reproductive_toxicity",
            status="abstained",
            reasons=[f"The {target} checkpoint could not be executed: {error}"],
            drug_id=drug_id,
        )

    metrics = _tox_metrics(target, metrics_path) or {}
    model_metrics = (metrics.get("models") or {}).get(model_name) or {}
    return _response(
        target,
        "nonclinical_reproductive_toxicity",
        status="ok",
        reasons=[],
        prediction=round(probability, 6),
        uncertainty=None,
        model={
            "name": model_name,
            "available": True,
            "path": str(_tox_model_path(target, model_name, model_dir)),
            "metrics": {
                key: model_metrics.get(key)
                for key in ("auroc", "auprc", "f1", "brier_score", "ece", "n")
            },
        },
        availability=["molecular (Morgan fingerprint)"],
        provenance={
            "task": "nonclinical animal toxicity endpoint",
            "endpoint": target,
            "description": TOX_ENDPOINT_DESCRIPTIONS[target],
            "warning": NONCLINICAL_WARNING,
            "uncertainty_note": (
                "No validated MC-dropout checkpoint or calibration study exists "
                "for this endpoint, so predictive uncertainty is not reported."
            ),
        },
        drug_id=drug_id,
    )


def predict(
    drug_id: str | None,
    target: str,
    *,
    smiles: str | None = None,
    query: str | None = None,
    model_dir: Path | str = TOX_MODEL_DIR,
    metrics_path: Path | str = TOX_METRICS_PATH,
) -> dict[str, Any]:
    """Run or abstain from a prediction for one target.

    ``drug_id`` may be omitted when ``query`` is given; the query is resolved
    first. A missing drug never produces a prediction.
    """
    if query and not drug_id:
        resolution = resolve_drug(query)
        if resolution["ambiguous"]:
            return _response(
                target,
                "unknown",
                status="abstained",
                reasons=[
                    f"'{query}' matches multiple drug identifiers; select one first."
                ],
            )
        drug_id = resolution.get("drug_id")
        if smiles is None:
            smiles = resolution.get("identity", {}).get("smiles")
    if not drug_id:
        return _response(
            target,
            "unknown",
            status="abstained",
            reasons=["The drug could not be resolved against the local mapping."],
        )
    if smiles is None:
        profile = build_drug_profile(drug_id)
        smiles = (profile.get("identity") or {}).get("smiles")

    if target in PREGNANCY_TARGETS:
        return _pregnancy_prediction(drug_id, target)
    if target in TOX_ENDPOINTS:
        return _toxicity_prediction(
            drug_id,
            target,
            smiles=smiles,
            model_dir=Path(model_dir),
            metrics_path=Path(metrics_path),
        )
    return _response(
        target,
        "unknown",
        status="unsupported",
        reasons=[
            f"No model or task is defined for target '{target}'.",
            "Supported targets: " + ", ".join(PREGNANCY_TARGETS + TOX_ENDPOINTS),
        ],
        drug_id=drug_id,
    )
