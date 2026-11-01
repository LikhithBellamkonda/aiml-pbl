from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = ROOT / "datasets" / "Supplementary_S3_Raw data.xlsx"
RESULTS_FILE = ROOT / "data" / "reproductive_tox_training_metrics.json"
MODEL_DIR = ROOT / "models" / "reproductive_tox"
LOGGER = logging.getLogger("reproductive_tox_training")
FINGERPRINT_BITS = 1024
N_SPLITS = 5

ENDPOINTS = {
    "TG414": ("414smiles4", "TG414"),
    "TG416": ("416smiles4", "TG416"),
    "TG421t": ("421smiles4", "TG421t"),
    "TG422t": ("422smiles4", "TG422t"),
    "TG443t": ("443smiles4", "TG443t"),
}


@dataclass(frozen=True)
class EndpointData:
    endpoint: str
    smiles: tuple[str, ...]
    labels: np.ndarray
    scaffolds: np.ndarray
    fingerprints: np.ndarray
    excluded_rows: int


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (probabilities >= lower) & (
            probabilities <= upper if upper == 1 else probabilities < upper
        )
        if in_bin.any():
            error += in_bin.mean() * abs(
                float(y_true[in_bin].mean()) - float(probabilities[in_bin].mean())
            )
    return float(error)


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predictions = probabilities >= 0.5
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "ece": expected_calibration_error(y_true, probabilities),
    }


def load_endpoint_data(
    source_path: str | Path = SOURCE_FILE,
    endpoint: str = "TG414",
) -> EndpointData:
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Unsupported endpoint: {endpoint}")
    sheet_name, label_column = ENDPOINTS[endpoint]
    frame = pd.read_excel(source_path, sheet_name=sheet_name)
    required = {"SMILES_wash", label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{sheet_name} is missing required columns: {sorted(missing)}")

    frame = frame[["SMILES_wash", label_column]].copy()
    frame[label_column] = pd.to_numeric(frame[label_column], errors="coerce")
    frame = frame.dropna(subset=["SMILES_wash", label_column])
    if not frame[label_column].isin((0, 1)).all():
        raise ValueError(
            f"{sheet_name} contains non-binary {label_column} values; "
            "do not infer threshold labels."
        )

    prepared_rows: list[tuple[str, int, str, np.ndarray]] = []
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=FINGERPRINT_BITS
    )
    for smiles_value, label_value in frame.itertuples(index=False, name=None):
        smiles = str(smiles_value).strip()
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        scaffold_group = (
            f"scaffold:{scaffold}"
            if scaffold
            else f"acyclic:{canonical_smiles}"
        )
        fingerprint = np.zeros(FINGERPRINT_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(
            fingerprint_generator.GetFingerprint(molecule), fingerprint
        )
        prepared_rows.append(
            (canonical_smiles, int(label_value), scaffold_group, fingerprint)
        )

    labels_by_smiles: dict[str, set[int]] = {}
    for smiles, label, _, _ in prepared_rows:
        labels_by_smiles.setdefault(smiles, set()).add(label)
    conflicts = [smiles for smiles, labels in labels_by_smiles.items() if len(labels) > 1]
    if conflicts:
        raise ValueError(
            f"{endpoint} contains conflicting labels for {len(conflicts)} canonical structures."
        )

    unique_rows: dict[str, tuple[int, str, np.ndarray]] = {}
    for smiles, label, scaffold, fingerprint in prepared_rows:
        unique_rows.setdefault(smiles, (label, scaffold, fingerprint))
    smiles = tuple(unique_rows)
    labels = np.asarray([unique_rows[value][0] for value in smiles], dtype=np.int64)
    scaffolds = np.asarray([unique_rows[value][1] for value in smiles], dtype=object)
    fingerprints = np.stack([unique_rows[value][2] for value in smiles])
    excluded_rows = len(frame) - len(smiles)

    if len(np.unique(labels)) != 2:
        raise ValueError(f"{endpoint} must contain both binary classes.")
    if len(np.unique(scaffolds)) < N_SPLITS:
        raise ValueError(
            f"{endpoint} has fewer than {N_SPLITS} independent scaffold groups."
        )

    return EndpointData(
        endpoint=endpoint,
        smiles=smiles,
        labels=labels,
        scaffolds=scaffolds,
        fingerprints=fingerprints,
        excluded_rows=excluded_rows,
    )


def _models(train_labels: np.ndarray) -> dict[str, Any]:
    positives = int(train_labels.sum())
    negatives = int(len(train_labels) - positives)
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            scale_pos_weight=negatives / max(positives, 1),
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        ),
    }


def cross_validate_endpoint(data: EndpointData) -> dict[str, Any]:
    splitter = GroupKFold(n_splits=N_SPLITS)
    oof = {
        name: np.full(len(data.labels), np.nan, dtype=float)
        for name in ("logistic_regression", "random_forest", "xgboost")
    }
    fold_details = []
    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(data.fingerprints, data.labels, groups=data.scaffolds), start=1
    ):
        train_labels = data.labels[train_indices]
        test_labels = data.labels[test_indices]
        if len(np.unique(train_labels)) != 2 or len(np.unique(test_labels)) != 2:
            raise RuntimeError(
                f"{data.endpoint} fold {fold_number} lacks both classes; "
                "reliable grouped evaluation is not possible."
            )
        fold_details.append(
            {
                "fold": fold_number,
                "train_records": int(len(train_indices)),
                "test_records": int(len(test_indices)),
                "train_positive": int(train_labels.sum()),
                "train_negative": int(len(train_labels) - train_labels.sum()),
                "test_positive": int(test_labels.sum()),
                "test_negative": int(len(test_labels) - test_labels.sum()),
                "test_scaffolds": int(len(np.unique(data.scaffolds[test_indices]))),
            }
        )
        for model_name, model in _models(train_labels).items():
            model.fit(data.fingerprints[train_indices], train_labels)
            oof[model_name][test_indices] = model.predict_proba(
                data.fingerprints[test_indices]
            )[:, 1]

    if any(not np.isfinite(values).all() for values in oof.values()):
        raise RuntimeError(f"{data.endpoint} did not receive a complete out-of-fold prediction.")

    majority_accuracy = float(max(data.labels.mean(), 1 - data.labels.mean()))
    metrics = {}
    for name, probabilities in oof.items():
        metrics[name] = binary_metrics(data.labels, probabilities)
        metrics[name]["majority_class_accuracy"] = majority_accuracy

    return {
        "records": int(len(data.labels)),
        "excluded_invalid_or_duplicate_rows": data.excluded_rows,
        "positive": int(data.labels.sum()),
        "negative": int(len(data.labels) - data.labels.sum()),
        "positive_percent": float(100 * data.labels.mean()),
        "scaffold_groups": int(len(np.unique(data.scaffolds))),
        "split_method": f"{N_SPLITS}-fold GroupKFold by Bemis-Murcko scaffold",
        "features": f"RDKit Morgan radius 2, {FINGERPRINT_BITS} bits from SMILES only",
        "label_semantics": (
            f"Binary labels read as provided from {data.endpoint} in the *_smiles4 "
            "workbook sheet; independent source definitions were not included."
        ),
        "metrics_are": "pooled out-of-fold estimates, not an external validation result",
        "folds": fold_details,
        "models": metrics,
    }


def train_all(source_path: str | Path = SOURCE_FILE) -> dict[str, Any]:
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Reproductive toxicity source file not found: {source_path}")

    all_results: dict[str, Any] = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for endpoint in ENDPOINTS:
        data = load_endpoint_data(source_path, endpoint)
        result = cross_validate_endpoint(data)
        result["source_file"] = str(source_path.relative_to(ROOT)).replace("\\", "/")
        all_results[endpoint] = result

        for model_name, model in _models(data.labels).items():
            model.fit(data.fingerprints, data.labels)
            model_path = MODEL_DIR / f"{endpoint}_{model_name}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "endpoint": endpoint,
                    "feature_type": "RDKit Morgan radius 2, 1024-bit",
                    "label_semantics": result["label_semantics"],
                    "training_records": len(data.labels),
                    "training_scaffolds": len(np.unique(data.scaffolds)),
                    "training_scope": "exploratory nonclinical toxicity endpoint only",
                },
                model_path,
            )
        LOGGER.info(
            "%s: %d compounds, %d positives, %d scaffolds; metrics saved",
            endpoint,
            len(data.labels),
            int(data.labels.sum()),
            len(np.unique(data.scaffolds)),
        )

    output = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "exploratory_nonclinical_toxicity_only",
        "clinical_pregnancy_model_trained": False,
        "source_file": str(source_path.relative_to(ROOT)).replace("\\", "/"),
        "important_limitations": [
            "This workbook's label definitions and source citation were not included; verify them before scientific use.",
            "OECD test-guideline endpoints are nonclinical toxicity labels, not observed human pregnancy outcomes.",
            "Do not interpret these models as pregnancy risk predictions or as clinical safety evidence.",
            "Metrics are scaffold-held-out cross-validation estimates; there is no independent external test cohort.",
            "Brier score and ECE are descriptive and do not establish calibration.",
        ],
        "endpoints": all_results,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def _render_summary(result: dict[str, Any]) -> None:
    for endpoint, details in result["endpoints"].items():
        print(
            f"{endpoint}: n={details['records']}, "
            f"positive={details['positive']}, negative={details['negative']}, "
            f"scaffolds={details['scaffold_groups']}"
        )
        for model_name, metrics in details["models"].items():
            print(
                f"  {model_name}: accuracy={metrics['accuracy']:.3f}, "
                f"AUROC={metrics['auroc']:.3f}, AUPRC={metrics['auprc']:.3f}, "
                f"precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, "
                f"F1={metrics['f1']:.3f}, Brier={metrics['brier_score']:.3f}, "
                f"ECE={metrics['ece']:.3f}"
            )


if __name__ == "__main__":
    result = train_all()
    _render_summary(result)
    print(f"Exploratory results: {RESULTS_FILE}")
    print(f"Exploratory models: {MODEL_DIR}")
