import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.services.inference import (
    TOX_ENDPOINTS,
    list_targets,
    model_status,
    predict,
)

TRETINOIN_SMILES = "CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC(=O)O)C)C"


def test_human_pregnancy_prediction_always_abstains():
    for target in ("fetal_death", "preterm_birth", "congenital_anomaly"):
        result = predict("CID000444795", target)
        assert result["target"] == target
        assert result["task"] == "human_pregnancy_outcome"
        assert result["status"] == "abstained"
        assert result["prediction"] is None
        assert result["uncertainty"] is None
        assert result["model"]["available"] is False
        assert any("pregnancy_labels.csv" in reason for reason in result["reasons"])


def test_unknown_target_is_unsupported():
    result = predict("CID000444795", "not_a_target")
    assert result["status"] == "unsupported"
    assert result["prediction"] is None
    assert "Supported targets" in result["reasons"][1]


def test_unresolved_drug_never_produces_a_prediction():
    result = predict(None, "fetal_death", query="zzz-not-a-drug")
    assert result["status"] == "abstained"
    assert result["prediction"] is None


def test_nonclinical_endpoint_without_checkpoint_abstains(tmp_path):
    result = predict(
        "CID000444795",
        "TG414",
        smiles=TRETINOIN_SMILES,
        model_dir=tmp_path / "missing",
        metrics_path=tmp_path / "missing.json",
    )
    assert result["status"] == "abstained"
    assert result["prediction"] is None
    assert any("No trained model checkpoint" in reason for reason in result["reasons"])


def test_missing_smiles_abstains_even_when_checkpoint_exists(tmp_path):
    _write_stub_checkpoint(tmp_path)
    result = predict(
        "CID000444795",
        "TG414",
        smiles="",
        model_dir=tmp_path / "tox",
        metrics_path=tmp_path / "metrics.json",
    )
    assert result["status"] == "abstained"
    assert any("no SMILES" in reason for reason in result["reasons"])


def test_stub_checkpoint_returns_a_real_probability(tmp_path):
    """A synthetic fixture exercises software behaviour only; it is not science."""
    _write_stub_checkpoint(tmp_path)
    result = predict(
        "CID000444795",
        "TG414",
        smiles=TRETINOIN_SMILES,
        model_dir=tmp_path / "tox",
        metrics_path=tmp_path / "metrics.json",
    )
    assert result["status"] == "ok"
    assert isinstance(result["prediction"], float)
    assert 0.0 <= result["prediction"] <= 1.0
    assert result["model"]["name"] == "logistic_regression"
    # No calibrated MC-dropout study exists for the stub, so uncertainty stays None.
    assert result["uncertainty"] is None
    assert "not a human pregnancy risk" in result["provenance"]["warning"]


def test_target_registry_lists_pregnancy_and_toxicity_tasks(tmp_path):
    targets = list_targets(tmp_path / "none", tmp_path / "none.json")
    pregnancy = [item for item in targets if item["task"] == "human_pregnancy_outcome"]
    toxicity = [
        item for item in targets if item["task"] == "nonclinical_reproductive_toxicity"
    ]
    assert len(pregnancy) == 4
    assert {item["target"] for item in toxicity} == set(TOX_ENDPOINTS)
    assert all(item["available"] is False for item in targets)


def test_model_status_reports_missing_checkpoints(tmp_path):
    status = model_status(tmp_path / "none", tmp_path / "none.json")
    assert status["human_pregnancy"]["available"] is False
    assert status["human_pregnancy"]["labels_present"] is False
    assert all(
        details["available"] is False
        for details in status["nonclinical_toxicity"].values()
    )


def _write_stub_checkpoint(tmp_path):
    rng = np.random.default_rng(0)
    features = rng.integers(0, 2, size=(60, 1024)).astype(np.uint8)
    labels = (features[:, 0] | features[:, 1]).astype(int)
    if len(np.unique(labels)) < 2:  # pragma: no cover - deterministic seed
        labels[0] = 1 - labels[0]
    classifier = LogisticRegression(max_iter=200).fit(features, labels)
    model_dir = tmp_path / "tox"
    model_dir.mkdir()
    joblib.dump(
        {"model": classifier, "endpoint": "TG414"},
        model_dir / "TG414_logistic_regression.joblib",
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "endpoints": {
                    "TG414": {
                        "models": {
                            "logistic_regression": {
                                "auroc": 0.5,
                                "auprc": 0.5,
                                "f1": 0.5,
                                "n": 60,
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
