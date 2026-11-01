from pathlib import Path

import pandas as pd
import pytest

from src.data_preprocessing import PregnancyDataset
from src.mepr_reasoning import evaluate_drug_safety
from src.process_real_datasets import _extract_faers_outcomes
from src.train import train_model


def test_human_feature_dataset_has_no_animal_feature_branch(tmp_path):
    adr_path = tmp_path / "adr.csv"
    chemical_path = tmp_path / "chemical.csv"
    pregnancy_path = tmp_path / "pregnancy.csv"
    labels_path = tmp_path / "labels.csv"
    faers_path = tmp_path / "faers.csv"
    pd.DataFrame([{"drug_id": "drug-1", "has_adr_data": 1}]).to_csv(
        adr_path, index=False
    )
    pd.DataFrame([{"drug_id": "drug-1", "smiles": ""}]).to_csv(
        chemical_path, index=False
    )
    pd.DataFrame(
        [{"drug_id": "drug-1", "has_pregnancy_outcome_data": 1}]
    ).to_csv(pregnancy_path, index=False)
    pd.DataFrame([{"drug_id": "drug-1", "faers_report_count": 5}]).to_csv(
        faers_path, index=False
    )
    pd.DataFrame(
        [
            {"drug_id": "drug-1", "label_congenital_anomaly": 0, "mask_congenital_anomaly": 0}
        ]
    ).to_csv(labels_path, index=False)

    dataset = PregnancyDataset(
        adr_path,
        chemical_path,
        pregnancy_path=pregnancy_path,
        labels_path=labels_path,
        faers_path=faers_path,
    )
    sample = dataset[0]

    assert not hasattr(sample, "tox_features")
    assert sample.availability_mask.shape == (5,)
    assert sample.y_mask.sum().item() == 0


def test_aggregate_faers_counts_cannot_be_used_as_binary_ground_truth():
    supplement = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "FAERS_JAAD_Mendeley Supplement.docx"
    )
    outcomes, totals = _extract_faers_outcomes(supplement)
    isotretinoin = totals.loc[
        totals["drug_name"].str.casefold() == "isotretinoin"
    ].iloc[0]

    assert not outcomes.empty
    assert outcomes["evidence_type"].eq("aggregate spontaneous-report summary").all()
    assert isotretinoin["outcomes"] > isotretinoin["reports"]


def test_reasoner_excludes_nonhuman_or_nonreport_sources_from_human_signals():
    evidence = pd.DataFrame(
        [
            {
                "drug_name": "ExampleDrug",
                "drug_id": "drug-1",
                "source_dataset": "Mendeley pregnancy report counts",
                "outcome_category": "congenital_anomaly",
                "trimester": "unknown",
                "evidence_granularity": "drug_level_aggregate_spontaneous_reports",
            },
            {
                "drug_name": "ExampleDrug",
                "drug_id": "drug-1",
                "source_dataset": "ReproTox-KG",
                "outcome_category": "congenital_anomaly",
                "trimester": "first_trimester",
                "evidence_granularity": "knowledge_graph_association_not_patient_outcome",
            },
            {
                "drug_name": "ExampleDrug",
                "drug_id": "drug-1",
                "source_dataset": "Dryad study",
                "outcome_category": "congenital_anomaly",
                "trimester": "first_trimester",
                "evidence_granularity": "study_level_aggregate_table",
            },
        ]
    )

    result = evaluate_drug_safety(
        "ExampleDrug",
        trimester="First Trimester (T1)",
        outcome_target="Fetal Outcomes Only",
        drug_id="drug-1",
        evidence=evidence,
    )

    assert result["records"].empty
    assert isinstance(result["risk_score"], float)
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["model_prediction_variance"] is None
    assert result["abstention_alert"] is not None


def test_training_remains_blocked_without_validated_human_labels(monkeypatch):
    class DatasetWithoutLabels:
        merged_df = pd.DataFrame()
        mask_cols = ["mask_congenital_anomaly"]

    monkeypatch.setattr("src.train.PregnancyDataset", lambda *args, **kwargs: DatasetWithoutLabels())

    with pytest.raises(RuntimeError, match="no validated binary human pregnancy-outcome labels"):
        train_model()
