import pandas as pd
import pytest

from src.mepr_reasoning import evaluate_drug_safety


@pytest.fixture
def trimester_evidence():
    return pd.DataFrame(
        [
            {
                "drug_name": "ExampleDrug",
                "drug_id": "NAME:exampledrug",
                "source_dataset": "Mendeley pregnancy study",
                "outcome_category": "congenital_anomaly",
                "outcome_term_original": "structural anomaly",
                "trimester": "first_trimester",
                "association_value": 99.0,
                "source_record_id": "t1-only",
            },
            {
                "drug_name": "ExampleDrug",
                "drug_id": "NAME:exampledrug",
                "source_dataset": "Mendeley pregnancy study",
                "outcome_category": "preterm_birth",
                "outcome_term_original": "preterm birth",
                "trimester": "second_trimester",
                "association_value": 0.3,
                "source_record_id": "t2-mendeley",
            },
            {
                "drug_name": "ExampleDrug",
                "drug_id": "NAME:exampledrug",
                "source_dataset": "ReproTox-KG",
                "outcome_category": "preterm_birth",
                "outcome_term_original": "preterm birth",
                "trimester": "second_trimester",
                "association_value": 0.4,
                "source_record_id": "t2-reprotox",
            },
            {
                "drug_name": "ExampleDrug",
                "drug_id": "NAME:exampledrug",
                "source_dataset": "FDA FAERS",
                "outcome_category": "preterm_birth",
                "outcome_term_original": "preterm birth",
                "trimester": "second_trimester",
                "association_value": 0.5,
                "source_record_id": "t2-faers",
            },
            {
                "drug_name": "ExampleDrug",
                "drug_id": "NAME:exampledrug",
                "source_dataset": "Mendeley pregnancy study",
                "outcome_category": "fetal_growth_outcome",
                "outcome_term_original": "fetal growth",
                "trimester": "third_trimester",
                "association_value": 0.2,
                "source_record_id": "t3-mendeley",
            },
            {
                "drug_name": "ExampleDrug",
                "drug_id": "NAME:exampledrug",
                "source_dataset": "ReproTox-KG",
                "outcome_category": "fetal_growth_outcome",
                "outcome_term_original": "fetal growth",
                "trimester": "third_trimester",
                "association_value": 0.5,
                "source_record_id": "t3-reprotox",
            },
            {
                "drug_name": "ExampleDrug",
                "drug_id": "NAME:exampledrug",
                "source_dataset": "OnSIDES",
                "outcome_category": "fetal_growth_outcome",
                "outcome_term_original": "fetal growth",
                "trimester": "third_trimester",
                "association_value": 0.6,
                "source_record_id": "t3-onsides",
            },
            {
                "drug_name": "ExampleDrug",
                "drug_id": "NAME:exampledrug",
                "source_dataset": "Dryad study",
                "outcome_category": "fetal_growth_outcome",
                "outcome_term_original": "fetal growth",
                "trimester": "third_trimester",
                "association_value": 0.7,
                "source_record_id": "t3-dryad",
            },
        ]
    )


@pytest.mark.parametrize(
    ("trimester", "expected_ids", "expected_coverage"),
    [
        ("First Trimester (T1)", ["t1-only"], 0.25),
        (
            "Second Trimester (T2)",
            ["t2-mendeley", "t2-faers"],
            0.5,
        ),
        (
            "Third Trimester (T3)",
            [
                "t3-mendeley",
                "t3-onsides",
            ],
            0.5,
        ),
    ],
)
def test_trimester_filters_produce_distinct_unleaked_coverage(
    trimester_evidence, trimester, expected_ids, expected_coverage
):
    result = evaluate_drug_safety(
        "ExampleDrug",
        trimester=trimester,
        outcome_target="Fetal Outcomes Only",
        evidence=trimester_evidence,
    )

    assert sorted(result["records"]["source_record_id"]) == sorted(expected_ids)
    assert result["evidence_coverage_score"] == expected_coverage
    assert result["risk_score"] is None
    assert result["model_prediction_variance"] is None
    assert not result["records"]["source_dataset"].str.contains(
        "ReproTox|Dryad", case=False, regex=True
    ).any()
    if trimester != "First Trimester (T1)":
        assert "t1-only" not in result["records"]["source_record_id"].tolist()


def test_timing_specific_abstention_alert_does_not_use_other_trimesters(
    trimester_evidence,
):
    result = evaluate_drug_safety(
        "ExampleDrug",
        trimester="Third Trimester (T3)",
        outcome_target="Maternal Outcomes Only",
        evidence=trimester_evidence,
    )

    assert result["records"].empty
    assert result["abstention_alert"] == (
        "Insufficient evidence available specifically for Third Trimester (T3). "
        "Model abstains from risk scoring."
    )
    assert result["evidence_coverage_score"] == 0


def test_outcome_target_excludes_unselected_endpoint_types(trimester_evidence):
    maternal = trimester_evidence.copy()
    maternal.loc[len(maternal)] = {
        "drug_name": "ExampleDrug",
        "drug_id": "NAME:exampledrug",
        "source_dataset": "Mendeley pregnancy study",
        "outcome_category": "maternal_adverse_reaction",
        "outcome_term_original": "maternal reaction",
        "trimester": "third_trimester",
        "association_value": 0.9,
        "source_record_id": "maternal-only",
    }

    result = evaluate_drug_safety(
        "ExampleDrug",
        trimester="Third Trimester (T3)",
        outcome_target="Maternal Outcomes Only",
        evidence=maternal,
    )

    assert result["records"]["source_record_id"].tolist() == ["maternal-only"]


def test_filter_values_are_validated(trimester_evidence):
    with pytest.raises(ValueError, match="Unsupported trimester"):
        evaluate_drug_safety(
            "ExampleDrug", trimester="Second trimester", evidence=trimester_evidence
        )
    with pytest.raises(ValueError, match="Unsupported outcome target"):
        evaluate_drug_safety(
            "ExampleDrug", outcome_target="Fetal", evidence=trimester_evidence
        )


def test_missing_master_dataset_is_not_reported_as_zero_coverage(monkeypatch, tmp_path):
    from src import mepr_reasoning

    monkeypatch.setattr(
        mepr_reasoning,
        "MASTER_EVIDENCE_PATH",
        tmp_path / "missing-master.csv",
    )
    result = evaluate_drug_safety(
        "ExampleDrug",
        trimester="First Trimester (T1)",
        outcome_target="Fetal Outcomes Only",
    )

    assert not result["evidence_source_available"]
    assert result["evidence_coverage_score"] is None
    assert result["abstention_alert"] is None
