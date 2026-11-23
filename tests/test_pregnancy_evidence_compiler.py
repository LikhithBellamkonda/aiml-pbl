import json

import pandas as pd

import src.services.pregnancy_evidence_compiler as compiler


def test_transfer_statement_classifier_preserves_uncertainty():
    assert (
        compiler.classify_narrative_transfer_statement(
            "The drug is known to cross the placental barrier."
        )
        == "narrative_says_crossing"
    )
    assert (
        compiler.classify_narrative_transfer_statement(
            "The drug does not cross the placenta."
        )
        == "narrative_says_no_crossing"
    )
    assert (
        compiler.classify_narrative_transfer_statement(
            "The drug may cross the placenta."
        )
        == "narrative_suggests_possible_crossing"
    )
    assert (
        compiler.classify_narrative_transfer_statement("No transfer details.")
        == "no_explicit_transfer_statement"
    )


def test_compiler_keeps_animal_evidence_separate_and_abstains_on_risk(
    tmp_path, monkeypatch
):
    profile = {
        "matched": True,
        "ambiguous": False,
        "drug_id": "CID000000001",
        "drug_name": "example drug",
        "name_key": "example drug",
        "adr": {
            "available": True,
            "term_count": 2,
            "source_datasets": ["SIDER"],
        },
        "pregnancy": {
            "available": True,
            "record_count": 1,
            "categories": {"fetal": [{"outcome": "congenital_anomaly"}], "maternal": []},
            "sources": ["local-summary.csv"],
            "trimesters": [],
        },
        "reproductive_toxicity": {
            "available": True,
            "record_count": 1,
            "species": ["rat"],
            "study_types": ["DEV"],
        },
        "research_priority": {"available": True, "score": 1, "max_score": 3},
    }
    monkeypatch.setattr(compiler, "build_drug_profile", lambda query, data_dir: profile)
    monkeypatch.setattr(
        compiler,
        "animal_evidence",
        lambda drug_id, drug_name, data_dir: {
            "available": True,
            "report_count": 3,
            "species_count": 1,
            "top_species": "Dog",
            "serious_report_count": 1,
        },
    )
    pd.DataFrame(
        [
            {
                "drug_id": "CID000000001",
                "name_key": "example drug",
                "Safety statement": "The drug is known to cross the placental barrier.",
            }
        ]
    ).to_csv(tmp_path / "pregnancy_classifications.csv", index=False)
    pd.DataFrame(
        [{"cord_blood_concentration": None}]
    ).to_csv(tmp_path / "real_toxicity_evidence.csv", index=False)

    report = compiler.compile_pregnancy_evidence("example drug", data_dir=tmp_path)

    assert report["status"] == "evidence_compiled_risk_unavailable"
    assert report["evidence"]["animal_spontaneous_adr"]["report_count"] == 3
    assert report["evidence"]["animal_experimental_toxicity"]["record_count"] == 1
    assert report["transfer_assessment"]["direct_measurement_status"] == (
        "no_direct_measurement_field_found_in_checked_tables"
    )
    assert report["transfer_assessment"]["narrative_claims"][0]["status"] == (
        "narrative_says_crossing"
    )
    assert report["fetal_risk"]["value"] is None
    assert report["fetal_risk"]["confidence"] is None
    assert report["maternal_risk"]["value"] is None
    assert report["maternal_risk"]["confidence"] is None
    json.dumps(report)


def test_transfer_measurement_fields_require_populated_local_values(tmp_path):
    empty_path = tmp_path / "real_adr_evidence.csv"
    pd.DataFrame({"cord_blood_concentration": [None, ""]}).to_csv(
        empty_path, index=False
    )
    populated_path = tmp_path / "real_toxicity_evidence.csv"
    pd.DataFrame({"cord_blood_concentration": [0.2]}).to_csv(
        populated_path, index=False
    )

    fields = compiler._transfer_measurement_fields(tmp_path)

    assert fields == ["real_toxicity_evidence.csv:cord_blood_concentration"]
