import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

from src.ingestion import dryad, onsides, reprotox
from src.ingestion.common import (
    IngestedSource,
    classify_pregnancy_outcome,
    normalize_trimester,
)
from src.ingestion.pipeline import (
    _availability_matrices,
    _make_identifier_map,
    _source_summary,
    _write_analysis,
)


def test_trimester_requires_explicit_timing_and_fetal_death_is_not_stillbirth():
    assert normalize_trimester()[0] == "unknown"
    assert normalize_trimester(gestational_weeks=10)[0] == "first_trimester"
    assert normalize_trimester(gestational_weeks=20)[0] == "second_trimester"
    assert normalize_trimester(gestational_weeks=32)[0] == "third_trimester"
    assert normalize_trimester(exposure_period="first trimester")[0] == "first_trimester"
    assert normalize_trimester(gestational_weeks="22+3 weeks") == (
        "second_trimester",
        22 + 3 / 7,
    )
    assert normalize_trimester(
        explicit_value="first trimester", gestational_weeks=10
    ) == ("first_trimester", 10.0)
    assert classify_pregnancy_outcome("fetal death") != "stillbirth"
    assert classify_pregnancy_outcome("stillbirth") == "stillbirth"


def test_dryad_supplement_remains_aggregate_study_evidence():
    path = Path(__file__).resolve().parents[1] / "datasets" / dryad.EXPECTED_FILENAME
    result = dryad.ingest(path)

    assert result.available
    assert result.status == "available_aggregate_tables_only"
    assert result.records["evidence_granularity"].eq("study_level_aggregate_table").all()
    assert result.records["outcome_category"].eq("").all()
    valproic_rows = result.records.loc[result.records["drug_name"] == "Valproic Acid"]
    assert not valproic_rows.empty
    assert set(valproic_rows["study_endpoint"]) == {"ASD", "ADHD"}
    assert not result.records["individual_outcome_observation"].any()


def test_onsides_aggregates_label_associations_without_creating_pregnancy_labels(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "small-onsides.zip"
    tables = {
        "csv/product_label.csv": pd.DataFrame(
            [
                {
                    "label_id": 1,
                    "source": "FDA",
                    "source_product_name": "Aspirin",
                    "source_product_id": "test-label-1",
                    "source_label_url": "https://example.test/label",
                },
                {
                    "label_id": 2,
                    "source": "FDA",
                    "source_product_name": "Aspirin / caffeine",
                    "source_product_id": "test-label-2",
                    "source_label_url": "https://example.test/label-2",
                },
            ]
        ),
        "csv/product_to_rxnorm.csv": pd.DataFrame(
            [
                {"label_id": 1, "rxnorm_product_id": "123"},
                {"label_id": 2, "rxnorm_product_id": "456"},
            ]
        ),
        "csv/vocab_rxnorm_product.csv": pd.DataFrame(
            [
                {"rxnorm_id": "123", "rxnorm_name": "aspirin", "rxnorm_term_type": "Ingredient"},
                {"rxnorm_id": "456", "rxnorm_name": "aspirin / caffeine", "rxnorm_term_type": "Clinical Drug"},
            ]
        ),
        "csv/vocab_rxnorm_ingredient.csv": pd.DataFrame(
            [
                {"rxnorm_id": "10600", "rxnorm_name": "aspirin", "rxnorm_term_type": "Ingredient"},
                {"rxnorm_id": "2019", "rxnorm_name": "caffeine", "rxnorm_term_type": "Ingredient"},
            ]
        ),
        "csv/vocab_rxnorm_ingredient_to_product.csv": pd.DataFrame(
            [
                {"product_id": "123", "ingredient_id": "10600"},
                {"product_id": "456", "ingredient_id": "10600"},
                {"product_id": "456", "ingredient_id": "2019"},
            ]
        ),
        "csv/vocab_meddra_adverse_effect.csv": pd.DataFrame(
            [{"meddra_id": 1001, "meddra_name": "Headache", "meddra_term_type": "PT"}]
        ),
        "csv/product_adverse_effect.csv": pd.DataFrame(
            [
                {
                    "product_label_id": 1,
                    "effect_id": 42,
                    "label_section": "AR",
                    "effect_meddra_id": 1001,
                    "match_method": "test",
                    "pred0": 0.1,
                    "pred1": 5.2,
                },
                {
                    "product_label_id": 2,
                    "effect_id": 43,
                    "label_section": "AR",
                    "effect_meddra_id": 1001,
                    "match_method": "test",
                    "pred0": 0.2,
                    "pred1": 4.2,
                },
            ]
        ),
    }
    with zipfile.ZipFile(source_path, "w") as archive:
        for name, frame in tables.items():
            archive.writestr(name, frame.to_csv(index=False))
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    monkeypatch.setattr(onsides, "EXPECTED_SHA256", digest)

    result = onsides.ingest(source_path, chunksize=1)

    assert result.available
    assert result.status == "available"
    assert len(result.records) == 2
    aspirin = result.records.loc[result.records["rxnorm_product_id"] == "123"].iloc[0]
    assert aspirin["drug_name"] == "aspirin"
    assert aspirin["generic_ingredient_name"] == "aspirin"
    assert aspirin["source_ingredient_count"] == 1
    assert not aspirin["is_combination"]
    assert aspirin["source_record_count"] == 1
    assert aspirin["evidence_status"] == "regulatory_label_adverse_effect_association"
    assert aspirin["outcome_category"] == ""
    assert aspirin["trimester"] == "unknown"
    assert aspirin["max_pred1"] == 5.2
    combination = result.records.loc[
        result.records["rxnorm_product_id"] == "456"
    ].iloc[0]
    assert combination["drug_name"] == "aspirin / caffeine"
    assert combination["generic_ingredient_name"] == ""
    assert combination["source_ingredient_count"] == 2
    assert combination["is_combination"]
    assert json.loads(combination["generic_ingredient_names"]) == [
        "aspirin",
        "caffeine",
    ]


def test_reprotox_edges_are_knowledge_graph_associations_not_observed_events(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "version": "test",
                "nodes": {
                    "drug": {
                        "type": "Drug",
                        "properties": {
                            "id": "CID:2244",
                            "label": "aspirin",
                            "InChI_key": "test",
                            "SMILES": "CCO",
                        },
                    },
                    "defect": {
                        "type": "BirthDefect",
                        "properties": {"id": "bd1", "label": "example defect"},
                    },
                },
                "edges": [
                    {
                        "source": "drug",
                        "target": "defect",
                        "relation": "associated_with",
                        "properties": {"score": 0.7, "resource": "test"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = reprotox.ingest(path)

    assert result.available
    assert result.records.iloc[0]["drug_id"] == "CID2244"
    assert result.records.iloc[0]["evidence_status"] == "curated_or_computational_association"
    assert result.records.iloc[0]["evidence_granularity"] == "knowledge_graph_association_not_patient_outcome"


def test_pipeline_availability_does_not_turn_unqueried_or_missing_into_negative():
    mendeley = pd.DataFrame(
        [
            {
                "drug_name": "Aspirin",
                "drug_id": "CID000002244",
                "source_record_id": "m1",
                "source_identifier": "",
                "outcome_category": "miscarriage",
                "evidence_status": "positive_safety_concern_evidence",
            }
        ]
    )
    pubchem = pd.DataFrame(
        [
            {
                "drug_name": "Aspirin",
                "drug_id": "CID000002244",
                "source_record_id": "aspirin",
                "source_identifier": "aspirin",
                "pubchem_cid": 2244,
                "resolution_status": "resolved",
                "outcome_category": "",
                "evidence_status": "",
            },
            {
                "drug_name": "Unknown",
                "drug_id": "NAME:unknown",
                "source_record_id": "unknown",
                "source_identifier": "unknown",
                "pubchem_cid": pd.NA,
                "resolution_status": "not_queried",
                "outcome_category": "",
                "evidence_status": "",
            },
        ]
    )
    sources = {
        "mendeley_pregnancy_v2": IngestedSource(
            "mendeley_pregnancy_v2", True, status="available"
        ),
        "dryad_antiepileptic_pregnancy": IngestedSource(
            "dryad_antiepileptic_pregnancy",
            False,
            status="source_unavailable",
        ),
        "pubchem": IngestedSource("pubchem", True, status="available_cached"),
    }
    canonical_frames = {
        "mendeley_pregnancy_v2": mendeley,
        "dryad_antiepileptic_pregnancy": pd.DataFrame(),
        "pubchem": pubchem,
    }
    drug_mapping = pd.DataFrame(
        [
            {"drug_id": "CID000002244", "preferred_name": "Aspirin"},
            {"drug_id": "NAME:unknown", "preferred_name": "Unknown"},
        ]
    )

    availability, outcome_matrix = _availability_matrices(
        sources, canonical_frames, drug_mapping
    )

    aspirin_miscarriage = outcome_matrix.loc[
        (outcome_matrix["drug_id"] == "CID000002244")
        & (outcome_matrix["source_dataset"] == "mendeley_pregnancy_v2")
        & (outcome_matrix["outcome"] == "miscarriage")
    ].iloc[0]
    assert aspirin_miscarriage["evidence_status"] == "positive_safety_concern_evidence"
    assert aspirin_miscarriage["observed_negative_records"] == 0
    assert aspirin_miscarriage["validated_binary_label_records"] == 0
    dryad_status = outcome_matrix.loc[
        (outcome_matrix["drug_id"] == "CID000002244")
        & (outcome_matrix["source_dataset"] == "dryad_antiepileptic_pregnancy")
    ]
    assert dryad_status["evidence_status"].eq("source_unavailable").all()
    pubchem_unknown = availability.loc[
        (availability["drug_id"] == "NAME:unknown")
        & (availability["source_dataset"] == "pubchem")
    ].iloc[0]
    assert pubchem_unknown["evidence_availability"] == "source_not_queried"


def test_identifier_map_deduplicates_repeated_evidence_rows():
    repeated = pd.DataFrame(
        [
            {
                "drug_name": "Aspirin",
                "source_identifier": "RxNorm:1",
                "source_record_id": "effect-1",
            },
            {
                "drug_name": "Aspirin",
                "source_identifier": "RxNorm:1",
                "source_record_id": "effect-2",
            },
        ]
    )
    pubchem = pd.DataFrame(
        [
            {
                "drug_name": "Aspirin",
                "pubchem_cid": 2244,
                "inchikey": "key",
                "smiles": "CCO",
                "resolution_status": "resolved",
            }
        ]
    )

    identifier_map, drug_mapping, _ = _make_identifier_map(
        {"onsides_general_adr": repeated},
        pd.DataFrame(),
        pubchem,
    )

    assert len(identifier_map) == 1
    assert drug_mapping.iloc[0]["drug_id"] == "CID000002244"


def test_identifier_map_uses_only_documented_single_ingredient_relations():
    source_records = pd.DataFrame(
        [
            {
                "drug_name": "Aspirin 500 mg tablet",
                "source_identifier": "RxNorm:single",
                "generic_ingredient_name": "aspirin",
                "generic_ingredient_ids": '["2244"]',
            },
            {
                "drug_name": "aspirin / caffeine",
                "source_identifier": "RxNorm:combination",
                "generic_ingredient_name": "",
                "generic_ingredient_ids": '["2244","2519"]',
            },
        ]
    )
    pubchem = pd.DataFrame(
        [
            {
                "drug_name": "aspirin",
                "pubchem_cid": 2244,
                "inchikey": "key",
                "smiles": "CCO",
                "resolution_status": "resolved",
            }
        ]
    )

    identifier_map, _, canonical_by_key = _make_identifier_map(
        {"onsides_general_adr": source_records},
        pd.DataFrame(),
        pubchem,
    )

    single = identifier_map.loc[
        identifier_map["source_identifier"] == "RxNorm:single"
    ].iloc[0]
    combination = identifier_map.loc[
        identifier_map["source_identifier"] == "RxNorm:combination"
    ].iloc[0]
    assert single["canonical_drug_id"] == "CID000002244"
    assert single["match_method"] == "rxnorm_single_ingredient_to_pubchem"
    assert combination["canonical_drug_id"].startswith("NAME:")
    assert combination["is_combination"]
    assert canonical_by_key["aspirin"] == "CID000002244"


def test_analysis_report_includes_overlap_label_and_missingness_tables(tmp_path):
    sources = {
        "mendeley_pregnancy_v2": IngestedSource(
            "mendeley_pregnancy_v2", True, status="available"
        ),
        "dryad_antiepileptic_pregnancy": IngestedSource(
            "dryad_antiepileptic_pregnancy",
            True,
            status="available_aggregate_tables_only",
        ),
    }
    summary = pd.DataFrame(
        [
            {
                "source_dataset": name,
                "status": source.status,
                "cleaned_record_count": 1,
                "unique_drug_count": 1,
                "pubchem_matched_drug_count": 1,
            }
            for name, source in sources.items()
        ]
    )
    overlaps = pd.DataFrame(
        [
            {
                "source_a": "mendeley_pregnancy_v2",
                "source_b": "dryad_antiepileptic_pregnancy",
                "unique_drugs_a": 1,
                "unique_drugs_b": 1,
                "overlapping_drug_count": 1,
            }
        ]
    )
    master = pd.DataFrame(
        [
            {
                "source_dataset": "mendeley_pregnancy_v2",
                "source_record_id": "m1",
                "drug_id": "CID000002244",
                "drug_name": "Aspirin",
                "outcome_category": "miscarriage",
                "evidence_status": "positive_safety_concern_evidence",
                "trimester": "unknown",
            }
        ]
    )
    identifier_map = pd.DataFrame(
        [
            {
                "source_dataset": "mendeley_pregnancy_v2",
                "source_record_id": "m1",
                "drug_name_key": "aspirin",
                "pubchem_cid": 2244,
            }
        ]
    )
    outcome_matrix = pd.DataFrame(
        [
            {
                "observed_positive_concern_records": 1,
                "observed_negative_records": 0,
            }
        ]
    )

    report = _write_analysis(
        tmp_path,
        sources,
        summary,
        overlaps,
        master,
        identifier_map,
        outcome_matrix,
    )

    report_path = tmp_path / "reports" / "dataset_integration_feasibility.md"
    text = report_path.read_text(encoding="utf-8")
    assert report["target_decisions"][0]["decision"] == "not_trainable"
    assert "Pairwise drug overlap" in text
    assert "validated_negative_label_count" in text
    assert "Master missingness" in text
    assert (tmp_path / "reports" / "figures" / "drug_coverage.png").is_file()


def test_pubchem_summary_counts_only_records_with_pubchem_matches():
    pubchem = IngestedSource(
        "pubchem",
        True,
        pd.DataFrame(
            [
                {"drug_id": "CID000000001", "pubchem_cid": 1},
                {"drug_id": "CID000000001", "pubchem_cid": 1},
                {"drug_id": "CID000000002", "pubchem_cid": pd.NA},
            ]
        ),
        status="available_cached",
    )

    summary = _source_summary({"pubchem": pubchem}, {"pubchem": pubchem.records})

    assert summary.loc[0, "unique_drug_count"] == 1
    assert summary.loc[0, "pubchem_matched_drug_count"] == 1
