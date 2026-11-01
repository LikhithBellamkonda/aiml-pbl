import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dashboard.data_access import find_drug, load_dashboard_data
from src.data_preprocessing import (
    BOND_FEATURES,
    OUTCOME_NAMES,
    PregnancyDataset,
    collate_drugs,
    get_dataloader,
)
from src.dataset_feasibility import build_feasibility_report
from src.models.framework import (
    MultimodalPregnancyADRModel,
    rank_outcomes,
)
from src.models.fusion import SourceGatedFusion
from src.process_real_datasets import (
    _extract_faers_outcomes,
    _query_pubchem,
    _sider_cid,
    drug_id_for_resolution,
    normalize_name,
    resolve_names,
)
from src.train import (
    binary_metrics,
    evaluate_missing_evidence,
    expected_calibration_error,
)


@pytest.fixture
def data_files(tmp_path):
    adr_path = tmp_path / "adr.csv"
    pubchem_path = tmp_path / "pubchem.csv"
    pregnancy_path = tmp_path / "pregnancy.csv"
    faers_path = tmp_path / "faers.csv"

    pd.DataFrame(
        [
            {"drug_id": "CID000002244", "has_adr_data": 1, "ADR_0": 1.0},
            {"drug_id": "NAME:unknown", "has_adr_data": 0, "ADR_0": 0.0},
        ]
    ).to_csv(adr_path, index=False)
    pd.DataFrame(
        [
            {
                "drug_id": "CID000002244",
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "resolution_status": "resolved",
            },
            {"drug_id": "NAME:unknown", "smiles": np.nan, "resolution_status": "unresolved"},
        ]
    ).to_csv(pubchem_path, index=False)
    pd.DataFrame(
        [
            {
                "drug_id": "CID000002244",
                "has_pregnancy_outcome_data": 1,
                "reports_fetal_death": 4,
                "reports_preterm_birth": 1,
                "reports_fetal_growth_restriction": 0,
                "reports_congenital_anomaly": 2,
            }
        ]
    ).to_csv(pregnancy_path, index=False)
    pd.DataFrame(
        [
            {
                "drug_id": "CID000002244",
                "faers_report_count": 8,
                "has_faers_data": 1,
            }
        ]
    ).to_csv(faers_path, index=False)
    return adr_path, pubchem_path, pregnancy_path, faers_path


def test_name_normalization_and_identifiers():
    assert normalize_name("  Aspirin  ") == "aspirin"
    assert _sider_cid("CID000010917") == 10917
    assert _sider_cid("CID100000085") == 85
    assert drug_id_for_resolution("unresolved medicine", {}) == "NAME:unresolved%20medicine"
    assert drug_id_for_resolution("aspirin", {"cid": 2244}) == "CID000002244"
    assert not drug_id_for_resolution("unresolved medicine", {}).startswith("CID999")


def test_pubchem_response_parsing(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "PropertyTable": {
                        "Properties": [
                            {
                                "CID": 2244,
                                "Title": "Aspirin",
                                "ConnectivitySMILES": "CC(=O)O",
                                "InChIKey": "test-key",
                                "MolecularFormula": "C9H8O4",
                                "MolecularWeight": 180.16,
                            }
                        ]
                    }
                }
            ).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse())
    record = _query_pubchem("aspirin")
    assert record["cid"] == 2244
    assert record["inchikey"] == "test-key"
    assert record["resolution_status"] == "resolved"


def test_pubchem_name_resolution_caches_multiple_names(monkeypatch, tmp_path):
    calls = []

    def fake_query(name):
        calls.append(name)
        return {
            "cid": len(calls),
            "resolution_status": "resolved",
            "resolution_source": "test",
        }

    monkeypatch.setattr("src.process_real_datasets._query_pubchem", fake_query)
    cache_path = tmp_path / "pubchem_cache.json"

    first = resolve_names(
        ["alpha medication", "beta medication"],
        cache_path=cache_path,
    )
    assert set(first) == {"alpha medication", "beta medication"}
    assert len(calls) == 2
    assert cache_path.exists()

    second = resolve_names(
        ["alpha medication", "beta medication"],
        cache_path=cache_path,
    )
    assert second == first
    assert len(calls) == 2


def test_dataset_feasibility_does_not_convert_reports_or_classifications_to_labels():
    report = build_feasibility_report()

    assert report["sources"]["dryad_antiepileptic"]["status"] == "not_found"
    assert report["sources"]["mendeley_dermatology"]["usable_binary_label_records"] == 0
    assert report["training_readiness"]["status"] == (
        "pregnancy_model_blocked_nonclinical_experiment_available"
    )
    assert all(
        target["decision"] == "not_trainable"
        for target in report["target_decisions"]
        if target["outcome"] in OUTCOME_NAMES
    )
    assert all(
        target["decision"] == "limited"
        for target in report["target_decisions"]
        if target["outcome"].startswith("nonclinical ")
    )


def test_dataset_preserves_missing_outcome_labels(data_files):
    adr, pubchem, pregnancy, faers = data_files
    dataset = PregnancyDataset(adr, pubchem, pregnancy, faers_path=faers)

    unknown = dataset[dataset.drug_ids.index("NAME:unknown")]
    assert unknown.y.shape == (len(OUTCOME_NAMES),)
    assert torch.equal(unknown.y_mask, torch.zeros(len(OUTCOME_NAMES)))
    assert unknown.availability_mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert unknown.faers_features.item() == 0.0

    aspirin = dataset[dataset.drug_ids.index("CID000002244")]
    assert aspirin.y_mask.sum().item() == 0
    assert aspirin.pregnancy_features[0].item() == 4.0
    assert aspirin.availability_mask.tolist() == [1, 1, 1, 1, 1]


def test_dataloader_and_outcome_specific_model(data_files):
    adr, pubchem, pregnancy, faers = data_files
    loader = get_dataloader(
        adr, pubchem, batch_size=2, shuffle=False,
        pregnancy_path=pregnancy, faers_path=faers
    )
    batch = next(iter(loader))
    assert batch.adr_features.shape == (2, 20)
    assert batch.availability_mask.shape == (2, 5)
    assert batch.y.shape == (2, len(OUTCOME_NAMES))
    assert batch.edge_attr.shape[1] == BOND_FEATURES

    model = MultimodalPregnancyADRModel(num_outcomes=len(OUTCOME_NAMES))
    predictions = model(batch)
    assert predictions.shape == (2, len(OUTCOME_NAMES))
    assert torch.isfinite(predictions).all()


def test_fusion_handles_all_sources_missing():
    fusion = SourceGatedFusion(rep_dim=8)
    representations = [torch.randn(2, 8) for _ in range(5)]
    missing = torch.zeros(2, 5)
    output = fusion(representations, missing, missing)
    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()
    assert torch.equal(output, torch.zeros_like(output))


def test_missing_evidence_evaluation_keeps_unlabeled_targets_masked(data_files):
    adr, pubchem, pregnancy, faers = data_files
    loader = get_dataloader(
        adr, pubchem, batch_size=2, shuffle=False,
        pregnancy_path=pregnancy, faers_path=faers
    )
    model = MultimodalPregnancyADRModel(num_outcomes=len(OUTCOME_NAMES))
    results = evaluate_missing_evidence(model, loader, torch.device("cpu"))
    assert "adr_and_chemical_only" in results
    assert results["complete_available_sources"]["fetal_death"]["n"] == 0


def test_molecular_graph_has_real_atom_and_bond_features():
    pytest.importorskip("rdkit")
    from src.data_preprocessing import _smiles_to_graph

    graph = _smiles_to_graph("CCO")
    assert graph is not None
    x, edge_index, edge_attr, descriptors = graph
    assert x.shape == (3, 29)
    assert edge_index.shape == (2, 4)
    assert edge_attr.shape == (4, BOND_FEATURES)
    assert descriptors.shape == (6,)
    assert torch.isfinite(x).all()


def test_uncertainty_ranking_and_calibration(data_files):
    adr, pubchem, pregnancy, faers = data_files
    batch = next(
        iter(
            get_dataloader(
                adr, pubchem, batch_size=2, shuffle=False,
                pregnancy_path=pregnancy, faers_path=faers
            )
        )
    )
    model = MultimodalPregnancyADRModel(num_outcomes=len(OUTCOME_NAMES))
    mean, variance = model.predict_with_uncertainty(batch, num_samples=4)
    assert mean.shape == variance.shape == (2, len(OUTCOME_NAMES))
    assert (variance >= 0).all()
    assert model.training is True
    ranked = rank_outcomes(torch.tensor([0.2, 0.8]), ["a", "b"])
    assert [name for name, _ in ranked] == ["b", "a"]
    assert [score for _, score in ranked] == pytest.approx([0.8, 0.2])
    metrics = binary_metrics([0, 1], [0.1, 0.9])
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["brier_score"] < 0.02
    assert expected_calibration_error([0, 1], [0.1, 0.9]) is not None


def test_local_faers_supplement_contains_aggregate_outcomes():
    path = Path(__file__).resolve().parents[1] / "datasets" / "FAERS_JAAD_Mendeley Supplement.docx"
    outcomes, totals = _extract_faers_outcomes(path)
    isotretinoin = totals[totals["drug_name"].str.casefold() == "isotretinoin"].iloc[0]
    assert isotretinoin["reports"] == 1713
    assert not outcomes[outcomes["drug_name"] == "Isotretinoin"].empty
    assert set(outcomes["evidence_type"]) == {"aggregate spontaneous-report summary"}


def test_dashboard_lookup_handles_unknown_name(tmp_path):
    mapping = pd.DataFrame(
        [{"original_name": "Aspirin", "drug_id": "CID000002244", "pubchem_cid": 2244}]
    )
    assert find_drug("aspirin", mapping)["drug_id"] == "CID000002244"
    assert find_drug("2244", mapping)["drug_id"] == "CID000002244"
    assert find_drug("unknown drug", mapping) is None
    (tmp_path / "data").mkdir()
    data = load_dashboard_data(tmp_path)
    assert data["mapping"].empty
