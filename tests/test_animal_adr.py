from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.animal_adr import acquisition, features, mapping, preprocessing
from src.animal_adr.pipeline import run as run_animal_pipeline
from src.data_preprocessing import OUTCOME_NAMES, PregnancyDataset, collate_drugs
from src.models.animal_adr_encoder import AnimalADRBranch
from src.models.framework import MultimodalPregnancyADRModel
from src.services.animal_adr_service import animal_evidence, evidence_distinction

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _report(**overrides):
    record = {
        "unique_aer_id_number": "USA-X-0001",
        "original_receive_date": "20260115",
        "onset_date": "20260110",
        "serious_ae": "true",
        "outcome": [{"medical_status": "Died"}],
        "animal": {"species": "Dog", "breed": "Labrador", "gender": "Female"},
        "drug": [
            {
                "brand_name": ["TestBrand"],
                "active_ingredients": [{"name": "Pyrantel Pamoate;Sarolaner"}],
                "manufacturer": {"name": "ACME"},
                "route": "Oral",
                "atc_vet_code": "QP54AB52",
            }
        ],
        "reaction": [
            {
                "veddra_term_name": "Vomiting",
                "veddra_term_code": "334",
                "number_of_animals_affected": "1",
                "accuracy": "Actual",
            }
        ],
    }
    record.update(overrides)
    return record


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
def test_manifest_records_provenance_and_hashes():
    manifest = acquisition.acquire(download_missing=False, copy_raw=False)
    fda = manifest["sources"]["fda_adae"]
    assert fda["status"] == "acquired"
    entry = fda["files"][0]
    assert len(entry["sha256"]) == 64
    assert entry["bytes"] > 0
    assert entry["retrieved_at"]
    assert entry["records"] > 0
    # Secondary sources that cannot be scripted are marked, not faked.
    assert manifest["sources"]["ema_evvet"]["status"] == "manual_access_required"
    assert manifest["sources"]["uk_vmd"]["status"] == "manual_access_required"
    assert (DATA_DIR / "animal_adr_acquisition_manifest.json").exists()


def test_fda_archive_validates_and_parses():
    paths = acquisition._find_local(acquisition.SOURCES["fda_adae"]["local_patterns"])
    assert paths, "the FDA ADAE archive is part of the repository"
    validation = acquisition.validate_fda_archive(paths[0])
    assert validation["records"] > 0
    records = acquisition.read_fda_reports(paths[0])
    assert len(records) == validation["records"]
    assert "unique_aer_id_number" in records[0]


def test_missing_secondary_source_raises_instead_of_inventing_data(tmp_path):
    with pytest.raises(acquisition.SourceUnavailable):
        acquisition.ingest_evvet(tmp_path / "absent.csv")
    with pytest.raises(acquisition.SourceUnavailable):
        acquisition.ingest_vmd(tmp_path / "absent.csv")


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def test_flatten_keeps_original_and_normalised_reaction():
    frame = preprocessing.flatten_reports([_report()])
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["reaction_term_original"] == "Vomiting"
    assert row["reaction_normalized"] == "vomiting"
    assert row["reaction_category"] == "Digestive tract disorders"
    assert row["report_date"] == "2026-01-15"
    assert row["outcomes"] == "Died"


def test_species_normalisation_and_human_separation():
    assert preprocessing.normalise_species("canine") == "Dog"
    assert preprocessing.normalise_species("FELINE") == "Cat"
    assert preprocessing.normalise_species("") == "Unknown"
    frame = preprocessing.flatten_reports(
        [_report(animal={"species": "Human", "breed": ""})]
    )
    assert frame.iloc[0]["species"] == "Human"
    assert bool(frame.iloc[0]["is_human_exposure"]) is True


def test_missing_reaction_description_is_retained_and_flagged():
    frame = preprocessing.flatten_reports([_report(reaction=[])])
    assert len(frame) == 1
    row = frame.iloc[0]
    assert bool(row["missing_reaction_description"]) is True
    assert row["reaction_term_original"] == ""


def test_duplicate_report_rows_are_removed_but_reports_counted():
    records = [_report(), _report(), _report(unique_aer_id_number="USA-X-0002")]
    frame = preprocessing.flatten_reports(records)
    cleaned, stats = preprocessing._deduplicate(frame)
    assert len(cleaned) == 2
    assert stats["duplicate_rows_removed"] == 1


def test_veddra_soc_is_preferred_over_keyword_category():
    veddra = pd.DataFrame(
        {"soc": ["Systemic disorders"], "pt": ["Vomiting"], "llt": ["Vomiting"]}
    )
    frame = preprocessing.flatten_reports([_report()], veddra)
    assert frame.iloc[0]["reaction_category"] == "Systemic disorders"


# --------------------------------------------------------------------------- #
# Identifier mapping
# --------------------------------------------------------------------------- #
def _index(entries):
    index: dict[str, list[dict[str, str]]] = {}
    for alias, drug_id in entries:
        index.setdefault(mapping.name_key(alias), []).append(
            {"drug_id": drug_id, "mapping_source": "test"}
        )
    return index


def test_combination_product_is_preserved_without_causal_attribution():
    cleaned = preprocessing.flatten_reports([_report()])
    index = _index([("pyrantel pamoate", "CID1"), ("sarolaner", "CID2")])
    result = mapping.map_products(cleaned, index)
    row = result.iloc[0]
    assert row["mapping_status"] == "combination_all_ingredients_matched"
    assert row["attribution"] == "product_level_only"
    assert row["standardized_drug_id"] == ""
    assert "pyrantel" in row["matched_ingredient"]
    assert "sarolaner" in row["matched_ingredient"]


def test_unmatched_and_ambiguous_products_are_kept():
    cleaned = preprocessing.flatten_reports(
        [
            _report(unique_aer_id_number="USA-X-1"),
            _report(
                unique_aer_id_number="USA-X-2",
                drug=[{"brand_name": ["Unknown"], "active_ingredients": [{"name": "Mysteryium"}]}],
            ),
        ]
    )
    index = _index([("pyrantel pamoate", "CID1"), ("pyrantel pamoate", "CID9")])
    result = mapping.map_products(cleaned, index)
    statuses = set(result["mapping_status"])
    assert "ambiguous" in statuses
    assert "unmatched" in statuses
    assert result["mapping_candidates"].astype(str).str.len().gt(0).any()


def test_ingredient_cleaning_drops_strengths_and_presentation_words():
    assert mapping.clean_ingredient("Fluralaner 13.64% 12-Week Chew") == "fluralaner"
    assert mapping.split_ingredients("Lotilaner,Moxidectin,Praziquantel") == [
        "lotilaner",
        "moxidectin",
        "praziquantel",
    ]


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def _mapped_rows():
    return pd.DataFrame(
        [
            {
                "standardized_drug_id": "CID1",
                "report_id": "R1",
                "product_name": "A",
                "active_ingredient_original": "pyrantel",
                "mapping_status": "matched",
                "mapping_confidence": 0.95,
                "attribution": "ingredient_level",
                "species": "Dog",
                "is_human_exposure": False,
                "reaction_normalized": "vomiting",
                "reaction_category": "Digestive tract disorders",
                "serious_ae": "true",
                "outcomes": "Died",
                "report_date": "2026-01-01",
                "record_completeness": 0.8,
            },
            {
                "standardized_drug_id": "CID1",
                "report_id": "R2",
                "product_name": "A",
                "active_ingredient_original": "pyrantel",
                "mapping_status": "matched",
                "mapping_confidence": 0.9,
                "attribution": "ingredient_level",
                "species": "Cat",
                "is_human_exposure": False,
                "reaction_normalized": "lethargy",
                "reaction_category": "Neurological disorders",
                "serious_ae": "false",
                "outcomes": "Recovered/Normal",
                "report_date": "2026-02-01",
                "record_completeness": 0.6,
            },
        ]
    )


def test_feature_counts_availability_and_no_incidence(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "FEATURES_PATH", tmp_path / "features.csv")
    monkeypatch.setattr(features, "SUMMARY_PATH", tmp_path / "summary.json")
    summary = features.build_features(_mapped_rows())
    assert summary["drugs_with_animal_adr_evidence"] == 1
    frame = pd.read_csv(tmp_path / "features.csv").set_index("drug_id")
    row = frame.loc["CID1"]
    assert row["animal_adr_report_count"] == 2
    assert row["animal_adr_reaction_count"] == 2
    assert row["animal_adr_species_count"] == 2
    assert row["serious_report_count"] == 1
    assert row["death_or_euthanasia_report_count"] == 1
    assert "incidence" not in frame.columns
    assert summary["not_incidence"] is True


def test_evidence_for_drug_reports_unavailable_without_safe_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "FEATURES_PATH", tmp_path / "features.csv")
    features.build_features(_mapped_rows())
    available = features.evidence_for_drug("CID1")
    assert available["available"] is True
    missing = features.evidence_for_drug("CID-unknown")
    assert missing["available"] is False
    assert "does not indicate that the drug is safe" in missing["message"]


# --------------------------------------------------------------------------- #
# Encoder and fusion
# --------------------------------------------------------------------------- #
def test_encoder_zeroes_missing_evidence_and_encodes_present_evidence():
    torch.manual_seed(0)
    branch = AnimalADRBranch(out_dim=16).eval()
    numeric = torch.rand(2, len(features.NUMERIC_FEATURES))
    species = torch.tensor([1, 0])
    category = torch.tensor([2, 0])
    availability = torch.tensor([1.0, 0.0])
    output = branch(numeric, species, category, availability)
    assert output.shape == (2, 16)
    assert torch.equal(output[1], torch.zeros(16))
    assert not torch.equal(output[0], torch.zeros(16))


def test_fusion_with_animal_evidence_masks_unavailable_drugs():
    dataset = PregnancyDataset(
        DATA_DIR / "real_faers_adr.csv",
        DATA_DIR / "real_pubchem.csv",
        pregnancy_path=DATA_DIR / "real_pregnancy_features.csv",
        faers_path=DATA_DIR / "real_faers_summary.csv",
        animal_adr_path=DATA_DIR / "animal_adr_features.csv",
    )
    batch = collate_drugs([dataset[index] for index in range(3)])
    assert batch.availability_mask.shape == (3, 6)
    assert batch.animal_features.shape[0] == 3
    assert batch.animal_availability.shape == (3,)

    model = MultimodalPregnancyADRModel(
        num_outcomes=len(OUTCOME_NAMES), include_animal_adr=True
    ).eval()
    with torch.no_grad():
        predictions = model(batch)
    assert predictions.shape == (3, len(OUTCOME_NAMES))
    assert torch.isfinite(predictions).all()


def test_five_source_model_unchanged_without_animal_path():
    dataset = PregnancyDataset(
        DATA_DIR / "real_faers_adr.csv",
        DATA_DIR / "real_pubchem.csv",
        pregnancy_path=DATA_DIR / "real_pregnancy_features.csv",
        faers_path=DATA_DIR / "real_faers_summary.csv",
    )
    batch = collate_drugs([dataset[0], dataset[1]])
    assert batch.availability_mask.shape == (2, 5)
    model = MultimodalPregnancyADRModel(num_outcomes=len(OUTCOME_NAMES)).eval()
    with torch.no_grad():
        predictions = model(batch)
    assert predictions.shape == (2, len(OUTCOME_NAMES))


# --------------------------------------------------------------------------- #
# Service and pipeline
# --------------------------------------------------------------------------- #
def test_service_distinguishes_evidence_categories():
    categories = {item["evidence_type"] for item in evidence_distinction()}
    assert categories == {
        "human_pregnancy_specific",
        "human_spontaneous_adr",
        "animal_spontaneous_adr",
        "animal_experimental_toxicity",
    }


def test_service_reports_availability_for_known_and_unknown_drugs():
    known = animal_evidence("CID005311068")
    assert known["available"] is True
    assert known["evidence_type"] == "animal_spontaneous_adr"
    unknown = animal_evidence("CID999999999")
    assert unknown["available"] is False
    assert "safe" in unknown["message"]


def test_pipeline_stage_is_registered_without_changing_required_outputs():
    from src.run_pipeline import REQUIRED_OUTPUTS, STAGES

    names = [name for name, _ in STAGES]
    assert "animal_adr" in names
    # Animal artefacts are optional, so the core pipeline contract is unchanged.
    assert "animal_adr_features.csv" not in REQUIRED_OUTPUTS


def test_animal_pipeline_runs_on_repository_data():
    record = run_animal_pipeline(download_missing=False)
    assert record["status"] == "ok"
    assert record["stages"]["preprocess"]["unique_reports"] > 0
    assert record["stages"]["acquire"]["records"] == 18673
    summary = json.loads((DATA_DIR / "animal_adr_dataset_summary.json").read_text())
    assert summary["source"] == "FDA CVM ADAE"
    assert summary["evidence_type"] == "animal_spontaneous_adr"
    assert (DATA_DIR / "animal_adr_overlap.csv").exists()
