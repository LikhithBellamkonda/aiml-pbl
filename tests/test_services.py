from pathlib import Path

import pandas as pd

from src.services.drug_service import (
    analyse_molecule,
    build_drug_profile,
    clear_cache,
    pregnancy_evidence,
    pregnancy_outcome_category,
    reproductive_toxicity_evidence,
    resolve_drug,
)


def test_resolve_by_name_identifier_and_cid():
    by_name = resolve_drug("tretinoin")
    assert by_name["matched"] and not by_name["ambiguous"]
    assert by_name["drug_id"] == "CID000444795"
    assert by_name["identity"]["smiles"]

    assert resolve_drug("444795")["drug_id"] == by_name["drug_id"]
    assert resolve_drug("CID000444795")["drug_id"] == by_name["drug_id"]
    assert resolve_drug("TRETINOIN")["drug_id"] == by_name["drug_id"]


def test_resolve_unknown_query_is_not_fabricated():
    clear_cache()
    result = resolve_drug("definitely-not-a-real-drug-xyz")
    assert result["matched"] is False
    assert result["drug_id"] is None
    assert result["warnings"]
    assert build_drug_profile("definitely-not-a-real-drug-xyz")["adr"]["available"] is False


def test_name_keyed_evidence_drug_is_resolvable_without_inventing_a_cid():
    profile = build_drug_profile("isotretinoin")
    assert profile["matched"] and not profile["ambiguous"]
    assert profile["drug_id"].startswith("NAME:")
    identity = profile["identity"]
    assert identity["pubchem_cid"] is None
    assert identity["resolution_status"] == "name_keyed_evidence_only"
    assert profile["pregnancy"]["available"]
    assert profile["pregnancy"]["record_count"] > 0
    assert any("pregnancy evidence index" in warning for warning in profile["warnings"])


def test_ambiguous_query_is_surfaced_not_guessed(tmp_path):
    pd.DataFrame(
        [
            {"original_name": "samename", "name_key": "samename", "drug_id": "CID000000001"},
            {"original_name": "samename", "name_key": "samename", "drug_id": "CID000000002"},
        ]
    ).to_csv(tmp_path / "drug_mapping.csv", index=False)

    result = resolve_drug("samename", data_dir=tmp_path)
    assert result["matched"] is False
    assert result["ambiguous"] is True
    assert len(result["candidates"]) == 2
    assert any("select one" in warning.lower() for warning in result["warnings"])


def test_missing_mapping_is_reported(tmp_path):
    result = resolve_drug("aspirin", data_dir=tmp_path)
    assert result["matched"] is False
    assert any("mapping is unavailable" in warning for warning in result["warnings"])


def test_molecular_descriptors_and_invalid_smiles():
    valid = analyse_molecule("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert valid["available"]
    assert valid["descriptors"]["molecular_weight"] > 100
    assert 0 < valid["fingerprint"]["on_bits"] <= 2048
    assert valid["scaffold"]

    invalid = analyse_molecule("this-is-not-a-smiles")
    assert invalid["available"] is False
    assert "could not be parsed" in invalid["reason"]

    missing = analyse_molecule(None)
    assert missing["available"] is False
    assert "No SMILES" in missing["reason"]


def test_human_and_nonclinical_evidence_stay_separate():
    pregnancy = pregnancy_evidence("CID000444795", "tretinoin")
    assert pregnancy["is_human_evidence"] is True
    assert pregnancy["records"]
    assert all("species" not in record for record in pregnancy["records"])
    assert set(pregnancy["category_counts"]) == {"maternal", "fetal", "neonatal"}

    toxicity = reproductive_toxicity_evidence("CID000011487")
    assert toxicity["is_human_evidence"] is False
    if toxicity["available"]:
        assert toxicity["species"]
        assert all(record["species"] for record in toxicity["records"])


def test_pregnancy_outcome_categories():
    assert pregnancy_outcome_category("congenital_anomaly") == "fetal"
    assert pregnancy_outcome_category("fetal_death") == "fetal"
    assert pregnancy_outcome_category("preterm_birth") == "neonatal"
    assert pregnancy_outcome_category("maternal_adverse_reaction") == "maternal"


def test_profile_reports_absent_evidence_without_zero_labels():
    profile = build_drug_profile("tretinoin")
    assert profile["matched"]
    assert profile["molecular"]["available"]
    assert profile["adr"]["available"]
    counts = profile["evidence_counts"]
    assert counts["adr_terms"] > 0
    assert counts["pregnancy_records"] >= 0
    # Evidence is a summary only; no risk score is exposed anywhere in the profile.
    assert "risk_score" not in profile
    assert "confidence" not in profile
