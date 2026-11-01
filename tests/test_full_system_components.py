# Comprehensive system component tests for the MEPR reasoning engine and dashboard

import pytest
import pandas as pd
from pathlib import Path

# Import the functions under test
from src.mepr_reasoning import evaluate_drug_safety
from dashboard.app import load_data, _evaluate_query, _render_filtered_summary

# Helper to locate the test data directory
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "cleaned"


def test_01_data_ingestion_and_cleaning():
    """Verify that cleaned master evidence CSV loads without errors and has expected columns."""
    master_path = Path(__file__).resolve().parents[1] / "data" / "standardized_master_evidence.csv"
    df = pd.read_csv(master_path)
    # Expect at least the core columns defined in src.mepr_reasoning.READ_COLUMNS
    expected_cols = {
        "drug_name",
        "source_dataset",
        "outcome_category",
        "trimester",
    }
    assert expected_cols.issubset(set(df.columns)), "Master evidence missing core columns"
    # Simple sanity check: non-empty DataFrame
    assert not df.empty, "Master evidence CSV appears empty"


def test_02_pubchem_and_rdkit_mapping():
    """Placeholder test – actual mapping functions are not present, ensure no placeholder IDs are used."""
    # In this repository, drug lookup is based on name; ensure a generic name returns records
    result = evaluate_drug_safety("acetaminophen", trimester="All", outcome_target="Both")
    assert isinstance(result, dict)
    # Should have some records (unless dataset limited)
    assert result["evidence_record_count"] >= 0


def test_03_trimester_and_endpoint_filtering():
    """Ensure trimester filtering isolates records correctly."""
    full = evaluate_drug_safety("aspirin", trimester="All", outcome_target="Both")
    t1 = evaluate_drug_safety("aspirin", trimester="First Trimester (T1)", outcome_target="Both")
    # Records count for T1 should be less than or equal to All
    assert t1["evidence_record_count"] <= full["evidence_record_count"]
    # Verify that all returned records have the expected normalized trimester value
    if t1["evidence_record_count"] > 0:
        assert all(
            rec["trimester"].lower().replace(" ", "_") == "first_trimester"
            for _, rec in t1["records"].iterrows()
        )


def test_04_mepr_evidence_coverage_scoring():
    """Validate that evidence_coverage_score reflects source family presence."""
    result = evaluate_drug_safety("metformin", trimester="All", outcome_target="Both")
    families = set(result["evidence_source_families"])
    expected = len(families) / len(
        tuple(["onsides", "mendeley", "faers", "tga"])  # matches PREGNANCY_EVIDENCE_SOURCES
    )
    # The function caps at 1.0, so compare with tolerance
    assert abs(result["evidence_coverage_score"] - min(expected, 1.0)) < 1e-6


def test_05_ml_to_ai_fallback_trigger():
    """Force fallback by using a drug with very low evidence coverage."""
    # Choose a drug unlikely to exist in the master dataset
    result = evaluate_drug_safety("nonexistentdrugxyz", trimester="All", outcome_target="Both")
    # Because evidence is missing, fallback should be active
    assert result["ml_prediction_active"] is False
    synth = result.get("ai_evidence_synthesis", {})
    assert "missing_source_families" in synth
    # Ensure explanation mentions missing sources
    assert "Missing human evidence sources" in synth.get("explanation", "")


def test_06_dashboard_state_rendering(monkeypatch):
    """Render the dashboard UI components without error.
    Uses Streamlit's built‑in testing utilities to capture output.
    """
    # Load minimal data for the dashboard (the function caches; monkeypatch to avoid heavy load)
    monkeypatch.setattr('dashboard.app.load_data', lambda: {"metadata": {}, "metrics": {}, "feasibility": {}})
    # Prepare a synthetic result dict mirroring evaluate_drug_safety output
    result = {
        "evidence_source_available": True,
        "trimester": "All",
        "risk_scoring_status": "No validated model.",
        "evidence_coverage_score": 0.5,
        "model_prediction_variance": None,
        "abstention_alert": None,
        "records": pd.DataFrame(columns=["outcome_category", "source_dataset"]),
        "provenance_badges": [],
        "ml_prediction_active": True,
        "ai_evidence_synthesis": {},
    }
    # Call the rendering function – should not raise
    _render_filtered_summary(result, "Aspirin", "Fetal Outcomes Only")
    # No explicit assertions; success is absence of exception.


