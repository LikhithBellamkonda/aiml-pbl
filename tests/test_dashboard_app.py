from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.mepr_reasoning import _load_side_effects_for_drug

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def _run_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=240)
    app.run()
    return app


def test_dashboard_starts_on_one_page_query_ui():
    app = _run_app()
    assert not app.exception
    assert any("Pregnancy Drug Evidence Explorer" in title.value for title in app.title)
    assert any("Research-support prototype" in warning.value for warning in app.warning)
    labels = [box.label for box in app.selectbox]
    assert "Select Medication Name" in labels
    metric_labels = [metric.label for metric in app.metric]
    assert "Missing evidence domains (out of 3)" in metric_labels
    assert any("not higher fetal or maternal risk" in item.value for item in app.info)


def test_dashboard_renders_animal_adr_evidence_section():
    app = _run_app()
    app.selectbox[0].set_value("tretinoin")
    app.run()
    assert not app.exception
    markdown = " ".join(item.value for item in app.markdown)
    captions = " ".join(item.value for item in app.caption)
    assert "Animal Adverse Drug Reaction Evidence" in markdown
    assert "Supplementary evidence only" in captions
    assert "Observed pregnancy evidence" in markdown
    assert "Pregnancy outcome model" in " ".join(item.label for item in app.metric)


def test_dashboard_search_resolves_identifier_and_shows_identity():
    app = _run_app()
    app.text_input[0].set_value("444795")
    app.run()
    assert not app.exception
    markdown = " ".join(item.value for item in app.markdown)
    assert "Standardized drug identity" in markdown
    assert "Evidence availability by source" in markdown
    metrics = {item.label: item.value for item in app.metric}
    assert "PubChem CID" in metrics
    assert metrics.get("Pregnancy outcome model") == "Not trained"
    assert "Predicted foetal risk" not in metrics
    assert "Prediction confidence" not in metrics
    assert "Chemical representation" in markdown
    assert "Molecular weight (g/mol)" in metrics


def test_side_effect_panel_reports_measured_counts_without_an_index():
    app = _run_app()
    app.selectbox[0].set_value("tretinoin")
    app.run()
    assert not app.exception
    captions = " ".join(item.value for item in app.caption)
    assert "source-table rows" in captions.lower()
    assert "report-volume level" not in captions.lower()


def test_sider_term_rows_are_not_mislabeled_as_patient_reports():
    terms = _load_side_effects_for_drug("CID000444795")
    abdominal_distension = next(
        term for term in terms if term["term"] == "Abdominal distension"
    )

    assert abdominal_distension["source_entries"] == 2
    assert abdominal_distension["concept_types"] == "LLT, PT"
    assert "report_count" not in abdominal_distension


def test_dashboard_handles_unmatched_search_query():
    app = _run_app()
    app.text_input[0].set_value("not-a-real-drug-xyz")
    app.run()
    assert not app.exception
    assert any("No exact keyword match" in item.value for item in app.info)


def test_dashboard_renders_for_drug_without_animal_records():
    app = _run_app()
    app.selectbox[0].set_value("abacavir")
    app.run()
    assert not app.exception
    messages = " ".join(item.value for item in app.info)
    assert (
        "Animal ADR evidence unavailable" in messages
        or "Animal Adverse Drug Reaction Evidence" in " ".join(
            item.value for item in app.markdown
        )
    )
