from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def _run_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=240)
    app.run()
    return app


def test_dashboard_starts_on_one_page_query_ui():
    app = _run_app()
    assert not app.exception
    assert any("Pregnancy-Related Drug Safety" in title.value for title in app.title)
    assert any("Research-support prototype" in warning.value for warning in app.warning)
    labels = [box.label for box in app.selectbox]
    assert "Select Medication Name" in labels


def test_dashboard_renders_animal_adr_evidence_section():
    app = _run_app()
    app.selectbox[0].set_value("tretinoin")
    app.run()
    assert not app.exception
    markdown = " ".join(item.value for item in app.markdown)
    captions = " ".join(item.value for item in app.caption)
    assert "Animal Adverse Drug Reaction Evidence" in markdown
    assert "Supplementary evidence only" in captions
    # The deterministic evidence panel and its caveat are present.
    assert "Pregnancy risk assessment" in markdown


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
    # No fabricated risk values are shown in the prediction panel.
    assert metrics.get("Predicted foetal risk") == "Unavailable"
    assert metrics.get("Prediction confidence") == "Unavailable"


def test_side_effect_panel_reports_measured_counts_without_an_index():
    app = _run_app()
    app.selectbox[0].set_value("tretinoin")
    app.run()
    assert not app.exception
    captions = " ".join(item.value for item in app.caption)
    assert "report-volume index" not in captions.lower()
    assert "report-volume level" in captions.lower()


def test_dashboard_handles_unmatched_search_query():
    app = _run_app()
    app.text_input[0].set_value("not-a-real-drug-xyz")
    app.run()
    assert not app.exception
    assert any("No local identifier mapping" in item.value for item in app.error)


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
