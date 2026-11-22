from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def _run_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    return app


def test_dashboard_starts_with_system_status():
    app = _run_app()
    assert not app.exception
    assert any("Pregnancy-Related Drug Safety" in title.value for title in app.title)
    assert any("Research-support prototype" in warning.value for warning in app.warning)


def test_dashboard_renders_evidence_tabs_for_a_drug():
    app = _run_app()
    app.text_input(key="drug_query").set_value("tretinoin")
    app.run()
    assert not app.exception
    labels = [tab.label for tab in app.tabs]
    for expected in (
        "Overview",
        "General ADR",
        "Molecular",
        "Reproductive toxicity",
        "Pregnancy evidence",
        "MEPRank results",
        "Evidence comparison",
        "System status",
    ):
        assert expected in labels
    # The abstention is visible rather than replaced by a fabricated score.
    assert any(
        "No validated human pregnancy-outcome model" in item.value
        for item in app.markdown
    ) or any(
        "abstained" in item.value.lower() for item in app.markdown
    )


def test_dashboard_handles_unknown_drug_without_crashing():
    app = _run_app()
    app.text_input(key="drug_query").set_value("zzzz-not-a-real-drug")
    app.run()
    assert not app.exception
    assert app.error
    assert any("Could not resolve" in error.value for error in app.error)


def test_dashboard_handles_name_keyed_evidence_drug():
    app = _run_app()
    app.text_input(key="drug_query").set_value("isotretinoin")
    app.run()
    assert not app.exception
    assert any("isotretinoin" in subheader.value.lower() for subheader in app.subheader)
