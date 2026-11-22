import pandas as pd

from src.evidence_inventory import build_inventory, write_inventory


def test_inventory_records_measured_coverage_without_labels():
    rows = build_inventory()
    assert len(rows) >= 10
    frame = pd.DataFrame(rows)
    # No column is a training label or prediction.
    assert not any(
        column in frame.columns
        for column in ("label", "risk", "prediction", "confidence")
    )
    assert {"human_pregnancy_records", "general_adr_terms", "mapping_status"} <= set(
        frame.columns
    )
    # At least the core controls resolve in the local mapping.
    matched_names = set(frame.loc[frame["matched"], "requested_name"])
    assert {"isotretinoin", "thalidomide", "warfarin"} <= matched_names


def test_inventory_writes_csv_and_markdown(tmp_path):
    summary = write_inventory(tmp_path)
    assert summary["controls"] == len(build_inventory())
    csv_path = tmp_path / "pregnancy_evidence_inventory.csv"
    md_path = tmp_path / "pregnancy_evidence_inventory.md"
    assert csv_path.is_file() and md_path.is_file()
    frame = pd.read_csv(csv_path)
    assert "literature_reference_class" in frame.columns
    markdown = md_path.read_text(encoding="utf-8")
    assert "not a" in markdown and "labelled benchmark" in markdown
