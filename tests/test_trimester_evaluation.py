import json

import src.trimester_evaluation as evaluation


def test_report_is_blocked_without_labels(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluation,
        "_evidence_reliability_summary",
        lambda: {
            "available": True,
            "definition": "Q_cov",
            "drugs": 3,
            "mean": 0.1,
            "median": 0.1,
            "p25": 0.0,
            "p75": 0.2,
            "evidence_gap_drugs": 3,
            "evidence_gap_fraction": 1.0,
        },
    )
    monkeypatch.setattr(evaluation, "LABELS_PATH", tmp_path / "missing_labels.csv")
    monkeypatch.setattr(evaluation, "PREDICTIONS_PATH", tmp_path / "missing_predictions.csv")

    report = evaluation.write_report(tmp_path)

    assert report["status"] == "blocked"
    assert report["results"] is None
    assert report["signals"]["prediction"]["available"] is False
    assert report["signals"]["uncertainty"]["available"] is False
    assert report["signals"]["evidence_reliability"]["available"] is True
    assert "auroc" in report["metric_spec"]["per_trimester"]
    assert "delta_auroc" in report["metric_spec"]["cross_trimester"][0]

    stored = json.loads((tmp_path / "trimester_evaluation_report.json").read_text())
    assert stored["status"] == "blocked"
    markdown = (tmp_path / "trimester_evaluation_report.md").read_text()
    assert "blocked evaluation" in markdown
    assert "not a zero-valued one" in markdown
