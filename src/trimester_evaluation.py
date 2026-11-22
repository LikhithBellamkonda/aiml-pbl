"""Trimester-focused evaluation report.

This module makes trimester reasoning the explicit evaluation centre:

* per-trimester AUROC / AUPRC,
* ΔAUROC between trimesters (does the model learn trimester differences or
  collapse to a single-stage predictor?),
* calibration error (ECE) and Brier score per trimester,
* Precision@10 per trimester.

It also reports the three signals the project keeps separate: prediction,
predictive uncertainty, and evidence reliability (Q_cov). Evidence reliability
is computed now; prediction and uncertainty are reported as unavailable because
no validated human pregnancy-outcome labels or trained checkpoint exist.

Outputs
-------
reports/trimester_evaluation_report.json
reports/trimester_evaluation_report.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation import (
    compute_batch_evidence_quality,
    evaluate_trimester_metrics,
    load_source_records_from_data_dir,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
LABELS_PATH = DATA_DIR / "pregnancy_labels.csv"
PREDICTIONS_PATH = DATA_DIR / "pregnancy_predictions.csv"
JSON_PATH = REPORT_DIR / "trimester_evaluation_report.json"
MD_PATH = REPORT_DIR / "trimester_evaluation_report.md"

BLOCKED_REASON = (
    "No validated human pregnancy-outcome labels (data/pregnancy_labels.csv) "
    "and no trained human pregnancy model exist, so trimester-wise AUROC, "
    "AUPRC, ΔAUROC, ECE, Brier and Precision@10 cannot be computed. They are "
    "reported as unavailable, not as zero."
)

METRIC_SPEC = {
    "per_trimester": [
        "auroc",
        "auprc",
        "ece",
        "brier",
        "precision_at_10",
    ],
    "cross_trimester": ["delta_auroc (max - min across trimesters)"],
    "interpretation": (
        "ΔAUROC > 0 indicates trimester-specific learning; ΔAUROC ≈ 0 means the "
        "model collapses to a single-stage predictor."
    ),
    "trimesters": [
        "first_trimester",
        "second_trimester",
        "third_trimester",
        "entire_pregnancy",
    ],
}


def _evidence_reliability_summary() -> dict[str, Any]:
    """Compute the Q_cov distribution over the local drug universe."""
    mapping_path = DATA_DIR / "drug_mapping.csv"
    if not mapping_path.is_file():
        return {"available": False, "reason": "data/drug_mapping.csv is missing."}
    drug_ids = (
        pd.read_csv(mapping_path, usecols=["drug_id"])["drug_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if not drug_ids:
        return {"available": False, "reason": "No drug identifiers in the mapping."}
    batch = compute_batch_evidence_quality(drug_ids, load_source_records_from_data_dir())
    values = [item["evidence_quality"] for item in batch.values()]
    gaps = sum(1 for item in batch.values() if item["evidence_gap"])
    series = pd.Series(values, dtype=float)
    return {
        "available": True,
        "definition": (
            "Q_cov = mean over source families of (availability mask × "
            "source quality), where source quality = 0.5 × populated content "
            "fields + 0.5 × log-scaled evidence volume for that drug"
        ),
        "not_a_risk_score": True,
        "drugs": len(values),
        "mean": round(float(series.mean()), 4),
        "median": round(float(series.median()), 4),
        "p25": round(float(series.quantile(0.25)), 4),
        "p75": round(float(series.quantile(0.75)), 4),
        "evidence_gap_drugs": int(gaps),
        "evidence_gap_fraction": round(gaps / max(len(values), 1), 4),
    }


def build_report() -> dict[str, Any]:
    """Assemble the trimester evaluation report from the current artefacts."""
    report: dict[str, Any] = {
        "status": "blocked",
        "reason": BLOCKED_REASON,
        "labels_present": LABELS_PATH.is_file(),
        "predictions_present": PREDICTIONS_PATH.is_file(),
        "metric_spec": METRIC_SPEC,
        "signals": {
            "prediction": {
                "available": False,
                "reason": "No trained human pregnancy-outcome model.",
                "value": None,
            },
            "uncertainty": {
                "available": False,
                "reason": "No trained model, so predictive uncertainty is undefined.",
                "value": None,
            },
            "evidence_reliability": _evidence_reliability_summary(),
        },
        "results": None,
    }

    if not LABELS_PATH.is_file():
        return report
    report["status"] = "labels_present_model_missing"
    report["reason"] = (
        "Labels are present but no trained model / prediction table exists. "
        "Run `python -m src.train` and export data/pregnancy_predictions.csv "
        "with columns drug_id, outcome, trimester, y_true, y_score."
    )
    if not PREDICTIONS_PATH.is_file():
        return report

    frame = pd.read_csv(PREDICTIONS_PATH)
    required = {"outcome", "trimester", "y_true", "y_score"}
    if not required.issubset(frame.columns):
        report["reason"] = (
            "data/pregnancy_predictions.csv is missing required columns: "
            + ", ".join(sorted(required - set(frame.columns)))
        )
        return report

    results: dict[str, Any] = {}
    for outcome, group in frame.groupby("outcome"):
        if len(group) < 2 or group["y_true"].nunique() < 2:
            results[str(outcome)] = {
                "status": "insufficient_data",
                "n": int(len(group)),
            }
            continue
        results[str(outcome)] = evaluate_trimester_metrics(
            group["y_true"].to_numpy(),
            group["y_score"].to_numpy(),
            group["trimester"].astype(str).to_numpy(),
        )
    report["status"] = "evaluated"
    report["reason"] = None
    report["results"] = results
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Trimester Evaluation Report",
        "",
        f"Status: **{report['status']}**",
        "",
    ]
    if report.get("reason"):
        lines += [report["reason"], ""]
    lines += [
        "## Metric specification",
        "",
        "- Per-trimester: " + ", ".join(METRIC_SPEC["per_trimester"]),
        "- Cross-trimester: " + ", ".join(METRIC_SPEC["cross_trimester"]),
        f"- {METRIC_SPEC['interpretation']}",
        "",
        "## Three separate signals",
        "",
    ]
    signals = report["signals"]
    for name, details in signals.items():
        if details.get("available"):
            lines.append(f"- **{name}**: available")
            if name == "evidence_reliability":
                lines.append(
                    "  - "
                    + details["definition"]
                    + f"; drugs={details['drugs']}, mean={details['mean']}, "
                    f"median={details['median']}, evidence-gap fraction="
                    f"{details['evidence_gap_fraction']}"
                )
        else:
            lines.append(f"- **{name}**: unavailable — {details.get('reason')}")
    lines += ["", "## Results", ""]
    if report["status"] != "evaluated":
        lines.append(
            "No trimester metrics are reported. This is a blocked evaluation, "
            "not a zero-valued one."
        )
    else:
        lines.append("```json")
        lines.append(json.dumps(report["results"], indent=2))
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_report(report_dir: Path | str = REPORT_DIR) -> dict[str, Any]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    (report_dir / JSON_PATH.name).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (report_dir / MD_PATH.name).write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    report = write_report()
    print(json.dumps({"status": report["status"], "reason": report["reason"]}, indent=2))


if __name__ == "__main__":
    main()
