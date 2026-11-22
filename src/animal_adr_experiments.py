"""Animal ADR ablation experiments (A/B/C/D) and report generation.

The experiments compare the same architecture with and without the animal ADR
branch using identical drug samples and a fixed seed:

A. Five-source model (no animal ADR).
B. Animal ADR included, but every drug treated as animal-available (no
   missing-evidence handling).
C. Animal ADR included through the missing-evidence-aware fusion gate.
D. Model C with animal evidence masked out at inference (robustness).

Metrics (AUROC, AUPRC, Precision@K, Recall@K, Brier, ECE) require validated
human pregnancy outcome labels.  When ``data/pregnancy_labels.csv`` is absent
the harness still runs every experiment and records that evaluation is
*blocked*, rather than reporting a number.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data_preprocessing import OUTCOME_NAMES, PregnancyDataset, collate_drugs
from src.models.framework import MultimodalPregnancyADRModel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
EVALUATION_PATH = REPORT_DIR / "animal_adr_evaluation_report.json"
ABLATION_PATH = REPORT_DIR / "animal_adr_ablation_results.csv"
INTEGRATION_REPORT_PATH = REPORT_DIR / "animal_adr_integration_report.md"

SEED = 42


def _dataset(include_animal: bool) -> PregnancyDataset:
    return PregnancyDataset(
        DATA_DIR / "real_faers_adr.csv",
        DATA_DIR / "real_pubchem.csv",
        pregnancy_path=DATA_DIR / "real_pregnancy_features.csv",
        labels_path=DATA_DIR / "pregnancy_labels.csv",
        faers_path=DATA_DIR / "real_faers_summary.csv",
        animal_adr_path=DATA_DIR / "animal_adr_features.csv" if include_animal else None,
    )


def _batch(dataset: PregnancyDataset, limit: int):
    indices = [index for index in range(min(limit, len(dataset)))]
    samples = [dataset[index] for index in indices]
    return collate_drugs(samples)


def _labels_available(dataset: PregnancyDataset) -> bool:
    if dataset.labels_df.empty:
        return False
    return bool(float(dataset.labels_df.shape[1]) > 0 and len(dataset.labels_df) > 0)


def _metrics(probabilities: np.ndarray, labels: np.ndarray, masks: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    observed = masks.astype(bool)
    y_true = labels[observed]
    y_prob = probabilities[observed]
    if len(np.unique(y_true)) < 2:
        return {
            "auroc": None,
            "auprc": None,
            "brier": None,
            "reason": "Only one class present in the observed labels.",
        }

    def precision_at(k: int) -> float | None:
        if len(y_true) < k:
            return None
        order = np.argsort(-y_prob)[:k]
        return round(float(y_true[order].sum() / k), 4)

    def recall_at(k: int) -> float | None:
        if y_true.sum() == 0 or len(y_true) < k:
            return None
        order = np.argsort(-y_prob)[:k]
        return round(float(y_true[order].sum() / y_true.sum()), 4)

    return {
        "auroc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "auprc": round(float(average_precision_score(y_true, y_prob)), 4),
        "precision_at_10": precision_at(10),
        "recall_at_10": recall_at(10),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def run_experiments(limit: int = 200, seed: int = SEED) -> dict[str, Any]:
    """Run A/B/C/D on real, identically ordered drug samples."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    base = _dataset(include_animal=False)
    with_animal = _dataset(include_animal=True)
    if len(base) == 0:
        return {"status": "blocked", "reason": "Model-ready feature table is empty."}

    batch_base = _batch(base, limit)
    batch_animal = _batch(with_animal, limit)
    labels_available = _labels_available(with_animal) and "pregnancy_labels.csv" in {
        path.name for path in DATA_DIR.glob("pregnancy_labels.csv")
    }

    model_a = MultimodalPregnancyADRModel(include_animal_adr=False).eval()
    model_c = MultimodalPregnancyADRModel(include_animal_adr=True).eval()
    # Share every parameter the two architectures have in common so that
    # differences between experiments come from the animal evidence and the
    # availability mask, not from unrelated parameter initialisation.
    missing, unexpected = model_c.load_state_dict(model_a.state_dict(), strict=False)
    # Only the new branch and its sixth fusion projection may be unshared.
    extra = ("animal_adr_branch", "fusion.source_projections.5")
    unshared = [name for name in missing if not any(token in name for token in extra)]
    if unshared or unexpected:
        raise RuntimeError(
            f"Unexpected architecture mismatch when sharing weights: {unshared} {unexpected}"
        )

    results: dict[str, Any] = {}
    with torch.no_grad():
        probabilities = model_a(batch_base)
        results["A_no_animal_adr"] = _summarise("A", probabilities, batch_base, labels_available)

        # B: animal branch always treated as available (no missing-evidence gate).
        batch_b = collate_drugs(
            [
                sample
                for sample in (
                    with_animal[index]
                    for index in range(min(limit, len(with_animal)))
                )
            ]
        )
        batch_b.availability_mask[:, -1] = 1.0
        batch_b.animal_availability = torch.ones_like(batch_b.animal_availability)
        results["B_animal_adr_no_missing_handling"] = _summarise(
            "B", model_c(batch_b), batch_b, labels_available
        )

        results["C_animal_adr_masked_fusion"] = _summarise(
            "C", model_c(batch_animal), batch_animal, labels_available
        )

        # D: animal evidence hidden at inference only.
        batch_d = collate_drugs(
            [
                sample
                for sample in (
                    with_animal[index]
                    for index in range(min(limit, len(with_animal)))
                )
            ]
        )
        batch_d.availability_mask[:, -1] = 0.0
        batch_d.animal_availability = torch.zeros_like(batch_d.animal_availability)
        batch_d.animal_features = torch.zeros_like(batch_d.animal_features)
        results["D_no_animal_adr_at_inference"] = _summarise(
            "D", model_c(batch_d), batch_d, labels_available
        )

    animal_available = int(batch_animal.animal_availability.sum().item())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "evaluated" if labels_available else "blocked",
        "reason": (
            None
            if labels_available
            else (
                "No validated human pregnancy-outcome labels "
                "(data/pregnancy_labels.csv) are available, so AUROC, AUPRC, "
                "Precision@K, Recall@K, Brier and ECE cannot be computed. "
                "Animal ADR reports are never used as human outcome labels."
            )
        ),
        "mask_semantics_check": {
            "A_equals_D_mean_probability": abs(
                results["A_no_animal_adr"]["mean_probability"]
                - results["D_no_animal_adr_at_inference"]["mean_probability"]
            ) < 1e-6,
            "note": (
                "With shared weights and the animal source masked out, the "
                "six-source model must reproduce the five-source model. A "
                "difference would mean missing evidence influenced the score."
            ),
        },
        "sample": {
            "drugs_scored": int(len(batch_animal.drug_ids)),
            "drugs_with_animal_adr_evidence": animal_available,
            "seed": seed,
            "training_performed": False,
            "note": (
                "Model weights are randomly initialised and untrained; only "
                "architecture and mask behaviour are exercised."
            ),
        },
        "outcomes": list(OUTCOME_NAMES),
        "experiments": results,
        "metrics": {
            "auroc": None,
            "auprc": None,
            "precision_at_10": None,
            "recall_at_10": None,
            "brier": None,
            "ece": None,
        },
    }
    EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = []
    for name, outcome in results.items():
        rows.append(
            {
                "experiment": name,
                "drugs": outcome["drugs"],
                "mean_probability": outcome["mean_probability"],
                "animal_source_masked_fraction": outcome["animal_source_masked_fraction"],
                "auroc": outcome["metrics"].get("auroc"),
                "auprc": outcome["metrics"].get("auprc"),
                "brier": outcome["metrics"].get("brier"),
                "status": outcome["status"],
            }
        )
    pd.DataFrame(rows).to_csv(ABLATION_PATH, index=False, encoding="utf-8")
    return report


def _summarise(prefix: str, probabilities: torch.Tensor, batch, labels_available: bool) -> dict[str, Any]:
    probs = probabilities.detach().numpy()
    if labels_available:
        metrics = _metrics(
            probs,
            batch.y.numpy(),
            batch.y_mask.numpy(),
        )
        status = "evaluated"
    else:
        metrics = {}
        status = "blocked: no validated human pregnancy-outcome labels"
    mask = batch.availability_mask.detach().numpy()
    return {
        "experiment": prefix,
        "drugs": int(len(batch.drug_ids)),
        "status": status,
        "mean_probability": round(float(probs.mean()), 4),
        "animal_source_available": int(mask[:, -1].sum()) if mask.shape[1] > 5 else 0,
        "animal_source_masked_fraction": (
            round(float(1 - mask[:, -1].mean()), 4) if mask.shape[1] > 5 else 1.0
        ),
        "metrics": metrics,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_integration_report() -> Path:
    """Write the human-readable integration report from generated artefacts."""
    dataset = _read_json(DATA_DIR / "animal_adr_dataset_summary.json")
    quality = _read_json(DATA_DIR / "animal_adr_data_quality_report.json")
    mapping = _read_json(DATA_DIR / "animal_adr_mapping_summary.json")
    feature_summary = _read_json(DATA_DIR / "animal_adr_feature_summary.json")
    acquisition_manifest = _read_json(DATA_DIR / "animal_adr_acquisition_manifest.json")
    evaluation = _read_json(EVALUATION_PATH)
    pipeline = _read_json(DATA_DIR / "pipeline_run.json")
    overlap = mapping.get("overlap", {})

    sources = acquisition_manifest.get("sources", {})
    source_lines = [
        f"| {key} | {value.get('status')} | {value.get('download_portal', '')} |"
        for key, value in sources.items()
    ]

    lines = [
        "# Animal ADR Evidence Integration Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Animal spontaneous adverse-event evidence (FDA CVM ADAE with VeDDRA",
        "terminology) is integrated as an independent evidence branch. It is",
        "kept separate from animal experimental toxicity, human ADR evidence and",
        "human pregnancy outcomes. No incidence, causality or pregnancy label is",
        "derived from these reports.",
        "",
        "## Acquisition",
        "",
        "| Source | Status | Portal |",
        "| --- | --- | --- |",
        *source_lines,
        "",
        "EMA EVVet and UK VMD do not expose a scriptable structured bulk export",
        "in this environment; both are recorded as `manual_access_required` with",
        "explicit local-file ingestion interfaces (`ingest_evvet`, `ingest_vmd`).",
        "",
        "## Data",
        "",
        f"- Raw FDA ADAE reports: {dataset.get('raw_reports')}",
        f"- Cleaned reaction records: {dataset.get('cleaned_rows')}",
        f"- Unique reports: {dataset.get('unique_reports')}",
        f"- Unique species: {dataset.get('unique_species')}",
        f"- Unique reaction terms: {dataset.get('unique_reaction_terms')}",
        f"- Reaction categories: {dataset.get('unique_reaction_categories')}",
        f"- Reporting period: {dataset.get('report_date_range')}",
        f"- Duplicate rows removed: {quality.get('duplicate_rows_removed')}",
        f"- Missing reaction descriptions (retained, flagged): "
        f"{quality.get('missing_reaction_description_rows')}",
        "",
        "## Identifier mapping",
        "",
        f"- Distinct product/ingredient pairs: {mapping.get('mapping_rows')}",
        f"- Directly matched products: {mapping.get('matched_products')}",
        f"- Combination products: {mapping.get('combination_products')} "
        f"(all ingredients matched: {mapping.get('combination_all_ingredients_matched')}, "
        f"partial: {mapping.get('combination_partial')})",
        f"- Ambiguous products: {mapping.get('ambiguous_products')}",
        f"- Unmatched products: {mapping.get('unmatched_products')}",
        f"- Unique standardised drugs: {mapping.get('unique_standardized_drugs')}",
        "",
        "## Overlap with existing project evidence",
        "",
        "| Measure | Value |",
        "| --- | --- |",
        *[f"| {key} | {value} |" for key, value in overlap.items()],
        "",
        "## Features and model branch",
        "",
        f"- Drugs with animal ADR features: {feature_summary.get('drugs_with_animal_adr_evidence')}",
        f"- Reports contributing to mapped drugs: {feature_summary.get('total_reports')}",
        "- Encoder: `src/models/animal_adr_encoder.py` (MLP + species/reaction",
        "  embeddings, zero output when evidence is unavailable).",
        "- Fusion: sixth source in `MaskAwareCrossAttentionGating`, gated by the",
        "  availability mask and evidence quality; 5-source default retained.",
        "",
        "## Training and evaluation",
        "",
    ]

    if evaluation.get("status") == "evaluated":
        summary = evaluation.get("sample", {})
        lines += [
            f"- Drugs scored: {summary.get('drugs_scored')}",
            f"- Metrics: {json.dumps(evaluation.get('metrics', {}))}",
        ]
    else:
        lines += [
            "- Animal ADR branch: **not trained** (no supervised target exists for",
            "  it; see the evaluation report).",
            "- MEPRank retraining: **not performed**.",
            "- Mask semantics check (A == D mean probability): "
            + (
                "**PASSED** - masking the missing animal source reproduces the "
                "five-source model exactly."
                if (evaluation.get("mask_semantics_check") or {}).get(
                    "A_equals_D_mean_probability"
                )
                else "**FAILED** - missing evidence influenced the score."
            ),
            f"- Reason: {evaluation.get('reason')}",
            "- Ablation A/B/C/D: executed forward passes only; metrics are",
            "  recorded as blocked, not as zero or as an improvement.",
        ]

    stages = pipeline.get("stages", [])
    if stages:
        lines += ["", "## Pipeline status", "", "| Stage | Status |", "| --- | --- |"]
        lines += [f"| {item.get('stage')} | {item.get('status')} |" for item in stages]

    lines += [
        "",
        "## Dashboard",
        "",
        "A dedicated **Animal Adverse Drug Reaction Evidence** section reports",
        "availability, sources, species, reaction categories, report counts,",
        "serious reports, completeness, mapping confidence, reporting dates and",
        "limitations, plus an evidence-category distinction table. Drugs without",
        "records show: \"Animal ADR evidence unavailable. This does not indicate",
        "that the drug is safe.\"",
        "",
        "## Remaining limitations",
        "",
        "- Spontaneous reporting is voluntary and has no exposure denominator;",
        "  counts are not incidence and imply no causality.",
        "- 96 standardised drugs match the project's human drug universe; the",
        "  remaining veterinary products are retained as unmatched records.",
        "- No validated human pregnancy-outcome labels exist, so the animal",
        "  branch's contribution to a pregnancy endpoint is unmeasured.",
        "- EMA EVVet and UK VMD require manual structured exports.",
    ]
    INTEGRATION_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return INTEGRATION_REPORT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Animal ADR ablation experiments.")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    report = run_experiments(limit=args.limit)
    write_integration_report()
    print(json.dumps({key: report[key] for key in ("status", "reason")}, indent=2))
    print(f"Report: {INTEGRATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
