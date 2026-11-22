"""Build a curated, evidence-tiered inventory of pregnancy drug-safety sources.

This is deliberately **not** a labelled benchmark. It records, for a fixed set
of drugs used as literature reference controls, exactly what each locally
acquired evidence source contains: human pregnancy-specific records, general
ADR terms, nonclinical developmental/reproductive toxicity studies, FAERS
reporting totals, mapping status and per-source availability.

The literature reference class (established teratogen, trimester-dependent,
generally considered lower risk) is an audit grouping taken from clinical
literature. It is never used as a training label or as a model metric, because
it is not a patient-level outcome observation.

Outputs
-------
reports/pregnancy_evidence_inventory.csv
reports/pregnancy_evidence_inventory.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.ml_risk_model import get_model
from src.services.drug_service import build_drug_profile


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
CSV_PATH = REPORT_DIR / "pregnancy_evidence_inventory.csv"
MD_PATH = REPORT_DIR / "pregnancy_evidence_inventory.md"

# (recorded query name, literature reference class, note)
# The reference class is a coverage-audit grouping from clinical literature,
# not an outcome label and not a training target.
CONTROL_DRUGS: tuple[tuple[str, str, str], ...] = (
    ("isotretinoin", "established teratogen", "retinoid embryopathy"),
    (
        "sodium valproate (valproic acid)",
        "established teratogen",
        "neural tube and neurodevelopmental effects",
    ),
    ("thalidomide", "established teratogen", "limb reduction defects"),
    ("warfarin", "established teratogen", "fetal warfarin syndrome"),
    ("carbamazepine", "established teratogen", "neural tube defects"),
    ("phenytoin sodium", "established teratogen", "fetal hydantoin syndrome"),
    ("lisinopril", "fetotoxic in later pregnancy", "renal and skull effects"),
    ("spironolactone", "possible anti-androgen effect", "feminisation reports"),
    (
        "ibuprofen",
        "trimester-dependent",
        "ductus arteriosus / oligohydramnios in late pregnancy",
    ),
    (
        "sertraline",
        "trimester-dependent",
        "neonatal adaptation syndrome",
    ),
    ("paracetamol", "generally considered lower risk", "first-line analgesic"),
    ("amoxicillin", "generally considered lower risk", "commonly used antibiotic"),
    ("metformin", "generally considered lower risk", "no established teratogenicity"),
    ("heparin", "generally considered lower risk", "does not cross the placenta"),
    ("thyroxine (levothyroxine)", "generally considered lower risk", "replacement therapy"),
    ("insulin glargine", "generally considered lower risk", "does not cross the placenta"),
)


def build_inventory() -> list[dict[str, Any]]:
    """Collect measured evidence coverage for every control drug."""
    rows: list[dict[str, Any]] = []
    engine = get_model()
    for query, reference_class, note in CONTROL_DRUGS:
        profile = build_drug_profile(query)
        quality = (
            engine.evidence_quality(profile.get("drug_id"))
            if profile.get("matched") and not profile.get("ambiguous")
            else {}
        )
        identity = profile.get("identity") or {}
        counts = profile.get("evidence_counts") or {}
        availability = profile.get("availability") or {}
        pregnancy = profile.get("pregnancy") or {}
        molecular = profile.get("molecular") or {}
        rows.append(
            {
                "requested_name": query,
                "literature_reference_class": reference_class,
                "reference_note": note,
                "matched": bool(profile.get("matched")),
                "ambiguous": bool(profile.get("ambiguous")),
                "drug_id": profile.get("drug_id"),
                "resolved_name": identity.get("name"),
                "mapping_status": identity.get("resolution_status"),
                "pubchem_cid": identity.get("pubchem_cid"),
                "inchikey": identity.get("inchikey"),
                "human_pregnancy_records": counts.get("pregnancy_records"),
                "general_adr_terms": counts.get("adr_terms"),
                "nonclinical_toxicity_records": counts.get(
                    "nonclinical_toxicity_records"
                ),
                "faers_reports": counts.get("faers_reports"),
                "trimesters_reported": ", ".join(pregnancy.get("trimesters") or [])
                or "none/unknown",
                "missing_source_families": ", ".join(
                    availability.get("missing_sources") or []
                )
                or "none",
                "structure_available": bool(molecular.get("available")),
                "evidence_quality": quality.get("evidence_quality"),
                "evidence_gap": bool(quality.get("evidence_gap", True)),
            }
        )
    return rows


def _markdown(rows: list[dict[str, Any]]) -> str:
    frame = pd.DataFrame(rows)
    matched = int(frame["matched"].sum()) if not frame.empty else 0
    lines = [
        "# Pregnancy Evidence Inventory (evidence tiers, not outcome labels)",
        "",
        "Generated from the current local data by `python -m src.evidence_inventory`.",
        "",
        "This inventory records **what each acquired source contains** for a fixed",
        "set of literature reference drugs. It is an evidence-coverage audit, not a",
        "labelled benchmark: no patient-level pregnancy outcome labels exist in this",
        "repository, so no AUROC/AUPRC or training target is derived from these rows.",
        "The literature reference class is a coverage grouping only and must not be",
        "read as a validated clinical label or as a model target.",
        "",
        f"- Control drugs listed: {len(rows)}",
        f"- Resolved by the local identifier mapping / evidence index: {matched}",
        "- Flagged with an evidence gap (fewer than two families or low coverage): "
        f"{int(frame['evidence_gap'].sum()) if not frame.empty else 0}",
        "",
        "## Coverage table",
        "",
        "| Drug | Reference class | Mapping | Human pregnancy records | ADR terms | Toxicity records | FAERS reports | Trimesters |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        def cell(value: Any) -> str:
            return "unavailable" if value is None or value == "" else str(value)

        lines.append(
            "| {name} | {klass} | {status} | {preg} | {adr} | {tox} | {faers} | {tri} |".format(
                name=row["requested_name"],
                klass=row["literature_reference_class"],
                status=cell(row["mapping_status"]),
                preg=cell(row["human_pregnancy_records"]),
                adr=cell(row["general_adr_terms"]),
                tox=cell(row["nonclinical_toxicity_records"]),
                faers=cell(row["faers_reports"]),
                tri=cell(row["trimesters_reported"]),
            )
        )

    missing = frame[frame["matched"] == False]  # noqa: E712 - explicit boolean column
    lines += [
        "",
        "## Unmatched controls",
        "",
        (
            "None: every control drug above resolves in the local mapping."
            if missing.empty
            else "The following controls are not present under that spelling: "
            + ", ".join(missing["requested_name"].tolist())
            + ". They are reported as unmatched rather than forced onto a neighbouring "
            "identifier."
        ),
        "",
        "## What this supports",
        "",
        "- Auditing which evidence families cover which drugs, and where the gaps are.",
        "- Selecting a small, manually reviewable subset of drugs for expert review.",
        "- Reproducible coverage comparisons when new sources are added.",
        "",
        "## What this does not support",
        "",
        "- Any pregnancy risk prediction, ranking or model metric.",
        "- Treating a literature reference class as a supervised label.",
        "- Converting spontaneous-report counts into incidence or causal effects.",
        "",
    ]
    return "\n".join(lines)


def write_inventory(report_dir: Path | str = REPORT_DIR) -> dict[str, Any]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = build_inventory()
    frame = pd.DataFrame(rows)
    csv_path = report_dir / CSV_PATH.name
    md_path = report_dir / MD_PATH.name
    frame.to_csv(csv_path, index=False)
    md_path.write_text(_markdown(rows), encoding="utf-8")
    return {
        "controls": len(rows),
        "matched": int(frame["matched"].sum()),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }


def main() -> None:
    summary = write_inventory()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
