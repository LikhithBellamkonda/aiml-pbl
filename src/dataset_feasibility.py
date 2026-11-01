from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile
from xml.etree import ElementTree

import pandas as pd
from rdkit import Chem

from src.process_real_datasets import _extract_faers_outcomes, _name_key
from src.train_reproductive_tox import ENDPOINTS, load_endpoint_data


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET_DIR = ROOT / "datasets"
OUTCOME_NAMES = (
    "fetal_death",
    "preterm_birth",
    "fetal_growth_restriction",
    "congenital_anomaly",
)

OUTCOME_LABELS = {
    "fetal_death": "Fetal death",
    "preterm_birth": "Preterm birth / prematurity",
    "fetal_growth_restriction": "Fetal growth restriction",
    "congenital_anomaly": "Congenital anomaly",
}


def _percentage_missing(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    return round(float(frame[column].isna().mean() * 100), 2)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _docx_text(path: Path) -> str:
    if not path.exists():
        return ""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    return " ".join(
        node.text or ""
        for node in document.findall(".//w:t", ns)
    )


def _study_supplement_summary(path: Path) -> dict[str, Any]:
    text = _docx_text(path)

    def count(pattern: str) -> int | None:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return int(match.group(1).replace(",", "")) if match else None

    return {
        "status": "aggregate_study_supplement_only" if text else "not_found",
        "file": str(path.relative_to(ROOT)).replace("\\", "/") if text else None,
        "study_title": (
            "Anti-seizure medication use during pregnancy and risk of ASD and ADHD in children"
            if text
            else None
        ),
        "overall_children": count(r"target sample of n\s*=\s*([\d,]+)\s+children"),
        "no_asm_children": count(r"No ASM\s+n\s*=\s*([\d,]+)"),
        "asm_children": count(r"ASM\s+n\s*=\s*([\d,]+)"),
        "epilepsy_subgroup_children": count(
            r"Associations among n\s*=\s*([\d,]+)\s+children"
        ),
        "outcome_names": ["autism spectrum disorder", "ADHD"],
        "medication_groups": [
            "Any ASM",
            "Valproic Acid",
            "Lamotrigine",
            "Carbamazepine",
        ],
        "outcome_event_counts_available": False,
        "patient_level_records_available": False,
        "drug_level_binary_labels_available": False,
        "available_effect_measures": "Hazard ratios and 95% confidence intervals in aggregate tables",
        "reason_not_trainable": (
            "The supplied tables provide aggregate hazard ratios and exposed-subgroup "
            "sizes, not patient-level outcomes, outcome event counts, or per-drug "
            "positive/negative labels."
        ),
    }


def _toxicity_assay_summary(source_path: Path, mapping: pd.DataFrame) -> dict[str, Any]:
    endpoints = []
    variant_consistency = []
    structures: set[str] = set()
    canonical_to_drug_ids: dict[str, list[str]] = {}
    for row in mapping.itertuples():
        smiles_value = getattr(row, "canonical_smiles", None)
        if not isinstance(smiles_value, str) or not smiles_value.strip():
            continue
        molecule = Chem.MolFromSmiles(smiles_value)
        if molecule is None:
            continue
        canonical = Chem.MolToSmiles(molecule, canonical=True)
        canonical_to_drug_ids.setdefault(canonical, []).append(str(row.drug_id))

    for endpoint in ENDPOINTS:
        data = load_endpoint_data(source_path, endpoint)
        structures.update(data.smiles)
        code = ENDPOINTS[endpoint][0].removesuffix("smiles4")
        continuous_sheet = f"{code}smiles3"
        binary_sheet = ENDPOINTS[endpoint][0]
        label_column = ENDPOINTS[endpoint][1]
        continuous = pd.read_excel(
            source_path, sheet_name=continuous_sheet, usecols=["SMILES_wash", label_column]
        )
        binary = pd.read_excel(
            source_path, sheet_name=binary_sheet, usecols=["SMILES_wash", label_column]
        )
        paired = continuous.merge(
            binary, on="SMILES_wash", suffixes=("_smiles3", "_smiles4")
        )
        continuous_labels = pd.to_numeric(
            paired[f"{label_column}_smiles3"], errors="coerce"
        )
        binary_labels = pd.to_numeric(
            paired[f"{label_column}_smiles4"], errors="coerce"
        )
        variant_consistency.append(
            {
                "endpoint": endpoint,
                "paired_structures": int(len(paired)),
                "label_disagreements": int(
                    (continuous_labels != binary_labels).fillna(False).sum()
                ),
                "nonbinary_smiles3_labels": int(
                    (~continuous_labels.isin((0, 1))).sum()
                ),
                "missing_smiles4_label_percent": _percentage_missing(
                    binary, label_column
                ),
                "missing_smiles_percent": _percentage_missing(binary, "SMILES_wash"),
            }
        )
        endpoints.append(
            {
                "endpoint": endpoint,
                "sheet": ENDPOINTS[endpoint][0],
                "records": int(len(data.labels)),
                "positive": int(data.labels.sum()),
                "negative": int(len(data.labels) - data.labels.sum()),
                "positive_percent": round(float(100 * data.labels.mean()), 2),
                "scaffold_groups": int(len(set(data.scaffolds))),
                "excluded_invalid_or_duplicate_rows": data.excluded_rows,
                "label_values_are_binary": bool(set(data.labels).issubset({0, 1})),
                "decision": "limited",
                "decision_reason": (
                    "Adequate labels for an exploratory structure-based split, but the "
                    "workbook's label provenance/definition is not supplied and these "
                    "are nonclinical toxicology endpoints."
                ),
            }
        )

    chemical_matches = {
        smiles: canonical_to_drug_ids[smiles]
        for smiles in structures
        if smiles in canonical_to_drug_ids
    }
    return {
        "status": "present",
        "file": str(source_path.relative_to(ROOT)).replace("\\", "/"),
        "sheets": 10,
        "sheet_variants": (
            "Each TG endpoint has a *_smiles3 and *_smiles4 variant. Modeling uses "
            "the *_smiles4 version because it contains binary target columns; "
            "the source does not document the binary-label derivation."
        ),
        "unique_structures_across_endpoints": int(len(structures)),
        "structures_matching_local_pubchem_smiles": int(len(chemical_matches)),
        "local_drug_ids_matching_assay_structures": int(
            len({drug_id for ids in chemical_matches.values() for drug_id in ids})
        ),
        "outcomes_are_human_pregnancy_outcomes": False,
        "endpoint_granularity": "Chemical-structure level",
        "endpoint_decisions": endpoints,
        "variant_consistency": variant_consistency,
        "source_definition_status": (
            "No dataset citation or codebook/label definition was found in the workbook "
            "or accompanying files."
        ),
    }


def _drug_target_summary(source_path: Path, mapping: pd.DataFrame) -> dict[str, Any]:
    if not source_path.exists():
        return {"status": "not_found"}
    frame = pd.read_csv(
        source_path,
        sep="\t",
        compression="gzip",
        usecols=["DRUG_NAME", "GENE", "TARGET_NAME"],
    )
    source_keys = set(frame["DRUG_NAME"].map(_name_key))
    mapping_keys = set(mapping["name_key"].astype(str))
    matched_keys = source_keys.intersection(mapping_keys)
    return {
        "status": "present",
        "file": str(source_path.relative_to(ROOT)).replace("\\", "/"),
        "records": int(len(frame)),
        "unique_drug_names": int(frame["DRUG_NAME"].map(_name_key).nunique()),
        "unique_targets": int(frame["TARGET_NAME"].nunique()),
        "drug_names_matching_local_mapping": int(len(matched_keys)),
        "drug_name_match_percent": round(
            100 * len(matched_keys) / max(1, len(source_keys)), 2
        ),
        "interpretation": "Pharmacology interaction evidence, not pregnancy outcome labels.",
    }


def build_feasibility_report(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    datasets = root / "datasets"
    data = root / "data"
    mendeley_path = datasets / "FAERS_JAAD_Mendeley Supplement.docx"
    outcome_rows, summary_rows = _extract_faers_outcomes(mendeley_path)
    mapping = pd.read_csv(data / "drug_mapping.csv")
    mapping = mapping.drop_duplicates("name_key").set_index("name_key")
    outcome_csv = pd.read_csv(data / "real_pregnancy_outcomes.csv")
    classification_path = datasets / "medicines-pregnancy-current-database-2026-10-21.csv"
    classifications = pd.read_csv(classification_path)
    assay_path = datasets / "Supplementary_S3_Raw data.xlsx"
    assay_summary = (
        _toxicity_assay_summary(assay_path, mapping.reset_index())
        if assay_path.exists()
        else {"status": "not_found", "endpoint_decisions": []}
    )
    antiseizure_summary = _study_supplement_summary(
        datasets / "Supplemental_tables.docx"
    )
    drug_target_summary = _drug_target_summary(
        datasets / "drug.target.interaction.tsv.gz", mapping.reset_index()
    )
    tox_training_metrics = _read_json(data / "reproductive_tox_training_metrics.json")

    outcome_keys = {
        _name_key(name) for name in outcome_rows.get("drug_name", pd.Series(dtype=str))
    }
    outcome_keys.discard("")
    summary_keys = {
        _name_key(name) for name in summary_rows.get("drug_name", pd.Series(dtype=str))
    }
    summary_keys.discard("")
    mapped_outcome_keys = outcome_keys.intersection(mapping.index.astype(str))
    mapped_summary_keys = summary_keys.intersection(mapping.index.astype(str))
    pubchem_outcome_keys = {
        key for key in mapped_outcome_keys if pd.notna(mapping.loc[key, "pubchem_cid"])
    }
    pubchem_summary_keys = {
        key for key in mapped_summary_keys if pd.notna(mapping.loc[key, "pubchem_cid"])
    }

    outcome_details = []
    for outcome in OUTCOME_NAMES:
        subset = outcome_rows[outcome_rows["outcome"] == outcome]
        outcome_details.append(
            {
                "outcome": outcome,
                "label": OUTCOME_LABELS[outcome],
                "aggregate_term_rows": int(len(subset)),
                "drugs_with_reported_terms": int(subset["drug_name"].nunique()),
                "reported_count_sum": int(subset["reported_count"].sum()),
                "reported_count_missing_percent": _percentage_missing(
                    subset, "reported_count"
                ),
                "verified_positive_labels": 0,
                "verified_negative_labels": 0,
                "decision": "not_trainable",
                "reason": (
                    "Only aggregate spontaneous-report term counts are present; no "
                    "exposed-pregnancy denominator or validated outcome-negative "
                    "observations are supplied."
                ),
            }
        )

    catalog_keys = set(classifications["Name"].map(_name_key))
    catalog_overlap = outcome_keys.intersection(catalog_keys)
    antiepileptic_mask = classifications["Classification 2"].astype(str).str.contains(
        r"antiepilept|anticonvuls", case=False, regex=True
    )
    antiepileptic_rows = classifications.loc[antiepileptic_mask]
    checkpoint_path = root / "src" / "models" / "pregnancy_adr_model.pt"
    metadata_path = root / "src" / "models" / "model_metadata.json"
    metrics_path = data / "training_metrics.json"
    model_metadata = _read_json(metadata_path)
    training_metrics = _read_json(metrics_path)
    compatible_evaluated_model = (
        checkpoint_path.exists()
        and model_metadata.get("outcomes") == list(OUTCOME_LABELS)
        and bool(training_metrics.get("model"))
    )

    dryad_candidates = sorted(
        path
        for path in datasets.rglob("*")
        if path.is_file()
        and any(
            token in path.name.casefold()
            for token in ("dryad", "antiepilep", "epilepsy", "antiepileptic")
        )
    )
    dryad_status = "raw_dataset_present" if dryad_candidates else "not_found"
    dryad_sources = [_relative(path) for path in dryad_candidates]
    dryad_source = {
        "status": dryad_status,
        "files": dryad_sources,
        "unique_drugs": None,
        "outcome_records": 0,
        "usable_binary_label_records": 0,
        "positive_labels": None,
        "negative_labels": None,
        "drug_identifier_match_success": None,
        "missing_value_percentages": None,
        "evidence_granularity": "Not assessable: no Dryad antiepileptic outcome file was found in datasets/.",
    }

    source_outcome_drugs = sorted(outcome_rows["drug_name"].dropna().astype(str).unique())
    source_summary_drugs = sorted(summary_rows["drug_name"].dropna().astype(str).unique())
    outcome_keys_by_name = {
        str(row.drug_name): _name_key(row.drug_name)
        for row in outcome_rows[["drug_name"]].drop_duplicates().itertuples(index=False)
    }
    mapping_rows = []
    for drug_name in source_summary_drugs:
        key = _name_key(drug_name)
        if key in mapping.index:
            row = mapping.loc[key]
            mapped_id = str(row["drug_id"])
            resolution_status = str(row["resolution_status"])
            pubchem_cid = (
                str(int(row["pubchem_cid"])) if pd.notna(row["pubchem_cid"]) else ""
            )
        else:
            mapped_id = ""
            resolution_status = "not_matched"
            pubchem_cid = ""
        mapping_rows.append(
            {
                "drug_name": drug_name,
                "has_outcome_terms": outcome_keys_by_name.get(drug_name) is not None,
                "drug_id": mapped_id,
                "pubchem_cid": pubchem_cid,
                "resolution_status": resolution_status,
            }
        )

    from zipfile import ZipFile
    from xml.etree import ElementTree

    with ZipFile(mendeley_path) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    word_ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = [
        "".join(node.text or "" for node in paragraph.findall(".//w:t", word_ns))
        for paragraph in document.findall(".//w:p", word_ns)
    ]
    methods = " ".join(paragraphs)
    expected_target_match = re.search(
        r"One hundred and twenty-one.*?remaining\s+(\d+)\s+dermatologic medications",
        methods,
        flags=re.IGNORECASE,
    )
    expected_analyzed = int(expected_target_match.group(1)) if expected_target_match else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "mendeley_dermatology": {
                "status": "present",
                "files": [_relative(mendeley_path)],
                "source_description": (
                    "Supplementary tables for a dermatology FAERS spontaneous-report "
                    "analysis. The document describes 121 medications and says 118 "
                    "were analyzed; the locally parsed report-total tables contain "
                    f"{len(summary_keys)} unique drug names."
                ),
                "reported_target_medications": 121,
                "reported_analyzed_medications": expected_analyzed,
                "parsed_report_total_unique_drugs": len(summary_keys),
                "unique_drugs_with_outcome_terms": len(outcome_keys),
                "aggregate_outcome_term_rows": int(len(outcome_rows)),
                "outcome_term_rows_with_reported_count": int(
                    outcome_rows["reported_count"].notna().sum()
                ),
                "outcome_term_rows_with_positive_count": int(
                    (outcome_rows["reported_count"] > 0).sum()
                ),
                "outcome_term_rows_with_zero_count": int(
                    (outcome_rows["reported_count"] == 0).sum()
                ),
                "unclassified_outcome_rows": int(
                    outcome_rows["outcome"].fillna("").eq("").sum()
                ),
                "drug_level_binary_positive_labels": 0,
                "drug_level_binary_negative_labels": 0,
                "usable_binary_label_records": 0,
                "evidence_granularity": (
                    "Drug-level aggregate reported-term counts in the supplied "
                    "supplement; no patient-level records or exposure denominator."
                ),
                "outcomes": outcome_details,
                "drug_identifier_matching": {
                    "unique_source_outcome_drugs": len(outcome_keys),
                    "matched_to_local_drug_mapping": len(mapped_outcome_keys),
                    "name_key_match_percent": round(
                        100 * len(mapped_outcome_keys) / max(1, len(outcome_keys)), 2
                    ),
                    "with_pubchem_cid": len(pubchem_outcome_keys),
                    "pubchem_cid_match_percent_of_source_drugs": round(
                        100 * len(pubchem_outcome_keys) / max(1, len(outcome_keys)), 2
                    ),
                    "outcome_rows_with_drug_id": int(outcome_csv["drug_id"].notna().sum()),
                    "outcome_rows_without_drug_id": int(outcome_csv["drug_id"].isna().sum()),
                    "outcome_rows_missing_drug_id_percent": _percentage_missing(
                        outcome_csv, "drug_id"
                    ),
                },
                "missing_value_percentages": {
                    "reported_count": _percentage_missing(outcome_rows, "reported_count"),
                    "outcome_category": _percentage_missing(outcome_rows, "outcome"),
                    "processed_drug_id": _percentage_missing(outcome_csv, "drug_id"),
                    "processed_outcome_category": _percentage_missing(
                        outcome_csv, "outcome"
                    ),
                },
                "parsed_drug_names": source_outcome_drugs,
                "parsed_report_total_drug_names": source_summary_drugs,
                "drug_mapping": mapping_rows,
            },
            "dryad_antiepileptic": dryad_source,
            "pregnancy_classification_catalog": {
                "status": "present",
                "files": [_relative(classification_path)],
                "rows": int(len(classifications)),
                "unique_drug_names": int(classifications["Name"].map(_name_key).nunique()),
                "antiepileptic_category_rows": int(antiepileptic_mask.sum()),
                "unique_antiepileptic_names": int(
                    antiepileptic_rows["Name"].map(_name_key).nunique()
                ),
                "outcome_records": 0,
                "usable_binary_label_records": 0,
                "missing_value_percentages": {
                    column: _percentage_missing(classifications, column)
                    for column in (
                        "Category",
                        "Safety statement",
                        "Classification 1",
                        "Classification 2",
                        "Classification 3",
                    )
                },
                "interpretation": (
                    "Drug pregnancy classifications and narrative statements, not "
                    "patient-level or drug-level observed pregnancy-outcome labels."
                ),
                "antiepileptic_names": sorted(
                    antiepileptic_rows["Name"].dropna().astype(str).unique()
                ),
            },
            "other_supplement": {
                "status": "present_but_not_dryad_outcome_data",
                "files": ["datasets/supplemental_data.docx"],
                "interpretation": (
                    "The document's tables define hospital discharge diagnoses and "
                    "drugs used to identify teratogenic infections; it is not an "
                    "antiepileptic pregnancy-outcome cohort."
                ),
            },
            "antiepileptic_study_supplement": antiseizure_summary,
            "nonclinical_toxicity_assay": assay_summary,
            "drug_target_interactions": drug_target_summary,
        },
        "compatibility": {
            "mendeley_outcome_drugs_vs_classification_catalog_exact_name_overlap": len(
                catalog_overlap
            ),
            "mendeley_outcome_drugs_vs_classification_catalog_overlap_percent": round(
                100 * len(catalog_overlap) / max(1, len(outcome_keys)), 2
            ),
            "mendeley_vs_dryad_overlap": None,
            "mendeley_vs_dryad_overlap_status": (
                "Not calculable: the Dryad antiepileptic outcome dataset is absent."
            ),
        },
        "target_decisions": outcome_details
        + [
            {
                "outcome": f"nonclinical {item['endpoint']} chemical assay label",
                "label": f"{item['endpoint']} encoded assay class",
                "aggregate_term_rows": item["records"],
                "drugs_with_reported_terms": 0,
                "verified_positive_labels": 0,
                "verified_negative_labels": 0,
                "decision": item["decision"],
                "reason": item["decision_reason"],
            }
            for item in assay_summary.get("endpoint_decisions", [])
        ]
        + [
            {
                "outcome": "ASD among offspring in the anti-seizure medication study",
                "label": "Autism spectrum disorder",
                "aggregate_term_rows": 0,
                "drugs_with_reported_terms": 0,
                "verified_positive_labels": 0,
                "verified_negative_labels": 0,
                "decision": "not_trainable",
                "reason": antiseizure_summary["reason_not_trainable"],
            },
            {
                "outcome": "ADHD among offspring in the anti-seizure medication study",
                "label": "ADHD",
                "aggregate_term_rows": 0,
                "drugs_with_reported_terms": 0,
                "verified_positive_labels": 0,
                "verified_negative_labels": 0,
                "decision": "not_trainable",
                "reason": antiseizure_summary["reason_not_trainable"],
            },
            {
                "outcome": "antiepileptic-study outcomes",
                "label": "Dryad antiepileptic study endpoints",
                "aggregate_term_rows": 0,
                "drugs_with_reported_terms": 0,
                "reported_count_sum": None,
                "verified_positive_labels": 0,
                "verified_negative_labels": 0,
                "decision": "not_trainable",
                "reason": (
                    "No Dryad study outcome data file is present in the local datasets "
                    "directory; the pregnancy-classification catalog is not an outcome "
                    "dataset."
                ),
            }
        ],
        "leakage_assessment": {
            "cross_dataset_overlap": (
                "Cannot assess the requested Mendeley-versus-Dryad overlap because the "
                "Dryad dataset was not found. Mendeley outcome drugs share "
                f"{len(catalog_overlap)} exact normalized names with the pregnancy "
                "classification catalog."
            ),
            "target_feature_leakage": (
                "The current model inputs include pregnancy/FAERS aggregate report "
                "features. Deriving targets from the same Mendeley report counts and "
                "feeding those counts to the model would leak target information."
            ),
            "split_requirement": (
                "If valid external labels are later added, keep every record for a "
                "drug together and use scaffold-grouped separation where structures "
                "are available; do not randomly split rows across datasets."
            ),
        },
        "training_readiness": {
            "status": "pregnancy_model_blocked_nonclinical_experiment_available",
            "reason": (
                "No supplied source provides compatible, validated binary human "
                "outcome labels with defensible negatives."
            ),
            "evaluation_results": None,
            "prediction_uncertainty": None,
            "nonclinical_toxicity_training": (
                "Exploratory scaffold-held-out models were trained for the supplied "
                "binary TG assay columns; see data/reproductive_tox_training_metrics.json."
                if tox_training_metrics
                else "Available via python -m src.train_reproductive_tox."
            ),
            "nonclinical_toxicity_metrics": tox_training_metrics.get("endpoints", {}),
            "model_artifacts": {
                "checkpoint_present": checkpoint_path.exists(),
                "metadata_present": bool(model_metadata),
                "evaluation_metrics_present": bool(training_metrics),
                "compatible_evaluated_model_available": compatible_evaluated_model,
            },
            "required_data": [
                "A documented, outcome-specific human pregnancy cohort or controlled study.",
                "Per-drug (or patient-level with a defensible drug-level aggregation) outcome and non-outcome observations.",
                "Exposure denominators or a suitable comparator cohort so absence of reports is never treated as a negative.",
                "Outcome definitions, follow-up window, pregnancy timing, source references, and a validated drug identifier such as PubChem CID/InChIKey.",
                "Separate labels and masks for each endpoint; unknown outcomes remain missing.",
                "Enough independent drugs/scaffolds and both classes in train, validation, and held-out test sets.",
            ],
        },
    }


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    mendeley = report["sources"]["mendeley_dermatology"]
    dryad = report["sources"]["dryad_antiepileptic"]
    catalog = report["sources"]["pregnancy_classification_catalog"]
    antiseizure = report["sources"]["antiepileptic_study_supplement"]
    assay = report["sources"]["nonclinical_toxicity_assay"]
    interactions = report["sources"]["drug_target_interactions"]
    matching = mendeley["drug_identifier_matching"]
    catalog_overlap_count = report["compatibility"][
        "mendeley_outcome_drugs_vs_classification_catalog_exact_name_overlap"
    ]
    artifacts = report["training_readiness"]["model_artifacts"]

    overview = _md_table(
        ["Local source", "Unique drugs / records", "Outcome data", "Usable binary labels"],
        [
            [
                "Mendeley dermatology FAERS supplement",
                (
                    f"{mendeley['parsed_report_total_unique_drugs']} parsed report-total "
                    f"drug names; {mendeley['unique_drugs_with_outcome_terms']} with "
                    "outcome-term rows"
                ),
                f"{mendeley['aggregate_outcome_term_rows']} aggregate outcome-term rows",
                "0",
            ],
            [
                "Anti-seizure medication study supplement",
                (
                    f"{antiseizure['overall_children']} children; "
                    f"{antiseizure['epilepsy_subgroup_children']} in epilepsy subgroup"
                    if antiseizure["status"] != "not_found"
                    else "Not available"
                ),
                "Aggregate ASD/ADHD hazard ratios; no individual outcome records",
                "0",
            ],
            [
                "Pregnancy classification catalog",
                f"{catalog['unique_drug_names']} names; {catalog['unique_antiepileptic_names']} antiepileptic-category names",
                "Classifications / narrative statements, not observations",
                "0",
            ],
            [
                "Nonclinical TG assay workbook",
                f"{assay.get('unique_structures_across_endpoints', 0)} unique structures",
                f"{len(assay.get('endpoint_decisions', []))} binary TG assay endpoints",
                "Exploratory labels only",
            ],
        ],
    )

    target_rows = [
        [
            item["label"],
            item["aggregate_term_rows"],
            item["drugs_with_reported_terms"],
            item.get("verified_positive_labels", 0),
            item.get("verified_negative_labels", 0),
            item["decision"],
        ]
        for item in report["target_decisions"]
    ]
    target_table = _md_table(
        [
            "Prediction target",
            "Aggregate term rows",
            "Drugs with report terms",
            "Validated positives",
            "Validated negatives",
            "Decision",
        ],
        target_rows,
    )
    match_rows = [
        [
            row["drug_name"],
            "Yes" if row["has_outcome_terms"] else "No",
            row["drug_id"] or "Unmatched",
            row["pubchem_cid"] or "Unavailable",
            row["resolution_status"],
        ]
        for row in mendeley["drug_mapping"]
    ]
    mapping_table = _md_table(
        [
            "Mendeley source drug name",
            "Outcome terms present",
            "Local drug ID",
            "PubChem CID",
            "Resolution",
        ],
        match_rows,
    )
    outcome_rows = [
        [
            item["label"],
            item["aggregate_term_rows"],
            item["drugs_with_reported_terms"],
            item["reported_count_sum"],
            f"{item['reported_count_missing_percent']}%",
            "Not available",
        ]
        for item in mendeley["outcomes"]
    ]
    outcome_table = _md_table(
        [
            "Outcome",
            "Term rows",
            "Drugs",
            "Sum of term counts*",
            "Count missing",
            "Binary negative labels",
        ],
        outcome_rows,
    )
    assay_rows = [
        [
            item["endpoint"],
            item["records"],
            item["positive"],
            item["negative"],
            f"{item['positive_percent']}%",
            item["scaffold_groups"],
            item["decision"],
        ]
        for item in assay.get("endpoint_decisions", [])
    ]
    assay_table = _md_table(
        [
            "Endpoint",
            "Usable structures",
            "Encoded positive",
            "Encoded negative",
            "Positive %",
            "Scaffold groups",
            "Readiness",
        ],
        assay_rows,
    )
    model_rows = []
    for endpoint, details in report["training_readiness"].get(
        "nonclinical_toxicity_metrics", {}
    ).items():
        for model_name, metrics in details.get("models", {}).items():
            model_rows.append(
                [
                    endpoint,
                    model_name,
                    f"{metrics['accuracy']:.3f}",
                    f"{metrics['majority_class_accuracy']:.3f}",
                    f"{metrics['auroc']:.3f}",
                    f"{metrics['auprc']:.3f}",
                    f"{metrics['precision']:.3f}",
                    f"{metrics['recall']:.3f}",
                    f"{metrics['f1']:.3f}",
                    f"{metrics['brier_score']:.3f}",
                    f"{metrics['ece']:.3f}",
                ]
            )
    model_table = (
        _md_table(
            [
                "Endpoint",
                "Model",
                "Accuracy",
                "Majority baseline",
                "AUROC",
                "AUPRC",
                "Precision",
                "Recall",
                "F1",
                "Brier",
                "ECE",
            ],
            model_rows,
        )
        if model_rows
        else "No nonclinical assay training run has been recorded yet."
    )

    return f"""# Pregnancy dataset feasibility and model-training readiness

Generated: {report['generated_at']}

## Executive decision

**No human pregnancy outcome target is trainable.** Preserve the pregnancy-model safeguard; do not train the pregnancy fusion model on aggregate tables. The new workbook does allow a separate, exploratory structure-based nonclinical assay task, but its label source/meaning is not documented. It is not a pregnancy prediction model, and the scaffold-held-out baselines did not achieve high accuracy over the majority-class baseline. No human pregnancy model accuracy, AUROC, AUPRC, precision, recall, F1, Brier/ECE, or calibrated uncertainty results exist.

The inventory contains an anti-seizure medication study supplementary-tables document and an additional raw assay workbook, but no patient-level Dryad cohort file. The anti-seizure document has aggregate study statistics only. The file `datasets/supplemental_data.docx` is an infection-diagnosis/drug definition supplement, while `datasets/medicines-pregnancy-current-database-2026-10-21.csv` contains pregnancy classifications and narrative statements.

## Dataset inventory and granularity

{overview}

### Mendeley dermatology FAERS supplement

Source: `{mendeley['files'][0]}`. The document describes **{mendeley['reported_target_medications']}** dermatologic medications and states that **{mendeley['reported_analyzed_medications']}** were analyzed. The local parser found **{mendeley['parsed_report_total_unique_drugs']}** unique names in report-total table headings and **{mendeley['unique_drugs_with_outcome_terms']}** names with outcome-term rows. This is a parsing/coverage discrepancy against the source narrative, so the extracted table inventory should not be treated as a complete medication list.

The supplement provides aggregate spontaneous-report counts grouped by drug and MedDRA outcome terms. It does not provide the underlying patient-level rows, a pregnancy-exposure denominator, or validated non-event controls. The paper's methods refer to case-level FAERS review, but those records are not included in the local supplement.

It contains **{mendeley['aggregate_outcome_term_rows']}** outcome-term rows. Of these, **{mendeley['outcome_term_rows_with_reported_count']}** have a count, **{mendeley['outcome_term_rows_with_positive_count']}** have a positive reported count, and **{mendeley['outcome_term_rows_with_zero_count']}** have zero. These are counts of reported terms, not binary drug-level outcome labels or independent pregnancy counts; one pregnancy may contribute multiple outcome terms.

{outcome_table}

\\* The term-count sums are descriptive source counts, not incidence, patient totals, or positive-label counts. The source does not identify valid drug-level negatives.

Identifier mapping for the **{matching['unique_source_outcome_drugs']}** source outcome-drug names: **{matching['matched_to_local_drug_mapping']} ({matching['name_key_match_percent']}%)** match a local normalized name key; **{matching['with_pubchem_cid']} ({matching['pubchem_cid_match_percent_of_source_drugs']}%)** have a PubChem CID. In the processed outcome-term CSV, **{matching['outcome_rows_without_drug_id']} / {mendeley['aggregate_outcome_term_rows']} ({matching['outcome_rows_missing_drug_id_percent']}%)** rows have no local `drug_id`.

Missing-value percentages in the processed outcome-term table: `reported_count` {mendeley['missing_value_percentages']['reported_count']}%; outcome category {mendeley['missing_value_percentages']['processed_outcome_category']}%; drug ID {mendeley['missing_value_percentages']['processed_drug_id']}%. Unclassified outcome terms are retained separately rather than forced into an endpoint.

#### Source drug names and matched identifiers

{mapping_table}

### Anti-seizure medication study supplement

No patient-level Dryad cohort file was found under `datasets/`. A related-looking supplementary document is present: `{antiseizure['file']}` (“{antiseizure['study_title']}”). It summarizes **{antiseizure['overall_children']}** children overall (**{antiseizure['no_asm_children']}** without ASM and **{antiseizure['asm_children']}** with ASM) and reports associations in a subgroup of **{antiseizure['epilepsy_subgroup_children']}** children. Its endpoints are autism spectrum disorder and ADHD; the medication groups include any ASM, valproic acid, lamotrigine, and carbamazepine. The tables report hazard ratios and confidence intervals, not event counts, participant-level labels, or drug-level positive/negative observations. It is therefore **not trainable** for either endpoint from the supplied data. The local medication classification catalog has {catalog['unique_antiepileptic_names']} antiepileptic-category names, but these classifications are not outcomes.

### New nonclinical TG assay workbook

Source: `{assay['file']}`. It contains `{assay['sheets']}` sheets: paired `*_smiles3` and `*_smiles4` versions of five TG endpoints. The training experiment reads the binary target from `*_smiles4` and uses only molecular structure (RDKit Morgan fingerprints) as input. The binary conversion/endpoint semantics are not documented in the workbook or accompanying files, so the models below are exploratory and the dataset is **limited**, not validated for scientific or clinical use. There are {assay['unique_structures_across_endpoints']} unique canonical structures across endpoints; {assay['structures_matching_local_pubchem_smiles']} also exactly match a local PubChem SMILES, corresponding to {assay['local_drug_ids_matching_assay_structures']} mapped local drug IDs. This structure overlap does not convert assay labels into human outcomes.

{assay_table}

### Exploratory scaffold-held-out baseline results

The command `python -m src.train_reproductive_tox` trains Logistic Regression, Random Forest, and XGBoost separately for each TG endpoint, using 5-fold scaffold-grouped out-of-fold predictions. The values below are estimates from that cross-validation, not an external holdout. Accuracy is compared to the majority-class accuracy; high accuracy is not established by these results.

{model_table}

The script saves full-data experimental model files under `models/reproductive_tox/`; those artifacts are strictly for the nonclinical endpoints above and are not loaded by the pregnancy dashboard. Brier and ECE are descriptive only; no calibration validation or confidence interval is provided.

The separate `{interactions['file']}` source contains {interactions['records']} drug-target records and {interactions['unique_drug_names']} drug names ({interactions['drug_names_matching_local_mapping']} exact normalized-name matches to the local drug map). These interactions are pharmacology features, not pregnancy outcome labels.

### Other local pregnancy classification data

`{catalog['files'][0]}` contains {catalog['rows']} rows and {catalog['unique_drug_names']} unique normalized names. It includes {catalog['antiepileptic_category_rows']} rows / {catalog['unique_antiepileptic_names']} names classified in antiepileptic categories. It has no patient-level or drug-level observed outcome records and cannot be converted into outcome labels. Missingness is descriptive metadata missingness: safety statement {catalog['missing_value_percentages']['Safety statement']}%, category {catalog['missing_value_percentages']['Category']}%, classification 2 {catalog['missing_value_percentages']['Classification 2']}%.

## Compatibility, overlap, and leakage

- Mendeley outcome-drug names overlap the classification catalog by exact normalized name for **{catalog_overlap_count} / {mendeley['unique_drugs_with_outcome_terms']}** names. This is a name-catalog overlap, not shared patient records and not outcome validation.
- Mendeley-versus-patient-level Dryad overlap cannot be calculated because the cohort records are absent. The anti-seizure supplement is not a per-drug outcome cohort.
- The model's current evidence inputs include the aggregate pregnancy/FAERS report-count features. If the same Mendeley counts were used to create target labels, those features would directly leak target information. Exclude source-derived outcome features or use labels from an independent cohort.
- If independent valid labels are obtained, split by drug and preferably chemical scaffold. Never randomly split rows from the same drug across train/test sets.

## Outcome-by-outcome training decision

{target_table}

For each Mendeley outcome, the listed term rows and drug counts are reported evidence only. They are **not** positive training labels, and missing reports cannot define a negative. The ASD and ADHD supplement contains only aggregate hazard-ratio results. Nonclinical TG labels support exploratory modeling only, with unverified label semantics and no clinical interpretation. No unrelated pregnancy outcomes have been combined.

## Required data before training

1. Supply the actual Dryad study data file and its data dictionary/citation, including the cohort and outcome columns.
2. Obtain an outcome-specific human cohort or controlled study with both outcome and non-outcome observations, suitable follow-up, and exposure/comparator denominators.
3. Keep endpoints separate and define ascertainment, pregnancy timing, and follow-up for fetal death, preterm birth, fetal growth restriction, and congenital anomaly.
4. Provide per-record or defensible per-drug aggregation, source references, outcome definitions, and stable identifiers (CID/InChIKey plus the original drug name).
5. Distinguish true observed non-events from unreported/missing outcomes. Encode unknowns as masked, never as zero.
6. Ensure enough independent drug/scaffold groups and positive/negative examples for train, validation, held-out test, and external validation. Check endpoint-specific class balance before selecting metrics.
7. Keep features independent of the target source to prevent target leakage.

## Model status and dashboard interpretation

- **Software pipeline validation:** tests validate implementation behavior only; they do not measure clinical or predictive performance.
- **Dataset validation:** this report records file presence, schemas/granularity, missingness, and identifier coverage.
- **Model training:** blocked because no compatible validated binary labels are present.
- **Model evaluation:** no evaluation metrics exist; do not report accuracy, precision, recall, or confidence as model results.
- **Nonclinical assay training:** exploratory scaffold-held-out models have been trained; their metrics are in the table above and are not pregnancy outcome performance.
- **Stored model files:** checkpoint present={artifacts['checkpoint_present']}; metadata present={artifacts['metadata_present']}; evaluation metrics present={artifacts['evaluation_metrics_present']}. A checkpoint by itself is not treated as a trained/evaluated model.
- **Prediction uncertainty:** unavailable for human pregnancy outcomes. Evidence completeness and MC-dropout variance are not calibrated confidence intervals.
"""


def write_report(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    report = build_feasibility_report(root)
    report_path = root / "data" / "dataset_feasibility.json"
    markdown_path = root / "reports" / "pregnancy_dataset_feasibility.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = write_report()
    print(f"Feasibility report: {ROOT / 'reports' / 'pregnancy_dataset_feasibility.md'}")
    print(
        "Training readiness: "
        f"{result['training_readiness']['status']} "
        f"({len(result['target_decisions'])} targets assessed)"
    )
