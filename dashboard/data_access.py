from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _read_optional(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_dashboard_data(root=ROOT):
    root = Path(root)
    data = root / "data"
    return {
        "mapping": _read_optional(data / "drug_mapping.csv"),
        "names": _read_optional(data / "drug_names.csv"),
        "pubchem": _read_optional(data / "real_pubchem.csv"),
        "adr": _read_optional(data / "real_adr_evidence.csv"),
        "toxicity": _read_optional(data / "real_toxicity_evidence.csv"),
        "pregnancy": _read_optional(data / "real_pregnancy_outcomes.csv"),
        "classifications": _read_optional(data / "pregnancy_classifications.csv"),
        "availability": _read_optional(data / "source_availability.csv"),
        "faers_summary": _read_optional(data / "real_faers_summary.csv"),
        "metrics": _read_optional_json(data / "training_metrics.json"),
        "toxicity_metrics": _read_optional_json(
            data / "reproductive_tox_training_metrics.json"
        ),
        "feasibility": _read_optional_json(data / "dataset_feasibility.json"),
        "metadata": _read_optional_json(root / "src" / "models" / "model_metadata.json"),
    }


def _read_optional_json(path):
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def normalize_query(value):
    return " ".join(str(value).casefold().split())


def find_drug(query, mapping):
    if mapping.empty or "original_name" not in mapping.columns:
        return None
    key = normalize_query(query)
    matches_mask = mapping["original_name"].map(normalize_query) == key
    for column in ("drug_id", "pubchem_cid"):
        if column in mapping:
            if column == "pubchem_cid":
                values = mapping[column].map(
                    lambda value: str(int(value)) if pd.notna(value) else ""
                )
            else:
                values = mapping[column].map(normalize_query)
            matches_mask |= values == key
    matches = mapping[matches_mask]
    return matches.iloc[0] if not matches.empty else None


def records_for_drug(frame, drug_id):
    if frame.empty or "drug_id" not in frame.columns:
        return frame.iloc[0:0]
    return frame[frame["drug_id"].astype(str) == str(drug_id)]
