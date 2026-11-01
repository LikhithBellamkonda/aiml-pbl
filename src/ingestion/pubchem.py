from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.common import IngestedSource, is_combination_name, normalize_drug_name
from src.process_real_datasets import resolve_names


LOGGER = logging.getLogger("pregnancy_dataset_ingestion.pubchem")


def _chemical_descriptors(smiles: Any) -> dict[str, float | None]:
    if not isinstance(smiles, str) or not smiles.strip():
        return {}
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return {}
        return {
            "rdkit_molecular_weight": float(Descriptors.MolWt(molecule)),
            "rdkit_logp": float(Descriptors.MolLogP(molecule)),
            "rdkit_tpsa": float(Descriptors.TPSA(molecule)),
            "rdkit_h_bond_donors": float(Descriptors.NumHDonors(molecule)),
            "rdkit_h_bond_acceptors": float(Descriptors.NumHAcceptors(molecule)),
            "rdkit_rotatable_bonds": float(Descriptors.NumRotatableBonds(molecule)),
        }
    except ImportError:
        LOGGER.exception("RDKit is required to calculate chemical descriptors.")
        raise


def ingest_names(names: list[str], *, offline: bool = False, max_queries: int | None = None) -> IngestedSource:
    unique_names = sorted({str(name).strip() for name in names if str(name).strip()}, key=str.casefold)
    resolved = resolve_names(unique_names, offline=offline, max_queries=max_queries)
    rows = []
    for original_name in unique_names:
        from src.process_real_datasets import _name_key

        record = resolved.get(_name_key(original_name), {})
        combination = is_combination_name(original_name)
        status = "combination_unresolved" if combination and record.get("resolution_status") == "combination_product_not_resolved" else record.get("resolution_status", "not_queried")
        row = {
            "source_dataset": "PubChem PUG REST",
            "source_record_id": original_name,
            "drug_name": original_name,
            "normalized_name": normalize_drug_name(original_name),
            "drug_name_key": _name_key(original_name),
            "drug_id": f"CID{int(record['cid']):09d}" if record.get("cid") else "",
            "pubchem_cid": record.get("cid"),
            "inchikey": record.get("inchikey", ""),
            "smiles": record.get("smiles", ""),
            "molecular_formula": record.get("molecular_formula", ""),
            "pubchem_molecular_weight": record.get("molecular_weight"),
            "resolution_status": status,
            "resolution_source": record.get("resolution_source", ""),
            "is_combination": combination,
            "source_url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/",
        }
        row.update(_chemical_descriptors(row["smiles"]))
        rows.append(row)
    frame = pd.DataFrame(rows)
    resolved_count = (
        int(frame["pubchem_cid"].notna().sum())
        if not frame.empty and "pubchem_cid" in frame
        else 0
    )
    source_status = (
        "available_cached"
        if offline and resolved_count
        else "available"
        if resolved_count or not offline
        else "not_queried"
    )
    return IngestedSource(
        "pubchem",
        bool(resolved_count or not offline),
        frame,
        status=source_status if unique_names else "available_no_names",
        source_url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/",
        source_version="PUG REST at retrieval time",
    )


def ingest_existing_mapping(path: Path) -> IngestedSource:
    path = Path(path)
    if not path.is_file():
        return IngestedSource("pubchem", False, status="source_unavailable", source_files=[str(path)])
    frame = pd.read_csv(path)
    normalized = frame.rename(
        columns={
            "original_name": "drug_name",
            "name_key": "drug_name_key",
            "pubchem_cid": "pubchem_cid",
            "canonical_smiles": "smiles",
        }
    )
    if "is_combination" not in normalized:
        normalized["is_combination"] = normalized["drug_name"].map(is_combination_name)
    return IngestedSource(
        "pubchem",
        True,
        normalized,
        status="available_cached",
        source_url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/",
        source_version="cached mapping",
        source_files=[str(path)],
    )
