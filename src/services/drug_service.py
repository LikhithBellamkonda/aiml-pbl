"""Drug resolution, molecular analysis and evidence assembly.

This module is the single place that turns a user query (a drug name, a
PubChem CID, an InChIKey or a project ``drug_id``) into the material the
dashboard displays: resolved identity, molecular features computed with
RDKit, and the evidence that each acquired source actually records.

Design rules enforced here
--------------------------
* Absent evidence is reported as absent (``None``/empty), never as zero and
  never as a negative label.
* Name-keyed drugs (``NAME:...``) and PubChem-keyed drugs (``CID...``) are
  reported as separate key spaces; a name is never silently joined to a CID.
* Nonclinical animal toxicity records and human pregnancy records are
  returned in separate structures, with species retained.
* No pregnancy-risk score, probability or confidence is computed. The
  research-priority index only counts missing evidence domains.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
LOGGER = logging.getLogger(__name__)

# Pregnancy outcome -> anatomical category used by the dashboard. This is a
# display grouping of the reported outcome term, not a clinical judgement.
OUTCOME_CATEGORIES = {
    "fetal_death": "fetal",
    "stillbirth": "fetal",
    "miscarriage": "fetal",
    "congenital_anomaly": "fetal",
    "fetal_growth_restriction": "fetal",
    "preterm_birth": "neonatal",
    "maternal_adverse_reaction": "maternal",
}

SOURCE_FILES = {
    "mapping": "drug_mapping.csv",
    "pubchem": "real_pubchem.csv",
    "adr": "real_adr_evidence.csv",
    "pregnancy": "real_pregnancy_outcomes.csv",
    "toxicity": "real_toxicity_evidence.csv",
    "faers": "real_faers_summary.csv",
    "availability": "source_availability.csv",
    "master": "standardized_master_evidence.csv",
}

# Module-level frame cache, keyed by (data directory, file name, mtime, size).
_CACHE: dict[tuple[str, str, float, int], pd.DataFrame] = {}


def clear_cache() -> None:
    """Drop cached tables (used by tests and after a pipeline re-run)."""
    _CACHE.clear()


def _read_table(name: str, data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    data_dir = Path(data_dir)
    file_name = SOURCE_FILES[name]
    path = data_dir / file_name
    if not path.is_file():
        LOGGER.warning("Source table unavailable: %s", path)
        return pd.DataFrame()
    stat = path.stat()
    key = (str(data_dir), file_name, stat.st_mtime, stat.st_size)
    if key not in _CACHE:
        _CACHE[key] = pd.read_csv(path, low_memory=False)
    return _CACHE[key].copy()


def normalise(value: Any) -> str:
    """Case- and whitespace-insensitive key used for matching only."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).casefold().split())


def _clean(value: Any) -> Any:
    """Convert pandas missing values to ``None`` for JSON/UI responses."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def _records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    subset = frame.head(limit) if limit else frame
    return [
        {column: _clean(row[column]) for column in subset.columns}
        for row in subset.to_dict("records")
    ]


def _cid(series_value: Any) -> int | None:
    value = _clean(series_value)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _contains(values: pd.Series, key: str) -> pd.Series:
    return values.fillna("").astype(str).map(normalise).eq(normalise(key))


# ---------------------------------------------------------------------------
# Drug resolution
# ---------------------------------------------------------------------------


def _mapping_matches(mapping: pd.DataFrame, query: str) -> pd.DataFrame:
    """All mapping rows matching ``query`` on any supported identifier."""
    key = normalise(query)
    if not key:
        return mapping.iloc[0:0]
    matches = pd.Series(False, index=mapping.index)
    if "original_name" in mapping.columns:
        matches |= _contains(mapping["original_name"], key)
    if "name_key" in mapping.columns:
        matches |= _contains(mapping["name_key"], key)
    if "drug_id" in mapping.columns:
        matches |= _contains(mapping["drug_id"], key)
    if "inchikey" in mapping.columns:
        matches |= _contains(mapping["inchikey"], key)
    if "pubchem_cid" in mapping.columns:
        cid_values = mapping["pubchem_cid"].map(_cid)
        numeric_key: int | None
        try:
            numeric_key = int(float(key))
        except (TypeError, ValueError):
            numeric_key = None
        if numeric_key is not None:
            matches |= cid_values.eq(numeric_key)
        # Also accept a fully padded identifier such as CID000441300.
        if key.startswith("cid"):
            digits = key[3:].lstrip("0")
            if digits.isdigit():
                matches |= cid_values.eq(int(digits))
    if "canonical_smiles" in mapping.columns:
        matches |= _contains(mapping["canonical_smiles"], key)
    return mapping.loc[matches]


def _evidence_index_matches(query: str, data_dir: Path) -> list[dict[str, Any]]:
    """Resolve names that exist in the acquired evidence but not in the mapping.

    The pregnancy-evidence table carries its own name keys (for example
    ``isotretinoin``, which the identifier mapping only lists as
    ``isotretinoin (systemic)``). Matching it here keeps that evidence
    reachable without inventing a PubChem identity or joining it to a CID.
    """
    frame = _read_table("pregnancy", data_dir)
    if frame.empty or "drug_name" not in frame.columns:
        return []
    key = normalise(query)
    matches = _contains(frame["drug_name"], key)
    if "name_key" in frame.columns:
        matches |= _contains(frame["name_key"], key)
    rows = frame.loc[matches]
    if rows.empty:
        return []

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in rows.iterrows():
        name = str(_clean(row.get("drug_name")) or query)
        evidence_key = normalise(_clean(row.get("name_key")) or name)
        raw_id = _clean(row.get("drug_id"))
        drug_id = str(raw_id) if raw_id not in (None, "") else f"NAME:{quote(evidence_key)}"
        if drug_id in seen:
            continue
        seen.add(drug_id)
        candidates.append(
            {
                "drug_id": drug_id,
                "name": name,
                "resolution_status": "name_keyed_evidence_only",
                "resolution_source": "data/real_pregnancy_outcomes.csv",
                "pubchem_cid": None,
                "inchikey": None,
                "smiles": None,
                "name_key": evidence_key,
                "warnings": [
                    "Resolved from the human pregnancy evidence index; this name "
                    "has no PubChem structure in the local mapping, so molecular "
                    "and nonclinical joins are unavailable."
                ],
            }
        )
    return candidates


def _identity_for(
    drug_id: str | None,
    mapping_row: pd.Series | None,
    pubchem: pd.DataFrame,
    data_dir: Path,
) -> dict[str, Any]:
    row = mapping_row
    if row is None and drug_id and "drug_id" in pubchem.columns:
        found = pubchem[pubchem["drug_id"].astype(str) == str(drug_id)]
        row = found.iloc[0] if not found.empty else None
    if row is None:
        return {
            "drug_id": drug_id,
            "name": None,
            "pubchem_cid": None,
            "inchikey": None,
            "smiles": None,
            "molecular_formula": None,
            "molecular_weight": None,
            "resolution_status": None,
            "resolution_source": None,
        }

    pubchem_row = None
    if drug_id and "drug_id" in pubchem.columns:
        found = pubchem[pubchem["drug_id"].astype(str) == str(drug_id)]
        if not found.empty:
            pubchem_row = found.iloc[0]

    def pick(column: str) -> Any:
        if column in row.index:
            value = _clean(row.get(column))
            if value not in (None, ""):
                return value
        if pubchem_row is not None and column in pubchem_row.index:
            return _clean(pubchem_row.get(column))
        return None

    return {
        "drug_id": drug_id,
        "name": pick("original_name") or pick("title") or drug_id,
        "pubchem_cid": _cid(pick("pubchem_cid")),
        "inchikey": pick("inchikey"),
        "smiles": pick("canonical_smiles") or pick("smiles"),
        "molecular_formula": pick("molecular_formula"),
        "molecular_weight": pick("molecular_weight"),
        "resolution_status": pick("resolution_status"),
        "resolution_source": pick("resolution_source"),
    }


def resolve_drug(query: str, data_dir: Path | str = DATA_DIR) -> dict[str, Any]:
    """Resolve a user query against the local identifier mapping.

    Returns a dict with ``matched``, ``ambiguous``, ``candidates`` and
    ``warnings``. Ambiguity is surfaced, never resolved silently.
    """
    data_dir = Path(data_dir)
    mapping = _read_table("mapping", data_dir)
    pubchem = _read_table("pubchem", data_dir)
    result: dict[str, Any] = {
        "query": str(query).strip(),
        "matched": False,
        "ambiguous": False,
        "drug_id": None,
        "drug_name": None,
        "candidates": [],
        "identity": _identity_for(None, None, pubchem, data_dir),
        "warnings": [],
    }
    if not str(query).strip():
        result["warnings"].append("Enter a drug name or a supported identifier.")
        return result
    if mapping.empty:
        result["warnings"].append(
            "The drug mapping is unavailable. Run `python src/process_real_datasets.py`."
        )
        return result

    matches = _mapping_matches(mapping, query)
    if matches.empty:
        fallback = _evidence_index_matches(query, data_dir)
        if fallback:
            result["candidates"] = fallback
            result["ambiguous"] = len({item["drug_id"] for item in fallback}) > 1
            if result["ambiguous"]:
                result["warnings"].append(
                    f"'{query}' matches {len(fallback)} evidence-index entries."
                )
                return result
            chosen = fallback[0]
            result.update(
                {
                    "matched": True,
                    "drug_id": chosen["drug_id"],
                    "drug_name": chosen["name"],
                    "identity": {
                        "drug_id": chosen["drug_id"],
                        "name": chosen["name"],
                        "pubchem_cid": chosen.get("pubchem_cid"),
                        "inchikey": chosen.get("inchikey"),
                        "smiles": chosen.get("smiles"),
                        "molecular_formula": None,
                        "molecular_weight": None,
                        "resolution_status": chosen.get("resolution_status"),
                        "resolution_source": chosen.get("resolution_source"),
                    },
                }
            )
            result["warnings"].extend(chosen.get("warnings") or [])
            return result
        result["warnings"].append(
            f"No local mapping entry matches '{query}'. The drug may not be covered "
            "by the acquired sources."
        )
        return result

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in matches.iterrows():
        drug_id = _clean(row.get("drug_id")) or ""
        name = _clean(row.get("original_name")) or drug_id
        signature = f"{drug_id}|{normalise(name)}"
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(
            {
                "drug_id": str(drug_id),
                "name": str(name),
                "resolution_status": _clean(row.get("resolution_status")),
                "pubchem_cid": _cid(row.get("pubchem_cid")),
                "inchikey": _clean(row.get("inchikey")),
            }
        )

    distinct_ids = {item["drug_id"] for item in candidates if item["drug_id"]}
    result["candidates"] = candidates
    result["ambiguous"] = len(distinct_ids) > 1
    if result["ambiguous"]:
        result["warnings"].append(
            f"'{query}' matches {len(distinct_ids)} distinct drug identifiers. "
            "Select one explicitly before interpreting evidence."
        )
        return result

    chosen = candidates[0]
    drug_id = chosen["drug_id"]
    mapping_row = mapping.loc[
        mapping["drug_id"].astype(str) == str(drug_id)
    ].iloc[0]
    identity = _identity_for(drug_id, mapping_row, pubchem, data_dir)
    result.update(
        {
            "matched": True,
            "drug_id": drug_id,
            "drug_name": identity.get("name") or chosen["name"],
            "identity": identity,
        }
    )
    status = normalise(identity.get("resolution_status"))
    if status and status != "resolved":
        result["warnings"].append(
            f"Identifier resolution status is '{identity.get('resolution_status')}'. "
            "No PubChem structure is available, so molecular analysis is limited."
        )
    if str(drug_id).startswith("NAME:"):
        result["warnings"].append(
            "This drug is keyed by name (NAME:...) rather than a PubChem CID. "
            "Evidence recorded under a CID key is not joined to it automatically."
        )
    return result


# ---------------------------------------------------------------------------
# Molecular analysis
# ---------------------------------------------------------------------------


def _structure_png(molecule) -> tuple[str | None, str | None]:
    """Render a 2D depiction as a base64 PNG without RDKit's drawing backend.

    RDKit's SVG/Cairo drawers need native graphics libraries (libXrender,
    Cairo) that a headless server may not provide. RDKit can still compute 2D
    coordinates, so the depiction is drawn with matplotlib from those
    coordinates. This is the real structure, not a placeholder image.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from rdkit import Chem
        from rdkit.Chem import rdDepictor

        mol = Chem.Mol(molecule)
        rdDepictor.Compute2DCoords(mol)
        conformer = mol.GetConformer()

        figure, axes = plt.subplots(figsize=(4.2, 3.2), dpi=150)
        figure.patch.set_facecolor("white")
        axes.set_facecolor("white")

        for bond in mol.GetBonds():
            begin = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
            end = conformer.GetAtomPosition(bond.GetEndAtomIdx())
            colour = "#4c1d95" if bond.GetIsAromatic() else "#374151"
            axes.add_line(
                Line2D(
                    [begin.x, end.x],
                    [begin.y, end.y],
                    color=colour,
                    linewidth=1.5,
                    solid_capstyle="round",
                )
            )
            if bond.GetBondTypeAsDouble() == 2 and not bond.GetIsAromatic():
                dx, dy = end.x - begin.x, end.y - begin.y
                norm = math.hypot(dx, dy) or 1.0
                offset_x, offset_y = -dy / norm * 0.13, dx / norm * 0.13
                axes.add_line(
                    Line2D(
                        [begin.x + offset_x, end.x + offset_x],
                        [begin.y + offset_y, end.y + offset_y],
                        color=colour,
                        linewidth=1.0,
                    )
                )

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol == "C":
                continue
            position = conformer.GetAtomPosition(atom.GetIdx())
            axes.text(
                position.x,
                position.y,
                symbol,
                ha="center",
                va="center",
                fontsize=9,
                color="#111827",
                bbox=dict(boxstyle="circle,pad=0.12", facecolor="white", edgecolor="none"),
            )

        axes.set_aspect("equal")
        axes.axis("off")
        buffer = io.BytesIO()
        figure.savefig(
            buffer, format="png", bbox_inches="tight", facecolor="white"
        )
        plt.close(figure)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}", None
    except Exception as error:  # rendering must never break descriptor output
        LOGGER.warning("Could not render structure: %s", error)
        return None, str(error)


def _formula_parts(formula: str | None) -> list[dict[str, Any]]:
    """Split a Hill-order molecular formula into element/count pairs.

    Used to render subscripts without inventing any chemistry: the counts come
    straight from RDKit's formula for the recorded SMILES.
    """
    if not formula:
        return []
    parts: list[dict[str, Any]] = []
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula)):
        if not element:
            continue
        parts.append({"element": element, "count": int(count) if count else 1})
    return parts


def analyse_molecule(smiles: str | None) -> dict[str, Any]:
    """Compute RDKit descriptors and a Morgan fingerprint summary.

    Returns ``available=False`` with a reason when SMILES is absent or cannot
    be parsed. All numbers are deterministic descriptors of the supplied
    structure; none of them is a biological prediction.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return {
            "available": False,
            "reason": "No SMILES is recorded for this drug.",
            "descriptors": {},
            "fingerprint": {},
            "svg": None,
            "structure_png": None,
            "drawing_error": None,
            "molecular_formula": None,
            "formula_parts": [],
        }
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as error:  # pragma: no cover - dependency is declared
        return {
            "available": False,
            "reason": f"RDKit is not installed: {error}",
            "descriptors": {},
            "fingerprint": {},
            "svg": None,
            "structure_png": None,
            "drawing_error": None,
            "molecular_formula": None,
            "formula_parts": [],
        }

    RDLogger.DisableLog("rdApp.*")
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None:
        return {
            "available": False,
            "reason": "The recorded SMILES could not be parsed by RDKit.",
            "descriptors": {},
            "fingerprint": {},
            "svg": None,
            "structure_png": None,
            "drawing_error": None,
            "molecular_formula": None,
            "formula_parts": [],
        }

    descriptors = {
        "molecular_weight": round(Descriptors.MolWt(molecule), 3),
        "clogp": round(Crippen.MolLogP(molecule), 3),
        "tpsa": round(rdMolDescriptors.CalcTPSA(molecule), 3),
        "h_bond_donors": int(Lipinski.NumHDonors(molecule)),
        "h_bond_acceptors": int(Lipinski.NumHAcceptors(molecule)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(molecule)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "fraction_csp3": round(rdMolDescriptors.CalcFractionCSP3(molecule), 3),
    }

    bit_info: dict[int, tuple[int, ...]] = {}
    fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        molecule, radius=2, nBits=2048, bitInfo=bit_info
    )
    on_bits = sorted(int(bit) for bit in fingerprint.GetOnBits())
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)

    # Structure rendering is optional: it needs the RDKit drawing backend and
    # its native graphics libraries, which a headless server may not provide.
    # Descriptors and fingerprints must still work without it.
    try:
        from rdkit.Chem.Draw import rdMolDraw2D

        drawer = rdMolDraw2D.MolDraw2DSVG(360, 260)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
    except Exception as error:
        # The matplotlib fallback below still renders the structure.
        LOGGER.debug("RDKit SVG drawing unavailable: %s", error)
        svg = None
        drawing_error = str(error)
    else:
        drawing_error = None

    structure_png, png_error = _structure_png(molecule)
    formula = rdMolDescriptors.CalcMolFormula(molecule)

    return {
        "available": True,
        "reason": None,
        "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True),
        "molecular_formula": formula,
        "formula_parts": _formula_parts(formula),
        "scaffold": scaffold or None,
        "structure_png": structure_png,
        "descriptors": descriptors,
        "fingerprint": {
            "type": "Morgan radius 2, 2048 bits (ECFP4-like)",
            "on_bits": len(on_bits),
            "bit_indices": on_bits,
            "density": round(len(on_bits) / 2048, 4),
        },
        "svg": svg,
        "drawing_error": None if structure_png else (drawing_error or png_error),
    }


# ---------------------------------------------------------------------------
# Evidence assembly
# ---------------------------------------------------------------------------


def _drug_rows(
    frame: pd.DataFrame,
    drug_id: str | None,
    name_key: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.iloc[0:0]
    rows = pd.Series(False, index=frame.index)
    if drug_id and "drug_id" in frame.columns:
        rows |= frame["drug_id"].fillna("").astype(str).eq(str(drug_id))
    if not rows.any() and name_key and "name_key" in frame.columns:
        rows |= _contains(frame["name_key"], name_key)
    return frame.loc[rows].copy()


def pregnancy_outcome_category(outcome: Any, group: Any = "") -> str:
    """Map a reported pregnancy outcome to maternal / fetal / neonatal."""
    key = normalise(outcome).replace(" ", "_")
    if key in OUTCOME_CATEGORIES:
        return OUTCOME_CATEGORIES[key]
    text = f"{normalise(outcome)} {normalise(group)}"
    if "maternal" in text:
        return "maternal"
    if "neonatal" in text or "preterm" in text or "prematur" in text:
        return "neonatal"
    return "fetal"


def adr_evidence(drug_id: str | None, name_key: str | None = None) -> dict[str, Any]:
    frame = _drug_rows(_read_table("adr"), drug_id, name_key)
    if frame.empty:
        return {
            "available": False,
            "term_count": 0,
            "source_datasets": [],
            "terms": [],
            "reason": "No general ADR (SIDER/FAERS term) records for this drug.",
        }
    term_column = "adverse_event_term" if "adverse_event_term" in frame.columns else None
    if term_column is None:
        return {
            "available": False,
            "term_count": 0,
            "source_datasets": [],
            "terms": [],
            "reason": "ADR table has no adverse_event_term column.",
        }
    grouped = (
        frame.assign(term=frame[term_column].fillna("").astype(str))
        .groupby("term")
        .agg(
            meddra_id=("meddra_id", "first") if "meddra_id" in frame else ("term", "size"),
            records=("term", "size"),
            sources=(
                "source_dataset",
                lambda values: ", ".join(sorted({str(v) for v in values.dropna()})),
            )
            if "source_dataset" in frame
            else ("term", "size"),
        )
        .reset_index()
        .sort_values(["records", "term"], ascending=[False, True])
    )
    grouped = grouped[grouped["term"].str.strip().ne("")]
    sources = (
        sorted({str(v) for v in frame["source_dataset"].dropna()})
        if "source_dataset" in frame
        else []
    )
    return {
        "available": not grouped.empty,
        "term_count": int(len(grouped)),
        "record_count": int(len(frame)),
        "source_datasets": sources,
        "terms": _records(grouped, limit=50),
        "reason": None if not grouped.empty else "No named adverse-event terms recorded.",
    }


def reproductive_toxicity_evidence(drug_id: str | None) -> dict[str, Any]:
    frame = _drug_rows(_read_table("toxicity"), drug_id)
    if frame.empty:
        return {
            "available": False,
            "record_count": 0,
            "species": [],
            "study_types": [],
            "records": [],
            "reason": "No nonclinical toxicity study records for this drug.",
            "is_human_evidence": False,
        }
    species = (
        sorted({str(v) for v in frame["species"].dropna()}) if "species" in frame else []
    )
    study_types = (
        sorted({str(v) for v in frame["study_type"].dropna()})
        if "study_type" in frame
        else []
    )
    counts = (
        frame.groupby(["species", "study_type"]).size().reset_index(name="records")
        if {"species", "study_type"}.issubset(frame.columns)
        else pd.DataFrame()
    )
    display_columns = [
        column
        for column in (
            "species",
            "strain",
            "study_type",
            "evidence_type",
            "study_year",
            "admin_route",
            "admin_method",
            "dose_start",
            "dose_start_unit",
            "dose_end",
            "dose_end_unit",
            "study_citation",
            "source_dataset",
        )
        if column in frame.columns
    ]
    return {
        "available": True,
        "record_count": int(len(frame)),
        "species": species,
        "study_types": study_types,
        "counts_by_species": _records(counts),
        "records": _records(frame[display_columns]),
        "sources": (
            sorted({str(v) for v in frame["source_dataset"].dropna()})
            if "source_dataset" in frame
            else []
        ),
        "reason": None,
        "is_human_evidence": False,
    }


def pregnancy_evidence(
    drug_id: str | None,
    name_key: str | None = None,
    data_dir: Path | str = DATA_DIR,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    frame = _drug_rows(_read_table("pregnancy", data_dir), drug_id, name_key)
    if frame.empty:
        return {
            "available": False,
            "record_count": 0,
            "categories": {"maternal": [], "fetal": [], "neonatal": []},
            "outcome_totals": [],
            "sources": [],
            "trimesters": [],
            "records": [],
            "reason": "No human pregnancy-outcome evidence records for this drug.",
            "is_human_evidence": True,
        }

    frame = frame.copy()
    frame["category"] = [
        pregnancy_outcome_category(outcome, group)
        for outcome, group in zip(
            frame.get("outcome", pd.Series("", index=frame.index)),
            frame.get("outcome_group", pd.Series("", index=frame.index)),
        )
    ]
    if "reported_count" in frame.columns:
        frame["reported_count"] = pd.to_numeric(
            frame["reported_count"], errors="coerce"
        )
    categories = {
        category: _records(frame[frame["category"] == category])
        for category in ("maternal", "fetal", "neonatal")
    }
    if "outcome" in frame.columns:
        totals = (
            frame.groupby(["category", "outcome"], dropna=False)
            .agg(
                records=("category", "size"),
                reported_count=(
                    "reported_count",
                    lambda values: float(values.sum()) if values.notna().any() else None,
                ),
            )
            .reset_index()
            .sort_values(["category", "outcome"])
        )
    else:
        totals = pd.DataFrame()

    # Trimester and study provenance come from the standardized evidence
    # master, which records timing explicitly. Missing timing stays unknown.
    master = _read_table("master", data_dir)
    trimesters: list[str] = []
    master_rows = _drug_rows(master, drug_id, name_key) if not master.empty else master
    if not master_rows.empty and "trimester" in master_rows.columns:
        trimesters = sorted(
            {str(value) for value in master_rows["trimester"].dropna()}
        )

    sources = (
        sorted({str(v) for v in frame["source_file"].dropna()})
        if "source_file" in frame
        else []
    )
    evidence_types = (
        sorted({str(v) for v in frame["evidence_type"].dropna()})
        if "evidence_type" in frame
        else []
    )
    return {
        "available": True,
        "record_count": int(len(frame)),
        "reported_count_total": (
            float(frame["reported_count"].sum())
            if "reported_count" in frame and frame["reported_count"].notna().any()
            else None
        ),
        "categories": categories,
        "category_counts": {
            category: int((frame["category"] == category).sum())
            for category in ("maternal", "fetal", "neonatal")
        },
        "outcome_totals": _records(totals),
        "sources": sources,
        "evidence_types": evidence_types,
        "trimesters": trimesters,
        "unknown_trimester": (
            not trimesters or all(normalise(value) == "unknown" for value in trimesters)
        ),
        "records": _records(
            frame[
                [
                    column
                    for column in (
                        "category",
                        "outcome_group",
                        "outcome",
                        "reported_term",
                        "reported_count",
                        "source_file",
                        "evidence_type",
                    )
                    if column in frame.columns
                ]
            ]
        ),
        "reason": None,
        "is_human_evidence": True,
    }


def research_priority_index(profile: dict[str, Any]) -> dict[str, Any]:
    """Count missing evidence domains for research triage, not risk estimation."""
    if not profile.get("matched") or profile.get("ambiguous"):
        return {
            "available": False,
            "score": None,
            "max_score": 3,
            "components": [],
            "reason": "Resolve one drug identity before assessing evidence gaps.",
        }

    pregnancy = profile.get("pregnancy") or {}
    toxicity = profile.get("reproductive_toxicity") or {}
    domains = [
        (
            "Drug-specific human pregnancy outcome records",
            bool(pregnancy.get("available")),
        ),
        (
            "Explicit gestational timing",
            bool(pregnancy.get("available"))
            and not bool(pregnancy.get("unknown_trimester", True)),
        ),
        (
            "Drug-specific developmental/reproductive toxicity records",
            bool(toxicity.get("available")),
        ),
    ]
    components = [
        {
            "domain": domain,
            "covered": covered,
            "gap_points": 0 if covered else 1,
        }
        for domain, covered in domains
    ]
    return {
        "available": True,
        "score": sum(component["gap_points"] for component in components),
        "max_score": len(components),
        "components": components,
        "reason": None,
    }


def structural_analogue_evidence(
    drug_id: str | None,
    smiles: str | None,
    data_dir: Path | str = DATA_DIR,
    limit: int = 5,
) -> dict[str, Any]:
    """Find structurally similar mapped compounds with pregnancy source rows.

    Similarity is a chemical descriptor only. Returned pregnancy records belong
    to the listed compound and must not be interpreted as outcomes for the query.
    """
    unavailable = {"available": False, "candidates": []}
    if not isinstance(smiles, str) or not smiles.strip():
        return {
            **unavailable,
            "reason": "No mapped structure is available for a structural comparison.",
        }

    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import rdMolDescriptors
    except ImportError as error:  # pragma: no cover - dependency is declared
        return {
            **unavailable,
            "reason": f"RDKit is not installed: {error}",
        }

    RDLogger.DisableLog("rdApp.*")
    query_molecule = Chem.MolFromSmiles(smiles.strip())
    if query_molecule is None:
        return {
            **unavailable,
            "reason": "The recorded SMILES could not be parsed for comparison.",
        }

    mapping = _read_table("mapping", data_dir)
    pregnancy = _read_table("pregnancy", data_dir)
    required_mapping = {"drug_id", "canonical_smiles"}
    if mapping.empty or not required_mapping.issubset(mapping.columns):
        return {
            **unavailable,
            "reason": "The local mapping has no structures for comparison.",
        }
    if pregnancy.empty or "drug_id" not in pregnancy.columns:
        return {
            **unavailable,
            "reason": "The local pregnancy evidence table has no linked records.",
        }

    query_fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        query_molecule, radius=2, nBits=2048
    )
    pregnancy = pregnancy.copy()
    pregnancy["_drug_id"] = pregnancy["drug_id"].fillna("").astype(str).str.strip()
    if "name_key" in pregnancy.columns:
        pregnancy["_name_key"] = pregnancy["name_key"].map(normalise)
    else:
        pregnancy["_name_key"] = ""

    candidates: list[dict[str, Any]] = []
    for candidate_id, group in mapping.groupby("drug_id", dropna=True, sort=False):
        candidate_id = str(candidate_id).strip()
        if not candidate_id or candidate_id == str(drug_id or ""):
            continue

        candidate_rows = group[
            group["canonical_smiles"].fillna("").astype(str).str.strip().ne("")
        ]
        if candidate_rows.empty:
            continue

        evidence = pregnancy[pregnancy["_drug_id"] == candidate_id]
        if "name_key" in candidate_rows.columns:
            exact_name_keys = {
                normalise(value)
                for value in candidate_rows["name_key"].dropna()
                if normalise(value)
            }
            if exact_name_keys:
                evidence = pd.concat(
                    [
                        evidence,
                        pregnancy[
                            pregnancy["_drug_id"].eq("")
                            & pregnancy["_name_key"].isin(exact_name_keys)
                        ],
                    ]
                ).drop_duplicates()
        if evidence.empty:
            continue

        candidate_row = candidate_rows.iloc[0]
        candidate_molecule = Chem.MolFromSmiles(
            str(candidate_row["canonical_smiles"]).strip()
        )
        if candidate_molecule is None:
            continue
        candidate_fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            candidate_molecule, radius=2, nBits=2048
        )

        outcome_summaries: list[dict[str, Any]] = []
        if "outcome" in evidence.columns:
            evidence = evidence.copy()
            evidence["_outcome"] = evidence["outcome"].fillna("").astype(str)
            for outcome, outcome_rows in evidence.groupby("_outcome", sort=True):
                if not outcome.strip():
                    continue
                reported_count = None
                if "reported_count" in outcome_rows.columns:
                    values = pd.to_numeric(
                        outcome_rows["reported_count"], errors="coerce"
                    )
                    if values.notna().any():
                        reported_count = float(values.sum())
                outcome_summaries.append(
                    {
                        "outcome": outcome,
                        "source_rows": int(len(outcome_rows)),
                        "reported_count": reported_count,
                    }
                )

        candidates.append(
            {
                "drug_id": candidate_id,
                "drug_name": _clean(candidate_row.get("original_name")) or candidate_id,
                "similarity": float(
                    DataStructs.TanimotoSimilarity(
                        query_fingerprint, candidate_fingerprint
                    )
                ),
                "source_record_count": int(len(evidence)),
                "outcomes": outcome_summaries,
                "sources": (
                    sorted(
                        {
                            str(value)
                            for value in evidence["source_file"].dropna()
                            if str(value).strip()
                        }
                    )
                    if "source_file" in evidence.columns
                    else []
                ),
                "timing": "Not recorded in the local pregnancy source",
            }
        )

    candidates.sort(key=lambda item: (-item["similarity"], item["drug_name"].casefold()))
    candidates = candidates[: max(0, limit)]
    return {
        "available": bool(candidates),
        "reason": (
            None
            if candidates
            else "No mapped structurally related compounds have linked pregnancy source rows."
        ),
        "similarity_method": "Morgan radius 2, 2048-bit fingerprint; Tanimoto similarity",
        "candidates": candidates,
    }


def faers_evidence(drug_id: str | None) -> dict[str, Any]:
    frame = _drug_rows(_read_table("faers"), drug_id)
    if frame.empty:
        return {
            "available": False,
            "report_count": None,
            "outcome_count": None,
            "reason": "No FAERS reporting summary for this drug.",
        }
    row = frame.iloc[0]
    return {
        "available": True,
        "report_count": _clean(row.get("faers_report_count")),
        "outcome_count": _clean(row.get("faers_outcome_count")),
        "source_name": _clean(row.get("faers_source_name")),
        "reason": None,
    }


def availability_for(drug_id: str | None) -> dict[str, Any]:
    frame = _drug_rows(_read_table("availability"), drug_id)
    if frame.empty:
        return {"available": False, "sources": {}, "missing_sources": []}
    row = frame.iloc[0]
    sources: dict[str, Any] = {}
    missing: list[str] = []
    for column in frame.columns:
        if column == "drug_id" or not column.endswith("_status"):
            continue
        source = column[: -len("_status")]
        status = _clean(row.get(column))
        quality_column = f"{source}_metadata_completeness"
        sources[source] = {
            "status": status,
            "metadata_completeness": _clean(row.get(quality_column)),
        }
        if status is None or str(status).lower() != "available":
            missing.append(source)
    return {"available": True, "sources": sources, "missing_sources": sorted(missing)}


def build_drug_profile(
    query: str,
    data_dir: Path | str = DATA_DIR,
) -> dict[str, Any]:
    """Assemble everything the dashboard shows for one query.

    The returned dict is JSON-serialisable and contains no prediction.
    """
    data_dir = Path(data_dir)
    resolution = resolve_drug(query, data_dir)
    profile: dict[str, Any] = {
        **resolution,
        "molecular": analyse_molecule(
            resolution.get("identity", {}).get("smiles")
        ),
        "adr": {"available": False, "terms": []},
        "reproductive_toxicity": {"available": False, "records": []},
        "pregnancy": {"available": False, "records": []},
        "related_pregnancy_evidence": {"available": False, "candidates": []},
        "faers": {"available": False},
        "availability": {"available": False, "sources": {}, "missing_sources": []},
    }
    if not resolution["matched"] or resolution["ambiguous"]:
        return profile

    drug_id = resolution["drug_id"]
    name_key = None
    mapping = _read_table("mapping", data_dir)
    if not mapping.empty and "name_key" in mapping.columns:
        rows = mapping[mapping["drug_id"].astype(str) == str(drug_id)]
        if not rows.empty:
            name_key = _clean(rows.iloc[0].get("name_key"))

    if not name_key and str(drug_id).startswith("NAME:"):
        name_key = normalise(unquote(str(drug_id)[5:]))
    profile["adr"] = adr_evidence(drug_id, name_key)
    profile["reproductive_toxicity"] = reproductive_toxicity_evidence(drug_id)
    profile["pregnancy"] = pregnancy_evidence(drug_id, name_key, data_dir)
    if not profile["pregnancy"]["available"]:
        profile["related_pregnancy_evidence"] = structural_analogue_evidence(
            drug_id,
            (profile.get("identity") or {}).get("smiles"),
            data_dir,
        )
    profile["faers"] = faers_evidence(drug_id)
    profile["availability"] = availability_for(drug_id)
    profile["name_key"] = name_key
    identity = profile.get("identity") or {}
    molecular = profile.get("molecular") or {}
    if not identity.get("molecular_formula") and molecular.get("available"):
        identity["molecular_formula"] = molecular.get("molecular_formula")
    if not identity.get("molecular_weight") and molecular.get("available"):
        identity["molecular_weight"] = (
            molecular.get("descriptors") or {}
        ).get("molecular_weight")
    profile["identity"] = identity
    profile["evidence_counts"] = {
        "adr_terms": profile["adr"].get("term_count", 0),
        "pregnancy_records": profile["pregnancy"].get("record_count", 0),
        "nonclinical_toxicity_records": profile["reproductive_toxicity"].get(
            "record_count", 0
        ),
        "faers_reports": profile["faers"].get("report_count"),
    }
    profile["research_priority"] = research_priority_index(profile)
    if not profile["pregnancy"]["available"]:
        profile["warnings"].append(
            "No human pregnancy-outcome evidence is recorded for this drug."
        )
    if not profile["reproductive_toxicity"]["available"]:
        profile["warnings"].append(
            "No nonclinical reproductive/developmental toxicity records for this drug."
        )
    return profile
