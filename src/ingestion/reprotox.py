from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.ingestion.common import IngestedSource, download_file


LOGGER = logging.getLogger("pregnancy_dataset_ingestion.reprotox")
SOURCE_URL = "https://s3.amazonaws.com/maayan-kg/reprotox/reprotox_serialization.valid.json"


def acquire(destination: Path) -> Path:
    return download_file(SOURCE_URL, destination)


def ingest(path: Path) -> IngestedSource:
    path = Path(path)
    if not path.is_file():
        return IngestedSource("reprotox_kg", False, status="source_unavailable", source_url=SOURCE_URL, source_files=[str(path)])
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
        nodes = graph.get("nodes", {})
        edges = graph.get("edges", [])
        records = []
        for edge_number, edge in enumerate(edges):
            source_node = nodes.get(str(edge.get("source")), {})
            target_node = nodes.get(str(edge.get("target")), {})
            source_props = source_node.get("properties", {})
            target_props = target_node.get("properties", {})
            drug_node, related_node = (
                (source_props, target_props)
                if source_node.get("type") == "Drug"
                else (target_props, source_props)
                if target_node.get("type") == "Drug"
                else (None, None)
            )
            if drug_node is None:
                continue
            relation = str(edge.get("relation", ""))
            related_type = (
                target_node.get("type")
                if source_node.get("type") == "Drug"
                else source_node.get("type")
            )
            related_kind = "birth_defect_term" if related_type == "BirthDefect" else "gene_or_target"
            records.append(
                {
                    "source_dataset": "ReproTox-KG",
                    "source_record_id": str(edge.get("properties", {}).get("id", edge_number)),
                    "drug_name": drug_node.get("label", ""),
                    "drug_id": str(drug_node.get("id", "")).replace("CID:", "CID"),
                    "source_identifier": str(drug_node.get("id", "")),
                    "pubchem_cid": str(drug_node.get("id", "")).removeprefix("CID:"),
                    "inchikey": drug_node.get("InChI_key", ""),
                    "smiles": drug_node.get("SMILES", ""),
                    "outcome_category": "congenital_anomaly" if related_kind == "birth_defect_term" else "",
                    "outcome_term_original": related_node.get("label", "") if related_kind == "birth_defect_term" else "",
                    "related_entity_type": related_type,
                    "related_entity_id": related_node.get("id", ""),
                    "relation": relation,
                    "source_relation": edge.get("properties", {}).get("resource", relation),
                    "association_value": edge.get("properties", {}).get("score"),
                    "evidence_status": "curated_or_computational_association",
                    "evidence_granularity": "knowledge_graph_association_not_patient_outcome",
                    "trimester": "unknown",
                    "gestational_weeks": None,
                    "raw_edge_json": json.dumps(edge, ensure_ascii=False, default=str, sort_keys=True),
                }
            )
        return IngestedSource(
            "reprotox_kg",
            True,
            pd.DataFrame(records),
            status="available",
            source_url=SOURCE_URL,
            source_version=str(graph.get("version", "")),
            source_files=[str(path)],
        )
    except Exception as error:
        LOGGER.exception("Failed parsing ReproTox-KG %s", path)
        return IngestedSource("reprotox_kg", True, status="parse_error", source_url=SOURCE_URL, error=str(error), source_files=[str(path)])
