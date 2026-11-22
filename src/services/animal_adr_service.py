"""Read-only Animal ADR evidence service for the dashboard and API layer.

The service returns *reported veterinary adverse-event evidence*. It never
returns a pregnancy risk value, a safety verdict or an incidence estimate for
a drug without records.
"""

from __future__ import annotations

import re
import unicodedata
import urllib.parse
from typing import Any

from src.animal_adr.features import evidence_for_drug, load_features

EVIDENCE_DISTINCTION = (
    {
        "evidence_type": "human_pregnancy_specific",
        "label": "Human pregnancy-specific evidence",
        "meaning": (
            "Reported human pregnancy outcomes, classifications or timed "
            "associations. This is the only category that can speak to "
            "pregnancy outcomes."
        ),
    },
    {
        "evidence_type": "human_spontaneous_adr",
        "label": "Human spontaneous ADR evidence",
        "meaning": (
            "Human adverse-event terms and FAERS reporting counts. Not "
            "pregnancy-specific and not a pregnancy label."
        ),
    },
    {
        "evidence_type": "animal_spontaneous_adr",
        "label": "Animal spontaneous ADR evidence",
        "meaning": (
            "Veterinary adverse-event reports (FDA CVM ADAE/VeDDRA). "
            "Supplementary signal for prioritisation only; spontaneous counts "
            "are not incidence and imply no causality."
        ),
    },
    {
        "evidence_type": "animal_experimental_toxicity",
        "label": "Animal experimental toxicity evidence",
        "meaning": (
            "Controlled developmental/reproductive toxicity assay endpoints. "
            "Separate from spontaneous veterinary reports and not a human "
            "pregnancy outcome label."
        ),
    },
)


def _name_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _name_keyed_id(drug_name: str) -> str:
    return "NAME:" + urllib.parse.quote(_name_key(drug_name), safe="")


def animal_evidence(
    drug_id: str | None = None,
    drug_name: str | None = None,
) -> dict[str, Any]:
    """Return Animal ADR evidence for a drug, with explicit unavailability."""
    features = load_features()
    candidate_ids = [str(drug_id)] if drug_id else []
    if drug_name:
        candidate_ids.append(_name_keyed_id(drug_name))
    for candidate in candidate_ids:
        if not features.empty and candidate in features.index:
            return evidence_for_drug(candidate, features)
    return evidence_for_drug(None, features) | {
        "drug_id": str(drug_id) if drug_id else "",
        "queried_name": drug_name or "",
    }


def evidence_distinction() -> list[dict[str, str]]:
    """The four evidence categories the dashboard must never merge."""
    return [dict(item) for item in EVIDENCE_DISTINCTION]
