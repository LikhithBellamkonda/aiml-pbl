"""Animal spontaneous adverse-event evidence (FDA CVM ADAE, VeDDRA).

This package keeps veterinary spontaneous-report evidence completely separate
from human ADR evidence, animal experimental toxicity and human pregnancy
outcomes. Nothing here derives incidence, causality or a pregnancy label.
"""

from __future__ import annotations

__all__ = [
    "acquisition",
    "features",
    "mapping",
    "pipeline",
    "preprocessing",
]
