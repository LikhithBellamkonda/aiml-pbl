"""Loading and validation of the scoring configuration file.

The configuration lives in ``src/scoring/scoring_config.json`` and stores all
provisional engineering weights used by the evidence-prioritization formula.
Weights are intentionally stored as data (not code) so they can be reviewed,
adjusted, and re-audited without changing code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "src" / "scoring" / "scoring_config.json"


class ScoringConfig:
    """Holds all scoring weights and thresholds.

    Every numeric value is a provisional engineering weight. None of these
    weights are derived from validated clinical data; they are transparent,
    documented assumptions that can be overridden by editing
    ``scoring_config.json``.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        self.version = str(data.get("version", "1.0"))
        self.normalization_factor = float(data.get("normalization_factor", 20.0))
        self.overall_weights = dict(data.get("overall_weights", {"maternal": 0.5, "fetal": 0.5}))
        self.endpoint_relevance = dict(data.get("endpoint_relevance", {}))
        self.source_reliability = dict(data.get("source_reliability", {}))
        self.timing_relevance = dict(data.get("timing_relevance", {}))
        self.volume = dict(data.get("volume", {"reference": 100.0, "cap": 1.0}))
        self.outcome_groups = dict(data.get("outcome_groups", {}))
        self.nonclinical_sources = list(data.get("nonclinical_sources", []))
        self.status_thresholds = dict(data.get("status_thresholds", {}))

    def endpoint_relevance_score(self, outcome: str, trimester: str) -> float:
        """Return the relevance of an outcome to a trimester (0.0-1.0)."""
        by_trimester = self.endpoint_relevance.get(outcome, {})
        return float(by_trimester.get(trimester, 0.0))

    def source_reliability_score(self, source: str) -> float:
        """Return the reliability weight of an evidence source (0.0-1.0)."""
        return float(self.source_reliability.get(source, 0.0))

    def timing_relevance_score(self, timing_class: str) -> float:
        """Return the timing relevance weight (0.0-1.0)."""
        return float(self.timing_relevance.get(timing_class, 0.0))

    def volume_factor(self, record_count: int, observed_total: float | None) -> float:
        """Return a bounded, log-scaled evidence-volume factor (0.0-1.0)."""
        reference = float(self.volume.get("reference", 100.0))
        cap = float(self.volume.get("cap", 1.0))
        if reference <= 0:
            return cap if record_count > 0 else 0.0
        value = max(float(record_count), float(observed_total or 0.0), 0.0)
        import math

        return min(cap, __import__("math").log1p(value) / __import__("math").log1p(reference))

    def is_nonclinical_source(self, source: str) -> bool:
        return source in self.nonclinical_sources


def load_config(path: Path | str | None = None) -> ScoringConfig:
    """Load the scoring configuration from JSON."""
    config_path = Path(path) if path else CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return ScoringConfig(data)
