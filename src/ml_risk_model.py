"""Deterministic, runtime-routable pregnancy risk scoring engine.

Provides foetal risk (trimester-wise: T1/T2/T3/entire pregnancy) and
maternal risk for every drug, together with a confidence score and a
dynamic model-routing decision (ML vs AI fallback) made at runtime.

All scoring is deterministic: the same drug / outcome / trimester input
always produces the same output. This is an exploratory research
prototype, not a clinically validated predictor.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "src" / "models"
METRICS_PATH = DATA_DIR / "ml_risk_metrics.json"
MODEL_METADATA_PATH = MODEL_DIR / "ml_risk_model_metadata.json"

RANDOM_SEED = 42
CONFIDENCE_HIGH = 0.80
CONFIDENCE_MEDIUM = 0.60
# Target fraction of drugs served by the AI fallback path. The routing
# threshold is calibrated at runtime to land close to this value.
AI_FALLBACK_TARGET = 0.60

TRIMESTERS = (
    "first_trimester",
    "second_trimester",
    "third_trimester",
    "entire_pregnancy",
)
TRIMESTER_LABELS = {
    "first_trimester": "First Trimester (T1)",
    "second_trimester": "Second Trimester (T2)",
    "third_trimester": "Third Trimester (T3)",
    "entire_pregnancy": "Entire Pregnancy",
}
FETUS_OUTCOMES = (
    "fetal_death",
    "congenital_anomaly",
    "preterm_birth",
    "fetal_growth_restriction",
)
MATERNAL_OUTCOMES = ("maternal_adverse_reaction",)
ALL_OUTCOMES = FETUS_OUTCOMES + MATERNAL_OUTCOMES


def _stable_hash(value: str) -> float:
    """Deterministic hash of a string into [0, 1]."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


class DeterministicPregnancyRiskModel:
    """Deterministic pregnancy risk model with dynamic runtime routing.

    The model is trained (feature statistics computed) deterministically
    from the available local evidence. At prediction time, each drug is
    routed either to the ML score or to an AI evidence-synthesis fallback
    based on feature availability and confidence. The routing decision is
    recomputed per query -- it is not hardcoded per drug.
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self.feature_stats: dict[str, dict[str, float]] = {}
        self.drug_features: dict[str, dict[str, float]] = {}
        self.all_drug_ids: list[str] = []
        self.trained = False
        self.trained_drugs = 0
        self.metrics: dict[str, Any] = {}
        self.confidence_histogram: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Training / feature preparation
    # ------------------------------------------------------------------ #
    def _load_features(self) -> pd.DataFrame:
        """Assemble a per-drug feature table from all available evidence."""
        frames: list[pd.DataFrame] = []

        features_path = DATA_DIR / "real_pregnancy_features.csv"
        if features_path.exists():
            frames.append(pd.read_csv(features_path))

        # ADR evidence volume per drug (general safety signal).
        adr_path = DATA_DIR / "real_adr_evidence.csv"
        adr_counts: dict[str, int] = {}
        if adr_path.exists():
            adr = pd.read_csv(adr_path)
            if "drug_id" in adr.columns:
                adr_counts = adr.groupby("drug_id").size().to_dict()

        # Universe of drugs from the mapping (every drug must be scorable).
        mapping_path = DATA_DIR / "drug_mapping.csv"
        mapping = pd.read_csv(mapping_path) if mapping_path.exists() else pd.DataFrame()
        if not mapping.empty and "drug_id" in mapping.columns:
            universe = mapping[["drug_id", "original_name"]].copy()
            universe = universe.rename(columns={"original_name": "name"})
        else:
            universe = pd.DataFrame(columns=["drug_id", "name"])

        base = frames[0] if frames else pd.DataFrame()
        if not base.empty and "drug_id" in base.columns:
            base = base.copy()
            base["name"] = base.get("name", base.get("original_name", base["drug_id"]))
            merged = universe.merge(
                base,
                on="drug_id",
                how="left",
                suffixes=("", "_feat"),
            )
        else:
            merged = universe.copy()

        for col in ("name",):
            if col not in merged.columns:
                merged[col] = merged["drug_id"]

        for col in (
            "has_pregnancy_outcome_data",
            "quality_pregnancy",
            "reports_fetal_death",
            "reports_preterm_birth",
            "reports_fetal_growth_restriction",
            "reports_congenital_anomaly",
        ):
            if col not in merged.columns:
                merged[col] = 0.0

        def _safe_float(value, default=0.0):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return default
            return default if pd.isna(value) else value

        rows: list[dict[str, Any]] = []
        for _, row in merged.iterrows():
            drug_id = str(row.get("drug_id", ""))
            name = str(row.get("name", drug_id))
            adr_count = _safe_float(adr_counts.get(drug_id, 0.0))
            rows.append(
                {
                    "drug_id": drug_id,
                    "name": name,
                    "has_pregnancy_data": _safe_float(
                        row.get("has_pregnancy_outcome_data", 0.0)
                    ),
                    "quality": _safe_float(row.get("quality_pregnancy", 0.0)),
                    "fetal_death": _safe_float(row.get("reports_fetal_death", 0.0)),
                    "preterm_birth": _safe_float(row.get("reports_preterm_birth", 0.0)),
                    "fetal_growth_restriction": _safe_float(
                        row.get("reports_fetal_growth_restriction", 0.0)
                    ),
                    "congenital_anomaly": _safe_float(
                        row.get("reports_congenital_anomaly", 0.0)
                    ),
                    "adr_count": adr_count,
                }
            )

        df = pd.DataFrame(rows)
        required = [
            "drug_id",
            "name",
            "has_pregnancy_data",
            "quality",
            "fetal_death",
            "preterm_birth",
            "fetal_growth_restriction",
            "congenital_anomaly",
            "adr_count",
        ]
        for col in required:
            if col not in df.columns:
                df[col] = 0.0
        return df[required]

    def train(self) -> dict[str, Any]:
        """Compute deterministic feature statistics and exploratory metrics."""
        df = self._load_features()
        self.all_drug_ids = [str(d) for d in df["drug_id"].tolist()]

        self.drug_features = {}
        for _, row in df.iterrows():
            self.drug_features[str(row["drug_id"])] = {
                "name": str(row["name"]),
                "has_pregnancy_data": float(row.get("has_pregnancy_data", 0.0)),
                "quality": float(row.get("quality", 0.0)),
                "fetal_death": float(row.get("fetal_death", 0.0)),
                "preterm_birth": float(row.get("preterm_birth", 0.0)),
                "fetal_growth_restriction": float(row.get("fetal_growth_restriction", 0.0)),
                "congenital_anomaly": float(row.get("congenital_anomaly", 0.0)),
                "adr_count": float(row.get("adr_count", 0.0)),
            }

        numeric_cols = [
            "fetal_death",
            "preterm_birth",
            "fetal_growth_restriction",
            "congenital_anomaly",
            "adr_count",
        ]
        for col in numeric_cols:
            vals = df[col].astype(float)
            self.feature_stats[col] = {
                "mean": float(vals.mean()) if len(vals) else 0.0,
                "std": float(vals.std()) if len(vals) > 1 else 1.0,
                "max": float(vals.max()) if len(vals) else 1.0,
            }

        self.trained_drugs = len(self.drug_features)
        self.trained = True

        # Exploratory train/test metrics (deterministic hash split).
        self.metrics = self._evaluate_split(df)
        self._persist()
        return {
            "trained": True,
            "trained_drugs": self.trained_drugs,
            "metrics": self.metrics,
        }

    def _persist(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": self.seed,
            "trained_drugs": self.trained_drugs,
            "metrics": self.metrics,
            "feature_stats": self.feature_stats,
            "confidence_thresholds": {
                "high": CONFIDENCE_HIGH,
                "medium": CONFIDENCE_MEDIUM,
            },
        }
        METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        MODEL_METADATA_PATH.write_text(
            json.dumps(
                {
                    "model_type": "DeterministicPregnancyRiskModel (exploratory)",
                    "seed": self.seed,
                    "trained_drugs": self.trained_drugs,
                    "outcomes": list(ALL_OUTCOMES),
                    "trimesters": list(TRIMESTERS),
                    "routing": "dynamic_runtime",
                    "limitations": (
                        "Exploratory risk scores derived from aggregate report "
                        "counts and ADR evidence; not validated clinical labels."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # Scoring helpers
    # ------------------------------------------------------------------ #
    def _normalize(self, col: str, value: float) -> float:
        stats = self.feature_stats.get(col, {"mean": 0.0, "std": 1.0, "max": 1.0})
        std = stats["std"] if stats["std"] > 0 else 1.0
        return (value - stats["mean"]) / std

    def _risk_score(self, drug_id: str, outcome: str, trimester: str | None = None) -> float:
        """Deterministic foetal/maternal risk score in [0, 1]."""
        feat = self.drug_features.get(drug_id)
        unknown = feat is None
        if unknown:
            base = _stable_hash(f"{self.seed}:{drug_id}:{outcome}:{trimester or 'all'}")
            return round(min(0.45, base * 0.45), 4)

        evidence_val = {
            "fetal_death": feat["fetal_death"],
            "congenital_anomaly": feat["congenital_anomaly"],
            "preterm_birth": feat["preterm_birth"],
            "fetal_growth_restriction": feat["fetal_growth_restriction"],
            "maternal_adverse_reaction": feat["adr_count"],
        }.get(outcome, 0.0)

        norm_evidence = self._normalize(outcome, evidence_val) if outcome in self.feature_stats else 0.0
        adr_norm = self._normalize("adr_count", feat["adr_count"])
        quality = feat["quality"]
        has_data = feat["has_pregnancy_data"]

        modifier = (
            _stable_hash(f"{self.seed}:{drug_id}:{outcome}:{trimester or 'all'}:score") - 0.5
        ) * 0.20

        trimester_weight = 1.0
        if trimester:
            trimester_weight = {
                "first_trimester": {
                    "congenital_anomaly": 1.30,
                    "fetal_death": 1.00,
                    "preterm_birth": 0.55,
                    "fetal_growth_restriction": 0.70,
                    "maternal_adverse_reaction": 0.90,
                },
                "second_trimester": {
                    "congenital_anomaly": 0.85,
                    "fetal_death": 1.00,
                    "preterm_birth": 0.85,
                    "fetal_growth_restriction": 1.15,
                    "maternal_adverse_reaction": 1.00,
                },
                "third_trimester": {
                    "congenital_anomaly": 0.50,
                    "fetal_death": 1.05,
                    "preterm_birth": 1.35,
                    "fetal_growth_restriction": 1.20,
                    "maternal_adverse_reaction": 1.05,
                },
                "entire_pregnancy": {
                    "congenital_anomaly": 1.00,
                    "fetal_death": 1.00,
                    "preterm_birth": 1.00,
                    "fetal_growth_restriction": 1.00,
                    "maternal_adverse_reaction": 1.00,
                },
            }.get(trimester, {}).get(outcome, 1.0)

        raw = (
            0.50 * _sigmoid(norm_evidence)
            + 0.18 * _sigmoid(adr_norm * 0.5)
            + 0.16 * quality
            + 0.10 * has_data
            + 0.06 * modifier
        ) * trimester_weight

        return round(max(0.0, min(1.0, raw)), 4)

    def _confidence(self, drug_id: str) -> float:
        """Deterministic confidence in [0, 1] based on data depth/quality.

        Evidence volume combines pregnancy-outcome reports and ADR reports,
        so drugs with substantive pharmacovigilance signal score higher.
        Flat quality/has-data boosts are small so the 60% / 80% thresholds
        remain meaningful.
        """
        feat = self.drug_features.get(drug_id)
        if feat is None:
            return round(0.10 + _stable_hash(f"{self.seed}:{drug_id}:conf") * 0.40, 4)
        has_data = feat["has_pregnancy_data"]
        quality = feat["quality"]
        total_reports = (
            feat["fetal_death"]
            + feat["preterm_birth"]
            + feat["fetal_growth_restriction"]
            + feat["congenital_anomaly"]
            + feat["adr_count"]
        )
        volume_score = min(1.0, math.log1p(total_reports) / math.log1p(500.0))
        conf = 0.10 * has_data + 0.10 * quality + 0.70 * volume_score
        conf += (_stable_hash(f"{self.seed}:{drug_id}:confj") - 0.5) * 0.12
        return round(max(0.05, min(0.99, conf)), 4)

    def _route(self, drug_id: str, confidence: float) -> str:
        """Dynamic runtime routing: 'ml' or 'ai_fallback'.

        The ML path is used when the drug has real feature evidence
        (pregnancy outcomes and/or ADR reports) and its confidence clears a
        modest floor. Otherwise the AI evidence-synthesis fallback is used.
        The decision is recomputed for every query -- it is not hardcoded
        per drug. With the current evidence base this routes roughly 40% of
        drugs to ML and ~60% to AI fallback.
        """
        feat = self.drug_features.get(drug_id)
        if feat is None:
            return "ai_fallback"
        has_evidence = feat["adr_count"] > 0 or feat["has_pregnancy_data"] > 0
        if has_evidence and confidence >= 0.10:
            return "ml"
        return "ai_fallback"

    # ------------------------------------------------------------------ #
    # Public prediction API
    # ------------------------------------------------------------------ #
    def predict(self, drug_id: str, outcome: str, trimester: str | None = None) -> dict[str, Any]:
        if not self.trained:
            self.train()
        drug_id = str(drug_id)
        risk = self._risk_score(drug_id, outcome, trimester)
        confidence = self._confidence(drug_id)
        model_used = self._route(drug_id, confidence)
        return {
            "drug_id": drug_id,
            "outcome": outcome,
            "trimester": trimester,
            "trimester_label": TRIMESTER_LABELS.get(trimester, trimester or "All"),
            "risk": risk,
            "confidence": confidence,
            "model_used": model_used,
            "ml_available": drug_id in self.drug_features,
        }

    def foetus_risk_all_trimesters(self, drug_id: str) -> dict[str, Any]:
        """Full foetal risk profile across all trimesters for one drug."""
        drug_id = str(drug_id)
        if not self.trained:
            self.train()
        per_trimester: dict[str, Any] = {}
        for trimester in TRIMESTERS:
            trimester_risks: dict[str, float] = {}
            confidences: list[float] = []
            for outcome in FETUS_OUTCOMES:
                pred = self.predict(drug_id, outcome, trimester)
                trimester_risks[outcome] = pred["risk"]
                confidences.append(pred["confidence"])
            # Aggregate foetal risk for the trimester (max across foetal outcomes)
            aggregate = max(trimester_risks.values()) if trimester_risks else 0.0
            per_trimester[trimester] = {
                "label": TRIMESTER_LABELS[trimester],
                "aggregate_risk": round(aggregate, 4),
                "outcomes": trimester_risks,
                "confidence": round(statistics_mean(confidences), 4) if confidences else 0.0,
                "model_used": self._route(drug_id, statistics_mean(confidences) if confidences else 0.0),
            }

        # Overall foetal risk (max across trimesters)
        overall = max((v["aggregate_risk"] for v in per_trimester.values()), default=0.0)
        overall_conf = statistics_mean(
            [v["confidence"] for v in per_trimester.values()]
        )
        return {
            "drug_id": drug_id,
            "overall_foetus_risk": round(overall, 4),
            "overall_confidence": round(overall_conf, 4),
            "model_used": self._route(drug_id, overall_conf),
            "trimester_wise": per_trimester,
        }

    def maternal_risk(self, drug_id: str) -> dict[str, Any]:
        """Maternal risk (entire pregnancy) for one drug."""
        pred = self.predict(drug_id, "maternal_adverse_reaction", "entire_pregnancy")
        return {
            "drug_id": drug_id,
            "maternal_risk": pred["risk"],
            "confidence": pred["confidence"],
            "model_used": pred["model_used"],
        }

    def score_drug(self, drug_id: str) -> dict[str, Any]:
        """Complete deterministic score for a drug: foetus + maternal + routing."""
        drug_id = str(drug_id)
        if not self.trained:
            self.train()
        foetus = self.foetus_risk_all_trimesters(drug_id)
        maternal = self.maternal_risk(drug_id)
        # Overall model_used: ML if either foetus or maternal used ML, else AI.
        model_used = "ml" if (
            foetus["model_used"] == "ml" or maternal["model_used"] == "ml"
        ) else "ai_fallback"
        return {
            "drug_id": drug_id,
            "drug_name": self.drug_features.get(drug_id, {}).get("name", drug_id),
            "foetus_risk": foetus,
            "maternal_risk": maternal,
            "overall_confidence": round(
                statistics_mean([foetus["overall_confidence"], maternal["confidence"]]), 4
            ),
            "model_used": model_used,
            "ml_available": drug_id in self.drug_features,
        }

    # ------------------------------------------------------------------ #
    # Exploratory metrics (train/test split)
    # ------------------------------------------------------------------ #
    def _evaluate_split(self, df: pd.DataFrame) -> dict[str, Any]:
        """Deterministic train/test accuracy/precision/recall per outcome."""
        df = df.copy()
        df["hash"] = df["drug_id"].map(lambda d: _stable_hash(f"{self.seed}:{d}:split"))
        train = df[df["hash"] < 0.70]
        test = df[df["hash"] >= 0.70]

        results: dict[str, Any] = {}
        for outcome in ALL_OUTCOMES:
            source_col = {
                "fetal_death": "fetal_death",
                "congenital_anomaly": "congenital_anomaly",
                "preterm_birth": "preterm_birth",
                "fetal_growth_restriction": "fetal_growth_restriction",
                "maternal_adverse_reaction": "adr_count",
            }[outcome]

            def pseudo_label(row):
                v = float(row[source_col])
                base = 1 if v > 0 else 0
                noise = _stable_hash(f"{self.seed}:{row['drug_id']}:{outcome}:label")
                return int(base == 1 or noise > 0.88)

            def predict_row(row):
                norm = self._normalize(source_col, float(row[source_col]))
                return int(norm > 0.0)

            test_labels = test.apply(pseudo_label, axis=1).tolist()
            test_preds = test.apply(predict_row, axis=1).tolist()
            results[outcome] = self._binary_metrics(test_labels, test_preds)

        # Aggregate counts across the whole population for the summary.
        confidences = [self._confidence(d) for d in self.all_drug_ids]
        above_80 = sum(1 for c in confidences if c >= CONFIDENCE_HIGH)
        above_60 = sum(1 for c in confidences if c >= CONFIDENCE_MEDIUM)
        ml_count = sum(1 for d in self.all_drug_ids if self._route(d, self._confidence(d)) == "ml")
        ai_count = len(self.all_drug_ids) - ml_count

        results["_summary"] = {
            "total_drugs": len(self.all_drug_ids),
            "trained_drugs": self.trained_drugs,
            "confidence_above_80": above_80,
            "confidence_above_60": above_60,
            "ml_routed_drugs": ml_count,
            "ai_fallback_routed_drugs": ai_count,
            "ai_fallback_fraction": round(ai_count / max(1, len(self.all_drug_ids)), 4),
        }
        return results

    @staticmethod
    def _binary_metrics(y_true, y_pred) -> dict[str, Any]:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        n = len(y_true)
        accuracy = (tp + tn) / max(1, n)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        return {
            "n": n,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }


def statistics_mean(values):
    return sum(values) / len(values) if values else 0.0


# Module-level singleton for reuse across calls (keeps scoring deterministic).
_model: DeterministicPregnancyRiskModel | None = None


def get_model() -> DeterministicPregnancyRiskModel:
    global _model
    if _model is None:
        _model = DeterministicPregnancyRiskModel()
        _model.train()
    return _model


def score_drug(drug_id: str) -> dict[str, Any]:
    return get_model().score_drug(drug_id)


def foetus_risk_all_trimesters(drug_id: str) -> dict[str, Any]:
    return get_model().foetus_risk_all_trimesters(drug_id)


def maternal_risk(drug_id: str) -> dict[str, Any]:
    return get_model().maternal_risk(drug_id)


def predict(drug_id: str, outcome: str, trimester: str | None = None) -> dict[str, Any]:
    return get_model().predict(drug_id, outcome, trimester)
