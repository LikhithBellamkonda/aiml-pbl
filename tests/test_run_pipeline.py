import json
from pathlib import Path

from src.run_pipeline import REQUIRED_OUTPUTS, _stage_features, _validate_outputs

ROOT = Path(__file__).resolve().parents[1]


def test_required_pipeline_outputs_are_present():
    outputs = _validate_outputs()
    assert outputs["missing"] == [], outputs["missing"]
    assert outputs["empty"] == []
    assert set(outputs["present"]) == set(REQUIRED_OUTPUTS)


def test_feature_stage_writes_honest_model_ready_summary():
    summary = _stage_features()
    assert summary["drugs"] > 0
    assert summary["adr_features"] == 20
    assert summary["pregnancy_features"] == 4
    # No human outcome labels exist, so no label may be reported as validated.
    assert summary["validated_labels"] == 0
    manifest = ROOT / "data" / "model_ready_features.json"
    assert manifest.is_file()
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    assert stored["drugs"] == summary["drugs"]
    assert stored["validated_labels"] == 0
