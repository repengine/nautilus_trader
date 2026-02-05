from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.training.event_driven.guardrails import validation_bundle


def _write_manifest(tmp_path: Path, *, feature_names: list[str]) -> Path:
    metadata_path = tmp_path / "dataset_metadata.json"
    metadata_payload = {
        "column_info": {
            "group_id_col": "instrument_id",
            "time_idx_col": "time_index",
            "target_col": "y",
            "static_categoricals": ["instrument_id", "exchange"],
            "static_reals": ["tick_size"],
            "time_varying_known_reals": ["time_index", *feature_names],
            "time_varying_unknown_reals": [],
        },
    }
    metadata_path.write_text(json.dumps(metadata_payload), encoding="utf-8")

    manifest_payload = {
        "cohort_run": {"plan_id": "plan-1", "completed_at": "2026-01-01T00:00:00Z"},
        "dataset": {"paths": {"metadata": str(metadata_path)}},
    }
    manifest_path = tmp_path / "plan-1_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    return manifest_path


def test_validate_manifest_coverage_accepts_canonical_names(tmp_path: Path) -> None:
    _write_manifest(tmp_path, feature_names=["return_1", "hour_sin", "hour_cos"])

    validation_bundle.validate_manifest_coverage(tmp_path, limit=1)


def test_validate_manifest_coverage_rejects_legacy_names(tmp_path: Path) -> None:
    _write_manifest(tmp_path, feature_names=["tod_sin", "return_1"])

    with pytest.raises(RuntimeError, match="tod_sin"):
        validation_bundle.validate_manifest_coverage(tmp_path, limit=1)
