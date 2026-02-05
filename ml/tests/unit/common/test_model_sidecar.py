"""
Unit tests for model sidecar metadata helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

from ml.common.model_sidecar import extract_inference_metadata
from ml.common.model_sidecar import load_sidecar_metadata
from ml.common.model_sidecar import resolve_model_sidecar_metadata


def test_load_sidecar_metadata_returns_none_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.meta.json"
    assert load_sidecar_metadata(missing) is None


def test_extract_inference_metadata_maps_calibrator_fields() -> None:
    sidecar = {
        "output_schema": {"kind": "binary_proba", "shape": [None, 1]},
        "calibrator_kind": "platt",
        "calibrator_params": {"coef": 1.1, "intercept": -0.2},
    }
    output_schema, calibration = extract_inference_metadata(sidecar)

    assert output_schema == {"kind": "binary_proba", "shape": [None, 1]}
    assert calibration == {"kind": "platt", "params": {"coef": 1.1, "intercept": -0.2}}


def test_resolve_model_sidecar_metadata_prefers_existing_candidates(tmp_path: Path) -> None:
    model_path = tmp_path / "student.onnx"
    model_path.write_bytes(b"onnx")
    sidecar_path = model_path.with_suffix(".meta.json")
    payload = {"output_schema": {"kind": "binary_proba"}}
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    resolved = resolve_model_sidecar_metadata(model_path)

    assert resolved == payload
