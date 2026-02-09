"""Unit tests for export-side reproducibility provenance serialization."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from ml.common.reproducibility import build_configured_reproducibility_provenance
from ml.training.export import ModelType
from ml.training.export import create_model_manifest_stub
from ml.training.export import save_model_with_metadata


@pytest.mark.unit
def test_save_model_with_metadata_serializes_reproducibility_payload(
    tmp_path: Path,
) -> None:
    """Verify sidecar metadata persists canonical reproducibility payload."""
    artifact_path = tmp_path / "model.onnx"
    artifact_path.write_bytes(b"onnx-artifact")
    payload = build_configured_reproducibility_provenance(
        primary_seed=19,
        deterministic_mode=True,
        context="unit export seed",
    )

    with patch("ml.training.export.detect_model_type", return_value=ModelType.ONNX):
        with patch("ml.training.export._save_onnx_model", return_value=artifact_path):
            saved_path = save_model_with_metadata(
                model=MagicMock(),
                path=tmp_path / "model",
                reproducibility_provenance=payload,
            )

    metadata_path = saved_path.with_suffix(saved_path.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reproducibility = metadata["reproducibility"]
    assert reproducibility["seed"] == 19
    assert reproducibility["deterministic_mode"] is True
    assert isinstance(reproducibility["python_version"], str)


@pytest.mark.unit
def test_save_model_with_metadata_when_reproducibility_payload_invalid_raises_value_error(
    tmp_path: Path,
) -> None:
    """Verify invalid reproducibility payload fails fast at export boundary."""
    artifact_path = tmp_path / "model.onnx"
    artifact_path.write_bytes(b"onnx-artifact")

    with patch("ml.training.export.detect_model_type", return_value=ModelType.ONNX):
        with patch("ml.training.export._save_onnx_model", return_value=artifact_path):
            with pytest.raises(ValueError, match="deterministic_mode"):
                save_model_with_metadata(
                    model=MagicMock(),
                    path=tmp_path / "model",
                    reproducibility_provenance={"seed": 3},
                )


@pytest.mark.unit
def test_create_model_manifest_stub_includes_reproducibility_payload() -> None:
    """Verify manifest stubs serialize reproducibility under training_config."""
    payload = build_configured_reproducibility_provenance(
        primary_seed=7,
        deterministic_mode=True,
        context="unit manifest seed",
    )

    manifest_data = create_model_manifest_stub(
        model=object(),
        feature_names=["f1"],
        architecture="onnx",
        reproducibility_provenance=payload,
    )

    training_config = manifest_data["training_config"]
    assert training_config["reproducibility"]["seed"] == 7
    assert training_config["reproducibility"]["deterministic_mode"] is True


@pytest.mark.unit
def test_create_model_manifest_stub_when_reproducibility_invalid_raises_value_error() -> None:
    """Verify manifest serialization validates reproducibility payload shape."""
    with pytest.raises(ValueError, match="deterministic_mode"):
        create_model_manifest_stub(
            model=object(),
            feature_names=["f1"],
            architecture="onnx",
            reproducibility_provenance={"seed": 1, "deterministic_mode": "yes"},
        )
