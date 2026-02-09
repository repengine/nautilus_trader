from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId

from ml.actors.common.model import ModelComponent
from ml.config.base import MLActorConfig
from ml.tests.utils.db import build_postgres_url


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")

TEST_DB_CONNECTION = build_postgres_url()


class _IO:
    def __init__(self, name: str) -> None:
        self.name = name
        self.shape = [None, 2]
        self.type = "tensor(float)"


class _Session:
    def __init__(self) -> None:
        self._inputs = [_IO("input")]
        self._outputs = [_IO("output")]

    def get_inputs(self) -> list[_IO]:
        return self._inputs

    def get_outputs(self) -> list[_IO]:
        return self._outputs


class _OnnxChecker:
    @staticmethod
    def check_model(_model: object) -> None:
        return None


class _OnnxModule:
    checker = _OnnxChecker

    @staticmethod
    def load(_path: str) -> object:
        return object()


class _OrtGraphOptimizationLevel:
    ORT_ENABLE_ALL = 1


class _OrtSessionOptions:
    def __init__(self) -> None:
        self.graph_optimization_level: int | None = None


class _OrtModule:
    GraphOptimizationLevel = _OrtGraphOptimizationLevel
    SessionOptions = _OrtSessionOptions


def _make_config(model_path: Path) -> MLActorConfig:
    return MLActorConfig(
        model_path=str(model_path),
        model_id="direct-path-model",
        bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-LAST-EXTERNAL"),
        instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
        db_connection=TEST_DB_CONNECTION,
        use_dummy_stores=True,
    )


def _patch_onnx_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    import ml._imports as ml_imports

    monkeypatch.setattr(ml_imports, "HAS_ONNX", True)
    monkeypatch.setattr(ml_imports, "HAS_ONNX_CORE", True)
    monkeypatch.setattr(ml_imports, "ONNX_CORE_IMPORT_ERROR", None)
    monkeypatch.setattr(ml_imports, "onnx", _OnnxModule)
    monkeypatch.setattr(ml_imports, "ort", _OrtModule)


def test_model_component_direct_load_applies_sidecar_digest_and_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    model_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "integrity": {"sha256": "sha256:" + ("f" * 64)},
                "output_schema": {"kind": "binary_proba", "shape": [None, 1]},
                "calibration": {"kind": "platt", "params": {"coef": 1.0}},
            },
        ),
        encoding="utf-8",
    )
    _patch_onnx_imports(monkeypatch)

    captured_kwargs: dict[str, object] = {}

    def _secure_load(**kwargs: object) -> _Session:
        captured_kwargs.update(kwargs)
        return _Session()

    monkeypatch.setattr("ml.common.security.secure_onnx_load", _secure_load)

    component = ModelComponent(_make_config(model_path), logging.getLogger("test_model_policy"))
    component.load_model()

    assert captured_kwargs["expected_digest"] == "f" * 64
    assert captured_kwargs["strict_integrity"] is True
    assert component.model_metadata["artifact_sha256_digest"] == "f" * 64
    assert component.model_metadata["output_schema"] == {"kind": "binary_proba", "shape": [None, 1]}
    assert component.model_metadata["calibration"] == {"kind": "platt", "params": {"coef": 1.0}}


def test_model_component_direct_load_strict_missing_digest_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    _patch_onnx_imports(monkeypatch)
    monkeypatch.setenv("ML_STRICT_MODEL_COMPATIBILITY", "1")
    monkeypatch.setenv("ML_ALLOW_COMPATIBILITY_MIGRATION_OVERRIDE", "0")
    monkeypatch.setenv("ML_ALLOW_UNSIGNED_ARTIFACTS", "0")

    component = ModelComponent(_make_config(model_path), logging.getLogger("test_model_policy"))

    with pytest.raises(RuntimeError, match="No SHA-256 digest available"):
        component.load_model()
