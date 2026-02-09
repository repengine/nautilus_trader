"""
Unit tests for model loader helpers in base actor module.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml.actors.base import ONNXModelLoader, ProductionModelLoader
from ml.tests.utils.model_artifacts import write_stub_onnx_artifact


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry")


class _IO:
    def __init__(self, name: str) -> None:
        self.name = name


class _Session:
    def __init__(self) -> None:
        self._inputs = [_IO("input")]
        self._outputs = [_IO("output")]
        self._providers = ["CPUExecutionProvider"]

    def get_inputs(self) -> list[_IO]:
        return self._inputs

    def get_outputs(self) -> list[_IO]:
        return self._outputs

    def get_providers(self) -> list[str]:
        return self._providers


@pytest.mark.unit
def test_production_model_loader_falls_back_to_json_when_xgboost_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"alpha": 1}
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(payload), encoding="utf-8")

    class _Booster:
        def load_model(self, _path: str) -> None:
            raise RuntimeError("boom")

    import ml._imports as ml_imports

    monkeypatch.setattr(ml_imports, "HAS_XGBOOST", True)
    monkeypatch.setattr(ml_imports, "xgb", SimpleNamespace(Booster=_Booster))

    loader = ProductionModelLoader()
    model, metadata = loader.load_model(str(model_path))

    assert model == payload
    assert metadata == {"type": "json", "format": "json"}


@pytest.mark.unit
def test_production_model_loader_rejects_pickle_in_onnx_only_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"stub")
    monkeypatch.setenv("ML_ONNX_ONLY", "1")

    loader = ProductionModelLoader()

    with pytest.raises(ValueError, match="Pickle model formats are forbidden"):
        loader.load_model(str(model_path))


@pytest.mark.unit
def test_production_model_loader_rejects_pickle_in_standard_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.pickle"
    model_path.write_bytes(b"stub")
    monkeypatch.delenv("ML_ONNX_ONLY", raising=False)

    loader = ProductionModelLoader()

    with pytest.raises(ValueError, match="Pickle model formats"):
        loader.load_model(str(model_path))


@pytest.mark.unit
def test_production_model_loader_rejects_joblib_when_not_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"stub")

    monkeypatch.delenv("ML_ALLOW_JOBLIB", raising=False)
    monkeypatch.delenv("ML_TESTING", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    loader = ProductionModelLoader()

    with pytest.raises(ValueError, match="Joblib model format"):
        loader.load_model(str(model_path))


@pytest.mark.unit
def test_production_model_loader_loads_joblib_when_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"stub")
    monkeypatch.setenv("ML_ALLOW_JOBLIB", "1")

    class _StubModel:
        pass

    def _load(_path: str) -> _StubModel:
        return _StubModel()

    import ml._imports as ml_imports

    monkeypatch.setattr(ml_imports, "joblib", SimpleNamespace(load=_load))

    loader = ProductionModelLoader()
    model, metadata = loader.load_model(str(model_path))

    assert isinstance(model, _StubModel)
    assert metadata["format"] == "joblib"
    assert metadata["model_class"] == "_StubModel"


@pytest.mark.unit
def test_production_model_loader_loads_onnx_with_secure_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    write_stub_onnx_artifact(model_path, content=b"onnx")
    session = _Session()

    monkeypatch.setattr(
        "ml.common.security.secure_onnx_load",
        lambda **_kwargs: session,
    )

    loader = ProductionModelLoader()
    loaded, metadata = loader.load_model(str(model_path))

    assert loaded is session
    assert metadata["format"] == "onnx"
    assert metadata["input_names"] == ["input"]
    assert metadata["output_names"] == ["output"]


@pytest.mark.unit
def test_production_model_loader_strict_missing_digest_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    monkeypatch.setenv("ML_STRICT_MODEL_COMPATIBILITY", "1")
    monkeypatch.setenv("ML_ALLOW_COMPATIBILITY_MIGRATION_OVERRIDE", "0")
    monkeypatch.setenv("ML_ALLOW_UNSIGNED_ARTIFACTS", "0")

    loader = ProductionModelLoader()

    with pytest.raises(ValueError, match="No SHA-256 digest available"):
        loader.load_model(str(model_path))


@pytest.mark.unit
def test_production_model_loader_onnx_uses_sidecar_digest_and_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    model_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "integrity": {"sha256": "sha256:" + ("a" * 64)},
                "output_schema": {"kind": "binary_proba", "shape": [None, 1]},
                "calibration": {"kind": "platt", "params": {"coef": 1.0}},
            },
        ),
        encoding="utf-8",
    )
    captured_kwargs: dict[str, object] = {}
    session = _Session()

    def _secure_load(**kwargs: object) -> _Session:
        captured_kwargs.update(kwargs)
        return session

    monkeypatch.setattr("ml.common.security.secure_onnx_load", _secure_load)

    loader = ProductionModelLoader()
    loaded, metadata = loader.load_model(str(model_path))

    assert loaded is session
    assert captured_kwargs["expected_digest"] == "a" * 64
    assert captured_kwargs["strict_integrity"] is True
    assert metadata["output_schema"] == {"kind": "binary_proba", "shape": [None, 1]}
    assert metadata["calibration"] == {"kind": "platt", "params": {"coef": 1.0}}
    assert metadata["artifact_sha256_digest"] == "a" * 64


@pytest.mark.unit
def test_onnx_model_loader_loads_model_with_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    write_stub_onnx_artifact(model_path, content=b"onnx")
    session = _Session()

    monkeypatch.setattr(
        "ml.actors.base.to_session_options",
        lambda _cfg: (object(), ["CPUExecutionProvider"]),
    )
    monkeypatch.setattr(
        "ml.common.security.secure_onnx_load",
        lambda **_kwargs: session,
    )

    loader = ONNXModelLoader()
    loader._onnx_available = True

    loaded, metadata = loader.load_model(str(model_path))

    assert loaded is session
    assert metadata["type"] == "onnx"
    assert metadata["providers"] == ["CPUExecutionProvider"]
    assert metadata["input_names"] == ["input"]
    assert metadata["output_names"] == ["output"]


@pytest.mark.unit
def test_onnx_model_loader_require_output_semantics_raises_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    model_path.with_suffix(".meta.json").write_text(
        json.dumps({"artifact_sha256_digest": "b" * 64}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ML_REQUIRE_OUTPUT_SEMANTICS", "1")
    monkeypatch.setenv("ML_ALLOW_COMPATIBILITY_MIGRATION_OVERRIDE", "0")

    loader = ONNXModelLoader()
    loader._onnx_available = True

    with pytest.raises(ValueError, match="Output semantics validation failed"):
        loader.load_model(str(model_path))


@pytest.mark.unit
def test_onnx_model_loader_version_hash_is_stable(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    fixed_mtime = 12345.0
    import os

    os.utime(model_path, (fixed_mtime, fixed_mtime))

    loader = ONNXModelLoader()
    loader._onnx_available = True

    version_first = loader.get_model_version(str(model_path))
    version_second = loader.get_model_version(str(model_path))

    assert version_first == version_second
    assert len(version_first) == 8
