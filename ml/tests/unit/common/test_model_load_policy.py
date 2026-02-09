from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.common.model_load_policy import apply_direct_model_load_policy


pytestmark = pytest.mark.unit


def _write_sidecar(model_path: Path, payload: dict[str, object]) -> None:
    sidecar_path = model_path.with_suffix(".meta.json")
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")


def test_apply_direct_model_load_policy_permissive_allows_missing_digest(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")

    result = apply_direct_model_load_policy(
        model_path=model_path,
        env={
            "ML_STRICT_MODEL_COMPATIBILITY": "false",
            "ML_REQUIRE_OUTPUT_SEMANTICS": "false",
        },
    )

    assert result.expected_digest is None
    assert result.strict_integrity is False


def test_apply_direct_model_load_policy_strict_missing_digest_raises(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")

    with pytest.raises(ValueError, match="No SHA-256 digest available"):
        apply_direct_model_load_policy(
            model_path=model_path,
            env={
                "ML_STRICT_MODEL_COMPATIBILITY": "true",
                "ML_ALLOW_COMPATIBILITY_MIGRATION_OVERRIDE": "false",
                "ML_ALLOW_UNSIGNED_ARTIFACTS": "false",
            },
        )


def test_apply_direct_model_load_policy_require_output_semantics_raises_when_missing(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    _write_sidecar(model_path, {"artifact_sha256_digest": "a" * 64})

    with pytest.raises(ValueError, match="Output semantics validation failed"):
        apply_direct_model_load_policy(
            model_path=model_path,
            env={
                "ML_REQUIRE_OUTPUT_SEMANTICS": "true",
                "ML_ALLOW_COMPATIBILITY_MIGRATION_OVERRIDE": "false",
            },
        )


def test_apply_direct_model_load_policy_merges_sidecar_metadata_and_digest(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    _write_sidecar(
        model_path,
        {
            "integrity": {"sha256": f"sha256:{'b' * 64}"},
            "inference": {
                "output_schema": {"kind": "binary_proba", "shape": [None, 1]},
                "calibration": {"kind": "platt", "params": {"coef": 1.0}},
            },
        },
    )

    result = apply_direct_model_load_policy(model_path=model_path)

    assert result.expected_digest == "b" * 64
    assert result.strict_integrity is True
    assert result.metadata["output_schema"] == {"kind": "binary_proba", "shape": [None, 1]}
    assert result.metadata["calibration"] == {"kind": "platt", "params": {"coef": 1.0}}

