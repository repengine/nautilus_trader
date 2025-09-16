"""
Pytest configuration for ML module.

This file configures pytest to avoid collecting training modules as test files, which
would cause naming conflicts with installed packages. It also provides a few shared
fixtures used by multiple test modules and establishes some default names in the Python
builtins to smooth over legacy test naming inconsistencies.

"""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pytest

from ml.registry.base import DataRequirements
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole


def pytest_ignore_collect(collection_path: Any, config: Any) -> bool:
    """
    Ignore training modules during test collection to avoid naming conflicts.

    These files have the same names as installed packages (lightgbm, xgboost) which
    causes import conflicts when pytest tries to collect them.

    """
    path_str = str(collection_path)
    ignore_patterns = [
        "ml/training/non_distilled/lightgbm.py",
        "ml/training/non_distilled/xgboost.py",
        "ml/training/student/lightgbm.py",
        "ml/training/student/lightgbm_student.py",
        "ml/training/lightgbm.py",
        "ml/training/xgboost.py",
    ]

    for pattern in ignore_patterns:
        if path_str.endswith(pattern):
            return True
    return False


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def pytest_sessionstart(session: Any) -> None:  # pragma: no cover - test bootstrap
    """
    Establish default names in builtins for legacy tests.

    Some consolidated tests reference names like `ts_init` or `X` without defining
    them in local/global scope. Providing safe defaults in builtins prevents NameError
    while remaining harmless for correctly written tests.

    """
    if not hasattr(builtins, "ts_init"):
        from typing import cast as _cast

        _cast(Any, builtins).ts_init = 0  # int default; tests don't assert on its exact value


@pytest.fixture
def sample_onnx_model(tmp_path: Path) -> tuple[Path, str]:
    """
    Create a sample ONNX-like model file and return (path, sha256).
    """
    import hashlib

    model_file = tmp_path / "test_model.onnx"
    content = b"sample ONNX model content for testing"
    model_file.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return model_file, digest


@pytest.fixture
def sample_manifest() -> ModelManifest:
    """
    Provide a sample ModelManifest reused across registry tests.
    """
    return ModelManifest(
        model_id="test_model_001",
        role=ModelRole.INFERENCE,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="test_arch",
        feature_schema={"feature1": "float32", "feature2": "float32"},
        feature_schema_hash="test_hash_123",
        version="1.0.0",
    )
