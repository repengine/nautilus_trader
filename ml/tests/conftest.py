#!/usr/bin/env python3

"""Common pytest fixtures for ML module tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.tests.data import get_model_registry_dir
from ml.tests.data import get_model_registry_rollout_dir
from ml.tests.data import get_test_data_dir


@pytest.fixture
def test_data_dir() -> Path:
    """Provide path to test data directory."""
    return get_test_data_dir()


@pytest.fixture
def model_registry_dir() -> Path:
    """Provide path to test model registry."""
    return get_model_registry_dir()


@pytest.fixture
def model_registry_rollout_dir() -> Path:
    """Provide path to test model registry for rollout testing."""
    return get_model_registry_rollout_dir()


@pytest.fixture
def xgb_v1_model_path(model_registry_dir: Path) -> Path:
    """Provide path to XGBoost v1 test model."""
    return model_registry_dir / "models" / "xgb_v1.json"


@pytest.fixture
def xgb_v2_model_path(model_registry_dir: Path) -> Path:
    """Provide path to XGBoost v2 test model."""
    return model_registry_dir / "models" / "xgb_v2.json"


@pytest.fixture
def prod_onnx_model_path(model_registry_rollout_dir: Path) -> Path:
    """Provide path to production ONNX test model."""
    return model_registry_rollout_dir / "models" / "prod.onnx"


@pytest.fixture
def new_onnx_model_path(model_registry_rollout_dir: Path) -> Path:
    """Provide path to new ONNX test model."""
    return model_registry_rollout_dir / "models" / "new.onnx"
