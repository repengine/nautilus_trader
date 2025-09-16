#!/usr/bin/env python3
# ruff: noqa: RUF022
"""
Test fixtures for ML testing.

This module provides centralized test utilities and fixtures for ML tests.

"""

# Import existing factories
from ml.tests.fixtures.model_factory import TestDataFactory
from ml.tests.fixtures.model_factory import TestModelFactory

# Import common fixtures (for programmatic use, not just pytest)
from ml.tests.fixtures.common import (
    alternative_bar_type,
    alternative_instrument_id,
    base_feature_config,
    base_ml_config,
    base_signal_config,
    default_bar_type,
    default_instrument_id,
    default_venue,
    dummy_onnx_model,
    dummy_xgboost_model,
    mock_data_store,
    mock_feature_registry,
    mock_model_registry,
    mock_stores_bundle,
    model_registry_config,
    sample_feature_array,
    sample_feature_manifest,
    sample_features,
    sample_model_manifest,
    sample_predictions,
    test_component_id,
    test_timestamps,
)

# Builder classes are imported lazily to avoid circular imports during test discovery


__all__ = [
    # Factory classes
    "TestDataFactory",
    "TestModelFactory",
    # Builder classes
    "DataBuilder",
    "MLConfigBuilder",
    "MockBuilder",
    "RegistryBuilder",
    # Common fixtures (for programmatic use)
    "alternative_bar_type",
    "alternative_instrument_id",
    "base_feature_config",
    "base_ml_config",
    "base_signal_config",
    "default_bar_type",
    "default_instrument_id",
    "default_venue",
    "dummy_onnx_model",
    "dummy_xgboost_model",
    "mock_data_store",
    "mock_feature_registry",
    "mock_model_registry",
    "mock_stores_bundle",
    "model_registry_config",
    "sample_feature_array",
    "sample_feature_manifest",
    "sample_features",
    "sample_model_manifest",
    "sample_predictions",
    "test_component_id",
    "test_timestamps",
]


def __getattr__(name: str):  # pragma: no cover - utility for import-time behavior
    if name in {"DataBuilder", "MLConfigBuilder", "MockBuilder", "RegistryBuilder"}:
        from ml.tests import builders as _builders

        return getattr(_builders, name)
    raise AttributeError(name)
