#!/usr/bin/env python3

"""Test data directory for ML module tests."""

from pathlib import Path


def get_test_data_dir() -> Path:
    """Get the path to the test data directory."""
    return Path(__file__).parent


def get_model_registry_dir() -> Path:
    """Get the path to the test model registry."""
    return get_test_data_dir() / "model_registry"


def get_model_registry_rollout_dir() -> Path:
    """Get the path to the test model registry for rollout testing."""
    return get_test_data_dir() / "model_registry_rollout"