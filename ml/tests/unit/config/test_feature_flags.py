"""Unit tests for feature flags module.

These tests verify that feature flags correctly control which implementation
is used during god class decomposition refactoring.

CRITICAL: These tests must pass BEFORE implementing any facade.
"""

from __future__ import annotations

import os

import pytest

from ml.config.feature_flags import use_legacy_data_store
from ml.config.feature_flags import use_legacy_feature_engineer
from ml.config.feature_flags import use_legacy_ml_pipeline_orchestrator
from ml.config.feature_flags import use_legacy_ml_signal_actor


class TestFeatureEngineerFlag:
    """Test use_legacy_feature_engineer flag."""

    def test_feature_flag_legacy_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test feature flag returns True when env var = '1'."""
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "1")
        assert use_legacy_feature_engineer() is True

    def test_feature_flag_facade_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test feature flag returns False when env var = '0'."""
        monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", "0")
        assert use_legacy_feature_engineer() is False

    def test_feature_flag_default_is_facade(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test feature flag defaults to False (facade) when env var not set."""
        monkeypatch.delenv("ML_USE_LEGACY_FEATURE_ENGINEER", raising=False)
        assert use_legacy_feature_engineer() is False

    def test_feature_flag_ignores_non_1_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test feature flag only accepts '1' for legacy mode."""
        # Any value other than "1" should return False
        for value in ["true", "True", "yes", "YES", "on", "ON", "2"]:
            monkeypatch.setenv("ML_USE_LEGACY_FEATURE_ENGINEER", value)
            assert use_legacy_feature_engineer() is False, f"Value '{value}' should be False"


class TestDataStoreFlag:
    """Test use_legacy_data_store flag."""

    def test_feature_flag_legacy_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test feature flag returns True when env var = '1'."""
        monkeypatch.setenv("ML_USE_LEGACY_DATA_STORE", "1")
        assert use_legacy_data_store() is True

    def test_feature_flag_facade_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test feature flag returns False when env var = '0'."""
        monkeypatch.setenv("ML_USE_LEGACY_DATA_STORE", "0")
        assert use_legacy_data_store() is False

    def test_feature_flag_default_is_facade(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test feature flag defaults to False (facade) when env var not set."""
        monkeypatch.delenv("ML_USE_LEGACY_DATA_STORE", raising=False)
        assert use_legacy_data_store() is False


class TestMLPipelineOrchestratorFlag:
    """Test use_legacy_ml_pipeline_orchestrator flag."""

    def test_feature_flag_legacy_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test feature flag returns True when env var = '1'."""
        monkeypatch.setenv("ML_USE_LEGACY_ML_PIPELINE_ORCHESTRATOR", "1")
        assert use_legacy_ml_pipeline_orchestrator() is True

    def test_feature_flag_facade_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test feature flag returns False when env var = '0'."""
        monkeypatch.setenv("ML_USE_LEGACY_ML_PIPELINE_ORCHESTRATOR", "0")
        assert use_legacy_ml_pipeline_orchestrator() is False

    def test_feature_flag_default_is_facade(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test feature flag defaults to False (facade) when env var not set."""
        monkeypatch.delenv("ML_USE_LEGACY_ML_PIPELINE_ORCHESTRATOR", raising=False)
        assert use_legacy_ml_pipeline_orchestrator() is False


class TestMLSignalActorFlag:
    """Test use_legacy_ml_signal_actor flag."""

    def test_feature_flag_legacy_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test feature flag returns True when env var = '1'."""
        monkeypatch.setenv("ML_USE_LEGACY_ML_SIGNAL_ACTOR", "1")
        assert use_legacy_ml_signal_actor() is True

    def test_feature_flag_facade_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test feature flag returns False when env var = '0'."""
        monkeypatch.setenv("ML_USE_LEGACY_ML_SIGNAL_ACTOR", "0")
        assert use_legacy_ml_signal_actor() is False

    def test_feature_flag_default_is_facade(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test feature flag defaults to False (facade) when env var not set."""
        monkeypatch.delenv("ML_USE_LEGACY_ML_SIGNAL_ACTOR", raising=False)
        assert use_legacy_ml_signal_actor() is False
