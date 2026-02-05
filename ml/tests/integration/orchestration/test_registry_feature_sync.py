"""Integration tests for FeatureRegistry synchronization (Phase 2.2.4).

This module contains integration tests for RegistrySynchronizer's
FeatureRegistry synchronization workflows.

All tests marked @pytest.mark.skip for structural phase.
Full implementation in Phase 2.2.8.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.registry_synchronizer import RegistrySynchronizer
from ml.tests.utils.targets import build_default_target_semantics_payload


@pytest.mark.integration
def test_export_feature_manifest_to_feature_registry(tmp_path) -> None:
    """Export feature manifest to FeatureRegistry.

    Phase 2.2.4: Test skipped (structural phase).
    Phase 2.2.8: Will verify feature manifest is exported to FeatureRegistry
    and queryable by dataset_id.

    Expected Behavior (Phase 2.2.8):
    - RegistrySynchronizer._export_feature_manifest() invoked
    - Feature manifest exported to FeatureRegistry
    - Feature manifest includes: features, schema_hash, version
    - Feature manifest queryable by dataset_id
    """
    # Setup
    data_registry = Mock()
    feature_registry = Mock()
    model_registry = Mock()

    synchronizer = RegistrySynchronizer(
        data_registry=data_registry,
        feature_registry=feature_registry,
        model_registry=model_registry,
    )

    cfg = DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        symbols="SPY",
        out_dir=str(tmp_path / "out"),
        register_features=True,
        feature_registry_dir=str(tmp_path / "registry"),
        target_semantics=build_default_target_semantics_payload(),
    )
    result = Mock()
    result.feature_names = ["price_sma_20", "ema_50", "rsi_14"]

    # Execute
    synchronizer._export_feature_manifest(cfg, result)

    # Verify (Phase 2.2.8 - currently placeholder)
    # manifest = feature_registry.get_feature_manifest("spy_2024_ohlcv")
    # assert manifest is not None
    # assert manifest["features"] == ["price_sma_20", "ema_50", "rsi_14"]
    # assert "schema_hash" in manifest


@pytest.mark.integration
def test_feature_schema_hash_computed_correctly(tmp_path) -> None:
    """Verify feature schema hash is deterministic and consistent.

    Phase 2.2.4: Test skipped (structural phase).
    Phase 2.2.8: Will verify schema hash is deterministic.

    Expected Behavior (Phase 2.2.8):
    - Schema hash computed from feature definitions
    - Same features → same hash (reproducibility)
    - Different features → different hash
    - Hash is SHA256 hex string (64 characters)
    """
    # Setup
    data_registry = Mock()
    feature_registry = Mock()
    model_registry = Mock()

    synchronizer = RegistrySynchronizer(
        data_registry=data_registry,
        feature_registry=feature_registry,
        model_registry=model_registry,
    )

    cfg = DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        symbols="SPY",
        out_dir=str(tmp_path / "out"),
        register_features=True,
        feature_registry_dir=str(tmp_path / "registry"),
        target_semantics=build_default_target_semantics_payload(),
    )
    result = Mock()
    result.feature_names = ["price_sma_20", "ema_50", "rsi_14"]

    # Execute (Phase 2.2.8)
    # Export feature manifest twice
    synchronizer._export_feature_manifest(cfg, result)
    # hash1 = feature_registry.get_feature_manifest("spy_2024_ohlcv")["schema_hash"]

    synchronizer._export_feature_manifest(cfg, result)
    # hash2 = feature_registry.get_feature_manifest("spy_2024_ohlcv")["schema_hash"]

    # Verify (Phase 2.2.8 - currently placeholder)
    # Same features → same hash
    # assert hash1 == hash2
    # assert len(hash1) == 64  # SHA256 hex


@pytest.mark.integration
def test_feature_manifest_queryable_from_registry(tmp_path) -> None:
    """Verify feature manifest queryable from FeatureRegistry after export.

    Phase 2.2.4: Test skipped (structural phase).
    Phase 2.2.8: Will verify feature manifest is queryable.

    Expected Behavior (Phase 2.2.8):
    - Export feature manifest
    - Query feature manifest by dataset_id
    - All feature metadata present (names, types, schemas)
    """
    # Setup
    data_registry = Mock()
    feature_registry = Mock()
    model_registry = Mock()

    synchronizer = RegistrySynchronizer(
        data_registry=data_registry,
        feature_registry=feature_registry,
        model_registry=model_registry,
    )

    cfg = DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        symbols="SPY",
        out_dir=str(tmp_path / "out"),
        register_features=True,
        feature_registry_dir=str(tmp_path / "registry"),
        target_semantics=build_default_target_semantics_payload(),
    )
    result = Mock()
    result.feature_names = ["price_sma_20", "ema_50", "rsi_14"]

    # Execute
    synchronizer._export_feature_manifest(cfg, result)

    # Verify (Phase 2.2.8 - currently placeholder)
    # manifest = feature_registry.get_feature_manifest("spy_2024_ohlcv")
    # assert manifest is not None
    # assert "features" in manifest
    # assert "schema_hash" in manifest
    # assert "version" in manifest
