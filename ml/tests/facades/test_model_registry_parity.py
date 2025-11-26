#!/usr/bin/env python3

"""
Parity tests for ModelRegistry: Legacy vs Facade implementations.

These tests ensure that the ModelRegistryFacade produces identical results
to the legacy ModelRegistry class for all operations.

Feature flag: ML_USE_LEGACY_MODEL_REGISTRY
- "1": Use legacy ModelRegistry
- "0" (default): Use ModelRegistryFacade
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml.registry.base import (
    DataRequirements,
    DeploymentStatus,
    ModelInfo,
    ModelManifest,
    ModelRole,
)
from ml.registry.dataclasses import CanaryConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_onnx_model(tmp_path: Path) -> tuple[Path, str]:
    """Create a sample ONNX model file with known SHA-256 digest."""
    # Create parent dir for the model within registry path
    model_file = tmp_path / "test_model.onnx"
    content = b"sample ONNX model content for parity testing"
    model_file.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return model_file, digest


@pytest.fixture
def create_manifest() -> Any:
    """Factory fixture to create manifests with unique IDs."""
    counter = [0]

    def _create(
        model_id: str | None = None,
        role: ModelRole = ModelRole.INFERENCE,
        architecture: str = "XGBoost",
        version: str = "1.0.0",
        performance_metrics: dict[str, float] | None = None,
    ) -> ModelManifest:
        if model_id is None:
            model_id = f"model_{counter[0]}"
            counter[0] += 1

        return ModelManifest(
            model_id=model_id,
            role=role,
            data_requirements=DataRequirements.L1_ONLY,
            architecture=architecture,
            feature_schema={"price": "float64", "volume": "float64"},
            feature_schema_hash="parity_test_hash_123",
            version=version,
            created_at=time.time(),
            last_modified=time.time(),
            serveable=True,
            artifact_format="onnx",
            performance_metrics=performance_metrics or {},
        )

    return _create


@pytest.fixture
def legacy_registry(tmp_path: Path) -> Any:
    """Create legacy ModelRegistry instance."""
    # Ensure legacy mode
    os.environ["ML_USE_LEGACY_MODEL_REGISTRY"] = "1"
    try:
        from ml.registry.model_registry import ModelRegistry

        registry_path = tmp_path / "legacy_registry"
        registry_path.mkdir(parents=True, exist_ok=True)
        return ModelRegistry(registry_path)
    finally:
        os.environ.pop("ML_USE_LEGACY_MODEL_REGISTRY", None)


@pytest.fixture
def facade_registry(tmp_path: Path) -> Any:
    """Create ModelRegistryFacade instance."""
    os.environ["ML_USE_LEGACY_MODEL_REGISTRY"] = "0"
    try:
        from ml.registry.model_registry_facade import ModelRegistryFacade

        registry_path = tmp_path / "facade_registry"
        registry_path.mkdir(parents=True, exist_ok=True)
        return ModelRegistryFacade(registry_path)
    finally:
        os.environ.pop("ML_USE_LEGACY_MODEL_REGISTRY", None)


def copy_model_to_registry(
    model_path: Path,
    registry: Any,
) -> Path:
    """Copy model file to registry path for security validation."""
    dest = registry.registry_path / model_path.name
    dest.write_bytes(model_path.read_bytes())
    return dest


# =============================================================================
# Core Operation Parity Tests
# =============================================================================


class TestRegisterModelParity:
    """Verify register_model produces identical results."""

    def test_parity_register_model_basic(
        self,
        legacy_registry: Any,
        facade_registry: Any,
        sample_onnx_model: tuple[Path, str],
        create_manifest: Any,
    ) -> None:
        """Verify registration produces identical model_id and digest."""
        _, expected_digest = sample_onnx_model

        # Copy model to each registry
        legacy_model = copy_model_to_registry(sample_onnx_model[0], legacy_registry)
        facade_model = copy_model_to_registry(sample_onnx_model[0], facade_registry)

        # Create identical manifests with explicit IDs
        legacy_manifest = create_manifest(model_id="parity_model_1")
        facade_manifest = create_manifest(model_id="parity_model_1")

        # Register in both
        legacy_id = legacy_registry.register_model(legacy_model, legacy_manifest)
        facade_id = facade_registry.register_model(facade_model, facade_manifest)

        # Verify model IDs match
        assert legacy_id == facade_id

        # Verify digests match
        legacy_info = legacy_registry.get_model(legacy_id)
        facade_info = facade_registry.get_model(facade_id)

        assert legacy_info is not None
        assert facade_info is not None
        assert legacy_info.manifest.artifact_sha256_digest == expected_digest
        assert facade_info.manifest.artifact_sha256_digest == expected_digest

    def test_parity_register_model_auto_version(
        self,
        legacy_registry: Any,
        facade_registry: Any,
        sample_onnx_model: tuple[Path, str],
        create_manifest: Any,
    ) -> None:
        """Verify auto-versioning produces identical versions."""
        legacy_model = copy_model_to_registry(sample_onnx_model[0], legacy_registry)
        facade_model = copy_model_to_registry(sample_onnx_model[0], facade_registry)

        # First model with explicit version
        m1_legacy = create_manifest(model_id="m1", version="1.0.0")
        m1_facade = create_manifest(model_id="m1", version="1.0.0")
        legacy_registry.register_model(legacy_model, m1_legacy)
        facade_registry.register_model(facade_model, m1_facade)

        # Second model without version (should auto-increment)
        m2_legacy = create_manifest(model_id="m2", version="")
        m2_facade = create_manifest(model_id="m2", version="")
        legacy_registry.register_model(legacy_model, m2_legacy)
        facade_registry.register_model(facade_model, m2_facade)

        # Verify auto-assigned versions match
        legacy_m2 = legacy_registry.get_model("m2")
        facade_m2 = facade_registry.get_model("m2")
        assert legacy_m2 is not None
        assert facade_m2 is not None
        assert legacy_m2.manifest.version == facade_m2.manifest.version


class TestDeployModelParity:
    """Verify deploy_model produces identical behavior."""

    def test_parity_deploy_model(
        self,
        legacy_registry: Any,
        facade_registry: Any,
        sample_onnx_model: tuple[Path, str],
        create_manifest: Any,
    ) -> None:
        """Verify deployment behavior is identical."""
        # Setup
        legacy_model = copy_model_to_registry(sample_onnx_model[0], legacy_registry)
        facade_model = copy_model_to_registry(sample_onnx_model[0], facade_registry)

        legacy_manifest = create_manifest(model_id="deploy_test")
        facade_manifest = create_manifest(model_id="deploy_test")

        legacy_registry.register_model(legacy_model, legacy_manifest)
        facade_registry.register_model(facade_model, facade_manifest)

        # Deploy in both
        legacy_result = legacy_registry.deploy_model("deploy_test", "target_1")
        facade_result = facade_registry.deploy_model("deploy_test", "target_1")

        # Verify results match
        assert legacy_result == facade_result == True

        # Verify deployment state matches
        legacy_info = legacy_registry.get_model("deploy_test")
        facade_info = facade_registry.get_model("deploy_test")

        assert legacy_info is not None
        assert facade_info is not None
        assert legacy_info.deployment_status == facade_info.deployment_status
        assert legacy_info.deployed_to == facade_info.deployed_to


class TestRollbackParity:
    """Verify rollback produces identical behavior."""

    def test_parity_rollback(
        self,
        legacy_registry: Any,
        facade_registry: Any,
        sample_onnx_model: tuple[Path, str],
        create_manifest: Any,
    ) -> None:
        """Verify rollback behavior is identical."""
        # Setup - register two models in each
        legacy_model = copy_model_to_registry(sample_onnx_model[0], legacy_registry)
        facade_model = copy_model_to_registry(sample_onnx_model[0], facade_registry)

        for registry, model_path in [(legacy_registry, legacy_model), (facade_registry, facade_model)]:
            registry.register_model(model_path, create_manifest(model_id="rollback_v1"))
            registry.register_model(model_path, create_manifest(model_id="rollback_v2"))
            registry.deploy_model("rollback_v2", "target")

        # Perform rollback
        legacy_result = legacy_registry.rollback("target", "rollback_v1")
        facade_result = facade_registry.rollback("target", "rollback_v1")

        # Verify results match
        assert legacy_result == facade_result == True

        # Verify state matches
        legacy_v1 = legacy_registry.get_model("rollback_v1")
        facade_v1 = facade_registry.get_model("rollback_v1")

        assert legacy_v1 is not None and facade_v1 is not None
        assert legacy_v1.deployment_status == facade_v1.deployment_status == DeploymentStatus.ACTIVE


class TestConfigureABTestParity:
    """Verify configure_ab_test produces identical results."""

    def test_parity_configure_ab_test(
        self,
        legacy_registry: Any,
        facade_registry: Any,
        sample_onnx_model: tuple[Path, str],
        create_manifest: Any,
    ) -> None:
        """Verify A/B test configuration is identical."""
        # Setup
        legacy_model = copy_model_to_registry(sample_onnx_model[0], legacy_registry)
        facade_model = copy_model_to_registry(sample_onnx_model[0], facade_registry)

        for registry, model_path in [(legacy_registry, legacy_model), (facade_registry, facade_model)]:
            registry.register_model(model_path, create_manifest(model_id="ab_model_a"))
            registry.register_model(model_path, create_manifest(model_id="ab_model_b"))

        # Configure A/B test
        legacy_config = legacy_registry.configure_ab_test(
            models=["ab_model_a", "ab_model_b"],
            split_ratio=0.5,
            duration_hours=24,
            target="ab_target",
        )
        facade_config = facade_registry.configure_ab_test(
            models=["ab_model_a", "ab_model_b"],
            split_ratio=0.5,
            duration_hours=24,
            target="ab_target",
        )

        # Verify configurations match (ignoring timestamps)
        assert legacy_config is not None
        assert facade_config is not None
        assert legacy_config["model_a"] == facade_config["model_a"]
        assert legacy_config["model_b"] == facade_config["model_b"]
        assert legacy_config["split_ratio"] == facade_config["split_ratio"]
        assert legacy_config["status"] == facade_config["status"]


class TestCompareModelsStatisticallyParity:
    """Verify statistical comparison produces identical results."""

    def test_parity_compare_models_statistically(
        self,
        legacy_registry: Any,
        facade_registry: Any,
        sample_onnx_model: tuple[Path, str],
        create_manifest: Any,
    ) -> None:
        """Verify statistical comparison results are identical."""
        # Setup
        legacy_model = copy_model_to_registry(sample_onnx_model[0], legacy_registry)
        facade_model = copy_model_to_registry(sample_onnx_model[0], facade_registry)

        for registry, model_path in [(legacy_registry, legacy_model), (facade_registry, facade_model)]:
            registry.register_model(model_path, create_manifest(model_id="stat_model_a"))
            registry.register_model(model_path, create_manifest(model_id="stat_model_b"))

            # Add identical performance history
            for i in range(10):
                registry.track_performance("stat_model_a", {"accuracy": 0.85 + 0.01 * i})
                registry.track_performance("stat_model_b", {"accuracy": 0.90 + 0.01 * i})

        # Compare statistically
        legacy_result = legacy_registry.compare_models_statistically(
            model_ids=["stat_model_a", "stat_model_b"],
            metric="accuracy",
        )
        facade_result = facade_registry.compare_models_statistically(
            model_ids=["stat_model_a", "stat_model_b"],
            metric="accuracy",
        )

        # Verify results match
        assert legacy_result is not None
        assert facade_result is not None
        assert np.isclose(
            legacy_result["p_value_approx"],
            facade_result["p_value_approx"],
            rtol=1e-6,
        )
        assert legacy_result["statistically_significant"] == facade_result["statistically_significant"]


class TestGetModelLineageParity:
    """Verify lineage queries produce identical results."""

    def test_parity_get_model_lineage(
        self,
        legacy_registry: Any,
        facade_registry: Any,
        sample_onnx_model: tuple[Path, str],
        create_manifest: Any,
    ) -> None:
        """Verify lineage results are identical."""
        # Setup parent-child relationships
        legacy_model = copy_model_to_registry(sample_onnx_model[0], legacy_registry)
        facade_model = copy_model_to_registry(sample_onnx_model[0], facade_registry)

        for registry, model_path in [(legacy_registry, legacy_model), (facade_registry, facade_model)]:
            # Register parent
            parent = create_manifest(model_id="lineage_parent")
            parent.serveable = False
            parent.artifact_format = "none"
            registry.register_model(model_path, parent)

            # Register child with parent_id
            child = create_manifest(model_id="lineage_child")
            child.parent_id = "lineage_parent"
            registry.register_model(model_path, child)

        # Get lineage
        legacy_lineage = legacy_registry.get_model_lineage("lineage_child")
        facade_lineage = facade_registry.get_model_lineage("lineage_child")

        # Verify lineage matches
        assert len(legacy_lineage) == len(facade_lineage)
        for legacy_model, facade_model in zip(legacy_lineage, facade_lineage):
            assert legacy_model.manifest.model_id == facade_model.manifest.model_id


class TestCanaryDeploymentParity:
    """Verify canary deployment workflow is identical."""

    def test_parity_canary_deployment_flow(
        self,
        legacy_registry: Any,
        facade_registry: Any,
        sample_onnx_model: tuple[Path, str],
        create_manifest: Any,
    ) -> None:
        """Verify canary deployment decisions are identical."""
        # Setup
        legacy_model = copy_model_to_registry(sample_onnx_model[0], legacy_registry)
        facade_model = copy_model_to_registry(sample_onnx_model[0], facade_registry)

        for registry, model_path in [(legacy_registry, legacy_model), (facade_registry, facade_model)]:
            registry.register_model(model_path, create_manifest(model_id="canary_model"))

        # Start canary
        config = CanaryConfig(
            traffic_percentage=5.0,
            success_metric="accuracy",
            min_samples=10,
        )

        legacy_deployment_id = legacy_registry.start_canary_deployment(
            model_id="canary_model",
            target="canary_target",
            config=config,
        )
        facade_deployment_id = facade_registry.start_canary_deployment(
            model_id="canary_model",
            target="canary_target",
            config=config,
        )

        # Add metrics
        for registry, deployment_id in [
            (legacy_registry, legacy_deployment_id),
            (facade_registry, facade_deployment_id),
        ]:
            for i in range(5):
                registry.update_canary_metrics(deployment_id, metric_value=0.95 + 0.01 * i)

        # Evaluate
        legacy_should_promote, legacy_reason = legacy_registry.evaluate_canary(legacy_deployment_id)
        facade_should_promote, facade_reason = facade_registry.evaluate_canary(facade_deployment_id)

        # Verify decisions match
        assert legacy_should_promote == facade_should_promote
        assert legacy_reason == facade_reason


class TestGradualRolloutParity:
    """Verify gradual rollout workflow is identical."""

    def test_parity_gradual_rollout(
        self,
        legacy_registry: Any,
        facade_registry: Any,
        sample_onnx_model: tuple[Path, str],
        create_manifest: Any,
    ) -> None:
        """Verify rollout stage progression is identical."""
        # Setup
        legacy_model = copy_model_to_registry(sample_onnx_model[0], legacy_registry)
        facade_model = copy_model_to_registry(sample_onnx_model[0], facade_registry)

        stages = [0.1, 0.25, 0.5, 1.0]

        for registry, model_path in [(legacy_registry, legacy_model), (facade_registry, facade_model)]:
            registry.register_model(model_path, create_manifest(model_id="rollout_current"))
            registry.register_model(model_path, create_manifest(model_id="rollout_new"))
            registry.deploy_model("rollout_current", "rollout_target")

        # Start rollout
        legacy_rollout_id = legacy_registry.start_gradual_rollout(
            current_model_id="rollout_current",
            new_model_id="rollout_new",
            target="rollout_target",
            stages=stages,
            stage_duration_minutes=60,
        )
        facade_rollout_id = facade_registry.start_gradual_rollout(
            current_model_id="rollout_current",
            new_model_id="rollout_new",
            target="rollout_target",
            stages=stages,
            stage_duration_minutes=60,
        )

        # Verify initial status matches
        legacy_status = legacy_registry.get_rollout_status(legacy_rollout_id)
        facade_status = facade_registry.get_rollout_status(facade_rollout_id)

        assert legacy_status is not None
        assert facade_status is not None
        assert legacy_status["current_stage"] == facade_status["current_stage"]
        assert legacy_status["stages"] == facade_status["stages"]

        # Advance stage
        legacy_advanced = legacy_registry.advance_rollout_stage(legacy_rollout_id)
        facade_advanced = facade_registry.advance_rollout_stage(facade_rollout_id)

        assert legacy_advanced == facade_advanced == True

        # Verify stages match after advancement
        legacy_status = legacy_registry.get_rollout_status(legacy_rollout_id)
        facade_status = facade_registry.get_rollout_status(facade_rollout_id)

        assert legacy_status["current_stage"] == facade_status["current_stage"]


# =============================================================================
# Query Parity Tests
# =============================================================================


class TestQueryParity:
    """Verify query operations produce identical results."""

    def test_parity_get_all_models(
        self,
        legacy_registry: Any,
        facade_registry: Any,
        sample_onnx_model: tuple[Path, str],
        create_manifest: Any,
    ) -> None:
        """Verify get_all_models returns identical model count."""
        legacy_model = copy_model_to_registry(sample_onnx_model[0], legacy_registry)
        facade_model = copy_model_to_registry(sample_onnx_model[0], facade_registry)

        # Register same number of models
        for i in range(3):
            legacy_registry.register_model(legacy_model, create_manifest(model_id=f"query_model_{i}"))
            facade_registry.register_model(facade_model, create_manifest(model_id=f"query_model_{i}"))

        legacy_models = legacy_registry.get_all_models()
        facade_models = facade_registry.get_all_models()

        assert len(legacy_models) == len(facade_models) == 3

    def test_parity_list_compatible(
        self,
        legacy_registry: Any,
        facade_registry: Any,
        sample_onnx_model: tuple[Path, str],
        create_manifest: Any,
    ) -> None:
        """Verify list_compatible returns identical results."""
        legacy_model = copy_model_to_registry(sample_onnx_model[0], legacy_registry)
        facade_model = copy_model_to_registry(sample_onnx_model[0], facade_registry)

        for registry, model_path in [(legacy_registry, legacy_model), (facade_registry, facade_model)]:
            registry.register_model(model_path, create_manifest(model_id="compat_1"))
            registry.register_model(model_path, create_manifest(model_id="compat_2"))

        legacy_compat = legacy_registry.list_compatible("parity_test_hash_123")
        facade_compat = facade_registry.list_compatible("parity_test_hash_123")

        assert len(legacy_compat) == len(facade_compat)

    def test_parity_resolve_latest(
        self,
        legacy_registry: Any,
        facade_registry: Any,
        sample_onnx_model: tuple[Path, str],
        create_manifest: Any,
    ) -> None:
        """Verify resolve_latest returns identical model."""
        legacy_model = copy_model_to_registry(sample_onnx_model[0], legacy_registry)
        facade_model = copy_model_to_registry(sample_onnx_model[0], facade_registry)

        for registry, model_path in [(legacy_registry, legacy_model), (facade_registry, facade_model)]:
            registry.register_model(model_path, create_manifest(model_id="latest_1", version="1.0.0"))
            registry.register_model(model_path, create_manifest(model_id="latest_2", version="1.1.0"))
            registry.register_model(model_path, create_manifest(model_id="latest_3", version="1.0.5"))

        legacy_latest = legacy_registry.resolve_latest(
            role=ModelRole.INFERENCE,
            architecture="XGBoost",
            schema_hash="parity_test_hash_123",
        )
        facade_latest = facade_registry.resolve_latest(
            role=ModelRole.INFERENCE,
            architecture="XGBoost",
            schema_hash="parity_test_hash_123",
        )

        assert legacy_latest is not None
        assert facade_latest is not None
        assert legacy_latest.manifest.version == facade_latest.manifest.version == "1.1.0"
