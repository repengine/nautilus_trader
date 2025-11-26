#!/usr/bin/env python3

"""
Unit tests for DeploymentManagerComponent.

Tests cover:
- Basic deployment operations (deploy, rollback, retire, hot reload)
- Canary deployments (start, update metrics, evaluate, promote)
- Gradual rollouts (start, status, advance stage)
- Error conditions and edge cases

Coverage target: 90%
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ml.registry.base import (
    DataRequirements,
    DeploymentStatus,
    ModelInfo,
    ModelManifest,
    ModelRole,
)
from ml.registry.common.deployment_manager import DeploymentManagerComponent
from ml.registry.common.model_persistence import ModelPersistenceComponent
from ml.registry.dataclasses import CanaryConfig, CanaryDeployment, RolloutPlan
from ml.registry.persistence import BackendType, PersistenceConfig


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def persistence_component(tmp_path: Path) -> ModelPersistenceComponent:
    """
    ModelPersistenceComponent with JSON backend for unit tests.

    Uses temporary directory to avoid side effects.
    """
    config = PersistenceConfig(
        backend=BackendType.JSON,
        json_path=tmp_path,
    )
    component = ModelPersistenceComponent(config, tmp_path)
    component.load_registry()
    return component


@pytest.fixture
def deployment_manager(
    persistence_component: ModelPersistenceComponent,
) -> DeploymentManagerComponent:
    """
    DeploymentManagerComponent with test persistence.
    """
    return DeploymentManagerComponent(persistence_component)


@pytest.fixture
def sample_model_info(tmp_path: Path) -> ModelInfo:
    """
    Create a sample ModelInfo for testing.
    """
    model_file = tmp_path / "test_model.onnx"
    model_file.write_bytes(b"test model content")

    manifest = ModelManifest(
        model_id="model_1",
        role=ModelRole.INFERENCE,
        data_requirements=DataRequirements.L1_ONLY,
        architecture="test_arch",
        feature_schema={"input": "float32"},
        feature_schema_hash="hash_123",
        version="1.0.0",
        created_at=time.time(),
        last_modified=time.time(),
        performance_metrics={"accuracy": 0.95},
    )

    return ModelInfo(
        manifest=manifest,
        model_path=model_file,
        deployment_status=DeploymentStatus.INACTIVE,
        deployed_to=[],
        performance_history=[{"accuracy": 0.95}],
        metadata={},
    )


@pytest.fixture
def registered_models(
    persistence_component: ModelPersistenceComponent,
    tmp_path: Path,
) -> dict[str, ModelInfo]:
    """
    Pre-registered models for deployment/comparison tests.

    Creates 3 models with different versions and performance metrics.
    """
    models: dict[str, ModelInfo] = {}
    for i, version in enumerate(["1.0.0", "1.0.1", "1.1.0"]):
        model_file = tmp_path / f"model_{i}.onnx"
        model_file.write_bytes(f"model content {i}".encode())

        manifest = ModelManifest(
            model_id=f"model_{i}",
            role=ModelRole.INFERENCE,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="test_arch",
            feature_schema={"input": "float32"},
            feature_schema_hash="hash_123",
            version=version,
            created_at=time.time(),
            last_modified=time.time(),
            performance_metrics={"accuracy": 0.9 + i * 0.01},
        )

        model_info = ModelInfo(
            manifest=manifest,
            model_path=model_file,
            deployment_status=DeploymentStatus.INACTIVE,
            deployed_to=[],
            performance_history=[{"accuracy": 0.9 + i * 0.01}],
            metadata={},
        )

        models[manifest.model_id] = model_info
        persistence_component.set_model(manifest.model_id, model_info)

    persistence_component.save_registry(immediate=True)
    return models


# ============================================================================
# Happy Path Tests - Basic Deployment Operations
# ============================================================================


class TestDeployModel:
    """Tests for deploy_model method."""

    def test_deploy_model_success(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """Verify successful model deployment."""
        model_id = "model_0"
        target = "ml_signal_actor"

        result = deployment_manager.deploy_model(model_id, target)

        assert result is True
        model_info = persistence_component.get_model(model_id)
        assert model_info is not None
        assert model_info.deployment_status == DeploymentStatus.ACTIVE
        assert target in model_info.deployed_to
        assert persistence_component.deployments[target] == [model_id]

    def test_deploy_model_with_config(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """Verify deployment configuration stored in metadata."""
        model_id = "model_0"
        target = "ml_signal_actor"
        config = {"traffic_percentage": 100.0, "timeout_ms": 500}

        result = deployment_manager.deploy_model(model_id, target, config=config)

        assert result is True
        model_info = persistence_component.get_model(model_id)
        assert model_info is not None
        assert model_info.metadata["deployment_config"] == config

    def test_deploy_model_not_found(
        self,
        deployment_manager: DeploymentManagerComponent,
    ) -> None:
        """Verify handling of non-existent model."""
        result = deployment_manager.deploy_model("nonexistent_model", "target")

        assert result is False


class TestRollback:
    """Tests for rollback method."""

    def test_rollback_success(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """Verify rollback to previous model."""
        # First deploy model_1
        deployment_manager.deploy_model("model_1", "ml_signal_actor")

        # Then rollback to model_0
        result = deployment_manager.rollback("ml_signal_actor", "model_0")

        assert result is True

        # Check previous model is inactive
        model_1 = persistence_component.get_model("model_1")
        assert model_1 is not None
        assert model_1.deployment_status == DeploymentStatus.INACTIVE
        assert "ml_signal_actor" not in model_1.deployed_to

        # Check rollback model is active
        model_0 = persistence_component.get_model("model_0")
        assert model_0 is not None
        assert model_0.deployment_status == DeploymentStatus.ACTIVE
        assert "ml_signal_actor" in model_0.deployed_to

        # Check deployments tracking
        assert persistence_component.deployments["ml_signal_actor"] == ["model_0"]

    def test_rollback_model_not_found(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """Verify handling of rollback to non-existent model."""
        deployment_manager.deploy_model("model_0", "ml_signal_actor")

        result = deployment_manager.rollback("ml_signal_actor", "nonexistent")

        assert result is False


class TestRetireModel:
    """Tests for retire_model method."""

    def test_retire_model_success(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """Verify model retirement."""
        # First deploy model
        deployment_manager.deploy_model("model_0", "ml_signal_actor")
        deployment_manager.deploy_model("model_0", "ml_trading_strategy")

        # Retire model
        result = deployment_manager.retire_model("model_0")

        assert result is True
        model_info = persistence_component.get_model("model_0")
        assert model_info is not None
        assert model_info.deployment_status == DeploymentStatus.RETIRED
        assert len(model_info.deployed_to) == 0

    def test_retire_model_not_found(
        self,
        deployment_manager: DeploymentManagerComponent,
    ) -> None:
        """Verify handling of non-existent model retirement."""
        result = deployment_manager.retire_model("nonexistent")

        assert result is False


class TestHotReloadModel:
    """Tests for hot_reload_model method."""

    def test_hot_reload_model_success(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """Verify hot reload swaps models atomically."""
        target = "ml_signal_actor"

        # Deploy model_0
        deployment_manager.deploy_model("model_0", target)

        # Hot reload with model_1
        result = deployment_manager.hot_reload_model(target, "model_1")

        assert result is True

        # Old model should be retired
        model_0 = persistence_component.get_model("model_0")
        assert model_0 is not None
        assert model_0.deployment_status == DeploymentStatus.RETIRED

        # New model should be active
        model_1 = persistence_component.get_model("model_1")
        assert model_1 is not None
        assert model_1.deployment_status == DeploymentStatus.ACTIVE
        assert target in model_1.deployed_to

    def test_hot_reload_feature_schema_mismatch(
        self,
        deployment_manager: DeploymentManagerComponent,
        persistence_component: ModelPersistenceComponent,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify warning logged for schema mismatch during hot reload."""
        # Create two models with different schema hashes
        for i, schema_hash in enumerate(["hash_A", "hash_B"]):
            model_file = tmp_path / f"schema_model_{i}.onnx"
            model_file.write_bytes(f"model {i}".encode())

            manifest = ModelManifest(
                model_id=f"schema_model_{i}",
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="test_arch",
                feature_schema={"input": "float32"},
                feature_schema_hash=schema_hash,
                version="1.0.0",
                created_at=time.time(),
                last_modified=time.time(),
            )

            model_info = ModelInfo(
                manifest=manifest,
                model_path=model_file,
                deployment_status=DeploymentStatus.INACTIVE,
                deployed_to=[],
            )
            persistence_component.set_model(manifest.model_id, model_info)

        persistence_component.save_registry(immediate=True)

        # Deploy first model
        deployment_manager.deploy_model("schema_model_0", "test_target")

        # Hot reload with mismatched schema
        import logging

        with caplog.at_level(logging.WARNING):
            result = deployment_manager.hot_reload_model("test_target", "schema_model_1")

        assert result is True
        assert "schema mismatch" in caplog.text.lower()

    def test_hot_reload_no_current_model(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """Verify hot reload when no model currently deployed."""
        target = "empty_target"

        # Hot reload to empty target
        result = deployment_manager.hot_reload_model(target, "model_0")

        assert result is True
        model_info = persistence_component.get_model("model_0")
        assert model_info is not None
        assert model_info.deployment_status == DeploymentStatus.ACTIVE
        assert target in model_info.deployed_to


# ============================================================================
# Happy Path Tests - Canary Deployment Operations
# ============================================================================


class TestCanaryDeployment:
    """Tests for canary deployment operations."""

    def test_start_canary_deployment(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """Verify canary deployment creation."""
        config = CanaryConfig(
            traffic_percentage=5.0,
            success_metric="accuracy",
            min_samples=100,
        )

        deployment_id = deployment_manager.start_canary_deployment(
            model_id="model_0",
            target="ml_signal_actor",
            config=config,
        )

        assert deployment_id.startswith("canary_")
        assert "model_0" in deployment_id

        # Check canary was stored
        canary = deployment_manager.get_canary_deployment(deployment_id)
        assert canary is not None
        assert canary.status == "active"
        assert canary.config == config

        # Check model status updated
        model_info = persistence_component.get_model("model_0")
        assert model_info is not None
        assert model_info.deployment_status == DeploymentStatus.TESTING

    def test_start_canary_deployment_with_baseline(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """Verify canary deployment with explicit baseline model."""
        # Deploy baseline model first
        deployment_manager.deploy_model("model_0", "ml_signal_actor")

        config = CanaryConfig(
            traffic_percentage=5.0,
            success_metric="accuracy",
            min_samples=100,
        )

        deployment_id = deployment_manager.start_canary_deployment(
            model_id="model_1",
            target="ml_signal_actor",
            config=config,
            baseline_model_id="model_0",
        )

        canary = deployment_manager.get_canary_deployment(deployment_id)
        assert canary is not None
        assert canary.baseline_model_id == "model_0"
        assert canary.baseline_performance == 0.9  # model_0's accuracy

    def test_get_canary_deployment_not_found(
        self,
        deployment_manager: DeploymentManagerComponent,
    ) -> None:
        """Verify handling of non-existent canary deployment."""
        result = deployment_manager.get_canary_deployment("nonexistent")

        assert result is None

    def test_update_canary_metrics(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """Verify canary metric recording."""
        config = CanaryConfig(min_samples=10)
        deployment_id = deployment_manager.start_canary_deployment(
            model_id="model_0",
            target="ml_signal_actor",
            config=config,
        )

        # Record some metrics
        deployment_manager.update_canary_metrics(
            deployment_id, metric_value=0.95, latency_ms=2.5
        )
        deployment_manager.update_canary_metrics(
            deployment_id, metric_value=0.92, latency_ms=3.0, error_occurred=False
        )
        deployment_manager.update_canary_metrics(
            deployment_id, metric_value=0.0, error_occurred=True
        )

        canary = deployment_manager.get_canary_deployment(deployment_id)
        assert canary is not None
        assert canary.metrics["sample_count"] == 3
        assert canary.metrics["success_count"] == 2
        assert canary.metrics["error_count"] == 1
        assert len(canary.metrics["metric_values"]) == 2
        assert len(canary.metrics["latency_values"]) == 2

    def test_evaluate_canary_insufficient_samples(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """Verify canary not promoted with insufficient data."""
        config = CanaryConfig(min_samples=100)
        deployment_id = deployment_manager.start_canary_deployment(
            model_id="model_0",
            target="ml_signal_actor",
            config=config,
        )

        # Add only a few samples
        for _ in range(10):
            deployment_manager.update_canary_metrics(deployment_id, metric_value=0.95)

        should_promote, reason = deployment_manager.evaluate_canary(deployment_id)

        assert should_promote is False
        assert reason == "insufficient_samples"

    def test_evaluate_canary_for_rollback_high_error_rate(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """Verify rollback triggered by high error rate."""
        config = CanaryConfig(min_samples=30, error_rate_threshold=0.05)
        deployment_id = deployment_manager.start_canary_deployment(
            model_id="model_0",
            target="ml_signal_actor",
            config=config,
        )

        # Add samples with high error rate (>5%)
        for _ in range(25):
            deployment_manager.update_canary_metrics(deployment_id, metric_value=0.95)
        for _ in range(10):
            deployment_manager.update_canary_metrics(
                deployment_id, metric_value=0.0, error_occurred=True
            )

        should_rollback, reason = deployment_manager.evaluate_canary_for_rollback(
            deployment_id
        )

        assert should_rollback is True
        assert reason == "high_error_rate"

    def test_auto_promote_canary(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """Verify automatic canary promotion."""
        # Deploy baseline model first
        deployment_manager.deploy_model("model_0", "ml_signal_actor")

        config = CanaryConfig(min_samples=10)
        deployment_id = deployment_manager.start_canary_deployment(
            model_id="model_1",
            target="ml_signal_actor",
            config=config,
            baseline_model_id="model_0",
        )

        # Promote canary
        success = deployment_manager.auto_promote_canary(deployment_id)

        assert success is True

        # Check canary status
        canary = deployment_manager.get_canary_deployment(deployment_id)
        assert canary is not None
        assert canary.status == "promoted"

        # Check model deployment
        model_1 = persistence_component.get_model("model_1")
        assert model_1 is not None
        assert model_1.deployment_status == DeploymentStatus.ACTIVE

        # Check baseline was retired
        model_0 = persistence_component.get_model("model_0")
        assert model_0 is not None
        assert model_0.deployment_status == DeploymentStatus.RETIRED


# ============================================================================
# Happy Path Tests - Gradual Rollout Operations
# ============================================================================


class TestGradualRollout:
    """Tests for gradual rollout operations."""

    def test_start_gradual_rollout(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """Verify gradual rollout initialization."""
        stages = [0.1, 0.25, 0.5, 1.0]

        rollout_id = deployment_manager.start_gradual_rollout(
            current_model_id="model_0",
            new_model_id="model_1",
            target="ml_signal_actor",
            stages=stages,
            stage_duration_minutes=60,
        )

        assert rollout_id.startswith("rollout_")

        status = deployment_manager.get_rollout_status(rollout_id)
        assert status is not None
        assert status["current_stage"] == 0
        assert status["traffic_split"] == 0.1
        assert status["stages"] == stages

    def test_get_rollout_status_not_found(
        self,
        deployment_manager: DeploymentManagerComponent,
    ) -> None:
        """Verify handling of non-existent rollout."""
        result = deployment_manager.get_rollout_status("nonexistent")

        assert result is None

    def test_advance_rollout_stage(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """Verify rollout stage advancement."""
        stages = [0.1, 0.25, 0.5, 1.0]

        rollout_id = deployment_manager.start_gradual_rollout(
            current_model_id="model_0",
            new_model_id="model_1",
            target="ml_signal_actor",
            stages=stages,
            stage_duration_minutes=60,
        )

        # Advance to next stage
        result = deployment_manager.advance_rollout_stage(rollout_id)

        assert result is True

        status = deployment_manager.get_rollout_status(rollout_id)
        assert status is not None
        assert status["current_stage"] == 1
        assert status["traffic_split"] == 0.25

    def test_advance_rollout_stage_complete(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """Verify advance returns False when rollout is complete."""
        stages = [0.5, 1.0]

        rollout_id = deployment_manager.start_gradual_rollout(
            current_model_id="model_0",
            new_model_id="model_1",
            target="ml_signal_actor",
            stages=stages,
            stage_duration_minutes=60,
        )

        # Advance through all stages
        deployment_manager.advance_rollout_stage(rollout_id)  # stage 1

        # Try to advance beyond final stage
        result = deployment_manager.advance_rollout_stage(rollout_id)

        assert result is False


# ============================================================================
# Error Condition Tests
# ============================================================================


class TestErrorConditions:
    """Tests for error conditions and edge cases."""

    def test_start_canary_deployment_model_not_found(
        self,
        deployment_manager: DeploymentManagerComponent,
    ) -> None:
        """Verify ValueError raised for non-existent model."""
        config = CanaryConfig()

        with pytest.raises(ValueError, match="Model nonexistent not found"):
            deployment_manager.start_canary_deployment(
                model_id="nonexistent",
                target="ml_signal_actor",
                config=config,
            )

    def test_start_gradual_rollout_model_not_found(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
    ) -> None:
        """Verify ValueError raised when models not found."""
        with pytest.raises(ValueError, match="One or both models not found"):
            deployment_manager.start_gradual_rollout(
                current_model_id="nonexistent",
                new_model_id="model_0",
                target="ml_signal_actor",
                stages=[0.5, 1.0],
                stage_duration_minutes=60,
            )

    def test_evaluate_canary_not_found(
        self,
        deployment_manager: DeploymentManagerComponent,
    ) -> None:
        """Verify handling of non-existent canary evaluation."""
        should_promote, reason = deployment_manager.evaluate_canary("nonexistent")

        assert should_promote is False
        assert reason == "deployment_not_found"

    def test_evaluate_canary_for_rollback_not_found(
        self,
        deployment_manager: DeploymentManagerComponent,
    ) -> None:
        """Verify handling of non-existent canary rollback evaluation."""
        should_rollback, reason = deployment_manager.evaluate_canary_for_rollback(
            "nonexistent"
        )

        assert should_rollback is False
        assert reason == "deployment_not_found"

    def test_auto_promote_canary_not_found(
        self,
        deployment_manager: DeploymentManagerComponent,
    ) -> None:
        """Verify auto_promote returns False for non-existent canary."""
        result = deployment_manager.auto_promote_canary("nonexistent")

        assert result is False

    def test_advance_rollout_stage_not_found(
        self,
        deployment_manager: DeploymentManagerComponent,
    ) -> None:
        """Verify advance_rollout_stage returns False for non-existent rollout."""
        result = deployment_manager.advance_rollout_stage("nonexistent")

        assert result is False


# ============================================================================
# Integration Tests - Component Interaction
# ============================================================================


class TestComponentInteraction:
    """Tests for component interaction scenarios."""

    def test_full_deployment_lifecycle(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """Test complete deployment lifecycle: deploy -> update -> rollback."""
        target = "ml_signal_actor"

        # 1. Deploy model_0
        assert deployment_manager.deploy_model("model_0", target) is True
        model_0 = persistence_component.get_model("model_0")
        assert model_0 is not None
        assert model_0.deployment_status == DeploymentStatus.ACTIVE

        # 2. Deploy model_1 (replaces model_0)
        assert deployment_manager.deploy_model("model_1", target) is True
        model_1 = persistence_component.get_model("model_1")
        assert model_1 is not None
        assert model_1.deployment_status == DeploymentStatus.ACTIVE

        # 3. Rollback to model_0
        assert deployment_manager.rollback(target, "model_0") is True
        model_0 = persistence_component.get_model("model_0")
        model_1 = persistence_component.get_model("model_1")
        assert model_0 is not None
        assert model_1 is not None
        assert model_0.deployment_status == DeploymentStatus.ACTIVE
        assert model_1.deployment_status == DeploymentStatus.INACTIVE

        # 4. Retire model_0
        assert deployment_manager.retire_model("model_0") is True
        model_0 = persistence_component.get_model("model_0")
        assert model_0 is not None
        assert model_0.deployment_status == DeploymentStatus.RETIRED

    def test_canary_to_full_promotion_flow(
        self,
        deployment_manager: DeploymentManagerComponent,
        registered_models: dict[str, ModelInfo],
        persistence_component: ModelPersistenceComponent,
    ) -> None:
        """Test canary deployment to full promotion flow."""
        target = "ml_signal_actor"

        # 1. Deploy baseline model
        deployment_manager.deploy_model("model_0", target)

        # 2. Start canary with new model
        config = CanaryConfig(min_samples=5)
        deployment_id = deployment_manager.start_canary_deployment(
            model_id="model_1",
            target=target,
            config=config,
            baseline_model_id="model_0",
        )

        # 3. Collect metrics
        for _ in range(10):
            deployment_manager.update_canary_metrics(deployment_id, metric_value=0.95)

        # 4. Promote canary
        assert deployment_manager.auto_promote_canary(deployment_id) is True

        # 5. Verify final state
        model_0 = persistence_component.get_model("model_0")
        model_1 = persistence_component.get_model("model_1")
        assert model_0 is not None
        assert model_1 is not None
        assert model_0.deployment_status == DeploymentStatus.RETIRED
        assert model_1.deployment_status == DeploymentStatus.ACTIVE
