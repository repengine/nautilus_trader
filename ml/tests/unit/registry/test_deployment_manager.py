#!/usr/bin/env python3

"""
Tests for LocalModelRegistry deployment functionality.

These tests verify the registry correctly handles model deployment,
hot reload, and lifecycle management.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.local_registry import LocalModelRegistry


class TestRegistryDeployment:
    """Test LocalModelRegistry deployment functionality."""

    def test_registry_basic_deploy(self) -> None:
        """Test LocalModelRegistry deployment functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            model_path = registry_path / "model.onnx"
            model_path.write_bytes(b"ONNX_MODEL")

            registry = LocalModelRegistry(registry_path)

            # Create manifest
            manifest = ModelManifest(
                model_id="deploy_test",
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="TestModel",
                feature_schema={"sma_10": "float32"},
                feature_schema_hash="test_hash",
                version="1.0.0",
                created_at=time.time(),
                last_modified=time.time(),
            )

            # Register model
            model_id = registry.register_model(
                model_path=model_path,
                manifest=manifest,
            )

            # Deploy directly through registry
            deployment_config = {
                "instruments": ["EURUSD", "GBPUSD"],
                "max_positions": 5,
            }

            success = registry.deploy_model(
                model_id=model_id,
                target="ml_signal_actor",
                config=deployment_config
            )

            assert success is True

            # Check deployment status
            model_info = registry.get_model(model_id)
            assert model_info is not None
            assert model_info.deployment_status == DeploymentStatus.ACTIVE
            assert "ml_signal_actor" in model_info.deployed_to
            assert model_info.metadata.get("deployment_config") == deployment_config

    def test_registry_hot_reload(self) -> None:
        """Test hot reload updates model without downtime."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            model_v1_path = registry_path / "model_v1.onnx"
            model_v2_path = registry_path / "model_v2.onnx"
            model_v1_path.write_bytes(b"ONNX_V1")
            model_v2_path.write_bytes(b"ONNX_V2")

            registry = LocalModelRegistry(registry_path)

            # Deploy v1
            manifest_v1 = ModelManifest(
                model_id="model_v1",
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="TestModel",
                feature_schema={"sma_10": "float32"},
                feature_schema_hash="v1_hash",
                version="1.0.0",
                created_at=time.time(),
                last_modified=time.time(),
            )

            model_v1 = registry.register_model(
                model_path=model_v1_path,
                manifest=manifest_v1,
            )

            # Deploy v1
            success = registry.deploy_model(
                model_id=model_v1,
                target="ml_signal_actor"
            )
            assert success is True

            # Register v2
            manifest_v2 = ModelManifest(
                model_id="model_v2",
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="TestModel",
                feature_schema={"sma_10": "float32", "rsi_14": "float32"},
                feature_schema_hash="v2_hash",
                version="2.0.0",
                created_at=time.time(),
                last_modified=time.time(),
            )

            model_v2 = registry.register_model(
                model_path=model_v2_path,
                manifest=manifest_v2,
            )

            # Hot reload to v2
            success = registry.hot_reload_model(
                target="ml_signal_actor",
                new_model_id=model_v2
            )

            assert success is True

            # Check v2 is active and v1 is retired
            model_v2_info = registry.get_model(model_v2)
            assert model_v2_info is not None
            assert model_v2_info.deployment_status == DeploymentStatus.ACTIVE
            assert "ml_signal_actor" in model_v2_info.deployed_to

            model_v1_info = registry.get_model(model_v1)
            assert model_v1_info is not None
            assert model_v1_info.deployment_status == DeploymentStatus.RETIRED

    def test_registry_gradual_rollout(self) -> None:
        """Test gradual rollout functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            model_current_path = registry_path / "model_current.onnx"
            model_new_path = registry_path / "model_new.onnx"
            model_current_path.write_bytes(b"ONNX_CURRENT")
            model_new_path.write_bytes(b"ONNX_NEW")

            registry = LocalModelRegistry(registry_path)

            # Register and deploy current model
            manifest_current = ModelManifest(
                model_id="model_current",
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="TestModel",
                feature_schema={"sma_10": "float32"},
                feature_schema_hash="current_hash",
                version="1.0.0",
            )

            current_id = registry.register_model(
                model_path=model_current_path,
                manifest=manifest_current,
            )

            registry.deploy_model(current_id, "ml_signal_actor")

            # Register new model
            manifest_new = ModelManifest(
                model_id="model_new",
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="TestModel",
                feature_schema={"sma_10": "float32"},
                feature_schema_hash="current_hash",  # Same schema for compatibility
                version="2.0.0",
            )

            new_id = registry.register_model(
                model_path=model_new_path,
                manifest=manifest_new,
            )

            # Start gradual rollout
            rollout_id = registry.start_gradual_rollout(
                current_model_id=current_id,
                new_model_id=new_id,
                target="ml_signal_actor",
                stages=[0.1, 0.25, 0.5, 1.0],
                stage_duration_minutes=30,
            )

            assert rollout_id is not None
            assert rollout_id.startswith("rollout_")

            # Check rollout status
            status = registry.get_rollout_status(rollout_id)
            assert status is not None
            assert status["current_stage"] == 0
            assert status["traffic_split"] == 0.1

            # Advance rollout
            advanced = registry.advance_rollout_stage(rollout_id)
            assert advanced is True

            status = registry.get_rollout_status(rollout_id)
            assert status is not None
            assert status["current_stage"] == 1
            assert status["traffic_split"] == 0.25
