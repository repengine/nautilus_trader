#!/usr/bin/env python3

"""
Test contracts for model registry module.

This ensures the registry properly orchestrates all ML components:
- Model registration with metadata
- Version management
- Deployment tracking
- Performance monitoring
- A/B testing support
- Rollback functionality
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelInfo
from ml.registry.deployment import ModelDeploymentManager
from ml.registry.local_registry import LocalModelRegistry


class TestModelRegistryContracts:
    """Test that model registry correctly orchestrates all ML components."""

    def test_model_info_structure(self) -> None:
        """Test ModelInfo data structure contains required fields."""
        model_info = ModelInfo(
            model_id="test_model_v1",
            model_path=Path("/models/test.onnx"),
            version="1.0.0",
            metadata={
                "features": ["sma_10", "rsi_14"],
                "training_metrics": {"accuracy": 0.95},
            },
            deployment_status=DeploymentStatus.INACTIVE,
            deployed_to=[],
            created_at=time.time(),
            last_modified=time.time(),
        )

        assert model_info.model_id == "test_model_v1"
        assert model_info.version == "1.0.0"
        assert model_info.deployment_status == DeploymentStatus.INACTIVE
        assert "features" in model_info.metadata
        assert model_info.deployed_to == []

    def test_local_registry_initialization(self) -> None:
        """Test LocalModelRegistry initializes correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = LocalModelRegistry(registry_path)

            assert registry.registry_path == registry_path
            assert (registry_path / "registry.json").exists()
            assert registry.get_active_models() == []

    def test_register_model_creates_entry(self) -> None:
        """Test registering a model creates proper registry entry."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            model_path = registry_path / "model.onnx"
            model_path.touch()  # Create dummy model file

            registry = LocalModelRegistry(registry_path)

            metadata = {
                "features": ["sma_10", "rsi_14", "volume"],
                "training_metrics": {"accuracy": 0.92, "auc": 0.88},
                "trainer_class": "XGBoostTrainer",
            }

            model_id = registry.register_model(
                model_path=model_path,
                metadata=metadata,
                version="1.0.0"
            )

            assert model_id is not None
            assert model_id.startswith("model_")

            # Check registry was updated
            models = registry.get_all_models()
            assert len(models) == 1
            assert models[0].model_id == model_id
            assert models[0].metadata["features"] == ["sma_10", "rsi_14", "volume"]

    def test_deploy_model_updates_status(self) -> None:
        """Test deploying a model updates its status and target."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            model_path = registry_path / "model.onnx"
            model_path.touch()

            registry = LocalModelRegistry(registry_path)

            # Register model
            model_id = registry.register_model(
                model_path=model_path,
                metadata={"features": ["sma_10"]},
            )

            # Deploy model
            success = registry.deploy_model(
                model_id=model_id,
                target="ml_signal_actor",
                config={"instrument": "EURUSD"}
            )

            assert success is True

            # Check deployment status
            active_models = registry.get_active_models()
            assert len(active_models) == 1
            assert active_models[0].model_id == model_id
            assert active_models[0].deployment_status == DeploymentStatus.ACTIVE
            assert "ml_signal_actor" in active_models[0].deployed_to

    def test_track_performance_stores_metrics(self) -> None:
        """Test performance tracking stores metrics correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            model_path = registry_path / "model.onnx"
            model_path.touch()

            registry = LocalModelRegistry(registry_path)

            # Register and deploy model
            model_id = registry.register_model(
                model_path=model_path,
                metadata={"features": ["sma_10"]},
            )
            registry.deploy_model(model_id, "ml_signal_actor")

            # Track performance
            metrics = {
                "live_accuracy": 0.91,
                "pnl": 1500.0,
                "trades": 25,
                "timestamp": time.time(),
            }
            registry.track_performance(model_id, metrics)

            # Get performance history
            history = registry.get_performance_history(model_id)
            assert len(history) == 1
            assert history[0]["live_accuracy"] == 0.91
            assert history[0]["pnl"] == 1500.0

    def test_rollback_model_restores_previous(self) -> None:
        """Test rollback functionality restores previous model version."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            model_v1_path = registry_path / "model_v1.onnx"
            model_v2_path = registry_path / "model_v2.onnx"
            model_v1_path.touch()
            model_v2_path.touch()

            registry = LocalModelRegistry(registry_path)

            # Register two versions
            model_id_v1 = registry.register_model(
                model_path=model_v1_path,
                metadata={"features": ["sma_10"]},
                version="1.0.0"
            )

            model_id_v2 = registry.register_model(
                model_path=model_v2_path,
                metadata={"features": ["sma_10", "rsi_14"]},
                version="2.0.0"
            )

            # Deploy v2
            registry.deploy_model(model_id_v2, "ml_signal_actor")

            # Rollback to v1
            success = registry.rollback(
                target="ml_signal_actor",
                to_model_id=model_id_v1
            )

            assert success is True

            # Check v1 is now active
            active_models = registry.get_active_models()
            assert len(active_models) == 1
            assert active_models[0].model_id == model_id_v1
            assert active_models[0].deployment_status == DeploymentStatus.ACTIVE

    def test_ab_test_configuration(self) -> None:
        """Test A/B test configuration for multiple models."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            model_a_path = registry_path / "model_a.onnx"
            model_b_path = registry_path / "model_b.onnx"
            model_a_path.touch()
            model_b_path.touch()

            registry = LocalModelRegistry(registry_path)

            # Register two models
            model_a = registry.register_model(
                model_path=model_a_path,
                metadata={"features": ["sma_10"]},
                version="1.0.0"
            )

            model_b = registry.register_model(
                model_path=model_b_path,
                metadata={"features": ["sma_10", "rsi_14"]},
                version="2.0.0"
            )

            # Configure A/B test
            ab_config = registry.configure_ab_test(
                models=[model_a, model_b],
                split_ratio=0.5,
                duration_hours=24,
                target="ml_signal_actor"
            )

            assert ab_config is not None
            assert ab_config["model_a"] == model_a
            assert ab_config["model_b"] == model_b
            assert ab_config["split_ratio"] == 0.5
            assert ab_config["duration_hours"] == 24

            # Both models should be in testing status
            all_models = registry.get_all_models()
            testing_models = [m for m in all_models if m.deployment_status == DeploymentStatus.TESTING]
            assert len(testing_models) == 2

    def test_thread_safety_concurrent_operations(self) -> None:
        """Test registry is thread-safe for concurrent operations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = LocalModelRegistry(registry_path)

            results = []
            errors = []

            def register_model(i: int) -> None:
                try:
                    model_path = registry_path / f"model_{i}.onnx"
                    model_path.touch()

                    model_id = registry.register_model(
                        model_path=model_path,
                        metadata={"index": i},
                        version=f"1.0.{i}"
                    )
                    results.append(model_id)
                except Exception as e:
                    errors.append(e)

            # Create multiple threads
            threads = []
            for i in range(10):
                thread = threading.Thread(target=register_model, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Check results
            assert len(errors) == 0
            assert len(results) == 10
            assert len(set(results)) == 10  # All unique IDs

            # Verify registry state
            all_models = registry.get_all_models()
            assert len(all_models) == 10

    def test_model_lifecycle_tracking(self) -> None:
        """Test complete model lifecycle from registration to retirement."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            model_path = registry_path / "model.onnx"
            model_path.touch()

            registry = LocalModelRegistry(registry_path)

            # 1. Register model
            model_id = registry.register_model(
                model_path=model_path,
                metadata={"features": ["sma_10"]},
                version="1.0.0"
            )

            model = registry.get_model(model_id)
            assert model is not None
            assert model.deployment_status == DeploymentStatus.INACTIVE

            # 2. Deploy model
            registry.deploy_model(model_id, "ml_signal_actor")
            model = registry.get_model(model_id)
            assert model is not None
            assert model.deployment_status == DeploymentStatus.ACTIVE

            # 3. Track performance
            for i in range(5):
                registry.track_performance(model_id, {
                    "accuracy": 0.90 + i * 0.01,
                    "timestamp": time.time() + i
                })

            history = registry.get_performance_history(model_id)
            assert len(history) == 5

            # 4. Retire model
            registry.retire_model(model_id)
            model = registry.get_model(model_id)
            assert model is not None
            assert model.deployment_status == DeploymentStatus.RETIRED

            # Should not be in active models
            active_models = registry.get_active_models()
            assert len(active_models) == 0

    def test_deployment_manager_integration(self) -> None:
        """Test ModelDeploymentManager coordinates with registry."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            model_path = registry_path / "model.onnx"
            model_path.touch()

            registry = LocalModelRegistry(registry_path)
            deployment_manager = ModelDeploymentManager(registry)

            # Register model
            model_id = registry.register_model(
                model_path=model_path,
                metadata={"features": ["sma_10"]},
            )

            # Deploy through manager
            deployment_config = {
                "target": "ml_signal_actor",
                "instruments": ["EURUSD", "GBPUSD"],
                "max_positions": 5,
            }

            deployment_id = deployment_manager.deploy(
                model_id=model_id,
                config=deployment_config
            )

            assert deployment_id is not None

            # Check deployment status
            assert deployment_id is not None  # Type guard
            status = deployment_manager.get_deployment_status(deployment_id)
            assert status is not None
            assert status["model_id"] == model_id
            assert status["is_active"] is True
            assert status["config"] == deployment_config

    def test_hot_reload_capability(self) -> None:
        """Test hot reload updates model without downtime."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            model_v1_path = registry_path / "model_v1.onnx"
            model_v2_path = registry_path / "model_v2.onnx"
            model_v1_path.touch()
            model_v2_path.touch()

            registry = LocalModelRegistry(registry_path)
            deployment_manager = ModelDeploymentManager(registry)

            # Deploy v1
            model_v1 = registry.register_model(
                model_path=model_v1_path,
                metadata={"features": ["sma_10"]},
                version="1.0.0"
            )

            deployment_id = deployment_manager.deploy(
                model_id=model_v1,
                config={"target": "ml_signal_actor"}
            )

            # Register v2
            model_v2 = registry.register_model(
                model_path=model_v2_path,
                metadata={"features": ["sma_10", "rsi_14"]},
                version="2.0.0"
            )

            # Hot reload to v2
            assert deployment_id is not None  # Type guard
            success = deployment_manager.hot_reload(
                deployment_id=deployment_id,
                new_model_id=model_v2
            )

            assert success is True

            # Check v2 is active
            assert deployment_id is not None  # Type guard
            status = deployment_manager.get_deployment_status(deployment_id)
            assert status is not None
            assert status["model_id"] == model_v2
            assert status["is_active"] is True

    def test_model_comparison(self) -> None:
        """Test comparing performance between models."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            model_a_path = registry_path / "model_a.onnx"
            model_b_path = registry_path / "model_b.onnx"
            model_a_path.touch()
            model_b_path.touch()

            registry = LocalModelRegistry(registry_path)

            # Register two models
            model_a = registry.register_model(
                model_path=model_a_path,
                metadata={"features": ["sma_10"]},
            )

            model_b = registry.register_model(
                model_path=model_b_path,
                metadata={"features": ["sma_10", "rsi_14"]},
            )

            # Track performance for both
            registry.track_performance(model_a, {
                "accuracy": 0.91,
                "pnl": 1000.0,
                "sharpe": 1.5
            })

            registry.track_performance(model_b, {
                "accuracy": 0.93,
                "pnl": 1500.0,
                "sharpe": 1.8
            })

            # Compare models
            comparison = registry.compare_models([model_a, model_b], metric="accuracy")

            assert comparison is not None
            assert comparison["best_model"] == model_b
            assert comparison["rankings"][0]["model_id"] == model_b
            assert comparison["rankings"][0]["accuracy"] == 0.93
