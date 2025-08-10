#!/usr/bin/env python3

"""
Integration tests showing how the registry orchestrates all ML components.

This demonstrates the complete flow from training to deployment to monitoring.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ml.registry import DeploymentStatus
from ml.registry import LocalModelRegistry
from ml.registry import ModelDeploymentManager


class TestRegistryIntegration:
    """Test registry integration with all ML components."""

    def test_training_to_deployment_flow(self) -> None:
        """Test complete flow from training to deployment."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = LocalModelRegistry(registry_path)
            deployment_manager = ModelDeploymentManager(registry)

            # Step 1: Training completes (Agent 3's domain)
            model_path = registry_path / "xgb_model.json"
            model_path.write_text('{"model": "xgboost"}')

            training_metadata = {
                "trainer_class": "XGBoostTrainer",
                "features": ["sma_10", "rsi_14", "volume"],
                "training_metrics": {
                    "accuracy": 0.92,
                    "auc": 0.88,
                    "f1_score": 0.85,
                },
                "model_type": "xgboost",
                "input_shape": [None, 3],
            }

            # Registry registers the model
            model_id = registry.register_model(
                model_path=model_path,
                metadata=training_metadata,
            )

            assert model_id is not None

            # Step 2: Deploy to Actor (Agent 1's domain)
            deployment_config = {
                "target": "ml_signal_actor",
                "instruments": ["EURUSD"],
                "confidence_threshold": 0.7,
            }

            deployment_id = deployment_manager.deploy(
                model_id=model_id,
                config=deployment_config,
            )

            assert deployment_id is not None

            # Verify model is active
            active_models = registry.get_active_models()
            assert len(active_models) == 1
            assert active_models[0].model_id == model_id

            # Step 3: Track performance (Strategy's feedback - Agent 4)
            performance_metrics = {
                "live_accuracy": 0.91,
                "pnl": 1500.0,
                "trades": 25,
                "model_id": model_id,  # Strategy tracks which model
            }

            registry.track_performance(model_id, performance_metrics)

            # Verify performance is tracked
            history = registry.get_performance_history(model_id)
            assert len(history) == 1
            assert history[0]["pnl"] == 1500.0

    def test_multi_model_ab_testing(self) -> None:
        """Test A/B testing orchestration for multiple models."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = LocalModelRegistry(registry_path)

            # Register two competing models
            model_a_path = registry_path / "model_a.onnx"
            model_b_path = registry_path / "model_b.onnx"
            model_a_path.write_text("model_a")
            model_b_path.write_text("model_b")

            # Model A: Current production model
            model_a = registry.register_model(
                model_path=model_a_path,
                metadata={
                    "features": ["sma_10", "rsi_14"],
                    "accuracy": 0.90,
                },
                version="1.0.0"
            )

            # Model B: New challenger model
            model_b = registry.register_model(
                model_path=model_b_path,
                metadata={
                    "features": ["sma_10", "rsi_14", "atr_20"],
                    "accuracy": 0.92,
                },
                version="2.0.0"
            )

            # Configure A/B test
            ab_config = registry.configure_ab_test(
                models=[model_a, model_b],
                split_ratio=0.3,  # 30% to model A, 70% to model B
                duration_hours=48,
                target="ml_signal_actor"
            )

            assert ab_config is not None
            assert ab_config["model_a"] == model_a
            assert ab_config["model_b"] == model_b

            # Both models should be in testing status
            model_a_info = registry.get_model(model_a)
            model_b_info = registry.get_model(model_b)

            assert model_a_info is not None
            assert model_b_info is not None
            assert model_a_info.deployment_status == DeploymentStatus.TESTING
            assert model_b_info.deployment_status == DeploymentStatus.TESTING

    def test_hot_reload_with_validation(self) -> None:
        """Test hot reload validates model compatibility."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = LocalModelRegistry(registry_path)
            deployment_manager = ModelDeploymentManager(registry)

            # Deploy v1
            model_v1_path = registry_path / "v1.json"
            model_v1_path.write_text("v1")

            model_v1 = registry.register_model(
                model_path=model_v1_path,
                metadata={
                    "features": ["sma_10", "rsi_14"],
                    "version": "1.0.0",
                }
            )

            deployment_id = deployment_manager.deploy(
                model_id=model_v1,
                config={"target": "ml_signal_actor"}
            )

            # Create v2 with additional features
            model_v2_path = registry_path / "v2.json"
            model_v2_path.write_text("v2")

            model_v2 = registry.register_model(
                model_path=model_v2_path,
                metadata={
                    "features": ["sma_10", "rsi_14", "volume"],
                    "version": "2.0.0",
                }
            )

            # Hot reload should succeed but warn about feature mismatch
            assert deployment_id is not None
            success = deployment_manager.hot_reload(
                deployment_id=deployment_id,
                new_model_id=model_v2
            )

            assert success is True

            # Verify v2 is now active
            model_v2_info = registry.get_model(model_v2)
            assert model_v2_info is not None
            assert model_v2_info.deployment_status == DeploymentStatus.ACTIVE

            # Verify v1 is retired
            model_v1_info = registry.get_model(model_v1)
            assert model_v1_info is not None
            assert model_v1_info.deployment_status == DeploymentStatus.RETIRED

    def test_rollback_on_performance_degradation(self) -> None:
        """Test automatic rollback when performance degrades."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = LocalModelRegistry(registry_path)

            # Deploy stable model
            stable_model_path = registry_path / "stable.onnx"
            stable_model_path.write_text("stable")

            stable_model = registry.register_model(
                model_path=stable_model_path,
                metadata={"features": ["sma_10"]},
            )

            registry.deploy_model(stable_model, "ml_signal_actor")

            # Track good performance
            registry.track_performance(stable_model, {
                "accuracy": 0.92,
                "pnl": 1000,
            })

            # Deploy new model
            new_model_path = registry_path / "new.onnx"
            new_model_path.write_text("new")

            new_model = registry.register_model(
                model_path=new_model_path,
                metadata={"features": ["sma_10", "rsi_14"]},
            )

            registry.deploy_model(new_model, "ml_signal_actor")

            # Track degraded performance
            registry.track_performance(new_model, {
                "accuracy": 0.75,  # Degraded
                "pnl": -500,  # Loss
            })

            # Compare performance
            comparison = registry.compare_models(
                model_ids=[stable_model, new_model],
                metric="pnl"
            )

            assert comparison is not None
            assert comparison["best_model"] == stable_model

            # Rollback to stable model
            success = registry.rollback(
                target="ml_signal_actor",
                to_model_id=stable_model
            )

            assert success is True

            # Verify stable model is active again
            stable_info = registry.get_model(stable_model)
            assert stable_info is not None
            assert stable_info.deployment_status == DeploymentStatus.ACTIVE

    def test_gradual_rollout_stages(self) -> None:
        """Test gradual rollout with multiple stages."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = LocalModelRegistry(registry_path)
            deployment_manager = ModelDeploymentManager(registry)

            # Create models
            current_path = registry_path / "current.onnx"
            new_path = registry_path / "new.onnx"
            current_path.write_text("current")
            new_path.write_text("new")

            current_model = registry.register_model(
                model_path=current_path,
                metadata={"name": "current"},
            )

            new_model = registry.register_model(
                model_path=new_path,
                metadata={"name": "new"},
            )

            # Deploy current model
            deployment_id = deployment_manager.deploy(
                model_id=current_model,
                config={"target": "ml_signal_actor"}
            )

            # Configure gradual rollout
            assert deployment_id is not None
            rollout_id = deployment_manager.gradual_rollout(
                deployment_id=deployment_id,
                new_model_id=new_model,
                stages=[0.1, 0.25, 0.5, 1.0],
                stage_duration_minutes=30,
            )

            assert rollout_id is not None
            assert rollout_id.startswith("rollout_")

            # Verify A/B test is configured for first stage
            all_models = registry.get_all_models()
            testing_models = [
                m for m in all_models
                if m.deployment_status == DeploymentStatus.TESTING
            ]

            # Both models should be in testing during rollout
            assert len(testing_models) == 2

    def test_model_metadata_preservation(self) -> None:
        """Test that model metadata is preserved through lifecycle."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = LocalModelRegistry(registry_path)

            # Register model with comprehensive metadata
            model_path = registry_path / "model.onnx"
            model_path.write_text("model")

            metadata = {
                # From training (Agent 3)
                "trainer_class": "XGBoostTrainer",
                "features": ["sma_10", "rsi_14", "volume"],
                "training_accuracy": 0.92,

                # For loading (Agent 2)
                "model_type": "xgboost",
                "input_shape": [None, 3],
                "output_shape": [None, 1],

                # For actors (Agent 1)
                "confidence_threshold": 0.7,
                "max_positions": 5,

                # For strategies (Agent 4)
                "target_instruments": ["EURUSD", "GBPUSD"],
                "aggregation_mode": "weighted_average",
            }

            model_id = registry.register_model(
                model_path=model_path,
                metadata=metadata,
            )

            # Retrieve and verify metadata
            model_info = registry.get_model(model_id)
            assert model_info is not None

            # All metadata should be preserved
            assert model_info.metadata["trainer_class"] == "XGBoostTrainer"
            assert model_info.metadata["features"] == ["sma_10", "rsi_14", "volume"]
            assert model_info.metadata["confidence_threshold"] == 0.7
            assert model_info.metadata["target_instruments"] == ["EURUSD", "GBPUSD"]

            # Deploy and verify metadata still accessible
            registry.deploy_model(model_id, "ml_signal_actor")

            active_models = registry.get_active_models()
            assert len(active_models) == 1
            assert active_models[0].metadata["features"] == ["sma_10", "rsi_14", "volume"]
