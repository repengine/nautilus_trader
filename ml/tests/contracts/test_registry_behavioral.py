#!/usr/bin/env python3
"""
Behavioral tests for model registry.

These tests verify BEHAVIORS, not implementation details. They use the new ModelManifest
API directly without helper functions.

"""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from ml.registry.base import DataRequirements
from ml.registry.base import DeploymentStatus
from ml.registry.base import ModelRole
from ml.registry import ModelManifest
from ml.registry import ModelRegistry
from ml.tests.builders import RegistryBuilder


@pytest.mark.skip(
    reason="Complex behavioral tests - disable during test reset for event-driven refactor",
)
@pytest.mark.flaky
@pytest.mark.slow
class TestRegistryBehaviors:
    """
    Test that the registry behaves correctly in real-world scenarios.
    """

    def test_thread_safety_concurrent_operations(self) -> None:
        """
        Test that registry is thread-safe for concurrent operations.

        This is a critical BEHAVIOR - the registry must handle concurrent
        access correctly in production.

        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = ModelRegistry(registry_path)

            results: dict[str, list[Any]] = {"registered": [], "errors": [], "deployed": []}
            lock = threading.Lock()

            def register_and_deploy(index: int) -> None:
                """
                Register and deploy a model concurrently.
                """
                try:
                    # Create model file
                    model_path = registry_path / f"model_{index}.onnx"
                    model_path.touch()

                    # Create manifest with unique features
                    feature_schema = {f"feature_{index}": "float32"}
                    schema_json = json.dumps(feature_schema, sort_keys=True)
                    schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()

                    manifest = RegistryBuilder.model_manifest(
                        model_id=f"concurrent_model_{index}",
                        architecture="ConcurrentTest",
                        feature_schema=feature_schema,
                        feature_schema_hash=schema_hash,
                        performance_metrics={"accuracy": 0.8 + index * 0.01},
                        version=f"{index}.0.0",
                    )

                    # Register model
                    model_id = registry.register_model(model_path, manifest)

                    with lock:
                        results["registered"].append(model_id)

                    # Try to deploy
                    if index % 2 == 0:  # Deploy every other model
                        success = registry.deploy_model(
                            model_id=model_id,
                            target=f"actor_{index % 3}",  # 3 different targets
                        )
                        if success:
                            with lock:
                                results["deployed"].append(model_id)

                except Exception as e:
                    with lock:
                        results["errors"].append(str(e))

            # Create multiple threads
            threads = []
            n_threads = 20
            for i in range(n_threads):
                thread = threading.Thread(target=register_and_deploy, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Verify results
            assert len(results["errors"]) == 0, f"Errors occurred: {results['errors']}"
            assert len(results["registered"]) == n_threads
            assert len(set(results["registered"])) == n_threads  # All unique

            # Verify registry state is consistent
            all_models = registry.get_all_models()
            assert len(all_models) == n_threads

            # Check deployed models
            active_models = registry.get_active_models()
            assert len(active_models) == len(results["deployed"])

    def test_rollback_restores_previous_model(self) -> None:
        """
        Test that rollback correctly restores a previous model version.

        This is a critical production BEHAVIOR for handling bad deployments.

        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = ModelRegistry(registry_path)

            # Create v1 - good model
            model_v1_path = registry_path / "model_v1.onnx"
            model_v1_path.write_text("model_v1_weights")

            manifest_v1 = RegistryBuilder.model_manifest(
                model_id="prod_model_v1",
                architecture="XGBoost",
                feature_schema={"close": "float32", "volume": "int64"},
                feature_schema_hash="v1_hash",
                performance_metrics={"accuracy": 0.92, "latency_ms": 2.5},
                version="1.0.0",
            )

            model_id_v1 = registry.register_model(model_v1_path, manifest_v1)
            registry.deploy_model(model_id_v1, "production")

            # Track good performance for v1
            registry.track_performance(
                model_id_v1,
                {
                    "live_accuracy": 0.91,
                    "pnl": 5000.0,
                    "trades": 100,
                },
            )

            # Create v2 - problematic model
            time.sleep(0.1)  # Ensure different timestamp
            model_v2_path = registry_path / "model_v2.onnx"
            model_v2_path.write_text("model_v2_weights")

            manifest_v2 = RegistryBuilder.model_manifest(
                model_id="prod_model_v2",
                architecture="XGBoost",
                feature_schema={"close": "float32", "volume": "int64", "rsi": "float32"},
                feature_schema_hash="v2_hash",
                performance_metrics={"accuracy": 0.94, "latency_ms": 3.0},  # Slower!
                version="2.0.0",
            )

            model_id_v2 = registry.register_model(model_v2_path, manifest_v2)
            registry.deploy_model(model_id_v2, "production")

            # Track bad performance for v2
            registry.track_performance(
                model_id_v2,
                {
                    "live_accuracy": 0.85,  # Worse than expected!
                    "pnl": -1000.0,  # Losing money!
                    "trades": 50,
                },
            )

            # ROLLBACK to v1
            success = registry.rollback(target="production", to_model_id=model_id_v1)

            assert success is True

            # Verify v1 is now active
            active_models = registry.get_active_models()
            assert len(active_models) == 1
            assert active_models[0].manifest.model_id == model_id_v1
            assert active_models[0].deployment_status.value == DeploymentStatus.ACTIVE.value

            # Verify v2 is no longer active
            v2_info = registry.get_model(model_id_v2)
            assert v2_info is not None
            assert v2_info.deployment_status.value == DeploymentStatus.INACTIVE.value

            # Verify performance history is preserved
            v1_history = registry.get_performance_history(model_id_v1)
            assert len(v1_history) == 1
            assert v1_history[0]["pnl"] == 5000.0

    def test_ab_test_splits_traffic(self) -> None:
        """
        Test A/B testing configuration for comparing models.

        This is a production BEHAVIOR for safe model rollouts.

        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = ModelRegistry(registry_path)

            # Create control model (current production)
            control_path = registry_path / "control_model.onnx"
            control_path.write_text("control_weights")

            control_manifest = RegistryBuilder.model_manifest(
                model_id="control_model",
                architecture="LightGBM",
                feature_schema={"close": "float32", "sma_20": "float32"},
                feature_schema_hash="control_hash",
                performance_metrics={"accuracy": 0.88},
                version="1.0.0",
            )

            control_id = registry.register_model(control_path, control_manifest)

            # Create treatment model (challenger)
            treatment_path = registry_path / "treatment_model.onnx"
            treatment_path.write_text("treatment_weights")

            treatment_manifest = RegistryBuilder.model_manifest(
                model_id="treatment_model",
                architecture="XGBoost",
                feature_schema={"close": "float32", "sma_20": "float32", "rsi_14": "float32"},
                feature_schema_hash="treatment_hash",
                performance_metrics={"accuracy": 0.90},  # Claims to be better
                version="1.0.0",
            )

            treatment_id = registry.register_model(treatment_path, treatment_manifest)

            # Configure A/B test with 30/70 split
            ab_config = registry.configure_ab_test(
                models=[control_id, treatment_id],
                split_ratio=0.3,  # 30% to control, 70% to treatment
                duration_hours=24,
                target="production",
            )

            assert ab_config is not None
            assert ab_config["model_a"] == control_id
            assert ab_config["model_b"] == treatment_id
            assert ab_config["split_ratio"] == 0.3
            assert ab_config["duration_hours"] == 24

            # Both models should be in TESTING status
            all_models = registry.get_all_models()
            testing_models = [
                m for m in all_models if m.deployment_status.value == DeploymentStatus.TESTING.value
            ]
            assert len(testing_models) == 2

            # Track performance for both during A/B test
            registry.track_performance(
                control_id,
                {
                    "ab_test_accuracy": 0.87,
                    "ab_test_pnl": 3000.0,
                    "ab_test_trades": 300,  # 30% of traffic
                },
            )

            registry.track_performance(
                treatment_id,
                {
                    "ab_test_accuracy": 0.91,  # Better!
                    "ab_test_pnl": 8000.0,  # Much better!
                    "ab_test_trades": 700,  # 70% of traffic
                },
            )

            # Compare models
            comparison = registry.compare_models(
                model_ids=[control_id, treatment_id],
                metric="ab_test_pnl",
            )

            assert comparison is not None
            assert comparison["best_model"] == treatment_id
            assert comparison["metric"] == "ab_test_pnl"

    def test_hot_reload_updates_without_downtime(self) -> None:
        """
        Test hot reload capability for zero-downtime model updates.

        This is a critical production BEHAVIOR.

        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = ModelRegistry(registry_path)

            # Deploy initial model
            model_v1_path = registry_path / "model_v1.onnx"
            model_v1_path.write_text("v1_weights")

            manifest_v1 = RegistryBuilder.model_manifest(
                model_id="live_model_v1",
                architecture="ONNX",
                feature_schema={"feature": "float32"},
                feature_schema_hash="v1_hash",
                performance_metrics={"latency_ms": 1.5},
                version="1.0.0",
            )

            model_id_v1 = registry.register_model(model_v1_path, manifest_v1)
            registry.deploy_model(model_id_v1, "live_trading")

            # Simulate model serving
            active = registry.get_active_models()
            assert len(active) == 1
            current_model = active[0]
            assert current_model.manifest.model_id == model_id_v1

            # Prepare new model
            model_v2_path = registry_path / "model_v2.onnx"
            model_v2_path.write_text("v2_weights_improved")

            manifest_v2 = RegistryBuilder.model_manifest(
                model_id="live_model_v2",
                architecture="ONNX",
                feature_schema={"feature": "float32"},  # Same schema for compatibility
                feature_schema_hash="v1_hash",  # Same hash = compatible
                performance_metrics={"latency_ms": 1.2},  # Faster!
                version="2.0.0",
            )

            model_id_v2 = registry.register_model(model_v2_path, manifest_v2)

            # Hot reload - deploy new model to same target
            success = registry.deploy_model(model_id_v2, "live_trading")
            assert success is True

            # Both models may be active during transition
            active_after = registry.get_active_models()
            active_ids = [m.manifest.model_id for m in active_after]
            assert model_id_v2 in active_ids  # New model is active

            # Verify new model is deployed to target
            v2_info = registry.get_model(model_id_v2)
            assert v2_info is not None
            assert "live_trading" in v2_info.deployed_to
            assert v2_info.deployment_status.value == DeploymentStatus.ACTIVE.value

            # Old model is still available for rollback if needed
            v1_info = registry.get_model(model_id_v1)
            assert v1_info is not None  # Still exists for rollback

    def test_deployment_validates_constraints(self) -> None:
        """
        Test that deployment validates model constraints.

        This ensures models meet production requirements before deployment.

        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir)
            registry = ModelRegistry(registry_path)

            # Create model with strict constraints
            model_path = registry_path / "constrained_model.onnx"
            model_path.write_text("model_weights")

            manifest = RegistryBuilder.model_manifest(
                model_id="constrained_model",
                role=ModelRole.STUDENT,  # Students have strict requirements
                architecture="LightGBM",
                feature_schema={"close": "float32", "volume": "int64"},
                feature_schema_hash="constrained_hash",
                parent_id="teacher_model_123",  # Students need a parent
                performance_metrics={
                    "accuracy": 0.85,
                    "inference_latency_ms": 4.5,  # Under 5ms requirement
                },
                deployment_constraints={
                    "max_latency_ms": 5.0,
                    "min_accuracy": 0.80,
                    "max_memory_mb": 256,
                },
                version="1.0.0",
            )

            model_id = registry.register_model(model_path, manifest)

            # Deployment should succeed - meets constraints
            success = registry.deploy_model(model_id, "production")
            assert success is True

            # Create model violating constraints
            bad_model_path = registry_path / "slow_model.onnx"
            bad_model_path.write_text("slow_model_weights")

            bad_manifest = RegistryBuilder.model_manifest(
                model_id="slow_model",
                role=ModelRole.STUDENT,
                data_requirements=DataRequirements.L1_L2,  # Wrong! Students need L1-only
                architecture="DeepNN",
                feature_schema={"close": "float32", "orderbook": "float32"},
                feature_schema_hash="slow_hash",
                parent_id=None,  # Missing parent!
                performance_metrics={
                    "accuracy": 0.95,  # Good accuracy but...
                    "inference_latency_ms": 10.0,  # Too slow!
                },
                deployment_constraints={
                    "max_latency_ms": 5.0,  # Can't meet this
                },
                version="1.0.0",
            )

            bad_model_id = registry.register_model(bad_model_path, bad_manifest)

            # Auto-deployment should fail validation for student with wrong data requirements
            bad_student_model_id = registry.register_model(
                model_path=bad_model_path,
                manifest=bad_manifest,
                auto_deploy=True,  # Should fail
            )

            # Check it wasn't deployed
            bad_model_info = registry.get_model(bad_student_model_id)
            assert bad_model_info is not None
            assert bad_model_info.deployment_status.value == DeploymentStatus.INACTIVE.value
