"""
Hypothesis-based property tests for model registry.

These tests verify registry invariants that must hold regardless of implementation.
Only includes tests for properties that are still valid in the new API.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from ml.registry.base import DataRequirements
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.local_registry import LocalModelRegistry


class TestRegistryProperties:
    """Test algebraic properties of the registry."""

    @given(
        metrics=st.dictionaries(
            st.sampled_from(["accuracy", "precision", "recall", "f1", "loss"]),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=20, deadline=5000)
    def test_metrics_persistence(self, metrics: dict[str, float]) -> None:
        """
        Property: Performance metrics should persist across registry reloads.

        This is a behavioral property - not an implementation detail.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir)
            model_path = Path(tmpdir) / "model.onnx"
            model_path.write_text("dummy_model")

            # Create manifest with proper structure
            manifest = ModelManifest(
                model_id="test_model",
                role=ModelRole.INFERENCE,
                data_requirements=DataRequirements.L1_ONLY,
                architecture="TestModel",
                feature_schema={"close": "float32"},
                feature_schema_hash="test_hash",
                performance_metrics=metrics,
                version="1.0.0",
            )

            # Register model
            registry1 = LocalModelRegistry(registry_path)
            model_id = registry1.register_model(model_path, manifest)

            # Track additional performance
            registry1.track_performance(model_id, {"live_metric": 0.95})

            # Flush to ensure data is persisted
            registry1.flush()

            # Reload registry
            registry2 = LocalModelRegistry(registry_path)
            model_info = registry2.get_model(model_id)

            # Original metrics should be in manifest
            assert model_info is not None
            assert model_info.manifest.performance_metrics == metrics

            # Tracked metrics should be in history
            assert len(model_info.performance_history) == 1
            assert model_info.performance_history[0]["live_metric"] == 0.95

    @given(
        n_operations=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=10, deadline=5000)
    def test_rollback_idempotency(self, n_operations: int) -> None:
        """
        Property: Rolling back multiple times to the same model is idempotent.

        rollback(X) ∘ rollback(X) = rollback(X)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = LocalModelRegistry(Path(tmpdir))

            # Create multiple models
            model_ids = []
            for i in range(3):
                model_path = Path(tmpdir) / f"model_{i}.onnx"
                model_path.write_text(f"model_{i}")

                manifest = ModelManifest(
                    model_id=f"model_{i}",
                    role=ModelRole.INFERENCE,
                    data_requirements=DataRequirements.L1_ONLY,
                    architecture="TestModel",
                    feature_schema={"feature": "float32"},
                    feature_schema_hash=f"hash_{i}",
                    version=f"{i}.0.0",
                )

                model_id = registry.register_model(model_path, manifest)
                model_ids.append(model_id)

            # Deploy latest
            registry.deploy_model(model_ids[-1], "target")

            # Rollback multiple times to first model
            for _ in range(n_operations):
                registry.rollback("target", model_ids[0])

            # Check state - should have first model deployed
            active = registry.get_active_models()
            assert len(active) == 1
            assert active[0].manifest.model_id == model_ids[0]

    @given(
        model_names=st.lists(
            st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
            min_size=2,
            max_size=5,
            unique=True,
        )
    )
    @settings(max_examples=10, deadline=5000)
    def test_model_isolation(self, model_names: list[str]) -> None:
        """
        Property: Operations on one model should not affect others.

        This tests that registry maintains proper isolation between models.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = LocalModelRegistry(Path(tmpdir))

            # Register all models
            model_ids = {}
            for name in model_names:
                model_path = Path(tmpdir) / f"{name}.onnx"
                model_path.write_text(name)

                manifest = ModelManifest(
                    model_id=name,
                    role=ModelRole.INFERENCE,
                    data_requirements=DataRequirements.L1_ONLY,
                    architecture="TestModel",
                    feature_schema={f"feature_{name}": "float32"},
                    feature_schema_hash=f"hash_{name}",
                    version="1.0.0",
                )

                model_id = registry.register_model(model_path, manifest)
                model_ids[name] = model_id

            # Deploy first model
            first_name = model_names[0]
            registry.deploy_model(model_ids[first_name], "target1")

            # Track performance for first model
            registry.track_performance(model_ids[first_name], {"metric": 0.9})

            # Operations on first model shouldn't affect others
            for name in model_names[1:]:
                model_info = registry.get_model(model_ids[name])
                assert model_info is not None
                assert model_info.deployment_status.value == "inactive"
                assert len(model_info.performance_history) == 0
                assert model_info.manifest.model_id == name
