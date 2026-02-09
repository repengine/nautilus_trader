from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from ml.config.constants import Versions
from ml.registry.base import DataRequirements
from ml.registry.dataclasses import QualityGate
from ml.registry.feature_operations import FeaturePromotionGate
from ml.registry.feature_operations import deprecate_feature_set
from ml.registry.feature_operations import promote_feature_set
from ml.registry.feature_operations import register_default_feature_set
from ml.registry.feature_registry import FeatureRole


def test_feature_promotion_gate_to_quality_gate_maps_all_fields() -> None:
    gate = FeaturePromotionGate(
        metric_name="pr_auc",
        threshold=0.72,
        comparison="gte",
        required=False,
    )

    quality_gate = gate.to_quality_gate()
    assert isinstance(quality_gate, QualityGate)
    assert quality_gate.metric_name == "pr_auc"
    assert quality_gate.threshold == 0.72
    assert quality_gate.comparison == "gte"
    assert quality_gate.required is False


def test_register_default_feature_set_uses_default_manifest_version() -> None:
    with (
        patch("ml.registry.feature_operations.FeatureEngineer") as engineer_cls,
        patch("ml.registry.feature_operations.FeatureRegistry") as registry_cls,
    ):
        engineer = engineer_cls.return_value
        manifest = object()
        engineer.generate_feature_manifest.return_value = manifest
        registry = registry_cls.return_value
        registry.register_feature_set.return_value = "feature_set_1"

        feature_set_id = register_default_feature_set(Path("/tmp/features"))

    assert feature_set_id == "feature_set_1"
    engineer.generate_feature_manifest.assert_called_once_with(
        name="default",
        version=Versions.DEFAULT_MANIFEST_VERSION,
        role=FeatureRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
    )
    registry.register_feature_set.assert_called_once_with(manifest)


def test_register_default_feature_set_honors_explicit_version() -> None:
    with (
        patch("ml.registry.feature_operations.FeatureEngineer") as engineer_cls,
        patch("ml.registry.feature_operations.FeatureRegistry") as registry_cls,
    ):
        engineer = engineer_cls.return_value
        engineer.generate_feature_manifest.return_value = object()
        registry = registry_cls.return_value
        registry.register_feature_set.return_value = "feature_set_2"

        register_default_feature_set(
            Path("/tmp/features"),
            name="l2_features",
            version="9.9.9",
            role=FeatureRole.INFERENCE_SUPPORT,
            data_requirements=DataRequirements.L1_L2,
        )

    engineer.generate_feature_manifest.assert_called_once_with(
        name="l2_features",
        version="9.9.9",
        role=FeatureRole.INFERENCE_SUPPORT,
        data_requirements=DataRequirements.L1_L2,
    )


def test_promote_feature_set_converts_typed_gates_before_registry_call() -> None:
    with patch("ml.registry.feature_operations.FeatureRegistry") as registry_cls:
        registry = registry_cls.return_value
        registry.validate_and_promote.return_value = True

        result = promote_feature_set(
            Path("/tmp/features"),
            feature_set_id="feature_123",
            gates=[
                FeaturePromotionGate(metric_name="p99_latency_ms", threshold=1.5, comparison="lte"),
                FeaturePromotionGate(metric_name="pr_auc", threshold=0.75),
            ],
        )

    assert result is True
    args = registry.validate_and_promote.call_args.args
    assert args[0] == "feature_123"
    assert len(args[1]) == 2
    assert all(isinstance(gate, QualityGate) for gate in args[1])
    assert args[0] == "feature_123"
    assert args[1][0].comparison == "lte"


def test_deprecate_feature_set_forwards_reason_to_registry() -> None:
    with patch("ml.registry.feature_operations.FeatureRegistry") as registry_cls:
        registry = registry_cls.return_value
        deprecate_feature_set(
            Path("/tmp/features"),
            feature_set_id="feature_123",
            reason="superseded",
        )

    registry.deprecate.assert_called_once_with("feature_123", reason="superseded")
