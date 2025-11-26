from __future__ import annotations

from pathlib import Path

from ml.common.protocols import MLComponentProtocol
from ml.registry import DataRegistry
from ml.registry import FeatureRegistry
from ml.registry import ModelRegistry
from ml.registry import StrategyRegistry
from ml.tests.utils.protocol_helpers import assert_implements_ml_component


def test_registries_implement_ml_component_protocol(tmp_path: Path) -> None:
    root = tmp_path / "registry"

    feature_registry = FeatureRegistry(registry_path=root / "features")
    model_registry = ModelRegistry(registry_path=root / "models")
    strategy_registry = StrategyRegistry(base_path=root)
    data_registry = DataRegistry(registry_path=root / "datasets")

    for comp in (feature_registry, model_registry, strategy_registry, data_registry):
        assert isinstance(comp, MLComponentProtocol)
        assert_implements_ml_component(comp)


def test_mixin_defaults_shape() -> None:
    # Simple class using the mixin via a registry instance
    # (defaults already exercised above). This test checks expected keys exist.
    class _Dummy:
        pass

    # Use a real component for realistic behavior
    comp = FeatureRegistry(registry_path=Path("/tmp/feature_registry_test"))
    health = comp.get_health_status()
    assert "component" in health and "status" in health and "timestamp" in health
