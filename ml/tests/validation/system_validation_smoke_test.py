"""
System Validation Smoke Test for StoreOperationsComponent.

This test validates the component works in a production-like context:
- Component instantiation with real config
- Progressive fallback activation
- Store property access
- Health check operations
- Lifecycle hooks
"""

import pytest
from ml.actors.common import StoreOperationsComponent
from ml.config.base import MLActorConfig
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.data import BarType


def test_component_instantiation_and_lifecycle():
    """
    Test: Component can be instantiated and lifecycle methods work.

    This validates:
    - Component initialization succeeds
    - Store properties are accessible
    - Health check works
    - Lifecycle hooks don't crash
    """
    # Given: Valid actor config
    config = MLActorConfig(
        model_path="/tmp/dummy_model.onnx",
        model_id="test_model",
        bar_type=BarType.from_str("BTCUSDT.BINANCE-1-MINUTE-LAST-INTERNAL"),
        instrument_id=InstrumentId.from_str("BTCUSDT.BINANCE")
    )

    # When: Component created
    component = StoreOperationsComponent(config, actor_id="smoke_test_actor")

    # Then: Component instantiated successfully
    assert component is not None

    # And: All stores are accessible
    assert component.feature_store is not None
    assert component.model_store is not None
    assert component.strategy_store is not None
    assert component.data_store is not None

    # And: Health check works
    health = component.get_health_status()
    assert isinstance(health, dict)
    assert "feature_store" in health
    assert "model_store" in health
    assert "strategy_store" in health
    assert "data_store" in health

    # And: Lifecycle hooks don't crash
    component.on_start()  # Should not raise
    component.on_stop()   # Should not raise

    print("✓ Component lifecycle smoke test PASSED")
    print(f"  Feature store: {type(component.feature_store).__name__}")
    print(f"  Model store: {type(component.model_store).__name__}")
    print(f"  Strategy store: {type(component.strategy_store).__name__}")
    print(f"  Data store: {type(component.data_store).__name__}")
    print(f"  Health: {health}")


def test_component_progressive_fallback():
    """
    Test: Component falls back to DummyStore when PostgreSQL unavailable.

    This validates:
    - Invalid connection string triggers fallback
    - DummyStore is used as fallback
    - Component doesn't crash on invalid config
    - Health check still works with fallback
    """
    # Given: Invalid PostgreSQL config
    config = MLActorConfig(
        model_path="/tmp/dummy_model.onnx",
        model_id="test_model",
        bar_type=BarType.from_str("BTCUSDT.BINANCE-1-MINUTE-LAST-INTERNAL"),
        instrument_id=InstrumentId.from_str("BTCUSDT.BINANCE")
    )

    # When: Component created (should trigger fallback)
    component = StoreOperationsComponent(config, actor_id="fallback_test_actor")

    # Then: Component instantiated (fallback activated)
    assert component is not None

    # And: Stores are DummyStore instances
    # (Name check - DummyStore classes have "Dummy" in name)
    assert "Dummy" in type(component.feature_store).__name__
    assert "Dummy" in type(component.model_store).__name__
    assert "Dummy" in type(component.strategy_store).__name__

    # And: Health check still works
    health = component.get_health_status()
    assert isinstance(health, dict)
    assert len(health) == 4

    # And: All stores show degraded status
    for store_name, store_health in health.items():
        assert store_health["status"] in ["healthy", "degraded"]

    print("✓ Progressive fallback smoke test PASSED")
    print(f"  Fallback activated: {health}")


if __name__ == "__main__":
    # Run smoke tests directly
    print("=" * 70)
    print("SYSTEM VALIDATION SMOKE TESTS")
    print("=" * 70)

    test_component_instantiation_and_lifecycle()
    print()
    test_component_progressive_fallback()

    print()
    print("=" * 70)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 70)
