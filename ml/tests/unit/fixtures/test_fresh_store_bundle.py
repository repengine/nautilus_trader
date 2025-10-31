"""
Test fresh_store_bundle fixture provides complete isolation.
"""

import pytest


@pytest.mark.database
@pytest.mark.serial
def test_fresh_store_bundle_isolation(fresh_store_bundle):
    """
    Verify fresh_store_bundle provides isolated stores per test.
    """
    # Write data to feature store
    fresh_store_bundle.feature_store.write_features(
        feature_set_id="test_isolation",
        instrument_id="TEST.SIM",
        features={"test_key": 1.0},
        ts_event=1_000_000_000,
        ts_init=1_000_000_001,
    )
    fresh_store_bundle.feature_store.flush()

    # Data should exist in THIS test's store
    from sqlalchemy import text

    with fresh_store_bundle.engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM ml_feature_values WHERE feature_set_id = 'test_isolation'"),
        )
        count = result.scalar()

    assert count == 1, "Data should exist in this test's store"


@pytest.mark.database
@pytest.mark.serial
def test_fresh_store_bundle_no_pollution(fresh_store_bundle):
    """
    Verify previous test's data does NOT pollute this test.
    """
    # This test should NOT see data from previous test
    from sqlalchemy import text

    with fresh_store_bundle.engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM ml_feature_values WHERE feature_set_id = 'test_isolation'"),
        )
        count = result.scalar()

    # Should be 0 because fresh_store_bundle cleanup truncated the table
    assert count == 0, "fresh_store_bundle should provide isolated stores with no cross-test data"


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.parametrize("run", [1, 2, 3])
def test_fresh_store_bundle_consistency(fresh_store_bundle, run):
    """
    Verify consistent behavior across multiple runs.
    """
    # Each parameterized run should see clean state
    from sqlalchemy import text

    with fresh_store_bundle.engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM ml_feature_values WHERE feature_set_id = 'test_isolation'"),
        )
        count = result.scalar()

    # Should always be 0, never polluted from previous parameterized runs
    assert count == 0, f"Run {run}: fresh_store_bundle should provide clean state"


@pytest.mark.database
@pytest.mark.serial
def test_fresh_store_bundle_attributes(fresh_store_bundle):
    """
    Verify fresh_store_bundle has all expected attributes.
    """
    assert hasattr(fresh_store_bundle, "feature_store"), "Should have feature_store"
    assert hasattr(fresh_store_bundle, "model_store"), "Should have model_store"
    assert hasattr(fresh_store_bundle, "strategy_store"), "Should have strategy_store"
    assert hasattr(fresh_store_bundle, "persistence_manager"), "Should have persistence_manager"
    assert hasattr(fresh_store_bundle, "engine"), "Should have engine"

    # Verify stores are not None
    assert fresh_store_bundle.feature_store is not None, "feature_store should be initialized"
    assert fresh_store_bundle.model_store is not None, "model_store should be initialized"
    assert fresh_store_bundle.strategy_store is not None, "strategy_store should be initialized"
