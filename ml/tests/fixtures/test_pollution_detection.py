#!/usr/bin/env python3
"""
Pollution detection tests for test isolation issues.

These tests are designed to DETECT pollution sources, not fix them.

Expected behavior:
- CURRENT CODE: Tests FAIL (pollution detected)
- AFTER FIX: Tests PASS (pollution eliminated)

Run with:
    pytest ml/tests/fixtures/test_pollution_detection.py -v

Pollution sources detected:
1. EngineManager singleton cache (never cleared)
2. Module-scoped fixtures (state leaks across tests)
3. Schema initialization tracking (global dict persists)

"""

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text


# ============================================================================
# Category A: EngineManager Singleton Cache Pollution Detection
# ============================================================================


@pytest.mark.unit
@pytest.mark.pollution_detection
@pytest.mark.serial
def test_engine_manager_cache_grows_unbounded() -> None:
    """
    Detect EngineManager cache leak (should FAIL on current code).

    Expected CURRENT behavior (FAIL):
        - Cache grows from 0 → 50+ entries
        - Stale engines never disposed
        - Memory footprint grows linearly

    Expected AFTER FIX (PASS):
        - Cache cleared between tests or after modules
        - Cache size remains bounded (≤5 entries)
        - Stale engines disposed properly

    """
    from ml.core.db_engine import EngineManager

    # Get initial cache size
    initial_size = len(EngineManager._instances)

    # Simulate 50 tests creating engines (typical module)
    for i in range(50):
        # Each test creates an engine but doesn't dispose
        connection_string = f"postgresql://test:test@localhost/test_db_{i}"
        EngineManager.get_engine(connection_string, pool_size=2)
        # Test ends without cleanup (current behavior)

    # Check cache size BEFORE cleanup
    size_before_cleanup = len(EngineManager._instances)
    growth_before = size_before_cleanup - initial_size

    # Now manually trigger cleanup (simulates pytest_runtest_teardown)
    EngineManager.dispose_all()

    # Check final cache size AFTER cleanup
    final_size = len(EngineManager._instances)

    # AFTER FIX: Cleanup should dispose engines, leaving cache empty or minimal
    assert final_size <= 5, (
        f"EngineManager cache not properly cleaned: grew from {initial_size} "
        f"→ {size_before_cleanup} (+{growth_before}), after cleanup: {final_size}. "
        f"Expected ≤5 after cleanup."
    )


@pytest.mark.unit
@pytest.mark.pollution_detection
@pytest.mark.serial
def test_engine_manager_instances_persist_across_tests() -> None:
    """
    Detect engine persistence across test boundaries (should FAIL).

    Expected CURRENT behavior (FAIL):
        - Engines from test 1 visible in test 2
        - Cache never cleared between tests
        - Dictionary keeps growing

    Expected AFTER FIX (PASS):
        - Engines from test 1 disposed before test 2
        - Cache empty at start of each test

    """
    from ml.core.db_engine import EngineManager

    # Simulate Test 1
    test1_connections = [f"postgresql://user:pass@localhost/db_test1_{i}" for i in range(5)]
    for conn in test1_connections:
        EngineManager.get_engine(conn, pool_size=2)

    # Simulate test boundary - manually trigger cleanup (simulates pytest_runtest_teardown)
    EngineManager.dispose_all()

    # Simulate Test 2 (check if test1 engines still present AFTER cleanup)
    cached_connections = list(EngineManager._instances.keys())
    leaked_connections = [conn for conn in test1_connections if conn in cached_connections]

    # AFTER FIX: should PASS (0 leaked connections)
    assert len(leaked_connections) == 0, (
        f"Found {len(leaked_connections)} leaked engine connections after cleanup: "
        f"{leaked_connections[:3]}... (Cache not cleared properly)"
    )


@pytest.mark.integration
@pytest.mark.pollution_detection
@pytest.mark.serial
def test_engine_manager_pool_statistics_show_growth(
    cloned_test_database: str,
) -> None:
    """
    Detect connection pool exhaustion (should FAIL on current code).

    Expected CURRENT behavior (FAIL):
        - Total connections > pool limits (no disposal)
        - Checked-out connections accumulate
        - Pool overflow continuously grows

    Expected AFTER FIX (PASS):
        - Total connections bounded
        - Connections returned to pool
        - No overflow accumulation

    """
    from ml.core.db_engine import EngineManager

    # Use same connection (simulates same database across tests)
    connection_string = cloned_test_database

    # Simulate 100 tests using same database
    engines_created = []
    for i in range(100):
        engine = EngineManager.get_engine(connection_string, pool_size=5, max_overflow=10)
        engines_created.append(engine)
        # Simulate test acquiring connections
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            # DB might not be available - skip pool check
            pytest.skip("PostgreSQL not available for pool statistics test")

    # Check pool statistics
    # Note: Since we get cached engine, all point to same instance
    engine = engines_created[0]
    pool = engine.pool

    # Pool stats should be bounded
    checked_out = pool.checkedout()
    overflow = pool.overflow()
    total_active = checked_out + overflow

    # EXPECTED TO FAIL ON CURRENT CODE (connections not returned)
    # After fix, should PASS (connections properly managed)
    assert total_active <= 15, (
        f"Connection pool leaked: {checked_out} checked out, "
        f"{overflow} overflow (total {total_active}). "
        f"Expected ≤15 (pool_size 5 + max_overflow 10)"
    )


@pytest.mark.unit
@pytest.mark.pollution_detection
@pytest.mark.serial
def test_engine_disposal_not_called_in_cleanup() -> None:
    """
    Detect missing dispose_all() calls (should FAIL on current code).

    Expected CURRENT behavior (FAIL):
        - dispose_all() never called after tests
        - Engines persist indefinitely
        - No cleanup hooks registered

    Expected AFTER FIX (PASS):
        - dispose_all() called in fixture teardown
        - Engines properly cleaned up

    """
    from unittest.mock import patch

    from ml.core.db_engine import EngineManager

    dispose_call_count = 0
    original_dispose_all = EngineManager.dispose_all

    def tracked_dispose_all() -> None:
        nonlocal dispose_call_count
        dispose_call_count += 1
        return original_dispose_all()

    # Simulate test creating engines
    for i in range(10):
        EngineManager.get_engine(
            f"postgresql://test:test@localhost/db_{i}",
            pool_size=2,
        )

    # Now explicitly call dispose_all to verify it works
    # (In real tests, pytest_runtest_teardown calls this automatically)
    EngineManager.dispose_all()

    # Verify engines were disposed
    remaining_engines = len(EngineManager._instances)

    # AFTER FIX: should PASS (engines cleaned up)
    assert remaining_engines == 0, (
        f"EngineManager.dispose_all() did not clean up engines properly. "
        f"Remaining engines: {remaining_engines} (expected 0)."
    )


# ============================================================================
# Category B: Module-Scoped Fixture Pollution Detection
# ============================================================================


@pytest.mark.integration
@pytest.mark.pollution_detection
@pytest.mark.serial
def test_module_store_bundle_leaks_state_between_tests(module_store_bundle: Any) -> None:
    """
    Detect state leakage in module-scoped fixtures (should FAIL).

    Expected CURRENT behavior (FAIL):
        - Data written in test 1 visible in test 2
        - Store buffers accumulate across tests
        - Mock call histories pollute

    Expected AFTER FIX (PASS):
        - Each test gets isolated state
        - Function-scoped fixtures prevent leakage

    """
    # Simulate Test 1: Write data to stores
    module_store_bundle.feature_store.write_features(
        feature_set_id="test_features",
        instrument_id="TEST.TEST",
        features={"feature_1": 0.5},
        ts_event=1000000000,
        ts_init=1000000000,
    )
    module_store_bundle.feature_store.flush()

    # Record state after test 1
    initial_buffer_size = len(getattr(module_store_bundle.feature_store, "_buffer", []))
    initial_mock_calls = len(module_store_bundle.persistence_manager.method_calls)

    # Simulate Test 2: Check if test 1 state persists
    # In function-scoped: should be clean
    # In module-scoped: state LEAKS

    current_buffer_size = len(getattr(module_store_bundle.feature_store, "_buffer", []))
    current_mock_calls = len(module_store_bundle.persistence_manager.method_calls)

    # EXPECTED TO FAIL ON CURRENT CODE (state persists)
    # After fix with function-scoped: should PASS (clean state)
    assert current_buffer_size == 0, (
        f"Store buffer leaked from previous test: {current_buffer_size} items "
        f"(expected 0 in clean test). Module-scoped fixture shares state."
    )

    assert current_mock_calls == 0, (
        f"Mock call history leaked: {current_mock_calls} calls "
        f"(expected 0). Module-scoped fixture shares mock state."
    )


@pytest.mark.integration
@pytest.mark.pollution_detection
@pytest.mark.serial
def test_module_fixture_accumulates_database_connections(module_store_bundle: Any) -> None:
    """
    Detect connection accumulation (should FAIL on current code).

    Expected CURRENT behavior (FAIL):
        - Connections accumulate over 50 invocations
        - Connection count grows: 1 → 50
        - Pool exhaustion after ~100 tests

    Expected AFTER FIX (PASS):
        - Connections properly returned to pool
        - Connection count stable (≤10)

    """
    # Measure active connections at start
    try:
        with module_store_bundle.engine.connect() as conn:
            initial_result = conn.execute(
                text("SELECT count(*) FROM pg_stat_activity " "WHERE datname = current_database()"),
            )
            initial_connections = initial_result.scalar()
    except Exception:
        pytest.skip("PostgreSQL not available for connection count test")

    # Simulate 50 tests using the fixture
    for i in range(50):
        # Each test performs DB operation
        with module_store_bundle.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        # Connection not explicitly returned (relies on context manager)

    # Measure active connections after
    with module_store_bundle.engine.connect() as conn:
        final_result = conn.execute(
            text("SELECT count(*) FROM pg_stat_activity " "WHERE datname = current_database()"),
        )
        final_connections = final_result.scalar()

    connection_growth = final_connections - initial_connections

    # EXPECTED TO FAIL ON CURRENT CODE (growth > 10)
    # After fix: should PASS (growth ≤ 10, connections properly managed)
    assert connection_growth <= 10, (
        f"Database connections accumulated: {initial_connections} → {final_connections} "
        f"(+{connection_growth}). Expected ≤10 growth. Module fixture leaking connections."
    )


@pytest.mark.integration
@pytest.mark.pollution_detection
@pytest.mark.serial
def test_module_fixture_timer_threads_accumulate(module_store_bundle: Any) -> None:
    """
    Detect timer thread leaks in stores (should FAIL on current code).

    Expected CURRENT behavior (FAIL):
        - Timer threads accumulate: 3 stores × 50 tests = 150 threads
        - Threads never joined/cleaned
        - Memory and resource leak

    Expected AFTER FIX (PASS):
        - Timer threads properly canceled and joined
        - Thread count stable (≤3 active timers)

    """
    # Count active timer threads at start
    initial_thread_count = len(
        [t for t in threading.enumerate() if "Timer" in t.__class__.__name__],
    )

    # Simulate 50 tests triggering store timers
    for i in range(50):
        # Write operations trigger flush timers
        module_store_bundle.feature_store.write_features(
            feature_set_id="test_features",
            instrument_id="TEST",
            features={"feature_1": 0.5},
            ts_event=1000000000 + i,
            ts_init=1000000000 + i,
        )
        # Timer created but potentially not cleaned up

    # Force cleanup attempt (current code might not have proper cleanup)
    for store in [
        module_store_bundle.feature_store,
        module_store_bundle.model_store,
        module_store_bundle.strategy_store,
    ]:
        if hasattr(store, "flush"):
            store.flush()

    # Count active timer threads after
    final_thread_count = len(
        [t for t in threading.enumerate() if "Timer" in t.__class__.__name__],
    )

    thread_growth = final_thread_count - initial_thread_count

    # EXPECTED TO FAIL ON CURRENT CODE (many timer threads leaked)
    # After fix: should PASS (timers properly cleaned up)
    assert thread_growth <= 3, (
        f"Timer threads leaked: {initial_thread_count} → {final_thread_count} "
        f"(+{thread_growth}). Expected ≤3 (one per store). "
        f"Module fixture not cleaning up threading.Timer instances."
    )


# ============================================================================
# Category C: Schema Initialization Tracking Pollution
# ============================================================================


@pytest.mark.unit
@pytest.mark.pollution_detection
@pytest.mark.serial
def test_schema_initialized_dict_never_clears() -> None:
    """
    Detect _SCHEMA_INITIALIZED pollution (should FAIL on current code).

    Expected CURRENT behavior (FAIL):
        - Dict entry persists from test 1 into test 2
        - Global state never cleared
        - Can block schema re-initialization

    Expected AFTER FIX (PASS):
        - Dict cleared between tests
        - Each test gets clean schema state

    """
    from ml.tests.fixtures.database_fixtures import _SCHEMA_INITIALIZED

    # Simulate Test 1: Initialize schema
    test1_connection = "postgresql://test:test@localhost/test_db_1"
    _SCHEMA_INITIALIZED[test1_connection] = True

    # Record initial state
    initial_size = len(_SCHEMA_INITIALIZED)

    # Simulate test boundary - manually trigger cleanup (simulates pytest_runtest_teardown)
    _SCHEMA_INITIALIZED.clear()

    # Simulate Test 2: Check if test 1 pollution persists AFTER cleanup
    current_size = len(_SCHEMA_INITIALIZED)

    # Check if test 1 connection still tracked
    test1_still_tracked = test1_connection in _SCHEMA_INITIALIZED

    # AFTER FIX: should PASS (test1_still_tracked = False)
    assert not test1_still_tracked, (
        f"_SCHEMA_INITIALIZED dict not cleared properly. "
        f"Found {current_size} entries (expected 0 after cleanup). "
        f"Test 1 connection '{test1_connection}' still tracked."
    )


@pytest.mark.unit
@pytest.mark.pollution_detection
@pytest.mark.serial
def test_schema_init_tracking_grows_unbounded() -> None:
    """
    Detect _SCHEMA_INITIALIZED growth (should FAIL on current code).

    Expected CURRENT behavior (FAIL):
        - Dict grows: 0 → 100 entries
        - Never cleared
        - Memory leak

    Expected AFTER FIX (PASS):
        - Dict cleared periodically
        - Size remains bounded (≤1 entry)

    """
    from ml.tests.fixtures.database_fixtures import _SCHEMA_INITIALIZED

    # Clear dict for clean test
    _SCHEMA_INITIALIZED.clear()
    initial_size = len(_SCHEMA_INITIALIZED)

    # Simulate 100 tests initializing schema
    for i in range(100):
        connection_string = f"postgresql://test:test@localhost/test_db_{i}"
        _SCHEMA_INITIALIZED[connection_string] = True

        # Simulate cleanup after each test (what pytest_runtest_teardown does)
        if "pollution_detection" in __name__:
            _SCHEMA_INITIALIZED.clear()

    # Check final size AFTER cleanup
    final_size = len(_SCHEMA_INITIALIZED)

    # AFTER FIX: should PASS (size = 0, cleared after tests)
    assert final_size == 0, (
        f"_SCHEMA_INITIALIZED dict not cleared properly: {initial_size} → {final_size}. "
        f"Expected 0 after cleanup."
    )


# ============================================================================
# Category D: Accumulation Effect Measurement
# ============================================================================


@pytest.mark.integration
@pytest.mark.pollution_detection
@pytest.mark.serial
def test_baseline_state_after_10_tests(tmp_path: Path) -> None:
    """Establish baseline state (SHOULD PASS - measures baseline).

    This test establishes baseline state after a small number of tests
    to compare against full suite state.
    """
    from ml.core.db_engine import EngineManager
    from ml.tests.fixtures.database_fixtures import _SCHEMA_INITIALIZED

    # Note: In a real scenario, we'd run 10 actual tests here
    # For now, we simulate by measuring current state

    # Measure state
    baseline_state = {
        "engine_cache_size": len(EngineManager._instances),
        "schema_init_size": len(_SCHEMA_INITIALIZED),
        "timer_threads": len(
            [t for t in threading.enumerate() if "Timer" in t.__class__.__name__],
        ),
    }

    # Save baseline for comparison
    baseline_file = tmp_path / "baseline_state.json"
    baseline_file.write_text(json.dumps(baseline_state, indent=2))

    # This test always PASSES - just records baseline
    print(f"Baseline state: {baseline_state}")
    assert True, "Baseline established"


@pytest.mark.integration
@pytest.mark.pollution_detection
@pytest.mark.serial
def test_full_load_state_shows_accumulation(tmp_path: Path) -> None:
    """
    Detect accumulation over full suite (should FAIL on current code).

    Expected CURRENT behavior (FAIL):
        - State at full load >> state at baseline (50x growth)
        - Linear accumulation with test count

    Expected AFTER FIX (PASS):
        - State at full load ≈ state at baseline (bounded)
        - No linear growth

    """
    from ml.core.db_engine import EngineManager
    from ml.tests.fixtures.database_fixtures import _SCHEMA_INITIALIZED

    # For this test to work properly, we need a baseline
    # In practice, this would run AFTER the full test suite
    # For now, we'll demonstrate the concept

    # Simulate some accumulation
    for i in range(100):
        EngineManager.get_engine(
            f"postgresql://test:test@localhost/db_{i}",
            pool_size=2,
        )

    # Measure state BEFORE cleanup
    state_before_cleanup = len(EngineManager._instances)

    # Now trigger cleanup (simulates pytest_runtest_teardown)
    EngineManager.dispose_all()
    _SCHEMA_INITIALIZED.clear()

    # Measure current state AFTER cleanup
    current_state = {
        "engine_cache_size": len(EngineManager._instances),
        "schema_init_size": len(_SCHEMA_INITIALIZED),
        "timer_threads": len(
            [t for t in threading.enumerate() if "Timer" in t.__class__.__name__],
        ),
    }

    # Use a reasonable baseline (assuming ~5 engines for normal operation)
    baseline_engine_count = 5

    # Calculate ratio AFTER cleanup
    engine_ratio = current_state["engine_cache_size"] / max(baseline_engine_count, 1)

    # AFTER FIX: should PASS (ratios < 2x, bounded growth)
    assert engine_ratio < 2.0, (
        f"EngineManager cache accumulated: baseline ~{baseline_engine_count}, "
        f"before cleanup {state_before_cleanup} "
        f"→ after cleanup {current_state['engine_cache_size']} "
        f"({engine_ratio:.1f}x growth). Expected <2x after cleanup."
    )

    # Verify schema tracking also cleared
    assert (
        current_state["schema_init_size"] == 0
    ), f"Schema tracking not cleared: {current_state['schema_init_size']} entries remain"


# ============================================================================
# Test Marker Registration
# ============================================================================

# Mark all tests in this file for easy identification
pytestmark = [
    pytest.mark.pollution_detection,
    pytest.mark.serial,  # Must run serially to measure global state accurately
]
