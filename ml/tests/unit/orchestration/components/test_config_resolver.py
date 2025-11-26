#!/usr/bin/env python3
"""
Unit tests for ConfigResolver component.

Phase 2.2.6: STRUCTURAL PHASE
- All tests marked @pytest.mark.skip
- Tests verify component structure (instantiation, method signatures)
- Full implementation deferred to Phase 2.2.8
"""

from collections import OrderedDict
from pathlib import Path

import pytest

from ml.orchestration.config_types import AutoFillUniverseConfig, DatasetBuildConfig


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def dataset_config() -> DatasetBuildConfig:
    """Provides minimal DatasetBuildConfig for testing."""
    return DatasetBuildConfig(
        data_dir="/tmp/test_data",
        out_dir="/tmp/test_out",
        dataset_id="test.dataset",
        start_iso="2024-01-01",
        end_iso="2024-12-31",
        symbols="SPY,QQQ",
        instrument_ids=("SPY.NASDAQ", "QQQ.NASDAQ"),
        market_dataset_id="databento.equities.us",
    )


@pytest.fixture
def auto_fill_config() -> AutoFillUniverseConfig:
    """Provides AutoFillUniverseConfig for testing."""
    return AutoFillUniverseConfig(
        enabled=True,
        dataset_id="test.dataset",
        instrument_ids=["SPY.NASDAQ"],
        include_bars=True,
        include_tbbo=True,
        include_trades=True,
        include_l2=False,
        include_l3=False,
    )


@pytest.fixture
def config_resolver():
    """Provides ConfigResolver instance for testing."""
    from ml.orchestration.components.config_resolver import ConfigResolver

    return ConfigResolver()


# ============================================================================
# STRUCTURAL TESTS (3 tests)
# ============================================================================


@pytest.mark.unit
def test_config_resolver_initializes_successfully():
    """
    Verify ConfigResolver can be instantiated without arguments.

    Phase 2.2.6: STRUCTURAL TEST
    - Component instantiates without errors
    - No required constructor parameters
    """
    from ml.orchestration.components.config_resolver import ConfigResolver

    resolver = ConfigResolver()
    assert resolver is not None
    assert isinstance(resolver, ConfigResolver)


@pytest.mark.unit
def test_config_resolver_has_all_ten_methods(config_resolver):
    """
    Verify all 10 methods exist with correct callable signatures.

    Phase 2.2.6: STRUCTURAL TEST
    - All 10 methods are callable
    - Methods accept expected parameter types (verified by mypy in Phase 3)
    """
    resolver = config_resolver

    # Resolution methods (4)
    assert callable(resolver._resolve_window_bounds_ns)
    assert callable(resolver._resolve_instrument_ids)
    assert callable(resolver._resolve_market_inputs)
    assert callable(resolver._symbol_to_instruments)

    # Inference methods (2)
    assert callable(resolver._infer_default_schema)
    assert callable(resolver._infer_feature_names)

    # Auto-fill methods (3)
    assert callable(resolver._auto_fill_universe)
    assert callable(resolver._auto_fill_schema)
    assert callable(resolver._auto_fill_l2)

    # Helper methods (1)
    assert callable(resolver._collect_instrument_ids)


@pytest.mark.unit
def test_config_resolver_methods_return_safe_defaults(
    config_resolver,
    dataset_config,
    auto_fill_config,
):
    """
    Verify all methods return safe default values without errors.

    Phase 2.2.6: STRUCTURAL TEST
    - All methods callable with minimal inputs
    - All methods return safe defaults (no exceptions)
    - Return types match annotations
    """
    resolver = config_resolver

    # Resolution methods return tuples/dicts
    result_bounds = resolver._resolve_window_bounds_ns(
        dataset_config.start_iso,
        dataset_config.end_iso,
    )
    assert isinstance(result_bounds, tuple)
    assert len(result_bounds) == 2

    result_instruments = resolver._resolve_instrument_ids(
        config_ids=tuple(dataset_config.instrument_ids),
        binding_ids=(),
    )
    assert isinstance(result_instruments, tuple)

    result_market = resolver._resolve_market_inputs(market_inputs=None)
    assert isinstance(result_market, OrderedDict)

    result_mapping = resolver._symbol_to_instruments(
        symbols=["SPY", "QQQ"],
        venue=None,
    )
    assert isinstance(result_mapping, tuple)

    # Inference methods return strings/tuples or None
    result_schema = resolver._infer_default_schema(dataset_config)
    assert result_schema is None or isinstance(result_schema, (str, tuple))

    result_features = resolver._infer_feature_names(Path("/tmp"))
    assert isinstance(result_features, tuple) or result_features is None

    # Auto-fill methods return tuples/dicts
    result_universe = resolver._auto_fill_universe(universe=["SPY", "QQQ"])
    assert isinstance(result_universe, tuple)

    result_schema_fill = resolver._auto_fill_schema(
        schema=None,
        config=dataset_config,
        feature_dir=None,
    )
    assert isinstance(result_schema_fill, tuple)

    result_l2 = resolver._auto_fill_l2(l2_schemas=None)
    assert isinstance(result_l2, dict)

    # Helper methods return tuples
    result_collect = resolver._collect_instrument_ids(market_inputs={})
    assert isinstance(result_collect, tuple)


# ============================================================================
# METHOD TESTS (10 tests - one per method)
# ============================================================================


@pytest.mark.unit
def test_resolve_window_bounds_ns_returns_tuple_of_ints(
    config_resolver,
    dataset_config,
):
    """
    Verify _resolve_window_bounds_ns() returns tuple of two integers or None values.

    Phase 2.2.6: Placeholder returns (None, None)
    Phase 2.2.8: Will parse ISO timestamps, apply defaults, validate bounds
    """
    resolver = config_resolver

    result = resolver._resolve_window_bounds_ns(
        dataset_config.start_iso,
        dataset_config.end_iso,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    # Placeholder returns (None, None)
    assert result[0] is None or isinstance(result[0], int)
    assert result[1] is None or isinstance(result[1], int)


@pytest.mark.unit
def test_resolve_instrument_ids_returns_tuple_of_strings(
    config_resolver,
    dataset_config,
):
    """
    Verify _resolve_instrument_ids() returns tuple of instrument IDs.

    Phase 2.2.6: Placeholder returns empty tuple ()
    Phase 2.2.8: Will merge config/param instruments, validate, deduplicate
    """
    resolver = config_resolver

    result = resolver._resolve_instrument_ids(
        config_ids=tuple(dataset_config.instrument_ids),
        binding_ids=(),
    )
    assert isinstance(result, tuple)
    # Placeholder returns empty tuple
    assert len(result) == 0


@pytest.mark.unit
def test_resolve_market_inputs_returns_tuple_of_inputs(
    config_resolver,
    dataset_config,
):
    """
    Verify _resolve_market_inputs() returns OrderedDict of market inputs.

    Phase 2.2.6: Placeholder returns empty OrderedDict()
    Phase 2.2.8: Will resolve market bindings, validate inputs
    """
    resolver = config_resolver

    result = resolver._resolve_market_inputs(market_inputs=None)
    assert isinstance(result, OrderedDict)
    # Placeholder returns empty OrderedDict
    assert len(result) == 0


@pytest.mark.unit
def test_symbol_to_instruments_returns_ordered_dict(
    config_resolver,
    dataset_config,
):
    """
    Verify _symbol_to_instruments() returns tuple of instrument IDs.

    Phase 2.2.6: Placeholder returns empty tuple ()
    Phase 2.2.8: Will parse symbols, build mapping, preserve order
    """
    resolver = config_resolver

    result = resolver._symbol_to_instruments(
        symbols=["SPY", "QQQ"],
        venue=None,
    )
    assert isinstance(result, tuple)
    # Placeholder returns empty tuple
    assert len(result) == 0


@pytest.mark.unit
def test_infer_default_schema_returns_string(dataset_config):
    """
    Verify _infer_default_schema() returns schema tuple or None.

    Phase 2.2.6: Returns None (placeholder)
    Phase 2.2.8: Will analyze config to infer appropriate schema
    """
    from ml.orchestration.components.config_resolver import ConfigResolver

    result = ConfigResolver._infer_default_schema(dataset_config)
    # Placeholder returns None
    assert result is None or isinstance(result, (str, tuple))


@pytest.mark.unit
def test_infer_feature_names_returns_tuple_of_strings():
    """
    Verify _infer_feature_names() returns tuple of feature name strings or None.

    Phase 2.2.6: Placeholder returns empty tuple ()
    Phase 2.2.8: Will scan directory, extract feature names from manifests
    """
    from ml.orchestration.components.config_resolver import ConfigResolver

    result = ConfigResolver._infer_feature_names(Path("/tmp/test_features"))
    # Placeholder returns empty tuple or None
    assert result is None or isinstance(result, tuple)
    if isinstance(result, tuple):
        assert all(isinstance(name, str) for name in result)


@pytest.mark.unit
def test_auto_fill_universe_returns_none(
    config_resolver,
    dataset_config,
    auto_fill_config,
):
    """
    Verify _auto_fill_universe() returns tuple of instrument IDs.

    Phase 2.2.6: Placeholder returns empty tuple ()
    Phase 2.2.8: Will resolve instruments, trigger schema auto-fill for each
    """
    resolver = config_resolver

    result = resolver._auto_fill_universe(universe=["SPY", "QQQ"])
    assert isinstance(result, tuple)
    # Placeholder returns empty tuple
    assert len(result) == 0


@pytest.mark.unit
def test_auto_fill_schema_returns_none(config_resolver, dataset_config):
    """
    Verify _auto_fill_schema() returns tuple of schema strings.

    Phase 2.2.6: Placeholder returns empty tuple () or schema unchanged
    Phase 2.2.8: Will resolve market bindings, trigger ingestion
    """
    resolver = config_resolver

    result = resolver._auto_fill_schema(
        schema=None,
        config=dataset_config,
        feature_dir=None,
    )
    assert isinstance(result, tuple)
    # Placeholder returns empty tuple when schema is None
    assert len(result) == 0


@pytest.mark.unit
def test_auto_fill_l2_returns_none(
    config_resolver,
    dataset_config,
    auto_fill_config,
):
    """
    Verify _auto_fill_l2() returns dict of L2 schemas.

    Phase 2.2.6: Placeholder returns empty dict {} or l2_schemas unchanged
    Phase 2.2.8: Will auto-fill depth and MBP schemas for L2 data
    """
    resolver = config_resolver

    result = resolver._auto_fill_l2(l2_schemas=None)
    assert isinstance(result, dict)
    # Placeholder returns empty dict when l2_schemas is None
    assert len(result) == 0


@pytest.mark.unit
def test_collect_instrument_ids_returns_tuple_of_strings():
    """
    Verify _collect_instrument_ids() returns tuple of instrument IDs.

    Phase 2.2.6: Placeholder returns empty tuple ()
    Phase 2.2.8: Will merge bindings and existing, deduplicate, preserve order
    """
    from ml.orchestration.components.config_resolver import ConfigResolver

    result = ConfigResolver._collect_instrument_ids(market_inputs={})
    assert isinstance(result, tuple)
    # Placeholder returns empty tuple
    assert len(result) == 0
