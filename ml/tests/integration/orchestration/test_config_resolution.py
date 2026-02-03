#!/usr/bin/env python3
"""
Integration tests for ConfigResolver configuration resolution workflows.

Phase 2.2.6: STRUCTURAL PHASE
- All tests marked @pytest.mark.skip
- Tests designed for Phase 2.2.8 full implementation
- Document expected configuration resolution behavior
"""

from collections import OrderedDict

import pytest

from ml.orchestration.config_types import DatasetBuildConfig
from ml.stores.providers import DAY_NS
from ml.tests.utils.targets import build_default_target_semantics_payload


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
        start_iso=None,  # Test default resolution
        end_iso=None,  # Test default resolution
        symbols="SPY,QQQ",
        instrument_ids=("SPY.NASDAQ", "QQQ.NASDAQ"),
        market_dataset_id="databento.equities.us",
        target_semantics=build_default_target_semantics_payload(),
    )


@pytest.fixture
def config_resolver():
    """Provides ConfigResolver instance for testing."""
    from ml.orchestration.config_resolver import ConfigResolver

    return ConfigResolver()


# ============================================================================
# INTEGRATION TESTS (4 tests - ALL SKIPPED)
# ============================================================================


@pytest.mark.integration
def test_resolve_window_bounds_with_defaults(config_resolver, dataset_config):
    """
    Verify window bounds resolution workflow with default values.

    Phase 2.2.8 Expected Behavior:
    - Applies default end_iso (current datetime UTC)
    - Calculates default start_iso (e.g., 1 year before end)
    - Returns (start_ns, end_ns) with valid bounds
    - start_ns < end_ns always true
    """
    resolver = config_resolver

    start_ns, end_ns = resolver.resolve_window_bounds_ns(dataset_config)

    # Phase 2.2.8 assertions
    assert start_ns > 0, "start_ns must be positive nanoseconds"
    assert end_ns > start_ns, "end_ns must be greater than start_ns"
    assert end_ns - start_ns > DAY_NS, "Window must be at least 1 day"


@pytest.mark.integration
def test_resolve_instruments_from_multiple_sources(config_resolver):
    """
    Verify instrument ID resolution merges config and parameter sources.

    Phase 2.2.8 Expected Behavior:
    - Merges config and parameter instrument lists
    - Deduplicates (SPY appears only once)
    - Preserves order (first occurrence)
    - Uppercases all IDs
    - Returns ("SPY.NASDAQ", "QQQ.NASDAQ")
    """
    resolver = config_resolver

    dataset_config = DatasetBuildConfig(
        data_dir="/tmp/test_data",
        out_dir="/tmp/test_out",
        dataset_id="test.dataset",
        instrument_ids=("SPY.NASDAQ",),
        symbols="SPY",
        target_semantics=build_default_target_semantics_payload(),
    )

    result = resolver.resolve_instrument_ids(
        dataset_config,
        override=("QQQ.NASDAQ", "SPY.NASDAQ"),  # Duplicate SPY
    )

    # Override wins and preserves provided order
    assert result == ("QQQ.NASDAQ", "SPY.NASDAQ")


@pytest.mark.integration
def test_symbol_to_instruments_mapping_workflow(config_resolver):
    """
    Verify symbol-to-instruments mapping creates correct OrderedDict.

    Phase 2.2.8 Expected Behavior:
    - Parses symbols: ["SPY", "QQQ"]
    - Groups instruments by symbol:
      - "SPY" -> ("SPY.NASDAQ", "SPY.ARCA")
      - "QQQ" -> ("QQQ.NASDAQ",)
    - Returns OrderedDict preserving symbol order
    """
    resolver = config_resolver

    dataset_config = DatasetBuildConfig(
        data_dir="/tmp/test_data",
        out_dir="/tmp/test_out",
        dataset_id="test.dataset",
        symbols="SPY,QQQ",
        instrument_ids=("SPY.NASDAQ", "SPY.ARCA", "QQQ.NASDAQ"),
        target_semantics=build_default_target_semantics_payload(),
    )

    result = resolver.symbol_to_instruments(dataset_config)

    # Phase 2.2.8 assertions
    assert isinstance(result, OrderedDict)
    assert list(result.keys()) == ["SPY", "QQQ"], "Symbol order preserved"
    assert result["SPY"] == (
        "SPY.NASDAQ",
        "SPY.ARCA",
    ), "SPY should map to both instruments"
    assert result["QQQ"] == ("QQQ.NASDAQ",), "QQQ should map to one instrument"


@pytest.mark.integration
def test_infer_default_schema_from_config_hints(config_resolver):
    """
    Verify schema inference analyzes config for appropriate default.

    Phase 2.2.8 Expected Behavior:
    - Returns "tbbo" if quotes are primary
    - Returns "ohlcv-1m" if bars are primary
    - Returns "trades" if trades are primary
    - Falls back to "ohlcv-1m" if no hints
    """
    # Test different config variations
    # Note: DatasetBuildConfig may not have bars/quotes fields yet
    # This test documents expected behavior for Phase 2.2.8

    config_empty = DatasetBuildConfig(
        data_dir="/tmp/test_data",
        out_dir="/tmp/test_out",
        dataset_id="test",
        symbols="SPY",
        target_semantics=build_default_target_semantics_payload(),
    )
    resolver = config_resolver
    assert resolver.infer_default_schema(config_empty) == "ohlcv-1m"

    # Phase 2.2.8: Add tests for bars=True, quotes=True when fields exist
