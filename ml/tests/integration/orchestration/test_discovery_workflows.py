#!/usr/bin/env python3
"""
Integration tests for DiscoveryService workflows.

Phase 2.2.7 (Structural): All tests marked @pytest.mark.skip.
Phase 2.2.8 (Full Implementation): Un-skip tests and verify PASS.
"""

from collections import OrderedDict
from unittest.mock import MagicMock

import pytest

from ml.config.market_data import MarketDatasetInput
from ml.data.ingest.discovery import DatasetDiscoveryService
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.discovery_service import DiscoveryService
from ml.registry.dataclasses import DatasetType
from ml.registry.dataclasses import StorageKind

# Import will be available after Phase 2 implementation
# from ml.orchestration.discovery_service import DiscoveryService


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def discovery_service():
    """Provide DiscoveryService instance for testing.

    Phase 2.2.7: Placeholder implementation.
    Phase 2.2.8: Full implementation.
    """
    return DiscoveryService()


@pytest.fixture
def dataset_config() -> DatasetBuildConfig:
    """Provide DatasetBuildConfig with symbols and instruments."""
    return DatasetBuildConfig(
        symbols="SPY,QQQ,AAPL",
        instrument_ids=("SPY.XNAS", "QQQ.XNAS", "AAPL.XNAS"),
        start_iso="2024-01-01",
        end_iso="2024-01-31",
        market_inputs=None,
        market_dataset_id=None,
    )


@pytest.fixture
def mock_dataset_service() -> MagicMock:
    """Provide mock DatasetDiscoveryService for testing."""
    service = MagicMock(spec=DatasetDiscoveryService)

    # Mock discover method to return market inputs
    service.discover.return_value = (
        MarketDatasetInput(
            dataset_id="test-dataset-spy",
            schema="ohlcv-1m",
            symbols=("SPY",),
        ),
        MarketDatasetInput(
            dataset_id="test-dataset-qqq",
            schema="ohlcv-1m",
            symbols=("QQQ",),
        ),
    )

    # Mock discover_one method to return symbol discovery
    service.discover_one.return_value = MagicMock(
        dataset_id="test-dataset",
        schema="ohlcv-1m",
        storage_kind=StorageKind.POSTGRES,
        symbol="SPY",
        requested_symbol="SPY",
        available_start_ns=0,
        available_end_ns=1000000000000,
        cost_usd=0.0,
        instrument_id="SPY.XNAS",
    )

    # Mock policy
    service.policy = MagicMock()
    service.policy.coverage = MagicMock()
    service.policy.coverage.allow_dataset = MagicMock()

    return service


@pytest.fixture
def mock_coverage_policy() -> MagicMock:
    """Provide mock CoveragePolicy for testing."""
    policy = MagicMock()
    policy.allow_dataset = MagicMock()
    return policy


@pytest.fixture
def resolved_bindings() -> tuple[ResolvedMarketBinding, ...]:
    """Provide tuple of ResolvedMarketBinding objects for testing."""
    return (
        ResolvedMarketBinding(
            binding_id="test-binding-spy",
            symbol="SPY",
            instrument_ids=("SPY.XNAS",),
            dataset_id="test-dataset-spy",
            descriptor_id=None,
            schema="ohlcv-1m",
            storage_kind=StorageKind.POSTGRES,
            license_start=None,
            license_end=None,
            start=None,
            end=None,
            source="test",
        ),
        ResolvedMarketBinding(
            binding_id="test-binding-qqq",
            symbol="QQQ",
            instrument_ids=("QQQ.XNAS",),
            dataset_id="test-dataset-qqq",
            descriptor_id=None,
            schema="ohlcv-1m",
            storage_kind=StorageKind.POSTGRES,
            license_start=None,
            license_end=None,
            start=None,
            end=None,
            source="test",
        ),
    )


# =============================================================================
# Integration Tests (3 tests)
# =============================================================================


@pytest.mark.integration
def test_service_discovery_workflow(
    discovery_service,
    mock_dataset_service,
    mock_coverage_policy,
):
    """Test service discovery end-to-end workflow.

    Phase 2.2.7: Returns empty tuple (placeholder).
    Phase 2.2.8: Discovers market inputs and updates coverage policy.

    Workflow:
    1. Create DiscoveryService with mock DatasetDiscoveryService
    2. Call discover_market_inputs with symbol_map
    3. Verify market inputs discovered
    4. Check coverage policy updated
    """
    # Phase 2.2.7: Placeholder returns empty tuple
    result = discovery_service.discover_market_inputs(
        symbol_map={
            "SPY": ("SPY.XNAS",),
            "QQQ": ("QQQ.XNAS",),
        },
        schema="ohlcv-1m",
        start_ns=0,
        end_ns=1000000000000,
        dataset_hint="test-hint",
    )

    # Phase 2.2.7 assertions
    assert result == ()
    assert isinstance(result, tuple)

    # Phase 2.2.8 assertions (uncomment after implementation):
    # assert len(result) == 2
    # assert all(isinstance(inp, MarketDatasetInput) for inp in result)
    # assert result[0].dataset_id == "test-dataset-spy"
    # assert result[1].dataset_id == "test-dataset-qqq"

    # Verify coverage policy updated (Phase 2.2.8)
    # mock_coverage_policy.allow_dataset.assert_called()


@pytest.mark.integration
def test_resource_discovery_workflow(
    discovery_service,
    dataset_config,
    resolved_bindings,
):
    """Test resource discovery (symbols and instruments) workflow.

    Phase 2.2.7: Returns empty collections (placeholder).
    Phase 2.2.8: Returns populated symbol map and instrument IDs.

    Workflow:
    1. Create DiscoveryService
    2. Call symbol_to_instruments with config containing symbols
    3. Call collect_instrument_ids with bindings
    4. Verify instrument IDs collected
    """
    # Step 1: Discover symbol-to-instrument mapping
    symbol_map = discovery_service.symbol_to_instruments(dataset_config)

    # Phase 2.2.7 assertions
    assert symbol_map == OrderedDict()
    assert isinstance(symbol_map, OrderedDict)

    # Phase 2.2.8 assertions (uncomment after implementation):
    # assert len(symbol_map) == 3
    # assert "SPY" in symbol_map
    # assert "QQQ" in symbol_map
    # assert "AAPL" in symbol_map
    # assert symbol_map["SPY"] == ("SPY.XNAS",)

    # Step 2: Collect instrument IDs from bindings
    instrument_ids = discovery_service.collect_instrument_ids(
        bindings=resolved_bindings,
        existing=("AAPL.XNAS",),
    )

    # Phase 2.2.7 assertions
    assert instrument_ids == ()
    assert isinstance(instrument_ids, tuple)

    # Phase 2.2.8 assertions (uncomment after implementation):
    # assert len(instrument_ids) == 3
    # assert "SPY.XNAS" in instrument_ids
    # assert "QQQ.XNAS" in instrument_ids
    # assert "AAPL.XNAS" in instrument_ids


@pytest.mark.integration
def test_schema_discovery_and_mapping_workflow(
    discovery_service,
    dataset_config,
):
    """Test schema inference and mapping end-to-end workflow.

    Phase 2.2.7: Returns safe defaults (placeholder).
    Phase 2.2.8: Returns correct schema mappings.

    Workflow:
    1. Create DiscoveryService
    2. Call infer_default_schema for config
    3. Call map_schema_to_dataset_type with inferred schema
    4. Call normalise_schema_for_lookback with schema
    5. Verify all steps complete without error
    """
    # Step 1: Infer default schema
    default_schema = discovery_service.infer_default_schema(dataset_config)

    # Phase 2.2.7 assertions
    assert default_schema == "ohlcv-1m"
    assert isinstance(default_schema, str)

    # Step 2: Map schema to dataset type
    dataset_type = discovery_service.map_schema_to_dataset_type(default_schema)

    # Phase 2.2.7 assertions
    assert dataset_type == DatasetType.BARS

    # Phase 2.2.8 assertions (uncomment after implementation):
    # Test multiple schema types:
    # assert discovery_service.map_schema_to_dataset_type("tbbo") == DatasetType.TBBO
    # assert discovery_service.map_schema_to_dataset_type("trades") == DatasetType.TRADES
    # assert discovery_service.map_schema_to_dataset_type("mbp-1") == DatasetType.MBP1

    # Step 3: Normalize schema for lookback
    normalised = discovery_service.normalise_schema_for_lookback(default_schema)

    # Phase 2.2.7 assertions
    assert normalised == "bars"
    assert isinstance(normalised, str)

    # Phase 2.2.8 assertions (uncomment after implementation):
    # Test multiple schema normalizations:
    # assert discovery_service.normalise_schema_for_lookback("ohlcv-5m") == "bars"
    # assert discovery_service.normalise_schema_for_lookback("tbbo") == "quotes"
    # assert discovery_service.normalise_schema_for_lookback("trades") == "trades"
    # assert discovery_service.normalise_schema_for_lookback("mbp-1") == "mbp"
    # assert discovery_service.normalise_schema_for_lookback(None) == "bars"
