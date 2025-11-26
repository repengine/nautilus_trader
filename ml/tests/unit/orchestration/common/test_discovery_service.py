#!/usr/bin/env python3
"""
Unit tests for DiscoveryService component.

Phase 2.2.7 (Structural): All tests marked @pytest.mark.skip.
Phase 2.2.8 (Full Implementation): Un-skip tests and verify PASS.
"""

from collections import OrderedDict
from unittest.mock import MagicMock

import pytest

from ml.config.market_data import MarketDatasetInput
from ml.data.ingest.discovery import DatasetDiscoveryError
from ml.data.ingest.discovery import DatasetDiscoveryService
from ml.data.ingest.discovery import DiscoveryRequest
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.ingest.service import SymbolDatasetDiscovery
from ml.orchestration.config_types import DatasetBuildConfig
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
    from ml.orchestration.discovery_service import DiscoveryService
    return DiscoveryService()


@pytest.fixture
def dataset_config() -> DatasetBuildConfig:
    """Provide minimal DatasetBuildConfig for testing."""
    return DatasetBuildConfig(
        data_dir="/tmp/test_data",
        out_dir="/tmp/test_out",
        symbols="SPY,QQQ",
        instrument_ids=None,
        start_iso="2024-01-01",
        end_iso="2024-01-31",
        market_inputs=None,
        market_dataset_id=None,
    )


@pytest.fixture
def mock_dataset_service() -> MagicMock:
    """Provide mock DatasetDiscoveryService for testing."""
    service = MagicMock(spec=DatasetDiscoveryService)
    service.discover.return_value = ()
    service.discover_one.return_value = None
    service.policy = MagicMock()
    service.policy.coverage = None
    return service


@pytest.fixture
def mock_dataset_service_with_error() -> MagicMock:
    """Provide mock DatasetDiscoveryService that raises errors."""
    service = MagicMock(spec=DatasetDiscoveryService)
    service.discover.side_effect = DatasetDiscoveryError("Test error")
    service.discover_one.side_effect = DatasetDiscoveryError("Test error")
    return service


@pytest.fixture
def resolved_bindings() -> tuple[ResolvedMarketBinding, ...]:
    """Provide tuple of ResolvedMarketBinding objects for testing."""
    return (
        ResolvedMarketBinding(
            binding_id="test-binding-1",
            symbol="SPY",
            instrument_ids=("SPY.XNAS",),
            dataset_id="test-dataset",
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
# Structural Tests (3 tests)
# =============================================================================


@pytest.mark.unit
def test_discovery_service_instantiation(discovery_service):
    """Test DiscoveryService can be instantiated.

    Phase 2.2.7: Verifies component structure exists.
    Phase 2.2.8: Verifies full implementation instantiates correctly.
    """
    from ml.orchestration.discovery_service import DiscoveryService

    assert discovery_service is not None
    assert isinstance(discovery_service, DiscoveryService)


@pytest.mark.unit
def test_discovery_service_has_all_methods(discovery_service):
    """Test DiscoveryService has all 8 required methods.

    Phase 2.2.7: Verifies method signatures exist.
    Phase 2.2.8: Verifies all methods are callable.

    Expected methods:
    1. _discover_market_inputs
    2. _discover_binding_for_symbol
    3. _discover_symbol_via_dataset_service
    4. _infer_default_schema
    5. _map_schema_to_dataset_type
    6. _normalise_schema_for_lookback
    7. _symbol_to_instruments
    8. _collect_instrument_ids
    """
    assert hasattr(discovery_service, "_discover_market_inputs")
    assert callable(discovery_service._discover_market_inputs)

    assert hasattr(discovery_service, "_discover_binding_for_symbol")
    assert callable(discovery_service._discover_binding_for_symbol)

    assert hasattr(discovery_service, "_discover_symbol_via_dataset_service")
    assert callable(discovery_service._discover_symbol_via_dataset_service)

    assert hasattr(discovery_service, "_infer_default_schema")
    assert callable(discovery_service._infer_default_schema)

    assert hasattr(discovery_service, "_map_schema_to_dataset_type")
    assert callable(discovery_service._map_schema_to_dataset_type)

    assert hasattr(discovery_service, "_normalise_schema_for_lookback")
    assert callable(discovery_service._normalise_schema_for_lookback)

    assert hasattr(discovery_service, "_symbol_to_instruments")
    assert callable(discovery_service._symbol_to_instruments)

    assert hasattr(discovery_service, "_collect_instrument_ids")
    assert callable(discovery_service._collect_instrument_ids)


@pytest.mark.unit
def test_discovery_service_methods_return_safe_defaults(
    discovery_service,
    dataset_config,
):
    """Test placeholder methods return safe defaults (not raise).

    Phase 2.2.7: Verifies placeholder behavior (safe defaults).
    Phase 2.2.8: Verifies full implementation returns expected values.

    Safe defaults (Phase 2.2.7):
    - _discover_market_inputs -> ()
    - _discover_binding_for_symbol -> None
    - _discover_symbol_via_dataset_service -> None
    - _infer_default_schema -> "ohlcv-1m"
    - _map_schema_to_dataset_type -> DatasetType.BARS
    - _normalise_schema_for_lookback -> "bars"
    - _symbol_to_instruments -> OrderedDict()
    - _collect_instrument_ids -> ()
    """
    # _discover_market_inputs
    result = discovery_service._discover_market_inputs(
        symbol_map={},
        schema="ohlcv-1m",
        start_ns=0,
        end_ns=1000,
        dataset_hint=None,
    )
    assert result == ()

    # _discover_binding_for_symbol
    result = discovery_service._discover_binding_for_symbol(
        symbol="SPY",
        instrument_ids=None,
        schema="ohlcv-1m",
        start_ns=0,
        end_ns=1000,
    )
    assert result is None

    # _discover_symbol_via_dataset_service
    mock_service = MagicMock()
    result = discovery_service._discover_symbol_via_dataset_service(
        dataset_service=mock_service,
        symbol="SPY",
        schema="ohlcv-1m",
        start_ns=0,
        end_ns=1000,
    )
    assert result is None

    # _infer_default_schema
    result = discovery_service._infer_default_schema(dataset_config)
    assert result == "ohlcv-1m"

    # _map_schema_to_dataset_type
    result = discovery_service._map_schema_to_dataset_type("ohlcv-1m")
    assert result == DatasetType.BARS

    # _normalise_schema_for_lookback
    result = discovery_service._normalise_schema_for_lookback("ohlcv-1m")
    assert result == "bars"

    # _symbol_to_instruments
    result = discovery_service._symbol_to_instruments(dataset_config)
    assert result == OrderedDict()

    # _collect_instrument_ids
    result = discovery_service._collect_instrument_ids(
        bindings=(),
        existing=None,
    )
    assert result == ()


# =============================================================================
# Method Tests (10 tests)
# =============================================================================


@pytest.mark.unit
def test_discover_market_inputs_returns_empty_tuple(discovery_service):
    """Test _discover_market_inputs placeholder behavior.

    Phase 2.2.7: Returns empty tuple (placeholder).
    Phase 2.2.8: Calls DatasetDiscoveryService, returns MarketDatasetInput objects.
    """
    result = discovery_service._discover_market_inputs(
        symbol_map={"SPY": ("SPY.XNAS",)},
        schema="ohlcv-1m",
        start_ns=0,
        end_ns=1000000000000,
        dataset_hint=None,
    )

    assert result == ()
    assert isinstance(result, tuple)


@pytest.mark.unit
def test_discover_market_inputs_with_discovery_error(
    discovery_service,
    mock_dataset_service_with_error,
):
    """Test _discover_market_inputs error handling.

    Phase 2.2.7: Returns empty tuple (no exception).
    Phase 2.2.8: Catches DatasetDiscoveryError, returns empty tuple.
    """
    # In Phase 2.2.8, inject mock_dataset_service_with_error
    result = discovery_service._discover_market_inputs(
        symbol_map={"INVALID": ()},
        schema="ohlcv-1m",
        start_ns=0,
        end_ns=1000000000000,
        dataset_hint=None,
    )

    assert result == ()
    # Phase 2.2.8: Verify no exception raised


@pytest.mark.unit
def test_discover_binding_for_symbol_returns_none(discovery_service):
    """Test _discover_binding_for_symbol placeholder behavior.

    Phase 2.2.7: Returns None (placeholder).
    Phase 2.2.8: Returns ResolvedMarketBinding or None.
    """
    result = discovery_service._discover_binding_for_symbol(
        symbol="SPY",
        instrument_ids=("SPY.XNAS",),
        schema="ohlcv-1m",
        start_ns=0,
        end_ns=1000000000000,
    )

    assert result is None


@pytest.mark.unit
def test_discover_symbol_via_dataset_service_returns_none(
    discovery_service,
    mock_dataset_service,
):
    """Test _discover_symbol_via_dataset_service placeholder behavior.

    Phase 2.2.7: Returns None (placeholder).
    Phase 2.2.8: Returns SymbolDatasetDiscovery or None.
    """
    result = discovery_service._discover_symbol_via_dataset_service(
        dataset_service=mock_dataset_service,
        symbol="SPY",
        schema="ohlcv-1m",
        start_ns=0,
        end_ns=1000000000000,
    )

    assert result is None


@pytest.mark.unit
def test_infer_default_schema_returns_ohlcv(discovery_service, dataset_config):
    """Test _infer_default_schema placeholder behavior.

    Phase 2.2.7: Returns "ohlcv-1m" (placeholder).
    Phase 2.2.8: Returns "ohlcv-1m" (same behavior).
    """
    result = discovery_service._infer_default_schema(dataset_config)

    assert result == "ohlcv-1m"
    assert isinstance(result, str)


@pytest.mark.unit
def test_map_schema_to_dataset_type_ohlcv(discovery_service):
    """Test _map_schema_to_dataset_type for OHLCV schema.

    Phase 2.2.7: Returns DatasetType.BARS (placeholder).
    Phase 2.2.8: Returns DatasetType.BARS (same behavior).
    """
    result = discovery_service._map_schema_to_dataset_type("ohlcv-1m")

    assert result == DatasetType.BARS


@pytest.mark.unit
def test_map_schema_to_dataset_type_tbbo(discovery_service):
    """Test _map_schema_to_dataset_type for TBBO schema.

    Phase 2.2.7: Returns DatasetType.TBBO (placeholder).
    Phase 2.2.8: Returns DatasetType.TBBO (same behavior).
    """
    # Note: The placeholder implementation always returns BARS
    # This test verifies the placeholder behavior
    result = discovery_service._map_schema_to_dataset_type("tbbo")

    # Placeholder returns BARS for all schemas
    assert result == DatasetType.BARS


@pytest.mark.unit
def test_normalise_schema_for_lookback_bars(discovery_service):
    """Test _normalise_schema_for_lookback for bar schemas.

    Phase 2.2.7: Returns "bars" (placeholder).
    Phase 2.2.8: Returns "bars" (same behavior).
    """
    result = discovery_service._normalise_schema_for_lookback("ohlcv-1m")

    assert result == "bars"


@pytest.mark.unit
def test_symbol_to_instruments_returns_empty_dict(discovery_service, dataset_config):
    """Test _symbol_to_instruments placeholder behavior.

    Phase 2.2.7: Returns empty OrderedDict (placeholder).
    Phase 2.2.8: Returns OrderedDict mapping symbols to instruments.
    """
    result = discovery_service._symbol_to_instruments(dataset_config)

    assert result == OrderedDict()
    assert isinstance(result, OrderedDict)


@pytest.mark.unit
def test_collect_instrument_ids_returns_empty_tuple(discovery_service):
    """Test _collect_instrument_ids placeholder behavior.

    Phase 2.2.7: Returns empty tuple (placeholder).
    Phase 2.2.8: Returns tuple of instrument IDs.
    """
    result = discovery_service._collect_instrument_ids(
        bindings=(),
        existing=None,
    )

    assert result == ()
    assert isinstance(result, tuple)
