#!/usr/bin/env python3

"""
Tests for DiscoveryClient component.

Ensures comprehensive coverage of discovery operations, error handling,
and service integration.

"""

from datetime import UTC
from datetime import datetime
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml.config.market_data import MarketDatasetInput
from ml.data.ingest.discovery import DatasetDiscoveryError
from ml.data.ingest.discovery import DiscoveryRequest
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.ingest.service import SymbolDatasetDiscovery
from ml.orchestration.discovery_client import DiscoveryClient
from ml.registry.dataclasses import StorageKind


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def mock_dataset_discovery():
    """Create mock dataset discovery service."""
    service = Mock()
    service.policy = Mock()
    service.policy.coverage = Mock()
    service.policy.coverage.allow_dataset = Mock()
    return service


@pytest.fixture
def mock_ingestion_service():
    """Create mock ingestion service."""
    service = Mock()
    service.discover_symbol_dataset = Mock()
    return service


@pytest.fixture
def discovery_client(mock_dataset_discovery, mock_ingestion_service):
    """Create discovery client with mocked services."""
    return DiscoveryClient(
        dataset_discovery=mock_dataset_discovery,
        ingestion_service=mock_ingestion_service,
    )


@pytest.fixture
def discovery_client_no_services():
    """Create discovery client with no services."""
    return DiscoveryClient(
        dataset_discovery=None,
        ingestion_service=None,
    )


# ========================================================================
# Test Initialization
# ========================================================================


def test_discovery_client_init():
    """Test discovery client initialization."""
    client = DiscoveryClient()
    assert client.dataset_discovery is None
    assert client.service is None
    assert hasattr(client, "discovery_requests_counter")
    assert hasattr(client, "discovery_errors_counter")


def test_discovery_client_init_with_services(mock_dataset_discovery, mock_ingestion_service):
    """Test discovery client initialization with services."""
    client = DiscoveryClient(
        dataset_discovery=mock_dataset_discovery,
        ingestion_service=mock_ingestion_service,
    )
    assert client.dataset_discovery is mock_dataset_discovery
    assert client.service is mock_ingestion_service


# ========================================================================
# Test ns_to_datetime
# ========================================================================


def test_ns_to_datetime():
    """Test nanoseconds to datetime conversion."""
    ns = 1_700_000_000_000_000_000  # Nov 14, 2023
    dt = DiscoveryClient.ns_to_datetime(ns)
    assert isinstance(dt, datetime)
    assert dt.tzinfo is UTC
    assert dt.year == 2023
    assert dt.month == 11


def test_ns_to_datetime_zero():
    """Test nanoseconds to datetime conversion with zero."""
    ns = 0
    dt = DiscoveryClient.ns_to_datetime(ns)
    assert isinstance(dt, datetime)
    assert dt.tzinfo is UTC
    assert dt.year == 1970


# ========================================================================
# Test discover_market_inputs
# ========================================================================


def test_discover_market_inputs_success(discovery_client, mock_dataset_discovery):
    """Test successful market input discovery."""
    symbol_map = {"AAPL": ("AAPL.XNAS",), "MSFT": ("MSFT.XNAS",)}
    schema = "ohlcv-1m"
    start_ns = 1_700_000_000_000_000_000
    end_ns = 1_700_086_400_000_000_000

    # Mock discovery response
    mock_inputs = (
        MarketDatasetInput(
            descriptor_id="test",
            dataset_id="XNAS.ITCH",
            symbols=("AAPL",),
            schema_override="ohlcv-1m",
        ),
        MarketDatasetInput(
            descriptor_id="test",
            dataset_id="XNAS.ITCH",
            symbols=("MSFT",),
            schema_override="ohlcv-1m",
        ),
    )
    mock_dataset_discovery.discover.return_value = mock_inputs

    result = discovery_client.discover_market_inputs(
        symbol_map=symbol_map,
        schema=schema,
        start_ns=start_ns,
        end_ns=end_ns,
        dataset_hint="XNAS.ITCH",
    )

    assert result == mock_inputs
    assert mock_dataset_discovery.discover.called


def test_discover_market_inputs_no_service(discovery_client_no_services):
    """Test market input discovery with no service."""
    symbol_map = {"AAPL": ("AAPL.XNAS",)}
    schema = "ohlcv-1m"
    start_ns = 1_700_000_000_000_000_000
    end_ns = 1_700_086_400_000_000_000

    result = discovery_client_no_services.discover_market_inputs(
        symbol_map=symbol_map,
        schema=schema,
        start_ns=start_ns,
        end_ns=end_ns,
    )

    assert result == ()


def test_discover_market_inputs_invalid_time_range(discovery_client):
    """Test market input discovery with invalid time range."""
    symbol_map = {"AAPL": ("AAPL.XNAS",)}
    schema = "ohlcv-1m"
    start_ns = 1_700_086_400_000_000_000
    end_ns = 1_700_000_000_000_000_000  # end < start

    result = discovery_client.discover_market_inputs(
        symbol_map=symbol_map,
        schema=schema,
        start_ns=start_ns,
        end_ns=end_ns,
    )

    assert result == ()


def test_discover_market_inputs_empty_symbol_map(discovery_client):
    """Test market input discovery with empty symbol map."""
    symbol_map = {}
    schema = "ohlcv-1m"
    start_ns = 1_700_000_000_000_000_000
    end_ns = 1_700_086_400_000_000_000

    result = discovery_client.discover_market_inputs(
        symbol_map=symbol_map,
        schema=schema,
        start_ns=start_ns,
        end_ns=end_ns,
    )

    assert result == ()


def test_discover_market_inputs_discovery_error(discovery_client, mock_dataset_discovery):
    """Test market input discovery with discovery error."""
    symbol_map = {"AAPL": ("AAPL.XNAS",)}
    schema = "ohlcv-1m"
    start_ns = 1_700_000_000_000_000_000
    end_ns = 1_700_086_400_000_000_000

    mock_dataset_discovery.discover.side_effect = DatasetDiscoveryError("Service unavailable")

    result = discovery_client.discover_market_inputs(
        symbol_map=symbol_map,
        schema=schema,
        start_ns=start_ns,
        end_ns=end_ns,
    )

    assert result == ()


def test_discover_market_inputs_applies_coverage_policy(discovery_client, mock_dataset_discovery):
    """Test market input discovery applies coverage policy."""
    symbol_map = {"AAPL": ("AAPL.XNAS",)}
    schema = "ohlcv-1m"
    start_ns = 1_700_000_000_000_000_000
    end_ns = 1_700_086_400_000_000_000

    mock_inputs = (
        MarketDatasetInput(
            descriptor_id="test",
            dataset_id="XNAS.ITCH",
            symbols=("AAPL",),
            schema_override="ohlcv-1m",
        ),
    )
    mock_dataset_discovery.discover.return_value = mock_inputs

    result = discovery_client.discover_market_inputs(
        symbol_map=symbol_map,
        schema=schema,
        start_ns=start_ns,
        end_ns=end_ns,
    )

    assert result == mock_inputs
    mock_dataset_discovery.policy.coverage.allow_dataset.assert_called_with("XNAS.ITCH")


def test_discover_market_inputs_no_coverage_policy(discovery_client, mock_dataset_discovery):
    """Test market input discovery without coverage policy."""
    symbol_map = {"AAPL": ("AAPL.XNAS",)}
    schema = "ohlcv-1m"
    start_ns = 1_700_000_000_000_000_000
    end_ns = 1_700_086_400_000_000_000

    mock_inputs = (
        MarketDatasetInput(
            descriptor_id="test",
            dataset_id="XNAS.ITCH",
            symbols=("AAPL",),
            schema_override="ohlcv-1m",
        ),
    )
    mock_dataset_discovery.discover.return_value = mock_inputs
    mock_dataset_discovery.policy = None  # No policy

    result = discovery_client.discover_market_inputs(
        symbol_map=symbol_map,
        schema=schema,
        start_ns=start_ns,
        end_ns=end_ns,
    )

    assert result == mock_inputs


# ========================================================================
# Test discover_binding_for_symbol
# ========================================================================


def test_discover_binding_for_symbol_success(discovery_client, mock_ingestion_service):
    """Test successful binding discovery for symbol."""
    symbol = "AAPL"
    instrument_ids = ("AAPL.XNAS",)
    schema = "ohlcv-1m"
    start_ns = 1_700_000_000_000_000_000
    end_ns = 1_700_086_400_000_000_000

    # Mock discovery result
    mock_discovery = SymbolDatasetDiscovery(
        dataset_id="XNAS.ITCH",
        schema="ohlcv-1m",
        storage_kind=StorageKind.POSTGRES,
        symbol="AAPL",
        requested_symbol="AAPL",
        available_start_ns=start_ns,
        available_end_ns=end_ns,
        cost_usd=0.0,
        instrument_id="AAPL.XNAS",
    )
    mock_ingestion_service.discover_symbol_dataset.return_value = mock_discovery

    result = discovery_client.discover_binding_for_symbol(
        symbol=symbol,
        instrument_ids=instrument_ids,
        schema=schema,
        start_ns=start_ns,
        end_ns=end_ns,
    )

    assert isinstance(result, ResolvedMarketBinding)
    assert result.symbol == "AAPL"
    assert result.dataset_id == "XNAS.ITCH"
    assert result.schema == "ohlcv-1m"
    assert result.source == "discovered"


def test_discover_binding_for_symbol_no_service(discovery_client_no_services):
    """Test binding discovery with no service."""
    result = discovery_client_no_services.discover_binding_for_symbol(
        symbol="AAPL",
        instrument_ids=("AAPL.XNAS",),
        schema="ohlcv-1m",
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
    )

    assert result is None


def test_discover_binding_for_symbol_empty_schema(discovery_client):
    """Test binding discovery with empty schema."""
    result = discovery_client.discover_binding_for_symbol(
        symbol="AAPL",
        instrument_ids=("AAPL.XNAS",),
        schema="",
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
    )

    assert result is None


def test_discover_binding_for_symbol_discovery_returns_none(discovery_client, mock_ingestion_service):
    """Test binding discovery when discovery returns None."""
    mock_ingestion_service.discover_symbol_dataset.return_value = None

    result = discovery_client.discover_binding_for_symbol(
        symbol="AAPL",
        instrument_ids=("AAPL.XNAS",),
        schema="ohlcv-1m",
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
    )

    assert result is None


def test_discover_binding_for_symbol_discovery_raises(discovery_client, mock_ingestion_service):
    """Test binding discovery when discovery raises exception."""
    mock_ingestion_service.discover_symbol_dataset.side_effect = Exception("Discovery failed")

    result = discovery_client.discover_binding_for_symbol(
        symbol="AAPL",
        instrument_ids=("AAPL.XNAS",),
        schema="ohlcv-1m",
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
    )

    assert result is None


def test_discover_binding_for_symbol_uses_dataset_service_fallback(discovery_client, mock_dataset_discovery, mock_ingestion_service):
    """Test binding discovery uses dataset service fallback."""
    # Remove discovery function from ingestion service
    del mock_ingestion_service.discover_symbol_dataset

    # Mock dataset service discovery
    mock_discovered = Mock()
    mock_discovered.dataset_id = "XNAS.ITCH"
    mock_discovered.schema = "ohlcv-1m"
    mock_discovered.storage_kind = StorageKind.POSTGRES
    mock_discovered.symbol = "AAPL"
    mock_discovered.requested_symbol = "AAPL"
    mock_discovered.available_start_ns = 1_700_000_000_000_000_000
    mock_discovered.available_end_ns = 1_700_086_400_000_000_000
    mock_discovered.cost_usd = 0.0
    mock_discovered.instrument_id = "AAPL.XNAS"
    mock_dataset_discovery.discover_one.return_value = mock_discovered

    result = discovery_client.discover_binding_for_symbol(
        symbol="AAPL",
        instrument_ids=("AAPL.XNAS",),
        schema="ohlcv-1m",
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
    )

    assert isinstance(result, ResolvedMarketBinding)
    assert result.symbol == "AAPL"
    assert result.dataset_id == "XNAS.ITCH"


# ========================================================================
# Test _discover_symbol_via_dataset_service
# ========================================================================


def test_discover_symbol_via_dataset_service_success(discovery_client, mock_dataset_discovery):
    """Test successful symbol discovery via dataset service."""
    mock_discovered = Mock()
    mock_discovered.dataset_id = "XNAS.ITCH"
    mock_discovered.schema = "ohlcv-1m"
    mock_discovered.storage_kind = None  # Test default
    mock_discovered.symbol = "AAPL"
    mock_discovered.requested_symbol = "AAPL"
    mock_discovered.available_start_ns = 1_700_000_000_000_000_000
    mock_discovered.available_end_ns = 1_700_086_400_000_000_000
    mock_discovered.cost_usd = 0.0
    mock_discovered.instrument_id = "AAPL.XNAS"
    mock_dataset_discovery.discover_one.return_value = mock_discovered

    result = discovery_client._discover_symbol_via_dataset_service(
        dataset_service=mock_dataset_discovery,
        symbol="AAPL",
        schema="ohlcv-1m",
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
    )

    assert isinstance(result, SymbolDatasetDiscovery)
    assert result.dataset_id == "XNAS.ITCH"
    assert result.symbol == "AAPL"
    assert result.storage_kind == StorageKind.POSTGRES  # Default


def test_discover_symbol_via_dataset_service_invalid_time_range(discovery_client, mock_dataset_discovery):
    """Test symbol discovery with invalid time range."""
    result = discovery_client._discover_symbol_via_dataset_service(
        dataset_service=mock_dataset_discovery,
        symbol="AAPL",
        schema="ohlcv-1m",
        start_ns=1_700_086_400_000_000_000,
        end_ns=1_700_000_000_000_000_000,  # end < start
    )

    assert result is None


def test_discover_symbol_via_dataset_service_discovery_error(discovery_client, mock_dataset_discovery):
    """Test symbol discovery with discovery error."""
    mock_dataset_discovery.discover_one.side_effect = DatasetDiscoveryError("Not found")

    result = discovery_client._discover_symbol_via_dataset_service(
        dataset_service=mock_dataset_discovery,
        symbol="AAPL",
        schema="ohlcv-1m",
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
    )

    assert result is None


def test_discover_symbol_via_dataset_service_generic_error(discovery_client, mock_dataset_discovery):
    """Test symbol discovery with generic error."""
    mock_dataset_discovery.discover_one.side_effect = Exception("Service error")

    result = discovery_client._discover_symbol_via_dataset_service(
        dataset_service=mock_dataset_discovery,
        symbol="AAPL",
        schema="ohlcv-1m",
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
    )

    assert result is None


# ========================================================================
# Test Protocol Conformance
# ========================================================================


def test_discovery_client_conforms_to_protocol():
    """Test that DiscoveryClient conforms to DiscoveryClientProtocol."""
    # Protocol conformance is validated by mypy at type-check time
    # Runtime checking of protocols is not reliable in Python
    client = DiscoveryClient()
    assert hasattr(client, "discover_market_inputs")
    assert hasattr(client, "discover_binding_for_symbol")
