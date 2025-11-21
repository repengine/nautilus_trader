#!/usr/bin/env python3

"""
Tests for BindingResolver component.

Ensures comprehensive coverage of binding resolution, coverage validation,
priority selection, and policy enforcement.

"""

from datetime import UTC
from datetime import datetime
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from ml.config.market_data import MarketDatasetInput
from ml.data.ingest.service import IngestionError
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.orchestration.binding_resolver import BindingResolver
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.discovery_client import DiscoveryClient
from ml.registry.dataclasses import StorageKind

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

# ========================================================================
# Fixtures
# ========================================================================

@pytest.fixture
def mock_coverage_provider():
    """Create mock coverage provider."""
    provider = Mock()
    provider.read_bucket_coverage = Mock()
    return provider

@pytest.fixture
def mock_ingestion_service():
    """Create mock ingestion service."""
    service = Mock()
    service.get_available_range_ns = Mock()
    service.estimate_cost_usd = Mock()
    return service

@pytest.fixture
def mock_discovery_client():
    """Create mock discovery client."""
    client = Mock(spec=DiscoveryClient)
    client.discover_market_inputs = Mock()
    client.discover_binding_for_symbol = Mock()
    return client

@pytest.fixture
def binding_resolver(mock_coverage_provider, mock_ingestion_service, mock_discovery_client):
    """Create binding resolver with mocked dependencies."""
    return BindingResolver(
        coverage_provider=mock_coverage_provider,
        ingestion_service=mock_ingestion_service,
        discovery_client=mock_discovery_client,
    )

@pytest.fixture
def binding_resolver_no_deps():
    """Create binding resolver with no dependencies."""
    return BindingResolver(
        coverage_provider=None,
        ingestion_service=None,
        discovery_client=None,
    )

@pytest.fixture
def sample_binding():
    """Create sample resolved market binding."""
    return ResolvedMarketBinding(
        binding_id="test-binding",
        symbol="AAPL",
        instrument_ids=("AAPL.XNAS",),
        dataset_id="XNAS.ITCH",
        descriptor_id="test-descriptor",
        schema="ohlcv-1m",
        storage_kind=StorageKind.POSTGRES,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="test",
    )

@pytest.fixture
def sample_config(tmp_path):
    """Create sample dataset build config."""
    return DatasetBuildConfig(
        symbols="AAPL,MSFT",
        start_iso="2023-01-01",
        end_iso="2023-12-31",
        data_dir=str(tmp_path / "data"),
        out_dir=str(tmp_path / "out"),
    )

# ========================================================================
# Test Initialization
# ========================================================================

def test_binding_resolver_init():
    """Test binding resolver initialization."""
    resolver = BindingResolver()
    assert resolver.coverage is None
    assert resolver.service is None
    assert resolver.discovery_client is None
    assert hasattr(resolver, "bindings_resolved_counter")
    assert hasattr(resolver, "binding_selection_time")

def test_binding_resolver_init_with_deps(mock_coverage_provider, mock_ingestion_service, mock_discovery_client):
    """Test binding resolver initialization with dependencies."""
    resolver = BindingResolver(
        coverage_provider=mock_coverage_provider,
        ingestion_service=mock_ingestion_service,
        discovery_client=mock_discovery_client,
    )
    assert resolver.coverage is mock_coverage_provider
    assert resolver.service is mock_ingestion_service
    assert resolver.discovery_client is mock_discovery_client

# ========================================================================
# Test _binding_priority_key
# ========================================================================

def test_binding_priority_key_equs_mini():
    """Test priority key for EQUS.MINI dataset."""
    binding = ResolvedMarketBinding(
        binding_id="test",
        symbol="AAPL",
        instrument_ids=(),
        dataset_id="EQUS.MINI",
        descriptor_id=None,
        schema="ohlcv-1m",
        storage_kind=StorageKind.POSTGRES,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="test",
    )
    priority = BindingResolver._binding_priority_key(binding)
    assert priority == (0, "EQUS.MINI")

def test_binding_priority_key_xnas_itch():
    """Test priority key for XNAS.ITCH dataset."""
    binding = ResolvedMarketBinding(
        binding_id="test",
        symbol="AAPL",
        instrument_ids=(),
        dataset_id="XNAS.ITCH",
        descriptor_id=None,
        schema="ohlcv-1m",
        storage_kind=StorageKind.POSTGRES,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="test",
    )
    priority = BindingResolver._binding_priority_key(binding)
    assert priority == (1, "XNAS.ITCH")

def test_binding_priority_key_other_dataset():
    """Test priority key for other datasets."""
    binding = ResolvedMarketBinding(
        binding_id="test",
        symbol="AAPL",
        instrument_ids=(),
        dataset_id="OTHER.DATASET",
        descriptor_id=None,
        schema="ohlcv-1m",
        storage_kind=StorageKind.POSTGRES,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="test",
    )
    priority = BindingResolver._binding_priority_key(binding)
    assert priority == (2, "OTHER.DATASET")

# ========================================================================
# Test _binding_allowed
# ========================================================================

def test_binding_allowed_no_schema(binding_resolver, sample_binding):
    """Test binding allowed with no schema and no default."""
    binding = ResolvedMarketBinding(
        binding_id="test",
        symbol="AAPL",
        instrument_ids=(),
        dataset_id="XNAS.ITCH",
        descriptor_id=None,
        schema="",  # Empty schema
        storage_kind=StorageKind.POSTGRES,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="test",
    )
    result = binding_resolver._binding_allowed(
        binding=binding,
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
        symbol="AAPL",
        default_schema="",  # No default either
    )
    assert result is False

def test_binding_allowed_no_service(binding_resolver_no_deps, sample_binding):
    """Test binding allowed with no ingestion service."""
    result = binding_resolver_no_deps._binding_allowed(
        binding=sample_binding,
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
        symbol="AAPL",
        default_schema="ohlcv-1m",
    )
    assert result is True

def test_binding_allowed_ingestion_error(binding_resolver, mock_ingestion_service, sample_binding):
    """Test binding allowed with ingestion error."""
    mock_ingestion_service.get_available_range_ns.side_effect = IngestionError("Not available")

    result = binding_resolver._binding_allowed(
        binding=sample_binding,
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
        symbol="AAPL",
        default_schema="ohlcv-1m",
    )
    assert result is False

def test_binding_allowed_outside_coverage_start(binding_resolver, mock_ingestion_service, sample_binding):
    """Test binding allowed outside coverage (before start)."""
    mock_ingestion_service.get_available_range_ns.return_value = (
        1_700_086_400_000_000_000,  # available_start > end
        1_700_172_800_000_000_000,
    )

    result = binding_resolver._binding_allowed(
        binding=sample_binding,
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_043_200_000_000_000,  # Before available_start
        symbol="AAPL",
        default_schema="ohlcv-1m",
    )
    assert result is False

def test_binding_allowed_outside_coverage_end(binding_resolver, mock_ingestion_service, sample_binding):
    """Test binding allowed outside coverage (after end)."""
    mock_ingestion_service.get_available_range_ns.return_value = (
        1_700_000_000_000_000_000,
        1_700_086_400_000_000_000,  # available_end < start
    )

    result = binding_resolver._binding_allowed(
        binding=sample_binding,
        start_ns=1_700_172_800_000_000_000,  # After available_end
        end_ns=1_700_259_200_000_000_000,
        symbol="AAPL",
        default_schema="ohlcv-1m",
    )
    assert result is False

def test_binding_allowed_cost_policy_rejected(binding_resolver, mock_ingestion_service, sample_binding):
    """Test binding allowed rejected by cost policy."""
    mock_ingestion_service.get_available_range_ns.return_value = (
        1_700_000_000_000_000_000,
        1_700_259_200_000_000_000,
    )
    mock_ingestion_service.estimate_cost_usd.side_effect = IngestionError("Cost policy")

    result = binding_resolver._binding_allowed(
        binding=sample_binding,
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
        symbol="AAPL",
        default_schema="ohlcv-1m",
    )
    assert result is False

def test_binding_allowed_nonzero_cost(binding_resolver, mock_ingestion_service, sample_binding):
    """Test binding allowed rejected due to non-zero cost."""
    mock_ingestion_service.get_available_range_ns.return_value = (
        1_700_000_000_000_000_000,
        1_700_259_200_000_000_000,
    )
    mock_ingestion_service.estimate_cost_usd.return_value = 10.50

    result = binding_resolver._binding_allowed(
        binding=sample_binding,
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
        symbol="AAPL",
        default_schema="ohlcv-1m",
    )
    assert result is False

def test_binding_allowed_zero_cost(binding_resolver, mock_ingestion_service, sample_binding):
    """Test binding allowed with zero cost."""
    mock_ingestion_service.get_available_range_ns.return_value = (
        1_700_000_000_000_000_000,
        1_700_259_200_000_000_000,
    )
    mock_ingestion_service.estimate_cost_usd.return_value = 0.0

    result = binding_resolver._binding_allowed(
        binding=sample_binding,
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
        symbol="AAPL",
        default_schema="ohlcv-1m",
    )
    assert result is True

def test_binding_allowed_availability_check_fails(binding_resolver, mock_ingestion_service, sample_binding):
    """Test binding allowed when availability check fails."""
    mock_ingestion_service.get_available_range_ns.side_effect = Exception("Service error")

    # Should continue and not reject immediately
    result = binding_resolver._binding_allowed(
        binding=sample_binding,
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
        symbol="AAPL",
        default_schema="ohlcv-1m",
    )
    assert result is True

# ========================================================================
# Test filter_candidate_bindings
# ========================================================================

def test_filter_candidate_bindings_empty(binding_resolver):
    """Test filtering empty candidate list."""
    result = binding_resolver.filter_candidate_bindings(
        candidates=(),
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
        symbol="AAPL",
        default_schema="ohlcv-1m",
    )
    assert result == ()

def test_filter_candidate_bindings_all_allowed(binding_resolver, mock_ingestion_service):
    """Test filtering when all candidates are allowed."""
    candidates = (
        ResolvedMarketBinding(
            binding_id="test1",
            symbol="AAPL",
            instrument_ids=(),
            dataset_id="EQUS.MINI",
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
            binding_id="test2",
            symbol="AAPL",
            instrument_ids=(),
            dataset_id="XNAS.ITCH",
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

    mock_ingestion_service.get_available_range_ns.return_value = (
        1_700_000_000_000_000_000,
        1_700_259_200_000_000_000,
    )
    mock_ingestion_service.estimate_cost_usd.return_value = 0.0

    result = binding_resolver.filter_candidate_bindings(
        candidates=candidates,
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
        symbol="AAPL",
        default_schema="ohlcv-1m",
    )

    assert len(result) == 2
    # Should be sorted by priority (EQUS.MINI first)
    assert result[0].dataset_id == "EQUS.MINI"
    assert result[1].dataset_id == "XNAS.ITCH"

def test_filter_candidate_bindings_some_rejected(binding_resolver, mock_ingestion_service):
    """Test filtering when some candidates are rejected."""
    candidates = (
        ResolvedMarketBinding(
            binding_id="test1",
            symbol="AAPL",
            instrument_ids=(),
            dataset_id="EQUS.MINI",
            descriptor_id=None,
            schema="",  # Will use default schema
            storage_kind=StorageKind.POSTGRES,
            license_start=None,
            license_end=None,
            start=None,
            end=None,
            source="test",
        ),
        ResolvedMarketBinding(
            binding_id="test2",
            symbol="AAPL",
            instrument_ids=(),
            dataset_id="XNAS.ITCH",
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

    # First binding will have non-zero cost and be rejected
    def mock_cost_side_effect(dataset, schema, symbols, start, end):
        if dataset == "EQUS.MINI":
            return 10.0  # Non-zero cost - will be rejected
        return 0.0

    mock_ingestion_service.get_available_range_ns.return_value = (
        1_700_000_000_000_000_000,
        1_700_259_200_000_000_000,
    )
    mock_ingestion_service.estimate_cost_usd.side_effect = mock_cost_side_effect

    result = binding_resolver.filter_candidate_bindings(
        candidates=candidates,
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
        symbol="AAPL",
        default_schema="ohlcv-1m",
    )

    assert len(result) == 1
    assert result[0].dataset_id == "XNAS.ITCH"

# ========================================================================
# Test select_binding_with_coverage
# ========================================================================

def test_select_binding_with_coverage_no_provider(binding_resolver_no_deps, sample_binding):
    """Test selecting binding with no coverage provider."""
    result = binding_resolver_no_deps.select_binding_with_coverage(
        candidates=(sample_binding,),
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
    )
    assert result is None

def test_select_binding_with_coverage_found(binding_resolver, mock_coverage_provider):
    """Test selecting binding with coverage found."""
    binding = ResolvedMarketBinding(
        binding_id="test",
        symbol="AAPL",
        instrument_ids=("AAPL.XNAS",),
        dataset_id="XNAS.ITCH",
        descriptor_id=None,
        schema="ohlcv-1m",
        storage_kind=StorageKind.POSTGRES,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="test",
    )

    mock_coverage_provider.read_bucket_coverage.return_value = {"bucket1", "bucket2"}

    result = binding_resolver.select_binding_with_coverage(
        candidates=(binding,),
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
    )

    assert result is binding

def test_select_binding_with_coverage_not_found(binding_resolver, mock_coverage_provider):
    """Test selecting binding with no coverage found."""
    binding = ResolvedMarketBinding(
        binding_id="test",
        symbol="AAPL",
        instrument_ids=("AAPL.XNAS",),
        dataset_id="XNAS.ITCH",
        descriptor_id=None,
        schema="ohlcv-1m",
        storage_kind=StorageKind.POSTGRES,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="test",
    )

    mock_coverage_provider.read_bucket_coverage.return_value = set()

    result = binding_resolver.select_binding_with_coverage(
        candidates=(binding,),
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
    )

    assert result is None

def test_select_binding_with_coverage_no_schema(binding_resolver, mock_coverage_provider):
    """Test selecting binding with no schema."""
    binding = ResolvedMarketBinding(
        binding_id="test",
        symbol="AAPL",
        instrument_ids=("AAPL.XNAS",),
        dataset_id="XNAS.ITCH",
        descriptor_id=None,
        schema="",  # Empty schema
        storage_kind=StorageKind.POSTGRES,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="test",
    )

    result = binding_resolver.select_binding_with_coverage(
        candidates=(binding,),
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
    )

    assert result is None

def test_select_binding_with_coverage_lookup_fails(binding_resolver, mock_coverage_provider):
    """Test selecting binding when coverage lookup fails."""
    binding = ResolvedMarketBinding(
        binding_id="test",
        symbol="AAPL",
        instrument_ids=("AAPL.XNAS",),
        dataset_id="XNAS.ITCH",
        descriptor_id=None,
        schema="ohlcv-1m",
        storage_kind=StorageKind.POSTGRES,
        license_start=None,
        license_end=None,
        start=None,
        end=None,
        source="test",
    )

    mock_coverage_provider.read_bucket_coverage.side_effect = Exception("Lookup failed")

    result = binding_resolver.select_binding_with_coverage(
        candidates=(binding,),
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
    )

    assert result is None

# ========================================================================
# Test resolve_market_inputs
# ========================================================================

def test_resolve_market_inputs_from_config(binding_resolver, sample_config, tmp_path):
    """Test resolving market inputs from config."""
    market_inputs = (
        MarketDatasetInput(
            descriptor_id="test",
            dataset_id="XNAS.ITCH",
            symbols=("AAPL",),
            schema_override="ohlcv-1m",
        ),
    )

    cfg = DatasetBuildConfig(
        symbols="AAPL",
        start_iso="2023-01-01",
        end_iso="2023-12-31",
        data_dir=str(tmp_path / "data"),
        out_dir=str(tmp_path / "out"),
        market_inputs=market_inputs,
    )

    # When market_inputs are provided in config, resolve should use them
    resolved_inputs, resolved_bindings = binding_resolver.resolve_market_inputs(
        cfg=cfg,
        symbol_map={"AAPL": ("AAPL.XNAS",)},
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
    )

    assert resolved_inputs == market_inputs
    # Bindings may or may not be resolved depending on the availability of resolve_market_dataset_bindings
    assert isinstance(resolved_bindings, tuple)

def test_resolve_market_inputs_no_discovery_client(binding_resolver_no_deps, sample_config):
    """Test resolving market inputs with no discovery client."""
    resolved_inputs, resolved_bindings = binding_resolver_no_deps.resolve_market_inputs(
        cfg=sample_config,
        symbol_map={"AAPL": ("AAPL.XNAS",)},
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
    )

    assert resolved_inputs is None
    assert resolved_bindings == ()

# ========================================================================
# Test Protocol Conformance
# ========================================================================

def test_binding_resolver_conforms_to_protocol():
    """Test that BindingResolver conforms to BindingResolverProtocol."""
    # Protocol conformance is validated by mypy at type-check time
    # Runtime checking of protocols is not reliable in Python
    resolver = BindingResolver()
    assert hasattr(resolver, "resolve_market_inputs")
    assert hasattr(resolver, "filter_candidate_bindings")
    assert hasattr(resolver, "select_binding_with_coverage")
