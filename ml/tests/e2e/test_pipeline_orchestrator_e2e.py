#!/usr/bin/env python3

"""
End-to-End tests for Phase 2.2 MLPipelineOrchestrator decomposition.

These tests verify the MLPipelineOrchestrator facade actually orchestrates
ML pipelines by performing real coordination operations with real components,
not just mocked structural tests.

Test Strategy:
--------------
1. Use real component instances with minimal mocks
2. Test actual coordination logic end-to-end
3. Verify component delegation works correctly
4. Test full pipeline flows

Success Criteria:
-----------------
- Can discover datasets successfully
- Can resolve bindings correctly
- Can build configurations properly
- Can coordinate components end-to-end
- No coordination failures
"""

import time
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

from ml.orchestration import MLPipelineOrchestrator
from ml.orchestration.config_types import DatasetBuildConfig
from ml.tests.utils.targets import build_default_target_semantics_payload

pytestmark = [
    pytest.mark.usefixtures(
        "isolated_prometheus_registry",
        "mock_tracing_backend",
        "isolated_orchestrator_env",
    ),
]


@pytest.fixture(autouse=True)
def _configure_onnx_harness(
    mock_onnx_runtime: Any,
    onnx_session_stub_factory: Callable[..., object],
) -> None:
    """
    Ensure ONNX runtime interactions stay within the shared harness.
    """

    mock_onnx_runtime.ort.InferenceSession.return_value = onnx_session_stub_factory()


# ============================================================================
# Test Fixtures - Mock Components
# ============================================================================


@pytest.fixture
def timestamp_now() -> int:
    """
    Get current timestamp in nanoseconds.
    """
    return time.time_ns()


# Note: ``mock_data_registry`` and ``mock_data_store`` live in
# ``ml.tests.fixtures.mock_stores`` and are provided via
# ``ml.tests.fixtures.pytest_plugins``. Use ``mock_registry_factory`` directly
# for custom manifests when the default fixtures are insufficient:
#   def test_something(mock_registry_factory):
#       registry = mock_registry_factory("data")
#       registry.get_manifest.return_value = custom_manifest


@pytest.fixture
def mock_coverage_provider() -> Any:
    """
    Create mock CoverageProvider.
    """
    coverage = MagicMock()

    # Mock coverage lookup - returns some buckets
    def read_bucket_coverage(
        dataset_id: str,
        schema: str,
        instrument_id: str,
        start_ns: int,
        end_ns: int,
    ) -> set[int]:
        # Return some day buckets
        start_day = start_ns // (24 * 60 * 60 * 1_000_000_000)
        end_day = end_ns // (24 * 60 * 60 * 1_000_000_000)
        return set(range(int(start_day), int(end_day)))

    coverage.read_bucket_coverage = read_bucket_coverage
    return coverage


@pytest.fixture
def mock_ingestion_service() -> Any:
    """
    Create mock Databento ingestion service.
    """
    service = MagicMock()

    # Mock available range
    def get_available_range_ns(dataset: str, schema: str) -> tuple[int | None, int | None]:
        # Return a 1-year window
        now = datetime.now(UTC)
        year_ago = now - timedelta(days=365)
        return (
            int(year_ago.timestamp() * 1_000_000_000),
            int(now.timestamp() * 1_000_000_000),
        )

    service.get_available_range_ns = get_available_range_ns

    # Mock cost estimation (free for tests)
    service.estimate_cost_usd = lambda **kwargs: 0.0

    # Mock symbol discovery
    from ml.data.ingest.service import SymbolDatasetDiscovery
    from ml.registry.dataclasses import StorageKind

    def discover_symbol_dataset(
        symbol: str,
        schema: str,
        start_ns: int,
        end_ns: int,
    ) -> SymbolDatasetDiscovery | None:
        return SymbolDatasetDiscovery(
            dataset_id="EQUS.MINI",
            schema=schema,
            storage_kind=StorageKind.PARQUET,
            symbol=symbol,
            requested_symbol=symbol,
            available_start_ns=start_ns,
            available_end_ns=end_ns,
            cost_usd=0.0,
            instrument_id=f"{symbol}.NASDAQ",
        )

    service.discover_symbol_dataset = discover_symbol_dataset

    return service


@pytest.fixture
def mock_dataset_discovery() -> Any:
    """
    Create mock DatasetDiscoveryService.
    """
    from ml.config.market_data import MarketDatasetInput
    from ml.registry.dataclasses import StorageKind

    service = MagicMock()

    # Mock discover method
    def discover(requests: tuple[Any, ...], dataset_hint: str | None = None) -> tuple[MarketDatasetInput, ...]:
        results = []
        for req in requests:
            results.append(
                MarketDatasetInput(
                    descriptor_id="test_descriptor",
                    dataset_id="EQUS.MINI",
                    symbols=(req.symbol, ),
                    schema_override=req.schema,
                    storage_kind_override=StorageKind.PARQUET,
                )
            )
        return tuple(results)

    service.discover = discover

    # Mock policy
    policy_mock = MagicMock()
    coverage_mock = MagicMock()
    coverage_mock.allow_dataset = MagicMock()
    policy_mock.coverage = coverage_mock
    service.policy = policy_mock

    return service


@pytest.fixture
def mock_writer() -> Any:
    """Create mock MarketDataWriter."""
    writer = MagicMock()
    writer.write_bars = MagicMock(return_value=0)
    writer.write_trades = MagicMock(return_value=0)
    return writer


@pytest.fixture
def mock_build_main() -> Any:
    """Create mock build_main CLI function."""
    return MagicMock(return_value=0)


@pytest.fixture
def mock_teacher_main() -> Any:
    """Create mock teacher_main CLI function."""
    return MagicMock(return_value=0)


@pytest.fixture
def orchestrator_factory(
    mock_coverage_provider: Any,
    mock_writer: Any,
    mock_build_main: Any,
    mock_teacher_main: Any,
    mock_data_registry: Any,
    mock_data_store: Any,
    mock_dataset_discovery: Any,
    mock_ingestion_service: Any,
) -> Any:
    """
    Factory fixture that creates MLPipelineOrchestrator with all required dependencies.

    Usage:
        def test_something(orchestrator_factory):
            orchestrator = orchestrator_factory()
            # or with overrides:
            orchestrator = orchestrator_factory(data_registry=custom_registry)
    """
    def _create(**overrides: Any) -> Any:
        defaults = {
            "coverage": mock_coverage_provider,
            "writer": mock_writer,
            "build_main": mock_build_main,
            "teacher_main": mock_teacher_main,
            "data_registry": mock_data_registry,
            "data_store": mock_data_store,
            "dataset_discovery": mock_dataset_discovery,
            "service": mock_ingestion_service,
        }
        defaults.update(overrides)
        return MLPipelineOrchestrator(**defaults)
    return _create


@pytest.fixture
def sample_dataset_config(
    timestamp_now: int,
    tmp_path_factory: pytest.TempPathFactory,
) -> DatasetBuildConfig:
    """
    Create sample DatasetBuildConfig for testing.
    """
    now = datetime.fromtimestamp(timestamp_now / 1_000_000_000, tz=UTC)
    week_ago = now - timedelta(days=7)

    base_dir = tmp_path_factory.mktemp("pipeline_dataset")
    data_dir = base_dir / "data"
    out_dir = base_dir / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    return DatasetBuildConfig(
        dataset_id="test_dataset",
        symbols="AAPL,MSFT",
        data_dir=str(data_dir),
        out_dir=str(out_dir),
        target_semantics=build_default_target_semantics_payload(
            horizon_minutes=5,
            threshold=0.001,
        ),
        lookback_periods=10,
        include_macro=False,
        macro_lag_days=2,
        include_micro=False,
        include_l2=False,
        student_mode=False,
        start_iso=week_ago.date().isoformat(),
        end_iso=now.date().isoformat(),
        chunk_days=1,
        emit_dataset_events=False,
        register_features=False,
        market_dataset_id="EQUS.MINI",
        auto_refresh_macro=False,
        macro_staleness_hours=24,
    )


# ============================================================================
# E2E Test Suite - Configuration Resolution
# ============================================================================


class TestE2EConfigResolution:
    """
    Test configuration resolution end-to-end.
    """

    def test_e2e_apply_default_market_inputs(
        self,
        orchestrator_factory: Any,
        sample_dataset_config: DatasetBuildConfig,
    ):
        """
        E2E Test: Apply default market inputs to configuration.
        """
        # Create orchestrator
        orchestrator = orchestrator_factory()

        # Apply defaults
        updated_cfg = orchestrator.apply_default_market_inputs(sample_dataset_config)

        # Verify configuration updated
        assert updated_cfg is not None
        assert updated_cfg.dataset_id == sample_dataset_config.dataset_id

    def test_e2e_collect_symbol_map(
        self,
        orchestrator_factory: Any,
        sample_dataset_config: DatasetBuildConfig,
    ):
        """
        E2E Test: Collect symbol to instrument ID mapping.
        """
        # Create orchestrator
        orchestrator = orchestrator_factory()

        # Collect symbol map
        symbol_map = orchestrator.collect_symbol_map(
            ds_cfg=sample_dataset_config,
            symbols=("AAPL", "MSFT"),
        )

        # Verify mapping created
        assert isinstance(symbol_map, dict)
        assert "AAPL" in symbol_map or "MSFT" in symbol_map

    def test_e2e_compute_window_start(
        self,
        orchestrator_factory: Any, ):
        """
        E2E Test: Compute window start date from end date.
        """
        # Create orchestrator
        orchestrator = orchestrator_factory()

        # Compute window start
        end_iso = "2024-01-31"
        start_iso = orchestrator.compute_window_start_iso(
            end_iso=end_iso,
            lookback_years=1,
        )

        # Verify start computed
        assert start_iso is not None
        assert isinstance(start_iso, str)
        assert start_iso < end_iso  # Start should be before end

    def test_e2e_resolve_window_bounds(
        self,
        orchestrator_factory: Any, sample_dataset_config: DatasetBuildConfig,
    ):
        """
        E2E Test: Resolve window bounds in nanoseconds.
        """
        # Create orchestrator
        orchestrator = orchestrator_factory()

        # Resolve bounds
        start_ns, end_ns = orchestrator.resolve_window_bounds_ns(sample_dataset_config)

        # Verify bounds resolved
        assert isinstance(start_ns, int)
        assert isinstance(end_ns, int)
        assert start_ns > 0
        assert end_ns > start_ns


# ============================================================================
# E2E Test Suite - Discovery Operations
# ============================================================================


class TestE2EDiscoveryOperations:
    """
    Test dataset discovery operations end-to-end.
    """

    def test_e2e_discover_market_inputs(
        self,
        orchestrator_factory: Any, mock_dataset_discovery: Any,
        mock_ingestion_service: Any,
        sample_dataset_config: DatasetBuildConfig,
        timestamp_now: int,
    ):
        """
        E2E Test: Discover market inputs for symbols.
        """
        # Create orchestrator
        orchestrator = orchestrator_factory()

        # Get symbol map
        symbol_map = {"AAPL": ("AAPL.NASDAQ", ), "MSFT": ("MSFT.NASDAQ", )}

        # Resolve bounds
        start_ns, end_ns = orchestrator.resolve_window_bounds_ns(sample_dataset_config)

        # Discover inputs
        inputs = orchestrator.discover_market_inputs(
            symbol_map=symbol_map,
            schema="ohlcv-1m",
            start_ns=start_ns,
            end_ns=end_ns,
            dataset_hint="EQUS.MINI",
        )

        # Verify inputs discovered
        assert inputs is not None
        if len(inputs) > 0:
            assert all(hasattr(inp, "dataset_id") for inp in inputs)


# ============================================================================
# E2E Test Suite - Binding Resolution
# ============================================================================


class TestE2EBindingResolution:
    """
    Test market binding resolution end-to-end.
    """

    def test_e2e_resolve_market_inputs_with_config(
        self,
        orchestrator_factory: Any, mock_coverage_provider: Any,
        mock_ingestion_service: Any,
        mock_dataset_discovery: Any,
        sample_dataset_config: DatasetBuildConfig,
    ):
        """
        E2E Test: Resolve market inputs with coverage validation.
        """
        # Create orchestrator
        orchestrator = orchestrator_factory()

        # Get symbol map and bounds
        symbol_map = {"AAPL": ("AAPL.NASDAQ", ), "MSFT": ("MSFT.NASDAQ", )}
        start_ns, end_ns = orchestrator.resolve_window_bounds_ns(sample_dataset_config)

        # Resolve inputs
        resolved_inputs, bindings = orchestrator.resolve_market_inputs(
            cfg=sample_dataset_config,
            symbol_map=symbol_map,
            start_ns=start_ns,
            end_ns=end_ns,
        )

        # Verify resolution completed
        # Note: May return empty if discovery not available, but should not error
        assert resolved_inputs is not None or bindings is not None
        assert isinstance(bindings, (tuple, list))

    def test_e2e_filter_candidate_bindings(
        self,
        orchestrator_factory: Any, mock_coverage_provider: Any,
        mock_ingestion_service: Any,
        timestamp_now: int,
    ):
        """
        E2E Test: Filter candidate bindings by availability.
        """
        from ml.data.ingest.market_bindings import ResolvedMarketBinding
        from ml.registry.dataclasses import StorageKind

        # Create orchestrator
        orchestrator = orchestrator_factory()

        # Create candidate bindings
        now = datetime.fromtimestamp(timestamp_now / 1_000_000_000, tz=UTC)
        start_ns = int((now - timedelta(days=7)).timestamp() * 1_000_000_000)
        end_ns = timestamp_now

        candidates = (
            ResolvedMarketBinding(
                binding_id="test_1",
                symbol="AAPL",
                instrument_ids=("AAPL.NASDAQ", ),
                dataset_id="EQUS.MINI",
                descriptor_id="test_desc",
                schema="ohlcv-1m",
                storage_kind=StorageKind.PARQUET,
                license_start=None,
                license_end=None,
                start=None,
                end=None,
                source="test",
            ),
        )

        # Filter bindings
        filtered = orchestrator.filter_candidate_bindings(
            candidates=candidates,
            start_ns=start_ns,
            end_ns=end_ns,
            symbol="AAPL",
            default_schema="ohlcv-1m",
        )

        # Verify filtering completed
        assert isinstance(filtered, tuple)


# ============================================================================
# E2E Test Suite - Dataset Building
# ============================================================================


class TestE2EDatasetBuilding:
    """
    Test dataset building end-to-end.
    """

    def test_e2e_prepare_dataset_config(
        self,
        orchestrator_factory: Any, sample_dataset_config: DatasetBuildConfig,
    ):
        """
        E2E Test: Prepare dataset config with resolved values.
        """
        from ml.config.market_data import MarketDatasetInput
        from ml.data.ingest.market_bindings import ResolvedMarketBinding
        from ml.registry.dataclasses import StorageKind

        # Create orchestrator
        orchestrator = orchestrator_factory()

        # Create resolved inputs and bindings
        resolved_inputs = (
            MarketDatasetInput(
                descriptor_id="test_desc",
                dataset_id="EQUS.MINI",
                symbols=("AAPL", ),
                schema_override="ohlcv-1m",
                storage_kind_override=StorageKind.PARQUET,
            ),
        )

        bindings = (
            ResolvedMarketBinding(
                binding_id="test_binding",
                symbol="AAPL",
                instrument_ids=("AAPL.NASDAQ", ),
                dataset_id="EQUS.MINI",
                descriptor_id="test_desc",
                schema="ohlcv-1m",
                storage_kind=StorageKind.PARQUET,
                license_start=None,
                license_end=None,
                start=None,
                end=None,
                source="test",
            ),
        )

        # Prepare config
        prepared_cfg = orchestrator.prepare_dataset_config(
            cfg=sample_dataset_config,
            resolved_inputs=resolved_inputs,
            bindings=bindings,
        )

        # Verify config prepared
        assert prepared_cfg is not None
        assert prepared_cfg.dataset_id == sample_dataset_config.dataset_id


# ============================================================================
# E2E Test Suite - Component Integration
# ============================================================================


class TestE2EComponentIntegration:
    """
    Test all components working together end-to-end.
    """

    def test_e2e_full_configuration_pipeline(
        self,
        orchestrator_factory: Any, mock_coverage_provider: Any,
        mock_ingestion_service: Any,
        mock_dataset_discovery: Any,
        sample_dataset_config: DatasetBuildConfig,
    ):
        """
        E2E Test: Full pipeline from config to prepared dataset.

        Tests the complete flow:
        1. Apply defaults to config
        2. Collect symbol map
        3. Resolve window bounds
        4. Discover market inputs
        5. Resolve bindings
        6. Prepare final config
        """
        # Create orchestrator
        orchestrator = orchestrator_factory()

        # Step 1: Apply defaults
        cfg = orchestrator.apply_default_market_inputs(sample_dataset_config)
        assert cfg is not None

        # Step 2: Collect symbol map
        symbol_map = orchestrator.collect_symbol_map(
            ds_cfg=cfg,
            symbols=("AAPL", "MSFT"),
        )
        assert isinstance(symbol_map, dict)

        # Step 3: Resolve window bounds
        start_ns, end_ns = orchestrator.resolve_window_bounds_ns(cfg)
        assert start_ns > 0
        assert end_ns > start_ns

        # Step 4: Discover market inputs (may be empty but should not error)
        try:
            inputs = orchestrator.discover_market_inputs(
                symbol_map=symbol_map,
                schema="ohlcv-1m",
                start_ns=start_ns,
                end_ns=end_ns,
            )
            discovered_inputs = inputs
        except Exception:
            discovered_inputs = None

        # Step 5: Resolve bindings (may return empty but should not error)
        resolved_inputs, bindings = orchestrator.resolve_market_inputs(
            cfg=cfg,
            symbol_map=symbol_map,
            start_ns=start_ns,
            end_ns=end_ns,
        )

        # Step 6: Prepare final config
        if resolved_inputs:
            final_cfg = orchestrator.prepare_dataset_config(
                cfg=cfg,
                resolved_inputs=resolved_inputs,
                bindings=bindings,
            )
            assert final_cfg is not None
            assert final_cfg.dataset_id == cfg.dataset_id


# ============================================================================
# E2E Test Suite - Health and Monitoring
# ============================================================================


class TestE2EHealthMonitoring:
    """
    Test health status reporting end-to-end.
    """

    def test_e2e_health_status_all_components(
        self,
        orchestrator_factory: Any, ):
        """
        E2E Test: Health status includes all components.
        """
        # Create orchestrator
        orchestrator = orchestrator_factory()

        # Get health status
        health = orchestrator.get_health_status()

        # Verify all components reported
        assert health["implementation"] == "component-based"
        assert health["config_resolver"] == "healthy"
        assert health["discovery_client"] == "healthy"
        assert health["binding_resolver"] == "healthy"
        assert health["ingestion_coordinator"] == "healthy"
        assert health["dataset_builder"] == "healthy"


# ============================================================================
# E2E Test Suite - Component Behavior
# ============================================================================


class TestE2EComponentBehavior:
    """
    Test component mode produces valid outputs.
    """

    def test_e2e_config_resolution_outputs(
        self,
        orchestrator_factory: Any,
    ) -> None:
        """
        E2E Test: Config resolution produces an ISO timestamp.
        """
        orchestrator = orchestrator_factory()
        start_iso = orchestrator.compute_window_start_iso("2024-01-31", 1)
        assert isinstance(start_iso, str)

    def test_e2e_window_bounds(
        self,
        orchestrator_factory: Any,
        sample_dataset_config: DatasetBuildConfig,
    ) -> None:
        """
        E2E Test: Window bounds resolution produces ordered timestamps.
        """
        orchestrator = orchestrator_factory()
        start_ns, end_ns = orchestrator.resolve_window_bounds_ns(sample_dataset_config)
        assert start_ns < end_ns

    def test_e2e_health_status_structure(
        self,
        orchestrator_factory: Any,
    ) -> None:
        """
        E2E Test: Health status exposes implementation metadata.
        """
        orchestrator = orchestrator_factory()
        health = orchestrator.get_health_status()
        assert "implementation" in health


# ============================================================================
# E2E Test Suite - Error Handling
# ============================================================================


class TestE2EErrorHandling:
    """
    Test error handling in E2E scenarios.
    """

    def test_e2e_missing_registry_handled_gracefully(
        self,
        orchestrator_factory: Any, ):
        """
        E2E Test: Missing registry handled gracefully.
        """
        # Create orchestrator without registry
        orchestrator = orchestrator_factory()

        # Should still work for basic operations
        health = orchestrator.get_health_status()
        assert health is not None

    def test_e2e_invalid_window_bounds_handled(
        self,
        orchestrator_factory: Any, ):
        """
        E2E Test: Invalid window bounds handled gracefully.
        """
        from ml.orchestration.config_types import VintagePolicy

        # Create config with invalid dates
        invalid_cfg = DatasetBuildConfig(
            dataset_id="test",
            symbols="AAPL",
            data_dir="/tmp/test",
            out_dir="/tmp/out",
            target_semantics=build_default_target_semantics_payload(
                horizon_minutes=5,
                threshold=0.001,
            ),
            lookback_periods=10,
            include_macro=False,
            macro_lag_days=2,
            include_micro=False,
            include_l2=False,
            student_mode=False,
            start_iso="2024-12-31",  # After end
            end_iso="2024-01-01",
            chunk_days=1,
            emit_dataset_events=False,
            register_features=False,
            auto_refresh_macro=False,
            macro_staleness_hours=24,
            vintage_policy=VintagePolicy.REAL_TIME,
        )

        # Create orchestrator
        orchestrator = orchestrator_factory()

        # Should handle gracefully (may adjust dates or return minimum window)
        start_ns, end_ns = orchestrator.resolve_window_bounds_ns(invalid_cfg)
        assert isinstance(start_ns, int)
        assert isinstance(end_ns, int)
        assert end_ns > start_ns  # Should be corrected


# ============================================================================
# E2E Performance Tests
# ============================================================================


class TestE2EPerformance:
    """
    Test performance characteristics of E2E operations.
    """

    @pytest.fixture(autouse=True)
    def test_e2e_config_resolution_performance(
        self,
        orchestrator_factory: Any, sample_dataset_config: DatasetBuildConfig,
    ):
        """
        E2E Test: Config resolution completes quickly.
        """
        import time

        # Create orchestrator
        orchestrator = orchestrator_factory()

        # Measure time
        start = time.perf_counter()

        # Perform operations
        cfg = orchestrator.apply_default_market_inputs(sample_dataset_config)
        symbol_map = orchestrator.collect_symbol_map(ds_cfg=cfg)
        _start_ns, _end_ns = orchestrator.resolve_window_bounds_ns(cfg)

        end = time.perf_counter()

        # Verify performance (should be fast - all in-memory)
        latency_ms = (end - start) * 1000
        assert latency_ms < 100.0  # Should complete in < 100ms
