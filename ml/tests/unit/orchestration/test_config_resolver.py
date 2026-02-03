#!/usr/bin/env python3

"""
Unit tests for ConfigResolver component.

Tests all configuration resolution methods including market input defaults,
symbol mapping, window bounds computation, and dataset config preparation.

"""

from __future__ import annotations

from collections import OrderedDict
from datetime import UTC
from datetime import datetime

import pytest

from ml.config.market_data import MarketDatasetInput
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.orchestration.config_resolver import ConfigResolver
from ml.orchestration.config_types import DatasetBuildConfig
from ml.registry.dataclasses import StorageKind
from ml.tests.utils.targets import build_default_target_semantics_payload

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

TARGET_SEMANTICS = build_default_target_semantics_payload()

@pytest.fixture
def resolver() -> ConfigResolver:
    """Create ConfigResolver instance for testing."""
    return ConfigResolver()

@pytest.fixture
def base_dataset_config() -> DatasetBuildConfig:
    """Create base dataset configuration for testing."""
    return DatasetBuildConfig(
        data_dir="data/test",
        symbols="SPY,AAPL",
        out_dir="output/test",
        dataset_id="test_dataset",
        target_semantics=TARGET_SEMANTICS,
    )

def create_test_binding(
    symbol: str = "SPY",
    instrument_ids: tuple[str, ...] | None = ("SPY.XNAS",),
    dataset_id: str = "test_ds",
    descriptor_id: str | None = "test_desc",
    schema: str | None = "ohlcv-1m",
    storage_kind: StorageKind | None = StorageKind.PARQUET,
    source: str = "test",
) -> ResolvedMarketBinding:
    """Helper to create test ResolvedMarketBinding with all required fields."""
    return ResolvedMarketBinding(
        binding_id=f"{symbol}_{dataset_id}",
        symbol=symbol,
        instrument_ids=instrument_ids or (symbol,),
        dataset_id=dataset_id,
        descriptor_id=descriptor_id,
        schema=schema,
        storage_kind=storage_kind,
        license_start="2020-01-01",
        license_end="2030-12-31",
        start="2020-01-01",
        end="2030-12-31",
        source=source,
    )

class TestApplyDefaultMarketInputs:
    """Test apply_default_market_inputs method."""

    def test_apply_default_market_inputs_when_inputs_exist_returns_unchanged(
        self,
        resolver: ConfigResolver,
        base_dataset_config: DatasetBuildConfig,
    ) -> None:
        """Test that existing market inputs are not overwritten."""
        from dataclasses import replace

        existing_input = MarketDatasetInput(
            descriptor_id="test_desc",
            dataset_id="test_ds",
            symbols=("SPY",),
        )
        cfg = replace(base_dataset_config, market_inputs=(existing_input,))

        result = resolver.apply_default_market_inputs(cfg)

        assert result.market_inputs == (existing_input,)

    def test_apply_default_market_inputs_when_no_dataset_id_returns_unchanged(
        self,
        resolver: ConfigResolver,
        base_dataset_config: DatasetBuildConfig,
    ) -> None:
        """Test that config without market_dataset_id is unchanged."""
        import msgspec

        result = resolver.apply_default_market_inputs(base_dataset_config)
        assert msgspec.to_builtins(result) == msgspec.to_builtins(base_dataset_config)

    def test_apply_default_market_inputs_when_descriptor_not_found_returns_unchanged(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that unknown descriptor returns unchanged config."""
        import msgspec

        cfg = DatasetBuildConfig(
            data_dir="data/test",
            symbols="SPY",
            out_dir="output/test",
            dataset_id="test_dataset",
            market_dataset_id="UNKNOWN_DESCRIPTOR",
            target_semantics=TARGET_SEMANTICS,
        )

        result = resolver.apply_default_market_inputs(cfg)
        assert msgspec.to_builtins(result) == msgspec.to_builtins(cfg)

    def test_apply_default_market_inputs_when_no_symbols_returns_unchanged(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that config without symbols returns unchanged."""
        import msgspec

        cfg = DatasetBuildConfig(
            data_dir="data/test",
            symbols="",
            out_dir="output/test",
            dataset_id="test_dataset",
            market_dataset_id="EQUS.MINI",
            target_semantics=TARGET_SEMANTICS,
        )

        result = resolver.apply_default_market_inputs(cfg)
        assert msgspec.to_builtins(result) == msgspec.to_builtins(cfg)

class TestCollectSymbolMap:
    """Test collect_symbol_map method."""

    def test_collect_symbol_map_from_symbols_only(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test symbol map collection from symbols only."""
        result = resolver.collect_symbol_map(
            ds_cfg=None,
            symbols=("SPY", "AAPL"),
            instruments=None,
            instrument_ids=None,
            market_inputs=None,
        )

        assert result == {"SPY": (), "AAPL": ()}

    def test_collect_symbol_map_from_instruments(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test symbol map collection from instruments."""
        result = resolver.collect_symbol_map(
            ds_cfg=None,
            symbols=None,
            instruments=("SPY.XNAS", "AAPL.XNAS"),
            instrument_ids=None,
            market_inputs=None,
        )

        assert result == {"SPY": ("SPY.XNAS",), "AAPL": ("AAPL.XNAS",)}

    def test_collect_symbol_map_from_instrument_ids(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test symbol map collection from instrument_ids."""
        result = resolver.collect_symbol_map(
            ds_cfg=None,
            symbols=None,
            instruments=None,
            instrument_ids=("SPY.XNAS", "AAPL.XNAS"),
            market_inputs=None,
        )

        assert result == {"SPY": ("SPY.XNAS",), "AAPL": ("AAPL.XNAS",)}

    def test_collect_symbol_map_from_dataset_config(
        self,
        resolver: ConfigResolver,
        base_dataset_config: DatasetBuildConfig,
    ) -> None:
        """Test symbol map collection from dataset config."""
        result = resolver.collect_symbol_map(
            ds_cfg=base_dataset_config,
            symbols=None,
            instruments=None,
            instrument_ids=None,
            market_inputs=None,
        )

        assert "SPY" in result
        assert "AAPL" in result

    def test_collect_symbol_map_from_market_inputs(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test symbol map collection from market inputs."""
        market_input = MarketDatasetInput(
            descriptor_id="test_desc",
            dataset_id="test_ds",
            symbols=("SPY", "AAPL"),
        )

        result = resolver.collect_symbol_map(
            ds_cfg=None,
            symbols=None,
            instruments=None,
            instrument_ids=None,
            market_inputs=(market_input,),
        )

        assert result == {"SPY": (), "AAPL": ()}

    def test_collect_symbol_map_merges_multiple_sources(
        self,
        resolver: ConfigResolver,
        base_dataset_config: DatasetBuildConfig,
    ) -> None:
        """Test that symbol map merges from multiple sources."""
        result = resolver.collect_symbol_map(
            ds_cfg=base_dataset_config,
            symbols=("TSLA",),
            instruments=("SPY.XNAS",),
            instrument_ids=("AAPL.XNAS",),
            market_inputs=None,
        )

        assert "SPY" in result
        assert "AAPL" in result
        assert "TSLA" in result
        assert result["SPY"] == ("SPY.XNAS",)

    def test_collect_symbol_map_deduplicates_instruments(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that duplicate instruments are removed."""
        result = resolver.collect_symbol_map(
            ds_cfg=None,
            symbols=None,
            instruments=("SPY.XNAS", "SPY.XNAS"),
            instrument_ids=("SPY.XNAS",),
            market_inputs=None,
        )

        assert result == {"SPY": ("SPY.XNAS",)}

    def test_collect_symbol_map_handles_empty_inputs(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that empty inputs return empty map."""
        result = resolver.collect_symbol_map(
            ds_cfg=None,
            symbols=None,
            instruments=None,
            instrument_ids=None,
            market_inputs=None,
        )

        assert result == {}

    def test_collect_symbol_map_normalizes_case(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that symbols are normalized to uppercase."""
        result = resolver.collect_symbol_map(
            ds_cfg=None,
            symbols=("spy", "aapl"),
            instruments=None,
            instrument_ids=None,
            market_inputs=None,
        )

        assert result == {"SPY": (), "AAPL": ()}

    def test_collect_symbol_map_strips_whitespace(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that whitespace is stripped."""
        result = resolver.collect_symbol_map(
            ds_cfg=None,
            symbols=(" SPY ", " AAPL "),
            instruments=None,
            instrument_ids=None,
            market_inputs=None,
        )

        assert result == {"SPY": (), "AAPL": ()}

class TestComputeWindowStartIso:
    """Test compute_window_start_iso method."""

    def test_compute_window_start_iso_subtracts_years(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that lookback years are subtracted correctly."""
        end_iso = "2024-01-15"
        result = resolver.compute_window_start_iso(end_iso, lookback_years=3)

        assert result == "2021-01-15"

    def test_compute_window_start_iso_handles_leap_year(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that leap year dates are handled correctly."""
        end_iso = "2024-02-29"
        result = resolver.compute_window_start_iso(end_iso, lookback_years=1)

        # 2023 is not a leap year, so Feb 29 becomes Feb 28
        assert result == "2023-02-28"

    def test_compute_window_start_iso_handles_month_boundaries(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that month boundaries are handled correctly."""
        end_iso = "2024-01-31"
        result = resolver.compute_window_start_iso(end_iso, lookback_years=1)

        assert result == "2023-01-31"

    def test_compute_window_start_iso_with_default_lookback(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test default lookback period."""
        end_iso = "2024-01-15"
        result = resolver.compute_window_start_iso(end_iso)

        # Default is 7 years
        assert result == "2017-01-15"

class TestResolveWindowBoundsNs:
    """Test resolve_window_bounds_ns method."""

    def test_resolve_window_bounds_ns_with_explicit_dates(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test window bounds with explicit start and end dates."""
        cfg = DatasetBuildConfig(
            data_dir="data/test",
            symbols="SPY",
            out_dir="output/test",
            dataset_id="test_dataset",
            start_iso="2023-01-01",
            end_iso="2023-12-31",
            target_semantics=TARGET_SEMANTICS,
        )

        start_ns, end_ns = resolver.resolve_window_bounds_ns(cfg)

        assert start_ns > 0
        assert end_ns > start_ns

        # Verify approximate correctness (within a day)
        start_dt = datetime.fromtimestamp(start_ns / 1_000_000_000, tz=UTC)
        end_dt = datetime.fromtimestamp(end_ns / 1_000_000_000, tz=UTC)

        assert start_dt.year == 2023
        assert start_dt.month == 1
        assert end_dt.year == 2023
        assert end_dt.month == 12

    def test_resolve_window_bounds_ns_with_no_dates_uses_defaults(
        self,
        resolver: ConfigResolver,
        base_dataset_config: DatasetBuildConfig,
    ) -> None:
        """Test that missing dates use defaults."""
        start_ns, end_ns = resolver.resolve_window_bounds_ns(base_dataset_config)

        assert start_ns > 0
        assert end_ns > start_ns

        # End should be approximately now
        end_dt = datetime.fromtimestamp(end_ns / 1_000_000_000, tz=UTC)
        now = datetime.now(tz=UTC)
        assert abs((now - end_dt).total_seconds()) < 86400  # Within a day

    def test_resolve_window_bounds_ns_ensures_end_after_start(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that end is always after start."""
        cfg = DatasetBuildConfig(
            data_dir="data/test",
            symbols="SPY",
            out_dir="output/test",
            dataset_id="test_dataset",
            start_iso="2024-01-01",
            end_iso="2024-01-01",  # Same as start
            target_semantics=TARGET_SEMANTICS,
        )

        start_ns, end_ns = resolver.resolve_window_bounds_ns(cfg)

        # Should add one day to end
        assert end_ns > start_ns

class TestPrepareDatasetConfig:
    """Test prepare_dataset_config method."""

    def test_prepare_dataset_config_with_resolved_inputs(
        self,
        resolver: ConfigResolver,
        base_dataset_config: DatasetBuildConfig,
    ) -> None:
        """Test config preparation with resolved inputs."""
        resolved_input = MarketDatasetInput(
            descriptor_id="test_desc",
            dataset_id="test_ds",
            symbols=("SPY",),
        )
        binding = create_test_binding()

        result = resolver.prepare_dataset_config(
            base_dataset_config,
            resolved_inputs=(resolved_input,),
            bindings=(binding,),
        )

        assert result.market_inputs == (resolved_input,)
        assert result.instrument_ids == ("SPY.XNAS",)

    def test_prepare_dataset_config_with_no_resolved_inputs(
        self,
        resolver: ConfigResolver,
        base_dataset_config: DatasetBuildConfig,
    ) -> None:
        """Test config preparation without resolved inputs."""
        result = resolver.prepare_dataset_config(
            base_dataset_config,
            resolved_inputs=None,
            bindings=(),
        )

        # Should apply defaults but not change structure
        assert result.symbols == base_dataset_config.symbols

class TestSymbolToInstruments:
    """Test symbol_to_instruments method."""

    def test_symbol_to_instruments_extracts_from_symbols(
        self,
        resolver: ConfigResolver,
        base_dataset_config: DatasetBuildConfig,
    ) -> None:
        """Test extraction from symbols field."""
        result = resolver.symbol_to_instruments(base_dataset_config)

        assert "SPY" in result
        assert "AAPL" in result
        assert isinstance(result, OrderedDict)

    def test_symbol_to_instruments_extracts_from_instrument_ids(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test extraction from instrument_ids field."""
        cfg = DatasetBuildConfig(
            data_dir="data/test",
            symbols="SPY",
            out_dir="output/test",
            dataset_id="test_dataset",
            instrument_ids=("SPY.XNAS", "SPY.ARCX"),
            target_semantics=TARGET_SEMANTICS,
        )

        result = resolver.symbol_to_instruments(cfg)

        assert result["SPY"] == ("SPY.XNAS", "SPY.ARCX")

    def test_symbol_to_instruments_handles_dotted_symbols(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that dotted symbols are split correctly."""
        cfg = DatasetBuildConfig(
            data_dir="data/test",
            symbols="SPY.XNAS,AAPL.XNAS",
            out_dir="output/test",
            dataset_id="test_dataset",
            target_semantics=TARGET_SEMANTICS,
        )

        result = resolver.symbol_to_instruments(cfg)

        assert "SPY" in result
        assert "AAPL" in result

    def test_symbol_to_instruments_preserves_order(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that symbol order is preserved."""
        cfg = DatasetBuildConfig(
            data_dir="data/test",
            symbols="AAPL,SPY,TSLA",
            out_dir="output/test",
            dataset_id="test_dataset",
            target_semantics=TARGET_SEMANTICS,
        )

        result = resolver.symbol_to_instruments(cfg)

        assert list(result.keys()) == ["AAPL", "SPY", "TSLA"]

class TestCollectInstrumentIds:
    """Test collect_instrument_ids method."""

    def test_collect_instrument_ids_from_existing(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test collection from existing instrument IDs."""
        result = resolver.collect_instrument_ids(
            bindings=(),
            existing=("SPY.XNAS", "AAPL.XNAS"),
        )

        assert result == ("SPY.XNAS", "AAPL.XNAS")

    def test_collect_instrument_ids_from_bindings(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test collection from bindings."""
        binding = create_test_binding()

        result = resolver.collect_instrument_ids(
            bindings=(binding,),
            existing=None,
        )

        assert result == ("SPY.XNAS",)

    def test_collect_instrument_ids_uses_symbol_fallback(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that symbol is used when instrument_ids is empty."""
        binding = create_test_binding(instrument_ids=None)

        result = resolver.collect_instrument_ids(
            bindings=(binding,),
            existing=None,
        )

        assert result == ("SPY",)

    def test_collect_instrument_ids_deduplicates(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that duplicate IDs are removed."""
        binding1 = create_test_binding()
        binding2 = create_test_binding()

        result = resolver.collect_instrument_ids(
            bindings=(binding1, binding2),
            existing=("SPY.XNAS",),
        )

        assert result == ("SPY.XNAS",)

    def test_collect_instrument_ids_merges_existing_and_bindings(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that existing and bindings are merged."""
        binding = create_test_binding(symbol="AAPL", instrument_ids=("AAPL.XNAS",))

        result = resolver.collect_instrument_ids(
            bindings=(binding,),
            existing=("SPY.XNAS",),
        )

        assert result == ("SPY.XNAS", "AAPL.XNAS")

class TestInferDefaultSchema:
    """Test infer_default_schema method."""

    def test_infer_default_schema_returns_ohlcv(
        self,
        resolver: ConfigResolver,
        base_dataset_config: DatasetBuildConfig,
    ) -> None:
        """Test that default schema is ohlcv-1m."""
        result = resolver.infer_default_schema(base_dataset_config)

        assert result == "ohlcv-1m"

class TestResolveInstrumentIds:
    """Test resolve_instrument_ids method."""

    def test_resolve_instrument_ids_uses_override(
        self,
        resolver: ConfigResolver,
        base_dataset_config: DatasetBuildConfig,
    ) -> None:
        """Test that override takes precedence."""
        result = resolver.resolve_instrument_ids(
            base_dataset_config,
            override=("TSLA.XNAS",),
        )

        assert result == ("TSLA.XNAS",)

    def test_resolve_instrument_ids_uses_config_instrument_ids(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that config instrument_ids are used."""
        cfg = DatasetBuildConfig(
            data_dir="data/test",
            symbols="SPY",
            out_dir="output/test",
            dataset_id="test_dataset",
            instrument_ids=("SPY.XNAS",),
            target_semantics=TARGET_SEMANTICS,
        )

        result = resolver.resolve_instrument_ids(cfg)

        assert result == ("SPY.XNAS",)

    def test_resolve_instrument_ids_falls_back_to_symbols(
        self,
        resolver: ConfigResolver,
        base_dataset_config: DatasetBuildConfig,
    ) -> None:
        """Test that symbols are used as fallback."""
        result = resolver.resolve_instrument_ids(base_dataset_config)

        assert result == ("SPY", "AAPL")

    def test_resolve_instrument_ids_strips_whitespace(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that whitespace is stripped."""
        cfg = DatasetBuildConfig(
            data_dir="data/test",
            symbols=" SPY , AAPL ",
            out_dir="output/test",
            dataset_id="test_dataset",
            target_semantics=TARGET_SEMANTICS,
        )

        result = resolver.resolve_instrument_ids(cfg)

        assert result == ("SPY", "AAPL")

    def test_resolve_instrument_ids_normalizes_case(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that symbols are normalized to uppercase."""
        cfg = DatasetBuildConfig(
            data_dir="data/test",
            symbols="spy,aapl",
            out_dir="output/test",
            dataset_id="test_dataset",
            target_semantics=TARGET_SEMANTICS,
        )

        result = resolver.resolve_instrument_ids(cfg)

        assert result == ("SPY", "AAPL")

class TestNsToDatetime:
    """Test ns_to_datetime static method."""

    def test_ns_to_datetime_converts_correctly(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test nanoseconds to datetime conversion."""
        # Jan 1, 2024 00:00:00 UTC
        ns = 1704067200_000_000_000

        result = ConfigResolver.ns_to_datetime(ns)

        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1
        assert result.tzinfo == UTC

    def test_ns_to_datetime_handles_subsecond_precision(
        self,
        resolver: ConfigResolver,
    ) -> None:
        """Test that subsecond precision is handled."""
        # Jan 1, 2024 00:00:00.500 UTC
        ns = 1704067200_500_000_000

        result = ConfigResolver.ns_to_datetime(ns)

        assert result.microsecond == 500_000
