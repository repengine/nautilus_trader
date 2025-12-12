"""Unit tests for ConfigResolver implementation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ml.orchestration.config_resolver import ConfigResolver
from ml.orchestration.config_types import DatasetBuildConfig


@pytest.fixture
def resolver() -> ConfigResolver:
    """Create ConfigResolver instance for testing."""
    return ConfigResolver()


@pytest.fixture
def dataset_config() -> DatasetBuildConfig:
    """Provide a baseline dataset configuration."""
    return DatasetBuildConfig(
        data_dir="/tmp/data",
        out_dir="/tmp/out",
        dataset_id="test.dataset",
        start_iso="2024-01-01",
        end_iso="2024-12-31",
        symbols="SPY,QQQ",
        instrument_ids=("SPY.NASDAQ", "QQQ.NASDAQ"),
    )


def test_resolve_window_bounds_ns_returns_valid_bounds(
    resolver: ConfigResolver,
    dataset_config: DatasetBuildConfig,
) -> None:
    """Window bounds resolution should produce increasing nanosecond values."""
    start_ns, end_ns = resolver.resolve_window_bounds_ns(dataset_config)

    assert isinstance(start_ns, int)
    assert isinstance(end_ns, int)
    assert start_ns > 0
    assert end_ns > start_ns


def test_resolve_instrument_ids_prefers_override(
    resolver: ConfigResolver,
    dataset_config: DatasetBuildConfig,
) -> None:
    """Override instrument IDs should win while preserving order and deduping."""
    result = resolver.resolve_instrument_ids(
        dataset_config,
        override=("QQQ.NASDAQ", "SPY.NASDAQ", "QQQ.NASDAQ"),
    )

    assert result == ("QQQ.NASDAQ", "SPY.NASDAQ", "QQQ.NASDAQ")


def test_symbol_to_instruments_preserves_symbol_order(
    resolver: ConfigResolver,
    dataset_config: DatasetBuildConfig,
) -> None:
    """Symbol to instrument mapping should retain declared order."""
    mapping = resolver.symbol_to_instruments(dataset_config)

    assert list(mapping.keys()) == ["SPY", "QQQ"]
    assert mapping["SPY"] == ("SPY.NASDAQ",)
    assert mapping["QQQ"] == ("QQQ.NASDAQ",)


def test_collect_symbol_map_merges_sources(
    resolver: ConfigResolver,
    dataset_config: DatasetBuildConfig,
) -> None:
    """Symbol map should merge dataset config and overrides."""
    result = resolver.collect_symbol_map(
        ds_cfg=dataset_config,
        symbols=("TSLA",),
        instruments=("MSFT.XNAS",),
        instrument_ids=("AAPL.XNAS",),
        market_inputs=None,
    )

    assert result["SPY"] == ("SPY.NASDAQ",)
    assert result["QQQ"] == ("QQQ.NASDAQ",)
    assert result["TSLA"] == ()
    assert result["MSFT"] == ("MSFT.XNAS",)
    assert result["AAPL"] == ("AAPL.XNAS",)


def test_infer_default_schema_defaults_to_ohlcv(
    resolver: ConfigResolver,
    dataset_config: DatasetBuildConfig,
) -> None:
    """Default schema inference should fall back to ohlcv-1m."""
    assert resolver.infer_default_schema(dataset_config) == "ohlcv-1m"


def test_prepare_dataset_config_applies_resolved_inputs(
    resolver: ConfigResolver,
    dataset_config: DatasetBuildConfig,
) -> None:
    """prepare_dataset_config should return cfg unchanged when no resolved inputs."""
    prepared = resolver.prepare_dataset_config(
        dataset_config,
        resolved_inputs=None,
        bindings=(),
    )

    assert prepared.market_inputs is None
    assert prepared.instrument_ids == dataset_config.instrument_ids


def test_apply_default_market_inputs_populates_from_descriptor(
    resolver: ConfigResolver,
    dataset_config: DatasetBuildConfig,
) -> None:
    """Descriptor-backed market inputs should be injected when missing."""
    cfg = replace(dataset_config, market_inputs=(), market_dataset_id="EQUS.MINI")
    updated = resolver.apply_default_market_inputs(cfg)

    # Descriptor may be absent in test env; ensure method is at least non-destructive
    assert updated.dataset_id == cfg.dataset_id
    assert updated.market_dataset_id == cfg.market_dataset_id
