"""Chunked dataset build streaming tests."""

from __future__ import annotations

import importlib
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from ml.data import DatasetBuildConfig
from ml.data import DatasetValidationConfig
from ml.data import build_tft_dataset
from ml.data.ingest.market_bindings import ResolvedMarketBinding

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


def _patch_market_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    def _resolver(**_: Any) -> tuple[ResolvedMarketBinding, ...]:
        binding = ResolvedMarketBinding(
            binding_id="binding-001",
            dataset_id="EQUS.MINI",
            descriptor_id="EQUS.MINI",
            symbol="SPY",
            instrument_ids=("SPY.XNAS",),
            schema="ohlcv-1m",
            storage_kind=None,
            source="descriptor",
            license_start=None,
            license_end=None,
            start=None,
            end=None,
        )
        return (binding,)

    monkeypatch.setattr("ml.data.resolve_market_dataset_bindings", _resolver)


def _patch_dataset_bars_ns(
    monkeypatch: pytest.MonkeyPatch,
    sample_bars_dataframe_factory,
    sample_bar_series_config_factory,
    config,
) -> None:
    def _stub(
        _catalog: object,
        instrument_ids: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        del _catalog, start, end
        instrument = instrument_ids[0] if instrument_ids else config.instrument_id
        resolved = sample_bar_series_config_factory(
            instrument_id=instrument,
            start=config.start,
            rows=config.rows,
            freq_minutes=config.freq_minutes,
        )
        frame = sample_bars_dataframe_factory(resolved)
        return frame.with_columns(
            pl.col("timestamp").cast(pl.Datetime("ns", "UTC")),
        )

    for module_path in ("ml.data.catalog_utils", "ml.data.tft_dataset_builder"):
        module = importlib.import_module(module_path)
        monkeypatch.setattr(module, "bars_to_dataframe", _stub)


def test_build_tft_dataset_when_chunked_returns_sequential_time_index(
    monkeypatch: pytest.MonkeyPatch,
    sample_bars_dataframe_factory,
    sample_bar_series_config_factory,
    tmp_path: Path,
) -> None:
    bar_config = sample_bar_series_config_factory(
        instrument_id="SPY",
        rows=32,
        start=datetime(2025, 1, 1, 9, 30, tzinfo=UTC),
    )
    _patch_dataset_bars_ns(
        monkeypatch,
        sample_bars_dataframe_factory,
        sample_bar_series_config_factory,
        bar_config,
    )
    _patch_market_bindings(monkeypatch)

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    cfg = DatasetBuildConfig(
        data_dir=data_dir,
        out_dir=tmp_path / "out",
        symbols=["SPY"],
        include_macro=False,
        include_events=False,
        include_calendar=False,
        include_earnings=False,
        include_micro=False,
        include_l2=False,
        horizon_minutes=1,
        lookback_periods=1,
        threshold=0.0,
        chunk_days=1,
        start=bar_config.start,
        end=bar_config.start + timedelta(days=2),
        validation=DatasetValidationConfig(
            min_rows=0,
            min_positive_rate=None,
            max_positive_rate=None,
            min_feature_coverage=0.0,
            require_macro_series=(),
            macro_min_vintage_observations=None,
        ),
    )

    result = build_tft_dataset(cfg)

    assert result.dataset_parquet.exists()
    df = pl.read_parquet(result.dataset_parquet)
    assert "time_index" in df.columns

    time_index = df.get_column("time_index").to_numpy()
    assert time_index.size == df.height
    assert time_index[0] == 0
    assert np.array_equal(time_index, np.arange(df.height, dtype=time_index.dtype))
