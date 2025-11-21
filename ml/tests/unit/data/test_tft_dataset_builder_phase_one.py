from __future__ import annotations

from datetime import datetime
from typing import Any
from typing import cast

import pytest

from ml.data.tft_dataset_builder import TFTDatasetBuilder
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry", "mock_tracing_backend")


def _make_builder(**overrides: Any) -> TFTDatasetBuilder:
    catalog = cast(ParquetDataCatalog, object())
    defaults: dict[str, Any] = {
        "catalog": catalog,
        "symbols": ["SPY"],
        "feature_store": None,
        "data_store": None,
        "market_bindings": (),
    }
    defaults.update(overrides)
    return TFTDatasetBuilder(**defaults)


def test_append_macro_delta_features_polars_computes_differences() -> None:
    pl = pytest.importorskip("polars")
    builder = _make_builder(
        include_macro=True,
        include_macro_deltas=True,
        macro_series_ids=("PAYEMS",),
    )
    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 3),
        interval="1d",
        eager=True,
    )
    frame = pl.DataFrame(
        {
            "timestamp": timestamps,
            "instrument_id": ["SPY"] * len(timestamps),
            "PAYEMS": [100.0, 110.0, 130.0],
        },
    )

    enriched = builder._append_macro_delta_features_polars(frame)

    assert "PAYEMS_delta_1d" in enriched.columns
    assert enriched.get_column("PAYEMS_delta_1d").to_list() == [0.0, 10.0, 20.0]





def test_event_features_join_when_calendar_lags_enabled() -> None:
    pl = pytest.importorskip("polars")

    class _StubProvider:
        def compute_features(self, ts_series: pl.Series, symbols: list[str]) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "timestamp": ts_series,
                    "hours_to_fed_meeting": [4.0] * len(ts_series),
                    "event_clustering_score": [0.2] * len(ts_series),
                },
            )

    builder = _make_builder(
        include_macro=False,
        include_events=False,
        include_calendar=False,
        include_calendar_lags=True,
        include_clustering_tags=True,
        include_context_features=True,
    )
    builder._event_provider = _StubProvider()

    timestamps = pl.datetime_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 1, minute=2),
        interval="1m",
        eager=True,
    )
    frame = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [100.0, 110.0, 120.0],
        },
    )
    frame = frame.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))

    dataset = builder._process_symbol_polars(
        frame,
        symbol="SPY",
        horizon_minutes=1,
        threshold=0.0,
        lookback_periods=0,
    )

    assert dataset is not None
    assert "hours_to_fed_meeting" in dataset.columns
    assert "event_clustering_score" in dataset.columns
