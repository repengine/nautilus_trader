from __future__ import annotations

from datetime import UTC
from datetime import datetime
from pathlib import Path

import polars as pl

from ml.data.common.pipeline_batch import PipelineBatchContext
from ml.data.common.pipeline_batch import PipelineBatchExecutor
from ml.data.providers.calendar import MarketCalendarProvider
from ml.data.providers.events import EventScheduleProvider
from ml.data.sources.calendar import MockCalendarSource
from ml.data.sources.events import MockEventSource
from ml.features.config import FeatureConfig
from ml.features.config import build_pipeline_spec_from_feature_config
from ml.features.pipeline import PipelineRunner
from ml.registry.base import DataRequirements


def _sample_bars() -> pl.DataFrame:
    timestamps = [
        datetime(2025, 1, 1, 9, 30, tzinfo=UTC),
        datetime(2025, 1, 1, 9, 31, tzinfo=UTC),
        datetime(2025, 1, 1, 9, 32, tzinfo=UTC),
        datetime(2025, 1, 1, 9, 33, tzinfo=UTC),
    ]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 100.1, 100.2, 100.3],
            "high": [100.2, 100.3, 100.4, 100.5],
            "low": [99.8, 99.9, 100.0, 100.1],
            "close": [100.05, 100.15, 100.25, 100.35],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
        },
    )


def _write_macro_parquet(tmp_path: Path, series_ids: list[str]) -> str:
    timestamps = [
        datetime(2025, 1, 1, 9, 30, tzinfo=UTC),
        datetime(2025, 1, 2, 9, 30, tzinfo=UTC),
    ]
    rows: list[dict[str, object]] = []
    for idx, ts in enumerate(timestamps):
        for series_id in series_ids:
            rows.append(
                {
                    "timestamp": ts,
                    "series_id": series_id,
                    "value": float(idx + 1),
                },
            )
    df = pl.DataFrame(rows)
    path = tmp_path / "fred_test.parquet"
    df.write_parquet(path)
    return str(path)


def test_pipeline_batch_executor_appends_ohlcv_features_polars() -> None:
    cfg = FeatureConfig()
    spec = build_pipeline_spec_from_feature_config(cfg)
    context = PipelineBatchContext(feature_config=cfg)
    executor = PipelineBatchExecutor(
        spec,
        allowable=cfg.resolved_data_requirements(),
        context=context,
    )

    df = _sample_bars()
    out = executor.execute_polars(df)

    expected = PipelineRunner(spec, cfg.resolved_data_requirements()).compute_feature_names()
    for name in expected:
        assert name in out.columns


def test_pipeline_batch_executor_appends_calendar_features() -> None:
    cfg = FeatureConfig(include_calendar=True)
    spec = build_pipeline_spec_from_feature_config(cfg)
    calendar_provider = MarketCalendarProvider(calendar_source=MockCalendarSource())
    context = PipelineBatchContext(feature_config=cfg, calendar_provider=calendar_provider)
    executor = PipelineBatchExecutor(
        spec,
        allowable=cfg.resolved_data_requirements(),
        context=context,
    )

    df = _sample_bars()
    out = executor.execute_polars(df)

    for col in [
        "hour_sin",
        "minute_sin",
        "is_trading_day",
        "is_market_hours",
        "minutes_to_close",
    ]:
        assert col in out.columns


def test_pipeline_batch_executor_appends_event_features() -> None:
    cfg = FeatureConfig(include_event_schedule=True)
    spec = build_pipeline_spec_from_feature_config(cfg)
    event_provider = EventScheduleProvider(event_source=MockEventSource())
    context = PipelineBatchContext(feature_config=cfg, event_provider=event_provider)
    executor = PipelineBatchExecutor(
        spec,
        allowable=cfg.resolved_data_requirements(),
        context=context,
    )

    df = _sample_bars()
    out = executor.execute_polars(df)

    for col in [
        "hours_to_earnings",
        "earnings_within_24h",
        "total_events_24h",
        "is_fomc_week",
    ]:
        assert col in out.columns


def test_pipeline_batch_executor_appends_macro_features_and_deltas(tmp_path: Path) -> None:
    series_ids = ["DGS10", "DGS2"]
    fred_path = _write_macro_parquet(tmp_path, series_ids)
    cfg = FeatureConfig(
        include_macro=True,
        include_macro_deltas=True,
        macro_series_ids=series_ids,
    )
    spec = build_pipeline_spec_from_feature_config(cfg)
    context = PipelineBatchContext(
        feature_config=cfg,
        fred_path=fred_path,
        macro_series_ids=tuple(series_ids),
    )
    executor = PipelineBatchExecutor(
        spec,
        allowable=cfg.resolved_data_requirements(),
        context=context,
    )

    df = _sample_bars()
    out = executor.execute_polars(df)

    for col in [
        "DGS10__value_real_time",
        "DGS2__value_real_time",
        "DGS10_delta_1d",
        "DGS2_delta_1d",
        "is_macro_available",
    ]:
        assert col in out.columns


def test_pipeline_batch_executor_appends_microstructure_trade_flow() -> None:
    cfg = FeatureConfig(
        include_microstructure=True,
        include_trade_flow=True,
        data_requirements=DataRequirements.L1_L2_L3,
    )
    spec = build_pipeline_spec_from_feature_config(cfg)
    context = PipelineBatchContext(feature_config=cfg)
    executor = PipelineBatchExecutor(
        spec,
        allowable=cfg.resolved_data_requirements(),
        context=context,
    )

    df = _sample_bars()
    out = executor.execute_polars(df)

    for col in [
        "spread_mean",
        "spread_std",
        "trade_flow_imbalance",
        "vwap",
    ]:
        assert col in out.columns


def test_pipeline_batch_executor_matches_feature_config_names(tmp_path: Path) -> None:
    series_ids = ["DGS10", "DGS2"]
    fred_path = _write_macro_parquet(tmp_path, series_ids)
    cfg = FeatureConfig(
        include_microstructure=True,
        include_trade_flow=True,
        include_macro=True,
        include_macro_deltas=True,
        include_macro_composites=True,
        include_calendar=True,
        include_event_schedule=True,
        macro_series_ids=series_ids,
        data_requirements=DataRequirements.L1_L2_L3,
    )
    spec = build_pipeline_spec_from_feature_config(cfg)
    calendar_provider = MarketCalendarProvider(calendar_source=MockCalendarSource())
    event_provider = EventScheduleProvider(event_source=MockEventSource())
    context = PipelineBatchContext(
        feature_config=cfg,
        fred_path=fred_path,
        macro_series_ids=tuple(series_ids),
        calendar_provider=calendar_provider,
        event_provider=event_provider,
    )
    executor = PipelineBatchExecutor(
        spec,
        allowable=cfg.resolved_data_requirements(),
        context=context,
    )

    df = _sample_bars()
    out = executor.execute_polars(df)

    expected = cfg.get_feature_names()
    for name in expected:
        assert name in out.columns
