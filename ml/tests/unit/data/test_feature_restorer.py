#!/usr/bin/env python3

"""
Unit tests for the feature coverage restorer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pandas")

import pandas as pd

from ml.config.dataset_ids import FEATURE_VALUES_DATASET_ID
from ml.config.dataset_ids import L2_MINUTE_DATASET_ID
from ml.config.dataset_ids import MICRO_MINUTE_DATASET_ID
from ml.data.coverage.feature_restorer import FeatureCoverageRestorer
from ml.data.coverage.manager import BucketSpec
from ml.data.coverage.types import DAY_NS
from ml.stores.providers import ParquetCoverageSpec

class _StubWriter:
    def __init__(self) -> None:
        self.actual_calls: list[dict[str, Any]] = []
        self.estimate_calls: list[dict[str, Any]] = []
        self.feature_calls: list[dict[str, Any]] = []
        self.ingestion_calls: list[dict[str, Any]] = []

    def write_earnings_actual(self, **kwargs: Any) -> object:
        self.actual_calls.append(kwargs)
        return object()

    def write_earnings_estimate(self, **kwargs: Any) -> object:
        self.estimate_calls.append(kwargs)
        return object()

    def write_features(self, **kwargs: Any) -> object:
        self.feature_calls.append(kwargs)
        return object()

    def write_ingestion(self, **kwargs: Any) -> object:
        self.ingestion_calls.append(kwargs)
        return object()


def _bucket_spec(
    dataset_id: str,
    *,
    instrument: str,
    ts_event: int,
    entity_field: str = "ticker",
) -> BucketSpec:
    bucket_start_ns = (ts_event // DAY_NS) * DAY_NS
    return BucketSpec(
        dataset_id=dataset_id,
        schema=f"{dataset_id}_schema",
        instrument_id=instrument,
        bucket_start_ns=bucket_start_ns,
        entity_field=entity_field,
    )


def _parquet_spec(
    dataset_id: str,
    base_path: Path,
    *,
    partition_field: str = "ticker",
    timestamp_field: str = "ts_event",
    partition_template: str | None = None,
) -> ParquetCoverageSpec:
    return ParquetCoverageSpec(
        dataset_id=dataset_id,
        base_path=base_path,
        partition_field=partition_field,
        timestamp_field=timestamp_field,
        partition_template=partition_template,
    )


def test_restorer_replays_earnings_actuals(tmp_path: Path) -> None:
    dataset_id = "ml.earnings_actuals"
    ts_event = 1_762_665_600_000_000_000
    base_path = tmp_path / "actuals"
    partition = base_path / "ticker=AAPL"
    partition.mkdir(parents=True)
    frame = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "period_end": "2025-03-29",
                "filing_date": "2025-05-02",
                "ts_event": ts_event,
                "ts_init": ts_event + 10,
                "eps_diluted": 1.5,
                "revenue": 1.0,
            },
        ],
    )
    frame.to_parquet(partition / "earnings.parquet")

    spec = _parquet_spec(dataset_id, base_path)
    bucket_spec = _bucket_spec(dataset_id, instrument="AAPL", ts_event=ts_event)
    writer = _StubWriter()
    restorer = FeatureCoverageRestorer(
        db_connection="postgresql://stub",
        parquet_specs={dataset_id: spec},
        writer_factory=lambda _: writer,
    )

    result = restorer.restore((bucket_spec,))

    assert result.rows_written == 1
    assert result.buckets_restored == 1
    assert result.failures == {}
    assert writer.actual_calls and writer.actual_calls[0]["ticker"] == "AAPL"
    assert writer.actual_calls[0]["source"] == "backfill"


def test_restorer_handles_missing_partitions(tmp_path: Path) -> None:
    dataset_id = "ml.earnings_actuals"
    base_path = tmp_path / "missing"
    base_path.mkdir(parents=True)
    spec = _parquet_spec(dataset_id, base_path)
    bucket_spec = _bucket_spec(dataset_id, instrument="AAPL", ts_event=1_762_665_600_000_000_000)
    writer = _StubWriter()
    restorer = FeatureCoverageRestorer(
        db_connection="postgresql://stub",
        parquet_specs={dataset_id: spec},
        writer_factory=lambda _: writer,
    )

    result = restorer.restore((bucket_spec,))

    assert result.rows_written == 0
    assert result.buckets_restored == 0
    assert "ml.earnings_actuals:AAPL" in result.failures
    assert writer.actual_calls == []


def test_restorer_filters_to_requested_buckets(tmp_path: Path) -> None:
    dataset_id = "ml.earnings_estimates"
    ts_event = 1_762_665_600_000_000_000
    base_path = tmp_path / "estimates"
    partition = base_path / "ticker=AAPL"
    partition.mkdir(parents=True)
    frame = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "estimate_date": "2025-11-08",
                "period_end": "2025-12-31",
                "ts_event": ts_event,
                "ts_init": ts_event + 99,
                "eps_consensus": 2.66,
            },
            {
                "ticker": "AAPL",
                "estimate_date": "2025-11-09",
                "period_end": "2025-12-31",
                "ts_event": ts_event + DAY_NS,
                "ts_init": ts_event + 199,
                "eps_consensus": 2.7,
            },
        ],
    )
    frame.to_parquet(partition / "estimates.parquet")

    spec = _parquet_spec(dataset_id, base_path)
    bucket_spec = _bucket_spec(dataset_id, instrument="AAPL", ts_event=ts_event)
    writer = _StubWriter()
    restorer = FeatureCoverageRestorer(
        db_connection="postgresql://stub",
        parquet_specs={dataset_id: spec},
        writer_factory=lambda _: writer,
    )

    result = restorer.restore((bucket_spec,))

    assert result.rows_written == 1
    assert result.buckets_restored == 1
    assert not result.failures
    assert len(writer.estimate_calls) == 1
    assert writer.estimate_calls[0]["estimate_date"] == "2025-11-08"


def test_restorer_replays_feature_values(tmp_path: Path) -> None:
    dataset_id = FEATURE_VALUES_DATASET_ID
    ts_event = 1_736_352_000_000_000_000
    base_path = tmp_path / "feature_values"
    partition = base_path / "AAPL" / "year=2024" / "month=12"
    partition.mkdir(parents=True)
    frame = pd.DataFrame(
        [
            {
                "instrument_id": "AAPL",
                "feature_set_id": "price_snapshot",
                "ts_event": ts_event,
                "ts_init": ts_event + 10,
                "values": '{"alpha": 1.5, "beta": 2.0}',
            },
        ],
    )
    frame.to_parquet(partition / "day=31.parquet")

    spec = _parquet_spec(
        dataset_id,
        base_path,
        partition_field="instrument_id",
        timestamp_field="ts_event",
        partition_template="{value}",
    )
    bucket_spec = _bucket_spec(
        dataset_id,
        instrument="AAPL",
        ts_event=ts_event,
        entity_field="instrument_id",
    )
    writer = _StubWriter()
    restorer = FeatureCoverageRestorer(
        db_connection="postgresql://stub",
        parquet_specs={dataset_id: spec},
        writer_factory=lambda _: writer,
    )

    result = restorer.restore((bucket_spec,))

    assert result.rows_written == 1
    assert result.buckets_restored == 1
    assert result.failures == {}
    assert len(writer.feature_calls) == 1
    call = writer.feature_calls[0]
    assert call["instrument_id"] == "AAPL"
    assert call["source"] == "backfill"
    features = call["features"]
    assert len(features) == 1
    payload = features[0]
    assert payload.feature_set_id == "price_snapshot"
    assert payload.instrument_id == "AAPL"
    assert payload.ts_event == ts_event
    assert payload.feature_values["alpha"] == 1.5


def test_restorer_replays_macro_release_dataset(tmp_path: Path) -> None:
    dataset_id = "ml.macro_release_calendar"
    base_path = tmp_path / "vintages"
    partition = base_path / "CPIAUCSL"
    partition.mkdir(parents=True)
    ts_release = 1_758_912_000_000_000_000
    frame = pd.DataFrame(
        [
            {
                "release_ts": ts_release,
                "observation_ts": ts_release - DAY_NS,
                "release_end_ts": ts_release + 1_000_000,
                "value": 2.5,
                "series_id": "CPIAUCSL",
            },
        ],
    )
    frame.to_parquet(partition / "release_calendar.parquet")
    spec = _parquet_spec(
        dataset_id,
        base_path,
        partition_field="series_id",
        timestamp_field="release_ts",
        partition_template="{value}/release_calendar.parquet",
    )
    bucket_spec = _bucket_spec(
        dataset_id,
        instrument="CPIAUCSL",
        ts_event=ts_release,
        entity_field="series_id",
    )
    writer = _StubWriter()
    restorer = FeatureCoverageRestorer(
        db_connection="postgresql://stub",
        parquet_specs={dataset_id: spec},
        writer_factory=lambda _: writer,
    )

    result = restorer.restore((bucket_spec,))

    assert result.rows_written == 1
    assert result.buckets_restored == 1
    assert not result.failures
    assert writer.ingestion_calls
    call = writer.ingestion_calls[0]
    assert call["dataset_id"] == dataset_id
    records = call["records"]
    assert isinstance(records, pd.DataFrame)
    assert set(records["series_id"]) == {"CPIAUCSL"}


def test_restorer_filters_file_backed_events(tmp_path: Path) -> None:
    dataset_id = "ml.events_calendar"
    base_path = tmp_path / "events.parquet"
    ts_aapl = 1_760_000_000_000_000_000
    ts_msft = ts_aapl + DAY_NS
    frame = pd.DataFrame(
        [
            {
                "event_timestamp": ts_aapl,
                "event_type": "earnings",
                "name": "AAPL call",
                "instrument_id": "AAPL",
                "importance": 1,
                "source": "ref",
                "metadata": "{}",
            },
            {
                "event_timestamp": ts_msft,
                "event_type": "earnings",
                "name": "MSFT call",
                "instrument_id": "MSFT",
                "importance": 1,
                "source": "ref",
                "metadata": "{}",
            },
        ],
    )
    frame.to_parquet(base_path)
    spec = _parquet_spec(
        dataset_id,
        base_path,
        partition_field="instrument_id",
        timestamp_field="event_timestamp",
        partition_template="",
    )
    bucket_spec = _bucket_spec(
        dataset_id,
        instrument="AAPL",
        ts_event=ts_aapl,
        entity_field="instrument_id",
    )
    writer = _StubWriter()
    restorer = FeatureCoverageRestorer(
        db_connection="postgresql://stub",
        parquet_specs={dataset_id: spec},
        writer_factory=lambda _: writer,
    )

    result = restorer.restore((bucket_spec,))

    assert result.rows_written == 1
    assert result.buckets_restored == 1
    assert not result.failures
    assert writer.ingestion_calls
    call = writer.ingestion_calls[0]
    records = call["records"]
    assert isinstance(records, pd.DataFrame)
    assert set(records["instrument_id"]) == {"AAPL"}


def test_restorer_replays_micro_dataset(tmp_path: Path) -> None:
    dataset_id = MICRO_MINUTE_DATASET_ID
    base_path = tmp_path / "micro"
    partition = base_path / "instrument_id=AAPL"
    partition.mkdir(parents=True)
    ts_event = 1_760_000_000_000_000_000
    frame = pd.DataFrame(
        [
            {
                "instrument_id": "AAPL",
                "timestamp": ts_event,
                "ts_event": ts_event,
                "ts_init": ts_event + 1_000,
                "midprice": 101.0,
                "spread_bps": 4.0,
                "quote_imbalance": 0.1,
                "trade_imbalance": -0.05,
                "realized_vol": 0.2,
            },
        ],
    )
    frame.to_parquet(partition / "micro.parquet")
    spec = _parquet_spec(
        dataset_id,
        base_path,
        partition_field="instrument_id",
        timestamp_field="timestamp",
        partition_template="{field}={value}",
    )
    bucket_spec = _bucket_spec(
        dataset_id,
        instrument="AAPL",
        ts_event=ts_event,
        entity_field="instrument_id",
    )
    writer = _StubWriter()
    restorer = FeatureCoverageRestorer(
        db_connection="postgresql://stub",
        parquet_specs={dataset_id: spec},
        writer_factory=lambda _: writer,
    )

    result = restorer.restore((bucket_spec,))

    assert result.rows_written == 1
    assert result.failures == {}
    assert writer.ingestion_calls
    assert writer.ingestion_calls[0]["dataset_id"] == dataset_id


def test_restorer_replays_l2_dataset(tmp_path: Path) -> None:
    dataset_id = L2_MINUTE_DATASET_ID
    base_path = tmp_path / "l2"
    partition = base_path / "instrument_id=MSFT"
    partition.mkdir(parents=True)
    ts_event = 1_761_000_000_000_000_000
    frame = pd.DataFrame(
        [
            {
                "instrument_id": "MSFT",
                "timestamp": ts_event,
                "ts_event": ts_event,
                "ts_init": ts_event + 5_000,
                "midprice": 299.0,
                "spread_bps": 2.0,
                "microprice_bps": 1.0,
                "depth_imbalance_top1": 0.2,
                "dwp_bps_top1": 0.3,
                "bid_slope_top1": 0.4,
                "ask_slope_top1": 0.5,
            },
        ],
    )
    frame.to_parquet(partition / "l2.parquet")
    spec = _parquet_spec(
        dataset_id,
        base_path,
        partition_field="instrument_id",
        timestamp_field="timestamp",
        partition_template="{field}={value}",
    )
    bucket_spec = _bucket_spec(
        dataset_id,
        instrument="MSFT",
        ts_event=ts_event,
        entity_field="instrument_id",
    )
    writer = _StubWriter()
    restorer = FeatureCoverageRestorer(
        db_connection="postgresql://stub",
        parquet_specs={dataset_id: spec},
        writer_factory=lambda _: writer,
    )

    result = restorer.restore((bucket_spec,))

    assert result.rows_written == 1
    assert result.failures == {}
    assert writer.ingestion_calls
    assert writer.ingestion_calls[0]["dataset_id"] == dataset_id


def test_restorer_marks_unsupported_dataset(tmp_path: Path) -> None:
    dataset_id = "ml.unsupported_dataset"
    base_path = tmp_path / "unsupported"
    partition = base_path / "instrument_id=FOO"
    partition.mkdir(parents=True)
    frame = pd.DataFrame(
        [
            {
                "instrument_id": "FOO",
                "ts_event": 100,
                "ts_init": 200,
            },
        ],
    )
    frame.to_parquet(partition / "data.parquet")
    spec = _parquet_spec(dataset_id, base_path)
    bucket_spec = _bucket_spec(dataset_id, instrument="FOO", ts_event=1_000_000)
    writer = _StubWriter()
    restorer = FeatureCoverageRestorer(
        db_connection="postgresql://stub",
        parquet_specs={dataset_id: spec},
        writer_factory=lambda _: writer,
    )

    with pytest.raises(ValueError):
        restorer.restore((bucket_spec,))
    assert writer.actual_calls == []
    assert writer.estimate_calls == []
    assert writer.ingestion_calls == []
