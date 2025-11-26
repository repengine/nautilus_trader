from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from ml.stores.providers import DAY_NS
from ml.stores.providers import ParquetCoverageSpec
from ml.stores.providers import PartitionedParquetCoverageProvider
from ml.stores.providers import SqlCoverageOverride
from ml.stores.providers import SqlCoverageProvider


def _sqlite_conn_str(db_path: Path) -> str:
    return f"sqlite:///{db_path}"


def test_sql_coverage_provider_respects_overrides(tmp_path: Path) -> None:
    db_path = tmp_path / "coverage.db"
    conn_str = _sqlite_conn_str(db_path)
    provider = SqlCoverageProvider(
        connection_string=conn_str,
        dataset_overrides={
            "ml.earnings_actuals": SqlCoverageOverride(
                table_name="earnings_actuals",
                schema=None,
                ts_field="ts_event",
                entity_field="ticker",
            ),
        },
    )
    with provider._engine.begin() as conn:  # type: ignore[attr-defined]
        conn.execute(
            text(
                """
                CREATE TABLE earnings_actuals (
                    ticker TEXT NOT NULL,
                    ts_event BIGINT NOT NULL
                )
                """,
            ),
        )
        conn.execute(
            text("INSERT INTO earnings_actuals (ticker, ts_event) VALUES (:ticker, :ts_event)"),
            [{"ticker": "AAPL", "ts_event": 1_700_000_000_000_000_000}],
        )

    buckets = provider.read_bucket_coverage(
        dataset_id="ml.earnings_actuals",
        schema="earnings",
        instrument_id="AAPL",
        start_ns=1_699_999_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
        entity_field="ticker",
    )
    assert buckets, "Expected at least one bucket from override-backed table"


def test_sql_coverage_provider_latest_timestamp(tmp_path: Path) -> None:
    db_path = tmp_path / "latest.db"
    conn_str = _sqlite_conn_str(db_path)
    provider = SqlCoverageProvider(connection_string=conn_str)
    now_ns = int(datetime.now(tz=UTC).timestamp() * 1_000_000_000)
    past_ns = now_ns - DAY_NS
    with provider._engine.begin() as conn:  # type: ignore[attr-defined]
        conn.execute(
            text(
                """
                CREATE TABLE market_data (
                    instrument_id TEXT NOT NULL,
                    ts_event BIGINT NOT NULL,
                    ts_init BIGINT NOT NULL
                )
                """,
            ),
        )
        conn.execute(
            text("INSERT INTO market_data (instrument_id, ts_event, ts_init) VALUES (:iid, :ts, :ts)"),
            [
                {"iid": "SPY.XNAS", "ts": past_ns},
                {"iid": "SPY.XNAS", "ts": now_ns},
            ],
        )

    latest = provider.latest_timestamp_ns(dataset_id="EQUS.MINI", instrument_id="SPY.XNAS")
    assert latest == now_ns


def test_partitioned_parquet_coverage_provider_detects_buckets(tmp_path: Path) -> None:
    base_path = tmp_path / "earnings_actuals"
    partition = base_path / "ticker=AAPL"
    partition.mkdir(parents=True)
    timestamps = [
        int(datetime(2024, 1, 10, tzinfo=UTC).timestamp() * 1_000_000_000),
        int(datetime(2024, 1, 11, tzinfo=UTC).timestamp() * 1_000_000_000),
    ]
    frame = pd.DataFrame({"ts_event": timestamps})
    frame.to_parquet(partition / "sample.parquet", index=False)
    spec = ParquetCoverageSpec(
        dataset_id="ml.earnings_actuals",
        base_path=base_path,
        partition_field="ticker",
        timestamp_field="ts_event",
    )
    provider = PartitionedParquetCoverageProvider(specs={"ml.earnings_actuals": spec})
    start_ns = int(datetime(2024, 1, 10, tzinfo=UTC).timestamp() * 1_000_000_000)
    buckets = provider.read_bucket_coverage(
        dataset_id="ml.earnings_actuals",
        schema="earnings",
        instrument_id="AAPL",
        start_ns=start_ns,
        end_ns=start_ns + 2 * DAY_NS,
        entity_field="ticker",
    )
    assert buckets == {start_ns // DAY_NS, (start_ns + DAY_NS) // DAY_NS}


def test_partitioned_parquet_provider_honors_custom_template(tmp_path: Path) -> None:
    base_path = tmp_path / "vintages"
    target = base_path / "CPIAUCSL" / "release_calendar.parquet"
    target.parent.mkdir(parents=True)
    ts_release = int(datetime(2024, 2, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
    frame = pd.DataFrame({"release_ts": [ts_release]})
    frame.to_parquet(target, index=False)
    spec = ParquetCoverageSpec(
        dataset_id="ml.macro_release_calendar",
        base_path=base_path,
        partition_field="series_id",
        timestamp_field="release_ts",
        partition_template="{value}/release_calendar.parquet",
    )
    provider = PartitionedParquetCoverageProvider({"ml.macro_release_calendar": spec})
    buckets = provider.read_bucket_coverage(
        dataset_id="ml.macro_release_calendar",
        schema="macro_release_calendar",
        instrument_id="CPIAUCSL",
        start_ns=ts_release,
        end_ns=ts_release + DAY_NS,
        entity_field="series_id",
    )
    assert buckets == {ts_release // DAY_NS}


def test_partitioned_parquet_provider_filters_file_backed_datasets(tmp_path: Path) -> None:
    base_path = tmp_path / "events.parquet"
    ts_aapl = int(datetime(2024, 3, 5, tzinfo=UTC).timestamp() * 1_000_000_000)
    ts_msft = ts_aapl + DAY_NS
    frame = pd.DataFrame(
        [
            {"instrument_id": "AAPL", "event_timestamp": ts_aapl},
            {"instrument_id": "MSFT", "event_timestamp": ts_msft},
        ],
    )
    frame.to_parquet(base_path, index=False)
    spec = ParquetCoverageSpec(
        dataset_id="ml.events_calendar",
        base_path=base_path,
        partition_field="instrument_id",
        timestamp_field="event_timestamp",
        partition_template="",
    )
    provider = PartitionedParquetCoverageProvider({"ml.events_calendar": spec})
    aapl_buckets = provider.read_bucket_coverage(
        dataset_id="ml.events_calendar",
        schema="events_calendar",
        instrument_id="AAPL",
        start_ns=ts_aapl,
        end_ns=ts_msft + DAY_NS,
        entity_field="instrument_id",
    )
    assert aapl_buckets == {ts_aapl // DAY_NS}
    msft_buckets = provider.read_bucket_coverage(
        dataset_id="ml.events_calendar",
        schema="events_calendar",
        instrument_id="MSFT",
        start_ns=ts_aapl,
        end_ns=ts_msft + DAY_NS,
        entity_field="instrument_id",
    )
    assert msft_buckets == {ts_msft // DAY_NS}


def test_parquet_spec_rejects_bad_partition_templates(tmp_path: Path) -> None:
    spec = ParquetCoverageSpec(
        dataset_id="ml.events_calendar",
        base_path=tmp_path,
        partition_field="instrument_id",
        timestamp_field="event_timestamp",
        partition_template="{value",
    )
    with pytest.raises(ValueError):
        spec.files_for_instrument("AAPL")
