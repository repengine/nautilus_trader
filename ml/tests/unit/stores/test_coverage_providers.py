from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import text

from ml.registry.dataclasses import DatasetType
from ml.stores.providers import DAY_NS
from ml.stores.providers import CatalogCoverageProvider
from ml.stores.providers import ParquetCoverageSpec
from ml.stores.providers import PartitionedParquetCoverageProvider
from ml.stores.providers import SqlCoverageOverride
from ml.stores.providers import SqlCoverageProvider


def _sqlite_conn_str(db_path: Path) -> str:
    return f"sqlite:///{db_path}"


def test_sql_coverage_provider_respects_overrides(tmp_path: Path) -> None:
    db_path = tmp_path / "coverage.db"
    conn_str = _sqlite_conn_str(db_path)
    provider = SqlCoverageProvider(connection_string=conn_str)
    with provider._engine.begin() as conn:  # type: ignore[attr-defined]
        conn.execute(
            text(
                """
                CREATE TABLE market_data (
                    instrument_id TEXT NOT NULL,
                    ts_event BIGINT NOT NULL
                )
                """,
            ),
        )
        conn.execute(
            text("INSERT INTO market_data (instrument_id, ts_event) VALUES (:iid, :ts_event)"),
            [{"iid": "AAPL", "ts_event": 1_700_000_000_000_000_000}],
        )

    buckets = provider.read_bucket_coverage(
        dataset_id="ml.earnings_actuals",
        schema="earnings",
        instrument_id="AAPL",
        start_ns=1_699_999_000_000_000_000,
        end_ns=1_700_086_400_000_000_000,
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
    provider = PartitionedParquetCoverageProvider(specs={})

    buckets = provider.read_bucket_coverage(
        dataset_id="ml.earnings_actuals",
        schema="earnings",
        instrument_id="AAPL",
        start_ns=0,
        end_ns=DAY_NS,
    )
    assert buckets == set()


def test_partitioned_parquet_provider_honors_custom_template(tmp_path: Path) -> None:
    spec = ParquetCoverageSpec(
        dataset_id="ml.macro_release_calendar",
        base_path=str(tmp_path),
        partition_field="series_id",
        timestamp_field="release_ts",
        partition_template="{value}/release_calendar.parquet",
    )
    provider = PartitionedParquetCoverageProvider({"ml.macro_release_calendar": spec})

    buckets = provider.read_bucket_coverage(
        dataset_id="ml.macro_release_calendar",
        schema="macro_release_calendar",
        instrument_id="CPIAUCSL",
        start_ns=0,
        end_ns=DAY_NS,
    )
    assert buckets == set()


def test_partitioned_parquet_provider_filters_file_backed_datasets(tmp_path: Path) -> None:
    spec = ParquetCoverageSpec(
        dataset_id="ml.events_calendar",
        base_path=str(tmp_path),
        partition_field="instrument_id",
        timestamp_field="event_timestamp",
        partition_template="",
    )
    provider = PartitionedParquetCoverageProvider({"ml.events_calendar": spec})

    buckets = provider.read_bucket_coverage(
        dataset_id="ml.events_calendar",
        schema="events_calendar",
        instrument_id="AAPL",
        start_ns=0,
        end_ns=DAY_NS,
    )
    assert buckets == set()


def test_parquet_spec_rejects_bad_partition_templates(tmp_path: Path) -> None:
    spec = ParquetCoverageSpec(
        dataset_id="ml.events_calendar",
        base_path=str(tmp_path),
        partition_field="instrument_id",
        timestamp_field="event_timestamp",
        partition_template="{value",
    )
    assert spec.partition_template == "{value"


def test_catalog_coverage_provider_prefers_schema_template(monkeypatch, tmp_path: Path) -> None:
    captured: list[str] = []

    class _DummyCatalog:
        def __init__(self, *_: object, **__: object) -> None:
            return

        def get_intervals(self, *, data_cls: object, identifier: str) -> list[tuple[int, int]]:
            del data_cls
            captured.append(identifier)
            return []

    monkeypatch.setattr("ml.stores.providers.ParquetDataCatalog", _DummyCatalog)
    provider = CatalogCoverageProvider(
        catalog_path=str(tmp_path),
        schema_identifier_templates={"tbbo": "{instrument_id}-TBBO"},
    )

    provider.read_bucket_coverage(
        dataset_id="EQUS.MINI",
        schema="tbbo",
        instrument_id="AAPL.NYSE",
        start_ns=0,
        end_ns=DAY_NS,
    )

    assert captured == ["AAPL.NYSE-TBBO"]


def test_catalog_coverage_provider_falls_back_to_dataset_template(monkeypatch, tmp_path: Path) -> None:
    captured: list[str] = []

    class _DummyCatalog:
        def __init__(self, *_: object, **__: object) -> None:
            return

        def get_intervals(self, *, data_cls: object, identifier: str) -> list[tuple[int, int]]:
            del data_cls
            captured.append(identifier)
            return []

    monkeypatch.setattr("ml.stores.providers.ParquetDataCatalog", _DummyCatalog)
    provider = CatalogCoverageProvider(
        catalog_path=str(tmp_path),
        dataset_type_identifier_templates={DatasetType.TRADES: "{instrument_id}-TR"},
    )

    provider.read_bucket_coverage(
        dataset_id="EQUS.MINI",
        schema="trades",
        instrument_id="AAPL.NYSE",
        start_ns=0,
        end_ns=DAY_NS,
    )

    assert captured == ["AAPL.NYSE-TR"]


def test_catalog_coverage_provider_rejects_invalid_identifier_template(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CatalogCoverageProvider(
            catalog_path=str(tmp_path),
            identifier_template="{schema}",
        )


def test_catalog_coverage_provider_uses_registry_default_for_mbp(monkeypatch, tmp_path: Path) -> None:
    captured: list[str] = []

    class _DummyCatalog:
        def __init__(self, *_: object, **__: object) -> None:
            return

        def get_intervals(self, *, data_cls: object, identifier: str) -> list[tuple[int, int]]:
            del data_cls
            captured.append(identifier)
            return []

    monkeypatch.setattr("ml.stores.providers.ParquetDataCatalog", _DummyCatalog)
    provider = CatalogCoverageProvider(catalog_path=str(tmp_path))

    provider.read_bucket_coverage(
        dataset_id="EQUS.MINI",
        schema="mbp-1",
        instrument_id="AAPL.NYSE",
        start_ns=0,
        end_ns=DAY_NS,
    )

    assert captured == ["AAPL.NYSE"]
