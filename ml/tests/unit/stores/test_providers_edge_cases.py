from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from ml.config.market_data import MarketDataTableConfig
from ml.config.market_data import MarketDataTableProfile
from ml.data.coverage.types import GLOBAL_ENTITY_ID
from ml.stores import providers


def test_sql_coverage_provider_missing_table_returns_empty(tmp_path: Path) -> None:
    db_path = tmp_path / "coverage_missing.db"
    provider = providers.SqlCoverageProvider(connection_string=f"sqlite:///{db_path}")

    buckets = provider.read_bucket_coverage(
        dataset_id="ml.bars",
        schema="bars",
        instrument_id="AAPL",
        start_ns=0,
        end_ns=providers.DAY_NS,
    )
    assert buckets == set()

    latest = provider.latest_timestamp_ns(
        dataset_id="ml.bars",
        instrument_id="AAPL",
        schema="bars",
    )
    assert latest is None


def test_sql_coverage_provider_rejects_invalid_identifiers() -> None:
    with pytest.raises(ValueError):
        providers.SqlCoverageProvider(connection_string="sqlite://", table_name="bad-name")


def test_bucket_from_path_parses_partition_fields() -> None:
    path = Path("/tmp/year=2024/month=2/day=3/data.parquet")
    bucket = providers._bucket_from_path(path)
    expected = int(datetime(2024, 2, 3, tzinfo=UTC).timestamp() * 1_000_000_000) // providers.DAY_NS
    assert bucket == expected


def test_parquet_spec_files_for_instrument_file_backed(tmp_path: Path) -> None:
    file_path = tmp_path / "events.parquet"
    file_path.write_text("test")

    spec = providers.ParquetCoverageSpec(
        dataset_id="ml.events",
        base_path=str(file_path),
        partition_field="instrument_id",
        timestamp_field="event_ts",
    )

    assert spec.files_for_instrument("AAPL") == [str(file_path)]


def test_parquet_spec_template_render_failure_returns_empty(tmp_path: Path) -> None:
    spec = providers.ParquetCoverageSpec(
        dataset_id="ml.events",
        base_path=str(tmp_path),
        partition_field="instrument_id",
        timestamp_field="event_ts",
        partition_template="{value",
    )

    assert spec.files_for_instrument("AAPL") == []


def test_parquet_spec_global_entity_returns_all_files(tmp_path: Path) -> None:
    root_file = tmp_path / "root.parquet"
    root_file.write_text("root")
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_file = nested_dir / "nested.parquet"
    nested_file.write_text("nested")

    spec = providers.ParquetCoverageSpec(
        dataset_id="ml.events",
        base_path=str(tmp_path),
        partition_field="instrument_id",
        timestamp_field="event_ts",
    )

    files = spec.files_for_instrument(GLOBAL_ENTITY_ID)
    assert set(files) == {str(root_file), str(nested_file)}


def test_coerce_parquet_stat_to_ns_handles_dates() -> None:
    naive_dt = datetime(2024, 1, 1)
    aware_dt = datetime(2024, 1, 2, tzinfo=UTC)
    day = date(2024, 1, 3)

    assert providers._coerce_parquet_stat_to_ns(True) is None
    assert providers._coerce_parquet_stat_to_ns(1.25) == 1
    assert providers._coerce_parquet_stat_to_ns(naive_dt) == int(
        naive_dt.replace(tzinfo=UTC).timestamp() * 1_000_000_000,
    )
    assert providers._coerce_parquet_stat_to_ns(aware_dt) == int(
        aware_dt.timestamp() * 1_000_000_000,
    )
    assert providers._coerce_parquet_stat_to_ns(day) == int(
        datetime(2024, 1, 3, tzinfo=UTC).timestamp() * 1_000_000_000,
    )


def test_relation_kind_prefers_tables_then_views() -> None:
    class _Inspector:
        default_schema_name = "public"

        def __init__(self) -> None:
            self._tables = {"public": ["market_data"]}
            self._views = {"public": ["market_view"]}

        def get_table_names(self, *, schema: str | None = None):
            return self._tables.get(schema, [])

        def get_view_names(self, *, schema: str | None = None):
            return self._views.get(schema, [])

    inspector = _Inspector()
    assert providers._relation_kind(inspector, "market_data") == "table"
    assert providers._relation_kind(inspector, "market_view") == "view"
    assert providers._relation_kind(inspector, "missing") is None


def test_resolve_market_data_profile_respects_explicit_profile() -> None:
    engine = create_engine("sqlite://")
    config = MarketDataTableConfig(profile=MarketDataTableProfile.CLASS_TABLES)
    assert (
        providers._resolve_market_data_profile(engine, config=config)
        == MarketDataTableProfile.CLASS_TABLES
    )
