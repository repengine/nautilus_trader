from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import sqlite3

import pytest
from sqlalchemy import create_engine

from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.core.nautilus_pyo3 import DataBackendSession
from nautilus_trader.model.identifiers import InstrumentId

from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.config.market_data import MarketDataTableConfig
from ml.config.market_data import MarketDataTableProfile
from ml.data.coverage.manager import BucketSpec
from ml.registry.dataclasses import DatasetType

if TYPE_CHECKING:
    from ml.tests.fixtures.model_factory import TestDataFactory

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry", "mock_tracing_backend")

try:
    from ml.data.rehydration.catalog_rehydrator import (
        CatalogRehydrationConfig,
        CatalogRehydrationResult,
        ParquetCatalogRehydrator,
    )
except ModuleNotFoundError:  # pragma: no cover - module under test not yet implemented
    CatalogRehydrationConfig = None  # type: ignore[assignment]
    CatalogRehydrationResult = None  # type: ignore[assignment]
    ParquetCatalogRehydrator = None  # type: ignore[assignment]


class _StubWriter:
    def write(self, *args: object, **kwargs: object) -> int:
        _ = args, kwargs
        return 0


class _StubCoverage:
    def read_bucket_coverage(self, *args: object, **kwargs: object) -> set[int]:
        _ = args, kwargs
        return set()


class _StubRegistry:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.watermarks: list[dict[str, object]] = []

    def emit_event(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: object,
        source: object,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: object,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.events.append(
            {
                "dataset_id": dataset_id,
                "instrument_id": instrument_id,
                "stage": stage,
                "source": source,
                "run_id": run_id,
                "ts_min": ts_min,
                "ts_max": ts_max,
                "count": count,
                "status": status,
                "metadata": metadata,
            },
        )

    def update_watermark(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        source: object,
        last_success_ns: int,
        count: int,
        completeness_pct: float,
    ) -> None:
        self.watermarks.append(
            {
                "dataset_id": dataset_id,
                "instrument_id": instrument_id,
                "source": source,
                "last_success_ns": last_success_ns,
                "count": count,
                "completeness_pct": completeness_pct,
            },
        )


class _StubBackendSession:
    def to_query_result(self) -> list[object]:
        return []


class _StubCatalog:
    def __init__(self) -> None:
        self.session = None

    def backend_session(self, *args: object, **kwargs: object) -> _StubBackendSession:
        self.session = kwargs.get("session")
        return _StubBackendSession()


@contextmanager
def _patched_sqlite_engine(connection: str, patch_engine_manager) -> Iterator[None]:
    with ExitStack() as stack:
        engine = create_engine(connection)
        stack.callback(engine.dispose)
        stack.enter_context(patch_engine_manager(engine=engine))
        yield

def _build_catalog_with_bars(
    catalog_path: Path,
    *,
    symbol: str,
    start: datetime,
    count: int,
    data_factory: TestDataFactory,
) -> ParquetDataCatalog:
    catalog = ParquetDataCatalog(str(catalog_path))
    bars = data_factory.bars(
        n=count,
        instrument_id=symbol,
        bar_type=f"{symbol}-1-MINUTE-LAST-EXTERNAL",
        start_date=start,
    )
    catalog.write_data(bars)
    return catalog

@pytest.mark.skipif(ParquetCatalogRehydrator is None, reason="rehydrator not implemented yet")
class TestParquetCatalogRehydrator:
    def test_resolve_identifier_prefers_schema_template(self) -> None:
        config = CatalogRehydrationConfig(
            enabled=True,
            lookback_days=1,
            batch_size=10,
            schema_identifier_templates={"tbbo": "{instrument_id}-TBBO"},
        )
        rehydrator = ParquetCatalogRehydrator(
            catalog=ParquetDataCatalog(":memory:"),
            db_connection="sqlite://",
            config=config,
            writer=_StubWriter(),
            coverage_provider=_StubCoverage(),
        )

        identifier = rehydrator._resolve_identifier(
            schema="tbbo",
            instrument_id="SPY.XNAS",
        )

        assert identifier == "SPY.XNAS-TBBO"

    def test_resolve_identifier_prefers_dataset_template(self) -> None:
        config = CatalogRehydrationConfig(
            enabled=True,
            lookback_days=1,
            batch_size=10,
            dataset_type_identifier_templates={DatasetType.TRADES: "{instrument_id}-TR"},
        )
        rehydrator = ParquetCatalogRehydrator(
            catalog=ParquetDataCatalog(":memory:"),
            db_connection="sqlite://",
            config=config,
            writer=_StubWriter(),
            coverage_provider=_StubCoverage(),
        )

        identifier = rehydrator._resolve_identifier(
            schema="trades",
            instrument_id="SPY.XNAS",
        )

        assert identifier == "SPY.XNAS-TR"

    def test_resolve_identifier_defaults_to_registry_template(self) -> None:
        config = CatalogRehydrationConfig(
            enabled=True,
            lookback_days=1,
            batch_size=10,
        )
        rehydrator = ParquetCatalogRehydrator(
            catalog=ParquetDataCatalog(":memory:"),
            db_connection="sqlite://",
            config=config,
            writer=_StubWriter(),
            coverage_provider=_StubCoverage(),
        )

        identifier = rehydrator._resolve_identifier(
            schema="mbp-1",
            instrument_id="SPY.XNAS",
        )

        assert identifier == "SPY.XNAS"

    def test_stream_chunk_size_passed_to_backend_session(self) -> None:
        config = CatalogRehydrationConfig(
            enabled=True,
            lookback_days=1,
            batch_size=10,
            stream_chunk_size=5_000,
        )
        catalog = _StubCatalog()
        rehydrator = ParquetCatalogRehydrator(
            catalog=catalog,  # type: ignore[arg-type]
            db_connection="sqlite://",
            config=config,
            writer=_StubWriter(),
            coverage_provider=_StubCoverage(),
        )

        frames = list(
            rehydrator._iter_bucket_frames(
                dataset_type=DatasetType.QUOTES,
                instrument=InstrumentId.from_str("SPY.XNAS"),
                identifier="SPY.XNAS",
                dataset_id="EQUS.MINI_QUOTES",
                bucket_start_ns=0,
                bucket_end_ns=1,
            ),
        )

        assert frames == []
        assert catalog.session is not None
        assert isinstance(catalog.session, DataBackendSession)

    def test_rehydrator_routes_tables_by_schema(self, patch_engine_manager) -> None:
        config = CatalogRehydrationConfig(
            enabled=True,
            lookback_days=1,
            batch_size=10,
            table_config=MarketDataTableConfig(profile=MarketDataTableProfile.CLASS_TABLES),
        )
        connection = "sqlite://"
        with _patched_sqlite_engine(connection, patch_engine_manager):
            rehydrator = ParquetCatalogRehydrator(
                catalog=ParquetDataCatalog(":memory:"),
                db_connection=connection,
                config=config,
            )

            assert rehydrator._writer._resolve_table_name("tbbo") == "market_data_tbbo"
            assert rehydrator._writer._resolve_table_name("ohlcv-1m") == "market_data_bar"
            assert rehydrator._writer._resolve_table_name("mbp-10") == "market_data_mbp10"
            assert rehydrator._writer._resolve_table_name("mbo") == "market_data_mbo"
            assert rehydrator._coverage._resolve_table_name("tbbo") == "market_data_tbbo"

    def test_rehydrate_restores_missing_buckets(
        self,
        tmp_path: Path,
        patch_engine_manager,
        test_data_factory: TestDataFactory,
    ) -> None:
        symbol = "NVDA.XNAS"
        start = datetime(2024, 1, 1, tzinfo=UTC)
        catalog = _build_catalog_with_bars(
            tmp_path / "catalog",
            symbol=symbol,
            start=start,
            count=32,
            data_factory=test_data_factory,
        )

        db_path = tmp_path / "rehydrate.db"
        connection = f"sqlite:///{db_path}"

        with _patched_sqlite_engine(connection, patch_engine_manager):
            config = CatalogRehydrationConfig(enabled=True, lookback_days=7, batch_size=500)
            rehydrator = ParquetCatalogRehydrator(
                catalog=catalog,
                db_connection=connection,
                config=config,
            )

            reference_time = start + timedelta(days=2)
            result = rehydrator.rehydrate_missing_data(
                dataset_id="EQUS.MINI",
                schema="ohlcv-1m",
                instrument_ids=[symbol],
                reference_time=reference_time,
            )

        assert isinstance(result, CatalogRehydrationResult)
        assert result.rows_written == 32
        assert result.buckets_restored >= 1
        assert result.failures == {}

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM market_data WHERE instrument_id = ?", (symbol,))
            row_count = cursor.fetchone()[0]
            assert row_count == 32

    def test_rehydrate_emits_events_when_registry_present_returns_expected(
        self,
        tmp_path: Path,
        patch_engine_manager,
        test_data_factory: TestDataFactory,
    ) -> None:
        symbol = "AMD.XNAS"
        start = datetime(2024, 3, 1, tzinfo=UTC)
        catalog = _build_catalog_with_bars(
            tmp_path / "catalog",
            symbol=symbol,
            start=start,
            count=12,
            data_factory=test_data_factory,
        )

        db_path = tmp_path / "rehydrate_events.db"
        connection = f"sqlite:///{db_path}"
        registry = _StubRegistry()

        with _patched_sqlite_engine(connection, patch_engine_manager):
            config = CatalogRehydrationConfig(enabled=True, lookback_days=7, batch_size=500)
            rehydrator = ParquetCatalogRehydrator(
                catalog=catalog,
                db_connection=connection,
                config=config,
                registry=registry,
            )

            reference_time = start + timedelta(days=2)
            result = rehydrator.rehydrate_missing_data(
                dataset_id="EQUS.MINI",
                schema="ohlcv-1m",
                instrument_ids=[symbol],
                reference_time=reference_time,
            )

        assert result.rows_written == 12
        assert registry.events
        assert registry.watermarks
        event = registry.events[0]
        watermark = registry.watermarks[0]
        assert event["dataset_id"] == "EQUS.MINI"
        assert event["instrument_id"] == symbol
        assert getattr(event["stage"], "value", None) == Stage.DATA_INGESTED.value
        assert getattr(event["source"], "value", None) == Source.BACKFILL.value
        assert getattr(event["status"], "value", None) == EventStatus.SUCCESS.value
        assert event["run_id"] == "catalog_rehydrate"
        metadata = event.get("metadata") or {}
        assert metadata.get("schema") == "ohlcv-1m"
        assert "bucket" in metadata
        assert watermark["dataset_id"] == "EQUS.MINI"
        assert watermark["instrument_id"] == symbol
        assert getattr(watermark["source"], "value", None) == Source.BACKFILL.value
        assert watermark["completeness_pct"] == 100.0

    def test_rehydrate_skips_existing_coverage(
        self,
        tmp_path: Path,
        patch_engine_manager,
        test_data_factory: TestDataFactory,
    ) -> None:
        symbol = "AAPL.XNAS"
        start = datetime(2024, 2, 1, tzinfo=UTC)
        catalog = _build_catalog_with_bars(
            tmp_path / "catalog",
            symbol=symbol,
            start=start,
            count=16,
            data_factory=test_data_factory,
        )

        db_path = tmp_path / "rehydrate.db"
        connection = f"sqlite:///{db_path}"

        with _patched_sqlite_engine(connection, patch_engine_manager):
            config = CatalogRehydrationConfig(enabled=True, lookback_days=7, batch_size=500)
            rehydrator = ParquetCatalogRehydrator(
                catalog=catalog,
                db_connection=connection,
                config=config,
            )

            reference_time = start + timedelta(days=2)

            first_result = rehydrator.rehydrate_missing_data(
                dataset_id="EQUS.MINI",
                schema="ohlcv-1m",
                instrument_ids=[symbol],
                reference_time=reference_time,
            )
            assert first_result.rows_written == 16

            second_result = rehydrator.rehydrate_missing_data(
                dataset_id="EQUS.MINI",
                schema="ohlcv-1m",
                instrument_ids=[symbol],
                reference_time=reference_time,
            )

        assert second_result.rows_written == 0
        assert second_result.buckets_restored == 0
        assert second_result.failures == {}

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM market_data WHERE instrument_id = ?", (symbol,))
            row_count = cursor.fetchone()[0]
            assert row_count == 16

    def test_exhaustive_mode_expands_window(
        self,
        tmp_path: Path,
        patch_engine_manager,
        test_data_factory: TestDataFactory,
    ) -> None:
        symbol = "MSFT.XNAS"
        start = datetime(2023, 1, 1, tzinfo=UTC)
        catalog = _build_catalog_with_bars(
            tmp_path / "catalog",
            symbol=symbol,
            start=start,
            count=32,
            data_factory=test_data_factory,
        )

        db_path = tmp_path / "rehydrate_exhaustive.db"
        connection = f"sqlite:///{db_path}"
        reference_time = start + timedelta(days=60)

        with _patched_sqlite_engine(connection, patch_engine_manager):
            baseline_config = CatalogRehydrationConfig(enabled=True, lookback_days=1, batch_size=500)
            baseline_rehydrator = ParquetCatalogRehydrator(
                catalog=catalog,
                db_connection=connection,
                config=baseline_config,
            )
            baseline_result = baseline_rehydrator.rehydrate_missing_data(
                dataset_id="EQUS.MINI",
                schema="ohlcv-1m",
                instrument_ids=[symbol],
                reference_time=reference_time,
            )
            assert baseline_result.rows_written == 0

            exhaustive_config = CatalogRehydrationConfig(
                enabled=True,
                lookback_days=1,
                batch_size=500,
                exhaustive=True,
            )
            exhaustive_rehydrator = ParquetCatalogRehydrator(
                catalog=catalog,
                db_connection=connection,
                config=exhaustive_config,
            )
            exhaustive_result = exhaustive_rehydrator.rehydrate_missing_data(
                dataset_id="EQUS.MINI",
                schema="ohlcv-1m",
                instrument_ids=[symbol],
                reference_time=reference_time,
            )

        assert exhaustive_result.rows_written == 32
        assert exhaustive_result.buckets_restored >= 1

    def test_rehydrate_accepts_explicit_bucket_filters(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        patch_engine_manager,
        test_data_factory: TestDataFactory,
    ) -> None:
        symbol = "META.XNAS"
        start = datetime(2024, 5, 1, tzinfo=UTC)
        catalog = _build_catalog_with_bars(
            tmp_path / "catalog",
            symbol=symbol,
            start=start,
            count=16,
            data_factory=test_data_factory,
        )
        db_path = tmp_path / "rehydrate_bucket.db"
        connection = f"sqlite:///{db_path}"

        captured: dict[str, set[int] | None] = {}

        with _patched_sqlite_engine(connection, patch_engine_manager):
            config = CatalogRehydrationConfig(enabled=True, lookback_days=7, batch_size=500)
            rehydrator = ParquetCatalogRehydrator(
                catalog=catalog,
                db_connection=connection,
                config=config,
            )

            def _fake_rehydrate_instrument(**kwargs: object) -> tuple[int, int, int]:
                captured["target_buckets"] = kwargs.get("target_buckets")
                return 0, 0, 0

            monkeypatch.setattr(rehydrator, "_rehydrate_instrument", _fake_rehydrate_instrument)  # type: ignore[arg-type]

            bucket = BucketSpec(
                dataset_id="EQUS.MINI",
                schema="ohlcv-1m",
                instrument_id=symbol,
                bucket_start_ns=int(start.timestamp() * 1_000_000_000),
            )
            rehydrator.rehydrate_missing_data(
                dataset_id="EQUS.MINI",
                schema="ohlcv-1m",
                instrument_ids=[symbol],
                buckets=(bucket,),
            )

        assert captured["target_buckets"] == {bucket.bucket_index}

    def test_identifier_resolution_respects_schema(
        self,
        tmp_path: Path,
        patch_engine_manager,
    ) -> None:
        catalog = ParquetDataCatalog(str(tmp_path / "catalog"))
        db_path = tmp_path / "identifiers.db"
        connection = f"sqlite:///{db_path}"

        with _patched_sqlite_engine(connection, patch_engine_manager):
            config = CatalogRehydrationConfig(
                enabled=True,
                lookback_days=2,
                batch_size=100,
                identifier_template="{instrument_id}-1-MINUTE-LAST-EXTERNAL",
            )
            rehydrator = ParquetCatalogRehydrator(
                catalog=catalog,
                db_connection=connection,
                config=config,
            )

            bar_identifier = rehydrator._resolve_identifier(
                schema="ohlcv-1m",
                instrument_id="AAPL.XNAS",
            )
            tbbo_identifier = rehydrator._resolve_identifier(
                schema="tbbo",
                instrument_id="AAPL.XNAS",
            )

        assert bar_identifier == "AAPL.XNAS-1-MINUTE-LAST-EXTERNAL"
        assert tbbo_identifier == "AAPL.XNAS"
