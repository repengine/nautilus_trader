from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Sequence

import pandas as pd
import pytest

from ml.config.databento_policy import DatabentoSafetyConfig
from ml.config.databento_policy import SchemaSafetyConfig
from ml.data.ingest.api import IngestionJob
from ml.data.ingest.api import fetch_symbol_data
from ml.data.ingest.api import run_jobs
from ml.data.ingest.policy import DatabentoCoveragePolicy
from ml.data.ingest.calibration import CalibrationBundle
from ml.data.ingest.calibration import SymbolCalibration
from ml.data.ingest.service import CostViolationError
from ml.data.ingest.service import IngestionError
from ml.data.ingest.service import DatabentoIngestionService
from ml.data.ingest.service import IngestionChunk
from ml.data.ingest.service import IngestionRequest
from ml.data.ingest.service import build_historical_adapter
from ml.data.ingest.service import build_like_client
from ml.registry.dataclasses import StorageKind


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)


class _FakeTimeseries:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_range(self, **kwargs: Any) -> _FakeResult:
        self.calls.append(kwargs)
        rows = [
            {
                "ts_event": int(datetime.fromisoformat(kwargs["start"]).timestamp() * 1e9),
                "price": 1.0,
            },
            {
                "ts_event": int(datetime.fromisoformat(kwargs["end"]).timestamp() * 1e9) - 1,
                "price": 2.0,
            },
        ]
        return _FakeResult(rows)


class _FakeMetadata:
    def __init__(self, cost: float) -> None:
        self.cost = float(cost)
        self.requests: list[dict[str, Any]] = []

    def get_cost(self, **kwargs: Any) -> float:
        self.requests.append(kwargs)
        return self.cost

    def get_dataset_range(self, dataset: str) -> dict[str, Any]:
        return {"start": "2025-09-01T00:00:00Z", "end": "2025-09-02T00:00:00Z"}


class _SchemaAwareMetadata:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def get_cost(self, **_: Any) -> float:
        return 0.0

    def get_dataset_range(self, dataset: str) -> dict[str, Any]:
        self.calls += 1
        return self.payload


class _FakeHistorical:
    def __init__(self, *, cost: float) -> None:
        self.metadata = _FakeMetadata(cost)
        self.timeseries = _FakeTimeseries()
        self.symbology = _SimpleSymbology()


class _SimpleSymbology:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def resolve(
        self,
        *,
        symbols: Sequence[str],
        dataset: str,
        stype_in: str,
        stype_out: str,
        start_date: str,
        end_date: str | None = None,
    ) -> dict[str, object]:
        del stype_in, stype_out, start_date, end_date
        symbol = symbols[0]
        dataset_id = dataset or ""
        root = symbol.split(".")[0]
        self.calls.append((dataset_id, root))
        return {"result": {symbol: ({"s": "1", "symbol": root},)}}


class _MetadataHistorical:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.metadata = _SchemaAwareMetadata(payload)
        self.timeseries = _FakeTimeseries()
        self.symbology: Any | None = None


class _VariantMetadata:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def get_cost(
        self,
        *,
        dataset: str,
        symbols: Sequence[str],
        schema: str,
        start: str | datetime,
        end: str | datetime,
    ) -> float:
        entry = (dataset, tuple(symbols))
        self.calls.append(entry)
        if dataset == "XNAS.ITCH" and symbols[0] == "INTC":
            return 0.0
        raise ValueError("unsupported symbol")

    def get_dataset_range(self, dataset: str) -> dict[str, Any]:
        base_start = datetime(2025, 9, 1, tzinfo=UTC)
        base_end = datetime(2025, 9, 2, tzinfo=UTC)
        return {
            "start": base_start.isoformat(),
            "end": base_end.isoformat(),
        }

    def list_datasets(self) -> Sequence[str]:
        return ("XNAS.ITCH",)

    def list_schemas(self, *, dataset: str) -> Sequence[str]:
        return ("trades",)


class _VariantHistorical:
    def __init__(self) -> None:
        self.metadata = _VariantMetadata()
        self.timeseries = _FakeTimeseries()
        self.symbology = _VariantSymbology()


class _VariantSymbology:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def resolve(
        self,
        *,
        symbols: Sequence[str],
        dataset: str,
        stype_in: str,
        stype_out: str,
        start_date: str,
        end_date: str | None = None,
    ) -> dict[str, object]:
        del stype_in, stype_out, start_date, end_date
        symbol = symbols[0]
        dataset_id = dataset or ""
        self.calls.append((dataset_id, symbol))
        base = symbol.split(".")[0]
        if dataset_id == "XNAS.ITCH":
            return {
                "result": {
                    symbol: (
                        {"s": "4182", "symbol": f"{base}.XNAS"},
                        {"s": "4182", "symbol": base},
                    ),
                },
            }
        return {"result": {symbol: ({"s": "0", "symbol": base},)}}


class _FallbackTimeseries:
    def __init__(
        self,
        *,
        fallback_rows_pre: list[dict[str, object]],
        fallback_rows_reference: list[dict[str, object]],
        trade_rows: list[dict[str, object]],
        eq_reference_rows: list[dict[str, object]],
        eq_reference_start: datetime,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self._fallback_rows_pre = fallback_rows_pre
        self._fallback_rows_reference = fallback_rows_reference
        self._trade_rows = trade_rows
        self._eq_reference_rows = eq_reference_rows
        self._eq_reference_start = eq_reference_start

    @property
    def fallback_rows(self) -> list[dict[str, object]]:
        return self._fallback_rows_pre

    def get_range(self, **kwargs: object) -> pd.DataFrame:
        record = {key: kwargs[key] for key in ("dataset", "symbols", "schema") if key in kwargs}
        self.calls.append(record)
        dataset = str(kwargs.get("dataset") or "")
        schema = str(kwargs.get("schema") or "").lower()
        start_raw = kwargs.get("start")
        start_dt = pd.to_datetime(start_raw, utc=True) if start_raw is not None else None
        if dataset == "EQUS.MINI":
            if start_dt is not None and start_dt >= self._eq_reference_start:
                return pd.DataFrame(self._eq_reference_rows)
            return pd.DataFrame()
        if dataset == "XNAS.ITCH":
            if schema == "trades":
                if start_dt is not None and start_dt < self._eq_reference_start:
                    return pd.DataFrame(self._trade_rows)
                return pd.DataFrame()
            if start_dt is not None and start_dt >= self._eq_reference_start:
                return pd.DataFrame(self._fallback_rows_reference)
            return pd.DataFrame(self._fallback_rows_pre)
        return pd.DataFrame()


class _FallbackMetadata:
    def __init__(self, eq_start: datetime) -> None:
        self.eq_start = eq_start

    def get_cost(self, **_: object) -> float:
        return 0.0

    def get_dataset_range(self, dataset: str) -> dict[str, object]:
        if dataset == "EQUS.MINI":
            start_iso = self.eq_start.isoformat()
            end_iso = datetime(2024, 1, 1, tzinfo=UTC).isoformat()
            return {
                "start": start_iso,
                "end": end_iso,
                "schema": {
                    "ohlcv-1m": {
                        "start": start_iso,
                        "end": end_iso,
                    },
                },
            }
        return {
            "start": datetime(2018, 1, 1, tzinfo=UTC).isoformat(),
            "end": datetime(2024, 1, 1, tzinfo=UTC).isoformat(),
        }


class _FallbackHistorical:
    def __init__(self) -> None:
        fallback_rows_pre = [
            {
                "ts_event": int(datetime(2020, 5, 1, 14, 30, tzinfo=UTC).timestamp() * 1_000_000_000),
                "ts_init": int(datetime(2020, 5, 1, 14, 30, tzinfo=UTC).timestamp() * 1_000_000_000),
                "rtype": 33,
                "publisher_id": 2,
                "instrument_id": "INTC.XNAS",
                "open": 29.0,
                "high": 29.1,
                "low": 28.9,
                "close": 29.05,
                "volume": 400.0,
                "symbol": "INTC",
            },
            {
                "ts_event": int(datetime(2020, 5, 1, 14, 31, tzinfo=UTC).timestamp() * 1_000_000_000),
                "ts_init": int(datetime(2020, 5, 1, 14, 31, tzinfo=UTC).timestamp() * 1_000_000_000),
                "rtype": 33,
                "publisher_id": 2,
                "instrument_id": "INTC.XNAS",
                "open": 29.05,
                "high": 29.07,
                "low": 29.02,
                "close": 29.06,
                "volume": 250.0,
                "symbol": "INTC",
            },
        ]
        trade_rows = [
            {
                "ts_event": int(datetime(2020, 5, 1, 14, 30, 0, tzinfo=UTC).timestamp() * 1_000_000_000),
                "price": 29.00,
                "size": 100,
                "sale_condition": "@",
            },
            {
                "ts_event": int(datetime(2020, 5, 1, 14, 30, 10, tzinfo=UTC).timestamp() * 1_000_000_000),
                "price": 29.05,
                "size": 150,
                "sale_condition": "A",
            },
            {
                "ts_event": int(datetime(2020, 5, 1, 14, 30, 30, tzinfo=UTC).timestamp() * 1_000_000_000),
                "price": 29.04,
                "size": 150,
                "sale_condition": "@",
            },
            {
                "ts_event": int(datetime(2020, 5, 1, 14, 31, 0, tzinfo=UTC).timestamp() * 1_000_000_000),
                "price": 29.05,
                "size": 120,
                "sale_condition": "@",
            },
            {
                "ts_event": int(datetime(2020, 5, 1, 14, 31, 20, tzinfo=UTC).timestamp() * 1_000_000_000),
                "price": 29.06,
                "size": 130,
                "sale_condition": "Z",
            },
        ]
        eq_reference_start = datetime(2023, 1, 1, tzinfo=UTC)
        fallback_reference_rows = [
            {
                "ts_event": int(datetime(2023, 1, 2, 14, 30, tzinfo=UTC).timestamp() * 1_000_000_000),
                "ts_init": int(datetime(2023, 1, 2, 14, 30, tzinfo=UTC).timestamp() * 1_000_000_000),
                "rtype": 33,
                "publisher_id": 2,
                "instrument_id": "INTC.XNAS",
                "open": 30.0,
                "high": 30.1,
                "low": 29.9,
                "close": 30.05,
                "volume": 420.0,
                "symbol": "INTC",
            },
            {
                "ts_event": int(datetime(2023, 1, 2, 14, 31, tzinfo=UTC).timestamp() * 1_000_000_000),
                "ts_init": int(datetime(2023, 1, 2, 14, 31, tzinfo=UTC).timestamp() * 1_000_000_000),
                "rtype": 33,
                "publisher_id": 2,
                "instrument_id": "INTC.XNAS",
                "open": 30.05,
                "high": 30.08,
                "low": 29.96,
                "close": 30.02,
                "volume": 360.0,
                "symbol": "INTC",
            },
        ]
        eq_reference_rows = [
            {
                "ts_event": int(datetime(2023, 1, 2, 14, 30, tzinfo=UTC).timestamp() * 1_000_000_000),
                "ts_init": int(datetime(2023, 1, 2, 14, 30, tzinfo=UTC).timestamp() * 1_000_000_000),
                "rtype": 33,
                "publisher_id": 95,
                "instrument_id": "INTC.XNAS",
                "open": 30.0,
                "high": 30.12,
                "low": 29.95,
                "close": 30.07,
                "volume": 840.0,
                "symbol": "INTC",
            },
            {
                "ts_event": int(datetime(2023, 1, 2, 14, 31, tzinfo=UTC).timestamp() * 1_000_000_000),
                "ts_init": int(datetime(2023, 1, 2, 14, 31, tzinfo=UTC).timestamp() * 1_000_000_000),
                "rtype": 33,
                "publisher_id": 95,
                "instrument_id": "INTC.XNAS",
                "open": 30.05,
                "high": 30.1,
                "low": 29.98,
                "close": 30.03,
                "volume": 720.0,
                "symbol": "INTC",
            },
        ]
        self.timeseries = _FallbackTimeseries(
            fallback_rows_pre=fallback_rows_pre,
            fallback_rows_reference=fallback_reference_rows,
            trade_rows=trade_rows,
            eq_reference_rows=eq_reference_rows,
            eq_reference_start=eq_reference_start,
        )
        self.metadata = _FallbackMetadata(datetime(2023, 1, 1, tzinfo=UTC))
        self.symbology = _SimpleSymbology()


@pytest.fixture()
def safety_config() -> DatabentoSafetyConfig:
    return DatabentoSafetyConfig(
        datasets=("EQUS.MINI", "XNAS.ITCH"),
        schemas={
            "trades": SchemaSafetyConfig(max_days=2),
        },
        max_cost_usd=0.0,
        max_symbols=100,
    )


def _request(start: datetime, end: datetime) -> IngestionRequest:
    return IngestionRequest(
        dataset="EQUS.MINI",
        schema="trades",
        symbols=("SPY",),
        start=start,
        end=end,
        reason="test",
    )


def test_ingest_returns_summary_and_invokes_callback(safety_config: DatabentoSafetyConfig) -> None:
    service = DatabentoIngestionService(
        client=_FakeHistorical(cost=0.0),
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
    )
    start = datetime(2025, 9, 1, tzinfo=UTC)
    end = start + timedelta(days=3)
    chunks: list[IngestionChunk] = []
    summaries = service.ingest(_request(start, end), on_chunk=chunks.append)
    assert len(summaries) == 1
    summary = summaries[0]
    # Schema policy max_days=2 should split into two windows
    assert len(summary.requested_windows) == 2
    assert summary.frames_returned == 2
    assert summary.rows_returned == 4
    assert len(chunks) == 2
    assert all(chunk.frame.shape[0] == 2 for chunk in chunks)


def test_ingest_cost_violation_raises(safety_config: DatabentoSafetyConfig) -> None:
    service = DatabentoIngestionService(
        client=_FakeHistorical(cost=12.5),
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
    )
    start = datetime(2025, 9, 1, tzinfo=UTC)
    with pytest.raises(CostViolationError):
        service.ingest(_request(start, start + timedelta(days=1)))


def test_ingest_allow_cost_bypasses_guard(safety_config: DatabentoSafetyConfig) -> None:
    service = DatabentoIngestionService(
        client=_FakeHistorical(cost=99.0),
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
    )
    start = datetime(2025, 9, 1, tzinfo=UTC)
    request = _request(start, start + timedelta(days=1))
    request = IngestionRequest(
        dataset=request.dataset,
        schema=request.schema,
        symbols=request.symbols,
        start=request.start,
        end=request.end,
        allow_cost=True,
        reason=request.reason,
    )
    summaries = service.ingest(request)
    assert summaries[0].rows_returned == 2


def test_ingest_fallback_canonicalizes_itch(safety_config: DatabentoSafetyConfig) -> None:
    client = _FallbackHistorical()
    calibration = CalibrationBundle(
        generated_at=datetime(2025, 1, 1, tzinfo=UTC),
        symbols={
            "INTC": SymbolCalibration(
                sale_condition_allowlist=frozenset({"@", "A"}),
                volume_scale_by_minute={870: 0.5, 871: 1.0},
                price_scaling_by_minute={870: 2.0, 871: 1.0},
                split_events={},
                exclude_auction_minutes=frozenset(),
            ),
        },
    )
    service = DatabentoIngestionService(
        client=client,
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
        calibration=calibration,
    )
    start = datetime(2020, 5, 1, tzinfo=UTC)
    end = start + timedelta(minutes=2)
    request = IngestionRequest(
        dataset="EQUS.MINI",
        schema="ohlcv-1m",
        symbols=("INTC",),
        start=start,
        end=end,
        reason="fallback",
    )
    chunks: list[IngestionChunk] = []
    summaries = service.ingest(request, on_chunk=chunks.append)
    assert summaries
    assert client.timeseries.calls[0]["dataset"] == "EQUS.MINI"
    assert client.timeseries.calls[1]["dataset"] == "XNAS.ITCH"
    assert any(call.get("schema") == "trades" for call in client.timeseries.calls)
    assert summaries[0].rows_returned == len(client.timeseries.fallback_rows)
    assert chunks
    frame = chunks[0].frame
    assert not frame.empty
    assert set(frame["publisher_id"]) == {95}
    assert frame["volume"].dtype == "int64"
    assert list(frame["volume"]) == [200, 120]
    assert list(frame.get("trade_count", [])) == [3, 1]
    assert frame["source_dataset"].unique().tolist() == ["XNAS.ITCH"]
    assert frame["aggregation_mode"].unique().tolist() == ["calibrated_reaggregated_trades"]
    assert frame["close"].iloc[0] == pytest.approx(58.08, rel=1e-6)
    assert frame["calibration_version"].unique().tolist() == ["2025-01-01T00:00:00+00:00"]


def test_ingest_fallback_scaling_applies_volume_factor(
    monkeypatch: pytest.MonkeyPatch,
    safety_config: DatabentoSafetyConfig,
) -> None:
    monkeypatch.setenv("ML_EQUS_ENABLE_TRADE_REAGG", "0")
    monkeypatch.setenv("ML_EQUS_ENABLE_VOLUME_SCALING", "1")
    client = _FallbackHistorical()
    service = DatabentoIngestionService(
        client=client,
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
    )
    start = datetime(2020, 5, 1, tzinfo=UTC)
    end = start + timedelta(minutes=2)
    request = IngestionRequest(
        dataset="EQUS.MINI",
        schema="ohlcv-1m",
        symbols=("INTC",),
        start=start,
        end=end,
        reason="fallback_scale",
    )
    chunks: list[IngestionChunk] = []
    summaries = service.ingest(request, on_chunk=chunks.append)
    assert summaries
    assert chunks
    frame = chunks[0].frame
    assert list(frame["volume"]) == [800, 500]
    assert frame["aggregation_mode"].unique().tolist() == ["scaled_volume"]
    assert frame["source_dataset"].unique().tolist() == ["XNAS.ITCH"]
    assert frame["calibration_version"].isna().all()


def test_ingest_fallback_applies_split_adjustment(
    safety_config: DatabentoSafetyConfig,
) -> None:
    client = _FallbackHistorical()
    calibration = CalibrationBundle(
        generated_at=datetime(2025, 1, 1, tzinfo=UTC),
        symbols={
            "INTC": SymbolCalibration(
                sale_condition_allowlist=frozenset({"@", "A", "Z"}),
                volume_scale_by_minute={},
                price_scaling_by_minute={},
                split_events={"2020-05-01": 0.5},
                exclude_auction_minutes=frozenset(),
            ),
        },
    )
    service = DatabentoIngestionService(
        client=client,
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
        calibration=calibration,
    )
    start = datetime(2020, 5, 1, tzinfo=UTC)
    end = start + timedelta(minutes=2)
    request = IngestionRequest(
        dataset="EQUS.MINI",
        schema="ohlcv-1m",
        symbols=("INTC",),
        start=start,
        end=end,
        reason="fallback_split",
    )
    chunks: list[IngestionChunk] = []
    service.ingest(request, on_chunk=chunks.append)
    assert chunks
    frame = chunks[0].frame
    assert list(frame["volume"]) == [800, 500]
    assert frame["close"].iloc[0] == pytest.approx(14.52, rel=1e-6)
    assert frame["close"].iloc[1] == pytest.approx(14.53, rel=1e-6)
    assert frame["calibration_version"].unique().tolist() == ["2025-01-01T00:00:00+00:00"]


def test_cost_guard_symbol_variants(safety_config: DatabentoSafetyConfig) -> None:
    client = _VariantHistorical()
    service = DatabentoIngestionService(
        client=client,
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
    )
    start = datetime(2025, 9, 1, tzinfo=UTC)
    end = start + timedelta(days=1)
    request = IngestionRequest(
        dataset="XNAS.ITCH",
        schema="trades",
        symbols=("INTC.XNAS",),
        start=start,
        end=end,
    )
    summaries = service.ingest(request)
    assert summaries
    metadata = service._client.metadata  # type: ignore[attr-defined]
    assert ("XNAS.ITCH", ("INTC",)) in metadata.calls
    assert summaries[0].symbol == "INTC"
    assert any(call.get("symbols") == "INTC" for call in client.timeseries.calls)
    assert ("XNAS.ITCH", "INTC") in client.symbology.calls


def test_discover_symbol_dataset_returns_resolution(safety_config: DatabentoSafetyConfig) -> None:
    client = _VariantHistorical()
    service = DatabentoIngestionService(
        client=client,
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
    )
    start = datetime(2025, 9, 1, tzinfo=UTC)
    end = start + timedelta(days=1)
    start_ns = int(start.timestamp() * 1_000_000_000)
    end_ns = int(end.timestamp() * 1_000_000_000)
    discovery = service.discover_symbol_dataset(
        symbol="INTC.XNAS",
        schema="trades",
        start_ns=start_ns,
        end_ns=end_ns,
    )
    assert discovery is not None
    assert discovery.dataset_id == "XNAS.ITCH"
    assert discovery.symbol == "INTC"
    assert discovery.requested_symbol == "INTC.XNAS"
    assert discovery.storage_kind == StorageKind.POSTGRES
    assert discovery.instrument_id == "4182"


def test_build_historical_adapter_provides_timeseries(safety_config: DatabentoSafetyConfig) -> None:
    service = DatabentoIngestionService(
        client=_FakeHistorical(cost=0.0),
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
    )
    adapter = build_historical_adapter(service)
    result = adapter.timeseries.get_range(
        dataset="EQUS.MINI",
        symbols="SPY",
        schema="trades",
        start=datetime(2025, 9, 1, tzinfo=UTC),
        end=datetime(2025, 9, 2, tzinfo=UTC),
    )
    df = result.to_df()
    assert not df.empty


def test_build_like_client_returns_dataframe(safety_config: DatabentoSafetyConfig) -> None:
    service = DatabentoIngestionService(
        client=_FakeHistorical(cost=0.0),
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
    )
    like_client = build_like_client(service)
    df = like_client.get_data(
        dataset="EQUS.MINI",
        symbols=["SPY"],
        schema="trades",
        start=datetime(2025, 9, 1, tzinfo=UTC),
        end=datetime(2025, 9, 2, tzinfo=UTC),
    )
    assert not df.empty


def test_fetch_symbol_data_concatenates_chunks(safety_config: DatabentoSafetyConfig) -> None:
    service = DatabentoIngestionService(
        client=_FakeHistorical(cost=0.0),
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
    )
    start = datetime(2025, 9, 1, tzinfo=UTC)
    end = start + timedelta(days=2)
    df = fetch_symbol_data(
        service=service,
        dataset="EQUS.MINI",
        schema="trades",
        symbol="SPY",
        start=start,
        end=end,
        chunk_days=1,
    )
    assert len(df.index) == 4


def test_run_jobs_executes_all_jobs(safety_config: DatabentoSafetyConfig) -> None:
    service = DatabentoIngestionService(
        client=_FakeHistorical(cost=0.0),
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
    )
    start = datetime(2025, 9, 1, tzinfo=UTC)
    end = start + timedelta(days=1)
    jobs = [
        IngestionJob(
            dataset="EQUS.MINI",
            schema="trades",
            symbols=("SPY",),
            start=start,
            end=end,
            reason="job1",
        ),
        IngestionJob(
            dataset="EQUS.MINI",
            schema="trades",
            symbols=("QQQ",),
            start=start,
            end=end,
            reason="job2",
        ),
    ]
    summaries = run_jobs(jobs, service=service)
    assert len(summaries) == 2


def test_get_available_range_ns_prefers_schema_bounds(
    safety_config: DatabentoSafetyConfig,
) -> None:
    dataset_start = datetime(2025, 9, 1, tzinfo=UTC)
    dataset_end = datetime(2025, 9, 30, tzinfo=UTC)
    schema_start = datetime(2025, 9, 5, tzinfo=UTC)
    schema_end = datetime(2025, 9, 10, tzinfo=UTC)
    payload = {
        "start": dataset_start.isoformat(),
        "end": dataset_end.isoformat(),
        "schema": {
            "trades": {
                "start": schema_start.isoformat(),
                "end": schema_end.isoformat(),
            },
        },
    }
    service = DatabentoIngestionService(
        client=_MetadataHistorical(payload),
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
    )

    start_ns, end_ns = service.get_available_range_ns(dataset="EQUS.MINI", schema="TRADES")
    assert start_ns is not None and end_ns is not None

    observed_start = datetime.fromtimestamp(start_ns / 1_000_000_000, tz=UTC)
    observed_end = datetime.fromtimestamp(end_ns / 1_000_000_000, tz=UTC)
    assert observed_start == schema_start
    assert observed_end == schema_end

    # Cached result should avoid extra metadata lookups
    _ = service.get_available_range_ns(dataset="EQUS.MINI", schema="trades")
    metadata_client = service.metadata_client
    assert isinstance(metadata_client, _SchemaAwareMetadata)
    assert metadata_client.calls == 1


def test_from_env_injects_provided_key(
    monkeypatch: pytest.MonkeyPatch,
    safety_config: DatabentoSafetyConfig,
) -> None:
    stub_module = SimpleNamespace(Historical=lambda key: SimpleNamespace(api_key=key))
    monkeypatch.setitem(sys.modules, "databento", stub_module)
    monkeypatch.setattr(
        "ml.data.ingest.service.load_databento_safety_config",
        lambda _path=None: safety_config,
    )
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)

    service = DatabentoIngestionService.from_env(api_key=" provided ")

    assert isinstance(service, DatabentoIngestionService)
    assert os.getenv("DATABENTO_API_KEY") == "provided"
    client = getattr(service, "_client")
    assert getattr(client, "api_key") == "provided"


def test_from_env_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_module = SimpleNamespace(Historical=lambda key: SimpleNamespace(api_key=key))
    monkeypatch.setitem(sys.modules, "databento", stub_module)
    monkeypatch.setattr(
        "ml.data.ingest.service.load_databento_safety_config",
        lambda _path=None: DatabentoSafetyConfig(
            datasets=("EQUS.MINI",),
            schemas={},
            max_cost_usd=0.0,
            max_symbols=1,
        ),
    )
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)

    with pytest.raises(IngestionError):
        DatabentoIngestionService.from_env()
