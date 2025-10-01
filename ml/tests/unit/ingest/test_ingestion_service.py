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
    result_df = result.to_df()
    assert not result_df.empty


def test_build_like_client_returns_dataframe(safety_config: DatabentoSafetyConfig) -> None:
    service = DatabentoIngestionService(
        client=_FakeHistorical(cost=0.0),
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
    )
    like_client = build_like_client(service)
    like_df = like_client.get_data(
        dataset="EQUS.MINI",
        symbols=["SPY"],
        schema="trades",
        start=datetime(2025, 9, 1, tzinfo=UTC),
        end=datetime(2025, 9, 2, tzinfo=UTC),
    )
    assert not like_df.empty


def test_fetch_symbol_data_concatenates_chunks(safety_config: DatabentoSafetyConfig) -> None:
    service = DatabentoIngestionService(
        client=_FakeHistorical(cost=0.0),
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
    )
    start = datetime(2025, 9, 1, tzinfo=UTC)
    end = start + timedelta(days=2)
    fetched_df = fetch_symbol_data(
        service=service,
        dataset="EQUS.MINI",
        schema="trades",
        symbol="SPY",
        start=start,
        end=end,
        chunk_days=1,
    )
    assert len(fetched_df.index) == 4


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


def test_ingest_tags_source_dataset(safety_config: DatabentoSafetyConfig) -> None:
    service = DatabentoIngestionService(
        client=_FakeHistorical(cost=0.0),
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
    )
    start = datetime(2025, 9, 1, tzinfo=UTC)
    end = start + timedelta(days=1)
    frames: list[pd.DataFrame] = []

    service.ingest(_request(start, end), on_chunk=lambda chunk: frames.append(chunk.frame))

    assert frames, "expected at least one chunk"
    for frame in frames:
        assert "source_dataset" in frame.columns
        assert frame["source_dataset"].dropna().unique().tolist() == ["EQUS.MINI"]


def test_ingest_fallback_tags_source_dataset(safety_config: DatabentoSafetyConfig) -> None:
    class _FallbackTimeseriesSimple:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get_range(self, *, dataset: str, symbols: Sequence[str], schema: str, start: str | datetime, end: str | datetime, **_: Any) -> _FakeResult:
            del symbols, schema, end
            self.calls.append(dataset)
            if isinstance(start, datetime):
                start_dt = start
            else:
                start_dt = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
            ts_event = int(start_dt.replace(tzinfo=UTC).timestamp() * 1_000_000_000)
            if dataset == "EQUS.MINI":
                return _FakeResult([])
            return _FakeResult(
                [
                    {
                        "ts_event": ts_event,
                        "open": 1.0,
                        "high": 1.1,
                        "low": 0.9,
                        "close": 1.05,
                        "volume": 100,
                    },
                    {
                        "ts_event": ts_event + 60_000_000_000,
                        "open": 1.05,
                        "high": 1.2,
                        "low": 1.0,
                        "close": 1.1,
                        "volume": 120,
                    },
                ],
            )

    class _FallbackMetadataSimple:
        def get_cost(self, **_: Any) -> float:
            return 0.0

        def get_dataset_range(self, dataset: str) -> dict[str, object]:
            if dataset == "EQUS.MINI":
                start = datetime(2025, 1, 1, tzinfo=UTC)
                return {
                    "start": start.isoformat(),
                    "end": datetime(2025, 12, 31, tzinfo=UTC).isoformat(),
                    "schema": {
                        "ohlcv-1m": {
                            "start": start.isoformat(),
                            "end": datetime(2025, 12, 31, tzinfo=UTC).isoformat(),
                        },
                    },
                }
            return {
                "start": datetime(2018, 1, 1, tzinfo=UTC).isoformat(),
                "end": datetime(2025, 12, 31, tzinfo=UTC).isoformat(),
            }

    class _FallbackClient:
        def __init__(self) -> None:
            self.timeseries = _FallbackTimeseriesSimple()
            self.metadata = _FallbackMetadataSimple()
            self.symbology = _SimpleSymbology()

    service = DatabentoIngestionService(
        client=_FallbackClient(),
        safety_config=safety_config,
        policy=DatabentoCoveragePolicy(),
    )
    request = IngestionRequest(
        dataset="EQUS.MINI",
        schema="ohlcv-1m",
        symbols=("INTC",),
        start=datetime(2024, 12, 1, tzinfo=UTC),
        end=datetime(2024, 12, 2, tzinfo=UTC),
        reason="fallback-test",
    )
    frames: list[pd.DataFrame] = []

    service.ingest(request, on_chunk=lambda chunk: frames.append(chunk.frame))

    assert frames, "expected fallback chunk"
    for frame in frames:
        assert not frame.empty
        assert frame["source_dataset"].dropna().unique().tolist() == ["XNAS.ITCH"]
