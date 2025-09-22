from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from ml.config.databento_policy import DatabentoSafetyConfig
from ml.config.databento_policy import SchemaSafetyConfig
from ml.data.ingest.api import IngestionJob
from ml.data.ingest.api import fetch_symbol_data
from ml.data.ingest.api import run_jobs
from ml.data.ingest.policy import DatabentoCoveragePolicy
from ml.data.ingest.service import CostViolationError
from ml.data.ingest.service import DatabentoIngestionService
from ml.data.ingest.service import IngestionChunk
from ml.data.ingest.service import IngestionRequest
from ml.data.ingest.service import build_historical_adapter
from ml.data.ingest.service import build_like_client


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


class _MetadataHistorical:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.metadata = _SchemaAwareMetadata(payload)
        self.timeseries = _FakeTimeseries()


@pytest.fixture()
def safety_config() -> DatabentoSafetyConfig:
    return DatabentoSafetyConfig(
        datasets=("EQUS.MINI",),
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
