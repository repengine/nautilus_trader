from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Callable

import pandas as pd
import pytest

from ml.common.databento_credentials import CredentialResolution
from ml.common.databento_credentials import CredentialSource
from ml.data.ingest.api import IngestionJob
from ml.data.ingest.api import ensure_service
from ml.data.ingest.api import fetch_symbol_data
from ml.data.ingest.api import run_jobs
from ml.data.ingest.service import IngestionChunk
from ml.data.ingest.service import IngestionRequest
from ml.data.ingest.service import IngestionWindow
from ml.data.ingest.service import SymbolIngestionSummary


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry", "mock_tracing_backend")


class _RecordingService:
    def __init__(self, chunks: list[IngestionChunk]) -> None:
        self._chunks = chunks
        self.requests: list[IngestionRequest] = []

    def ingest(
        self,
        request: IngestionRequest,
        on_chunk: Callable[[IngestionChunk], None] | None = None,
    ) -> list[SymbolIngestionSummary]:
        self.requests.append(request)
        if on_chunk is not None:
            for chunk in self._chunks:
                on_chunk(chunk)
        return []


class _SummaryService:
    def __init__(self) -> None:
        self.requests: list[IngestionRequest] = []
        self.on_chunk_calls = 0

    def ingest(
        self,
        request: IngestionRequest,
        on_chunk: Callable[[IngestionChunk], None] | None = None,
    ) -> list[SymbolIngestionSummary]:
        self.requests.append(request)
        if on_chunk is not None:
            window = IngestionWindow(start=request.start, end=request.end)
            on_chunk(
                IngestionChunk(
                    symbol=request.symbols[0],
                    window=window,
                    frame=pd.DataFrame(),
                ),
            )
            self.on_chunk_calls += 1
        return [
            SymbolIngestionSummary(
                symbol=request.symbols[0],
                requested_windows=(IngestionWindow(start=request.start, end=request.end),),
                frames_returned=0,
                rows_returned=0,
            )
        ]


def test_ensure_service_returns_existing_service(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()

    def _boom() -> CredentialResolution:
        raise AssertionError("resolve_databento_api_key should not be called")

    monkeypatch.setattr("ml.data.ingest.api.resolve_databento_api_key", _boom)
    assert ensure_service(sentinel) is sentinel


def test_ensure_service_constructs_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    resolution = CredentialResolution(
        value="test-token",
        source=CredentialSource.EXPLICIT,
        injected=False,
    )
    monkeypatch.setattr(
        "ml.data.ingest.api.resolve_databento_api_key",
        lambda: resolution,
    )

    created = object()

    def _from_env(*, api_key: str | None = None) -> object:
        assert api_key == "test-token"
        return created

    monkeypatch.setattr("ml.data.ingest.api.DatabentoIngestionService.from_env", _from_env)

    assert ensure_service() is created


def test_fetch_symbol_data_concatenates_non_empty_frames() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=1)
    window = IngestionWindow(start=start, end=end)
    chunks = [
        IngestionChunk(symbol="SPY", window=window, frame=pd.DataFrame()),
        IngestionChunk(
            symbol="SPY",
            window=window,
            frame=pd.DataFrame({"price": [1.0, 2.0]}),
        ),
        IngestionChunk(
            symbol="SPY",
            window=window,
            frame=pd.DataFrame({"price": [3.0]}),
        ),
    ]
    service = _RecordingService(chunks)

    result = fetch_symbol_data(
        service=service,
        dataset="XNAS.ITCH",
        schema="ohlcv-1m",
        symbol="SPY",
        start=start,
        end=end,
        chunk_days=1,
        allow_cost=True,
        rate_limit_per_min=2,
        reason="test",
    )

    assert result["price"].tolist() == [1.0, 2.0, 3.0]
    assert len(service.requests) == 1
    request = service.requests[0]
    assert request.dataset == "XNAS.ITCH"
    assert request.schema == "ohlcv-1m"
    assert request.symbols == ("SPY",)
    assert request.chunk_days == 1
    assert request.allow_cost is True
    assert request.rate_limit_per_min == 2
    assert request.reason == "test"


def test_fetch_symbol_data_returns_empty_when_no_frames() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=1)
    service = _RecordingService([])

    result = fetch_symbol_data(
        service=service,
        dataset="XNAS.ITCH",
        schema="ohlcv-1m",
        symbol="SPY",
        start=start,
        end=end,
    )

    assert result.empty


def test_run_jobs_executes_all_requests() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=1)
    jobs = [
        IngestionJob(
            dataset="XNAS.ITCH",
            schema="ohlcv-1m",
            symbols=("SPY",),
            start=start,
            end=end,
            chunk_days=1,
            allow_cost=True,
            rate_limit_per_min=5,
            reason="alpha",
        ),
        IngestionJob(
            dataset="XNAS.ITCH",
            schema="trades",
            symbols=("QQQ",),
            start=start,
            end=end,
        ),
    ]
    service = _SummaryService()
    chunk_calls: list[str] = []

    def _on_chunk(_: IngestionChunk) -> None:
        chunk_calls.append("called")

    summaries = run_jobs(jobs, service=service, on_chunk=_on_chunk)

    assert len(summaries) == 2
    assert service.on_chunk_calls == 2
    assert len(chunk_calls) == 2
    assert service.requests[0].schema == "ohlcv-1m"
    assert service.requests[1].schema == "trades"
