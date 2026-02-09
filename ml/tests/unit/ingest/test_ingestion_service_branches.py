from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any, Sequence, cast

import pandas as pd
import pytest

from ml.config.databento_policy import DatabentoSafetyConfig
from ml.config.databento_policy import SchemaSafetyConfig
from ml.data.ingest.discovery import DatasetDiscoveryError
from ml.data.ingest.service import CostViolationError
from ml.data.ingest.service import DatabentoIngestionService
from ml.data.ingest.service import IngestionChunk
from ml.data.ingest.service import IngestionError
from ml.data.ingest.service import IngestionRequest
from ml.data.ingest.service import IngestionWindow
from ml.data.ingest.service import ServiceDatabentoLikeClient
from ml.data.ingest.service import ServiceHistoricalAdapter
from ml.data.ingest.service import SymbolDatasetDiscovery
from ml.data.ingest.symbology import SymbolResolution


@pytest.fixture()
def safety_config() -> DatabentoSafetyConfig:
    return DatabentoSafetyConfig(
        datasets=("EQUS.MINI", "XNAS.ITCH"),
        schemas={"trades": SchemaSafetyConfig(max_days=2, max_cost_usd=1.0)},
        max_cost_usd=2.0,
        max_symbols=100,
    )


@dataclass
class _Metadata:
    cost: float = 0.0
    range_payload: dict[str, Any] | None = None
    fail_range: bool = False
    fail_symbols: set[str] | None = None

    def get_cost(
        self,
        *,
        dataset: str,
        symbols: Sequence[str],
        schema: str,
        start: str | datetime,
        end: str | datetime,
    ) -> float:
        del dataset, schema, start, end
        symbol = symbols[0]
        if self.fail_symbols and symbol in self.fail_symbols:
            raise RuntimeError("cost lookup failed")
        return float(self.cost)

    def get_dataset_range(self, dataset: str) -> dict[str, Any]:
        del dataset
        if self.fail_range:
            raise RuntimeError("range lookup failed")
        if self.range_payload is not None:
            return self.range_payload
        return {
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-12-31T00:00:00Z",
            "schema": {},
        }

    def list_datasets(self) -> tuple[str, ...]:
        return ("XNAS.ITCH",)

    def list_schemas(self, *, dataset: str) -> tuple[str, ...]:
        del dataset
        return ("trades",)


@dataclass
class _Timeseries:
    result: Any
    calls: list[dict[str, Any]]

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls = []

    def get_range(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if callable(self.result):
            return self.result(**kwargs)
        return self.result


@dataclass
class _Client:
    metadata: _Metadata
    timeseries: _Timeseries
    symbology: Any | None = None


@dataclass
class _Resolver:
    resolution: SymbolResolution
    raise_ingestion_error: bool = False

    def resolve(
        self,
        *,
        dataset: str,
        symbol: str,
        schema: str,
        start: datetime | None,
        end: datetime | None,
    ) -> SymbolResolution:
        del dataset, symbol, schema, start, end
        if self.raise_ingestion_error:
            raise IngestionError("resolver failed")
        return self.resolution


@dataclass
class _PolicyAllowAll:
    allowed_datasets: set[str] | None = None

    def validate_dataset_schema(self, *, dataset: str, schema: str) -> None:
        del dataset, schema

    def filter_symbols(self, symbols: Sequence[str]) -> list[str]:
        return list(symbols)


@dataclass
class _PolicyFilterAll:
    allowed_datasets: set[str] | None = None

    def validate_dataset_schema(self, *, dataset: str, schema: str) -> None:
        del dataset, schema

    def filter_symbols(self, symbols: Sequence[str]) -> list[str]:
        del symbols
        return []


def _service(
    safety_config: DatabentoSafetyConfig,
    *,
    metadata: _Metadata | None = None,
    timeseries_result: Any = None,
) -> DatabentoIngestionService:
    md = metadata or _Metadata()
    ts = _Timeseries(timeseries_result if timeseries_result is not None else [{"ts_event": 1, "price": 1.0}])
    return DatabentoIngestionService(
        client=cast(Any, _Client(metadata=md, timeseries=ts, symbology=None)),
        safety_config=safety_config,
    )


def _resolution(symbol: str = "AAPL") -> SymbolResolution:
    return SymbolResolution(
        original=symbol,
        dataset="EQUS.MINI",
        schema="trades",
        candidates=(symbol,),
        preferred=symbol,
        instrument_id="42",
    )


def test_ingest_returns_empty_for_missing_or_blank_symbols(
    safety_config: DatabentoSafetyConfig,
) -> None:
    service = _service(safety_config)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)

    assert service.ingest(
        IngestionRequest(
            dataset="EQUS.MINI",
            schema="trades",
            symbols=(),
            start=start,
            end=end,
        ),
    ) == []
    assert service.ingest(
        IngestionRequest(
            dataset="EQUS.MINI",
            schema="trades",
            symbols=(" ",),
            start=start,
            end=end,
        ),
    ) == []


def test_ingest_uses_rate_limiter_and_skips_filtered_symbols(
    monkeypatch: pytest.MonkeyPatch,
    safety_config: DatabentoSafetyConfig,
) -> None:
    service = _service(safety_config)
    service._policy = cast(Any, _PolicyAllowAll())
    service._symbology = cast(Any, _Resolver(_resolution("AAPL")))
    waited: list[bool] = []

    def _wait(_self: Any) -> None:
        waited.append(True)

    monkeypatch.setattr(
        "ml.data.ingest.service.RateLimiter.wait",
        _wait,
    )

    request = IngestionRequest(
        dataset="EQUS.MINI",
        schema="trades",
        symbols=("AAPL",),
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 2, tzinfo=UTC),
        chunk_days=1,
        rate_limit_per_min=1,
    )
    summaries = service.ingest(request)
    assert summaries and waited

    filtered_service = _service(safety_config, timeseries_result=[])
    filtered_service._policy = cast(Any, _PolicyFilterAll())
    filtered_service._symbology = cast(Any, _Resolver(_resolution("AAPL")))
    assert filtered_service.ingest(request) == []


def test_discover_symbol_dataset_handles_guard_and_error_paths(
    monkeypatch: pytest.MonkeyPatch,
    safety_config: DatabentoSafetyConfig,
) -> None:
    service = _service(safety_config)
    start_ns = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
    end_ns = start_ns + 1_000
    assert service.discover_symbol_dataset(symbol="AAPL", schema="trades", start_ns=end_ns, end_ns=start_ns) is None

    monkeypatch.setattr(service, "_get_discovery_service", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert service.discover_symbol_dataset(symbol="AAPL", schema="trades", start_ns=start_ns, end_ns=end_ns) is None

    class _DiscoveryErrorService:
        def discover_one(self, *, request: Any) -> Any:
            del request
            raise DatasetDiscoveryError("no dataset")

    monkeypatch.setattr(service, "_get_discovery_service", lambda: _DiscoveryErrorService())
    assert service.discover_symbol_dataset(symbol="AAPL", schema="trades", start_ns=start_ns, end_ns=end_ns) is None

    class _DiscoveryCrashService:
        def discover_one(self, *, request: Any) -> Any:
            del request
            raise RuntimeError("fail")

    monkeypatch.setattr(service, "_get_discovery_service", lambda: _DiscoveryCrashService())
    assert service.discover_symbol_dataset(symbol="AAPL", schema="trades", start_ns=start_ns, end_ns=end_ns) is None


def test_estimate_cost_and_dataset_validation_branches(
    safety_config: DatabentoSafetyConfig,
) -> None:
    service = _service(safety_config)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)
    assert service.estimate_cost_usd(
        dataset="EQUS.MINI",
        schema="trades",
        symbols=(),
        start=start,
        end=end,
    ) == 0.0
    assert service.estimate_cost_usd(
        dataset="EQUS.MINI",
        schema="trades",
        symbols=(" ",),
        start=start,
        end=end,
    ) == 0.0

    service._policy = cast(Any, _PolicyAllowAll(allowed_datasets=None))
    service._validate_dataset("UNKNOWN.DATASET")

    service._policy = cast(Any, _PolicyAllowAll(allowed_datasets={"BLOCKED"}))
    service._safety = DatabentoSafetyConfig(
        datasets=("EQUS.MINI",),
        schemas={},
        max_cost_usd=1.0,
        max_symbols=1,
    )
    service._validate_dataset("EQUS.MINI")

    with pytest.raises(IngestionError):
        service._validate_dataset("NOT_ALLOWED")


def test_window_limit_cost_and_range_helpers_cover_edge_paths(
    safety_config: DatabentoSafetyConfig,
) -> None:
    service = _service(safety_config, metadata=_Metadata(fail_range=True))
    assert service.get_available_range_ns(dataset="EQUS.MINI", schema="trades") == (None, None)
    assert service.get_available_range_ns(dataset="EQUS.MINI", schema="trades") == (None, None)

    start = datetime(2025, 1, 1, tzinfo=UTC)
    with pytest.raises(IngestionError):
        service._sanitize_window(start, start)
    assert service._ensure_utc(datetime(2025, 1, 1)).tzinfo is not None

    assert DatabentoIngestionService._resolve_chunk_days(3, SchemaSafetyConfig(max_days=2)) == 3
    assert DatabentoIngestionService._resolve_chunk_days(None, SchemaSafetyConfig(max_days=2)) == 2
    assert DatabentoIngestionService._resolve_chunk_days(None, SchemaSafetyConfig()) == 365

    assert service._resolve_cost_limit(4.0, SchemaSafetyConfig()) == 4.0
    assert service._resolve_cost_limit(None, SchemaSafetyConfig(max_cost_usd=3.0)) == 3.0
    assert service._resolve_cost_limit(None, SchemaSafetyConfig()) == 2.0
    service._safety = DatabentoSafetyConfig(datasets=(), schemas={}, max_cost_usd=None, max_symbols=1)
    assert service._resolve_cost_limit(None, SchemaSafetyConfig()) == 0.0


def test_enforce_cost_guard_and_fetch_chunk_and_canonicalize(
    safety_config: DatabentoSafetyConfig,
) -> None:
    service = _service(
        safety_config,
        metadata=_Metadata(cost=0.5, fail_symbols={"FAIL"}),
        timeseries_result=pd.DataFrame({"ts": ["2025-01-01T00:00:00Z"]}),
    )
    window = IngestionWindow(
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 2, tzinfo=UTC),
    )

    no_candidates = SymbolResolution(
        original="AAPL",
        dataset="EQUS.MINI",
        schema="trades",
        candidates=(),
        preferred="AAPL",
        instrument_id=None,
    )
    assert service._enforce_cost_guard(
        dataset="EQUS.MINI",
        schema="trades",
        resolution=no_candidates,
        window=window,
        limit=1.0,
    ) == (None, None)

    failed_candidates = SymbolResolution(
        original="AAPL",
        dataset="EQUS.MINI",
        schema="trades",
        candidates=("FAIL",),
        preferred="FAIL",
        instrument_id=None,
    )
    assert service._enforce_cost_guard(
        dataset="EQUS.MINI",
        schema="trades",
        resolution=failed_candidates,
        window=window,
        limit=1.0,
    ) == (None, None)

    violating = SymbolResolution(
        original="AAPL",
        dataset="EQUS.MINI",
        schema="trades",
        candidates=("AAPL",),
        preferred="AAPL",
        instrument_id=None,
    )
    with pytest.raises(CostViolationError):
        service._enforce_cost_guard(
            dataset="EQUS.MINI",
            schema="trades",
            resolution=violating,
            window=window,
            limit=0.1,
        )

    frame = service._fetch_chunk(dataset="EQUS.MINI", schema="trades", symbol="AAPL", window=window)
    assert "ts_event" in frame.columns
    assert "ts_init" in frame.columns
    empty = service._canonicalize_chunk(
        dataset="EQUS.MINI",
        schema="trades",
        frame=pd.DataFrame(),
        symbol="AAPL",
        instrument_id=None,
        source_dataset=None,
    )
    assert empty.empty


def test_fallback_helpers_and_parse_extract_coerce_paths(
    safety_config: DatabentoSafetyConfig,
) -> None:
    service = _service(
        safety_config,
        metadata=_Metadata(
            range_payload={
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-12-31T00:00:00Z",
                "schema": {"ohlcv-1m": {"start": "2025-01-01T00:00:00Z", "end": "2025-12-31T00:00:00Z"}},
            },
        ),
        timeseries_result=[],
    )
    window = IngestionWindow(
        start=datetime(2025, 1, 2, tzinfo=UTC),
        end=datetime(2025, 1, 2, 1, tzinfo=UTC),
    )
    assert not service._should_use_itch_fallback(dataset="OTHER", schema="ohlcv-1m", window=window)
    assert service._attempt_fallback_to_itch(
        schema="ohlcv-1m",
        symbol="AAPL",
        download_symbol="AAPL",
        window=window,
        instrument_id="42",
    ) is None

    parsed_default = DatabentoIngestionService._parse_dataset_ranges("xnas.itch", object())
    assert parsed_default == {("xnas.itch", None): (None, None)}

    parsed = DatabentoIngestionService._parse_dataset_ranges(
        "xnas.itch",
        {
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-12-31T00:00:00Z",
            "schema": {"trades": {"start": "2025-01-05T00:00:00Z", "end": "2025-01-06T00:00:00Z"}},
            "schemas": [{"schema": "tbbo", "start": "2025-01-07T00:00:00Z", "end": "2025-01-08T00:00:00Z"}],
        },
    )
    assert ("xnas.itch", "trades") in parsed
    assert ("xnas.itch", "tbbo") in parsed

    assert DatabentoIngestionService._coerce_timestamp_ns(None) is None
    assert DatabentoIngestionService._coerce_timestamp_ns(5) == 5
    assert DatabentoIngestionService._coerce_timestamp_ns(5.2) == 5
    assert isinstance(DatabentoIngestionService._coerce_timestamp_ns(datetime(2025, 1, 1)), int)
    assert DatabentoIngestionService._coerce_timestamp_ns("bad") is None
    assert DatabentoIngestionService._coerce_timestamp_ns(pd.Series([], dtype="datetime64[ns]")) is None
    assert DatabentoIngestionService._coerce_timestamp_ns(pd.DatetimeIndex([])) is None


def test_service_adapters_cover_empty_single_and_multi_frame_paths(
    safety_config: DatabentoSafetyConfig,
) -> None:
    service = _service(safety_config)

    def _ingest_no_frames(
        request: IngestionRequest,
        *,
        on_chunk: Any | None = None,
    ) -> list[Any]:
        del request, on_chunk
        return []

    service.ingest = _ingest_no_frames  # type: ignore[assignment]
    like_client = ServiceDatabentoLikeClient(service)
    empty = like_client.get_data(
        dataset="EQUS.MINI",
        symbols=["AAPL"],
        schema="trades",
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 1, 1, tzinfo=UTC),
    )
    assert empty.empty

    frames = [
        pd.DataFrame(),
        pd.DataFrame({"ts_event": [1], "price": [1.0]}),
        pd.DataFrame({"ts_event": [2], "price": [2.0]}),
    ]

    def _ingest_with_frames(
        request: IngestionRequest,
        *,
        on_chunk: Any | None = None,
    ) -> list[Any]:
        del request
        if on_chunk is not None:
            for frame in frames:
                on_chunk(
                    IngestionChunk(
                        symbol="AAPL",
                        window=IngestionWindow(
                            start=datetime(2025, 1, 1, tzinfo=UTC),
                            end=datetime(2025, 1, 1, 1, tzinfo=UTC),
                        ),
                        frame=frame,
                    ),
                )
        return []

    service.ingest = _ingest_with_frames  # type: ignore[assignment]
    concatenated = like_client.get_data(
        dataset="EQUS.MINI",
        symbols=["AAPL"],
        schema="trades",
        start=datetime(2025, 1, 1),
        end="2025-01-01T01:00:00Z",
    )
    assert len(concatenated.index) == 2

    adapter = ServiceHistoricalAdapter(service)
    result = adapter.timeseries.get_range(
        dataset="EQUS.MINI",
        symbols=("AAPL",),
        schema="trades",
        start=datetime(2025, 1, 1),
        end=datetime(2025, 1, 1, 1, tzinfo=UTC),
    )
    df = result.to_df()
    assert len(df.index) == 2
    assert adapter.metadata is service.metadata_client


def test_discover_symbol_dataset_maps_discovery_response(
    monkeypatch: pytest.MonkeyPatch,
    safety_config: DatabentoSafetyConfig,
) -> None:
    service = _service(safety_config)
    start_ns = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
    end_ns = start_ns + 3_600_000_000_000

    class _DiscoveryService:
        def discover_one(self, *, request: Any) -> Any:
            del request
            return type(
                "Discovered",
                (),
                {
                    "dataset_id": "XNAS.ITCH",
                    "schema": "trades",
                    "storage_kind": None,
                    "symbol": "",
                    "requested_symbol": "AAPL",
                    "available_start_ns": start_ns,
                    "available_end_ns": end_ns,
                    "cost_usd": 0.0,
                    "instrument_id": "1234",
                },
            )()

    monkeypatch.setattr(service, "_get_discovery_service", lambda: _DiscoveryService())
    discovered = service.discover_symbol_dataset(
        symbol="AAPL",
        schema="trades",
        start_ns=start_ns,
        end_ns=end_ns,
    )
    assert isinstance(discovered, SymbolDatasetDiscovery)
    assert discovered is not None
    assert discovered.symbol == "AAPL"
