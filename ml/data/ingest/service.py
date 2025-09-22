"""
Databento historical ingestion service.

This module provides a single, configurable entry point for all historical data
ingestion within the ML stack.  It centralises safety checks (dataset/schema
allowlists, lookback limits), performs Databento cost estimation, and offers a
streaming interface that callers can use to persist data however they choose
(e.g. Parquet catalog, tier1 fallback files, SQL stores).

The service is intentionally cold-path only.  It relies on ``structlog`` for
structured logging and ``ml.common.metrics_bootstrap`` for Prometheus metrics so
that every ingestion job can be monitored uniformly.

Example
-------
>>> from datetime import datetime, UTC
>>> from ml.data.ingest.service import DatabentoIngestionService, IngestionRequest
>>> service = DatabentoIngestionService.from_env()
>>> request = IngestionRequest(
...     dataset="EQUS.MINI",
...     schema="trades",
...     symbols=("SPY",),
...     start=datetime(2025, 8, 1, tzinfo=UTC),
...     end=datetime(2025, 8, 2, tzinfo=UTC),
... )
>>> results = service.ingest(request)
>>> len(results)
1

All ingestion-capable CLIs and orchestrators must route requests through this
service to guarantee consistent safeguards across the codebase.

"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
import structlog

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.config.databento_policy import DatabentoSafetyConfig
from ml.config.databento_policy import DatabentoSafetyConfigError
from ml.config.databento_policy import SchemaSafetyConfig
from ml.config.databento_policy import load_databento_safety_config
from ml.data.ingest.common import RateLimiter
from ml.data.ingest.policy import DatabentoCoveragePolicy
from ml.data.ingest.resume import DatabentoLikeClient


logger = structlog.get_logger(__name__)


class DatabentoTimeseriesClient(Protocol):
    """
    Protocol for ``Historical.timeseries`` used by the ingestion service.
    """

    def get_range(
        self,
        *,
        dataset: str,
        symbols: str | Sequence[str],
        schema: str,
        start: str | datetime,
        end: str | datetime,
        **kwargs: Any,
    ) -> Any: ...


class DatabentoMetadataClient(Protocol):
    """
    Protocol for ``Historical.metadata`` used by the ingestion service.
    """

    def get_cost(
        self,
        *,
        dataset: str,
        symbols: Sequence[str],
        schema: str,
        start: str | datetime,
        end: str | datetime,
    ) -> float: ...

    def get_dataset_range(self, dataset: str) -> dict[str, Any]: ...


class DatabentoHistoricalClient(Protocol):
    """
    Minimal surface of ``databento.Historical`` used for ingestion.
    """

    @property
    def timeseries(self) -> DatabentoTimeseriesClient: ...

    @property
    def metadata(self) -> DatabentoMetadataClient: ...


@dataclass(slots=True, frozen=True)
class IngestionWindow:
    """
    UTC time window to download (inclusive start, exclusive end).
    """

    start: datetime
    end: datetime


@dataclass(slots=True, frozen=True)
class IngestionRequest:
    """
    Ingestion parameters for a single dataset/schema window.
    """

    dataset: str
    schema: str
    symbols: tuple[str, ...]
    start: datetime
    end: datetime
    chunk_days: int | None = None
    rate_limit_per_min: int | None = None
    allow_cost: bool = False
    max_cost_usd: float | None = None
    cost_sample_size: int = 5
    reason: str | None = None


@dataclass(slots=True, frozen=True)
class IngestionChunk:
    """
    Single chunk of data retrieved for a symbol.
    """

    symbol: str
    window: IngestionWindow
    frame: pd.DataFrame


@dataclass(slots=True, frozen=True)
class SymbolIngestionSummary:
    """
    Summary statistics for an ingested symbol.
    """

    symbol: str
    requested_windows: tuple[IngestionWindow, ...]
    frames_returned: int
    rows_returned: int


class IngestionError(RuntimeError):
    """
    Base error for ingestion failures.
    """


class CostViolationError(IngestionError):
    """
    Raised when the estimated Databento cost exceeds the allowed budget.
    """

    def __init__(
        self,
        *,
        dataset: str,
        schema: str,
        symbols: Sequence[str],
        cost: float,
        limit: float,
    ) -> None:
        message = (
            "Databento cost estimate exceeds allowed budget: "
            f"dataset={dataset} schema={schema} symbols={list(symbols)[:5]} cost={cost:.2f} limit={limit:.2f}"
        )
        super().__init__(message)
        self.dataset = dataset
        self.schema = schema
        self.symbols = tuple(symbols)
        self.cost = float(cost)
        self.limit = float(limit)


class SafetyConfigError(IngestionError):
    """
    Raised when the Databento safety configuration cannot be loaded.
    """


_CHUNK_COUNTER = get_counter(
    "ml_ingestion_chunks_total",
    "Total ingestion chunks downloaded",
    labelnames=("dataset", "schema"),
)
_ROWS_COUNTER = get_counter(
    "ml_ingestion_rows_total",
    "Total rows downloaded via ingestion service",
    labelnames=("dataset", "schema"),
)
_CHUNK_LATENCY = get_histogram(
    "ml_ingestion_chunk_latency_seconds",
    "Latency per Databento chunk download",
    labelnames=("dataset", "schema"),
    buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32),
)


class DatabentoIngestionService:
    """
    Centralised Databento historical ingestion service.

    Parameters
    ----------
    client:
        Instance of ``databento.Historical`` (or protocol-compatible fake).
    safety_config:
        Repository safety configuration loaded from ``databento_safe_config.json``.
    policy:
        Environment-driven coverage policy providing per-run overrides.

    """

    def __init__(
        self,
        *,
        client: DatabentoHistoricalClient,
        safety_config: DatabentoSafetyConfig,
        policy: DatabentoCoveragePolicy | None = None,
    ) -> None:
        self._client = client
        self._safety = safety_config
        self._policy = policy or DatabentoCoveragePolicy.from_env()
        self._dataset_range_cache: dict[tuple[str, str | None], tuple[int | None, int | None]] = {}

    # ---------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def from_env(
        cls,
        *,
        safety_path: Path | None = None,
        policy: DatabentoCoveragePolicy | None = None,
    ) -> DatabentoIngestionService:
        """
        Instantiate the service using ``DATABENTO_API_KEY`` from the environment.
        """
        import os

        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            raise IngestionError("DATABENTO_API_KEY is required for Databento ingestion")
        try:
            import databento as db
        except Exception as exc:  # pragma: no cover - optional dependency
            raise IngestionError("databento package is required for ingestion") from exc

        client = cast(DatabentoHistoricalClient, db.Historical(api_key))
        try:
            safety_config = load_databento_safety_config(safety_path)
        except DatabentoSafetyConfigError as exc:
            raise SafetyConfigError(str(exc)) from exc
        return cls(client=client, safety_config=safety_config, policy=policy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest(
        self,
        request: IngestionRequest,
        *,
        on_chunk: Callable[[IngestionChunk], None] | None = None,
    ) -> list[SymbolIngestionSummary]:
        """
        Execute the requested ingestion job.

        Parameters
        ----------
        request:
            Dataclass describing dataset, schema, symbols, and time window.
        on_chunk:
            Optional callback invoked for every non-empty chunk fetched.  The
            callback receives an :class:`IngestionChunk` which includes the
            Pandas ``DataFrame`` for further processing.

        Returns
        -------
        list[SymbolIngestionSummary]
            Summary information per symbol.

        """
        if not request.symbols:
            return []
        dataset = request.dataset
        schema = request.schema
        symbol_list = tuple(sym.strip() for sym in request.symbols if sym.strip())
        if not symbol_list:
            return []

        self._policy.validate_dataset_schema(dataset=dataset, schema=schema)
        self._validate_dataset(dataset)
        window = self._sanitize_window(request.start, request.end)
        schema_policy = self._safety.schemas.get(schema, SchemaSafetyConfig())
        chunk_days = self._resolve_chunk_days(request.chunk_days, schema_policy)
        limiter = (
            RateLimiter(per_minute=request.rate_limit_per_min)
            if request.rate_limit_per_min
            else None
        )

        summaries: list[SymbolIngestionSummary] = []
        for symbol in symbol_list:
            allowed_symbols = self._policy.filter_symbols([symbol])
            if not allowed_symbols:
                logger.info(
                    "Symbol filtered by policy",
                    symbol=symbol,
                    dataset=dataset,
                    schema=schema,
                )
                continue
            symbol_effective = allowed_symbols[0]
            cost_limit = self._resolve_cost_limit(request.max_cost_usd, schema_policy)
            if request.allow_cost is False:
                self._enforce_cost_guard(
                    dataset=dataset,
                    schema=schema,
                    symbols=(symbol_effective,),
                    window=window,
                    limit=cost_limit,
                    sample_size=max(1, min(request.cost_sample_size, len(symbol_list))),
                )

            windows = self._split_windows(window, chunk_days)
            frames_returned = 0
            rows_returned = 0
            for chunk_window in windows:
                if limiter is not None:
                    limiter.wait()
                chunk_frame = self._fetch_chunk(
                    dataset=dataset,
                    schema=schema,
                    symbol=symbol_effective,
                    window=chunk_window,
                )
                if chunk_frame.empty:
                    continue
                chunk = IngestionChunk(
                    symbol=symbol_effective,
                    window=chunk_window,
                    frame=chunk_frame,
                )
                if on_chunk is not None:
                    on_chunk(chunk)
                frames_returned += 1
                rows_returned += len(chunk_frame.index)
                _CHUNK_COUNTER.labels(dataset=dataset, schema=schema).inc()
                _ROWS_COUNTER.labels(dataset=dataset, schema=schema).inc(len(chunk_frame.index))

            summaries.append(
                SymbolIngestionSummary(
                    symbol=symbol_effective,
                    requested_windows=tuple(windows),
                    frames_returned=frames_returned,
                    rows_returned=rows_returned,
                ),
            )
            logger.info(
                "ingestion.symbol.completed",
                dataset=dataset,
                schema=schema,
                symbol=symbol_effective,
                frames=frames_returned,
                rows=rows_returned,
                reason=request.reason,
            )
        return summaries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @property
    def metadata_client(self) -> DatabentoMetadataClient:
        """
        Expose the underlying metadata client for supplementary queries.
        """
        return self._client.metadata

    def get_available_range_ns(
        self,
        *,
        dataset: str,
        schema: str | None = None,
    ) -> tuple[int | None, int | None]:
        """
        Fetch and cache the provider's available range for a dataset/schema pair.

        Returns
        -------
        tuple[int | None, int | None]
            Start/end nanosecond timestamps in UTC. ``None`` indicates the provider did
            not supply a bound.
        """
        dataset_key = dataset.lower()
        schema_key = schema.lower() if schema is not None else None
        cache_key = (dataset_key, schema_key)
        cached = self._dataset_range_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            range_info = self.metadata_client.get_dataset_range(dataset)
        except Exception:
            result = (None, None)
            self._dataset_range_cache.setdefault(cache_key, result)
            return result

        parsed = self._parse_dataset_ranges(dataset_key, range_info)
        self._dataset_range_cache.update(parsed)
        return self._dataset_range_cache.get(cache_key, (None, None))

    def _validate_dataset(self, dataset: str) -> None:
        if self._safety.datasets and dataset not in self._safety.datasets:
            raise IngestionError(f"Dataset '{dataset}' is not permitted by safety configuration")

    @staticmethod
    def _sanitize_window(start: datetime, end: datetime) -> IngestionWindow:
        s = DatabentoIngestionService._ensure_utc(start)
        e = DatabentoIngestionService._ensure_utc(end)
        if e <= s:
            raise IngestionError("end must be after start for ingestion window")
        return IngestionWindow(start=s, end=e)

    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    @staticmethod
    def _resolve_chunk_days(chunk_days: int | None, schema_policy: SchemaSafetyConfig) -> int:
        if chunk_days is not None and chunk_days > 0:
            candidate = chunk_days
        elif schema_policy.max_days is not None and schema_policy.max_days > 0:
            candidate = schema_policy.max_days
        else:
            candidate = 365  # fallback to a reasonable default year window
        return max(1, candidate)

    def _resolve_cost_limit(
        self,
        override: float | None,
        schema_policy: SchemaSafetyConfig,
    ) -> float:
        if override is not None:
            return float(override)
        if schema_policy.max_cost_usd is not None:
            return float(schema_policy.max_cost_usd)
        if self._safety.max_cost_usd is not None:
            return float(self._safety.max_cost_usd)
        return 0.0

    def _enforce_cost_guard(
        self,
        *,
        dataset: str,
        schema: str,
        symbols: Sequence[str],
        window: IngestionWindow,
        limit: float,
        sample_size: int,
    ) -> None:
        sample = list(symbols[:sample_size])
        if not sample:
            return
        estimate = float(
            self._client.metadata.get_cost(
                dataset=dataset,
                symbols=sample,
                schema=schema,
                start=window.start.isoformat(),
                end=window.end.isoformat(),
            ),
        )
        if estimate > limit:
            logger.warning(
                "ingestion.cost_violation",
                dataset=dataset,
                schema=schema,
                estimate=estimate,
                limit=limit,
                symbols=sample,
            )
            raise CostViolationError(
                dataset=dataset,
                schema=schema,
                symbols=sample,
                cost=estimate,
                limit=limit,
            )

    @staticmethod
    def _split_windows(window: IngestionWindow, chunk_days: int) -> list[IngestionWindow]:
        windows: list[IngestionWindow] = []
        delta = timedelta(days=chunk_days)
        cursor = window.start
        while cursor < window.end:
            next_cursor = min(window.end, cursor + delta)
            windows.append(IngestionWindow(start=cursor, end=next_cursor))
            cursor = next_cursor
        return windows

    def _fetch_chunk(
        self,
        *,
        dataset: str,
        schema: str,
        symbol: str,
        window: IngestionWindow,
    ) -> pd.DataFrame:
        start_iso = window.start.isoformat()
        end_iso = window.end.isoformat()
        histogram = _CHUNK_LATENCY.labels(dataset=dataset, schema=schema)
        with histogram.time():
            result = self._client.timeseries.get_range(
                dataset=dataset,
                symbols=symbol,
                schema=schema,
                start=start_iso,
                end=end_iso,
            )
        if hasattr(result, "to_df"):
            df = cast(pd.DataFrame, result.to_df())
        else:
            df = pd.DataFrame(result)

        if isinstance(df.index, pd.DatetimeIndex) and df.index.name == "ts_event" and "ts_event" not in df.columns:
            df = df.reset_index()

        if "ts_event" not in df.columns and "ts" in df.columns:
            try:
                df["ts_event"] = pd.to_datetime(df["ts"], utc=True)
            except Exception:
                pass

        if "ts_event" in df.columns and pd.api.types.is_datetime64_any_dtype(df["ts_event"]):
            df["ts_event"] = df["ts_event"].astype("int64")

        if "ts_init" not in df.columns and "ts_event" in df.columns:
            df["ts_init"] = df["ts_event"]

        return df

    @staticmethod
    def _parse_dataset_ranges(
        dataset_key: str,
        range_info: Mapping[str, Any] | Any,
    ) -> dict[tuple[str, str | None], tuple[int | None, int | None]]:
        if not isinstance(range_info, Mapping):
            return {(dataset_key, None): (None, None)}

        dataset_start = DatabentoIngestionService._coerce_timestamp_ns(range_info.get("start"))
        dataset_end = DatabentoIngestionService._coerce_timestamp_ns(range_info.get("end"))
        parsed: dict[tuple[str, str | None], tuple[int | None, int | None]] = {
            (dataset_key, None): (dataset_start, dataset_end),
        }

        schema_ranges = DatabentoIngestionService._extract_schema_ranges(range_info)
        for schema_name, (schema_start, schema_end) in schema_ranges.items():
            start_ns = schema_start if schema_start is not None else dataset_start
            end_ns = schema_end if schema_end is not None else dataset_end
            parsed[(dataset_key, schema_name)] = (start_ns, end_ns)

        return parsed

    @staticmethod
    def _extract_schema_ranges(range_info: Mapping[str, Any]) -> dict[str, tuple[int | None, int | None]]:
        result: dict[str, tuple[int | None, int | None]] = {}

        raw_schema = range_info.get("schema")
        if isinstance(raw_schema, Mapping):
            for key, value in raw_schema.items():
                if isinstance(key, str) and isinstance(value, Mapping):
                    schema_key = key.lower()
                    result[schema_key] = (
                        DatabentoIngestionService._coerce_timestamp_ns(value.get("start")),
                        DatabentoIngestionService._coerce_timestamp_ns(value.get("end")),
                    )

        raw_schemas = range_info.get("schemas")
        if isinstance(raw_schemas, Sequence):
            for entry in raw_schemas:
                if not isinstance(entry, Mapping):
                    continue
                key_raw = entry.get("schema") or entry.get("name") or entry.get("id")
                if not isinstance(key_raw, str):
                    continue
                schema_key = key_raw.lower()
                result[schema_key] = (
                    DatabentoIngestionService._coerce_timestamp_ns(entry.get("start")),
                    DatabentoIngestionService._coerce_timestamp_ns(entry.get("end")),
                )

        return result

    @staticmethod
    def _coerce_timestamp_ns(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return int(value)
        if isinstance(value, datetime):
            dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
            return int(dt.astimezone(UTC).timestamp() * 1_000_000_000)
        try:
            ts = pd.to_datetime(value, utc=True)
        except Exception:
            return None
        if isinstance(ts, pd.Timestamp):
            return int(ts.value)
        if isinstance(ts, pd.Series):
            if ts.empty:
                return None
            return int(ts.iloc[0].value)
        if isinstance(ts, pd.DatetimeIndex):
            if ts.empty:
                return None
            return int(ts[0].value)
        return None


class _ServiceResult:
    """
    Adapter result providing a ``to_df`` method for legacy callsites.
    """

    def __init__(self, frames: list[pd.DataFrame]) -> None:
        self._frames = frames

    def to_df(self) -> pd.DataFrame:
        if not self._frames:
            return pd.DataFrame()
        if len(self._frames) == 1:
            return self._frames[0]
        return pd.concat(self._frames, ignore_index=True)


class _ServiceTimeseries:
    """
    Timeseries facade backed by :class:`DatabentoIngestionService`.
    """

    def __init__(self, service: DatabentoIngestionService) -> None:
        self._service = service

    def get_range(
        self,
        *,
        dataset: str,
        symbols: Sequence[str] | str,
        schema: str,
        start: str | datetime,
        end: str | datetime,
        **_: Any,
    ) -> _ServiceResult:
        symbol_tuple: tuple[str, ...]
        if isinstance(symbols, str):
            symbol_tuple = (symbols,)
        else:
            symbol_tuple = tuple(symbols)

        start_dt = _parse_datetime(start)
        end_dt = _parse_datetime(end)

        frames: list[pd.DataFrame] = []

        def _collect(chunk: IngestionChunk) -> None:
            if chunk.frame.empty:
                return
            frames.append(chunk.frame)

        request = IngestionRequest(
            dataset=dataset,
            schema=schema,
            symbols=symbol_tuple,
            start=start_dt,
            end=end_dt,
            reason="service_adapter",
        )
        self._service.ingest(request, on_chunk=_collect)
        return _ServiceResult(frames)


def _parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return pd.to_datetime(value, utc=True).to_pydatetime()


class ServiceHistoricalAdapter:
    """
    Adapter exposing ``metadata``/``timeseries`` like ``databento.Historical``.
    """

    def __init__(self, service: DatabentoIngestionService) -> None:
        self._service = service
        self.metadata = service.metadata_client
        self.timeseries = _ServiceTimeseries(service)


def build_historical_adapter(service: DatabentoIngestionService) -> DatabentoHistoricalClient:
    """
    Create a historical client adapter backed by :class:`DatabentoIngestionService`.
    """
    return ServiceHistoricalAdapter(service)


class ServiceDatabentoLikeClient(DatabentoLikeClient):
    """
    Adapter implementing :class:`DatabentoLikeClient` via the ingestion service.
    """

    def __init__(self, service: DatabentoIngestionService) -> None:
        self._service = service

    def get_data(
        self,
        dataset: str,
        symbols: list[str],
        schema: str,
        start: str | datetime,
        end: str | datetime,
        **_: Any,
    ) -> pd.DataFrame:
        symbol_tuple = tuple(symbols)
        frames: list[pd.DataFrame] = []

        def _collect(chunk: IngestionChunk) -> None:
            if chunk.frame.empty:
                return
            frames.append(chunk.frame)

        request = IngestionRequest(
            dataset=dataset,
            schema=schema,
            symbols=symbol_tuple,
            start=_parse_datetime(start),
            end=_parse_datetime(end),
            reason="service_like_client",
        )
        self._service.ingest(request, on_chunk=_collect)
        if not frames:
            return pd.DataFrame()
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True)


def build_like_client(service: DatabentoIngestionService) -> DatabentoLikeClient:
    """
    Return an object implementing :class:`DatabentoLikeClient` backed by the service.
    """
    return ServiceDatabentoLikeClient(service)


__all__ = [
    "CostViolationError",
    "DatabentoHistoricalClient",
    "DatabentoIngestionService",
    "DatabentoMetadataClient",
    "DatabentoTimeseriesClient",
    "IngestionChunk",
    "IngestionError",
    "IngestionRequest",
    "IngestionWindow",
    "SafetyConfigError",
    "ServiceDatabentoLikeClient",
    "ServiceHistoricalAdapter",
    "SymbolIngestionSummary",
    "build_historical_adapter",
    "build_like_client",
]
