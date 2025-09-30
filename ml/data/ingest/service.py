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

import os
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, Final, Protocol, cast

import numpy as np
import pandas as pd
import structlog

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.config.databento_policy import DatabentoSafetyConfig
from ml.config.databento_policy import DatabentoSafetyConfigError
from ml.config.databento_policy import SchemaSafetyConfig
from ml.config.databento_policy import load_databento_safety_config
from ml.data.ingest.calibration import CalibrationBundle
from ml.data.ingest.calibration import SymbolCalibration
from ml.data.ingest.calibration import load_calibration_bundle
from ml.data.ingest.canonicalization import CanonicalizationResult
from ml.data.ingest.canonicalization import CanonicalizationStats
from ml.data.ingest.canonicalization import canonicalize_equities_minute_bars
from ml.data.ingest.common import RateLimiter
from ml.data.ingest.discovery import DatasetDiscoveryError
from ml.data.ingest.discovery import DatasetDiscoveryService
from ml.data.ingest.discovery import DiscoveredInput
from ml.data.ingest.discovery import DiscoveryPolicy
from ml.data.ingest.discovery import DiscoveryRequest
from ml.data.ingest.policy import DatabentoCoveragePolicy
from ml.data.ingest.resume import DatabentoLikeClient
from ml.data.ingest.symbology import DatabentoSymbologyClient
from ml.data.ingest.symbology import DatabentoSymbologyResolver
from ml.data.ingest.symbology import SymbolResolution
from ml.registry.dataclasses import StorageKind


logger = structlog.get_logger(__name__)


DAY_NS: Final[int] = 86_400_000_000_000


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

    def list_datasets(self) -> Sequence[str] | str: ...

    def list_schemas(self, *, dataset: str) -> Sequence[str] | str: ...


class DatabentoHistoricalClient(Protocol):
    """
    Minimal surface of ``databento.Historical`` used for ingestion.
    """

    @property
    def timeseries(self) -> DatabentoTimeseriesClient: ...

    @property
    def metadata(self) -> DatabentoMetadataClient: ...

    @property
    def symbology(self) -> DatabentoSymbologyClient | None: ...


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


@dataclass(slots=True, frozen=True)
class SymbolDatasetDiscovery:
    """Result of dataset discovery for a specific symbol window."""

    dataset_id: str
    schema: str
    storage_kind: StorageKind
    symbol: str
    requested_symbol: str
    available_start_ns: int | None
    available_end_ns: int | None
    cost_usd: float | None
    instrument_id: str | None


@dataclass(slots=True, frozen=True)
class VolumeScaleEstimate:
    """Volume scaling factor derived from overlapping EQUS/ITCH periods."""

    symbol: str
    factor: float
    sample_minutes: int
    reference_start_ns: int
    reference_end_ns: int
    residual_abs: float | None
    residual_rel: float | None


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
_CANONICALIZED_ROWS_COUNTER = get_counter(
    "ml_ingestion_canonicalized_rows_total",
    "Rows canonicalized by the ingestion service",
    labelnames=("dataset", "source_dataset"),
)
_VOLUME_RESIDUAL = get_histogram(
    "ml_canonicalization_volume_residual",
    "Canonicalization volume residual (absolute/relative)",
    labelnames=("mode", "residual_type", "dataset"),
    buckets=(
        0.01,
        0.05,
        0.1,
        0.5,
        1,
        5,
        10,
        50,
        100,
        500,
        1_000,
        5_000,
        10_000,
        50_000,
        100_000,
        500_000,
        1_000_000,
    ),
)
_FALLBACK_COUNTER = get_counter(
    "ml_fallback_activations_total",
    "Fallback activations",
    labelnames=("component", "level"),
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
        resolver: DatabentoSymbologyResolver | None = None,
        calibration: CalibrationBundle | None = None,
    ) -> None:
        self._client = client
        self._safety = safety_config
        self._policy = policy or DatabentoCoveragePolicy.from_env()
        self._dataset_range_cache: dict[tuple[str, str | None], tuple[int | None, int | None]] = {}
        self._discovery_service: DatasetDiscoveryService | None = None
        sym_client = None
        try:
            sym_client = getattr(client, "symbology", None)
        except Exception:  # pragma: no cover - defensive guard
            sym_client = None
        self._symbology = resolver or DatabentoSymbologyResolver(
            client=cast(DatabentoSymbologyClient | None, sym_client),
        )
        fallback_dataset_env = os.getenv("ML_EQUS_FALLBACK_DATASET", "XNAS.ITCH").strip() or "XNAS.ITCH"
        self._fallback_dataset: Final[str] = fallback_dataset_env
        self._calibration = calibration
        self._calibration_version = (
            calibration.generated_at.astimezone(UTC).isoformat()
            if calibration is not None
            else None
        )
        self._enable_trade_reaggregation = os.getenv("ML_EQUS_ENABLE_TRADE_REAGG", "1") != "0"
        self._enable_volume_scaling = os.getenv("ML_EQUS_ENABLE_VOLUME_SCALING", "1") != "0"
        try:
            reference_days = int(os.getenv("ML_EQUS_SCALING_REFERENCE_DAYS", "5"))
        except ValueError:
            reference_days = 5
        self._scaling_reference_days = max(1, reference_days)
        try:
            min_ratio = float(os.getenv("ML_EQUS_SCALING_MIN_RATIO", "0.1"))
        except ValueError:
            min_ratio = 0.1
        try:
            max_ratio = float(os.getenv("ML_EQUS_SCALING_MAX_RATIO", "10.0"))
        except ValueError:
            max_ratio = 10.0
        self._scaling_min_ratio = max(0.0, min_ratio)
        self._scaling_max_ratio = max(self._scaling_min_ratio if self._scaling_min_ratio > 0 else 0.1, max_ratio)
        self._volume_scale_cache: dict[str, VolumeScaleEstimate] = {}

    # ---------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def from_env(
        cls,
        *,
        safety_path: Path | None = None,
        policy: DatabentoCoveragePolicy | None = None,
        api_key: str | None = None,
    ) -> DatabentoIngestionService:
        """
        Instantiate the service with a Databento API key.

        When ``api_key`` is provided it is normalised and injected into
        ``os.environ['DATABENTO_API_KEY']`` so that downstream helpers relying on the
        environment continue to operate.
        """
        import os

        resolved_key = (api_key or os.getenv("DATABENTO_API_KEY") or "").strip()
        if not resolved_key:
            raise IngestionError("DATABENTO_API_KEY is required for Databento ingestion")
        if os.getenv("DATABENTO_API_KEY") != resolved_key:
            os.environ["DATABENTO_API_KEY"] = resolved_key
        try:
            import databento as db
        except Exception as exc:  # pragma: no cover - optional dependency
            raise IngestionError("databento package is required for ingestion") from exc

        client = cast(DatabentoHistoricalClient, db.Historical(resolved_key))
        try:
            safety_config = load_databento_safety_config(safety_path)
        except DatabentoSafetyConfigError as exc:
            raise SafetyConfigError(str(exc)) from exc

        calibration_path = os.getenv("ML_EQUS_CALIBRATION_PATH")
        calibration: CalibrationBundle | None = None
        if calibration_path:
            try:
                calibration = load_calibration_bundle(Path(calibration_path))
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Failed to load EQUS calibration bundle", path=calibration_path, error=str(exc))
        return cls(client=client, safety_config=safety_config, policy=policy, calibration=calibration)

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
            resolution = self._resolve_symbol(
                dataset=dataset,
                schema=schema,
                symbol=symbol_effective,
                window=window,
            )
            cost_limit = self._resolve_cost_limit(request.max_cost_usd, schema_policy)
            download_symbol = resolution.preferred
            cost_symbol: str | None = None
            cost_estimate: float | None = None
            if request.allow_cost is False:
                cost_symbol, cost_estimate = self._enforce_cost_guard(
                    dataset=dataset,
                    schema=schema,
                    resolution=resolution,
                    window=window,
                    limit=cost_limit,
                )
                if cost_symbol is not None:
                    download_symbol = cost_symbol

            windows = self._split_windows(window, chunk_days)
            frames_returned = 0
            rows_returned = 0
            instrument_id = resolution.instrument_id
            for chunk_window in windows:
                if limiter is not None:
                    limiter.wait()
                chunk_frame = self._fetch_chunk(
                    dataset=dataset,
                    schema=schema,
                    symbol=download_symbol,
                    window=chunk_window,
                )
                canonical_stats: CanonicalizationStats | None = None
                if chunk_frame.empty and self._should_use_itch_fallback(
                    dataset=dataset,
                    schema=schema,
                    window=chunk_window,
                ):
                    fallback = self._attempt_fallback_to_itch(
                        schema=schema,
                        symbol=symbol_effective,
                        download_symbol=download_symbol,
                        window=chunk_window,
                        instrument_id=instrument_id,
                    )
                    if fallback is not None:
                        chunk_frame, canonical_stats = fallback
                        level = f"itch_{(canonical_stats.aggregation_mode or 'fallback')}"
                        _FALLBACK_COUNTER.labels(
                            component="databento_ingestion_service",
                            level=level,
                        ).inc()
                if canonical_stats is None:
                    chunk_frame, canonical_stats = self._canonicalize_chunk(
                        dataset=dataset,
                        schema=schema,
                        frame=chunk_frame,
                        symbol=download_symbol,
                        instrument_id=instrument_id,
                        source_dataset=None,
                        aggregation_mode="native",
                    )
                if chunk_frame.empty:
                    continue
                chunk = IngestionChunk(
                    symbol=download_symbol,
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
                    symbol=download_symbol,
                    requested_windows=tuple(windows),
                    frames_returned=frames_returned,
                    rows_returned=rows_returned,
                ),
            )
            logger.info(
                "ingestion.symbol.completed",
                dataset=dataset,
                schema=schema,
                symbol=download_symbol,
                input_symbol=symbol_effective,
                instrument_id=resolution.instrument_id,
                cost_estimate=cost_estimate,
                frames=frames_returned,
                rows=rows_returned,
                reason=request.reason,
            )
        return summaries

    def discover_symbol_dataset(
        self,
        *,
        symbol: str,
        schema: str,
        start_ns: int,
        end_ns: int,
    ) -> SymbolDatasetDiscovery | None:
        if start_ns >= end_ns:
            return None
        try:
            discovery_service = self._get_discovery_service()
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Discovery service unavailable", exc_info=True)
            return None

        start_dt = datetime.fromtimestamp(start_ns / 1_000_000_000, tz=UTC)
        end_dt = datetime.fromtimestamp(end_ns / 1_000_000_000, tz=UTC)
        request = DiscoveryRequest(
            symbol=symbol,
            schema=schema,
            start=start_dt,
            end=end_dt,
        )
        try:
            discovered: DiscoveredInput = discovery_service.discover_one(request=request)
        except DatasetDiscoveryError as exc:
            logger.debug(
                "Symbol discovery rejected",
                symbol=symbol,
                schema=schema,
                start_ns=start_ns,
                end_ns=end_ns,
                reason=str(exc),
            )
            return None
        except Exception:  # pragma: no cover - defensive guard
            logger.debug(
                "Symbol discovery failed",
                exc_info=True,
                symbol=symbol,
                schema=schema,
                start_ns=start_ns,
                end_ns=end_ns,
            )
            return None

        storage_kind = discovered.storage_kind or StorageKind.POSTGRES
        resolved_symbol = discovered.symbol or symbol
        return SymbolDatasetDiscovery(
            dataset_id=discovered.dataset_id,
            schema=discovered.schema,
            storage_kind=storage_kind,
            symbol=resolved_symbol,
            requested_symbol=discovered.requested_symbol,
            available_start_ns=discovered.available_start_ns,
            available_end_ns=discovered.available_end_ns,
            cost_usd=discovered.cost_usd,
            instrument_id=discovered.instrument_id,
        )

    def estimate_cost_usd(
        self,
        *,
        dataset: str,
        schema: str,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> float:
        """
        Estimate the Databento cost for a dataset/schema window.
        """
        if not symbols:
            return 0.0
        self._validate_dataset(dataset)
        window = self._sanitize_window(start, end)
        sample = tuple(symbol.strip() for symbol in symbols if symbol.strip())
        if not sample:
            return 0.0
        estimate = self.metadata_client.get_cost(
            dataset=dataset,
            symbols=sample,
            schema=schema,
            start=window.start.isoformat(),
            end=window.end.isoformat(),
        )
        return float(estimate)

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
        allowed_by_policy = (
            self._policy.allowed_datasets is None or dataset in self._policy.allowed_datasets
        )
        if allowed_by_policy:
            if self._safety.datasets and dataset not in self._safety.datasets:
                logger.debug(
                    "Dataset permitted by policy but absent from safety config",
                    dataset=dataset,
                )
            return
        if self._safety.datasets and dataset in self._safety.datasets:
            return
        raise IngestionError(f"Dataset '{dataset}' is not permitted by policy or safety configuration")

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
        resolution: SymbolResolution,
        window: IngestionWindow,
        limit: float,
    ) -> tuple[str | None, float | None]:
        candidates = resolution.candidates
        if not candidates:
            return None, None
        estimate: float | None = None
        chosen_symbol: str | None = None
        for candidate in candidates:
            try:
                estimate_candidate = float(
                    self._client.metadata.get_cost(
                        dataset=dataset,
                        symbols=[candidate],
                        schema=schema,
                        start=window.start.isoformat(),
                        end=window.end.isoformat(),
                    ),
                )
            except Exception as exc:  # pragma: no cover - provider/network errors
                logger.debug(
                    "Cost estimate variant failed",
                    dataset=dataset,
                    schema=schema,
                    symbol=candidate,
                    error=str(exc),
                )
                continue
            chosen_symbol = candidate
            estimate = estimate_candidate
            break
        if estimate is None:
            logger.debug(
                "Cost estimate unavailable",
                dataset=dataset,
                schema=schema,
                symbols=list(candidates),
            )
            return None, None
        if estimate > limit:
            context_symbol = [chosen_symbol] if chosen_symbol else list(candidates)
            logger.warning(
                "ingestion.cost_violation",
                dataset=dataset,
                schema=schema,
                estimate=estimate,
                limit=limit,
                symbols=context_symbol,
            )
            raise CostViolationError(
                dataset=dataset,
                schema=schema,
                symbols=context_symbol,
                cost=estimate,
                limit=limit,
            )
        logger.debug(
            "ingestion.cost_estimate",
            dataset=dataset,
            schema=schema,
            symbol=chosen_symbol,
            cost=estimate,
            limit=limit,
        )
        return chosen_symbol, estimate

    def _resolve_symbol(
        self,
        *,
        dataset: str,
        schema: str,
        symbol: str,
        window: IngestionWindow | None,
    ) -> SymbolResolution:
        start_dt = window.start if window is not None else None
        end_dt = window.end if window is not None else None
        return self._symbology.resolve(
            dataset=dataset,
            symbol=symbol,
            schema=schema,
            start=start_dt,
            end=end_dt,
        )

    def _get_discovery_service(self) -> DatasetDiscoveryService:
        if self._discovery_service is None:
            policy = self._build_discovery_policy()
            self._discovery_service = DatasetDiscoveryService(
                metadata=self._client.metadata,
                policy=policy,
                resolver=self._symbology,
            )
        return self._discovery_service

    def _build_discovery_policy(self) -> DiscoveryPolicy:
        env_policy = DiscoveryPolicy.from_env(os.environ)
        return DiscoveryPolicy(
            coverage=self._policy,
            dataset_allowlist=env_policy.dataset_allowlist,
            dataset_denylist=env_policy.dataset_denylist,
            max_cost_usd=env_policy.max_cost_usd,
            max_candidates=env_policy.max_candidates,
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
            result_frame = cast(pd.DataFrame, result.to_df())
        else:
            result_frame = pd.DataFrame(result)

        if (
            isinstance(result_frame.index, pd.DatetimeIndex)
            and result_frame.index.name == "ts_event"
            and "ts_event" not in result_frame.columns
        ):
            result_frame = result_frame.reset_index()

        if "ts_event" not in result_frame.columns and "ts" in result_frame.columns:
            try:
                result_frame["ts_event"] = pd.to_datetime(result_frame["ts"], utc=True)
            except Exception:
                pass

        if "ts_event" in result_frame.columns and pd.api.types.is_datetime64_any_dtype(result_frame["ts_event"]):
            result_frame["ts_event"] = result_frame["ts_event"].astype("int64")

        if "ts_init" not in result_frame.columns and "ts_event" in result_frame.columns:
            result_frame["ts_init"] = result_frame["ts_event"]

        return result_frame

    def _canonicalize_chunk(
        self,
        *,
        dataset: str,
        schema: str,
        frame: pd.DataFrame,
        symbol: str,
        instrument_id: str | None,
        source_dataset: str | None,
        aggregation_mode: str | None = None,
        scaling_factor: float | None = None,
        volume_residual_abs: float | None = None,
        volume_residual_rel: float | None = None,
        calibration_version: str | None = None,
    ) -> tuple[pd.DataFrame, CanonicalizationStats | None]:
        if frame.empty:
            return frame, None
        normalized_schema = schema.lower()
        canonical_source = source_dataset or dataset
        if dataset == "EQUS.MINI" and normalized_schema == "ohlcv-1m":
            publisher_override = 95 if canonical_source != dataset else None
            result = canonicalize_equities_minute_bars(
                frame,
                source_dataset=canonical_source,
                symbol=symbol,
                instrument_id=instrument_id,
                publisher_id=publisher_override,
                aggregation_mode=aggregation_mode,
                scaling_factor=scaling_factor,
                volume_residual_abs=volume_residual_abs,
                volume_residual_rel=volume_residual_rel,
                calibration_version=calibration_version,
            )
            frame_out = result.frame
            frame_out = frame_out.copy()
            frame_out["source_dataset"] = canonical_source
            frame_out["aggregation_mode"] = aggregation_mode or "native"
            frame_out["scaling_factor"] = (
                float(scaling_factor) if scaling_factor is not None else None
            )
            frame_out["calibration_version"] = calibration_version
            result = CanonicalizationResult(frame=frame_out, stats=result.stats)
            self._record_canonicalization(
                dataset=dataset,
                schema=schema,
                symbol=symbol,
                stats=result.stats,
            )
            return result.frame, result.stats
        return frame, None

    def _record_canonicalization(
        self,
        *,
        dataset: str,
        schema: str,
        symbol: str,
        stats: CanonicalizationStats,
    ) -> None:
        _CANONICALIZED_ROWS_COUNTER.labels(
            dataset=dataset,
            source_dataset=stats.source_dataset,
        ).inc(stats.rows_out)
        mode = stats.aggregation_mode or "native"
        if stats.volume_residual_abs is not None:
            _VOLUME_RESIDUAL.labels(
                mode=mode,
                residual_type="abs",
                dataset=dataset,
            ).observe(max(float(stats.volume_residual_abs), 0.0))
        if stats.volume_residual_rel is not None:
            _VOLUME_RESIDUAL.labels(
                mode=mode,
                residual_type="rel",
                dataset=dataset,
            ).observe(max(float(stats.volume_residual_rel), 0.0))
        logger.info(
            "ingestion.canonicalize.applied",
            dataset=dataset,
            schema=schema,
            symbol=symbol,
            source_dataset=stats.source_dataset,
            rows_in=stats.rows_in,
            rows_out=stats.rows_out,
            rows_trimmed=stats.rows_trimmed,
            rows_deduped=stats.rows_deduped,
            session_start=str(stats.session_start),
            session_end=str(stats.session_end),
            timezone=stats.timezone,
            aggregation_mode=mode,
            scaling_factor=stats.scaling_factor,
            volume_residual_abs=stats.volume_residual_abs,
            volume_residual_rel=stats.volume_residual_rel,
            calibration_version=stats.calibration_version,
        )

    def _should_use_itch_fallback(
        self,
        *,
        dataset: str,
        schema: str,
        window: IngestionWindow,
    ) -> bool:
        if dataset != "EQUS.MINI" or schema.lower() != "ohlcv-1m":
            return False
        start_ns, _ = self.get_available_range_ns(dataset=dataset, schema=schema)
        if start_ns is None:
            return False
        window_end_ns = int(window.end.timestamp() * 1_000_000_000)
        return window_end_ns <= start_ns

    def _attempt_fallback_to_itch(
        self,
        *,
        schema: str,
        symbol: str,
        download_symbol: str,
        window: IngestionWindow,
        instrument_id: str | None,
    ) -> tuple[pd.DataFrame, CanonicalizationStats] | None:
        fallback_dataset = self._fallback_dataset
        try:
            self._validate_dataset(fallback_dataset)
        except IngestionError:
            return None
        fallback_resolution = self._resolve_symbol(
            dataset=fallback_dataset,
            schema=schema,
            symbol=symbol,
            window=window,
        )
        fallback_symbol = fallback_resolution.preferred
        fallback_frame = self._fetch_chunk(
            dataset=fallback_dataset,
            schema=schema,
            symbol=fallback_symbol,
            window=window,
        )
        resolved_instrument = fallback_resolution.instrument_id or instrument_id
        symbol_calibration = self._get_symbol_calibration(download_symbol)
        aggregation_mode = "raw_fallback"
        scaling_factor: float | None = None
        volume_residual_abs: float | None = None
        volume_residual_rel: float | None = None
        bars_to_use = fallback_frame
        fallback_reference = fallback_frame if not fallback_frame.empty else None
        calibration_version = self._calibration_version if symbol_calibration is not None else None

        if self._enable_trade_reaggregation:
            trades_frame = self._fetch_chunk(
                dataset=fallback_dataset,
                schema="trades",
                symbol=fallback_symbol,
                window=window,
            )
            reaggregated = self._reaggregate_trades_to_minute_bars(
                trades_frame,
                symbol=fallback_symbol,
                instrument_id=resolved_instrument,
            )
            if not reaggregated.empty:
                aggregation_mode = "reaggregated_trades"
                bars_to_use = reaggregated
                if not fallback_frame.empty:
                    volume_residual_abs, volume_residual_rel = self._volume_residual(
                        fallback_frame,
                        reaggregated,
                    )

        calibration_applied = False
        if symbol_calibration is not None and not bars_to_use.empty:
            calibrated = self._apply_calibration_to_bars(
                bars_to_use,
                symbol_calibration,
            )
            if not calibrated.equals(bars_to_use):
                bars_to_use = calibrated
                calibration_applied = True
                aggregation_mode = f"calibrated_{aggregation_mode}"
                if fallback_reference is not None:
                    volume_residual_abs, volume_residual_rel = self._volume_residual(
                        fallback_reference,
                        bars_to_use,
                    )

        pre_scaling_frame = bars_to_use
        if (
            not calibration_applied
            and self._enable_volume_scaling
            and not bars_to_use.empty
        ):
            scale_estimate = self._resolve_volume_scale_factor(
                symbol=download_symbol,
                fallback_symbol=fallback_symbol,
            )
            if scale_estimate is not None:
                scaled = self._apply_volume_scale(bars_to_use, scale_estimate.factor)
                residual_base = fallback_reference if fallback_reference is not None else pre_scaling_frame
                if residual_base is not None:
                    volume_residual_abs, volume_residual_rel = self._volume_residual(
                        residual_base,
                        scaled,
                    )
                bars_to_use = scaled
                scaling_factor = scale_estimate.factor
                aggregation_mode = (
                    "reaggregated_trades_scaled"
                    if aggregation_mode == "reaggregated_trades"
                    else "scaled_volume"
                )

        if fallback_reference is None and volume_residual_abs is None and not bars_to_use.empty:
            volume_residual_abs, volume_residual_rel = self._volume_residual(
                pre_scaling_frame,
                bars_to_use,
            )

        if bars_to_use.empty:
            return None

        canonical_frame, stats = self._canonicalize_chunk(
            dataset="EQUS.MINI",
            schema=schema,
            frame=bars_to_use,
            symbol=download_symbol,
            instrument_id=resolved_instrument,
            source_dataset=fallback_dataset,
            aggregation_mode=aggregation_mode,
            scaling_factor=scaling_factor,
            volume_residual_abs=volume_residual_abs,
            volume_residual_rel=volume_residual_rel,
            calibration_version=calibration_version,
        )
        if stats is None:
            return None
        logger.info(
            "ingestion.canonicalize.fallback_applied",
            dataset="EQUS.MINI",
            fallback_dataset=fallback_dataset,
            schema=schema,
            symbol=download_symbol,
            aggregation_mode=aggregation_mode,
            scaling_factor=scaling_factor,
            volume_residual_abs=volume_residual_abs,
            volume_residual_rel=volume_residual_rel,
        )
        return canonical_frame, stats

    def _reaggregate_trades_to_minute_bars(
        self,
        frame: pd.DataFrame,
        *,
        symbol: str,
        instrument_id: str | None,
    ) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()

        working = frame.copy(deep=True)
        if "ts_event" not in working.columns:
            if "ts" in working.columns:
                working["ts_event"] = working["ts"]
            elif isinstance(working.index, pd.DatetimeIndex):
                working = working.reset_index().rename(columns={working.index.name or "index": "ts_event"})
            else:
                return pd.DataFrame()

        symbol_calibration = self._get_symbol_calibration(symbol)
        if (
            symbol_calibration is not None
            and symbol_calibration.sale_condition_allowlist
            and "sale_condition" in working.columns
        ):
            sale_series = working["sale_condition"].astype(str).str.strip()
            working = working[sale_series.isin(symbol_calibration.sale_condition_allowlist)]
            if working.empty:
                return pd.DataFrame()

        event_series = working["ts_event"]
        event_dt = pd.to_datetime(event_series, utc=True, errors="coerce")
        working = working.assign(_event_dt=event_dt)
        working = working[working["_event_dt"].notna()]
        if working.empty:
            return pd.DataFrame()

        price_col = next((col for col in ("price", "trade_px", "last", "px_last") if col in working.columns), None)
        volume_col = next((col for col in ("size", "quantity", "volume", "trade_size") if col in working.columns), None)
        if price_col is None or volume_col is None:
            return pd.DataFrame()

        working = working.assign(
            _price=pd.to_numeric(working[price_col], errors="coerce"),
            _volume=pd.to_numeric(working[volume_col], errors="coerce").fillna(0.0),
        )
        working = working[working["_price"].notna()]
        if working.empty:
            return pd.DataFrame()

        working = working.assign(_bucket=working["_event_dt"].dt.floor("min"))
        grouped = working.groupby("_bucket", sort=True)
        aggregated = pd.DataFrame(
            {
                "open": grouped["_price"].first(),
                "high": grouped["_price"].max(),
                "low": grouped["_price"].min(),
                "close": grouped["_price"].last(),
                "volume": grouped["_volume"].sum(),
                "trade_count": grouped.size(),
            },
        )
        if aggregated.empty:
            return pd.DataFrame()
        aggregated = aggregated.reset_index().rename(columns={"_bucket": "timestamp"})
        aggregated["ts_event"] = aggregated["timestamp"].astype("int64")
        aggregated["ts_init"] = aggregated["ts_event"]
        aggregated = aggregated.drop(columns=["timestamp"], errors="ignore")
        aggregated["volume"] = aggregated["volume"].round().astype("int64", errors="ignore")
        aggregated["trade_count"] = aggregated["trade_count"].astype("int64", errors="ignore")
        aggregated["rtype"] = 33
        aggregated["publisher_id"] = 95
        aggregated["symbol"] = symbol
        if instrument_id is not None:
            aggregated["instrument_id"] = instrument_id
        return aggregated.sort_values("ts_event").reset_index(drop=True)

    def _resolve_volume_scale_factor(
        self,
        *,
        symbol: str,
        fallback_symbol: str,
    ) -> VolumeScaleEstimate | None:
        cache_key = symbol.upper()
        cached = self._volume_scale_cache.get(cache_key)
        if cached is not None:
            return cached

        available = self.get_available_range_ns(dataset="EQUS.MINI", schema="ohlcv-1m")
        eq_start_ns, eq_end_ns = available
        if eq_start_ns is None or eq_end_ns is None:
            return None
        reference_end_ns = min(eq_end_ns, eq_start_ns + self._scaling_reference_days * DAY_NS)
        if reference_end_ns <= eq_start_ns:
            return None

        start_dt = datetime.fromtimestamp(eq_start_ns / 1_000_000_000, tz=UTC)
        end_dt = datetime.fromtimestamp(reference_end_ns / 1_000_000_000, tz=UTC)
        reference_window = IngestionWindow(start=start_dt, end=end_dt)
        eq_frame = self._fetch_chunk(
            dataset="EQUS.MINI",
            schema="ohlcv-1m",
            symbol=symbol,
            window=reference_window,
        )
        if eq_frame.empty:
            return None
        eq_canon = canonicalize_equities_minute_bars(
            eq_frame,
            source_dataset="EQUS.MINI",
            symbol=symbol,
            publisher_id=None,
            aggregation_mode="native",
        ).frame
        fallback_frame = self._fetch_chunk(
            dataset=self._fallback_dataset,
            schema="ohlcv-1m",
            symbol=fallback_symbol,
            window=reference_window,
        )
        if fallback_frame.empty:
            return None
        fallback_canon = canonicalize_equities_minute_bars(
            fallback_frame,
            source_dataset=self._fallback_dataset,
            symbol=fallback_symbol,
        ).frame
        if "volume" not in eq_canon.columns or "volume" not in fallback_canon.columns:
            return None
        eq_series = eq_canon.set_index("ts_event")["volume"].astype(float)
        fallback_series = fallback_canon.set_index("ts_event")["volume"].astype(float)
        joined = eq_series.to_frame("eq").join(fallback_series.to_frame("fallback"), how="inner")
        joined = joined[(joined["eq"] > 0) & (joined["fallback"] > 0)]
        if joined.empty:
            return None
        ratios = joined["eq"] / joined["fallback"]
        factor = float(ratios.median())
        factor = min(max(factor, self._scaling_min_ratio), self._scaling_max_ratio)
        scaled_total = (joined["fallback"] * factor).sum()
        eq_total = joined["eq"].sum()
        residual_abs = abs(scaled_total - eq_total)
        residual_rel = residual_abs / eq_total if eq_total > 0 else None
        estimate = VolumeScaleEstimate(
            symbol=symbol,
            factor=factor,
            sample_minutes=int(joined.shape[0]),
            reference_start_ns=eq_start_ns,
            reference_end_ns=reference_end_ns,
            residual_abs=residual_abs,
            residual_rel=residual_rel,
        )
        self._volume_scale_cache[cache_key] = estimate
        logger.info(
            "ingestion.fallback.scale.estimated",
            symbol=symbol,
            fallback_symbol=fallback_symbol,
            factor=factor,
            sample_minutes=estimate.sample_minutes,
            residual_abs=residual_abs,
            residual_rel=residual_rel,
        )
        return estimate

    @staticmethod
    def _volume_residual(
        base: pd.DataFrame,
        adjusted: pd.DataFrame,
    ) -> tuple[float | None, float | None]:
        if "ts_event" not in base.columns or "volume" not in base.columns:
            return None, None
        if "ts_event" not in adjusted.columns or "volume" not in adjusted.columns:
            return None, None
        base_series = pd.to_numeric(base["volume"], errors="coerce")
        adjusted_series = pd.to_numeric(adjusted["volume"], errors="coerce")
        base_ts = pd.to_numeric(base["ts_event"], errors="coerce")
        adjusted_ts = pd.to_numeric(adjusted["ts_event"], errors="coerce")
        base_df = pd.DataFrame({"ts_event": base_ts, "volume": base_series}).dropna()
        adjusted_df = pd.DataFrame({"ts_event": adjusted_ts, "volume": adjusted_series}).dropna()
        merged = base_df.merge(adjusted_df, on="ts_event", suffixes=("_base", "_adjusted"))
        if merged.empty:
            return None, None
        base_total = float(merged["volume_base"].sum())
        adjusted_total = float(merged["volume_adjusted"].sum())
        diff = abs(adjusted_total - base_total)
        if base_total <= 0:
            return diff, None
        return diff, diff / base_total

    @staticmethod
    def _apply_volume_scale(
        frame: pd.DataFrame,
        factor: float,
    ) -> pd.DataFrame:
        scaled = frame.copy(deep=True)
        if "volume" not in scaled.columns:
            return scaled
        volume_series = pd.to_numeric(scaled["volume"], errors="coerce").fillna(0.0)
        adjusted = (volume_series * factor).round()
        try:
            scaled["volume"] = adjusted.astype("int64")
        except (TypeError, ValueError):
            scaled["volume"] = adjusted
        return scaled

    def _apply_calibration_to_bars(
        self,
        frame: pd.DataFrame,
        calibration: SymbolCalibration,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame
        working = frame.copy(deep=True)
        minute_values = self._compute_minute_of_day(working)
        if calibration.exclude_auction_minutes:
            mask = ~np.isin(minute_values, list(calibration.exclude_auction_minutes))
            working = working.loc[mask].reset_index(drop=True)
            minute_values = minute_values[mask]
            if working.empty:
                return working
        working = self._apply_calibrated_volume_scaling(
            working,
            calibration,
            minute_values=minute_values,
        )
        working = self._apply_calibrated_price_scaling(
            working,
            calibration,
            minute_values=minute_values,
        )
        working = self._apply_split_adjustments(working, calibration)
        return working

    def _apply_calibrated_volume_scaling(
        self,
        frame: pd.DataFrame,
        calibration: SymbolCalibration,
        *,
        minute_values: np.ndarray | None = None,
    ) -> pd.DataFrame:
        if frame.empty or not calibration.volume_scale_by_minute:
            return frame
        working = frame.copy(deep=True)
        if minute_values is None:
            minute_values = self._compute_minute_of_day(working)
        scale_values = np.fromiter(
            (calibration.scale_for_minute(value) for value in minute_values),
            dtype=float,
        )
        if np.allclose(scale_values, 1.0):
            return working
        if "volume" in working.columns:
            volume_series = pd.to_numeric(working["volume"], errors="coerce").fillna(0.0)
        else:
            volume_series = pd.Series(0.0, index=working.index, dtype="float64")
        adjusted = np.rint(volume_series.to_numpy(dtype=float) * scale_values)
        working["volume"] = adjusted.astype("int64")
        return working

    def _apply_calibrated_price_scaling(
        self,
        frame: pd.DataFrame,
        calibration: SymbolCalibration,
        *,
        minute_values: np.ndarray | None = None,
    ) -> pd.DataFrame:
        if frame.empty or not calibration.price_scaling_by_minute:
            return frame
        working = frame.copy(deep=True)
        if minute_values is None:
            minute_values = self._compute_minute_of_day(working)
        scale_values = np.fromiter(
            (calibration.price_scale_for_minute(value) for value in minute_values),
            dtype=float,
        )
        if np.allclose(scale_values, 1.0):
            return working
        price_columns = [
            column
            for column in ("open", "high", "low", "close")
            if column in working.columns
        ]
        if not price_columns:
            return working
        scaled = scale_values.astype(float)
        for column in price_columns:
            series = pd.to_numeric(working[column], errors="coerce")
            working[column] = series.to_numpy(dtype=float) * scaled
        return working

    def _apply_split_adjustments(
        self,
        frame: pd.DataFrame,
        calibration: SymbolCalibration,
    ) -> pd.DataFrame:
        if frame.empty or not calibration.split_events:
            return frame
        working = frame.copy(deep=True)
        ts_series = pd.to_numeric(working["ts_event"], errors="coerce")
        ts_datetimes = pd.to_datetime(ts_series, utc=True, unit="ns", errors="coerce")
        if ts_datetimes.isna().all():
            return working
        split_points: list[tuple[datetime, float]] = []
        for raw_date, factor in calibration.split_events.items():
            parsed = raw_date.replace("Z", "+00:00")
            split_dt = datetime.fromisoformat(parsed)
            if split_dt.tzinfo is None:
                split_dt = split_dt.replace(tzinfo=UTC)
            split_points.append((split_dt, float(factor)))
        if not split_points:
            return working
        split_points.sort()
        factors = np.ones(len(working.index), dtype=float)
        for split_dt, factor in split_points:
            mask = ts_datetimes >= split_dt
            factors[mask.to_numpy()] *= factor
        if np.allclose(factors, 1.0):
            return working
        price_columns = [
            column
            for column in ("open", "high", "low", "close")
            if column in working.columns
        ]
        if price_columns:
            for column in price_columns:
                series = pd.to_numeric(working[column], errors="coerce")
                working[column] = series.to_numpy(dtype=float) * factors
        if "volume" in working.columns:
            volume_series = pd.to_numeric(working["volume"], errors="coerce").fillna(0.0)
            adjusted = np.where(
                factors != 0.0,
                np.rint(volume_series.to_numpy(dtype=float) / factors),
                0.0,
            )
            working["volume"] = adjusted.astype("int64")
        return working

    @staticmethod
    def _compute_minute_of_day(
        frame: pd.DataFrame,
    ) -> np.ndarray:
        ts_numeric = pd.to_numeric(frame["ts_event"], errors="coerce").fillna(0)
        ts_array = ts_numeric.to_numpy(dtype=np.int64)
        return ((ts_array // 60_000_000_000) % 1440).astype(int)

    def _get_symbol_calibration(self, symbol: str) -> SymbolCalibration | None:
        if self._calibration is None:
            return None
        return self._calibration.for_symbol(symbol)

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
    def _extract_schema_ranges(
        range_info: Mapping[str, Any],
    ) -> dict[str, tuple[int | None, int | None]]:
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
        try:
            sym_client = getattr(service._client, "symbology", None)
        except Exception:  # pragma: no cover - defensive guard
            sym_client = None
        self.symbology = cast(DatabentoSymbologyClient | None, sym_client)


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
    "SymbolDatasetDiscovery",
    "SymbolIngestionSummary",
    "build_historical_adapter",
    "build_like_client",
]
