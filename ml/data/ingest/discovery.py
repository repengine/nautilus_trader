"""Dynamic Databento dataset discovery helpers."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import structlog

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.config.market_data import MarketDatasetInput
from ml.data.ingest.policy import DatabentoCoveragePolicy
from ml.data.ingest.symbology import DatabentoSymbologyResolver
from ml.data.ingest.symbology import SymbologyResolutionError
from ml.data.ingest.symbology import SymbolResolution
from ml.registry.dataclasses import StorageKind


if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from ml.data.ingest.service import DatabentoMetadataClient


logger = structlog.get_logger(__name__)


@dataclass(slots=True, frozen=True)
class DiscoveryRequest:
    """Parameters describing discovery for a single symbol/schema."""

    symbol: str
    schema: str
    start: datetime
    end: datetime


@dataclass(slots=True, frozen=True)
class DiscoveryPolicy:
    """Runtime policy controls for dataset discovery."""

    coverage: DatabentoCoveragePolicy
    dataset_allowlist: tuple[str, ...] | None
    dataset_denylist: frozenset[str] | None
    max_cost_usd: float | None
    max_candidates: int | None

    @staticmethod
    def from_env(env: Mapping[str, str]) -> DiscoveryPolicy:
        coverage = DatabentoCoveragePolicy.from_env()
        allow_env = env.get("DATABENTO_DISCOVERY_DATASETS", "").strip()
        allow_tokens = tuple(
            token
            for token in (item.strip() for item in allow_env.split(","))
            if token
        )
        deny_env = env.get("DATABENTO_DISCOVERY_DENYLIST", "").strip()
        deny_tokens = frozenset(
            token
            for token in (item.strip() for item in deny_env.split(","))
            if token
        )
        max_cost_env = env.get("DATABENTO_DISCOVERY_MAX_COST_USD", "").strip()
        max_cost = float(max_cost_env) if max_cost_env else None
        max_candidates_env = env.get("DATABENTO_DISCOVERY_MAX_CANDIDATES", "").strip()
        max_candidates = int(max_candidates_env) if max_candidates_env.isdigit() else None
        default_allow = ("XNAS.ITCH",)
        allowlist = allow_tokens if allow_tokens else default_allow
        return DiscoveryPolicy(
            coverage=coverage,
            dataset_allowlist=allowlist,
            dataset_denylist=deny_tokens if deny_tokens else None,
            max_cost_usd=max_cost,
            max_candidates=max_candidates,
        )

    def candidates(self, available: Sequence[str]) -> Iterable[str]:
        """Yield dataset ids that satisfy allow/deny rules."""
        if self.dataset_allowlist:
            base_iterable = [dataset for dataset in self.dataset_allowlist if dataset in available]
        else:
            base_iterable = list(available)
        filtered = [
            dataset
            for dataset in base_iterable
            if not self.dataset_denylist or dataset not in self.dataset_denylist
        ]
        if self.max_candidates is None:
            return tuple(filtered)
        return tuple(filtered[: int(self.max_candidates)])

    def cost_allowed(self, cost: float | None) -> bool:
        if cost is None or self.max_cost_usd is None:
            return True
        return cost <= self.max_cost_usd


@dataclass(slots=True, frozen=True)
class DiscoveredInput:
    """Resolved dataset binding derived from discovery."""

    requested_symbol: str
    symbol: str
    dataset_id: str
    schema: str
    storage_kind: StorageKind
    available_start_ns: int | None
    available_end_ns: int | None
    cost_usd: float | None
    instrument_id: str | None

    def to_market_input(self) -> MarketDatasetInput:
        return MarketDatasetInput(
            dataset_id=self.dataset_id,
            symbols=(self.symbol,),
            schema_override=self.schema,
            storage_kind_override=self.storage_kind,
        )


class DatasetDiscoveryError(RuntimeError):
    """Raised when no dataset satisfies discovery constraints."""


class DatasetDiscoveryService:
    """
    Runtime Databento dataset discovery helper.

    Example
    -------
    >>> from datetime import UTC, datetime, timedelta
    >>> from ml.data.ingest.discovery import DiscoveryPolicy, DiscoveryRequest
    >>> historical = ...  # databento.Historical instance
    >>> metadata_client = historical.metadata
    >>> resolver = DatabentoSymbologyResolver(client=historical.symbology)
    >>> policy = DiscoveryPolicy.from_env({})
    >>> service = DatasetDiscoveryService(
    ...     metadata=metadata_client,
    ...     policy=policy,
    ...     resolver=resolver,
    ... )
    >>> end = datetime.now(tz=UTC)
    >>> start = end - timedelta(days=7)
    >>> requests = (DiscoveryRequest(symbol="INTC", schema="ohlcv-1m", start=start, end=end),)
    >>> market_inputs = service.discover(requests=requests)
    >>> market_inputs[0].dataset_id
    'XNAS.ITCH'
    """

    def __init__(
        self,
        *,
        metadata: DatabentoMetadataClient,
        policy: DiscoveryPolicy,
        resolver: DatabentoSymbologyResolver | None = None,
    ) -> None:
        self._metadata = metadata
        self._policy = policy
        if resolver is None:
            raise ValueError("DatasetDiscoveryService requires a symbology resolver")
        self._resolver = resolver
        self._metrics = _DiscoveryMetrics()

    @property
    def policy(self) -> DiscoveryPolicy:
        """Expose the discovery policy for runtime adjustments."""
        return self._policy

    def discover(
        self,
        *,
        requests: Sequence[DiscoveryRequest],
        dataset_hint: str | None = None,
    ) -> tuple[MarketDatasetInput, ...]:
        if not requests:
            return ()
        with self._metrics.discovery_latency.time():
            dataset_listing = self._list_datasets()
            ordered_candidates = self._ordered_candidates(dataset_listing, dataset_hint)
            discovered: list[MarketDatasetInput] = []
            for request in requests:
                discovered_input = self._discover_for_request(
                    request=request,
                    ordered_candidates=ordered_candidates,
                )
                discovered.append(discovered_input.to_market_input())
            return tuple(discovered)

    def discover_one(
        self,
        *,
        request: DiscoveryRequest,
        dataset_hint: str | None = None,
    ) -> DiscoveredInput:
        with self._metrics.discovery_latency.time():
            dataset_listing = self._list_datasets()
            ordered_candidates = self._ordered_candidates(dataset_listing, dataset_hint)
            return self._discover_for_request(
                request=request,
                ordered_candidates=ordered_candidates,
            )

    def _ordered_candidates(
        self,
        datasets: Sequence[str],
        dataset_hint: str | None,
    ) -> tuple[str, ...]:
        if dataset_hint:
            normalized_hint = dataset_hint.strip()
            prioritized = [normalized_hint]
            prioritized.extend(dataset for dataset in datasets if dataset != normalized_hint)
            base_sequence = tuple(prioritized)
        else:
            base_sequence = tuple(datasets)
        return tuple(self._policy.candidates(base_sequence))

    def _discover_for_request(
        self,
        *,
        request: DiscoveryRequest,
        ordered_candidates: Sequence[str],
    ) -> DiscoveredInput:
        symbol = request.symbol.strip().upper()
        schema = request.schema.strip()
        window_start, window_end = self._policy.coverage.clamp_range(
            request.start,
            request.end,
            dataset=None,
            schema=schema,
        )
        if window_end <= window_start:
            raise DatasetDiscoveryError(
                f"Empty window after policy clamp for {symbol}/{schema}: start={request.start}, end={request.end}",
            )
        best: DiscoveredInput | None = None
        best_cost: float = math.inf
        for dataset_id in ordered_candidates:
            if not dataset_id:
                continue
            if not self._schema_supported(dataset_id, schema):
                continue
            available_start_ns, available_end_ns = self._dataset_bounds(dataset_id, schema)
            clamped_start_ns, clamped_end_ns = self._intersect_bounds(
                window_start,
                window_end,
                available_start_ns,
                available_end_ns,
            )
            if clamped_start_ns is None or clamped_end_ns is None:
                continue
            iso_start, iso_end = (
                _ns_to_iso(clamped_start_ns),
                _ns_to_iso(clamped_end_ns),
            )
            start_dt = datetime.fromtimestamp(clamped_start_ns / 1_000_000_000, tz=UTC)
            end_dt = datetime.fromtimestamp(clamped_end_ns / 1_000_000_000, tz=UTC)
            try:
                resolution = self._resolve_symbol(
                    original_symbol=request.symbol,
                    dataset_id=dataset_id,
                    schema=schema,
                    start=start_dt,
                    end=end_dt,
                )
            except SymbologyResolutionError as exc:
                logger.debug(
                    "Symbology resolution rejected",
                    dataset=dataset_id,
                    symbol=request.symbol,
                    schema=schema,
                    reason=str(exc),
                )
                continue
            cost, resolved_symbol = self._estimate_cost(
                dataset_id=dataset_id,
                schema=schema,
                symbol_variants=resolution.candidates,
                start_iso=iso_start,
                end_iso=iso_end,
            )
            if not self._policy.cost_allowed(cost):
                logger.info(
                    "Discovery candidate rejected due to cost",
                    dataset=dataset_id,
                    symbol=resolved_symbol or resolution.preferred,
                    schema=schema,
                    cost=cost,
                    max_cost=self._policy.max_cost_usd,
                )
                self._metrics.candidates_rejected_cost.inc()
                continue
            if cost is not None and cost < best_cost:
                best_cost = cost
                best = DiscoveredInput(
                    requested_symbol=resolution.original,
                    symbol=resolved_symbol or resolution.preferred,
                    dataset_id=dataset_id,
                    schema=schema,
                    storage_kind=StorageKind.POSTGRES,
                    available_start_ns=available_start_ns,
                    available_end_ns=available_end_ns,
                    cost_usd=cost,
                    instrument_id=resolution.instrument_id,
                )
                logger.info(
                    "Dataset discovery resolved",
                    dataset=dataset_id,
                    symbol=resolved_symbol or resolution.preferred,
                    requested_symbol=resolution.original,
                    schema=schema,
                    cost=cost,
                    instrument_id=resolution.instrument_id,
                )
                if cost == 0.0:
                    break
        if best is None:
            raise DatasetDiscoveryError(
                f"No dataset satisfies discovery rules for symbol={symbol} schema={schema}",
            )
        self._metrics.candidates_selected.inc()
        return best

    def _schema_supported(self, dataset_id: str, schema: str) -> bool:
        schemas = self._list_schemas(dataset_id)
        return schema in schemas

    def _dataset_bounds(
        self,
        dataset_id: str,
        schema: str,
    ) -> tuple[int | None, int | None]:
        range_info = self._dataset_range(dataset_id)
        schema_payload = range_info.get("schema", {})
        schema_info = schema_payload.get(schema, {})
        start_iso = schema_info.get("start") or range_info.get("start")
        end_iso = schema_info.get("end") or range_info.get("end")
        return _iso_to_ns(start_iso), _iso_to_ns(end_iso)

    def _intersect_bounds(
        self,
        start_dt: datetime,
        end_dt: datetime,
        available_start_ns: int | None,
        available_end_ns: int | None,
    ) -> tuple[int | None, int | None]:
        start_ns = _datetime_to_ns(start_dt)
        end_ns = _datetime_to_ns(end_dt)
        if available_start_ns is not None and end_ns <= available_start_ns:
            return None, None
        if available_end_ns is not None and start_ns >= available_end_ns:
            return None, None
        clamped_start = max(start_ns, available_start_ns) if available_start_ns is not None else start_ns
        clamped_end = min(end_ns, available_end_ns) if available_end_ns is not None else end_ns
        if clamped_end <= clamped_start:
            return None, None
        return clamped_start, clamped_end

    def _resolve_symbol(
        self,
        *,
        original_symbol: str,
        dataset_id: str,
        schema: str,
        start: datetime,
        end: datetime,
    ) -> SymbolResolution:
        return self._resolver.resolve(
            dataset=dataset_id,
            symbol=original_symbol,
            schema=schema,
            start=start,
            end=end,
        )

    def _estimate_cost(
        self,
        *,
        dataset_id: str,
        schema: str,
        symbol_variants: tuple[str, ...],
        start_iso: str,
        end_iso: str,
    ) -> tuple[float | None, str | None]:
        histogram = self._metrics.cost_latency.labels(dataset=dataset_id, schema=schema)
        for candidate in symbol_variants or ("",):
            if not candidate:
                continue
            try:
                with histogram.time():
                    cost = self._metadata.get_cost(
                        dataset=dataset_id,
                        schema=schema,
                        symbols=[candidate],
                        start=start_iso,
                        end=end_iso,
                    )
                return float(cost), candidate
            except Exception as exc:  # pragma: no cover - network error surface
                logger.debug(
                    "Cost estimation failed",
                    dataset=dataset_id,
                    schema=schema,
                    symbol=candidate,
                    start=start_iso,
                    end=end_iso,
                    error=str(exc),
                )
                continue
        self._metrics.cost_failures.inc()
        return None, None

    @lru_cache(maxsize=64)
    def _list_datasets(self) -> tuple[str, ...]:
        try:
            list_datasets = getattr(self._metadata, "list_datasets")
        except AttributeError as exc:  # pragma: no cover - defensive guard
            raise DatasetDiscoveryError("Metadata client does not expose list_datasets") from exc
        try:
            listing_raw = list_datasets()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Failed to list datasets", error=str(exc))
            raise
        if isinstance(listing_raw, str):
            candidates = [token.strip() for token in listing_raw.splitlines() if token.strip()]
            return tuple(candidates)
        if isinstance(listing_raw, (list, tuple)):
            return tuple(str(item) for item in listing_raw if str(item).strip())
        try:
            return tuple(json.loads(str(listing_raw)))
        except Exception as exc:  # pragma: no cover - fallback path
            logger.error("Unable to parse dataset list", error=str(exc))
            raise

    @lru_cache(maxsize=64)
    def _list_schemas(self, dataset_id: str) -> frozenset[str]:
        try:
            list_schemas = getattr(self._metadata, "list_schemas")
        except AttributeError as exc:  # pragma: no cover - defensive guard
            raise DatasetDiscoveryError("Metadata client does not expose list_schemas") from exc
        try:
            raw = list_schemas(dataset=dataset_id)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Failed to list schemas", dataset=dataset_id, error=str(exc))
            return frozenset()
        if isinstance(raw, (list, tuple)):
            return frozenset(str(item) for item in raw)
        try:
            parsed = json.loads(str(raw))
        except Exception:
            return frozenset()
        if isinstance(parsed, list):
            return frozenset(str(item) for item in parsed)
        return frozenset()

    @lru_cache(maxsize=64)
    def _dataset_range(self, dataset_id: str) -> Mapping[str, Any]:
        histogram = self._metrics.range_latency.labels(dataset=dataset_id)
        with histogram.time():
            return self._metadata.get_dataset_range(dataset_id)


class _DiscoveryMetrics:
    """Prometheus metrics bundle for discovery service."""

    def __init__(self) -> None:
        self.discovery_latency = get_histogram(
            "nautilus_ml_discovery_latency_seconds",
            "Latency of dataset discovery operations",
        )
        self.range_latency = get_histogram(
            "nautilus_ml_discovery_range_latency_seconds",
            "Latency for dataset range lookups",
            labelnames=("dataset",),
        )
        self.cost_latency = get_histogram(
            "nautilus_ml_discovery_cost_latency_seconds",
            "Latency for cost estimation",
            labelnames=("dataset", "schema"),
        )
        self.cost_failures = get_counter(
            "nautilus_ml_discovery_cost_failures_total",
            "Count of failed cost estimation attempts",
        )
        self.candidates_selected = get_counter(
            "nautilus_ml_discovery_candidates_selected_total",
            "Count of candidate datasets selected",
        )
        self.candidates_rejected_cost = get_counter(
            "nautilus_ml_discovery_candidates_rejected_cost_total",
            "Candidates rejected due to policy cost limits",
        )


def _iso_to_ns(value: str | None) -> int | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _datetime_to_ns(dt)


def _datetime_to_ns(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    value_utc = value.astimezone(UTC)
    return int(value_utc.timestamp() * 1_000_000_000)


def _ns_to_iso(value: int) -> str:
    seconds = value / 1_000_000_000
    dt = datetime.fromtimestamp(seconds, tz=UTC)
    return dt.isoformat().replace("+00:00", "Z")


__all__ = [
    "DatasetDiscoveryError",
    "DatasetDiscoveryService",
    "DiscoveryPolicy",
    "DiscoveryRequest",
]
