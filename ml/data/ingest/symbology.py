"""Databento symbology resolution utilities."""

from __future__ import annotations

import time
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from functools import lru_cache
from typing import Any, Protocol

import structlog
from databento.common.error import BentoClientError
from databento.common.error import BentoHttpError
from databento.common.error import BentoServerError

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram


logger = structlog.get_logger(__name__)

_ALIAS_OVERRIDES: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "EQUS.MINI": {
        "BRK": ("BRK.B",),
    },
}


class DatabentoSymbologyClient(Protocol):
    """Protocol covering the subset of ``Historical.symbology`` we depend on."""

    def resolve(
        self,
        *,
        symbols: Sequence[str],
        dataset: str,
        stype_in: str,
        stype_out: str,
        start_date: str,
        end_date: str | None = None,
    ) -> Any:
        del symbols, dataset, stype_in, stype_out, start_date, end_date
        raise NotImplementedError


@dataclass(slots=True, frozen=True)
class SymbolResolution:
    """
    Resolved symbol candidates for a dataset/schema.

    Attributes
    ----------
    original:
        User-supplied symbol (upper-cased, stripped).
    dataset:
        Dataset identifier used for resolution (upper-cased, stripped).
    schema:
        Optional schema hint provided to the resolver.
    candidates:
        Ordered tuple of symbol candidates to try with Databento APIs.
    preferred:
        First candidate, intended for primary ingestion attempts.
    """

    original: str
    dataset: str
    schema: str | None
    candidates: tuple[str, ...]
    preferred: str
    instrument_id: str | None


class SymbologyResolutionError(RuntimeError):
    """Raised when Databento symbology lookup cannot resolve a symbol."""


class DatabentoSymbologyResolver:
    """
    High-level resolver providing dataset-aware symbol variants.

    The resolver favours deterministic, allocation-light heuristics, layering
    Databento symbology lookups when the client provides them. Results are
    cached aggressively to keep hot-paths allocation free while preserving
    observability via Prometheus metrics and structured logs.
    """

    def __init__(
        self,
        *,
        client: DatabentoSymbologyClient | None,
        cache_size: int = 512,
        retry_attempts: int = 3,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self._client = client
        self._cache_size = cache_size
        self._retry_attempts = max(1, retry_attempts)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._metrics = _SymbologyMetrics()
        self._instrument_cache: dict[tuple[str, str, str, str, str], str | None] = {}
        self._last_instrument_id: str | None = None

    @classmethod
    def from_optional_client(
        cls,
        client: DatabentoSymbologyClient | None,
    ) -> DatabentoSymbologyResolver:
        """Factory helper retaining optional client support."""
        return cls(client=client)

    def resolve(
        self,
        *,
        dataset: str,
        symbol: str,
        schema: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> SymbolResolution:
        dataset_norm = dataset.strip().upper()
        symbol_norm = symbol.strip().upper()
        schema_norm = schema.strip() if schema is not None else None
        symbol_root = self._normalize_symbol(symbol_norm)
        self._last_instrument_id = None
        alias_candidates = self._alias_candidates(dataset=dataset_norm, symbol=symbol_norm)
        heuristic_candidates = self._heuristic_candidates(
            symbol=symbol_norm,
            dataset=dataset_norm,
            alias_candidates=alias_candidates,
        )
        remote_candidates = self._remote_candidates(
            dataset=dataset_norm,
            symbol=symbol_root,
            schema=schema_norm,
            start=start,
            end=end,
            alias_candidates=alias_candidates,
        )
        merged = self._deduplicate((*remote_candidates, *heuristic_candidates))
        if not merged:
            merged = (symbol_root,) if symbol_root else (dataset_norm,)
        preferred = merged[0]
        instrument_id = self._last_instrument_id
        if preferred != symbol_norm:
            logger.debug(
                "symbology.resolved",
                dataset=dataset_norm,
                schema=schema_norm,
                original=symbol_norm,
                resolved=preferred,
                candidate_count=len(merged),
            )
        return SymbolResolution(
            original=symbol_norm,
            dataset=dataset_norm,
            schema=schema_norm,
            candidates=merged,
            preferred=preferred,
            instrument_id=instrument_id,
        )

    def _remote_candidates(
        self,
        *,
        dataset: str,
        symbol: str,
        schema: str | None,
        start: datetime | None,
        end: datetime | None,
        alias_candidates: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        client = self._client
        if client is None:
            raise SymbologyResolutionError("Symbology client unavailable")
        cache_key = self._build_cache_key(
            dataset=dataset,
            symbol=symbol,
            schema=schema,
            start=start,
            end=end,
        )
        resolver = self._cached_resolve()

        def _execute_resolve(key: tuple[str, str, str, str, str]) -> tuple[str, ...]:
            symbols = resolver(key)
            self._last_instrument_id = self._instrument_cache.get(key)
            return symbols

        try:
            return _execute_resolve(cache_key)
        except SymbologyResolutionError:
            fallback = self._attempt_alias_resolution(
                dataset=dataset,
                schema=schema,
                start=start,
                end=end,
                alias_candidates=alias_candidates,
                resolver=_execute_resolve,
            )
            if fallback is not None:
                self._metrics.alias_hits.labels(dataset=dataset).inc()
                return fallback
            self._metrics.resolve_miss.labels(dataset=dataset).inc()
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug(
                "symbology.remote_failure",
                dataset=dataset,
                schema=schema,
                symbol=symbol,
                error=str(exc),
                exc_info=True,
            )
            self._metrics.resolve_miss.labels(dataset=dataset).inc()
            raise SymbologyResolutionError(
                f"Databento symbology request failed for {symbol} on {dataset}",
            ) from exc

    def _build_cache_key(
        self,
        *,
        dataset: str,
        symbol: str,
        schema: str | None,
        start: datetime | None,
        end: datetime | None,
    ) -> tuple[str, str, str, str, str]:
        start_dt = (start or datetime.now(tz=UTC)).astimezone(UTC).date()
        end_dt = (end or start or datetime.now(tz=UTC)).astimezone(UTC).date()
        return (
            dataset,
            symbol,
            schema or "",
            start_dt.isoformat(),
            end_dt.isoformat(),
        )

    def _cached_resolve(
        self,
    ) -> Callable[[tuple[str, str, str, str, str]], tuple[str, ...]]:
        client = self._client
        if client is None:
            raise RuntimeError("Symbology client unavailable for resolution cache")
        cache_size = self._cache_size
        retry_attempts = self._retry_attempts
        retry_backoff = self._retry_backoff_seconds

        @lru_cache(maxsize=cache_size)
        def _inner(key: tuple[str, str, str, str, str]) -> tuple[str, ...]:
            dataset, symbol, _schema_token, start_date, end_date = key
            histogram = self._metrics.resolve_latency.labels(dataset=dataset)
            with histogram.time():
                payload: Any | None = None
                for attempt in range(1, retry_attempts + 1):
                    try:
                        payload = client.resolve(
                            symbols=(symbol,),
                            dataset=dataset,
                            stype_in="raw_symbol",
                            stype_out="instrument_id",
                            start_date=start_date,
                            end_date=end_date,
                        )
                        break
                    except BentoServerError as exc:  # pragma: no cover - remote failure
                        self._handle_retryable_http_error(
                            dataset=dataset,
                            symbol=symbol,
                            exc=exc,
                            attempt=attempt,
                            retry_attempts=retry_attempts,
                            backoff=retry_backoff,
                        )
                        continue
                    except BentoHttpError as exc:  # pragma: no cover - defensive
                        if exc.http_status >= 500:
                            self._handle_retryable_http_error(
                                dataset=dataset,
                                symbol=symbol,
                                exc=exc,
                                attempt=attempt,
                                retry_attempts=retry_attempts,
                                backoff=retry_backoff,
                            )
                            continue
                        raise SymbologyResolutionError(str(exc)) from exc
                    except BentoClientError as exc:  # pragma: no cover - remote failure
                        raise SymbologyResolutionError(str(exc)) from exc
                if payload is None:
                    raise SymbologyResolutionError(
                        f"Databento symbology retries exhausted for {symbol} on {dataset}",
                    )
            instrument_id: str | None
            extracted_symbols: tuple[str, ...]
            result_map = payload.get("result", {}) if isinstance(payload, Mapping) else {}
            entries = result_map.get(symbol, ()) if isinstance(result_map, Mapping) else ()
            if entries:
                first_entry = entries[0]
                instrument_id = str(first_entry.get("s")) if isinstance(first_entry, Mapping) else None
                extracted_symbols = (symbol,)
            else:
                if isinstance(payload, Mapping) and payload.get("not_found"):
                    raise SymbologyResolutionError(
                        f"Symbol {symbol} not found in dataset {dataset}",
                    )
                raise SymbologyResolutionError(
                    f"Unexpected symbology payload for {symbol} on {dataset}",
                )
            # record instrument id for retrieval by resolve()
            self._instrument_cache[key] = instrument_id
            self._metrics.resolve_success.labels(dataset=dataset).inc()
            return extracted_symbols

        return _inner

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        token = symbol.strip().upper()
        return token.split(".", 1)[0] if token else token

    def _heuristic_candidates(
        self,
        *,
        symbol: str,
        dataset: str,
        alias_candidates: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        variants: list[str] = []
        base = self._normalize_symbol(symbol)
        if base:
            variants.append(base)
        if symbol and symbol not in variants:
            variants.append(symbol)
        if alias_candidates:
            for alias in alias_candidates:
                if alias not in variants:
                    variants.append(alias)
        return tuple(variants)

    def _alias_candidates(self, *, dataset: str, symbol: str) -> tuple[str, ...]:
        alias_map = _ALIAS_OVERRIDES.get(dataset)
        if not alias_map:
            return ()
        base = self._normalize_symbol(symbol)
        candidates = alias_map.get(base) or alias_map.get(symbol)
        if not candidates:
            return ()
        normalized = tuple(token.strip().upper() for token in candidates if token.strip())
        return normalized

    def _attempt_alias_resolution(
        self,
        *,
        dataset: str,
        schema: str | None,
        start: datetime | None,
        end: datetime | None,
        alias_candidates: tuple[str, ...] | None,
        resolver: Callable[[tuple[str, str, str, str, str]], tuple[str, ...]],
    ) -> tuple[str, ...] | None:
        if not alias_candidates:
            return None
        for alias_symbol in alias_candidates:
            cache_key = self._build_cache_key(
                dataset=dataset,
                symbol=alias_symbol,
                schema=schema,
                start=start,
                end=end,
            )
            try:
                return resolver(cache_key)
            except SymbologyResolutionError:
                continue
        return None

    def _deduplicate(self, candidates: Sequence[str]) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for token in candidates:
            token_norm = token.strip().upper()
            if not token_norm or token_norm in seen:
                continue
            seen.add(token_norm)
            ordered.append(token_norm)
        return tuple(ordered)

    def _handle_retryable_http_error(
        self,
        *,
        dataset: str,
        symbol: str,
        exc: BentoHttpError,
        attempt: int,
        retry_attempts: int,
        backoff: float,
    ) -> None:
        if attempt >= retry_attempts:
            raise SymbologyResolutionError(str(exc)) from exc
        status = getattr(exc, "http_status", None)
        self._metrics.retry_total.labels(dataset=dataset, status=str(status or "unknown")).inc()
        delay_seconds = max(backoff, 0.0) * attempt
        logger.warning(
            "symbology.retrying_http_error",
            dataset=dataset,
            symbol=symbol,
            status=status,
            attempt=attempt,
            max_attempts=retry_attempts,
            delay_seconds=delay_seconds,
            exc_info=True,
        )
        if delay_seconds > 0.0:
            time.sleep(delay_seconds)


class _SymbologyMetrics:
    """Prometheus collectors for symbology resolution."""

    def __init__(self) -> None:
        self.resolve_latency = get_histogram(
            "nautilus_ml_symbology_resolve_latency_seconds",
            "Latency of Databento symbology.resolve calls",
            labelnames=("dataset",),
        )
        self.resolve_success = get_counter(
            "nautilus_ml_symbology_resolve_success_total",
            "Total successful symbology resolves",
            labelnames=("dataset",),
        )
        self.resolve_miss = get_counter(
            "nautilus_ml_symbology_resolve_miss_total",
            "Total symbology resolves returning no candidates",
            labelnames=("dataset",),
        )
        self.alias_hits = get_counter(
            "nautilus_ml_symbology_alias_hits_total",
            "Total symbology resolves satisfied via alias overrides",
            labelnames=("dataset",),
        )
        self.retry_total = get_counter(
            "nautilus_ml_symbology_retry_total",
            "Total Databento symbology retries grouped by dataset and HTTP status.",
            labelnames=("dataset", "status"),
        )


__all__ = [
    "DatabentoSymbologyClient",
    "DatabentoSymbologyResolver",
    "SymbolResolution",
    "SymbologyResolutionError",
]
