"""
Helpers for deriving ingestion windows from DataRegistry watermarks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import tzinfo
from typing import Protocol, runtime_checkable

from ml.config.events import Source
from ml.config.ingestion_windows import WatermarkWindowConfig
from ml.registry.watermark import Watermark


@runtime_checkable
class WatermarkRegistryProtocol(Protocol):
    """
    Protocol for registry watermark lookups used by ingestion windows.
    """

    def get_watermark(
        self,
        dataset_id: str,
        instrument_id: str,
        source: Source | str,
    ) -> Watermark | None:
        """
        Return a watermark for the dataset/instrument/source tuple.
        """


@dataclass(frozen=True)
class WatermarkWindowResult:
    """
    Result of watermark window resolution.

    Attributes
    ----------
    start : datetime | None
        Resolved start datetime for ingestion.
    reason : str
        Reason label describing how the start was computed.
    """

    start: datetime | None
    reason: str


def resolve_watermark_start_datetime(
    *,
    registry: WatermarkRegistryProtocol | None,
    dataset_id: str,
    instrument_ids: Sequence[str],
    source: Source,
    end: datetime,
    config: WatermarkWindowConfig,
    start: datetime | None = None,
) -> WatermarkWindowResult:
    """
    Resolve an ingestion start datetime using registry watermarks.

    Args:
        registry: Registry instance providing ``get_watermark`` or ``None``.
        dataset_id: Dataset identifier for watermark lookup.
        instrument_ids: Instrument identifiers to query for watermarks.
        source: Data source enum used in the registry watermark.
        end: End datetime for the requested ingest window.
        config: Window configuration controlling lookback and caps.
        start: Optional baseline start datetime (e.g., CLI-provided).

    Returns:
        WatermarkWindowResult with the resolved start datetime and a reason label.

    Example:
        >>> end = datetime(2024, 6, 1, tzinfo=UTC)
        >>> result = resolve_watermark_start_datetime(
        ...     registry=registry,
        ...     dataset_id="ml.events_calendar",
        ...     instrument_ids=[""],
        ...     source=Source.HISTORICAL,
        ...     end=end,
        ...     config=WatermarkWindowConfig(lookback_days=30, max_window_days=365),
        ... )
        >>> result.start is not None
        True
    """
    if not instrument_ids:
        return WatermarkWindowResult(start=start, reason="no_instruments")

    tzinfo = end.tzinfo
    end_normalized = _normalize_datetime(end, tzinfo=tzinfo)
    resolved_start = _normalize_datetime(start, tzinfo=tzinfo) if start is not None else None
    reason = "baseline"

    if config.use_watermark and registry is not None:
        watermark_ns = _min_watermark_ns(
            registry=registry,
            dataset_id=dataset_id,
            instrument_ids=instrument_ids,
            source=source,
        )
        if watermark_ns is not None:
            watermark_dt = _normalize_datetime(
                datetime.fromtimestamp(watermark_ns / 1_000_000_000, tz=UTC),
                tzinfo=tzinfo,
            )
            candidate = watermark_dt - timedelta(days=config.lookback_days)
            if resolved_start is None or candidate > resolved_start:
                resolved_start = candidate
                reason = "watermark"

    if resolved_start is None and config.fallback_start_days is not None:
        resolved_start = end_normalized - timedelta(days=config.fallback_start_days)
        reason = "fallback"

    if config.max_window_days is not None:
        window_start = end_normalized - timedelta(days=config.max_window_days)
        if resolved_start is None or resolved_start < window_start:
            resolved_start = window_start
            reason = "max_window"

    if resolved_start is not None and resolved_start > end_normalized:
        resolved_start = end_normalized
        reason = "clamped_end"

    return WatermarkWindowResult(start=resolved_start, reason=reason)


def resolve_watermark_start_date(
    *,
    registry: WatermarkRegistryProtocol | None,
    dataset_id: str,
    instrument_ids: Sequence[str],
    source: Source,
    end: date,
    config: WatermarkWindowConfig,
    start: date | None = None,
) -> WatermarkWindowResult:
    """
    Resolve an ingestion start date using registry watermarks.

    Args:
        registry: Registry instance providing ``get_watermark`` or ``None``.
        dataset_id: Dataset identifier for watermark lookup.
        instrument_ids: Instrument identifiers to query for watermarks.
        source: Data source enum used in the registry watermark.
        end: End date for the requested ingest window.
        config: Window configuration controlling lookback and caps.
        start: Optional baseline start date (e.g., CLI-provided).

    Returns:
        WatermarkWindowResult with the resolved start datetime (UTC midnight) and reason.

    Example:
        >>> result = resolve_watermark_start_date(
        ...     registry=registry,
        ...     dataset_id="ml.microstructure_minute",
        ...     instrument_ids=["SPY"],
        ...     source=Source.HISTORICAL,
        ...     end=date(2024, 6, 1),
        ...     config=WatermarkWindowConfig(lookback_days=7, max_window_days=30),
        ... )
        >>> isinstance(result.start, datetime)
        True
    """
    end_dt = datetime(end.year, end.month, end.day, tzinfo=UTC)
    start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC) if start is not None else None
    result = resolve_watermark_start_datetime(
        registry=registry,
        dataset_id=dataset_id,
        instrument_ids=instrument_ids,
        source=source,
        end=end_dt,
        config=config,
        start=start_dt,
    )
    return result


def _min_watermark_ns(
    *,
    registry: WatermarkRegistryProtocol,
    dataset_id: str,
    instrument_ids: Sequence[str],
    source: Source,
) -> int | None:
    watermark_ns: int | None = None
    for instrument_id in instrument_ids:
        try:
            watermark = registry.get_watermark(dataset_id, instrument_id, source)
        except Exception:
            continue
        if watermark is None:
            continue
        try:
            candidate = int(watermark.last_success_ns)
        except (TypeError, ValueError):
            continue
        if watermark_ns is None or candidate < watermark_ns:
            watermark_ns = candidate
    return watermark_ns


def _normalize_datetime(value: datetime, *, tzinfo: tzinfo | None) -> datetime:
    if tzinfo is None:
        return value.replace(tzinfo=None)
    if value.tzinfo is None:
        return value.replace(tzinfo=tzinfo)
    return value.astimezone(tzinfo)


__all__ = [
    "WatermarkRegistryProtocol",
    "WatermarkWindowResult",
    "resolve_watermark_start_date",
    "resolve_watermark_start_datetime",
]
