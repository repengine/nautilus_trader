"""Canonicalization helpers for Databento ingestion outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Final
from zoneinfo import ZoneInfo

import pandas as pd


_EASTERN_TZ: Final[ZoneInfo] = ZoneInfo("America/New_York")


@dataclass(slots=True, frozen=True)
class CanonicalizationStats:
    """Summary of canonicalization adjustments applied to a frame."""

    source_dataset: str
    rows_in: int
    rows_out: int
    rows_trimmed: int
    rows_deduped: int
    timezone: str
    session_start: time
    session_end: time
    aggregation_mode: str | None = None
    scaling_factor: float | None = None
    volume_residual_abs: float | None = None
    volume_residual_rel: float | None = None
    calibration_version: str | None = None


@dataclass(slots=True, frozen=True)
class CanonicalizationResult:
    """Canonicalization output including the transformed frame and stats."""

    frame: pd.DataFrame
    stats: CanonicalizationStats


def canonicalize_equities_minute_bars(
    frame: pd.DataFrame,
    *,
    source_dataset: str,
    symbol: str | None = None,
    instrument_id: str | None = None,
    timezone: ZoneInfo = _EASTERN_TZ,
    session_start: time = time(8, 0),
    session_end: time = time(16, 0),
    publisher_id: int | None = 95,
    rtype: int | None = 33,
    aggregation_mode: str | None = None,
    scaling_factor: float | None = None,
    volume_residual_abs: float | None = None,
    volume_residual_rel: float | None = None,
    calibration_version: str | None = None,
) -> CanonicalizationResult:
    """Canonicalize Databento equities minute bars into EQUS conventions."""
    df = frame.copy(deep=True)
    rows_in = len(df.index)
    if rows_in == 0:
        stats = CanonicalizationStats(
            source_dataset=source_dataset,
            rows_in=0,
            rows_out=0,
            rows_trimmed=0,
            rows_deduped=0,
            timezone=str(timezone),
            session_start=session_start,
            session_end=session_end,
            aggregation_mode=aggregation_mode,
            scaling_factor=scaling_factor,
            volume_residual_abs=volume_residual_abs,
            volume_residual_rel=volume_residual_rel,
            calibration_version=calibration_version,
        )
        return CanonicalizationResult(frame=df, stats=stats)

    if "ts_event" in df.columns:
        raw_events = df["ts_event"]
        if pd.api.types.is_datetime64_any_dtype(raw_events):
            event_series = pd.to_datetime(raw_events, utc=True)
        else:
            event_series = pd.to_datetime(raw_events, unit="ns", utc=True, errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        index = df.index
        if index.tzinfo is None:
            normalized_index = index.tz_localize("UTC")
        else:
            normalized_index = index.tz_convert("UTC")
        event_series = pd.Series(normalized_index, index=df.index)
    else:
        raise ValueError("frame requires ts_event column or datetime index")

    df = df.assign(ts_event=event_series.view("int64"))
    df = df.reset_index(drop=True)
    df["ts_init"] = df.get("ts_init", df["ts_event"])
    local_dt = event_series.dt.tz_convert(timezone)
    weekday_mask = local_dt.dt.dayofweek < 5
    start_mask = local_dt.dt.time >= session_start
    end_mask = local_dt.dt.time < session_end
    session_mask = weekday_mask & start_mask & end_mask
    filtered = df.loc[session_mask].copy()
    rows_after_session = len(filtered.index)
    trimmed = rows_in - rows_after_session
    filtered = filtered.dropna(subset=["ts_event", "open", "high", "low", "close", "volume"])
    filtered = filtered.sort_values("ts_event")
    deduped = filtered.drop_duplicates(subset="ts_event", keep="last")
    deduped = deduped.reset_index(drop=True)
    deduped = deduped.assign(
        open=pd.to_numeric(deduped["open"], errors="coerce").round(4),
        high=pd.to_numeric(deduped["high"], errors="coerce").round(4),
        low=pd.to_numeric(deduped["low"], errors="coerce").round(4),
        close=pd.to_numeric(deduped["close"], errors="coerce").round(4),
        volume=pd.to_numeric(deduped["volume"], errors="coerce").round(),
    )
    deduped = deduped.dropna(subset=["open", "high", "low", "close", "volume"])
    deduped["volume"] = deduped["volume"].astype("int64")
    deduped["ts_event"] = deduped["ts_event"].astype("int64")
    deduped["ts_init"] = deduped["ts_init"].astype("int64")
    if publisher_id is not None:
        deduped["publisher_id"] = int(publisher_id)
    if rtype is not None:
        deduped["rtype"] = int(rtype)
    if symbol is not None:
        deduped["symbol"] = str(symbol).upper()
    if instrument_id is not None:
        deduped["instrument_id"] = str(instrument_id)
    rows_out = len(deduped.index)
    stats = CanonicalizationStats(
        source_dataset=source_dataset,
        rows_in=rows_in,
        rows_out=rows_out,
        rows_trimmed=max(trimmed, 0),
        rows_deduped=max(rows_after_session - rows_out, 0),
        timezone=str(timezone),
        session_start=session_start,
        session_end=session_end,
        aggregation_mode=aggregation_mode,
        scaling_factor=scaling_factor,
        volume_residual_abs=volume_residual_abs,
        volume_residual_rel=volume_residual_rel,
        calibration_version=calibration_version,
    )
    return CanonicalizationResult(frame=deduped, stats=stats)


__all__ = [
    "CanonicalizationResult",
    "CanonicalizationStats",
    "canonicalize_equities_minute_bars",
]
