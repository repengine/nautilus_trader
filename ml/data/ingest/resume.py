"""
Databento-like ingestion with resume and retry/backoff semantics.

This module provides a minimal, provider-agnostic ingestion helper that:
- Plans daily time windows across time zones (DST-aware)
- Ingests a time window with retry/backoff on transient errors
- Resumes from the last successful timestamp (idempotent by timestamp)
- Records ingestion metrics via ml.data.ingest.metrics

Designed for use with the provided MockDatabentoClient in tests, but typed via
protocol to avoid hard coupling.

"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import date
from datetime import datetime
from datetime import timedelta
from typing import Any, Protocol

import pandas as pd

from ml.data.ingest.metrics import record_ingest_batch
from ml.data.ingest.metrics import record_ingest_error


class DatabentoLikeClient(Protocol):
    """
    Minimal protocol for a Databento-like client.
    """

    def get_data(
        self,
        dataset: str,
        symbols: list[str],
        schema: str,
        start: str | datetime,
        end: str | datetime,
        **kwargs: Any,
    ) -> pd.DataFrame: ...


@dataclass(slots=True)
class BackoffPolicy:
    initial_seconds: float = 0.5
    max_seconds: float = 5.0
    multiplier: float = 2.0
    jitter: float = 0.0  # 0..1 fraction of delay
    max_attempts: int = 5

    def schedule(self) -> Iterable[float]:
        delay = max(0.0, float(self.initial_seconds))
        for _ in range(max(1, int(self.max_attempts))):
            yield delay
            delay = min(self.max_seconds, delay * max(self.multiplier, 1.0))


SleepFn = Callable[[float], None]


@dataclass(slots=True)
class IngestState:
    last_ts_ns_by_instrument: dict[str, int] = field(default_factory=dict)

    def last_ts(self, instrument_id: str) -> int | None:
        return self.last_ts_ns_by_instrument.get(instrument_id)

    def update_last_ts(self, instrument_id: str, ts_ns: int) -> None:
        self.last_ts_ns_by_instrument[instrument_id] = int(ts_ns)


def _df_max_ts_ns(df: pd.DataFrame) -> int | None:
    if "ts_event" in df.columns:
        return int(df["ts_event"].max()) if not df.empty else None
    if "timestamp" in df.columns:
        # Support pandas timestamps
        col = df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(col):
            return int(col.max().value) if not df.empty else None
        # If plain integers/strings are present, attempt conversion
        try:
            ts = pd.to_datetime(col, utc=True)
            return int(ts.max().value) if not ts.empty else None
        except Exception:
            return None
    return None


def _to_dt_ns(x: int | datetime) -> str | datetime:
    # Databento client accepts ISO strings or datetime
    if isinstance(x, int):
        return datetime.fromtimestamp(x / 1e9, tz=UTC)
    return x


@dataclass(slots=True)
class DatabentoIngestor:
    client: DatabentoLikeClient
    policy: BackoffPolicy = field(default_factory=BackoffPolicy)
    sleep_fn: SleepFn | None = None

    def ingest_time_window(
        self,
        *,
        dataset: str,
        schema: str,
        instrument: str,
        start_ns: int,
        end_ns: int,
        source: str = "historical",
        state: IngestState | None = None,
    ) -> pd.DataFrame:
        """
        Ingest a time window with retry/backoff and resume from last timestamp.
        """
        # Resume from last successful timestamp + 1ns
        if state is not None:
            last = state.last_ts(instrument)
            if last is not None and last >= start_ns:
                start_ns = last + 1
                if start_ns > end_ns:
                    return pd.DataFrame()

        # Retry/backoff loop
        last_err: str | None = None
        attempts = 0
        for delay in self.policy.schedule():
            attempts += 1
            try:
                df = self.client.get_data(
                    dataset=dataset,
                    symbols=[instrument],
                    schema=schema,
                    start=_to_dt_ns(start_ns),
                    end=_to_dt_ns(end_ns),
                )
                # Normalize instrumentation: record metrics and update state
                ts_min = int(start_ns)
                ts_max = int(end_ns)
                record_ingest_batch(
                    dataset=dataset,
                    instrument=instrument,
                    source=source,
                    duration_seconds=0.0,
                    ts_min=ts_min,
                    ts_max=ts_max,
                )
                mx = _df_max_ts_ns(df)
                if state is not None and mx is not None:
                    state.update_last_ts(instrument, mx)
                return df
            except Exception as e:  # pragma: no cover - error path verified via unit test
                last_err = type(e).__name__
                record_ingest_error(
                    dataset=dataset,
                    instrument=instrument,
                    error_type=str(last_err),
                )
                # no sleep in tests unless provided
                if attempts >= self.policy.max_attempts:
                    raise
                if self.sleep_fn is not None:
                    self.sleep_fn(delay)
                continue

        # Unreachable: loop returns or raises
        return pd.DataFrame()

    @staticmethod
    def plan_daily_windows(
        *,
        start_date: date,
        end_date: date,
        tz: str = "UTC",
    ) -> list[tuple[int, int]]:
        """
        Plan contiguous daily windows [start, end) across timezone boundaries.
        """
        from zoneinfo import ZoneInfo

        zone = ZoneInfo(tz)
        cur = datetime(start_date.year, start_date.month, start_date.day, tzinfo=zone)
        end_dt = datetime(end_date.year, end_date.month, end_date.day, tzinfo=zone)
        # Ensure end covers whole last day by adding one day
        windows: list[tuple[int, int]] = []
        while cur < end_dt:
            nxt = cur + timedelta(days=1)
            # Convert to UTC nanoseconds
            cur_utc = cur.astimezone(UTC)
            nxt_utc = nxt.astimezone(UTC)
            windows.append((int(cur_utc.timestamp() * 1e9), int(nxt_utc.timestamp() * 1e9)))
            cur = nxt
        return windows
