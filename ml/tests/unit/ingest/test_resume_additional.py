from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from ml.data.ingest.resume import BackoffPolicy
from ml.data.ingest.resume import DatabentoIngestor
from ml.data.ingest.resume import IngestState
from ml.data.ingest.resume import _df_max_ts_ns
from ml.data.ingest.resume import _to_dt_ns


def test_df_max_ts_ns_supports_timestamp_column_shapes() -> None:
    ts_frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T00:01:00Z"],
                utc=True,
            ),
        },
    )
    ts_ns = _df_max_ts_ns(ts_frame)
    assert ts_ns is not None
    assert ts_ns == int(ts_frame["timestamp"].max().value)

    string_frame = pd.DataFrame({"timestamp": ["2025-01-01T00:00:00Z"]})
    converted_ns = _df_max_ts_ns(string_frame)
    assert converted_ns is not None

    invalid_frame = pd.DataFrame({"timestamp": ["bad-value"]})
    assert _df_max_ts_ns(invalid_frame) is None


def test_to_dt_ns_preserves_datetime_objects() -> None:
    dt = datetime(2025, 1, 1, tzinfo=UTC)
    assert _to_dt_ns(dt) is dt
    converted = _to_dt_ns(1_000_000_000)
    assert isinstance(converted, datetime)


@dataclass
class _NeverCalledClient:
    calls: int = 0

    def get_data(
        self,
        dataset: str,
        symbols: list[str],
        schema: str,
        start: str | datetime,
        end: str | datetime,
        **kwargs: Any,
    ) -> pd.DataFrame:
        del dataset, symbols, schema, start, end, kwargs
        self.calls += 1
        return pd.DataFrame()


def test_ingest_window_short_circuits_when_resume_point_exceeds_end() -> None:
    client = _NeverCalledClient()
    state = IngestState(last_ts_ns_by_instrument={"AAPL.XNAS": 20})
    ingestor = DatabentoIngestor(client=client)

    frame = ingestor.ingest_time_window(
        dataset="XNAS.ITCH",
        schema="trades",
        instrument="AAPL.XNAS",
        start_ns=10,
        end_ns=20,
        state=state,
    )

    assert frame.empty
    assert client.calls == 0


@dataclass
class _AlwaysFailingClient:
    calls: int = 0

    def get_data(
        self,
        dataset: str,
        symbols: list[str],
        schema: str,
        start: str | datetime,
        end: str | datetime,
        **kwargs: Any,
    ) -> pd.DataFrame:
        del dataset, symbols, schema, start, end, kwargs
        self.calls += 1
        raise RuntimeError("transient failure")


def test_ingestor_raises_after_max_attempts_and_records_sleep_delays() -> None:
    client = _AlwaysFailingClient()
    observed_delays: list[float] = []
    ingestor = DatabentoIngestor(
        client=client,
        policy=BackoffPolicy(initial_seconds=0.1, max_seconds=0.2, multiplier=2.0, max_attempts=2),
        sleep_fn=lambda delay: observed_delays.append(delay),
    )

    with pytest.raises(RuntimeError, match="transient failure"):
        ingestor.ingest_time_window(
            dataset="XNAS.ITCH",
            schema="trades",
            instrument="AAPL.XNAS",
            start_ns=1,
            end_ns=2,
            state=IngestState(),
        )

    assert client.calls == 2
    assert observed_delays == [0.1]
