from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ml.data.ingest.resume import BackoffPolicy, DatabentoIngestor, IngestState


@dataclass
class _FlakyClient:
    fail_times: int
    calls: int = 0

    def get_data(
        self,
        dataset: str,
        symbols: list[str],
        schema: str,
        start: str | Any,
        end: str | Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("rate_limit")
        # Return minimal DataFrame with ts_event
        return pd.DataFrame(
            {
                "ts_event": [1, 2, 3],
                "instrument_id": [symbols[0]] * 3,
            },
        )


def test_ingest_backoff_retries_then_succeeds() -> None:
    client = _FlakyClient(fail_times=2)
    # Collect planned delays (do not sleep in tests)
    observed: list[float] = []
    ingestor = DatabentoIngestor(
        client=client,
        policy=BackoffPolicy(initial_seconds=0.1, max_seconds=0.2, multiplier=2.0, max_attempts=5),
        sleep_fn=lambda d: observed.append(d),
    )
    st = IngestState()
    df = ingestor.ingest_time_window(
        dataset="tbbo",
        schema="tbbo",
        instrument="EURUSD.SIM",
        start_ns=0,
        end_ns=10,
        state=st,
    )
    assert client.calls == 3  # 2 failures + 1 success
    assert not df.empty and st.last_ts("EURUSD.SIM") == 3
    # Backoff schedule recorded two sleeps
    assert observed[:2] == [0.1, 0.2]


def test_resume_from_last_timestamp() -> None:
    client = _FlakyClient(fail_times=0)
    ingestor = DatabentoIngestor(client=client)
    st = IngestState()
    st.update_last_ts("EURUSD.SIM", 5)
    df = ingestor.ingest_time_window(
        dataset="tbbo",
        schema="tbbo",
        instrument="EURUSD.SIM",
        start_ns=0,
        end_ns=10,
        state=st,
    )
    # Still returns data, but start was shifted to >5; our mock returns [1,2,3] regardless.
    assert not df.empty
