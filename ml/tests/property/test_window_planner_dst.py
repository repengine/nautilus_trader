from __future__ import annotations

from datetime import date

from hypothesis import given, strategies as st
from itertools import pairwise

from ml.data.ingest.resume import DatabentoIngestor


@given(
    start_day=st.integers(min_value=7, max_value=10),
)
def test_plan_daily_windows_dst_transition(start_day: int) -> None:
    # Choose a US DST transition period (e.g., 2021-03-14 America/New_York)
    start = date(2021, 3, start_day)
    end = date(2021, 3, start_day + 3)
    windows = DatabentoIngestor.plan_daily_windows(start_date=start, end_date=end, tz="America/New_York")
    # Expect number of windows == days span
    assert len(windows) == (end - start).days
    # Contiguity: next.start == prev.end, and increasing order
    for (s0, e0), (s1, e1) in pairwise(windows):
        assert s1 == e0
        assert s0 < e0 <= s1 < e1
