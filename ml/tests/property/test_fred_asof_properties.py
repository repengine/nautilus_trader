from __future__ import annotations

import polars as pl
import pytest


try:
    from hypothesis import given
    from hypothesis import settings
    from hypothesis import strategies as st
except Exception:  # pragma: no cover
    pytest.skip("hypothesis not available", allow_module_level=True)

from ml.data.fred_join import join_fred_asof


def _timeseries(start: int, n: int, step: int) -> list[int]:
    return [start + i * step for i in range(n)]


@settings(max_examples=25)
@given(
    n_releases=st.integers(min_value=1, max_value=5),
    n_rows=st.integers(min_value=3, max_value=20),
    step=st.integers(min_value=60_000_000_000, max_value=300_000_000_000),  # 1-5 minutes in ns
)
def test_fred_lag_increases_nulls(n_releases: int, n_rows: int, step: int) -> None:
    # Build synthetic FRED (single series) and left timestamps
    base_ns = 1_600_000_000_000_000_000
    fred_ts = _timeseries(base_ns, n_releases, step * 5)
    fred = pl.DataFrame(
        {
            "timestamp": pl.Series(fred_ts).cast(pl.Datetime("ns")),
            "series_id": ["S"] * n_releases,
            "value": list(range(n_releases)),
        }
    )
    left_ts = _timeseries(base_ns, n_rows, step)
    left = pl.DataFrame(
        {"timestamp": pl.Series(left_ts).cast(pl.Datetime("ns")), "x": list(range(n_rows))}
    )

    out0 = join_fred_asof(left, timestamp_col="timestamp", lag_days=0)
    out1 = join_fred_asof(left, timestamp_col="timestamp", lag_days=1)
    s0 = out0.get_column("S") if "S" in out0.columns else pl.Series("S", [None] * n_rows)
    s1 = out1.get_column("S") if "S" in out1.columns else pl.Series("S", [None] * n_rows)
    # Nulls should not decrease with added lag
    assert s1.null_count() >= s0.null_count()
    # Where both present, values should match
    for i in range(n_rows):
        if s0[i] is not None and s1[i] is not None:
            assert s0[i] == s1[i]
