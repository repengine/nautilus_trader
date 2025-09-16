from __future__ import annotations

import pytest

from ml._imports import HAS_POLARS, pl


@pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
def test_join_fred_asof_polars_smoke(monkeypatch) -> None:
    assert pl is not None
    # Left frame: daily timestamps
    left = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 1),
                end=pl.datetime(2024, 1, 3),
                interval="1d",
                eager=True,
            ),
            "price": [1.0, 1.1, 1.2],
        },
    )

    # Create a small FRED ML-format file substitute in-memory by monkeypatching loader if needed.
    # For smoke test, call function and ensure it returns a DataFrame with same row count.
    from ml.data import fred_join as fred_mod

    # Return an empty FRED frame so the join is a no-op
    monkeypatch.setattr(
        fred_mod,
        "_load_fred_ml_pl",
        lambda fred_path=None: pl.DataFrame({"timestamp": [], "series_id": [], "value": []}),
    )

    join_fred_asof = fred_mod.join_fred_asof

    out = join_fred_asof(left, timestamp_col="timestamp", lag_days=1, fred_path=None)
    assert isinstance(out, type(left))
    assert out.height == left.height
