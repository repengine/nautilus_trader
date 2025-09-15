from __future__ import annotations

import pytest

from ml._imports import HAS_PANDAS, HAS_POLARS


@pytest.mark.parametrize("use_polars", [True, False])
def test_asof_join_basic(use_polars: bool) -> None:
    if use_polars and not HAS_POLARS:
        pytest.skip("polars not available")
    if not use_polars and not HAS_PANDAS:
        pytest.skip("pandas not available")

    if use_polars:
        import polars as pl

        left = pl.DataFrame({"timestamp": [100, 200, 300], "instrument_id": ["SPY", "SPY", "SPY"], "price": [400.0, 401.0, 402.0]})
        right = pl.DataFrame({"timestamp": [150, 250], "instrument_id": ["SPY", "SPY"], "event": ["earnings", "fed"]})
    else:
        import pandas as pd

        left = pd.DataFrame({"timestamp": [100, 200, 300], "instrument_id": ["SPY", "SPY", "SPY"], "price": [400.0, 401.0, 402.0]})
        right = pd.DataFrame({"timestamp": [150, 250], "instrument_id": ["SPY", "SPY"], "event": ["earnings", "fed"]})

    from ml.preprocessing.joins import asof_join

    joined = asof_join(left, right, on="timestamp", by="instrument_id")
    # Should have same number of rows as left
    if use_polars:
        assert joined.height == 3
    else:
        assert len(joined) == 3


def test_embargo_window_polars_or_pandas() -> None:
    if HAS_POLARS:
        import polars as pl

        df = pl.DataFrame({"ts_event": [100, 200, 300, 400], "price": [1.0, 2.0, 3.0, 4.0]})
        from ml.preprocessing.joins import embargo_window

        out = embargo_window(df, event_timestamps=[250], embargo_before_ns=100, embargo_after_ns=100)
        assert out.select(pl.col("embargo").sum()).item() == 2
    elif HAS_PANDAS:
        import pandas as pd

        df = pd.DataFrame({"ts_event": [100, 200, 300, 400], "price": [1.0, 2.0, 3.0, 4.0]})
        from ml.preprocessing.joins import embargo_window

        out = embargo_window(df, event_timestamps=[250], embargo_before_ns=100, embargo_after_ns=100)
        assert int(out["embargo"].sum()) == 2
    else:
        pytest.skip("no dataframe backend available")

