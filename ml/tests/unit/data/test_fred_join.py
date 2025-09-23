from __future__ import annotations

import pytest
from datetime import datetime
from pathlib import Path
from pytest import MonkeyPatch

from ml._imports import HAS_POLARS, pl


@pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
def test_join_fred_asof_polars_smoke(monkeypatch: MonkeyPatch) -> None:
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
    assert isinstance(out, pl.DataFrame)
    assert out.height == left.height


@pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
def test_join_fred_asof_vintage_selection(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    assert pl is not None
    from ml.data import fred_join as fred_mod

    # Create synthetic vintage release calendar
    vintage_dir = Path(tmp_path)
    series_dir = vintage_dir / "CPI"
    series_dir.mkdir(parents=True, exist_ok=True)
    calendar = pl.DataFrame(
        {
            "series_id": ["CPI", "CPI", "CPI"],
            "observation_ts": [
                datetime(2023, 12, 1),
                datetime(2023, 12, 1),
                datetime(2024, 1, 1),
            ],
            "value": [100.0, 110.0, 200.0],
            "release_ts": [
                datetime(2024, 1, 10),
                datetime(2024, 1, 20),
                datetime(2024, 1, 20),
            ],
            "release_end_ts": [
                datetime(2024, 1, 20),
                datetime(2024, 2, 15),
                datetime(2024, 2, 15),
            ],
        },
    )
    calendar.write_parquet(series_dir / "release_calendar.parquet")

    # Ensure base FRED loader returns empty to exercise pure vintage path
    monkeypatch.setattr(
        fred_mod,
        "_load_fred_ml_pl",
        lambda fred_path=None: pl.DataFrame({"timestamp": [], "series_id": [], "value": []}),
    )

    join_fred_asof = fred_mod.join_fred_asof

    left = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                start=pl.datetime(2024, 1, 5),
                end=pl.datetime(2024, 1, 26),
                interval="10d",
                eager=True,
            ),
        },
    )

    out = join_fred_asof(
        left,
        timestamp_col="timestamp",
        lag_days=0,
        fred_path=None,
        vintage_base_dir=vintage_dir,
    )
    assert isinstance(out, pl.DataFrame)

    values = out.get_column("CPI").to_list()
    assert values[0] is None  # before first release
    assert values[1] == 100.0  # between releases
    assert values[2] == 200.0  # after second release uses latest observation
