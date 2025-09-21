from __future__ import annotations

from datetime import datetime

import pandas as pd
import polars as pl

from ml.data.loaders import ALFREDConfig
from ml.data.loaders import ALFREDDataLoader


class _StubFred:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def get_series_all_releases(
        self,
        series_id: str,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> pd.DataFrame:
        assert series_id == "CPI"
        return self._frame


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "realtime_start": [
                "2024-01-10",
                "2024-01-20",
                "2024-01-20",
            ],
            "realtime_end": [
                "2024-01-20",
                "2024-02-15",
                "2024-02-15",
            ],
            "date": [
                "2023-12-01",
                "2023-12-01",
                "2024-01-01",
            ],
            "value": [100.0, 110.0, 200.0],
        },
    )


def test_alfred_loader_persists_vintages(tmp_path) -> None:
    cfg = ALFREDConfig(series_ids=("CPI",), out_dir=tmp_path, api_key="dummy")
    loader = ALFREDDataLoader(cfg, fred_client=_StubFred(_frame()))

    stats = loader.refresh()

    assert stats["CPI"]["releases"] == 2
    assert stats["CPI"]["rows"] == 3

    series_dir = tmp_path / "CPI"
    calendar_path = series_dir / "release_calendar.parquet"
    assert calendar_path.exists()

    calendar = pl.read_parquet(calendar_path).sort(["release_ts", "observation_ts"])
    assert calendar.height == 3
    assert set(calendar.get_column("series_id").to_list()) == {"CPI"}

    first_release = pl.read_parquet(series_dir / "20240110.parquet")
    assert first_release.height == 1
    assert first_release["value"].item() == 100.0

    second_release = pl.read_parquet(series_dir / "20240120.parquet").sort("observation_ts")
    assert second_release.height == 2
    assert second_release["value"].to_list() == [110.0, 200.0]

    release_timestamps = [row["release_ts"] for row in calendar.sort("release_ts").to_dicts()]
    assert release_timestamps[0] == datetime(2024, 1, 10)
    assert release_timestamps[-1] == datetime(2024, 1, 20)
