from __future__ import annotations

from datetime import datetime
from pathlib import Path

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

    def get_series(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> pd.Series:
        raise AssertionError("get_series should not be invoked in this scenario")


class _WindowedStubFred(_StubFred):
    def __init__(self, frame: pd.DataFrame) -> None:
        super().__init__(frame)
        self.calls: list[tuple[str | None, str | None]] = []

    def get_series_all_releases(
        self,
        series_id: str,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> pd.DataFrame:
        self.calls.append((realtime_start, realtime_end))
        return super().get_series_all_releases(
            series_id,
            realtime_start=realtime_start,
            realtime_end=realtime_end,
        )

    def get_series(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> pd.Series:
        raise AssertionError("get_series should not be invoked in this scenario")


class _FallbackFred:
    def __init__(self, fred_series: pd.Series) -> None:
        self._series = fred_series
        self.fallback_calls = 0

    def get_series_all_releases(
        self,
        series_id: str,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> pd.DataFrame:
        raise ValueError("Series does not exist in ALFRED")

    def get_series(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> pd.Series:
        self.fallback_calls += 1
        return self._series


class _FailingFallbackFred(_FallbackFred):
    def __init__(self) -> None:
        super().__init__(fred_series=pd.Series(dtype=float))

    def get_series(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> pd.Series:
        raise ValueError("Bad Request.  The series does not exist.")


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


def test_alfred_loader_persists_vintages(tmp_path: Path) -> None:
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


def test_alfred_loader_windowed_fetch_normalizes_datetimes(tmp_path: Path) -> None:
    cfg = ALFREDConfig(
        series_ids=("CPI",),
        out_dir=tmp_path,
        api_key="dummy",
        start_date="2015-01-01",
        window_days=365,
    )
    stub = _WindowedStubFred(_frame())
    loader = ALFREDDataLoader(cfg, fred_client=stub)

    stats = loader.refresh()

    assert stats["CPI"]["rows"] == 3
    assert stub.calls, "windowed fetch should invoke client at least once"
    for realtime_start, realtime_end in stub.calls:
        assert realtime_start is not None
        assert realtime_end is not None
        assert len(realtime_start) == 10
        assert len(realtime_end) == 10


def test_alfred_loader_falls_back_to_fred_series(tmp_path: Path) -> None:
    fred_series = pd.Series(
        [10.5, 11.25],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
    )
    client = _FallbackFred(fred_series)
    cfg = ALFREDConfig(
        series_ids=("CRBINDX",),
        out_dir=tmp_path,
        api_key="dummy",
        fallback_to_fred_series=("CRBINDX",),
    )
    loader = ALFREDDataLoader(cfg, fred_client=client)

    stats = loader.refresh()

    assert client.fallback_calls == 1
    assert stats["CRBINDX"]["rows"] == 2
    series_dir = tmp_path / "CRBINDX"
    calendar = pl.read_parquet(series_dir / "release_calendar.parquet").sort("release_ts")
    assert calendar.height == 2
    assert calendar["release_ts"].to_list() == calendar["observation_ts"].to_list()
    release_files = sorted(
        p.name for p in series_dir.glob("*.parquet") if p.name != "release_calendar.parquet"
    )
    assert release_files == ["20240102.parquet", "20240103.parquet"]


def test_alfred_loader_fallback_handles_missing_series(tmp_path: Path) -> None:
    client = _FailingFallbackFred()
    cfg = ALFREDConfig(
        series_ids=("GOLDAMGBD228NLBM",),
        out_dir=tmp_path,
        api_key="dummy",
        fallback_to_fred_series=("GOLDAMGBD228NLBM",),
    )
    loader = ALFREDDataLoader(cfg, fred_client=client)

    stats = loader.refresh()

    assert stats["GOLDAMGBD228NLBM"]["rows"] == 0
    calendar = pl.read_parquet(tmp_path / "GOLDAMGBD228NLBM" / "release_calendar.parquet")
    assert calendar.is_empty()
