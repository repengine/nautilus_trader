"""Tests for FRED join validation guards."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ml.data.fred_join import _iter_vintage_series_dirs
from ml.data.fred_join import _load_vintage_release_pl
from ml.data.fred_join import join_fred_asof
from ml.data.vintage import VintagePolicy

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry", "mock_tracing_backend")

def test_iter_vintage_series_dirs_raises_when_directory_missing() -> None:
    """Verify FileNotFoundError raised when vintage directory doesn't exist."""
    nonexistent_dir = Path("/tmp/nonexistent_vintage_dir_test_12345")

    with pytest.raises(FileNotFoundError, match="Vintage directory not found"):
        list(_iter_vintage_series_dirs(nonexistent_dir, None))

def test_iter_vintage_series_dirs_with_valid_directory(tmp_path: Path) -> None:
    """Verify function works correctly with valid directory structure."""
    # Create mock vintage directory structure
    vintage_dir = tmp_path / "vintage"
    vintage_dir.mkdir()

    # Create some series subdirectories
    (vintage_dir / "GDP").mkdir()
    (vintage_dir / "CPI").mkdir()
    (vintage_dir / "not_a_dir.txt").touch()  # File, should be ignored

    result = list(_iter_vintage_series_dirs(vintage_dir, None))

    # Should return only directories, not files
    assert len(result) == 2
    series_ids = {series_id for series_id, _ in result}
    assert series_ids == {"GDP", "CPI"}

def test_iter_vintage_series_dirs_with_filter() -> None:
    """Verify series_filter correctly filters results."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        vintage_dir = Path(tmpdir) / "vintage"
        vintage_dir.mkdir()

        # Create series directories
        (vintage_dir / "GDP").mkdir()
        (vintage_dir / "CPI").mkdir()
        (vintage_dir / "UNRATE").mkdir()

        # Filter to only GDP and CPI
        series_filter = {"GDP", "CPI"}
        result = list(_iter_vintage_series_dirs(vintage_dir, series_filter))

        assert len(result) == 2
        series_ids = {series_id for series_id, _ in result}
        assert series_ids == {"GDP", "CPI"}

def test_load_vintage_release_pl_normalizes_schema(tmp_path: Path) -> None:
    """Ensure mismatched release calendar schemas are normalized before concatenation."""
    pl = pytest.importorskip("polars")
    vintage_dir = tmp_path / "vintage"
    (vintage_dir / "FOO").mkdir(parents=True)
    (vintage_dir / "BAR").mkdir(parents=True)

    foo_calendar = pl.DataFrame(
        {
            "release_ts": [datetime(2024, 1, 15, tzinfo=UTC)],
            "value": [1.0],
            "observation_ts": [datetime(2023, 12, 31, tzinfo=UTC)],
        },
    ).select(["release_ts", "value", "observation_ts"])
    foo_calendar.write_parquet(vintage_dir / "FOO" / "release_calendar.parquet")

    bar_calendar = pl.DataFrame(
        {
            "series_id": ["BAR"],
            "observation_ts": [datetime(2024, 1, 1, tzinfo=UTC)],
            "value": [2.0],
            "release_ts": [datetime(2024, 2, 1, tzinfo=UTC)],
            "release_end_ts": [datetime(2024, 2, 2, tzinfo=UTC)],
        },
    ).select(["series_id", "observation_ts", "value", "release_ts", "release_end_ts"])
    bar_calendar.write_parquet(vintage_dir / "BAR" / "release_calendar.parquet")

    result = _load_vintage_release_pl(vintage_dir, None)

    assert result.columns == [
        "series_id",
        "observation_ts",
        "value",
        "release_ts",
        "release_end_ts",
    ]
    assert set(result.get_column("series_id").to_list()) == {"FOO", "BAR"}
    assert result.filter(pl.col("series_id") == "FOO").select("release_ts").drop_nulls().height == 1

def test_join_fred_asof_real_time_uses_vintages(tmp_path: Path) -> None:
    """Validate REAL_TIME policy leverages normalized release calendars."""
    pl = pytest.importorskip("polars")
    fred_path = tmp_path / "fred.parquet"
    fred_df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
            "series_id": ["CPIA_TEST"],
            "value": [123.45],
        },
    )
    fred_df.write_parquet(fred_path)

    vintage_root = tmp_path / "vintages"
    series_dir = vintage_root / "CPIA_TEST"
    series_dir.mkdir(parents=True)
    release_df = pl.DataFrame(
        {
            "release_ts": [datetime(2024, 1, 10, tzinfo=UTC)],
            "observation_ts": [datetime(2023, 12, 1, tzinfo=UTC)],
            "value": [123.45],
        },
    ).select(["release_ts", "observation_ts", "value"])
    release_df.write_parquet(series_dir / "release_calendar.parquet")

    market = pl.DataFrame({"timestamp": [datetime(2024, 2, 1, tzinfo=UTC)]})
    joined = join_fred_asof(
        market,
        fred_path=fred_path,
        vintage_base_dir=vintage_root,
        series_filter={"CPIA_TEST"},
        vintage_policy=VintagePolicy.REAL_TIME,
    )

    col = "CPIA_TEST__value_vintage_ts"
    assert col in joined.columns
    assert len(joined[col].drop_nulls()) == joined.height
