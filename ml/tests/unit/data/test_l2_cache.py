from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from ml.data.l2_cache import L2MinuteCache
from ml.features.l2_aggregate import L2_MINUTE_COLUMNS

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry", "mock_tracing_backend")

def _full_row(timestamp: datetime) -> pl.DataFrame:
    data: dict[str, list[float]] = {"timestamp": [timestamp]}
    for col in L2_MINUTE_COLUMNS:
        if col == "timestamp":
            continue
        data[col] = [0.0]
    return pl.DataFrame(data)

def test_ensure_day_rebuilds_placeholder_partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = L2MinuteCache(tmp_path)
    target_day = date(2024, 1, 2)
    partition = cache.path_for("SPY", target_day)
    partition.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"timestamp": []}).write_parquet(partition)

    calls = {"count": 0}

    def fake_compute(self, symbol: str, start: datetime | None = None, end: datetime | None = None) -> pl.DataFrame:  # type: ignore[override]
        calls["count"] += 1
        return _full_row(datetime(2024, 1, 2, tzinfo=UTC))

    monkeypatch.setattr("ml.data.l2_cache.L2Aggregator.compute_for_symbol", fake_compute)

    cache.ensure_day("SPY", target_day, tmp_path)

    assert calls["count"] == 1
    df = pl.read_parquet(str(partition))
    assert set(df.columns) == set(L2_MINUTE_COLUMNS)
    assert df.height == 1

def test_ensure_day_skips_when_partition_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = L2MinuteCache(tmp_path)
    target_day = date(2024, 1, 3)
    partition = cache.path_for("SPY", target_day)
    partition.parent.mkdir(parents=True, exist_ok=True)
    valid_df = _full_row(datetime(2024, 1, 3, tzinfo=UTC))
    valid_df.write_parquet(partition)

    calls = {"count": 0}

    def fake_compute(self, symbol: str, start: datetime | None = None, end: datetime | None = None) -> pl.DataFrame:  # type: ignore[override]
        calls["count"] += 1
        return valid_df

    monkeypatch.setattr("ml.data.l2_cache.L2Aggregator.compute_for_symbol", fake_compute)

    cache.ensure_day("SPY", target_day, tmp_path)

    assert calls["count"] == 0
