from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from ml.data.micro_cache import MicroMinuteCache
from ml.features.micro_aggregate import MICRO_COLUMNS

pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry", "mock_tracing_backend")


def _full_row(timestamp: datetime) -> pl.DataFrame:
    data: dict[str, list[float]] = {"timestamp": [timestamp]}
    for col in MICRO_COLUMNS:
        data[col] = [0.0]
    return pl.DataFrame(data)


def test_ensure_day_rebuilds_invalid_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = MicroMinuteCache(tmp_path)
    target_day = date(2024, 1, 2)
    partition = cache.path_for("SPY", target_day)
    partition.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"timestamp": []}).write_parquet(partition)

    def fake_compute(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        _ = (symbol, start, end)
        return _full_row(datetime(2024, 1, 2, tzinfo=UTC))

    monkeypatch.setattr(
        "ml.data.micro_cache.MicrostructureAggregator.compute_for_symbol",
        fake_compute,
    )

    cache.ensure_day("SPY", target_day, tmp_path)

    df = pl.read_parquet(str(partition))
    assert set(df.columns) >= {"timestamp", *MICRO_COLUMNS}
    assert df.height == 1


def test_ensure_day_skips_empty_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = MicroMinuteCache(tmp_path)
    target_day = date(2024, 1, 3)
    partition = cache.path_for("SPY", target_day)

    def fake_compute(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        _ = (symbol, start, end)
        return pl.DataFrame({"timestamp": []})

    monkeypatch.setattr(
        "ml.data.micro_cache.MicrostructureAggregator.compute_for_symbol",
        fake_compute,
    )

    cache.ensure_day("SPY", target_day, tmp_path)

    assert not partition.exists()
