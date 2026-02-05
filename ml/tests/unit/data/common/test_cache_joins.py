from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from ml.data.common.cache_joins import join_l2_cache_polars
from ml.data.common.cache_joins import join_micro_cache_pandas
from ml.data.common.cache_joins import join_micro_cache_polars


pytestmark = pytest.mark.usefixtures("isolated_prometheus_registry", "mock_tracing_backend")


def _micro_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 2, 0, 1, tzinfo=UTC),
            ],
            "midprice": [1.0, None],
            "spread_bps": [None, 2.5],
        },
    )


def _l2_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 2, 0, 1, tzinfo=UTC),
            ],
            "midprice": [1.0, None],
            "spread_bps": [1.1, None],
        },
    )


def test_join_micro_cache_polars_fills_numeric_nulls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 2, 0, 1, tzinfo=UTC),
            ],
            "close": [1.0, 2.0],
        },
    )

    def fake_get_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        raw_base_dir: Path,
        allow_compute: bool = True,
    ) -> pl.DataFrame:
        _ = (self, symbol, start, end, raw_base_dir)
        assert allow_compute is True
        return _micro_frame()

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("live aggregation should not run for cache_first")

    monkeypatch.setattr(
        "ml.data.micro_cache.MicroMinuteCache.get_range",
        fake_get_range,
    )
    monkeypatch.setattr(
        "ml.features.micro_aggregate.MicrostructureAggregator.compute_for_symbol",
        boom,
    )

    result = join_micro_cache_polars(
        dataset,
        symbol="SPY",
        raw_base_dir=tmp_path,
        cache_dir=tmp_path,
        policy="cache_first",
    )

    assert "midprice" in result.columns
    assert "spread_bps" in result.columns
    assert result["midprice"].null_count() == 0
    assert result["spread_bps"].null_count() == 0


def test_join_micro_cache_polars_adds_columns_when_cache_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 2, 0, 1, tzinfo=UTC),
            ],
            "close": [1.0, 2.0],
        },
    )

    def empty_get_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        raw_base_dir: Path,
        allow_compute: bool = True,
    ) -> pl.DataFrame:
        _ = (self, symbol, start, end, raw_base_dir)
        assert allow_compute is False
        return pl.DataFrame({"timestamp": []})

    monkeypatch.setattr(
        "ml.data.micro_cache.MicroMinuteCache.get_range",
        empty_get_range,
    )

    result = join_micro_cache_polars(
        dataset,
        symbol="SPY",
        raw_base_dir=tmp_path,
        cache_dir=tmp_path,
        policy="cache_only",
    )

    assert "midprice" in result.columns
    assert result["midprice"].to_list() == [0.0, 0.0]


def test_join_l2_cache_polars_fills_numeric_nulls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 2, 0, 1, tzinfo=UTC),
            ],
            "close": [1.0, 2.0],
        },
    )

    def fake_get_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        raw_base_dir: Path,
        allow_compute: bool = True,
    ) -> pl.DataFrame:
        _ = (self, symbol, start, end, raw_base_dir)
        assert allow_compute is True
        return _l2_frame()

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("live aggregation should not run for cache_first")

    monkeypatch.setattr(
        "ml.data.l2_cache.L2MinuteCache.get_range",
        fake_get_range,
    )
    monkeypatch.setattr(
        "ml.features.l2_aggregate.L2Aggregator.compute_for_symbol",
        boom,
    )

    result = join_l2_cache_polars(
        dataset,
        symbol="SPY",
        raw_base_dir=tmp_path,
        cache_dir=tmp_path,
        policy="cache_first",
    )

    assert "midprice" in result.columns
    assert "spread_bps" in result.columns
    assert result["midprice"].null_count() == 0
    assert result["spread_bps"].null_count() == 0


def test_join_l2_cache_polars_adds_columns_when_cache_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 2, 0, 1, tzinfo=UTC),
            ],
            "close": [1.0, 2.0],
        },
    )

    def empty_get_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        raw_base_dir: Path,
        allow_compute: bool = True,
    ) -> pl.DataFrame:
        _ = (self, symbol, start, end, raw_base_dir)
        assert allow_compute is False
        return pl.DataFrame({"timestamp": []})

    monkeypatch.setattr(
        "ml.data.l2_cache.L2MinuteCache.get_range",
        empty_get_range,
    )

    result = join_l2_cache_polars(
        dataset,
        symbol="SPY",
        raw_base_dir=tmp_path,
        cache_dir=tmp_path,
        policy="cache_only",
    )

    assert "midprice" in result.columns
    assert result["midprice"].to_list() == [0.0, 0.0]


def test_join_micro_cache_pandas_live_only_uses_aggregator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    dataset = pd.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 0, 0, tzinfo=UTC),
                datetime(2024, 1, 2, 0, 1, tzinfo=UTC),
            ],
            "close": [1.0, 2.0],
        },
    )

    def fake_compute(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        _ = (self, symbol, start, end)
        return _micro_frame()

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cache should not be used for live_only")

    monkeypatch.setattr(
        "ml.features.micro_aggregate.MicrostructureAggregator.compute_for_symbol",
        fake_compute,
    )
    monkeypatch.setattr(
        "ml.data.micro_cache.MicroMinuteCache.get_range",
        boom,
    )

    result = join_micro_cache_pandas(
        dataset,
        symbol="SPY",
        raw_base_dir=tmp_path,
        cache_dir=tmp_path,
        policy="live_only",
    )

    assert "midprice" in result.columns
    assert "spread_bps" in result.columns
