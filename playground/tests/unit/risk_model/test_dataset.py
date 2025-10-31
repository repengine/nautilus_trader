"""Tests for the sector dataset assembly utilities."""
# ruff: noqa: E402

from __future__ import annotations

import importlib.util
import json
import sys
import types
from datetime import UTC
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest


def _should_stub_nautilus_trader() -> bool:
    spec = importlib.util.find_spec("nautilus_trader")
    return spec is None


if "nautilus_trader" not in sys.modules and _should_stub_nautilus_trader():
    nt_module = types.ModuleType("nautilus_trader")
    nt_module.__path__ = []  # type: ignore[attr-defined]
    sys.modules["nautilus_trader"] = nt_module
    persistence_module = types.ModuleType("nautilus_trader.persistence")
    catalog_module = types.ModuleType("nautilus_trader.persistence.catalog")
    parquet_module = types.ModuleType("nautilus_trader.persistence.catalog.parquet")

    class _StubParquetDataCatalog:
        def __init__(self, *args: object, **kwargs: object) -> None:  # pragma: no cover
            self.args = args
            self.kwargs = kwargs

    parquet_module.ParquetDataCatalog = _StubParquetDataCatalog
    sys.modules["nautilus_trader.persistence"] = persistence_module
    sys.modules["nautilus_trader.persistence.catalog"] = catalog_module
    sys.modules["nautilus_trader.persistence.catalog.parquet"] = parquet_module
    nt_module.persistence = persistence_module
    persistence_module.catalog = catalog_module
    catalog_module.parquet = parquet_module

from ml.common.metrics_manager import MetricsManager
from ml.data.ingest.yfinance_adapter import PriceFetcherProtocol
from ml.data.ingest.yfinance_adapter import YFinanceIngestConfig
from playground.risk_model.dataset import FactorDataRequest
from playground.risk_model.dataset import FactorReturnFetcher
from playground.risk_model.dataset import SectorDataRequest
from playground.risk_model.dataset import SectorDatasetAssembler
from playground.risk_model.dataset import SectorReturnFetcher
from playground.risk_model.fetchers import SectorFetcherConfig
from playground.risk_model.fetchers import YFinanceSectorFetcher


class _StaticSectorFetcher(SectorReturnFetcher):
    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    def __call__(self, request: SectorDataRequest) -> pl.DataFrame:
        return self._frame


class _StaticFactorFetcher(FactorReturnFetcher):
    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    def __call__(self, request: FactorDataRequest) -> pl.DataFrame:
        return self._frame


class _StubMetrics(MetricsManager):
    def __init__(self) -> None:
        super().__init__()
        self.observed: list[tuple[str, float, dict[str, object]]] = []
        self.incremented: list[tuple[str, float, dict[str, object]]] = []

    def observe(
        self,
        name: str,
        description: str,
        value: float,
        *,
        labels: dict[str, object] | None = None,
        labelnames: tuple[str, ...] | None = None,
        buckets: tuple[float, ...] | None = None,
    ) -> None:
        del description, labelnames, buckets
        self.observed.append((name, float(value), dict(labels or {})))

    def inc(
        self,
        name: str,
        description: str,
        *,
        labels: dict[str, object] | None = None,
        amount: float = 1.0,
        labelnames: tuple[str, ...] | None = None,
    ) -> None:
        del description, labelnames
        self.incremented.append((name, float(amount), dict(labels or {})))


def test_dataset_assembler_aligns_and_persists(tmp_path: Path) -> None:
    timestamps = [
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 2, tzinfo=UTC),
        datetime(2020, 1, 3, tzinfo=UTC),
    ]
    sector_data = pl.DataFrame(
        {
            "timestamp": timestamps * 2,
            "symbol": ["XLF"] * 3 + ["XLK"] * 3,
            "return": [0.01, 0.015, -0.005, 0.007, 0.01, 0.012],
        },
    )
    factor_data = pl.DataFrame(
        {
            "timestamp": timestamps,
            "factor_duration": [0.0, 0.001, 0.002],
            "factor_credit": [0.002, 0.003, 0.001],
        },
    )

    sector_request = SectorDataRequest(
        sectors=("XLF", "XLK"),
        start=timestamps[0],
        end=timestamps[-1],
    )
    factor_request = FactorDataRequest(
        factor_columns=("factor_duration", "factor_credit"),
        start=timestamps[0],
        end=timestamps[-1],
    )

    metrics = _StubMetrics()
    assembler = SectorDatasetAssembler(
        sector_fetcher=_StaticSectorFetcher(sector_data),
        factor_fetcher=_StaticFactorFetcher(factor_data),
        metrics=metrics,
    )

    dataset = assembler.build(sector_request, factor_request, persist_dir=tmp_path)

    assert dataset.sector_returns.height == 6
    assert dataset.factor_returns.height == 3
    assert set(dataset.factor_returns.columns) == {"timestamp", "factor_duration", "factor_credit"}
    null_flag = (
        dataset.sector_returns
        .select(pl.col("return").is_null().any())
        .to_series()
        .item()
    )
    assert null_flag is False
    assert dataset.coverage.sector_expected_days == 2
    assert dataset.coverage.factor_expected_days == 2
    assert dataset.coverage.calendar_name in {"XNYS", "WEEKDAY"}
    assert dataset.coverage.sector_coverage["XLF"] == pytest.approx(1.0)
    assert dataset.coverage.factor_coverage["factor_duration"] == pytest.approx(1.0)
    assert dataset.coverage.composite_coverage == {}

    sector_path = tmp_path / "sector_returns.parquet"
    factor_path = tmp_path / "factor_returns.parquet"
    coverage_path = tmp_path / "coverage_summary.json"
    assert sector_path.exists()
    assert factor_path.exists()
    assert coverage_path.exists()
    assert sector_path.stat().st_size > 0
    assert factor_path.stat().st_size > 0
    coverage_payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert coverage_payload["sector_expected_days"] == dataset.coverage.sector_expected_days
    assert coverage_payload["factor_expected_days"] == dataset.coverage.factor_expected_days
    assert coverage_payload["composite_coverage"] == {}


def test_dataset_assembler_clips_future_observations() -> None:
    """Assembler should drop any observations past the requested end date."""
    sector_data = pl.DataFrame(
        {
            "timestamp": [
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 2, tzinfo=UTC),
                datetime(2020, 1, 5, tzinfo=UTC),  # Beyond requested end
            ],
            "symbol": ["XLK", "XLK", "XLK"],
            "return": [0.01, 0.02, 0.50],
        }
    )
    factor_data = pl.DataFrame(
        {
            "timestamp": [
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2020, 1, 2, tzinfo=UTC),
                datetime(2020, 1, 7, tzinfo=UTC),
            ],
            "factor_duration": [0.0, 0.0, 10.0],
        }
    )
    stub_metrics = _StubMetrics()
    assembler = SectorDatasetAssembler(
        sector_fetcher=_StaticSectorFetcher(sector_data),
        factor_fetcher=_StaticFactorFetcher(factor_data),
        metrics=stub_metrics,
    )
    end = datetime(2020, 1, 2, tzinfo=UTC)
    dataset = assembler.build(
        SectorDataRequest(
            sectors=("XLK",),
            start=datetime(2020, 1, 1, tzinfo=UTC),
            end=end,
        ),
        FactorDataRequest(
            factor_columns=("factor_duration",),
            start=datetime(2020, 1, 1, tzinfo=UTC),
            end=end,
        ),
    )
    assert dataset.sector_returns.get_column("timestamp").max() <= end
    assert dataset.factor_returns.get_column("timestamp").max() <= end
    trim_entries = [
        entry
        for entry in stub_metrics.observed
        if entry[0] == "playground_dataset_trim_ratio" and entry[2].get("reason") == "future"
    ]
    assert any(entry[2].get("kind") == "sector" for entry in trim_entries)
    assert any(entry[2].get("kind") == "factor" for entry in trim_entries)


def test_sector_fetcher_uses_proxy_with_better_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = [
        datetime(2000, 1, 3, tzinfo=UTC),
        datetime(2000, 1, 4, tzinfo=UTC),
        datetime(2000, 1, 5, tzinfo=UTC),
        datetime(2000, 1, 6, tzinfo=UTC),
        datetime(2000, 1, 7, tzinfo=UTC),
    ]
    price_rows = {
        "timestamp": [timestamps[0], timestamps[1], *timestamps],
        "symbol": ["AAA", "AAA", "BBB", "BBB", "BBB", "BBB", "BBB"],
        "open": [10.0, 10.1, 20.0, 20.2, 20.3, 20.4, 20.5],
        "high": [10.2, 10.3, 20.2, 20.4, 20.5, 20.6, 20.7],
        "low": [9.8, 9.9, 19.8, 20.0, 20.1, 20.2, 20.3],
        "close": [10.1, 10.2, 20.1, 20.3, 20.4, 20.5, 20.6],
        "adj_close": [10.1, 10.2, 20.1, 20.3, 20.4, 20.5, 20.6],
        "volume": [1, 1, 1, 1, 1, 1, 1],
    }
    prices = pl.DataFrame(price_rows)

    def _fake_fetch(
        config: YFinanceIngestConfig,
        fetcher: PriceFetcherProtocol | None = None,
    ) -> pl.DataFrame:
        return prices

    monkeypatch.setattr(
        "playground.risk_model.fetchers.fetch_asset_history",
        _fake_fetch,
    )

    fetcher = YFinanceSectorFetcher(
        config=SectorFetcherConfig(
            ticker_overrides={"AAA": ("AAA", "BBB")},
            min_coverage_ratio=0.9,
        ),
    )
    request = SectorDataRequest(
        sectors=("AAA",),
        start=timestamps[0],
        end=timestamps[-1],
        price_column="close",
    )

    frame = fetcher(request)

    assert frame.select("symbol").unique().to_series().to_list() == ["AAA"]
    assert frame.height == 4


def test_sector_fetcher_adjusts_for_corporate_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Corporate actions should not introduce artificial jumps in returns."""
    timestamps = [
        datetime(2020, 1, 2, tzinfo=UTC),
        datetime(2020, 1, 3, tzinfo=UTC),
    ]
    prices = pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["XLK", "XLK"],
            "open": [100.0, 51.0],
            "high": [101.0, 52.0],
            "low": [99.0, 50.0],
            "close": [100.0, 50.0],  # Split causes close to drop
            "adj_close": [50.0, 50.0],  # Adjusted close remains continuous
            "volume": [1_000, 2_000],
        }
    )
    captured_auto_adjust: dict[str, bool] = {}

    def _fake_fetch(
        config: YFinanceIngestConfig,
        fetcher: PriceFetcherProtocol | None = None,
    ) -> pl.DataFrame:
        captured_auto_adjust["auto_adjust"] = bool(config.auto_adjust)
        return prices

    monkeypatch.setattr(
        "playground.risk_model.fetchers.fetch_asset_history",
        _fake_fetch,
    )

    fetcher = YFinanceSectorFetcher()
    request = SectorDataRequest(
        sectors=("XLK",),
        start=timestamps[0],
        end=timestamps[-1],
    )
    frame = fetcher(request)

    assert captured_auto_adjust["auto_adjust"] is True
    assert frame.height == 1
    assert frame.get_column("return").to_list()[0] == pytest.approx(0.0)
