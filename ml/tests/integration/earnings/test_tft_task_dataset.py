from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ml.data import DatasetBuildConfig
from ml.data import build_tft_dataset
from ml.tests.builders import DataBuilder
from ml.tests.utils.earnings_facade import build_test_data_store


def _fake_bars_to_dataframe(
    catalog: object,
    instrument_ids: list[str],
    start: datetime | None = None,
    end: datetime | None = None,
):
    """Return a deterministic bars DataFrame for testing."""
    del catalog, start, end
    pl = pytest.importorskip("polars")

    base = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    base_ns = int(base.timestamp() * 1_000_000_000)
    ts_ns = DataBuilder.time_series(n_points=5, start_time=base_ns, interval_ns=60_000_000_000)
    timestamps = [datetime.fromtimestamp(value / 1_000_000_000, tz=UTC) for value in ts_ns]
    return pl.DataFrame(
        {
            "instrument_id": [instrument_ids[0]] * len(timestamps),
            "timestamp": timestamps,
            "open": [100.0, 100.1, 100.2, 100.3, 100.4],
            "high": [100.1, 100.2, 100.3, 100.4, 100.5],
            "low": [99.9, 100.0, 100.1, 100.2, 100.3],
            "close": [100.05, 100.15, 100.25, 100.35, 100.45],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        },
    )


def _stub_descriptor_loader() -> object:
    class _DescriptorNamespace:
        def as_mapping(self) -> dict[str, object]:
            return {}

    return _DescriptorNamespace()


def _make_timestamp_ns(value: datetime) -> int:
    value_utc = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return int(value_utc.timestamp() * 1_000_000_000)


@pytest.mark.integration
def test_task_builds_dataset_with_earnings_columns(tmp_path, monkeypatch):
    pl = pytest.importorskip("polars")

    monkeypatch.setattr("ml.data.catalog_utils.bars_to_dataframe", _fake_bars_to_dataframe)
    monkeypatch.setattr("ml.data.tft_dataset_builder.bars_to_dataframe", _fake_bars_to_dataframe)
    monkeypatch.setattr("ml.data.load_market_feed_descriptors", lambda: _stub_descriptor_loader())
    monkeypatch.setattr("ml.data.resolve_market_dataset_bindings", lambda **_: [])

    store = build_test_data_store()

    quarters = [
        ("2023-06-30", "2023-07-20", 1.10, 1.05),
        ("2023-09-30", "2023-10-25", 1.25, 1.18),
        ("2023-12-31", "2024-02-01", 1.37, 1.30),
        ("2024-03-31", "2024-04-25", 1.52, 1.45),
    ]

    for period_end, filing_date, eps_actual, eps_consensus in quarters:
        filing_dt = datetime.fromisoformat(filing_date).replace(tzinfo=UTC)
        store.write_earnings_actual(
            ticker="AAPL",
            period_end=period_end,
            filing_date=filing_date,
            eps_diluted=eps_actual,
            revenue=95_000_000_000.0,
            ts_event=_make_timestamp_ns(filing_dt),
            ts_init=_make_timestamp_ns(filing_dt + timedelta(minutes=1)),
        )
        estimate_dt = filing_dt - timedelta(days=30)
        store.write_earnings_estimate(
            ticker="AAPL",
            estimate_date=estimate_dt.date().isoformat(),
            period_end=period_end,
            eps_consensus=eps_consensus,
            ts_event=_make_timestamp_ns(estimate_dt),
            ts_init=_make_timestamp_ns(estimate_dt + timedelta(minutes=1)),
        )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    cfg = DatasetBuildConfig(
        data_dir=data_dir,
        out_dir=out_dir,
        symbols=["AAPL"],
        include_macro=False,
        include_micro=False,
        include_l2=False,
        include_events=False,
        include_calendar=False,
        include_earnings=True,
        earnings_lag_days=0,
        horizon_minutes=1,
        threshold=0.0,
        lookback_periods=1,
    )

    result = build_tft_dataset(cfg, data_store=store)

    dataset = pl.read_parquet(result.dataset_parquet)
    assert not dataset.is_empty()

    expected_columns = {
        "eps_surprise_q0_AAPL",
        "eps_surprise_pct_q0_AAPL",
        "eps_growth_yoy_AAPL",
        "earnings_beat_streak_AAPL",
        "is_earnings_available",
    }
    assert expected_columns.issubset(set(dataset.columns))
    assert dataset.select(pl.col("is_earnings_available").sum())[0, 0] > 0
