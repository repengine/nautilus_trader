from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from ml.config.scheduler_config import DatabentoConfig
from ml.config.scheduler_config import SchedulerConfig
from ml.data.coverage.manager import BucketSpec
from ml.data.coverage.types import DAY_NS
from ml.data.scheduler import DataScheduler

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

class _StubHistorical:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

def _install_databento_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_module = SimpleNamespace(Historical=_StubHistorical)
    monkeypatch.setitem(sys.modules, "databento", stub_module)

def test_run_targeted_update_invokes_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_databento_stub(monkeypatch)
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")
    catalog = MagicMock()
    scheduler = DataScheduler(
        catalog=catalog,
        config=SchedulerConfig(symbols=("AAPL.XNAS",)),
        start_metrics_server=False,
    )

    calls: list[tuple[str, datetime]] = []

    def _fake_collect(
        self: DataScheduler,
        *,
        client: Any,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        target_date: datetime,
        temp_data_dir: Any,
    ) -> bool:
        calls.append((symbol, start_date))
        return True

    monkeypatch.setattr(DataScheduler, "_collect_symbol_data", _fake_collect, raising=False)

    bucket_start = int(datetime(2024, 1, 10, tzinfo=UTC).timestamp() * 1_000_000_000)
    spec = BucketSpec(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="AAPL.XNAS",
        bucket_start_ns=bucket_start,
    )

    scheduler.run_targeted_update([spec, spec])

    assert len(calls) == 1
    assert calls[0][0] == "AAPL.XNAS"
    assert calls[0][1] == datetime(2024, 1, 10, tzinfo=UTC)

def test_run_targeted_update_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_databento_stub(monkeypatch)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    catalog = MagicMock()
    scheduler = DataScheduler(catalog=catalog, start_metrics_server=False)
    spec = BucketSpec(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="AAPL.XNAS",
        bucket_start_ns=int(datetime(2024, 1, 9, tzinfo=UTC).timestamp() * 1_000_000_000),
    )
    with pytest.raises(ValueError):
        scheduler.run_targeted_update([spec])

def test_run_targeted_update_uses_orchestrator_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = MagicMock()
    scheduler = DataScheduler(
        catalog=catalog,
        config=SchedulerConfig(symbols=("AAPL.XNAS",)),
        start_metrics_server=False,
        use_orchestrator=True,
    )

    class _Result:
        persisted_window_count = 1

    orchestrator = MagicMock()
    orchestrator.backfill_gaps.return_value = _Result()
    monkeypatch.setattr(
        DataScheduler,
        "_build_orchestrator",
        lambda self: (orchestrator, 2),
        raising=False,
    )

    bucket = BucketSpec(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="AAPL.XNAS",
        bucket_start_ns=int(datetime(2024, 1, 9, tzinfo=UTC).timestamp() * 1_000_000_000),
    )

    scheduler.run_targeted_update([bucket])

    orchestrator.backfill_gaps.assert_called_once()
    kwargs = orchestrator.backfill_gaps.call_args.kwargs
    assert kwargs["dataset_id"] == "EQUS.MINI"
    assert kwargs["schema"] == "ohlcv-1m"
    assert kwargs["instrument_id"] == "AAPL.XNAS"

def test_run_targeted_update_expands_orchestrator_lookback(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = MagicMock()
    scheduler = DataScheduler(
        catalog=catalog,
        config=SchedulerConfig(symbols=("AAPL.XNAS",)),
        start_metrics_server=False,
        use_orchestrator=True,
    )

    class _Result:
        persisted_window_count = 0

    orchestrator = MagicMock()
    orchestrator.backfill_gaps.return_value = _Result()
    monkeypatch.setattr(
        DataScheduler,
        "_build_orchestrator",
        lambda self: (orchestrator, 1),
        raising=False,
    )

    bucket_start = datetime.now(tz=UTC) - timedelta(days=10)
    bucket = BucketSpec(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_id="AAPL.XNAS",
        bucket_start_ns=int(bucket_start.timestamp() * 1_000_000_000),
    )

    scheduler.run_targeted_update([bucket])

    lookback_days = orchestrator.backfill_gaps.call_args.kwargs["lookback_days"]
    assert lookback_days > 1

def test_collect_symbol_data_handles_timezone_aware_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_databento_stub(monkeypatch)
    monkeypatch.setenv("DATABENTO_API_KEY", "test-key")

    catalog = MagicMock()
    scheduler = DataScheduler(
        catalog=catalog,
        config=SchedulerConfig(
            symbols=("SPY.XNAS",),
            databento=DatabentoConfig(use_temporary_files=True),
        ),
        start_metrics_server=False,
    )
    scheduler._data_registry = None
    scheduler.catalog.write_data = lambda data: None  # type: ignore[assignment]
    monkeypatch.setattr(scheduler, "_ensure_dataset_registered", lambda **_: None)
    monkeypatch.setattr(scheduler, "_load_from_dbn_file", lambda *_, **__: [object()])

    class _Response:
        def to_file(self, path: str) -> None:
            Path(path).write_text("dummy", encoding="utf-8")

    class _Timeseries:
        def get_range(self, **_: Any) -> _Response:
            return _Response()

    client = SimpleNamespace(timeseries=_Timeseries())
    target_date = datetime(2025, 11, 7, tzinfo=UTC)
    start_date = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    result = scheduler._collect_symbol_data(
        client=client,
        symbol="SPY.XNAS",
        start_date=start_date,
        end_date=end_date,
        target_date=target_date,
        temp_data_dir=tmp_path,
    )

    assert result is True

def test_apply_trading_day_padding_weekday(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = MagicMock()
    scheduler = DataScheduler(
        catalog=catalog,
        config=SchedulerConfig(symbols=("AAPL.XNAS",)),
        start_metrics_server=False,
        use_orchestrator=True,
    )

    reference = datetime(2025, 11, 12, tzinfo=UTC)  # Wednesday
    lookback = scheduler._apply_trading_day_padding(
        base_lookback_days=1,
        reference_time=reference,
    )

    assert lookback == 1

def test_apply_trading_day_padding_sunday(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = MagicMock()
    scheduler = DataScheduler(
        catalog=catalog,
        config=SchedulerConfig(symbols=("AAPL.XNAS",)),
        start_metrics_server=False,
        use_orchestrator=True,
    )

    reference = datetime(2025, 11, 9, tzinfo=UTC)  # Sunday
    lookback = scheduler._apply_trading_day_padding(
        base_lookback_days=1,
        reference_time=reference,
    )

    assert lookback == 2

def test_apply_trading_day_padding_monday(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = MagicMock()
    scheduler = DataScheduler(
        catalog=catalog,
        config=SchedulerConfig(symbols=("AAPL.XNAS",)),
        start_metrics_server=False,
        use_orchestrator=True,
    )

    reference = datetime(2025, 11, 10, tzinfo=UTC)  # Monday -> previous trading day Friday
    lookback = scheduler._apply_trading_day_padding(
        base_lookback_days=1,
        reference_time=reference,
    )

    assert lookback == 3

def test_derive_catalog_lookback_days_uses_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = MagicMock()
    scheduler = DataScheduler(
        catalog=catalog,
        config=SchedulerConfig(symbols=("SPY.XNAS", "AAPL.XNAS")),
        start_metrics_server=False,
        use_orchestrator=True,
    )

    reference = datetime(2025, 11, 9, tzinfo=UTC)
    now_bucket = int(int(reference.timestamp() * 1_000_000_000) // DAY_NS)

    class _Provider:
        def read_bucket_coverage(
            self,
            *,
            dataset_id: str,
            schema: str,
            instrument_id: str,
            start_ns: int,
            end_ns: int,
        ) -> set[int]:
            if instrument_id == "SPY.XNAS":
                return {20300, 20400}
            return set()

    monkeypatch.setattr(
        scheduler,
        "_catalog_coverage_provider",
        lambda: _Provider(),
        raising=False,
    )
    monkeypatch.setattr(
        scheduler,
        "_catalog_identifier_for_instrument",
        lambda **kwargs: kwargs["instrument_id"],
        raising=False,
    )

    lookback = scheduler._derive_catalog_lookback_days(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_ids=("SPY.XNAS", "AAPL.XNAS"),
        reference_time=reference,
    )

    assert lookback == now_bucket - 20300

def test_derive_catalog_lookback_days_without_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = MagicMock()
    scheduler = DataScheduler(
        catalog=catalog,
        config=SchedulerConfig(symbols=("SPY.XNAS",)),
        start_metrics_server=False,
        use_orchestrator=True,
    )
    monkeypatch.setattr(scheduler, "_catalog_coverage_provider", lambda: None, raising=False)
    monkeypatch.setattr(
        scheduler,
        "_catalog_identifier_for_instrument",
        lambda **kwargs: kwargs["instrument_id"],
        raising=False,
    )
    lookback = scheduler._derive_catalog_lookback_days(
        dataset_id="EQUS.MINI",
        schema="ohlcv-1m",
        instrument_ids=("SPY.XNAS",),
    )
    assert lookback == 0
