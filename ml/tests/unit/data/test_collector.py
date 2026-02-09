from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import builtins

import pandas as pd
import pytest

import ml.data.collector as collector_module
from ml.config.base import DataCollectorConfig
from ml.data.collector import DataCollector
from ml.data.collector import _ensure_directory


@dataclass
class _FakeResult:
    frame: pd.DataFrame

    def to_df(self) -> pd.DataFrame:
        return self.frame.copy()


@dataclass
class _FakeTimeseries:
    frames: dict[tuple[str, str], pd.DataFrame]
    failures: set[tuple[str, str]]

    def get_range(
        self,
        *,
        dataset: str,
        symbols: list[str],
        start: Any,
        end: Any,
        schema: str,
        limit: int,
    ) -> _FakeResult:
        del dataset, start, end, limit
        symbol = symbols[0]
        key = (schema, symbol)
        if key in self.failures:
            raise RuntimeError("simulated client failure")
        return _FakeResult(self.frames.get(key, pd.DataFrame()))


@dataclass
class _FakeClient:
    timeseries: _FakeTimeseries


def _install_parquet_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def _to_parquet(
        self: pd.DataFrame,
        path: str | Path,
        *_: Any,
        **__: Any,
    ) -> None:
        Path(path).write_bytes(b"parquet-bytes")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _to_parquet)


def test_ensure_directory_and_storage_helpers(tmp_path: Path) -> None:
    target_dir = _ensure_directory(tmp_path / "new_dir")
    assert target_dir.is_dir()

    file_path = tmp_path / "not_a_dir"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _ensure_directory(file_path)

    collector = DataCollector(storage_limit_gb=1.0, data_dir=tmp_path / "data")
    symbol_dir = collector.data_dir / "AAPL"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    (symbol_dir / "bars.parquet").write_bytes(b"x" * 1024)
    assert collector._get_current_storage_gb() > 0.0

    base = collector._estimate_data_size_gb("ohlcv-1m", ["AAPL"], 10)
    boosted = collector._estimate_data_size_gb("trades", ["AAPL"], 10)
    assert boosted > base


def test_collect_methods_return_early_when_client_missing(tmp_path: Path) -> None:
    collector = DataCollector(storage_limit_gb=1.0, data_dir=tmp_path / "data")
    collector.client = None

    collector.collect_l2_depth(symbols=["AAPL"], days=1)
    collector.collect_l1_trades(symbols=["AAPL"], years=1)
    collector.collect_tbbo_quotes(symbols=["AAPL"], days=1)
    collector.collect_minute_bars(symbols=["AAPL"], days=1)

    assert collector.stats["l2_depth"]["count"] == 0
    assert collector.stats["l1_trades"]["count"] == 0
    assert collector.stats["tbbo_quotes"]["count"] == 0
    assert collector.stats["minute_bars"]["count"] == 0


def test_collect_l2_depth_covers_success_empty_skip_and_error_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_parquet_stub(monkeypatch)
    monkeypatch.setattr(collector_module.time, "sleep", lambda _: None)

    frames = {
        (
            "mbp-1",
            "AAPL",
        ): pd.DataFrame({"bid_px_00": [100.0], "ask_px_00": [100.1]}),
        ("mbp-1", "MSFT"): pd.DataFrame(),
    }
    client = _FakeClient(timeseries=_FakeTimeseries(frames=frames, failures={("mbp-1", "NVDA")}))
    collector = DataCollector(storage_limit_gb=5.0, data_dir=tmp_path / "data")
    collector.client = client
    collector.existing_symbols = ["AAPL", "MSFT", "NVDA", "QQQ"]
    (collector.data_dir / "QQQ").mkdir(parents=True, exist_ok=True)
    (collector.data_dir / "QQQ" / "l2_depth_1d.parquet").write_bytes(b"existing")

    monkeypatch.setattr(collector, "_get_current_storage_gb", lambda: 0.0)
    collector.collect_l2_depth(symbols=["AAPL", "MSFT", "NVDA", "QQQ"], days=1)

    assert collector.stats["l2_depth"]["count"] == 1
    assert collector.stats["l2_depth"]["size_gb"] > 0.0


def test_collect_l1_trades_adjusts_scope_and_collects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_parquet_stub(monkeypatch)
    monkeypatch.setattr(collector_module.time, "sleep", lambda _: None)

    frames = {
        ("trades", "AAPL"): pd.DataFrame({"price": [100.0, 101.0], "size": [10, 11]}),
        ("trades", "MSFT"): pd.DataFrame(),
    }
    client = _FakeClient(timeseries=_FakeTimeseries(frames=frames, failures=set()))
    collector = DataCollector(storage_limit_gb=0.001, data_dir=tmp_path / "data")
    collector.client = client
    monkeypatch.setattr(collector, "_get_current_storage_gb", lambda: 0.0)
    collector.collect_l1_trades(symbols=["AAPL", "MSFT"], years=2)

    assert collector.stats["l1_trades"]["count"] >= 1
    assert collector.stats["l1_trades"]["size_gb"] >= 0.0


def test_collect_tbbo_and_minute_bars_cover_success_and_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_parquet_stub(monkeypatch)
    monkeypatch.setattr(collector_module.time, "sleep", lambda _: None)

    frames = {
        ("tbbo", "AAPL"): pd.DataFrame({"bid_px": [100.0], "ask_px": [100.2]}),
        ("tbbo", "MSFT"): pd.DataFrame(),
        ("ohlcv-1m", "AAPL"): pd.DataFrame({"open": [1.0], "close": [1.1]}),
        ("ohlcv-1m", "MSFT"): pd.DataFrame(),
    }
    client = _FakeClient(
        timeseries=_FakeTimeseries(
            frames=frames,
            failures={("tbbo", "NVDA"), ("ohlcv-1m", "NVDA")},
        ),
    )
    collector = DataCollector(storage_limit_gb=5.0, data_dir=tmp_path / "data")
    collector.client = client
    monkeypatch.setattr(collector, "_get_current_storage_gb", lambda: 0.0)

    collector.collect_tbbo_quotes(symbols=["AAPL", "MSFT", "NVDA"], days=1)
    collector.collect_minute_bars(symbols=["AAPL", "MSFT", "NVDA"], days=1)

    assert collector.stats["tbbo_quotes"]["count"] == 1
    assert collector.stats["minute_bars"]["count"] == 1


def test_run_collection_main_and_summary_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector = DataCollector(storage_limit_gb=5.0, data_dir=tmp_path / "data")
    collector.existing_symbols = ["AAPL", "MSFT"]
    calls: list[str] = []

    monkeypatch.setattr(collector, "collect_l2_depth", lambda *args, **kwargs: calls.append("l2"))
    monkeypatch.setattr(collector, "collect_l1_trades", lambda *args, **kwargs: calls.append("l1"))
    monkeypatch.setattr(collector, "collect_tbbo_quotes", lambda *args, **kwargs: calls.append("tbbo"))
    monkeypatch.setattr(collector, "collect_minute_bars", lambda *args, **kwargs: calls.append("bars"))
    monkeypatch.setattr(collector, "_print_final_summary", lambda: calls.append("summary"))

    storage_points = iter([1.0, 2.0, 3.0, 4.0, 4.2])
    monkeypatch.setattr(collector, "_get_current_storage_gb", lambda: next(storage_points))
    collector.run_collection()
    assert calls == ["l2", "l1", "tbbo", "bars", "l1", "summary"]

    summary_collector = DataCollector(storage_limit_gb=5.0, data_dir=tmp_path / "summary_data")
    summary_collector.existing_symbols = ["AAPL", "MSFT"]
    summary_collector.PRIORITY_SYMBOLS = ["AAPL", "MSFT"]
    summary_collector.stats["l2_depth"]["count"] = 1
    summary_collector.stats["l2_depth"]["size_gb"] = 0.1
    summary_collector.stats["l1_trades"]["count"] = 2
    summary_collector.stats["l1_trades"]["size_gb"] = 0.2
    summary_collector.stats["tbbo_quotes"]["count"] = 3
    summary_collector.stats["tbbo_quotes"]["size_gb"] = 0.3
    summary_collector.stats["minute_bars"]["count"] = 4
    summary_collector.stats["minute_bars"]["size_gb"] = 0.4
    monkeypatch.setattr(summary_collector, "_get_current_storage_gb", lambda: 1.0)
    summary_collector._print_final_summary()
    metadata_file = summary_collector.data_dir / "collection_metadata.json"
    assert metadata_file.exists()

    class _CollectorStub:
        def __init__(self, storage_limit_gb: float) -> None:
            self.storage_limit_gb = storage_limit_gb
            self.ran = False

        def run_collection(self) -> None:
            self.ran = True

    stub = _CollectorStub(storage_limit_gb=1000.0)
    monkeypatch.setattr(collector_module, "DataCollector", lambda storage_limit_gb=1000.0: stub)
    monkeypatch.setattr(collector_module.sys.stdin, "isatty", lambda: False)
    collector_module.main()
    assert stub.ran


def test_main_cancelled_in_tty_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    del tmp_path
    called: list[bool] = []

    class _CollectorStub:
        def run_collection(self) -> None:
            called.append(True)

    monkeypatch.setattr(collector_module, "DataCollector", lambda storage_limit_gb=1000.0: _CollectorStub())
    monkeypatch.setattr(collector_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _: "no")
    collector_module.main()
    assert not called


def test_main_runs_in_tty_mode_when_user_confirms(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []

    class _CollectorStub:
        def run_collection(self) -> None:
            called.append(True)

    monkeypatch.setattr(collector_module, "DataCollector", lambda storage_limit_gb=1000.0: _CollectorStub())
    monkeypatch.setattr(collector_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _: "yes")
    collector_module.main()
    assert called == [True]


def test_collector_handles_end_date_config_and_storage_gates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_parquet_stub(monkeypatch)
    monkeypatch.setattr(collector_module.time, "sleep", lambda _: None)

    existing_dir = tmp_path / "existing_dir"
    existing_dir.mkdir()
    assert _ensure_directory(existing_dir) == existing_dir

    collector = DataCollector(
        data_dir=tmp_path / "collector",
        config=DataCollectorConfig(end_date_iso="invalid-date"),
    )
    assert isinstance(collector.end_date, datetime)

    dated_collector = DataCollector(
        data_dir=tmp_path / "dated",
        config=DataCollectorConfig(end_date_iso="2025-01-01T00:00:00"),
    )
    assert dated_collector.end_date == datetime.fromisoformat("2025-01-01T00:00:00")

    explicit_end = datetime(2025, 1, 10, 0, 0, 0)
    explicit_collector = DataCollector(
        data_dir=tmp_path / "explicit",
        config=DataCollectorConfig(end_date_iso="2024-01-01T00:00:00"),
        end_date=explicit_end,
    )
    assert explicit_collector.end_date == explicit_end

    collector.client = _FakeClient(timeseries=_FakeTimeseries(frames={}, failures=set()))
    monkeypatch.setattr(collector, "_estimate_data_size_gb", lambda schema, symbols, days: 9999.0)
    collector.collect_l2_depth(symbols=["AAPL"], days=1)
    assert collector.stats["l2_depth"]["count"] == 0

    monkeypatch.setattr(collector, "_estimate_data_size_gb", lambda schema, symbols, days: 0.0)
    monkeypatch.setattr(collector, "_get_current_storage_gb", lambda: collector.storage_limit_gb * 0.96)
    collector.collect_l2_depth(symbols=["AAPL"], days=1)
    assert collector.stats["l2_depth"]["count"] == 0


def test_collect_l1_trades_limits_symbol_count_to_ten_when_storage_constrained(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_parquet_stub(monkeypatch)
    monkeypatch.setattr(collector_module.time, "sleep", lambda _: None)

    class _CountingTimeseries(_FakeTimeseries):
        def __init__(self) -> None:
            super().__init__(frames={("trades", f"S{i:02d}"): pd.DataFrame({"price": [1.0], "size": [1]}) for i in range(12)}, failures=set())
            self.requested_symbols: list[str] = []

        def get_range(
            self,
            *,
            dataset: str,
            symbols: list[str],
            start: Any,
            end: Any,
            schema: str,
            limit: int,
        ) -> _FakeResult:
            self.requested_symbols.append(symbols[0])
            return super().get_range(
                dataset=dataset,
                symbols=symbols,
                start=start,
                end=end,
                schema=schema,
                limit=limit,
            )

    timeseries = _CountingTimeseries()
    collector = DataCollector(storage_limit_gb=0.001, data_dir=tmp_path / "trades")
    collector.client = _FakeClient(timeseries=timeseries)
    monkeypatch.setattr(collector, "_estimate_data_size_gb", lambda schema, symbols, days: 9999.0)
    monkeypatch.setattr(collector, "_get_current_storage_gb", lambda: 0.0)

    symbols = [f"S{i:02d}" for i in range(12)]
    collector.collect_l1_trades(symbols=symbols, years=1)

    assert len(set(timeseries.requested_symbols)) == 10


def test_run_collection_skips_phases_when_storage_is_high(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector = DataCollector(storage_limit_gb=5.0, data_dir=tmp_path / "high_storage")
    collector.existing_symbols = ["AAPL", "MSFT"]

    calls: list[str] = []
    monkeypatch.setattr(collector, "collect_l2_depth", lambda *args, **kwargs: calls.append("l2"))
    monkeypatch.setattr(collector, "collect_l1_trades", lambda *args, **kwargs: calls.append("l1"))
    monkeypatch.setattr(collector, "collect_tbbo_quotes", lambda *args, **kwargs: calls.append("tbbo"))
    monkeypatch.setattr(collector, "collect_minute_bars", lambda *args, **kwargs: calls.append("bars"))
    monkeypatch.setattr(collector, "_print_final_summary", lambda: calls.append("summary"))

    monkeypatch.setattr(collector, "_get_current_storage_gb", lambda: 4.9)
    collector.run_collection()

    assert calls == ["l2", "summary"]
