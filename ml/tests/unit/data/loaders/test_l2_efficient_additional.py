from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Callable
from typing import cast

import pandas as pd
import pytest

from ml.data.ingest import l2_efficient
from ml.data.ingest.l2_efficient import L2PopulateConfig
from ml.data.ingest.l2_efficient import L2PopulateResult
from ml.data.ingest.l2_efficient import PopulateL2TaskConfig
from ml.data.ingest.service import DatabentoIngestionService


class _StubService:  # pragma: no cover - protocol stub
    pass


def _build_config(tmp_path: Path, **overrides: Any) -> L2PopulateConfig:
    payload: dict[str, Any] = {
        "symbols": ("SPY",),
        "data_dir": tmp_path,
        "progress_file": tmp_path / "progress.json",
        "resume": True,
        "start_date": datetime(2024, 1, 2, 0, 0, 0),
        "end_date": datetime(2024, 1, 3, 0, 0, 0),
        "check_gaps": False,
        "force": False,
        "max_symbols": None,
        "symbol_offset": 0,
        "shuffle": False,
        "rate_limit": 60,
        "dataset": "DBEQ.BASIC",
        "schema": "mbp-10",
        "sleep_between_symbols": 0.0,
    }
    payload.update(overrides)
    return L2PopulateConfig(**payload)


def test_save_progress_delegates_to_helper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[Path, dict[str, list[str]]]] = []

    def _save(path: Path, progress: dict[str, list[str]]) -> None:
        calls.append((path, progress))

    monkeypatch.setattr(l2_efficient, "save_progress_json", _save)
    progress = {"SPY": ["2024-01-02"]}
    l2_efficient._save_progress(tmp_path / "progress.json", progress)

    assert calls == [(tmp_path / "progress.json", progress)]


def test_validate_data_integrity_edge_cases(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.parquet"
    pd.DataFrame({"ts_event": []}).to_parquet(empty_file)
    assert l2_efficient.validate_data_integrity(empty_file, "SPY", datetime(2024, 1, 2, 0, 0, 0)) is False

    none_file = tmp_path / "none.parquet"
    pd.DataFrame({"ts_event": [None, None]}).to_parquet(none_file)
    assert l2_efficient.validate_data_integrity(none_file, "SPY", datetime(2024, 1, 2, 0, 0, 0)) is False

    mismatch_file = tmp_path / "mismatch.parquet"
    pd.DataFrame(
        {
            "ts_event": [
                int(pd.Timestamp("2024-01-02T10:00:00Z").value),
                int(pd.Timestamp("2024-01-03T10:00:00Z").value),
            ],
        },
    ).to_parquet(mismatch_file)
    assert l2_efficient.validate_data_integrity(mismatch_file, "SPY", datetime(2024, 1, 2, 0, 0, 0)) is False

    low_market_file = tmp_path / "low_market.parquet"
    ts_values = [int(pd.Timestamp("2024-01-02T03:00:00Z").value) + idx for idx in range(1_100)]
    pd.DataFrame({"ts_event": ts_values}).to_parquet(low_market_file)
    assert l2_efficient.validate_data_integrity(low_market_file, "SPY", datetime(2024, 1, 2, 0, 0, 0)) is True


def test_detect_data_gaps_handles_empty_final_file(tmp_path: Path) -> None:
    final_file = tmp_path / "SPY_mbp-10.parquet"
    pd.DataFrame({"ts_event": []}).to_parquet(final_file)

    gaps = l2_efficient.detect_data_gaps(
        symbol="SPY",
        output_dir=tmp_path,
        start_date=datetime(2024, 1, 2, 0, 0, 0),
        end_date=datetime(2024, 1, 3, 0, 0, 0),
        schema="mbp-10",
    )

    assert [date.strftime("%Y-%m-%d") for date in gaps] == ["2024-01-02", "2024-01-03"]


def test_merge_new_with_existing_additional_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    l2_efficient.merge_new_with_existing("SPY", tmp_path, schema="mbp-10")

    symbol = "SPY"
    final_file = tmp_path / f"{symbol}_mbp-10.parquet"
    pd.DataFrame({"ts_event": [1], "value": [1.0]}).to_parquet(final_file)
    daily_file = tmp_path / f"{symbol}_mbp10_20240102.parquet"
    pd.DataFrame({"ts_event": [2], "value": [2.0], "extra": [3.0]}).to_parquet(daily_file)

    original_unlink = Path.unlink

    def _unlink_with_error(self: Path, *args: Any, **kwargs: Any) -> None:
        if self.name.endswith("20240102.parquet"):
            raise OSError("cannot unlink")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _unlink_with_error)
    l2_efficient.merge_new_with_existing(symbol, tmp_path, schema="mbp-10")
    assert final_file.exists()


def test_merge_new_with_existing_cleans_tmp_on_replace_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    symbol = "SPY"
    daily_file = tmp_path / f"{symbol}_mbp10_20240102.parquet"
    pd.DataFrame({"ts_event": [1], "value": [1.0]}).to_parquet(daily_file)

    original_replace = Path.replace

    def _replace_fail(self: Path, target: Path) -> Path:
        del target
        raise RuntimeError("replace failed")

    monkeypatch.setattr(Path, "replace", _replace_fail)

    with pytest.raises(RuntimeError, match="replace failed"):
        l2_efficient.merge_new_with_existing(symbol, tmp_path, schema="mbp-10")

    tmp_file = tmp_path / f"{symbol}_mbp-10.tmp.parquet"
    assert not tmp_file.exists()
    monkeypatch.setattr(Path, "replace", original_replace)


def test_populate_l2_data_check_gaps_and_no_download_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    symbol = "SPY"
    final_file = tmp_path / symbol / "l2" / f"{symbol}_mbp-10.parquet"
    final_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ts_event": [1], "value": [1.0]}).to_parquet(final_file)

    monkeypatch.setattr(l2_efficient, "_load_progress", lambda _: {})
    monkeypatch.setattr(l2_efficient, "detect_data_gaps", lambda *args, **kwargs: [])

    config = _build_config(tmp_path, check_gaps=True)
    result = l2_efficient.populate_l2_data(config, service=cast(DatabentoIngestionService, _StubService()))
    assert result.total_records == 0
    assert result.total_size_mb > 0

    final_file.unlink()
    result_no_file = l2_efficient.populate_l2_data(
        config,
        service=cast(DatabentoIngestionService, _StubService()),
    )
    assert result_no_file.total_records == 0


def test_populate_l2_data_force_mode_and_signal_termination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    symbol = "SPY"
    final_file = tmp_path / symbol / "l2" / f"{symbol}_mbp-10.parquet"
    final_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ts_event": [1], "value": [1.0]}).to_parquet(final_file)

    sleep_calls: list[float] = []
    saved_progress: list[dict[str, list[str]]] = []
    handlers: list[Callable[[int, Any], None]] = []

    def _sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    def _save(path: Path, progress: dict[str, list[str]]) -> None:  # noqa: ARG001
        saved_progress.append({key: list(value) for key, value in progress.items()})

    def _signal(_sig: int, handler: Callable[[int, Any], None]) -> None:
        handlers.append(handler)
        if len(handlers) == 2:
            handler(15, None)

    download_calls: list[datetime] = []

    def _download(*args: Any, **kwargs: Any) -> int:
        if "date" in kwargs:
            download_calls.append(cast(datetime, kwargs["date"]))
        else:
            download_calls.append(cast(datetime, args[2]))
        return 1

    def _combine(symbol_name: str, output_dir: Path, *, schema: str) -> None:
        output = output_dir / f"{symbol_name}_mbp-10.parquet"
        pd.DataFrame({"ts_event": [1], "value": [1.0]}).to_parquet(output)

    monkeypatch.setattr(l2_efficient.time, "sleep", _sleep)
    monkeypatch.setattr(l2_efficient, "_save_progress", _save)
    monkeypatch.setattr(l2_efficient.signal, "signal", _signal)
    monkeypatch.setattr(l2_efficient, "download_l2_daily", _download)
    monkeypatch.setattr(l2_efficient, "combine_daily_files", _combine)
    monkeypatch.setattr(l2_efficient, "_load_progress", lambda _: {symbol: ["2024-01-02"]})

    now_values = iter([0.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(l2_efficient.time, "time", lambda: next(now_values, 0.0))

    config = _build_config(
        tmp_path,
        force=True,
        check_gaps=False,
        rate_limit=1,
        sleep_between_symbols=0.5,
    )

    result = l2_efficient.populate_l2_data(config, service=cast(DatabentoIngestionService, _StubService()))
    assert result.total_records >= 0
    assert saved_progress
    assert sleep_calls


def test_populate_l2_efficient_symbol_resolution_and_dates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="No symbols resolved"):
        l2_efficient.populate_l2_efficient(
            PopulateL2TaskConfig(
                data_dir=tmp_path,
                progress_file=tmp_path / "progress.json",
                symbols=("", None),
            ),
        )

    captured: dict[str, object] = {}

    def _populate(config: L2PopulateConfig, *, service: DatabentoIngestionService) -> L2PopulateResult:
        captured["start"] = config.start_date
        captured["end"] = config.end_date
        captured["symbols"] = tuple(config.symbols)
        captured["service"] = service
        return L2PopulateResult(total_records=0, total_size_mb=0.0, symbols_processed=len(config.symbols))

    service = cast(DatabentoIngestionService, _StubService())
    monkeypatch.setattr("ml.data.ingest.api.ensure_service", lambda: service)
    monkeypatch.setattr(l2_efficient, "populate_l2_data", _populate)

    start = datetime(2024, 1, 2, 0, 0, 0)
    end = datetime(2024, 1, 3, 0, 0, 0)

    result = l2_efficient.populate_l2_efficient(
        PopulateL2TaskConfig(
            data_dir=tmp_path,
            progress_file=tmp_path / "progress.json",
            symbols=("spy",),
            start_date=start,
            end_date=end,
        ),
    )

    assert result.symbols_processed == 1
    assert captured["symbols"] == ("SPY",)
    assert captured["start"] == start
    assert captured["end"] == end
    assert captured["service"] is service
