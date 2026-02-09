from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import pandas as pd
import pyarrow.parquet as pq
import pytest

from ml.data.ingest import l2_efficient
from ml.data.ingest.l2_efficient import L2PopulateConfig
from ml.data.ingest.l2_efficient import PopulateL2TaskConfig
from ml.data.ingest.service import DatabentoIngestionService


class _StubService:  # pragma: no cover - behaviourless stub
    pass


@pytest.fixture(autouse=True)
def _no_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ml.data.ingest.l2_efficient.signal.signal",
        lambda *args, **kwargs: None,
    )


def _build_config(tmp_path: Path, **overrides: Any) -> L2PopulateConfig:
    payload: dict[str, Any] = {
        "symbols": ("SPY",),
        "data_dir": tmp_path,
        "progress_file": tmp_path / "progress.json",
        "resume": True,
        "start_date": datetime(2024, 1, 2, 0, 0, 0),
        "end_date": datetime(2024, 1, 2, 0, 0, 0),
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


def _write_parquet(path: Path, ts_events: list[int], values: list[float] | None = None) -> None:
    frame = pd.DataFrame({"ts_event": ts_events, "value": values or [1.0] * len(ts_events)})
    frame.to_parquet(path)


def _ts_ns(date_str: str, hour: int = 10) -> int:
    return int(pd.Timestamp(f"{date_str} {hour:02d}:00:00", tz="UTC").value)


def test_populate_l2_data_returns_zero_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(l2_efficient, "fetch_symbol_data", lambda **_: pd.DataFrame())
    config = _build_config(tmp_path)

    result = l2_efficient.populate_l2_data(
        config,
        service=cast(DatabentoIngestionService, _StubService()),
    )

    assert result.total_records == 0
    assert result.symbols_processed == 1


def test_load_progress_normalizes_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        l2_efficient,
        "load_progress_json",
        lambda _: {"SPY": ["2024-01-02", 1], 101: ["2024-01-03"], "BAD": "value"},
    )

    result = l2_efficient._load_progress(tmp_path / "progress.json")

    assert result == {"SPY": ["2024-01-02"], "101": ["2024-01-03"], "BAD": []}


def test_load_progress_returns_empty_when_source_is_not_dict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(l2_efficient, "load_progress_json", lambda _: ["bad"])

    assert l2_efficient._load_progress(tmp_path / "progress.json") == {}


def test_get_business_dates_skips_weekends() -> None:
    dates = l2_efficient.get_business_dates(
        datetime(2024, 1, 5, 0, 0, 0),
        datetime(2024, 1, 9, 0, 0, 0),
    )

    assert [item.strftime("%Y-%m-%d") for item in dates] == ["2024-01-05", "2024-01-08", "2024-01-09"]


def test_validate_data_integrity_handles_missing_and_small_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"
    assert l2_efficient.validate_data_integrity(missing, "SPY", datetime(2024, 1, 2, 0, 0, 0)) is False

    low_records_file = tmp_path / "SPY_mbp-10.parquet"
    _write_parquet(
        low_records_file,
        ts_events=[_ts_ns("2024-01-02") + idx for idx in range(10)],
    )
    assert (
        l2_efficient.validate_data_integrity(
            low_records_file,
            "SPY",
            datetime(2024, 1, 2, 0, 0, 0),
        )
        is False
    )


def test_validate_data_integrity_handles_reader_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "SPY_mbp-10.parquet"
    file_path.write_bytes(b"placeholder")

    def _boom(path: Path) -> Any:  # noqa: ARG001
        raise ValueError("boom")

    monkeypatch.setattr(l2_efficient.pl, "read_parquet", _boom)

    assert l2_efficient.validate_data_integrity(file_path, "SPY", datetime(2024, 1, 2, 0, 0, 0)) is False


def test_detect_data_gaps_returns_all_dates_when_final_file_missing(tmp_path: Path) -> None:
    gaps = l2_efficient.detect_data_gaps(
        symbol="SPY",
        output_dir=tmp_path,
        start_date=datetime(2024, 1, 2, 0, 0, 0),
        end_date=datetime(2024, 1, 3, 0, 0, 0),
        schema="mbp-10",
    )

    assert [item.strftime("%Y-%m-%d") for item in gaps] == ["2024-01-02", "2024-01-03"]


def test_detect_data_gaps_flags_corrupt_daily_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_file = tmp_path / "SPY_mbp-10.parquet"
    _write_parquet(final_file, ts_events=[_ts_ns("2024-01-02"), _ts_ns("2024-01-03")])
    corrupt_daily = tmp_path / "SPY_mbp10_20240103.parquet"
    _write_parquet(corrupt_daily, ts_events=[_ts_ns("2024-01-03")])

    def _validate(file_path: Path, symbol: str, expected_date: datetime) -> bool:  # noqa: ARG001
        return file_path != corrupt_daily

    monkeypatch.setattr(l2_efficient, "validate_data_integrity", _validate)

    gaps = l2_efficient.detect_data_gaps(
        symbol="SPY",
        output_dir=tmp_path,
        start_date=datetime(2024, 1, 2, 0, 0, 0),
        end_date=datetime(2024, 1, 3, 0, 0, 0),
        schema="mbp-10",
    )

    assert [item.strftime("%Y-%m-%d") for item in gaps] == ["2024-01-03"]


def test_detect_data_gaps_falls_back_on_read_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_file = tmp_path / "SPY_mbp-10.parquet"
    _write_parquet(final_file, ts_events=[_ts_ns("2024-01-02")])

    def _boom(path: Path) -> Any:  # noqa: ARG001
        raise RuntimeError("bad")

    monkeypatch.setattr(l2_efficient.pl, "read_parquet", _boom)

    gaps = l2_efficient.detect_data_gaps(
        symbol="SPY",
        output_dir=tmp_path,
        start_date=datetime(2024, 1, 2, 0, 0, 0),
        end_date=datetime(2024, 1, 3, 0, 0, 0),
        schema="mbp-10",
    )

    assert len(gaps) == 2


def test_merge_new_with_existing_merges_and_cleans_daily_files(tmp_path: Path) -> None:
    symbol = "SPY"
    _write_parquet(tmp_path / f"{symbol}_mbp10_20240102.parquet", ts_events=[_ts_ns("2024-01-02")])
    _write_parquet(tmp_path / f"{symbol}_mbp10_20240103.parquet", ts_events=[_ts_ns("2024-01-03")])

    l2_efficient.merge_new_with_existing(symbol, tmp_path, schema="mbp-10")

    final_file = tmp_path / f"{symbol}_mbp-10.parquet"
    assert final_file.exists()
    assert list(tmp_path.glob(f"{symbol}_mbp10_*.parquet")) == []


def test_merge_new_with_existing_cleans_temp_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symbol = "SPY"
    _write_parquet(tmp_path / f"{symbol}_mbp10_20240102.parquet", ts_events=[_ts_ns("2024-01-02")])

    def _raise(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise ValueError("boom")

    monkeypatch.setattr(l2_efficient.pq, "ParquetFile", _raise)

    with pytest.raises(Exception):
        l2_efficient.merge_new_with_existing(symbol, tmp_path, schema="mbp-10")
    assert not (tmp_path / f"{symbol}_mbp-10.tmp.parquet").exists()


def test_get_tier1_symbols_prefers_progress_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tier1_l1_progress.json").write_text(
        json.dumps({"completed_bbo": ["QQQ", "SPY", "QQQ"]}),
        encoding="utf-8",
    )

    assert l2_efficient.get_tier1_symbols() == ["QQQ", "SPY"]


def test_get_tier1_symbols_falls_back_for_invalid_progress_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tier1_l1_progress.json").write_text("{not-json", encoding="utf-8")

    assert l2_efficient.get_tier1_symbols() == list(l2_efficient.TIER1_CORE)


def test_download_l2_daily_writes_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_frame = pd.DataFrame({"ts_event": [1], "value": [1.0]})
    monkeypatch.setattr(l2_efficient, "fetch_symbol_data", lambda **_: data_frame)
    output_dir = tmp_path
    output_dir.mkdir(parents=True, exist_ok=True)
    records = l2_efficient.download_l2_daily(
        service=cast(DatabentoIngestionService, _StubService()),
        symbol="SPY",
        date=datetime(2024, 1, 2, 0, 0, 0),
        output_dir=output_dir,
        dataset="DBEQ.BASIC",
        schema="mbp-10",
    )
    assert records == 1
    assert any(output_dir.iterdir())


def test_download_l2_daily_skips_weekends(tmp_path: Path) -> None:
    records = l2_efficient.download_l2_daily(
        service=cast(DatabentoIngestionService, _StubService()),
        symbol="SPY",
        date=datetime(2024, 1, 6, 0, 0, 0),
        output_dir=tmp_path,
        dataset="DBEQ.BASIC",
        schema="mbp-10",
    )

    assert records == 0


def test_download_l2_daily_handles_non_transient_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(l2_efficient, "fetch_symbol_data", lambda **_: pd.DataFrame())

    def _retry_impl(_func: Callable[..., Any], **kwargs: Any) -> int:
        on_exception = cast(Callable[[int, BaseException], None], kwargs["on_exception"])
        on_exception(0, RuntimeError("403 forbidden"))
        return 1

    records = l2_efficient.download_l2_daily(
        service=cast(DatabentoIngestionService, _StubService()),
        symbol="SPY",
        date=datetime(2024, 1, 2, 0, 0, 0),
        output_dir=tmp_path,
        dataset="DBEQ.BASIC",
        schema="mbp-10",
        retry_impl=_retry_impl,
    )

    assert records == 0


def test_download_l2_daily_handles_transient_error_then_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(l2_efficient, "fetch_symbol_data", lambda **_: pd.DataFrame({"ts_event": [1, 2]}))

    def _retry_impl(func: Callable[..., Any], **kwargs: Any) -> int:
        on_exception = cast(Callable[[int, BaseException], None], kwargs["on_exception"])
        on_exception(0, RuntimeError("504 timeout"))
        return int(func())

    records = l2_efficient.download_l2_daily(
        service=cast(DatabentoIngestionService, _StubService()),
        symbol="SPY",
        date=datetime(2024, 1, 2, 0, 0, 0),
        output_dir=tmp_path,
        dataset="DBEQ.BASIC",
        schema="mbp-10",
        retry_impl=_retry_impl,
    )

    assert records == 2


def test_validate_daily_file_reports_corruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "broken.parquet"
    target.write_bytes(b"broken")

    def _raise(path: Path) -> Any:  # noqa: ARG001
        raise ValueError("bad")

    monkeypatch.setattr(l2_efficient.pq, "ParquetFile", _raise)

    assert l2_efficient._validate_daily_file(target) is False


def test_stream_merge_daily_files_raises_when_all_files_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daily = tmp_path / "SPY_mbp10_20240102.parquet"
    _write_parquet(daily, ts_events=[_ts_ns("2024-01-02")])
    monkeypatch.setattr(l2_efficient, "_validate_daily_file", lambda _: False)

    with pytest.raises(ValueError, match="No valid daily files"):
        l2_efficient._stream_merge_daily_files([daily], tmp_path / "tmp.parquet")


def test_combine_daily_files_merges_and_cleans(tmp_path: Path) -> None:
    symbol = "SPY"
    _write_parquet(tmp_path / f"{symbol}_mbp10_20240102.parquet", ts_events=[_ts_ns("2024-01-02")])
    _write_parquet(tmp_path / f"{symbol}_mbp10_20240103.parquet", ts_events=[_ts_ns("2024-01-03")])

    l2_efficient.combine_daily_files(symbol, tmp_path, schema="mbp-10")

    final_file = tmp_path / f"{symbol}_mbp-10.parquet"
    assert final_file.exists()
    assert list(tmp_path.glob(f"{symbol}_mbp10_*.parquet")) == []
    parquet = pq.ParquetFile(final_file)
    assert parquet.metadata is not None
    assert int(parquet.metadata.num_rows) == 2


def test_combine_daily_files_removes_tmp_output_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symbol = "SPY"
    _write_parquet(tmp_path / f"{symbol}_mbp10_20240102.parquet", ts_events=[_ts_ns("2024-01-02")])

    def _boom(_daily_files: Sequence[Path], tmp_output: Path) -> None:
        tmp_output.write_bytes(b"tmp")
        raise RuntimeError("merge failed")

    monkeypatch.setattr(l2_efficient, "_stream_merge_daily_files", _boom)

    with pytest.raises(RuntimeError, match="merge failed"):
        l2_efficient.combine_daily_files(symbol, tmp_path, schema="mbp-10")
    assert not (tmp_path / f"{symbol}_mbp-10.tmp.parquet").exists()


def test_populate_l2_data_respects_offset_limit_shuffle_and_resume_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_symbol = "BBB"
    final_file = tmp_path / target_symbol / "l2" / f"{target_symbol}_mbp-10.parquet"
    final_file.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(final_file, ts_events=[_ts_ns("2024-01-02")])

    shuffled: dict[str, bool] = {"called": False}

    def _reverse(symbols: list[str]) -> None:
        shuffled["called"] = True
        symbols.reverse()

    def _unexpected(*args: Any, **kwargs: Any) -> int:  # noqa: ARG001
        raise AssertionError("unexpected download")

    monkeypatch.setattr(l2_efficient, "_shuffle", _reverse)
    monkeypatch.setattr(l2_efficient, "download_l2_daily", _unexpected)
    config = _build_config(
        tmp_path,
        symbols=("AAA", "BBB", "CCC"),
        shuffle=True,
        symbol_offset=1,
        max_symbols=1,
    )

    result = l2_efficient.populate_l2_data(
        config,
        service=cast(DatabentoIngestionService, _StubService()),
    )

    assert shuffled["called"] is True
    assert result.symbols_processed == 1
    assert result.total_records == 0
    assert result.total_size_mb > 0.0


def test_populate_l2_data_check_gaps_downloads_and_merges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symbol = "SPY"
    final_file = tmp_path / symbol / "l2" / f"{symbol}_mbp-10.parquet"
    final_file.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(final_file, ts_events=[_ts_ns("2024-01-02")])

    saved_progress: list[dict[str, list[str]]] = []
    merge_calls: list[str] = []

    def _save(path: Path, progress: dict[str, list[str]]) -> None:  # noqa: ARG001
        saved_progress.append({key: list(value) for key, value in progress.items()})

    def _download(*args: Any, **kwargs: Any) -> int:  # noqa: ARG001
        return 7

    def _merge(merged_symbol: str, output_dir: Path, *, schema: str) -> None:  # noqa: ARG001
        merge_calls.append(f"{merged_symbol}:{schema}")

    def _fail_combine(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise AssertionError("combine should not run")

    monkeypatch.setattr(l2_efficient, "_load_progress", lambda _: {symbol: ["2024-01-02"]})
    monkeypatch.setattr(l2_efficient, "_save_progress", _save)
    monkeypatch.setattr(l2_efficient, "download_l2_daily", _download)
    monkeypatch.setattr(l2_efficient, "merge_new_with_existing", _merge)
    monkeypatch.setattr(l2_efficient, "combine_daily_files", _fail_combine)

    config = _build_config(
        tmp_path,
        check_gaps=True,
        start_date=datetime(2024, 1, 2, 0, 0, 0),
        end_date=datetime(2024, 1, 3, 0, 0, 0),
    )

    result = l2_efficient.populate_l2_data(
        config,
        service=cast(DatabentoIngestionService, _StubService()),
    )

    assert result.total_records == 7
    assert merge_calls == ["SPY:mbp-10"]
    assert saved_progress
    assert "2024-01-03" in saved_progress[-1][symbol]


def test_populate_l2_efficient_resolves_tier_symbols_and_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    service = cast(DatabentoIngestionService, _StubService())

    monkeypatch.setattr(l2_efficient, "get_tier1_symbols", lambda: ["SPY", "QQQ"])
    monkeypatch.setattr("ml.data.ingest.api.ensure_service", lambda: service)

    def _populate(config: L2PopulateConfig, *, service: DatabentoIngestionService) -> l2_efficient.L2PopulateResult:
        captured["symbols"] = tuple(config.symbols)
        captured["dataset"] = config.dataset
        captured["schema"] = config.schema
        captured["service"] = service
        return l2_efficient.L2PopulateResult(
            total_records=1,
            total_size_mb=1.0,
            symbols_processed=2,
        )

    monkeypatch.setattr(l2_efficient, "populate_l2_data", _populate)

    result = l2_efficient.populate_l2_efficient(
        PopulateL2TaskConfig(
            data_dir=tmp_path,
            progress_file=tmp_path / "progress.json",
            tier=1,
            days=2,
            dataset="DBEQ.BASIC",
            schema="mbp-1",
        ),
    )

    assert result.total_records == 1
    assert captured["symbols"] == ("SPY", "QQQ")
    assert captured["dataset"] == "DBEQ.BASIC"
    assert captured["schema"] == "mbp-1"
    assert captured["service"] is service


def test_populate_l2_efficient_requires_symbols_or_tier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Populate L2 requires --symbols or tier=1 configuration"):
        l2_efficient.populate_l2_efficient(
            PopulateL2TaskConfig(
                data_dir=tmp_path,
                progress_file=tmp_path / "progress.json",
            ),
        )
