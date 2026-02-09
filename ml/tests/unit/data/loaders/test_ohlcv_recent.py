from __future__ import annotations

import json
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import cast

import pandas as pd
import polars as pl
import pytest

from ml.data.ingest.policy import DatabentoCoveragePolicy
from ml.data.ingest.service import DatabentoIngestionService
from ml.data.loaders import ohlcv_recent as ohlcv
from ml.data.loaders.ohlcv_recent import OhlcvRecentBackfillConfig
from ml.data.loaders.ohlcv_recent import OhlcvRecentBackfillResult
from ml.data.loaders.ohlcv_recent import SymbolBackfillSummary
from ml.data.loaders.ohlcv_recent import SymbolBackfillStatus
from ml.data.loaders.ohlcv_recent import backfill_recent_ohlcv


class _StubService:  # pragma: no cover - behaviour-less stub
    """
    Minimal stub standing in for DatabentoIngestionService.
    """


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01T00:00:00")],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [100],
        },
    )


def test_backfill_recent_creates_parquet(tmp_path: Path) -> None:
    symbol_dir = tmp_path / "SPY"
    symbol_dir.mkdir()

    config = OhlcvRecentBackfillConfig(
        data_dir=tmp_path,
        symbols=None,
        tier=None,
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 2, 0, 0, 0),
        lookback_days=1,
    )
    policy = DatabentoCoveragePolicy()

    result = backfill_recent_ohlcv(
        config,
        service=cast(DatabentoIngestionService, _StubService()),
        policy=policy,
        fetch_fn=lambda **_: _sample_frame(),
    )

    assert len(result.summaries) == 1
    summary = result.summaries[0]
    assert summary.symbol == "SPY"
    assert summary.status.value == SymbolBackfillStatus.SUCCESS.value
    parquet_path = tmp_path / "SPY" / "l0" / "SPY_ohlcv.parquet"
    assert parquet_path.exists()


def test_backfill_recent_skips_disallowed_symbol(tmp_path: Path) -> None:
    config = OhlcvRecentBackfillConfig(
        data_dir=tmp_path,
        symbols=("SPY",),
        tier=None,
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 1, 0, 0),
    )
    policy = DatabentoCoveragePolicy(allowed_symbols={"AAPL"})

    result = backfill_recent_ohlcv(
        config,
        service=cast(DatabentoIngestionService, _StubService()),
        policy=policy,
        fetch_fn=lambda **_: _sample_frame(),
    )

    assert len(result.summaries) == 1
    summary = result.summaries[0]
    assert summary.status.value == SymbolBackfillStatus.SKIPPED.value


def test_backfill_recent_resolves_service_when_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = OhlcvRecentBackfillConfig(
        data_dir=tmp_path,
        symbols=("SPY",),
        tier=None,
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 2, 0, 0, 0),
        lookback_days=1,
    )
    policy = DatabentoCoveragePolicy()
    service_stub = cast(DatabentoIngestionService, _StubService())

    monkeypatch.setattr("ml.data.ingest.api.ensure_service", lambda: service_stub)
    result = backfill_recent_ohlcv(
        config,
        policy=policy,
        fetch_fn=lambda **_: _sample_frame(),
    )

    assert len(result.summaries) == 1
    assert result.summaries[0].status.value == SymbolBackfillStatus.SUCCESS.value


def test_backfill_result_properties_partition_symbols() -> None:
    result = OhlcvRecentBackfillResult(
        summaries=(
            SymbolBackfillSummary(
                symbol="SPY",
                status=SymbolBackfillStatus.SUCCESS,
                requested_start=None,
                requested_end=None,
            ),
            SymbolBackfillSummary(
                symbol="QQQ",
                status=SymbolBackfillStatus.SKIPPED,
                requested_start=None,
                requested_end=None,
            ),
            SymbolBackfillSummary(
                symbol="IWM",
                status=SymbolBackfillStatus.EMPTY,
                requested_start=None,
                requested_end=None,
            ),
        ),
        dataset="EQUS.MINI",
        schema="ohlcv-1m",
    )

    assert result.successful_symbols == ("SPY",)
    assert result.skipped_symbols == ("QQQ",)


def test_symbols_from_universe_file_parses_direct_and_nested_payloads(tmp_path: Path) -> None:
    direct_path = tmp_path / "direct.json"
    direct_path.write_text(json.dumps({"symbols": ["spy", "AAPL"]}), encoding="utf-8")
    assert ohlcv._symbols_from_universe_file(direct_path) == ["AAPL", "SPY"]

    nested_path = tmp_path / "nested.json"
    nested_path.write_text(
        json.dumps(
            {
                "tier1": [{"symbol": "qqq"}, {"symbol": "spy"}],
                "tier2": [{"symbol": "iwm"}],
            },
        ),
        encoding="utf-8",
    )
    assert ohlcv._symbols_from_universe_file(nested_path) == ["IWM", "QQQ", "SPY"]


def test_resolve_symbols_uses_tier_universe_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universe_path = tmp_path / "universe_tier1.json"
    universe_path.write_text(json.dumps({"symbols": ["spy", "aapl"]}), encoding="utf-8")
    monkeypatch.setattr(ohlcv, "Path", lambda _raw: universe_path)

    resolved = ohlcv._resolve_symbols(
        OhlcvRecentBackfillConfig(
            data_dir=tmp_path / "missing",
            symbols=None,
            tier=1,
        ),
    )
    assert resolved == ("AAPL", "SPY")


def test_last_bar_timestamp_reads_l0_alias_column(tmp_path: Path) -> None:
    symbol = "SPY"
    l0_path = tmp_path / symbol / "l0" / f"{symbol}_ohlcv.parquet"
    l0_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "ts_event": [
                datetime(2024, 1, 1, 0, 0, 0),
                datetime(2024, 1, 1, 0, 5, 0),
            ],
        },
    ).write_parquet(l0_path)

    with pytest.raises(TypeError, match="Unsupported timestamp type"):
        ohlcv._last_bar_timestamp(tmp_path, symbol)


def test_last_bar_timestamp_handles_int_and_invalid_timestamp_types(tmp_path: Path) -> None:
    symbol = "SPY"
    fallback = tmp_path / symbol / "ohlcv-1m_recent.parquet"
    fallback.parent.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({"timestamp": [1_700_000_000]}).write_parquet(fallback)
    parsed = ohlcv._last_bar_timestamp(tmp_path, symbol)
    assert parsed == datetime.fromtimestamp(1_700_000_000.0)

    pl.DataFrame({"timestamp": ["not-a-time"]}).write_parquet(fallback)
    with pytest.raises(TypeError, match="Unsupported timestamp type"):
        ohlcv._last_bar_timestamp(tmp_path, symbol)


def test_merge_save_handles_alias_columns_and_existing_rows(tmp_path: Path) -> None:
    symbol = "SPY"
    out_path = tmp_path / symbol / "l0" / f"{symbol}_ohlcv.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "timestamp": pl.Series(
                "timestamp",
                [datetime(2024, 1, 1, 0, 0, 0)],
                dtype=pl.Datetime("ns"),
            ),
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [100.0],
        },
    ).write_parquet(out_path)

    incoming = pd.DataFrame(
        {
            "time": [datetime(2024, 1, 1, 0, 1, 0), datetime(2024, 1, 1, 0, 1, 0)],
            "open": [1.1, 1.1],
            "high": [2.1, 2.1],
            "low": [0.6, 0.6],
            "close": [1.6, 1.6],
            "volume": [101.0, 101.0],
        },
    )

    ohlcv._merge_save(tmp_path, symbol, incoming)
    merged = pl.read_parquet(out_path)

    assert merged.height == 2
    timestamps = set(merged.get_column("timestamp").to_list())
    assert timestamps == {
        datetime(2024, 1, 1, 0, 0, 0),
        datetime(2024, 1, 1, 0, 1, 0),
    }


def test_merge_save_falls_back_to_new_frame_on_read_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symbol = "SPY"
    out_path = tmp_path / symbol / "l0" / f"{symbol}_ohlcv.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("not-parquet", encoding="utf-8")

    original_read_parquet = ohlcv.PL.read_parquet

    def _patched_read(path: str) -> pl.DataFrame:
        if path == str(out_path):
            raise RuntimeError("broken existing parquet")
        return cast(pl.DataFrame, original_read_parquet(path))

    monkeypatch.setattr(ohlcv.PL, "read_parquet", _patched_read)

    incoming = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 0, 2, 0)],
            "open": [1.2],
            "high": [2.2],
            "low": [0.7],
            "close": [1.7],
            "volume": [102.0],
        },
    )
    ohlcv._merge_save(tmp_path, symbol, incoming)
    merged = pl.read_parquet(out_path)
    assert merged.height == 1


def test_backfill_recent_no_symbols_returns_empty_summary(tmp_path: Path) -> None:
    config = OhlcvRecentBackfillConfig(
        data_dir=tmp_path / "missing",
        symbols=None,
        tier=None,
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 1, 0, 0),
    )
    result = backfill_recent_ohlcv(
        config,
        service=cast(DatabentoIngestionService, _StubService()),
        policy=DatabentoCoveragePolicy(),
        fetch_fn=lambda **_: _sample_frame(),
    )
    assert result.summaries == tuple()


def test_backfill_recent_uses_last_bar_and_skips_empty_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = OhlcvRecentBackfillConfig(
        data_dir=tmp_path,
        symbols=("SPY",),
        tier=None,
        start=None,
        end=datetime(2024, 1, 1, 0, 0, 0),
        lookback_days=1,
    )
    monkeypatch.setattr(
        ohlcv,
        "_last_bar_timestamp",
        lambda _base, _symbol: datetime(2024, 1, 1, 0, 0, 0),
    )
    policy = DatabentoCoveragePolicy()

    result = backfill_recent_ohlcv(
        config,
        service=cast(DatabentoIngestionService, _StubService()),
        policy=policy,
        fetch_fn=lambda **_: _sample_frame(),
    )

    assert len(result.summaries) == 1
    summary = result.summaries[0]
    assert summary.status is SymbolBackfillStatus.SKIPPED
    assert summary.message == "empty-window"


def test_backfill_recent_marks_empty_when_fetch_returns_empty(tmp_path: Path) -> None:
    config = OhlcvRecentBackfillConfig(
        data_dir=tmp_path,
        symbols=("SPY",),
        tier=None,
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 1, 1, 0, 0),
    )
    policy = DatabentoCoveragePolicy()

    result = backfill_recent_ohlcv(
        config,
        service=cast(DatabentoIngestionService, _StubService()),
        policy=policy,
        fetch_fn=lambda **_: pd.DataFrame(),
    )

    assert len(result.summaries) == 1
    summary = result.summaries[0]
    assert summary.status is SymbolBackfillStatus.EMPTY
    assert summary.message == "no-rows"


def test_backfill_recent_respects_minimum_lookback_days(tmp_path: Path) -> None:
    observed: dict[str, datetime] = {}
    end_dt = datetime(2024, 1, 10, 0, 0, 0)

    config = OhlcvRecentBackfillConfig(
        data_dir=tmp_path,
        symbols=("SPY",),
        tier=None,
        start=None,
        end=end_dt,
        lookback_days=0,
    )
    policy = DatabentoCoveragePolicy()

    def _fetch(**kwargs: object) -> pd.DataFrame:
        observed["start"] = cast(datetime, kwargs["start"])
        observed["end"] = cast(datetime, kwargs["end"])
        return _sample_frame()

    result = backfill_recent_ohlcv(
        config,
        service=cast(DatabentoIngestionService, _StubService()),
        policy=policy,
        fetch_fn=_fetch,
    )

    assert result.summaries[0].status is SymbolBackfillStatus.SUCCESS
    assert observed["end"] == end_dt
    assert observed["start"] == end_dt - timedelta(days=1)
