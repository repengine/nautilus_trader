from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd

from ml.data.ingest.policy import DatabentoCoveragePolicy
from ml.data.ingest.service import DatabentoIngestionService
from ml.data.loaders.ohlcv_recent import OhlcvRecentBackfillConfig
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
    assert summary.status is SymbolBackfillStatus.SUCCESS
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
    assert summary.status is SymbolBackfillStatus.SKIPPED
