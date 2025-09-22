from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from ml.data.ingest import l2_efficient
from ml.data.ingest.l2_efficient import L2PopulateConfig
from ml.data.ingest.service import DatabentoIngestionService


class _StubService:  # pragma: no cover - behaviourless stub
    pass


@pytest.fixture(autouse=True)
def _no_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ml.data.ingest.l2_efficient.signal.signal",
        lambda *args, **kwargs: None,
    )


def test_populate_l2_data_returns_zero_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(l2_efficient, "fetch_symbol_data", lambda **_: pd.DataFrame())
    config = L2PopulateConfig(
        symbols=("SPY",),
        data_dir=tmp_path,
        progress_file=tmp_path / "progress.json",
        resume=True,
        start_date=datetime(2024, 1, 2, 0, 0, 0),
        end_date=datetime(2024, 1, 2, 0, 0, 0),
        check_gaps=False,
        force=False,
        max_symbols=None,
        symbol_offset=0,
        shuffle=False,
        rate_limit=60,
        dataset="DBEQ.BASIC",
        schema="mbp-10",
        sleep_between_symbols=0.0,
    )

    result = l2_efficient.populate_l2_data(
        config,
        service=cast(DatabentoIngestionService, _StubService()),
    )

    assert result.total_records == 0
    assert result.symbols_processed == 1


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
