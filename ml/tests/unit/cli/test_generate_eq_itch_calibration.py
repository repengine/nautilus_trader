from __future__ import annotations

from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from pathlib import Path

import pandas as pd

from ml.cli.generate_eq_itch_calibration import main
from ml.data.ingest.calibration import load_calibration_bundle
from ml.data.ingest.service import IngestionChunk
from ml.data.ingest.service import IngestionRequest
from ml.data.ingest.service import IngestionWindow


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)


class _StubIngestionService:
    def __init__(self, frames: dict[tuple[str, str, str], list[pd.DataFrame]]) -> None:
        self._frames = frames

    def ingest(
        self,
        request: IngestionRequest,
        *,
        on_chunk: Callable[[IngestionChunk], None] | None = None,
    ) -> list[object]:
        key = (request.dataset, request.schema, request.symbols[0])
        window = IngestionWindow(start=request.start, end=request.end)
        for frame in self._frames.get(key, []):
            if on_chunk is None:
                continue
            on_chunk(
                IngestionChunk(
                    symbol=request.symbols[0],
                    window=window,
                    frame=frame.copy(deep=True),
                ),
            )
        return []


def test_cli_generates_calibration(monkeypatch, tmp_path: Path) -> None:
    symbol = "INTC"
    start = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    end = datetime(2025, 1, 2, 14, 31, tzinfo=UTC)
    minute = _ns(start.replace(second=0, microsecond=0))
    eq_frame = pd.DataFrame(
        {
            "ts_event": [minute],
            "volume": [400.0],
            "close": [58.0],
        },
    )
    trades_frame = pd.DataFrame(
        {
            "ts_event": [minute, minute + 30_000_000_000],
            "price": [29.0, 29.0],
            "size": [100, 100],
            "sale_condition": ["@", "@"],
        },
    )
    depth_frame = pd.DataFrame(
        {
            "ts_event": [_ns(datetime(2025, 1, 1, 12, 0, tzinfo=UTC))],
            "split_factor": [1.0],
        },
    )
    frames = {
        ("EQUS.MINI", "ohlcv-1m", symbol): [eq_frame],
        ("XNAS.ITCH", "trades", symbol): [trades_frame],
        ("XNAS.ITCH", "mbp-1", symbol): [depth_frame],
    }
    stub_service = _StubIngestionService(frames)

    def _fake_from_env(cls, **_: object) -> _StubIngestionService:  # noqa: D401 - signature matches classmethod
        return stub_service

    monkeypatch.setattr(
        "ml.cli.generate_eq_itch_calibration.DatabentoIngestionService.from_env",
        classmethod(_fake_from_env),
    )

    output_path = tmp_path / "bundle.json"
    monkeypatch.setenv("ML_EQUS_CALIBRATION_PATH", str(output_path))

    exit_code = main(
        [
            "--symbol",
            symbol,
            "--start",
            start.isoformat(),
            "--end",
            end.isoformat(),
            "--min-ratio-minutes",
            "1",
        ],
    )
    assert exit_code == 0
    assert output_path.exists()
    bundle = load_calibration_bundle(output_path)
    calibration = bundle.for_symbol(symbol)
    assert calibration is not None
    assert calibration.sale_condition_allowlist == frozenset({"@"})
    assert calibration.volume_scale_by_minute
