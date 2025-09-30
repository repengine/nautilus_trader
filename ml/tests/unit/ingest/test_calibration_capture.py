from __future__ import annotations

from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from ml.data.ingest.calibration import load_calibration_bundle
from ml.data.ingest.calibration_capture import CalibrationCaptureConfig
from ml.data.ingest.calibration_capture import CalibrationCaptureService
from ml.data.ingest.service import IngestionChunk
from ml.data.ingest.service import IngestionRequest
from ml.data.ingest.service import IngestionWindow


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)


class _StubIngestionService:
    def __init__(self, frames: dict[tuple[str, str, str], list[pd.DataFrame]]) -> None:
        self._frames = frames
        self.requests: list[IngestionRequest] = []

    def ingest(
        self,
        request: IngestionRequest,
        *,
        on_chunk: Callable[[IngestionChunk], None] | None = None,
    ) -> list[object]:
        self.requests.append(request)
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


def test_capture_generates_expected_calibration(tmp_path: Path) -> None:
    symbol = "INTC"
    start = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    end = datetime(2025, 1, 2, 14, 32, tzinfo=UTC)
    minute_0 = _ns(start.replace(second=0, microsecond=0))
    minute_1 = _ns((start + (end - start) / 2).replace(second=0, microsecond=0))
    eq_frame = pd.DataFrame(
        {
            "ts_event": [minute_0, minute_1],
            "volume": [200.0, 120.0],
            "close": [29.1, 29.225],
        },
    )
    trades_frame = pd.DataFrame(
        {
            "ts_event": [
                minute_0,
                minute_0 + 30_000_000_000,
                minute_0 + 45_000_000_000,
                minute_1,
                minute_1 + 20_000_000_000,
                minute_1 + 40_000_000_000,
            ],
            "price": [29.1, 29.1, 29.3, 29.2, 29.25, 29.3],
            "size": [100, 100, 50, 60, 60, 30],
            "sale_condition": ["@", "A", "Z", "@", "A", "Z"],
        },
    )
    depth_frame = pd.DataFrame(
        {
            "ts_event": [
                _ns(datetime(2025, 1, 1, 12, 0, tzinfo=UTC)),
                _ns(datetime(2025, 1, 2, 13, 0, tzinfo=UTC)),
            ],
            "split_factor": [0.5, None],
            "is_auction": [False, True],
        },
    )
    frames = {
        ("EQUS.MINI", "ohlcv-1m", symbol): [eq_frame],
        ("XNAS.ITCH", "trades", symbol): [trades_frame],
        ("XNAS.ITCH", "mbp-1", symbol): [depth_frame],
    }
    ingestion = _StubIngestionService(frames)
    service = CalibrationCaptureService(ingestion)
    output_path = tmp_path / "calibration.json"
    config = CalibrationCaptureConfig(
        symbols=(symbol,),
        start=start,
        end=end,
        output_path=output_path,
        min_ratio_minutes=1,
    )
    result = service.capture(config)
    assert result.output_path == output_path
    assert output_path.exists()

    bundle = load_calibration_bundle(output_path)
    calibration = bundle.for_symbol(symbol)
    assert calibration is not None
    assert calibration.sale_condition_allowlist == frozenset({"@", "A"})
    assert calibration.volume_scale_by_minute[870] == pytest.approx(1.0, rel=1e-6)
    assert calibration.volume_scale_by_minute[871] == pytest.approx(1.0, rel=1e-6)
    assert calibration.price_scaling_by_minute[870] == pytest.approx(1.0, rel=1e-6)
    assert calibration.price_scaling_by_minute[871] == pytest.approx(1.0, rel=1e-6)
    assert calibration.split_events == {"2025-01-01": 0.5}
    assert calibration.exclude_auction_minutes == frozenset({780})


def test_capture_handles_missing_trades(tmp_path: Path) -> None:
    symbol = "XYZ"
    start = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    end = datetime(2025, 1, 2, 14, 35, tzinfo=UTC)
    eq_frame = pd.DataFrame(
        {
            "ts_event": [_ns(start.replace(second=0, microsecond=0))],
            "volume": [100.0],
            "close": [10.0],
        },
    )
    frames = {
        ("EQUS.MINI", "ohlcv-1m", symbol): [eq_frame],
    }
    ingestion = _StubIngestionService(frames)
    service = CalibrationCaptureService(ingestion)
    config = CalibrationCaptureConfig(
        symbols=(symbol,),
        start=start,
        end=end,
        output_path=None,
        min_ratio_minutes=1,
    )
    result = service.capture(config)
    calibration = result.bundle.for_symbol(symbol)
    assert calibration is not None
    assert not calibration.sale_condition_allowlist
    assert calibration.volume_scale_by_minute == {}
    assert calibration.price_scaling_by_minute == {}
    assert calibration.split_events == {}
    assert calibration.exclude_auction_minutes == frozenset()
