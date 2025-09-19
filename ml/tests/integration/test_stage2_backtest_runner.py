from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ml.orchestration.promotions import Stage2Config
from ml.orchestration.promotions import run_promotion_stage2


@pytest.mark.integration
def test_stage2_backtest_engine_smoke(tmp_path: Path) -> None:
    try:
        from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
        from nautilus_trader.test_kit.providers import TestInstrumentProvider
        from nautilus_trader.model.data import Bar, BarSpecification, BarType
        from nautilus_trader.model.enums import BarAggregation, PriceType
        from nautilus_trader.model.objects import Price, Quantity
    except Exception:
        pytest.skip("Backtest dependencies unavailable")

    # Prepare a tiny catalog with 10 minute bars for one instrument
    catalog_dir = tmp_path / "catalog"
    catalog = ParquetDataCatalog(str(catalog_dir))

    instrument = TestInstrumentProvider.equity(symbol="SPY", venue="XNAS")
    bar_type = BarType(instrument.id, BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST))

    # 10 bars => 10 timestamps
    ts0 = 1_700_000_000_000_000_000
    bars: list[Bar] = []
    close = 100.0
    for i in range(10):
        ts = ts0 + i * 60_000_000_000
        prev = close
        close = prev * (1.0 + (0.001 if i % 2 == 0 else -0.001))
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_double(prev),
                high=Price.from_double(max(prev, close)),
                low=Price.from_double(min(prev, close)),
                close=Price.from_double(close),
                volume=Quantity.from_double(1_000.0),
                ts_event=ts,
                ts_init=ts,
            ),
        )
    catalog.write_data(bars)

    # Build dataset.csv tail aligned to last 10 bars
    import pandas as pd

    ds_path = tmp_path / "dataset.csv"
    df = pd.DataFrame(
        {
            "time_index": list(range(10)),
            "timestamp": [ts0 + i * 60_000_000_000 for i in range(10)],
            "instrument_id": [str(instrument.id)] * 10,
        },
    )
    df.to_csv(ds_path, index=False)

    # Teacher preds: simple alternating longs/shorts
    out_dir = tmp_path
    q = np.asarray([1.0 if i % 2 == 0 else 0.0 for i in range(10)], dtype=np.float32)
    np.savez_compressed(out_dir / "teacher_preds.npz", q_val=q, y_val_true=q)

    cfg = Stage2Config(
        out_dir=str(out_dir),
        dataset_csv=str(ds_path),
        data_dir=str(catalog_dir),
        horizon_minutes=1,
        engine_mode="backtest",
        gates=(),
    )
    result = run_promotion_stage2(cfg)
    status = str(result.get("status", ""))
    # Either the backtest metrics were computed or the runner fell back to returns
    assert status in {"passed", "failed", "skipped"}
    if status != "skipped":
        from typing import Any, cast
        metrics = cast(dict[str, float], result.get("metrics", {}))
        assert "sharpe_ratio" in metrics and "max_drawdown" in metrics
