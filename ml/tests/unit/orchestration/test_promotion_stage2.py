from __future__ import annotations

from pathlib import Path

import json
import numpy as np

from ml.orchestration.promotions import Stage2Config
from ml.orchestration.promotions import run_promotion_stage2


def _write_csv(path: Path, n: int = 100, symbol: str = "SPY") -> None:
    # Minimal dataset tail with required columns
    import pandas as pd

    ts0 = 1_700_000_000_000_000_000  # arbitrary ns
    rows = []
    for i in range(n):
        rows.append(
            {
                "time_index": i,
                "timestamp": ts0 + i * 60_000_000_000,  # minute steps
                "instrument_id": symbol,
            },
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_stage2_returns_engine(tmp_path: Path) -> None:
    out_dir = tmp_path / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # teacher_preds.npz with q_val/y_val_true
    q = np.concatenate([np.zeros(50, dtype=np.float32), np.ones(50, dtype=np.float32)])
    y = np.concatenate([np.zeros(50, dtype=np.float32), np.ones(50, dtype=np.float32)])
    np.savez_compressed(out_dir / "teacher_preds.npz", q_val=q, y_val_true=y)

    # dataset.csv minimal tail
    csv_path = out_dir / "dataset.csv"
    _write_csv(csv_path, n=100, symbol="SPY")

    cfg = Stage2Config(
        out_dir=str(out_dir),
        dataset_csv=str(csv_path),
        data_dir=str(tmp_path),  # not used by this test
        horizon_minutes=1,
        engine_mode="returns",
        cost_bps=0.0,
        model_id_hint=None,
        gates=(),
        auto_promote=False,
        deploy_target=None,
    )
    result = run_promotion_stage2(cfg)
    assert result["status"] in {"passed", "failed", "skipped"}
    if result["status"] != "skipped":
        metrics = result["metrics"]
        assert "sharpe_ratio" in metrics and "max_drawdown" in metrics


def test_stage2_backtest_engine_fallback(tmp_path: Path) -> None:
    out_dir = tmp_path / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Minimal artifacts
    np.savez_compressed(out_dir / "teacher_preds.npz", q_val=np.ones(10, dtype=np.float32), y_val_true=np.ones(10, dtype=np.float32))
    csv_path = out_dir / "dataset.csv"
    _write_csv(csv_path, n=10, symbol="SPY")

    cfg = Stage2Config(
        out_dir=str(out_dir),
        dataset_csv=str(csv_path),
        data_dir=str(tmp_path),
        horizon_minutes=1,
        engine_mode="backtest",
        gates=(),
    )
    result = run_promotion_stage2(cfg)
    # Engine mode may fall back; ensure it still returns a result
    assert result["status"] in {"passed", "failed", "skipped"}
