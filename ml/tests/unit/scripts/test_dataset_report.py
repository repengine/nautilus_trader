from __future__ import annotations

import json
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import polars as pl

from ml.cli.dataset_report import main as report_main


def _make_df() -> pl.DataFrame:
    t0 = datetime(2024, 1, 1)
    ts = [t0 + timedelta(minutes=i) for i in range(5)]
    return pl.DataFrame(
        {
            "instrument_id": ["SPY", "SPY", "SPY", "QQQ", "QQQ"],
            "time_index": [0, 1, 2, 3, 4],
            "timestamp": ts,
            "y": [1, 0, 1, 0, 0],
            # Features with a null to test coverage
            "f1": [1.0, 2.0, None, 4.0, 5.0],
            "f2": [10.0, 11.0, 12.0, None, 14.0],
            # Macro columns (subset of known series)
            "DGS10": [4.0, 4.1, 4.0, 4.2, None],
            "VIXCLS": [15.0, None, 16.0, 15.5, 15.2],
        },
    )


def test_dataset_report_generates_stats(tmp_path: Path) -> None:
    df = _make_df()
    ds = tmp_path / "dataset.parquet"
    df.write_parquet(ds)

    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    rc = report_main(["--dataset", str(ds), "--out_json", str(out_json), "--out_md", str(out_md)])
    assert rc == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text())

    # Shape and sections present
    assert data["shape"] == [5, len(df.columns)]
    assert "macro_null_rates" in data and "feature_coverage" in data and "target" in data

    # Macro null rates computed for known columns
    mr = data["macro_null_rates"]
    assert set(mr.keys()) >= {"DGS10", "VIXCLS"}
    # There is one null in DGS10 across 5 rows
    assert abs(mr["DGS10"] - (1.0 / 5.0)) < 1e-9

    # Feature coverage overall should reflect non-null ratios
    fc_overall = data["feature_coverage"]["overall"]
    assert "f1" in fc_overall and "f2" in fc_overall
    # f1 has 4 non-nulls of 5 => 0.8
    assert abs(fc_overall["f1"] - 0.8) < 1e-9

    # Target distribution
    tgt = data["target"]["overall"]
    assert tgt["total"] == 5
    assert tgt["positives"] == 2
    assert abs(tgt["positive_rate"] - (2.0 / 5.0)) < 1e-9

    # Markdown file produced
    assert out_md.exists() and out_md.stat().st_size > 0
