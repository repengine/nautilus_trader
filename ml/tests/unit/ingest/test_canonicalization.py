from __future__ import annotations

import json
import os
from datetime import UTC
from datetime import datetime
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.ingest.canonicalization import canonicalize_equities_minute_bars


_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "ingest"
_REPORT_DIR = Path(__file__).resolve().parents[2] / "validation_reports"
_PARITY_REPORT_PATH = _REPORT_DIR / "equs_itch_parity_summary.json"

PRICE_P99_BOUND = float(os.getenv("ML_EQUS_PARITY_PRICE_P99_BOUND", "0.06"))
VOLUME_CORR_FLOOR = float(os.getenv("ML_EQUS_PARITY_VOLUME_CORR_FLOOR", "0.7"))
VOLUME_RATIO_LOWER = float(os.getenv("ML_EQUS_PARITY_RATIO_LOWER", "0.1"))
VOLUME_RATIO_UPPER = float(os.getenv("ML_EQUS_PARITY_RATIO_UPPER", "1.6"))


def _load_csv(name: str) -> pd.DataFrame:
    path = _FIXTURE_DIR / name
    return pd.read_csv(path)


def _build_synthetic_minute_bars(
    *,
    symbol: str,
    start: datetime,
    minutes: int,
    price_base: float,
    price_step: float,
    volume_base: float,
    volume_step: float,
) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=minutes, freq="1min", tz="UTC")
    ts_values = (index.view("int64")).astype("int64")
    increments = np.arange(minutes, dtype=np.float64)
    price_series = price_base + price_step * increments + 0.05 * np.sin(increments / 5.0)
    df = pd.DataFrame(
        {
            "ts_event": ts_values,
            "ts_init": ts_values,
            "open": price_series,
            "high": price_series + 0.1,
            "low": price_series - 0.1,
            "close": price_series + 0.02,
            "volume": volume_base + volume_step * increments,
            "symbol": symbol,
        },
    )
    return df


def _compute_parity_summary(
    eq_df: pd.DataFrame,
    fallback_df: pd.DataFrame,
    *,
    symbol: str,
    label: str,
    fallback_dataset: str,
    fallback_mode: str = "raw_fallback",
    scaling_factor: float | None = None,
) -> dict[str, object]:
    eq_result = canonicalize_equities_minute_bars(
        eq_df,
        source_dataset="EQUS.MINI",
        symbol=symbol,
        aggregation_mode="native",
    )
    fallback_result = canonicalize_equities_minute_bars(
        fallback_df,
        source_dataset=fallback_dataset,
        symbol=symbol,
        aggregation_mode=fallback_mode,
        scaling_factor=scaling_factor,
    )
    eq_frame = eq_result.frame
    fallback_frame = fallback_result.frame
    merged = eq_frame.merge(
        fallback_frame,
        on="ts_event",
        how="inner",
        suffixes=("_eq", "_fallback"),
    )
    merged = merged.sort_values("ts_event")
    assert not merged.empty, "Merged canonicalization frame should not be empty"

    price_diff = (merged["close_eq"] - merged["close_fallback"]).abs()
    volume_corr = merged["volume_eq"].corr(merged["volume_fallback"])
    price_corr = merged["close_eq"].corr(merged["close_fallback"])

    fallback_volumes = merged["volume_fallback"].replace(0, np.nan)
    volume_ratio = merged["volume_eq"] / fallback_volumes
    ratio_clean = volume_ratio.dropna()

    eq_volume_total = float(merged["volume_eq"].sum())
    fallback_volume_total = float(merged["volume_fallback"].sum())
    volume_residual_abs = abs(eq_volume_total - fallback_volume_total)
    volume_residual_rel = (
        volume_residual_abs / eq_volume_total if eq_volume_total else None
    )

    ratio_bounds: dict[str, float] = {}
    if not ratio_clean.empty:
        ratio_bounds = {
            "min": float(ratio_clean.min()),
            "max": float(ratio_clean.max()),
            "median": float(ratio_clean.median()),
            "p05": float(ratio_clean.quantile(0.05)),
            "p95": float(ratio_clean.quantile(0.95)),
        }

    summary: dict[str, object] = {
        "label": label,
        "symbol": symbol,
        "samples": len(merged.index),
        "price_close_p99_abs_diff": float(price_diff.quantile(0.99)),
        "price_close_correlation": float(price_corr) if not np.isnan(price_corr) else None,
        "volume_correlation": float(volume_corr) if not np.isnan(volume_corr) else None,
        "volume_ratio_bounds": ratio_bounds,
        "volume_residual_abs": volume_residual_abs,
        "volume_residual_rel": volume_residual_rel,
        "fallback_source_dataset": fallback_result.stats.source_dataset,
        "fallback_aggregation_mode": fallback_result.stats.aggregation_mode,
    }

    if fallback_result.stats.scaling_factor is not None:
        summary["fallback_scaling_factor"] = float(fallback_result.stats.scaling_factor)

    return summary


def test_canonicalized_itch_aligns_with_equs() -> None:
    eq_df = _load_csv("INTC_EQUS_20230515.csv")
    itch_df = _load_csv("INTC_XNAS_ITCH_20230515.csv")

    eq_canon = canonicalize_equities_minute_bars(
        eq_df,
        source_dataset="EQUS.MINI",
        symbol="INTC",
    ).frame
    itch_canon = canonicalize_equities_minute_bars(
        itch_df,
        source_dataset="XNAS.ITCH",
        symbol="INTC",
    ).frame

    missing = set(eq_canon["ts_event"]) - set(itch_canon["ts_event"])
    assert len(missing) <= 5

    merged = eq_canon.merge(itch_canon, on="ts_event", suffixes=("_eq", "_itch"))
    assert not merged.empty

    price_diff = (merged["close_eq"] - merged["close_itch"]).abs()
    assert price_diff.quantile(0.99) <= 0.06
    assert merged["close_eq"].corr(merged["close_itch"]) > 0.999
    assert merged["volume_eq"].corr(merged["volume_itch"]) > 0.7


def test_canonicalization_trims_out_of_session() -> None:
    itch_df = _load_csv("INTC_XNAS_ITCH_20230515.csv")
    canonical = canonicalize_equities_minute_bars(
        itch_df,
        source_dataset="XNAS.ITCH",
        symbol="INTC",
    ).frame

    timestamps = pd.to_datetime(canonical["ts_event"], unit="ns", utc=True)
    local = timestamps.dt.tz_convert("America/New_York")
    assert local.dt.dayofweek.max() <= 4
    assert local.dt.time.min() >= time(8, 0)
    assert local.dt.time.max() < time(16, 0)


def test_eq_itch_parity_metrics_report() -> None:
    eq_df = _load_csv("INTC_EQUS_20230515.csv")
    itch_df = _load_csv("INTC_XNAS_ITCH_20230515.csv")
    summaries: list[dict[str, object]] = []

    real_summary = _compute_parity_summary(
        eq_df,
        itch_df,
        symbol="INTC",
        label="INTC_20230515",
        fallback_dataset="XNAS.ITCH",
    )
    summaries.append(real_summary)

    synthetic_start = datetime(2017, 1, 2, 8, 0, tzinfo=UTC)
    synthetic_minutes = (8 * 60)
    spy_eq = _build_synthetic_minute_bars(
        symbol="SPY",
        start=synthetic_start,
        minutes=synthetic_minutes,
        price_base=220.0,
        price_step=0.02,
        volume_base=1_500.0,
        volume_step=15.0,
    )
    spy_fallback = spy_eq.copy(deep=True)
    spy_fallback["open"] = spy_fallback["open"] + 0.01
    spy_fallback["high"] = spy_fallback["high"] + 0.01
    spy_fallback["low"] = spy_fallback["low"] + 0.01
    spy_fallback["close"] = spy_fallback["close"] + 0.01
    spy_fallback["volume"] = (spy_fallback["volume"] / 1.05).round().clip(lower=1)

    synthetic_summary = _compute_parity_summary(
        spy_eq,
        spy_fallback,
        symbol="SPY",
        label="SPY_SYNTH_20170102",
        fallback_dataset="XNAS.ITCH",
        fallback_mode="scaled_volume",
        scaling_factor=1.05,
    )
    summaries.append(synthetic_summary)

    synthetic_reagg = spy_fallback.copy(deep=True)
    synthetic_reagg["volume"] = synthetic_reagg["volume"] * 1.02

    reagg_summary = _compute_parity_summary(
        spy_eq,
        synthetic_reagg,
        symbol="SPY",
        label="SPY_SYNTH_REAGG",
        fallback_dataset="XNAS.ITCH",
        fallback_mode="reaggregated_trades",
    )
    summaries.append(reagg_summary)

    for summary in summaries:
        assert summary["price_close_p99_abs_diff"] <= PRICE_P99_BOUND
        volume_corr = summary.get("volume_correlation")
        assert volume_corr is not None and volume_corr >= VOLUME_CORR_FLOOR
        ratio_bounds = summary["volume_ratio_bounds"]
        if ratio_bounds:
            assert ratio_bounds["p95"] <= VOLUME_RATIO_UPPER
            assert ratio_bounds["p05"] >= VOLUME_RATIO_LOWER

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_payload: dict[str, object]
    if _PARITY_REPORT_PATH.exists():
        try:
            report_payload = json.loads(_PARITY_REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:
            report_payload = {}
    else:
        report_payload = {}

    for summary in summaries:
        report_payload[summary["label"]] = summary

    _PARITY_REPORT_PATH.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
