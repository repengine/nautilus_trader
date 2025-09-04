"""
L2 order book feature aggregation (per-minute) from MBP-10 snapshots.

This module computes robust per-minute features from Databento MBP-10 snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl


TOPKS: tuple[int, ...] = (1, 3, 5, 10)


def _ensure_polars() -> None:
    if not HAS_POLARS:
        check_ml_dependencies(["polars"])  # pragma: no cover


def _cols(prefix: str, k: int) -> list[str]:
    return [f"{prefix}_{i:02d}" for i in range(k)]


def _safe_div(numer: pl.Expr, denom: pl.Expr) -> pl.Expr:
    return numer / pl.when(denom > 0).then(denom).otherwise(1.0)


def _slope_approx(p0: pl.Expr, pk: pl.Expr, k: int) -> pl.Expr:
    # Approximate slope across levels: (p_{k-1} - p_0) / (k-1)
    return (pk - p0) / max(k - 1, 1)


def aggregate_l2_minute_pl(l2: pl.DataFrame, *, timestamp_col: str = "ts_event") -> pl.DataFrame:
    """
    Aggregate MBP-10 L2 snapshots to per-minute features using Polars.
    """
    _ensure_polars()
    if l2.is_empty():
        return pl.DataFrame({"timestamp": []})

    df = l2
    if df[timestamp_col].dtype != pl.Datetime:
        df = df.with_columns(pl.col(timestamp_col).cast(pl.Datetime("ns", "UTC")))

    # Base mid and spread at level 0
    mid = (pl.col("ask_px_00") + pl.col("bid_px_00")) / 2.0
    spread_bps = 10000.0 * _safe_div(pl.col("ask_px_00") - pl.col("bid_px_00"), mid)

    # Level 0 microprice
    microprice = _safe_div(
        pl.col("ask_px_00") * pl.col("bid_sz_00") + pl.col("bid_px_00") * pl.col("ask_sz_00"),
        pl.col("bid_sz_00") + pl.col("ask_sz_00"),
    )
    microprice_bps = 10000.0 * _safe_div(microprice - mid, mid)

    # Build per-minute aggregations
    aggs: list[pl.Expr] = [
        mid.mean().alias("midprice"),
        spread_bps.mean().alias("spread_bps"),
        microprice_bps.mean().alias("microprice_bps"),
    ]

    for k in TOPKS:
        bid_sz_cols = _cols("bid_sz", k)
        ask_sz_cols = _cols("ask_sz", k)
        bid_px_cols = _cols("bid_px", k)
        ask_px_cols = _cols("ask_px", k)

        sum_bid_sz = pl.sum_horizontal([pl.col(c).cast(pl.Float64) for c in bid_sz_cols])
        sum_ask_sz = pl.sum_horizontal([pl.col(c).cast(pl.Float64) for c in ask_sz_cols])
        total_sz = sum_bid_sz + sum_ask_sz
        depth_imb = _safe_div(sum_bid_sz - sum_ask_sz, total_sz)

        # Depth-weighted price across top-k
        dwp_num = pl.sum_horizontal(
            [pl.col(px).cast(pl.Float64) * pl.col(sz).cast(pl.Float64) for px, sz in zip(bid_px_cols, bid_sz_cols)]
            + [pl.col(px).cast(pl.Float64) * pl.col(sz).cast(pl.Float64) for px, sz in zip(ask_px_cols, ask_sz_cols)]
        )
        dwp = _safe_div(dwp_num, total_sz)
        dwp_bps = 10000.0 * _safe_div(dwp - mid, mid)

        # Price slope approximation across k levels
        bid_slope = _slope_approx(pl.col("bid_px_00"), pl.col(f"bid_px_{k-1:02d}"), k)
        ask_slope = _slope_approx(pl.col("ask_px_00"), pl.col(f"ask_px_{k-1:02d}"), k)

        aggs.extend(
            [
                depth_imb.mean().alias(f"depth_imbalance_top{k}"),
                dwp_bps.mean().alias(f"dwp_bps_top{k}"),
                bid_slope.mean().alias(f"bid_slope_top{k}"),
                ask_slope.mean().alias(f"ask_slope_top{k}"),
            ],
        )

    out = (
        df.group_by_dynamic(index_column=timestamp_col, every="1m", period="1m")
        .agg(aggs)
        .rename({timestamp_col: "timestamp"})
        .sort("timestamp")
    )
    return out


@dataclass(slots=True)
class L2Aggregator:
    base_dir: Path

    def _load_l2(self, symbol: str) -> pl.DataFrame | None:
        _ensure_polars()
        l2_dir = self.base_dir / symbol / "l2"
        if not l2_dir.exists():
            return None
        paths = sorted(p for p in l2_dir.glob("*.parquet"))
        if not paths:
            return None
        try:
            return pl.read_parquet(str(paths[-1]))
        except Exception:
            return None

    def compute_for_symbol(self, symbol: str) -> pl.DataFrame:
        df = self._load_l2(symbol)
        if df is None or df.is_empty():
            return pl.DataFrame({"timestamp": []})
        return aggregate_l2_minute_pl(df)
