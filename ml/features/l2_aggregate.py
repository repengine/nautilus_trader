"""
L2 order book feature aggregation (per-minute) from MBP-10 snapshots.

This module computes robust per-minute features from Databento MBP-10 snapshots.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.common.safe_math import safe_divide_expr


TOPKS: tuple[int, ...] = (1, 3, 5, 10)


def _ensure_polars() -> None:
    if not HAS_POLARS:
        check_ml_dependencies(["polars"])  # pragma: no cover


def _cols(prefix: str, k: int) -> list[str]:
    return [f"{prefix}_{i:02d}" for i in range(k)]


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
    spread_bps = 10000.0 * safe_divide_expr(
        pl.col("ask_px_00") - pl.col("bid_px_00"),
        mid,
    )

    # Level 0 microprice
    microprice = safe_divide_expr(
        pl.col("ask_px_00") * pl.col("bid_sz_00") + pl.col("bid_px_00") * pl.col("ask_sz_00"),
        pl.col("bid_sz_00") + pl.col("ask_sz_00"),
    )
    microprice_bps = 10000.0 * safe_divide_expr(microprice - mid, mid)

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
        depth_imb = safe_divide_expr(sum_bid_sz - sum_ask_sz, total_sz)

        # Depth-weighted price across top-k
        dwp_num = pl.sum_horizontal(
            [
                pl.col(px).cast(pl.Float64) * pl.col(sz).cast(pl.Float64)
                for px, sz in zip(bid_px_cols, bid_sz_cols)
            ]
            + [
                pl.col(px).cast(pl.Float64) * pl.col(sz).cast(pl.Float64)
                for px, sz in zip(ask_px_cols, ask_sz_cols)
            ],
        )
        dwp = safe_divide_expr(dwp_num, total_sz)
        dwp_bps = 10000.0 * safe_divide_expr(dwp - mid, mid)

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

    df = df.sort(timestamp_col)
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

    def compute_for_symbol(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        _ensure_polars()
        logger = logging.getLogger(__name__)
        l2_dir = self.base_dir / symbol / "l2"
        if not l2_dir.exists():
            logger.debug("L2 dir missing for %s: %s", symbol, l2_dir)
            return pl.DataFrame({"timestamp": []})
        paths = sorted(p for p in l2_dir.glob("*.parquet"))
        if not paths:
            logger.debug("No L2 parquet files for %s", symbol)
            return pl.DataFrame({"timestamp": []})
        latest = paths[-1]
        try:
            logger.info("L2: using %s (%0.2f MB)", latest, latest.stat().st_size / (1024 * 1024))
            # Read only necessary columns to reduce memory
            cols = ["ts_event"]
            for i in range(10):
                cols += [f"bid_px_{i:02d}", f"ask_px_{i:02d}", f"bid_sz_{i:02d}", f"ask_sz_{i:02d}"]
            lf = pl.scan_parquet(str(latest)).select(cols)
            # Ensure datetime type on ts_event
            lf = lf.with_columns(pl.col("ts_event").cast(pl.Datetime("ns", "UTC")))
            if start is not None or end is not None:
                cond = pl.lit(True)
                if start is not None:
                    from ml.common.timestamps import sanitize_timestamp_ns

                    start_ns = sanitize_timestamp_ns(
                        int(start.timestamp() * 1_000_000_000),
                        context="l2_aggregate.scan.start",
                    )
                    cond = cond & (pl.col("ts_event").cast(pl.Int64) >= start_ns)
                if end is not None:
                    from ml.common.timestamps import sanitize_timestamp_ns

                    end_ns = sanitize_timestamp_ns(
                        int(end.timestamp() * 1_000_000_000),
                        context="l2_aggregate.scan.end",
                    )
                    cond = cond & (pl.col("ts_event").cast(pl.Int64) < end_ns)
                lf = lf.filter(cond)
            # Materialize only filtered rows
            df = lf.collect()
            est = getattr(df, "estimated_size", None)
            size_b = est() if callable(est) else 0
            logger.info(
                "L2 scan for %s produced %d rows, est_size=%s bytes",
                symbol,
                len(df),
                size_b,
            )
        except Exception:
            # Fallback to eager read
            logger.warning(
                "L2 lazy scan failed; falling back to eager read for %s",
                symbol,
                exc_info=True,
            )
            df_fallback = self._load_l2(symbol)
            if df_fallback is None or df_fallback.is_empty():
                return pl.DataFrame({"timestamp": []})
            if start is not None or end is not None:
                if df_fallback["ts_event"].dtype != pl.Datetime:
                    df_fallback = df_fallback.with_columns(
                        pl.col("ts_event").cast(pl.Datetime("ns", "UTC")),
                    )
                cond = pl.lit(True)
                if start is not None:
                    from ml.common.timestamps import sanitize_timestamp_ns

                    start_ns = sanitize_timestamp_ns(
                        int(start.timestamp() * 1_000_000_000),
                        context="l2_aggregate.eager.start",
                    )
                    cond = cond & (pl.col("ts_event").cast(pl.Int64) >= start_ns)
                if end is not None:
                    from ml.common.timestamps import sanitize_timestamp_ns

                    end_ns = sanitize_timestamp_ns(
                        int(end.timestamp() * 1_000_000_000),
                        context="l2_aggregate.eager.end",
                    )
                    cond = cond & (pl.col("ts_event").cast(pl.Int64) < end_ns)
                df_fallback = df_fallback.filter(cond)
            df_final = df_fallback
        else:
            df_final = df
        out = aggregate_l2_minute_pl(df_final)
        logger.info("L2 aggregated %s: %d rows -> %d minutes", symbol, len(df_final), len(out))
        return out
