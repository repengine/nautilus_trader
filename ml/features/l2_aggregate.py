"""
L2 order book feature aggregation (per-minute) from MBP-10 snapshots.

This module computes robust per-minute features from Databento MBP-10 snapshots.
Streaming execution remains gated until a dedicated order-book backend is wired
and the Databento subscription is active again, to avoid hot-path IO/allocations
and preserve batch/stream parity guarantees.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from typing import cast as _cast

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.common import resolve_symbol_data_dir
from ml.common.safe_math import safe_divide_expr


if TYPE_CHECKING:
    from polars import DataFrame as PlDataFrame
    from polars import Expr as PlExpr
else:  # pragma: no cover - typing only
    PlDataFrame = Any  # type: ignore[assignment]
    PlExpr = Any  # type: ignore[assignment]

# Cast runtime handle to avoid Optional union complaints in type checking
PL = _cast(Any, pl)


TOPKS: tuple[int, ...] = (1, 3, 5, 10)

# Columns produced by aggregate_l2_minute_pl
L2_MINUTE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "midprice",
    "spread_bps",
    "microprice_bps",
    *[f"depth_imbalance_top{k}" for k in TOPKS],
    *[f"dwp_bps_top{k}" for k in TOPKS],
    *[f"bid_slope_top{k}" for k in TOPKS],
    *[f"ask_slope_top{k}" for k in TOPKS],
)


def _ensure_polars() -> None:
    if not HAS_POLARS:
        check_ml_dependencies(["polars"])  # pragma: no cover


def _cols(prefix: str, k: int) -> list[str]:
    return [f"{prefix}_{i:02d}" for i in range(k)]


def _slope_approx(p0: PlExpr, pk: PlExpr, k: int) -> PlExpr:
    # Approximate slope across levels: (p_{k-1} - p_0) / (k-1)
    return (pk - p0) / max(k - 1, 1)


def aggregate_l2_minute_pl(l2: PlDataFrame, *, timestamp_col: str = "ts_event") -> PlDataFrame:
    """
    Aggregate MBP-10 L2 snapshots to per-minute features using Polars.
    """
    _ensure_polars()
    if l2.is_empty():
        schema = [(L2_MINUTE_COLUMNS[0], PL.Datetime("ns", "UTC"))]
        schema.extend((name, PL.Float64) for name in L2_MINUTE_COLUMNS[1:])
        return _cast(PlDataFrame, PL.DataFrame(schema=schema))

    df = l2
    if df[timestamp_col].dtype != PL.Datetime:
        df = df.with_columns(PL.col(timestamp_col).cast(PL.Datetime("ns", "UTC")))

    # Base mid and spread at level 0
    mid = (PL.col("ask_px_00") + PL.col("bid_px_00")) / 2.0
    spread_bps = 10000.0 * safe_divide_expr(
        PL.col("ask_px_00") - PL.col("bid_px_00"),
        mid,
    )

    # Level 0 microprice
    microprice = safe_divide_expr(
        PL.col("ask_px_00") * PL.col("bid_sz_00") + PL.col("bid_px_00") * PL.col("ask_sz_00"),
        PL.col("bid_sz_00") + PL.col("ask_sz_00"),
    )
    microprice_bps = 10000.0 * safe_divide_expr(microprice - mid, mid)

    # Build per-minute aggregations
    aggs: list[PlExpr] = [
        mid.mean().alias("midprice"),
        spread_bps.mean().alias("spread_bps"),
        microprice_bps.mean().alias("microprice_bps"),
    ]

    for k in TOPKS:
        bid_sz_cols = _cols("bid_sz", k)
        ask_sz_cols = _cols("ask_sz", k)
        bid_px_cols = _cols("bid_px", k)
        ask_px_cols = _cols("ask_px", k)

        sum_bid_sz = PL.sum_horizontal([PL.col(c).cast(PL.Float64) for c in bid_sz_cols])
        sum_ask_sz = PL.sum_horizontal([PL.col(c).cast(PL.Float64) for c in ask_sz_cols])
        total_sz = sum_bid_sz + sum_ask_sz
        depth_imb = safe_divide_expr(sum_bid_sz - sum_ask_sz, total_sz)

        # Depth-weighted price across top-k
        dwp_num = PL.sum_horizontal(
            [
                PL.col(px).cast(PL.Float64) * PL.col(sz).cast(PL.Float64)
                for px, sz in zip(bid_px_cols, bid_sz_cols)
            ]
            + [
                PL.col(px).cast(PL.Float64) * PL.col(sz).cast(PL.Float64)
                for px, sz in zip(ask_px_cols, ask_sz_cols)
            ],
        )
        dwp = safe_divide_expr(dwp_num, total_sz)
        dwp_bps = 10000.0 * safe_divide_expr(dwp - mid, mid)

        # Price slope approximation across k levels
        bid_slope = _slope_approx(PL.col("bid_px_00"), PL.col(f"bid_px_{k-1:02d}"), k)
        ask_slope = _slope_approx(PL.col("ask_px_00"), PL.col(f"ask_px_{k-1:02d}"), k)

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

    def _resolve_l2_dir(self, symbol: str) -> Path | None:
        resolved = resolve_symbol_data_dir(self.base_dir, symbol)
        if resolved is None:
            return None
        l2_dir = resolved / "l2"
        return l2_dir if l2_dir.exists() else None

    def _load_l2(self, symbol: str) -> PlDataFrame | None:
        _ensure_polars()
        l2_dir = self._resolve_l2_dir(symbol)
        if l2_dir is None:
            return None
        paths = sorted(p for p in l2_dir.glob("*.parquet"))
        if not paths:
            return None
        try:
            return _cast(PlDataFrame, PL.read_parquet(str(paths[-1])))
        except Exception:
            return None

    def compute_for_symbol(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> PlDataFrame:
        _ensure_polars()
        logger = logging.getLogger(__name__)
        l2_dir = self._resolve_l2_dir(symbol)
        if l2_dir is None:
            logger.debug("L2 dir missing for %s: %s", symbol, self.base_dir)
            return _cast(PlDataFrame, PL.DataFrame({"timestamp": []}))
        paths = sorted(p for p in l2_dir.glob("*.parquet"))
        if not paths:
            logger.debug("No L2 parquet files for %s", symbol)
            return _cast(PlDataFrame, PL.DataFrame({"timestamp": []}))
        latest = paths[-1]
        try:
            logger.info("L2: using %s (%0.2f MB)", latest, latest.stat().st_size / (1024 * 1024))
            # Read only necessary columns to reduce memory
            cols = ["ts_event"]
            for i in range(10):
                cols += [f"bid_px_{i:02d}", f"ask_px_{i:02d}", f"bid_sz_{i:02d}", f"ask_sz_{i:02d}"]
            lf = PL.scan_parquet(str(latest)).select(cols)
            # Ensure datetime type on ts_event
            lf = lf.with_columns(PL.col("ts_event").cast(PL.Datetime("ns", "UTC")))
            if start is not None or end is not None:
                cond = PL.lit(True)
                if start is not None:
                    from ml.common.timestamps import sanitize_timestamp_ns

                    start_ns = sanitize_timestamp_ns(
                        int(start.timestamp() * 1_000_000_000),
                        context="l2_aggregate.scan.start",
                    )
                    cond = cond & (PL.col("ts_event").cast(PL.Int64) >= start_ns)
                if end is not None:
                    from ml.common.timestamps import sanitize_timestamp_ns

                    end_ns = sanitize_timestamp_ns(
                        int(end.timestamp() * 1_000_000_000),
                        context="l2_aggregate.scan.end",
                    )
                    cond = cond & (PL.col("ts_event").cast(PL.Int64) < end_ns)
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
                return _cast(PlDataFrame, PL.DataFrame({"timestamp": []}))
            if start is not None or end is not None:
                if df_fallback["ts_event"].dtype != PL.Datetime:
                    df_fallback = df_fallback.with_columns(
                        PL.col("ts_event").cast(PL.Datetime("ns", "UTC")),
                    )
                cond = PL.lit(True)
                if start is not None:
                    from ml.common.timestamps import sanitize_timestamp_ns

                    start_ns = sanitize_timestamp_ns(
                        int(start.timestamp() * 1_000_000_000),
                        context="l2_aggregate.eager.start",
                    )
                    cond = cond & (PL.col("ts_event").cast(PL.Int64) >= start_ns)
                if end is not None:
                    from ml.common.timestamps import sanitize_timestamp_ns

                    end_ns = sanitize_timestamp_ns(
                        int(end.timestamp() * 1_000_000_000),
                        context="l2_aggregate.eager.end",
                    )
                    cond = cond & (PL.col("ts_event").cast(PL.Int64) < end_ns)
                df_fallback = df_fallback.filter(cond)
            df_final = df_fallback
        else:
            df_final = df
        out = aggregate_l2_minute_pl(df_final)
        logger.info("L2 aggregated %s: %d rows -> %d minutes", symbol, len(df_final), len(out))
        return out
