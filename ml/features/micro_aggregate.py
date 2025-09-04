"""
Microstructure feature aggregation utilities (per-minute from L1/L2).

Separated from advanced L2 feature calculators to keep aggregation lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl


MICRO_COLUMNS = [
    "midprice",
    "spread_bps",
    "quote_imbalance",
    "trade_imbalance",
    "realized_vol",
]


def _ensure_polars() -> None:
    if not HAS_POLARS:
        check_ml_dependencies(["polars"])  # pragma: no cover


def aggregate_microstructure_minute_pl(
    quotes: pl.DataFrame | None,
    trades: pl.DataFrame | None,
    *,
    timestamp_col: str = "ts_event",
    bid_col: str = "bid_px_00",
    ask_col: str = "ask_px_00",
    bid_sz_col: str = "bid_sz_00",
    ask_sz_col: str = "ask_sz_00",
) -> pl.DataFrame:
    _ensure_polars()
    if quotes is None and trades is None:
        return pl.DataFrame({"timestamp": []})

    frames: list[pl.DataFrame] = []

    if quotes is not None and len(quotes) > 0:
        q = quotes
        if q[timestamp_col].dtype != pl.Datetime:
            q = q.with_columns(pl.col(timestamp_col).cast(pl.Datetime("ns", "UTC")))
        mid_expr = ((pl.col(bid_col) + pl.col(ask_col)) / 2.0)
        denom = (pl.col(bid_sz_col) + pl.col(ask_sz_col)).cast(pl.Float64)
        denom_safe = pl.when(denom > 0).then(denom).otherwise(1.0)
        q = q.with_columns(
            [
                mid_expr.alias("midprice"),
                (10000.0 * (pl.col(ask_col) - pl.col(bid_col)) / mid_expr).alias("spread_bps"),
                ((pl.col(bid_sz_col) - pl.col(ask_sz_col)) / denom_safe).alias("quote_imbalance"),
            ],
        )
        q_min = (
            q.group_by_dynamic(index_column=timestamp_col, every="1m", period="1m")
            .agg([
                pl.col("midprice").mean().alias("midprice"),
                pl.col("spread_bps").mean().alias("spread_bps"),
                pl.col("quote_imbalance").mean().alias("quote_imbalance"),
            ])
            .rename({timestamp_col: "timestamp"})
        )
        frames.append(q_min)

    if trades is not None and len(trades) > 0:
        t = trades
        if t[timestamp_col].dtype != pl.Datetime:
            t = t.with_columns(pl.col(timestamp_col).cast(pl.Datetime("ns", "UTC")))
        sign = pl.when(pl.col("side").str.contains("SELL")).then(-1).otherwise(1)
        t = t.with_columns((sign * pl.col("size").cast(pl.Float64)).alias("signed_size"))
        t = t.with_columns(pl.col("price").cast(pl.Float64))
        t = t.with_columns((pl.col("price").log() - pl.col("price").log().shift(1)).alias("log_ret"))
        denom_sum = pl.sum("size").cast(pl.Float64)
        denom_sum_safe = pl.when(denom_sum > 0).then(denom_sum).otherwise(1.0)
        t_min = (
            t.group_by_dynamic(index_column=timestamp_col, every="1m", period="1m")
            .agg([
                (pl.col("signed_size").sum() / denom_sum_safe).alias("trade_imbalance"),
                pl.col("log_ret").std().fill_null(0.0).alias("realized_vol"),
            ])
            .rename({timestamp_col: "timestamp"})
        )
        frames.append(t_min)

    if not frames:
        return pl.DataFrame({"timestamp": []})
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on="timestamp", how="outer")
    return out.sort("timestamp").fill_null(strategy="forward")


@dataclass(slots=True)
class MicrostructureAggregator:
    base_dir: Path

    def _load_l1_quotes(self, symbol: str) -> pl.DataFrame | None:
        _ensure_polars()
        l1_dir = self.base_dir / symbol / "l1"
        if not l1_dir.exists():
            return None
        paths = sorted(p for p in l1_dir.glob("*_bbo_*.parquet"))
        if not paths:
            return None
        try:
            df = pl.read_parquet(paths[-1])
            needed = [c for c in df.columns if c in {"ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"}]
            return df.select(needed)
        except Exception:
            return None

    def _load_l1_trades(self, symbol: str) -> pl.DataFrame | None:
        _ensure_polars()
        l1_dir = self.base_dir / symbol / "l1"
        if not l1_dir.exists():
            return None
        paths = sorted(p for p in l1_dir.glob("*_trades_*.parquet"))
        if not paths:
            return None
        try:
            df = pl.read_parquet(paths[-1])
            needed = [c for c in df.columns if c in {"ts_event", "price", "size", "side"}]
            return df.select(needed)
        except Exception:
            return None

    def compute_for_symbol(self, symbol: str) -> pl.DataFrame:
        _ensure_polars()
        q = self._load_l1_quotes(symbol)
        t = self._load_l1_trades(symbol)
        if q is None and t is None:
            return pl.DataFrame({"timestamp": []})
        return aggregate_microstructure_minute_pl(q, t)
