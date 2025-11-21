"""
Microstructure feature aggregation utilities (per-minute from L1/L2).

Separated from advanced L2 feature calculators to keep aggregation lightweight.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import cast as _cast

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.features._symbol_utils import resolve_symbol_data_dir
from ml.features._symbol_utils import select_latest_symbol_file
from ml.ml_types import PolarsDF


MICRO_COLUMNS = [
    "midprice",
    "spread_bps",
    "quote_imbalance",
    "trade_imbalance",
    "realized_vol",
]


def _ensure_polars() -> Any:
    if not HAS_POLARS:
        check_ml_dependencies(["polars"])  # pragma: no cover - raises if missing
    module = pl
    if module is None:
        check_ml_dependencies(["polars"])  # pragma: no cover - raises if missing
        module = pl
    if module is None:
        raise RuntimeError("Polars dependency 'polars' is required for microstructure aggregation")
    return module


def aggregate_microstructure_minute_pl(
    quotes: PolarsDF | None,
    trades: PolarsDF | None,
    *,
    timestamp_col: str = "ts_event",
    bid_col: str = "bid_px_00",
    ask_col: str = "ask_px_00",
    bid_sz_col: str = "bid_sz_00",
    ask_sz_col: str = "ask_sz_00",
) -> PolarsDF:
    _pl = _ensure_polars()
    if quotes is None and trades is None:
        return _cast(PolarsDF, _pl.DataFrame({"timestamp": []}))

    frames: list[PolarsDF] = []

    if quotes is not None and len(quotes) > 0:
        q = quotes
        if q[timestamp_col].dtype != _pl.Datetime:
            q = q.with_columns(_pl.col(timestamp_col).cast(_pl.Datetime("ns", "UTC")))
        q = q.drop_nulls(subset=[timestamp_col])
        mid_expr = (_pl.col(bid_col) + _pl.col(ask_col)) / 2.0
        denom = (_pl.col(bid_sz_col) + _pl.col(ask_sz_col)).cast(_pl.Float64)
        denom_safe = _pl.when(denom > 0).then(denom).otherwise(1.0)
        q = q.with_columns(
            [
                mid_expr.alias("midprice"),
                (10000.0 * (_pl.col(ask_col) - _pl.col(bid_col)) / mid_expr).alias("spread_bps"),
                ((_pl.col(bid_sz_col) - _pl.col(ask_sz_col)) / denom_safe).alias("quote_imbalance"),
            ],
        )
        q = q.sort(timestamp_col)
        q_min = (
            q.group_by_dynamic(index_column=timestamp_col, every="1m", period="1m")
            .agg(
                [
                    _pl.col("midprice").mean().alias("midprice"),
                    _pl.col("spread_bps").mean().alias("spread_bps"),
                    _pl.col("quote_imbalance").mean().alias("quote_imbalance"),
                ],
            )
            .rename({timestamp_col: "timestamp"})
        )
        frames.append(q_min)

    if trades is not None and len(trades) > 0:
        t = trades
        if t[timestamp_col].dtype != _pl.Datetime:
            t = t.with_columns(_pl.col(timestamp_col).cast(_pl.Datetime("ns", "UTC")))
        t = t.drop_nulls(subset=[timestamp_col])
        sign = _pl.when(_pl.col("side").str.contains("SELL")).then(-1).otherwise(1)
        t = t.with_columns((sign * _pl.col("size").cast(_pl.Float64)).alias("signed_size"))
        t = t.with_columns(_pl.col("price").cast(_pl.Float64))
        t = t.with_columns(
            (_pl.col("price").log() - _pl.col("price").log().shift(1)).alias("log_ret"),
        )
        t = t.sort(timestamp_col)
        denom_sum = _pl.sum("size").cast(_pl.Float64)
        denom_sum_safe = _pl.when(denom_sum > 0).then(denom_sum).otherwise(1.0)
        t_min = (
            t.group_by_dynamic(index_column=timestamp_col, every="1m", period="1m")
            .agg(
                [
                    (_pl.col("signed_size").sum() / denom_sum_safe).alias("trade_imbalance"),
                    _pl.col("log_ret").std().fill_null(0.0).alias("realized_vol"),
                ],
            )
            .rename({timestamp_col: "timestamp"})
        )
        frames.append(t_min)

    if not frames:
        return _cast(PolarsDF, _pl.DataFrame({"timestamp": []}))
    out = frames[0]
    for f in frames[1:]:
        # Polars 0.20.29: use 'full' instead of deprecated 'outer'
        out = out.join(f, on="timestamp", how="full")
    return out.sort("timestamp").fill_null(strategy="forward")


@dataclass(slots=True)
class MicrostructureAggregator:
    base_dir: Path

    def _load_l1_quotes(self, symbol: str) -> PolarsDF | None:
        _pl = _ensure_polars()
        root_dir = resolve_symbol_data_dir(self.base_dir, symbol)
        if root_dir is None:
            return None
        l1_dir = root_dir / "l1"
        if not l1_dir.exists():
            return None
        path = select_latest_symbol_file(l1_dir, root_dir.name, "bbo")
        if path is None:
            return None
        try:
            df = _pl.read_parquet(str(path))
            needed = [
                c
                for c in df.columns
                if c in {"ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"}
            ]
            return _cast(PolarsDF, df.select(needed))
        except Exception:
            return None

    def _load_l1_trades(self, symbol: str) -> PolarsDF | None:
        _pl = _ensure_polars()
        root_dir = resolve_symbol_data_dir(self.base_dir, symbol)
        if root_dir is None:
            return None
        l1_dir = root_dir / "l1"
        if not l1_dir.exists():
            return None
        path = select_latest_symbol_file(l1_dir, root_dir.name, "trades")
        if path is None:
            return None
        try:
            df = _pl.read_parquet(str(path))
            needed = [c for c in df.columns if c in {"ts_event", "price", "size", "side"}]
            return _cast(PolarsDF, df.select(needed))
        except Exception:
            return None

    def compute_for_symbol(self, symbol: str) -> PolarsDF:
        _pl = _ensure_polars()
        q = self._load_l1_quotes(symbol)
        t = self._load_l1_trades(symbol)
        if q is None and t is None:
            return _cast(PolarsDF, _pl.DataFrame({"timestamp": []}))
        return aggregate_microstructure_minute_pl(q, t)
