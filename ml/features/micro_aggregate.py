"""
Microstructure feature aggregation utilities (per-minute from L1/L2).

Separated from advanced L2 feature calculators to keep aggregation lightweight.

"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import cast as _cast

from ml._imports import HAS_POLARS
from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.common import resolve_symbol_data_dir
from ml.ml_types import PolarsDF
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.funcs import urisafe_identifier


MICRO_COLUMNS = [
    "midprice",
    "spread_bps",
    "quote_imbalance",
    "trade_imbalance",
    "realized_vol",
]


logger = logging.getLogger(__name__)


def _ensure_polars() -> None:
    if not HAS_POLARS:
        check_ml_dependencies(["polars"])  # pragma: no cover


def aggregate_microstructure_minute_pl(
    quotes: PolarsDF | None,
    trades: PolarsDF | None,
    *,
    timestamp_col: str = "ts_event",
    bid_col: str = "bid_px_00",
    ask_col: str = "ask_px_00",
    bid_sz_col: str = "bid_sz_00",
    ask_sz_col: str = "ask_sz_00",
    trade_side_col: str = "side",
) -> PolarsDF:
    _ensure_polars()
    _pl = pl
    assert _pl is not None
    if quotes is None and trades is None:
        return _cast(PolarsDF, _pl.DataFrame({"timestamp": []}))

    frames: list[PolarsDF] = []

    if quotes is not None and len(quotes) > 0:
        q = quotes
        if q[timestamp_col].dtype != _pl.Datetime:
            q = q.with_columns(_pl.col(timestamp_col).cast(_pl.Datetime("ns", "UTC")))
        q = q.filter(_pl.col(timestamp_col).is_not_null()).sort(timestamp_col)
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
        t = t.filter(_pl.col(timestamp_col).is_not_null()).sort(timestamp_col)
        side_expr = _pl.col(trade_side_col).cast(_pl.Utf8)
        sign = _pl.when(side_expr.str.contains("SELL")).then(-1).otherwise(1)
        t = t.with_columns((sign * _pl.col("size").cast(_pl.Float64)).alias("signed_size"))
        t = t.with_columns(_pl.col("price").cast(_pl.Float64))
        t = t.with_columns(
            (_pl.col("price").log() - _pl.col("price").log().shift(1)).alias("log_ret"),
        )
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
    catalog_path: Path | None = None
    prefer_catalog: bool = True
    catalog_symbol_suffix: str | None = None

    def _resolve_catalog_root(self) -> Path | None:
        if not self.prefer_catalog:
            return None
        if self.catalog_path is not None:
            return self.catalog_path.expanduser()
        env_path = os.getenv("CATALOG_PATH")
        if env_path:
            return Path(env_path).expanduser()
        default = Path("data/catalog")
        return default if default.exists() else None

    def _catalog_kind_dir(self, kind: str) -> Path | None:
        root = self._resolve_catalog_root()
        if root is None:
            return None
        base = root / "data" / kind
        return base if base.exists() else None

    def _catalog_symbol_candidates(self, symbol: str) -> tuple[str, ...]:
        normalized = symbol.strip().upper()
        if not normalized:
            return ()
        candidates = [normalized]
        suffix = self.catalog_symbol_suffix or os.getenv("CATALOG_SYMBOL_SUFFIX", ".EQUS")
        if suffix and not normalized.endswith(suffix):
            if normalized.endswith(".XNAS"):
                base = normalized[: -len(".XNAS")]
                if base:
                    candidates.append(f"{base}{suffix}")
            else:
                candidates.append(f"{normalized}{suffix}")
        return tuple(dict.fromkeys(candidates))

    def _resolve_catalog_dir(self, symbol: str, kind: str) -> Path | None:
        base = self._catalog_kind_dir(kind)
        if base is None:
            return None
        for candidate in self._catalog_symbol_candidates(symbol):
            safe = urisafe_identifier(candidate)
            path = base / safe
            if path.exists():
                return path
        head = symbol.strip().upper()
        if not head:
            return None
        head_token = head.rsplit(".", 1)[0] if "." in head else head
        matches = [
            entry
            for entry in base.iterdir()
            if entry.is_dir() and entry.name.upper().startswith(f"{head_token}.")
        ]
        if matches:
            return sorted(matches)[0]
        return None

    def _load_catalog_quotes(
        self,
        symbol: str,
        *,
        start: datetime | None,
        end: datetime | None,
    ) -> PolarsDF | None:
        root = self._resolve_catalog_root()
        if root is None:
            return None
        base = self._catalog_kind_dir("quote_tick")
        if base is None:
            return None
        catalog = ParquetDataCatalog(str(root))
        for candidate in self._catalog_symbol_candidates(symbol):
            safe = urisafe_identifier(candidate)
            if not (base / safe).exists():
                continue
            try:
                from ml.data.catalog_utils import quotes_to_dataframe

                df = quotes_to_dataframe(catalog, [candidate], start=start, end=end)
            except Exception:
                logger.debug(
                    "catalog.quotes_load_failed",
                    exc_info=True,
                    extra={"symbol": candidate},
                )
                continue
            if df.is_empty():
                continue
            if "timestamp" in df.columns and "ts_event" not in df.columns:
                df = df.rename({"timestamp": "ts_event"})
            return self._normalize_quote_frame(df)
        return None

    def _load_catalog_trades(
        self,
        symbol: str,
        *,
        start: datetime | None,
        end: datetime | None,
    ) -> PolarsDF | None:
        root = self._resolve_catalog_root()
        if root is None:
            return None
        base = self._catalog_kind_dir("trade_tick")
        if base is None:
            return None
        catalog = ParquetDataCatalog(str(root))
        for candidate in self._catalog_symbol_candidates(symbol):
            safe = urisafe_identifier(candidate)
            if not (base / safe).exists():
                continue
            try:
                from ml.data.catalog_utils import trades_to_dataframe

                df = trades_to_dataframe(catalog, [candidate], start=start, end=end)
            except Exception:
                logger.debug(
                    "catalog.trades_load_failed",
                    exc_info=True,
                    extra={"symbol": candidate},
                )
                continue
            if df.is_empty():
                continue
            if "timestamp" in df.columns and "ts_event" not in df.columns:
                df = df.rename({"timestamp": "ts_event"})
            return self._normalize_trade_frame(df)
        return None

    def _apply_time_window(
        self,
        lf: Any,
        *,
        start: datetime | None,
        end: datetime | None,
        column: str,
    ) -> Any:
        if start is None and end is None:
            return lf
        _ensure_polars()
        _pl = pl
        assert _pl is not None
        cond = _pl.lit(True)
        if start is not None:
            start_ns = int(start.timestamp() * 1_000_000_000)
            cond = cond & (_pl.col(column).cast(_pl.Int64) >= start_ns)
        if end is not None:
            end_ns = int(end.timestamp() * 1_000_000_000)
            cond = cond & (_pl.col(column).cast(_pl.Int64) < end_ns)
        return lf.filter(cond)

    def _normalize_quote_frame(self, quotes: PolarsDF) -> PolarsDF:
        _ensure_polars()
        _pl = pl
        assert _pl is not None
        mapping: dict[str, str] = {}
        if "bid_price" in quotes.columns:
            mapping["bid_price"] = "bid_px_00"
        elif "bid" in quotes.columns:
            mapping["bid"] = "bid_px_00"
        if "ask_price" in quotes.columns:
            mapping["ask_price"] = "ask_px_00"
        elif "ask" in quotes.columns:
            mapping["ask"] = "ask_px_00"
        if "bid_size" in quotes.columns:
            mapping["bid_size"] = "bid_sz_00"
        elif "bid_sz" in quotes.columns:
            mapping["bid_sz"] = "bid_sz_00"
        if "ask_size" in quotes.columns:
            mapping["ask_size"] = "ask_sz_00"
        elif "ask_sz" in quotes.columns:
            mapping["ask_sz"] = "ask_sz_00"
        if mapping:
            quotes = quotes.rename(mapping)
        return quotes

    def _normalize_trade_frame(self, trades: PolarsDF) -> PolarsDF:
        if "side" in trades.columns:
            return trades
        if "aggressor_side" in trades.columns:
            return trades.rename({"aggressor_side": "side"})
        return trades

    def _resolve_l1_dir(self, symbol: str) -> Path | None:
        resolved = resolve_symbol_data_dir(self.base_dir, symbol)
        if resolved is None:
            return None
        l1_dir = resolved / "l1"
        if l1_dir.exists():
            return l1_dir
        return None

    def _load_l1_quotes(self, symbol: str) -> PolarsDF | None:
        _ensure_polars()
        _pl = pl
        assert _pl is not None
        l1_dir = self._resolve_l1_dir(symbol)
        if l1_dir is None:
            return None
        paths = [p for p in l1_dir.glob("*_bbo*.parquet") if p.is_file()]
        if not paths:
            return None
        latest = max(paths, key=lambda path: (path.stat().st_mtime, path.name))
        try:
            df = _pl.read_parquet(latest)
            needed = [
                c
                for c in df.columns
                if c in {"ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"}
            ]
            return _cast(PolarsDF, df.select(needed))
        except Exception:
            logger.debug("L1 quotes load failed for %s (%s)", symbol, l1_dir, exc_info=True)
            return None

    def _load_l1_trades(self, symbol: str) -> PolarsDF | None:
        _ensure_polars()
        _pl = pl
        assert _pl is not None
        l1_dir = self._resolve_l1_dir(symbol)
        if l1_dir is None:
            return None
        paths = [p for p in l1_dir.glob("*_trades*.parquet") if p.is_file()]
        if not paths:
            return None
        latest = max(paths, key=lambda path: (path.stat().st_mtime, path.name))
        try:
            df = _pl.read_parquet(latest)
            needed = [c for c in df.columns if c in {"ts_event", "price", "size", "side"}]
            return _cast(PolarsDF, df.select(needed))
        except Exception:
            logger.debug("L1 trades load failed for %s (%s)", symbol, l1_dir, exc_info=True)
            return None

    def compute_for_symbol(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> PolarsDF:
        _ensure_polars()
        _pl = pl
        assert _pl is not None
        q = self._load_catalog_quotes(symbol, start=start, end=end)
        t = self._load_catalog_trades(symbol, start=start, end=end)
        if q is None and t is None:
            q = self._load_l1_quotes(symbol)
            t = self._load_l1_trades(symbol)
        if q is not None:
            q = _cast(
                PolarsDF,
                self._apply_time_window(q, start=start, end=end, column="ts_event"),
            )
        if t is not None:
            t = _cast(
                PolarsDF,
                self._apply_time_window(t, start=start, end=end, column="ts_event"),
            )
        if q is None and t is None:
            return _cast(PolarsDF, _pl.DataFrame({"timestamp": []}))
        return aggregate_microstructure_minute_pl(q, t)
