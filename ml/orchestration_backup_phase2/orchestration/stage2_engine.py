from __future__ import annotations


# ruff: noqa: E402  # Allow module docstring before imports per project style

"""
Stage 2 promotion engines (cold path only).

Defines a small, typed interface for computing trading metrics used by promotion
gates. Two engines are exposed:

- ReturnsStage2Engine: Computes realized forward returns from catalog bars aligned to
  the teacher validation tail and applies a simple threshold + cost model.
- BacktestStage2EngineRunner: Advisory hook for Nautilus Trader BacktestEngine. The
  current implementation validates environment capability and may fall back to the
  returns engine when the backtest stack is unavailable.

All logic here is cold-path. No hot-path imports or work occur in this module.
"""

import logging
import os
from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray


if TYPE_CHECKING:  # pragma: no cover - avoid runtime coupling
    from ml.orchestration.promotions import Stage2Config as _Stage2Config


@dataclass(slots=True, frozen=True)
class Stage2Result:
    status: Literal["passed", "failed", "skipped"]
    metrics: dict[str, float]
    reason: str | None = None


class Stage2Engine(Protocol):
    def run(self, cfg: _Stage2Config) -> Stage2Result: ...


def _load_validation_arrays(out_dir: str) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """
    Load q_val and y_val_true arrays from teacher_preds.npz under out_dir.

    Returns (q_val, y_val_true) or None if unavailable.

    """
    try:
        from pathlib import Path as _Path

        import numpy as _np

        p = _Path(out_dir) / "teacher_preds.npz"
        if not p.exists():
            return None
        npz = _np.load(str(p))
        if "q_val" not in npz or "y_val_true" not in npz:
            return None
        q_val = _np.asarray(npz["q_val"], dtype=_np.float64).reshape(-1)
        y_val_true = _np.asarray(npz["y_val_true"], dtype=_np.float64).reshape(-1)
        return (q_val, y_val_true)
    except Exception:
        return None


def _load_validation_tail(dataset_csv: str, n_tail: int) -> tuple[Any, Any] | None:
    """
    Load last n_tail rows from dataset_csv sorted by time_index.

    Returns (df_tail: pandas.DataFrame, pandas_module) or None.

    """
    try:
        from ml._imports import pd as _pd

        if _pd is None:
            from ml._imports import check_ml_dependencies as _check

            _check(["pandas"])  # cold-path guard
            from ml._imports import pd as _pd
        pd_mod = cast(Any, _pd)
        df = pd_mod.read_csv(dataset_csv)
        if "time_index" not in df.columns:
            return None
        df_sorted = df.sort_values("time_index")
        tail = df_sorted.tail(int(n_tail))
        if "timestamp" not in tail.columns or "instrument_id" not in tail.columns:
            return None
        return (tail, pd_mod)
    except Exception:
        return None


class ReturnsStage2Engine:
    """
    Computes trading metrics using a returns-based decision policy on the validation
    tail.
    """

    def run(
        self,
        cfg: _Stage2Config,
    ) -> Stage2Result:  # pragma: no cover - exercised via promotions
        arrays = _load_validation_arrays(cfg.out_dir)
        if arrays is None:
            return Stage2Result(status="skipped", metrics={}, reason="teacher_preds.npz missing")
        q_val, _y_val_true = arrays
        if q_val.size == 0:
            return Stage2Result(status="skipped", metrics={}, reason="empty validation window")

        tail = _load_validation_tail(cfg.dataset_csv, int(q_val.size))
        if tail is None:
            return Stage2Result(status="skipped", metrics={}, reason="dataset tail missing cols")
        df_val, pd_mod = tail
        df_val = cast(Any, df_val)
        pd_mod = cast(Any, pd_mod)
        # Build realized forward returns R[t] = (close[t+h] - close[t]) / close[t]
        try:
            from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog as _Cat
        except Exception as exc:
            return Stage2Result(
                status="skipped",
                metrics={},
                reason=f"catalog import failed: {exc}",
            )

        cat = _Cat(str(cfg.data_dir))
        by_inst = df_val.groupby("instrument_id")
        realized: list[float] = []
        horizon_ns = int(cfg.horizon_minutes) * 60 * 1_000_000_000

        for inst, g in by_inst:
            ts_ns = pd_mod.to_datetime(g["timestamp"]).astype("int64")
            ts_min = int(ts_ns.min())
            ts_max = int(ts_ns.max() + horizon_ns)
            try:
                from datetime import datetime as _dt

                from ml.data.catalog_utils import bars_to_dataframe as _bars_to_df

                start_dt = _dt.fromtimestamp(ts_min / 1e9, tz=UTC)
                end_dt = _dt.fromtimestamp(ts_max / 1e9, tz=UTC)
                bars_pl = _bars_to_df(cat, [str(inst)], start=start_dt, end=end_dt)
            except Exception as exc:  # pragma: no cover
                return Stage2Result(status="skipped", metrics={}, reason=f"bars load failed: {exc}")

            # Build mapping ts -> close
            try:
                import polars as _pl

                if not isinstance(bars_pl, _pl.DataFrame):
                    bars_pl = _pl.DataFrame(bars_pl)
                bdf = bars_pl.select(
                    [
                        _pl.col("timestamp").cast(_pl.Int64).alias("ts"),
                        _pl.col("close").alias("close"),
                    ],
                )
                ts_list = bdf["ts"].to_list()
                close_list = bdf["close"].to_list()
                m_ts_to_close = dict(zip(ts_list, close_list))
            except Exception:
                bpdf = (
                    bars_pl.to_pandas()
                    if hasattr(bars_pl, "to_pandas")
                    else pd_mod.DataFrame(bars_pl)
                )
                bpdf["ts"] = pd_mod.to_datetime(bpdf["timestamp"]).astype("int64")
                m_ts_to_close = dict(zip(bpdf["ts"].tolist(), bpdf["close"].tolist()))

            for ts in ts_ns.tolist():
                c0 = m_ts_to_close.get(int(ts))
                c1 = m_ts_to_close.get(int(ts + horizon_ns))
                realized.append(
                    0.0 if (c0 is None or c1 is None or c0 == 0) else float((c1 - c0) / c0),
                )

        # Align and compute strategy returns
        if len(realized) != int(q_val.size):
            realized = realized[-int(q_val.size) :]
        returns = np.asarray(realized, dtype=np.float64)
        signals = np.where(q_val >= 0.5, 1.0, -1.0)

        # Apply cost model (bps) on entries and turns
        costs = np.zeros_like(signals)
        # Combine cost components (bps) into a single effective rate
        eff_bps = (
            float(getattr(cfg, "cost_bps", 0.0) or 0.0)
            + float(getattr(cfg, "commission_bps", 0.0) or 0.0)
            + float(getattr(cfg, "slippage_bps", 0.0) or 0.0)
        )
        if eff_bps > 0.0:
            bp = float(eff_bps) / 10_000.0
            costs += (abs(signals) > 0).astype(np.float64) * bp
            turns = np.abs(np.diff(signals, prepend=0.0))
            costs += (turns > 0).astype(np.float64) * bp
        strat_ret = returns * signals - costs
        mask = np.isfinite(strat_ret)
        strat_ret = strat_ret[mask]
        if strat_ret.size == 0:
            return Stage2Result(status="skipped", metrics={}, reason="no finite strategy returns")

        # Metrics
        import math

        mu = float(np.mean(strat_ret))
        sigma = float(np.std(strat_ret))
        n = int(strat_ret.size)
        periods_per_year = (
            252.0 * 390.0
            if int(cfg.horizon_minutes) <= 1
            else 252.0 * (390.0 / int(cfg.horizon_minutes))
        )
        sharpe = float((math.sqrt(periods_per_year) * mu / sigma) if sigma > 0 else 0.0)
        cum = np.cumprod(1.0 + strat_ret)
        run_max = np.maximum.accumulate(cum)
        dd = (cum - run_max) / run_max
        max_dd = float(abs(np.min(dd))) if dd.size > 0 else 0.0
        ann_return = float((cum[-1] ** (periods_per_year / max(n, 1))) - 1.0) if n > 0 else 0.0
        calmar = float((ann_return / max_dd) if max_dd > 0 else 0.0)
        t_stat = float((math.sqrt(n) * mu / sigma) if sigma > 0 else 0.0)

        metrics = {
            "sharpe_ratio": sharpe,
            "calmar_ratio": calmar,
            "t_stat": t_stat,
            "max_drawdown": max_dd,
            "annualized_return": ann_return,
            "mean_return": mu,
            "volatility": sigma,
        }
        return Stage2Result(status="passed", metrics=metrics)


class BacktestStage2EngineRunner:
    """
    Advisory backtest engine.

    Validates environment and may fall back in promotions.

    """

    def run(
        self,
        cfg: _Stage2Config,
    ) -> Stage2Result:  # pragma: no cover - exercised via integration
        # Load arrays and validation tail
        arrays = _load_validation_arrays(cfg.out_dir)
        if arrays is None:
            return Stage2Result(status="skipped", metrics={}, reason="teacher_preds.npz missing")
        q_val, _ = arrays
        if q_val.size == 0:
            return Stage2Result(status="skipped", metrics={}, reason="empty validation window")
        tail = _load_validation_tail(cfg.dataset_csv, int(q_val.size))
        if tail is None:
            return Stage2Result(status="skipped", metrics={}, reason="dataset tail missing cols")
        df_val, pd_mod = tail
        df_val = cast(Any, df_val)
        pd_mod = cast(Any, pd_mod)

        # Build per-instrument q series mapping
        try:
            df_val = df_val.reset_index(drop=True)
            insts = df_val["instrument_id"].tolist()
            ts_list = pd_mod.to_datetime(df_val["timestamp"]).astype("int64").tolist()
            q_list = q_val.tolist()
            q_map: dict[str, dict[int, float]] = {}
            for inst, ts, q in zip(insts, ts_list, q_list):
                m = q_map.setdefault(str(inst), {})
                m[int(ts)] = float(q)
        except Exception as exc:
            return Stage2Result(status="skipped", metrics={}, reason=f"q mapping failed: {exc}")

        # Import backtest engine and required types
        try:
            from nautilus_trader.backtest.engine import BacktestEngine
            from nautilus_trader.backtest.engine import BacktestEngineConfig
            from nautilus_trader.model.data import Bar
            from nautilus_trader.model.data import BarSpecification
            from nautilus_trader.model.data import BarType
            from nautilus_trader.model.identifiers import TraderId
            from nautilus_trader.model.identifiers import Venue
            from nautilus_trader.model.objects import Money
            from nautilus_trader.model.objects import Price
            from nautilus_trader.model.objects import Quantity

            from nautilus_trader.config import LoggingConfig
            from nautilus_trader.model.currencies import USD
            from nautilus_trader.model.enums import AccountType
            from nautilus_trader.model.enums import BarAggregation
            from nautilus_trader.model.enums import OmsType
            from nautilus_trader.model.enums import PriceType
            from nautilus_trader.test_kit.providers import TestInstrumentProvider
        except Exception as exc:
            return Stage2Result(
                status="skipped",
                metrics={},
                reason=f"backtest engine unavailable: {exc}",
            )

        # Build engine (use MARGIN to allow shorting)
        engine = BacktestEngine(
            config=BacktestEngineConfig(
                trader_id=TraderId("STAGE2"),
                logging=LoggingConfig(log_level="ERROR"),
            ),
        )
        venue = Venue("SIM")
        starting_money = Money(100_000, USD)
        engine.add_venue(
            venue=venue,
            oms_type=OmsType.HEDGING,
            account_type=AccountType.MARGIN,
            base_currency=USD,
            starting_balances=[starting_money],
        )

        # Add instruments present in the validation tail
        unique_insts = sorted({str(inst) for inst in insts})
        inst_obj_map: dict[str, Any] = {}
        for inst in unique_insts:
            # Parse symbol + venue
            if "." in inst:
                symbol, ven = inst.split(".", 1)
            else:
                symbol, ven = inst, "XNAS"
            try:
                instrument = TestInstrumentProvider.equity(symbol=symbol, venue=ven)
            except Exception:
                # Fallback to venue SIM if unknown
                instrument = TestInstrumentProvider.equity(symbol=symbol, venue="SIM")
            inst_obj_map[inst] = instrument
            engine.add_instrument(instrument)

        # Load bars from catalog and replay into engine
        from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog as _Cat

        cat = _Cat(str(cfg.data_dir))
        # Define bar types per instrument (1-minute LAST)
        bar_types: dict[str, Any] = {}
        for inst, instrument in inst_obj_map.items():
            spec = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
            bar_types[inst] = BarType(instrument.id, spec)

        # Determine min/max timestamps of the tail
        ts_tail_min = min(ts_list)
        ts_tail_max = max(ts_list)

        # Replay bars; collect per-instrument sequence covering [ts_tail_min, ts_tail_max]
        to_replay: list[Bar] = []
        for inst, instrument in inst_obj_map.items():
            try:
                # Catalog accepts InstrumentId; use API to read a wide window and filter
                bars_iter = cat.bars(
                    instrument_ids=[instrument.id],
                    start=ts_tail_min,
                    end=ts_tail_max,
                )
                # cat.bars returns iterable of Bar; ensure list
                seq = list(bars_iter)
                # If catalog returned objects not Bar, reconstruct from dict-like (fallback)
                if seq and not isinstance(seq[0], Bar):
                    seq = []
            except Exception:
                # Fallback: attempt via ml.data.catalog_utils
                from datetime import datetime as _dt

                from ml.data.catalog_utils import bars_to_dataframe as _bars_to_df

                start_dt = _dt.fromtimestamp(ts_tail_min / 1e9, tz=UTC)
                end_dt = _dt.fromtimestamp(ts_tail_max / 1e9, tz=UTC)
                df = _bars_to_df(cat, [str(instrument.id)], start=start_dt, end=end_dt)
                try:
                    # Normalize to pandas-like for iteration
                    if hasattr(df, "to_pandas"):
                        pdf = df.to_pandas()
                        cols = list(pdf.columns)
                        cix = {name: i for i, name in enumerate(cols)}
                        for r in pdf.itertuples(index=False, name=None):
                            ts_ns = int(pd_mod.to_datetime(r[cix["timestamp"]]).to_datetime64())
                            to_replay.append(
                                Bar(
                                    bar_type=bar_types[inst],
                                    open=Price.from_double(float(r[cix["open"]])),
                                    high=Price.from_double(float(r[cix["high"]])),
                                    low=Price.from_double(float(r[cix["low"]])),
                                    close=Price.from_double(float(r[cix["close"]])),
                                    volume=Quantity.from_double(
                                        float(r[cix.get("volume", -1)] if "volume" in cix else 0.0),
                                    ),
                                    ts_event=ts_ns,
                                    ts_init=ts_ns,
                                ),
                            )
                    else:
                        # Assume records() like iteration
                        try:
                            for row in df.iter_rows(named=True):
                                ts_ns = int(pd_mod.to_datetime(row["timestamp"]).to_datetime64())
                                to_replay.append(
                                    Bar(
                                        bar_type=bar_types[inst],
                                        open=Price.from_double(float(row["open"])),
                                        high=Price.from_double(float(row["high"])),
                                        low=Price.from_double(float(row["low"])),
                                        close=Price.from_double(float(row["close"])),
                                        volume=Quantity.from_double(float(row.get("volume", 0.0))),
                                        ts_event=ts_ns,
                                        ts_init=ts_ns,
                                    ),
                                )
                        except Exception:
                            return Stage2Result(
                                status="skipped",
                                metrics={},
                                reason="unsupported df iterator",
                            )
                except Exception as exc2:
                    return Stage2Result(
                        status="skipped",
                        metrics={},
                        reason=f"bar replay failed: {exc2}",
                    )
                continue

            to_replay.extend(seq)

        # Minimal strategy that submits orders on side changes and builds equity snapshots from portfolio PnL
        from nautilus_trader.core.correctness import UUID4
        from nautilus_trader.model.identifiers import InstrumentId
        from nautilus_trader.model.objects import Price
        from nautilus_trader.model.orders.market import MarketOrder
        from nautilus_trader.trading.strategy import Strategy as _RuntimeStrategy

        from nautilus_trader.model.enums import OrderSide
        from nautilus_trader.model.enums import TimeInForce

        strategy_base = cast(type[Any], _RuntimeStrategy)

        class _QThresholdStrategy(strategy_base):  # type: ignore[misc,valid-type]
            def __init__(self, qmap: dict[str, dict[int, float]], starting_balance: float) -> None:
                super().__init__()
                self._qmap = qmap
                self._last_close: dict[str, float] = {}
                self._last_side: dict[str, int] = {}
                self._equity: list[float] = [float(starting_balance)]
                self._flip_marks: list[int] = []

            def on_bar(self, bar: Any) -> None:
                inst_id_obj = bar.bar_type.instrument_id
                inst = str(inst_id_obj)
                ts = int(bar.ts_event)
                close = (
                    float(bar.close.as_double())
                    if hasattr(bar.close, "as_double")
                    else float(bar.close)
                )
                q = self._qmap.get(inst, {}).get(ts)
                desired = 1 if (q is not None and q >= 0.5) else -1
                current = self._last_side.get(inst, 0)

                # Side change → submit market order for delta quantity
                flip = 0
                if desired != current:
                    delta = desired - current
                    try:
                        qty = Quantity.from_double(abs(float(delta)))
                        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                        order = MarketOrder(
                            trader_id=self.trader_id,
                            strategy_id=self.id,
                            instrument_id=inst_id_obj,
                            client_order_id=self.cache.client_order_id(),
                            order_side=side,
                            quantity=qty,
                            init_id=UUID4(),
                            ts_init=self.clock.timestamp_ns(),
                            time_in_force=TimeInForce.GTC,
                            reduce_only=False,
                        )
                        self.submit_order(order)
                    except Exception:
                        logger.debug(
                            "stage2_engine.order_submit_failed instrument=%s delta=%s",
                            inst,
                            delta,
                            exc_info=True,
                            extra={"instrument_id": inst, "delta": float(delta)},
                        )
                    self._last_side[inst] = desired
                    flip = 1

                # Update last close
                self._last_close[inst] = close

                # Equity snapshot from portfolio PnL (sum over instruments seen so far)
                try:
                    total_pnl = 0.0
                    for k, last_c in self._last_close.items():
                        price_obj = Price.from_double(float(last_c))
                        try:
                            iid = inst_id_obj if k == inst else InstrumentId.from_str(k)
                        except Exception:
                            iid = inst_id_obj
                        if getattr(self, "portfolio", None) is not None:
                            m = self.portfolio.total_pnl(iid, price_obj)
                            try:
                                total_pnl += float(m)
                            except Exception:
                                total_pnl += (
                                    float(m.as_double()) if hasattr(m, "as_double") else 0.0
                                )
                    equity = float(starting_money) + float(total_pnl)
                    self._equity.append(equity)
                    self._flip_marks.append(flip)
                except Exception:
                    logger.debug(
                        "stage2_engine.equity_snapshot_failed instrument=%s",
                        inst,
                        exc_info=True,
                        extra={"instrument_id": inst},
                    )

        strat = _QThresholdStrategy(q_map, float(starting_money))
        engine.add_strategy(strat)

        # Replay
        if to_replay:
            engine.add_data(to_replay)
        try:
            engine.run()
        except Exception as exc:
            return Stage2Result(status="skipped", metrics={}, reason=f"engine run failed: {exc}")

        # Compute metrics from equity snapshots → returns
        eq = getattr(strat, "_equity", [])
        flips = getattr(strat, "_flip_marks", [])
        if not eq or len(eq) < 2:
            return Stage2Result(
                status="skipped",
                metrics={},
                reason="no equity snapshots collected",
            )
        arr = np.diff(np.asarray(eq, dtype=np.float64)) / np.asarray(eq[:-1], dtype=np.float64)
        # Apply cost on flips
        try:
            total_bps = (
                float(getattr(cfg, "cost_bps", 0.0) or 0.0)
                + float(getattr(cfg, "commission_bps", 0.0) or 0.0)
                + float(getattr(cfg, "slippage_bps", 0.0) or 0.0)
            )
            bp = float(total_bps) / 10_000.0
            if flips and len(flips) == arr.size:
                arr = arr - (np.asarray(flips, dtype=np.float64) * bp)
        except Exception:
            logger.debug(
                "stage2_engine.cost_adjustment_failed",
                exc_info=True,
                extra={
                    "cost_bps": getattr(cfg, "cost_bps", None),
                    "commission_bps": getattr(cfg, "commission_bps", None),
                    "slippage_bps": getattr(cfg, "slippage_bps", None),
                },
            )
        import math

        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        n = int(arr.size)
        periods_per_year = (
            252.0 * 390.0
            if int(cfg.horizon_minutes) <= 1
            else 252.0 * (390.0 / int(cfg.horizon_minutes))
        )
        sharpe = float((math.sqrt(periods_per_year) * mu / sigma) if sigma > 0 else 0.0)
        cum = np.cumprod(1.0 + arr)
        run_max = np.maximum.accumulate(cum)
        dd = (cum - run_max) / run_max
        max_dd = float(abs(np.min(dd))) if dd.size > 0 else 0.0
        ann_return = float((cum[-1] ** (periods_per_year / max(n, 1))) - 1.0) if n > 0 else 0.0
        calmar = float((ann_return / max_dd) if max_dd > 0 else 0.0)
        t_stat = float((math.sqrt(n) * mu / sigma) if sigma > 0 else 0.0)

        metrics = {
            "sharpe_ratio": sharpe,
            "calmar_ratio": calmar,
            "t_stat": t_stat,
            "max_drawdown": max_dd,
            "annualized_return": ann_return,
            "mean_return": mu,
            "volatility": sigma,
        }
        return Stage2Result(status="passed", metrics=metrics)


_BACKTEST_ENABLED = os.environ.get("ML_ENABLE_STAGE2_BACKTEST", "0") == "1"


def build_engine(mode: Literal["returns", "backtest"]) -> Stage2Engine:
    if mode == "backtest" and _BACKTEST_ENABLED:
        return BacktestStage2EngineRunner()
    return ReturnsStage2Engine()
logger = logging.getLogger(__name__)
