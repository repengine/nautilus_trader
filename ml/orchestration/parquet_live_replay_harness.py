"""
Parquet live replay harness for end-to-end pipeline validation.

This harness replays historical bars from a Parquet catalog through the
MLSignalActor + MLTradingStrategy hot path using the BacktestEngine. It is a
fast/TestClock-style execution (no live pacing). Live-paced replay remains
deferred until the fast path is validated.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.engine import BacktestEngineConfig
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity

from ml.actors import MLSignalActor
from ml.config.actors import MLSignalActorConfig
from ml.config.base import AccountMode
from ml.config.base import ExitPolicyConfig
from ml.config.base import MLStrategyConfig
from ml.config.replay_harness import ActorReplayConfig
from ml.config.replay_harness import ParquetLiveReplayHarnessConfig
from ml.config.replay_harness import StrategyReplayConfig
from ml.strategies.execution import ExecutionConfig
from ml.strategies.ml_strategy import MLTradingStrategy
from ml.strategies.risk import RiskConfig
from nautilus_trader.backtest.results import BacktestResult
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.instruments import Equity
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ParquetLiveReplayHarnessResult:
    """
    Summary result for a replay harness run.

    Attributes
    ----------
    run_id : str
        Identifier for the replay run.
    instrument_ids : tuple[str, ...]
        Instruments included in the replay.
    bars_loaded : int
        Total bars loaded from the Parquet catalog.
    quote_ticks_loaded : int
        Total quote ticks loaded from the Parquet catalog.
    output_path : Path | None
        File store output path, if configured.
    """

    run_id: str
    instrument_ids: tuple[str, ...]
    bars_loaded: int
    quote_ticks_loaded: int
    output_path: Path | None


def run_parquet_live_replay_harness(
    config: ParquetLiveReplayHarnessConfig,
) -> ParquetLiveReplayHarnessResult:
    """
    Run the parquet live replay harness.

    Args:
        config: Replay harness configuration.

    Returns:
        ParquetLiveReplayHarnessResult summarizing the run.

    Example:
        >>> from ml.config.replay_harness import ParquetLiveReplayHarnessConfig
        >>> cfg = ParquetLiveReplayHarnessConfig(
        ...     catalog_path="data/catalog",
        ...     instrument_ids=["SPY.XNAS"],
        ...     model_id="demo",
        ...     model_path="ml_registry/models/demo.onnx",
        ... )
        >>> result = run_parquet_live_replay_harness(cfg)
        >>> assert result.bars_loaded >= 0
    """
    run_id = config.run_id or f"replay_{time.time_ns()}"
    output_path = _resolve_output_path(config, run_id)
    _configure_environment(config, output_path)

    if config.strategy.serialize_order_intents and not config.strategy.execute_trades:
        logger.warning(
            "serialize_order_intents_enabled_but_execute_trades_false",
            extra={"run_id": run_id},
        )
    if config.strategy.serialize_order_intents:
        logger.warning(
            "serialize_order_intents_bypasses_fills",
            extra={
                "run_id": run_id,
                "note": "disable to validate fills/exits; enable quote ticks for pricing",
            },
        )

    instrument_ids = _normalize_instrument_ids(
        config.instrument_ids,
        fallback_venue=config.fallback_venue,
    )
    bar_spec = BarSpecification.from_str(config.bar_spec)
    bar_types = {inst: BarType(inst, bar_spec) for inst in instrument_ids}

    catalog = ParquetDataCatalog(config.catalog_path)
    bars = _load_bars(catalog, bar_types, config.start_time, config.end_time)
    if not bars:
        raise ValueError("No bars loaded for replay; check catalog path and bar_spec")

    quote_ticks: list[QuoteTick] = []
    if config.strategy.subscribe_quote_ticks:
        quote_ticks = _load_quote_ticks(
            catalog=catalog,
            instrument_ids=instrument_ids,
            start_time=config.start_time,
            end_time=config.end_time,
        )

    price_precisions = _infer_price_precisions(bars, quote_ticks)
    instruments = _resolve_instruments(
        catalog=catalog,
        instrument_ids=instrument_ids,
        fallback_venue=Venue(config.fallback_venue),
        price_precisions=price_precisions,
    )

    engine = _build_engine(config, instruments)
    if quote_ticks:
        engine.add_data(quote_ticks, sort=True)

    engine.add_data(bars, sort=True)

    _attach_components(engine, config, instrument_ids, bar_types)

    try:
        engine.run(start=config.start_time, end=config.end_time)
        backtest_result = engine.get_result()
        _persist_backtest_result(backtest_result, output_path)
    finally:
        try:
            engine.dispose()
        except Exception:
            logger.debug("replay_engine_dispose_failed", exc_info=True)

    return ParquetLiveReplayHarnessResult(
        run_id=run_id,
        instrument_ids=tuple(str(inst) for inst in instrument_ids),
        bars_loaded=len(bars),
        quote_ticks_loaded=len(quote_ticks),
        output_path=output_path,
    )


def _resolve_output_path(
    config: ParquetLiveReplayHarnessConfig,
    run_id: str,
) -> Path | None:
    """
    Resolve the file store output path for a run.
    """
    if config.output_dir is None:
        return None
    base = Path(config.output_dir)
    output_path = base / run_id if run_id else base
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error(
            "replay_output_path_unavailable",
            exc_info=True,
            extra={"error": str(exc), "output_path": str(output_path)},
        )
        raise
    return output_path


def _configure_environment(
    config: ParquetLiveReplayHarnessConfig,
    output_path: Path | None,
) -> None:
    """
    Apply environment overrides required for replay execution.
    """
    if output_path is not None:
        os.environ["ML_FILE_STORE_PATH"] = str(output_path)
    if config.allow_parquet_fallback:
        os.environ.setdefault("ML_TFT_ALLOW_PARQUET_FALLBACK", "1")


def _persist_backtest_result(
    result: BacktestResult,
    output_path: Path | None,
) -> None:
    """
    Persist a JSON summary of the backtest results to the output directory.
    """
    if output_path is None:
        return
    result_path = output_path / "backtest_result.json"
    payload = asdict(result)
    try:
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
    except Exception as exc:
        logger.error(
            "replay_backtest_result_write_failed",
            exc_info=True,
            extra={"error": str(exc), "output_path": str(result_path)},
        )
        raise
    logger.info(
        "replay_backtest_result_written",
        extra={"output_path": str(result_path)},
    )


def _normalize_instrument_ids(
    instrument_ids: Iterable[str],
    *,
    fallback_venue: str,
) -> tuple[InstrumentId, ...]:
    """
    Normalize and validate instrument identifiers, adding a fallback venue when missing.
    """
    seen: set[str] = set()
    results: list[InstrumentId] = []
    for raw in instrument_ids:
        text = raw.strip()
        if not text:
            raise ValueError("instrument_ids cannot contain empty values")
        if "." not in text:
            text = f"{text}.{fallback_venue}"
        if text in seen:
            continue
        try:
            inst = InstrumentId.from_str(text)
        except ValueError as exc:
            raise ValueError(f"Invalid instrument_id: {text}") from exc
        results.append(inst)
        seen.add(text)
    if not results:
        raise ValueError("instrument_ids must contain at least one entry")
    return tuple(results)


def _resolve_instruments(
    catalog: ParquetDataCatalog,
    instrument_ids: Iterable[InstrumentId],
    *,
    fallback_venue: Venue,
    price_precisions: dict[InstrumentId, int] | None = None,
) -> dict[InstrumentId, object]:
    """
    Resolve instrument definitions from the catalog, with a fallback instrument.
    """
    resolved: dict[InstrumentId, object] = {}
    precision_map = price_precisions or {}
    for inst in instrument_ids:
        instrument = _try_catalog_instrument(catalog, inst)
        if instrument is None:
            symbol, venue = _split_symbol_and_venue(str(inst), str(fallback_venue))
            try:
                price_precision = precision_map.get(inst, 2)
                instrument = _build_fallback_equity(
                    inst,
                    symbol=symbol,
                    venue=venue,
                    price_precision=price_precision,
                )
                logger.info(
                    "replay_instrument_fallback",
                    extra={
                        "instrument_id": str(inst),
                        "venue": venue,
                        "price_precision": price_precision,
                    },
                )
            except Exception as exc:
                logger.error(
                    "replay_instrument_fallback_failed",
                    exc_info=True,
                    extra={"error": str(exc), "instrument_id": str(inst)},
                )
                raise ValueError(f"Unable to resolve instrument {inst}") from exc
        resolved[inst] = instrument
    return resolved


def _try_catalog_instrument(
    catalog: ParquetDataCatalog,
    instrument_id: InstrumentId,
) -> object | None:
    """
    Attempt to load instrument metadata from the Parquet catalog.
    """
    try:
        instruments = catalog.instruments(instrument_ids=[str(instrument_id)])
        if instruments:
            instrument: object = instruments[0]
            return instrument
    except Exception as exc:
        logger.debug(
            "replay_catalog_instrument_lookup_failed",
            exc_info=True,
            extra={"error": str(exc), "instrument_id": str(instrument_id)},
        )
    return None


def _split_symbol_and_venue(
    instrument_id: str,
    fallback_venue: str,
) -> tuple[str, str]:
    """
    Split an instrument identifier into symbol and venue.
    """
    if "." in instrument_id:
        symbol, venue = instrument_id.split(".", 1)
        if symbol and venue:
            return symbol, venue
    return instrument_id, fallback_venue


def _infer_price_precisions(
    bars: Iterable[Bar],
    quote_ticks: Iterable[QuoteTick],
) -> dict[InstrumentId, int]:
    """
    Infer the maximum price precision per instrument from bars and quote ticks.
    """
    precisions: dict[InstrumentId, int] = {}
    for bar in bars:
        instrument_id = bar.bar_type.instrument_id
        _update_price_precision(precisions, instrument_id, bar.open)
        _update_price_precision(precisions, instrument_id, bar.high)
        _update_price_precision(precisions, instrument_id, bar.low)
        _update_price_precision(precisions, instrument_id, bar.close)
    for tick in quote_ticks:
        instrument_id = tick.instrument_id
        _update_price_precision(precisions, instrument_id, tick.bid_price)
        _update_price_precision(precisions, instrument_id, tick.ask_price)
    return precisions


def _update_price_precision(
    precisions: dict[InstrumentId, int],
    instrument_id: InstrumentId,
    price: Price | None,
) -> None:
    """
    Update the tracked price precision for an instrument.
    """
    if price is None:
        return
    precision = getattr(price, "precision", None)
    if precision is None:
        return
    current = precisions.get(instrument_id)
    new_precision = int(precision)
    if current is None or new_precision > current:
        precisions[instrument_id] = new_precision


def _build_fallback_equity(
    instrument_id: InstrumentId,
    *,
    symbol: str,
    venue: str,
    price_precision: int,
) -> Equity:
    """
    Build an Equity instrument with a precision aligned to the input data.
    """
    price_increment = _price_increment_from_precision(price_precision)
    return Equity(
        instrument_id=instrument_id,
        raw_symbol=Symbol(symbol),
        currency=USD,
        price_precision=price_precision,
        price_increment=price_increment,
        lot_size=Quantity.from_int(1),
        ts_event=0,
        ts_init=0,
    )


def _price_increment_from_precision(price_precision: int) -> Price:
    """
    Build a price increment (tick size) from a precision value.
    """
    if price_precision < 0:
        raise ValueError("price_precision must be >= 0")
    if price_precision == 0:
        return Price.from_str("1")
    zeros = "0" * (price_precision - 1)
    return Price.from_str(f"0.{zeros}1")


def _build_engine(
    config: ParquetLiveReplayHarnessConfig,
    instruments: dict[InstrumentId, object],
) -> BacktestEngine:
    """
    Build a backtest engine with venues and instruments attached.
    """
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId(config.trader_id),
            logging=LoggingConfig(log_level=config.engine_log_level),
        ),
    )

    venue_names = {str(inst.venue) for inst in instruments.keys()}
    account_mode = getattr(config.strategy, "account_mode", AccountMode.CASH)
    account_type = AccountType.MARGIN if account_mode is AccountMode.MARGIN else AccountType.CASH
    starting_balance = Money(float(config.starting_balance), USD)
    for venue_name in venue_names:
        engine.add_venue(
            venue=Venue(venue_name),
            oms_type=OmsType.NETTING,
            account_type=account_type,
            base_currency=USD,
            starting_balances=[starting_balance],
        )

    for instrument in instruments.values():
        engine.add_instrument(instrument)

    return engine


def _load_bars(
    catalog: ParquetDataCatalog,
    bar_types: dict[InstrumentId, BarType],
    start_time: str | int | None,
    end_time: str | int | None,
) -> list[Bar]:
    """
    Load bars from the catalog for the given bar types.
    """
    bars: list[Bar] = []
    missing: list[str] = []
    for bar_type in bar_types.values():
        bar_type_str = str(bar_type)
        try:
            seq = catalog.bars(
                bar_types=[bar_type_str],
                start=start_time,
                end=end_time,
            )
        except Exception as exc:
            logger.error(
                "replay_catalog_read_failed",
                exc_info=True,
                extra={"error": str(exc), "bar_type": bar_type_str},
            )
            raise
        if not seq:
            missing.append(bar_type_str)
            continue
        bars.extend(seq)

    if missing:
        logger.warning(
            "replay_catalog_bars_missing",
            extra={"bar_types": tuple(missing)},
        )

    bars.sort(key=lambda bar: bar.ts_event)
    return bars


def _load_quote_ticks(
    catalog: ParquetDataCatalog,
    instrument_ids: Iterable[InstrumentId],
    start_time: str | int | None,
    end_time: str | int | None,
) -> list[QuoteTick]:
    """
    Load quote ticks from the catalog for the given instruments.
    """
    quote_ticks: list[QuoteTick] = []
    missing: list[str] = []
    for instrument_id in instrument_ids:
        instrument_id_str = str(instrument_id)
        try:
            seq = catalog.quote_ticks(
                instrument_ids=[instrument_id_str],
                start=start_time,
                end=end_time,
            )
        except Exception as exc:
            logger.error(
                "replay_catalog_quotes_read_failed",
                exc_info=True,
                extra={"error": str(exc), "instrument_id": instrument_id_str},
            )
            raise
        if not seq:
            missing.append(instrument_id_str)
            continue
        quote_ticks.extend(seq)

    if missing:
        logger.warning(
            "replay_catalog_quote_ticks_missing",
            extra={"instrument_ids": tuple(missing)},
        )

    quote_ticks.sort(key=lambda tick: tick.ts_event)
    return quote_ticks


def _build_strategy_config(
    *,
    strategy_config: StrategyReplayConfig,
    instrument_id: InstrumentId,
    actor_id: str,
    strategy_id: str,
) -> MLStrategyConfig:
    """
    Build MLStrategyConfig for the replay harness.
    """
    risk_config = _resolve_risk_config(strategy_config)
    execution_config: ExecutionConfig | None = None
    if strategy_config.execution_validation_mode is not None:
        execution_config = ExecutionConfig(
            validation_mode=strategy_config.execution_validation_mode,
        )
    exit_policy_config = ExitPolicyConfig(
        stop_loss_pct=strategy_config.stop_loss_pct,
        take_profit_pct=strategy_config.take_profit_pct,
        max_holding_ms=strategy_config.max_holding_ms,
    )
    return MLStrategyConfig(
        strategy_id=strategy_id,
        instrument_id=instrument_id,
        ml_signal_source=actor_id,
        position_size_pct=strategy_config.position_size_pct,
        min_confidence=strategy_config.min_confidence,
        max_positions=strategy_config.max_positions,
        account_mode=strategy_config.account_mode,
        short_entry_policy=strategy_config.short_entry_policy,
        stop_loss_pct=strategy_config.stop_loss_pct,
        take_profit_pct=strategy_config.take_profit_pct,
        exit_policy_config=exit_policy_config,
        model_exit_config=strategy_config.model_exit_config,
        risk_config=risk_config,
        execution_config=execution_config,
        use_strategy_store=strategy_config.use_strategy_store,
        persist_all_signals=strategy_config.persist_all_signals,
        execute_trades=strategy_config.execute_trades,
        serialize_order_intents=strategy_config.serialize_order_intents,
        order_intent_path=strategy_config.order_intent_path,
        subscribe_quote_ticks=strategy_config.subscribe_quote_ticks,
        quote_schema=strategy_config.quote_schema,
        max_quote_age_ms=strategy_config.max_quote_age_ms,
    )


def _build_actor_config(
    *,
    actor_config: ActorReplayConfig,
    model_id: str,
    model_path: str,
    bar_type: BarType,
    instrument_id: InstrumentId,
    actor_id: str,
) -> MLSignalActorConfig:
    """
    Build MLSignalActorConfig for the replay harness.
    """
    signal_strategy = actor_config.signal_strategy or "threshold"
    min_signal_separation_bars = actor_config.min_signal_separation_bars or 3
    return MLSignalActorConfig(
        model_id=model_id,
        model_path=model_path,
        bar_type=bar_type,
        instrument_id=instrument_id,
        component_id=actor_id,
        actor_id=actor_id,
        prediction_threshold=actor_config.prediction_threshold,
        warm_up_period=actor_config.warm_up_period,
        publish_signals=actor_config.publish_signals,
        log_predictions=actor_config.log_predictions,
        signal_strategy=signal_strategy,
        min_signal_separation_bars=min_signal_separation_bars,
        feature_config=actor_config.feature_config,
        feature_set_id=actor_config.feature_set_id,
        registry_path=actor_config.registry_path,
        use_registry_features=actor_config.use_registry_features,
        db_connection=actor_config.db_connection,
        use_dummy_stores=actor_config.use_dummy_stores,
    )


def _resolve_risk_config(strategy_config: StrategyReplayConfig) -> RiskConfig | None:
    """
    Resolve the risk configuration for replay runs.
    """
    if strategy_config.risk_config is not None:
        return strategy_config.risk_config
    if strategy_config.liquidation_config is None:
        return None
    return RiskConfig(
        allow_reduce_only_when_halted=strategy_config.allow_reduce_only_when_halted,
        liquidation_config=strategy_config.liquidation_config,
    )


def _attach_components(
    engine: BacktestEngine,
    config: ParquetLiveReplayHarnessConfig,
    instrument_ids: Iterable[InstrumentId],
    bar_types: dict[InstrumentId, BarType],
) -> None:
    """
    Attach ML actors and strategies for each instrument.
    """
    for inst in instrument_ids:
        actor_id = f"{config.actor.component_id_prefix}-{inst}"
        strategy_id = f"{config.strategy.id_prefix}-{inst}"
        bar_type = bar_types[inst]

        actor_config = _build_actor_config(
            actor_config=config.actor,
            model_id=config.model_id,
            model_path=config.model_path,
            bar_type=bar_type,
            instrument_id=inst,
            actor_id=actor_id,
        )

        strategy_config = _build_strategy_config(
            strategy_config=config.strategy,
            instrument_id=inst,
            actor_id=actor_id,
            strategy_id=strategy_id,
        )

        actor = MLSignalActor(config=actor_config)
        strategy = MLTradingStrategy(config=strategy_config)

        engine.add_actor(actor)
        engine.add_strategy(strategy)

        actor.subscribe_bars(bar_type)
        strategy.subscribe_bars(bar_type)


__all__ = [
    "ParquetLiveReplayHarnessResult",
    "run_parquet_live_replay_harness",
]
