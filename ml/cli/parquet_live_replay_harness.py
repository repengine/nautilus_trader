#!/usr/bin/env python3
"""Run the parquet live replay harness (fast/TestClock mode)."""

from __future__ import annotations

import argparse
import logging

from ml.common.logging_config import configure_logging
from ml.config.base import AccountMode
from ml.config.base import ExecutionValidationMode
from ml.config.base import ModelExitConfig
from ml.config.base import ShortEntryPolicy
from ml.config.replay_harness import ActorReplayConfig
from ml.config.replay_harness import ParquetLiveReplayHarnessConfig
from ml.config.replay_harness import StrategyReplayConfig
from ml.features.config import FeatureConfig
from ml.orchestration.parquet_live_replay_harness import run_parquet_live_replay_harness
from ml.strategies.risk import RiskConfig
from ml.strategies.risk import RiskLiquidationConfig


logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay Parquet catalog bars through ML actor + strategy.",
    )
    parser.add_argument(
        "--catalog-path",
        required=True,
        help="Path to the Parquet data catalog.",
    )
    parser.add_argument(
        "--instrument-id",
        action="append",
        required=True,
        help="Instrument ID to replay (repeat for multiple).",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model identifier for MLSignalActor.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="ONNX model artifact path.",
    )
    parser.add_argument(
        "--bar-spec",
        default="1-MINUTE-LAST",
        help="Bar specification string (default: 1-MINUTE-LAST).",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Optional start time (ISO string or nanoseconds).",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="Optional end time (ISO string or nanoseconds).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier for outputs/logging.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base output directory for JSONL stores.",
    )
    parser.add_argument(
        "--fallback-venue",
        default="SIM",
        help="Fallback venue for instrument IDs without venue segments.",
    )
    parser.add_argument(
        "--engine-log-level",
        default="INFO",
        help="Backtest engine log level (default: INFO).",
    )
    parser.add_argument(
        "--lookback-window",
        type=int,
        default=None,
        help="Optional lookback window override for features.",
    )
    parser.add_argument(
        "--warm-up-period",
        type=int,
        default=50,
        help="Warm-up period before predictions (default: 50).",
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.5,
        help="Prediction threshold for MLSignalActor.",
    )
    parser.add_argument(
        "--signal-strategy",
        choices=["threshold", "extremes", "momentum", "ensemble", "adaptive"],
        default=None,
        help="Signal strategy override (default: MLSignalActorConfig default).",
    )
    parser.add_argument(
        "--min-signal-separation-bars",
        type=int,
        default=None,
        help="Minimum bars between signals (default: actor config default).",
    )
    parser.add_argument(
        "--log-predictions",
        action="store_true",
        help="Log individual model predictions for debugging.",
    )
    parser.add_argument(
        "--disable-publish-signals",
        action="store_true",
        help="Disable publishing ML signals to the message bus.",
    )
    parser.add_argument(
        "--use-dummy-stores",
        action="store_true",
        help="Use dummy stores instead of persistence backends.",
    )
    parser.add_argument(
        "--execute-trades",
        action="store_true",
        help="Execute trades (required to emit order intents).",
    )
    parser.add_argument(
        "--serialize-order-intents",
        action="store_true",
        help=(
            "Serialize order intents to JSONL instead of broker submission. "
            "Use for live safety; bypasses simulated fills in replay."
        ),
    )
    parser.add_argument(
        "--order-intent-path",
        default=None,
        help="Explicit JSONL output path for order intents.",
    )
    parser.add_argument(
        "--subscribe-quote-ticks",
        action="store_true",
        help=(
            "Subscribe to quote ticks for execution market state "
            "(recommended for execution validation and intent pricing)."
        ),
    )
    parser.add_argument(
        "--quote-schema",
        default=None,
        help="Quote schema override (e.g., mbp-1, bbo-1s, bbo-1m).",
    )
    parser.add_argument(
        "--max-quote-age-ms",
        type=int,
        default=None,
        help="Maximum quote age in milliseconds allowed for execution.",
    )
    parser.add_argument(
        "--execution-validation-mode",
        choices=[mode.value for mode in ExecutionValidationMode],
        default=None,
        help=(
            "Replay-only execution validation mode: "
            "disabled, cross_bbo, or market."
        ),
    )
    parser.add_argument(
        "--position-size-pct",
        type=float,
        default=0.1,
        help="Position size percentage (default: 0.1).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence for strategy decisions (default: 0.7).",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=1,
        help="Maximum concurrent positions (default: 1).",
    )
    parser.add_argument(
        "--account-mode",
        choices=[mode.value for mode in AccountMode],
        default=AccountMode.CASH.value,
        help="Account mode for short-entry defaults (default: cash).",
    )
    parser.add_argument(
        "--short-entry-policy",
        choices=[policy.value for policy in ShortEntryPolicy],
        default=None,
        help="Optional short-entry policy override (allow, exit_only, deny).",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.02,
        help="Stop loss percentage (default: 0.02).",
    )
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=0.04,
        help="Take profit percentage (default: 0.04).",
    )
    parser.add_argument(
        "--max-holding-ms",
        type=int,
        default=None,
        help="Maximum holding time in milliseconds before forcing an exit.",
    )
    parser.add_argument(
        "--liquidation-enabled",
        action="store_true",
        help="Enable staged liquidation safeguards.",
    )
    parser.add_argument(
        "--liquidation-daily-loss-limit-pct",
        type=float,
        default=None,
        help="Liquidate when daily loss exceeds this percentage.",
    )
    parser.add_argument(
        "--liquidation-drawdown-limit-pct",
        type=float,
        default=None,
        help="Liquidate when drawdown exceeds this percentage.",
    )
    parser.add_argument(
        "--liquidation-unrealized-loss-limit-pct",
        type=float,
        default=None,
        help="Liquidate when unrealized loss exceeds this percentage.",
    )
    parser.add_argument(
        "--liquidation-cooldown-ms",
        type=int,
        default=None,
        help="Minimum milliseconds between liquidation attempts.",
    )
    parser.add_argument(
        "--liquidation-require-full-positions",
        action="store_true",
        help="Require full positions list before liquidation.",
    )
    parser.add_argument(
        "--disable-reduce-only-when-halted",
        action="store_true",
        help="Disallow reduce-only orders during risk halts.",
    )
    parser.add_argument(
        "--model-exit-on-flip",
        action="store_true",
        help="Enable model-driven exits on prediction flips.",
    )
    parser.add_argument(
        "--model-reverse-on-flip",
        action="store_true",
        help="Reverse on prediction flips when model exits are enabled.",
    )
    parser.add_argument(
        "--model-exit-confidence-threshold",
        type=float,
        default=None,
        help="Exit when model confidence drops below this threshold.",
    )
    parser.add_argument(
        "--model-exit-prediction-band",
        type=float,
        default=None,
        help="Neutral-zone band around 0.5 for model-driven exits.",
    )
    parser.add_argument(
        "--model-exit-min-hold-ms",
        type=int,
        default=None,
        help="Minimum holding time in milliseconds before model exits.",
    )
    parser.add_argument(
        "--persist-all-signals",
        action="store_true",
        help="Persist HOLD signals in addition to BUY/SELL.",
    )
    parser.add_argument(
        "--feature-set-id",
        default=None,
        help="Feature registry set ID (required when using registry features).",
    )
    parser.add_argument(
        "--registry-path",
        default=None,
        help="Registry root path override.",
    )
    parser.add_argument(
        "--use-registry-features",
        action="store_true",
        help="Align inference features to registry manifest.",
    )
    return parser


def _parse_time(value: str | None) -> str | int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def main() -> None:
    configure_logging()
    args = _build_parser().parse_args()

    feature_config = None
    if args.lookback_window is not None:
        feature_config = FeatureConfig(lookback_window=args.lookback_window)

    actor_config = ActorReplayConfig(
        component_id_prefix="MLSignalActor",
        prediction_threshold=args.prediction_threshold,
        warm_up_period=args.warm_up_period,
        publish_signals=not args.disable_publish_signals,
        log_predictions=args.log_predictions,
        signal_strategy=args.signal_strategy,
        min_signal_separation_bars=args.min_signal_separation_bars,
        feature_config=feature_config,
        feature_set_id=args.feature_set_id,
        registry_path=args.registry_path,
        use_registry_features=args.use_registry_features,
        use_dummy_stores=args.use_dummy_stores,
    )
    model_exit_config = None
    if (
        args.model_exit_on_flip
        or args.model_reverse_on_flip
        or args.model_exit_confidence_threshold is not None
        or args.model_exit_prediction_band is not None
        or args.model_exit_min_hold_ms is not None
    ):
        model_exit_config = ModelExitConfig(
            exit_on_flip=args.model_exit_on_flip or args.model_reverse_on_flip,
            reverse_on_flip=args.model_reverse_on_flip,
            exit_confidence_threshold=args.model_exit_confidence_threshold,
            exit_prediction_band=(
                args.model_exit_prediction_band
                if args.model_exit_prediction_band is not None
                else 0.0
            ),
            min_hold_ms=args.model_exit_min_hold_ms,
        )

    liquidation_config = None
    liquidation_thresholds = (
        args.liquidation_daily_loss_limit_pct is not None
        or args.liquidation_drawdown_limit_pct is not None
        or args.liquidation_unrealized_loss_limit_pct is not None
    )
    if (
        args.liquidation_enabled
        or liquidation_thresholds
        or args.liquidation_cooldown_ms is not None
        or args.liquidation_require_full_positions
    ):
        liquidation_config = RiskLiquidationConfig(
            enabled=args.liquidation_enabled or liquidation_thresholds,
            daily_loss_limit_pct=args.liquidation_daily_loss_limit_pct,
            drawdown_limit_pct=args.liquidation_drawdown_limit_pct,
            unrealized_loss_limit_pct=args.liquidation_unrealized_loss_limit_pct,
            cooldown_ms=args.liquidation_cooldown_ms,
            require_full_positions=args.liquidation_require_full_positions,
        )

    risk_config = None
    allow_reduce_only_when_halted = not args.disable_reduce_only_when_halted
    if liquidation_config is not None or args.disable_reduce_only_when_halted:
        risk_config = RiskConfig(
            allow_reduce_only_when_halted=allow_reduce_only_when_halted,
            liquidation_config=liquidation_config,
        )

    account_mode = AccountMode(args.account_mode)
    short_entry_policy = (
        ShortEntryPolicy(args.short_entry_policy) if args.short_entry_policy else None
    )
    execution_validation_mode = (
        ExecutionValidationMode(args.execution_validation_mode)
        if args.execution_validation_mode
        else None
    )

    strategy_config = StrategyReplayConfig(
        id_prefix="MLStrategy",
        position_size_pct=args.position_size_pct,
        min_confidence=args.min_confidence,
        max_positions=args.max_positions,
        account_mode=account_mode,
        short_entry_policy=short_entry_policy,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        max_holding_ms=args.max_holding_ms,
        model_exit_config=model_exit_config,
        risk_config=risk_config,
        persist_all_signals=args.persist_all_signals,
        execute_trades=args.execute_trades,
        serialize_order_intents=args.serialize_order_intents,
        order_intent_path=args.order_intent_path,
        subscribe_quote_ticks=args.subscribe_quote_ticks,
        quote_schema=args.quote_schema,
        max_quote_age_ms=args.max_quote_age_ms,
        execution_validation_mode=execution_validation_mode,
    )

    config = ParquetLiveReplayHarnessConfig(
        catalog_path=args.catalog_path,
        instrument_ids=args.instrument_id,
        model_id=args.model_id,
        model_path=args.model_path,
        bar_spec=args.bar_spec,
        start_time=_parse_time(args.start_time),
        end_time=_parse_time(args.end_time),
        run_id=args.run_id,
        output_dir=args.output_dir,
        fallback_venue=args.fallback_venue,
        engine_log_level=args.engine_log_level,
        actor=actor_config,
        strategy=strategy_config,
    )

    result = run_parquet_live_replay_harness(config)

    logger.info(
        "parquet_live_replay_complete",
        extra={
            "run_id": result.run_id,
            "bars_loaded": result.bars_loaded,
            "quote_ticks_loaded": result.quote_ticks_loaded,
            "output_path": str(result.output_path) if result.output_path else None,
            "instrument_ids": result.instrument_ids,
        },
    )


if __name__ == "__main__":
    main()
