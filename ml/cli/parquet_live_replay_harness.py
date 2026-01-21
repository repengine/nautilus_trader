#!/usr/bin/env python3
"""Run the parquet live replay harness (fast/TestClock mode)."""

from __future__ import annotations

import argparse
import logging

from ml.common.logging_config import configure_logging
from ml.config.replay_harness import ActorReplayConfig
from ml.config.replay_harness import ParquetLiveReplayHarnessConfig
from ml.config.replay_harness import StrategyReplayConfig
from ml.features.config import FeatureConfig
from ml.orchestration.parquet_live_replay_harness import run_parquet_live_replay_harness


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
        "--execute-trades",
        action="store_true",
        help="Execute trades (required to emit order intents).",
    )
    parser.add_argument(
        "--serialize-order-intents",
        action="store_true",
        help="Serialize order intents to JSONL instead of broker submission.",
    )
    parser.add_argument(
        "--order-intent-path",
        default=None,
        help="Explicit JSONL output path for order intents.",
    )
    parser.add_argument(
        "--subscribe-quote-ticks",
        action="store_true",
        help="Subscribe to quote ticks for execution market state.",
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
        feature_config=feature_config,
        feature_set_id=args.feature_set_id,
        registry_path=args.registry_path,
        use_registry_features=args.use_registry_features,
    )
    strategy_config = StrategyReplayConfig(
        id_prefix="MLStrategy",
        position_size_pct=args.position_size_pct,
        min_confidence=args.min_confidence,
        max_positions=args.max_positions,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        persist_all_signals=args.persist_all_signals,
        execute_trades=args.execute_trades,
        serialize_order_intents=args.serialize_order_intents,
        order_intent_path=args.order_intent_path,
        subscribe_quote_ticks=args.subscribe_quote_ticks,
        quote_schema=args.quote_schema,
        max_quote_age_ms=args.max_quote_age_ms,
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
