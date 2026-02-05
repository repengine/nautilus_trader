#!/usr/bin/env python3
"""
Command-line entrypoint for earnings ingestion.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.config import WatermarkWindowConfig
from ml.config import earnings_window_defaults
from ml.config.earnings_ingestion import DEFAULT_SKIP_ACTUALS_TICKERS
from ml.config.earnings_ingestion import EarningsIngestionConfig
from ml.config.edgar_smoke import EdgarSmokeTestConfig
from ml.config.sec_identity import SecIdentityConfig
from ml.core.common.registry_initialization import RegistryInitializationComponent
from ml.core.common.store_initialization import StoreInitializationComponent
from ml.features.earnings.ingestion.edgar_smoke import run_edgar_smoke_test
from ml.features.earnings.ingestion.service import EarningsIngestionService
from ml.features.earnings.raw_writer import EarningsParquetRawWriter
from ml.registry.base import DummyRegistry
from ml.stores.protocols import EarningsStoreProtocol
from ml.stores.protocols import FeatureStoreProtocol
from ml.stores.protocols import ModelStoreProtocol
from ml.stores.protocols import StrategyStoreProtocol


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest earnings data from EDGAR and Yahoo Finance.")
    parser.add_argument("--dsn", default=os.getenv("NAUTILUS_DB"), help="PostgreSQL connection string (default: NAUTILUS_DB env).")
    parser.add_argument("--parquet-root", default="data/features/earnings_raw", help="Root directory for Parquet mirrors.")
    parser.add_argument("--universe-mode", default="postgres", choices=("postgres", "tier1_full", "tier1", "fallback"), help="Universe source mode.")
    parser.add_argument("--symbol", action="append", dest="symbols", help="Explicit ticker to ingest (can be repeated).")
    parser.add_argument("--skip-actuals", action="append", dest="skip_actuals", help="Ticker to skip EDGAR ingestion (can be repeated).")
    parser.add_argument("--quarters", type=int, default=8, help="Number of quarters to fetch from EDGAR.")
    parser.add_argument("--edgar-rate-limit", type=float, default=1.0, help="Delay between EDGAR API calls.")
    parser.add_argument("--edgar-retries", type=int, default=3, help="Maximum EDGAR retries.")
    parser.add_argument("--edgar-min-quarters", type=int, default=1, help="Minimum EDGAR quarters when using watermarks.")
    parser.add_argument("--edgar-max-quarters", type=int, default=None, help="Maximum EDGAR quarters when using watermarks.")
    parser.add_argument("--edgar-quarter-days", type=int, default=90, help="Days per quarter for watermark conversion.")
    parser.add_argument("--yahoo-rate-limit", type=float, default=0.5, help="Delay between Yahoo requests.")
    parser.add_argument("--yahoo-retries", type=int, default=3, help="Maximum Yahoo retries.")
    parser.add_argument("--no-yahoo", action="store_true", help="Disable Yahoo consensus ingestion.")
    parser.add_argument(
        "--use-watermark",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use registry watermarks to scope incremental ingestion.",
    )
    parser.add_argument("--watermark-lookback-days", type=int, default=None, help="Lookback days for watermark window.")
    parser.add_argument("--watermark-max-window-days", type=int, default=None, help="Max days per incremental window.")
    parser.add_argument("--watermark-fallback-days", type=int, default=None, help="Fallback days when no watermark exists.")
    parser.add_argument("--sec-identity", default=None, help="SEC identity string passed to edgartools.")
    parser.add_argument("--sec-user-agent-name", default=None, help="SEC User-Agent contact name.")
    parser.add_argument("--sec-user-agent-email", default=None, help="SEC User-Agent contact email.")
    parser.add_argument("--sec-user-agent-phone", default=None, help="SEC User-Agent contact phone.")
    parser.add_argument("--edgar-smoke-test", action="store_true", help="Run a minimal EDGAR submissions smoke test and exit.")
    parser.add_argument("--edgar-smoke-cik", default=None, help="10-digit CIK for EDGAR smoke test (default: Apple CIK).")
    parser.add_argument("--edgar-smoke-timeout", type=float, default=None, help="Timeout (seconds) for EDGAR smoke test.")
    parser.add_argument("--partition-key", action="append", dest="partition_keys", help="Partition key for Parquet output (can be repeated).")
    parser.add_argument("--log-level", default=os.getenv("ML_LOG_LEVEL", "INFO"), help="Logging level.")
    return parser.parse_args(argv)


def resolve_sec_identity(args: argparse.Namespace) -> str | None:
    """
    Resolve the SEC identity string from CLI args or environment.

    Args:
        args: Parsed CLI arguments.

    Returns:
        SEC identity string, if available.
    """
    identity = cast(str | None, args.sec_identity)
    if identity:
        return identity
    explicit = SecIdentityConfig(
        name=cast(str | None, args.sec_user_agent_name),
        email=cast(str | None, args.sec_user_agent_email),
        phone=cast(str | None, args.sec_user_agent_phone),
    ).resolved_identity()
    if explicit:
        return explicit
    return SecIdentityConfig.from_env().resolved_identity()


def build_config(args: argparse.Namespace) -> EarningsIngestionConfig:
    if not args.dsn:
        raise SystemExit("Database connection string required (use --dsn or set NAUTILUS_DB).")

    parquet_root = Path(args.parquet_root)
    override_symbols = tuple(args.symbols) if args.symbols else None
    skip_actuals = tuple(args.skip_actuals) if args.skip_actuals else DEFAULT_SKIP_ACTUALS_TICKERS
    partition_keys = tuple(args.partition_keys) if args.partition_keys else ("ticker",)
    watermark_base = earnings_window_defaults()
    watermark_config = WatermarkWindowConfig(
        use_watermark=(
            watermark_base.use_watermark if args.use_watermark is None else bool(args.use_watermark)
        ),
        lookback_days=(
            watermark_base.lookback_days
            if args.watermark_lookback_days is None
            else int(args.watermark_lookback_days)
        ),
        max_window_days=(
            watermark_base.max_window_days
            if args.watermark_max_window_days is None
            else int(args.watermark_max_window_days)
        ),
        fallback_start_days=(
            watermark_base.fallback_start_days
            if args.watermark_fallback_days is None
            else int(args.watermark_fallback_days)
        ),
    )

    return EarningsIngestionConfig(
        postgres_dsn=args.dsn,
        parquet_root=parquet_root,
        universe_mode=args.universe_mode,
        override_symbols=override_symbols,
        skip_actuals=skip_actuals,
        edgar_quarters=args.quarters,
        edgar_min_quarters=int(args.edgar_min_quarters),
        edgar_max_quarters=(
            int(args.edgar_max_quarters) if args.edgar_max_quarters is not None else None
        ),
        edgar_quarter_days=int(args.edgar_quarter_days),
        edgar_rate_limit=args.edgar_rate_limit,
        edgar_max_retries=args.edgar_retries,
        yahoo_rate_limit=args.yahoo_rate_limit,
        yahoo_max_retries=args.yahoo_retries,
        enable_yahoo=not args.no_yahoo,
        sec_identity=resolve_sec_identity(args),
        parquet_partition_keys=partition_keys,
        watermark_config=watermark_config,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(level=args.log_level.upper())
    bind_log_context(component="ml.cli.ingest_earnings")

    if args.edgar_smoke_test:
        sec_identity = resolve_sec_identity(args)
        if not sec_identity:
            raise SystemExit(
                "SEC identity missing. Set SEC_IDENTITY or SEC_USER_AGENT_* env vars."
            )
        smoke_defaults = EdgarSmokeTestConfig()
        smoke_cik = cast(str | None, args.edgar_smoke_cik) or smoke_defaults.cik
        smoke_timeout = cast(float | None, args.edgar_smoke_timeout) or smoke_defaults.timeout_seconds
        smoke_cfg = EdgarSmokeTestConfig(
            cik=smoke_cik,
            timeout_seconds=smoke_timeout,
        )
        smoke_result = run_edgar_smoke_test(
            cik=smoke_cfg.cik,
            identity=sec_identity,
            timeout_seconds=smoke_cfg.timeout_seconds,
        )
        logger = logging.getLogger("ml.cli.ingest_earnings")
        logger.info(
            "EDGAR smoke test succeeded",
            extra={
                "url": smoke_result.url,
                "status": smoke_result.status,
                "cik": smoke_result.cik,
                "filings_count": smoke_result.filings_count,
            },
        )
        return

    config = build_config(args)

    raw_writer = EarningsParquetRawWriter(
        base_path=config.parquet_root,
        partition_keys=config.parquet_partition_keys,
    )

    store_init = StoreInitializationComponent(db_connection=config.postgres_dsn)
    store_init.init_stores()
    if store_init.file_fallback or store_init.json_fallback:
        raise RuntimeError("Earnings ingestion requires PostgreSQL-backed stores")

    registry_init = RegistryInitializationComponent(db_connection=config.postgres_dsn)
    registry_init.init_registries()
    if isinstance(registry_init.data_registry, DummyRegistry):
        raise RuntimeError("DataRegistry not initialized")

    registry_init.inject_data_registry_into_stores(
        store_init.feature_store,
        store_init.model_store,
    )

    store = registry_init.create_data_store(
        feature_store=cast(FeatureStoreProtocol, store_init.feature_store),
        feature_dataset_store=store_init.feature_dataset_store,
        model_store=cast(ModelStoreProtocol, store_init.model_store),
        strategy_store=cast(StrategyStoreProtocol, store_init.strategy_store),
        earnings_store=cast(EarningsStoreProtocol, store_init.earnings_store),
        raw_writer=raw_writer,
    )

    service = EarningsIngestionService(config=config, writer=store)
    result = service.run()

    logger = logging.getLogger("ml.cli.ingest_earnings")
    logger.info(
        "Earnings ingestion complete",
        extra={
            "tickers": result.tickers_attempted,
            "actuals_written": result.actuals_written,
            "estimates_written": result.estimates_written,
            "skipped_actuals": result.skipped_actuals,
            "failures": result.failures,
            "duration_seconds": round(result.duration_seconds, 2),
            "universe_source": result.universe.source,
        },
    )


if __name__ == "__main__":
    main()
