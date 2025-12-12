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

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.config.earnings_ingestion import DEFAULT_SKIP_ACTUALS_TICKERS
from ml.config.earnings_ingestion import EarningsIngestionConfig
from ml.data.earnings.ingestion_service import EarningsIngestionService
from ml.stores.data_store import DataStore
from ml.stores.earnings_raw_writer import EarningsParquetRawWriter


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest earnings data from EDGAR and Yahoo Finance.")
    parser.add_argument("--dsn", default=os.getenv("NAUTILUS_DB"), help="PostgreSQL connection string (default: NAUTILUS_DB env).")
    parser.add_argument("--parquet-root", default="ml_out/earnings_raw", help="Root directory for Parquet mirrors.")
    parser.add_argument("--universe-mode", default="postgres", choices=("postgres", "tier1_full", "tier1", "fallback"), help="Universe source mode.")
    parser.add_argument("--symbol", action="append", dest="symbols", help="Explicit ticker to ingest (can be repeated).")
    parser.add_argument("--skip-actuals", action="append", dest="skip_actuals", help="Ticker to skip EDGAR ingestion (can be repeated).")
    parser.add_argument("--quarters", type=int, default=8, help="Number of quarters to fetch from EDGAR.")
    parser.add_argument("--edgar-rate-limit", type=float, default=1.0, help="Delay between EDGAR API calls.")
    parser.add_argument("--edgar-retries", type=int, default=3, help="Maximum EDGAR retries.")
    parser.add_argument("--yahoo-rate-limit", type=float, default=0.5, help="Delay between Yahoo requests.")
    parser.add_argument("--yahoo-retries", type=int, default=3, help="Maximum Yahoo retries.")
    parser.add_argument("--no-yahoo", action="store_true", help="Disable Yahoo consensus ingestion.")
    parser.add_argument("--sec-identity", default=os.getenv("SEC_IDENTITY"), help="SEC identity string passed to edgartools.")
    parser.add_argument("--partition-key", action="append", dest="partition_keys", help="Partition key for Parquet output (can be repeated).")
    parser.add_argument("--log-level", default=os.getenv("ML_LOG_LEVEL", "INFO"), help="Logging level.")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> EarningsIngestionConfig:
    if not args.dsn:
        raise SystemExit("Database connection string required (use --dsn or set NAUTILUS_DB).")

    parquet_root = Path(args.parquet_root)
    override_symbols = tuple(args.symbols) if args.symbols else None
    skip_actuals = tuple(args.skip_actuals) if args.skip_actuals else DEFAULT_SKIP_ACTUALS_TICKERS
    partition_keys = tuple(args.partition_keys) if args.partition_keys else ("ticker",)

    return EarningsIngestionConfig(
        postgres_dsn=args.dsn,
        parquet_root=parquet_root,
        universe_mode=args.universe_mode,
        override_symbols=override_symbols,
        skip_actuals=skip_actuals,
        edgar_quarters=args.quarters,
        edgar_rate_limit=args.edgar_rate_limit,
        edgar_max_retries=args.edgar_retries,
        yahoo_rate_limit=args.yahoo_rate_limit,
        yahoo_max_retries=args.yahoo_retries,
        enable_yahoo=not args.no_yahoo,
        sec_identity=args.sec_identity,
        parquet_partition_keys=partition_keys,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(level=args.log_level.upper())
    bind_log_context(component="ml.cli.ingest_earnings")

    config = build_config(args)

    raw_writer = EarningsParquetRawWriter(
        base_path=config.parquet_root,
        partition_keys=config.parquet_partition_keys,
    )
    store = DataStore(connection_string=config.postgres_dsn, raw_writer=raw_writer)

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
