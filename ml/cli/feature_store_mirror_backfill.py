#!/usr/bin/env python3
"""
CLI for backfilling the FeatureStore parquet mirror from SQL.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ml.config import FeatureStoreMirrorBackfillConfig
from ml.config import FeatureStoreMirrorConfig
from ml.config._env_utils import resolve_db_connection
from ml.stores import backfill_feature_store_mirror


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill feature store parquet mirrors from SQL",
    )
    parser.add_argument("--dsn", help="PostgreSQL DSN (defaults to env)")
    parser.add_argument(
        "--mirror-dir",
        help="Override mirror base directory (defaults to env or config)",
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--start-ts", type=int, default=None)
    parser.add_argument("--end-ts", type=int, default=None)
    parser.add_argument("--feature-set-id", default=None)
    parser.add_argument("--instrument-id", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dsn = args.dsn or resolve_db_connection(os.environ)
    if not dsn:
        raise SystemExit("Missing DB connection (set ML_DB_CONNECTION or pass --dsn)")

    config = FeatureStoreMirrorBackfillConfig(
        db_connection=dsn,
        batch_size=int(args.batch_size),
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        feature_set_id=args.feature_set_id,
        instrument_id=args.instrument_id,
    )

    mirror_config = FeatureStoreMirrorConfig.from_env()
    if args.mirror_dir:
        mirror_config = FeatureStoreMirrorConfig(
            enabled=mirror_config.enabled,
            base_dir=Path(args.mirror_dir),
            partition_field=mirror_config.partition_field,
            timestamp_field=mirror_config.timestamp_field,
            values_field=mirror_config.values_field,
        )

    backfill_feature_store_mirror(config, mirror_config=mirror_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
