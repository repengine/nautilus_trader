#!/usr/bin/env python3
"""
CLI to refresh feature dataset parquet mirrors from SQL.
"""

from __future__ import annotations

import argparse
import os

from ml.config import FeatureDatasetMirrorConfig
from ml.config._env_utils import resolve_db_connection
from ml.stores import FeatureDatasetMirrorExportConfig
from ml.stores import refresh_feature_dataset_mirrors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh feature dataset parquet mirrors from SQL",
    )
    parser.add_argument("--dsn", help="PostgreSQL DSN (defaults to env)")
    parser.add_argument(
        "--series-id",
        action="append",
        default=None,
        help="Limit macro export to specific series (repeatable)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dsn = args.dsn or resolve_db_connection(os.environ)
    if not dsn:
        raise SystemExit("Missing DB connection (set ML_DB_CONNECTION or pass --dsn)")

    series_ids = tuple(args.series_id) if args.series_id else None
    config = FeatureDatasetMirrorExportConfig(
        db_connection=dsn,
        series_ids=series_ids,
    )
    mirror_config = FeatureDatasetMirrorConfig.from_env()
    refresh_feature_dataset_mirrors(config, mirror_config=mirror_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
