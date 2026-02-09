#!/usr/bin/env python3
"""
CLI wrapper for the production dataset pipeline.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.data import ProductionDatasetConfig
from ml.data import build_production_dataset


__all__ = ["main"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build production ML dataset")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific phase",
    )
    parser.add_argument("--full", action="store_true", help="Run complete pipeline")
    parser.add_argument("--estimate", action="store_true", help="Estimate requirements only")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory",
    )
    parser.add_argument(
        "--databento-api-key",
        type=str,
        default=None,
        help="Override Databento API key (defaults to env)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    bind_log_context(component="ml.cli.build_production_dataset")

    if ProductionDatasetConfig is None or build_production_dataset is None:
        raise RuntimeError(
            "Production dataset tooling requires optional dependencies; install ml[production]",
        )

    config = ProductionDatasetConfig(
        data_dir=args.data_dir,
        phase=args.phase,
        run_full=args.full,
        estimate_only=args.estimate,
        databento_api_key=args.databento_api_key,
    )
    build_production_dataset(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
