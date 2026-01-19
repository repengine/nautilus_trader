#!/usr/bin/env python3
"""Run Chronos inference using the adapter wrapper."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from ml._imports import HAS_AUTOGLUON
from ml._imports import HAS_PANDAS
from ml._imports import TimeSeriesPredictor
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml.actors.common.chronos_inference import ChronosInferenceAdapter
from ml.common.logging_config import configure_logging
from ml.config.autogluon import AutoGluonDataConfig


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as _pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Chronos inference via adapter.")
    parser.add_argument(
        "--predictor-path",
        type=Path,
        required=True,
        help="Path to AutoGluon TimeSeriesPredictor directory.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset file (parquet or csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npy file for predictions.",
    )
    parser.add_argument(
        "--item-id-column",
        type=str,
        default="instrument_id",
        help="Column name for series identifiers.",
    )
    parser.add_argument(
        "--timestamp-column",
        type=str,
        default="ts_event",
        help="Column name for timestamps (nanoseconds).",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="forward_return",
        help="Target column name used for conversion.",
    )
    return parser


def _load_frame(path: Path) -> _pd.DataFrame:
    if not HAS_PANDAS or pd is None:
        check_ml_dependencies(["pandas"])
        raise ImportError("pandas is required to load Chronos inference datasets")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return cast(_pd.DataFrame, pd.read_parquet(path))
    if suffix in {".csv", ".tsv"}:
        return cast(_pd.DataFrame, pd.read_csv(path))
    raise ValueError(f"Unsupported dataset format: {suffix}")


def main() -> None:
    configure_logging()
    args = _build_parser().parse_args()

    if not HAS_AUTOGLUON or TimeSeriesPredictor is None:
        check_ml_dependencies(["autogluon"])
        raise ImportError("AutoGluon TimeSeries is required for Chronos inference")

    predictor = TimeSeriesPredictor.load(str(args.predictor_path))
    data_config = AutoGluonDataConfig(
        item_id_column=str(args.item_id_column),
        timestamp_column=str(args.timestamp_column),
        target_column=str(args.target_column),
    )

    frame = _load_frame(args.dataset)
    adapter = ChronosInferenceAdapter(predictor=predictor, data_config=data_config)
    predictions = adapter.predict(frame)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, predictions)

    logger.info(
        "chronos_inference_complete",
        extra={
            "predictor_path": str(args.predictor_path),
            "dataset": str(args.dataset),
            "output": str(output_path),
            "rows": int(predictions.shape[0]),
        },
    )


if __name__ == "__main__":
    main()
