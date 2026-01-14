#!/usr/bin/env python3

from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

from datetime import datetime, timedelta
import importlib
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest

from ml._imports import HAS_ONNX
from ml._imports import HAS_ONNX_CORE
from ml._imports import HAS_ONNX_EXPORT
from ml._imports import HAS_POLARS
from ml._imports import HAS_XGBOOST
from ml._imports import onnxmltools
from ml._imports import onnx
from ml._imports import pl
from ml.common.security import secure_onnx_load
from ml.config.runtime import OnnxRuntimeConfig
from ml.config.runtime import to_session_options
from ml.config.xgboost import XGBoostTrainingConfig
from ml.data import BuildResult
from ml.data import DatasetBuildConfig
from ml.data import build_tft_dataset
from ml.data import load_dataset_metadata
from ml.data.validation import DatasetValidationConfig
from ml.data.validation import DatasetValidationError
from ml.data.validation import validate_dataset
from ml.data.vintage import parse_dt
from ml.features.common.feature_metrics_collector import FeatureMetricsCollector
from ml.ml_types import PolarsDF
from ml.training.export import DEFAULT_ONNX_OPSET
from ml.training.non_distilled.xgboost import XGBoostTrainer


pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures(
        "isolated_prometheus_registry",
        "mock_tracing_backend",
        "isolated_orchestrator_env",
    ),
]


class _CatalogSlice(Protocol):
    """
    Protocol describing the real-catalog slice fixture payload.
    """

    catalog_path: Path
    instrument_id: str
    symbol: str
    start: datetime
    end: datetime


REAL_CATALOG_HORIZON_MINUTES = 5


def _require_polars() -> None:
    """
    Skip tests when Polars is unavailable.
    """
    if not HAS_POLARS or pl is None:
        pytest.skip("Polars not installed")


def _load_dataset_frame(result: BuildResult) -> PolarsDF:
    """
    Load the dataset parquet for assertions.
    """
    if pl is None:
        msg = "Polars is required to load dataset artifacts"
        raise RuntimeError(msg)
    return pl.read_parquet(str(result.dataset_parquet))


def _assert_finite_numeric_values(df: PolarsDF) -> None:
    """
    Assert numeric columns contain only finite values.
    """
    if pl is None:
        msg = "Polars is required for numeric checks"
        raise RuntimeError(msg)
    numeric_cols = [name for name, dtype in df.schema.items() if dtype.is_numeric()]
    if not numeric_cols:
        pytest.skip("No numeric columns available for finite-value checks")
    values = df.select(numeric_cols).to_numpy()
    assert np.isfinite(values).all()


def _build_window_dataset(
    *,
    slice_cfg: _CatalogSlice,
    out_dir: Path,
    start: datetime,
    end: datetime,
    horizon_minutes: int,
    threshold: float,
    lookback_periods: int,
) -> BuildResult:
    """
    Build a bounded dataset for walk-forward testing.
    """
    cfg = DatasetBuildConfig(
        data_dir=slice_cfg.catalog_path,
        out_dir=out_dir,
        symbols=[slice_cfg.symbol],
        instrument_ids=[slice_cfg.instrument_id],
        include_macro=False,
        include_micro=False,
        include_l2=False,
        include_events=False,
        include_calendar=False,
        include_earnings=False,
        auto_refresh_macro=False,
        horizon_minutes=horizon_minutes,
        threshold=threshold,
        lookback_periods=lookback_periods,
        start=start,
        end=end,
    )
    return build_tft_dataset(cfg)


def _extract_timestamp_ns(df: PolarsDF) -> np.ndarray:
    """
    Return timestamp column as int64 nanoseconds.
    """
    if pl is None:
        msg = "Polars is required to extract timestamps"
        raise RuntimeError(msg)
    ts_col = "timestamp" if "timestamp" in df.columns else "ts_event"
    return df.select(pl.col(ts_col).cast(pl.Int64)).to_series().to_numpy()


def _validation_config(*, horizon_minutes: int | None) -> DatasetValidationConfig:
    """
    Return dataset validation defaults for real-catalog checks.
    """
    return DatasetValidationConfig(
        min_rows=1,
        min_positive_rate=None,
        max_positive_rate=None,
        forward_return_horizon=horizon_minutes,
        forward_return_price_column="close",
    )


def _pick_numeric_feature_column(df: PolarsDF) -> str | None:
    """
    Select a numeric feature column suitable for data-quality checks.
    """
    exclude = {
        "y",
        "forward_return",
        "time_index",
        "instrument_id",
        "timestamp",
        "ts_event",
    }
    preferred = ("close", "return_1")
    for name in preferred:
        if name in df.columns and df.schema[name].is_numeric():
            return name
    for name, dtype in df.schema.items():
        if name in exclude:
            continue
        if dtype.is_numeric():
            return name
    return None


def test_real_catalog_dataset_smoke(real_catalog_dataset: BuildResult) -> None:
    """
    Assert real catalog dataset has required columns and finite numeric values.
    """
    _require_polars()
    df = _load_dataset_frame(real_catalog_dataset)
    if df.is_empty():
        pytest.skip("Real catalog dataset is empty")

    timestamp_col = "timestamp" if "timestamp" in df.columns else "ts_event"
    required = {"instrument_id", "time_index", "y", "forward_return", timestamp_col}
    assert required.issubset(set(df.columns))

    ts_values = _extract_timestamp_ns(df)
    assert np.all(np.diff(ts_values) >= 0)

    _assert_finite_numeric_values(df)


@pytest.mark.slow
def test_real_catalog_train_eval_infer_smoke(real_catalog_dataset: BuildResult) -> None:
    """
    Train/evaluate/infer on a real dataset slice and assert finite outputs.
    """
    _require_polars()
    if not HAS_XGBOOST:
        pytest.skip("XGBoost not installed")

    df = _load_dataset_frame(real_catalog_dataset)
    if df.is_empty():
        pytest.skip("Real catalog dataset is empty")

    feature_names = list(real_catalog_dataset.feature_names or [])
    if not feature_names:
        pytest.skip("Real catalog dataset has no feature columns")
    target_col = "y"
    required_cols = set(feature_names) | {target_col}
    if not required_cols.issubset(set(df.columns)):
        pytest.skip("Dataset missing required training columns")

    train_df = df.select(list(required_cols))
    if train_df.height < 50:
        pytest.skip("Insufficient rows for training smoke")

    labels = train_df[target_col].to_numpy()
    if len(np.unique(labels)) < 2:
        pytest.skip("Training labels are single-class for this slice")

    config = XGBoostTrainingConfig(
        data_source=str(real_catalog_dataset.dataset_parquet),
        target_column=target_col,
        train_test_split=0.8,
        random_seed=7,
        n_estimators=20,
        max_depth=3,
        learning_rate=0.2,
    )
    trainer = XGBoostTrainer(config)
    results = trainer.train(train_df)

    metrics = results["metrics"]
    numeric_metrics = [value for value in metrics.values() if isinstance(value, int | float)]
    assert numeric_metrics
    assert np.isfinite(np.asarray(numeric_metrics, dtype=float)).all()

    split_idx = int(train_df.height * config.train_test_split)
    val_df = train_df[split_idx:]
    if val_df.is_empty():
        pytest.skip("No validation rows available")

    X_val, y_val, _ = trainer.prepare_data(val_df, target_col)
    preds = trainer.predict(results["model"], X_val)
    assert preds.shape[0] == X_val.shape[0]
    assert np.isfinite(preds).all()

    eval_metrics = trainer.evaluate(results["model"], X_val, y_val)
    eval_values = [value for value in eval_metrics.values() if isinstance(value, int | float)]
    assert eval_values
    assert np.isfinite(np.asarray(eval_values, dtype=float)).all()


@pytest.mark.slow
def test_real_catalog_inference_serving_smoke(
    real_catalog_dataset: BuildResult,
    tmp_path: Path,
) -> None:
    """
    Export a model and run ONNX inference on real inputs.
    """
    _require_polars()
    if not HAS_XGBOOST:
        pytest.skip("XGBoost not installed")
    if not (HAS_ONNX and HAS_ONNX_CORE and HAS_ONNX_EXPORT):
        pytest.skip("ONNX runtime/export dependencies not installed")

    df = _load_dataset_frame(real_catalog_dataset)
    if df.is_empty():
        pytest.skip("Real catalog dataset is empty")

    feature_names = list(real_catalog_dataset.feature_names or [])
    if not feature_names:
        pytest.skip("Real catalog dataset has no feature columns")
    target_col = "y"
    required_cols = feature_names + [target_col]
    if not set(required_cols).issubset(set(df.columns)):
        pytest.skip("Dataset missing required training columns")

    train_df = df.select(required_cols)
    if train_df.height < 80:
        pytest.skip("Insufficient rows for inference smoke")

    labels = train_df[target_col].to_numpy()
    if len(np.unique(labels)) < 2:
        pytest.skip("Training labels are single-class for this slice")

    config = XGBoostTrainingConfig(
        data_source=str(real_catalog_dataset.dataset_parquet),
        target_column=target_col,
        train_test_split=0.8,
        random_seed=7,
        n_estimators=20,
        max_depth=3,
        learning_rate=0.2,
    )
    trainer = XGBoostTrainer(config)
    trainer.train(train_df)

    opset = DEFAULT_ONNX_OPSET
    if onnx is not None:
        try:
            opset = min(opset, int(onnx.defs.onnx_opset_version()))
        except Exception:
            opset = DEFAULT_ONNX_OPSET
    if onnxmltools is not None:
        try:
            topology = importlib.import_module("onnxmltools.convert.common._topology")
            max_supported = int(getattr(topology, "get_maximum_opset_supported")())
            opset = min(opset, max_supported)
        except Exception:
            pass

    model_path = trainer.save_for_production(
        tmp_path / "real_catalog_model",
        format="onnx",
        opset=opset,
    )

    session_options, providers = to_session_options(OnnxRuntimeConfig())
    session = secure_onnx_load(
        file_path=model_path,
        expected_digest=None,
        session_options=session_options,
        providers=providers,
        strict_integrity=False,
    )
    input_name = session.get_inputs()[0].name

    split_idx = int(train_df.height * config.train_test_split)
    holdout_df = train_df[split_idx:]
    if holdout_df.is_empty():
        pytest.skip("No holdout rows available for inference")

    features, _, _ = trainer.prepare_data(holdout_df, target_col)
    if features.size == 0:
        pytest.skip("Holdout features empty")
    batch_size = min(16, features.shape[0])
    inputs = features[:batch_size].astype(np.float32)
    if inputs.ndim == 1:
        inputs = inputs.reshape(1, -1)

    session.run(None, {input_name: inputs})
    runs = 3
    outputs = None
    start = time.perf_counter()
    for _ in range(runs):
        outputs = session.run(None, {input_name: inputs})
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / runs

    assert outputs is not None
    assert outputs
    preds = np.asarray(outputs[0], dtype=float)
    if preds.ndim > 1 and preds.shape[1] == 1:
        preds = preds.reshape(-1)
    assert preds.shape[0] == inputs.shape[0]
    assert np.isfinite(preds).all()

    max_latency_ms = 500.0
    assert elapsed_ms <= max_latency_ms


@pytest.mark.slow
def test_real_catalog_walk_forward_no_lookahead(
    real_catalog_slice: _CatalogSlice,
    tmp_path: Path,
) -> None:
    """
    Ensure train and prediction windows are disjoint and ordered.
    """
    _require_polars()

    horizon_minutes = 5
    lookback_periods = 30
    threshold = 0.0005
    train_duration = timedelta(minutes=90)
    gap_duration = timedelta(minutes=15)
    predict_duration = timedelta(minutes=60)

    train_start = real_catalog_slice.start
    train_end = train_start + train_duration
    predict_start = train_end + gap_duration + timedelta(minutes=horizon_minutes)
    predict_end = predict_start + predict_duration

    if predict_end > real_catalog_slice.end:
        pytest.skip("Catalog slice too small for walk-forward windows")

    train_result = _build_window_dataset(
        slice_cfg=real_catalog_slice,
        out_dir=tmp_path / "train",
        start=train_start,
        end=train_end,
        horizon_minutes=horizon_minutes,
        threshold=threshold,
        lookback_periods=lookback_periods,
    )
    pred_result = _build_window_dataset(
        slice_cfg=real_catalog_slice,
        out_dir=tmp_path / "predict",
        start=predict_start,
        end=predict_end,
        horizon_minutes=horizon_minutes,
        threshold=threshold,
        lookback_periods=lookback_periods,
    )

    train_df = _load_dataset_frame(train_result)
    pred_df = _load_dataset_frame(pred_result)
    if train_df.is_empty() or pred_df.is_empty():
        pytest.skip("Walk-forward dataset build produced empty outputs")

    train_ts = _extract_timestamp_ns(train_df)
    pred_ts = _extract_timestamp_ns(pred_df)
    assert train_ts.max() < pred_ts.min()
    assert train_end + timedelta(minutes=horizon_minutes) <= predict_start

    train_meta = load_dataset_metadata(
        train_result.dataset_parquet.with_name("dataset_metadata.json"),
    )
    pred_meta = load_dataset_metadata(
        pred_result.dataset_parquet.with_name("dataset_metadata.json"),
    )
    train_end_meta = parse_dt(train_meta.ts_event_end)
    pred_start_meta = parse_dt(pred_meta.ts_event_start)
    if train_end_meta is None or pred_start_meta is None:
        pytest.skip("Dataset metadata missing timestamp bounds")
    assert train_end_meta <= predict_start
    assert pred_start_meta >= predict_start
    assert train_end_meta < pred_start_meta


def test_real_catalog_dataset_validation_smoke(
    real_catalog_dataset: BuildResult,
) -> None:
    """
    Validate dataset rules against a real catalog slice.
    """
    _require_polars()
    df = _load_dataset_frame(real_catalog_dataset)
    if df.is_empty():
        pytest.skip("Real catalog dataset is empty")
    if "close" not in df.columns:
        pytest.skip("Dataset missing close column for forward-return validation")

    config = _validation_config(horizon_minutes=REAL_CATALOG_HORIZON_MINUTES)
    result = validate_dataset(df, config=config)
    assert result.row_count == df.height


def test_real_catalog_validation_detects_forward_return_misalignment(
    real_catalog_dataset: BuildResult,
) -> None:
    """
    Ensure forward-return validation catches misalignment.
    """
    _require_polars()
    if pl is None:
        raise RuntimeError("Polars is required for forward-return checks")

    df = _load_dataset_frame(real_catalog_dataset)
    if df.is_empty():
        pytest.skip("Real catalog dataset is empty")
    if "forward_return" not in df.columns or "close" not in df.columns:
        pytest.skip("Dataset missing forward_return or close columns")

    df_bad = df.with_columns(
        pl.when(pl.arange(0, df.height) == 0)
        .then(pl.col("forward_return") + 0.1)
        .otherwise(pl.col("forward_return"))
        .alias("forward_return"),
    )

    config = _validation_config(horizon_minutes=REAL_CATALOG_HORIZON_MINUTES)
    with pytest.raises(DatasetValidationError):
        validate_dataset(df_bad, config=config)


def test_real_catalog_validation_detects_timestamp_reversal(
    real_catalog_dataset: BuildResult,
) -> None:
    """
    Ensure timestamp reversals are rejected during validation.
    """
    _require_polars()
    if pl is None:
        raise RuntimeError("Polars is required for timestamp checks")

    df = _load_dataset_frame(real_catalog_dataset)
    if df.is_empty():
        pytest.skip("Real catalog dataset is empty")
    if "time_index" not in df.columns:
        pytest.skip("Dataset missing time_index column")

    ts_col = "timestamp" if "timestamp" in df.columns else "ts_event"
    symbol = df.select(pl.col("instrument_id").first()).item()
    min_index = (
        df.filter(pl.col("instrument_id") == symbol).select(pl.col("time_index").min()).item()
    )
    if min_index is None:
        pytest.skip("Dataset missing time_index values")
    target_index = int(min_index) + 1
    if df.filter(
        (pl.col("instrument_id") == symbol) & (pl.col("time_index") == target_index),
    ).is_empty():
        pytest.skip("Insufficient rows to introduce timestamp reversal")

    previous_ts = (
        df.filter(
            (pl.col("instrument_id") == symbol) & (pl.col("time_index") == int(min_index)),
        )
        .select(pl.col(ts_col).cast(pl.Int64))
        .item()
    )
    if previous_ts is None:
        pytest.skip("Unable to resolve previous timestamp value")

    df_bad = df.with_columns(
        pl.when(
            (pl.col("instrument_id") == symbol) & (pl.col("time_index") == target_index),
        )
        .then(pl.lit(int(previous_ts) - 1))
        .otherwise(pl.col(ts_col).cast(pl.Int64))
        .alias(ts_col),
    )

    config = _validation_config(horizon_minutes=None)
    with pytest.raises(DatasetValidationError):
        validate_dataset(df_bad, config=config)


def test_real_catalog_validation_detects_feature_gaps(
    real_catalog_dataset: BuildResult,
) -> None:
    """
    Ensure feature coverage checks detect injected gaps.
    """
    _require_polars()
    if pl is None:
        raise RuntimeError("Polars is required for feature-gap checks")

    df = _load_dataset_frame(real_catalog_dataset)
    if df.is_empty():
        pytest.skip("Real catalog dataset is empty")

    feature_col = _pick_numeric_feature_column(df)
    if feature_col is None:
        pytest.skip("No numeric feature column available for gap checks")

    df_gap = df.with_columns(
        pl.when(pl.arange(0, df.height) % 5 == 0)
        .then(None)
        .otherwise(pl.col(feature_col))
        .alias(feature_col),
    )
    config = DatasetValidationConfig(
        min_rows=1,
        min_positive_rate=None,
        max_positive_rate=None,
        min_feature_coverage=0.95,
        require_monotonic_timestamps=True,
        forward_return_horizon=None,
    )
    with pytest.raises(DatasetValidationError):
        validate_dataset(df_gap, config=config)


def test_real_catalog_spike_outlier_rate_increases(
    real_catalog_dataset: BuildResult,
) -> None:
    """
    Verify outlier metrics respond to injected spikes on real data.
    """
    _require_polars()
    if pl is None:
        raise RuntimeError("Polars is required for outlier checks")

    df = _load_dataset_frame(real_catalog_dataset)
    if df.is_empty():
        pytest.skip("Real catalog dataset is empty")

    feature_col = _pick_numeric_feature_column(df)
    if feature_col is None:
        pytest.skip("No numeric feature column available for spike checks")

    series = df.get_column(feature_col).cast(pl.Float64)
    if len(series) == 0:
        pytest.skip("Feature column is empty")

    collector = FeatureMetricsCollector()
    values = series.to_list()
    spike_index = len(values) // 2
    if values[spike_index] is None:
        pytest.skip("Spike target value is null")
    values[spike_index] = float(values[spike_index]) * 50.0
    spiked_series = pl.Series(feature_col, values, dtype=pl.Float64)
    spiked_rate = collector._calculate_outlier_rate(spiked_series, total_rows=len(values))
    assert spiked_rate >= (1.0 / float(len(values)))
