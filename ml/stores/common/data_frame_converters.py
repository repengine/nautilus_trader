#!/usr/bin/env python3

"""
Helpers to convert DataFrame-like inputs into store payload objects.

These utilities keep conversion logic shared between legacy and facade writers.

"""

from __future__ import annotations

from typing import Any, cast

from ml.ml_types import DataFrameLike
from ml.stores.base import FeatureData
from ml.stores.base import ModelPrediction
from ml.stores.base import StrategySignal


def data_frame_to_feature_data(
    data_frame: DataFrameLike,
    instrument_id: str,
) -> list[FeatureData]:
    """
    Convert DataFrame to list of FeatureData.

    Args:
        data_frame: DataFrame with feature data.
        instrument_id: Instrument identifier to attach to rows.

    Returns:
        List of FeatureData objects.

    Example:
        >>> features = data_frame_to_feature_data(df, "EUR/USD")
        >>> assert features[0].instrument_id == "EUR/USD"

    """
    features: list[FeatureData] = []
    data_frame_any = cast(Any, data_frame)

    if hasattr(data_frame_any, "iter_rows"):
        for row in data_frame_any.iter_rows(named=True):
            ts_event_raw = row.get("ts_event")
            if ts_event_raw is None:
                raise ValueError("Missing ts_event in feature row")
            ts_event = int(ts_event_raw)
            ts_init_raw = row.get("ts_init", ts_event_raw)
            ts_init = int(ts_init_raw)
            features.append(
                FeatureData(
                    feature_set_id=row.get("feature_set_id", "default"),
                    instrument_id=instrument_id,
                    values=row.get("values", {}),
                    _ts_event=ts_event,
                    _ts_init=ts_init,
                ),
            )
    elif hasattr(data_frame_any, "iterrows"):
        for _, row in data_frame_any.iterrows():
            ts_event_raw = row.get("ts_event")
            if ts_event_raw is None:
                raise ValueError("Missing ts_event in feature row")
            ts_event = int(ts_event_raw)
            ts_init_raw = row.get("ts_init", ts_event_raw)
            ts_init = int(ts_init_raw)
            features.append(
                FeatureData(
                    feature_set_id=row.get("feature_set_id", "default"),
                    instrument_id=instrument_id,
                    values=row.get("values", {}),
                    _ts_event=ts_event,
                    _ts_init=ts_init,
                ),
            )
    else:
        for row in data_frame_any:
            if isinstance(row, dict):
                ts_event_raw = row.get("ts_event")
                if ts_event_raw is None:
                    raise ValueError("Missing ts_event in feature row")
                ts_event = int(ts_event_raw)
                ts_init_raw = row.get("ts_init", ts_event_raw)
                ts_init = int(ts_init_raw)
                features.append(
                    FeatureData(
                        feature_set_id=row.get("feature_set_id", "default"),
                        instrument_id=instrument_id,
                        values=row.get("values", {}),
                        _ts_event=ts_event,
                        _ts_init=ts_init,
                    ),
                )

    return features


def data_frame_to_predictions(
    data_frame: DataFrameLike | list[dict[str, Any]],
) -> list[ModelPrediction]:
    """
    Convert DataFrame to list of ModelPrediction.

    Args:
        data_frame: DataFrame or list of dicts with prediction data.

    Returns:
        List of ModelPrediction objects.

    Example:
        >>> preds = data_frame_to_predictions(df)
        >>> assert preds[0].model_id

    """
    predictions: list[ModelPrediction] = []
    data_frame_any = cast(Any, data_frame)

    if hasattr(data_frame_any, "iter_rows"):
        for row in data_frame_any.iter_rows(named=True):
            features_used = cast(
                dict[str, float],
                row.get("features_used", row.get("features", {})) or {},
            )
            inference_time_ms = float(row.get("inference_time_ms", 0.0))
            ts_event_raw = row.get("ts_event")
            if ts_event_raw is None:
                raise ValueError("Missing ts_event in prediction row")
            ts_event = int(ts_event_raw)
            ts_init_raw = row.get("ts_init", ts_event_raw)
            ts_init = int(ts_init_raw)
            prediction_raw = row.get("prediction")
            if prediction_raw is None:
                prediction_raw = row.get("value", 0.0)
            prediction = float(0.0 if prediction_raw is None else prediction_raw)
            predictions.append(
                ModelPrediction(
                    model_id=row["model_id"],
                    instrument_id=row["instrument_id"],
                    prediction=prediction,
                    confidence=float(row.get("confidence", 0.0)),
                    features_used=features_used,
                    inference_time_ms=inference_time_ms,
                    _ts_event=ts_event,
                    _ts_init=ts_init,
                    is_live=bool(row.get("is_live", False)),
                ),
            )
    elif hasattr(data_frame_any, "iterrows"):
        for _, row in data_frame_any.iterrows():
            features_used = cast(
                dict[str, float],
                row.get("features_used", row.get("features", {})) or {},
            )
            inference_time_ms = float(row.get("inference_time_ms", 0.0))
            ts_event_raw = row.get("ts_event")
            if ts_event_raw is None:
                raise ValueError("Missing ts_event in prediction row")
            ts_event = int(ts_event_raw)
            ts_init_raw = row.get("ts_init", ts_event_raw)
            ts_init = int(ts_init_raw)
            prediction_raw = row.get("prediction")
            if prediction_raw is None:
                prediction_raw = row.get("value", 0.0)
            prediction = float(0.0 if prediction_raw is None else prediction_raw)
            predictions.append(
                ModelPrediction(
                    model_id=row["model_id"],
                    instrument_id=row["instrument_id"],
                    prediction=prediction,
                    confidence=float(row.get("confidence", 0.0)),
                    features_used=features_used,
                    inference_time_ms=inference_time_ms,
                    _ts_event=ts_event,
                    _ts_init=ts_init,
                    is_live=bool(row.get("is_live", False)),
                ),
            )
    else:
        for row in data_frame_any:
            if isinstance(row, dict):
                features_used = cast(
                    dict[str, float],
                    row.get("features_used", row.get("features", {})) or {},
                )
                inference_time_ms = float(row.get("inference_time_ms", 0.0))
                ts_event_raw = row.get("ts_event")
                if ts_event_raw is None:
                    raise ValueError("Missing ts_event in prediction row")
                ts_event = int(ts_event_raw)
                ts_init_raw = row.get("ts_init", ts_event_raw)
                ts_init = int(ts_init_raw)
                prediction_raw = row.get("prediction")
                if prediction_raw is None:
                    prediction_raw = row.get("value", 0.0)
                prediction = float(0.0 if prediction_raw is None else prediction_raw)
                predictions.append(
                    ModelPrediction(
                        model_id=row["model_id"],
                        instrument_id=row["instrument_id"],
                        prediction=prediction,
                        confidence=float(row.get("confidence", 0.0)),
                        features_used=features_used,
                        inference_time_ms=inference_time_ms,
                        _ts_event=ts_event,
                        _ts_init=ts_init,
                        is_live=bool(row.get("is_live", False)),
                    ),
                )

    return predictions


def data_frame_to_signals(
    data_frame: DataFrameLike | list[dict[str, Any]],
) -> list[StrategySignal]:
    """
    Convert DataFrame to list of StrategySignal.

    Args:
        data_frame: DataFrame with signal data.

    Returns:
        List of StrategySignal objects.

    Example:
        >>> signals = data_frame_to_signals(df)
        >>> assert signals[0].strategy_id

    """
    signals: list[StrategySignal] = []
    data_frame_any = cast(Any, data_frame)

    if hasattr(data_frame_any, "iter_rows"):
        for row in data_frame_any.iter_rows(named=True):
            ts_event_raw = row.get("ts_event")
            if ts_event_raw is None:
                raise ValueError("Missing ts_event in signal row")
            ts_event = int(ts_event_raw)
            ts_init_raw = row.get("ts_init", ts_event_raw)
            ts_init = int(ts_init_raw)
            signals.append(
                StrategySignal(
                    strategy_id=row["strategy_id"],
                    instrument_id=row["instrument_id"],
                    signal_type=row["signal_type"],
                    strength=float(row.get("strength", row.get("signal_value", 0.0))),
                    model_predictions=row.get("model_predictions", {}),
                    risk_metrics=row.get("risk_metrics", {}),
                    execution_params=row.get("execution_params", {}),
                    _ts_event=ts_event,
                    _ts_init=ts_init,
                    run_id=row.get("run_id"),
                    ingested_at_ns=row.get("ingested_at_ns"),
                ),
            )
    elif hasattr(data_frame_any, "iterrows"):
        for _, row in data_frame_any.iterrows():
            ts_event_raw = row.get("ts_event")
            if ts_event_raw is None:
                raise ValueError("Missing ts_event in signal row")
            ts_event = int(ts_event_raw)
            ts_init_raw = row.get("ts_init", ts_event_raw)
            ts_init = int(ts_init_raw)
            signals.append(
                StrategySignal(
                    strategy_id=row["strategy_id"],
                    instrument_id=row["instrument_id"],
                    signal_type=row["signal_type"],
                    strength=float(row.get("strength", row.get("signal_value", 0.0))),
                    model_predictions=row.get("model_predictions", {}),
                    risk_metrics=row.get("risk_metrics", {}),
                    execution_params=row.get("execution_params", {}),
                    _ts_event=ts_event,
                    _ts_init=ts_init,
                    run_id=row.get("run_id"),
                    ingested_at_ns=row.get("ingested_at_ns"),
                ),
            )
    else:
        for row in data_frame_any:
            if isinstance(row, dict):
                ts_event_raw = row.get("ts_event")
                if ts_event_raw is None:
                    raise ValueError("Missing ts_event in signal row")
                ts_event = int(ts_event_raw)
                ts_init_raw = row.get("ts_init") or ts_event_raw
                ts_init = int(ts_init_raw)
                signals.append(
                    StrategySignal(
                        strategy_id=row["strategy_id"],
                        instrument_id=row["instrument_id"],
                        signal_type=row["signal_type"],
                        strength=float(row.get("strength", row.get("signal_value", 0.0)) or 0.0),
                        model_predictions=row.get("model_predictions", {}),
                        risk_metrics=row.get("risk_metrics", {}),
                        execution_params=row.get("execution_params", {}),
                        _ts_event=ts_event,
                        _ts_init=ts_init,
                        run_id=row.get("run_id"),
                        ingested_at_ns=row.get("ingested_at_ns"),
                    ),
                )

    return signals
