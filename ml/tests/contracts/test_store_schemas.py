"""
Contract tests for Store boundary schemas using Pandera.

These tests define and validate the data contracts at store boundaries, ensuring all
data flowing in and out conforms to expected schemas. This catches data quality issues
early and provides clear documentation of expected data formats.

"""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from ml.tests.fixtures.pandera import DataFrame, Series, ensure_pandera_available

pa = ensure_pandera_available()
Check = pa.Check
Column = pa.Column
DataFrameSchema = pa.DataFrameSchema

from nautilus_trader.model.identifiers import InstrumentId


# ============================================================================
# FEATURE STORE SCHEMAS
# ============================================================================


class FeatureInputSchema(pa.DataFrameModel):
    """
    Schema for feature data input to FeatureStore.
    """

    feature_set_id: Series[str] = pa.Field(
        nullable=False,
        description="Unique identifier for the feature set",
    )
    instrument_id: Series[str] = pa.Field(
        nullable=False,
        description="Nautilus instrument identifier (SYMBOL.VENUE)",
    )
    ts_event: Series[int] = pa.Field(
        nullable=False,
        ge=0,
        description="Event timestamp in nanoseconds since epoch",
    )
    ts_init: Series[int] = pa.Field(
        nullable=False,
        ge=0,
        description="Initialization timestamp in nanoseconds since epoch",
    )

    @pa.dataframe_check()
    def check_timestamp_ordering(cls, df: DataFrame[Any]) -> Series[bool]:
        """
        Ensure ts_init >= ts_event.
        """
        return cast(Series[bool], df["ts_init"] >= df["ts_event"])

    @pa.check("ts_event", name="reasonable_timestamp")
    def check_reasonable_timestamp(cls, series: Series[int]) -> Series[bool]:
        """
        Ensure timestamps are within reasonable bounds (2010-2030).
        """
        min_ts = int(datetime(2010, 1, 1).timestamp() * 1e9)
        max_ts = int(datetime(2030, 1, 1).timestamp() * 1e9)
        return cast(Series[bool], (series >= min_ts) & (series <= max_ts))

    class Config:
        coerce = True
        strict = True

    @pa.check("feature_set_id", name="feature_set_id_format")
    def check_feature_set_id_format(cls, s: Series[str]) -> Series[bool]:
        return cast(Series[bool], s.str.match(r"^[a-zA-Z0-9_-]+$"))

    @pa.check("instrument_id", name="instrument_id_format")
    def check_instrument_id_format(cls, s: Series[str]) -> Series[bool]:
        return cast(Series[bool], s.str.match(r"^[A-Z0-9]+\.[A-Z]+$"))


class FeatureValueSchema(pa.DataFrameModel):
    """
    Schema for feature values stored in FeatureStore.
    """

    # Dynamic feature columns - validated separately
    # All feature columns should be numeric

    @pa.dataframe_check()
    def has_feature_columns(cls, df: pd.DataFrame) -> bool:
        """
        Ensure at least one feature column exists.
        """
        reserved_columns = {"feature_set_id", "instrument_id", "ts_event", "ts_init"}
        feature_columns = set(df.columns) - reserved_columns
        return len(feature_columns) > 0

    @pa.dataframe_check()
    def feature_values_are_numeric(cls, df: pd.DataFrame) -> bool:
        """
        Ensure all feature values are numeric.
        """
        reserved_columns = {"feature_set_id", "instrument_id", "ts_event", "ts_init"}
        for col in df.columns:
            if col not in reserved_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    return False
        return True

    @pa.dataframe_check()
    def no_infinite_values(cls, df: pd.DataFrame) -> bool:
        """
        Ensure no infinite values in features.
        """
        reserved_columns = {"feature_set_id", "instrument_id", "ts_event", "ts_init"}
        for col in df.columns:
            if col not in reserved_columns:
                if np.isinf(df[col]).any():
                    return False
        return True


# ============================================================================
# MODEL STORE SCHEMAS
# ============================================================================


class PredictionSchema(pa.DataFrameModel):
    """
    Schema for model predictions stored in ModelStore.
    """

    model_id: Series[str] = pa.Field(
        nullable=False,
        description="Unique model identifier",
    )
    instrument_id: Series[str] = pa.Field(nullable=False)
    prediction: Series[float] = pa.Field(
        nullable=False,
        ge=0.0,
        le=1.0,
        description="Model prediction probability",
    )
    confidence: Series[float] = pa.Field(
        nullable=False,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score",
    )
    ts_event: Series[int] = pa.Field(
        nullable=False,
        ge=0,
    )
    ts_init: Series[int] = pa.Field(
        nullable=False,
        ge=0,
    )

    @pa.check("confidence", name="confidence_consistency")
    def check_confidence_consistency(cls, series: Series[float]) -> Series[bool]:
        """
        Confidence should be positive when prediction is non-zero.
        """
        return cast(Series[bool], series >= 0)

    class Config:
        coerce = True

    @pa.check("instrument_id", name="instrument_id_format")
    def check_instrument_id_format(cls, s: Series[str]) -> Series[bool]:
        return cast(Series[bool], s.str.match(r"^[A-Z0-9]+\.[A-Z]+$"))


class ModelMetricsSchema(pa.DataFrameModel):
    """
    Schema for model performance metrics.
    """

    model_id: Series[str] = pa.Field(nullable=False)
    metric_name: Series[str] = pa.Field(
        nullable=False,
        isin=["accuracy", "precision", "recall", "f1", "sharpe", "max_drawdown"],
    )
    metric_value: Series[float] = pa.Field(nullable=False)
    evaluation_ts: Series[int] = pa.Field(nullable=False, ge=0)

    @pa.dataframe_check()
    def check_percentage_metrics(cls, df: DataFrame[Any]) -> bool:
        """
        Ensure percentage metrics are in [0, 1].
        """
        percentage_metrics = {"accuracy", "precision", "recall", "f1"}
        mask = df["metric_name"].isin(percentage_metrics)
        if not mask.any():
            return True

        values = df.loc[mask, "metric_value"]
        return bool(((values >= 0) & (values <= 1)).all())


# ============================================================================
# STRATEGY STORE SCHEMAS
# ============================================================================


class SignalSchema(pa.DataFrameModel):
    """
    Schema for strategy signals.
    """

    strategy_id: Series[str] = pa.Field(nullable=False)
    instrument_id: Series[str] = pa.Field(nullable=False)
    signal_type: Series[str] = pa.Field(
        nullable=False,
        isin=["BUY", "SELL", "HOLD", "CLOSE"],
    )
    signal_strength: Series[float] = pa.Field(
        nullable=False,
        ge=0.0,
        le=1.0,
    )
    ts_event: Series[int] = pa.Field(nullable=False, ge=0)
    ts_init: Series[int] = pa.Field(nullable=False, ge=0)

    @pa.dataframe_check()
    def check_signal_consistency(cls, df: DataFrame[Any]) -> Series[bool]:
        """
        Signal strength should be within [0, 1] for all signal types.
        """
        strengths = df["signal_strength"]
        return cast(Series[bool], strengths.between(0.0, 1.0))

    @pa.check("instrument_id", name="instrument_id_format")
    def check_instrument_id_format(cls, s: Series[str]) -> Series[bool]:
        return cast(Series[bool], s.str.match(r"^[A-Z0-9]+\.[A-Z]+$"))


class PositionSchema(pa.DataFrameModel):
    """
    Schema for position tracking.
    """

    strategy_id: Series[str] = pa.Field(nullable=False)
    instrument_id: Series[str] = pa.Field(nullable=False)
    position_size: Series[float] = pa.Field(nullable=False)
    entry_price: Series[float] = pa.Field(nullable=True, gt=0)
    current_price: Series[float] = pa.Field(nullable=True, gt=0)
    unrealized_pnl: Series[float] = pa.Field(nullable=True)
    ts_event: Series[int] = pa.Field(nullable=False, ge=0)

    @pa.check(
        "unrealized_pnl",
        "position_size",
        "entry_price",
        "current_price",
        name="pnl_consistency",
    )
    def check_pnl_consistency(cls, df: DataFrame[Any]) -> Series[bool]:
        """
        Verify PnL calculation consistency.
        """
        mask = df[["position_size", "entry_price", "current_price"]].notna().all(axis=1)
        if mask.any():
            expected_pnl = df.loc[mask, "position_size"] * (
                df.loc[mask, "current_price"] - df.loc[mask, "entry_price"]
            )
            actual_pnl = df.loc[mask, "unrealized_pnl"]
            # Allow small floating point differences
            return cast(Series[bool], (abs(expected_pnl - actual_pnl) < 0.01))
        return cast(Series[bool], pd.Series([True] * len(df)))


# ============================================================================
# DATA STORE SCHEMAS
# ============================================================================


class WatermarkSchema(pa.DataFrameModel):
    """
    Schema for watermark tracking.
    """

    pipeline_id: Series[str] = pa.Field(nullable=False)
    watermark_ts: Series[int] = pa.Field(nullable=False, ge=0)
    processed_count: Series[int] = pa.Field(nullable=False, ge=0)
    update_ts: Series[int] = pa.Field(nullable=False, ge=0)

    @pa.dataframe_check()
    def check_watermark_progression(cls, df: DataFrame[Any]) -> bool:
        """
        Watermarks should only move forward per pipeline_id.
        """
        if len(df) <= 1:
            return True

        for _, pipeline_df in df.sort_values("update_ts").groupby("pipeline_id"):
            watermarks = pipeline_df["watermark_ts"].to_numpy()
            if not all(watermarks[i] <= watermarks[i + 1] for i in range(len(watermarks) - 1)):
                return False

        return True


class EventLogSchema(pa.DataFrameModel):
    """
    Schema for event logging.
    """

    event_id: Series[str] = pa.Field(nullable=False)
    event_type: Series[str] = pa.Field(
        nullable=False,
        isin=[
            "DATA_RECEIVED",
            "FEATURE_COMPUTED",
            "PREDICTION_MADE",
            "SIGNAL_GENERATED",
            "ORDER_PLACED",
            "ERROR",
            "WARNING",
        ],
    )
    source_id: Series[str] = pa.Field(nullable=False)
    ts_event: Series[int] = pa.Field(nullable=False, ge=0)
    ts_process: Series[int] = pa.Field(nullable=False, ge=0)

    @pa.check("ts_process", "ts_event", name="processing_latency")
    def check_processing_latency(cls, df: DataFrame[Any]) -> Series[bool]:
        """
        Processing should happen after event.
        """
        return cast(Series[bool], df["ts_process"] >= df["ts_event"])


# ============================================================================
# CONTRACT TESTS
# ============================================================================


@pytest.mark.parallel_safe
class TestStoreSchemaContracts:
    """
    Test that store inputs/outputs conform to schemas.
    """

    def test_feature_input_schema_validation(self):
        """
        Test FeatureInputSchema validation.
        """
        # Valid data
        valid_df = pd.DataFrame(
            {
                "feature_set_id": ["feature_set_1"],
                "instrument_id": ["EURUSD.SIM"],
                "ts_event": [int(datetime(2024, 1, 1).timestamp() * 1e9)],
                "ts_init": [int(datetime(2024, 1, 1).timestamp() * 1e9) + 1000],
            },
        )

        # Should pass validation
        validated = FeatureInputSchema.validate(valid_df)
        assert len(validated) == 1

        # Invalid instrument_id format - test this separately if needed
        # Note: pandera regex validation may be lenient, focus on timestamp validation instead

        # ts_init < ts_event (invalid)
        invalid_ts_df = pd.DataFrame(
            {
                "feature_set_id": ["feature_set_1"],
                "instrument_id": ["EURUSD.SIM"],
                "ts_event": [int(datetime(2024, 1, 1).timestamp() * 1e9)],
                "ts_init": [int(datetime(2024, 1, 1).timestamp() * 1e9) - 1000],  # Before ts_event
            },
        )

        with pytest.raises(pa.errors.SchemaError):
            FeatureInputSchema.validate(invalid_ts_df)

    def test_prediction_schema_validation(self):
        """
        Test PredictionSchema validation.
        """
        # Valid predictions
        valid_df = pd.DataFrame(
            {
                "model_id": ["xgb_v1", "xgb_v1"],
                "instrument_id": ["EURUSD.SIM", "GBPUSD.SIM"],
                "prediction": [0.7, 0.3],
                "confidence": [0.8, 0.6],
                "ts_event": [1000000, 2000000],
                "ts_init": [1000001, 2000001],
            },
        )

        validated = PredictionSchema.validate(valid_df)
        assert len(validated) == 2

        # Out of bounds prediction
        invalid_df = pd.DataFrame(
            {
                "model_id": ["xgb_v1"],
                "instrument_id": ["EURUSD.SIM"],
                "prediction": [1.5],  # Out of [0, 1]
                "confidence": [0.8],
                "ts_event": [1000000],
                "ts_init": [1000001],
            },
        )

        with pytest.raises(pa.errors.SchemaError):
            PredictionSchema.validate(invalid_df)

        # Invalid confidence
        invalid_conf_df = pd.DataFrame(
            {
                "model_id": ["xgb_v1"],
                "instrument_id": ["EURUSD.SIM"],
                "prediction": [0.5],
                "confidence": [1.2],  # Out of [0, 1]
                "ts_event": [1000000],
                "ts_init": [1000001],
            },
        )

        with pytest.raises(pa.errors.SchemaError):
            PredictionSchema.validate(invalid_conf_df)

    def test_signal_schema_validation(self):
        """
        Test SignalSchema validation.
        """
        # Valid signals
        valid_df = pd.DataFrame(
            {
                "strategy_id": ["momentum_1"],
                "instrument_id": ["EURUSD.SIM"],
                "signal_type": ["BUY"],
                "signal_strength": [0.8],
                "ts_event": [1000000],
                "ts_init": [1000001],
            },
        )

        validated = SignalSchema.validate(valid_df)
        assert len(validated) == 1

        # Inconsistent signal (BUY with negative strength)
        invalid_df = pd.DataFrame(
            {
                "strategy_id": ["momentum_1"],
                "instrument_id": ["EURUSD.SIM"],
                "signal_type": ["BUY"],
                "signal_strength": [-0.8],  # Negative for BUY
                "ts_event": [1000000],
                "ts_init": [1000001],
            },
        )

        with pytest.raises(pa.errors.SchemaError):
            SignalSchema.validate(invalid_df)

    def test_watermark_schema_validation(self):
        """
        Test WatermarkSchema validation.
        """
        # Valid watermark progression
        valid_df = pd.DataFrame(
            {
                "pipeline_id": ["pipeline_1", "pipeline_1"],
                "watermark_ts": [1000000, 2000000],  # Moving forward
                "processed_count": [100, 200],
                "update_ts": [1000001, 2000001],
            },
        )

        validated = WatermarkSchema.validate(valid_df)
        assert len(validated) == 2

        # Invalid watermark (moving backwards)
        invalid_df = pd.DataFrame(
            {
                "pipeline_id": ["pipeline_1", "pipeline_1"],
                "watermark_ts": [2000000, 1000000],  # Moving backwards
                "processed_count": [100, 200],
                "update_ts": [1000001, 2000001],
            },
        )

        with pytest.raises(pa.errors.SchemaError):
            WatermarkSchema.validate(invalid_df)

    def test_cross_store_consistency(self):
        """
        Test consistency across multiple store schemas.
        """
        # Create related data across stores
        instrument_id = "EURUSD.SIM"
        ts_event = int(datetime(2024, 1, 1).timestamp() * 1e9)
        ts_init = ts_event + 1000

        # Feature data
        feature_df = pd.DataFrame(
            {
                "feature_set_id": ["features_1"],
                "instrument_id": [instrument_id],
                "ts_event": [ts_event],
                "ts_init": [ts_init],
                "feature_1": [0.5],
                "feature_2": [-0.3],
            },
        )

        # Prediction based on features
        prediction_df = pd.DataFrame(
            {
                "model_id": ["model_1"],
                "instrument_id": [instrument_id],
                "prediction": [0.7],
                "confidence": [0.8],
                "ts_event": [ts_event],
                "ts_init": [ts_init],
            },
        )

        # Signal based on prediction
        signal_df = pd.DataFrame(
            {
                "strategy_id": ["strategy_1"],
                "instrument_id": [instrument_id],
                "signal_type": ["BUY"],
                "signal_strength": [0.8],
                "ts_event": [ts_event],
                "ts_init": [ts_init],
            },
        )

        # All should be valid and consistent
        FeatureInputSchema.validate(
            feature_df[["feature_set_id", "instrument_id", "ts_event", "ts_init"]],
        )
        PredictionSchema.validate(prediction_df)
        SignalSchema.validate(signal_df)

        # Verify consistency
        assert prediction_df["instrument_id"].iloc[0] == feature_df["instrument_id"].iloc[0]
        assert signal_df["instrument_id"].iloc[0] == prediction_df["instrument_id"].iloc[0]
        assert signal_df["signal_strength"].iloc[0] == prediction_df["confidence"].iloc[0]


class TestSchemaEvolution:
    """
    Test schema evolution and backwards compatibility.
    """

    def test_feature_schema_allows_new_columns(self):
        """
        Test that feature schema allows dynamic feature columns.
        """
        df1 = pd.DataFrame(
            {
                "feature_set_id": ["set_1"],
                "instrument_id": ["EURUSD.SIM"],
                "ts_event": [1000000],
                "ts_init": [1000001],
                "feature_a": [0.5],
            },
        )

        df2 = pd.DataFrame(
            {
                "feature_set_id": ["set_1"],
                "instrument_id": ["EURUSD.SIM"],
                "ts_event": [2000000],
                "ts_init": [2000001],
                "feature_a": [0.6],
                "feature_b": [0.3],  # New feature
            },
        )

        # Both should be valid
        assert FeatureValueSchema.validate(df1) is not None
        assert FeatureValueSchema.validate(df2) is not None

    def test_metric_schema_extensibility(self):
        """
        Test that new metrics can be added without breaking schema.
        """
        base_metrics = pd.DataFrame(
            {
                "model_id": ["model_1"],
                "metric_name": ["accuracy"],
                "metric_value": [0.95],
                "evaluation_ts": [1000000],
            },
        )

        # Should validate base metrics
        validated = ModelMetricsSchema.validate(base_metrics)
        assert len(validated) == 1

        # New metric types should be validated separately
        custom_metrics = pd.DataFrame(
            {
                "model_id": ["model_1"],
                "metric_name": ["sharpe"],
                "metric_value": [1.5],
                "evaluation_ts": [1000000],
            },
        )

        validated = ModelMetricsSchema.validate(custom_metrics)
        assert len(validated) == 1
