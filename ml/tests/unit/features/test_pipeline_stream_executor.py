"""Unit tests for PipelineStreamExecutor and config derivation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.config.constants import MLConstants
from ml.data.common.pipeline_batch import PipelineBatchContext
from ml.data.common.pipeline_batch import PipelineBatchExecutor
from ml.data.providers.calendar import MarketCalendarProvider
from ml.data.providers.events import EventScheduleProvider
from ml.data.sources.calendar import SimpleCalendarSource
from ml.data.sources.events import SimpleEventSource
from ml.features.common.feature_calculator import FeatureCalculator
from ml.features.config import FeatureConfig
from ml.features.config import derive_ohlcv_feature_config
from ml.features.indicators import IndicatorManager
from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec
from ml.features.pipeline import TransformSpec
from ml.features.pipeline_stream import PipelineStreamContext
from ml.features.pipeline_stream import PipelineStreamExecutor
from ml.registry.base import DataRequirements


@pytest.fixture
def base_feature_config() -> FeatureConfig:
    """Baseline FeatureConfig for streaming tests."""
    return FeatureConfig(
        return_periods=[1, 5],
        momentum_periods=[1, 5],
        volume_ma_periods=[5],
        ema_fast=12,
        ema_slow=26,
        rsi_period=14,
        bb_period=20,
        bb_std=2.0,
        atr_period=14,
        enable_returns=True,
        enable_momentum=True,
        enable_volatility=True,
        enable_technical=False,
        include_microstructure=False,
        include_trade_flow=False,
    )


@pytest.fixture
def indicator_manager_with_history(base_feature_config: FeatureConfig) -> IndicatorManager:
    """IndicatorManager seeded with synthetic history."""
    manager = IndicatorManager(base_feature_config)
    for i in range(30):
        manager.update_from_values(
            close=100.0 + i * 0.2,
            high=101.0 + i * 0.2,
            low=99.0 + i * 0.2,
            volume=1_000_000.0 + i * 500,
        )
    return manager


@pytest.fixture
def current_bar() -> dict[str, float]:
    """Current bar payload for streaming execution."""
    return {
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1_000_000.0,
    }


@pytest.fixture
def sample_ohlcv_dataframe() -> pd.DataFrame:
    """Synthetic OHLCV sample for batch/stream parity checks."""
    rng = np.random.default_rng(7)
    rows = 64
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="1min", tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.1, size=rows))
    open_ = close + rng.normal(0.0, 0.05, size=rows)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.02, size=rows))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.02, size=rows))
    volume = 1_000_000.0 + rng.integers(0, 5000, size=rows)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class StubMacroTransform:
    """Stub macro transform for streaming executor tests."""

    def __init__(self) -> None:
        self.macro_series_ids = ["PAYEMS"]
        self.include_revisions = False
        self.revision_mode = "core"
        self.include_composites = False
        self.last_ts_event: int | None = None

    def compute_realtime(self, bar: object | None = None, ts_event: int | None = None) -> dict[str, float]:
        self.last_ts_event = ts_event
        return {"PAYEMS__value_real_time": 42.0}


def test_derive_ohlcv_feature_config_when_trade_flow_enabled_promotes_requirements(
    base_feature_config: FeatureConfig,
) -> None:
    """Trade flow transforms should promote data requirements to L1_L2_L3."""
    transforms = [TransformSpec(name="trade_flow", params={})]

    derived = derive_ohlcv_feature_config(
        base_feature_config,
        transforms,
        allowable=DataRequirements.L1_ONLY,
    )

    assert derived.include_trade_flow is True
    assert derived.data_requirements == DataRequirements.L1_L2_L3


def test_stream_executor_rejects_calendar_without_provider(
    base_feature_config: FeatureConfig,
    indicator_manager_with_history: IndicatorManager,
) -> None:
    """Calendar transforms require a provider in streaming context."""
    spec = PipelineSpec(transforms=[TransformSpec(name="calendar", params={})])
    context = PipelineStreamContext(
        feature_config=base_feature_config,
        indicator_manager=indicator_manager_with_history,
    )

    with pytest.raises(ValueError, match="calendar_provider"):
        PipelineStreamExecutor(spec, allowable=DataRequirements.L1_ONLY, context=context)


def test_stream_executor_rejects_macro_and_technical_extras(
    base_feature_config: FeatureConfig,
    indicator_manager_with_history: IndicatorManager,
) -> None:
    """Unsupported transforms should still be rejected in streaming mode."""
    spec = PipelineSpec(
        transforms=[
            TransformSpec(name="macro", params={"series_ids": ["PAYEMS"]}),
            TransformSpec(name="keltner", params={}),
        ],
    )
    context = PipelineStreamContext(
        feature_config=base_feature_config,
        indicator_manager=indicator_manager_with_history,
        macro_transform=StubMacroTransform(),
    )

    with pytest.raises(ValueError, match="keltner"):
        PipelineStreamExecutor(spec, allowable=DataRequirements.L1_ONLY, context=context)


def test_stream_executor_supports_calendar_with_provider(
    base_feature_config: FeatureConfig,
    indicator_manager_with_history: IndicatorManager,
    current_bar: dict[str, float],
) -> None:
    """Calendar transforms should run when a provider is supplied."""
    spec = PipelineSpec(
        transforms=[
            TransformSpec(name="returns", params={"periods": [1]}),
            TransformSpec(name="calendar", params={}),
        ],
    )
    context = PipelineStreamContext(
        feature_config=base_feature_config,
        indicator_manager=indicator_manager_with_history,
        calendar_provider=MarketCalendarProvider(SimpleCalendarSource()),
    )
    executor = PipelineStreamExecutor(spec, allowable=DataRequirements.L1_ONLY, context=context)

    indicator_manager_with_history.update_from_values(
        close=current_bar["close"],
        high=current_bar["high"],
        low=current_bar["low"],
        volume=current_bar["volume"],
    )
    timestamp_ns = int(pd.Timestamp("2024-01-02T15:00:00Z").value)
    features = executor.execute(current_bar, timestamp_ns=timestamp_ns)
    names = list(executor.feature_names)

    hour_idx = names.index("hour_sin")
    assert -1.0 <= float(features[hour_idx]) <= 1.0


def test_stream_executor_supports_event_schedule_with_provider(
    base_feature_config: FeatureConfig,
    indicator_manager_with_history: IndicatorManager,
    current_bar: dict[str, float],
) -> None:
    """Event schedule transforms should run when a provider is supplied."""
    spec = PipelineSpec(
        transforms=[
            TransformSpec(name="returns", params={"periods": [1]}),
            TransformSpec(name="event_schedule", params={}),
        ],
    )
    context = PipelineStreamContext(
        feature_config=base_feature_config,
        indicator_manager=indicator_manager_with_history,
        event_provider=EventScheduleProvider(SimpleEventSource()),
        event_instruments=["AAPL"],
    )
    executor = PipelineStreamExecutor(spec, allowable=DataRequirements.L1_ONLY, context=context)

    indicator_manager_with_history.update_from_values(
        close=current_bar["close"],
        high=current_bar["high"],
        low=current_bar["low"],
        volume=current_bar["volume"],
    )
    timestamp_ns = int(pd.Timestamp("2024-01-15T15:00:00Z").value)
    features = executor.execute(current_bar, timestamp_ns=timestamp_ns)
    names = list(executor.feature_names)

    hours_idx = names.index("hours_to_earnings")
    assert np.isfinite(features[hours_idx])


def test_stream_executor_supports_macro_transform(
    base_feature_config: FeatureConfig,
    indicator_manager_with_history: IndicatorManager,
    current_bar: dict[str, float],
) -> None:
    """Macro transforms should run when a macro transform is supplied."""
    spec = PipelineSpec(
        transforms=[
            TransformSpec(
                name="macro",
                params={"series_ids": ["PAYEMS"], "include_revisions": False},
            ),
            TransformSpec(name="returns", params={"periods": [1]}),
        ],
    )
    macro_transform = StubMacroTransform()
    context = PipelineStreamContext(
        feature_config=base_feature_config,
        indicator_manager=indicator_manager_with_history,
        macro_transform=macro_transform,
    )
    executor = PipelineStreamExecutor(spec, allowable=DataRequirements.L1_ONLY, context=context)

    indicator_manager_with_history.update_from_values(
        close=current_bar["close"],
        high=current_bar["high"],
        low=current_bar["low"],
        volume=current_bar["volume"],
    )
    timestamp_ns = int(pd.Timestamp("2024-01-02T15:00:00Z").value)
    features = executor.execute(current_bar, timestamp_ns=timestamp_ns)
    names = list(executor.feature_names)

    macro_idx = names.index("PAYEMS__value_real_time")
    assert features[macro_idx] == pytest.approx(42.0)
    assert macro_transform.last_ts_event == timestamp_ns


def test_stream_executor_when_macro_transform_enabled_without_timestamp_raises(
    base_feature_config: FeatureConfig,
    indicator_manager_with_history: IndicatorManager,
    current_bar: dict[str, float],
) -> None:
    """Macro transforms require an explicit timestamp for causality bounds."""
    spec = PipelineSpec(
        transforms=[
            TransformSpec(
                name="macro",
                params={"series_ids": ["PAYEMS"], "include_revisions": False},
            ),
            TransformSpec(name="returns", params={"periods": [1]}),
        ],
    )
    context = PipelineStreamContext(
        feature_config=base_feature_config,
        indicator_manager=indicator_manager_with_history,
        macro_transform=StubMacroTransform(),
    )
    executor = PipelineStreamExecutor(spec, allowable=DataRequirements.L1_ONLY, context=context)

    indicator_manager_with_history.update_from_values(
        close=current_bar["close"],
        high=current_bar["high"],
        low=current_bar["low"],
        volume=current_bar["volume"],
    )
    with pytest.raises(ValueError, match="macro/calendar/event transforms"):
        executor.execute(current_bar)


def test_stream_executor_gates_microstructure_by_data_requirements(
    base_feature_config: FeatureConfig,
    indicator_manager_with_history: IndicatorManager,
) -> None:
    """Microstructure transforms should be gated by DataRequirements."""
    spec = PipelineSpec(transforms=[TransformSpec(name="microstructure", params={})])
    context = PipelineStreamContext(
        feature_config=base_feature_config,
        indicator_manager=indicator_manager_with_history,
    )

    with pytest.raises(ValueError, match="requires l1_l2"):
        PipelineStreamExecutor(spec, allowable=DataRequirements.L1_ONLY, context=context)


def test_stream_executor_gates_trade_flow_by_data_requirements(
    base_feature_config: FeatureConfig,
    indicator_manager_with_history: IndicatorManager,
) -> None:
    """Trade flow transforms should be gated by DataRequirements."""
    spec = PipelineSpec(transforms=[TransformSpec(name="trade_flow", params={})])
    context = PipelineStreamContext(
        feature_config=base_feature_config,
        indicator_manager=indicator_manager_with_history,
    )

    with pytest.raises(ValueError, match="requires l1_l2_l3"):
        PipelineStreamExecutor(spec, allowable=DataRequirements.L1_L2, context=context)


def test_stream_executor_matches_batch_for_calendar_features(
    base_feature_config: FeatureConfig,
    indicator_manager_with_history: IndicatorManager,
    current_bar: dict[str, float],
) -> None:
    """Calendar features should align between batch and stream paths."""
    spec = PipelineSpec(transforms=[TransformSpec(name="calendar", params={})])
    provider = MarketCalendarProvider(SimpleCalendarSource())
    allowable = DataRequirements.L1_ONLY

    batch_context = PipelineBatchContext(
        feature_config=base_feature_config,
        calendar_provider=provider,
    )
    batch_executor = PipelineBatchExecutor(spec, allowable=allowable, context=batch_context)
    timestamp = pd.Timestamp("2024-01-02T15:00:00Z")
    batch_df = batch_executor.execute_pandas(pd.DataFrame({"timestamp": [timestamp]}))
    feature_names = PipelineRunner(spec, allowable=allowable).compute_feature_names()
    batch_features = batch_df[feature_names].iloc[0].to_numpy(dtype=np.float32)

    indicator_manager_with_history.update_from_values(
        close=current_bar["close"],
        high=current_bar["high"],
        low=current_bar["low"],
        volume=current_bar["volume"],
    )
    stream_context = PipelineStreamContext(
        feature_config=base_feature_config,
        indicator_manager=indicator_manager_with_history,
        calendar_provider=provider,
    )
    stream_executor = PipelineStreamExecutor(spec, allowable=allowable, context=stream_context)
    stream_features = stream_executor.execute(
        current_bar,
        timestamp_ns=int(timestamp.value),
    )

    np.testing.assert_allclose(
        batch_features,
        stream_features,
        rtol=0.0,
        atol=MLConstants.FEATURE_PARITY_TOLERANCE,
    )


def test_stream_executor_matches_batch_for_event_schedule(
    base_feature_config: FeatureConfig,
    indicator_manager_with_history: IndicatorManager,
    current_bar: dict[str, float],
) -> None:
    """Event schedule features should align between batch and stream paths."""
    spec = PipelineSpec(transforms=[TransformSpec(name="event_schedule", params={})])
    provider = EventScheduleProvider(SimpleEventSource())
    allowable = DataRequirements.L1_ONLY

    batch_context = PipelineBatchContext(
        feature_config=base_feature_config,
        event_provider=provider,
        event_instruments=["AAPL"],
    )
    batch_executor = PipelineBatchExecutor(spec, allowable=allowable, context=batch_context)
    timestamp = pd.Timestamp("2024-01-15T15:00:00Z")
    batch_df = batch_executor.execute_pandas(pd.DataFrame({"timestamp": [timestamp]}))
    feature_names = PipelineRunner(spec, allowable=allowable).compute_feature_names()
    batch_features = batch_df[feature_names].iloc[0].to_numpy(dtype=np.float32)

    indicator_manager_with_history.update_from_values(
        close=current_bar["close"],
        high=current_bar["high"],
        low=current_bar["low"],
        volume=current_bar["volume"],
    )
    stream_context = PipelineStreamContext(
        feature_config=base_feature_config,
        indicator_manager=indicator_manager_with_history,
        event_provider=provider,
        event_instruments=["AAPL"],
    )
    stream_executor = PipelineStreamExecutor(spec, allowable=allowable, context=stream_context)
    stream_features = stream_executor.execute(
        current_bar,
        timestamp_ns=int(timestamp.value),
    )

    np.testing.assert_allclose(
        batch_features,
        stream_features,
        rtol=0.0,
        atol=MLConstants.FEATURE_PARITY_TOLERANCE,
    )


def test_stream_executor_returns_feature_vector_matching_names(
    base_feature_config: FeatureConfig,
    indicator_manager_with_history: IndicatorManager,
    current_bar: dict[str, float],
) -> None:
    """Stream execution returns a feature vector aligned to pipeline names."""
    spec = PipelineSpec(
        transforms=[
            TransformSpec(name="returns", params={"periods": [1, 5]}),
            TransformSpec(name="momentum", params={"periods": [1, 5]}),
            TransformSpec(name="volatility", params={}),
            TransformSpec(name="volume_ratio", params={"periods": [5]}),
            TransformSpec(name="core_indicators", params={}),
        ],
    )
    context = PipelineStreamContext(
        feature_config=base_feature_config,
        indicator_manager=indicator_manager_with_history,
    )
    executor = PipelineStreamExecutor(
        spec,
        allowable=DataRequirements.L1_ONLY,
        context=context,
    )

    indicator_manager_with_history.update_from_values(
        close=current_bar["close"],
        high=current_bar["high"],
        low=current_bar["low"],
        volume=current_bar["volume"],
    )
    features = executor.execute(current_bar)

    expected_names = PipelineRunner(spec, allowable=DataRequirements.L1_ONLY).compute_feature_names()
    assert list(executor.feature_names) == expected_names
    assert features.shape[0] == len(expected_names)


def test_stream_executor_matches_batch_for_ohlcv_transforms(
    base_feature_config: FeatureConfig,
    sample_ohlcv_dataframe: pd.DataFrame,
) -> None:
    """Batch and stream outputs should align for OHLCV transforms."""
    spec = PipelineSpec(
        transforms=[
            TransformSpec(name="returns", params={"periods": [1, 5]}),
            TransformSpec(name="momentum", params={"periods": [1, 5]}),
            TransformSpec(name="volatility", params={}),
            TransformSpec(name="volume_ratio", params={"periods": [5]}),
        ],
    )
    allowable = DataRequirements.L1_ONLY
    derived = derive_ohlcv_feature_config(base_feature_config, spec.transforms, allowable=allowable)

    batch_context = PipelineBatchContext(feature_config=derived)
    batch_executor = PipelineBatchExecutor(spec, allowable=allowable, context=batch_context)
    batch_df = batch_executor.execute_pandas(sample_ohlcv_dataframe)
    feature_names = PipelineRunner(spec, allowable=allowable).compute_feature_names()
    batch_features = batch_df[feature_names].to_numpy(dtype=np.float32)

    indicator_manager = IndicatorManager(derived)
    stream_context = PipelineStreamContext(
        feature_config=derived,
        indicator_manager=indicator_manager,
    )
    stream_executor = PipelineStreamExecutor(spec, allowable=allowable, context=stream_context)

    stream_rows: list[np.ndarray] = []
    for row in sample_ohlcv_dataframe.itertuples(index=False):
        indicator_manager.update_from_values(
            close=float(row.close),
            high=float(row.high),
            low=float(row.low),
            volume=float(row.volume),
        )
        current_bar = {
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        stream_rows.append(stream_executor.execute(current_bar).copy())

    stream_features = np.stack(stream_rows, axis=0)
    warmup = FeatureCalculator(derived)._required_warmup_history_len
    for idx in range(warmup, len(sample_ohlcv_dataframe)):
        np.testing.assert_allclose(
            batch_features[idx],
            stream_features[idx],
            rtol=0.0,
            atol=MLConstants.FEATURE_PARITY_TOLERANCE,
        )


def test_stream_executor_matches_batch_for_microstructure_transforms(
    base_feature_config: FeatureConfig,
    sample_ohlcv_dataframe: pd.DataFrame,
) -> None:
    """Batch and stream outputs should align for microstructure transforms."""
    spec = PipelineSpec(transforms=[TransformSpec(name="microstructure", params={})])
    allowable = DataRequirements.L1_L2
    derived = derive_ohlcv_feature_config(base_feature_config, spec.transforms, allowable=allowable)

    batch_context = PipelineBatchContext(feature_config=derived)
    batch_executor = PipelineBatchExecutor(spec, allowable=allowable, context=batch_context)
    batch_df = batch_executor.execute_pandas(sample_ohlcv_dataframe)
    feature_names = PipelineRunner(spec, allowable=allowable).compute_feature_names()
    batch_features = batch_df[feature_names].to_numpy(dtype=np.float32)

    indicator_manager = IndicatorManager(derived)
    stream_context = PipelineStreamContext(
        feature_config=derived,
        indicator_manager=indicator_manager,
    )
    stream_executor = PipelineStreamExecutor(spec, allowable=allowable, context=stream_context)

    stream_rows: list[np.ndarray] = []
    for row in sample_ohlcv_dataframe.itertuples(index=False):
        indicator_manager.update_from_values(
            close=float(row.close),
            high=float(row.high),
            low=float(row.low),
            volume=float(row.volume),
        )
        current_bar = {
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        stream_rows.append(stream_executor.execute(current_bar).copy())

    stream_features = np.stack(stream_rows, axis=0)
    warmup = FeatureCalculator(derived)._required_warmup_history_len
    for idx in range(warmup, len(sample_ohlcv_dataframe)):
        np.testing.assert_allclose(
            batch_features[idx],
            stream_features[idx],
            rtol=0.0,
            atol=MLConstants.FEATURE_PARITY_TOLERANCE,
        )


def test_stream_executor_matches_batch_for_trade_flow_transforms(
    base_feature_config: FeatureConfig,
    sample_ohlcv_dataframe: pd.DataFrame,
) -> None:
    """Batch and stream outputs should align for trade flow transforms."""
    spec = PipelineSpec(transforms=[TransformSpec(name="trade_flow", params={})])
    allowable = DataRequirements.L1_L2_L3
    derived = derive_ohlcv_feature_config(base_feature_config, spec.transforms, allowable=allowable)

    batch_context = PipelineBatchContext(feature_config=derived)
    batch_executor = PipelineBatchExecutor(spec, allowable=allowable, context=batch_context)
    batch_df = batch_executor.execute_pandas(sample_ohlcv_dataframe)
    feature_names = PipelineRunner(spec, allowable=allowable).compute_feature_names()
    batch_features = batch_df[feature_names].to_numpy(dtype=np.float32)

    indicator_manager = IndicatorManager(derived)
    stream_context = PipelineStreamContext(
        feature_config=derived,
        indicator_manager=indicator_manager,
    )
    stream_executor = PipelineStreamExecutor(spec, allowable=allowable, context=stream_context)

    stream_rows: list[np.ndarray] = []
    for row in sample_ohlcv_dataframe.itertuples(index=False):
        indicator_manager.update_from_values(
            close=float(row.close),
            high=float(row.high),
            low=float(row.low),
            volume=float(row.volume),
        )
        current_bar = {
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        stream_rows.append(stream_executor.execute(current_bar).copy())

    stream_features = np.stack(stream_rows, axis=0)
    warmup = FeatureCalculator(derived)._required_warmup_history_len
    for idx in range(warmup, len(sample_ohlcv_dataframe)):
        np.testing.assert_allclose(
            batch_features[idx],
            stream_features[idx],
            rtol=0.0,
            atol=MLConstants.FEATURE_PARITY_TOLERANCE,
        )
