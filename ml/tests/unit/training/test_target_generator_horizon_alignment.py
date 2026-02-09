"""Unit tests for wall-clock horizon alignment in target generation."""

from __future__ import annotations

import numpy as np
import pytest

from ml.config.targets import BinaryTargetConfig
from ml.config.targets import EXECUTION_UNRESOLVED_CONTEXT_FAIL
from ml.config.targets import HORIZON_RESOLUTION_WALL_CLOCK
from ml.config.targets import MulticlassTargetConfig
from ml.config.targets import RegressionTargetConfig
from ml.config.targets import TargetHorizonSpec
from ml.config.targets import TargetSemanticsConfig
from ml.config.targets import build_binary_target_column
from ml.config.targets import build_forward_return_column
from ml.data.common.target_generation import TargetGenerationComponent


@pytest.mark.unit
def test_generate_targets_wall_clock_polars_aligns_by_timestamp() -> None:
    """Verify Polars wall-clock horizons align using timestamps instead of row offsets."""
    pl = pytest.importorskip("polars")
    timestamps_ns = [0, 60_000_000_000, 180_000_000_000, 240_000_000_000]
    close = [100.0, 101.0, 103.0, 104.0]
    df = pl.DataFrame({"timestamp": timestamps_ns, "close": close})

    wall_clock_config = TargetSemanticsConfig(
        horizons=(TargetHorizonSpec(minutes=2),),
        binary=BinaryTargetConfig(enabled=True, threshold_bps=10.0),
        multiclass=MulticlassTargetConfig(enabled=False),
        regression=RegressionTargetConfig(enabled=False),
        horizon_resolution_mode=HORIZON_RESOLUTION_WALL_CLOCK,
        wall_clock_timestamp_column="timestamp",
    )
    bar_index_config = TargetSemanticsConfig(
        horizons=(TargetHorizonSpec(minutes=2),),
        binary=BinaryTargetConfig(enabled=True, threshold_bps=10.0),
        multiclass=MulticlassTargetConfig(enabled=False),
        regression=RegressionTargetConfig(enabled=False),
    )

    component = TargetGenerationComponent()
    wall_clock_result = component.generate_targets_with_semantics(
        df,
        wall_clock_config,
        use_polars=True,
    )
    bar_index_result = component.generate_targets_with_semantics(
        df,
        bar_index_config,
        use_polars=True,
    )

    forward_col = build_forward_return_column("2m")
    wall_clock_forward = np.asarray(
        wall_clock_result.frame.get_column(forward_col).to_list(),
        dtype=float,
    )
    bar_index_forward = np.asarray(
        bar_index_result.frame.get_column(forward_col).to_list(),
        dtype=float,
    )

    expected = np.array([0.03, (103.0 - 101.0) / 101.0, 0.0, 0.0], dtype=float)
    assert np.allclose(wall_clock_forward, expected, rtol=1e-6, atol=1e-9)
    assert not np.isclose(wall_clock_forward[1], bar_index_forward[1])

    binary_col = build_binary_target_column("2m")
    assert wall_clock_result.frame.get_column(binary_col).to_list() == [1, 1, 0, 0]


@pytest.mark.unit
def test_generate_targets_wall_clock_pandas_aligns_by_timestamp() -> None:
    """Verify Pandas wall-clock horizons align using timestamps and zero-fill missing future rows."""
    pd = pytest.importorskip("pandas")
    timestamps_ns = [0, 60_000_000_000, 180_000_000_000, 240_000_000_000]
    close = [100.0, 101.0, 103.0, 104.0]
    df = pd.DataFrame({"timestamp": timestamps_ns, "close": close})
    config = TargetSemanticsConfig(
        horizons=(TargetHorizonSpec(minutes=2),),
        binary=BinaryTargetConfig(enabled=True, threshold_bps=10.0),
        multiclass=MulticlassTargetConfig(enabled=False),
        regression=RegressionTargetConfig(enabled=False),
        horizon_resolution_mode=HORIZON_RESOLUTION_WALL_CLOCK,
        wall_clock_timestamp_column="timestamp",
    )

    result = TargetGenerationComponent().generate_targets_with_semantics(
        df,
        config,
        use_polars=False,
    )
    forward_col = build_forward_return_column("2m")
    actual = result.frame[forward_col].to_numpy(dtype=float)
    expected = np.array([0.03, (103.0 - 101.0) / 101.0, 0.0, 0.0], dtype=float)
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-9)
    assert float(actual[-1]) == 0.0


@pytest.mark.unit
def test_generate_targets_wall_clock_when_timestamp_missing_raises_key_error() -> None:
    """Verify wall-clock mode requires configured timestamp column."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    config = TargetSemanticsConfig(
        horizons=(TargetHorizonSpec(minutes=1),),
        horizon_resolution_mode=HORIZON_RESOLUTION_WALL_CLOCK,
        wall_clock_timestamp_column="timestamp",
    )

    with pytest.raises(KeyError, match="timestamp"):
        TargetGenerationComponent().generate_targets_with_semantics(df, config, use_polars=False)


@pytest.mark.unit
def test_build_target_semantics_metadata_includes_wall_clock_alignment_fields() -> None:
    """Verify emitted target semantics metadata includes wall-clock mode/alignment fields."""
    pd = pytest.importorskip("pandas")
    config = TargetSemanticsConfig(
        horizons=(TargetHorizonSpec(minutes=1),),
        horizon_resolution_mode=HORIZON_RESOLUTION_WALL_CLOCK,
        wall_clock_timestamp_column="ts_event",
    )
    result = TargetGenerationComponent().generate_targets_with_semantics(
        pd.DataFrame({"ts_event": [0, 60_000_000_000], "close": [100.0, 101.0]}),
        config,
        use_polars=False,
    )
    metadata = result.semantics
    assert metadata["horizon_resolution_mode"] == HORIZON_RESOLUTION_WALL_CLOCK
    alignment = metadata["horizon_alignment"]
    assert isinstance(alignment, dict)
    assert alignment["mode"] == HORIZON_RESOLUTION_WALL_CLOCK
    assert alignment["timestamp_column"] == "ts_event"
    assert alignment["future_anchor"] == "first_timestamp_at_or_after_horizon"
    assert alignment["insufficient_future_handling"] == "zero_return"
    execution = metadata["execution"]
    assert isinstance(execution, dict)
    assert execution["entry_price_column"] == "close"
    assert execution["exit_price_column"] == "close"
    assert execution["latency_bars"] == 0
    assert execution["latency_unit"] == "bars"
    assert execution["unresolved_context_mode"] == "zero_return"
    assert execution["unresolved_context_return"] == 0.0


@pytest.mark.unit
def test_generate_targets_with_execution_latency_zero_fills_unresolved_rows() -> None:
    """Verify unresolved execution context is deterministic under zero-return fallback."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]})
    config = TargetSemanticsConfig(
        horizons=(TargetHorizonSpec(minutes=1),),
        binary=BinaryTargetConfig(enabled=False),
        multiclass=MulticlassTargetConfig(enabled=False),
        regression=RegressionTargetConfig(enabled=False),
        execution_latency_bars=2,
        unresolved_execution_context_mode="zero_return",
    )

    result = TargetGenerationComponent().generate_targets_with_semantics(
        df,
        config,
        use_polars=False,
    )
    forward_col = build_forward_return_column("1m")
    actual = result.frame[forward_col].to_numpy(dtype=float)
    expected = np.array([(103.0 - 102.0) / 102.0, 0.0, 0.0, 0.0], dtype=float)
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-9)


@pytest.mark.unit
def test_generate_targets_when_unresolved_execution_mode_fail_raises_value_error() -> None:
    """Verify unresolved execution context fail mode raises deterministically."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]})
    config = TargetSemanticsConfig(
        horizons=(TargetHorizonSpec(minutes=1),),
        binary=BinaryTargetConfig(enabled=False),
        multiclass=MulticlassTargetConfig(enabled=False),
        regression=RegressionTargetConfig(enabled=False),
        execution_latency_bars=2,
        unresolved_execution_context_mode=EXECUTION_UNRESOLVED_CONTEXT_FAIL,
    )

    with pytest.raises(ValueError, match="execution context unresolved"):
        TargetGenerationComponent().generate_targets_with_semantics(
            df,
            config,
            use_polars=False,
        )
