"""Unit tests for target generator semantics."""

from __future__ import annotations

import numpy as np
import pytest

from ml.config.targets import BinaryTargetConfig
from ml.config.targets import MulticlassTargetConfig
from ml.config.targets import RegressionTargetConfig
from ml.config.targets import TargetCostModelConfig
from ml.config.targets import TargetHorizonSpec
from ml.config.targets import TargetSemanticsConfig
from ml.config.targets import build_binary_target_column
from ml.config.targets import build_cost_return_column
from ml.config.targets import build_forward_return_column
from ml.config.targets import build_multiclass_target_column
from ml.config.targets import build_regression_target_column
from ml.training.datasets.target_generator import TargetGenerator


def test_generate_targets_multi_horizon_values_polars() -> None:
    """Validate multi-horizon targets and labels for Polars inputs."""
    pl = pytest.importorskip("polars")

    df = pl.DataFrame({"close": [100.0, 101.0, 99.0, 100.0]})
    config = TargetSemanticsConfig(
        horizons=(TargetHorizonSpec(minutes=1), TargetHorizonSpec(minutes=2)),
        binary=BinaryTargetConfig(enabled=True, threshold_bps=10.0),
        multiclass=MulticlassTargetConfig(
            enabled=True,
            short_threshold_bps=10.0,
            long_threshold_bps=10.0,
        ),
        regression=RegressionTargetConfig(enabled=True),
    )

    result = TargetGenerator().generate_targets_with_semantics(df, config, use_polars=True)
    targets = result.frame

    expected_columns = {
        build_forward_return_column("1m"),
        build_forward_return_column("2m"),
        build_binary_target_column("1m"),
        build_binary_target_column("2m"),
        build_multiclass_target_column("1m"),
        build_multiclass_target_column("2m"),
        build_regression_target_column("1m"),
        build_regression_target_column("2m"),
    }
    assert expected_columns.issubset(set(targets.columns))

    forward_1 = np.asarray(
        targets.get_column(build_forward_return_column("1m")).to_list(),
        dtype=float,
    )
    forward_2 = np.asarray(
        targets.get_column(build_forward_return_column("2m")).to_list(),
        dtype=float,
    )

    expected_forward_1 = np.array([0.01, -0.019801980198, 0.010101010101, 0.0])
    expected_forward_2 = np.array([-0.01, -0.009900990099, 0.0, 0.0])
    assert np.allclose(forward_1, expected_forward_1, rtol=1e-6, atol=1e-9)
    assert np.allclose(forward_2, expected_forward_2, rtol=1e-6, atol=1e-9)

    regression_1 = np.asarray(
        targets.get_column(build_regression_target_column("1m")).to_list(),
        dtype=float,
    )
    regression_2 = np.asarray(
        targets.get_column(build_regression_target_column("2m")).to_list(),
        dtype=float,
    )
    assert np.allclose(regression_1, expected_forward_1, rtol=1e-6, atol=1e-9)
    assert np.allclose(regression_2, expected_forward_2, rtol=1e-6, atol=1e-9)

    bin_1 = targets.get_column(build_binary_target_column("1m")).to_list()
    bin_2 = targets.get_column(build_binary_target_column("2m")).to_list()
    assert bin_1 == [1, 0, 1, 0]
    assert bin_2 == [0, 0, 0, 0]

    cls_1 = targets.get_column(build_multiclass_target_column("1m")).to_list()
    cls_2 = targets.get_column(build_multiclass_target_column("2m")).to_list()
    assert cls_1 == [1, -1, 1, 0]
    assert cls_2 == [-1, -1, 0, 0]


def test_cost_aware_binary_labels_reduce_with_costs() -> None:
    """Cost-aware binary labels should not increase positives as costs rise."""
    pl = pytest.importorskip("polars")

    close = [100.0, 100.2, 100.4004, 100.6012008]
    df = pl.DataFrame({"close": close})

    def _config(cost_bps: float) -> TargetSemanticsConfig:
        return TargetSemanticsConfig(
            horizons=(TargetHorizonSpec(minutes=1),),
            cost_model=TargetCostModelConfig(cost_bps=cost_bps),
            binary=BinaryTargetConfig(
                enabled=True,
                threshold_bps=10.0,
                return_basis="cost",
            ),
        )

    generator = TargetGenerator()
    result_zero = generator.generate_targets_with_semantics(df, _config(0.0), use_polars=True)
    result_cost = generator.generate_targets_with_semantics(df, _config(10.0), use_polars=True)

    bin_col = build_binary_target_column("1m")
    positives_zero = int(sum(result_zero.frame.get_column(bin_col).to_list()))
    positives_cost = int(sum(result_cost.frame.get_column(bin_col).to_list()))
    assert positives_cost <= positives_zero

    cost_col = build_cost_return_column("1m")
    assert cost_col in result_cost.frame.columns

    forward_col = build_forward_return_column("1m")
    forward_returns = np.asarray(
        result_zero.frame.get_column(forward_col).to_list(),
        dtype=float,
    )
    expected_cost = forward_returns - TargetCostModelConfig(cost_bps=10.0).round_trip_decimal
    actual_cost = np.asarray(
        result_cost.frame.get_column(cost_col).to_list(),
        dtype=float,
    )
    assert np.allclose(actual_cost, expected_cost, rtol=1e-6, atol=1e-9)
