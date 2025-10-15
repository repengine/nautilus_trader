"""
Unit tests for portfolio strategy implementations.

These tests focus on the FactorTiltStrategy hybrid forecasting logic and
basic weight normalization guarantees, ensuring Phase 3 backtesting relies on
stable, bounded signals.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from playground.backtest.strategies import HYBRID_STABILITY_WINDOW
from playground.backtest.strategies import FactorTiltStrategy
from playground.risk_model.dataset import CoverageSummary
from playground.risk_model.dataset import SectorDataset


def _build_constant_dataset(num_days: int, constant_return: float) -> SectorDataset:
    """Create a dataset with constant sector and factor returns."""
    start = datetime(2020, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(days=i) for i in range(num_days)]

    factor_returns = pl.DataFrame({
        "timestamp": timestamps,
        "factor_duration": [constant_return] * num_days,
        "factor_credit": [constant_return] * num_days,
        "factor_liquidity": [constant_return] * num_days,
    })

    sector_rows: list[dict[str, object]] = []
    for timestamp in timestamps:
        for symbol in ("SPY", "XLK", "AGG"):
            sector_rows.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "return": constant_return,
            })

    sector_returns = pl.DataFrame(sector_rows)

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=num_days,
        factor_expected_days=num_days,
        sector_coverage={"SPY": 1.0, "XLK": 1.0, "AGG": 1.0},
        factor_coverage={
            "factor_duration": 1.0,
            "factor_credit": 1.0,
            "factor_liquidity": 1.0,
        },
    )

    return SectorDataset(
        sector_returns=sector_returns,
        factor_returns=factor_returns,
        coverage=coverage,
    )


def _build_negative_liquidity_dataset(num_days: int, negative_value: float) -> SectorDataset:
    """Dataset with persistent negative liquidity factor for scaling tests."""
    dataset = _build_constant_dataset(num_days=num_days, constant_return=0.0005)
    factor_returns = dataset.factor_returns.with_columns([
        pl.lit(negative_value).alias("factor_liquidity"),
    ])
    return SectorDataset(
        sector_returns=dataset.sector_returns,
        factor_returns=factor_returns,
        coverage=dataset.coverage,
    )

def test_hybrid_forecast_constant_series_full_history() -> None:
    """Hybrid forecast should respect constant signals when history is sufficient."""
    dataset = _build_constant_dataset(num_days=252, constant_return=0.001)
    strategy = FactorTiltStrategy()
    latest_date = dataset.factor_returns["timestamp"][-1]

    forecasts = strategy._forecast_factor_returns(latest_date, dataset)

    assert forecasts == pytest.approx({
        "factor_duration": 0.001,
        "factor_credit": 0.001,
        "factor_liquidity": 0.001,
    }, rel=1e-6)


def test_hybrid_forecast_shrinks_with_limited_history() -> None:
    """Forecasts should shrink toward zero when limited history is available."""
    dataset = _build_constant_dataset(num_days=90, constant_return=0.001)
    strategy = FactorTiltStrategy(min_observations=60)
    latest_date = dataset.factor_returns["timestamp"][-1]

    forecasts = strategy._forecast_factor_returns(latest_date, dataset)

    stability_window = max(strategy.min_observations, HYBRID_STABILITY_WINDOW)
    shrink_ratio = 90 / stability_window

    assert forecasts == pytest.approx({
        "factor_duration": 0.001 * shrink_ratio,
        "factor_credit": 0.001 * shrink_ratio,
        "factor_liquidity": 0.001 * shrink_ratio,
    }, rel=1e-6)


def test_regime_scaling_applies_multipliers() -> None:
    """Regime-aware scaling should apply overall and per-factor multipliers."""
    dataset = _build_constant_dataset(num_days=260, constant_return=0.001)
    latest_date = dataset.factor_returns["timestamp"][-1]

    def resolver(_: datetime) -> str | None:
        return "Rate Hiking Cycle"

    strategy = FactorTiltStrategy(
        regime_scaling=True,
        regime_scaling_map={"Rate Hiking Cycle": 0.8},
        regime_resolver=resolver,
        regime_factor_multipliers={"Rate Hiking Cycle": {"factor_liquidity": 0.5}},
    )

    forecasts = strategy._forecast_factor_returns(latest_date, dataset)

    assert forecasts["factor_duration"] == pytest.approx(0.001 * 0.8, rel=1e-6)
    assert forecasts["factor_credit"] == pytest.approx(0.001 * 0.8, rel=1e-6)
    assert forecasts["factor_liquidity"] == pytest.approx(0.001 * 0.8 * 0.5, rel=1e-6)


def test_regime_scaling_respects_floor() -> None:
    """Scaling multipliers should never drop below the configured floor."""
    dataset = _build_constant_dataset(num_days=260, constant_return=0.001)
    latest_date = dataset.factor_returns["timestamp"][-1]

    def resolver(_: datetime) -> str | None:
        return "Rate Hiking Cycle"

    strategy = FactorTiltStrategy(
        regime_scaling=True,
        regime_scaling_floor=0.45,
        regime_scaling_map={"Rate Hiking Cycle": 0.1},
        regime_resolver=resolver,
        regime_factor_multipliers={"Rate Hiking Cycle": {"factor_liquidity": 0.1}},
    )

    forecasts = strategy._forecast_factor_returns(latest_date, dataset)

    assert forecasts["factor_duration"] == pytest.approx(0.001 * 0.45, rel=1e-6)
    assert forecasts["factor_liquidity"] == pytest.approx(0.001 * 0.45 * 0.45, rel=1e-6)


def _build_linear_dataset(num_days: int) -> SectorDataset:
    """Create a dataset with deterministic factor structure for weight testing."""
    start = datetime(2020, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(days=i) for i in range(num_days)]

    duration = np.linspace(-0.001, 0.001, num_days)
    credit = np.linspace(0.001, -0.001, num_days)
    liquidity = np.sin(np.linspace(0.0, 4.0 * np.pi, num_days)) * 0.0005

    factor_returns = pl.DataFrame({
        "timestamp": timestamps,
        "factor_duration": duration,
        "factor_credit": credit,
        "factor_liquidity": liquidity,
    })

    sector_rows: list[dict[str, object]] = []
    for idx, timestamp in enumerate(timestamps):
        factor_duration = float(duration[idx])
        factor_credit = float(credit[idx])
        factor_liquidity = float(liquidity[idx])

        combinations = {
            "SPY": (0.5, 0.3, 0.2, 0.0010),
            "XLK": (0.2, 0.4, -0.1, 0.0008),
            "AGG": (0.1, -0.2, 0.5, 0.0004),
        }

        for symbol, (beta_dur, beta_cred, beta_liq, intercept) in combinations.items():
            sector_rows.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "return": (
                    beta_dur * factor_duration
                    + beta_cred * factor_credit
                    + beta_liq * factor_liquidity
                    + intercept
                ),
            })

    sector_returns = pl.DataFrame(sector_rows)

    coverage = CoverageSummary(
        calendar_name="XNYS",
        sector_expected_days=num_days,
        factor_expected_days=num_days,
        sector_coverage={"SPY": 1.0, "XLK": 1.0, "AGG": 1.0},
        factor_coverage={
            "factor_duration": 1.0,
            "factor_credit": 1.0,
            "factor_liquidity": 1.0,
        },
    )

    return SectorDataset(
        sector_returns=sector_returns,
        factor_returns=factor_returns,
        coverage=coverage,
    )


def test_compute_weights_respects_constraints() -> None:
    """Weights should sum to 1.0 and respect bounds under manual forecasts."""
    dataset = _build_linear_dataset(num_days=240)
    base_forecasts = {
        "factor_duration": 0.002,
        "factor_credit": -0.001,
        "factor_liquidity": 0.001,
    }
    strategy = FactorTiltStrategy(
        factor_forecasts=base_forecasts,
        min_weight=0.0,
        max_weight=0.50,
        min_observations=120,
        blend_to_equal=0.0,
    )
    latest_date = dataset.factor_returns["timestamp"][-1]

    weights = strategy.compute_weights(latest_date, dataset)

    assert weights
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    for weight in weights.values():
        assert 0.0 <= weight <= 0.50 + 1e-9


def test_blend_to_equal_softens_concentration() -> None:
    """Blending to equal weights should reduce extreme allocations."""
    dataset = _build_linear_dataset(num_days=240)
    base_forecasts = {
        "factor_duration": 0.003,
        "factor_credit": -0.002,
        "factor_liquidity": 0.001,
    }

    aggressive = FactorTiltStrategy(
        factor_forecasts=base_forecasts,
        min_weight=0.0,
        max_weight=0.60,
        blend_to_equal=0.0,
        min_observations=120,
    )
    blended = FactorTiltStrategy(
        factor_forecasts=base_forecasts,
        min_weight=0.0,
        max_weight=0.60,
        blend_to_equal=0.5,
        min_observations=120,
        turnover_smoothing=0.0,
    )
    date = dataset.factor_returns["timestamp"][-1]

    aggressive_weights = aggressive.compute_weights(date, dataset)
    blended_weights = blended.compute_weights(date, dataset)

    assert aggressive_weights
    assert blended_weights
    assert pytest.approx(sum(blended_weights.values()), rel=1e-6) == 1.0
    max_aggressive = max(aggressive_weights.values())
    max_blended = max(blended_weights.values())
    assert max_blended <= max_aggressive


def test_turnover_smoothing_limits_weight_changes() -> None:
    """Turnover smoothing should reduce allocation changes between periods."""
    dataset = _build_linear_dataset(num_days=260)
    dates = dataset.factor_returns["timestamp"]
    first_date = dates[200]
    second_date = dates[240]

    base_forecasts = {
        "factor_duration": 0.003,
        "factor_credit": -0.002,
        "factor_liquidity": 0.001,
    }

    no_smoothing = FactorTiltStrategy(
        factor_forecasts=base_forecasts,
        min_weight=0.0,
        max_weight=0.60,
        blend_to_equal=0.0,
        turnover_smoothing=0.0,
    )
    smoothed = FactorTiltStrategy(
        factor_forecasts=base_forecasts,
        min_weight=0.0,
        max_weight=0.60,
        blend_to_equal=0.0,
        turnover_smoothing=0.5,
    )

    weights1_raw = no_smoothing.compute_weights(first_date, dataset)
    weights2_raw = no_smoothing.compute_weights(second_date, dataset)
    delta_raw = sum(abs(weights2_raw.get(k, 0.0) - weights1_raw.get(k, 0.0)) for k in weights2_raw)

    weights1_smooth = smoothed.compute_weights(first_date, dataset)
    weights2_smooth = smoothed.compute_weights(second_date, dataset)
    delta_smooth = sum(abs(weights2_smooth.get(k, 0.0) - weights1_smooth.get(k, 0.0)) for k in weights2_smooth)

    assert delta_smooth <= delta_raw + 1e-9


def test_dynamic_factor_scaling_dampens_negative_factors() -> None:
    """Dynamic factor scaling should reduce positive forecasts when factors underperform."""
    dataset = _build_negative_liquidity_dataset(num_days=260, negative_value=-0.001)
    date = dataset.factor_returns["timestamp"][-1]

    base_strategy = FactorTiltStrategy(
        min_observations=120,
        dynamic_factor_scaling=False,
    )
    scaled_strategy = FactorTiltStrategy(
        min_observations=120,
        dynamic_factor_scaling=True,
        scaling_threshold=0.005,
        scaling_floor=0.3,
        scaling_lookback=126,
    )

    base_forecasts = base_strategy._forecast_factor_returns(date, dataset)
    scaled_forecasts = scaled_strategy._forecast_factor_returns(date, dataset)

    assert "factor_liquidity" in base_forecasts
    assert "factor_liquidity" in scaled_forecasts
    assert abs(scaled_forecasts["factor_liquidity"]) <= abs(base_forecasts["factor_liquidity"]) + 1e-12
