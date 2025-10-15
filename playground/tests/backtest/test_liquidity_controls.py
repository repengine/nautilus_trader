"""
Unit tests for liquidity control heuristics powering regime-aware scaling.
"""

from __future__ import annotations

from pathlib import Path

from playground.backtest.liquidity_controls import LiquidityScalingConfig
from playground.backtest.liquidity_controls import build_regime_scaling_maps
from playground.backtest.liquidity_controls import derive_liquidity_scaling
from playground.backtest.liquidity_controls import load_liquidity_contributions_from_csv


def test_derive_liquidity_scaling_severe_drag() -> None:
    """Severe drag should trigger the harshest multipliers within configured floor."""
    config = LiquidityScalingConfig(
        severe_threshold=-0.02,
        moderate_threshold=-0.01,
        severe_regime_multiplier=0.8,
        moderate_regime_multiplier=0.9,
        severe_liquidity_multiplier=0.5,
        moderate_liquidity_multiplier=0.75,
        neutral_liquidity_multiplier=1.0,
        floor=0.45,
    )
    decision = derive_liquidity_scaling(-0.025, config=config)
    assert decision.regime_multiplier == 0.8
    assert decision.factor_multipliers["factor_liquidity"] == 0.5


def test_derive_liquidity_scaling_clamps_floor() -> None:
    """Multipliers below the floor should be clamped."""
    config = LiquidityScalingConfig(
        severe_liquidity_multiplier=0.3,
        floor=0.4,
    )
    decision = derive_liquidity_scaling(-0.05, config=config)
    assert decision.factor_multipliers["factor_liquidity"] == 0.4
    assert decision.regime_multiplier == 0.85


def test_build_regime_scaling_maps() -> None:
    """Regime and factor maps should align for multiple regimes."""
    config = LiquidityScalingConfig(
        severe_threshold=-0.02,
        moderate_threshold=-0.01,
        severe_regime_multiplier=0.8,
        moderate_regime_multiplier=0.9,
        severe_liquidity_multiplier=0.5,
        moderate_liquidity_multiplier=0.75,
        neutral_liquidity_multiplier=1.0,
        floor=0.5,
    )
    regime_map, factor_map = build_regime_scaling_maps(
        {
            "Rate Hiking Cycle": -0.025,
            "Zero Rates": -0.012,
            "Recent": 0.0,
        },
        config=config,
    )
    assert regime_map == {
        "Rate Hiking Cycle": 0.8,
        "Zero Rates": 0.9,
        "Recent": 1.0,
    }
    assert factor_map["Rate Hiking Cycle"]["factor_liquidity"] == 0.5
    assert factor_map["Zero Rates"]["factor_liquidity"] == 0.75
    assert factor_map["Recent"]["factor_liquidity"] == 1.0


def test_load_liquidity_contributions_from_csv(tmp_path: Path) -> None:
    """Ensure loader parses regime attribution CSVs into mappings."""
    slug = "3d_factor_rolling_betas"
    csv_path = tmp_path / f"{slug}_regime_attribution.csv"
    csv_path.write_text(
        "regime,factor,beta,annualized_contribution,alpha,alpha_annualized\n"
        "Rate Hiking Cycle,factor_liquidity,0.1,-2.0,0.0,0.0\n"
        "Rate Hiking Cycle,alpha,,,\n"
        "Recent,factor_liquidity,0.02,0.5,0.0,0.0\n",
        encoding="utf-8",
    )

    contributions = load_liquidity_contributions_from_csv(tmp_path, slug)

    assert contributions == {
        "Rate Hiking Cycle": -2.0,
        "Recent": 0.5,
    }


def test_load_liquidity_contributions_missing_file(tmp_path: Path) -> None:
    """Missing CSV should return an empty mapping."""
    contributions = load_liquidity_contributions_from_csv(tmp_path, "unknown_strategy")
    assert contributions == {}
