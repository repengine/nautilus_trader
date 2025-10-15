"""
Heuristics for regime-aware liquidity scaling in the Phase 3 backtest suite.

This module encapsulates the rules used to dampen liquidity factor exposure when
regime attribution shows persistent drag. The helpers are intentionally pure and
typed so they can be unit-tested without loading historical CSV artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(slots=True)
class LiquidityScalingConfig:
    """
    Configuration for translating regime-level liquidity attribution into multipliers.

    Attributes
    ----------
    severe_threshold : float
        Annualised liquidity contribution (in decimal form) deemed severely negative.
    moderate_threshold : float
        Annualised contribution regarded as moderately negative.
    severe_regime_multiplier : float
        Broad regime scaling applied when contribution breaches ``severe_threshold``.
    moderate_regime_multiplier : float
        Broad regime scaling applied when contribution is between thresholds.
    severe_liquidity_multiplier : float
        Liquidity factor-specific multiplier under severe drag.
    moderate_liquidity_multiplier : float
        Liquidity factor-specific multiplier under moderate drag.
    neutral_liquidity_multiplier : float
        Liquidity multiplier when attribution is non-negative.
    floor : float
        Lower bound for any multiplier to maintain exposure continuity.
    """

    severe_threshold: float = -0.020
    moderate_threshold: float = -0.010
    severe_regime_multiplier: float = 0.85
    moderate_regime_multiplier: float = 0.92
    severe_liquidity_multiplier: float = 0.55
    moderate_liquidity_multiplier: float = 0.70
    neutral_liquidity_multiplier: float = 1.0
    floor: float = 0.40

    def clamp(self, value: float) -> float:
        """Clamp ``value`` to the inclusive range [floor, 1.0]."""
        return max(self.floor, min(1.0, value))


@dataclass(slots=True)
class LiquidityScalingDecision:
    """
    Regime-aware scaling decision for the factor model.

    Attributes
    ----------
    regime_multiplier : float
        Broad scaling applied to all factor forecasts within the regime.
    factor_multipliers : dict[str, float]
        Per-factor overrides (currently used for liquidity dampening).
    """

    regime_multiplier: float
    factor_multipliers: dict[str, float]


def derive_liquidity_scaling(
    annualized_contribution: float,
    *,
    config: LiquidityScalingConfig,
) -> LiquidityScalingDecision:
    """
    Translate a liquidity attribution contribution into regime and factor multipliers.

    Parameters
    ----------
    annualized_contribution : float
        Annualised liquidity contribution (e.g., -0.020 for -2.0%).
    config : LiquidityScalingConfig
        Thresholds and multipliers governing the decision tree.

    Returns
    -------
    LiquidityScalingDecision
        Multipliers suitable for consumption by ``FactorTiltStrategy``.
    """
    if annualized_contribution <= config.severe_threshold:
        regime_multiplier = config.clamp(config.severe_regime_multiplier)
        liquidity_multiplier = config.clamp(config.severe_liquidity_multiplier)
    elif annualized_contribution <= config.moderate_threshold:
        regime_multiplier = config.clamp(config.moderate_regime_multiplier)
        liquidity_multiplier = config.clamp(config.moderate_liquidity_multiplier)
    else:
        regime_multiplier = 1.0
        liquidity_multiplier = config.clamp(config.neutral_liquidity_multiplier)

    return LiquidityScalingDecision(
        regime_multiplier=regime_multiplier,
        factor_multipliers={"factor_liquidity": liquidity_multiplier},
    )


def build_regime_scaling_maps(
    liquidity_contributions: Mapping[str, float],
    *,
    config: LiquidityScalingConfig | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Generate regime-level and per-factor multipliers from attribution inputs.

    Parameters
    ----------
    liquidity_contributions : Mapping[str, float]
        Mapping of regime name to annualised liquidity contribution.
    config : LiquidityScalingConfig, optional
        Custom configuration; defaults to ``LiquidityScalingConfig()``.

    Returns
    -------
    tuple[dict[str, float], dict[str, dict[str, float]]]
        Regime multipliers and per-factor overrides keyed by regime.
    """
    resolved_config = config or LiquidityScalingConfig()
    regime_scaling: dict[str, float] = {}
    factor_scaling: dict[str, dict[str, float]] = {}

    for regime_name, contribution in liquidity_contributions.items():
        decision = derive_liquidity_scaling(contribution, config=resolved_config)
        regime_scaling[regime_name] = decision.regime_multiplier
        factor_scaling[regime_name] = decision.factor_multipliers

    return regime_scaling, factor_scaling


def load_liquidity_contributions_from_csv(
    directory: Path,
    strategy_slug: str,
) -> dict[str, float]:
    r"""
    Load liquidity contributions from a regime attribution CSV.

    Parameters
    ----------
    directory : Path
        Directory containing ``*_regime_attribution.csv`` exports.
    strategy_slug : str
        Slug identifying the strategy (e.g., ``\"3d_factor_rolling_betas\"``).

    Returns
    -------
    dict[str, float]
        Mapping of regime name to annualised liquidity contribution. Returns an empty
        dict if the file is absent or lacks the required columns.
    """
    file_path = directory / f"{strategy_slug}_regime_attribution.csv"
    if not file_path.exists():
        return {}

    frame = pl.read_csv(
        file_path,
        schema_overrides={"annualized_contribution": pl.Float64},
        ignore_errors=True,
    )

    required_columns = {"regime", "factor", "annualized_contribution"}
    if not required_columns.issubset(set(frame.columns)):
        return {}

    liquidity_rows = frame.filter(
        (pl.col("factor") == "factor_liquidity") & pl.col("annualized_contribution").is_not_null()
    )
    if liquidity_rows.is_empty():
        return {}

    contributions: dict[str, float] = {}
    for row in liquidity_rows.iter_rows(named=True):
        regime_name = str(row["regime"])
        contribution = float(row["annualized_contribution"])
        contributions[regime_name] = contribution

    return contributions
