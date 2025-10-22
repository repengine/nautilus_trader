"""
Configuration defaults for the playground 3D risk model backtesting suite.

The playground 3D risk model relies on a small number of tunable parameters that
govern turnover controls, liquidity scaling, and validation thresholds. To keep
these tunables discoverable and type-safe, we expose them via frozen dataclasses
under ``ml.config`` so application code can import a single source of truth.

Usage
-----
>>> defaults = ThreeDRiskBacktestDefaults()
>>> defaults.stable_turnover_smoothing
0.3
>>> liquidity_config = defaults.build_liquidity_config()
>>> round(liquidity_config.moderate_threshold, 3)
-0.01
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from playground.backtest.liquidity_controls import LiquidityScalingConfig
    from playground.backtest.splits import WalkForwardConfig


@dataclass(frozen=True)
class LiquidityScalingDefaults:
    """
    Immutable defaults for regime-aware liquidity scaling thresholds.

    Attributes
    ----------
    severe_threshold : float
        Annualised liquidity contribution deemed severely negative (decimal form).
    moderate_threshold : float
        Annualised contribution regarded as moderately negative.
    severe_regime_multiplier : float
        Regime-level multiplier applied when contribution breaches ``severe_threshold``.
    moderate_regime_multiplier : float
        Regime-level multiplier applied when contribution is between thresholds.
    severe_liquidity_multiplier : float
        Liquidity factor multiplier under severe drag.
    moderate_liquidity_multiplier : float
        Liquidity factor multiplier under moderate drag.
    neutral_liquidity_multiplier : float
        Liquidity factor multiplier when attribution is non-negative.
    floor : float
        Inclusive lower bound applied to any multiplier.
    """

    severe_threshold: float = -0.020
    moderate_threshold: float = -0.010
    severe_regime_multiplier: float = 0.85
    moderate_regime_multiplier: float = 0.92
    severe_liquidity_multiplier: float = 0.55
    moderate_liquidity_multiplier: float = 0.70
    neutral_liquidity_multiplier: float = 1.0
    floor: float = 0.40

    def __post_init__(self) -> None:
        """Validate logical ordering and ranges for multiplier configuration."""
        if self.severe_threshold > self.moderate_threshold:
            msg = "severe_threshold must be <= moderate_threshold"
            raise ValueError(msg)
        for attr_name in (
            "severe_regime_multiplier",
            "moderate_regime_multiplier",
            "severe_liquidity_multiplier",
            "moderate_liquidity_multiplier",
            "neutral_liquidity_multiplier",
            "floor",
        ):
            value = getattr(self, attr_name)
            if not 0.0 < value <= 1.0:
                msg = f"{attr_name} must be in (0, 1], received {value}"
                raise ValueError(msg)
        if self.neutral_liquidity_multiplier < self.floor:
            msg = "neutral_liquidity_multiplier must be >= floor"
            raise ValueError(msg)

    def to_kwargs(self) -> dict[str, float]:
        """
        Return values as keyword arguments for ``LiquidityScalingConfig``.

        Returns
        -------
        dict[str, float]
            Mapping compatible with ``playground.backtest.liquidity_controls.LiquidityScalingConfig``.
        """
        return {
            "severe_threshold": self.severe_threshold,
            "moderate_threshold": self.moderate_threshold,
            "severe_regime_multiplier": self.severe_regime_multiplier,
            "moderate_regime_multiplier": self.moderate_regime_multiplier,
            "severe_liquidity_multiplier": self.severe_liquidity_multiplier,
            "moderate_liquidity_multiplier": self.moderate_liquidity_multiplier,
            "neutral_liquidity_multiplier": self.neutral_liquidity_multiplier,
            "floor": self.floor,
        }


@dataclass(frozen=True)
class NestedWalkForwardDefaults:
    """
    Immutable defaults describing an inner walk-forward validation sweep.

    Attributes
    ----------
    train_years : int
        Number of years in each nested training window.
    test_years : int
        Number of years in each nested testing window.
    step_years : int
        Number of years between successive nested folds.
    min_folds : int
        Minimum number of folds required for the nested sweep to be considered
        valid. Nested evaluation is skipped when fewer folds are produced.
    """

    train_years: int
    test_years: int
    step_years: int
    min_folds: int = 2

    def __post_init__(self) -> None:
        """Validate positive horizons and reasonable fold requirements."""
        for attr_name in ("train_years", "test_years", "step_years"):
            value = getattr(self, attr_name)
            if value <= 0:
                msg = f"{attr_name} must be positive, received {value}"
                raise ValueError(msg)
        if self.min_folds <= 0:
            msg = "min_folds must be positive"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, int]:
        """
        Represent the nested configuration as serialisable dictionary.

        Returns
        -------
        dict[str, int]
            Mapping of configuration fields.
        """
        return {
            "train_years": self.train_years,
            "test_years": self.test_years,
            "step_years": self.step_years,
            "min_folds": self.min_folds,
        }


@dataclass(frozen=True)
class WalkForwardPermutationDefaults:
    """
    Immutable specification for a walk-forward validation permutation.

    Attributes
    ----------
    name : str
        Human-readable name describing the permutation.
    description : str
        Optional extended description of the validation goal.
    train_years : int
        Length of the training window expressed in calendar years.
    test_years : int
        Length of the testing window expressed in calendar years.
    step_years : int
        Stride applied between successive folds.
    nested : NestedWalkForwardDefaults | None
        Optional nested walk-forward sweep executed within each outer training
        window for additional validation.
    """

    name: str
    description: str
    train_years: int
    test_years: int
    step_years: int
    nested: NestedWalkForwardDefaults | None = None

    def __post_init__(self) -> None:
        """Validate permutation metadata and horizon parameters."""
        if not self.name.strip():
            msg = "Permutation name must be non-empty"
            raise ValueError(msg)
        if not self.description.strip():
            msg = "Permutation description must be non-empty"
            raise ValueError(msg)
        for attr_name in ("train_years", "test_years", "step_years"):
            value = getattr(self, attr_name)
            if value <= 0:
                msg = f"{attr_name} must be positive, received {value}"
                raise ValueError(msg)
        if self.nested is not None and self.nested.train_years >= self.train_years:
            msg = "Nested training window must be shorter than outer training window"
            raise ValueError(msg)

    @property
    def slug(self) -> str:
        """
        Canonical slug for filesystem-safe directory naming.

        Returns
        -------
        str
            Lower-case slug derived from the permutation name with fallback
            to a deterministic horizon signature.
        """
        candidate = re.sub(r"[^a-zA-Z0-9]+", "-", self.name.strip().lower()).strip("-")
        if candidate:
            return candidate
        return f"wf-{self.train_years}y-{self.test_years}y-step-{self.step_years}y"

    def to_config(self, start_date: datetime, end_date: datetime) -> WalkForwardConfig:
        """
        Build a :class:`WalkForwardConfig` from the stored horizons.

        Parameters
        ----------
        start_date : datetime
            Earliest available observation.
        end_date : datetime
            Latest observation to include in the outer test windows.

        Returns
        -------
        WalkForwardConfig
            Instantiated configuration ready for walk-forward generation.
        """
        from playground.backtest.splits import WalkForwardConfig

        return WalkForwardConfig(
            start_date=start_date,
            end_date=end_date,
            train_years=self.train_years,
            test_years=self.test_years,
            step_years=self.step_years,
        )

    def to_dict(self) -> dict[str, object]:
        """
        Serialise permutation details for metadata exports.

        Returns
        -------
        dict[str, object]
            Mapping describing permutation fields and nested configuration.
        """
        payload: dict[str, object] = {
            "name": self.name,
            "description": self.description,
            "train_years": self.train_years,
            "test_years": self.test_years,
            "step_years": self.step_years,
            "slug": self.slug,
        }
        if self.nested is not None:
            payload["nested"] = self.nested.to_dict()
        return payload


@dataclass(frozen=True)
class MonteCarloShockOverlayDefaults:
    """
    Immutable specification for a macro shock overlay applied during stress tests.

    Attributes
    ----------
    name : str
        Human-readable overlay identifier used in reports.
    probability : float
        Probability that the overlay triggers within a simulation path.
    magnitude : float
        Additive daily return shock applied on the first day the overlay triggers.
    duration_days : int
        Number of consecutive trading days the overlay remains active.
    decay : float
        Multiplicative decay factor (0-1) applied to the magnitude for each
        subsequent day while the overlay is active.
    max_applications : int
        Maximum number of times the overlay may trigger within a single simulation.
    regime_bias : tuple[str, ...] | None
        Optional list of regime names where the overlay should preferentially
        activate. When provided, the overlay only triggers if at least one of the
        named regimes exists in the synthetic path.
    """

    name: str
    probability: float
    magnitude: float
    duration_days: int
    decay: float = 0.50
    max_applications: int = 1
    regime_bias: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate overlay configuration."""
        if not self.name.strip():
            msg = "Overlay name must be non-empty"
            raise ValueError(msg)
        if not 0.0 <= self.probability <= 1.0:
            msg = f"probability must be within [0, 1], received {self.probability}"
            raise ValueError(msg)
        if self.duration_days <= 0:
            msg = "duration_days must be positive"
            raise ValueError(msg)
        if not 0.0 <= self.decay <= 1.0:
            msg = "decay must be within [0, 1]"
            raise ValueError(msg)
        if self.max_applications <= 0:
            msg = "max_applications must be positive"
            raise ValueError(msg)
        if self.regime_bias is not None:
            normalized = tuple(sorted({name.strip() for name in self.regime_bias if name.strip()}))
            object.__setattr__(self, "regime_bias", normalized or None)

    def to_dict(self) -> dict[str, object]:
        """
        Serialise overlay configuration for reporting.

        Returns
        -------
        dict[str, object]
            Dictionary representation of the overlay.
        """
        return {
            "name": self.name,
            "probability": self.probability,
            "magnitude": self.magnitude,
            "duration_days": self.duration_days,
            "decay": self.decay,
            "max_applications": self.max_applications,
            "regime_bias": list(self.regime_bias) if self.regime_bias is not None else None,
        }


@dataclass(frozen=True)
class MonteCarloStressDefaults:
    """
    Defaults controlling Monte Carlo regime reshuffling stress tests.

    Attributes
    ----------
    num_paths : int
        Number of Monte Carlo simulation paths to generate.
    random_seed : int
        Seed for deterministic random number generation.
    risk_free_rate : float
        Annualised risk-free rate used when computing Sharpe ratios.
    cvar_alpha : float
        Confidence level for Conditional Value-at-Risk calculations.
    sample_with_replacement : bool
        Whether regime blocks may repeat within a path (bootstrap style) when
        randomising regime order.
    target_strategy : str
        Display name of the strategy whose return stream is stressed.
    overlays : tuple[MonteCarloShockOverlayDefaults, ...]
        Collection of macro overlays evaluated for each simulation path.
    """

    num_paths: int = 64
    random_seed: int = 7_431
    risk_free_rate: float = 0.02
    cvar_alpha: float = 0.95
    sample_with_replacement: bool = True
    target_strategy: str = "3D Factor (Rolling Betas)"
    overlays: tuple[MonteCarloShockOverlayDefaults, ...] = field(
        default_factory=lambda: (
            MonteCarloShockOverlayDefaults(
                name="rate_hike_shock",
                probability=0.20,
                magnitude=-0.015,
                duration_days=5,
                decay=0.60,
                max_applications=1,
                regime_bias=("Rate Hiking Cycle",),
            ),
            MonteCarloShockOverlayDefaults(
                name="growth_scare",
                probability=0.30,
                magnitude=-0.010,
                duration_days=10,
                decay=0.75,
                max_applications=2,
                regime_bias=None,
            ),
        ),
    )

    def __post_init__(self) -> None:
        """Validate Monte Carlo stress configuration."""
        if self.num_paths <= 0:
            msg = "num_paths must be positive"
            raise ValueError(msg)
        if self.random_seed < 0:
            msg = "random_seed must be non-negative"
            raise ValueError(msg)
        if not 0.0 <= self.risk_free_rate <= 0.25:
            msg = "risk_free_rate must be within [0, 0.25]"
            raise ValueError(msg)
        if not 0.0 < self.cvar_alpha < 1.0:
            msg = "cvar_alpha must be within (0, 1)"
            raise ValueError(msg)
        if not self.target_strategy.strip():
            msg = "target_strategy must be non-empty"
            raise ValueError(msg)
        if len({overlay.name for overlay in self.overlays}) != len(self.overlays):
            msg = "Overlay names must be unique"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """
        Serialise Monte Carlo stress configuration for reporting.

        Returns
        -------
        dict[str, object]
            Dictionary describing Monte Carlo defaults.
        """
        return {
            "num_paths": self.num_paths,
            "random_seed": self.random_seed,
            "risk_free_rate": self.risk_free_rate,
            "cvar_alpha": self.cvar_alpha,
            "sample_with_replacement": self.sample_with_replacement,
            "target_strategy": self.target_strategy,
            "overlays": [overlay.to_dict() for overlay in self.overlays],
        }


@dataclass(frozen=True)
class ThreeDRiskBacktestDefaults:
    """
    Centralised defaults for the 3D risk model backtesting harness.

    Attributes
    ----------
    risk_free_rate : float
        Annual risk-free rate used when computing risk-adjusted metrics.
    stable_turnover_smoothing : float
        Turnover smoothing applied to the stable-beta factor strategy.
    rolling_turnover_smoothing : float
        Turnover smoothing applied to the rolling-beta factor strategy.
    min_training_days : int
        Minimum required trading days in the training window (approximate).
    min_testing_days : int
        Minimum required trading days in the testing window (approximate).
    coverage_tolerance_days : int
        Number of calendar days tolerated when dataset coverage does not align
        perfectly with split boundaries (e.g., weekly datasets).
    baseline_strategies : tuple[str, ...]
        Strategies treated as canonical benchmarks in comparison tables.
    liquidity_contribution_fallbacks : Mapping[str, float]
        Mapping of regime names to fallback liquidity contributions when CSV
        attribution exports are unavailable.
    liquidity_scaling : LiquidityScalingDefaults
        Liquidity scaling defaults for deriving mitigation multipliers.
    walk_forward_permutations : tuple[WalkForwardPermutationDefaults, ...]
        Collection of walk-forward validation permutations executed during Phase 3
        sweeps, including the canonical 5y/1y baseline and supplementary horizons.
    monte_carlo_stress : MonteCarloStressDefaults
        Configuration defaults for Monte Carlo stress testing sweeps, including
        regime reshuffling behaviour and macro shock overlays.
    """

    risk_free_rate: float = 0.02
    stable_turnover_smoothing: float = 0.30
    rolling_turnover_smoothing: float = 0.40
    min_training_days: int = 1_250  # ~5 years allowing calendar rounding
    min_testing_days: int = 250  # ~1 year allowing calendar rounding
    coverage_tolerance_days: int = 7
    baseline_strategies: tuple[str, ...] = ("Equal Weight", "60/40 Portfolio", "Risk Parity")
    liquidity_contribution_fallbacks: Mapping[str, float] = field(
        default_factory=lambda: {"Rate Hiking Cycle": -0.0204},
    )
    liquidity_scaling: LiquidityScalingDefaults = field(default_factory=LiquidityScalingDefaults)
    walk_forward_permutations: tuple[WalkForwardPermutationDefaults, ...] = field(
        default_factory=lambda: (
            WalkForwardPermutationDefaults(
                name="Baseline 5y/1y (stride 1y)",
                description="Canonical Phase 3 walk-forward harness using a 5-year train / 1-year test window with annual stride.",
                train_years=5,
                test_years=1,
                step_years=1,
                nested=NestedWalkForwardDefaults(
                    train_years=3,
                    test_years=1,
                    step_years=1,
                    min_folds=2,
                ),
            ),
            WalkForwardPermutationDefaults(
                name="Extended Horizon 7y/2y",
                description="Long-horizon validation with broader train/test windows to probe parameter persistence.",
                train_years=7,
                test_years=2,
                step_years=1,
                nested=NestedWalkForwardDefaults(
                    train_years=4,
                    test_years=1,
                    step_years=1,
                    min_folds=2,
                ),
            ),
            WalkForwardPermutationDefaults(
                name="Stride 2y Adaptation 4y/1y",
                description="Shorter estimation window with biennial stride to test resilience to reduced overlap.",
                train_years=4,
                test_years=1,
                step_years=2,
                nested=NestedWalkForwardDefaults(
                    train_years=2,
                    test_years=1,
                    step_years=1,
                    min_folds=2,
                ),
            ),
        ),
    )
    monte_carlo_stress: MonteCarloStressDefaults = field(default_factory=MonteCarloStressDefaults)

    def __post_init__(self) -> None:
        """Validate ranges, enforce immutability, and normalise mapping inputs."""
        if not 0.0 <= self.risk_free_rate <= 0.25:
            msg = f"risk_free_rate must be within [0, 0.25], received {self.risk_free_rate}"
            raise ValueError(msg)
        for attr_name in ("stable_turnover_smoothing", "rolling_turnover_smoothing"):
            value = getattr(self, attr_name)
            if not 0.0 <= value < 1.0:
                msg = f"{attr_name} must be in [0, 1), received {value}"
                raise ValueError(msg)
        if self.min_training_days <= 0 or self.min_testing_days <= 0:
            msg = "Training and testing day thresholds must be positive"
            raise ValueError(msg)
        if self.coverage_tolerance_days < 0:
            msg = "coverage_tolerance_days must be non-negative"
            raise ValueError(msg)
        if not self.baseline_strategies:
            msg = "baseline_strategies must contain at least one strategy name"
            raise ValueError(msg)
        if len(set(self.baseline_strategies)) != len(self.baseline_strategies):
            msg = "baseline_strategies must not contain duplicates"
            raise ValueError(msg)
        if not self.walk_forward_permutations:
            msg = "walk_forward_permutations must contain at least one permutation"
            raise ValueError(msg)

        normalized_fallbacks = {
            str(name): float(contribution)
            for name, contribution in self.liquidity_contribution_fallbacks.items()
        }
        object.__setattr__(
            self,
            "liquidity_contribution_fallbacks",
            MappingProxyType(normalized_fallbacks),
        )

    def build_liquidity_config(self) -> LiquidityScalingConfig:
        """
        Construct a ``LiquidityScalingConfig`` using the embedded defaults.

        Returns
        -------
        playground.backtest.liquidity_controls.LiquidityScalingConfig
            Configured instance ready for use when deriving regime multipliers.

        Example
        -------
        >>> defaults = ThreeDRiskBacktestDefaults()
        >>> config = defaults.build_liquidity_config()
        >>> round(config.severe_threshold, 3)
        -0.02
        """
        from playground.backtest.liquidity_controls import LiquidityScalingConfig

        return LiquidityScalingConfig(**self.liquidity_scaling.to_kwargs())

    @property
    def primary_walk_forward_permutation(self) -> WalkForwardPermutationDefaults:
        """
        Return the canonical walk-forward permutation used for nightly runs.

        Returns
        -------
        WalkForwardPermutationDefaults
            First permutation defined in ``walk_forward_permutations``.
        """
        return self.walk_forward_permutations[0]


__all__ = [
    "LiquidityScalingDefaults",
    "MonteCarloShockOverlayDefaults",
    "MonteCarloStressDefaults",
    "NestedWalkForwardDefaults",
    "ThreeDRiskBacktestDefaults",
    "WalkForwardPermutationDefaults",
]
