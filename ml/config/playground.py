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


ParameterValue = float | int | str


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


def _slugify(value: str) -> str:
    candidate = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    if candidate:
        return candidate
    return "default"


@dataclass(frozen=True)
class ParameterHeatmapSpecDefaults:
    """
    Immutable specification describing a single parameter heatmap run.

    Attributes
    ----------
    name : str
        Human-readable identifier for the heatmap.
    description : str
        Short description outlining the diagnostic goal.
    target_strategy : str
        Display name of the strategy whose metrics populate the heatmap.
    parameters : tuple[str, str]
        Tuple containing the row and column parameter names respectively. Parameter
        names may be prefixed with ``config.``, ``strategy_params.``,
        ``turnover_overrides.``, or ``liquidity_scaling.`` to indicate how the value
        is applied.
    grid : Mapping[str, tuple[ParameterValue, ...]]
        Parameter grid evaluated for the heatmap. Keys align with the prefixes above.
        Additional parameters not present in ``parameters`` are treated as auxiliary
        overrides and must contain exactly one value.
    metric : str
        Attribute name on :class:`playground.backtest.performance_metrics.PerformanceMetrics`
        used to populate the heatmap cells (default: ``"sharpe_ratio"``).
    base_overrides : Mapping[str, ParameterValue]
        Additional overrides applied to every configuration in the sweep. Uses the
        same prefix semantics as ``grid``.
    """

    name: str
    description: str
    target_strategy: str
    parameters: tuple[str, str]
    grid: Mapping[str, tuple[ParameterValue, ...]]
    metric: str = "sharpe_ratio"
    base_overrides: Mapping[str, ParameterValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            msg = "Heatmap name must be non-empty"
            raise ValueError(msg)
        if not self.description.strip():
            msg = "Heatmap description must be non-empty"
            raise ValueError(msg)
        if len(self.parameters) != 2:
            msg = "parameters must contain exactly two entries (row, column)"
            raise ValueError(msg)
        if len(set(self.parameters)) != 2:
            msg = "parameters must reference distinct keys"
            raise ValueError(msg)
        if not self.target_strategy.strip():
            msg = "target_strategy must be non-empty"
            raise ValueError(msg)
        normalized_grid: dict[str, tuple[ParameterValue, ...]] = {}
        for key, values in self.grid.items():
            cleaned_key = key.strip()
            if not cleaned_key:
                msg = "Grid keys must be non-empty"
                raise ValueError(msg)
            if cleaned_key in normalized_grid:
                msg = f"Duplicate grid key detected: {cleaned_key}"
                raise ValueError(msg)
            if not values:
                msg = f"Grid values for '{cleaned_key}' must be non-empty"
                raise ValueError(msg)
            normalized_grid[cleaned_key] = tuple(values)
        normalized_overrides = {
            key.strip(): value
            for key, value in self.base_overrides.items()
            if key.strip()
        }
        for parameter in self.parameters:
            if parameter not in normalized_grid:
                msg = f"Primary parameter '{parameter}' missing from grid specification"
                raise ValueError(msg)
            if len(normalized_grid[parameter]) < 2:
                msg = f"Grid for '{parameter}' must contain at least two values"
                raise ValueError(msg)
        object.__setattr__(self, "grid", MappingProxyType(normalized_grid))
        object.__setattr__(
            self,
            "base_overrides",
            MappingProxyType(normalized_overrides),
        )

    @property
    def slug(self) -> str:
        """Return a filesystem-safe slug derived from ``name``."""
        return _slugify(self.name)


@dataclass(frozen=True)
class ParameterHeatmapSuiteDefaults:
    """
    Collection of parameter heatmap specifications executed during nightly runs.
    """

    specs: tuple[ParameterHeatmapSpecDefaults, ...]
    output_dirname: str = "heatmaps"


@dataclass(frozen=True)
class ParameterSensitivitySpecDefaults:
    """
    Immutable specification describing a parameter sensitivity sweep.

    Attributes
    ----------
    name : str
        Human-readable identifier for the sweep.
    description : str
        Short description outlining the robustness objective.
    target_strategy : str
        Display name of the strategy whose metrics populate the sweep summary.
    parameter_grid : Mapping[str, tuple[ParameterValue, ...]]
        Parameter grid evaluated during the sweep. Keys support the same prefix
        semantics as :class:`ParameterHeatmapSpecDefaults` (``config.``,
        ``turnover_overrides.``, ``strategy_params.``, ``liquidity_scaling.``).
    metric : str, default "sharpe_ratio"
        Attribute name on :class:`playground.backtest.performance_metrics.PerformanceMetrics`
        used to score each configuration.
    base_overrides : Mapping[str, ParameterValue], optional
        Additional overrides applied to every configuration prior to exploring the grid.
    """

    name: str
    description: str
    target_strategy: str
    parameter_grid: Mapping[str, tuple[ParameterValue, ...]]
    metric: str = "sharpe_ratio"
    base_overrides: Mapping[str, ParameterValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            msg = "Sensitivity spec name must be non-empty"
            raise ValueError(msg)
        if not self.description.strip():
            msg = "Sensitivity spec description must be non-empty"
            raise ValueError(msg)
        if not self.target_strategy.strip():
            msg = "Sensitivity spec target_strategy must be non-empty"
            raise ValueError(msg)
        if not self.parameter_grid:
            msg = "parameter_grid must contain at least one parameter"
            raise ValueError(msg)
        normalized_grid: dict[str, tuple[ParameterValue, ...]] = {}
        for key, values in self.parameter_grid.items():
            cleaned_key = key.strip()
            if not cleaned_key:
                msg = "Parameter grid keys must be non-empty"
                raise ValueError(msg)
            if cleaned_key in normalized_grid:
                msg = f"Duplicate parameter key detected: {cleaned_key}"
                raise ValueError(msg)
            if not values:
                msg = f"Parameter grid for '{cleaned_key}' must contain at least one value"
                raise ValueError(msg)
            normalized_grid[cleaned_key] = tuple(values)
        normalized_overrides = {
            key.strip(): value
            for key, value in self.base_overrides.items()
            if key.strip()
        }
        object.__setattr__(self, "parameter_grid", MappingProxyType(normalized_grid))
        object.__setattr__(self, "base_overrides", MappingProxyType(normalized_overrides))

    @property
    def slug(self) -> str:
        """Return a filesystem safe slug for the specification."""
        return _slugify(self.name)


@dataclass(frozen=True)
class ParameterSensitivitySuiteDefaults:
    """
    Collection of parameter sensitivity specifications evaluated during robustness runs.
    """

    specs: tuple[ParameterSensitivitySpecDefaults, ...]
    output_dirname: str = "sensitivity"
    sharpe_delta_tolerance: float = 0.15

    def __post_init__(self) -> None:
        if not self.specs:
            msg = "ParameterHeatmapSuiteDefaults.specs must contain at least one spec"
            raise ValueError(msg)
        slugs = {spec.slug for spec in self.specs}
        if len(slugs) != len(self.specs):
            msg = "Heatmap specification names must be unique"
            raise ValueError(msg)
        if not self.output_dirname.strip():
            msg = "output_dirname must be non-empty"
            raise ValueError(msg)
        if self.sharpe_delta_tolerance <= 0:
            msg = "sharpe_delta_tolerance must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class DiagnosticsDefaults:
    """
    Defaults governing extended diagnostic calculations.

    Attributes
    ----------
    tail_quantiles : tuple[float, ...]
        Lower-tail quantiles (expressed in decimal form) used for VaR/CVaR style metrics.
    turnover_bins : tuple[float, ...]
        Turnover bucket edges (exclusive upper bounds) used when aggregating distributions.
    alternative_benchmarks : tuple[str, ...]
        Strategy names treated as benchmarks when computing deltas for diagnostics.
    turnover_window_days : int
        Window length (in trading days) for rolling turnover diagnostics.
    benchmark_delta_metrics : tuple[str, ...]
        Metric fields compared against alternative benchmarks.
    """

    tail_quantiles: tuple[float, ...] = (0.01, 0.05)
    turnover_bins: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
    alternative_benchmarks: tuple[str, ...] = ("Equal Weight", "60/40 Portfolio", "Risk Parity", "Minimum Variance")
    turnover_window_days: int = 252
    benchmark_delta_metrics: tuple[str, ...] = ("sharpe_ratio", "max_drawdown", "annualized_return", "calmar_ratio")

    def __post_init__(self) -> None:
        if not self.tail_quantiles:
            msg = "tail_quantiles must contain at least one value"
            raise ValueError(msg)
        for quantile in self.tail_quantiles:
            if not 0.0 < quantile < 0.5:
                msg = f"tail quantiles must be in (0, 0.5), received {quantile}"
                raise ValueError(msg)
        if list(self.tail_quantiles) != sorted(self.tail_quantiles):
            msg = "tail_quantiles must be sorted ascending"
            raise ValueError(msg)
        if not self.turnover_bins:
            msg = "turnover_bins must contain at least one threshold"
            raise ValueError(msg)
        last_value = 0.0
        for bin_edge in self.turnover_bins:
            if not 0.0 < bin_edge < 1.0:
                msg = f"turnover_bins must be within (0, 1), received {bin_edge}"
                raise ValueError(msg)
            if bin_edge <= last_value:
                msg = "turnover_bins must be strictly increasing"
                raise ValueError(msg)
            last_value = bin_edge
        if not self.alternative_benchmarks:
            msg = "alternative_benchmarks must contain at least one strategy name"
            raise ValueError(msg)
        if len({name.strip() for name in self.alternative_benchmarks}) != len(self.alternative_benchmarks):
            msg = "alternative_benchmarks must not contain duplicates"
            raise ValueError(msg)
        if self.turnover_window_days <= 0:
            msg = "turnover_window_days must be positive"
            raise ValueError(msg)
        if not self.benchmark_delta_metrics:
            msg = "benchmark_delta_metrics must contain at least one metric name"
            raise ValueError(msg)
        normalized_metrics: list[str] = []
        seen: set[str] = set()
        for metric in self.benchmark_delta_metrics:
            normalized = metric.strip()
            if not normalized:
                msg = "benchmark_delta_metrics must not contain blank entries"
                raise ValueError(msg)
            if normalized in seen:
                msg = f"Duplicate entry detected in benchmark_delta_metrics: {normalized}"
                raise ValueError(msg)
            normalized_metrics.append(normalized)
            seen.add(normalized)
        object.__setattr__(self, "benchmark_delta_metrics", tuple(normalized_metrics))


@dataclass(frozen=True)
class PhaseTwoValidationDefaults:
    """
    Thresholds governing Phase 2 regression diagnostics acceptance.

    Attributes
    ----------
    r2_threshold : float
        Minimum acceptable R² for an individual sector.
    min_sector_pass_rate : float
        Minimum fraction of sectors that must exceed ``r2_threshold``.
    min_significant_beta_pass_rate : float
        Minimum fraction of sectors that must have at least two significant betas.
    min_durbin_watson_pass_rate : float
        Minimum fraction of sectors that must have Durbin-Watson within bounds.
    significance_level : float
        P-value threshold used when determining beta significance (default 0.05).
    vif_threshold : float
        Maximum allowable Variance Inflation Factor for any factor.
    durbin_watson_lower : float
        Lower bound for acceptable Durbin-Watson statistics.
    durbin_watson_upper : float
        Upper bound for acceptable Durbin-Watson statistics.
    """

    r2_threshold: float = 0.30
    min_sector_pass_rate: float = 2 / 3  # ≥ 6/9 sectors
    min_significant_beta_pass_rate: float = 2 / 3
    min_durbin_watson_pass_rate: float = 2 / 3
    significance_level: float = 0.05
    vif_threshold: float = 5.0
    durbin_watson_lower: float = 1.5
    durbin_watson_upper: float = 2.5

    def __post_init__(self) -> None:
        """Validate threshold ranges and relative ordering."""
        if not 0.0 < self.r2_threshold <= 1.0:
            msg = f"r2_threshold must be within (0, 1], received {self.r2_threshold}"
            raise ValueError(msg)
        for attr_name in (
            "min_sector_pass_rate",
            "min_significant_beta_pass_rate",
            "min_durbin_watson_pass_rate",
        ):
            value = getattr(self, attr_name)
            if not 0.0 < value <= 1.0:
                msg = f"{attr_name} must be within (0, 1], received {value}"
                raise ValueError(msg)
        if not 0.0 < self.significance_level < 0.5:
            msg = f"significance_level must be within (0, 0.5), received {self.significance_level}"
            raise ValueError(msg)
        if self.vif_threshold <= 1.0:
            msg = f"vif_threshold must be greater than 1.0, received {self.vif_threshold}"
            raise ValueError(msg)
        if self.durbin_watson_lower <= 0 or self.durbin_watson_upper <= 0:
            msg = "Durbin-Watson bounds must be positive"
            raise ValueError(msg)
        if self.durbin_watson_lower >= self.durbin_watson_upper:
            msg = "durbin_watson_lower must be less than durbin_watson_upper"
            raise ValueError(msg)


@dataclass(frozen=True)
class VintageWindowDefaults:
    """
    Sliding vintage window specification used for sequential walk-forward validation.
    """

    label: str
    train_years: int
    test_years: int
    step_years: int
    min_folds: int = 2

    def __post_init__(self) -> None:
        if not self.label.strip():
            msg = "Vintage window label must be non-empty"
            raise ValueError(msg)
        for attr in ("train_years", "test_years", "step_years", "min_folds"):
            value = getattr(self, attr)
            if value <= 0:
                msg = f"{attr} must be positive, received {value}"
                raise ValueError(msg)

    @property
    def slug(self) -> str:
        """Return slugified label."""
        return _slugify(self.label)

    def to_dict(self) -> dict[str, int | str]:
        """Serialise the vintage window specification."""
        return {
            "label": self.label,
            "train_years": self.train_years,
            "test_years": self.test_years,
            "step_years": self.step_years,
            "min_folds": self.min_folds,
            "slug": self.slug,
        }


@dataclass(frozen=True)
class ProxyDatasetSpecDefaults:
    """
    Proxy dataset configuration describing alternative evaluation universes.
    """

    name: str
    relative_path: str
    description: str
    allow_missing: bool = True
    min_train_years: int = 5
    min_test_years: int = 1
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name.strip():
            msg = "Proxy dataset name must be non-empty"
            raise ValueError(msg)
        if not self.relative_path.strip():
            msg = "Proxy dataset relative_path must be non-empty"
            raise ValueError(msg)
        if not self.description.strip():
            msg = "Proxy dataset description must be non-empty"
            raise ValueError(msg)
        if self.min_train_years <= 0 or self.min_test_years <= 0:
            msg = "Vintage constraints must be positive"
            raise ValueError(msg)
        if self.tags:
            normalized_tags = tuple(sorted({tag.strip() for tag in self.tags if tag.strip()}))
            object.__setattr__(self, "tags", normalized_tags)

    @property
    def slug(self) -> str:
        """Filesystem-safe slug derived from the dataset name."""
        return _slugify(self.name)

    def to_dict(self) -> dict[str, object]:
        """Serialise the proxy specification for reporting."""
        return {
            "name": self.name,
            "relative_path": self.relative_path,
            "description": self.description,
            "allow_missing": self.allow_missing,
            "min_train_years": self.min_train_years,
            "min_test_years": self.min_test_years,
            "slug": self.slug,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class ProxyDatasetSuiteDefaults:
    """
    Aggregated defaults for proxy dataset validation and vintage simulations.
    """

    specs: tuple[ProxyDatasetSpecDefaults, ...]
    vintage_windows: tuple[VintageWindowDefaults, ...]

    def __post_init__(self) -> None:
        slugs = {spec.slug for spec in self.specs}
        if len(slugs) != len(self.specs):
            msg = "Proxy dataset names must be unique"
            raise ValueError(msg)
        vintage_slugs = {window.slug for window in self.vintage_windows}
        if len(vintage_slugs) != len(self.vintage_windows):
            msg = "Vintage window labels must be unique"
            raise ValueError(msg)


@dataclass(frozen=True)
class MonitoringExportDefaults:
    """
    Defaults describing the consolidated monitoring snapshot export.
    """

    filename: str = "phase3_monitoring_snapshot.json"
    include_sections: tuple[str, ...] = (
        "walk_forward",
        "monte_carlo",
        "parameter_heatmaps",
        "extended_diagnostics",
        "proxy_datasets",
        "vintage_simulations",
        "benchmarks",
        "phase4_sensitivity",
        "phase4_data_quality",
        "phase4_outliers",
    )
    alert_channels: tuple[str, ...] = ("grafana", "pagerduty")
    dashboard_targets: Mapping[str, str] = field(
        default_factory=lambda: {
            "grafana": "dashboards/phase3_risk_model",
            "pagerduty": "services/phase3-risk-monitor",
        },
    )
    alert_rules: Mapping[str, str] = field(
        default_factory=lambda: {
            "grafana": "alerts/phase3_risk_ruleset.yml",
            "pagerduty": "alerts/phase3-risk-critical",
        },
    )
    automation_targets: Mapping[str, str] = field(
        default_factory=lambda: {
            "airflow": "dags/phase3_monitoring_refresh.py",
            "github_actions": ".github/workflows/phase3_monitoring.yml",
        },
    )

    def __post_init__(self) -> None:
        if not self.filename.strip():
            msg = "Monitoring export filename must be non-empty"
            raise ValueError(msg)
        if not self.include_sections:
            msg = "include_sections must reference at least one section"
            raise ValueError(msg)
        if len({section.strip() for section in self.include_sections}) != len(self.include_sections):
            msg = "include_sections must not contain duplicates"
            raise ValueError(msg)
        seen_channels: set[str] = set()
        normalized_channels: list[str] = []
        for channel in self.alert_channels:
            normalized = channel.strip()
            if not normalized:
                msg = "alert_channels must not contain blank entries"
                raise ValueError(msg)
            if normalized in seen_channels:
                msg = f"Duplicate alert channel detected: {normalized}"
                raise ValueError(msg)
            seen_channels.add(normalized)
            normalized_channels.append(normalized)
        object.__setattr__(self, "alert_channels", tuple(normalized_channels))
        cleaned_targets: dict[str, str] = {}
        for key, value in self.dashboard_targets.items():
            normalized_key = key.strip()
            normalized_value = value.strip()
            if not normalized_key or not normalized_value:
                msg = "dashboard_targets keys and values must be non-empty strings"
                raise ValueError(msg)
            cleaned_targets[normalized_key] = normalized_value
        missing_channels = {channel for channel in self.alert_channels if channel not in cleaned_targets}
        if missing_channels:
            missing_desc = ", ".join(sorted(missing_channels))
            msg = f"dashboard_targets missing entries for alert_channels: {missing_desc}"
            raise ValueError(msg)
        object.__setattr__(self, "dashboard_targets", MappingProxyType(cleaned_targets))
        cleaned_rules: dict[str, str] = {}
        for key, value in self.alert_rules.items():
            normalized_key = key.strip()
            normalized_value = value.strip()
            if not normalized_key or not normalized_value:
                msg = "alert_rules keys and values must be non-empty strings"
                raise ValueError(msg)
            cleaned_rules[normalized_key] = normalized_value
        missing_rules = {channel for channel in self.alert_channels if channel not in cleaned_rules}
        if missing_rules:
            missing_desc = ", ".join(sorted(missing_rules))
            msg = f"alert_rules missing entries for alert_channels: {missing_desc}"
            raise ValueError(msg)
        object.__setattr__(self, "alert_rules", MappingProxyType(cleaned_rules))
        cleaned_automation: dict[str, str] = {}
        for key, value in self.automation_targets.items():
            normalized_key = key.strip()
            normalized_value = value.strip()
            if not normalized_key or not normalized_value:
                msg = "automation_targets keys and values must be non-empty strings"
                raise ValueError(msg)
            cleaned_automation[normalized_key] = normalized_value
        if not cleaned_automation:
            msg = "automation_targets must contain at least one entry"
            raise ValueError(msg)
        object.__setattr__(self, "automation_targets", MappingProxyType(cleaned_automation))


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
    category : str
        High-level classification for the overlay (e.g., "rates", "growth", "liquidity").
    description : str | None
        Optional human-readable description summarising the overlay narrative.
    tags : tuple[str, ...]
        Additional structured labels used for reporting or telemetry.
    """

    name: str
    probability: float
    magnitude: float
    duration_days: int
    decay: float = 0.50
    max_applications: int = 1
    regime_bias: tuple[str, ...] | None = None
    category: str = "macro"
    description: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

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
        if not self.category.strip():
            msg = "category must be non-empty"
            raise ValueError(msg)
        normalized_description = None
        if self.description is not None:
            stripped = self.description.strip()
            normalized_description = stripped or None
        object.__setattr__(self, "description", normalized_description)
        if self.regime_bias is not None:
            normalized = tuple(sorted({name.strip() for name in self.regime_bias if name.strip()}))
            object.__setattr__(self, "regime_bias", normalized or None)
        if self.tags:
            normalized_tags = tuple(sorted({tag.strip() for tag in self.tags if tag.strip()}))
            object.__setattr__(self, "tags", normalized_tags)

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
            "category": self.category,
            "description": self.description,
            "tags": list(self.tags),
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
    report_quantiles : tuple[float, ...]
        Quantiles recorded in Monte Carlo summary exports.
    report_metrics : tuple[str, ...]
        Additional metric names persisted in summary metadata for downstream reporting.
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
                category="rates",
                description="Rates hike shock aligned to tightening regimes.",
                tags=("macro", "rates"),
            ),
            MonteCarloShockOverlayDefaults(
                name="growth_scare",
                probability=0.30,
                magnitude=-0.010,
                duration_days=10,
                decay=0.75,
                max_applications=2,
                regime_bias=None,
                category="growth",
                description="Growth scare dragging returns lower over two weeks.",
                tags=("macro", "growth"),
            ),
            MonteCarloShockOverlayDefaults(
                name="liquidity_crunch",
                probability=0.15,
                magnitude=-0.020,
                duration_days=3,
                decay=0.55,
                max_applications=1,
                regime_bias=("Liquidity Stress", "Rate Hiking Cycle"),
                category="liquidity",
                description="Acute liquidity withdrawal aligned with stressed regimes.",
                tags=("macro", "liquidity"),
            ),
            MonteCarloShockOverlayDefaults(
                name="volatility_breakout",
                probability=0.22,
                magnitude=-0.012,
                duration_days=6,
                decay=0.70,
                max_applications=2,
                regime_bias=None,
                category="volatility",
                description="Volatility spike causing multi-session drawdowns.",
                tags=("macro", "volatility"),
            ),
            MonteCarloShockOverlayDefaults(
                name="cross_asset_contagion",
                probability=0.18,
                magnitude=-0.013,
                duration_days=7,
                decay=0.68,
                max_applications=1,
                regime_bias=("Rate Normalization", "Rate Hiking Cycle"),
                category="cross_asset",
                description="Cross-asset contagion where equity, credit, and commodities de-risk together.",
                tags=("macro", "cross-asset", "credit"),
            ),
            MonteCarloShockOverlayDefaults(
                name="compound_liquidity_growth",
                probability=0.12,
                magnitude=-0.009,
                duration_days=8,
                decay=0.72,
                max_applications=3,
                regime_bias=("Zero Rates", "Rate Hiking Cycle"),
                category="compound",
                description="Sequential liquidity withdrawal followed by a growth shock to mimic cascading stresses.",
                tags=("macro", "compound", "sequenced"),
            ),
            MonteCarloShockOverlayDefaults(
                name="credit_spread_widening",
                probability=0.16,
                magnitude=-0.011,
                duration_days=8,
                decay=0.65,
                max_applications=2,
                regime_bias=("GFC Aftermath", "Rate Normalization"),
                category="credit",
                description="Credit spread widening led by cyclical deleveraging and rate volatility.",
                tags=("macro", "credit", "spread"),
            ),
            MonteCarloShockOverlayDefaults(
                name="inflation_repricing",
                probability=0.14,
                magnitude=-0.012,
                duration_days=4,
                decay=0.60,
                max_applications=2,
                regime_bias=("Rate Hiking Cycle",),
                category="inflation",
                description="Inflation surprise repricing hitting duration-sensitive equities and bonds.",
                tags=("macro", "inflation"),
            ),
            MonteCarloShockOverlayDefaults(
                name="energy_supply_shock",
                probability=0.10,
                magnitude=-0.014,
                duration_days=5,
                decay=0.58,
                max_applications=1,
                regime_bias=("Rate Normalization", "Rate Hiking Cycle"),
                category="commodities",
                description="Energy supply disruption spilling into equities and credit risk premia.",
                tags=("macro", "energy", "commodities"),
            ),
        ),
    )
    report_quantiles: tuple[float, ...] = (
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
    )
    report_metrics: tuple[str, ...] = ("sharpe_ratio", "max_drawdown", "terminal_value", "overlay_total_impact")

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
        if not self.report_quantiles:
            msg = "report_quantiles must contain at least one entry"
            raise ValueError(msg)
        if tuple(sorted(self.report_quantiles)) != self.report_quantiles:
            msg = "report_quantiles must be sorted ascending"
            raise ValueError(msg)
        for quantile in self.report_quantiles:
            if not 0.0 < quantile < 1.0:
                msg = f"report_quantiles must be within (0, 1), received {quantile}"
                raise ValueError(msg)
        if not self.report_metrics:
            msg = "report_metrics must contain at least one metric name"
            raise ValueError(msg)
        seen: set[str] = set()
        ordered_metrics: list[str] = []
        for metric in self.report_metrics:
            normalized = metric.strip()
            if not normalized:
                msg = "report_metrics must not contain blank entries"
                raise ValueError(msg)
            if normalized in seen:
                msg = f"Duplicate metric detected in report_metrics: {normalized}"
                raise ValueError(msg)
            seen.add(normalized)
            ordered_metrics.append(normalized)
        object.__setattr__(self, "report_metrics", tuple(ordered_metrics))

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
            "report_quantiles": list(self.report_quantiles),
            "report_metrics": list(self.report_metrics),
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
    parameter_heatmaps : ParameterHeatmapSuiteDefaults
        Phase 3 parameter interaction sweeps rendered as heatmaps.
    parameter_sensitivity : ParameterSensitivitySuiteDefaults
        Phase 4 sensitivity specifications covering single and multi-parameter robustness tests.
    """

    risk_free_rate: float = 0.02
    stable_turnover_smoothing: float = 0.30
    rolling_turnover_smoothing: float = 0.40
    min_training_days: int = 1_250  # ~5 years allowing calendar rounding
    min_testing_days: int = 250  # ~1 year allowing calendar rounding
    coverage_tolerance_days: int = 7
    baseline_strategies: tuple[str, ...] = (
        "Equal Weight",
        "60/40 Portfolio",
        "Risk Parity",
        "Minimum Variance",
        "3D Factor (Rolling Betas)",
    )
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
    parameter_heatmaps: ParameterHeatmapSuiteDefaults = field(
        default_factory=lambda: ParameterHeatmapSuiteDefaults(
            specs=(
                ParameterHeatmapSpecDefaults(
                    name="Turnover vs Transaction Costs",
                    description="Assess rolling beta stability across turnover smoothing and execution costs.",
                    target_strategy="3D Factor (Rolling Betas)",
                    parameters=("turnover_overrides.3d_factor_rolling", "config.transaction_cost_bps"),
                    grid={
                        "turnover_overrides.3d_factor_rolling": (0.30, 0.35, 0.40, 0.45),
                        "config.transaction_cost_bps": (5.0, 10.0, 15.0, 20.0),
                    },
                    base_overrides={"config.random_seed": 42},
                ),
                ParameterHeatmapSpecDefaults(
                    name="Liquidity Scaling Stress",
                    description="Probe liquidity multiplier responsiveness to attribution thresholds.",
                    target_strategy="3D Factor (Rolling Betas)",
                    parameters=("liquidity_scaling.severe_threshold", "liquidity_scaling.moderate_threshold"),
                    grid={
                        "liquidity_scaling.severe_threshold": (-0.030, -0.025, -0.020, -0.015),
                        "liquidity_scaling.moderate_threshold": (-0.020, -0.015, -0.010, -0.005),
                    },
                    base_overrides={
                        "liquidity_scaling.severe_liquidity_multiplier": 0.50,
                        "liquidity_scaling.moderate_liquidity_multiplier": 0.70,
                    },
                ),
                ParameterHeatmapSpecDefaults(
                    name="Beta Window vs Liquidity Floor",
                    description="Contrast rolling beta window length against liquidity floor assumptions.",
                    target_strategy="3D Factor (Rolling Betas)",
                    parameters=("strategy_params.3d_factor_rolling.beta_window_days", "liquidity_scaling.floor"),
                    grid={
                        "strategy_params.3d_factor_rolling.beta_window_days": (126, 189, 252, 378),
                        "liquidity_scaling.floor": (0.35, 0.40, 0.45, 0.50),
                        "config.transaction_cost_bps": (10.0,),
                    },
                    base_overrides={
                        "config.random_seed": 99,
                        "turnover_overrides.3d_factor_rolling": 0.38,
                    },
                ),
                ParameterHeatmapSpecDefaults(
                    name="Turnover vs Liquidity Multipliers",
                    description="Balance turnover smoothing against liquidity multipliers to gauge mitigation robustness.",
                    target_strategy="3D Factor (Rolling Betas)",
                    parameters=("turnover_overrides.3d_factor_rolling", "liquidity_scaling.neutral_liquidity_multiplier"),
                    grid={
                        "turnover_overrides.3d_factor_rolling": (0.30, 0.35, 0.40, 0.45),
                        "liquidity_scaling.neutral_liquidity_multiplier": (0.95, 1.0, 1.05),
                    },
                    metric="calmar_ratio",
                    base_overrides={
                        "liquidity_scaling.moderate_liquidity_multiplier": 0.70,
                        "liquidity_scaling.severe_liquidity_multiplier": 0.55,
                        "liquidity_scaling.floor": 0.40,
                        "config.transaction_cost_bps": 10.0,
                        "config.random_seed": 57,
                    },
                ),
                ParameterHeatmapSpecDefaults(
                    name="Transaction Cost Envelope",
                    description="Evaluate combined transaction cost and slippage assumptions for execution robustness.",
                    target_strategy="3D Factor (Rolling Betas)",
                    parameters=("config.transaction_cost_bps", "config.slippage_bps"),
                    grid={
                        "config.transaction_cost_bps": (5.0, 10.0, 15.0, 20.0),
                        "config.slippage_bps": (0.0, 5.0, 10.0),
                        "turnover_overrides.3d_factor_rolling": (0.38,),
                    },
                    metric="sortino_ratio",
                    base_overrides={
                        "config.rebalance_threshold": 0.05,
                        "config.random_seed": 123,
                    },
                ),
            ),
        ),
    )
    parameter_sensitivity: ParameterSensitivitySuiteDefaults = field(
        default_factory=lambda: ParameterSensitivitySuiteDefaults(
            specs=(
                ParameterSensitivitySpecDefaults(
                    name="Rolling Window Sensitivity",
                    description="Evaluate rolling beta estimation window robustness for the 3D factor strategy.",
                    target_strategy="3D Factor (Rolling Betas)",
                    parameter_grid={
                        "strategy_params.3d_factor_rolling.rolling_window": (126, 252, 504),
                    },
                    base_overrides={"config.random_seed": 42},
                ),
                ParameterSensitivitySpecDefaults(
                    name="Rebalance Cadence Sweep",
                    description="Assess performance across slower rebalance cadences to test execution resilience.",
                    target_strategy="3D Factor (Rolling Betas)",
                    parameter_grid={
                        "config.rebalance_frequency": ("monthly", "quarterly", "semi_annual"),
                    },
                    base_overrides={"config.transaction_cost_bps": 10.0},
                ),
                ParameterSensitivitySpecDefaults(
                    name="Transaction Cost Sensitivity",
                    description="Stress transaction cost assumptions to quantify performance degradation.",
                    target_strategy="3D Factor (Rolling Betas)",
                    parameter_grid={
                        "config.transaction_cost_bps": (5.0, 10.0, 20.0),
                    },
                    base_overrides={"config.rebalance_frequency": "monthly"},
                ),
                ParameterSensitivitySpecDefaults(
                    name="Weight Constraint Sweep",
                    description="Test sector weight bounds to ensure diversification remains stable.",
                    target_strategy="3D Factor (Rolling Betas)",
                    parameter_grid={
                        "strategy_params.3d_factor_rolling.max_weight": (0.25, 0.30, 0.35),
                        "strategy_params.3d_factor_rolling.min_weight": (0.00, 0.02, 0.05),
                    },
                    base_overrides={"config.rebalance_frequency": "monthly"},
                ),
                ParameterSensitivitySpecDefaults(
                    name="Turnover Smoothing vs Blend",
                    description="Explore turnover control and blend-to-equal trade-offs for live readiness.",
                    target_strategy="3D Factor (Rolling Betas)",
                    parameter_grid={
                        "strategy_params.3d_factor_rolling.turnover_smoothing": (0.20, 0.30, 0.40, 0.50),
                        "strategy_params.3d_factor_rolling.blend_to_equal": (0.10, 0.20, 0.30),
                    },
                    base_overrides={
                        "config.rebalance_frequency": "monthly",
                        "config.transaction_cost_bps": 10.0,
                    },
                ),
            ),
        ),
    )
    diagnostics: DiagnosticsDefaults = field(default_factory=DiagnosticsDefaults)
    phase_two_validation: PhaseTwoValidationDefaults = field(default_factory=PhaseTwoValidationDefaults)
    proxy_datasets: ProxyDatasetSuiteDefaults = field(
        default_factory=lambda: ProxyDatasetSuiteDefaults(
            specs=(
                ProxyDatasetSpecDefaults(
                    name="International Sectors",
                    relative_path="playground/data/sector_dataset_intl",
                    description="International sector ETF universe for robustness checks.",
                    allow_missing=True,
                    min_train_years=4,
                    min_test_years=1,
                    tags=("international", "etf"),
                ),
                ProxyDatasetSpecDefaults(
                    name="Factor ETF Proxy",
                    relative_path="playground/data/factor_etf_dataset",
                    description="Factor ETF proxy dataset for liquidity-constrained simulation.",
                    allow_missing=True,
                    min_train_years=4,
                    min_test_years=1,
                    tags=("factor", "etf"),
                ),
                ProxyDatasetSpecDefaults(
                    name="Global Macro Overlay Proxy",
                    relative_path="playground/data/global_macro_overlay",
                    description="Global macro overlay dataset incorporating rates and commodities proxies.",
                    allow_missing=True,
                    min_train_years=5,
                    min_test_years=1,
                    tags=("macro", "rates", "commodities"),
                ),
                ProxyDatasetSpecDefaults(
                    name="Treasury Futures Hedge Proxy",
                    relative_path="playground/data/treasury_futures_dataset",
                    description="UST futures ladder capturing duration hedges for cross-asset validation.",
                    allow_missing=True,
                    min_train_years=5,
                    min_test_years=1,
                    tags=("rates", "futures", "hedge"),
                ),
            ),
            vintage_windows=(
                VintageWindowDefaults(
                    label="Five-Year Rolling",
                    train_years=5,
                    test_years=1,
                    step_years=1,
                    min_folds=3,
                ),
                VintageWindowDefaults(
                    label="Seven-Year Rolling",
                    train_years=7,
                    test_years=1,
                    step_years=1,
                    min_folds=2,
                ),
                VintageWindowDefaults(
                    label="Three-Year Rolling High Frequency",
                    train_years=3,
                    test_years=1,
                    step_years=1,
                    min_folds=3,
                ),
                VintageWindowDefaults(
                    label="Crisis Response 2y/1y",
                    train_years=2,
                    test_years=1,
                    step_years=1,
                    min_folds=4,
                ),
            ),
        ),
    )
    monitoring: MonitoringExportDefaults = field(default_factory=MonitoringExportDefaults)

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
    "DiagnosticsDefaults",
    "LiquidityScalingDefaults",
    "MonitoringExportDefaults",
    "MonteCarloShockOverlayDefaults",
    "MonteCarloStressDefaults",
    "NestedWalkForwardDefaults",
    "ParameterHeatmapSpecDefaults",
    "ParameterHeatmapSuiteDefaults",
    "ParameterSensitivitySpecDefaults",
    "ParameterSensitivitySuiteDefaults",
    "PhaseTwoValidationDefaults",
    "ProxyDatasetSpecDefaults",
    "ProxyDatasetSuiteDefaults",
    "ThreeDRiskBacktestDefaults",
    "VintageWindowDefaults",
    "WalkForwardPermutationDefaults",
]
