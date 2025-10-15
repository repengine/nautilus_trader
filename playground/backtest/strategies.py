"""
Portfolio allocation strategies for backtesting.

This module implements various portfolio allocation strategies including:
- Factor-based tilting using 3D risk model
- Equal-weight benchmark
- Risk parity (future enhancement)
- Minimum variance (future enhancement)

Performance Notes:
- Cold path only (strategy computation happens during rebalancing)
- Target: < 100ms per weight computation
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable
from typing import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import structlog


if TYPE_CHECKING:
    from playground.risk_model.dataset import SectorDataset


LOGGER = structlog.get_logger(__name__)

HYBRID_SHORT_WINDOW = 21
HYBRID_MEDIUM_WINDOW = 63
HYBRID_LONG_WINDOW = 126
HYBRID_STABILITY_WINDOW = 189


# ===== Strategy Implementations =====


class FactorTiltStrategy:
    """
    3D Factor Model strategy with factor tilts.

    Computes optimal sector weights based on:
    1. Sector factor betas (duration, credit, liquidity)
    2. Factor forecasts (user-provided or momentum-based)
    3. Position size constraints

    This strategy tilts portfolio weights toward sectors with favorable
    factor exposures given expected factor returns.

    Examples
    --------
    >>> strategy = FactorTiltStrategy(use_rolling_betas=False)
    >>> weights = strategy.compute_weights(date, dataset)
    >>> print(weights)
    {'XLK': 0.15, 'XLF': 0.12, ...}
    """

    def __init__(
        self,
        use_rolling_betas: bool = False,
        rolling_window: int = 252,  # 1 year
        factor_forecasts: dict[str, float] | None = None,
        min_weight: float = 0.0,
        max_weight: float | None = 0.30,
        volatility_floor: float = 1e-3,
        min_observations: int = 60,
        score_exponent: float = 0.75,
        blend_to_equal: float = 0.25,
        turnover_smoothing: float = 0.0,
        dynamic_factor_scaling: bool = False,
        scaling_lookback: int = 126,
        scaling_threshold: float = 0.01,
        scaling_floor: float = 0.4,
        regime_scaling: bool = False,
        regime_scaling_map: Mapping[str, float] | None = None,
        regime_scaling_floor: float = 0.3,
        regime_resolver: Callable[[datetime], str | None] | None = None,
        regime_factor_multipliers: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        """
        Initialize factor tilt strategy.

        Parameters
        ----------
        use_rolling_betas : bool, default False
            If True, use rolling betas; else use stable (full-sample) betas.
        rolling_window : int, default 252
            Window size for rolling beta estimation (trading days).
        factor_forecasts : dict[str, float], optional
            User-provided factor return forecasts.
            Keys: "factor_duration", "factor_credit", "factor_liquidity"
            If None, use simple momentum signals.
        min_weight : float, default 0.0
            Minimum weight per sector after optimization.
        max_weight : float | None, default 0.30
            Optional maximum weight per sector (None to disable cap).
        volatility_floor : float, default 1e-3
            Floor applied to volatility when computing risk-adjusted scores
            to prevent division-by-zero.
        min_observations : int, default 60
            Minimum number of observations required when estimating betas.
        score_exponent : float, default 0.75
            Exponent applied to normalized scores to temper concentration.
        blend_to_equal : float, default 0.25
            Fraction of weight blended back to equal-weight baseline to reduce turnover.
        turnover_smoothing : float, default 0.0
            Weight applied to the previous allocation when computing the new target to curb turnover.
        dynamic_factor_scaling : bool, default False
            If True, scale factor forecasts when recent realized factor performance is weak.
        scaling_lookback : int, default 126
            Lookback window (days) when computing realized factor performance for scaling.
        scaling_threshold : float, default 0.01
            Annualized threshold below which factor exposure is dampened.
        scaling_floor : float, default 0.4
            Minimum scaling multiplier applied when a factor breaches the threshold.
        regime_scaling : bool, default False
            If True, apply additional scaling based on regime-specific attribution signals.
        regime_scaling_map : Mapping[str, float] | None
            Optional mapping from regime name to scaling multiplier (< 1 reduces exposure).
        regime_scaling_floor : float, default 0.3
            Floor applied to regime-supplied scaling factors.
        regime_resolver : Callable[[datetime], str | None], optional
            Callable returning the active regime for a given date. Required when `regime_scaling`
            is enabled in order to apply the appropriate controls.
        regime_factor_multipliers : Mapping[str, Mapping[str, float]] | None
            Optional nested mapping of regime name to per-factor multipliers for fine-grained
            dampening (e.g., {"Rate Hiking Cycle": {"factor_liquidity": 0.6}}).
        """
        self.use_rolling_betas = use_rolling_betas
        self.rolling_window = rolling_window
        self.factor_forecasts = factor_forecasts
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.volatility_floor = volatility_floor
        self.min_observations = max(min_observations, 10)
        self.score_exponent = score_exponent
        self.blend_to_equal = blend_to_equal
        self.turnover_smoothing = turnover_smoothing
        self.dynamic_factor_scaling = dynamic_factor_scaling
        self.scaling_lookback = scaling_lookback
        self.scaling_threshold = scaling_threshold
        self.scaling_floor = scaling_floor
        self.regime_scaling = regime_scaling
        self.regime_scaling_map = dict(regime_scaling_map) if regime_scaling_map is not None else {}
        self.regime_scaling_floor = regime_scaling_floor
        self.regime_resolver = regime_resolver
        self.regime_factor_multipliers: dict[str, dict[str, float]] = {}
        if regime_factor_multipliers is not None:
            for regime_name, factor_map in regime_factor_multipliers.items():
                normalized: dict[str, float] = {}
                for factor, multiplier in factor_map.items():
                    normalized[factor] = float(multiplier)
                self.regime_factor_multipliers[regime_name] = normalized
        if self.rolling_window <= 0:
            msg = "rolling_window must be positive"
            raise ValueError(msg)
        if self.volatility_floor <= 0:
            msg = "volatility_floor must be positive"
            raise ValueError(msg)
        if self.min_weight < 0:
            msg = "min_weight must be non-negative"
            raise ValueError(msg)
        if self.max_weight is not None and self.max_weight <= 0:
            msg = "max_weight must be positive when provided"
            raise ValueError(msg)
        if self.max_weight is not None and self.max_weight < self.min_weight:
            msg = "max_weight cannot be less than min_weight"
            raise ValueError(msg)
        if self.score_exponent <= 0:
            msg = "score_exponent must be positive"
            raise ValueError(msg)
        if not (0.0 <= self.blend_to_equal < 1.0):
            msg = "blend_to_equal must be in [0, 1)"
            raise ValueError(msg)
        if not (0.0 <= self.turnover_smoothing < 1.0):
            msg = "turnover_smoothing must be in [0, 1)"
            raise ValueError(msg)
        if self.scaling_lookback <= 0:
            msg = "scaling_lookback must be positive"
            raise ValueError(msg)
        if self.scaling_threshold <= 0:
            msg = "scaling_threshold must be positive"
            raise ValueError(msg)
        if self.scaling_floor <= 0 or self.scaling_floor > 1.0:
            msg = "scaling_floor must be in (0, 1]"
            raise ValueError(msg)
        if self.regime_scaling_floor <= 0 or self.regime_scaling_floor > 1.0:
            msg = "regime_scaling_floor must be in (0, 1]"
            raise ValueError(msg)
        for regime_name, multiplier in self.regime_scaling_map.items():
            if multiplier <= 0:
                msg = f"regime_scaling_map multiplier for {regime_name} must be positive"
                raise ValueError(msg)
        for regime_name, factor_map in self.regime_factor_multipliers.items():
            for factor, multiplier in factor_map.items():
                if multiplier <= 0:
                    msg = f"regime_factor_multipliers[{regime_name!r}][{factor!r}] must be positive"
                    raise ValueError(msg)
        self.logger = LOGGER.bind(strategy="factor_tilt")
        self._beta_cache: dict[tuple[str, datetime], dict[str, dict[str, float]]] = {}
        self._vol_cache: dict[tuple[str, datetime], dict[str, float]] = {}
        self._last_factor_forecasts: dict[str, float] | None = None
        self._previous_weights: dict[str, float] | None = None

    def compute_weights(
        self,
        date: datetime,
        dataset: SectorDataset,
    ) -> dict[str, float]:
        """
        Compute optimal sector weights for given date.

        Algorithm:
        1. Estimate factor betas for each sector (stable or rolling)
        2. Forecast factor returns (momentum or user-provided)
        3. Compute expected sector returns: E[R_i] = β_i' × E[f]
        4. Optimize weights to maximize expected return subject to constraints

        Parameters
        ----------
        date : datetime
            Current rebalance date.
        dataset : SectorDataset
            Historical sector returns and factor data.

        Returns
        -------
        dict[str, float]
            Sector weights (sum to 1.0).

        """
        sectors = self._get_available_sectors(date, dataset)
        if not sectors:
            return {}

        betas, volatilities = self._estimate_betas(
            date=date,
            dataset=dataset,
            sectors=sectors,
        )

        if not betas:
            self.logger.warning(
                "Unable to estimate betas; falling back to equal weight",
                date=date.isoformat(),
            )
            return self._equal_weight(sectors)

        forecasts = self._forecast_factor_returns(date, dataset)
        if not forecasts:
            self.logger.warning(
                "Unable to generate factor forecasts; falling back to equal weight",
                date=date.isoformat(),
            )
            return self._equal_weight(sectors)

        sector_scores = self._score_sectors(
            sectors=sectors,
            betas=betas,
            forecasts=forecasts,
            volatilities=volatilities,
        )

        if not sector_scores:
            self.logger.warning(
                "All sector scores non-positive; using equal weight",
                date=date.isoformat(),
            )
            return self._equal_weight(sectors)

        weights = self._normalize_scores(sector_scores)
        constrained = self._apply_weight_constraints(weights)
        smoothed = self._apply_turnover_smoothing(constrained)
        blended = self._blend_with_equal(smoothed)

        self.logger.debug(
            "Computed factor tilt weights",
            date=date.isoformat(),
            use_rolling=self.use_rolling_betas,
            forecasts=forecasts,
        )

        self._previous_weights = blended.copy()

        return blended

    def _get_available_sectors(
        self,
        date: datetime,
        dataset: SectorDataset,
    ) -> list[str]:
        """Get list of sectors with data available up to date."""
        # Filter to data before or on date (no look-ahead bias)
        historical_data = dataset.sector_returns.filter(
            pl.col("timestamp") <= date
        )

        if historical_data.is_empty():
            return []

        return sorted(historical_data["symbol"].unique().to_list())

    def _estimate_betas(
        self,
        date: datetime,
        dataset: SectorDataset,
        sectors: list[str],
    ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        """
        Estimate factor betas for each sector.
        """
        cache_key = ("rolling" if self.use_rolling_betas else "stable", date)

        if cache_key in self._beta_cache and cache_key in self._vol_cache:
            return self._beta_cache[cache_key], self._vol_cache[cache_key]

        factor_columns = self._get_factor_columns(dataset.factor_returns)
        if not factor_columns:
            self.logger.warning("No factor columns available for beta estimation")
            return {}, {}

        sector_returns = dataset.sector_returns.filter(pl.col("timestamp") <= date)
        factor_returns = dataset.factor_returns.filter(pl.col("timestamp") <= date)

        if sector_returns.is_empty() or factor_returns.is_empty():
            return {}, {}

        joined = (
            sector_returns.join(factor_returns, on="timestamp", how="inner")
            .filter(pl.col("symbol").is_in(sectors))
            .sort(["symbol", "timestamp"])
            .drop_nulls(subset=["return", *factor_columns])
        )

        if joined.is_empty():
            return {}, {}

        betas: dict[str, dict[str, float]] = {}
        volatilities: dict[str, float] = {}

        # Partition per sector for regression
        for frame in joined.partition_by("symbol", maintain_order=True):
            sector = str(frame["symbol"][0])

            sector_frame = frame.select(["timestamp", "return", *factor_columns]).sort("timestamp")

            if self.use_rolling_betas:
                sector_frame = sector_frame.tail(self.rolling_window)

            if sector_frame.height < max(len(factor_columns) + 1, self.min_observations):
                continue

            y = sector_frame["return"].to_numpy()
            X = np.column_stack([
                sector_frame[col].to_numpy()
                for col in factor_columns
            ])

            # Add intercept term
            X = np.column_stack([X, np.ones(len(y))])

            try:
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                self.logger.warning("Unable to solve regression", sector=sector)
                continue

            betas[sector] = {
                factor: float(coef[i])
                for i, factor in enumerate(factor_columns)
            }

            volatility = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
            volatilities[sector] = volatility

        self._beta_cache[cache_key] = betas
        self._vol_cache[cache_key] = volatilities

        return betas, volatilities

    def _forecast_factor_returns(
        self,
        date: datetime,
        dataset: SectorDataset,
    ) -> dict[str, float]:
        """
        Forecast factor returns using a hybrid momentum signal.

        The hybrid signal blends short-, medium-, and long-horizon momentum with
        mild shrinkage toward zero when limited history is available. Results are
        clipped via z-score normalization to avoid extreme leverage on outliers.
        """
        if self.factor_forecasts is not None:
            return self.factor_forecasts.copy()
        factor_columns = self._get_factor_columns(dataset.factor_returns)
        if not factor_columns:
            return {}

        factor_history = (
            dataset.factor_returns
            .filter(pl.col("timestamp") <= date)
            .sort("timestamp")
        )

        if factor_history.is_empty():
            return {}

        forecasts: dict[str, float] = {}
        for factor in factor_columns:
            series = factor_history[factor].drop_nulls().drop_nans()
            if series.is_empty():
                forecasts[factor] = 0.0
                continue

            values = series.to_numpy()
            if values.size < self.min_observations:
                forecasts[factor] = 0.0
                continue

            short_avg = self._window_mean(values, HYBRID_SHORT_WINDOW)
            medium_avg = self._window_mean(values, HYBRID_MEDIUM_WINDOW)
            long_avg = self._window_mean(values, HYBRID_LONG_WINDOW)

            components: list[tuple[float, float]] = []
            if short_avg is not None:
                components.append((short_avg, 0.6))
            if medium_avg is not None:
                components.append((medium_avg, 0.3))
            if long_avg is not None:
                components.append((long_avg, 0.1))

            if components:
                weight_sum = sum(weight for _, weight in components)
                momentum_signal = sum(avg * weight for avg, weight in components) / weight_sum
            else:
                momentum_signal = float(values[-1])

            trailing_window = values[-HYBRID_LONG_WINDOW:] if values.size >= HYBRID_LONG_WINDOW else values
            baseline_mean = float(trailing_window.mean())
            baseline_std = float(trailing_window.std(ddof=1)) if trailing_window.size > 1 else 0.0

            if baseline_std > 0:
                zscore = (momentum_signal - baseline_mean) / baseline_std
                clipped_z = float(np.clip(zscore, -2.5, 2.5))
                normalized_signal = baseline_mean + clipped_z * baseline_std
            else:
                normalized_signal = momentum_signal

            stability_window = max(self.min_observations, HYBRID_STABILITY_WINDOW)
            stability_ratio = min(1.0, values.size / stability_window)
            forecasts[factor] = normalized_signal * stability_ratio

        if self.dynamic_factor_scaling:
            scaling = self._compute_dynamic_scaling(date, dataset, factor_columns)
            for factor, scale in scaling.items():
                forecasts[factor] = forecasts.get(factor, 0.0) * scale
        if self.regime_scaling:
            forecasts = self._apply_regime_scaling(date=date, forecasts=forecasts)
        self._last_factor_forecasts = forecasts
        return forecasts

    def _score_sectors(
        self,
        sectors: Iterable[str],
        betas: dict[str, dict[str, float]],
        forecasts: dict[str, float],
        volatilities: dict[str, float],
    ) -> dict[str, float]:
        """Compute risk-adjusted scores for each sector."""
        scores: dict[str, float] = {}
        factor_keys = forecasts.keys()

        for sector in sectors:
            sector_betas = betas.get(sector)
            if sector_betas is None:
                continue

            expected_return = sum(
                sector_betas.get(factor, 0.0) * forecasts.get(factor, 0.0)
                for factor in factor_keys
            )

            volatility = max(volatilities.get(sector, 0.0), self.volatility_floor)

            # Risk-adjusted score (similar to information ratio proxy)
            score = expected_return / volatility
            scores[sector] = score

        return scores

    def _normalize_scores(
        self,
        scores: dict[str, float],
    ) -> dict[str, float]:
        """Normalize risk-adjusted scores to weights."""
        if not scores:
            return {}

        # Shift scores so they are non-negative
        min_score = min(scores.values())
        shift = abs(min_score) + 1e-6 if min_score < 0 else 0.0

        adjusted = {
            sector: max(score + shift, 0.0)
            for sector, score in scores.items()
        }

        if self.score_exponent != 1.0:
            adjusted = {
                sector: value ** self.score_exponent
                for sector, value in adjusted.items()
            }

        total = sum(adjusted.values())
        if total <= 0:
            return {}

        return {sector: value / total for sector, value in adjusted.items()}

    def _apply_weight_constraints(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Apply min/max weight constraints and renormalize."""
        if not weights:
            return {}

        sectors = list(weights.keys())
        num_sectors = len(sectors)

        min_bound = max(self.min_weight, 0.0)
        max_bound = self.max_weight if self.max_weight is not None else 1.0

        if min_bound > max_bound:
            self.logger.warning(
                "Inconsistent bounds detected; reverting to equal weight",
                min_bound=min_bound,
                max_bound=max_bound,
            )
            return self._equal_weight(sectors)

        if num_sectors * min_bound > 1.0 + 1e-9:
            self.logger.warning(
                "Minimum weight constraint infeasible; reverting to equal weight",
                min_bound=min_bound,
                sectors=num_sectors,
            )
            return self._equal_weight(sectors)

        if self.max_weight is not None and num_sectors * max_bound < 1.0 - 1e-9:
            self.logger.warning(
                "Maximum weight constraint infeasible; reverting to equal weight",
                max_bound=max_bound,
                sectors=num_sectors,
            )
            return self._equal_weight(sectors)

        clipped = {
            sector: min(max(weight, min_bound), max_bound)
            for sector, weight in weights.items()
        }

        total = sum(clipped.values())
        if total <= 0:
            return self._equal_weight(sectors)

        target_total = 1.0
        tolerance = 1e-9

        if abs(total - target_total) <= tolerance:
            return clipped

        if total < target_total:
            deficit = target_total - total
            adjustable = [sector for sector in sectors if clipped[sector] < max_bound - tolerance]
            while adjustable and deficit > tolerance:
                per_sector = deficit / len(adjustable)
                next_round: list[str] = []
                for sector in adjustable:
                    room = max_bound - clipped[sector]
                    allocation = min(room, per_sector)
                    clipped[sector] += allocation
                    deficit -= allocation
                    if clipped[sector] < max_bound - tolerance:
                        next_round.append(sector)
                if len(next_round) == len(adjustable):
                    break
                adjustable = next_round
            if deficit > tolerance:
                total = sum(clipped.values())
                if total <= 0:
                    return self._equal_weight(sectors)
                return {sector: weight / total for sector, weight in clipped.items()}
        else:
            excess = total - target_total
            adjustable = [sector for sector in sectors if clipped[sector] > min_bound + tolerance]
            while adjustable and excess > tolerance:
                per_sector = excess / len(adjustable)
                next_round = []
                for sector in adjustable:
                    room = clipped[sector] - min_bound
                    reduction = min(room, per_sector)
                    clipped[sector] -= reduction
                    excess -= reduction
                    if clipped[sector] > min_bound + tolerance:
                        next_round.append(sector)
                if len(next_round) == len(adjustable):
                    break
                adjustable = next_round
            if excess > tolerance:
                total = sum(clipped.values())
                if total <= 0:
                    return self._equal_weight(sectors)
                return {sector: weight / total for sector, weight in clipped.items()}

        final_total = sum(clipped.values())
        if abs(final_total - target_total) > tolerance and final_total > 0:
            return {
                sector: weight / final_total
                for sector, weight in clipped.items()
            }

        return clipped

    def _blend_with_equal(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Blend computed weights with equal-weight baseline to control turnover."""
        if not weights or self.blend_to_equal <= 0:
            return weights

        equal_weights = self._equal_weight(list(weights.keys()))
        blend = self.blend_to_equal
        blended = {
            sector: (1.0 - blend) * weights.get(sector, 0.0) + blend * equal_weights.get(sector, 0.0)
            for sector in equal_weights
        }
        total = sum(blended.values())
        if total <= 0:
            return self._equal_weight(list(equal_weights.keys()))

        return {
            sector: value / total
            for sector, value in blended.items()
        }

    def _apply_turnover_smoothing(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Blend weights with previous allocation to discourage turnover."""
        if not weights or self.turnover_smoothing <= 0.0 or self._previous_weights is None:
            return weights

        smoothing = self.turnover_smoothing
        smoothed = {
            sector: (1.0 - smoothing) * weights.get(sector, 0.0) + smoothing * self._previous_weights.get(sector, 0.0)
            for sector in weights
        }
        total = sum(smoothed.values())
        if total <= 0:
            return weights

        return {
            sector: value / total
            for sector, value in smoothed.items()
        }

    def _compute_dynamic_scaling(
        self,
        date: datetime,
        dataset: SectorDataset,
        factors: list[str],
    ) -> dict[str, float]:
        """Compute scaling multipliers based on recent factor performance."""
        history = (
            dataset.factor_returns
            .filter(pl.col("timestamp") <= date)
            .sort("timestamp")
        )
        window = history.tail(self.scaling_lookback)
        if window.is_empty():
            return {}

        scaling: dict[str, float] = {}
        for factor in factors:
            if factor not in window.columns:
                continue
            series = window[factor].drop_nulls().drop_nans()
            if series.is_empty():
                continue
            mean_value = series.mean()
            mean_daily = float(cast(float, mean_value)) if mean_value is not None else 0.0
            annualized = mean_daily * 252.0
            if annualized >= self.scaling_threshold:
                multiplier = 1.0
            elif annualized <= -self.scaling_threshold:
                multiplier = self.scaling_floor
            else:
                upper = self.scaling_threshold
                lower = -self.scaling_threshold
                ratio = (annualized - lower) / (upper - lower)
                multiplier = self.scaling_floor + ratio * (1.0 - self.scaling_floor)
            scaling[factor] = max(self.scaling_floor, min(1.0, multiplier))

        return scaling

    def _apply_regime_scaling(
        self,
        date: datetime,
        forecasts: dict[str, float],
    ) -> dict[str, float]:
        """Apply regime-aware scaling heuristics to factor forecasts."""
        if self.regime_resolver is None:
            return forecasts
        regime_name = self.regime_resolver(date)
        if regime_name is None:
            return forecasts
        scaled = forecasts.copy()
        base_multiplier = self.regime_scaling_map.get(regime_name)
        if base_multiplier is not None:
            clamped = self._clamp_scaling_multiplier(base_multiplier)
            for factor, value in scaled.items():
                scaled[factor] = value * clamped
        factor_overrides = self.regime_factor_multipliers.get(regime_name)
        if factor_overrides:
            for factor, multiplier in factor_overrides.items():
                if factor not in scaled:
                    continue
                clamped = self._clamp_scaling_multiplier(multiplier)
                scaled[factor] = scaled[factor] * clamped
        return scaled

    def _clamp_scaling_multiplier(self, multiplier: float) -> float:
        """Clamp scaling multipliers to the configured safe range."""
        if multiplier <= 0:
            return self.regime_scaling_floor
        return max(self.regime_scaling_floor, min(1.0, multiplier))

    def _equal_weight(self, sectors: list[str]) -> dict[str, float]:
        """Return equal weights across sectors."""
        if not sectors:
            return {}
        weight = 1.0 / len(sectors)
        return dict.fromkeys(sectors, weight)

    @staticmethod
    def _get_factor_columns(factor_returns: pl.DataFrame) -> list[str]:
        """Return factor column names from the factor returns frame."""
        return [
            column
            for column in factor_returns.columns
            if column != "timestamp"
        ]

    @staticmethod
    def _window_mean(values: np.ndarray, window: int) -> float | None:
        """Compute trailing mean for the specified window size."""
        if values.size < window:
            return None
        return float(values[-window:].mean())


class EqualWeightStrategy:
    """
    Simple equal-weight (1/N) benchmark.

    Allocates 1/N to each available sector with monthly rebalancing.
    This is a standard benchmark for active portfolio strategies.

    Examples
    --------
    >>> strategy = EqualWeightStrategy()
    >>> weights = strategy.compute_weights(date, dataset)
    >>> print(weights)
    {'XLK': 0.111, 'XLF': 0.111, ...}  # 1/9 for 9 sectors
    """

    def __init__(self) -> None:
        """Initialize equal-weight strategy."""
        self.logger = LOGGER.bind(strategy="equal_weight")

    def compute_weights(
        self,
        date: datetime,
        dataset: SectorDataset,
    ) -> dict[str, float]:
        """
        Return equal weights across all sectors.

        Parameters
        ----------
        date : datetime
            Current rebalance date.
        dataset : SectorDataset
            Historical sector returns and factor data.

        Returns
        -------
        dict[str, float]
            Equal weights for each sector (sum to 1.0).
        """
        # Get sectors with data available up to date (no look-ahead bias)
        historical_data = dataset.sector_returns.filter(
            pl.col("timestamp") <= date
        )

        if historical_data.is_empty():
            self.logger.warning("No historical data available", date=date.isoformat())
            return {}

        sectors = sorted(historical_data["symbol"].unique().to_list())

        if not sectors:
            self.logger.warning("No sectors found", date=date.isoformat())
            return {}

        # Equal weight
        weight = 1.0 / len(sectors)
        weights = dict.fromkeys(sectors, weight)

        self.logger.debug(
            "Computed equal weights",
            date=date.isoformat(),
            num_sectors=len(sectors),
            weight=f"{weight:.4f}",
        )

        return weights


__all__ = [
    "EqualWeightStrategy",
    "FactorTiltStrategy",
]
