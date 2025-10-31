"""Helpers for computing economic and stability diagnostics for streaming runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt

import numpy as np
import numpy.typing as npt


def _as_1d(array: npt.NDArray[np.float64] | npt.NDArray[np.float32] | None) -> npt.NDArray[np.float64]:
    if array is None:
        return np.asarray([], dtype=np.float64)
    coerced = np.asarray(array, dtype=np.float64).reshape(-1)
    return coerced


def _default_returns(labels: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # Use a conservative +/- 10 bps profile when explicit returns are unavailable.
    up = 0.001
    down = -0.001
    return np.where(labels >= 0.5, up, down)


def _compute_turnover(signals: npt.NDArray[np.float64]) -> float | None:
    if signals.size <= 1:
        return None
    transitions = np.abs(np.diff(signals)) > 1e-9
    if transitions.size == 0:
        return 0.0
    return float(np.mean(transitions))


def _compute_drawdown(returns: npt.NDArray[np.float64]) -> float | None:
    if returns.size == 0:
        return None
    equity_curve = np.cumprod(1.0 + returns)
    if equity_curve.size == 0:
        return None
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / running_max - 1.0
    return float(np.min(drawdown)) if drawdown.size > 0 else None


def _two_sample_ks_statistic(
    sample_a: npt.NDArray[np.float64],
    sample_b: npt.NDArray[np.float64],
) -> float | None:
    if sample_a.size == 0 or sample_b.size == 0:
        return None
    combined = np.concatenate((sample_a, sample_b))
    if combined.size == 0:
        return None
    grid = np.sort(np.unique(combined))
    if grid.size == 0:
        return None
    cdf_a = np.searchsorted(np.sort(sample_a), grid, side="right") / sample_a.size
    cdf_b = np.searchsorted(np.sort(sample_b), grid, side="right") / sample_b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


@dataclass(slots=True, frozen=True)
class EconomicMetricBundle:
    """Economic diagnostics derived from validation predictions."""

    slippage_adjusted_sharpe: float | None
    hit_rate: float | None
    turnover: float | None
    max_drawdown: float | None

    def as_dict(self) -> dict[str, float]:
        """Return serialisable mapping without ``None`` entries."""
        mapping: dict[str, float] = {}
        if self.slippage_adjusted_sharpe is not None:
            mapping["economic_slippage_adjusted_sharpe"] = float(self.slippage_adjusted_sharpe)
        if self.hit_rate is not None:
            mapping["economic_hit_rate"] = float(self.hit_rate)
        if self.turnover is not None:
            mapping["economic_turnover"] = float(self.turnover)
        if self.max_drawdown is not None:
            mapping["economic_max_drawdown"] = float(self.max_drawdown)
        return mapping


@dataclass(slots=True, frozen=True)
class StabilityMetricBundle:
    """Stability diagnostics comparing train/validation distributions."""

    ks_statistic: float | None
    calibration_drift: float | None

    def as_dict(self) -> dict[str, float]:
        """Return serialisable mapping without ``None`` entries."""
        mapping: dict[str, float] = {}
        if self.ks_statistic is not None:
            mapping["stability_ks_statistic"] = float(self.ks_statistic)
        if self.calibration_drift is not None:
            mapping["stability_calibration_drift"] = float(self.calibration_drift)
        return mapping


def compute_economic_and_stability_metrics(
    *,
    validation_probabilities: npt.NDArray[np.float64] | npt.NDArray[np.float32],
    validation_labels: npt.NDArray[np.float64] | npt.NDArray[np.float32],
    training_probabilities: npt.NDArray[np.float64] | npt.NDArray[np.float32] | None = None,
    validation_returns: npt.NDArray[np.float64] | npt.NDArray[np.float32] | None = None,
    slippage_bps: float | None = None,
    calibration_metrics: Mapping[str, float] | None = None,
    baseline_calibration_ece: float | None = None,
) -> tuple[EconomicMetricBundle, StabilityMetricBundle, dict[str, float]]:
    """
    Compute economic and stability diagnostics for validation predictions.

    Args:
        validation_probabilities: Validation probabilities after blending.
        validation_labels: Binary validation labels aligned with probabilities.
        training_probabilities: Optional training probabilities for drift checks.
        validation_returns: Optional forward returns aligned with validation rows.
        slippage_bps: Estimated round-trip slippage in basis points.
        calibration_metrics: Mapping containing ``calibration_ece_20`` for drift comparison.
        baseline_calibration_ece: Reference calibration error for drift deltas.

    Returns:
        Economic bundle, stability bundle, and flattened metric mapping.
    """
    probs = _as_1d(validation_probabilities)
    labels = _as_1d(validation_labels)
    if probs.size == 0 or labels.size == 0:
        empty_economic = EconomicMetricBundle(None, None, None, None)
        empty_stability = StabilityMetricBundle(None, None)
        return empty_economic, empty_stability, {}

    returns = _as_1d(validation_returns)
    if returns.size == 0:
        returns = _default_returns(labels)
    if returns.size != labels.size:
        aligned = min(int(returns.size), int(labels.size))
        if aligned <= 0:
            returns = _default_returns(labels)
        else:
            returns = returns[:aligned]
            probs = probs[:aligned]
            labels = labels[:aligned]

    signals = np.where(probs >= 0.5, 1.0, -1.0)
    realized_direction = np.where(labels >= 0.5, 1.0, -1.0)
    gross_returns = realized_direction * signals * returns

    slippage_cost = 0.0 if slippage_bps is None else float(slippage_bps) / 10_000.0
    strategy_returns = gross_returns - slippage_cost * np.abs(signals)
    sharpe: float | None
    if strategy_returns.size == 0 or np.std(strategy_returns) == 0.0:
        sharpe = None
    else:
        sharpe = float(np.mean(strategy_returns) / np.std(strategy_returns) * sqrt(252.0))
    hit_rate: float | None
    if strategy_returns.size == 0:
        hit_rate = None
    else:
        hit_rate = float(np.mean(strategy_returns > 0.0))
    turnover = _compute_turnover(signals)
    drawdown = _compute_drawdown(strategy_returns)
    economic_bundle = EconomicMetricBundle(
        slippage_adjusted_sharpe=sharpe,
        hit_rate=hit_rate,
        turnover=turnover,
        max_drawdown=abs(drawdown) if drawdown is not None else None,
    )

    train_probs = _as_1d(training_probabilities)
    ks_stat = _two_sample_ks_statistic(train_probs, probs) if train_probs.size else None
    ece_value: float | None = None
    if calibration_metrics is not None:
        try:
            ece_value = float(calibration_metrics.get("calibration_ece_20", None))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            ece_value = None
    calibration_drift: float | None = None
    if ece_value is not None:
        if baseline_calibration_ece is not None:
            calibration_drift = ece_value - float(baseline_calibration_ece)
        else:
            calibration_drift = ece_value
    stability_bundle = StabilityMetricBundle(
        ks_statistic=ks_stat,
        calibration_drift=calibration_drift,
    )

    merged_metrics = {**economic_bundle.as_dict(), **stability_bundle.as_dict()}
    return economic_bundle, stability_bundle, merged_metrics


__all__ = [
    "EconomicMetricBundle",
    "StabilityMetricBundle",
    "compute_economic_and_stability_metrics",
]
