"""
Helpers for applying training-time transaction cost models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt


if TYPE_CHECKING:
    from ml.config.base import MLTrainingConfig


def resolve_round_trip_cost_decimal(config: MLTrainingConfig) -> float | None:
    """
    Resolve the round-trip cost in decimal return units from the training config.

    Args:
        config: Training configuration holding target semantics.

    Returns:
        Round-trip cost in decimal units, or ``None`` when no cost model is configured.
    """
    target_semantics = getattr(config, "target_semantics", None)
    if target_semantics is None:
        return None
    cost_model = getattr(target_semantics, "cost_model", None)
    if cost_model is None:
        return None
    return float(cost_model.round_trip_decimal)


def apply_round_trip_costs(
    *,
    strategy_returns: npt.NDArray[np.float64],
    signals: npt.NDArray[np.float64],
    cost_decimal: float | None,
) -> npt.NDArray[np.float64]:
    """
    Apply round-trip transaction costs to strategy returns.

    Args:
        strategy_returns: Gross strategy returns per period.
        signals: Trading signals aligned with returns (e.g., -1, 0, 1).
        cost_decimal: Round-trip cost in decimal return units.

    Returns:
        Cost-adjusted strategy returns.
    """
    if cost_decimal is None or cost_decimal <= 0.0:
        return strategy_returns
    return strategy_returns - float(cost_decimal) * np.abs(signals)


__all__ = [
    "apply_round_trip_costs",
    "resolve_round_trip_cost_decimal",
]
