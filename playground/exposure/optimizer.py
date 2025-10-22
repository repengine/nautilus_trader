"""Portfolio optimizer for aligning exposures to a target risk point."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import polars as pl


if TYPE_CHECKING:  # pragma: no cover - type checking only
    import cvxpy  # noqa: F401


_CP_MODULE: ModuleType | None = None


def _require_cvxpy() -> ModuleType:
    global _CP_MODULE
    if _CP_MODULE is None:
        try:
            module = import_module("cvxpy")
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("cvxpy is required for constrained optimisation") from exc
        _CP_MODULE = module
    return _CP_MODULE


@dataclass(slots=True)
class RiskPoint:
    """Target risk coordinates keyed by factor column name."""

    coordinates: dict[str, float]

    def to_vector(self, factor_names: Iterable[str]) -> np.ndarray:
        try:
            return np.array([self.coordinates[name] for name in factor_names], dtype=float)
        except KeyError as exc:  # pragma: no cover - debug aid
            msg = f"Target risk point missing coordinate for factor '{exc.args[0]}'"
            raise ValueError(msg) from exc


def default_target_point() -> RiskPoint:
    """Return the default target risk point for duration/credit/liquidity axes."""
    return RiskPoint(
        {
            "factor_duration": 0.4,
            "factor_credit": 0.35,
            "factor_liquidity": 0.25,
        },
    )


def compute_optimal_weights(
    exposures: pl.DataFrame,
    target: RiskPoint,
    *,
    min_weight: float = 0.0,
    max_weight: float | None = None,
    weight_caps: Mapping[str, float] | None = None,
    regularization: float = 1e-6,
    solver: str | None = None,
) -> dict[str, float]:
    """
    Compute portfolio weights that minimize squared distance to the target point.

    Parameters
    ----------
    exposures
        Long-form EWMA beta observations keyed by ``asset_id`` and ``benchmark_id``.
    target
        Desired factor coordinates (Duration/Credit/Liquidity).
    min_weight
        Lower bound applied element-wise prior to normalisation.
    max_weight
        Optional global upper bound for each asset weight (``None`` leaves the
        upper bound unconstrained apart from the simplex).
    weight_caps
        Optional per-asset caps overriding ``max_weight``. Keys must match
        ``asset_id`` values present in the exposure frame.
    regularization
        Ridge-style regularisation term applied inside the quadratic program.
    solver
        Optional cvxpy solver identifier (defaults to the library heuristic).
    """
    if exposures.is_empty():
        raise ValueError("Exposures DataFrame is empty")

    latest = exposures.sort("ts_event").group_by(["asset_id", "benchmark_id"]).tail(1)

    wide = latest.pivot(
        index="asset_id",
        on="benchmark_id",
        values="ewma_beta",
    )

    factor_names = [col for col in wide.columns if col != "asset_id"]
    if not factor_names:
        raise ValueError("No factor columns found in exposures")

    matrix = wide.select(factor_names).to_numpy()
    target_vector = target.to_vector(factor_names)

    if matrix.size == 0:
        raise ValueError("Exposure matrix is empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Exposure matrix contains non-finite values")
    if not np.all(np.isfinite(target_vector)):
        raise ValueError("Target vector contains non-finite values")

    asset_ids = wide["asset_id"].to_list()
    lower_bounds = np.full(len(asset_ids), min_weight, dtype=float)

    if max_weight is not None and max_weight <= 0:
        raise ValueError("max_weight must be positive when provided")

    upper_bounds = None
    if weight_caps:
        upper_bounds = np.array(
            [
                float(weight_caps.get(asset, max_weight if max_weight is not None else 1.0))
                for asset in asset_ids
            ],
            dtype=float,
        )
    elif max_weight is not None:
        upper_bounds = np.full(len(asset_ids), max_weight, dtype=float)

    if upper_bounds is not None and np.any(upper_bounds < lower_bounds):
        raise ValueError("Weight caps must be greater than or equal to min_weight")

    cp_module = _require_cvxpy()

    weights_var = cp_module.Variable(len(asset_ids))
    residual = matrix.T @ weights_var - target_vector
    objective = cp_module.Minimize(
        cp_module.sum_squares(residual) + regularization * cp_module.sum_squares(weights_var),
    )
    constraints = [cp_module.sum(weights_var) == 1, weights_var >= lower_bounds]
    if upper_bounds is not None:
        constraints.append(weights_var <= upper_bounds)

    problem = cp_module.Problem(objective, constraints)
    try:
        problem.solve(solver=solver, warm_start=True)
    except Exception as exc:  # pragma: no cover - cvxpy failure path
        raise ValueError("Constrained optimisation failed") from exc

    if problem.status not in {cp_module.OPTIMAL, cp_module.OPTIMAL_INACCURATE}:
        raise ValueError(f"Optimizer status not optimal: {problem.status}")

    solution = np.array(weights_var.value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(solution)):
        raise ValueError("Optimiser produced non-finite weights")

    solution = np.maximum(solution, lower_bounds)
    if upper_bounds is not None:
        solution = np.minimum(solution, upper_bounds)

    total = solution.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Computed weights are degenerate (sum <= 0)")

    solution = solution / total
    return dict(zip(asset_ids, solution.tolist()))
