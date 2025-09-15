"""
Safe math utilities shared across features and data processing modules.

Provides safe division for scalar values and Polars expressions to avoid divide-by-zero
pitfalls and to keep implementations consistent.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml._imports import pl


if TYPE_CHECKING:  # pragma: no cover - typing only
    from polars import Expr as PlExpr


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Perform safe scalar division with zero/None guard.

    Returns `numerator / denominator` when `denominator` is truthy and non-zero,
    otherwise returns `default`.

    """
    if denominator is None or denominator == 0:
        return default
    return numerator / denominator


def safe_divide_expr(numer: PlExpr, denom: PlExpr) -> PlExpr:
    """
    Perform safe division for Polars expressions.

    Uses a guarded denominator which substitutes 1.0 when the input is <= 0 to avoid
    divide-by-zero while remaining differentiable for common transforms.

    """
    if pl is None:  # pragma: no cover - runtime dep guard
        raise RuntimeError("Polars is required for safe_divide_expr")
    from typing import cast as _cast

    return _cast("PlExpr", numer / pl.when(denom > 0).then(denom).otherwise(1.0))


__all__ = ["safe_divide", "safe_divide_expr"]
