from __future__ import annotations


# ruff: noqa: E402 - allow module docstring before imports in typing stubs

"""
Internal typing aliases to keep runtime dependencies optional while providing
precise static types for mypy/ruff.

This module should contain ONLY type aliases and imports guarded by
`TYPE_CHECKING` so importing it has no heavy runtime side effects.
"""

from typing import TYPE_CHECKING, Any, TypeAlias


if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as _pd
    import polars as _pl
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    PandasDF: TypeAlias = _pd.DataFrame
    PandasSeries: TypeAlias = _pd.Series
    PolarsDF: TypeAlias = _pl.DataFrame
    PolarsSeries: TypeAlias = _pl.Series
    StandardScaler = _StandardScaler
else:  # Fallbacks for runtime when deps may be absent
    # At runtime, these aliases are only used for annotations (from __future__ import annotations)
    # so we can safely assign to Any to avoid import-time failures.
    PandasDF: TypeAlias = Any
    PandasSeries: TypeAlias = Any
    PolarsDF: TypeAlias = Any
    PolarsSeries: TypeAlias = Any

    class StandardScaler:  # pragma: no cover - runtime stub
        pass


# Union convenience types
DataFrameLike: TypeAlias = PandasDF | PolarsDF
SeriesLike: TypeAlias = PandasSeries | PolarsSeries
