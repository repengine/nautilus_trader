from __future__ import annotations


"""
Internal typing aliases to keep runtime dependencies optional while providing
precise static types for mypy/ruff.

This module should contain ONLY type aliases and imports guarded by
`TYPE_CHECKING` so importing it has no heavy runtime side effects.
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as _pd
    import polars as _pl
    from sklearn.preprocessing import StandardScaler as _StandardScaler
else:  # Fallbacks for runtime when deps may be absent
    class _pd:  # type: ignore[too-many-function-args]
        class DataFrame:  # pragma: no cover - runtime stub
            pass

        class Series:  # pragma: no cover - runtime stub
            pass

    class _pl:  # type: ignore[too-many-function-args]
        class DataFrame:  # pragma: no cover - runtime stub
            pass

        class Series:  # pragma: no cover - runtime stub
            pass

    class _StandardScaler:  # pragma: no cover - runtime stub
        pass


# Public aliases
PandasDF = _pd.DataFrame
PandasSeries = _pd.Series
PolarsDF = _pl.DataFrame
PolarsSeries = _pl.Series

# Union convenience types
DataFrameLike = PandasDF | PolarsDF
SeriesLike = PandasSeries | PolarsSeries

# Sklearn scaler alias (type-only)
StandardScaler = _StandardScaler

