"""
Macro delta feature helpers for batch datasets.

These utilities are used by dataset builders to append 1-day deltas for
configured macro series after a join.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ml._imports import pl as pl_runtime


if TYPE_CHECKING:
    from polars import DataFrame as PlDataFrame
else:  # pragma: no cover - typing fallback
    PlDataFrame = Any  # type: ignore[assignment]


pl: Any = cast(Any, pl_runtime)


def append_macro_delta_features_polars(
    df: PlDataFrame,
    *,
    include_macro: bool,
    include_macro_deltas: bool,
    macro_series_ids: tuple[str, ...] | None,
) -> PlDataFrame:
    """
    Append 1-day delta features for configured macro series columns.

    The deltas are computed per-instrument, ordered by the timestamp column
    (``timestamp`` or ``ts_event``). The first delta in each instrument group
    is filled with ``0.0``.

    Args:
        df: Polars DataFrame containing macro series columns.
        include_macro: Whether macro features are enabled.
        include_macro_deltas: Whether macro delta features are enabled.
        macro_series_ids: Macro series identifiers to compute deltas for.

    Returns:
        DataFrame with appended ``*_delta_1d`` columns when enabled.
    """
    if not (include_macro and include_macro_deltas and macro_series_ids):
        return df
    if df.is_empty():
        return df

    series_ids = [series_id for series_id in macro_series_ids if series_id in df.columns]
    if not series_ids:
        return df

    if "timestamp" in df.columns:
        time_col = "timestamp"
    elif "ts_event" in df.columns:
        time_col = "ts_event"
    else:
        return df

    if "instrument_id" in df.columns:
        df_sorted = df.sort(["instrument_id", time_col])
        exprs = [
            pl.col(series_id)
            .diff()
            .over("instrument_id")
            .fill_null(0.0)
            .alias(f"{series_id}_delta_1d")
            for series_id in series_ids
        ]
        return df_sorted.with_columns(exprs)

    df_sorted = df.sort(time_col)
    exprs = [
        pl.col(series_id).diff().fill_null(0.0).alias(f"{series_id}_delta_1d")
        for series_id in series_ids
    ]
    return df_sorted.with_columns(exprs)


__all__ = ["append_macro_delta_features_polars"]
