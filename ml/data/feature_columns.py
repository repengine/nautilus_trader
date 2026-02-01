"""
Shared helpers for inferring numeric feature columns.

These utilities are used by dataset validation, feature manifest export, and dataset
artifact generation to keep feature inference consistent.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ml._imports import HAS_PANDAS
from ml._imports import pd
from ml._imports import pl


DEFAULT_FEATURE_EXCLUDE_COLUMNS: tuple[str, ...] = (
    "y",
    "forward_return",
    "cost_return",
    "time_index",
    "timestamp",
    "instrument_id",
    "ts_event",
    "asset_class",
    "exchange",
    "timestamp_right",
    "timestamp_left",
)

DEFAULT_FEATURE_EXCLUDE_SUFFIXES: tuple[str, ...] = ("_vintage_ts",)
DEFAULT_FEATURE_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "forward_return_",
    "cost_return_",
    "target_bin_",
    "target_class_",
    "target_reg_",
)


def split_feature_columns(
    df: Any,
    *,
    exclude: Sequence[str] | None = None,
    exclude_suffixes: Sequence[str] | None = None,
    exclude_prefixes: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Split candidate feature columns into numeric and non-numeric groups.

    Parameters
    ----------
    df : Any
        Polars or pandas DataFrame to inspect.
    exclude : Sequence[str] | None, optional
        Column names to exclude from feature candidates.
    exclude_suffixes : Sequence[str] | None, optional
        Column suffixes to exclude from feature candidates.

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple of (numeric_columns, non_numeric_columns).

    """
    exclude_set = {
        str(name) for name in (exclude if exclude is not None else DEFAULT_FEATURE_EXCLUDE_COLUMNS)
    }
    suffixes = (
        tuple(str(suffix) for suffix in exclude_suffixes)
        if exclude_suffixes is not None
        else DEFAULT_FEATURE_EXCLUDE_SUFFIXES
    )
    prefixes = (
        tuple(str(prefix) for prefix in exclude_prefixes)
        if exclude_prefixes is not None
        else DEFAULT_FEATURE_EXCLUDE_PREFIXES
    )

    def is_excluded(name: str) -> bool:
        if name in exclude_set:
            return True
        if any(name.endswith(suffix) for suffix in suffixes):
            return True
        return any(name.startswith(prefix) for prefix in prefixes)

    if pl is not None and isinstance(df, pl.DataFrame):
        candidates = [name for name in df.columns if not is_excluded(name)]
        numeric = [
            name
            for name in candidates
            if df[name].dtype.is_numeric() or df[name].dtype == pl.Boolean
        ]
        non_numeric = [name for name in candidates if name not in numeric]
        return numeric, non_numeric

    if HAS_PANDAS and pd is not None and isinstance(df, pd.DataFrame):  # pragma: no cover
        candidates = [name for name in df.columns if not is_excluded(str(name))]
        numeric = []
        non_numeric = []
        for name in candidates:
            series = df[name]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                numeric.append(str(name))
            else:
                non_numeric.append(str(name))
        return numeric, non_numeric

    return [], []


def infer_numeric_feature_columns(
    df: Any,
    *,
    exclude: Sequence[str] | None = None,
    exclude_suffixes: Sequence[str] | None = None,
    exclude_prefixes: Sequence[str] | None = None,
) -> list[str]:
    """
    Infer numeric feature columns after exclusions.

    Parameters
    ----------
    df : Any
        Polars or pandas DataFrame to inspect.
    exclude : Sequence[str] | None, optional
        Column names to exclude from feature candidates.
    exclude_suffixes : Sequence[str] | None, optional
        Column suffixes to exclude from feature candidates.

    Returns
    -------
    list[str]
        Numeric feature column names.

    """
    numeric, _ = split_feature_columns(
        df,
        exclude=exclude,
        exclude_suffixes=exclude_suffixes,
        exclude_prefixes=exclude_prefixes,
    )
    return numeric


__all__ = [
    "DEFAULT_FEATURE_EXCLUDE_COLUMNS",
    "DEFAULT_FEATURE_EXCLUDE_PREFIXES",
    "DEFAULT_FEATURE_EXCLUDE_SUFFIXES",
    "infer_numeric_feature_columns",
    "split_feature_columns",
]
