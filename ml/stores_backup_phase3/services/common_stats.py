"""
Common SQL helpers for stats services (model/strategy), keeping stores thin.

These helpers centralize time-window WHERE clause construction with precise typing
and no import-time side effects. They return SQL fragments and parameter dicts
ready for use with SQLAlchemy `text(...)` queries.
"""

from __future__ import annotations

from typing import Any


def resolve_table_name(
    deps: Any,
    *,
    attr: str,
    base: str,
    allowed: set[str] | None,
) -> str:
    """Return a schema-qualified table name resilient to engine differences."""
    table_obj = getattr(deps, attr, None)
    if table_obj is not None:
        fullname = getattr(table_obj, "fullname", None)
        if fullname:
            return str(fullname)
        name = getattr(table_obj, "name", None)
        if name:
            schema = getattr(table_obj, "schema", None)
            if schema:
                return f"{schema}.{name}"
            return str(name)

    safe_fn = getattr(deps, "_safe_table", None)
    if callable(safe_fn) and allowed is not None:
        from typing import cast as _cast

        return _cast(str, safe_fn(base, allowed))

    qualified_fn = getattr(deps, "_qualified_table", None)
    if callable(qualified_fn):
        from typing import cast as _cast

        return _cast(str, qualified_fn(base))

    return base


def build_time_conditions(
    start_ns: int | None,
    end_ns: int | None,
    *,
    field: str = "ts_event",
) -> tuple[list[str], dict[str, int]]:
    """
    Return (conditions, params) for an AND-joined WHERE using provided field.

    - Only includes params for non-None bounds.
    - Example output: (["ts_event >= :start_ns", "ts_event < :end_ns"], {"start_ns": 1, "end_ns": 2})
    """
    conditions: list[str] = []
    params: dict[str, int] = {}
    if start_ns is not None:
        conditions.append(f"{field} >= :start_ns")
        params["start_ns"] = int(start_ns)
    if end_ns is not None:
        conditions.append(f"{field} < :end_ns")
        params["end_ns"] = int(end_ns)
    return conditions, params


def build_nullsafe_time_clause(
    start_ns: int | None,
    end_ns: int | None,
    *,
    field: str = "ts_event",
) -> tuple[str, dict[str, int | None]]:
    """
    Return (clause, params) for WHERE with NULL-safe time bounds on `field`.

    - Always includes both params (which may be None) to support `:param IS NULL` patterns.
    - Example output: ("(:start_ns IS NULL OR ts_event >= :start_ns) AND (:end_ns IS NULL OR ts_event < :end_ns)", {"start_ns": None, "end_ns": 2})
    """
    clause = (
        f"(:start_ns IS NULL OR {field} >= :start_ns) AND "
        f"(:end_ns IS NULL OR {field} < :end_ns)"
    )
    params: dict[str, int | None] = {
        "start_ns": int(start_ns) if start_ns is not None else None,
        "end_ns": int(end_ns) if end_ns is not None else None,
    }
    return clause, params


def select_signal_counts(*, include_avg_strength: bool = True) -> str:
    """
    Return a SELECT fragment for signal counts and optional avg strength.

    Example output (include_avg_strength=True):
    """
    base = (
        "COUNT(*) as signal_count,\n"
        "                SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_count,\n"
        "                SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_count,\n"
        "                SUM(CASE WHEN signal_type = 'HOLD' THEN 1 ELSE 0 END) as hold_count"
    )
    if include_avg_strength:
        base += ",\n                AVG(strength) as avg_strength"
    return base


def select_latency_summary(
    *,
    column: str = "inference_time_ms",
    include_avg: bool = True,
    percentiles: tuple[float, ...] = (0.50, 0.95, 0.99),
) -> str:
    """
    Return a SELECT fragment for latency summary statistics.

    Parameters
    ----------
    column : str
        Column name to compute latency summary over.
    include_avg : bool
        Whether to include the average latency line.
    percentiles : tuple[float, ...]
        Percentiles to include using PERCENTILE_CONT within group.

    Returns
    -------
    str
        SQL fragment suitable to inject into a SELECT list.
    """
    parts: list[str] = []
    if include_avg:
        parts.append(f"AVG({column}) as avg_latency_ms")
    for p in percentiles:
        # Map 0.50 -> p50, 0.95 -> p95, etc.
        pct_label = round(p * 100)
        parts.append(
            f"PERCENTILE_CONT({p:.2f}) WITHIN GROUP (ORDER BY {column}) as p{pct_label}_latency_ms"
        )
    return ", ".join(parts)


def select_min_max_ts(
    field: str = "ts_event",
    *,
    min_alias: str = "min_ts",
    max_alias: str = "max_ts",
) -> str:
    """
    Return a SELECT fragment for min/max of a timestamp-like column.

    Parameters
    ----------
    field : str
        Column name to aggregate.
    min_alias : str
        Alias for min column.
    max_alias : str
        Alias for max column.
    """
    return f"MIN({field}) as {min_alias}, MAX({field}) as {max_alias}"


def select_numeric_stats(
    *,
    column: str,
    prefix: str,
    include_avg: bool = True,
    include_stddev: bool = True,
    include_min_max: bool = True,
) -> str:
    """
    Return a SELECT fragment for basic numeric stats on a column.

    Aliases follow the pattern: avg_{prefix}, std_{prefix}, min_{prefix}, max_{prefix}.
    """
    parts: list[str] = []
    if include_avg:
        parts.append(f"AVG({column}) as avg_{prefix}")
    if include_stddev:
        parts.append(f"STDDEV({column}) as std_{prefix}")
    if include_min_max:
        parts.append(f"MIN({column}) as min_{prefix}")
        parts.append(f"MAX({column}) as max_{prefix}")
    return ", ".join(parts)


def select_avg_std(
    column: str,
    *,
    avg_alias: str = "avg_value",
    std_alias: str = "std_value",
) -> str:
    """
    Return a SELECT fragment for AVG/STDDEV of an arbitrary SQL expression.

    Parameters
    ----------
    column : str
        SQL expression or column name to aggregate, e.g., "(bid + ask) / 2".
    avg_alias : str
        Alias for the average result.
    std_alias : str
        Alias for the standard deviation result.
    """
    return f"AVG({column}) as {avg_alias}, STDDEV({column}) as {std_alias}"
