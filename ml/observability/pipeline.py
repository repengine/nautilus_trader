"""
Unified Observability Pipeline DTO builders.

This module provides typed builders that produce pandas DataFrames representing
latency watermarks, metrics collection, event correlation/lineage, and health
scores. These are used by tests and integration layers to validate observability
contracts while keeping production code decoupled from Pandera.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import pandas as pd


def build_latency_watermarks(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """
    Build a latency watermark DataFrame from stage rows.

    Each row should provide:
    - correlation_id: str
    - instrument_id: str
    - pipeline_stage: str
    - ts_stage_start: int
    - ts_stage_end: int

    The builder adds:
    - stage_latency_ns (int)
    - cumulative_latency_ns (int) in input order
    """
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df.assign(stage_latency_ns=pd.Series(dtype="int64"), cumulative_latency_ns=pd.Series(dtype="int64"))

    df = df.copy()
    df["stage_latency_ns"] = (df["ts_stage_end"].astype("int64") - df["ts_stage_start"].astype("int64")).clip(lower=0)
    df["cumulative_latency_ns"] = df["stage_latency_ns"].cumsum()
    return df


def build_metrics_collection(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """
    Build a metrics collection DataFrame with type normalization.

    Expected fields:
    - metric_name: str
    - metric_type: str (counter, histogram, gauge, summary)
    - value: float
    - timestamp: int
    - labels: dict[str, Any] | str (will be JSON-encoded string)
    """
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    df = df.copy()
    # Ensure labels is a JSON string
    def _enc(x: Any) -> str:
        return x if isinstance(x, str) else json.dumps(x or {})

    df["labels"] = df["labels"].map(_enc)
    df["value"] = df["value"].astype(float)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def build_event_correlation(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """
    Build an event correlation/lineage DataFrame.

    Fields:
    - correlation_id, event_id, parent_event_id (nullable), instrument_id
    - domain (data/features/models/strategies)
    - lineage_depth (int >= 0)
    - ts_event (int)
    - propagation_path (list[str] | str) -> JSON string
    """
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    df = df.copy()
    # Normalize propagation_path to JSON string
    def _enc_list(x: Any) -> str:
        return x if isinstance(x, str) else json.dumps(list(x or []))

    df["propagation_path"] = df["propagation_path"].map(_enc_list)
    df["lineage_depth"] = df["lineage_depth"].astype("int64").clip(lower=0)
    return df


def build_health_scores(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """
    Build a health score aggregation DataFrame.

    Fields:
    - component_id: str
    - health_score: float in [0, 1]
    - subsystem_scores: dict[str, float] | str (JSON-encoded)
    - timestamp: int
    - measurement_window_ms: int
    """
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    df = df.copy()

    def _enc_obj(x: Any) -> str:
        return x if isinstance(x, str) else json.dumps(x or {})

    df["subsystem_scores"] = df["subsystem_scores"].map(_enc_obj)
    df["health_score"] = df["health_score"].astype(float).clip(lower=0.0, upper=1.0)
    df["timestamp"] = df["timestamp"].astype("int64")
    df["measurement_window_ms"] = df["measurement_window_ms"].astype("int64")
    # Provide a default alert threshold if not present to satisfy contracts
    if "alert_threshold" not in df.columns:
        df["alert_threshold"] = 0.8
    else:
        df["alert_threshold"] = df["alert_threshold"].astype(float).clip(lower=0.0, upper=1.0)
    return df


__all__ = [
    "build_event_correlation",
    "build_health_scores",
    "build_latency_watermarks",
    "build_metrics_collection",
]


def aggregate_metrics_by_window(
    rows: Iterable[dict[str, Any]],
    *,
    window_ns: int,
) -> pd.DataFrame:
    """
    Aggregate metric rows by fixed windows while preserving totals.

    Groups by (metric_name, domain, instrument_id, window_start). The `labels`
    field is not carried through to avoid exponential cardinality; callers can
    join if needed. Returns a DataFrame with columns:
      - metric_name, domain, instrument_id, window_start, total_value, sample_count
    """
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(
            columns=[
                "metric_name",
                "domain",
                "instrument_id",
                "window_start",
                "total_value",
                "sample_count",
            ],
        )
    # Ensure types
    df = df.copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    df["value"] = df["value"].astype(float)
    # Window floor
    df["window_start"] = (df["timestamp"] // int(window_ns)) * int(window_ns)
    group_cols = ["metric_name", "domain", "instrument_id", "window_start"]
    out = (
        df.groupby(group_cols, dropna=False)["value"]
        .agg(total_value="sum", sample_count="count")
        .reset_index()
    )
    return out


def scale_health_scores(
    rows: Iterable[dict[str, Any]],
    *,
    factor: float,
) -> pd.DataFrame:
    """
    Scale health scores uniformly and clip to [0, 1].

    Returns a DataFrame with same columns as input, where `health_score` is
    multiplied by `factor` and then clipped to [0, 1].
    """
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    df = df.copy()
    df["health_score"] = (df["health_score"].astype(float) * float(factor)).clip(lower=0.0, upper=1.0)
    return df
