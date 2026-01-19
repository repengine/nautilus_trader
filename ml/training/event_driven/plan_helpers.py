"""Shared helpers for streaming dataset plan construction."""

from __future__ import annotations

from dataclasses import replace

from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.training.teacher.streaming_loader import TFTStreamingConfig


def combine_limit(service_value: int | None, request_value: int | None) -> int | None:
    """Return the tighter of two optional limit values."""
    if service_value is None:
        return request_value
    if request_value is None:
        return service_value
    return min(service_value, request_value)


def apply_service_caps(
    service_config: DatasetServiceConfig,
    request_config: TFTStreamingConfig,
) -> TFTStreamingConfig:
    """
    Apply service-level caps and feature toggles to the request config.

    Args:
        service_config: Dataset service configuration with global caps.
        request_config: Requested streaming configuration to adjust.

    Returns:
        TFTStreamingConfig: Config with service caps applied.
    """
    merged = replace(
        request_config,
        max_total_rows=combine_limit(service_config.max_total_rows, request_config.max_total_rows),
        max_total_sequences=combine_limit(
            service_config.max_total_sequences,
            request_config.max_total_sequences,
        ),
        max_shards=combine_limit(service_config.max_shards, request_config.max_shards),
    )
    if service_config.include_macro:
        merged = replace(merged, include_macro=True)
    if service_config.include_calendar:
        merged = replace(merged, include_calendar=True)
    if service_config.include_events:
        merged = replace(merged, include_events=True)
    if service_config.include_earnings:
        merged = replace(merged, include_earnings=True)
    if service_config.include_micro:
        merged = replace(merged, include_micro=True)
    if service_config.include_l2:
        merged = replace(merged, include_l2=True)
    if service_config.include_macro_revisions:
        merged = replace(merged, include_macro_revisions=True)
    if service_config.include_macro_deltas:
        merged = replace(merged, include_macro_deltas=True)
    if service_config.include_calendar_lags:
        merged = replace(merged, include_calendar_lags=True)
    if service_config.include_clustering_tags:
        merged = replace(merged, include_clustering_tags=True)
    if service_config.include_context_features:
        merged = replace(merged, include_context_features=True)
    if merged.include_l2 and not merged.include_micro:
        merged = replace(merged, include_micro=True)
    return merged


def ensure_target_in_numeric(
    numeric_columns: tuple[str, ...],
    target_col: str,
) -> tuple[str, ...]:
    """
    Ensure the target column is included in numeric feature columns.

    Args:
        numeric_columns: Numeric feature columns from the request.
        target_col: Target column that must be included.

    Returns:
        tuple[str, ...]: Numeric columns with the target column appended when missing.
    """
    if target_col in numeric_columns:
        return numeric_columns
    ordered = list(numeric_columns)
    ordered.append(target_col)
    return tuple(dict.fromkeys(ordered))


__all__ = [
    "apply_service_caps",
    "combine_limit",
    "ensure_target_in_numeric",
]
