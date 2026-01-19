"""Dataset planning service for streaming TFT training."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Final
from uuid import uuid4

from ml.config.events import EventStatus
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.training.event_driven.guardrails import enforce_dataset_guardrails
from ml.training.event_driven.services import DatasetPlanEvent
from ml.training.event_driven.services import DatasetPlanner
from ml.training.event_driven.services import DatasetPlanRequest
from ml.training.teacher import streaming_loader as stream
from ml.training.teacher.streaming_loader import TFTStreamingConfig


logger = logging.getLogger(__name__)

_CAP_KEYS: Final[tuple[str, ...]] = ("max_shards", "max_total_rows", "max_total_sequences")


def _combine_limit(service_value: int | None, request_value: int | None) -> int | None:
    if service_value is None:
        return request_value
    if request_value is None:
        return service_value
    return min(service_value, request_value)


def _apply_service_caps(
    service_config: DatasetServiceConfig,
    request_config: TFTStreamingConfig,
) -> TFTStreamingConfig:
    merged = replace(
        request_config,
        max_total_rows=_combine_limit(service_config.max_total_rows, request_config.max_total_rows),
        max_total_sequences=_combine_limit(
            service_config.max_total_sequences,
            request_config.max_total_sequences,
        ),
        max_shards=_combine_limit(service_config.max_shards, request_config.max_shards),
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


def _ensure_target_in_numeric(
    numeric_columns: tuple[str, ...],
    target_col: str,
) -> tuple[str, ...]:
    if target_col in numeric_columns:
        return numeric_columns
    ordered = list(numeric_columns)
    ordered.append(target_col)
    return tuple(dict.fromkeys(ordered))


class StreamingDatasetPlanner(DatasetPlanner):
    """Concrete dataset planner using parquet metadata scans."""

    def __init__(self, config: DatasetServiceConfig) -> None:
        super().__init__(config)

    def plan(self, request: DatasetPlanRequest) -> DatasetPlanEvent:
        parquet_path = self._resolve_parquet_path(request)
        numeric_columns = _ensure_target_in_numeric(
            request.numeric_columns,
            request.streaming_config.target_col,
        )

        metadata = stream.collect_streaming_metadata(
            parquet_path,
            feature_names=request.feature_names,
            categorical_columns=request.categorical_columns,
            numeric_columns=numeric_columns,
            target_col=request.streaming_config.target_col,
            group_id_col=request.streaming_config.group_id_col,
            time_index_col=request.streaming_config.time_idx_col,
            shard_row_budget=int(self.config.shard_row_budget),
            phase_one_signals=request.phase_one_signals,
        )
        planner_config = _apply_service_caps(self.config, request.streaming_config)
        limited_metadata, limits = stream.apply_streaming_limits(metadata, planner_config)
        limited_summary = stream.summarize_metadata(limited_metadata)

        status = EventStatus.SUCCESS if limited_metadata.shard_indices else EventStatus.DEFERRED
        plan_id = f"{request.dataset_id}-{uuid4().hex[:12]}"
        caps: dict[str, float | int | None] = {
            "shard_row_budget": self.config.shard_row_budget,
            "max_shards": planner_config.max_shards,
            "max_total_rows": planner_config.max_total_rows,
            "max_total_sequences": planner_config.max_total_sequences,
        }
        if planner_config.seed is not None:
            caps["dataset_seed"] = int(planner_config.seed)

        logger.info(
            "dataset plan ready",
            extra={
                "plan_id": plan_id,
                "dataset_id": request.dataset_id,
                "status": status.value,
                "selected_shards": limited_summary.total_shards,
                "selected_rows": limited_summary.total_rows,
                "skipped_shards": limits.skipped_shards,
                "skipped_rows": limits.skipped_rows,
            },
        )
        plan_event = DatasetPlanEvent(
            plan_id=plan_id,
            dataset_id=request.dataset_id,
            parquet_path=parquet_path,
            metadata=limited_metadata,
            metadata_summary=limited_summary,
            limits=limits,
            streaming_config=planner_config,
            caps=caps,
            phase_one_signals=limited_metadata.phase_one_signals,
            status=status,
        )
        enforce_dataset_guardrails(
            plan_event,
            request=request,
            service_config=self.config,
        )
        return plan_event

    def _resolve_parquet_path(self, request: DatasetPlanRequest) -> Path:
        if request.parquet_path is not None:
            parquet_path = Path(request.parquet_path)
        else:
            parquet_path = Path(self.config.parquet_root) / request.dataset_id
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet dataset not found at {parquet_path}")
        return parquet_path


__all__ = [
    "StreamingDatasetPlanner",
]
