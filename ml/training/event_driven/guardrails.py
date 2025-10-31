"""Guardrail checks for streaming dataset plans."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from ml.common.metrics_bootstrap import get_counter
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.features.validation import validate_known_future_effective_times
from ml.training.event_driven.services import DatasetPlanEvent
from ml.training.event_driven.services import DatasetPlanRequest
from ml.training.teacher.streaming_loader import TFTStreamingMetadata


logger = logging.getLogger(__name__)

_GUARDRAIL_COUNTER = get_counter(
    "ml_tft_streaming_dataset_guardrails_total",
    "Dataset guardrail outcomes grouped by dimension and result.",
    labelnames=("dimension", "outcome"),
)


class DatasetGuardrailError(RuntimeError):
    """Raised when a dataset plan violates guardrail constraints."""


def enforce_dataset_guardrails(
    plan_event: DatasetPlanEvent,
    *,
    request: DatasetPlanRequest,
    service_config: DatasetServiceConfig,
) -> DatasetPlanEvent:
    """
    Run dataset-level guardrails and return the (possibly updated) plan event.

    Args:
        plan_event: Dataset plan to validate.
        request: Original planning request (used for schema hints).
        service_config: Configuration supplying guardrail parameters.

    Returns:
        DatasetPlanEvent: Potentially updated event (status adjustments).

    Raises:
        DatasetGuardrailError: When mandatory guardrails fail.
    """
    dataset_id = plan_event.dataset_id
    target_col = plan_event.streaming_config.target_col
    positive_rate = _positive_rate(plan_event.metadata, target_col)
    _validate_positive_rate(
        dataset_id=dataset_id,
        plan_id=plan_event.plan_id,
        positive_rate=positive_rate,
        config=service_config,
    )
    _validate_schema(
        dataset_id=dataset_id,
        plan_id=plan_event.plan_id,
        metadata=plan_event.metadata,
        request=request,
        config=service_config,
    )
    _validate_known_future_pairs(
        parquet_path=plan_event.parquet_path,
        dataset_id=dataset_id,
        plan_id=plan_event.plan_id,
        config=service_config,
    )
    # Guardrails may adjust event status in the future; keep behaviour pluggable
    return plan_event


def _positive_rate(metadata: TFTStreamingMetadata, target_col: str) -> float | None:
    stats = metadata.numeric_stats.get(target_col)
    if stats is None or stats.count <= 0:
        return None
    return float(stats.mean)


def _validate_positive_rate(
    *,
    dataset_id: str,
    plan_id: str,
    positive_rate: float | None,
    config: DatasetServiceConfig,
) -> None:
    if positive_rate is None:
        logger.warning(
            "streaming dataset guardrail skipped positive rate check (missing stats)",
            extra={"dataset_id": dataset_id, "plan_id": plan_id},
        )
        _GUARDRAIL_COUNTER.labels("positive_rate_threshold", "skipped").inc()
        return

    if (
        config.min_positive_rate is not None
        and positive_rate < float(config.min_positive_rate)
    ):
        _GUARDRAIL_COUNTER.labels("positive_rate_threshold", "error").inc()
        msg = (
            f"target positive rate {positive_rate:.6f} below configured minimum "
            f"{float(config.min_positive_rate):.6f}"
        )
        logger.error(
            "streaming dataset guardrail failed",
            extra={
                "dataset_id": dataset_id,
                "plan_id": plan_id,
                "constraint": "positive_rate_min",
                "positive_rate": positive_rate,
                "threshold": float(config.min_positive_rate),
            },
        )
        raise DatasetGuardrailError(msg)
    if (
        config.max_positive_rate is not None
        and positive_rate > float(config.max_positive_rate)
    ):
        _GUARDRAIL_COUNTER.labels("positive_rate_threshold", "error").inc()
        msg = (
            f"target positive rate {positive_rate:.6f} above configured maximum "
            f"{float(config.max_positive_rate):.6f}"
        )
        logger.error(
            "streaming dataset guardrail failed",
            extra={
                "dataset_id": dataset_id,
                "plan_id": plan_id,
                "constraint": "positive_rate_max",
                "positive_rate": positive_rate,
                "threshold": float(config.max_positive_rate),
            },
        )
        raise DatasetGuardrailError(msg)
    _GUARDRAIL_COUNTER.labels("positive_rate_threshold", "ok").inc()

    baseline = config.positive_rate_baseline
    tolerance = float(config.positive_rate_drift_tolerance)
    if baseline is None:
        return
    drift = abs(positive_rate - float(baseline))
    if drift > tolerance:
        _GUARDRAIL_COUNTER.labels("positive_rate_drift", "alert").inc()
        logger.warning(
            "streaming dataset positive rate drift detected",
            extra={
                "dataset_id": dataset_id,
                "plan_id": plan_id,
                "positive_rate": positive_rate,
                "baseline": float(baseline),
                "tolerance": tolerance,
                "drift": drift,
            },
        )
        return
    _GUARDRAIL_COUNTER.labels("positive_rate_drift", "ok").inc()


def _validate_schema(
    *,
    dataset_id: str,
    plan_id: str,
    metadata: TFTStreamingMetadata,
    request: DatasetPlanRequest,
    config: DatasetServiceConfig,
) -> None:
    reference = tuple(config.schema_reference_columns)
    if not reference:
        return

    available = _available_feature_columns(metadata)
    missing = tuple(sorted(column for column in reference if column not in available))
    if missing:
        _GUARDRAIL_COUNTER.labels("schema_required", "error").inc()
        logger.error(
            "streaming dataset missing required features",
            extra={
                "dataset_id": dataset_id,
                "plan_id": plan_id,
                "required": reference,
                "missing": missing,
                "requested_features": request.feature_names,
            },
        )
        raise DatasetGuardrailError(
            f"Dataset {dataset_id} missing required features: {missing}",
        )
    _GUARDRAIL_COUNTER.labels("schema_required", "ok").inc()

    unexpected = tuple(sorted(column for column in available if column not in reference))
    if unexpected:
        outcome = "alert" if config.schema_alert_on_unexpected else "ok"
        _GUARDRAIL_COUNTER.labels("schema_unexpected", outcome).inc()
        if config.schema_alert_on_unexpected:
            logger.warning(
                "streaming dataset schema drift detected",
                extra={
                    "dataset_id": dataset_id,
                    "plan_id": plan_id,
                    "expected": reference,
                    "unexpected": unexpected,
                    "requested_features": request.feature_names,
                },
            )
    else:
        _GUARDRAIL_COUNTER.labels("schema_unexpected", "ok").inc()


def _available_feature_columns(metadata: TFTStreamingMetadata) -> set[str]:
    numeric_columns = {column for column in metadata.numeric_stats}
    categorical_columns = {column for column in metadata.categorical_vocab}
    return numeric_columns | categorical_columns


def _validate_known_future_pairs(
    *,
    parquet_path: Path,
    dataset_id: str,
    plan_id: str,
    config: DatasetServiceConfig,
) -> None:
    pairs = _resolve_known_future_pairs(config.known_future_pairs)
    if not pairs:
        return

    try:
        import pyarrow.dataset as pa_dataset
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning(
            "pyarrow dataset unavailable; skipping known-future guardrail",
            extra={"dataset_id": dataset_id, "plan_id": plan_id},
            exc_info=True,
        )
        _GUARDRAIL_COUNTER.labels("known_future", "skipped").inc()
        return

    dataset = pa_dataset.dataset(str(parquet_path), format="parquet")
    limit = config.known_future_sample_rows

    for evaluation_column, effective_column in pairs:
        rows_checked = 0
        try:
            scanner = dataset.scanner(columns=[evaluation_column, effective_column])
        except Exception as exc:  # pragma: no cover - defensive guard
            _GUARDRAIL_COUNTER.labels("known_future", "error").inc()
            logger.error(
                "known-future guardrail columns missing",
                extra={
                    "dataset_id": dataset_id,
                    "plan_id": plan_id,
                    "evaluation_column": evaluation_column,
                    "effective_column": effective_column,
                },
                exc_info=True,
            )
            raise DatasetGuardrailError(str(exc)) from exc

        for batch in scanner.to_batches():
            payload = batch.to_pydict()
            evaluation_values = payload.get(evaluation_column, [])
            effective_values = payload.get(effective_column, [])
            if not evaluation_values or not effective_values:
                continue

            if limit is not None:
                remaining = int(limit) - rows_checked
                if remaining <= 0:
                    break
                if remaining < len(evaluation_values):
                    evaluation_values = evaluation_values[:remaining]
                    effective_values = effective_values[:remaining]
            try:
                validate_known_future_effective_times(
                    evaluation_series=evaluation_values,
                    effective_series=effective_values,
                    context=f"{dataset_id}:{evaluation_column}->{effective_column}",
                )
            except Exception as exc:
                _GUARDRAIL_COUNTER.labels("known_future", "error").inc()
                logger.error(
                    "known-future guardrail failed",
                    extra={
                        "dataset_id": dataset_id,
                        "plan_id": plan_id,
                        "evaluation_column": evaluation_column,
                        "effective_column": effective_column,
                    },
                    exc_info=True,
                )
                raise DatasetGuardrailError(str(exc)) from exc
            rows_checked += len(evaluation_values)
            if limit is not None and rows_checked >= int(limit):
                break
        if rows_checked == 0:
            logger.warning(
                "known-future guardrail scanned zero rows",
                extra={
                    "dataset_id": dataset_id,
                    "plan_id": plan_id,
                    "evaluation_column": evaluation_column,
                    "effective_column": effective_column,
                },
            )
            _GUARDRAIL_COUNTER.labels("known_future", "skipped").inc()
        else:
            _GUARDRAIL_COUNTER.labels("known_future", "ok").inc()


def _resolve_known_future_pairs(pairs: Sequence[str]) -> tuple[tuple[str, str], ...]:
    resolved: list[tuple[str, str]] = []
    for item in pairs:
        cleaned = item.strip()
        if not cleaned or ":" not in cleaned:
            continue
        evaluation, effective = cleaned.split(":", 1)
        evaluation = evaluation.strip()
        effective = effective.strip()
        if evaluation and effective:
            resolved.append((evaluation, effective))
    return tuple(resolved)


__all__ = [
    "DatasetGuardrailError",
    "enforce_dataset_guardrails",
]
