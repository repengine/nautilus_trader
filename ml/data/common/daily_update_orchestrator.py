"""
Daily update orchestrator component extracted from DataScheduler.

This component handles the orchestration of the complete daily update pipeline:
- Coordinate data collection, feature computation, and cleanup stages
- Track pipeline metrics and timing
- Handle failures with proper status recording

Extracted from legacy DataScheduler (lines 502-547):
- run_daily_update() - Orchestrate the complete daily update process

"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Protocol

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram


logger = logging.getLogger(__name__)


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

# Import module-level metrics from scheduler module for compatibility
# These are created at module load and must be the same instances
try:
    from ml.data.scheduler import pipeline_runs_total
    from ml.data.scheduler import pipeline_stage_latency
except ImportError:
    # Fallback for isolated testing - create local metrics
    pipeline_stage_latency = get_histogram(
        "nautilus_ml_pipeline_stage_latency_seconds",
        "Pipeline stage execution latency in seconds",
        ["stage"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    pipeline_runs_total = get_counter(
        "nautilus_ml_pipeline_runs_total",
        "Total pipeline runs",
        ["status"],
    )


@contextmanager
def track_pipeline_stage(stage: str) -> Generator[None, None, None]:
    """
    Context manager to track pipeline stage execution time.

    Records the duration of each pipeline stage in the pipeline_stage_latency
    histogram metric with the stage name as a label.

    Args:
        stage: Name of the pipeline stage to track (e.g., "data_collection",
            "feature_computation", "data_cleanup").

    Yields:
        None - Used as a context manager.

    Example:
        >>> with track_pipeline_stage("data_collection"):
        ...     # Perform data collection
        ...     pass

    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        pipeline_stage_latency.labels(stage=stage).observe(duration)


class DailyUpdateOrchestratorProtocol(Protocol):
    """
    Protocol for daily update orchestration operations.

    This protocol defines the contract for daily update orchestrator components,
    enabling duck typing for testing and alternative implementations.

    Methods
    -------
    run_daily_update
        Run the complete daily update process.

    """

    def run_daily_update(
        self,
        use_orchestrator: bool,
        feature_engineer: Any | None,
        collect_latest_data_fn: Callable[[], None],
        collect_via_orchestrator_fn: Callable[[], None],
        compute_features_fn: Callable[[], None],
        clean_old_data_fn: Callable[[], None],
    ) -> None:
        """
        Run the complete daily update process.

        Args:
            use_orchestrator: Whether to use orchestrator mode for data collection.
            feature_engineer: Feature engineer instance (if any).
            collect_latest_data_fn: Function to collect latest data directly.
            collect_via_orchestrator_fn: Function to collect via orchestrator.
            compute_features_fn: Function to compute features.
            clean_old_data_fn: Function to clean old data.

        Raises:
            Exception: Re-raised after recording failure metric.

        """
        ...


class DailyUpdateOrchestratorComponent:
    """
    Component for orchestrating the daily update pipeline.

    This component extracts daily update orchestration responsibilities from
    DataScheduler, providing a focused method for running the complete daily
    update process including:
    - Data collection (via direct API or orchestrator)
    - Feature computation (if feature engineer is configured)
    - Data cleanup (retention policy enforcement)

    All pipeline operations record Prometheus metrics for observability:
    - pipeline_runs_total: Counter tracking total runs by status
    - pipeline_stage_latency: Histogram tracking stage durations

    Example:
        >>> from ml.data.common.daily_update_orchestrator import (
        ...     DailyUpdateOrchestratorComponent,
        ... )
        >>> component = DailyUpdateOrchestratorComponent()
        >>> component.run_daily_update(
        ...     use_orchestrator=False,
        ...     feature_engineer=my_feature_engineer,
        ...     collect_latest_data_fn=my_collect_fn,
        ...     collect_via_orchestrator_fn=my_orchestrator_fn,
        ...     compute_features_fn=my_compute_fn,
        ...     clean_old_data_fn=my_cleanup_fn,
        ... )

    """

    def run_daily_update(
        self,
        use_orchestrator: bool,
        feature_engineer: Any | None,
        collect_latest_data_fn: Callable[[], None],
        collect_via_orchestrator_fn: Callable[[], None],
        compute_features_fn: Callable[[], None],
        clean_old_data_fn: Callable[[], None],
    ) -> None:
        """
        Run the complete daily update process.

        Orchestrates the three-stage daily pipeline:
        1. Data Collection: Collects latest data from Databento (via direct API
           or orchestrator mode depending on use_orchestrator flag)
        2. Feature Computation: Computes features if feature_engineer is configured
        3. Data Cleanup: Cleans old data based on retention policy

        Each stage is tracked with Prometheus metrics via track_pipeline_stage().
        On completion, records overall pipeline duration and run count.
        On failure, records failure status before re-raising.

        Args:
            use_orchestrator: Whether to use orchestrator mode for data collection.
                If True, uses collect_via_orchestrator_fn; otherwise uses
                collect_latest_data_fn.
            feature_engineer: Feature engineer instance (if any). If not None,
                feature computation stage will be executed.
            collect_latest_data_fn: Function to collect latest data directly
                from the Databento API.
            collect_via_orchestrator_fn: Function to collect data via the
                IngestionOrchestrator.
            compute_features_fn: Function to compute features for newly
                collected data.
            clean_old_data_fn: Function to clean old data based on
                retention policy.

        Raises:
            Exception: Re-raised after recording failure metric. The pipeline
                status is set to "failure" and metrics are recorded before
                propagating the exception.

        Example:
            >>> component = DailyUpdateOrchestratorComponent()
            >>> # With orchestrator mode
            >>> component.run_daily_update(
            ...     use_orchestrator=True,
            ...     feature_engineer=my_feature_engineer,
            ...     collect_latest_data_fn=lambda: None,
            ...     collect_via_orchestrator_fn=orchestrator.collect,
            ...     compute_features_fn=my_feature_fn,
            ...     clean_old_data_fn=my_cleanup_fn,
            ... )
            >>> # Without feature computation
            >>> component.run_daily_update(
            ...     use_orchestrator=False,
            ...     feature_engineer=None,  # Skips feature computation
            ...     collect_latest_data_fn=direct_collect,
            ...     collect_via_orchestrator_fn=lambda: None,
            ...     compute_features_fn=lambda: None,
            ...     clean_old_data_fn=cleanup_fn,
            ... )

        """
        logger.info("Starting daily data update...")
        pipeline_start_time = time.perf_counter()
        pipeline_status = "success"

        try:
            # Step 1: Collect latest data
            with track_pipeline_stage("data_collection"):
                if use_orchestrator:
                    collect_via_orchestrator_fn()
                else:
                    collect_latest_data_fn()

            # Step 2: Compute features if configured
            if feature_engineer is not None:
                with track_pipeline_stage("feature_computation"):
                    compute_features_fn()

            # Step 3: Clean old data
            with track_pipeline_stage("data_cleanup"):
                clean_old_data_fn()

            logger.info("Daily data update completed successfully")

        except Exception:
            pipeline_status = "failure"
            logger.error(
                "Daily data update failed",
                exc_info=True,
            )
            raise
        finally:
            # Record overall pipeline metrics
            pipeline_duration = time.perf_counter() - pipeline_start_time
            pipeline_runs_total.labels(status=pipeline_status).inc()
            pipeline_stage_latency.labels(stage="complete_pipeline").observe(
                pipeline_duration
            )


__all__ = [
    "DailyUpdateOrchestratorComponent",
    "DailyUpdateOrchestratorProtocol",
    "track_pipeline_stage",
]
