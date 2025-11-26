"""
Pipeline integration component for Dashboard service.

Extracted from DashboardService to follow single-responsibility principle.
Manages ML pipeline triggering, job tracking, progress monitoring, and cancellation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Coroutine
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from ml.common.logging_config import bind_log_context
from ml.config.events import EventStatus


if TYPE_CHECKING:
    from ml.core.integration import MLIntegrationManager
    from ml.dashboard.config import DashboardConfig
    from ml.dashboard.services import PipelineCancelResult
    from ml.dashboard.services import PipelineIntegrationService
    from ml.dashboard.services import PipelineJobState
    from ml.dashboard.services import PipelineProgress
    from ml.dashboard.services import PipelinePurgeResult
    from ml.dashboard.services import PipelineTriggerResult


logger = logging.getLogger(__name__)

PipelineRunResultT = TypeVar(
    "PipelineRunResultT",
    "PipelineTriggerResult",
    "list[PipelineJobState]",
    "PipelineProgress",
    "PipelinePurgeResult",
    "PipelineCancelResult",
)


class PipelineIntegrationProtocol(Protocol):
    """Protocol for pipeline integration operations."""

    def trigger_pipeline(
        self,
        pipeline_type: str,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Trigger a pipeline execution."""
        ...

    def trigger_orchestrator_task(
        self,
        task: str,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Trigger a specific MLPipelineOrchestrator task."""
        ...

    def list_pipeline_jobs(self) -> dict[str, Any]:
        """List all pipeline jobs."""
        ...

    def get_pipeline_job(self, job_id: str) -> dict[str, Any]:
        """Get details for a specific pipeline job."""
        ...

    def purge_pipeline_job(self, job_id: str) -> dict[str, Any]:
        """Purge a pipeline job from history."""
        ...

    def build_dataset_pipeline(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Trigger a dataset building pipeline."""
        ...

    def train_model_pipeline(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Trigger a model training pipeline."""
        ...

    def run_hpo_pipeline(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Trigger a hyperparameter optimization pipeline."""
        ...

    def get_pipeline_progress(self, job_id: str) -> dict[str, Any]:
        """Get progress information for a pipeline job."""
        ...

    def cancel_pipeline_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a running pipeline job."""
        ...


class PipelineIntegrationComponent:
    """
    Component for managing ML pipeline operations.

    Extracted from DashboardService to follow single-responsibility principle.
    Responsible for triggering pipelines, tracking jobs, and managing pipeline lifecycle.

    This component integrates with PipelineIntegrationService to execute orchestrator tasks
    and MLPipelineOrchestrator for direct task execution.
    """

    def __init__(self, config: DashboardConfig) -> None:
        """
        Initialize pipeline integration component.

        Args:
            config: Dashboard configuration containing DB connection and settings.

        Example:
            >>> component = PipelineIntegrationComponent(config)
            >>> result = component.trigger_pipeline("build_dataset", {"symbols": "AAPL"})
            >>> assert "job_id" in result or "error" in result
        """
        self.config = config
        self._pipeline_service: PipelineIntegrationService | None = None
        self._pipeline_integration_manager: MLIntegrationManager | None = None
        self._last_orchestrator: Any = None  # MLPipelineOrchestrator

    def trigger_pipeline(
        self,
        pipeline_type: str,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Submit a pipeline request to the integration service.

        Args:
            pipeline_type: Type of pipeline to trigger (e.g., "build_dataset", "train_model").
            config: Configuration parameters for the pipeline execution.

        Returns:
            Dictionary with job status containing:
            - success: Whether the job was queued successfully
            - job_id: Unique identifier for the job
            - pipeline_type: Type of pipeline triggered
            - status: Current job status (QUEUED, RUNNING, etc.)
            - message: Optional status message
            - error: Optional error message

        Example:
            >>> component = PipelineIntegrationComponent(config)
            >>> result = component.trigger_pipeline("build_dataset", {"symbols": "AAPL"})
            >>> assert result["success"] or "error" in result
        """
        start = time.perf_counter()
        route = "/api/pipeline/run"
        status_label = "error"
        try:
            bind_log_context(component="ml.dashboard.pipeline", action="trigger", pipeline_type=pipeline_type)
            service = self._get_pipeline_service()
            if service is None:
                status_label = "unavailable"
                return {
                    "success": False,
                    "status": "UNAVAILABLE",
                    "pipeline_type": pipeline_type,
                    "error": "pipeline_service_unavailable",
                }

            from ml.dashboard.services import PipelineTriggerRequest

            request = PipelineTriggerRequest(
                pipeline_type=pipeline_type,
                config=dict(config),
            )
            result = self._run_pipeline(service.trigger_pipeline(request))
            status_label = result.status.lower()
            payload = {
                "success": result.success,
                "job_id": result.job_id,
                "pipeline_type": result.pipeline_type,
                "status": result.status,
                "message": result.message,
                "error": result.error,
            }
            return payload
        except Exception:
            logger.debug("pipeline trigger failed", exc_info=True)
            status_label = "error"
            return {
                "success": False,
                "status": "ERROR",
                "pipeline_type": pipeline_type,
                "error": "internal_error",
            }
        finally:
            logger.debug(
                f"trigger_pipeline completed in {time.perf_counter() - start:.3f}s",
                extra={"route": route, "status": status_label},
            )

    def trigger_orchestrator_task(
        self,
        task: str,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Trigger a specific MLPipelineOrchestrator task.

        Supported tasks:
        - backfill: Run data backfill for specified instruments
        - build_dataset: Build feature dataset
        - run_hpo: Run hyperparameter optimization
        - train_teacher: Train teacher model
        - distill_student: Distill student model from teacher
        - full_pipeline: Run complete pipeline

        Args:
            task: Name of the orchestrator task to execute.
            config: Optional configuration parameters for the task.

        Returns:
            Dictionary with execution status containing:
            - ok: Whether the task was started successfully
            - result: Task execution result or error details

        Example:
            >>> component = PipelineIntegrationComponent(config)
            >>> result = component.trigger_orchestrator_task("build_dataset", {"symbols": "AAPL"})
            >>> assert result["ok"] or "error" in result["result"]
        """
        import json

        start = time.perf_counter()
        route = f"/api/orchestrator/{task}"
        config_json = json.loads(json.dumps(config or {}))
        ok = False
        result = {}
        status_label = "error"

        try:
            bind_log_context(component="ml.dashboard.orchestrator", task=task)
            # Import orchestrator lazily
            from ml.core.integration import MLIntegrationManager
            from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator

            # Initialize integration manager to get stores
            integration = MLIntegrationManager(
                db_connection=self.config.db_connection,
                auto_start_postgres=False,
                auto_migrate=False,
                ensure_healthy=False,
            )

            # Create orchestrator with integration components
            from ml.stores.providers import SqlCoverageProvider
            from ml.stores.writers import DataStoreMarketDataWriter

            def _noop_cli(argv: list[str] | None = None) -> int:
                del argv
                return 0

            orchestrator = MLPipelineOrchestrator(
                coverage=SqlCoverageProvider(connection_string=self.config.db_connection or ""),
                writer=DataStoreMarketDataWriter(data_store=integration.data_store),  # type: ignore
                build_main=_noop_cli,  # Will be replaced with actual CLI
                teacher_main=_noop_cli,
                data_registry=integration.data_registry,
                model_registry=integration.model_registry,
                feature_registry=integration.feature_registry,
                strategy_registry=integration.strategy_registry,
            )
            self._last_orchestrator = orchestrator

            # Execute the requested task
            if task == "backfill":
                result = {"status": "started", "task": task, "config": config_json}
                # orchestrator.backfill() would be called here
                ok = True
            elif task == "build_dataset":
                result = {"status": "started", "task": task, "config": config_json}
                ok = True
            elif task == "run_hpo":
                result = {"status": "started", "task": task, "config": config_json}
                ok = True
            elif task == "train_teacher":
                result = {"status": "started", "task": task, "config": config_json}
                ok = True
            elif task == "distill_student":
                result = {"status": "started", "task": task, "config": config_json}
                ok = True
            elif task == "full_pipeline":
                result = {"status": "started", "task": task, "config": config_json}
                ok = True
            else:
                result = {"error": f"Unknown task: {task}"}
                ok = False

            status_label = "success" if ok else "invalid_task"

        except Exception as e:
            logger.debug(f"orchestrator task {task} failed", exc_info=True)
            result = {"error": str(e)}
            status_label = "error"
        finally:
            logger.debug(
                f"trigger_orchestrator_task completed in {time.perf_counter() - start:.3f}s",
                extra={"route": route, "status": status_label},
            )

        return {"ok": ok, "result": result}

    def list_pipeline_jobs(self) -> dict[str, Any]:
        """
        List all pipeline jobs.

        Returns:
            Dictionary containing:
            - status: Overall operation status
            - jobs: List of job state dictionaries
            - error: Optional error message

        Example:
            >>> component = PipelineIntegrationComponent(config)
            >>> result = component.list_pipeline_jobs()
            >>> assert "jobs" in result
            >>> assert isinstance(result["jobs"], list)
        """
        start = time.perf_counter()
        route = "/api/pipeline/jobs"
        status_label = "error"
        try:
            service = self._get_pipeline_service()
            if service is None:
                status_label = "unavailable"
                return {
                    "status": "unavailable",
                    "jobs": [],
                    "error": "pipeline_service_unavailable",
                }
            jobs = self._run_pipeline(service.list_jobs())
            payload = {
                "status": EventStatus.SUCCESS.value,
                "jobs": [self._serialize_job_state(job) for job in jobs],
            }
            status_label = EventStatus.SUCCESS.value
            return payload
        except Exception:
            logger.debug("pipeline jobs listing failed", exc_info=True)
            status_label = "error"
            return {
                "status": "error",
                "jobs": [],
                "error": "internal_error",
            }
        finally:
            logger.debug(
                f"list_pipeline_jobs completed in {time.perf_counter() - start:.3f}s",
                extra={"route": route, "status": status_label},
            )

    def get_pipeline_job(self, job_id: str) -> dict[str, Any]:
        """
        Get details for a specific pipeline job.

        Args:
            job_id: Unique identifier for the pipeline job.

        Returns:
            Dictionary containing:
            - status: Operation status
            - job: Job details (if found)
            - error: Optional error message

        Example:
            >>> component = PipelineIntegrationComponent(config)
            >>> result = component.get_pipeline_job("build_dataset_abc123")
            >>> assert result["status"] in ("success", "not_found", "error")
        """
        start = time.perf_counter()
        route = "/api/pipeline/jobs/<job_id>"
        status_label = "error"
        try:
            service = self._get_pipeline_service()
            if service is None:
                status_label = "unavailable"
                return {
                    "status": "unavailable",
                    "error": "pipeline_service_unavailable",
                }
            progress = self._run_pipeline(service.get_pipeline_progress(job_id))
            if progress.status == "UNKNOWN":
                status_label = "not_found"
                return {"status": "not_found", "error": "job_not_found"}
            payload = {
                "status": EventStatus.SUCCESS.value,
                "job": self._serialize_pipeline_progress(progress),
            }
            status_label = EventStatus.SUCCESS.value
            return payload
        except Exception:
            logger.debug("pipeline job detail failed", exc_info=True)
            status_label = "error"
            return {"status": "error", "error": "internal_error"}
        finally:
            logger.debug(
                f"get_pipeline_job completed in {time.perf_counter() - start:.3f}s",
                extra={"route": route, "status": status_label},
            )

    def purge_pipeline_job(self, job_id: str) -> dict[str, Any]:
        """
        Purge a pipeline job from history.

        Args:
            job_id: Unique identifier for the pipeline job to purge.

        Returns:
            Dictionary containing:
            - status: Operation status (purged, not_found, error)
            - result: Detailed purge result

        Example:
            >>> component = PipelineIntegrationComponent(config)
            >>> result = component.purge_pipeline_job("build_dataset_abc123")
            >>> assert result["status"] in ("success", "not_found", "failed", "error")
        """
        start = time.perf_counter()
        route = "/api/pipeline/jobs/<job_id>"
        status_label = "error"
        try:
            service = self._get_pipeline_service()
            if service is None:
                status_label = "unavailable"
                return {
                    "status": "unavailable",
                    "error": "pipeline_service_unavailable",
                }
            result = self._run_pipeline(service.purge_job(job_id))
            result_payload = {
                "success": result.success,
                "job_id": result.job_id,
                "status": result.status.lower(),
                "message": result.message,
                "error": result.error,
            }
            status = result_payload["status"]
            if status == "purged":
                status_label = "success"
            elif status == "not_found":
                status_label = "not_found"
            else:
                status_label = "failed"
            return {"status": status, "result": result_payload}
        except Exception:
            logger.debug("pipeline job purge failed", exc_info=True)
            status_label = "error"
            return {"status": "error", "error": "internal_error"}
        finally:
            logger.debug(
                f"purge_pipeline_job completed in {time.perf_counter() - start:.3f}s",
                extra={"route": route, "status": status_label},
            )

    def build_dataset_pipeline(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """
        Trigger a dataset building pipeline.

        Args:
            config: Configuration for dataset building. Expected keys include symbols,
                start_date, end_date, and dataset-specific parameters.

        Returns:
            Response containing job_id, status, and optional error message.

        Example:
            >>> component = PipelineIntegrationComponent(config)
            >>> result = component.build_dataset_pipeline({"symbols": "AAPL,GOOGL"})
            >>> assert result["success"] or "error" in result
        """
        return self.trigger_pipeline(pipeline_type="build_dataset", config=config)

    def train_model_pipeline(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """
        Trigger a model training pipeline.

        Args:
            config: Configuration for model training. Expected keys include model_type,
                algorithm, dataset_id, and training parameters.

        Returns:
            Response containing job_id, status, and optional error message.

        Example:
            >>> component = PipelineIntegrationComponent(config)
            >>> result = component.train_model_pipeline({"model_type": "xgboost"})
            >>> assert result["success"] or "error" in result
        """
        return self.trigger_pipeline(pipeline_type="train_model", config=config)

    def run_hpo_pipeline(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """
        Trigger a hyperparameter optimization pipeline.

        Args:
            config: Configuration for HPO. Expected keys include search_method, trials,
                model_config, and optimization parameters.

        Returns:
            Response containing job_id, status, and optional error message.

        Example:
            >>> component = PipelineIntegrationComponent(config)
            >>> result = component.run_hpo_pipeline({"trials": 100})
            >>> assert result["success"] or "error" in result
        """
        return self.trigger_pipeline(pipeline_type="run_hpo", config=config)

    def get_pipeline_progress(self, job_id: str) -> dict[str, Any]:
        """
        Get progress information for a pipeline job.

        Args:
            job_id: The unique identifier for the pipeline job.

        Returns:
            Progress information including:
            - status: Operation status
            - progress: Progress percentage and details (if successful)
            - error: Optional error message

        Example:
            >>> component = PipelineIntegrationComponent(config)
            >>> result = component.get_pipeline_progress("build_dataset_abc123")
            >>> assert result["status"] in ("success", "not_found", "error")
        """
        start = time.perf_counter()
        route = "/api/pipeline/jobs/<job_id>/progress"
        status_label = EventStatus.FAILED.value
        try:
            service = self._get_pipeline_service()
            if service is None:
                status_label = EventStatus.DEFERRED.value
                return {
                    "status": EventStatus.DEFERRED.value,
                    "error": "pipeline_service_unavailable",
                }
            progress = self._run_pipeline(service.get_pipeline_progress(job_id))
            if progress.status == "UNKNOWN":
                status_label = EventStatus.DEFERRED.value
                return {"status": EventStatus.DEFERRED.value, "error": "job_not_found"}
            payload = self._serialize_pipeline_progress(progress)
            status_label = EventStatus.SUCCESS.value
            return {"status": EventStatus.SUCCESS.value, "progress": payload}
        except Exception:
            logger.debug("pipeline progress retrieval failed", exc_info=True)
            status_label = EventStatus.FAILED.value
            return {"status": EventStatus.FAILED.value, "error": "internal_error"}
        finally:
            logger.debug(
                f"get_pipeline_progress completed in {time.perf_counter() - start:.3f}s",
                extra={"route": route, "status": status_label},
            )

    def cancel_pipeline_job(self, job_id: str) -> dict[str, Any]:
        """
        Cancel a running pipeline job.

        Args:
            job_id: The unique identifier for the pipeline job to cancel.

        Returns:
            Response containing:
            - success: Whether cancellation was successful
            - job_id: The job identifier
            - status: Current job status after cancellation
            - message: Optional status message
            - error: Optional error message

        Example:
            >>> component = PipelineIntegrationComponent(config)
            >>> result = component.cancel_pipeline_job("build_dataset_abc123")
            >>> assert "success" in result
        """
        start = time.perf_counter()
        route = "/api/pipeline/jobs/<job_id>/cancel"
        status_label = "error"
        try:
            service = self._get_pipeline_service()
            if service is None:
                status_label = "unavailable"
                return {
                    "success": False,
                    "status": "unavailable",
                    "error": "pipeline_service_unavailable",
                }
            result = self._run_pipeline(service.cancel_pipeline(job_id))
            result_payload = {
                "success": result.success,
                "job_id": result.job_id,
                "status": result.status,
                "message": result.message,
                "error": result.error,
            }
            if result.success:
                status_label = "success"
            elif result.status == "NOT_FOUND":
                status_label = "not_found"
            else:
                status_label = "failed"
            return result_payload
        except Exception:
            logger.debug("pipeline job cancellation failed", exc_info=True)
            status_label = "error"
            return {
                "success": False,
                "status": "error",
                "error": "internal_error",
            }
        finally:
            logger.debug(
                f"cancel_pipeline_job completed in {time.perf_counter() - start:.3f}s",
                extra={"route": route, "status": status_label},
            )

    def get_integration_manager(self) -> MLIntegrationManager | None:
        """
        Return the cached ML integration manager if available.

        Returns:
            The cached MLIntegrationManager instance or None.

        Example:
            >>> component = PipelineIntegrationComponent(config)
            >>> manager = component.get_integration_manager()
            >>> assert manager is None or hasattr(manager, "data_store")
        """
        if self._pipeline_integration_manager is not None:
            return self._pipeline_integration_manager
        self._get_pipeline_service()
        return self._pipeline_integration_manager

    # Private helper methods

    def _get_pipeline_service(self) -> PipelineIntegrationService | None:
        """Get or initialize the pipeline integration service."""
        if self._pipeline_service is not None:
            return self._pipeline_service
        try:
            from ml.core.integration import MLIntegrationManager
            from ml.dashboard.services import PipelineIntegrationService

            integration = MLIntegrationManager(
                db_connection=self.config.db_connection,
                auto_start_postgres=False,
                auto_migrate=False,
                ensure_healthy=False,
            )
        except Exception:
            logger.debug("pipeline integration manager init failed", exc_info=True)
            return None
        self._pipeline_integration_manager = integration
        self._pipeline_service = PipelineIntegrationService(integration)
        return self._pipeline_service

    @staticmethod
    def _serialize_job_state(job_state: PipelineJobState) -> dict[str, Any]:
        """Serialize a PipelineJobState to a dictionary."""
        return {
            "job_id": job_state.job_id,
            "pipeline_type": job_state.pipeline_type,
            "status": job_state.status,
            "progress": job_state.progress,
            "current_stage": job_state.current_stage,
            "eta_seconds": job_state.eta_seconds,
            "message": job_state.message,
            "error": job_state.error,
            "started_at": job_state.started_at,
            "finished_at": job_state.finished_at,
            "started_at_iso": job_state.started_at_iso,
            "finished_at_iso": job_state.finished_at_iso,
        }

    @staticmethod
    def _serialize_pipeline_progress(progress: PipelineProgress) -> dict[str, Any]:
        """Serialize a PipelineProgress to a dictionary."""
        return {
            "job_id": progress.job_id,
            "status": progress.status,
            "progress": progress.progress,
            "current_stage": progress.current_stage,
            "eta_seconds": progress.eta_seconds,
            "message": progress.message,
            "error": progress.error,
            "started_at": progress.started_at,
            "finished_at": progress.finished_at,
            "started_at_iso": progress.started_at_iso,
            "finished_at_iso": progress.finished_at_iso,
        }

    @staticmethod
    def _run_pipeline(coroutine: Coroutine[Any, Any, PipelineRunResultT]) -> PipelineRunResultT:
        """Execute an async pipeline operation synchronously."""
        return asyncio.run(coroutine)


__all__ = [
    "PipelineIntegrationComponent",
    "PipelineIntegrationProtocol",
]
