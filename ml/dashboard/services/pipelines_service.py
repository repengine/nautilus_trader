"""Typed integration service bridging the dashboard to pipeline orchestration."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, get_args, get_origin

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.dashboard.services.base_service import BaseIntegrationService


if TYPE_CHECKING:
    from ml.core.integration import MLIntegrationManager
    from ml.orchestration.config_loader import IngestionStageConfig
    from ml.orchestration.config_loader import OrchestratorRunConfig
    from ml.orchestration.config_loader import Stage
    from ml.orchestration.config_loader import TrainingStageConfig
    from ml.orchestration.config_types import AutoFillUniverseConfig
    from ml.orchestration.config_types import DatasetBuildConfig
    from ml.orchestration.config_types import HPOConfig
    from ml.orchestration.config_types import IntegrationConfig
    from ml.orchestration.config_types import OrchestratorConfig
    from ml.orchestration.config_types import PromotionsConfig
    from ml.orchestration.config_types import StudentDistillConfig
    from ml.orchestration.config_types import TeacherTrainConfig
    from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator


logger = logging.getLogger(__name__)

pipeline_jobs_total = get_counter(
    "ml_dashboard_pipeline_jobs_total",
    "Total pipeline jobs launched via dashboard",
    labelnames=["pipeline_type", "status"],
)

pipeline_job_latency = get_histogram(
    "ml_dashboard_pipeline_job_latency_seconds",
    "Pipeline job execution latency",
    labelnames=["pipeline_type"],
    buckets=(1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)

pipeline_job_store_failures_total = get_counter(
    "ml_dashboard_pipeline_job_store_failures_total",
    "Pipeline job store operation failures",
    labelnames=["operation"],
)


@dataclass(slots=True)
class PipelineTriggerRequest:
    """Request payload used to trigger a pipeline execution."""

    pipeline_type: str
    config: Mapping[str, Any]


@dataclass(slots=True)
class PipelineTriggerResult:
    """Response returned when a pipeline job is enqueued."""

    success: bool
    job_id: str
    pipeline_type: str
    status: str
    message: str | None = None
    error: str | None = None


@dataclass(slots=True)
class PipelineProgress:
    """
    Typed progress snapshot for pipeline tracking.

    Example
    -------
    >>> PipelineProgress(
    ...     job_id="training_1234abcd",
    ...     status="RUNNING",
    ...     progress=0.4,
    ...     current_stage="feature_ingestion",
    ...     eta_seconds=120,
    ...     message="Ingesting features",
    ...     error=None,
    ...     started_at=1_000.0,
    ...     finished_at=None,
    ...     started_at_iso="2025-01-01T00:00:00+00:00",
    ...     finished_at_iso=None,
    ... )
    PipelineProgress(job_id='training_1234abcd', status='RUNNING', progress=0.4, current_stage='feature_ingestion', eta_seconds=120, message='Ingesting features', error=None, started_at=1000.0, finished_at=None, started_at_iso='2025-01-01T00:00:00+00:00', finished_at_iso=None)
    """

    job_id: str
    status: str
    progress: float
    current_stage: str
    eta_seconds: int
    message: str | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    started_at_iso: str | None = None
    finished_at_iso: str | None = None


@dataclass(slots=True)
class PipelineJobState:
    """In-memory state for an active or completed pipeline job."""

    job_id: str
    pipeline_type: str
    status: str
    progress: float = 0.0
    current_stage: str = "QUEUED"
    eta_seconds: int = 0
    message: str | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    started_at_iso: str | None = None
    finished_at_iso: str | None = None


@dataclass(slots=True)
class PipelineCancelResult:
    """Outcome of cancelling a pipeline job."""

    success: bool
    job_id: str
    status: str
    message: str | None = None
    error: str | None = None


@dataclass(slots=True)
class PipelinePurgeResult:
    """
    Outcome of purging a pipeline job history record.

    Example
    -------
    >>> PipelinePurgeResult(success=True, job_id="training_deadbeef", status="PURGED")
    PipelinePurgeResult(success=True, job_id='training_deadbeef', status='PURGED', message=None, error=None)
    """

    success: bool
    job_id: str
    status: str
    message: str | None = None
    error: str | None = None


class PipelineJobStoreProtocol(Protocol):
    """Persistence contract for pipeline job state history."""

    def save(self, job_state: PipelineJobState) -> None: ...

    def delete(self, job_id: str) -> None: ...

    def get(self, job_id: str) -> PipelineJobState | None: ...

    def list_jobs(self) -> list[PipelineJobState]: ...


T = TypeVar("T")


class PipelineIntegrationService(BaseIntegrationService):
    """Integration facade that submits orchestration jobs on behalf of the dashboard."""

    _jobs: dict[str, PipelineJobState]
    _job_tasks: dict[str, asyncio.Task[None]]
    _job_store: PipelineJobStoreProtocol | None

    def __init__(
        self,
        integration_manager: MLIntegrationManager | None,
    ) -> None:
        super().__init__(integration_manager)
        self._jobs = {}
        self._job_tasks = {}
        self._job_store = None
        self._initialise_job_store(integration_manager)

    def get_service_name(self) -> str:
        return "pipeline_integration"

    async def health_check(self) -> dict[str, Any]:
        orchestrator = self._resolve_orchestrator()
        if orchestrator is None:
            return {"healthy": False, "reason": "Orchestrator not configured"}
        return {"healthy": True, "orchestrator": orchestrator.__class__.__name__}

    async def trigger_pipeline(self, request: PipelineTriggerRequest) -> PipelineTriggerResult:
        """
        Validate the request, submit a pipeline run, and return the queued job id.

        Parameters
        ----------
        request : PipelineTriggerRequest
            Typed request containing the pipeline type identifier and nested
            configuration mapping.
        """
        self._track_operation(operation=f"trigger_{request.pipeline_type}", status="started")

        orchestrator = self._resolve_orchestrator()
        if orchestrator is None:
            pipeline_jobs_total.labels(pipeline_type=request.pipeline_type, status="unavailable").inc()
            return PipelineTriggerResult(
                success=False,
                job_id="",
                pipeline_type=request.pipeline_type,
                status="UNAVAILABLE",
                error="Pipeline orchestrator is not configured",
            )

        try:
            run_config = self._build_run_config(
                pipeline_type=request.pipeline_type,
                payload=request.config,
            )
        except ValueError as exc:
            pipeline_jobs_total.labels(pipeline_type=request.pipeline_type, status="invalid_config").inc()
            logger.error("Pipeline configuration invalid", exc_info=True)
            self._track_operation(operation=f"trigger_{request.pipeline_type}", status="invalid_config")
            return PipelineTriggerResult(
                success=False,
                job_id="",
                pipeline_type=request.pipeline_type,
                status="INVALID",
                error=str(exc),
            )

        job_id = self._generate_job_id(request.pipeline_type)
        job_state = PipelineJobState(job_id=job_id, pipeline_type=request.pipeline_type, status="QUEUED")
        self._register_job_state(job_state)

        loop = asyncio.get_running_loop()
        task = loop.create_task(
            self._execute_pipeline_job(
                job_state=job_state,
                orchestrator=orchestrator,
                run_config=run_config,
            )
        )
        self._job_tasks[job_id] = task

        self._track_operation(operation=f"trigger_{request.pipeline_type}", status="queued")
        return PipelineTriggerResult(
            success=True,
            job_id=job_id,
            pipeline_type=request.pipeline_type,
            status="QUEUED",
            message=f"Pipeline {job_id} queued successfully",
        )

    async def get_pipeline_progress(self, job_id: str) -> PipelineProgress:
        """Return progress information for a given job id."""
        state = self._jobs.get(job_id)
        if state is None:
            persisted = self._load_job_state(job_id)
            if persisted is not None:
                self._jobs[job_id] = persisted
                state = persisted
        if state is None:
            return PipelineProgress(
                job_id=job_id,
                status="UNKNOWN",
                progress=0.0,
                current_stage="UNKNOWN",
                eta_seconds=0,
                message="Job id not found",
                error="not_found",
                started_at=None,
                finished_at=None,
            )
        return PipelineProgress(
            job_id=job_id,
            status=state.status,
            progress=state.progress,
            current_stage=state.current_stage,
            eta_seconds=state.eta_seconds,
            message=state.message,
            error=state.error,
            started_at=state.started_at,
            finished_at=state.finished_at,
            started_at_iso=state.started_at_iso,
            finished_at_iso=state.finished_at_iso,
        )

    async def cancel_pipeline(self, job_id: str) -> PipelineCancelResult:
        """Attempt to cancel a running pipeline job."""
        operation = f"cancel_{job_id}"
        self._track_operation(operation=operation, status="started")

        job_state = self._jobs.get(job_id)
        if job_state is None:
            persisted = self._load_job_state(job_id)
            if persisted is not None:
                self._jobs[job_id] = persisted
                job_state = persisted

        if job_state is None:
            self._track_operation(operation=operation, status="not_found")
            return PipelineCancelResult(
                success=False,
                job_id=job_id,
                status="NOT_FOUND",
                error="Job not found",
            )

        task = self._job_tasks.get(job_id)
        if task is None or task.done():
            if job_state.status in {"COMPLETED", "FAILED", "CANCELLED"}:
                self._track_operation(operation=operation, status="already_finished")
                return PipelineCancelResult(
                    success=True,
                    job_id=job_id,
                    status=job_state.status,
                    message="Job already finished",
                )

        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - defensive
                logger.debug("pipeline cancellation raised", exc_info=True)

        job_state.status = "CANCELLED"
        job_state.message = "Pipeline cancelled"
        job_state.error = None
        job_state.eta_seconds = 0
        loop = asyncio.get_event_loop()
        job_state.finished_at = loop.time()
        job_state.finished_at_iso = dt.datetime.now(dt.UTC).isoformat()
        self._persist_job_state(job_state)
        self._job_tasks.pop(job_id, None)
        pipeline_jobs_total.labels(pipeline_type=job_state.pipeline_type, status="cancelled").inc()
        self._track_operation(operation=operation, status="success")
        return PipelineCancelResult(
            success=True,
            job_id=job_id,
            status="CANCELLED",
            message="Cancellation request acknowledged",
        )

    async def list_jobs(self) -> list[PipelineJobState]:
        """Return a snapshot of known pipeline jobs."""
        snapshot: dict[str, PipelineJobState] = {
            job_id: replace(job_state)
            for job_id, job_state in self._jobs.items()
        }
        store = self._job_store
        if store is not None:
            try:
                for job_state in store.list_jobs():
                    snapshot.setdefault(job_state.job_id, replace(job_state))
            except Exception:  # pragma: no cover - defensive
                pipeline_job_store_failures_total.labels(operation="list").inc()
                logger.debug("pipeline job list retrieval failed", exc_info=True)
        return sorted(snapshot.values(), key=self._job_sort_key)

    async def purge_job(self, job_id: str) -> PipelinePurgeResult:
        """Remove a job record from in-memory and persisted history."""
        operation = f"purge_{job_id}"
        self._track_operation(operation=operation, status="started")

        job_state = self._jobs.pop(job_id, None)
        if job_state is None:
            job_state = self._load_job_state(job_id)

        if job_state is None:
            self._track_operation(operation=operation, status="not_found")
            return PipelinePurgeResult(
                success=False,
                job_id=job_id,
                status="NOT_FOUND",
                message="Job not found",
                error=None,
            )

        task = self._job_tasks.pop(job_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # pragma: no cover - defensive
                logger.debug("pipeline purge task cancellation raised", exc_info=True)

        removed = self._remove_job_state(job_id)
        if not removed:
            self._jobs[job_id] = job_state
            self._track_operation(operation=operation, status="store_delete_failed")
            return PipelinePurgeResult(
                success=False,
                job_id=job_id,
                status="FAILED",
                message="Unable to delete job from store",
                error="store_delete_failed",
            )

        self._track_operation(operation=operation, status="success")
        return PipelinePurgeResult(
            success=True,
            job_id=job_id,
            status="PURGED",
            message="Pipeline job purged",
            error=None,
        )

    async def _execute_pipeline_job(
        self,
        *,
        job_state: PipelineJobState,
        orchestrator: MLPipelineOrchestrator,
        run_config: OrchestratorRunConfig,
    ) -> None:
        loop = asyncio.get_event_loop()
        job_state.status = "RUNNING"
        job_state.started_at = loop.time()
        start_time = dt.datetime.now(dt.UTC)
        job_state.started_at_iso = start_time.isoformat()
        job_state.current_stage = f"stage:{run_config.stage.value}"
        job_state.message = f"Running {run_config.stage.value} stage"
        job_state.progress = 0.0
        self._persist_job_state(job_state)
        try:
            rc = await self._run_async(
                partial(self._dispatch_stage_run, orchestrator, run_config),
            )
        except asyncio.CancelledError:
            job_state.status = "CANCELLED"
            job_state.current_stage = "cancelled"
            job_state.message = "Pipeline execution cancelled"
            job_state.error = None
            pipeline_jobs_total.labels(pipeline_type=job_state.pipeline_type, status="cancelled").inc()
            self._persist_job_state(job_state)
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            job_state.status = "FAILED"
            job_state.error = str(exc)
            job_state.message = "Pipeline execution raised an exception"
            pipeline_jobs_total.labels(pipeline_type=job_state.pipeline_type, status="exception").inc()
            logger.exception("Pipeline execution failed", extra={"job_id": job_state.job_id})
            self._persist_job_state(job_state)
        else:
            if rc == 0:
                job_state.status = "COMPLETED"
                job_state.progress = 1.0
                job_state.current_stage = "completed"
                job_state.message = "Pipeline completed successfully"
                pipeline_jobs_total.labels(pipeline_type=job_state.pipeline_type, status="completed").inc()
            else:
                job_state.status = "FAILED"
                job_state.error = f"orchestrator returned code {rc}"
                job_state.current_stage = "failed"
                job_state.message = f"Pipeline failed with return code {rc}"
                pipeline_jobs_total.labels(pipeline_type=job_state.pipeline_type, status="failed").inc()
            self._persist_job_state(job_state)
        finally:
            finish_time = loop.time()
            finish_dt = dt.datetime.now(dt.UTC)
            job_state.finished_at = finish_time
            job_state.finished_at_iso = finish_dt.isoformat()
            job_state.eta_seconds = 0
            duration = max(finish_time - (job_state.started_at or finish_time), 0.0)
            pipeline_job_latency.labels(pipeline_type=job_state.pipeline_type).observe(duration)
            logger.info(
                "Pipeline job finished",
                extra={
                    "job_id": job_state.job_id,
                    "pipeline_type": job_state.pipeline_type,
                    "status": job_state.status,
                    "started_at": start_time.isoformat(),
                    "duration_seconds": duration,
                },
            )
            self._persist_job_state(job_state)
            self._job_tasks.pop(job_state.job_id, None)

    def _dispatch_stage_run(
        self,
        orchestrator: MLPipelineOrchestrator,
        run_config: OrchestratorRunConfig,
    ) -> int:
        """Execute the orchestrator stage defined by ``run_config``."""

        from ml.orchestration.pipeline_orchestrator import _dataset_only_config
        from ml.orchestration.pipeline_orchestrator import _run_ingestion_stage

        orch_cfg = run_config.compose_orchestrator_config() if run_config.dataset is not None else None
        stage_enum = self._import_stage_enum()
        stage = run_config.stage
        if stage is stage_enum.INGEST:
            ingestion_cls = self._import_ingestion_stage_config()
            ingestion_cfg = run_config.ingestion or ingestion_cls()
            auto_fill_cls = self._import_auto_fill_config()
            auto_fill_cfg = run_config.auto_fill or auto_fill_cls()
            return _run_ingestion_stage(
                orch=orchestrator,
                ds_cfg=run_config.dataset,
                auto_fill_cfg=auto_fill_cfg,
                ingestion_cfg=ingestion_cfg,
                ingestor=getattr(orchestrator, "ingestor", None),
                ingestion_service=getattr(orchestrator, "service", None),
            )
        if orch_cfg is None:
            raise ValueError("Dataset configuration is required for non-ingestion stages")
        if stage is stage_enum.DATASET:
            dataset_only_cfg = _dataset_only_config(orch_cfg)
            return orchestrator.run(dataset_only_cfg)
        if stage is stage_enum.TRAIN:
            return orchestrator.run_training_only(orch_cfg)
        return orchestrator.run(orch_cfg)

    def _resolve_orchestrator(self) -> MLPipelineOrchestrator | None:
        integration = self._integration
        if integration is None:
            return None
        orchestrator = getattr(integration, "orchestrator", None)
        if orchestrator is None:
            return None
        if not hasattr(orchestrator, "run"):
            return None
        return cast("MLPipelineOrchestrator", orchestrator)

    def _generate_job_id(self, pipeline_type: str) -> str:
        suffix = uuid.uuid4().hex[:8]
        return f"{pipeline_type}_{suffix}"

    def _register_job_state(self, job_state: PipelineJobState) -> None:
        self._jobs[job_state.job_id] = job_state
        self._persist_job_state(job_state)

    def _persist_job_state(self, job_state: PipelineJobState) -> None:
        store = self._job_store
        if store is None:
            return
        try:
            store.save(job_state)
        except Exception:  # pragma: no cover - defensive
            pipeline_job_store_failures_total.labels(operation="save").inc()
            logger.debug("pipeline job persistence failed", exc_info=True)

    def _remove_job_state(self, job_id: str) -> bool:
        store = self._job_store
        if store is None:
            return True
        try:
            store.delete(job_id)
        except Exception:  # pragma: no cover - defensive
            pipeline_job_store_failures_total.labels(operation="delete").inc()
            logger.debug("pipeline job removal failed", exc_info=True)
            return False
        return True

    def _load_job_state(self, job_id: str) -> PipelineJobState | None:
        store = self._job_store
        if store is None:
            return None
        try:
            return store.get(job_id)
        except Exception:  # pragma: no cover - defensive
            pipeline_job_store_failures_total.labels(operation="get").inc()
            logger.debug("pipeline job load failed", exc_info=True)
            return None

    def _initialise_job_store(self, integration_manager: MLIntegrationManager | None) -> None:
        if integration_manager is None:
            return
        store = getattr(integration_manager, "pipeline_job_store", None)
        if store is None:
            return
        if not hasattr(store, "save"):
            return
        self._job_store = cast(PipelineJobStoreProtocol, store)
        try:
            for persisted in self._job_store.list_jobs():
                if persisted.job_id not in self._jobs:
                    self._jobs[persisted.job_id] = persisted
        except Exception:  # pragma: no cover - defensive
            pipeline_job_store_failures_total.labels(operation="list").inc()
            logger.debug("pipeline job bootstrap failed", exc_info=True)

    def _build_run_config(
        self,
        *,
        pipeline_type: str,
        payload: Mapping[str, Any],
    ) -> OrchestratorRunConfig:
        run_config_cls = self._import_run_config_type()
        stage_enum = self._import_stage_enum()

        dataset_cfg = self._coerce_dataclass(
            payload.get("dataset"),
            self._import_dataset_config(),
        )
        ingestion_cfg = self._coerce_optional_dataclass(
            payload.get("ingestion"),
            self._import_ingestion_stage_config(),
        )

        teacher_cls = self._import_teacher_config()
        student_cls = self._import_student_config()
        hpo_cls = self._import_hpo_config()
        training_cls = self._import_training_stage_config()

        teacher_cfg = self._coerce_optional_dataclass(payload.get("teacher"), teacher_cls)
        student_cfg = self._coerce_optional_dataclass(payload.get("student"), student_cls)
        hpo_cfg = self._coerce_optional_dataclass(payload.get("hpo"), hpo_cls)
        training_cfg = self._coerce_optional_dataclass(payload.get("training"), training_cls)
        if training_cfg is None:
            training_cfg = training_cls(
                teacher=teacher_cfg or teacher_cls(),
                student=student_cfg or student_cls(),
                hpo=hpo_cfg or hpo_cls(),
            )
        else:
            training_cfg = training_cls(
                teacher=teacher_cfg or training_cfg.teacher,
                student=student_cfg or training_cfg.student,
                hpo=hpo_cfg or training_cfg.hpo,
            )

        promotions_cfg = self._coerce_optional_dataclass(
            payload.get("promotions"),
            self._import_promotions_config(),
        )
        auto_fill_cfg = self._coerce_optional_dataclass(
            payload.get("auto_fill"),
            self._import_auto_fill_config(),
        )
        integration_cfg = self._coerce_optional_dataclass(
            payload.get("integration"),
            self._import_integration_config(),
        )

        ingestion_cls = self._import_ingestion_stage_config()
        stage = self._infer_stage_from_pipeline_type(
            pipeline_type=pipeline_type,
            explicit_stage=payload.get("stage"),
            stage_enum=stage_enum,
        )

        ingestion_cfg_effective = ingestion_cfg
        if stage is stage_enum.INGEST:
            ingestion_cfg_effective = self._ensure_ingestion_enabled(
                ingestion_cfg=ingestion_cfg_effective,
                ingestion_cls=ingestion_cls,
                dataset_cfg=dataset_cfg,
            )

        return run_config_cls(
            stage=stage,
            dataset=dataset_cfg,
            ingestion=ingestion_cfg_effective,
            training=training_cfg,
            promotions=promotions_cfg,
            auto_fill=auto_fill_cfg,
            integration=integration_cfg,
        )

    @staticmethod
    def _job_sort_key(job_state: PipelineJobState) -> tuple[float, str]:
        timestamp_iso = job_state.finished_at_iso or job_state.started_at_iso
        if timestamp_iso is not None:
            try:
                iso_dt = dt.datetime.fromisoformat(timestamp_iso)
                return (-iso_dt.timestamp(), job_state.job_id)
            except ValueError:
                pass
        timestamp = job_state.finished_at
        if timestamp is None:
            timestamp = job_state.started_at or 0.0
        return (-float(timestamp), job_state.job_id)

    def _coerce_optional_dataclass(self, data: Any, cls: type[T]) -> T | None:
        if data is None:
            return None
        return self._coerce_dataclass(data, cls)

    def _coerce_dataclass(self, data: Any, cls: type[T]) -> T:
        if isinstance(data, cls):
            return data
        if not isinstance(data, Mapping):
            raise ValueError(f"Expected mapping to build {cls.__name__}")
        kwargs: dict[str, Any] = {}
        class_type = cast(type[Any], cls)
        for fld in fields(class_type):
            if fld.name not in data:
                continue
            value = data[fld.name]
            kwargs[fld.name] = self._coerce_value(fld.type, value)
        return cls(**kwargs)

    def _coerce_value(self, annotation: Any, value: Any) -> Any:
        origin = get_origin(annotation)
        if origin is None:
            if is_dataclass_type(annotation):
                return self._coerce_dataclass(value, annotation)
            if isinstance(annotation, type) and not isinstance(value, annotation):
                try:
                    return annotation(value)
                except Exception:
                    return value
            return value

        if origin is tuple:
            elem_type = get_args(annotation)[0]
            return tuple(self._coerce_value(elem_type, item) for item in value)

        if origin in {list, set, frozenset}:
            elem_type = get_args(annotation)[0]
            converted = [self._coerce_value(elem_type, item) for item in value]
            if origin is list:
                return converted
            if origin is set:
                return set(converted)
            return frozenset(converted)

        if origin in {dict, Mapping}:
            key_type, val_type = get_args(annotation)
            return {
                self._coerce_value(key_type, key): self._coerce_value(val_type, val)
                for key, val in value.items()
            }

        args = get_args(annotation)
        if args:
            for arg in args:
                if arg is type(None) and value is None:
                    return None
                try:
                    return self._coerce_value(arg, value)
                except ValueError:
                    continue
            return value

        return value

    def _import_dataset_config(self) -> type[DatasetBuildConfig]:
        from ml.orchestration.config_types import DatasetBuildConfig

        return DatasetBuildConfig

    def _import_hpo_config(self) -> type[HPOConfig]:
        from ml.orchestration.config_types import HPOConfig

        return HPOConfig

    def _import_teacher_config(self) -> type[TeacherTrainConfig]:
        from ml.orchestration.config_types import TeacherTrainConfig

        return TeacherTrainConfig

    def _import_student_config(self) -> type[StudentDistillConfig]:
        from ml.orchestration.config_types import StudentDistillConfig

        return StudentDistillConfig

    def _import_promotions_config(self) -> type[PromotionsConfig]:
        from ml.orchestration.config_types import PromotionsConfig

        return PromotionsConfig

    def _import_auto_fill_config(self) -> type[AutoFillUniverseConfig]:
        from ml.orchestration.config_types import AutoFillUniverseConfig

        return AutoFillUniverseConfig

    def _import_integration_config(self) -> type[IntegrationConfig]:
        from ml.orchestration.config_types import IntegrationConfig

        return IntegrationConfig

    def _import_ingestion_stage_config(self) -> type[IngestionStageConfig]:
        from ml.orchestration.config_loader import IngestionStageConfig

        return IngestionStageConfig

    def _import_training_stage_config(self) -> type[TrainingStageConfig]:
        from ml.orchestration.config_loader import TrainingStageConfig

        return TrainingStageConfig

    def _import_run_config_type(self) -> type[OrchestratorRunConfig]:
        from ml.orchestration.config_loader import OrchestratorRunConfig

        return OrchestratorRunConfig

    def _import_stage_enum(self) -> type[Stage]:
        from ml.orchestration.config_loader import Stage

        return Stage

    def _infer_stage_from_pipeline_type(
        self,
        *,
        pipeline_type: str,
        explicit_stage: str | Stage | None,
        stage_enum: type[Stage],
    ) -> Stage:
        if explicit_stage is not None:
            try:
                return stage_enum(explicit_stage)
            except ValueError as exc:
                raise ValueError(f"Unsupported stage '{explicit_stage}'") from exc

        normalized = pipeline_type.strip().lower().replace("-", "_")
        mapping: dict[str, Stage] = {
            "ingest": stage_enum.INGEST,
            "ingestion": stage_enum.INGEST,
            "ingest_only": stage_enum.INGEST,
            "dataset": stage_enum.DATASET,
            "build_dataset": stage_enum.DATASET,
            "dataset_only": stage_enum.DATASET,
            "train": stage_enum.TRAIN,
            "training": stage_enum.TRAIN,
            "training_only": stage_enum.TRAIN,
            "full": stage_enum.FULL,
            "full_pipeline": stage_enum.FULL,
            "pipeline": stage_enum.FULL,
        }
        inferred = mapping.get(normalized, stage_enum.FULL)
        if inferred is not stage_enum.FULL:
            logger.debug(
                "Inferred pipeline stage from type",
                extra={"pipeline_type": pipeline_type, "stage": inferred.value},
            )
        return inferred

    def _ensure_ingestion_enabled(
        self,
        *,
        ingestion_cfg: IngestionStageConfig | None,
        ingestion_cls: type[IngestionStageConfig],
        dataset_cfg: DatasetBuildConfig,
    ) -> IngestionStageConfig:
        default_cfg = ingestion_cls()
        cfg = ingestion_cfg or default_cfg
        updated = False
        if not cfg.enabled:
            cfg = replace(cfg, enabled=True)
            updated = True
        preferred_dataset_id = dataset_cfg.market_dataset_id or dataset_cfg.dataset_id
        if preferred_dataset_id:
            if cfg.dataset_id == "" or cfg.dataset_id == default_cfg.dataset_id:
                cfg = replace(cfg, dataset_id=preferred_dataset_id)
                updated = True

        if (not cfg.instruments or cfg.instruments == default_cfg.instruments) and dataset_cfg.instrument_ids:
            cfg = replace(cfg, instruments=dataset_cfg.instrument_ids)
            updated = True
        elif (not cfg.instruments or cfg.instruments == default_cfg.instruments) and dataset_cfg.symbols:
            inferred = tuple(
                symbol.strip().upper()
                for symbol in dataset_cfg.symbols.split(",")
                if symbol.strip()
            )
            if inferred:
                cfg = replace(cfg, instruments=inferred)
                updated = True
        if updated:
            logger.debug(
                "Adjusted ingestion configuration for stage",
                extra={
                    "dataset_id": cfg.dataset_id,
                    "instrument_count": len(cfg.instruments),
                },
            )
        return cfg


def is_dataclass_type(type_: Any) -> bool:
    """Return ``True`` when ``type_`` is a dataclass type."""
    try:
        return is_dataclass(cast(type[Any], type_))
    except Exception:
        return False


__all__ = [
    "PipelineCancelResult",
    "PipelineIntegrationService",
    "PipelineJobState",
    "PipelineProgress",
    "PipelineTriggerRequest",
    "PipelineTriggerResult",
]
