"""Tests for the pipeline integration service."""

from __future__ import annotations

import asyncio
import types
from dataclasses import replace
from typing import Any

import pytest

from ml.dashboard.services.pipelines_service import (
    PipelineIntegrationService,
    PipelineJobState,
    PipelineProgress,
    PipelineTriggerRequest,
    pipeline_job_store_failures_total,
)


class DummyOrchestrator:
    """Stub orchestrator capturing run invocations."""

    def __init__(self, return_code: int = 0) -> None:
        self.return_code = return_code
        self.run_calls: list[Any] = []

    def run(self, cfg: Any) -> int:
        self.run_calls.append(cfg)
        return self.return_code

    def run_training_only(self, cfg: Any) -> int:
        self.run_calls.append(cfg)
        return self.return_code


class DummyIntegrationManager:
    """Simple integration manager exposing orchestrator attribute."""

    def __init__(self, orchestrator: Any | None, pipeline_job_store: Any | None = None) -> None:
        self.orchestrator = orchestrator
        self.pipeline_job_store = pipeline_job_store


class InMemoryJobStore:
    """In-memory implementation of the pipeline job store protocol."""

    def __init__(self) -> None:
        self._jobs: dict[str, PipelineJobState] = {}

    def save(self, job_state: PipelineJobState) -> None:
        self._jobs[job_state.job_id] = replace(job_state)

    def delete(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def get(self, job_id: str) -> PipelineJobState | None:
        state = self._jobs.get(job_id)
        return replace(state) if state is not None else None

    def list_jobs(self) -> list[PipelineJobState]:
        return [replace(state) for state in self._jobs.values()]


class FaultyJobStore:
    """Job store that raises on all operations to exercise failure metrics."""

    def save(self, job_state: PipelineJobState) -> None:  # pragma: no cover - deterministic in tests
        raise RuntimeError("save failed")

    def delete(self, job_id: str) -> None:  # pragma: no cover - deterministic in tests
        raise RuntimeError("delete failed")

    def get(self, job_id: str) -> PipelineJobState | None:  # pragma: no cover - deterministic in tests
        raise RuntimeError("get failed")

    def list_jobs(self) -> list[PipelineJobState]:  # pragma: no cover - deterministic in tests
        raise RuntimeError("list failed")


@pytest.fixture
def pipeline_service() -> PipelineIntegrationService:
    orchestrator = DummyOrchestrator(return_code=0)
    manager = DummyIntegrationManager(orchestrator)
    service = PipelineIntegrationService(manager)

    async def fake_run_async(self: PipelineIntegrationService, func: Any) -> Any:
        return func()

    service._run_async = types.MethodType(fake_run_async, service)
    return service


@pytest.mark.asyncio
async def test_trigger_pipeline_success(pipeline_service: PipelineIntegrationService) -> None:
    request = PipelineTriggerRequest(
        pipeline_type="training",
        config={
            "dataset": {
                "data_dir": "/tmp/data",
                "symbols": "AAPL",
                "out_dir": "/tmp/out",
            },
            "hpo": {},
            "teacher": {"model_id": "teacher"},
            "student": {},
        },
    )

    result = await pipeline_service.trigger_pipeline(request)

    assert result.success is True
    assert result.status == "QUEUED"
    job_id = result.job_id

    await asyncio.sleep(0)
    progress = await pipeline_service.get_pipeline_progress(job_id)

    assert isinstance(progress, PipelineProgress)
    assert progress.status == "COMPLETED"
    assert progress.progress == pytest.approx(1.0)
    assert progress.started_at is not None
    assert progress.finished_at is not None
    assert progress.finished_at >= progress.started_at
    assert progress.started_at_iso is not None
    assert progress.finished_at_iso is not None


@pytest.mark.asyncio
async def test_trigger_pipeline_invalid_config(pipeline_service: PipelineIntegrationService) -> None:
    bad_request = PipelineTriggerRequest(pipeline_type="training", config={})

    result = await pipeline_service.trigger_pipeline(bad_request)

    assert result.success is False
    assert result.status == "INVALID"
    assert result.error is not None


@pytest.mark.asyncio
async def test_pipeline_failure_updates_state() -> None:
    orchestrator = DummyOrchestrator(return_code=1)
    manager = DummyIntegrationManager(orchestrator)
    service = PipelineIntegrationService(manager)

    async def fake_run_async(self: PipelineIntegrationService, func: Any) -> Any:
        return func()

    service._run_async = types.MethodType(fake_run_async, service)

    request = PipelineTriggerRequest(
        pipeline_type="training",
        config={
            "dataset": {
                "data_dir": "/tmp/data",
                "symbols": "MSFT",
                "out_dir": "/tmp/out",
            },
            "hpo": {},
            "teacher": {"model_id": "teacher"},
            "student": {},
        },
    )

    trigger = await service.trigger_pipeline(request)
    await asyncio.sleep(0)
    progress = await service.get_pipeline_progress(trigger.job_id)

    assert progress.status == "FAILED"
    assert progress.error is not None
    assert progress.started_at is not None
    assert progress.finished_at is not None
    assert progress.started_at_iso is not None
    assert progress.finished_at_iso is not None

    jobs = await service.list_jobs()
    failing_job = next(job for job in jobs if job.job_id == trigger.job_id)
    assert failing_job.message is not None
    assert failing_job.error is not None


@pytest.mark.asyncio
async def test_cancel_pipeline_marks_job_cancelled() -> None:
    orchestrator = DummyOrchestrator(return_code=0)
    job_store = InMemoryJobStore()
    manager = DummyIntegrationManager(orchestrator, pipeline_job_store=job_store)
    service = PipelineIntegrationService(manager)

    async def slow_run_async(self: PipelineIntegrationService, func: Any) -> Any:
        await asyncio.sleep(1.0)
        return func()

    service._run_async = types.MethodType(slow_run_async, service)

    request = PipelineTriggerRequest(
        pipeline_type="training",
        config={
            "dataset": {
                "data_dir": "/tmp/data",
                "symbols": "MSFT",
                "out_dir": "/tmp/out",
            },
            "hpo": {},
            "teacher": {"model_id": "teacher"},
            "student": {},
        },
    )

    result = await service.trigger_pipeline(request)
    await asyncio.sleep(0)
    cancel = await service.cancel_pipeline(result.job_id)

    assert cancel.success is True
    assert cancel.status == "CANCELLED"

    progress = await service.get_pipeline_progress(result.job_id)
    assert progress.status == "CANCELLED"
    assert progress.started_at is not None
    assert progress.finished_at is not None
    assert progress.started_at_iso is not None
    assert progress.finished_at_iso is not None

    persisted = job_store.get(result.job_id)
    assert persisted is not None
    assert persisted.status == "CANCELLED"


@pytest.mark.asyncio
async def test_pipeline_job_store_persistence_round_trip() -> None:
    job_store = InMemoryJobStore()
    orchestrator = DummyOrchestrator(return_code=0)
    manager = DummyIntegrationManager(orchestrator, pipeline_job_store=job_store)
    service = PipelineIntegrationService(manager)

    async def immediate_run_async(self: PipelineIntegrationService, func: Any) -> Any:
        return func()

    service._run_async = types.MethodType(immediate_run_async, service)

    request = PipelineTriggerRequest(
        pipeline_type="training",
        config={
            "dataset": {
                "data_dir": "/tmp/data",
                "symbols": "IBM",
                "out_dir": "/tmp/out",
            },
            "hpo": {},
            "teacher": {"model_id": "teacher"},
            "student": {},
        },
    )

    result = await service.trigger_pipeline(request)
    await asyncio.sleep(0)
    progress = await service.get_pipeline_progress(result.job_id)
    assert progress.status == "COMPLETED"
    assert progress.started_at is not None
    assert progress.finished_at is not None
    assert progress.started_at_iso is not None
    assert progress.finished_at_iso is not None

    reloaded_service = PipelineIntegrationService(
        DummyIntegrationManager(orchestrator=None, pipeline_job_store=job_store),
    )
    reloaded_progress = await reloaded_service.get_pipeline_progress(result.job_id)
    assert reloaded_progress.status == "COMPLETED"
    assert reloaded_progress.progress == pytest.approx(1.0)
    assert reloaded_progress.started_at is not None
    assert reloaded_progress.finished_at is not None
    assert reloaded_progress.started_at_iso is not None
    assert reloaded_progress.finished_at_iso is not None
    jobs = await reloaded_service.list_jobs()
    assert any(job.job_id == result.job_id and job.status == "COMPLETED" for job in jobs)


@pytest.mark.asyncio
async def test_pipeline_job_store_failure_metrics_increment() -> None:
    job_store = FaultyJobStore()
    orchestrator = DummyOrchestrator(return_code=0)
    manager = DummyIntegrationManager(orchestrator, pipeline_job_store=job_store)
    service = PipelineIntegrationService(manager)

    async def immediate_run_async(self: PipelineIntegrationService, func: Any) -> Any:
        return func()

    service._run_async = types.MethodType(immediate_run_async, service)

    save_counter = pipeline_job_store_failures_total.labels(operation="save")
    get_counter = pipeline_job_store_failures_total.labels(operation="get")
    list_counter = pipeline_job_store_failures_total.labels(operation="list")

    save_initial = save_counter._value.get()
    get_initial = get_counter._value.get()
    list_initial = list_counter._value.get()

    request = PipelineTriggerRequest(
        pipeline_type="training",
        config={
            "dataset": {
                "data_dir": "/tmp/data",
                "symbols": "ORCL",
                "out_dir": "/tmp/out",
            },
            "hpo": {},
            "teacher": {"model_id": "teacher"},
            "student": {},
        },
    )

    await service.trigger_pipeline(request)
    assert save_counter._value.get() == pytest.approx(save_initial + 1)

    await service.get_pipeline_progress("missing-job")
    assert get_counter._value.get() == pytest.approx(get_initial + 1)

    list_before = list_counter._value.get()
    await service.list_jobs()
    assert list_counter._value.get() == pytest.approx(list_before + 1)


@pytest.mark.asyncio
async def test_purge_job_removes_state() -> None:
    job_store = InMemoryJobStore()
    service = PipelineIntegrationService(
        DummyIntegrationManager(orchestrator=None, pipeline_job_store=job_store)
    )

    job_state = PipelineJobState(
        job_id="job_to_purge",
        pipeline_type="training",
        status="FAILED",
        message="Errored",
        error="boom",
        started_at=1.0,
        finished_at=2.0,
        started_at_iso="2025-01-01T02:00:00+00:00",
        finished_at_iso="2025-01-01T02:05:00+00:00",
    )
    job_store.save(job_state)
    service._jobs[job_state.job_id] = job_state

    result = await service.purge_job(job_state.job_id)

    assert result.success is True
    assert result.status == "PURGED"
    assert service._jobs.get(job_state.job_id) is None
    assert job_store.get(job_state.job_id) is None


@pytest.mark.asyncio
async def test_purge_job_records_delete_failure_metric() -> None:
    job_store = FaultyJobStore()
    service = PipelineIntegrationService(
        DummyIntegrationManager(orchestrator=None, pipeline_job_store=job_store)
    )

    job_state = PipelineJobState(
        job_id="job_failure",
        pipeline_type="training",
        status="FAILED",
    )
    service._jobs[job_state.job_id] = job_state

    delete_counter = pipeline_job_store_failures_total.labels(operation="delete")
    delete_initial = delete_counter._value.get()

    result = await service.purge_job(job_state.job_id)

    assert result.success is False
    assert result.status == "FAILED"
    assert service._jobs.get(job_state.job_id) is not None
    assert delete_counter._value.get() == pytest.approx(delete_initial + 1)


@pytest.mark.asyncio
async def test_list_jobs_returns_sorted_history() -> None:
    job_store = InMemoryJobStore()
    job_store.save(
        PipelineJobState(
            job_id="job_old",
            pipeline_type="training",
            status="COMPLETED",
            progress=1.0,
            current_stage="completed",
            eta_seconds=0,
            message="Old job",
            started_at=5.0,
            finished_at=10.0,
            started_at_iso="2025-01-01T00:01:00+00:00",
            finished_at_iso="2025-01-01T00:02:00+00:00",
        )
    )
    job_store.save(
        PipelineJobState(
            job_id="job_new",
            pipeline_type="training",
            status="COMPLETED",
            progress=1.0,
            current_stage="completed",
            eta_seconds=0,
            message="New job",
            started_at=15.0,
            finished_at=20.0,
            started_at_iso="2025-01-01T01:00:00+00:00",
            finished_at_iso="2025-01-01T01:05:00+00:00",
        )
    )

    service = PipelineIntegrationService(DummyIntegrationManager(orchestrator=None, pipeline_job_store=job_store))
    service._jobs["job_running"] = PipelineJobState(
        job_id="job_running",
        pipeline_type="training",
        status="RUNNING",
        progress=0.4,
        current_stage="stage",
        eta_seconds=30,
        started_at=17.5,
        started_at_iso="2025-01-01T01:02:30+00:00",
    )

    jobs = await service.list_jobs()
    assert [job.job_id for job in jobs] == ["job_new", "job_running", "job_old"]
