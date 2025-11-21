"""
Tests for the dashboard actor integration service.
"""

from __future__ import annotations

pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

import types
from collections.abc import Mapping
from typing import Any, Callable

import pytest

from ml.dashboard.services.actors_service import ActorDeploymentRequest
from ml.dashboard.services.actors_service import ActorIntegrationService
from ml.dashboard.services.actors_service import ActorPauseRequest
from ml.dashboard.services.actors_service import ActorResumeRequest
from ml.dashboard.services.actors_service import ActorStopRequest


pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)

class DummyLifecycleActor:
    """
    Dummy actor exposing lifecycle hooks for testing.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)
        self.actor_id = config.get("actor_id", "dummy-actor")
        self.paused = False
        self.disposed = False
        self.degrade_called = False
        self.fail_stop = bool(config.get("fail_stop"))
        self.fail_dispose = bool(config.get("fail_dispose"))

    def get_health_status(self) -> Mapping[str, Any]:
        return {"healthy": not self.disposed, "paused": self.paused}

    def stop(self) -> None:
        if self.fail_stop:
            raise RuntimeError("stop failure")
        self.paused = True

    def resume(self) -> None:
        if self.disposed:
            raise RuntimeError("disposed")
        self.paused = False

    def dispose(self) -> None:
        if self.fail_dispose:
            raise RuntimeError("dispose failure")
        self.disposed = True

    def degrade(self) -> None:
        self.degrade_called = True


class DummyModelRegistry:
    """
    Minimal registry verifying model availability.
    """

    def __init__(self, available_models: set[str]) -> None:
        self.available_models = available_models

    def get_model(self, model_id: str) -> object:
        if model_id not in self.available_models:
            raise ValueError(f"model {model_id} missing")
        return object()


class DummyIntegrationManager:
    """
    Stub integration manager exposing registry attributes.
    """

    def __init__(self, model_registry: object | None = None) -> None:
        self.model_registry = model_registry


@pytest.fixture
def actor_service() -> ActorIntegrationService:
    manager = DummyIntegrationManager(model_registry=DummyModelRegistry({"model_a"}))
    service = ActorIntegrationService(manager)

    def factory(config: Mapping[str, Any]) -> DummyLifecycleActor:
        model_id = config.get("model_id")
        registry = manager.model_registry
        if isinstance(registry, DummyModelRegistry):
            registry.get_model(str(model_id))
        return DummyLifecycleActor(config)

    service.register_actor_factory("MLSignalActor", factory)

    async def fake_run_async(
        self: ActorIntegrationService,
        func: Callable[[], Any],
    ) -> Any:
        return func()

    service._run_async = types.MethodType(fake_run_async, service)
    return service


@pytest.mark.asyncio
async def test_deploy_actor_success(actor_service: ActorIntegrationService) -> None:
    request = ActorDeploymentRequest(
        actor_type="MLSignalActor",
        config={
            "actor_id": "test-actor",
            "model_id": "model_a",
            "prediction_threshold": 0.7,
        },
    )

    result = await actor_service.deploy_actor(request)

    assert result.success is True
    assert result.actor_id == "test-actor"
    assert result.status == "DEPLOYED"


@pytest.mark.asyncio
async def test_deploy_actor_missing_model(actor_service: ActorIntegrationService) -> None:
    request = ActorDeploymentRequest(
        actor_type="MLSignalActor",
        config={
            "actor_id": "test-actor",
            "model_id": "missing",
        },
    )

    result = await actor_service.deploy_actor(request)

    assert result.success is False
    assert result.status == "FAILED"
    assert result.error is not None


@pytest.mark.asyncio
async def test_pause_actor_transitions_state(actor_service: ActorIntegrationService) -> None:
    await actor_service.deploy_actor(
        ActorDeploymentRequest(
            actor_type="MLSignalActor",
            config={"actor_id": "actor-1", "model_id": "model_a"},
        ),
    )

    result = await actor_service.pause_actor(
        ActorPauseRequest(actor_id="actor-1", reason="maintenance")
    )

    assert result.success is True
    assert result.status == "PAUSED"

    snapshot = await actor_service.get_actor_health()
    assert snapshot.paused_actors == 1
    assert snapshot.actors["actor-1"].get("status") == "PAUSED"


@pytest.mark.asyncio
async def test_resume_actor_restores_running_state(actor_service: ActorIntegrationService) -> None:
    await actor_service.deploy_actor(
        ActorDeploymentRequest(
            actor_type="MLSignalActor",
            config={"actor_id": "actor-2", "model_id": "model_a"},
        ),
    )
    await actor_service.pause_actor(ActorPauseRequest(actor_id="actor-2"))

    result = await actor_service.resume_actor(ActorResumeRequest(actor_id="actor-2"))

    assert result.success is True
    assert result.status == "RUNNING"
    actor = actor_service._running_actors["actor-2"]
    assert isinstance(actor, DummyLifecycleActor)
    assert actor.paused is False
    snapshot = await actor_service.get_actor_health()
    assert snapshot.paused_actors == 0


@pytest.mark.asyncio
async def test_stop_actor_removes_instance(actor_service: ActorIntegrationService) -> None:
    await actor_service.deploy_actor(
        ActorDeploymentRequest(
            actor_type="MLSignalActor",
            config={"actor_id": "actor-stop", "model_id": "model_a"},
        ),
    )

    response = await actor_service.stop_actor(ActorStopRequest(actor_id="actor-stop"))

    assert response.success is True
    assert response.status == "STOPPED"
    assert "actor-stop" not in actor_service._running_actors
    snapshot = await actor_service.get_actor_health()
    assert snapshot.total_actors == 0


@pytest.mark.asyncio
async def test_stop_actor_force_uses_degrade(actor_service: ActorIntegrationService) -> None:
    await actor_service.deploy_actor(
        ActorDeploymentRequest(
            actor_type="MLSignalActor",
            config={"actor_id": "actor-force", "model_id": "model_a", "fail_stop": True},
        ),
    )
    actor = actor_service._running_actors["actor-force"]
    assert isinstance(actor, DummyLifecycleActor)

    response = await actor_service.stop_actor(ActorStopRequest(actor_id="actor-force", force=True))

    assert response.success is True
    assert response.status == "STOPPED"
    assert actor.degrade_called is True
    assert actor.disposed is True


@pytest.mark.asyncio
async def test_stop_actor_failure_returns_error(actor_service: ActorIntegrationService) -> None:
    await actor_service.deploy_actor(
        ActorDeploymentRequest(
            actor_type="MLSignalActor",
            config={"actor_id": "actor-fail", "model_id": "model_a", "fail_dispose": True},
        ),
    )

    response = await actor_service.stop_actor(ActorStopRequest(actor_id="actor-fail"))

    assert response.success is False
    assert response.status == "FAILED"
    assert response.error is not None


@pytest.mark.asyncio
async def test_pause_actor_not_found(actor_service: ActorIntegrationService) -> None:
    result = await actor_service.pause_actor(ActorPauseRequest(actor_id="missing"))

    assert result.success is False
    assert result.status == "NOT_FOUND"
