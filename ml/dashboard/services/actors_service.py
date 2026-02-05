"""Actor integration service with typed deployment flows."""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, cast

from ml.common.metrics_bootstrap import get_counter
from ml.dashboard.services.base_service import BaseIntegrationService


if TYPE_CHECKING:
    from ml.actors.base import BaseMLInferenceActor
    from ml.core.integration import MLIntegrationManager


logger = logging.getLogger(__name__)

actor_failures_total = get_counter(
    "ml_dashboard_actor_failures_total",
    "Total actor deployment failures",
    labelnames=["reason"],
)

actor_lifecycle_total = get_counter(
    "ml_dashboard_actor_lifecycle_total",
    "Actor lifecycle operations",
    labelnames=["operation", "status"],
)


class LifecycleActorProtocol(Protocol):
    """Protocol covering lifecycle controls expected by the integration service."""

    actor_id: str

    def stop(self) -> None:  # pragma: no cover - typing only
        """Pause the actor while keeping resources allocated."""

    def resume(self) -> None:  # pragma: no cover - typing only
        """Resume a previously paused actor."""

    def dispose(self) -> None:  # pragma: no cover - typing only
        """Release actor resources."""

    def get_health_status(self) -> Mapping[str, Any]:  # pragma: no cover - typing only
        """Return health telemetry for the actor."""


@dataclass(slots=True)
class ActorDeploymentRequest:
    """Request payload required to deploy an actor."""

    actor_type: str
    config: Mapping[str, Any]
    run_id: str | None = None


@dataclass(slots=True)
class ActorDeploymentResult:
    """Response emitted after attempting actor deployment."""

    success: bool
    actor_id: str
    status: str
    message: str | None = None
    error: str | None = None


@dataclass(slots=True)
class ActorHotReloadResult:
    """Outcome of a hot reload operation."""

    success: bool
    actor_id: str
    new_model_id: str | None = None
    status: str = "UNKNOWN"
    error: str | None = None


@dataclass(slots=True)
class ActorHealthSnapshot:
    """Aggregated health status for tracked actors."""

    total_actors: int = 0
    healthy_actors: int = 0
    unhealthy_actors: int = 0
    paused_actors: int = 0
    actors: dict[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class ActorLifecycleState:
    """Internal lifecycle metadata tracked per actor."""

    status: str
    last_transition: str
    message: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActorPauseRequest:
    """Request model for pausing an actor instance."""

    actor_id: str
    reason: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(slots=True)
class ActorResumeRequest:
    """Request model for resuming a paused actor."""

    actor_id: str
    metadata: Mapping[str, Any] | None = None


@dataclass(slots=True)
class ActorStopRequest:
    """Request model for stopping and disposing an actor."""

    actor_id: str
    force: bool = False
    reason: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(slots=True)
class ActorLifecycleResult:
    """Standardized response payload for lifecycle operations."""

    success: bool
    actor_id: str
    status: str
    message: str | None = None
    error: str | None = None


class ActorIntegrationService(BaseIntegrationService):
    """
    Integration facade for actor lifecycle controls.

    The service provides structured deployment and management helpers that the
    dashboard routes can rely on. It enforces early validation against the model
    registry when available and records deployment metrics.

    Example
    -------
    >>> from ml.dashboard.services.actors_service import ActorIntegrationService, ActorDeploymentRequest
    >>> service = ActorIntegrationService(integration_manager=None)
    >>> request = ActorDeploymentRequest(actor_type="MLSignalActor", config={"model_id": "demo"})
    >>> # Raises ValueError because the default factory requires a valid registry
    >>> # service.deploy_actor(request)
    """

    def __init__(
        self,
        integration_manager: MLIntegrationManager | None,
    ) -> None:
        super().__init__(integration_manager)
        self._running_actors: dict[str, LifecycleActorProtocol] = {}
        self._actor_states: dict[str, ActorLifecycleState] = {}
        self._actor_factories: dict[str, Callable[[Mapping[str, Any]], LifecycleActorProtocol]] = {
            "MLSignalActor": cast(
                Callable[[Mapping[str, Any]], LifecycleActorProtocol],
                self._create_signal_actor,
            ),
        }

    def register_actor_factory(
        self,
        actor_type: str,
        factory: Callable[[Mapping[str, Any]], LifecycleActorProtocol],
    ) -> None:
        """Register or override the factory used to build a specific actor type."""
        self._actor_factories[actor_type] = factory

    def get_service_name(self) -> str:
        return "actor_integration"

    async def health_check(self) -> dict[str, Any]:
        snapshot = await self.get_actor_health()
        return {
            "total_actors": snapshot.total_actors,
            "healthy_actors": snapshot.healthy_actors,
            "unhealthy_actors": snapshot.unhealthy_actors,
            "actors": snapshot.actors,
        }

    async def get_actor_health(self) -> ActorHealthSnapshot:
        """Return typed health snapshot for dashboard consumption."""
        snapshot = ActorHealthSnapshot(total_actors=len(self._running_actors))
        for actor_id, actor in self._running_actors.items():
            lifecycle_state = self._actor_states.get(actor_id)
            actor_info: dict[str, Any]
            try:
                if hasattr(actor, "get_health_status"):
                    status = actor.get_health_status()
                    actor_info = dict(status) if isinstance(status, Mapping) else {"raw": status}
                else:
                    actor_info = {"healthy": True}
            except Exception as exc:  # pragma: no cover - defensive
                actor_info = {"healthy": False, "error": str(exc)}

            if lifecycle_state is not None:
                actor_info.setdefault("status", lifecycle_state.status)
                actor_info.setdefault("last_transition", lifecycle_state.last_transition)
                if lifecycle_state.message is not None:
                    actor_info.setdefault("message", lifecycle_state.message)
                if lifecycle_state.error is not None:
                    actor_info.setdefault("lifecycle_error", lifecycle_state.error)
                if lifecycle_state.metadata:
                    actor_info.setdefault("metadata", lifecycle_state.metadata)
            else:
                actor_info.setdefault("status", "UNKNOWN")

            snapshot.actors[actor_id] = actor_info

            if bool(actor_info.get("healthy")):
                snapshot.healthy_actors += 1
            else:
                snapshot.unhealthy_actors += 1

            if lifecycle_state is not None and lifecycle_state.status == "PAUSED":
                snapshot.paused_actors += 1
        return snapshot

    async def deploy_actor(self, request: ActorDeploymentRequest) -> ActorDeploymentResult:
        """
        Deploy an actor using the provided configuration.

        Parameters
        ----------
        request : ActorDeploymentRequest
            Typed payload containing the desired actor type and configuration
            mapping. For ``MLSignalActor`` deployments ``model_id`` is required.

        Returns
        -------
        ActorDeploymentResult
            Structured response describing the outcome. ``success`` is ``True``
            when an actor instance has been constructed and registered with the
            service.
        """
        self._track_operation(operation="deploy_actor", status="started")
        actor_type = request.actor_type
        factory = self._actor_factories.get(actor_type)
        if factory is None:
            self._track_operation(operation="deploy_actor", status="invalid_type")
            return ActorDeploymentResult(
                success=False,
                actor_id="",
                status="INVALID",
                error=f"Invalid actor type: {actor_type}",
                message=None,
            )

        try:
            actor = factory(request.config)
        except ValueError as exc:
            actor_failures_total.labels(reason="init_failed").inc()
            logger.error("actor initialization failed", exc_info=True, extra={"actor_type": actor_type})
            self._track_operation(operation="deploy_actor", status="init_failed")
            return ActorDeploymentResult(
                success=False,
                actor_id="",
                status="FAILED",
                error=str(exc),
                message=None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            actor_failures_total.labels(reason="unexpected_error").inc()
            logger.exception("unexpected actor initialization error", extra={"actor_type": actor_type})
            self._track_operation(operation="deploy_actor", status="unexpected_error")
            return ActorDeploymentResult(
                success=False,
                actor_id="",
                status="FAILED",
                error=str(exc),
                message="Unexpected error during actor deployment",
            )

        actor_id = getattr(actor, "actor_id", None) or f"{actor_type}_{request.config.get('model_id', 'unknown')}"
        self._running_actors[actor_id] = actor
        metadata: dict[str, Any] = {"actor_type": actor_type}
        model_id_value = request.config.get("model_id")
        if isinstance(model_id_value, str):
            metadata["model_id"] = model_id_value
        if request.run_id is not None:
            metadata["run_id"] = request.run_id
        self._update_actor_state(
            actor_id,
            status="RUNNING",
            message="Actor deployed",
            metadata=metadata,
        )
        self._track_operation(operation="deploy_actor", status="success")
        return ActorDeploymentResult(
            success=True,
            actor_id=actor_id,
            status="DEPLOYED",
            message=f"Actor {actor_id} deployed successfully",
            error=None,
        )

    async def pause_actor(self, request: ActorPauseRequest) -> ActorLifecycleResult:
        """Pause a running actor without disposing it."""
        operation = "pause_actor"
        actor_id = request.actor_id
        self._track_operation(operation=operation, status="started")

        actor = self._running_actors.get(actor_id)
        if actor is None:
            existing_state = self._actor_states.get(actor_id)
            if existing_state is not None and existing_state.status == "PAUSED":
                actor_lifecycle_total.labels(operation="pause", status="noop").inc()
                self._track_operation(operation=operation, status="noop")
                return ActorLifecycleResult(
                    success=True,
                    actor_id=actor_id,
                    status="PAUSED",
                    message="Actor already paused",
                    error=None,
                )
            actor_lifecycle_total.labels(operation="pause", status="not_found").inc()
            self._track_operation(operation=operation, status="actor_not_found")
            return ActorLifecycleResult(
                success=False,
                actor_id=actor_id,
                status="NOT_FOUND",
                message=None,
                error=f"Actor {actor_id} not found",
            )

        try:
            await self._run_async(actor.stop)
        except Exception as exc:  # pragma: no cover - defensive
            error_message = str(exc)
            actor_lifecycle_total.labels(operation="pause", status="failed").inc()
            self._track_operation(operation=operation, status="failed")
            self._update_actor_state(actor_id, status="ERROR", error=error_message)
            return ActorLifecycleResult(
                success=False,
                actor_id=actor_id,
                status="FAILED",
                message="Unable to pause actor",
                error=error_message,
            )

        self._update_actor_state(
            actor_id,
            status="PAUSED",
            message=request.reason or "Actor paused",
            metadata=request.metadata,
        )
        actor_lifecycle_total.labels(operation="pause", status="success").inc()
        self._track_operation(operation=operation, status="success")
        return ActorLifecycleResult(
            success=True,
            actor_id=actor_id,
            status="PAUSED",
            message=request.reason or "Actor paused",
            error=None,
        )

    async def resume_actor(self, request: ActorResumeRequest) -> ActorLifecycleResult:
        """Resume a paused actor."""
        operation = "resume_actor"
        actor_id = request.actor_id
        self._track_operation(operation=operation, status="started")

        actor = self._running_actors.get(actor_id)
        if actor is None:
            actor_lifecycle_total.labels(operation="resume", status="not_found").inc()
            self._track_operation(operation=operation, status="actor_not_found")
            return ActorLifecycleResult(
                success=False,
                actor_id=actor_id,
                status="NOT_FOUND",
                message=None,
                error=f"Actor {actor_id} not found",
            )

        try:
            await self._run_async(actor.resume)
        except Exception as exc:  # pragma: no cover - defensive
            error_message = str(exc)
            actor_lifecycle_total.labels(operation="resume", status="failed").inc()
            self._track_operation(operation=operation, status="failed")
            self._update_actor_state(actor_id, status="ERROR", error=error_message)
            return ActorLifecycleResult(
                success=False,
                actor_id=actor_id,
                status="FAILED",
                message="Unable to resume actor",
                error=error_message,
            )

        self._update_actor_state(
            actor_id,
            status="RUNNING",
            message="Actor resumed",
            metadata=request.metadata,
        )
        actor_lifecycle_total.labels(operation="resume", status="success").inc()
        self._track_operation(operation=operation, status="success")
        return ActorLifecycleResult(
            success=True,
            actor_id=actor_id,
            status="RUNNING",
            message="Actor resumed",
            error=None,
        )

    async def stop_actor(self, request: ActorStopRequest) -> ActorLifecycleResult:
        """Stop and dispose of a running actor."""
        operation = "stop_actor"
        actor_id = request.actor_id
        self._track_operation(operation=operation, status="started")

        actor = self._running_actors.get(actor_id)
        if actor is None:
            existing_state = self._actor_states.get(actor_id)
            if existing_state is not None and existing_state.status == "STOPPED":
                actor_lifecycle_total.labels(operation="stop", status="noop").inc()
                self._track_operation(operation=operation, status="noop")
                return ActorLifecycleResult(
                    success=True,
                    actor_id=actor_id,
                    status="STOPPED",
                    message="Actor already stopped",
                    error=None,
                )
            actor_lifecycle_total.labels(operation="stop", status="not_found").inc()
            self._track_operation(operation=operation, status="actor_not_found")
            return ActorLifecycleResult(
                success=False,
                actor_id=actor_id,
                status="NOT_FOUND",
                message=None,
                error=f"Actor {actor_id} not found",
            )

        stop_error: str | None = None
        if request.force:
            degrade_fn = getattr(actor, "degrade", None)
            if callable(degrade_fn):
                try:
                    await self._run_async(degrade_fn)
                except Exception as exc:  # pragma: no cover - defensive
                    stop_error = str(exc)
            else:
                stop_error = "Actor does not support degrade()"
        else:
            try:
                await self._run_async(actor.stop)
            except Exception as exc:  # pragma: no cover - defensive
                stop_error = str(exc)

        dispose_error: str | None = None
        try:
            await self._run_async(actor.dispose)
        except Exception as exc:  # pragma: no cover - defensive
            dispose_error = str(exc)

        if stop_error or dispose_error:
            combined_error = "; ".join(filter(None, (stop_error, dispose_error)))
            actor_lifecycle_total.labels(operation="stop", status="failed").inc()
            self._track_operation(operation=operation, status="failed")
            self._update_actor_state(actor_id, status="ERROR", error=combined_error)
            return ActorLifecycleResult(
                success=False,
                actor_id=actor_id,
                status="FAILED",
                message="Unable to stop actor",
                error=combined_error,
            )

        self._running_actors.pop(actor_id, None)
        self._update_actor_state(
            actor_id,
            status="STOPPED",
            message=request.reason or "Actor stopped",
            metadata=request.metadata,
        )
        actor_lifecycle_total.labels(operation="stop", status="success").inc()
        self._track_operation(operation=operation, status="success")
        return ActorLifecycleResult(
            success=True,
            actor_id=actor_id,
            status="STOPPED",
            message=request.reason or "Actor stopped",
            error=None,
        )

    def _create_signal_actor(self, config_mapping: Mapping[str, Any]) -> BaseMLInferenceActor:
        """Instantiate an :class:`MLSignalActor` from a configuration mapping."""
        from ml.actors.signal import create_signal_actor
        from ml.config.actors import MLSignalActorConfig

        model_id = config_mapping.get("model_id")
        if not isinstance(model_id, str):
            raise ValueError("model_id is required for MLSignalActor deployment")

        integration = self._integration
        model_registry = getattr(integration, "model_registry", None) if integration else None
        if model_registry is not None and hasattr(model_registry, "get_model"):
            try:
                _ = model_registry.get_model(model_id)
            except Exception as exc:  # pragma: no cover - defensive
                actor_failures_total.labels(reason="model_lookup_failed").inc()
                logger.error(
                    "model lookup failed prior to actor deployment",
                    exc_info=True,
                    extra={"model_id": model_id},
                )
                raise ValueError(f"Model {model_id} unavailable") from exc

        actor_config = MLSignalActorConfig(**dict(config_mapping))
        return create_signal_actor(actor_config)

    async def hot_reload_model(self, actor_id: str, new_model_id: str) -> ActorHotReloadResult:
        """Hot reload a model for a running actor."""
        self._track_operation(operation="hot_reload", status="started")

        actor = self._running_actors.get(actor_id)
        if actor is None:
            self._track_operation(operation="hot_reload", status="actor_not_found")
            return ActorHotReloadResult(
                success=False,
                actor_id=actor_id,
                status="NOT_FOUND",
                error=f"Actor {actor_id} not found",
            )

        reloader = getattr(actor, "_execute_hot_reload", None)
        if callable(reloader):
            try:
                await self._run_async(partial(reloader, new_model_id))
                current_status = self._actor_states.get(actor_id)
                next_status = current_status.status if current_status else "RUNNING"
                self._update_actor_state(
                    actor_id,
                    status=next_status,
                    message=f"Hot reloaded to {new_model_id}",
                    metadata={"model_id": new_model_id},
                )
                result = ActorHotReloadResult(
                    success=True,
                    actor_id=actor_id,
                    new_model_id=new_model_id,
                    status="RELOADED",
                )
                self._track_operation(operation="hot_reload", status="success")
                return result
            except Exception as exc:  # pragma: no cover - defensive
                self._track_operation(operation="hot_reload", status="failed")
                return ActorHotReloadResult(
                    success=False,
                    actor_id=actor_id,
                    new_model_id=new_model_id,
                    status="FAILED",
                    error=str(exc),
                )

        self._track_operation(operation="hot_reload", status="unsupported")
        return ActorHotReloadResult(
            success=False,
            actor_id=actor_id,
            new_model_id=new_model_id,
            status="UNSUPPORTED",
            error="Actor does not support hot reload",
        )

    def _update_actor_state(
        self,
        actor_id: str,
        *,
        status: str,
        message: str | None = None,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        existing_state = self._actor_states.get(actor_id)
        merged_metadata: dict[str, Any]
        if existing_state is None:
            merged_metadata = {}
        else:
            merged_metadata = dict(existing_state.metadata)
        if metadata:
            merged_metadata.update(dict(metadata))

        self._actor_states[actor_id] = ActorLifecycleState(
            status=status,
            last_transition=dt.datetime.now(dt.UTC).isoformat(),
            message=message,
            error=error,
            metadata=merged_metadata,
        )


__all__ = [
    "ActorDeploymentRequest",
    "ActorDeploymentResult",
    "ActorHealthSnapshot",
    "ActorHotReloadResult",
    "ActorIntegrationService",
    "ActorLifecycleResult",
    "ActorPauseRequest",
    "ActorResumeRequest",
    "ActorStopRequest",
]
