"""System connector service implementing the dashboard "Connect System" flow."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram
from ml.dashboard.services.base_service import BaseIntegrationService


if TYPE_CHECKING:
    from ml.core.integration import MLIntegrationManager


logger = logging.getLogger(__name__)

_system_connect_total = get_counter(
    "ml_dashboard_system_connect_total",
    "System connection attempts triggered by the dashboard",
    labelnames=["status"],
)

_system_disconnect_total = get_counter(
    "ml_dashboard_system_disconnect_total",
    "System disconnect calls initiated by the dashboard",
    labelnames=["status"],
)

_system_connect_latency = get_histogram(
    "ml_dashboard_system_connect_latency_seconds",
    "Latency for system connection attempts",
    labelnames=["status"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

_system_component_connected = get_gauge(
    "ml_dashboard_system_component_connected",
    "Connection state for system components (1 connected, 0 disconnected)",
    labelnames=["component"],
)

_COMPONENT_LABELS: tuple[str, ...] = ("build", "data_engine", "exec_engine", "cache")
_CRITICAL_COMPONENTS: frozenset[str] = frozenset({"build", "data_engine", "exec_engine"})


class EngineConnectorProtocol(Protocol):
    """Protocol describing the subset used from engine connectors."""

    def connect(self) -> None:  # pragma: no cover - protocol definition
        """Connect to the underlying service dependency."""

    def disconnect(self) -> None:  # pragma: no cover - protocol definition
        """Disconnect from the underlying service dependency."""


class NodeKernelProtocol(Protocol):
    """Protocol for the TradingNode kernel attributes accessed by the service."""

    data_engine: EngineConnectorProtocol | None
    exec_engine: EngineConnectorProtocol | None
    cache: Any | None


class TradingNodeProtocol(Protocol):
    """Protocol capturing the TradingNode operations used by the service."""

    kernel: NodeKernelProtocol

    def build(self) -> None:  # pragma: no cover - protocol definition
        """Build internal clients for the node."""

    def stop(self) -> None:  # pragma: no cover - protocol definition
        """Stop the node (synchronous convenience wrapper)."""

    def dispose(self) -> None:  # pragma: no cover - protocol definition
        """Dispose node resources."""


@dataclass(slots=True)
class ComponentStatus:
    """Status payload describing the state of a system component."""

    component: str
    connected: bool
    latency_seconds: float | None = None
    error: str | None = None


@dataclass(slots=True)
class SystemConnectRequest:
    """Request model for initiating a system connection."""

    timeout_seconds: float = 30.0
    component_timeout_seconds: float = 10.0
    force_reconnect: bool = False


@dataclass(slots=True)
class SystemConnectResult:
    """Outcome payload returned after attempting to connect the system."""

    success: bool
    status: str
    timestamp: str
    components: tuple[ComponentStatus, ...]
    message: str | None = None
    error: str | None = None


@dataclass(slots=True)
class SystemDisconnectResult:
    """Outcome payload for disconnect operations."""

    success: bool
    status: str
    timestamp: str
    message: str | None = None
    error: str | None = None


@dataclass(slots=True)
class SystemStatusSnapshot:
    """Aggregated snapshot surfaced to health endpoints."""

    connected: bool
    last_connected: str | None
    last_checked: str
    components: tuple[ComponentStatus, ...]
    message: str | None = None
    error: str | None = None


class SystemConnectorService(BaseIntegrationService):
    """
    Service responsible for connecting the Nautilus trading system on demand.

    The dashboard "Connect System" button invokes this service which will in turn
    instantiate a :class:`nautilus_trader.live.node.TradingNode`, build its clients,
    and request connectivity for the data and execution engines. Each step is
    guarded by a timeout and emits Prometheus metrics for observability.

    Example
    -------
    >>> from ml.dashboard.services.system_service import SystemConnectorService, SystemConnectRequest
    >>> service = SystemConnectorService(integration_manager=None)
    >>> result = await service.connect_system(SystemConnectRequest(timeout_seconds=5.0))
    >>> result.success
    False
    >>> status = service.get_system_status()
    >>> status.connected
    False
    """

    def __init__(
        self,
        integration_manager: MLIntegrationManager | None,
        *,
        node_factory: Callable[[], TradingNodeProtocol] | None = None,
    ) -> None:
        super().__init__(integration_manager)
        self._node_factory: Callable[[], TradingNodeProtocol] = (
            node_factory if node_factory is not None else self._default_node_factory
        )
        self._trading_node: TradingNodeProtocol | None = None
        now = dt.datetime.now(dt.UTC).isoformat()
        self._status_snapshot = SystemStatusSnapshot(
            connected=False,
            last_connected=None,
            last_checked=now,
            components=tuple(),
            message="System disconnected",
            error=None,
        )
        for component in _COMPONENT_LABELS:
            _system_component_connected.labels(component=component).set(0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_service_name(self) -> str:
        return "system_connector"

    async def health_check(self) -> dict[str, Any]:
        snapshot = self.get_system_status()
        return {
            "connected": snapshot.connected,
            "last_connected": snapshot.last_connected,
            "last_checked": snapshot.last_checked,
            "components": [asdict(component) for component in snapshot.components],
            "message": snapshot.message,
            "error": snapshot.error,
        }

    async def connect_system(
        self,
        request: SystemConnectRequest | None = None,
    ) -> SystemConnectResult:
        req = request or SystemConnectRequest()
        self._track_operation(operation="connect_system", status="started")
        start = time.perf_counter()
        timestamp = dt.datetime.now(dt.UTC).isoformat()
        status_label = "error"

        if self._trading_node is not None and not req.force_reconnect:
            snapshot = self.get_system_status()
            status_label = "already_connected" if snapshot.connected else "stale"
            _system_connect_total.labels(status=status_label).inc()
            _system_connect_latency.labels(status=status_label).observe(0.0)
            self._track_operation(operation="connect_system", status=status_label)
            return SystemConnectResult(
                success=snapshot.connected,
                status="ALREADY_CONNECTED" if snapshot.connected else "STALE",
                timestamp=timestamp,
                components=snapshot.components,
                message="System already connected" if snapshot.connected else snapshot.message,
                error=snapshot.error,
            )

        if req.force_reconnect and self._trading_node is not None:
            await self._disconnect_internal(timeout=req.component_timeout_seconds)

        deadline = time.perf_counter() + max(req.timeout_seconds, 0.1)
        components: list[ComponentStatus] = []
        node: TradingNodeProtocol | None = None
        error_message: str | None = None

        try:
            node = await asyncio.wait_for(self._run_async(self._node_factory), req.timeout_seconds)
        except TimeoutError:
            error_message = "Timed out instantiating TradingNode"
            status_label = "timeout"
        except Exception as exc:  # pragma: no cover - defensive
            error_message = str(exc)
            status_label = "init_failed"
            logger.debug("trading node instantiation failed", exc_info=True)

        if node is None:
            components_tuple = tuple(components)
            self._update_status(
                connected=False,
                message="System connection failed",
                components=components_tuple,
                timestamp=timestamp,
                error=error_message,
            )
            _system_connect_total.labels(status=status_label).inc()
            _system_connect_latency.labels(status=status_label).observe(time.perf_counter() - start)
            self._track_operation(operation="connect_system", status=status_label)
            return SystemConnectResult(
                success=False,
                status="FAILED",
                timestamp=timestamp,
                components=components_tuple,
                message="System connection failed",
                error=error_message,
            )

        try:
            components.extend(
                await self._connect_components(
                    node=node,
                    deadline=deadline,
                    component_timeout=req.component_timeout_seconds,
                )
            )
        except TimeoutError:
            error_message = "System connection timed out"
            status_label = "timeout"
        except Exception as exc:  # pragma: no cover - defensive
            error_message = str(exc)
            status_label = "error"
            logger.debug("system component connection failure", exc_info=True)

        success = error_message is None and all(
            status.connected for status in components if status.component in _CRITICAL_COMPONENTS
        )

        if success:
            self._trading_node = node
            status_str = "CONNECTED"
            status_label = "success"
            message = "System connected successfully"
            error_message = None
        else:
            await self._disconnect_internal(node=node, timeout=req.component_timeout_seconds)
            status_str = "DEGRADED"
            message = "System connected in degraded state" if components else "System connection failed"
            if error_message is None:
                failed_components = [s.component for s in components if not s.connected]
                error_message = "; ".join(failed_components) if failed_components else "Unknown failure"

        components_tuple = tuple(components)
        self._update_status(
            connected=success,
            message=message,
            components=components_tuple,
            timestamp=timestamp,
            error=error_message,
        )
        _system_connect_total.labels(status=status_label).inc()
        _system_connect_latency.labels(status=status_label).observe(time.perf_counter() - start)
        self._track_operation(operation="connect_system", status=status_label)
        return SystemConnectResult(
            success=success,
            status=status_str,
            timestamp=timestamp,
            components=components_tuple,
            message=message,
            error=error_message,
        )

    async def disconnect_system(self) -> SystemDisconnectResult:
        self._track_operation(operation="disconnect_system", status="started")
        timestamp = dt.datetime.now(dt.UTC).isoformat()
        status_label = "success"

        if self._trading_node is None:
            _system_disconnect_total.labels(status="noop").inc()
            self._track_operation(operation="disconnect_system", status="noop")
            self._update_status(
                connected=False,
                message="System disconnected",
                components=tuple(),
                timestamp=timestamp,
                error=None,
            )
            return SystemDisconnectResult(
                success=True,
                status="NOOP",
                timestamp=timestamp,
                message="System already disconnected",
                error=None,
            )

        try:
            await self._disconnect_internal(timeout=10.0)
        except Exception as exc:  # pragma: no cover - defensive
            status_label = "error"
            logger.debug("system disconnect raised", exc_info=True)
            _system_disconnect_total.labels(status=status_label).inc()
            self._track_operation(operation="disconnect_system", status=status_label)
            self._update_status(
                connected=False,
                message="System disconnect encountered errors",
                components=tuple(),
                timestamp=timestamp,
                error=str(exc),
            )
            return SystemDisconnectResult(
                success=False,
                status="ERROR",
                timestamp=timestamp,
                message="System disconnect encountered errors",
                error=str(exc),
            )

        _system_disconnect_total.labels(status=status_label).inc()
        self._track_operation(operation="disconnect_system", status=status_label)
        self._update_status(
            connected=False,
            message="System disconnected",
            components=tuple(),
            timestamp=timestamp,
            error=None,
        )
        return SystemDisconnectResult(
            success=True,
            status="DISCONNECTED",
            timestamp=timestamp,
            message="System disconnected",
            error=None,
        )

    def get_system_status(self) -> SystemStatusSnapshot:
        snapshot = self._status_snapshot
        return SystemStatusSnapshot(
            connected=snapshot.connected,
            last_connected=snapshot.last_connected,
            last_checked=snapshot.last_checked,
            components=tuple(snapshot.components),
            message=snapshot.message,
            error=snapshot.error,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _connect_components(
        self,
        *,
        node: TradingNodeProtocol,
        deadline: float,
        component_timeout: float,
    ) -> list[ComponentStatus]:
        statuses: list[ComponentStatus] = []

        if hasattr(node, "build"):
            statuses.append(
                await self._invoke_component(
                    component="build",
                    func=node.build,
                    deadline=deadline,
                    component_timeout=component_timeout,
                )
            )

        data_engine = getattr(node.kernel, "data_engine", None)
        if data_engine is not None and hasattr(data_engine, "connect"):
            statuses.append(
                await self._invoke_component(
                    component="data_engine",
                    func=data_engine.connect,
                    deadline=deadline,
                    component_timeout=component_timeout,
                )
            )
        else:
            statuses.append(ComponentStatus(component="data_engine", connected=False, error="Missing data engine"))
            _system_component_connected.labels(component="data_engine").set(0.0)

        exec_engine = getattr(node.kernel, "exec_engine", None)
        if exec_engine is not None and hasattr(exec_engine, "connect"):
            statuses.append(
                await self._invoke_component(
                    component="exec_engine",
                    func=exec_engine.connect,
                    deadline=deadline,
                    component_timeout=component_timeout,
                )
            )
        else:
            statuses.append(ComponentStatus(component="exec_engine", connected=False, error="Missing exec engine"))
            _system_component_connected.labels(component="exec_engine").set(0.0)

        cache = getattr(node.kernel, "cache", None)
        cache_connected = cache is not None
        cache_status = ComponentStatus(
            component="cache",
            connected=cache_connected,
            latency_seconds=0.0 if cache_connected else None,
            error=None if cache_connected else "Cache unavailable",
        )
        statuses.append(cache_status)
        _system_component_connected.labels(component="cache").set(1.0 if cache_connected else 0.0)

        return statuses

    async def _disconnect_internal(
        self,
        *,
        node: TradingNodeProtocol | None = None,
        timeout: float,
    ) -> None:
        target = node if node is not None else self._trading_node
        if target is None:
            return
        self._trading_node = None

        disconnect_timeout = max(timeout, 0.1)

        try:
            await asyncio.wait_for(self._run_async(target.stop), disconnect_timeout)
        except Exception:  # pragma: no cover - defensive
            logger.debug("trading node stop failed", exc_info=True)

        try:
            await asyncio.wait_for(self._run_async(target.dispose), disconnect_timeout)
        except Exception:  # pragma: no cover - defensive
            logger.debug("trading node dispose failed", exc_info=True)

        for component in _COMPONENT_LABELS:
            _system_component_connected.labels(component=component).set(0.0)

    async def _invoke_component(
        self,
        *,
        component: str,
        func: Callable[[], Any],
        deadline: float,
        component_timeout: float,
    ) -> ComponentStatus:
        start = time.perf_counter()
        try:
            timeout_seconds = self._compute_timeout(deadline=deadline, component_timeout=component_timeout)
            await asyncio.wait_for(self._run_async(func), timeout_seconds)
        except TimeoutError:
            latency = time.perf_counter() - start
            _system_component_connected.labels(component=component).set(0.0)
            return ComponentStatus(
                component=component,
                connected=False,
                latency_seconds=latency,
                error="timeout",
            )
        except Exception as exc:  # pragma: no cover - defensive
            latency = time.perf_counter() - start
            _system_component_connected.labels(component=component).set(0.0)
            logger.debug("component connection failed", exc_info=True, extra={"component": component})
            return ComponentStatus(
                component=component,
                connected=False,
                latency_seconds=latency,
                error=str(exc),
            )

        latency = time.perf_counter() - start
        _system_component_connected.labels(component=component).set(1.0)
        return ComponentStatus(
            component=component,
            connected=True,
            latency_seconds=latency,
            error=None,
        )

    def _compute_timeout(self, *, deadline: float, component_timeout: float) -> float:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("Connection deadline exceeded")
        timeout_seconds = min(max(component_timeout, 0.1), remaining)
        return timeout_seconds

    def _update_status(
        self,
        *,
        connected: bool,
        message: str | None,
        components: Sequence[ComponentStatus],
        timestamp: str,
        error: str | None,
    ) -> None:
        last_connected = timestamp if connected else self._status_snapshot.last_connected
        self._status_snapshot = SystemStatusSnapshot(
            connected=connected,
            last_connected=last_connected,
            last_checked=timestamp,
            components=tuple(components),
            message=message,
            error=error,
        )

    def _default_node_factory(self) -> TradingNodeProtocol:
        from nautilus_trader.config import TradingNodeConfig
        from nautilus_trader.live.node import TradingNode

        config = TradingNodeConfig()
        return cast(TradingNodeProtocol, TradingNode(config=config))


__all__ = [
    "ComponentStatus",
    "SystemConnectRequest",
    "SystemConnectResult",
    "SystemConnectorService",
    "SystemDisconnectResult",
    "SystemStatusSnapshot",
]
