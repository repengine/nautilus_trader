"""
Dashboard service implementation providing health aggregation and control actions.

All operations are cold-path only. Metrics are recorded via the centralized metrics
bootstrap. Logging uses structlog configuration from ml.common.

"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from threading import Lock
from threading import Event as ThreadEvent
from threading import Thread
from typing import Any, Generic, TypeVar, cast
from urllib.parse import urljoin

import requests

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.common.message_bus import publisher_from_config
from ml.common.message_topics import build_stage_topic
from ml.common.message_topics import build_topic_for_stage
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.common.retry_utils import retry_with_backoff
from ml.config.bus import MessageBusConfig
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.dashboard.config import DashboardConfig
from ml.dashboard.controllers import ComposeServiceController
from ml.dashboard.controllers import NoopServiceController
from ml.dashboard.controllers import ServiceControllerProtocol
from ml.dashboard.exceptions import ServiceControlUnsupportedError
from ml.dashboard.grafana import GrafanaConfig
from ml.dashboard.grafana import provision_dashboard
from ml.registry import BackendType
from ml.registry import DataRegistry
from ml.registry import DatasetLineageRecord
from ml.registry import FeatureRegistry
from ml.registry import ModelInfo
from ml.registry import ModelRegistry
from ml.registry import PersistenceConfig
from ml.registry import StrategyRegistry
from ml.registry import Watermark
from ml.registry.base import DummyRegistry
from ml.registry.dataclasses import QualityGate
from ml.registry.feature_registry import FeatureStage


logger = logging.getLogger(__name__)


_REQS_TOTAL = get_counter(
    "ml_dashboard_requests_total",
    "Total dashboard API requests",
    labels=["route", "method", "status"],
)
_LATENCY_SECONDS = get_histogram(
    "ml_dashboard_latency_seconds",
    "Dashboard API latency (seconds)",
    labels=["route"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

_REGISTRY_CACHE_HITS = get_counter(
    "ml_dashboard_registry_cache_hits_total",
    "Dashboard registry cache hits",
    labels=["entry"],
)
_REGISTRY_CACHE_MISSES = get_counter(
    "ml_dashboard_registry_cache_misses_total",
    "Dashboard registry cache misses",
    labels=["entry"],
)
_REGISTRY_FALLBACK_TOTAL = get_counter(
    "ml_dashboard_registry_fallback_total",
    "Dashboard registry fallback activations",
    labels=["registry", "reason"],
)
_REGISTRY_RETRY_TOTAL = get_counter(
    "ml_dashboard_registry_retry_total",
    "Dashboard registry retry attempts",
    labels=["registry"],
)

_EVENT_CACHE_HITS = get_counter(
    "ml_dashboard_events_cache_hits_total",
    "Dashboard events cache hits",
)
_EVENT_CACHE_MISSES = get_counter(
    "ml_dashboard_events_cache_misses_total",
    "Dashboard events cache misses",
)
_EVENT_POLLS_TOTAL = get_counter(
    "ml_dashboard_events_poll_total",
    "Dashboard events poll attempts",
)
_EVENT_FAILURES_TOTAL = get_counter(
    "ml_dashboard_events_failure_total",
    "Dashboard events polling failures",
    labels=["reason"],
)


_CacheValueT = TypeVar("_CacheValueT")


@dataclass(slots=True)
class _CacheEntry(Generic[_CacheValueT]):
    """Cache entry containing the value and its monotonic expiry."""

    value: _CacheValueT
    expires_at: float


@dataclass(slots=True)
class _TTLCache(Generic[_CacheValueT]):
    """Simple TTL cache intended for cold-path dashboard usage."""

    ttl_seconds: float
    max_entries: int
    _clock: Callable[[], float] = time.monotonic
    _entries: dict[str, _CacheEntry[_CacheValueT]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def get(self, key: str) -> _CacheValueT | None:
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop(key, None)
                return None
            return entry.value

    def put(self, key: str, value: _CacheValueT) -> None:
        expires_at = self._clock() + self.ttl_seconds
        with self._lock:
            if key not in self._entries and len(self._entries) >= self.max_entries:
                self._evict_locked()
            self._entries[key] = _CacheEntry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._entries.keys())

    def _evict_locked(self) -> None:
        if not self._entries:
            return
        lru_key = min(self._entries.items(), key=lambda item: item[1].expires_at)[0]
        self._entries.pop(lru_key, None)


def _env_allows_dummy() -> bool:
    """Return True when environment permits dummy fallback."""
    import os

    value = os.getenv("ML_ALLOW_DUMMY", "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _safe_get(url: str, timeout: float) -> tuple[bool, int]:
    def _call() -> tuple[bool, int]:
        resp = requests.get(url, timeout=timeout)
        return resp.ok, resp.status_code

    try:
        return retry_with_backoff(
            _call,
            max_attempts=3,
            initial_delay=0.1,
            max_delay=0.5,
            jitter=0.2,
        )
    except Exception:
        logger.debug("health probe request failed", extra={"url": url}, exc_info=True)
        return False, 0


def _to_url(host_port: int, path: str) -> str:
    return urljoin(f"http://localhost:{host_port}/", path.lstrip("/"))


@dataclass
class _EventCache:
    """Bounded TTL cache for dashboard event history."""

    ttl_seconds: float
    max_entries: int
    _clock: Callable[[], float] = time.monotonic
    _events: list[dict[str, Any]] = field(default_factory=list)
    _expires_at: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    def snapshot(self) -> tuple[list[dict[str, Any]], bool]:
        """Return cached events and whether they remain fresh."""

        now = self._clock()
        with self._lock:
            is_fresh = bool(self._events) and now < self._expires_at
            return list(self._events), is_fresh

    def update(self, events: list[dict[str, Any]]) -> None:
        trimmed = list(events[: self.max_entries])
        with self._lock:
            self._events = trimmed
            self._expires_at = self._clock() + self.ttl_seconds

    def stale_snapshot(self) -> list[dict[str, Any]]:
        """Return cached events irrespective of expiry (best effort)."""

        with self._lock:
            return list(self._events)


class DashboardService:
    """
    Service providing dashboard operations.
    """

    config: DashboardConfig
    controller: ServiceControllerProtocol
    _model_registry: ModelRegistry | None = field(default=None, init=False)
    _feature_registry: FeatureRegistry | None = field(default=None, init=False)
    _strategy_registry: StrategyRegistry | None = field(default=None, init=False)
    _data_registry: DataRegistry | None = field(default=None, init=False)
    _registry_cache: _TTLCache[object] = field(
        default_factory=lambda: _TTLCache(ttl_seconds=30.0, max_entries=32),
        init=False,
    )
    _allow_dummy_fallback: bool = field(default_factory=_env_allows_dummy, init=False)
    _event_cache: _EventCache = field(init=False)
    _event_poll_thread: Thread | None = field(default=None, init=False)
    _event_poll_stop: ThreadEvent | None = field(default=None, init=False)

    @classmethod
    def from_config(cls, config: DashboardConfig) -> DashboardService:
        controller: ServiceControllerProtocol
        if config.compose_enabled:
            controller = ComposeServiceController(config.compose_file)
        else:
            controller = NoopServiceController()
        return cls(config=config, controller=controller)

    def __post_init__(self) -> None:
        self._event_cache = _EventCache(
            ttl_seconds=self.config.events_cache_ttl_seconds,
            max_entries=self.config.events_cache_max_entries,
        )

    # -----------------
    # Health & metadata
    # -----------------
    def get_system_health(self) -> dict[str, Any]:
        """
        Aggregate health across core services and dependencies.

        This is a read-only aggregation that pings known endpoints for liveness.

        """
        start = time.perf_counter()
        route = "/api/health/system"
        try:
            cfg = self.config
            health: dict[str, Any] = {
                "services": {},
                "dependencies": {},
            }

            # ML services
            for name, port, hpath in (
                ("ml_signal_actor", cfg.actor_port, "/health"),
                ("ml_strategy", cfg.strategy_port, "/health"),
                ("ml_pipeline", cfg.pipeline_port, "/health"),
            ):
                ok, code = _safe_get(_to_url(port, hpath), cfg.request_timeout_seconds)
                health["services"][name] = {"healthy": ok, "status_code": code}

            # Observability
            ok_prom, code_prom = _safe_get(
                f"http://localhost:{cfg.prometheus_port}/-/healthy",
                cfg.request_timeout_seconds,
            )
            ok_graf, code_graf = _safe_get(
                f"http://localhost:{cfg.grafana_port}/api/health",
                cfg.request_timeout_seconds,
            )
            health["dependencies"]["prometheus"] = {"healthy": ok_prom, "status_code": code_prom}
            health["dependencies"]["grafana"] = {"healthy": ok_graf, "status_code": code_graf}

            return health
        finally:
            _LATENCY_SECONDS.labels(route=route).observe(time.perf_counter() - start)

    def list_services(self) -> list[dict[str, Any]]:
        cfg = self.config
        return [
            {
                "name": "ml_signal_actor",
                "ports": {"http": cfg.actor_port},
                "endpoints": {
                    "health": _to_url(cfg.actor_port, "/health"),
                    "metrics": _to_url(cfg.actor_port, "/metrics"),
                },
            },
            {
                "name": "ml_strategy",
                "ports": {"http": cfg.strategy_port},
                "endpoints": {
                    "health": _to_url(cfg.strategy_port, "/health"),
                    "metrics": _to_url(cfg.strategy_port, "/metrics"),
                },
            },
            {
                "name": "ml_pipeline",
                "ports": {"http": cfg.pipeline_port},
                "endpoints": {
                    "health": _to_url(cfg.pipeline_port, "/health"),
                    "metrics": _to_url(cfg.pipeline_port, "/metrics"),
                },
            },
        ]

    # -----------------
    # Registries (read/promote)
    # -----------------
    def _get_model_registry(self) -> ModelRegistry | None:
        if self._model_registry is not None:
            return self._model_registry
        registry = cast(
            ModelRegistry | None,
            self._build_registry(name="model", builder=self._build_model_registry),
        )
        self._model_registry = registry
        return registry

    def _get_feature_registry(self) -> FeatureRegistry | None:
        if self._feature_registry is not None:
            return self._feature_registry
        registry = cast(
            FeatureRegistry | None,
            self._build_registry(name="feature", builder=self._build_feature_registry),
        )
        self._feature_registry = registry
        return registry

    def _get_strategy_registry(self) -> StrategyRegistry | None:
        if self._strategy_registry is not None:
            return self._strategy_registry
        registry = cast(
            StrategyRegistry | None,
            self._build_registry(name="strategy", builder=self._build_strategy_registry),
        )
        self._strategy_registry = registry
        return registry

    def _get_data_registry(self) -> DataRegistry | None:
        if self._data_registry is not None:
            return self._data_registry
        registry = cast(
            DataRegistry | None,
            self._build_registry(name="data", builder=self._build_data_registry),
        )
        self._data_registry = registry
        return registry

    def _build_registry(
        self,
        *,
        name: str,
        builder: Callable[[], object | None],
    ) -> object | None:
        def _on_exception(attempt_index: int, exc: BaseException) -> None:
            _REGISTRY_RETRY_TOTAL.labels(registry=name).inc()
            logger.debug(
                "registry init retry",
                extra={"registry": name, "attempt": attempt_index + 1},
                exc_info=True,
            )

        registry: object | None = None
        try:
            registry = retry_with_backoff(
                builder,
                max_attempts=3,
                initial_delay=0.2,
                max_delay=1.0,
                jitter=0.25,
                on_exception=_on_exception,
            )
        except Exception:
            self._record_registry_error(name=name, reason="init_failed")
            logger.warning("registry init failed", extra={"registry": name}, exc_info=True)

        if registry is None:
            if self._allow_dummy_fallback:
                self._record_registry_error(name=name, reason="dummy_registry")
                logger.warning(
                    "using dummy registry fallback",
                    extra={"registry": name},
                )
                return DummyRegistry()
            return None
        return registry

    def _build_model_registry(self) -> ModelRegistry | None:
        import os

        db = os.getenv("DB_CONNECTION") or os.getenv("DATABASE_URL")
        if db:
            from pathlib import Path as _Path

            reg_path = _Path("./ml_registry/models")
            pc = PersistenceConfig(backend=BackendType.POSTGRES, connection_string=db)
        else:
            from pathlib import Path as _Path

            reg_path = _Path("./ml_registry/models")
            pc = PersistenceConfig(backend=BackendType.JSON, json_path=reg_path)
        return ModelRegistry(registry_path=reg_path, persistence_config=pc)

    def _build_feature_registry(self) -> FeatureRegistry | None:
        import os
        from pathlib import Path as _Path

        db = os.getenv("DB_CONNECTION") or os.getenv("DATABASE_URL")
        reg_path = _Path("./ml_registry/features")
        if db:
            pc = PersistenceConfig(backend=BackendType.POSTGRES, connection_string=db)
        else:
            pc = PersistenceConfig(backend=BackendType.JSON, json_path=reg_path)
        return FeatureRegistry(registry_path=reg_path, persistence_config=pc)

    def _build_strategy_registry(self) -> StrategyRegistry | None:
        import os
        from pathlib import Path as _Path

        db = os.getenv("DB_CONNECTION") or os.getenv("DATABASE_URL")
        base_path = _Path("./ml_registry/strategies")
        if db:
            pc = PersistenceConfig(backend=BackendType.POSTGRES, connection_string=db)
        else:
            pc = PersistenceConfig(backend=BackendType.JSON, json_path=base_path)
        return StrategyRegistry(base_path=base_path, persistence_config=pc)

    def _build_data_registry(self) -> DataRegistry | None:
        import os
        from pathlib import Path as _Path

        db = os.getenv("DB_CONNECTION") or os.getenv("DATABASE_URL")
        path = _Path("./ml_registry/datasets")
        if db:
            pc = PersistenceConfig(backend=BackendType.POSTGRES, connection_string=db)
        else:
            pc = PersistenceConfig(backend=BackendType.JSON, json_path=path)
        return DataRegistry(registry_path=path, persistence_config=pc)

    def _poll_events(self, *, limit: int) -> list[dict[str, Any]]:
        cfg = MessageBusConfig.from_env()
        if not cfg.enabled or cfg.backend != "redis":
            raise RuntimeError("events_disabled")
        try:
            import redis

            client: Any = redis.Redis.from_url(cfg.redis_url, decode_responses=True)
            rows: list[tuple[str, dict[str, str]]] = client.xrevrange(
                cfg.redis_stream,
                count=max(1, int(limit)),
            )
        except RuntimeError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("events_error") from exc

        events: list[dict[str, Any]] = []
        for entry_id, fields in rows:
            topic = fields.get("topic", "")
            payload_raw = fields.get("payload", "{}")
            try:
                payload = json.loads(payload_raw)
            except Exception:
                payload = {"raw": payload_raw}
            events.append({"id": entry_id, "topic": topic, "payload": payload})
        return events

    def _cache_key(self, entry: str, *parts: str | None) -> str:
        safe_parts = [part or "*" for part in parts]
        return f"{entry}:{'|'.join(safe_parts)}"

    def _cached_registry_call(
        self,
        *,
        key: str,
        fetch: Callable[[], _CacheValueT],
    ) -> _CacheValueT:
        cached = self._registry_cache.get(key)
        if cached is not None:
            _REGISTRY_CACHE_HITS.labels(entry=key).inc()
            return cast(_CacheValueT, cached)

        _REGISTRY_CACHE_MISSES.labels(entry=key).inc()
        value = fetch()
        self._registry_cache.put(key, value)
        return value

    def _invalidate_cache(self, *keys: str) -> None:
        if not keys:
            return
        existing_keys = self._registry_cache.keys()
        for key in keys:
            if key.endswith("*"):
                prefix = key[:-1]
                for existing in existing_keys:
                    if existing.startswith(prefix):
                        self._registry_cache.delete(existing)
            else:
                if key.endswith(":"):
                    for existing in existing_keys:
                        if existing.startswith(key):
                            self._registry_cache.delete(existing)
                else:
                    self._registry_cache.delete(key)

    def _record_registry_error(self, *, name: str, reason: str) -> None:
        _REGISTRY_FALLBACK_TOTAL.labels(registry=name, reason=reason).inc()

    def start_event_polling(self, interval_seconds: float) -> None:
        if interval_seconds <= 0.0 or self._event_poll_thread is not None:
            return
        stop = ThreadEvent()
        self._event_poll_stop = stop

        def _run() -> None:
            while not stop.wait(interval_seconds):
                try:
                    events = self._poll_events(limit=self.config.events_cache_max_entries)
                    _EVENT_POLLS_TOTAL.inc()
                    self._event_cache.update(events)
                except RuntimeError as exc:
                    reason = "disabled" if str(exc) == "events_disabled" else "error"
                    _EVENT_FAILURES_TOTAL.labels(reason=reason).inc()
                except Exception:
                    _EVENT_FAILURES_TOTAL.labels(reason="error").inc()
                    logger.debug("background event poll failed", exc_info=True)

        thread = Thread(target=_run, name="ml-dashboard-event-poll", daemon=True)
        thread.start()
        self._event_poll_thread = thread

    def stop_event_polling(self) -> None:
        if self._event_poll_stop is not None:
            self._event_poll_stop.set()
        self._event_poll_thread = None
        self._event_poll_stop = None

    def list_models(self) -> list[dict[str, Any]]:
        cache_key = self._cache_key("models")

        def _fetch() -> list[dict[str, Any]]:
            reg = self._get_model_registry()
            if reg is None:
                self._record_registry_error(name="model", reason="unavailable")
                return []
            try:
                models: list[ModelInfo] | None = reg.get_all_models()
            except Exception:
                self._record_registry_error(name="model", reason="list_failed")
                logger.warning("list models failed", exc_info=True)
                return []
            out: list[dict[str, Any]] = []
            for mi in models or []:
                out.append(
                    {
                        "model_id": mi.manifest.model_id,
                        "role": mi.manifest.role.value,
                        "version": mi.manifest.version,
                        "deployment_status": mi.deployment_status.value,
                        "deployed_to": list(mi.deployed_to),
                        "architecture": mi.manifest.architecture,
                        "feature_schema_hash": mi.manifest.feature_schema_hash,
                    },
                )
            return out

        return self._cached_registry_call(key=cache_key, fetch=_fetch)

    def list_deployments(self) -> dict[str, list[str]]:
        cache_key = self._cache_key("deployments")

        def _fetch() -> dict[str, list[str]]:
            reg = self._get_model_registry()
            if reg is None:
                self._record_registry_error(name="model", reason="deployments_unavailable")
                return {}
            try:
                active = reg.get_active_models()
            except Exception:
                self._record_registry_error(name="model", reason="deployments_failed")
                logger.warning("list deployments failed", exc_info=True)
                return {}
            deployments: dict[str, list[str]] = {}
            for mi in active or []:
                for tgt in mi.deployed_to:
                    deployments.setdefault(tgt, []).append(mi.manifest.model_id)
            return deployments

        return self._cached_registry_call(key=cache_key, fetch=_fetch)

    def deploy_model(self, model_id: str, target: str) -> dict[str, Any]:
        reg = self._get_model_registry()
        if reg is None:
            self._record_registry_error(name="model", reason="deploy_unavailable")
            return {"ok": False, "model_id": model_id, "target": target}
        ok = False
        try:
            ok = reg.deploy_model(model_id, target)
        except Exception:
            self._record_registry_error(name="model", reason="deploy_failed")
            logger.warning("deploy model failed", exc_info=True)
            ok = False
        if ok:
            self._invalidate_cache(self._cache_key("models"), self._cache_key("deployments"))
        return {"ok": ok, "model_id": model_id, "target": target}

    def hot_reload_model(self, target: str, new_model_id: str) -> dict[str, Any]:
        """
        Hot reload a deployment with a new model id.
        """
        reg = self._get_model_registry()
        if reg is None:
            self._record_registry_error(name="model", reason="hot_reload_unavailable")
            return {"ok": False, "target": target, "model_id": new_model_id}
        ok = False
        try:
            ok = reg.hot_reload_model(target=target, new_model_id=new_model_id)
        except Exception:
            self._record_registry_error(name="model", reason="hot_reload_failed")
            logger.warning("hot reload model failed", exc_info=True)
            ok = False
        if ok:
            self._invalidate_cache(self._cache_key("models"), self._cache_key("deployments"))
        return {"ok": ok, "target": target, "model_id": new_model_id}

    def rollback_deployment(self, target: str, to_model_id: str) -> dict[str, Any]:
        """
        Rollback a target to a specific model id.
        """
        reg = self._get_model_registry()
        if reg is None:
            self._record_registry_error(name="model", reason="rollback_unavailable")
            return {"ok": False, "target": target, "model_id": to_model_id}
        ok = False
        try:
            ok = reg.rollback(target=target, to_model_id=to_model_id)
        except Exception:
            self._record_registry_error(name="model", reason="rollback_failed")
            logger.warning("rollback model failed", exc_info=True)
            ok = False
        if ok:
            self._invalidate_cache(self._cache_key("models"), self._cache_key("deployments"))
        return {"ok": ok, "target": target, "model_id": to_model_id}

    # -----------------
    # Events (read-only)
    # -----------------
    def list_events(
        self,
        *,
        limit: int = 100,
        stage: str | None = None,
        source: str | None = None,
        instrument_substr: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return recent events from Redis Streams when bus is enabled.

        Best-effort; returns empty list if disabled or unavailable.

        """
        limit_value = max(1, int(limit))
        cached, is_fresh = self._event_cache.snapshot()
        events = cached
        if is_fresh:
            _EVENT_CACHE_HITS.inc()
        else:
            _EVENT_CACHE_MISSES.inc()
            try:
                polled = self._poll_events(limit=max(limit_value, self.config.events_cache_max_entries))
                _EVENT_POLLS_TOTAL.inc()
                self._event_cache.update(polled)
                events = polled
            except RuntimeError as exc:
                reason = "disabled" if str(exc) == "events_disabled" else "error"
                _EVENT_FAILURES_TOTAL.labels(reason=reason).inc()
                events = cached
            except Exception:
                _EVENT_FAILURES_TOTAL.labels(reason="error").inc()
                logger.debug("event polling failed", exc_info=True)
                events = cached

        out: list[dict[str, Any]] = []
        for entry in events:
            topic = entry.get("topic", "")
            payload = entry.get("payload", {})
            if source is not None and str(payload.get("source")) != source:
                continue
            if instrument_substr:
                instrument = None
                if isinstance(payload, dict):
                    params = payload.get("params")
                    if isinstance(params, dict):
                        instrument = params.get("instrument")
                if not instrument or instrument_substr not in str(instrument):
                    if instrument_substr not in topic:
                        continue
            if stage and stage not in topic:
                continue
            out.append(entry)
            if len(out) >= limit_value:
                break
        return out

    # -----------------
    # Feature/Strategy/Data registry listings
    # -----------------
    def list_features(
        self,
        *,
        role: str | None = None,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        cache_key = self._cache_key("features", role, stage)

        def _fetch() -> list[dict[str, Any]]:
            reg = self._get_feature_registry()
            if reg is None:
                self._record_registry_error(name="feature", reason="unavailable")
                return []
            try:
                infos = reg.list_all()
            except Exception:
                self._record_registry_error(name="feature", reason="list_failed")
                logger.warning("list features failed", exc_info=True)
                return []
            out: list[dict[str, Any]] = []
            for fi in (infos or []):
                if role and fi.manifest.role.value != role:
                    continue
                if stage and fi.manifest.stage.value != stage:
                    continue
                out.append(
                    {
                        "feature_set_id": fi.manifest.feature_set_id,
                        "role": fi.manifest.role.value,
                        "stage": fi.manifest.stage.value,
                        "schema_hash": fi.manifest.schema_hash,
                        "version": fi.manifest.version,
                    },
                )
            return out

        return self._cached_registry_call(key=cache_key, fetch=_fetch)

    def list_strategies(self) -> list[dict[str, Any]]:
        cache_key = self._cache_key("strategies")

        def _fetch() -> list[dict[str, Any]]:
            reg = self._get_strategy_registry()
            if reg is None:
                self._record_registry_error(name="strategy", reason="unavailable")
                return []
            try:
                strategies = reg.list_strategies()
            except Exception:
                self._record_registry_error(name="strategy", reason="list_failed")
                logger.warning("list strategies failed", exc_info=True)
                return []
            result: list[dict[str, Any]] = []
            for sinfo in strategies or []:
                manifest = sinfo.manifest
                result.append(
                    {
                        "strategy_id": manifest.strategy_id,
                        "type": manifest.strategy_type.value,
                        "version": manifest.version,
                        "required_models": manifest.required_models or [],
                    },
                )
            return result

        return self._cached_registry_call(key=cache_key, fetch=_fetch)

    def list_datasets(self) -> list[dict[str, Any]]:
        cache_key = self._cache_key("datasets")

        def _fetch() -> list[dict[str, Any]]:
            reg = self._get_data_registry()
            if reg is None:
                self._record_registry_error(name="data", reason="unavailable")
                return []
            try:
                manifests = reg.list_manifests()
            except Exception:
                self._record_registry_error(name="data", reason="list_failed")
                logger.warning("list datasets failed", exc_info=True)
                return []
            result: list[dict[str, Any]] = []
            for manifest in manifests or []:
                result.append(
                    {
                        "dataset_id": manifest.dataset_id,
                        "dataset_type": manifest.dataset_type.value,
                        "location": manifest.location,
                        "version": manifest.version,
                    },
                )
            return result

        return self._cached_registry_call(key=cache_key, fetch=_fetch)

    def get_feature_lineage(self, feature_set_id: str) -> list[dict[str, Any]]:
        """
        Return lineage (parent/child manifests) for a feature set.
        """
        reg = self._get_feature_registry()
        if reg is None:
            self._record_registry_error(name="feature", reason="lineage_unavailable")
            manifests = []
        else:
            try:
                manifests = reg.get_lineage(feature_set_id)
            except Exception:
                self._record_registry_error(name="feature", reason="lineage_failed")
                logger.warning("get feature lineage failed", exc_info=True)
                manifests = []
        out: list[dict[str, Any]] = []
        for m in manifests:
            out.append(
                {
                    "feature_set_id": m.feature_set_id,
                    "role": m.role.value,
                    "stage": m.stage.value,
                    "version": m.version,
                    "schema_hash": m.schema_hash,
                },
            )
        return out

    def list_watermarks(
        self,
        *,
        dataset_id: str,
        instrument: str | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Return dataset watermarks.

        Best‑effort for JSON backend; single lookup for DB.

        """
        reg = self._get_data_registry()
        if reg is None:
            self._record_registry_error(name="data", reason="watermarks_unavailable")
            return []
        limit_value = max(0, int(limit)) if limit >= 0 else 0

        def _serialize(wm: Watermark) -> dict[str, Any]:
            return {
                "dataset_id": wm.dataset_id,
                "instrument_id": wm.instrument_id,
                "source": wm.source,
                "last_success_ns": wm.last_success_ns,
                "last_attempt_ns": wm.last_attempt_ns,
                "last_count": wm.last_count,
                "completeness_pct": wm.completeness_pct,
                "updated_at": wm.updated_at,
            }

        if instrument and source:
            try:
                watermark = reg.get_watermark(dataset_id, instrument, source)
            except Exception:
                self._record_registry_error(name="data", reason="watermark_failed")
                logger.warning("get watermark failed", exc_info=True)
                return []
            if watermark is None:
                return []
            return [_serialize(watermark)]

        try:
            records: list[Watermark] = list(
                reg.iter_watermarks(
                    dataset_id=dataset_id,
                    instrument_id=instrument,
                    source=source,
                    limit=limit_value,
                ),
            )
        except Exception:
            self._record_registry_error(name="data", reason="watermark_failed")
            logger.warning("list watermarks failed", exc_info=True)
            return []

        return [_serialize(wm) for wm in records]

    def list_dataset_lineage(
        self,
        *,
        child: str | None = None,
        parent: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Return dataset lineage entries filtered by child/parent identifiers.
        """
        reg = self._get_data_registry()
        if reg is None:
            self._record_registry_error(name="data", reason="lineage_unavailable")
            return []
        limit_value = max(0, int(limit)) if limit >= 0 else 0

        try:
            records: list[DatasetLineageRecord] = list(
                reg.iter_lineage(
                    child=child,
                    parent=parent,
                    limit=limit_value,
                ),
            )
        except Exception:
            self._record_registry_error(name="data", reason="lineage_failed")
            logger.warning("list dataset lineage failed", exc_info=True)
            return []

        result: list[dict[str, Any]] = []
        for record in records:
            result.append(
                {
                    "transform_id": record.transform_id,
                    "child_dataset_id": record.child_dataset_id,
                    "parent_dataset_id": record.parent_dataset_id,
                    "ts_range": record.ts_range,
                    "parameters": record.parameters,
                    "created_at": record.created_at,
                },
            )
        return result

    def get_strategy_details(self, strategy_id: str) -> dict[str, Any] | None:
        """
        Return strategy manifest details if available.
        """
        reg = self._get_strategy_registry()
        if reg is None:
            self._record_registry_error(name="strategy", reason="details_unavailable")
            sinfo = None
        else:
            try:
                sinfo = reg.get_strategy(strategy_id)
            except Exception:
                self._record_registry_error(name="strategy", reason="details_failed")
                logger.warning("get strategy details failed", exc_info=True)
                sinfo = None
        if not sinfo:
            return None
        m = sinfo.manifest
        return {
            "strategy_id": m.strategy_id,
            "type": m.strategy_type.value,
            "version": m.version,
            "required_models": m.required_models or [],
            "required_features": list(m.required_features),
            "suitable_regimes": [r.value for r in m.suitable_regimes],
            "instrument_types": list(m.instrument_types),
        }

    def check_strategy_compatibility(self, strategy_id: str, active: list[str]) -> dict[str, Any]:
        """
        Return compatibility boolean for a strategy against active strategies.
        """
        reg = self._get_strategy_registry()
        if reg is None:
            self._record_registry_error(name="strategy", reason="compatibility_unavailable")
            compatible = False
        else:
            try:
                compatible = reg.check_compatibility(strategy_id, active)
            except Exception:
                self._record_registry_error(name="strategy", reason="compatibility_failed")
                logger.warning("check strategy compatibility failed", exc_info=True)
                compatible = False
        return {"strategy_id": strategy_id, "compatible": bool(compatible)}

    # -----------------
    # Feature actions: promote/deprecate
    # -----------------
    def promote_feature(
        self,
        feature_set_id: str,
        *,
        stage: str | None = None,
        gates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        reg = self._get_feature_registry()
        if reg is None:
            self._record_registry_error(name="feature", reason="promote_unavailable")
            return {"ok": False, "feature_set_id": feature_set_id, "stage": stage or "PROD"}
        ok = False
        new_stage = stage or "PROD"
        try:
            if gates:
                qg = [QualityGate(**g) for g in gates]
                ok = reg.validate_and_promote(feature_set_id, qg)
            else:
                fs = FeatureStage(new_stage)
                reg.promote(feature_set_id, fs)
                ok = True
        except Exception:
            self._record_registry_error(name="feature", reason="promote_failed")
            ok = False

        # Best-effort bus publish (stage-first topic)
        try:
            cfg = MessageBusConfig.from_env()
            pub = publisher_from_config(cfg)
            topic = build_stage_topic(Stage.FEATURE_COMPUTED, prefix=cfg.topic_prefix)
            _ = pub.publish(
                topic,
                {
                    "feature_set_id": feature_set_id,
                    "action": "promote",
                    "new_stage": new_stage,
                    "status": EventStatus.SUCCESS.value if ok else EventStatus.FAILED.value,
                },
            )
        except Exception:
            pass
        if ok:
            self._invalidate_cache(self._cache_key("features"))
        return {"ok": ok, "feature_set_id": feature_set_id, "stage": new_stage}

    def deprecate_feature(
        self,
        feature_set_id: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        reg = self._get_feature_registry()
        if reg is None:
            self._record_registry_error(name="feature", reason="deprecate_unavailable")
            return {"ok": False, "feature_set_id": feature_set_id}
        ok = False
        try:
            reg.deprecate(feature_set_id, reason)
            ok = True
        except Exception:
            self._record_registry_error(name="feature", reason="deprecate_failed")
            ok = False
        try:
            cfg = MessageBusConfig.from_env()
            pub = publisher_from_config(cfg)
            topic = build_stage_topic(Stage.FEATURE_COMPUTED, prefix=cfg.topic_prefix)
            _ = pub.publish(
                topic,
                {
                    "feature_set_id": feature_set_id,
                    "action": "deprecate",
                    "reason": reason or "",
                    "status": EventStatus.SUCCESS.value if ok else EventStatus.FAILED.value,
                },
            )
        except Exception:
            pass
        if ok:
            self._invalidate_cache(self._cache_key("features"))
        return {"ok": ok, "feature_set_id": feature_set_id}

    # -----------------
    # Grafana provisioning
    # -----------------
    def provision_grafana_dashboard(self, *, title: str | None = None) -> dict[str, Any]:
        import os

        url = os.getenv("GRAFANA_URL", "http://localhost:3000")
        token = os.getenv("GRAFANA_API_TOKEN")
        user = os.getenv("GF_ADMIN_USER") or os.getenv("GRAFANA_ADMIN_USER")
        pwd = os.getenv("GF_SECURITY_ADMIN_PASSWORD") or os.getenv("GRAFANA_ADMIN_PASSWORD")
        cfg = GrafanaConfig(url=url, api_token=token, username=user, password=pwd)
        ok, dash_url = provision_dashboard(cfg, overwrite=True, title=title)
        return {"ok": ok, "url": dash_url}

    # -----------------
    # Control actions
    # -----------------
    def control_service(self, name: str, action: str) -> dict[str, Any]:
        route = "/api/services/{name}:action"
        start = time.perf_counter()
        try:
            bind_log_context(component="ml.dashboard", action=action, service=name)
            if not isinstance(
                self.controller,
                ServiceControllerProtocol,
            ):  # runtime check for protocols
                raise ServiceControlUnsupportedError("service control unavailable")
            result = False
            if action == "start":
                result = self.controller.start(name)
            elif action == "stop":
                result = self.controller.stop(name)
            elif action == "restart":
                result = self.controller.restart(name)
            else:
                raise ValueError(f"unknown action: {action}")
            status = "success" if result else "noop"
            _REQS_TOTAL.labels(route=route, method="POST", status=status).inc()
            return {"ok": result, "action": action, "service": name}
        except ServiceControlUnsupportedError:
            _REQS_TOTAL.labels(route=route, method="POST", status="unsupported").inc()
            return {"ok": False, "action": action, "service": name, "error": "unsupported"}
        finally:
            _LATENCY_SECONDS.labels(route=route).observe(time.perf_counter() - start)

    def trigger_pipeline(
        self,
        mode: str,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Best-effort trigger notification for a pipeline run.

        This does not execute the run directly; instead it emits a bus event so an
        external orchestrator can react. For local use, you can wire this to a Compose
        controller or CLI runner in a follow-up iteration.

        """
        start = time.perf_counter()
        route = "/api/pipeline/run"
        params_json = json.loads(json.dumps(params or {}))  # ensure JSON-serializable
        ok = False
        topic = ""
        status_label = "error"
        try:
            cfg = MessageBusConfig.from_env()
            pub = publisher_from_config(cfg)
            stage = Stage.DATA_INGESTED if mode == "backfill" else Stage.CATALOG_WRITTEN
            topic = build_topic_for_stage(
                stage,
                instrument_id=params_json.get("instrument", "UNKNOWN"),
                scheme=cfg.scheme,
                prefix=cfg.topic_prefix,
            )
            payload = {
                "mode": mode,
                "params": params_json,
                "source": Source.BACKFILL.value if mode == "backfill" else Source.LIVE.value,
                "status": EventStatus.SUCCESS.value,
            }
            ok = bool(pub.publish(topic, payload))
            status_label = "success" if ok else "noop"
        except Exception:
            logger.debug("pipeline trigger publish failed", exc_info=True)
            status_label = "error"
        finally:
            _REQS_TOTAL.labels(route=route, method="POST", status=status_label).inc()
            _LATENCY_SECONDS.labels(route=route).observe(time.perf_counter() - start)
        return {"ok": ok, "topic": topic}


def _bootstrap_logging() -> None:
    try:
        configure_logging()
        bind_log_context(component="ml.dashboard")
    except Exception:  # pragma: no cover - defensive
        logger.debug("logging bootstrap failed (ignored)", exc_info=True)


__all__ = ["DashboardService"]
