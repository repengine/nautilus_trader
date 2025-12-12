"""
Registry manager component for Dashboard service.

Extracted from DashboardService to follow single-responsibility principle.
Manages all 4 registry interactions with TTL-based caching and progressive fallback.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from threading import Lock
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, cast

from ml.common.message_bus import publisher_from_config
from ml.common.message_topics import build_topic_for_stage
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.common.retry_utils import retry_with_backoff
from ml.config.bus import MessageBusConfig
from ml.config.events import EventStatus
from ml.config.events import Stage
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


if TYPE_CHECKING:
    from ml.dashboard.config import DashboardConfig


logger = logging.getLogger(__name__)


# Metrics
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
_REGISTRY_LATENCY_SECONDS = get_histogram(
    "ml_dashboard_registry_latency_seconds",
    "Dashboard registry cache latency (seconds)",
    labels=["entry"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0],
)


_CacheValueT = TypeVar("_CacheValueT")


@dataclass(slots=True)
class _CacheEntry(Generic[_CacheValueT]):
    """Cache entry containing the value and its monotonic expiry."""

    value: _CacheValueT
    expires_at: float


@dataclass(slots=True)
class _TTLCache(Generic[_CacheValueT]):
    """
    Simple TTL cache intended for cold-path dashboard usage.

    Provides get/put/delete/clear operations with automatic expiry.
    Thread-safe with internal locking.
    """

    ttl_seconds: float
    max_entries: int
    _clock: Callable[[], float] = time.monotonic
    _entries: dict[str, _CacheEntry[_CacheValueT]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def get(self, key: str) -> _CacheValueT | None:
        """Get cached value if present and not expired."""
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
        """Put value into cache with TTL expiry."""
        expires_at = self._clock() + self.ttl_seconds
        with self._lock:
            if key not in self._entries and len(self._entries) >= self.max_entries:
                self._evict_locked()
            self._entries[key] = _CacheEntry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> None:
        """Delete entry from cache."""
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._entries.clear()

    def keys(self) -> list[str]:
        """Return all current cache keys."""
        with self._lock:
            return list(self._entries.keys())

    def _evict_locked(self) -> None:
        """Evict oldest entry (assumes lock held)."""
        if not self._entries:
            return
        lru_key = min(self._entries.items(), key=lambda item: item[1].expires_at)[0]
        self._entries.pop(lru_key, None)


def _env_allows_dummy() -> bool:
    """Return True when environment permits dummy fallback."""
    import os

    value = os.getenv("ML_ALLOW_DUMMY", "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


class RegistryManagerProtocol(Protocol):
    """Protocol for registry management operations."""

    def list_models(self) -> list[dict[str, Any]]: ...
    def get_model_performance_history(self, model_id: str, *, limit: int = 100) -> list[dict[str, Any]]: ...
    def list_deployments(self) -> dict[str, list[str]]: ...
    def list_features(self, role: str | None = None, stage: str | None = None) -> list[dict[str, Any]]: ...
    def get_feature_lineage(self, feature_set_id: str) -> list[dict[str, Any]]: ...
    def list_strategies(self) -> list[dict[str, Any]]: ...
    def get_strategy_details(self, strategy_id: str) -> dict[str, Any] | None: ...
    def check_strategy_compatibility(self, strategy_id: str, active: list[str]) -> dict[str, Any]: ...
    def promote_feature(self, feature_set_id: str, *, stage: str | None = None, gates: list[dict[str, Any]] | None = None) -> dict[str, Any]: ...
    def deprecate_feature(self, feature_set_id: str, *, reason: str | None = None) -> dict[str, Any]: ...
    def list_datasets(self) -> list[dict[str, Any]]: ...
    def list_watermarks(self, *, dataset_id: str, instrument: str | None = None, source: str | None = None, limit: int = 100) -> list[dict[str, Any]]: ...
    def list_dataset_lineage(self, *, child: str | None = None, parent: str | None = None, limit: int = 100) -> list[dict[str, Any]]: ...


@dataclass
class RegistryManagerComponent:
    """
    Component for managing registry interactions.

    Extracted from DashboardService to follow single-responsibility principle.
    Manages all 4 registries (Model, Feature, Strategy, Data) with TTL-based caching
    and progressive fallback to DummyRegistry when PostgreSQL unavailable.

    Responsibilities:
    - Manage all 4 registry interactions
    - Cache registry queries with TTL (30 seconds default)
    - Invalidate cache on mutations
    - Fallback to DummyRegistry when PostgreSQL unavailable
    """

    config: DashboardConfig
    _model_registry: ModelRegistry | None = field(default=None, init=False, repr=False)
    _feature_registry: FeatureRegistry | None = field(default=None, init=False, repr=False)
    _strategy_registry: StrategyRegistry | None = field(default=None, init=False, repr=False)
    _data_registry: DataRegistry | None = field(default=None, init=False, repr=False)
    _registry_cache: _TTLCache[object] = field(init=False, repr=False)
    _allow_dummy_fallback: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize cache and fallback settings."""
        self._registry_cache = _TTLCache(ttl_seconds=30.0, max_entries=32)
        self._allow_dummy_fallback = _env_allows_dummy()

    # -----------------
    # Private helpers: Registry builders
    # -----------------
    def _get_model_registry(self) -> ModelRegistry | None:
        """Get or build model registry with lazy initialization."""
        if self._model_registry is not None:
            return self._model_registry
        registry = cast(
            ModelRegistry | None,
            self._build_registry(name="model", builder=self._build_model_registry),
        )
        self._model_registry = registry
        return registry

    def _get_feature_registry(self) -> FeatureRegistry | None:
        """Get or build feature registry with lazy initialization."""
        if self._feature_registry is not None:
            return self._feature_registry
        registry = cast(
            FeatureRegistry | None,
            self._build_registry(name="feature", builder=self._build_feature_registry),
        )
        self._feature_registry = registry
        return registry

    def _get_strategy_registry(self) -> StrategyRegistry | None:
        """Get or build strategy registry with lazy initialization."""
        if self._strategy_registry is not None:
            return self._strategy_registry
        registry = cast(
            StrategyRegistry | None,
            self._build_registry(name="strategy", builder=self._build_strategy_registry),
        )
        self._strategy_registry = registry
        return registry

    def _get_data_registry(self) -> DataRegistry | None:
        """Get or build data registry with lazy initialization."""
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
        """
        Build registry with retry and fallback logic.

        Args:
            name: Registry name for logging/metrics
            builder: Callable that builds the registry

        Returns:
            Registry instance or DummyRegistry fallback
        """
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
        """Build ModelRegistry with PostgreSQL or JSON backend."""
        import os
        from pathlib import Path as _Path

        db = os.getenv("DB_CONNECTION") or os.getenv("DATABASE_URL")
        reg_path = _Path("./ml_registry/models")
        if db:
            pc = PersistenceConfig(backend=BackendType.POSTGRES, connection_string=db)
        else:
            pc = PersistenceConfig(backend=BackendType.JSON, json_path=reg_path)
        return ModelRegistry(registry_path=reg_path, persistence_config=pc)

    def _build_feature_registry(self) -> FeatureRegistry | None:
        """Build FeatureRegistry with PostgreSQL or JSON backend."""
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
        """Build StrategyRegistry with PostgreSQL or JSON backend."""
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
        """Build DataRegistry with PostgreSQL or JSON backend."""
        import os
        from pathlib import Path as _Path

        db_cfg = self.config.db_connection
        if db_cfg is not None:
            db = db_cfg.strip()
        else:
            db = (os.getenv("DB_CONNECTION") or os.getenv("DATABASE_URL") or "").strip()
        if not db:
            # Avoid implicitly loading local/ambient registries when no backend is configured.
            return None
        path = _Path("./ml_registry/datasets")
        pc = PersistenceConfig(backend=BackendType.POSTGRES, connection_string=db)
        return DataRegistry(registry_path=path, persistence_config=pc)

    # -----------------
    # Private helpers: Caching
    # -----------------
    def _cache_key(self, entry: str, *parts: str | None) -> str:
        """Build cache key from entry name and optional parts."""
        safe_parts = [part or "*" for part in parts]
        return f"{entry}:{'|'.join(safe_parts)}"

    def _cached_registry_call(
        self,
        *,
        key: str,
        fetch: Callable[[], _CacheValueT],
    ) -> _CacheValueT:
        """
        Execute registry call with caching.

        Args:
            key: Cache key
            fetch: Callable to fetch fresh data on cache miss

        Returns:
            Cached or freshly fetched value
        """
        start = time.perf_counter()
        cached = self._registry_cache.get(key)
        if cached is not None:
            _REGISTRY_CACHE_HITS.labels(entry=key).inc()
            _REGISTRY_LATENCY_SECONDS.labels(entry=key).observe(time.perf_counter() - start)
            return cast(_CacheValueT, cached)

        _REGISTRY_CACHE_MISSES.labels(entry=key).inc()
        value = fetch()
        self._registry_cache.put(key, value)
        _REGISTRY_LATENCY_SECONDS.labels(entry=key).observe(time.perf_counter() - start)
        return value

    def _invalidate_cache(self, *keys: str) -> None:
        """
        Invalidate cache entries by key pattern.

        Supports wildcards: "models*" invalidates all keys starting with "models"
        """
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
        """Record registry error metric."""
        _REGISTRY_FALLBACK_TOTAL.labels(registry=name, reason=reason).inc()

    # -----------------
    # Public API: Model Registry
    # -----------------
    def list_models(self) -> list[dict[str, Any]]:
        """
        List all models from model registry.

        Returns:
            List of model dictionaries with id, role, version, status, etc.

        Example:
            >>> manager = RegistryManagerComponent(config)
            >>> models = manager.list_models()
            >>> assert isinstance(models, list)
        """
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

    def get_model_performance_history(
        self,
        model_id: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get performance history for a specific model.

        Args:
            model_id: Model identifier
            limit: Maximum number of history entries to return

        Returns:
            List of performance history dictionaries

        Example:
            >>> manager = RegistryManagerComponent(config)
            >>> history = manager.get_model_performance_history("model_v1", limit=10)
            >>> assert isinstance(history, list)
        """
        registry = self._get_model_registry()
        if registry is None:
            return []
        get_model = getattr(registry, "get_model", None)
        if not callable(get_model):
            return []
        try:
            model_info = get_model(model_id)
        except Exception:
            logger.debug("get_model failed", exc_info=True)
            return []
        if model_info is None:
            return []
        history = getattr(model_info, "performance_history", None)
        if not isinstance(history, list):
            return []
        if limit >= 0:
            history = history[-limit:]
        result: list[dict[str, Any]] = []
        for entry in history:
            if isinstance(entry, Mapping):
                result.append(dict(entry))
        return result

    def list_deployments(self) -> dict[str, list[str]]:
        """
        List model deployments by target.

        Returns:
            Dictionary mapping deployment targets to list of model IDs

        Example:
            >>> manager = RegistryManagerComponent(config)
            >>> deployments = manager.list_deployments()
            >>> assert isinstance(deployments, dict)
        """
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

    # -----------------
    # Public API: Feature Registry
    # -----------------
    def list_features(
        self,
        role: str | None = None,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List features from feature registry with optional filtering.

        Args:
            role: Optional role filter
            stage: Optional stage filter

        Returns:
            List of feature dictionaries

        Example:
            >>> manager = RegistryManagerComponent(config)
            >>> features = manager.list_features(role="primary")
            >>> assert isinstance(features, list)
        """
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
            for fi in infos or []:
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

    def get_feature_lineage(self, feature_set_id: str) -> list[dict[str, Any]]:
        """
        Get lineage (parent/child manifests) for a feature set.

        Args:
            feature_set_id: Feature set identifier

        Returns:
            List of lineage manifest dictionaries

        Example:
            >>> manager = RegistryManagerComponent(config)
            >>> lineage = manager.get_feature_lineage("feature_v1")
            >>> assert isinstance(lineage, list)
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

    def promote_feature(
        self,
        feature_set_id: str,
        *,
        stage: str | None = None,
        gates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Promote feature to a new stage with optional quality gates.

        Args:
            feature_set_id: Feature set identifier
            stage: Target stage (defaults to "PROD")
            gates: Optional quality gate checks

        Returns:
            Result dictionary with ok status

        Example:
            >>> manager = RegistryManagerComponent(config)
            >>> result = manager.promote_feature("feature_v1", stage="PROD")
            >>> assert "ok" in result
        """
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
            logger.warning("promote feature failed", exc_info=True)
            ok = False

        # Best-effort bus publish (env-configured topic scheme)
        try:
            cfg = MessageBusConfig.from_env()
            pub = publisher_from_config(cfg)
            topic = build_topic_for_stage(
                Stage.FEATURE_COMPUTED,
                feature_set_id,
                scheme=cfg.scheme,
                prefix=cfg.topic_prefix,
            )
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
            logger.debug("Failed to publish promote_feature event", exc_info=True)
        if ok:
            self._invalidate_cache(self._cache_key("features"))
        return {"ok": ok, "feature_set_id": feature_set_id, "stage": new_stage}

    def deprecate_feature(
        self,
        feature_set_id: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Deprecate a feature set.

        Args:
            feature_set_id: Feature set identifier
            reason: Optional deprecation reason

        Returns:
            Result dictionary with ok status

        Example:
            >>> manager = RegistryManagerComponent(config)
            >>> result = manager.deprecate_feature("feature_v1", reason="outdated")
            >>> assert "ok" in result
        """
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
            logger.warning("deprecate feature failed", exc_info=True)
            ok = False
        try:
            cfg = MessageBusConfig.from_env()
            pub = publisher_from_config(cfg)
            topic = build_topic_for_stage(
                Stage.FEATURE_COMPUTED,
                feature_set_id,
                scheme=cfg.scheme,
                prefix=cfg.topic_prefix,
            )
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
            logger.debug("Failed to publish deprecate_feature event", exc_info=True)
        if ok:
            self._invalidate_cache(self._cache_key("features"))
        return {"ok": ok, "feature_set_id": feature_set_id}

    # -----------------
    # Public API: Strategy Registry
    # -----------------
    def list_strategies(self) -> list[dict[str, Any]]:
        """
        List all strategies from strategy registry.

        Returns:
            List of strategy dictionaries

        Example:
            >>> manager = RegistryManagerComponent(config)
            >>> strategies = manager.list_strategies()
            >>> assert isinstance(strategies, list)
        """
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

    def get_strategy_details(self, strategy_id: str) -> dict[str, Any] | None:
        """
        Get detailed information for a specific strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Strategy details dictionary or None if not found

        Example:
            >>> manager = RegistryManagerComponent(config)
            >>> details = manager.get_strategy_details("strategy_v1")
            >>> if details:
            ...     assert "strategy_id" in details
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
        Check compatibility of a strategy with active strategies.

        Args:
            strategy_id: Strategy identifier
            active: List of active strategy IDs

        Returns:
            Compatibility result dictionary

        Example:
            >>> manager = RegistryManagerComponent(config)
            >>> result = manager.check_strategy_compatibility("strategy_v1", ["strategy_v2"])
            >>> assert "compatible" in result
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
    # Public API: Data Registry
    # -----------------
    def list_datasets(self) -> list[dict[str, Any]]:
        """
        List all datasets from data registry.

        Returns:
            List of dataset dictionaries

        Example:
            >>> manager = RegistryManagerComponent(config)
            >>> datasets = manager.list_datasets()
            >>> assert isinstance(datasets, list)
        """
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

    def list_watermarks(
        self,
        *,
        dataset_id: str,
        instrument: str | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        List watermarks for a dataset with optional filtering.

        Args:
            dataset_id: Dataset identifier
            instrument: Optional instrument filter
            source: Optional source filter
            limit: Maximum number of watermarks to return

        Returns:
            List of watermark dictionaries

        Example:
            >>> manager = RegistryManagerComponent(config)
            >>> watermarks = manager.list_watermarks(dataset_id="dataset_v1", limit=10)
            >>> assert isinstance(watermarks, list)
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
        List dataset lineage entries filtered by child/parent identifiers.

        Args:
            child: Optional child dataset filter
            parent: Optional parent dataset filter
            limit: Maximum number of lineage entries to return

        Returns:
            List of lineage record dictionaries

        Example:
            >>> manager = RegistryManagerComponent(config)
            >>> lineage = manager.list_dataset_lineage(child="dataset_v2", limit=10)
            >>> assert isinstance(lineage, list)
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


__all__ = [
    "RegistryManagerComponent",
    "RegistryManagerProtocol",
]
