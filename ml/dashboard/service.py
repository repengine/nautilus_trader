"""
Dashboard service implementation providing health aggregation and control actions.

All operations are cold-path only. Metrics are recorded via the centralized
metrics bootstrap. Logging uses structlog configuration from ml.common.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from urllib.parse import urljoin

import requests

from ml.common.logging_config import bind_log_context
from ml.common.logging_config import configure_logging
from ml.common.message_bus import publisher_from_config
from ml.common.message_topics import build_stage_topic
from ml.common.message_topics import build_topic_for_stage
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
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
from ml.registry import FeatureRegistry
from ml.registry import ModelInfo
from ml.registry import ModelRegistry
from ml.registry import PersistenceConfig
from ml.registry import StrategyRegistry
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


def _safe_get(url: str, timeout: float) -> tuple[bool, int]:
    try:
        resp = requests.get(url, timeout=timeout)
        return resp.ok, resp.status_code
    except Exception:
        return False, 0


def _to_url(host_port: int, path: str) -> str:
    return urljoin(f"http://localhost:{host_port}/", path.lstrip("/"))


@dataclass(slots=True)
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

    @classmethod
    def from_config(cls, config: DashboardConfig) -> DashboardService:
        controller: ServiceControllerProtocol
        if config.compose_enabled:
            controller = ComposeServiceController(config.compose_file)
        else:
            controller = NoopServiceController()
        return cls(config=config, controller=controller)

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
                f"http://localhost:{cfg.prometheus_port}/-/healthy", cfg.request_timeout_seconds
            )
            ok_graf, code_graf = _safe_get(
                f"http://localhost:{cfg.grafana_port}/api/health", cfg.request_timeout_seconds
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
                "endpoints": {"health": _to_url(cfg.actor_port, "/health"), "metrics": _to_url(cfg.actor_port, "/metrics")},
            },
            {
                "name": "ml_strategy",
                "ports": {"http": cfg.strategy_port},
                "endpoints": {"health": _to_url(cfg.strategy_port, "/health"), "metrics": _to_url(cfg.strategy_port, "/metrics")},
            },
            {
                "name": "ml_pipeline",
                "ports": {"http": cfg.pipeline_port},
                "endpoints": {"health": _to_url(cfg.pipeline_port, "/health"), "metrics": _to_url(cfg.pipeline_port, "/metrics")},
            },
        ]

    # -----------------
    # Registries (read/promote)
    # -----------------
    def _get_model_registry(self) -> ModelRegistry:
        if self._model_registry is not None:
            return self._model_registry
        # Prefer DB if available in env, otherwise JSON under ./ml_registry/models
        import os
        db = os.getenv("DB_CONNECTION") or os.getenv("DATABASE_URL")
        if db:
            pc = PersistenceConfig(backend=BackendType.POSTGRES, connection_string=db)
        else:
            from pathlib import Path as _Path

            pc = PersistenceConfig(backend=BackendType.JSON, json_path=_Path("./ml_registry/models"))
        reg_path = pc.json_path if pc.backend == BackendType.JSON else None
        if reg_path is None:
            from pathlib import Path as _Path

            reg_path = _Path("./ml_registry/models")
        self._model_registry = ModelRegistry(registry_path=reg_path, persistence_config=pc)
        return self._model_registry

    def _get_feature_registry(self) -> FeatureRegistry:
        if self._feature_registry is not None:
            return self._feature_registry
        import os
        db = os.getenv("DB_CONNECTION") or os.getenv("DATABASE_URL")
        from pathlib import Path as _Path

        reg_path = _Path("./ml_registry/features")
        if db:
            pc = PersistenceConfig(backend=BackendType.POSTGRES, connection_string=db)
        else:
            pc = PersistenceConfig(backend=BackendType.JSON, json_path=reg_path)
        self._feature_registry = FeatureRegistry(registry_path=reg_path, persistence_config=pc)
        return self._feature_registry

    def _get_strategy_registry(self) -> StrategyRegistry:
        if self._strategy_registry is not None:
            return self._strategy_registry
        import os
        db = os.getenv("DB_CONNECTION") or os.getenv("DATABASE_URL")
        from pathlib import Path as _Path

        base_path = _Path("./ml_registry/strategies")
        if db:
            pc = PersistenceConfig(backend=BackendType.POSTGRES, connection_string=db)
        else:
            pc = PersistenceConfig(backend=BackendType.JSON, json_path=base_path)
        self._strategy_registry = StrategyRegistry(base_path=base_path, persistence_config=pc)
        return self._strategy_registry

    def _get_data_registry(self) -> DataRegistry:
        if self._data_registry is not None:
            return self._data_registry
        import os
        db = os.getenv("DB_CONNECTION") or os.getenv("DATABASE_URL")
        from pathlib import Path as _Path

        path = _Path("./ml_registry/datasets")
        if db:
            pc = PersistenceConfig(backend=BackendType.POSTGRES, connection_string=db)
        else:
            pc = PersistenceConfig(backend=BackendType.JSON, json_path=path)
        self._data_registry = DataRegistry(registry_path=path, persistence_config=pc)
        return self._data_registry

    def list_models(self) -> list[dict[str, Any]]:
        reg = self._get_model_registry()
        models: list[ModelInfo] = reg.get_all_models()
        # Summarize to lightweight JSON structure
        out: list[dict[str, Any]] = []
        for mi in models:
            out.append(
                {
                    "model_id": mi.manifest.model_id,
                    "role": mi.manifest.role.value,
                    "version": mi.manifest.version,
                    "deployment_status": mi.deployment_status.value,
                    "deployed_to": list(mi.deployed_to),
                    "architecture": mi.manifest.architecture,
                    "feature_schema_hash": mi.manifest.feature_schema_hash,
                }
            )
        return out

    def list_deployments(self) -> dict[str, list[str]]:
        reg = self._get_model_registry()
        # Use public methods to derive deployments
        active = reg.get_active_models()
        deployments: dict[str, list[str]] = {}
        for mi in active:
            for tgt in mi.deployed_to:
                deployments.setdefault(tgt, []).append(mi.manifest.model_id)
        return deployments

    def deploy_model(self, model_id: str, target: str) -> dict[str, Any]:
        reg = self._get_model_registry()
        ok = reg.deploy_model(model_id, target)
        return {"ok": ok, "model_id": model_id, "target": target}

    def hot_reload_model(self, target: str, new_model_id: str) -> dict[str, Any]:
        """
        Hot reload a deployment with a new model id.
        """
        reg = self._get_model_registry()
        ok = reg.hot_reload_model(target=target, new_model_id=new_model_id)
        return {"ok": ok, "target": target, "model_id": new_model_id}

    def rollback_deployment(self, target: str, to_model_id: str) -> dict[str, Any]:
        """
        Rollback a target to a specific model id.
        """
        reg = self._get_model_registry()
        ok = reg.rollback(target=target, to_model_id=to_model_id)
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
        cfg = MessageBusConfig.from_env()
        if not cfg.enabled or cfg.backend != "redis":
            return []
        try:
            import redis

            client: Any = redis.Redis.from_url(cfg.redis_url, decode_responses=True)
            # Newest-first range; fetch extra and filter locally
            rows: list[tuple[str, dict[str, str]]] = client.xrevrange(cfg.redis_stream, count=max(1, int(limit)))
        except Exception:
            return []

        out: list[dict[str, Any]] = []
        for entry_id, fields in rows:
            topic = fields.get("topic", "")
            payload_raw = fields.get("payload", "{}")
            try:
                payload = json.loads(payload_raw)
            except Exception:
                payload = {"raw": payload_raw}
            # Simple filters
            if source is not None and str(payload.get("source")) != source:
                continue
            if instrument_substr:
                instr = payload.get("params", {}).get("instrument") if isinstance(payload.get("params"), dict) else None
                if not instr or instrument_substr not in str(instr):
                    # Try topic suffix match
                    if instrument_substr not in topic:
                        continue
            # Stage filter: best-effort from topic
            if stage:
                if stage not in topic:
                    continue
            out.append({"id": entry_id, "topic": topic, "payload": payload})
            if len(out) >= limit:
                break
        return out

    # -----------------
    # Feature/Strategy/Data registry listings
    # -----------------
    def list_features(self, *, role: str | None = None, stage: str | None = None) -> list[dict[str, Any]]:
        reg = self._get_feature_registry()
        infos = reg.list_all()
        out: list[dict[str, Any]] = []
        for fi in infos:
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
                }
            )
        return out

    def list_strategies(self) -> list[dict[str, Any]]:
        reg = self._get_strategy_registry()
        # StrategyRegistry does not expose list_all; use JSON registry listing where available
        result: list[dict[str, Any]] = []
        try:
            # For JSON backend, iterate registry dict
            registry = reg._load_registry()
            for sid in registry:
                sinfo = reg.get_strategy(sid)
                if not sinfo:
                    continue
                m = sinfo.manifest
                result.append(
                    {
                        "strategy_id": m.strategy_id,
                        "type": m.strategy_type.value,
                        "version": m.version,
                        "required_models": m.required_models or [],
                    }
                )
        except Exception:
            # Best effort only
            return result
        return result

    def list_datasets(self) -> list[dict[str, Any]]:
        reg = self._get_data_registry()
        result: list[dict[str, Any]] = []
        try:
            # For JSON backend, DataRegistry stores manifests internally
            manifests = getattr(reg, "_manifests", {})
            for ds_id, manifest in manifests.items():
                result.append(
                    {
                        "dataset_id": ds_id,
                        "dataset_type": manifest.dataset_type.value,
                        "location": manifest.location,
                        "version": manifest.version,
                    }
                )
        except Exception:
            return result
        return result

    def get_feature_lineage(self, feature_set_id: str) -> list[dict[str, Any]]:
        """
        Return lineage (parent/child manifests) for a feature set.
        """
        reg = self._get_feature_registry()
        try:
            manifests = reg.get_lineage(feature_set_id)
        except Exception:
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
                }
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
        Return dataset watermarks. Best‑effort for JSON backend; single lookup for DB.
        """
        reg = self._get_data_registry()
        out: list[dict[str, Any]] = []
        if instrument and source:
            try:
                wm = reg.get_watermark(dataset_id, instrument, source)
                if wm is not None:
                    out.append(
                        {
                            "dataset_id": wm.dataset_id,
                            "instrument_id": wm.instrument_id,
                            "source": wm.source,
                            "last_success_ns": wm.last_success_ns,
                            "last_attempt_ns": wm.last_attempt_ns,
                            "last_count": wm.last_count,
                            "completeness_pct": wm.completeness_pct,
                            "updated_at": wm.updated_at,
                        }
                    )
            except Exception:
                return []
            return out

        # JSON backend: scan cached dict for dataset matches
        try:
            cached: dict[str, Any] = getattr(reg, "_watermarks", {})
            for key, wm in cached.items():
                if not key.startswith(f"{dataset_id}:"):
                    continue
                if instrument and f":{instrument}:" not in key:
                    continue
                if source and key.split(":")[-1] != source:
                    continue
                out.append(
                    {
                        "dataset_id": wm.dataset_id,
                        "instrument_id": wm.instrument_id,
                        "source": wm.source,
                        "last_success_ns": wm.last_success_ns,
                        "last_attempt_ns": wm.last_attempt_ns,
                        "last_count": wm.last_count,
                        "completeness_pct": wm.completeness_pct,
                        "updated_at": wm.updated_at,
                    }
                )
                if len(out) >= limit:
                    break
        except Exception:
            return []
        return out

    def list_dataset_lineage(
        self,
        *,
        child: str | None = None,
        parent: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Return dataset lineage entries filtered by child/parent (JSON backend only).
        """
        reg = self._get_data_registry()
        try:
            entries: list[dict[str, Any]] = getattr(reg, "_lineage", [])
        except Exception:
            return []
        out: list[dict[str, Any]] = []
        for e in reversed(entries):
            if child and e.get("child_dataset_id") != child:
                continue
            if parent and e.get("parent_dataset_id") != parent:
                continue
            out.append(
                {
                    "transform_id": e.get("transform_id", ""),
                    "child_dataset_id": e.get("child_dataset_id", ""),
                    "parent_dataset_id": e.get("parent_dataset_id", ""),
                    "ts_range": e.get("ts_range", {}),
                    "parameters": e.get("parameters", {}),
                    "created_at": e.get("created_at", 0.0),
                }
            )
            if len(out) >= limit:
                break
        return out
    def get_strategy_details(self, strategy_id: str) -> dict[str, Any] | None:
        """
        Return strategy manifest details if available.
        """
        reg = self._get_strategy_registry()
        try:
            sinfo = reg.get_strategy(strategy_id)
        except Exception:
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
        try:
            compatible = reg.check_compatibility(strategy_id, active)
        except Exception:
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
        return {"ok": ok, "feature_set_id": feature_set_id, "stage": new_stage}

    def deprecate_feature(self, feature_set_id: str, *, reason: str | None = None) -> dict[str, Any]:
        reg = self._get_feature_registry()
        ok = False
        try:
            reg.deprecate(feature_set_id, reason)
            ok = True
        except Exception:
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
            if not isinstance(self.controller, ServiceControllerProtocol):  # runtime check for protocols
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

    def trigger_pipeline(self, mode: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """
        Best-effort trigger notification for a pipeline run.

        This does not execute the run directly; instead it emits a bus event so an
        external orchestrator can react. For local use, you can wire this to a
        Compose controller or CLI runner in a follow-up iteration.
        """
        start = time.perf_counter()
        route = "/api/pipeline/run"
        params_json = json.loads(json.dumps(params or {}))  # ensure JSON-serializable
        try:
            cfg = MessageBusConfig.from_env()
            pub = publisher_from_config(cfg)
            stage = Stage.DATA_INGESTED if mode == "backfill" else Stage.CATALOG_WRITTEN
            topic = build_topic_for_stage(stage, instrument_id=params_json.get("instrument", "UNKNOWN"), scheme=cfg.scheme, prefix=cfg.topic_prefix)
            payload = {
                "mode": mode,
                "params": params_json,
                "source": Source.BACKFILL.value if mode == "backfill" else Source.LIVE.value,
                "status": EventStatus.SUCCESS.value,
            }
            ok = pub.publish(topic, payload)
            _REQS_TOTAL.labels(route=route, method="POST", status=("success" if ok else "noop")).inc()
            return {"ok": ok, "topic": topic}
        finally:
            _LATENCY_SECONDS.labels(route=route).observe(time.perf_counter() - start)


def _bootstrap_logging() -> None:
    try:
        configure_logging()
        bind_log_context(component="ml.dashboard")
    except Exception:  # pragma: no cover - defensive
        logger.debug("logging bootstrap failed (ignored)", exc_info=True)


__all__ = ["DashboardService"]
