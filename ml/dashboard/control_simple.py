"""
Simplified in-memory control panel utilities for dashboard endpoints.

These helpers keep track of requested actor, pipeline, and ingestion operations
without requiring the heavyweight orchestration stack. They also surface best-
 effort store health using the ML integration manager. All state is persisted to
``/tmp/dashboard_control_state.json`` so operators can inspect or resume recent
 activity between CLI invocations.

The implementation is intentionally side-effect free beyond lightweight health
 probes. Real orchestration or actor lifecycle actions need to be executed by
 dedicated services; this module merely records requested actions.

"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from ml.core.integration import MLIntegrationManager


_STATE_PATH = Path(tempfile.gettempdir()) / "dashboard_control_state.json"

_PYTEST_ENV_FLAG = "PYTEST_CURRENT_TEST"
_DEFAULT_INTEGRATION_TIMEOUT = 10.0
_DEFAULT_HEALTH_TIMEOUT = 5.0

T = TypeVar("T")


def _timeout_from_env(env_var: str, fallback: float | None) -> float | None:
    value = os.getenv(env_var, "").strip()
    if value:
        try:
            parsed = float(value)
        except ValueError:
            return fallback
        return max(parsed, 0.0)
    if fallback is not None and os.getenv(_PYTEST_ENV_FLAG):
        return fallback
    return None


def _utc_now() -> datetime:
    """
    Return a timezone-aware UTC timestamp.
    """
    return datetime.now(tz=UTC)


@dataclass(slots=True)
class _ActorRecord:
    """
    Tracked dashboard actor entry.
    """

    actor_type: str
    config: dict[str, Any]
    started_at: datetime
    status: str = "running"
    last_model_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "actor_type": self.actor_type,
            "config": self.config,
            "started_at": self.started_at.isoformat(),
            "status": self.status,
        }
        if self.last_model_id is not None:
            payload["last_model_id"] = self.last_model_id
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> _ActorRecord:
        return cls(
            actor_type=str(raw["actor_type"]),
            config=dict(raw.get("config", {})),
            started_at=datetime.fromisoformat(str(raw["started_at"])),
            status=str(raw.get("status", "running")),
            last_model_id=(str(raw["last_model_id"]) if raw.get("last_model_id") else None),
        )


@dataclass(slots=True)
class _PipelineRecord:
    """
    Tracked dashboard pipeline run entry.
    """

    mode: str
    config: dict[str, Any]
    started_at: datetime
    status: str = "running"
    job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "mode": self.mode,
            "config": self.config,
            "started_at": self.started_at.isoformat(),
            "status": self.status,
        }
        if self.job_id is not None:
            payload["job_id"] = self.job_id
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> _PipelineRecord:
        return cls(
            mode=str(raw["mode"]),
            config=dict(raw.get("config", {})),
            started_at=datetime.fromisoformat(str(raw["started_at"])),
            status=str(raw.get("status", "running")),
            job_id=str(raw.get("job_id")) if raw.get("job_id") else None,
        )


@dataclass(slots=True)
class _IngestionRecord:
    """
    Tracked ingestion or backfill task.
    """

    symbols: list[str]
    source: str
    started_at: datetime
    status: str = "running"
    kind: str = "ingestion"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbols": list(self.symbols),
            "source": self.source,
            "started_at": self.started_at.isoformat(),
            "status": self.status,
            "kind": self.kind,
        }
        if self.details:
            payload["details"] = self.details
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> _IngestionRecord:
        symbols_raw = raw.get("symbols", [])
        symbols = [str(sym) for sym in symbols_raw]
        details = dict(raw.get("details", {}))
        return cls(
            symbols=symbols,
            source=str(raw.get("source", "unknown")),
            started_at=datetime.fromisoformat(str(raw["started_at"])),
            status=str(raw.get("status", "running")),
            kind=str(raw.get("kind", "ingestion")),
            details=details,
        )


class SimpleControlPanel:
    """
    Lightweight state tracker for dashboard control actions.
    """

    def __init__(
        self,
        *,
        state_path: Path | None = None,
        integration_timeout: float | None = None,
        health_timeout: float | None = None,
    ) -> None:
        self._state_path = state_path or _STATE_PATH
        self._actors: MutableMapping[str, _ActorRecord] = {}
        self._pipelines: MutableMapping[str, _PipelineRecord] = {}
        self._ingestion: MutableMapping[str, _IngestionRecord] = {}
        self._integration: MLIntegrationManager | None = None
        self._integration_failed = False
        self._integration_timeout = (
            integration_timeout
            if integration_timeout is not None
            else _timeout_from_env("ML_DASHBOARD_INTEGRATION_TIMEOUT", _DEFAULT_INTEGRATION_TIMEOUT)
        )
        self._health_timeout = (
            health_timeout
            if health_timeout is not None
            else _timeout_from_env("ML_DASHBOARD_HEALTH_TIMEOUT", _DEFAULT_HEALTH_TIMEOUT)
        )
        self._load_state()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        for actor_id, payload in raw.get("actors", {}).items():
            try:
                self._actors[str(actor_id)] = _ActorRecord.from_dict(payload)
            except Exception:
                continue
        for run_id, payload in raw.get("pipelines", {}).items():
            try:
                self._pipelines[str(run_id)] = _PipelineRecord.from_dict(payload)
            except Exception:
                continue
        for task_id, payload in raw.get("ingestion", {}).items():
            try:
                self._ingestion[str(task_id)] = _IngestionRecord.from_dict(payload)
            except Exception:
                continue

    def _save_state(self) -> None:
        payload = {
            "actors": {actor_id: record.to_dict() for actor_id, record in self._actors.items()},
            "pipelines": {run_id: record.to_dict() for run_id, record in self._pipelines.items()},
            "ingestion": {task_id: record.to_dict() for task_id, record in self._ingestion.items()},
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            # Persisting control state is best effort only.
            pass

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------
    def _maybe_get_integration(self) -> MLIntegrationManager | None:
        if self._integration is not None:
            return self._integration
        if self._integration_failed:
            return None
        try:
            self._integration = self._call_with_timeout(
                lambda: MLIntegrationManager(
                    auto_start_postgres=False,
                    auto_migrate=False,
                    ensure_healthy=False,
                ),
                timeout=self._integration_timeout,
            )
        except TimeoutError:
            self._integration_failed = True
            self._integration = None
        except Exception:
            self._integration_failed = True
            self._integration = None
        return self._integration

    def _collect_store_health(self) -> dict[str, dict[str, bool]]:
        integration = self._maybe_get_integration()
        result: dict[str, dict[str, bool]] = {}
        fallback_mode = (
            bool(
                getattr(integration, "_file_fallback", False)
                or getattr(integration, "_json_fallback", False),
            )
            if integration
            else False
        )
        mapping: Mapping[str, object | None] = {
            "data": getattr(integration, "data_store", None) if integration else None,
            "model": getattr(integration, "model_store", None) if integration else None,
            "feature": getattr(integration, "feature_store", None) if integration else None,
            "strategy": getattr(integration, "strategy_store", None) if integration else None,
        }
        for name, store in mapping.items():
            if fallback_mode:
                result[name] = {"healthy": False, "fallback": True}
            else:
                result[name] = self._evaluate_store_health(store)
        return result

    def _evaluate_store_health(self, store: object | None) -> dict[str, bool]:
        if store is None:
            return {"healthy": False, "fallback": True}
        health_callable = getattr(store, "health_check", None)
        if callable(health_callable):
            try:
                report = self._call_with_timeout(health_callable, timeout=self._health_timeout)
            except TimeoutError:
                return {"healthy": False, "fallback": True}
            except Exception:
                report = None
            if isinstance(report, Mapping):
                return {
                    "healthy": bool(report.get("healthy", True)),
                    "fallback": bool(report.get("fallback", False)),
                }
        is_healthy = getattr(store, "is_healthy", None)
        if callable(is_healthy):
            try:
                healthy_value = bool(
                    self._call_with_timeout(is_healthy, timeout=self._health_timeout)
                )
            except TimeoutError:
                return {"healthy": False, "fallback": True}
            except Exception:
                healthy_value = True
            return {"healthy": healthy_value, "fallback": False}
        return {"healthy": True, "fallback": False}

    def _call_with_timeout(self, func: Callable[[], T], *, timeout: float | None) -> T:
        if timeout is None or timeout <= 0.0:
            return func()

        result_holder: list[T] = []
        error_holder: list[BaseException] = []
        completed = threading.Event()

        def _run() -> None:
            try:
                result_holder.append(func())
            except BaseException as exc:  # pragma: no cover - defensive
                error_holder.append(exc)
            finally:
                completed.set()

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        if not completed.wait(timeout):
            raise TimeoutError

        if error_holder:
            raise error_holder[0]

        return result_holder[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_system_status(self) -> dict[str, Any]:
        """
        Return a snapshot of tracked actors, pipelines, ingestion, and store health.
        """
        stores = self._collect_store_health()
        return {
            "actors": {
                "active": len(self._actors),
                "max": 10,
                "instances": {k: v.to_dict() for k, v in self._actors.items()},
            },
            "pipelines": {
                "active": len(self._pipelines),
                "max": 5,
                "runs": {k: v.to_dict() for k, v in self._pipelines.items()},
            },
            "ingestion": {
                "active_tasks": len(self._ingestion),
                "tasks": {k: v.to_dict() for k, v in self._ingestion.items()},
            },
            "stores": stores,
            "timestamp": _utc_now().isoformat(),
        }

    def actor_count(self) -> int:
        """
        Return the number of tracked actors.
        """
        return len(self._actors)

    def start_actor(
        self, actor_id: str, actor_type: str, config: Mapping[str, Any]
    ) -> dict[str, Any]:
        """
        Record a newly requested actor.
        """
        if actor_id in self._actors:
            record = self._actors[actor_id]
            return {
                "success": True,
                "actor_id": actor_id,
                "status": record.status,
                "already_started": True,
            }
        record = _ActorRecord(
            actor_type=actor_type,
            config=dict(config),
            started_at=_utc_now(),
        )
        self._actors[actor_id] = record
        self._save_state()
        return {"success": True, "actor_id": actor_id, "status": record.status}

    def stop_actor(self, actor_id: str) -> dict[str, Any]:
        """
        Remove an actor from the active set.
        """
        if actor_id not in self._actors:
            return {"success": False, "error": "Actor not found"}
        del self._actors[actor_id]
        self._save_state()
        return {"success": True, "actor_id": actor_id, "status": "stopped"}

    def record_hot_reload(self, actor_id: str, model_id: str) -> dict[str, Any]:
        """
        Update the tracked model identifier for an actor.
        """
        record = self._actors.get(actor_id)
        if record is None:
            return {"success": False, "error": "Actor not found"}
        record.last_model_id = model_id
        record.status = "running"
        self._save_state()
        return {
            "success": True,
            "actor_id": actor_id,
            "model_id": model_id,
            "updated": record.started_at.isoformat(),
        }

    def trigger_pipeline(
        self,
        mode: str,
        config: Mapping[str, Any],
        *,
        job_id: str | None = None,
        status: str = "queued",
    ) -> dict[str, Any]:
        """
        Record a pipeline invocation with optional integration metadata.
        """
        run_key = job_id or f"run_{_utc_now().strftime('%Y%m%d_%H%M%S')}"
        record = _PipelineRecord(
            mode=mode,
            config=dict(config),
            started_at=_utc_now(),
            status=status,
            job_id=job_id,
        )
        self._pipelines[run_key] = record
        self._save_state()
        return {
            "success": True,
            "run_id": run_key,
            "job_id": job_id or run_key,
            "mode": mode,
            "status": record.status,
            "start_time": record.started_at.isoformat(),
        }

    def set_pipeline_status(self, run_id: str, status: str) -> None:
        """
        Update the stored status for a pipeline run if it exists.
        """
        record = self._pipelines.get(run_id)
        if record is None:
            return
        record.status = status
        self._save_state()

    def start_ingestion(self, symbols: Sequence[str], source: str) -> dict[str, Any]:
        """
        Record a streaming ingestion request.
        """
        task_id = f"ingest_{_utc_now().strftime('%Y%m%d_%H%M%S')}"
        record = _IngestionRecord(
            symbols=[str(sym) for sym in symbols],
            source=source,
            started_at=_utc_now(),
        )
        self._ingestion[task_id] = record
        self._save_state()
        return {
            "success": True,
            "task_id": task_id,
            "symbols": record.symbols,
            "source": source,
        }

    def trigger_backfill(
        self, symbols: Sequence[str], start: datetime, end: datetime
    ) -> dict[str, Any]:
        """
        Record a historical backfill request.
        """
        task_id = f"backfill_{_utc_now().strftime('%Y%m%d_%H%M%S')}"
        record = _IngestionRecord(
            symbols=[str(sym) for sym in symbols],
            source="backfill",
            started_at=_utc_now(),
            kind="backfill",
            details={
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )
        self._ingestion[task_id] = record
        self._save_state()
        return {
            "success": True,
            "task_id": task_id,
            "symbols": record.symbols,
            "date_range": f"{start.isoformat()} to {end.isoformat()}",
        }

    def emergency_stop_all(self) -> dict[str, Any]:
        """
        Clear all tracked actors, pipelines, and ingestion tasks.
        """
        stopped = {
            "actors": list(self._actors.keys()),
            "pipelines": list(self._pipelines.keys()),
            "ingestion": list(self._ingestion.keys()),
        }
        self._actors.clear()
        self._pipelines.clear()
        self._ingestion.clear()
        self._save_state()
        return {"success": True, "stopped_components": stopped, "stop_time": _utc_now().isoformat()}

    @classmethod
    def from_env(cls) -> SimpleControlPanel:
        """
        Construct a control panel using the default persisted state path.
        """
        return cls()


__all__ = ["SimpleControlPanel"]
