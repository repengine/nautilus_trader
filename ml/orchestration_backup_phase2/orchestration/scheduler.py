#!/usr/bin/env python3

from __future__ import annotations


# ruff: noqa: E402  # Allow module docstring preceding imports per project style

"""
Minimal pipeline scheduler (cold path).

This module provides a tiny loop which periodically invokes the existing
orchestrator CLI with arguments derived from a config file. It is designed
to be safe, typed, and easy to test.

Non-goals: DAG engines, bus consumers, hot-path changes.
"""

import logging
import tempfile
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

from ml.common.event_emitter import emit_dataset_event as _emit_dataset_event
from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_histogram
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.orchestration.config_loader import OrchestratorRunConfig
from ml.orchestration.config_loader import Stage as OrchestratorStage
from ml.registry.data_registry import DataRegistry


logger = logging.getLogger(__name__)


class _ConfigLoaderProtocol(Protocol):
    def load_orchestrator_run_config(
        self,
        path: str | Path,
        *,
        env: Mapping[str, str] | None = None,
    ) -> OrchestratorRunConfig: ...


class _EmitEventProtocol(Protocol):
    def __call__(
        self,
        registry: Any,
        *,
        dataset_id: str,
        instrument_id: str,
        stage: Stage,
        source: Source,
        run_id: str,
        ts_min: int,
        ts_max: int,
        count: int,
        status: EventStatus,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
        dataset_type: str | None = None,
        component: str | None = None,
    ) -> None: ...


@dataclass(slots=True, frozen=True)
class _Metrics:
    runs_total: Any
    phase_latency: Any

    @staticmethod
    def default() -> _Metrics:
        return _Metrics(
            runs_total=get_counter(
                "nautilus_ml_orch_runs_total",
                "Total orchestrator runs by status",
                ("status",),
            ),
            phase_latency=get_histogram(
                "nautilus_ml_orch_phase_latency_seconds",
                "Orchestrator phase durations (seconds)",
                ("phase",),
            ),
        )


def compute_next_run(
    schedule_time: str | None,
    interval_min: int | None,
    now: datetime,
) -> datetime:
    """
    Compute next run time from a daily UTC schedule or interval.

    Precedence: schedule_time > interval_min. Raises ValueError if neither
    is provided or inputs are invalid.

    """
    if schedule_time:
        st = schedule_time.strip().upper()
        if not st.endswith("Z"):
            raise ValueError("schedule_time must end with 'Z' and be in HH:MMZ (UTC)")
        hhmm = st[:-1]
        parts = hhmm.split(":")
        if len(parts) != 2:
            raise ValueError("schedule_time must be in HH:MMZ form")
        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except ValueError as exc:  # pragma: no cover - unlikely with argparse
            raise ValueError("Invalid HH:MMZ components") from exc
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError("Hour must be 0-23 and minute 0-59")

        # Normalize now to UTC aware
        now_utc = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
        candidate = now_utc.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now_utc:
            candidate = candidate + timedelta(days=1)
        return candidate

    if interval_min is not None:
        if interval_min <= 0:
            raise ValueError("interval_min must be > 0")
        base = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
        return base + timedelta(minutes=int(interval_min))

    raise ValueError("Either schedule_time or interval_min must be provided")


def _lock_is_stale(lock_path: Path, ttl_hours: int) -> bool:
    if not lock_path.exists():
        return True
    try:
        mtime = datetime.fromtimestamp(lock_path.stat().st_mtime, tz=UTC)
    except Exception:
        return True
    return (datetime.now(tz=UTC) - mtime) > timedelta(hours=ttl_hours)


def _try_acquire_lock(lock_path: Path, ttl_hours: int) -> bool:
    # Remove stale lock
    if lock_path.exists() and _lock_is_stale(lock_path, ttl_hours):
        try:
            lock_path.unlink()
        except Exception as exc:
            logger.debug("Unlink stale lock failed: %s", exc, exc_info=True)
    if lock_path.exists():
        return False
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(str(datetime.now(tz=UTC).isoformat()), encoding="utf-8")
        return True
    except Exception:
        return False


def _release_lock(lock_path: Path) -> None:
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception as exc:
        # best-effort
        logger.debug("Release lock failed: %s", exc, exc_info=True)


def run_forever(
    config_loader: _ConfigLoaderProtocol,
    invoke_pipeline: Callable[[list[str]], int],
    sleep_fn: Callable[[float], None],
    emit_event: _EmitEventProtocol = _emit_dataset_event,
    metrics: _Metrics | None = None,
) -> None:
    """
    Run the orchestrator on a schedule forever.

    Environment variables honored:
    - ORCH_SCHEDULE_TIME=HH:MMZ (UTC daily time)
    - ORCH_INTERVAL_MIN=N (minutes interval)
    - ORCH_CONFIG=path/to/config.{json|toml}
    - ORCH_DRY_RUN=1 (log only)
    - ORCH_FORCE=1 (ignore existing outputs)
    - ORCH_LOCK_PATH=custom lock file path
    - ORCH_LOCK_TTL_HOURS=12 (stale lock threshold)

    """
    import os
    import time

    m = metrics or _Metrics.default()

    schedule_time = os.getenv("ORCH_SCHEDULE_TIME")
    interval_env = os.getenv("ORCH_INTERVAL_MIN")
    interval_min = int(interval_env) if interval_env else None
    cfg_path = os.getenv("ORCH_CONFIG")
    if cfg_path is None:
        raise ValueError("ORCH_CONFIG must be set to run the orchestrator scheduler")
    force = os.getenv("ORCH_FORCE", "").strip() in {"1", "true", "yes"}
    lock_ttl_hours = int(os.getenv("ORCH_LOCK_TTL_HOURS", "12"))

    # Load orchestrator config once; callers can restart the service to pick up changes
    run_cfg = config_loader.load_orchestrator_run_config(cfg_path)
    cfg = run_cfg.compose_orchestrator_config()
    stage_override = run_cfg.stage
    args = ["--config", cfg_path]
    if stage_override is not OrchestratorStage.FULL:
        args += ["--stage", stage_override.value]

    # Determine default lock path from config out_dir
    try:
        out_dir = None
        if hasattr(cfg, "dataset") and hasattr(cfg.dataset, "out_dir"):
            out_dir = Path(str(cfg.dataset.out_dir))
        default_lock = (
            out_dir / ".orch.lock"
            if out_dir
            else Path(tempfile.gettempdir()) / "ml_orch.lock"
        )
    except Exception as exc:
        import logging as _logging

        _logging.getLogger(__name__).debug("Deriving default lock path failed: %s", exc)
        default_lock = Path(tempfile.gettempdir()) / "ml_orch.lock"

    lock_path = Path(os.getenv("ORCH_LOCK_PATH", str(default_lock)))

    # Integration manager provides a registry for events
    registry: DataRegistry | None = None
    try:
        from ml.core.integration import MLIntegrationManager

        mgr = MLIntegrationManager(
            auto_start_postgres=False,
            auto_migrate=False,
            ensure_healthy=False,
        )
        registry = mgr.data_registry
    except Exception as exc:
        import logging as _logging

        _logging.getLogger(__name__).debug("MLIntegrationManager not available: %s", exc)
        registry = None

    while True:
        now = datetime.now(tz=UTC)
        # Evaluate dry-run at each tick to honor env changes between invocations
        dry_run = os.getenv("ORCH_DRY_RUN", "").strip() in {"1", "true", "yes"}
        try:
            next_run = compute_next_run(schedule_time, interval_min, now)
        except Exception as exc:
            # If schedule invalid, wait a minute and retry
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "Invalid scheduler config; retrying in 60s: %s",
                exc,
            )
            sleep_fn(60.0)
            continue

        # Sleep until scheduled time
        delta = (next_run - now).total_seconds()
        if delta > 0:
            sleep_fn(delta)

        # Lock to prevent overlap
        if not _try_acquire_lock(lock_path, ttl_hours=lock_ttl_hours):
            # Another run in progress; skip this tick
            continue

        run_id = f"orch_{time.time_ns()}"
        t0 = time.perf_counter()
        status = EventStatus.SUCCESS
        error: str | None = None

        try:
            # Skip if outputs already exist unless forced
            if not force:
                try:
                    if out_dir is not None and (Path(out_dir) / "dataset.csv").exists():
                        # Skip this run quietly
                        status = EventStatus.SUCCESS
                        # Record a near-zero duration for the skipped phase
                        m.phase_latency.labels(phase="pipeline").observe(0.0)
                        continue
                except Exception as exc:
                    import logging as _logging

                    _logging.getLogger(__name__).debug(
                        "Output precheck failed (ignored): %s",
                        exc,
                    )

            # Execute orchestrator
            if dry_run:
                rc = 0
            else:
                rc = int(invoke_pipeline(args))
            if rc != 0:
                status = EventStatus.FAILED
                error = f"orchestrator exited with code {rc}"
        except Exception as exc:  # keep scheduler resilient
            status = EventStatus.FAILED
            error = str(exc)
        finally:
            # Observe total latency and emit a single summary event per run
            duration = max(0.0, time.perf_counter() - t0)
            try:
                m.phase_latency.labels(phase="pipeline").observe(duration)
                m.runs_total.labels(status=status.value).inc()
            except Exception as exc:
                import logging as _logging

                _logging.getLogger(__name__).debug(
                    "Emit orchestrator event failed (ignored): %s",
                    exc,
                )

            # Emit event (status only; no watermark updates in scheduler)
            try:
                from ml.common.timestamps import sanitize_timestamp_ns as _sanitize

                now_ns = _sanitize(
                    int(time.time_ns()),
                    context="orchestration.scheduler:emit_event.now",
                )
                metadata = {"phase": "pipeline", "run_id": run_id, "duration": duration}
                if error:
                    metadata["error"] = error
                if registry is None:
                    logger.debug("Skipping scheduler dataset event; registry unavailable")
                else:
                    emit_event(
                        registry,
                        dataset_id="ml_pipeline",
                        instrument_id="GLOBAL",
                        stage=Stage.FEATURE_COMPUTED,
                        source=Source.HISTORICAL,
                        run_id=run_id,
                        ts_min=now_ns,
                        ts_max=now_ns,
                        count=1,
                        status=status,
                        error=error,
                        metadata=metadata,
                        dataset_type="pipeline",
                        component="scheduler",
                    )
            except Exception as exc:
                logger.debug(
                    "Emit scheduler dataset event failed (ignored): %s",
                    exc,
                    exc_info=True,
                )

            # Always release the lock for the next schedule
            _release_lock(lock_path)


__all__ = ["compute_next_run", "run_forever"]
