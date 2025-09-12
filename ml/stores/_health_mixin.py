"""
Shared health probing mixin for SQL-backed stores.

Provides standardized connectivity and writeability probes and optional buffered
backlog reporting. Emits best-effort metrics via the metrics bootstrap helper.

This mixin avoids direct Prometheus collector instantiation: always acquire
metrics via ml.common.metrics_bootstrap.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

from ml.common.metrics_manager import MetricsManager


logger = logging.getLogger(__name__)


class HealthMixin:
    """
    Standard health checks for stores.

    Expects consumer classes to provide:
    - `engine`: SQLAlchemy Engine
    - `_write_buffer`: optional list-like for buffered writes
    - `__class__.__name__`: used for metric labels
    """

    engine: Any  # SQLAlchemy Engine at runtime

    # Gauges created via MetricsManager (delegates to bootstrap)
    _MM = MetricsManager.default()
    _health_gauge = _MM.gauge(
        "nautilus_ml_store_health_status",
        "Store health status (1=ok, 0=unhealthy)",
        ["store"],
    )
    _backlog_gauge = _MM.gauge(
        "nautilus_ml_store_buffer_backlog",
        "Buffered write backlog size",
        ["store"],
    )

    def _probe_connectivity(self) -> bool:
        try:
            eng = getattr(self, "engine", None)
            if eng is None:
                return True  # In-memory/dummy mode
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            logger.debug("Connectivity probe failed", exc_info=True)
            return False

    def _probe_writeability(self) -> bool:
        """
        Attempt a lightweight write in a transaction using a temporary table.

        Uses CREATE TEMP TABLE (Postgres) semantics; for other dialects, falls back
        to a no-op transaction block.
        """
        try:
            eng = getattr(self, "engine", None)
            if eng is None:
                return True
            dialect = getattr(eng, "dialect", None)
            name = getattr(dialect, "name", None) if dialect is not None else None
            with eng.begin() as conn:
                if name == "postgresql":
                    conn.execute(text("CREATE TEMP TABLE IF NOT EXISTS ml_health_probe(id INT)"))
                    conn.execute(text("INSERT INTO ml_health_probe (id) VALUES (1) ON CONFLICT DO NOTHING"))
                    # Clean up row; table drops at session end
                    conn.execute(text("DELETE FROM ml_health_probe WHERE id = 1"))
                else:
                    # For SQLite/others, a trivial BEGIN/COMMIT sequence suffices
                    conn.execute(text("SELECT 1"))
            return True
        except Exception:
            logger.debug("Writeability probe failed", exc_info=True)
            return False

    def _buffer_backlog(self) -> int:
        try:
            buf = getattr(self, "_write_buffer", None)
            if buf is None:
                return 0
            return len(buf)
        except Exception:
            return 0

    def health_details(self) -> dict[str, Any]:
        connectivity = self._probe_connectivity()
        write_ok = self._probe_writeability()
        backlog = self._buffer_backlog()
        details: dict[str, Any] = {
            "connectivity_ok": connectivity,
            "write_ok": write_ok,
            "buffer_backlog": backlog,
        }
        return details

    def is_healthy(self) -> bool:
        details = self.health_details()
        healthy = bool(details.get("connectivity_ok")) and bool(details.get("write_ok"))
        # Emit best-effort metrics; guard against metric init failures
        try:
            label = self.__class__.__name__
            self._health_gauge.labels(store=label).set(1.0 if healthy else 0.0)
            self._backlog_gauge.labels(store=label).set(float(details.get("buffer_backlog", 0)))
        except Exception:  # pragma: no cover - metrics optional
            logger.debug("Health metrics emission failed", exc_info=True)
        return healthy
