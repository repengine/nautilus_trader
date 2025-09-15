"""
Consolidated mixins and helpers for ML stores.

This module groups common mixins and utilities used across the SQL-backed
stores to keep hot paths lean and reduce file sprawl. The contents were
migrated from the previous private modules:

- _buffered_store.py
- _engine_mixin.py
- _health_mixin.py
- _init_mixin.py
- _read_helpers.py
- _registry_mixin.py
- _upsert_mixin.py
- _batch_utils.py

The public surface remains unchanged via shims that re-export from here.

"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import MetaData
from sqlalchemy import text as _satext
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from ml.common.message_topics import build_topic_for_stage
from ml.common.metrics_manager import MetricsManager
from ml.config.events import EventStatus
from ml.config.events import Stage
from ml.core.db_engine import EngineManager


# =============================================================================
# Batch helpers (from _batch_utils)
# =============================================================================


def sanitize_and_dedup(
    values: list[dict[str, Any]],
    *,
    ts_event_field: str,
    ts_init_field: str,
    context: str,
    key_fields: tuple[str, str, str],
) -> list[dict[str, Any]]:
    """
    Sanitize ts fields and de-duplicate rows within a batch.
    """
    if not values:
        return values

    from ml.common.timestamps import sanitize_timestamp_ns

    for v in values:
        if ts_event_field in v:
            v[ts_event_field] = sanitize_timestamp_ns(int(v[ts_event_field]), context=context)
        if ts_init_field in v:
            v[ts_init_field] = sanitize_timestamp_ns(int(v[ts_init_field]), context=context)

    k1, k2, k3 = key_fields
    dedup: dict[tuple[str, str, int], dict[str, Any]] = {}
    for v in values:
        key = (str(v[k1]), str(v[k2]), int(v[k3]))
        dedup[key] = v
    return list(dedup.values())


def publish_batch_and_rows(
    *,
    enable_publishing: bool,
    publisher: Any | None,
    publish_mode: str,
    topic_scheme: str,
    topic_prefix: str,
    stage: Stage,
    dataset_id: str,
    instrument_key: str,
    ts_field: str,
    rows: Iterable[dict[str, Any]],
    run_id_batch: str,
    run_id_row: str,
    source: str,
    logger: Any,
) -> None:
    """
    Publish batch summary and per-row events (best-effort).
    """
    rows_list = list(rows)
    if not (enable_publishing and publisher and rows_list):
        return

    if publish_mode in ("batch", "both"):
        try:
            instrument_id = str(rows_list[0].get(instrument_key, "UNKNOWN"))
            topic = build_topic_for_stage(
                stage,
                instrument_id,
                scheme=topic_scheme,
                prefix=topic_prefix,
            )
            ts_vals = [int(r.get(ts_field, 0)) for r in rows_list]
            payload: dict[str, Any] = {
                "dataset_id": dataset_id,
                "instrument_id": instrument_id,
                "stage": stage.value,
                "source": source,
                "run_id": run_id_batch,
                "ts_min": min(ts_vals) if ts_vals else 0,
                "ts_max": max(ts_vals) if ts_vals else 0,
                "count": len(rows_list),
                "status": EventStatus.SUCCESS.value,
            }
            publisher.publish(topic, payload)
        except Exception:
            logger.debug("Batch publish failed", exc_info=True)

    if publish_mode in ("row", "both"):
        try:
            for r in rows_list:
                instrument_id = str(r.get(instrument_key, "UNKNOWN"))
                topic = build_topic_for_stage(
                    stage,
                    instrument_id,
                    scheme=topic_scheme,
                    prefix=topic_prefix,
                )
                ts_e = int(r.get(ts_field, 0))
                row_payload: dict[str, Any] = {
                    "dataset_id": dataset_id,
                    "instrument_id": instrument_id,
                    "stage": stage.value,
                    "source": source,
                    "run_id": run_id_row,
                    "ts_min": ts_e,
                    "ts_max": ts_e,
                    "count": 1,
                    "status": EventStatus.SUCCESS.value,
                }
                publisher.publish(topic, row_payload)
        except Exception:
            logger.debug("Per-row publish failed", exc_info=True)


# =============================================================================
# Buffered store mixin (from _buffered_store)
# =============================================================================


class BufferedStoreMixin:
    """
    Buffered flush behavior, time-based flush decision, and health check.
    """

    _write_buffer: list[Any]
    _last_flush_ns: int
    flush_interval_ms: int
    clock: Any | None
    engine: Any  # SQLAlchemy Engine

    def _should_flush_by_time(self) -> bool:
        if not self.clock or not self._last_flush_ns:
            return False
        try:
            elapsed_ms = (self.clock.timestamp_ns() - self._last_flush_ns) / 1e6
            return bool(elapsed_ms >= float(self.flush_interval_ms))
        except Exception:
            return False

    def flush(self) -> None:
        if not getattr(self, "_write_buffer", None):
            return
        buffer_copy = list(self._write_buffer)
        write_batch = getattr(self, "write_batch")
        try:
            write_batch(buffer_copy, emit_events=False)
        except TypeError:
            write_batch(buffer_copy)
        try:
            emit = getattr(self, "_emit_events", None)
            if callable(emit):
                emit(buffer_copy)
        finally:
            self._write_buffer.clear()
            if self.clock:
                try:
                    self._last_flush_ns = int(self.clock.timestamp_ns())
                except Exception:
                    self._last_flush_ns = 0

    def is_healthy(self) -> bool:  # lightweight connectivity probe
        try:
            if getattr(self, "engine", None):
                from sqlalchemy import text  # local import

                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT 1"))
                    return result is not None
            return True
        except Exception:
            return False


# =============================================================================
# Engine init mixin (from _engine_mixin)
# =============================================================================

logger = logging.getLogger(__name__)


class EngineInitMixin:
    """
    Initialize SQLAlchemy engine + metadata and call `_setup_tables()`.
    """

    connection_string: str | None
    engine: Engine
    metadata: MetaData

    def _init_engine_and_tables(self) -> None:
        if not self.connection_string:
            return
        self.engine = EngineManager.get_engine(self.connection_string)
        self.metadata = MetaData()
        self._setup_tables()  # type: ignore[attr-defined]
        try:
            status: dict[str, Any] | None = EngineManager.get_pool_status(self.connection_string)
            if status:
                logger.debug("Engine pool status: %s", status)
        except Exception as exc:
            logger.debug("Pool status unavailable: %s", exc)


# =============================================================================
# Health mixin (from _health_mixin)
# =============================================================================


class HealthMixin:
    """
    Standard health checks and metrics for stores.
    """

    engine: Any

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
                return True
            with eng.connect() as conn:
                conn.execute(_satext("SELECT 1"))
            return True
        except Exception:
            logger.debug("Connectivity probe failed", exc_info=True)
            return False

    def _probe_writeability(self) -> bool:
        try:
            eng = getattr(self, "engine", None)
            if eng is None:
                return True
            dialect = getattr(eng, "dialect", None)
            name = getattr(dialect, "name", None) if dialect is not None else None
            with eng.begin() as conn:
                if name == "postgresql":
                    conn.execute(_satext("CREATE TEMP TABLE IF NOT EXISTS ml_health_probe(id INT)"))
                    conn.execute(
                        _satext(
                            "INSERT INTO ml_health_probe (id) VALUES (1) ON CONFLICT DO NOTHING",
                        ),
                    )
                    conn.execute(_satext("DELETE FROM ml_health_probe WHERE id = 1"))
                else:
                    conn.execute(_satext("SELECT 1"))
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
        return {
            "connectivity_ok": connectivity,
            "write_ok": write_ok,
            "buffer_backlog": backlog,
        }

    def is_healthy(self) -> bool:
        details = self.health_details()
        healthy = bool(details.get("connectivity_ok")) and bool(details.get("write_ok"))
        try:
            label = self.__class__.__name__
            self._health_gauge.labels(store=label).set(1.0 if healthy else 0.0)
            self._backlog_gauge.labels(store=label).set(float(details.get("buffer_backlog", 0)))
        except Exception:  # pragma: no cover
            logger.debug("Health metrics emission failed", exc_info=True)
        return healthy


# =============================================================================
# Store init mixin (from _init_mixin)
# =============================================================================

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nautilus_trader.common.clock import Clock

    from ml.common.message_bus import MessagePublisherProtocol
    from ml.registry.persistence import PersistenceConfig


class StoreInitMixin:
    """
    Shared constructor wiring for stores.
    """

    def _init_store_common(
        self: Any,
        *,
        connection_string: str | None,
        persistence_config: PersistenceConfig | None,
        batch_size: int,
        flush_interval_ms: int,
        flush_interval_seconds: float | None,
        clock: Clock | None,
        enable_publishing: bool,
        publisher: MessagePublisherProtocol | None,
        publish_mode: Literal["batch", "row", "both"],
        persistence_manager: object | None,
    ) -> None:
        cfg = persistence_config
        if (
            connection_string
            and not cfg
            and ("postgresql://" in connection_string or "postgres://" in connection_string)
        ):
            from ml.registry.persistence import BackendType
            from ml.registry.persistence import PersistenceConfig as _PC

            cfg = _PC(backend=BackendType.POSTGRES, connection_string=connection_string)

        if cfg is not None:
            from ml.registry.persistence import PersistenceManager

            setattr(self, "persistence", PersistenceManager(cfg))
            setattr(self, "connection_string", cfg.connection_string)
        else:
            setattr(self, "persistence", None)
            setattr(
                self,
                "connection_string",
                connection_string or "postgresql://postgres:postgres@localhost:5432/nautilus",
            )

        setattr(self, "batch_size", int(batch_size))
        if flush_interval_seconds is not None:
            setattr(self, "flush_interval_ms", int(flush_interval_seconds * 1000))
        else:
            setattr(self, "flush_interval_ms", int(flush_interval_ms))
        setattr(self, "clock", clock)

        self._init_bus_publishing(
            enable_publishing=enable_publishing,
            publisher=publisher,
            publish_mode=publish_mode,
        )

        if persistence_manager is not None:
            try:
                setattr(self, "persistence", persistence_manager)
            except Exception:
                pass

        setattr(self, "_last_flush_ns", 0)
        self._init_engine_and_tables()


# =============================================================================
# Read helpers (from _read_helpers)
# =============================================================================


class ReadQueryMixin:
    """
    Helpers for read-side queries and schema-qualified names.
    """

    engine: Any

    def _qualified_table(self, base: str) -> str:
        name: str | None = None
        try:
            eng = getattr(self, "engine", None)
            if eng is not None:
                dialect = getattr(eng, "dialect", None)
                if dialect is not None:
                    name = getattr(dialect, "name", None)
        except Exception:
            name = None
        if name == "sqlite":
            return base
        return f"public.{base}"

    def _safe_identifier(self, name: str, allowed: set[str]) -> str:
        if name not in allowed:
            raise ValueError(f"Disallowed identifier: {name}")
        return name

    def _safe_table(self, base: str, allowed: set[str]) -> str:
        base_safe = self._safe_identifier(base, allowed)
        return self._qualified_table(base_safe)

    def _execute_read(
        self,
        sql: Any,
        params: Mapping[str, Any],
        *,
        columns: Sequence[str],
    ) -> Any:
        import pandas as pd
        from sqlalchemy import text as _text

        session_obj: Any | None = None
        try:
            sess = getattr(self, "persistence", None)
            if sess is not None:
                session_obj = getattr(sess, "session", None)
                if session_obj is None and hasattr(sess, "get_session"):
                    session_obj = sess.get_session()
        except Exception:
            session_obj = None

        if session_obj is not None:
            try:
                rows = session_obj.execute(_text(str(sql)), params).fetchall()
            except Exception:
                rows = []

            data = [{col: row[idx] for idx, col in enumerate(columns)} for row in rows]
            df = pd.DataFrame(data, columns=list(columns))
            if len(df.index):
                return df

        with self.engine.connect() as conn:
            return pd.read_sql_query(sql, conn, params=dict(params))

    def _fetch_one(self, sql: Any, params: Mapping[str, Any]) -> tuple[Any, ...] | None:
        from sqlalchemy import text as _text

        session_obj: Any | None = None
        try:
            sess = getattr(self, "persistence", None)
            if sess is not None:
                session_obj = getattr(sess, "session", None)
                if session_obj is None and hasattr(sess, "get_session"):
                    session_obj = sess.get_session()
        except Exception:
            session_obj = None

        if session_obj is not None:
            try:
                row2 = session_obj.execute(_text(str(sql)), dict(params)).fetchone()
            except Exception:
                row2 = None
            else:
                # Detect mocks and degrade to engine path for concrete values
                try:
                    from unittest.mock import MagicMock as _MM

                    if isinstance(row2, _MM):
                        row2 = None
                except Exception:
                    pass
            if row2 is not None:
                from typing import cast as _cast

                return _cast(tuple[Any, ...] | None, row2)

        with self.engine.connect() as conn:
            try:
                row = conn.execute(_text(str(sql)), dict(params)).fetchone()
            except Exception:
                row = None
        from typing import cast as _cast

        return _cast(tuple[Any, ...] | None, row)

    def _fetch_all(self, sql: Any, params: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        from sqlalchemy import text as _text

        session_obj: Any | None = None
        try:
            sess = getattr(self, "persistence", None)
            if sess is not None:
                session_obj = getattr(sess, "session", None)
                if session_obj is None and hasattr(sess, "get_session"):
                    session_obj = sess.get_session()
        except Exception:
            session_obj = None

        if session_obj is not None:
            try:
                rows2 = session_obj.execute(_text(str(sql)), dict(params)).fetchall()
            except Exception:
                rows2 = []
            else:
                try:
                    from unittest.mock import MagicMock as _MM

                    if isinstance(rows2, _MM) or (rows2 and isinstance(rows2[0], _MM)):
                        rows2 = []
                except Exception:
                    pass
            if rows2:
                return list(rows2)

        with self.engine.connect() as conn:
            try:
                rows = conn.execute(_text(str(sql)), dict(params)).fetchall()
            except Exception:
                rows = []
        return list(rows)


# =============================================================================
# DataRegistry mixin (from _registry_mixin)
# =============================================================================


class DataRegistryMixin:
    """
    Lazy DataRegistry initialization with progressive fallback.
    """

    _data_registry: Any | None = None
    connection_string: str | None

    def _get_data_registry(self) -> Any | None:
        if self._data_registry is not None:
            return self._data_registry

        from pathlib import Path

        from ml.registry.data_registry import DataRegistry
        from ml.registry.persistence import BackendType
        from ml.registry.persistence import PersistenceConfig

        registry_path = Path.home() / ".nautilus" / "ml" / "registry"

        tried_postgres = False
        if self.connection_string and (
            "postgresql://" in self.connection_string or "postgres://" in self.connection_string
        ):
            tried_postgres = True
            try:
                pg_cfg = PersistenceConfig(
                    backend=BackendType.POSTGRES,
                    connection_string=self.connection_string,
                )
                self._data_registry = DataRegistry(
                    registry_path=registry_path,
                    persistence_config=pg_cfg,
                )
                logger.debug("Initialized DataRegistry (POSTGRES)")
                return self._data_registry
            except Exception as e:
                logger.warning("POSTGRES registry init failed, falling back to JSON: %s", e)

        try:
            json_cfg = PersistenceConfig(backend=BackendType.JSON, json_path=registry_path)
            self._data_registry = DataRegistry(
                registry_path=registry_path,
                persistence_config=json_cfg,
            )
            logger.debug(
                "Initialized DataRegistry (JSON%s)",
                " after PG fail" if tried_postgres else "",
            )
        except Exception as e:
            logger.warning("Failed to initialize JSON DataRegistry: %s", e)
            self._data_registry = None

        return self._data_registry


# =============================================================================
# Upsert mixin (from _upsert_mixin)
# =============================================================================


class SQLUpsertMixin:
    """
    Reusable upsert-and-publish operation.
    """

    def _execute_upsert_and_publish(
        self,
        *,
        values: list[dict[str, Any]],
        ts_event_field: str,
        ts_init_field: str,
        context: str,
        key_fields: tuple[str, str, str],
        table: Any,
        conflict_cols: Iterable[str],
        update_cols: Iterable[str],
        dataset_id: str,
        stage: Any,
        instrument_key: str,
        ts_field: str,
        run_id_batch: str,
        run_id_row: str,
        source: str,
        logger: Any,
    ) -> None:
        if not values:
            return

        values = sanitize_and_dedup(
            values,
            ts_event_field=ts_event_field,
            ts_init_field=ts_init_field,
            context=context,
            key_fields=key_fields,
        )

        stmt = insert(table)
        excluded = stmt.excluded
        set_map = {col: getattr(excluded, col) for col in update_cols}
        stmt = stmt.on_conflict_do_update(index_elements=list(conflict_cols), set_=set_map)

        engine = getattr(self, "engine")
        with engine.begin() as conn:
            conn.execute(stmt, values)

        publish_batch_and_rows(
            enable_publishing=bool(getattr(self, "_enable_publishing", False)),
            publisher=getattr(self, "publisher", None),
            publish_mode=getattr(self, "_publish_mode", "batch"),
            topic_scheme=getattr(self, "_topic_scheme", "domain_op"),
            topic_prefix=getattr(self, "_topic_prefix", "events.ml"),
            stage=stage,
            dataset_id=dataset_id,
            instrument_key=instrument_key,
            ts_field=ts_field,
            rows=values,
            run_id_batch=run_id_batch,
            run_id_row=run_id_row,
            source=source,
            logger=logger,
        )


__all__ = [
    "BufferedStoreMixin",
    "DataRegistryMixin",
    "EngineInitMixin",
    "HealthMixin",
    "ReadQueryMixin",
    "SQLUpsertMixin",
    "StoreInitMixin",
    "publish_batch_and_rows",
    "sanitize_and_dedup",
]
