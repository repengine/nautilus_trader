"""Store health summary helpers for the dashboard (cold path)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.engine import Engine


StoreName = Literal["feature", "model", "strategy", "data"]


@dataclass(slots=True, frozen=True)
class StoreItemSummary:
    """Per-entity freshness metadata (e.g., dataset or instrument)."""

    key: str
    latest_event_ns: int | None
    latest_event_iso: str | None
    age_seconds: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "latest_event_ns": self.latest_event_ns,
            "latest_event_iso": self.latest_event_iso,
            "age_seconds": self.age_seconds,
        }


@dataclass(slots=True, frozen=True)
class StoreHealthSummary:
    """Aggregated health telemetry for a store."""

    store: StoreName
    healthy: bool
    connectivity_ok: bool | None
    write_ok: bool | None
    buffer_backlog: int | None
    latest_event_ns: int | None
    latest_event_iso: str | None
    age_seconds: float | None
    items: tuple[StoreItemSummary, ...]
    fallback_active: bool
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "store": self.store,
            "healthy": self.healthy,
            "connectivity_ok": self.connectivity_ok,
            "write_ok": self.write_ok,
            "buffer_backlog": self.buffer_backlog,
            "latest_event_ns": self.latest_event_ns,
            "latest_event_iso": self.latest_event_iso,
            "age_seconds": self.age_seconds,
            "items": [item.as_dict() for item in self.items],
            "fallback_active": self.fallback_active,
            "error": self.error,
        }


def _ns_to_datetime(ns: int | None) -> dt.datetime | None:
    if ns is None:
        return None
    try:
        return dt.datetime.fromtimestamp(ns / 1_000_000_000, tz=dt.UTC)
    except Exception:
        return None


def _age_seconds(moment: dt.datetime | None, *, now: dt.datetime) -> float | None:
    if moment is None:
        return None
    delta = now - moment
    return max(delta.total_seconds(), 0.0)


def _safe_health_detail(store: object | None) -> tuple[bool | None, bool | None, int | None, str | None]:
    if store is None:
        return None, None, None, "store_unavailable"
    try:
        details = getattr(store, "health_details")()
    except Exception as exc:  # pragma: no cover - defensive
        return None, None, None, str(exc)
    if not isinstance(details, dict):
        return None, None, None, "invalid_health_details"
    backlog_raw = details.get("buffer_backlog")
    backlog = int(backlog_raw) if isinstance(backlog_raw, (int, float)) else None
    connectivity = details.get("connectivity_ok")
    write_ok = details.get("write_ok")
    return (
        bool(connectivity) if connectivity is not None else None,
        bool(write_ok) if write_ok is not None else None,
        backlog,
        None,
    )


def _fetch_max_timestamp(
    engine: Engine | None,
    table: str,
    *,
    where_clause: str | None = None,
) -> tuple[int | None, str | None]:
    if engine is None:
        return None, "engine_unavailable"
    query = f"SELECT MAX(ts_event) AS ts_event FROM {table}"
    if where_clause:
        query += f" WHERE {where_clause}"
    try:
        with engine.connect() as conn:
            row = conn.execute(text(query)).first()
    except Exception as exc:
        return None, str(exc)
    if not row:
        return None, None
    value = row["ts_event"] if isinstance(row, dict) else row[0]
    if value is None:
        return None, None
    try:
        return int(value), None
    except Exception:
        return None, "invalid_timestamp"


def _fetch_dataset_top(
    engine: Engine | None,
    *,
    limit: int,
    now: dt.datetime,
) -> tuple[StoreItemSummary, ...]:
    if engine is None or limit <= 0:
        return ()
    query = (
        "SELECT dataset_type, MAX(ts_event) AS ts_event "
        "FROM ml_data_events "
        "GROUP BY dataset_type "
        "ORDER BY ts_event DESC NULLS LAST "
        "LIMIT :limit"
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(query), {"limit": limit}).fetchall()
    except Exception:
        return ()
    items: list[StoreItemSummary] = []
    for row in rows:
        dataset_type = row["dataset_type"] if isinstance(row, dict) else row[0]
        ts_value = row["ts_event"] if isinstance(row, dict) else row[1]
        latest_ns: int | None
        try:
            latest_ns = int(ts_value) if ts_value is not None else None
        except Exception:
            latest_ns = None
        moment = _ns_to_datetime(latest_ns)
        age = _age_seconds(moment, now=now) if moment is not None else None
        items.append(
            StoreItemSummary(
                key=str(dataset_type),
                latest_event_ns=latest_ns,
                latest_event_iso=moment.isoformat() if moment else None,
                age_seconds=age,
            ),
        )
    return tuple(items)


def _derive_health(
    connectivity_ok: bool | None,
    write_ok: bool | None,
    fallback_active: bool,
    latest_event_ns: int | None,
) -> bool:
    if connectivity_ok is not None and write_ok is not None:
        return bool(connectivity_ok and write_ok)
    if fallback_active:
        return False
    return latest_event_ns is not None


def summarize_feature_store(
    store: object | None,
    engine: Engine | None,
    *,
    now: dt.datetime | None = None,
) -> StoreHealthSummary:
    now = now or dt.datetime.now(dt.UTC)
    connectivity_ok, write_ok, backlog, health_error = _safe_health_detail(store)
    latest_ns, latest_error = _fetch_max_timestamp(engine, "ml_feature_values", where_clause="is_live = TRUE")
    moment = _ns_to_datetime(latest_ns)
    age = _age_seconds(moment, now=now)
    fallback_active = store is None or engine is None
    error = health_error or latest_error
    healthy = _derive_health(connectivity_ok, write_ok, fallback_active, latest_ns)
    return StoreHealthSummary(
        store="feature",
        healthy=healthy,
        connectivity_ok=connectivity_ok,
        write_ok=write_ok,
        buffer_backlog=backlog,
        latest_event_ns=latest_ns,
        latest_event_iso=moment.isoformat() if moment else None,
        age_seconds=age,
        items=(),
        fallback_active=fallback_active,
        error=error,
    )


def summarize_model_store(
    store: object | None,
    engine: Engine | None,
    *,
    now: dt.datetime | None = None,
) -> StoreHealthSummary:
    now = now or dt.datetime.now(dt.UTC)
    connectivity_ok, write_ok, backlog, health_error = _safe_health_detail(store)
    latest_ns, latest_error = _fetch_max_timestamp(engine, "ml_model_predictions", where_clause="is_live = TRUE")
    moment = _ns_to_datetime(latest_ns)
    age = _age_seconds(moment, now=now)
    fallback_active = store is None or engine is None
    error = health_error or latest_error
    healthy = _derive_health(connectivity_ok, write_ok, fallback_active, latest_ns)
    return StoreHealthSummary(
        store="model",
        healthy=healthy,
        connectivity_ok=connectivity_ok,
        write_ok=write_ok,
        buffer_backlog=backlog,
        latest_event_ns=latest_ns,
        latest_event_iso=moment.isoformat() if moment else None,
        age_seconds=age,
        items=(),
        fallback_active=fallback_active,
        error=error,
    )


def summarize_strategy_store(
    store: object | None,
    engine: Engine | None,
    *,
    now: dt.datetime | None = None,
) -> StoreHealthSummary:
    now = now or dt.datetime.now(dt.UTC)
    connectivity_ok, write_ok, backlog, health_error = _safe_health_detail(store)
    latest_ns, latest_error = _fetch_max_timestamp(engine, "ml_strategy_signals", where_clause="is_live = TRUE")
    moment = _ns_to_datetime(latest_ns)
    age = _age_seconds(moment, now=now)
    fallback_active = store is None or engine is None
    error = health_error or latest_error
    healthy = _derive_health(connectivity_ok, write_ok, fallback_active, latest_ns)
    return StoreHealthSummary(
        store="strategy",
        healthy=healthy,
        connectivity_ok=connectivity_ok,
        write_ok=write_ok,
        buffer_backlog=backlog,
        latest_event_ns=latest_ns,
        latest_event_iso=moment.isoformat() if moment else None,
        age_seconds=age,
        items=(),
        fallback_active=fallback_active,
        error=error,
    )


def summarize_data_store(
    engine: Engine | None,
    *,
    top_limit: int,
    now: dt.datetime | None = None,
) -> StoreHealthSummary:
    now = now or dt.datetime.now(dt.UTC)
    latest_ns, latest_error = _fetch_max_timestamp(engine, "ml_data_events", where_clause=None)
    moment = _ns_to_datetime(latest_ns)
    age = _age_seconds(moment, now=now)
    top_items = _fetch_dataset_top(engine, limit=top_limit, now=now)
    fallback_active = engine is None
    healthy = latest_ns is not None and not fallback_active
    return StoreHealthSummary(
        store="data",
        healthy=healthy,
        connectivity_ok=None,
        write_ok=None,
        buffer_backlog=None,
        latest_event_ns=latest_ns,
        latest_event_iso=moment.isoformat() if moment else None,
        age_seconds=age,
        items=top_items,
        fallback_active=fallback_active,
        error=latest_error,
    )


def summarize_all_stores(
    *,
    feature_store: object | None,
    model_store: object | None,
    strategy_store: object | None,
    engine: Engine | None,
    top_dataset_limit: int,
    now: dt.datetime | None = None,
) -> tuple[StoreHealthSummary, ...]:
    """Collect summaries for all configured stores."""
    return (
        summarize_feature_store(feature_store, engine, now=now),
        summarize_model_store(model_store, engine, now=now),
        summarize_strategy_store(strategy_store, engine, now=now),
        summarize_data_store(engine, top_limit=top_dataset_limit, now=now),
    )


__all__ = [
    "StoreHealthSummary",
    "StoreItemSummary",
    "summarize_all_stores",
    "summarize_data_store",
    "summarize_feature_store",
    "summarize_model_store",
    "summarize_strategy_store",
]
