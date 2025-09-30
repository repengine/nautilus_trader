from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from ml.core.db_engine import EngineManager
from ml.core.integration import MLIntegrationManager


@contextmanager
def _env(var: str, value: str) -> Iterator[None]:
    old = os.environ.get(var)
    os.environ[var] = value
    try:
        yield
    finally:
        if old is None:
            del os.environ[var]
        else:
            os.environ[var] = old


def test_integration_manager_fallback_to_dummy(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force Postgres check to fail
    monkeypatch.setattr(MLIntegrationManager, "_is_postgres_running", lambda self: False)

    with _env("ML_ALLOW_DUMMY", "1"):
        mgr = MLIntegrationManager(config=None, ensure_healthy=False)
        # Stores/registries should be present even in dummy mode
        assert hasattr(mgr, "feature_store")
        assert hasattr(mgr, "model_store")
        assert hasattr(mgr, "strategy_store")
        assert hasattr(mgr, "data_registry")


def test_integration_manager_uses_env_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = "postgresql://postgres:postgres@localhost:5433/nautilus"

    monkeypatch.setattr(MLIntegrationManager, "_is_postgres_running", lambda self: True)
    monkeypatch.setattr(MLIntegrationManager, "_init_database", lambda self: None)
    monkeypatch.setattr(MLIntegrationManager, "_init_stores", lambda self: None)
    monkeypatch.setattr(MLIntegrationManager, "_init_registries", lambda self: None)
    monkeypatch.setattr(MLIntegrationManager, "_init_partition_manager", lambda self: None)
    monkeypatch.setattr(MLIntegrationManager, "ensure_healthy", lambda self: None)
    monkeypatch.setattr(
        MLIntegrationManager, "_validate_protocol_compliance", lambda self, strict=None: None
    )
    monkeypatch.setattr(MLIntegrationManager, "_maybe_run_backfill_on_start", lambda self: None)
    monkeypatch.delenv("NAUTILUS_DB", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("ML_DB_CONNECTION", raising=False)

    with _env("ML_DB_CONNECTION", expected):
        mgr = MLIntegrationManager(config=None, ensure_healthy=False)

    assert mgr.db_connection == expected


def test_is_postgres_running_switches_to_alternate_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSTGRES_HOST_PORT", raising=False)
    mgr = object.__new__(MLIntegrationManager)
    mgr.db_connection = "postgresql://postgres:postgres@localhost:5432/nautilus"
    mgr._connection_candidates = (
        "postgresql://postgres:postgres@localhost:5432/nautilus",
        "postgresql://postgres:postgres@localhost:5433/nautilus",
    )

    calls: list[str] = []

    def _fake_can_connect(self: MLIntegrationManager, conn: str) -> bool:
        calls.append(conn)
        return conn.endswith(":5433/nautilus")

    monkeypatch.setattr(MLIntegrationManager, "_can_connect", _fake_can_connect)
    disposed: list[str] = []
    monkeypatch.setattr(EngineManager, "dispose_engine", lambda conn: disposed.append(conn))

    assert MLIntegrationManager._is_postgres_running(mgr) is True
    assert mgr.db_connection.endswith(":5433/nautilus")
    assert disposed and disposed[0].endswith(":5432/nautilus")
    assert calls[0].endswith(":5432/nautilus")
    assert any(call.endswith(":5433/nautilus") for call in calls[1:])
