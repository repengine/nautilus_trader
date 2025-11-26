#!/usr/bin/env python3
"""
Runtime and environment fixtures for ML tests.

These fixtures centralize environment cleanup, logging configuration, and
Hypothesis adapters so test modules can rely on deterministic behaviour without
re-implementing boilerplate in ``conftest.py``.
"""

from __future__ import annotations

import gc
import logging
import os
import uuid
from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

ORCHESTRATOR_ENV_VARS: tuple[str, ...] = (
    "ORCH_CONFIG",
    "ORCH_DRY_RUN",
    "ORCH_FORCE",
    "ORCH_INTERVAL_MIN",
    "ORCH_LOCK_PATH",
    "ORCH_LOCK_TTL_HOURS",
    "ORCH_SCHEDULE_TIME",
)

@pytest.fixture(autouse=True)
def cleanup_after_test() -> Generator[None, None, None]:
    """Automatic cleanup after each test to avoid cross-test pollution."""

    yield

    try:
        from ml.core.cache import clear_all_caches

        clear_all_caches()
    except ImportError:
        pass

    try:
        from ml.config import reset_global_config

        reset_global_config()
    except ImportError:
        pass

    gc.collect()


@pytest.fixture(autouse=False)
def cleanup_engines() -> None:
    """Deprecated per-test engine cleanup retained for compatibility."""

    return None


@pytest.fixture(autouse=True, scope="session")
def configure_test_logging() -> Generator[None, None, None]:
    """Configure logging for tests and restore the original configuration afterwards."""

    original_levels = {
        "sqlalchemy.engine": logging.getLogger("sqlalchemy.engine").level,
        "ml": logging.getLogger("ml").level,
        "root": logging.root.level,
    }
    original_handlers = logging.root.handlers.copy()

    try:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("ml").setLevel(logging.INFO)

        test_run_id = str(uuid.uuid4())[:8]
        logging.basicConfig(
            format=f"[{test_run_id}] %(levelname)s %(name)s: %(message)s",
            level=logging.INFO,
        )

        yield
    finally:
        logging.getLogger("sqlalchemy.engine").setLevel(original_levels["sqlalchemy.engine"])
        logging.getLogger("ml").setLevel(original_levels["ml"])
        logging.root.setLevel(original_levels["root"])
        logging.root.handlers = original_handlers


@pytest.fixture
def valid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a minimal valid environment for deployment entrypoint tests."""

    monkeypatch.setenv("DB_CONNECTION", "postgresql://test:test@localhost:5432/test")
    monkeypatch.setenv("STRATEGY_ID", "MLStrategy-TEST-001")
    monkeypatch.setenv("ML_SIGNAL_SOURCE", "MLSignalActor-001")
    monkeypatch.setenv("INSTRUMENT_ID", "BTC-USDT.DATABENTO")
    monkeypatch.setenv("EXECUTE_TRADES", "false")
    monkeypatch.setenv("POSITION_SIZE_PCT", "0.02")
    monkeypatch.setenv("MIN_CONFIDENCE", "0.6")
    monkeypatch.setenv("MAX_POSITIONS", "3")
    monkeypatch.setenv("STOP_LOSS_PCT", "0.02")
    monkeypatch.setenv("TAKE_PROFIT_PCT", "0.04")
    monkeypatch.setenv("USE_STRATEGY_STORE", "true")
    monkeypatch.setenv("PERSIST_ALL_SIGNALS", "true")


@pytest.fixture(scope="session", autouse=True)
def _set_isolated_ml_registry_path(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[None, None, None]:
    """Force ML registries to use a temporary directory during tests."""

    previous_value = os.getenv("ML_REGISTRY_PATH")
    registry_dir = tmp_path_factory.mktemp("ml_registry")
    os.environ["ML_REGISTRY_PATH"] = str(registry_dir)
    try:
        yield
    finally:
        if previous_value is None:
            os.environ.pop("ML_REGISTRY_PATH", None)
        else:
            os.environ["ML_REGISTRY_PATH"] = previous_value


@pytest.fixture
def isolated_orchestrator_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Clear pipeline orchestrator environment variables for the duration of a test.

    Tests can opt into this fixture to ensure scheduler helpers never inherit
    stray ORCH_* values from other shards while still using monkeypatch to
    set explicit overrides.
    """

    for key in ORCHESTRATOR_ENV_VARS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def hypothesis_database_session() -> Generator[Session, None, None]:
    """Provide a lightweight SQLite session for Hypothesis-driven property tests."""

    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=NullPool,
    )

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        yield session
    finally:
        session.close()
        engine.dispose()


__all__ = [
    "_set_isolated_ml_registry_path",
    "cleanup_after_test",
    "cleanup_engines",
    "configure_test_logging",
    "hypothesis_database_session",
    "isolated_orchestrator_env",
    "valid_env",
]
