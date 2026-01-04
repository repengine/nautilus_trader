"""Tests for database utility helpers."""

from __future__ import annotations

from typing import Callable, ContextManager
from unittest.mock import MagicMock, Mock

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

import ml.common.db_utils as db_utils
from ml.common.db_utils import (
    STORE_PARTITIONED_TABLES,
    ensure_default_partition,
    ensure_partition_tables_ready,
    get_or_create_engine,
    get_default_pool_config,
)
from ml.tests.utils.db import build_postgres_url

pytestmark = pytest.mark.serial

PatchEngineManager = Callable[..., ContextManager[MagicMock]]

def _first_engine_call(engine: MagicMock) -> tuple[tuple[object, ...], dict[str, object]]:
    calls = getattr(engine, "_engine_manager_calls", [])
    assert calls, "EngineManager.get_engine was not invoked"
    return calls[0]

def test_get_default_pool_config():
    """Default pool config returns expected values."""
    config = get_default_pool_config()
    assert config["pool_size"] == 5
    assert config["max_overflow"] == 10
    assert config["pool_pre_ping"] is True
    assert config["pool_recycle"] == 3600

def test_get_or_create_engine_with_defaults(patch_engine_manager: PatchEngineManager):
    """Engine created with default pool settings."""
    with patch_engine_manager(record_calls=True) as mock_engine:
        engine = get_or_create_engine("postgresql://localhost/test")

    args, kwargs = _first_engine_call(mock_engine)
    assert engine == mock_engine
    assert args[0] == "postgresql://localhost/test"
    assert kwargs["pool_size"] == 5
    assert kwargs["max_overflow"] == 10
    assert kwargs["pool_pre_ping"] is True
    assert kwargs["pool_recycle"] == 3600

def test_get_or_create_engine_with_custom_settings(patch_engine_manager: PatchEngineManager):
    """Engine created with custom pool settings."""
    with patch_engine_manager(record_calls=True) as mock_engine:
        engine = get_or_create_engine(
            "postgresql://localhost/test",
            pool_size=10,
            max_overflow=20,
            pool_recycle=7200,
        )

    _, kwargs = _first_engine_call(mock_engine)
    assert engine == mock_engine
    assert kwargs["pool_size"] == 10
    assert kwargs["max_overflow"] == 20
    assert kwargs["pool_pre_ping"] is True
    assert kwargs["pool_recycle"] == 7200

def test_get_or_create_engine_empty_connection_string():
    """Raises ValueError for empty connection string."""
    with pytest.raises(ValueError, match="connection_string cannot be empty"):
        get_or_create_engine("")

def test_get_or_create_engine_handles_engine_manager_error(patch_engine_manager: PatchEngineManager):
    """RuntimeError raised when EngineManager fails."""
    with patch_engine_manager(side_effect=Exception("Connection failed")):
        with pytest.raises(RuntimeError, match="Database engine creation failed"):
            get_or_create_engine("postgresql://localhost/test")

def test_connection_string_sanitized_in_logs(caplog, patch_engine_manager: PatchEngineManager):
    """Connection string credentials not leaked in logs."""
    with patch_engine_manager():
        get_or_create_engine(
            build_postgres_url(
                user="user",
                password="secret",
                database="testdb",
            ),
        )

    for record in caplog.records:
        assert "secret" not in record.message
        assert "user" not in record.message

def test_get_or_create_engine_with_extra_kwargs(patch_engine_manager: PatchEngineManager):
    """Engine creation forwards extra kwargs to EngineManager."""
    with patch_engine_manager(record_calls=True) as mock_engine:
        engine = get_or_create_engine(
            "postgresql://localhost/test",
            echo=True,
            pool_timeout=30,
        )

    _, kwargs = _first_engine_call(mock_engine)
    assert engine == mock_engine
    assert kwargs["echo"] is True
    assert kwargs["pool_timeout"] == 30

def test_get_or_create_engine_preserves_pool_pre_ping_default(
    patch_engine_manager: PatchEngineManager,
):
    """Pool pre-ping default is preserved when not specified."""
    with patch_engine_manager(record_calls=True) as mock_engine:
        get_or_create_engine("postgresql://localhost/test")

    _, kwargs = _first_engine_call(mock_engine)
    assert kwargs["pool_pre_ping"] is True

def test_get_or_create_engine_allows_custom_pool_pre_ping(
    patch_engine_manager: PatchEngineManager,
):
    """Pool pre-ping can be overridden."""
    with patch_engine_manager(record_calls=True) as mock_engine:
        get_or_create_engine("postgresql://localhost/test", pool_pre_ping=False)

    _, kwargs = _first_engine_call(mock_engine)
    assert kwargs["pool_pre_ping"] is False

def test_get_or_create_engine_none_pool_size_uses_default(
    patch_engine_manager: PatchEngineManager,
):
    """None pool_size uses default value."""
    with patch_engine_manager(record_calls=True) as mock_engine:
        get_or_create_engine("postgresql://localhost/test", pool_size=None)

    _, kwargs = _first_engine_call(mock_engine)
    assert kwargs["pool_size"] == 5

def test_get_or_create_engine_none_max_overflow_uses_default(
    patch_engine_manager: PatchEngineManager,
):
    """None max_overflow uses default value."""
    with patch_engine_manager(record_calls=True) as mock_engine:
        get_or_create_engine("postgresql://localhost/test", max_overflow=None)

    _, kwargs = _first_engine_call(mock_engine)
    assert kwargs["max_overflow"] == 10

def test_get_or_create_engine_sqlite_connection(patch_engine_manager: PatchEngineManager):
    """SQLite connection strings are handled correctly."""
    with patch_engine_manager(record_calls=True) as mock_engine:
        engine = get_or_create_engine("sqlite:///test.db")

    args, _ = _first_engine_call(mock_engine)
    assert engine == mock_engine
    assert args[0] == "sqlite:///test.db"

@pytest.mark.database
def test_ensure_default_partition_idempotent(cloned_test_database: str):
    """Default partition creation is idempotent."""
    engine = get_or_create_engine(cloned_test_database)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS public.ml_feature_values_default CASCADE"))

    ensure_default_partition(engine, "ml_feature_values")
    ensure_default_partition(engine, "ml_feature_values")

    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT to_regclass('public.ml_feature_values_default')")
        ).scalar()

    assert exists in {"ml_feature_values_default", "public.ml_feature_values_default"}

@pytest.mark.database
def test_ensure_partition_tables_ready_seeds_partitions(cloned_test_database: str):
    """Ensure partition helper creates default partitions and monthly shards."""
    engine = get_or_create_engine(cloned_test_database)
    with engine.begin() as conn:
        partitions = conn.execute(
            text(
                "SELECT inhrelid::regclass FROM pg_inherits"
                " WHERE inhparent = 'ml_feature_values'::regclass"
            ),
        )
        for partition in partitions:
            conn.execute(text(f"DROP TABLE IF EXISTS {partition[0]} CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS public.ml_feature_values_default CASCADE"))

    ensure_partition_tables_ready(engine, ("ml_feature_values",), months_ahead=1)

    with engine.connect() as conn:
        default_exists = conn.execute(
            text("SELECT to_regclass('public.ml_feature_values_default')")
        ).scalar()
        partition_count = conn.execute(
            text(
                "SELECT COUNT(*) FROM pg_inherits WHERE inhparent = 'ml_feature_values'::regclass"
            ),
        ).scalar()

    assert default_exists in {"ml_feature_values_default", "public.ml_feature_values_default"}
    assert partition_count is not None

def test_ensure_partition_tables_ready_invalid_months():
    """Reject negative month horizon."""
    with pytest.raises(ValueError):
        ensure_partition_tables_ready(Mock(spec=Engine), ("ml_feature_values",), months_ahead=-1)

def test_ensure_default_partition_invalid_identifier():
    """Invalid identifiers raise ValueError."""
    engine = Mock(spec=Engine)
    with pytest.raises(ValueError):
        ensure_default_partition(engine, "invalid-name")

def test_get_default_pool_config_immutability():
    """get_default_pool_config returns new dict each time."""
    config1 = get_default_pool_config()
    config2 = get_default_pool_config()

    # Should be equal but not the same object
    assert config1 == config2
    assert config1 is not config2

    # Modifying one should not affect the other
    config1["pool_size"] = 999
    assert config2["pool_size"] == 5

def test_ensure_partition_tables_ready_acquires_lock(monkeypatch):
    """Partition helper serializes creation with advisory locks."""

    engine = MagicMock(spec=Engine)
    conn = MagicMock()
    conn.dialect.name = "postgresql"

    class _Ctx:
        def __enter__(self):
            return conn

        def __exit__(self, *args):
            return False

    engine.begin.return_value = _Ctx()

    lock_calls: list[tuple[str, str]] = []

    def _fake_lock(connection, schema, table):
        lock_calls.append((schema, table))
        class _LockCtx:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, *exc):
                return False

        return _LockCtx()

    monkeypatch.setattr(db_utils, "_partition_lock", _fake_lock)

    ensure_partition_tables_ready(engine, ("ml_feature_values", "ml_strategy_signals"))

    assert lock_calls == [("public", "ml_feature_values"), ("public", "ml_strategy_signals")]

def test_ensure_monthly_partitions_acquires_lock(monkeypatch):
    """Single-table helper uses the same advisory lock pattern."""

    engine = MagicMock(spec=Engine)
    conn = MagicMock()
    conn.dialect.name = "postgresql"

    class _Ctx:
        def __enter__(self):
            return conn

        def __exit__(self, *args):
            return False

    engine.begin.return_value = _Ctx()

    lock_calls: list[tuple[str, str]] = []

    def _fake_lock(connection, schema, table):
        lock_calls.append((schema, table))

        class _LockCtx:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, *exc):
                return False

        return _LockCtx()

    monkeypatch.setattr(db_utils, "_partition_lock", _fake_lock)

    db_utils.ensure_monthly_partitions(engine, "ml_feature_values")

    assert lock_calls == [("public", "ml_feature_values")]
