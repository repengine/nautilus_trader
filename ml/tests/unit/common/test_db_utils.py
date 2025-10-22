"""Tests for database utility helpers."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ml.common.db_utils import (
    STORE_PARTITIONED_TABLES,
    ensure_default_partition,
    ensure_partition_tables_ready,
    get_or_create_engine,
    get_default_pool_config,
)

pytestmark = pytest.mark.serial


def test_get_default_pool_config():
    """Default pool config returns expected values."""
    config = get_default_pool_config()
    assert config["pool_size"] == 5
    assert config["max_overflow"] == 10
    assert config["pool_pre_ping"] is True
    assert config["pool_recycle"] == 3600


def test_get_or_create_engine_with_defaults(monkeypatch):
    """Engine created with default pool settings."""
    mock_engine = Mock()
    call_data = {}

    def mock_get_engine(connection_string, **kwargs):
        call_data["connection_string"] = connection_string
        call_data["kwargs"] = kwargs
        return mock_engine

    monkeypatch.setattr("ml.core.db_engine.EngineManager.get_engine", mock_get_engine)

    engine = get_or_create_engine("postgresql://localhost/test")

    assert engine == mock_engine
    assert call_data["connection_string"] == "postgresql://localhost/test"
    assert call_data["kwargs"]["pool_size"] == 5
    assert call_data["kwargs"]["max_overflow"] == 10
    assert call_data["kwargs"]["pool_pre_ping"] is True
    assert call_data["kwargs"]["pool_recycle"] == 3600


def test_get_or_create_engine_with_custom_settings(monkeypatch):
    """Engine created with custom pool settings."""
    mock_engine = Mock()
    call_data = {}

    def mock_get_engine(connection_string, **kwargs):
        call_data["connection_string"] = connection_string
        call_data["kwargs"] = kwargs
        return mock_engine

    monkeypatch.setattr("ml.core.db_engine.EngineManager.get_engine", mock_get_engine)

    engine = get_or_create_engine(
        "postgresql://localhost/test",
        pool_size=10,
        max_overflow=20,
        pool_recycle=7200,
    )

    assert engine == mock_engine
    assert call_data["kwargs"]["pool_size"] == 10
    assert call_data["kwargs"]["max_overflow"] == 20
    assert call_data["kwargs"]["pool_pre_ping"] is True
    assert call_data["kwargs"]["pool_recycle"] == 7200


def test_get_or_create_engine_empty_connection_string():
    """Raises ValueError for empty connection string."""
    with pytest.raises(ValueError, match="connection_string cannot be empty"):
        get_or_create_engine("")


def test_get_or_create_engine_handles_engine_manager_error(monkeypatch):
    """RuntimeError raised when EngineManager fails."""
    def mock_failure(*args, **kwargs):
        raise Exception("Connection failed")

    monkeypatch.setattr("ml.core.db_engine.EngineManager.get_engine", mock_failure)

    with pytest.raises(RuntimeError, match="Database engine creation failed"):
        get_or_create_engine("postgresql://localhost/test")


def test_connection_string_sanitized_in_logs(monkeypatch, caplog):
    """Connection string credentials not leaked in logs."""
    mock_engine = Mock()
    monkeypatch.setattr(
        "ml.core.db_engine.EngineManager.get_engine",
        lambda *args, **kwargs: mock_engine
    )

    get_or_create_engine("postgresql://user:secret@localhost:5432/testdb")

    # Check logs don't contain password
    for record in caplog.records:
        assert "secret" not in record.message
        assert "user" not in record.message


def test_get_or_create_engine_with_extra_kwargs(monkeypatch):
    """Engine creation forwards extra kwargs to EngineManager."""
    mock_engine = Mock()
    call_data = {}

    def mock_get_engine(connection_string, **kwargs):
        call_data["kwargs"] = kwargs
        return mock_engine

    monkeypatch.setattr("ml.core.db_engine.EngineManager.get_engine", mock_get_engine)

    engine = get_or_create_engine(
        "postgresql://localhost/test",
        echo=True,
        pool_timeout=30,
    )

    assert engine == mock_engine
    assert call_data["kwargs"]["echo"] is True
    assert call_data["kwargs"]["pool_timeout"] == 30


def test_get_or_create_engine_preserves_pool_pre_ping_default(monkeypatch):
    """Pool pre-ping default is preserved when not specified."""
    call_data = {}

    def mock_get_engine(connection_string, **kwargs):
        call_data["kwargs"] = kwargs
        return Mock()

    monkeypatch.setattr("ml.core.db_engine.EngineManager.get_engine", mock_get_engine)

    get_or_create_engine("postgresql://localhost/test")

    assert call_data["kwargs"]["pool_pre_ping"] is True


def test_get_or_create_engine_allows_custom_pool_pre_ping(monkeypatch):
    """Pool pre-ping can be overridden."""
    call_data = {}

    def mock_get_engine(connection_string, **kwargs):
        call_data["kwargs"] = kwargs
        return Mock()

    monkeypatch.setattr("ml.core.db_engine.EngineManager.get_engine", mock_get_engine)

    get_or_create_engine("postgresql://localhost/test", pool_pre_ping=False)

    assert call_data["kwargs"]["pool_pre_ping"] is False


def test_get_or_create_engine_none_pool_size_uses_default(monkeypatch):
    """None pool_size uses default value."""
    call_data = {}

    def mock_get_engine(connection_string, **kwargs):
        call_data["kwargs"] = kwargs
        return Mock()

    monkeypatch.setattr("ml.core.db_engine.EngineManager.get_engine", mock_get_engine)

    get_or_create_engine("postgresql://localhost/test", pool_size=None)

    assert call_data["kwargs"]["pool_size"] == 5


def test_get_or_create_engine_none_max_overflow_uses_default(monkeypatch):
    """None max_overflow uses default value."""
    call_data = {}

    def mock_get_engine(connection_string, **kwargs):
        call_data["kwargs"] = kwargs
        return Mock()

    monkeypatch.setattr("ml.core.db_engine.EngineManager.get_engine", mock_get_engine)

    get_or_create_engine("postgresql://localhost/test", max_overflow=None)

    assert call_data["kwargs"]["max_overflow"] == 10


def test_get_or_create_engine_sqlite_connection(monkeypatch):
    """SQLite connection strings are handled correctly."""
    mock_engine = Mock()
    call_data = {}

    def mock_get_engine(connection_string, **kwargs):
        call_data["connection_string"] = connection_string
        return mock_engine

    monkeypatch.setattr("ml.core.db_engine.EngineManager.get_engine", mock_get_engine)

    engine = get_or_create_engine("sqlite:///test.db")

    assert engine == mock_engine
    assert call_data["connection_string"] == "sqlite:///test.db"


@pytest.mark.database
def test_ensure_default_partition_idempotent(test_database):
    """Default partition creation is idempotent."""
    engine = test_database.engine
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
def test_ensure_partition_tables_ready_seeds_partitions(test_database):
    """Ensure partition helper creates default partitions and monthly shards."""
    engine = test_database.engine
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
