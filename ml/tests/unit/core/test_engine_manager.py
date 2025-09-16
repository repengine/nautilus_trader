from __future__ import annotations

from sqlalchemy.pool import NullPool

from ml.core.db_engine import EngineManager


def test_sqlite_engine_uses_nullpool() -> None:
    engine = EngineManager.get_engine("sqlite:///:memory:")
    try:
        assert isinstance(engine.pool, NullPool)
    finally:
        EngineManager.dispose_all()


def test_engine_cache_reuse_and_dispose() -> None:
    dsn = "sqlite:///:memory:"
    e1 = EngineManager.get_engine(dsn)
    e2 = EngineManager.get_engine(dsn)
    assert e1 is e2  # cached instance
    assert EngineManager.has_engine(dsn)

    EngineManager.dispose_engine(dsn)
    assert not EngineManager.has_engine(dsn)
