#!/usr/bin/env python3
# ruff: noqa: RUF022
"""
Pytest configuration for ML module tests.

This module focuses on deterministic test orchestration (Hypothesis profiles,
marker registration, database gating, session hooks) while delegating actual
fixtures to the dedicated modules under ``ml.tests.fixtures``.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck
from hypothesis import settings

from ml.core.db_engine import EngineManager as _EngineManager
from ml.tests.fixtures.database_fixtures import (
    DATABASE_URL,
    _SCHEMA_INITIALIZED,
    acquire_db_lock,
    is_postgresql_running,
    release_db_lock,
    start_postgresql,
)

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
except ImportError:
    pass  # dotenv not installed, use system environment

# ============================================================================
# Prototype gating
# ============================================================================

_PROTOTYPE_PATH_SUFFIXES = [
    "ml/tests/property/test_domain_bookkeeping_phase1.py",
    "ml/tests/contracts/test_domain_bookkeeping_schemas.py",
    "ml/tests/metamorphic/test_domain_bookkeeping_event_flow.py",
    "ml/tests/property/test_domain_bookkeeping_phase2.py",
    "ml/tests/contracts/test_observability_pipeline_schemas.py",
    "ml/tests/metamorphic/test_observability_correlation.py",
    "ml/tests/combinatorial/test_domain_bookkeeping_configs.py",
    "ml/tests/property/test_domain_bookkeeping_stateful.py",
]


def _mark_prototypes(items: list[pytest.Item]) -> None:
    """Mark TDD prototype tests so they do not block installs by default."""

    for item in items:
        nodeid = item.nodeid.replace("::", "/")
        for suffix in _PROTOTYPE_PATH_SUFFIXES:
            if nodeid.endswith(suffix):
                item.add_marker(pytest.mark.prototype)
                break


# ============================================================================
# Hypothesis configuration
# ============================================================================

settings.register_profile(
    "ci",
    max_examples=50,
    deadline=5000,
    print_blob=True,
    report_multiple_bugs=True,
    derandomize=True,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)

settings.register_profile(
    "dev",
    max_examples=200,
    deadline=None,
    print_blob=True,
    report_multiple_bugs=True,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)

settings.register_profile(
    "debug",
    max_examples=10,
    deadline=None,
    print_blob=True,
    verbosity=2,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)

if os.getenv("CI"):
    settings.load_profile("ci")
else:
    try:
        settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci"))
    except Exception:
        settings.load_profile("ci")


# ============================================================================
# Pytest hook implementations
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers and sensible defaults."""

    config.addinivalue_line("markers", "database: requires PostgreSQL; may run serially")
    config.addinivalue_line("markers", "serial: run test in isolation (no xdist)")
    config.addinivalue_line("markers", "integration: integration test category")
    config.addinivalue_line(
        "markers",
        "pollution_detection: tests that detect test isolation pollution",
    )

    os.environ.setdefault("ML_DISABLE_METRICS_SERVER", "1")
    os.environ.setdefault("TEST_DB_SKIP_TRUNCATE", "1")

    try:
        import multiprocessing

        import xdist  # noqa: F401 - imported to detect availability

        cpu_count = multiprocessing.cpu_count()
        optimal_workers = max(1, cpu_count // 2)
        if not config.getoption("--numprocesses", default=None):
            config.option.numprocesses = optimal_workers

        current_dist = getattr(config.option, "dist", None)
        if getattr(config.option, "numprocesses", 0) and current_dist in (None, "load", "loadscope"):
            config.option.dist = "loadgroup"
    except ImportError:
        pass


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply prototype marks and gate DB tests on collection."""

    _mark_prototypes(items)

    if not is_postgresql_running():
        skip_reason = (
            f"PostgreSQL not reachable at {DATABASE_URL}; skipping @pytest.mark.database tests"
        )
        skip_db = pytest.mark.skip(reason=skip_reason)
        for item in items:
            if "database" in item.keywords:
                item.add_marker(skip_db)

    for item in items:
        node = item.nodeid.replace("::", "/")
        if "/ml/tests/integration/" in node:
            if "serial" not in item.keywords:
                item.add_marker(pytest.mark.serial)
            if "integration" not in item.keywords:
                item.add_marker(pytest.mark.integration)

    try:
        import xdist  # noqa: F401

        for item in items:
            if "database" in item.keywords or "serial" in item.keywords:
                try:
                    item.add_marker(pytest.mark.xdist_group("db"))  # type: ignore[attr-defined]
                except Exception:
                    pass
    except Exception:
        pass


_DB_LOCK_FH: dict[str, Any] = {}


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Serialize database/serial tests across xdist workers using a file lock."""

    if "database" in item.keywords or "serial" in item.keywords:
        if os.getenv("ML_TEST_DISABLE_DB_LOCK") in {"1", "true", "True"}:
            return
        fh = acquire_db_lock("db")
        if fh is None:
            item.add_marker(
                pytest.mark.skip(reason="DB lock contention timeout; skipping to avoid hang"),
            )
            return
        _DB_LOCK_FH[item.nodeid] = fh


def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
    """Release DB locks and clear connection caches after each test."""

    if item.nodeid in _DB_LOCK_FH:
        fh = _DB_LOCK_FH.pop(item.nodeid)
        release_db_lock(fh)

    _EngineManager.dispose_all()
    _SCHEMA_INITIALIZED.clear()


def pytest_sessionstart(session: pytest.Session) -> None:
    """Set up test database at session start."""

    os.environ.setdefault("ML_TEST_ALLOW_NON_ONNX", "1")
    os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

    try:
        from urllib.parse import urlparse

        url = os.getenv("DATABASE_URL", DATABASE_URL)
        parsed = urlparse(url)
        if not os.getenv("PGPASSWORD"):
            if parsed.password:
                os.environ["PGPASSWORD"] = parsed.password
            else:
                os.environ.setdefault("PGPASSWORD", "postgres")
        os.environ.setdefault("ML_YFINANCE_FIXTURE", "static")
    except Exception:
        os.environ.setdefault("PGPASSWORD", "postgres")
        os.environ.setdefault("ML_YFINANCE_FIXTURE", "static")

    if os.environ.get("SKIP_DB_INIT", "").lower() in ("1", "true", "yes"):
        print("Skipping database initialization (SKIP_DB_INIT is set)")
        return

    try:
        from shutil import rmtree

        cache_dir = Path.cwd() / ".pytest_cache"
        if cache_dir.exists():
            rmtree(cache_dir)
    except Exception:
        import logging

        logging.getLogger(__name__).debug(
            "Failed to clear pytest cache; continuing",
            exc_info=True,
        )

    start_postgresql()

    if is_postgresql_running():
        engine = _EngineManager.get_engine(DATABASE_URL)
        print("Database initialized, stores will create tables as needed...")
        engine.dispose()

        try:
            import ml.tests.fix_database_issues as _dbfix

            _dbfix.main()
        except Exception as exc:
            print(f"Warning: database fixes could not be applied: {exc}")

        try:
            from ml.stores.infrastructure import check_db_prereqs

            status = check_db_prereqs(DATABASE_URL)
            ok = bool(status.get("ok", False))
            if not ok:
                try:
                    from sqlalchemy import text as _text

                    _eng = _EngineManager.get_engine(DATABASE_URL)
                    with _eng.begin() as _conn:
                        _conn.execute(_text("SELECT auto_create_partitions()"))
                except Exception as exc:
                    import logging

                    logging.getLogger(__name__).debug(
                        "Partition function creation failed: %s",
                        exc,
                        exc_info=True,
                    )
                status = check_db_prereqs(DATABASE_URL)
                print(f"Warning: DB preflight failed: {status}")
        except Exception as exc:
            print(f"Warning: DB preflight error: {exc}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Finalize shared resources once the controlling pytest process exits."""

    if getattr(session.config, "workerinput", None) is not None:
        return

    import logging

    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        _EngineManager.dispose_all()
    finally:
        logging.disable(previous_disable)

    _SCHEMA_INITIALIZED.clear()

    logger = logging.getLogger(__name__)
    try:
        logger.info("Test session completed with exit status: %s", exitstatus)
    except ValueError:
        pass

