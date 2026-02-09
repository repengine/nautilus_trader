#!/usr/bin/env python3
# ruff: noqa: RUF022
"""
Pytest configuration for ML module tests.

This module focuses on deterministic test orchestration (Hypothesis profiles,
marker registration, database gating, session hooks) while delegating actual
fixtures to the dedicated modules under ``ml.tests.fixtures``.
"""

from __future__ import annotations

import gc
import logging
import os
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

# Default to CPU-only execution for deterministic test runs unless explicitly overridden.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
from hypothesis import HealthCheck
from hypothesis import settings

from ml.core.db_engine import EngineManager as _EngineManager

try:
    import execnet.gateway_base as _execnet_gateway
except ImportError:  # pragma: no cover - instrumentation optional
    _execnet_gateway = None

_EXECNET_LOGGER = logging.getLogger("ml.tests.execnet")
if not _EXECNET_LOGGER.handlers:
    log_path = Path(os.getenv("ML_EXECNET_LOG_PATH", "/tmp/ml-execnet.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s]: %(message)s"),
    )
    _EXECNET_LOGGER.addHandler(handler)
_EXECNET_LOGGER.setLevel(logging.DEBUG)
_EXECNET_LOGGER.propagate = False

_LOGGED_WORKERS: set[str] = set()


@pytest.fixture(autouse=True)
def _log_worker_pid(worker_id: str) -> None:
    """
    Emit the OS pid for each xdist worker so we can map node IDs to processes.
    """

    if worker_id in _LOGGED_WORKERS:
        return

    _LOGGED_WORKERS.add(worker_id)
    log_path = Path("/tmp") / "xdist-worker-pids.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{worker_id}:{os.getpid()}\n")
    except Exception:
        pass

def _install_execnet_message_logging() -> None:
    """Wrap execnet Message.from_io so channel failures emit a log before propagating."""

    if _execnet_gateway is None:
        return
    if getattr(_execnet_gateway, "_ml_message_from_io_wrapped", False):
        return

    original = _execnet_gateway.Message.from_io

    def _logged_message_from_io(io: object) -> object:
        try:
            return original(io)
        except Exception as exc:
            try:
                _EXECNET_LOGGER.exception("Execnet channel read failed", extra={"io": io})
            except (ValueError, OSError):
                pass
            raise

    _execnet_gateway.Message.from_io = _logged_message_from_io
    setattr(_execnet_gateway, "_ml_message_from_io_wrapped", True)


_install_execnet_message_logging()


_DB_FIXTURES_MODULE: ModuleType | None = None
_PYTEST_CONFIG: pytest.Config | None = None


def _db_fixtures() -> ModuleType:
    """
    Return the database fixtures module without forcing an early import.
    """
    global _DB_FIXTURES_MODULE
    if _DB_FIXTURES_MODULE is not None:
        return _DB_FIXTURES_MODULE
    if _PYTEST_CONFIG is not None:
        plugin_manager = _PYTEST_CONFIG.pluginmanager
        plugin = plugin_manager.get_plugin("ml.tests.fixtures.database_fixtures")
        if plugin is None:
            plugin_manager.import_plugin("ml.tests.fixtures.database_fixtures")
            plugin = plugin_manager.get_plugin("ml.tests.fixtures.database_fixtures")
        if plugin is not None:
            _DB_FIXTURES_MODULE = plugin
            return plugin

    from ml.tests.fixtures import database_fixtures

    _DB_FIXTURES_MODULE = database_fixtures
    return database_fixtures

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
    global _PYTEST_CONFIG

    _PYTEST_CONFIG = config

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

    # Auto-mark tests as database when they request DB-backed fixtures.
    db_fixtures = {
        # Database engines/sessions
        "database_engine",
        "database_session_factory",
        "database_session",
        "postgres_connection",
        "clean_postgres_db",
        "clean_postgres_db_class",
        "clean_postgres_db_module",
        "test_database",
        "module_test_database",
        "template_database",
        "cloned_test_database",
        # Store bundles/helpers that are Postgres-backed
        "module_store_bundle",
        "fresh_store_bundle",
        "store_bundle",
        "module_store_bundle",
        "component_data_store_factory",
        "module_store_bundle",
        "db_engine",
    }
    for item in items:
        if any(fixture in item.fixturenames for fixture in db_fixtures):
            item.add_marker(pytest.mark.database)
            if "serial" not in item.keywords:
                item.add_marker(pytest.mark.serial)

    db_fixtures = _db_fixtures()
    if not db_fixtures.is_postgresql_running():
        skip_reason = (
            f"PostgreSQL not reachable at {db_fixtures.DATABASE_URL}; skipping @pytest.mark.database tests"
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

    if item.get_closest_marker("skip") is not None:
        return

    if "database" in item.keywords or "serial" in item.keywords:
        if os.getenv("ML_TEST_DISABLE_DB_LOCK") in {"1", "true", "True"}:
            return
        db_fixtures = _db_fixtures()
        fh = db_fixtures.acquire_db_lock("db")
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
        db_fixtures = _db_fixtures()
        db_fixtures.release_db_lock(fh)

    _EngineManager.dispose_all()
    if "database" in item.keywords or "serial" in item.keywords:
        gc.collect()
    _db_fixtures()._SCHEMA_INITIALIZED.clear()


def pytest_sessionstart(session: pytest.Session) -> None:
    """Set up test database at session start."""

    os.environ.setdefault("ML_TEST_ALLOW_NON_ONNX", "1")
    os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
    db_fixtures = _db_fixtures()

    try:
        from urllib.parse import urlparse

        url = os.getenv("DATABASE_URL", db_fixtures.DATABASE_URL)
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

    db_fixtures.start_postgresql()

    skip_db_preflight = os.getenv("ML_SKIP_DB_PREFLIGHT", "").lower() in {"1", "true", "yes"}

    if db_fixtures.is_postgresql_running() and not skip_db_preflight:
        engine = _EngineManager.get_engine(db_fixtures.DATABASE_URL)
        print("Database initialized, stores will create tables as needed...")
        engine.dispose()

        try:
            import ml.tests.fix_database_issues as _dbfix

            _dbfix.main()
        except Exception as exc:
            print(f"Warning: database fixes could not be applied: {exc}")

        try:
            from ml.stores.infrastructure import check_db_prereqs

            status = check_db_prereqs(db_fixtures.DATABASE_URL)
            ok = bool(status.get("ok", False))
            if not ok:
                try:
                    from sqlalchemy import text as _text

                    _eng = _EngineManager.get_engine(db_fixtures.DATABASE_URL)
                    with _eng.begin() as _conn:
                        _conn.execute(_text("SELECT auto_create_partitions()"))
                except Exception as exc:
                    import logging

                    logging.getLogger(__name__).debug(
                        "Partition function creation failed: %s",
                        exc,
                        exc_info=True,
                    )
                status = check_db_prereqs(db_fixtures.DATABASE_URL)
                print(f"Warning: DB preflight failed: {status}")
        except Exception as exc:
            print(f"Warning: DB preflight error: {exc}")
    elif skip_db_preflight:
        print("Skipping DB preflight (ML_SKIP_DB_PREFLIGHT=1)")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Finalize shared resources once the controlling pytest process exits."""

    if getattr(session.config, "workerinput", None) is not None:
        return

    import logging

    logging.disable(logging.CRITICAL)
    _EngineManager.dispose_all()
    gc.collect()

    _db_fixtures()._SCHEMA_INITIALIZED.clear()
