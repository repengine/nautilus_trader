"""
Singleton database engine manager for ML store implementations.

This module provides a thread-safe singleton pattern for managing SQLAlchemy database
engines with proper connection pooling to prevent pool exhaustion in the ML test suite
and production deployments.

The EngineManager ensures that each unique connection string gets exactly one engine
instance, preventing the creation of multiple connection pools to the same database
which can lead to "too many clients already" errors in PostgreSQL.

"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.pool import NullPool
from sqlalchemy.pool import QueuePool


if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


logger = logging.getLogger(__name__)


class EngineManager:
    """
    Thread-safe singleton manager for SQLAlchemy database engines.

    This class ensures that each unique connection string gets exactly one engine
    instance, preventing connection pool exhaustion. The manager is particularly
    important for the ML test suite where hypothesis tests can create many store
    instances rapidly.

    Attributes
    ----------
    _instances : dict[str, Engine]
        Cache of engine instances keyed by connection string
    _lock : threading.Lock
        Lock for thread-safe access to the instances cache

    Examples
    --------
    >>> # Get an engine (creates if not exists)
    >>> engine = EngineManager.get_engine("postgresql://user:pass@localhost/db")

    >>> # Get the same engine instance (cached)
    >>> engine2 = EngineManager.get_engine("postgresql://user:pass@localhost/db")
    >>> assert engine is engine2  # Same instance

    >>> # Dispose all engines during cleanup
    >>> EngineManager.dispose_all()

    """

    _instances: dict[str, Engine] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_engine(
        cls,
        connection_string: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_pre_ping: bool = True,
        pool_recycle: int = 3600,
        echo: bool = False,
        **kwargs: object,
    ) -> Engine:
        """
        Get or create a SQLAlchemy engine for the given connection string.

        This method implements a thread-safe singleton pattern, ensuring that each
        unique connection string gets exactly one engine instance. For test
        environments, conservative pool settings are used to prevent exhaustion.

        Parameters
        ----------
        connection_string : str
            Database connection string (e.g., "postgresql://user:pass@host/db")
        pool_size : int, default=5
            Number of persistent connections to maintain in the pool.
            For tests, use 2. For production, use 5-10.
        max_overflow : int, default=10
            Maximum overflow connections above pool_size.
            For tests, use 3. For production, use 10-20.
        pool_pre_ping : bool, default=True
            Test connections before using them from the pool.
            Recommended for reliability, especially in tests.
        pool_recycle : int, default=3600
            Number of seconds after which to recycle connections.
            Prevents timeout issues with long-lived connections.
        echo : bool, default=False
            Whether to log SQL statements. Useful for debugging.
        **kwargs : Any
            Additional arguments passed to create_engine.

        Returns
        -------
        Engine
            SQLAlchemy engine instance for the connection string

        Raises
        ------
        ValueError
            If connection_string is empty or None
        RuntimeError
            If engine creation fails

        Examples
        --------
        >>> # Production configuration
        >>> engine = EngineManager.get_engine(
        ...     "postgresql://prod_user:pass@prod_host/nautilus",
        ...     pool_size=10,
        ...     max_overflow=20
        ... )

        >>> # Test configuration (conservative pooling)
        >>> test_engine = EngineManager.get_engine(
        ...     "postgresql://test_user:pass@localhost/test_db",
        ...     pool_size=2,
        ...     max_overflow=3
        ... )

        Notes
        -----
        The engine cache is keyed by the connection string alone, not by the pool
        parameters. This means that the first call with a given connection string
        determines the pool settings for all subsequent calls with the same string.

        In test environments, it's recommended to use smaller pool sizes (2-3) and
        lower max_overflow (3-5) to prevent connection exhaustion during parallel
        test execution or hypothesis property-based testing.

        """
        if not connection_string:
            msg = "Connection string cannot be empty"
            raise ValueError(msg)

        # Fast path: check if engine exists without locking
        if connection_string in cls._instances:
            return cls._instances[connection_string]

        # If the incoming URL has a masked password (e.g., "***"), attempt to
        # find an already-created engine that matches the same target (driver,
        # user, host, port, database) regardless of password. This situation
        # occurs when callers pass `str(engine.url)` which hides the password.
        try:
            parsed = make_url(connection_string)
            masked = isinstance(parsed.password, str) and parsed.password.strip() == "***"
        except Exception:
            parsed = None
            masked = False

        if masked and parsed is not None:
            for existing in cls._instances.values():
                try:
                    eurl = existing.url
                    if (
                        str(eurl.drivername) == str(parsed.drivername)
                        and str(eurl.username or "") == str(parsed.username or "")
                        and str(eurl.host or "") == str(parsed.host or "")
                        and int(eurl.port or 0) == int(parsed.port or 0)
                        and str(eurl.database or "") == str(parsed.database or "")
                    ):
                        return existing
                except Exception:
                    continue

        # Slow path: acquire lock and create engine if needed
        with cls._lock:
            # Double-check after acquiring lock
            if connection_string in cls._instances:
                return cls._instances[connection_string]

            try:
                logger.debug(
                    f"Creating new engine for connection: {connection_string[:30]}... "
                    f"(pool_size={pool_size}, max_overflow={max_overflow})",
                )

                # Determine if this is a test environment
                is_test = any(
                    marker in connection_string.lower()
                    for marker in ["test", "temp", "tmp", ":memory:"]
                )

                if is_test:
                    # Use conservative settings for tests
                    actual_pool_size = min(pool_size, 2)
                    actual_max_overflow = min(max_overflow, 3)
                    logger.debug(
                        f"Test environment detected, using conservative pool settings: "
                        f"pool_size={actual_pool_size}, max_overflow={actual_max_overflow}",
                    )
                else:
                    actual_pool_size = pool_size
                    actual_max_overflow = max_overflow

                # Create engine with appropriate pooling configuration
                if ":memory:" in connection_string or "sqlite" in connection_string:
                    # SQLite doesn't benefit from connection pooling
                    engine = create_engine(
                        connection_string,
                        poolclass=NullPool,
                        echo=echo,
                        **kwargs,
                    )
                else:
                    # Prepare connect_args and enforce safe timeouts in tests
                    connect_args_obj = kwargs.pop("connect_args", None)
                    connect_args: dict[str, Any] = (
                        {} if connect_args_obj is None else cast(dict[str, Any], connect_args_obj)
                    )
                    is_pg = connection_string.lower().startswith("postgresql")
                    is_test_runtime = is_test or bool(os.getenv("PYTEST_CURRENT_TEST"))
                    if is_pg and is_test_runtime:
                        # Bound statement execution to 60s to prevent hangs in CI/local tests
                        # Only set if caller hasn't provided custom options
                        if "options" not in connect_args:
                            connect_args["options"] = "-c statement_timeout=60000"

                    # PostgreSQL/MySQL benefit from connection pooling
                    engine = create_engine(
                        connection_string,
                        poolclass=QueuePool,
                        pool_size=actual_pool_size,
                        max_overflow=actual_max_overflow,
                        pool_pre_ping=pool_pre_ping,
                        pool_recycle=pool_recycle,
                        echo=echo,
                        connect_args=connect_args,
                        **kwargs,
                    )

                cls._instances[connection_string] = engine
                logger.info(
                    f"Created engine for {connection_string[:30]}... "
                    f"(total engines: {len(cls._instances)})",
                )

                # Best-effort: ensure default partitions exist for core ML tables so
                # direct SQL inserts used in integration tests work even when the
                # monthly partitions have not been created yet by migrations.
                try:
                    if str(engine.url).startswith("postgresql"):
                        from sqlalchemy import text as _text  # local import to avoid global deps

                        with engine.begin() as _conn:
                            for parent in (
                                "ml_feature_values",
                                "ml_model_predictions",
                                "ml_strategy_signals",
                            ):
                                try:
                                    _conn.execute(
                                        _text(
                                            f"CREATE TABLE IF NOT EXISTS {parent}_default "
                                            f"PARTITION OF {parent} DEFAULT",
                                        ),
                                    )
                                except Exception as pexc:
                                    # Parent may not exist or may be non-partitioned in some envs
                                    logger.debug(
                                        "Default partition ensure skipped for %s: %s",
                                        parent,
                                        pexc,
                                        exc_info=True,
                                    )
                except Exception as exc:
                    logger.debug("Partition ensure pass skipped: %s", exc, exc_info=True)
                return engine

            except Exception as e:
                msg = f"Failed to create engine for {connection_string[:30]}...: {e}"
                logger.error(msg)
                raise RuntimeError(msg) from e

    @classmethod
    def dispose_engine(cls, connection_string: str) -> None:
        """
        Dispose a specific engine and remove it from the cache.

        This method closes all connections in the engine's connection pool and
        removes the engine from the internal cache. Use this for targeted cleanup
        of specific database connections.

        Parameters
        ----------
        connection_string : str
            The connection string of the engine to dispose

        Examples
        --------
        >>> engine = EngineManager.get_engine("postgresql://user:pass@localhost/db")
        >>> # ... use engine ...
        >>> EngineManager.dispose_engine("postgresql://user:pass@localhost/db")

        """
        with cls._lock:
            if connection_string in cls._instances:
                try:
                    engine = cls._instances[connection_string]
                    engine.dispose()
                    del cls._instances[connection_string]
                    logger.debug(f"Disposed engine for {connection_string[:30]}...")
                except Exception as e:
                    logger.warning(f"Error disposing engine for {connection_string[:30]}...: {e}")

    @classmethod
    def dispose_all(cls) -> None:
        """
        Dispose all cached engines and clear the instance cache.

        This method should be called during test teardown or application shutdown
        to ensure all database connections are properly closed. It's particularly
        important in test suites to prevent connection leaks between test runs.

        Examples
        --------
        >>> # In pytest fixture teardown
        >>> def teardown_function():
        ...     EngineManager.dispose_all()

        >>> # In application shutdown
        >>> try:
        ...     run_application()
        ... finally:
        ...     EngineManager.dispose_all()

        Notes
        -----
        This method is thread-safe and will dispose all engines even if called
        concurrently from multiple threads. Errors during disposal are logged
        but do not prevent other engines from being disposed.

        """
        with cls._lock:
            if not cls._instances:
                return

            logger.info(f"Disposing {len(cls._instances)} cached engine(s)")
            errors = []

            for connection_string, engine in list(cls._instances.items()):
                try:
                    engine.dispose()
                    logger.debug(f"Disposed engine for {connection_string[:30]}...")
                except Exception as e:
                    error_msg = f"Error disposing engine for {connection_string[:30]}...: {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)

            cls._instances.clear()

            if errors:
                logger.warning(f"Encountered {len(errors)} error(s) during disposal")

    @classmethod
    def get_engine_count(cls) -> int:
        """
        Get the number of cached engine instances.

        Returns
        -------
        int
            Number of engines currently in the cache

        Examples
        --------
        >>> count = EngineManager.get_engine_count()
        >>> print(f"Currently managing {count} database engine(s)")

        """
        with cls._lock:
            return len(cls._instances)

    @classmethod
    def has_engine(cls, connection_string: str) -> bool:
        """
        Check if an engine exists for the given connection string.

        Parameters
        ----------
        connection_string : str
            The connection string to check

        Returns
        -------
        bool
            True if an engine exists for the connection string, False otherwise

        Examples
        --------
        >>> if not EngineManager.has_engine("postgresql://user:pass@localhost/db"):
        ...     engine = EngineManager.get_engine("postgresql://user:pass@localhost/db")

        """
        with cls._lock:
            return connection_string in cls._instances

    @classmethod
    def get_pool_status(cls, connection_string: str) -> dict[str, Any] | None:
        """
        Get connection pool status for a specific engine.

        Parameters
        ----------
        connection_string : str
            The connection string of the engine to check

        Returns
        -------
        dict[str, Any] | None
            Dictionary with pool status information or None if engine not found.
            Contains 'size', 'checked_in', 'checked_out', 'overflow', 'total' keys.

        Examples
        --------
        >>> status = EngineManager.get_pool_status("postgresql://user:pass@localhost/db")
        >>> if status:
        ...     print(f"Connections in use: {status['checked_out']}/{status['total']}")

        """
        with cls._lock:
            if connection_string not in cls._instances:
                return None

            engine = cls._instances[connection_string]
            pool = engine.pool

            # Handle different pool types
            if hasattr(pool, "size"):
                # QueuePool has these attributes
                try:
                    size_fn = getattr(pool, "size", lambda: 0)
                    overflow_fn = getattr(pool, "overflow", lambda: 0)
                    return {
                        "size": size_fn(),
                        "checked_in": getattr(pool, "checkedin", lambda: 0)(),
                        "checked_out": getattr(pool, "checkedout", lambda: 0)(),
                        "overflow": overflow_fn(),
                        "total": size_fn() + overflow_fn(),
                    }
                except AttributeError:
                    # Fallback if attributes don't exist
                    return {
                        "size": 0,
                        "checked_in": 0,
                        "checked_out": 0,
                        "overflow": 0,
                        "total": 0,
                        "pool_type": type(pool).__name__,
                    }
            else:
                # NullPool or other pool types without these attributes
                return {
                    "size": 0,
                    "checked_in": 0,
                    "checked_out": 0,
                    "overflow": 0,
                    "total": 0,
                    "pool_type": type(pool).__name__,
                }
