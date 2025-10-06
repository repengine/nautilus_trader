"""
Standardized error handling utilities for ML module.

Provides context managers and decorators to eliminate duplicated try/except patterns
across stores, registries, and actors. All error handling should use these utilities
for consistency, proper logging, and fallback behavior.

"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, TypeVar


__all__ = [
    "db_operation_handler",
    "registry_operation_handler",
    "with_db_error_handling",
    "with_fallback",
]

T = TypeVar("T")


@contextmanager
def db_operation_handler(
    operation_name: str,
    logger: logging.Logger,
    fallback: Any = None,
    re_raise: bool = True,
) -> Generator[None, None, None]:
    """
    Context manager for database operations with standardized error handling.

    Parameters
    ----------
    operation_name : str
        Human-readable operation description (e.g., "write features").
    logger : logging.Logger
        Logger instance for error messages.
    fallback : Any, optional
        Value to return on error if not re-raising.
    re_raise : bool, optional
        Whether to re-raise exception after logging. Defaults to True.

    Yields
    ------
    None

    Raises
    ------
    Exception
        Original exception if re_raise=True.

    Examples
    --------
    >>> with db_operation_handler("write features", logger):
    ...     with engine.begin() as conn:
    ...         conn.execute(insert_stmt)

    >>> with db_operation_handler("load data", logger, fallback=[], re_raise=False):
    ...     data = fetch_from_db()
    ...     return data
    """
    try:
        yield
    except Exception as e:
        logger.error("Failed to %s: %s", operation_name, e, exc_info=True)
        if not re_raise:
            # Context managers cannot return values; caller must handle fallback
            pass
        else:
            raise


@contextmanager
def registry_operation_handler(
    operation_name: str,
    registry_name: str,
    logger: logging.Logger,
    fallback: Any = None,
    re_raise: bool = False,
) -> Generator[None, None, None]:
    """
    Context manager for registry operations with fallback support.

    Parameters
    ----------
    operation_name : str
        Operation description (e.g., "load manifest").
    registry_name : str
        Registry identifier (e.g., "ModelRegistry").
    logger : logging.Logger
        Logger instance.
    fallback : Any, optional
        Fallback value on error.
    re_raise : bool, optional
        Whether to re-raise. Defaults to False (registry ops are non-critical).

    Yields
    ------
    None

    Examples
    --------
    >>> with registry_operation_handler("load manifest", "ModelRegistry", logger):
    ...     manifest = self._load_from_db()
    """
    try:
        yield
    except Exception as e:
        logger.warning(
            "%s failed to %s: %s (using fallback)",
            registry_name,
            operation_name,
            e,
        )
        if re_raise:
            raise
        # Context managers cannot return values; swallow exception


def with_db_error_handling(
    operation_name: str | None = None,
    fallback_value: Any = None,
    re_raise: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for database operation error handling.

    Parameters
    ----------
    operation_name : str | None, optional
        Operation description. If None, uses function name.
    fallback_value : Any, optional
        Value to return on error if not re-raising.
    re_raise : bool, optional
        Whether to re-raise exceptions. Defaults to True.

    Returns
    -------
    Callable
        Decorated function with error handling.

    Examples
    --------
    >>> @with_db_error_handling("write predictions")
    ... def write_predictions(self, predictions):
    ...     with self.engine.begin() as conn:
    ...         conn.execute(insert_stmt)

    >>> @with_db_error_handling(fallback_value=[], re_raise=False)
    ... def load_features(self, instrument_id):
    ...     return self._fetch_from_db(instrument_id)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get logger from first arg (usually self)
            logger = getattr(args[0], "logger", logging.getLogger(__name__))
            op_name = operation_name or func.__name__

            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "Failed to %s: %s",
                    op_name,
                    e,
                    exc_info=True,
                )
                if re_raise:
                    raise
                # Cast needed: fallback_value is Any, but we're returning T
                return fallback_value  # type: ignore[no-any-return]

        return wrapper
    return decorator


def with_fallback(
    fallback_value: Any,
    log_level: str = "warning",
    operation_name: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Generic decorator for any operation with fallback on error.

    Parameters
    ----------
    fallback_value : Any
        Value to return on any exception.
    log_level : str, optional
        Log level for errors ("debug", "info", "warning", "error"). Defaults to "warning".
    operation_name : str | None, optional
        Operation description. If None, uses function name.

    Returns
    -------
    Callable
        Decorated function that never raises, always returns fallback on error.

    Examples
    --------
    >>> @with_fallback(fallback_value={}, log_level="debug")
    ... def load_optional_config(self):
    ...     return self._load_from_file()

    >>> @with_fallback(fallback_value=None, operation_name="fetch metrics")
    ... def get_prometheus_metrics(self):
    ...     return self.registry.get_sample_value("ml_predictions_total")
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            logger = getattr(args[0], "logger", logging.getLogger(__name__))
            op_name = operation_name or func.__name__
            log_func = getattr(logger, log_level, logger.warning)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_func(
                    "Failed to %s: %s (using fallback: %r)",
                    op_name,
                    e,
                    fallback_value,
                )
                # Cast needed: fallback_value is Any, but we're returning T
                return fallback_value  # type: ignore[no-any-return]

        return wrapper
    return decorator
