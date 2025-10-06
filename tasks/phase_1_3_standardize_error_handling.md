# Task: [Phase 1.3] Standardize Error Handling

## Context
**Phase:** 1 - DRY Violations - Critical Path
**Task ID:** 1.3
**Depends On:** 1.1, 1.2
**Estimated Effort:** 10 hours
**Impact Score:** 680 (213 files affected)

## Scope
Create standardized error handling utilities in `ml/common/error_handlers.py` to eliminate duplicated `try/except Exception as e:` blocks across 213 files. Provide context managers and decorators for database operations, registry interactions, and store operations.

## Required Reading
- [x] REFACTORING_PLAN.md (Phase 1.3)
- [x] DRY Violations Report (Error Handling Blocks section)
- [x] ml/docs/development/CODING_STANDARDS.md
- [x] ml/docs/architecture/universal_patterns_guide.md

## Definition of Done
- [ ] New file created: `ml/common/error_handlers.py` with utilities
- [ ] Context manager `db_operation_handler()` implemented
- [ ] Decorator `@with_db_error_handling` implemented
- [ ] Context manager `registry_operation_handler()` implemented
- [ ] Decorator `@with_fallback` implemented (generic)
- [ ] Top 50 files with most duplicated patterns updated
- [ ] Comprehensive test suite created
- [ ] All tests pass
- [ ] Ruff check passes
- [ ] MyPy strict passes
- [ ] Pattern validation passes
- [ ] Error messages remain informative

## Files to Modify

### Create New (2 files)
- [ ] `ml/common/error_handlers.py` - Error handling utilities
- [ ] `ml/tests/unit/common/test_error_handlers.py` - Comprehensive tests

### Update (50 files - highest duplication)
Top priority files (grep to find):
- `ml/stores/feature_store.py`
- `ml/stores/model_store.py`
- `ml/stores/strategy_store.py`
- `ml/stores/data_store.py`
- `ml/registry/model_registry.py`
- `ml/registry/feature_registry.py`
- `ml/actors/base.py`
- ... (47 more - identified via grep)

## Implementation Steps

### Step 1: Create ml/common/error_handlers.py

```python
"""
Standardized error handling utilities for ML module.

Provides context managers and decorators to eliminate duplicated try/except patterns
across stores, registries, and actors. All error handling should use these utilities
for consistency, proper logging, and fallback behavior.

"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

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
        if re_raise:
            raise
        return fallback


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
        return fallback


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
                return fallback_value  # type: ignore[return-value]

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
                return fallback_value  # type: ignore[return-value]

        return wrapper
    return decorator
```

### Step 2: Update ml/common/__init__.py

Add imports:
```python
from ml.common.error_handlers import (
    db_operation_handler,
    registry_operation_handler,
    with_db_error_handling,
    with_fallback,
)
```

### Step 3: Find files with most duplication

```bash
# Count try/except patterns per file
grep -r "except Exception as e:" ml/ --include="*.py" -c | sort -t: -k2 -nr | head -50

# This gives you the top 50 files to update
```

### Step 4: Refactor patterns in stores

**Example: ml/stores/feature_store.py**

**OLD:**
```python
def write_features(self, features):
    try:
        with self.engine.begin() as conn:
            conn.execute(insert_stmt)
    except Exception as e:
        self.logger.error("Failed to write features: %s", e)
        raise
```

**NEW (Context Manager):**
```python
from ml.common.error_handlers import db_operation_handler

def write_features(self, features):
    with db_operation_handler("write features", self.logger):
        with self.engine.begin() as conn:
            conn.execute(insert_stmt)
```

**NEW (Decorator):**
```python
from ml.common.error_handlers import with_db_error_handling

@with_db_error_handling("write features")
def write_features(self, features):
    with self.engine.begin() as conn:
        conn.execute(insert_stmt)
```

### Step 5: Refactor patterns in registries

**Example: ml/registry/model_registry.py**

**OLD:**
```python
def load_manifest(self, model_id):
    try:
        return self._load_from_db(model_id)
    except Exception as e:
        logger.warning("Failed to load manifest: %s (using fallback)", e)
        return None
```

**NEW:**
```python
from ml.common.error_handlers import with_fallback

@with_fallback(fallback_value=None, log_level="warning")
def load_manifest(self, model_id):
    return self._load_from_db(model_id)
```

### Step 6: Create comprehensive test suite

Create `ml/tests/unit/common/test_error_handlers.py`:

```python
"""Tests for error handling utilities."""

import logging
import pytest
from unittest.mock import Mock

from ml.common.error_handlers import (
    db_operation_handler,
    registry_operation_handler,
    with_db_error_handling,
    with_fallback,
)


def test_db_operation_handler_success(caplog):
    """Context manager passes through on success."""
    logger = logging.getLogger(__name__)

    with db_operation_handler("test operation", logger):
        result = 42

    assert result == 42
    assert len(caplog.records) == 0


def test_db_operation_handler_error_re_raises(caplog):
    """Context manager logs and re-raises by default."""
    logger = logging.getLogger(__name__)

    with pytest.raises(ValueError, match="test error"):
        with db_operation_handler("test operation", logger):
            raise ValueError("test error")

    assert any("Failed to test operation" in r.message for r in caplog.records)


def test_db_operation_handler_fallback(caplog):
    """Context manager returns fallback when re_raise=False."""
    logger = logging.getLogger(__name__)

    with db_operation_handler("test", logger, fallback=[], re_raise=False):
        raise ValueError("error")

    # Should have logged error
    assert any("Failed to test" in r.message for r in caplog.records)


def test_with_db_error_handling_decorator_success():
    """Decorator passes through on success."""
    class TestClass:
        logger = logging.getLogger(__name__)

        @with_db_error_handling("test op")
        def method(self):
            return 42

    obj = TestClass()
    assert obj.method() == 42


def test_with_db_error_handling_decorator_error():
    """Decorator logs and re-raises by default."""
    class TestClass:
        logger = logging.getLogger(__name__)

        @with_db_error_handling("test op")
        def method(self):
            raise ValueError("error")

    obj = TestClass()
    with pytest.raises(ValueError):
        obj.method()


def test_with_fallback_decorator():
    """Fallback decorator returns fallback on error."""
    class TestClass:
        logger = logging.getLogger(__name__)

        @with_fallback(fallback_value={}, log_level="debug")
        def method(self):
            raise ValueError("error")

    obj = TestClass()
    result = obj.method()
    assert result == {}
```

### Step 7: Run validation

```bash
# Unit tests
pytest ml/tests/unit/common/test_error_handlers.py -v

# Updated store tests
pytest ml/tests/unit/stores/ -v

# Full test suite
pytest ml/tests/ -v

# Linting
ruff check ml/common/error_handlers.py
mypy ml/common/error_handlers.py --strict

# Pattern validation
make validate-nautilus-patterns
```

## Testing Requirements

- [ ] Test context managers (success, error, fallback)
- [ ] Test decorators (success, error, fallback)
- [ ] Test logger integration
- [ ] Test re_raise behavior
- [ ] Test fallback values
- [ ] Test operation name formatting
- [ ] Integration test: verify stores still handle errors correctly

## Rollback Plan

```bash
git checkout ml/common/error_handlers.py ml/common/__init__.py
git checkout ml/stores/
git checkout ml/registry/
git checkout ml/tests/unit/common/test_error_handlers.py
```

## Success Metrics
- Lines reduced: ~1,400 (680 blocks × ~2 lines avg)
- DRY impact score: 680 → ~70 (90% reduction)
- Files affected: 52 (50 updated + 2 new)
- Test coverage: New module at 100%
- Error logging: Consistent across all modules
- Fallback behavior: Standardized

## Notes
- This is the largest impact task in Phase 1 (213 files affected)
- Update top 50 files initially, remaining can be done incrementally
- Error messages must remain informative - don't lose context
- Fallback behavior is critical for registry operations (non-blocking)
- Database operations should re-raise by default (data integrity)
- Use appropriate log levels (error for DB, warning for registry)
- Context managers are clearer than decorators for multi-statement blocks
- Decorators are better for single-function error handling
