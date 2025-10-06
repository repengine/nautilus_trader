# Task: [Phase 1.1] Centralize Database Engine Creation

## Context
**Phase:** 1 - DRY Violations - Critical Path
**Task ID:** 1.1
**Depends On:** Phase 0 complete
**Estimated Effort:** 8 hours
**Impact Score:** 1,953 (63 files affected)

## Scope
Create a centralized database engine creation utility in `ml/common/db_utils.py` to eliminate duplicated `create_engine()` wrapper functions across 8 store modules and standardize usage across 63 files.

## Required Reading
- [x] REFACTORING_PLAN.md (Phase 1.1)
- [x] DRY Violations Report (Database Connection Patterns section)
- [x] ml/docs/development/CODING_STANDARDS.md
- [x] ml/docs/architecture/universal_patterns_guide.md

## Definition of Done
- [ ] New file created: `ml/common/db_utils.py` with centralized utilities
- [ ] Function `get_or_create_engine()` implemented with standard error handling
- [ ] All 8 module-level `create_engine()` wrappers removed:
  - [ ] ml/stores/feature_store.py
  - [ ] ml/stores/model_store.py
  - [ ] ml/stores/strategy_store.py
  - [ ] ml/stores/earnings_store.py
  - [ ] ml/stores/instrument_metadata_store.py
  - [ ] ml/stores/data_processor.py
  - [ ] ml/stores/infrastructure.py
  - [ ] ml/observability/db_persistence.py
- [ ] All 63 files updated to use centralized function
- [ ] Comprehensive test suite created
- [ ] All tests pass
- [ ] Ruff check passes
- [ ] MyPy strict passes
- [ ] Pattern validation passes
- [ ] Backward compatibility maintained

## Files to Modify

### Create New (2 files)
- [ ] `ml/common/db_utils.py` - Centralized database utilities
- [ ] `ml/tests/unit/common/test_db_utils.py` - Comprehensive test suite

### Modify (8 stores - remove wrappers)
- [ ] `ml/stores/feature_store.py`
- [ ] `ml/stores/model_store.py`
- [ ] `ml/stores/strategy_store.py`
- [ ] `ml/stores/earnings_store.py`
- [ ] `ml/stores/instrument_metadata_store.py`
- [ ] `ml/stores/data_processor.py`
- [ ] `ml/stores/infrastructure.py`
- [ ] `ml/observability/db_persistence.py`

### Update (55 other files - update imports)
- Search with: `grep -r "create_engine(" ml/ --include="*.py" | grep -v __pycache__`

## Implementation Steps

### Step 1: Create ml/common/db_utils.py

```python
"""
Database utilities for ML module.

Provides centralized database engine creation with standardized connection pooling,
error handling, and configuration. All ML components should use these utilities
instead of creating engines directly.

Usage Example
-------------
    from ml.common.db_utils import get_or_create_engine

    # Standard usage
    engine = get_or_create_engine("postgresql://user:pass@localhost/db")

    # Custom pool settings
    engine = get_or_create_engine(
        "postgresql://user:pass@localhost/db",
        pool_size=10,
        max_overflow=20,
    )

"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.engine import Engine

__all__ = [
    "get_or_create_engine",
    "get_default_pool_config",
]

logger = logging.getLogger(__name__)


def get_default_pool_config() -> dict[str, Any]:
    """
    Get default database connection pool configuration.

    Returns
    -------
    dict[str, Any]
        Default pool configuration parameters.

    """
    return {
        "pool_size": 5,
        "max_overflow": 10,
        "pool_pre_ping": True,
        "pool_recycle": 3600,
    }


def get_or_create_engine(
    connection_string: str,
    pool_size: int | None = None,
    max_overflow: int | None = None,
    pool_pre_ping: bool = True,
    pool_recycle: int = 3600,
    **kwargs: Any,
) -> Engine:
    """
    Create or retrieve a SQLAlchemy engine with standardized configuration.

    This function wraps EngineManager.get_engine() to provide:
    - Consistent connection pooling across all ML components
    - Standardized error handling
    - Default configuration that works for most use cases
    - Single source of truth for engine creation

    Parameters
    ----------
    connection_string : str
        SQLAlchemy database URL (e.g., "postgresql://user:pass@host/db").
    pool_size : int | None, optional
        Number of connections to keep in pool. Defaults to 5.
    max_overflow : int | None, optional
        Max connections beyond pool_size. Defaults to 10.
    pool_pre_ping : bool, optional
        Test connections before using. Defaults to True.
    pool_recycle : int, optional
        Recycle connections after N seconds. Defaults to 3600 (1 hour).
    **kwargs : Any
        Additional arguments passed to EngineManager.get_engine().

    Returns
    -------
    Engine
        Configured SQLAlchemy engine.

    Raises
    ------
    ValueError
        If connection_string is empty or invalid.
    RuntimeError
        If engine creation fails.

    Notes
    -----
    - Uses EngineManager for connection pooling and lifecycle management
    - All ML stores should use this function instead of creating engines directly
    - Pool settings are optimized for ML workload patterns
    - Pre-ping prevents "connection closed" errors in long-running processes

    """
    if not connection_string:
        raise ValueError("connection_string cannot be empty")

    from ml.core.db_engine import EngineManager

    # Use defaults if not specified
    defaults = get_default_pool_config()
    final_pool_size = pool_size if pool_size is not None else defaults["pool_size"]
    final_max_overflow = max_overflow if max_overflow is not None else defaults["max_overflow"]

    try:
        engine = EngineManager.get_engine(
            connection_string,
            pool_size=final_pool_size,
            max_overflow=final_max_overflow,
            pool_pre_ping=pool_pre_ping,
            pool_recycle=pool_recycle,
            **kwargs,
        )
        logger.debug(
            "Created engine for %s with pool_size=%d, max_overflow=%d",
            connection_string.split("@")[-1],  # Log host only, not credentials
            final_pool_size,
            final_max_overflow,
        )
        return engine
    except Exception as e:
        logger.error("Failed to create database engine: %s", e)
        raise RuntimeError(f"Database engine creation failed: {e}") from e
```

### Step 2: Update ml/common/__init__.py

Add to imports (alphabetically):
```python
from ml.common.db_utils import (
    get_or_create_engine,
    get_default_pool_config,
)
```

Add to `__all__`

### Step 3: Remove duplicate wrappers from stores

For each store file, find and remove the `create_engine()` function, then update imports:

**Example for ml/stores/feature_store.py:**
```python
# OLD:
def create_engine(connection_string: str, **kwargs: Any) -> Engine:
    return EngineManager.get_engine(connection_string, **kwargs)

# In __init__:
self.engine = create_engine(connection_string)

# NEW:
from ml.common.db_utils import get_or_create_engine

# In __init__:
self.engine = get_or_create_engine(connection_string)
```

### Step 4: Search and update all usages

```bash
# Find all create_engine calls
grep -r "create_engine(" ml/ --include="*.py" | grep -v __pycache__ | grep -v "\.pyc"

# For each file:
# 1. Check if it imports the local create_engine wrapper
# 2. Replace with: from ml.common.db_utils import get_or_create_engine
# 3. Update the call
```

### Step 5: Create comprehensive test suite

Create `ml/tests/unit/common/test_db_utils.py` with tests for:
- Default pool configuration
- Engine creation with defaults
- Engine creation with custom settings
- Error handling (empty connection string)
- Error handling (invalid connection string)
- Integration with EngineManager
- Connection string sanitization in logs (no credentials leaked)

### Step 6: Run validation

```bash
# Unit tests
pytest ml/tests/unit/common/test_db_utils.py -v

# Integration tests
pytest ml/tests/integration/stores/ -v -k "engine"

# Full store tests
pytest ml/tests/unit/stores/ -v

# Linting
ruff check ml/common/db_utils.py
mypy ml/common/db_utils.py --strict

# Pattern validation
make validate-nautilus-patterns
```

## Testing Requirements

### Unit Tests (ml/tests/unit/common/test_db_utils.py)
```python
"""Tests for database utilities."""

import pytest
from unittest.mock import Mock, patch

from ml.common.db_utils import (
    get_or_create_engine,
    get_default_pool_config,
)


def test_get_default_pool_config():
    """Default pool config returns expected values."""
    config = get_default_pool_config()
    assert config["pool_size"] == 5
    assert config["max_overflow"] == 10
    assert config["pool_pre_ping"] is True
    assert config["pool_recycle"] == 3600


def test_get_or_create_engine_with_defaults(mocker):
    """Engine created with default pool settings."""
    mock_engine = Mock()
    mock_get_engine = mocker.patch(
        "ml.core.db_engine.EngineManager.get_engine",
        return_value=mock_engine
    )

    engine = get_or_create_engine("postgresql://localhost/test")

    assert engine == mock_engine
    mock_get_engine.assert_called_once_with(
        "postgresql://localhost/test",
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600,
    )


def test_get_or_create_engine_with_custom_settings(mocker):
    """Engine created with custom pool settings."""
    mock_engine = Mock()
    mock_get_engine = mocker.patch(
        "ml.core.db_engine.EngineManager.get_engine",
        return_value=mock_engine
    )

    engine = get_or_create_engine(
        "postgresql://localhost/test",
        pool_size=10,
        max_overflow=20,
        pool_recycle=7200,
    )

    assert engine == mock_engine
    mock_get_engine.assert_called_once_with(
        "postgresql://localhost/test",
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=7200,
    )


def test_get_or_create_engine_empty_connection_string():
    """Raises ValueError for empty connection string."""
    with pytest.raises(ValueError, match="connection_string cannot be empty"):
        get_or_create_engine("")


def test_get_or_create_engine_handles_engine_manager_error(mocker):
    """RuntimeError raised when EngineManager fails."""
    mocker.patch(
        "ml.core.db_engine.EngineManager.get_engine",
        side_effect=Exception("Connection failed")
    )

    with pytest.raises(RuntimeError, match="Database engine creation failed"):
        get_or_create_engine("postgresql://localhost/test")


def test_connection_string_sanitized_in_logs(mocker, caplog):
    """Connection string credentials not leaked in logs."""
    mock_engine = Mock()
    mocker.patch(
        "ml.core.db_engine.EngineManager.get_engine",
        return_value=mock_engine
    )

    get_or_create_engine("postgresql://user:secret@localhost:5432/testdb")

    # Check logs don't contain password
    for record in caplog.records:
        assert "secret" not in record.message
        assert "user" not in record.message
```

### Integration Test Updates
- [ ] Verify existing store tests pass with new utility
- [ ] Add test that all stores use centralized function

## Rollback Plan

```bash
# If issues arise, revert changes
git checkout ml/common/db_utils.py ml/common/__init__.py
git checkout ml/stores/
git checkout ml/observability/db_persistence.py
git checkout ml/tests/unit/common/test_db_utils.py
```

## Success Metrics
- Lines reduced: ~150 (8 duplicate functions × ~18 lines each)
- DRY impact score: 1,953 → ~200 (87% reduction)
- Files affected: 65 (8 modified + 55 updated + 2 new)
- Test coverage: New module at 100%
- Single source of truth: ✅
- Configuration consistency: ✅

## Notes
- This is the **highest impact** DRY violation fix
- Affects 63 files - careful testing required
- EngineManager.get_engine() already does pooling - we're just standardizing the wrapper
- Default pool settings are optimized for ML workloads
- Backward compatible - same signature as old wrappers
- Credentials sanitization in logs prevents security issues
- Pool recycle prevents stale connections in long-running processes

## Dependencies
- Requires Phase 0 complete (no circular dependencies)
- Must not break existing store functionality
- Connection pooling behavior must remain identical
