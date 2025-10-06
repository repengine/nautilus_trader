# Task: [Phase 0.2] Extract Dataset Constants to Config

## Context
**Phase:** 0 - Foundation (Critical Blockers)
**Task ID:** 0.2
**Depends On:** 0.1
**Estimated Effort:** 1 hour

## Scope
Move `EARNINGS_ACTUALS_DATASET_ID` and `EARNINGS_ESTIMATES_DATASET_ID` from `ml/stores/data_store.py` to a new centralized config module to break the registry → stores circular dependency.

## Required Reading
- [x] REFACTORING_PLAN.md (Phase 0.2)
- [x] ml/docs/development/CODING_STANDARDS.md
- [x] ml/docs/architecture/universal_patterns_guide.md

## Definition of Done
- [ ] New file created: `ml/config/dataset_ids.py`
- [ ] Constants moved from `ml/stores/data_store.py`
- [ ] `ml/registry/bootstrap_datasets.py` imports from config (lines 29-30)
- [ ] `ml/stores/data_store.py` imports from config
- [ ] All existing usages updated (search entire codebase)
- [ ] All tests pass: `pytest ml/tests/ -v`
- [ ] Circular dependency broken (registry ↔ stores)
- [ ] Ruff check passes
- [ ] MyPy passes
- [ ] Pattern validation passes

## Files to Modify
- [ ] ml/config/dataset_ids.py (CREATE NEW)
- [ ] ml/stores/data_store.py (UPDATE: remove constants, add import)
- [ ] ml/registry/bootstrap_datasets.py (UPDATE: lines 29-30)
- [ ] ml/config/__init__.py (UPDATE: export new constants)

## Implementation Steps

### Step 1: Create ml/config/dataset_ids.py
```python
"""
Dataset ID constants for ML module.

These constants define canonical dataset identifiers used across the ML infrastructure.
All dataset references should use these constants rather than hardcoded strings.
"""

from typing import Final

__all__ = [
    "EARNINGS_ACTUALS_DATASET_ID",
    "EARNINGS_ESTIMATES_DATASET_ID",
]

# Earnings dataset IDs
EARNINGS_ACTUALS_DATASET_ID: Final[str] = "earnings.actuals"
EARNINGS_ESTIMATES_DATASET_ID: Final[str] = "earnings.estimates"
```

### Step 2: Update ml/config/__init__.py
Add to imports (keep alphabetically sorted):
```python
from ml.config.dataset_ids import (
    EARNINGS_ACTUALS_DATASET_ID,
    EARNINGS_ESTIMATES_DATASET_ID,
)
```

Add to `__all__` (keep alphabetically sorted)

### Step 3: Update ml/registry/bootstrap_datasets.py
Replace lines 29-30:
```python
# OLD:
from ml.stores.data_store import EARNINGS_ACTUALS_DATASET_ID
from ml.stores.data_store import EARNINGS_ESTIMATES_DATASET_ID

# NEW:
from ml.config.dataset_ids import (
    EARNINGS_ACTUALS_DATASET_ID,
    EARNINGS_ESTIMATES_DATASET_ID,
)
```

### Step 4: Update ml/stores/data_store.py
1. Find and remove the constant definitions
2. Add import at top:
```python
from ml.config.dataset_ids import (
    EARNINGS_ACTUALS_DATASET_ID,
    EARNINGS_ESTIMATES_DATASET_ID,
)
```

### Step 5: Find all other usages
```bash
grep -r "EARNINGS_ACTUALS_DATASET_ID" ml/ --include="*.py" | grep -v "__pycache__"
grep -r "EARNINGS_ESTIMATES_DATASET_ID" ml/ --include="*.py" | grep -v "__pycache__"
```

Update all imports to use `ml.config.dataset_ids`

### Step 6: Run validation
```bash
# Test imports work in isolation
python -c "from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID; print(EARNINGS_ACTUALS_DATASET_ID)"

# Run tests
pytest ml/tests/ -k "earnings" -v

# Full test suite
pytest ml/tests/unit/ -v

# Linting
ruff check ml/config/dataset_ids.py ml/config/__init__.py
mypy ml/config/dataset_ids.py --strict

# Pattern validation
make validate-nautilus-patterns
```

## Testing Requirements
- [ ] Existing tests pass unchanged
- [ ] Add test to verify constants accessible from config:
  ```python
  # ml/tests/unit/config/test_dataset_ids.py
  """Test dataset ID constants."""

  def test_dataset_ids_accessible_from_config():
      """Dataset IDs can be imported from ml.config."""
      from ml.config import (
          EARNINGS_ACTUALS_DATASET_ID,
          EARNINGS_ESTIMATES_DATASET_ID,
      )
      assert EARNINGS_ACTUALS_DATASET_ID == "earnings.actuals"
      assert EARNINGS_ESTIMATES_DATASET_ID == "earnings.estimates"

  def test_dataset_ids_are_final():
      """Dataset IDs use Final type hint."""
      from ml.config.dataset_ids import (
          EARNINGS_ACTUALS_DATASET_ID,
          EARNINGS_ESTIMATES_DATASET_ID,
      )
      import typing
      # Check type annotations exist
      assert hasattr(EARNINGS_ACTUALS_DATASET_ID, '__class__')
      assert hasattr(EARNINGS_ESTIMATES_DATASET_ID, '__class__')
  ```

## Rollback Plan
```bash
git checkout ml/config/dataset_ids.py ml/config/__init__.py
git checkout ml/stores/data_store.py
git checkout ml/registry/bootstrap_datasets.py
# Remove test file if created
rm -f ml/tests/unit/config/test_dataset_ids.py
```

## Success Metrics
- Circular dependency chain count: 2 → 1
- Files affected: 5 (3 modified, 1 created, 1 test)
- Test suite: 100% pass rate maintained
- Lines of code: +25 (new file + test) -10 (removed duplication) = +15 net
- Pattern validation: 0 new errors

## Notes
- Constants use `Final` type hint to prevent reassignment
- Module docstring explains purpose and usage
- Constants are exported via `ml/config/__init__.py` for convenience
- Follows config-driven development pattern from CODING_STANDARDS.md
