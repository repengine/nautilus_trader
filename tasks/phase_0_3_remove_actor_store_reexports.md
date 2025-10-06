# Task: [Phase 0.3] Remove Concrete Store Re-exports from Actors

## Context
**Phase:** 0 - Foundation (Critical Blockers)
**Task ID:** 0.3
**Depends On:** 0.1, 0.2
**Estimated Effort:** 0.5 hours

## Scope
Remove runtime re-exports of concrete store classes from `ml/actors/base.py` (lines 2035-2038) to reduce coupling and break transitive circular dependency chains.

## Required Reading
- [x] REFACTORING_PLAN.md (Phase 0.3)
- [x] ml/docs/development/CODING_STANDARDS.md
- [x] ml/docs/architecture/universal_patterns_guide.md

## Definition of Done
- [ ] Lines 2035-2038 of `ml/actors/base.py` removed
- [ ] Concrete stores NOT re-exported from actors module
- [ ] TYPE_CHECKING imports remain (lines 73-76)
- [ ] All tests updated to import stores directly from `ml.stores`
- [ ] All tests pass: `pytest ml/tests/ -v`
- [ ] No runtime dependencies on concrete store imports in actors
- [ ] Ruff check passes
- [ ] MyPy passes
- [ ] Pattern validation passes

## Files to Modify
- [ ] ml/actors/base.py (lines 2035-2038 - REMOVE)
- [ ] Any test files importing stores from actors module

## Implementation Steps

### Step 1: Identify the re-exports
Read `ml/actors/base.py` lines 2035-2038 to confirm:
```python
# Expected to find something like:
from ml.stores.data_store import DataStore as DataStore
from ml.stores.feature_store import FeatureStore as FeatureStore
from ml.stores.model_store import ModelStore as ModelStore
from ml.stores.strategy_store import StrategyStore as StrategyStore
```

### Step 2: Verify TYPE_CHECKING imports exist
Check lines 73-76 for:
```python
if TYPE_CHECKING:
    from ml.stores.protocols import FeatureStoreProtocol
    # ... etc
```
These should REMAIN (type hints only)

### Step 3: Remove runtime re-exports
Delete lines 2035-2038 completely

### Step 4: Find affected test files
```bash
# Search for tests importing stores from actors
grep -r "from ml.actors.base import.*Store" ml/tests/ --include="*.py"
grep -r "from ml.actors import.*Store" ml/tests/ --include="*.py"
```

### Step 5: Update test imports
Replace any imports like:
```python
# OLD (incorrect):
from ml.actors.base import FeatureStore

# NEW (correct):
from ml.stores import FeatureStore
```

### Step 6: Verify no runtime usage in actors
```bash
# Should only find TYPE_CHECKING imports
grep -n "import.*Store" ml/actors/base.py
```

### Step 7: Run validation
```bash
# Test imports
python -c "import ml.actors.base; print('✓ actors imports')"

# Run tests
pytest ml/tests/unit/actors/ -v
pytest ml/tests/integration/actors/ -v

# Full suite
pytest ml/tests/ -v

# Linting
ruff check ml/actors/base.py
mypy ml/actors/base.py --strict

# Pattern validation
make validate-nautilus-patterns
```

## Testing Requirements
- [ ] All existing tests pass (possibly with import updates)
- [ ] Verify stores can't be imported from actors at runtime:
  ```python
  # ml/tests/test_no_circular_imports.py (add to existing file)

  def test_stores_not_reexported_from_actors():
      """Store classes should not be importable from actors module at runtime."""
      import ml.actors.base as actors_base

      # These should NOT be accessible
      assert not hasattr(actors_base, 'FeatureStore')
      assert not hasattr(actors_base, 'ModelStore')
      assert not hasattr(actors_base, 'StrategyStore')
      assert not hasattr(actors_base, 'DataStore')

  def test_stores_available_from_stores_module():
      """Store classes should be imported from ml.stores."""
      from ml.stores import (
          FeatureStore,
          ModelStore,
          StrategyStore,
          DataStore,
      )
      assert FeatureStore is not None
      assert ModelStore is not None
      assert StrategyStore is not None
      assert DataStore is not None
  ```

## Rollback Plan
```bash
git checkout ml/actors/base.py
# If test files were modified:
git checkout ml/tests/
```

## Success Metrics
- Circular dependency chain count: 1 → 0 ✅
- Coupling reduced: actors no longer reference concrete stores
- Test suite: 100% pass rate maintained
- Lines removed: ~4 from base.py
- Import updates: ~5-10 test files (estimate)
- Pattern validation: 0 new errors

## Notes
- This completes Phase 0, breaking ALL circular dependencies
- Actors should depend on store protocols, not concrete implementations
- Tests importing stores from actors module are doing it wrong - fix them
- TYPE_CHECKING imports are fine and should remain
- This aligns with Protocol-First Interface Design (Universal Pattern 2)
- After this task, all Phase 1+ refactoring can proceed safely

## Dependencies Eliminated
After this task, dependency graph becomes:
```
actors/ → stores/protocols (TYPE_CHECKING only)
actors/ → registry/ (allowed)
stores/ → registry/ (allowed)
config/ → [nothing in ml/]
```

All circular dependencies resolved! 🎉
