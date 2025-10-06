# Task: [Phase 0.1] Remove stores → actors Circular Dependency

## Context
**Phase:** 0 - Foundation (Critical Blockers)
**Task ID:** 0.1
**Depends On:** none
**Estimated Effort:** 0.5 hours

## Scope
Remove the circular import between `ml/stores/__init__.py` and `ml/actors/base.py` by eliminating the runtime import of `BaseMLInferenceActor` in the stores module.

## Required Reading
- [x] REFACTORING_PLAN.md (Phase 0.1)
- [x] ml/docs/development/CODING_STANDARDS.md
- [x] ml/docs/architecture/universal_patterns_guide.md

## Definition of Done
- [ ] `ml/stores/__init__.py` does NOT import `BaseMLInferenceActor` at runtime
- [ ] `BaseMLInferenceActor` can be imported for TYPE_CHECKING only (if needed)
- [ ] All tests pass: `pytest ml/tests/ -v`
- [ ] No import errors when `import ml.stores` is executed standalone
- [ ] No import errors when `import ml.actors` is executed standalone
- [ ] Circular dependency broken (verify with import order test)
- [ ] Ruff check passes: `ruff check ml/stores/__init__.py`
- [ ] MyPy passes: `mypy ml/stores/__init__.py --strict`

## Files to Modify
- [ ] ml/stores/__init__.py (line 20 - remove or move to TYPE_CHECKING)

## Implementation Steps
1. Read `ml/stores/__init__.py` to understand current imports
2. Locate line 20 where `BaseMLInferenceActor` is imported
3. Determine if import is needed at runtime or only for type hints
4. If only for type hints: Move inside `if TYPE_CHECKING:` block
5. If in `__all__`: Remove from exports (tests should import from ml.actors directly)
6. Test standalone imports:
   ```bash
   python -c "import ml.stores; print('✓ ml.stores')"
   python -c "import ml.actors; print('✓ ml.actors')"
   python -c "import ml.actors; import ml.stores; print('✓ order 1')"
   python -c "import ml.stores; import ml.actors; print('✓ order 2')"
   ```
7. Run tests: `pytest ml/tests/unit/stores/ -v`
8. Run pattern validation: `make validate-nautilus-patterns`

## Testing Requirements
- [ ] All existing tests pass unchanged
- [ ] Create test: `ml/tests/test_no_circular_imports.py`
  ```python
  """Test that circular imports are eliminated."""

  def test_stores_import_standalone():
      """Stores module can be imported without actors."""
      import ml.stores
      assert ml.stores is not None

  def test_actors_import_standalone():
      """Actors module can be imported without stores."""
      import ml.actors
      assert ml.actors is not None

  def test_import_order_independence():
      """Imports work in either order."""
      # Clean slate
      import sys
      mods_to_remove = [k for k in sys.modules.keys() if k.startswith('ml.')]
      for mod in mods_to_remove:
          del sys.modules[mod]

      # Test order 1
      import ml.stores
      import ml.actors

      # Clean again
      mods_to_remove = [k for k in sys.modules.keys() if k.startswith('ml.')]
      for mod in mods_to_remove:
          del sys.modules[mod]

      # Test order 2
      import ml.actors
      import ml.stores
  ```

## Rollback Plan
```bash
git checkout ml/stores/__init__.py
```

## Success Metrics
- Circular dependency chain count: 3 → 2
- Import time: (measure with `time python -c "import ml.stores"`)
- Test suite: 100% pass rate maintained
- Pattern validation: 0 new errors

## Notes
- This is a **critical blocker** - must be completed before any other refactoring
- The import likely exists for re-export purposes in `__all__`
- Tests may need to import `BaseMLInferenceActor` directly from `ml.actors.base`
