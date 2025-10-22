# Task: [Protocol Remediation 2.3] create_data_store Explicit Signature

## Context
**Phase:** Protocol Remediation - High Priority API Type Safety
**Task ID:** 2.3
**Depends On:** 2.2 (Health TypedDict)
**Estimated Effort:** 1 hour 10 minutes

## Scope
Replace `create_data_store(**kwargs: object)` type erasure with explicit parameter signature and direct import. Remove dynamic import workaround that bypasses mypy type checking.

## Required Reading
- [x] AGENT_TASK_FRAMEWORK.md (5-phase workflow)
- [x] CRITICAL_SAFEGUARDS.md (100% pass rate, test execution verification)
- [x] INTEGRATION_GENERIC_TYPES_REMEDIATION.md (lines 427-585, Violation 5)
- [x] CLAUDE.md (Protocol-First pattern, type safety requirements)

## Definition of Done
- [ ] Function signature uses explicit parameters (no `**kwargs: object`)
- [ ] Direct import replaces dynamic `importlib` + `getattr`
- [ ] Unsafe string literal cast removed
- [ ] Proper protocol imports added (`RawReaderProtocol`, `RawIngestionWriterProtocol`)
- [ ] All tests pass: 100% pass rate (no exceptions)
- [ ] MyPy --strict: 0 errors
- [ ] Ruff check: 0 violations
- [ ] No call site changes needed (already use keyword args)
- [ ] Test coverage ≥90%

## Files to Modify
- [ ] `ml/core/integration.py` (lines 2087-2101: function signature + implementation)
- [ ] `ml/core/integration.py` (top of file: add protocol imports if not present)

## Implementation Steps

### Discovery Phase
1. ✅ Verify function location: `ml/core/integration.py:2087`
2. ✅ Verify call sites already use keyword arguments (lines 628-633, 2058)
3. ✅ Verify protocols exist:
   - `RawReaderProtocol` in `ml/stores/raw_protocols.py`
   - `RawIngestionWriterProtocol` in `ml/stores/raw_protocols.py`
   - `DataRegistry` already imported

### Implementation Phase
1. Update function signature at line 2087:
   ```python
   def create_data_store(
       *,
       registry: DataRegistry,
       connection_string: str,
       raw_reader: RawReaderProtocol | None = None,
       raw_writer: RawIngestionWriterProtocol | None = None,
   ) -> DataStore:
   ```

2. Replace dynamic import (lines 2096-2101):
   ```python
   # OLD (lines 2096-2101):
   import importlib
   from typing import Any as _Any, cast as _cast
   DataStore = getattr(importlib.import_module("ml.stores.data_store"), "DataStore")
   return _cast("DataStoreFacadeProtocol", DataStore(**_cast(dict[str, _Any], kwargs)))

   # NEW:
   from ml.stores.data_store import DataStore

   return DataStore(
       connection_string=connection_string,
       registry=registry,
       raw_reader=raw_reader,
       raw_writer=raw_writer,
   )
   ```

3. Add protocol imports at top of file (if not present):
   ```python
   from ml.stores.raw_protocols import RawIngestionWriterProtocol
   from ml.stores.raw_protocols import RawReaderProtocol
   ```

4. Update docstring with parameter documentation:
   ```python
   """
   Create a DataStore instance with proper type safety.

   This factory function initializes a DataStore with automatic integration
   of all registry and data reader/writer components. It returns the instance
   with full type information for IDE autocomplete and mypy verification.

   Parameters
   ----------
   registry : DataRegistry
       The data registry for dataset manifest and lineage tracking
   connection_string : str
       PostgreSQL connection string for raw market data queries
   raw_reader : RawReaderProtocol | None, optional
       Reader for raw market data (e.g., SQL or Parquet catalog)
   raw_writer : RawIngestionWriterProtocol | None, optional
       Writer for market data (used for backfill and sync operations)

   Returns
   -------
   DataStore
       Initialized data store with full type information

   Example
   -------
   >>> from ml.core.integration import create_data_store
   >>> from ml.registry import DataRegistry
   >>> from ml.stores.io_raw import ParquetCatalogRawReader
   >>>
   >>> registry = DataRegistry(...)
   >>> reader = ParquetCatalogRawReader(...)
   >>> store = create_data_store(
   ...     registry=registry,
   ...     connection_string="postgresql://...",
   ...     raw_reader=reader,
   ... )
   >>> store.read_range(...)  # Proper IDE autocomplete!
   """
   ```

5. Verify call sites still work (no changes needed):
   - Line 628-633: ✅ Already passes all 4 parameters by name
   - Line 2058: ✅ Already passes registry + connection_string by name

### Validation Phase
6. Run static validation:
   ```bash
   poetry run mypy ml/core/integration.py --strict
   ruff check ml/core/integration.py
   python -c "from ml.core.integration import create_data_store; print('✓ Import works')"
   ```

7. Run integration tests (verify "X passed" not "X collected"):
   ```bash
   pytest ml/tests/unit/core/test_create_data_store_signature.py -v
   pytest ml/tests/unit/core -v --tb=short  # No regressions
   ```

## Testing Requirements

### Unit Tests (100% pass rate required)
- [ ] Test function signature has explicit parameters (not `object`)
- [ ] Test parameters are correctly typed:
  - `registry` → `DataRegistry`
  - `connection_string` → `str`
  - `raw_reader` → `RawReaderProtocol | None`
  - `raw_writer` → `RawIngestionWriterProtocol | None`
- [ ] Test return type is `DataStore` (not `object`)
- [ ] Test function accepts keyword-only arguments
- [ ] Test default values work (raw_reader=None, raw_writer=None)
- [ ] Test actual instantiation succeeds
- [ ] Test all call sites still work

### Test Strategy
Follow **Pattern from Task 1.1** (flexible assertions, behavior not implementation):
- ✅ Check type is NOT `object` (behavior)
- ✅ Check type contains expected name component (flexible)
- ❌ Avoid exact type name checks (brittle due to runtime aliases)

Example:
```python
def test_create_data_store_registry_param_not_object():
    """Verify registry parameter is not generic object type."""
    from typing import get_type_hints
    from ml.core.integration import create_data_store

    hints = get_type_hints(create_data_store)
    registry_type_name = hints['registry'].__name__

    # Flexible: check it's not object
    assert registry_type_name != "object"
    # Sanity: check it's registry-related
    assert "Registry" in registry_type_name
```

## Rollback Plan
```bash
git checkout ml/core/integration.py
```

If tests fail:
1. Return to Phase 2 (Implementation Agent)
2. Fix issues based on Phase 4 failure report
3. Re-run Phase 3 → Phase 4

## Success Metrics
- **Type Safety**: MyPy errors reduced (dynamic import workaround removed)
- **Maintainability**: Function signature self-documents parameters
- **IDE Support**: Full autocomplete for parameters and return value
- **Test Pass Rate**: 100% (mandatory, no exceptions)
- **Pattern 2 Compliance**: Direct import enables full type checking
- **Coverage**: ≥90% for new test file

## Critical Reminders (from CRITICAL_SAFEGUARDS.md)

### ⚠️ NO STUBS (Category 2)
- NO `raise NotImplementedError`
- NO `# TODO: implement`
- NO placeholder code
- Full working implementation ONLY

### ⚠️ TEST EXECUTION (Category 3)
- "X passed" required, NOT "X collected"
- Phase 4 agent MUST verify tests actually RUN
- Automatic rejection if tests only collected

### ⚠️ 100% PASS RATE (Lesson from Task 1.1)
- User rejected 71% pass rate with "noooooooope"
- 100% is non-negotiable
- No exceptions

### ⚠️ FLEXIBLE ASSERTIONS (Lesson from Task 1.1)
- Test behavior, not implementation details
- Avoid exact type name checks (brittle)
- Handle runtime type variations (aliases, Python version differences)

## Notes
- **Call sites already correct**: No breaking changes needed
- **Protocols exist**: `RawReaderProtocol` and `RawIngestionWriterProtocol` already defined
- **DataRegistry import**: Already present in file (line 52 runtime, line 38 TYPE_CHECKING)
- **Risk**: LOW - Purely additive type information, same runtime behavior
- **Backward Compatibility**: PRESERVED - Call sites unchanged, same parameters accepted
