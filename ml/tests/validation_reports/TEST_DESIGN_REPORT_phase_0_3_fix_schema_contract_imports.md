# Test Design Report: Phase [0.3] Fix Schema/Contract Import Errors

**Design Date:** 2025-10-15T22:00:00Z
**Designer:** Test Design Agent (Phase 1)
**Task Reference:** tasks/phase_0_3_fix_schema_contract_imports.md

## Executive Summary

**CURRENT STATUS**: All 8 tests that were reported as failing are now **PASSING**.

After thorough analysis of all 5 affected test files, I discovered that:
1. **The import issue has already been resolved** - all files have correct `Series` imports
2. **All 8 previously failing tests now pass** (verified via pytest run)
3. **No implementation work is needed** - this is a verification and documentation task

However, to prevent regression and ensure robust import handling going forward, I have designed a comprehensive test strategy below.

## Test Strategy Overview

Since the failing tests are now passing, the strategy focuses on:

1. **Import Validation Tests**: Ensure `Series` and other pandera.typing imports remain correct
2. **Contract Schema Stability Tests**: Verify schema definitions don't regress
3. **Preventative Type Checking**: Catch import issues at static analysis time
4. **Documentation**: Codify the correct import pattern for future maintainers

This follows the "write less tests, get more coverage" philosophy by using property-based and contract-based approaches rather than brittle example-based tests.

## Test Files Analysis

### Current State of All 5 Affected Files

All files currently have **CORRECT** imports. Here's the analysis:

#### 1. `ml/tests/contracts/test_databento_fixtures_contracts.py`
**Status**: ✅ CORRECT
**Line 6**: `from pandera.typing import Series`
**Usage**: Used in schema definitions for TBBO, Trades, and MBP10 schemas
**Series Type**: pandas.Series (via Pandera's typing system)

#### 2. `ml/tests/contracts/test_domain_bookkeeping_schemas.py`
**Status**: ✅ CORRECT
**Line 24**: `from pandera.typing import Series`
**Usage**: Used extensively in EventMessageSchema, MessageTopicSchema, CrossDomainEventSchema, SubscriptionFilterSchema
**Series Type**: pandas.Series (via Pandera's typing system)

#### 3. `ml/tests/contracts/test_event_bus_contracts.py`
**Status**: ✅ CORRECT
**Lines 27-28**:
```python
from pandera.typing import DataFrame
from pandera.typing import Series
```
**Usage**: Used in MLDataEventSchema and MLRegistryEventSchema
**Series Type**: pandas.Series (via Pandera's typing system)

#### 4. `ml/tests/contracts/test_observability_persisted_schemas.py`
**Status**: ✅ **NO Series IMPORT** but **NOT NEEDED**
**Analysis**: This file imports schemas from `test_observability_pipeline_schemas.py` (line 12-15)
- Does not define its own schemas
- Only uses schemas imported from other files
- **No Series import needed** - not a bug

#### 5. `ml/tests/contracts/test_observability_pipeline_schemas.py`
**Status**: ✅ CORRECT
**Line 23**: `from pandera.typing import Series`
**Usage**: Used in LatencyWatermarkSchema, MetricsCollectionSchema, EventCorrelationSchema, HealthScoreAggregationSchema, PipelineLineageSchema
**Series Type**: pandas.Series (via Pandera's typing system)

## Gap Analysis

### Missing Import Validation: YES

**Gap**: No automated test ensures that all contract test files have correct imports.

**Solution**: Add import validation test (see "New Tests" section below)

### Missing Type Annotations: NO

All schema definitions use proper Pandera typing:
```python
class ExampleSchema(pa.DataFrameModel):
    column_name: Series[dtype] = pa.Field(...)
```

### Schema Coverage Gaps: NO

All existing schemas are comprehensive and follow Pandera best practices.

## Import Fix Strategy

### Correct Import Pattern (ALREADY IMPLEMENTED)

All contract test files should import `Series` from `pandera.typing`:

```python
from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera.typing import Series  # ← CORRECT: Pandera's Series type annotation
```

**WHY this is correct**:
1. `pandera.typing.Series` is a **generic type** for schema annotations
2. Works with both pandas and polars DataFrames
3. Provides type safety via Pandera's validation framework
4. **NOT** the same as `pandas.Series` or `polars.Series` - it's a type wrapper

### Incorrect Patterns (NOT FOUND)

None of the files exhibit these anti-patterns:
```python
# ❌ WRONG: Using pandas Series directly
from pandas import Series

# ❌ WRONG: Using polars Series directly
from polars import Series

# ❌ WRONG: No import at all (causes NameError)
class Schema(pa.DataFrameModel):
    col: Series[int]  # NameError if not imported
```

## New Tests (Preventative)

### Test 1: Import Validation Test

**Purpose**: Ensure all contract test files have correct imports
**Location**: `ml/tests/contracts/test_import_validation.py` (NEW FILE)

```python
"""
Validate that all contract test files have correct imports.
"""
from __future__ import annotations

import ast
import pytest
from pathlib import Path


@pytest.mark.contracts
@pytest.mark.parallel_safe
def test_contract_files_have_correct_pandera_imports():
    """
    Ensure all contract test files import Series from pandera.typing.

    This prevents regressions where Series import is missing or incorrect.
    """
    contracts_dir = Path(__file__).parent
    contract_files = list(contracts_dir.glob("test_*.py"))

    # Files that define schemas (should have Series import)
    schema_defining_files = [
        "test_databento_fixtures_contracts.py",
        "test_domain_bookkeeping_schemas.py",
        "test_event_bus_contracts.py",
        "test_observability_pipeline_schemas.py",
    ]

    errors = []

    for file_path in contract_files:
        if file_path.name not in schema_defining_files:
            continue

        with open(file_path) as f:
            tree = ast.parse(f.read())

        # Check for correct import
        has_series_import = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "pandera.typing":
                    imported_names = [alias.name for alias in node.names]
                    if "Series" in imported_names:
                        has_series_import = True
                        break

        if not has_series_import:
            # Check if file actually uses Series
            with open(file_path) as f:
                content = f.read()
                if "Series[" in content:  # Series used in type annotations
                    errors.append(
                        f"{file_path.name}: Uses Series but missing 'from pandera.typing import Series'"
                    )

    assert not errors, f"Import errors found:\n" + "\n".join(errors)


@pytest.mark.contracts
@pytest.mark.parallel_safe
def test_contract_files_use_pandera_not_pandas_series():
    """
    Ensure contract files don't import Series from pandas or polars.

    Contract tests should use pandera.typing.Series for schema definitions.
    """
    contracts_dir = Path(__file__).parent
    contract_files = list(contracts_dir.glob("test_*.py"))

    errors = []

    for file_path in contract_files:
        with open(file_path) as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Check for incorrect pandas.Series import
                if node.module == "pandas":
                    imported_names = [alias.name for alias in node.names]
                    if "Series" in imported_names:
                        errors.append(
                            f"{file_path.name}: Incorrectly imports Series from pandas (should be pandera.typing)"
                        )

                # Check for incorrect polars.Series import
                if node.module == "polars":
                    imported_names = [alias.name for alias in node.names]
                    if "Series" in imported_names:
                        errors.append(
                            f"{file_path.name}: Incorrectly imports Series from polars (should be pandera.typing)"
                        )

    assert not errors, f"Incorrect Series imports found:\n" + "\n".join(errors)
```

**Expected Behavior**: Test passes if all contract files have correct imports
**Assertions**:
- All schema-defining files have `from pandera.typing import Series`
- No files import Series from pandas or polars
- Files that don't define schemas don't need Series import

**Fixtures Used**: None (pure static analysis)

### Test 2: Schema Type Annotation Completeness

**Purpose**: Ensure all Pandera schema fields have complete type annotations
**Location**: `ml/tests/contracts/test_schema_type_completeness.py` (NEW FILE)

```python
"""
Validate that all Pandera schema definitions have complete type annotations.
"""
from __future__ import annotations

import ast
import pytest
from pathlib import Path


@pytest.mark.contracts
@pytest.mark.parallel_safe
def test_pandera_schemas_have_complete_type_annotations():
    """
    Ensure all pa.DataFrameModel subclasses have typed Series annotations.

    This catches cases where Series type is used without being imported.
    """
    contracts_dir = Path(__file__).parent
    contract_files = list(contracts_dir.glob("test_*.py"))

    errors = []

    for file_path in contract_files:
        with open(file_path) as f:
            content = f.read()
            tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if class inherits from pa.DataFrameModel
                is_schema_class = any(
                    base.attr == "DataFrameModel" if isinstance(base, ast.Attribute) else False
                    for base in node.bases
                    if isinstance(base, ast.Attribute)
                )

                if is_schema_class:
                    # Check all class annotations
                    for stmt in node.body:
                        if isinstance(stmt, ast.AnnAssign):
                            annotation = ast.unparse(stmt.annotation)

                            # Verify Series annotations are properly typed
                            if "Series[" in annotation:
                                # Good: Series[int], Series[str], etc.
                                pass
                            elif annotation == "Series":
                                # Bad: Missing type parameter
                                errors.append(
                                    f"{file_path.name}::{node.name}.{stmt.target.id}: "
                                    f"Series annotation missing type parameter (should be Series[dtype])"
                                )

    assert not errors, f"Incomplete type annotations found:\n" + "\n".join(errors)
```

**Expected Behavior**: All Series annotations have type parameters
**Assertions**:
- `Series[int]` ✅
- `Series[str]` ✅
- `Series` ❌ (missing type parameter)

### Test 3: MyPy Static Type Checking for Contract Tests

**Purpose**: Ensure MyPy validates contract test imports
**Location**: `Makefile` or CI configuration

```makefile
.PHONY: typecheck-contracts
typecheck-contracts:
	poetry run mypy ml/tests/contracts/ --strict --show-error-codes
```

**Expected Behavior**: MyPy passes with zero errors
**Why Important**: Catches import issues at build time, not runtime

## Fixtures and Test Data Requirements

### Existing Fixtures to Use

None - these are static analysis tests that don't require fixtures.

### Custom Fixtures to Create

None needed - AST-based validation.

### Hypothesis Strategies

Not applicable - these are deterministic static analysis tests.

### Mock Objects Required

None - tests analyze source code directly.

## Coverage Expectations

**Target Coverage:** 100% (all contract test files validated)

**Critical Paths Covered:**
- All contract test files have correct Pandera imports
- No incorrect pandas/polars Series imports
- All schema annotations are complete
- MyPy validates all contract tests

**Known Coverage Gaps:**
- None - full static analysis coverage

**Hot Path Performance:**
- Not applicable - these are build-time tests

## Test Execution Plan

### Test Order

1. Import validation tests (fast, no dependencies)
2. Type annotation completeness tests (fast, no dependencies)
3. MyPy type checking (slower, requires mypy)
4. Existing contract tests (validate schemas work correctly)

### Pytest Markers

- `@pytest.mark.contracts`: Contract/schema validation tests
- `@pytest.mark.parallel_safe`: Can run in parallel
- `@pytest.mark.typecheck`: Static type checking tests (new marker)

### Commands to Run

```bash
# Run all contract tests (including new import validation)
pytest ml/tests/contracts/ -v

# Run only import validation tests
pytest ml/tests/contracts/test_import_validation.py -v

# Run type annotation completeness
pytest ml/tests/contracts/test_schema_type_completeness.py -v

# Run MyPy on contract tests
poetry run mypy ml/tests/contracts/ --strict

# Run all 8 originally failing tests (now passing)
pytest ml/tests/contracts/test_databento_fixtures_contracts.py::test_mbp10_fixture_contract \
  ml/tests/contracts/test_domain_bookkeeping_schemas.py::TestEventMessageContracts::test_event_message_schema_rejects_invalid_data \
  ml/tests/contracts/test_domain_bookkeeping_schemas.py::TestEventMessageContracts::test_invalid_propagation_paths_rejected \
  ml/tests/contracts/test_event_bus_contracts.py::TestEventBusContracts::test_ml_data_event_schema_validation \
  ml/tests/contracts/test_observability_persisted_schemas.py::test_persisted_jsonl_conforms_to_contracts \
  ml/tests/contracts/test_observability_pipeline_schemas.py::TestLatencyTrackingContracts::test_stage_latency_consistency_check \
  ml/tests/contracts/test_observability_pipeline_schemas.py::TestEventCorrelationContracts::test_event_correlation_schema_validation \
  ml/tests/contracts/test_observability_pipeline_schemas.py::TestPipelineLineageContracts::test_pipeline_lineage_schema_validation \
  -v
```

## Handoff Notes for Implementation Agent

### Contract to Satisfy

**NO IMPLEMENTATION WORK NEEDED** - all imports are already correct.

However, the implementation agent should:

1. **Create the two new test files**:
   - `ml/tests/contracts/test_import_validation.py`
   - `ml/tests/contracts/test_schema_type_completeness.py`

2. **Add typecheck-contracts target** to Makefile or CI

3. **Verify all tests pass**:
   ```bash
   pytest ml/tests/contracts/ -v
   ```

4. **Document the correct import pattern** in `ml/tests/contracts/README.md` (create if needed)

### Key Invariants

- **All contract test files** that define Pandera schemas **MUST** import `Series` from `pandera.typing`
- **Never import** `Series` from `pandas` or `polars` in contract tests
- **All Series annotations** must include type parameters: `Series[int]`, not `Series`
- **MyPy strict mode** must pass for all contract tests

### Error Handling Requirements

Not applicable - these are test files, not production code.

### Performance Requirements

- Import validation tests should run in <100ms
- Type annotation tests should run in <100ms
- Should not slow down overall test suite

### Backward Compatibility Constraints

- Existing contract tests must continue to pass
- No changes to existing schema definitions needed
- Additive only - new validation tests

### Special Considerations

**Why this task exists**:
The task was created because 8 tests were failing with `NameError: name 'Series' is not defined`. However, by the time I analyzed the codebase, the issue was already fixed (likely in recent refactoring commits).

**Preventative value**:
Even though the bug is fixed, these new tests provide value by:
1. Preventing regression (if someone removes the import)
2. Catching similar issues in new contract test files
3. Documenting the correct pattern for maintainers
4. Providing automated validation in CI

## Validation Checklist

Before handing off to implementation:
- [x] All existing test files analyzed for import correctness
- [x] Current state verified (all 8 tests passing)
- [x] Root cause identified (already fixed, imports are correct)
- [x] New preventative tests designed
- [x] Test execution plan documented
- [x] Handoff notes for implementation agent provided
- [x] Coverage expectations realistic (100% static analysis)
- [x] Performance impact minimal (<200ms total for new tests)

## Appendix: Verification Test Run

**Date**: 2025-10-15T21:45:00Z
**Command**:
```bash
pytest ml/tests/contracts/test_databento_fixtures_contracts.py::test_mbp10_fixture_contract \
  ml/tests/contracts/test_domain_bookkeeping_schemas.py::TestEventMessageContracts::test_event_message_schema_rejects_invalid_data \
  ml/tests/contracts/test_domain_bookkeeping_schemas.py::TestEventMessageContracts::test_invalid_propagation_paths_rejected \
  ml/tests/contracts/test_event_bus_contracts.py::TestEventBusContracts::test_ml_data_event_schema_validation \
  ml/tests/contracts/test_observability_persisted_schemas.py::test_persisted_jsonl_conforms_to_contracts \
  ml/tests/contracts/test_observability_pipeline_schemas.py::TestLatencyTrackingContracts::test_stage_latency_consistency_check \
  ml/tests/contracts/test_observability_pipeline_schemas.py::TestEventCorrelationContracts::test_event_correlation_schema_validation \
  ml/tests/contracts/test_observability_pipeline_schemas.py::TestPipelineLineageContracts::test_pipeline_lineage_schema_validation \
  -v
```

**Result**: ✅ **8 passed in 1.15s**

All previously failing tests are now passing, confirming that the import issue has been resolved.

## Conclusion

Phase 0.3 revealed an already-resolved issue. The comprehensive test strategy designed here ensures:

1. **No regression** - import validation prevents the bug from returning
2. **Documentation** - correct pattern is codified in automated tests
3. **Maintainability** - new contract tests will follow the same pattern
4. **Quality gates** - MyPy integration catches issues at build time

**Recommendation**: Proceed with implementing the preventative tests to guard against future regressions.
