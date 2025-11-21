# Nautilus Trader ML Refactoring Plan

**Generated:** 2025-10-04
**Status:** Draft - Awaiting approval
**Estimated Total Effort:** 110-150 hours (7-10 weeks)


## Executive Summary

Analysis of 120K LOC revealed three critical categories of technical debt:

1. **God Classes:** 8 files ranging from 922-3,424 lines with SRP violations
2. **DRY Violations:** 2,847 total impact score across 6 categories
3. **Circular Dependencies:** 3 circular import chains + 23 layer violations

This plan provides a phased approach to address these issues while maintaining backward compatibility and minimizing disruption.

---

## Phase 0: Foundation (Week 0 - IMMEDIATE)

**Goal:** Break critical circular dependencies that block everything else

### 0.1 Remove stores → actors circular dependency
**File:** `ml/stores/__init__.py:20`
**Action:** Remove `from ml.actors.base import BaseMLInferenceActor`
**Effort:** 30 minutes
**Impact:** Breaks actors ↔ stores cycle

### 0.2 Extract dataset constants to config
**Files:**
- Create `ml/config/dataset_ids.py`
- Update `ml/registry/bootstrap_datasets.py:29-30`
- Update `ml/stores/data_store.py`

**Action:**
```python
# ml/config/dataset_ids.py
EARNINGS_ACTUALS_DATASET_ID = "earnings.actuals"
EARNINGS_ESTIMATES_DATASET_ID = "earnings.estimates"
```

**Effort:** 1 hour
**Impact:** Breaks registry ↔ stores cycle

### 0.3 Remove concrete store re-exports from actors
**File:** `ml/actors/base.py:2035-2038`
**Action:** Remove runtime re-exports, keep only TYPE_CHECKING imports
**Effort:** 30 minutes
**Impact:** Reduces coupling, breaks transitive cycles

**Total Phase 0 Effort:** 2 hours

---

## Phase 1: DRY Violations - Critical Path (Weeks 1-2)

**Goal:** Eliminate highest-impact code duplication

### 1.1 Centralize database engine creation (Week 1)
**Impact Score:** 1,953 (63 files affected)

**Actions:**
1. Create `ml/common/db_utils.py`:
```python
def get_or_create_engine(
    connection_string: str,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_pre_ping: bool = True,
    **kwargs: Any
) -> Engine:
    """Centralized engine creation with standard error handling."""
    from ml.core.db_engine import EngineManager
    return EngineManager.get_engine(
        connection_string,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=pool_pre_ping,
        **kwargs
    )
```

2. Replace 8 module-level `create_engine()` wrappers
3. Update 63 files to use centralized function

**Effort:** 8 hours
**Benefit:** -300 lines, single point of configuration

### 1.2 Create table schema factory (Week 1)
**Impact Score:** 567 (6 store files affected)

**Actions:**
1. Create `ml/stores/table_factory.py`:
```python
def get_schema_name(engine: Engine) -> str | None:
    """Get schema name based on dialect."""
    dialect = getattr(getattr(engine, "dialect", None), "name", None)
    return "public" if dialect and dialect != "sqlite" else None

def build_nautilus_timestamp_columns() -> list[Column]:
    """Standard timestamp columns for all ML tables."""
    return [
        Column("ts_event", BIGINT, primary_key=True),
        Column("ts_init", BIGINT)
    ]

def create_ml_table(
    name: str,
    columns: list[Column],
    engine: Engine,
    indexes: list[Index] | None = None
) -> Table:
    """Factory for ML tables with standard schema."""
    # Implementation
```

2. Refactor `_setup_tables()` in:
   - `ml/stores/feature_store.py`
   - `ml/stores/model_store.py`
   - `ml/stores/strategy_store.py`

**Effort:** 6 hours
**Benefit:** -500 lines, consistent table definitions

### 1.3 Standardize error handling (Week 2)
**Impact Score:** 680 (213 files affected)

**Actions:**
1. Create `ml/common/error_handlers.py`:
```python
from contextlib import contextmanager
from typing import Any, Callable

@contextmanager
def db_operation_handler(
    operation_name: str,
    logger: logging.Logger,
    fallback: Any = None
):
    """Context manager for database operations with standard error handling."""
    try:
        yield
    except Exception as e:
        logger.error("Failed to %s: %s", operation_name, e)
        if fallback is not None:
            return fallback
        raise

def with_db_error_handling(fallback_value: Any = None):
    """Decorator for database operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = getattr(args[0], 'logger', logging.getLogger(__name__))
                logger.error("Failed to execute %s: %s", func.__name__, e)
                return fallback_value
        return wrapper
    return decorator
```

2. Update top 50 files with most duplicated error patterns

**Effort:** 10 hours
**Benefit:** -1,400 lines, consistent error handling

**Total Phase 1 Effort:** 24 hours

---

## Phase 2: God Class Decomposition - Priority Queue (Weeks 3-6)

**Goal:** Break down largest classes using Strangler Fig pattern

### 2.1 DataStore decomposition (Week 3-4)
**Current:** 3,731 lines
**Target:** 5 components + facade (~800 lines total in facade)

**Extraction targets:**
1. `ml/stores/schema_validator.py` (lines 972-1176)
2. `ml/stores/data_writer.py` (lines 1202-1500)
3. `ml/stores/data_reader.py` (lines 537-707)
4. `ml/stores/contract_enforcer.py` (lines 1800+)
5. `ml/stores/data_store_facade.py` (lines 337-532 + coordination)

**Approach:**
- Week 3: Extract non-dependent components (schema_validator, data_reader)
- Week 4: Extract writers and contract enforcer
- Maintain backward compatibility via facade
- Add feature flag: `ML_USE_LEGACY_DATA_STORE=1`

**Effort:** 20 hours
**Benefit:** Testability, maintainability

### 2.2 MLPipelineOrchestrator decomposition (Week 5)
**Current:** 4,592 lines (LARGEST FILE)
**Target:** 5 components + orchestrator (~600 lines)

**Extraction targets:**
1. `ml/orchestration/config_resolver.py` (lines 173-279)
2. `ml/orchestration/ingestion_coordinator.py` (lines 501-677)
3. `ml/orchestration/dataset_builder.py` (lines 1130-1500+)
4. `ml/orchestration/binding_resolver.py` (lines 787-1109)
5. `ml/orchestration/discovery_client.py` (lines 678-1109)

**Effort:** 25 hours

### 2.3 ModelRegistry decomposition (Week 6)
**Current:** 2,256 lines
**Target:** 5 components

**Extraction targets:**
1. `ml/registry/model_persistence.py`
2. `ml/registry/model_deployment_mgr.py`
3. `ml/registry/model_quality_validator.py`
4. `ml/registry/ab_testing_manager.py`
5. `ml/registry/canary_deployment_mgr.py`

**Effort:** 15 hours

**Total Phase 2 Effort:** 60 hours

---

## Phase 3: Remaining God Classes (Weeks 7-9)

### 3.1 DataRegistry (Week 7)
**Effort:** 10 hours

### 3.2 FeatureStore (Week 8)
**Effort:** 12 hours

### 3.3 DataScheduler + TFTDatasetBuilder (Week 9)
**Effort:** 22 hours (10h + 12h)

**Total Phase 3 Effort:** 44 hours

---

## Phase 4: Documentation & Testing Consolidation (Week 10)

### 4.1 Consolidate conftest.py files
**Current:** 26 files
**Target:** 4 canonical files

Keep:
- `/conftest.py` (root)
- `/tests/conftest.py` (core Nautilus)
- `/ml/conftest.py` (ML module bootstrap)
- `/ml/tests/conftest.py` (ML test fixtures)

**Actions:**
1. Audit all 26 conftest.py files for unique fixtures
2. Promote reusable fixtures to parent conftest.py
3. Delete redundant conftest.py files
4. Update import statements in tests

**Effort:** 6 hours

### 4.2 Documentation consolidation
**Current:** Markdown files scattered across root, /docs, /ml/docs
**Target:** Single documentation tree in /docs

**Actions:**
1. Move all ML docs to `/docs/ml/`
2. Create `/docs/INDEX.md` with complete navigation
3. Delete stale markdown files at root (except README, CLAUDE.md, CONTRIBUTING)
4. Generate:
   - Module dependency graphs
   - Database schema docs from migrations
   - API reference for actors/stores/registries

**Effort:** 8 hours

**Total Phase 4 Effort:** 14 hours

---

## Implementation Guidelines

### Backward Compatibility Strategy

1. **Maintain Facades:** Keep original class names as thin wrappers during migration
2. **Feature Flags:** Environment variables to toggle old/new implementations
   ```python
   if os.getenv("ML_USE_LEGACY_DATA_STORE") == "1":
       from ml.stores.data_store_legacy import DataStore
   else:
       from ml.stores.data_store_facade import DataStore
   ```
3. **Deprecation Warnings:** Log warnings when legacy paths are used
4. **Version Timeline:** Remove legacy code in next major version (v2.0.0)

### Testing Strategy

1. **Pre-refactoring:** Capture current behavior with characterization tests
2. **During refactoring:** Run full test suite with both old and new implementations
3. **Post-refactoring:** Add unit tests for new components
4. **Integration tests:** Ensure end-to-end workflows unchanged

### Code Review Process

1. **One PR per component extraction** (not per god class)
2. **Maximum PR size:** 500 lines changed
3. **Required reviews:** 2 approvals (technical lead + domain expert)
4. **CI gates:**
   - All tests pass
   - Coverage ≥ current (no regression)
   - Ruff + MyPy pass
   - `make validate-nautilus-patterns` passes

---

## Risk Mitigation

### Risk 1: Breaking existing integrations
**Mitigation:**
- Facade pattern maintains API compatibility
- Feature flags allow rollback
- Extensive integration testing

### Risk 2: Performance regression
**Mitigation:**
- Benchmark critical paths before/after
- Profile memory usage
- Monitor production metrics

### Risk 3: Database migration issues
**Mitigation:**
- All schema changes via versioned migrations
- Test on staging database first
- Rollback plan for each migration

### Risk 4: Developer confusion during transition
**Mitigation:**
- Clear documentation of new vs old
- Team training sessions
- Pair programming for first extractions

---

## Success Metrics

### Code Quality
- [ ] God classes: 0 classes >1000 lines
- [ ] DRY violations: <50 impact score (from 2,847)
- [ ] Circular dependencies: 0 (from 3)
- [ ] Layer violations: <5 (from 23)

### Test Coverage
- [ ] ML module coverage: ≥90% (current: ~85%)
- [ ] Store tests: ≥95%
- [ ] Registry tests: ≥90%

### Performance
- [ ] Hot path latency: <5ms P99 (maintained)
- [ ] Test suite time: ≤ current + 10%
- [ ] Memory usage: ≤ current + 5%

### Documentation
- [ ] All public APIs documented
- [ ] Architecture decision records (ADRs) for major changes
- [ ] Migration guide for downstream users

---

## Timeline Summary

| Phase | Description | Duration | Effort (hours) |
|-------|-------------|----------|----------------|
| 0 | Break circular dependencies | Week 0 | 2 |
| 1 | DRY violations - critical | Weeks 1-2 | 24 |
| 2 | God classes - priority | Weeks 3-6 | 60 |
| 3 | God classes - remaining | Weeks 7-9 | 44 |
| 4 | Documentation & testing | Week 10 | 14 |
| **Total** | | **10 weeks** | **144 hours** |

---

## Next Steps

1. **Review this plan** with team
2. **Create GitHub issues** for Phase 0 tasks (2 hours of work)
3. **Schedule kickoff meeting** to assign ownership
4. **Set up tracking board** (GitHub Projects or Jira)
5. **Begin Phase 0** immediately after approval

---

## Appendix A: Detailed Analysis Reports

Full analysis reports are available in the agent outputs:
- God class analysis: See Task 1 output
- DRY violations: See Task 2 output
- Dependency graph: See Task 3 output

---

## Appendix B: Pre-Commit Hook Updates

Add to `.pre-commit-config.yaml`:

```yaml
  - id: check-circular-imports
    name: Check for circular imports
    entry: python -m scripts.check_circular_imports
    language: python
    files: '^ml/.*\.py$'

  - id: check-god-classes
    name: Check for god classes (>1000 lines)
    entry: bash -c 'find ml -name "*.py" -exec wc -l {} \; | awk "\$1 > 1000 {print; exit 1}"'
    language: system
    files: '^ml/.*\.py$'

  - id: enforce-layer-boundaries
    name: Enforce architectural layers
    entry: python .pre-commit-hooks/check_layer_violations.py
    language: python
    files: '^ml/.*\.py$'
```

---

**Author:** Claude (Sonnet 4.5) + Agentic Analysis
**Approved by:** _[Pending]_
**Last updated:** 2025-10-04
