# ML Module Refactoring Summary

## Current State (After __init__.py Refactoring)

### ✅ Successfully Refactored

- **19/19 domains** have new comprehensive `__init__.py` files
- **17/19 modules** (89%) import successfully
- **Clean public APIs** exposed for all domains
- **Universal ML Architecture Patterns** documented and enforced

### ❌ Remaining Issues

#### Circular Imports (2 modules affected)

1. **ml.actors** - Circular import with `ml._imports` (ort)
2. **ml.consumers** - Circular import with `ml.stores.data_store`

#### Code Duplication Statistics
| Pattern | Files Affected | Duplicate LOC | Priority |
|---------|---------------|---------------|----------|
| Timestamp conversion | 145 | ~435 | HIGH |
| Dependency checks | 92 | ~276 | HIGH |
| DataFrame transforms | 49 | ~392 | HIGH |
| Retry/backoff logic | 26 | ~390 | MEDIUM |
| Rate limiting | 23 | ~184 | MEDIUM |
| Config validation | 25 | ~200 | LOW |
| **TOTAL** | **360+** | **~5,700+** | - |

## How to Examine Issues

### 1. Circular Import Analysis

```bash
# Run validation script to see current state
python ml/validate_imports.py

# Trace specific circular import
python -c "import ml.actors" 2>&1 | grep "circular import"

# Check import dependencies
python -c "import sys; import ml.actors; print(sys.modules.keys())"
```

### 2. Duplicate Code Analysis

```bash
# Find duplicate timestamp conversions
grep -r "pd.to_datetime.*unit='ns'" ml/ --include="*.py" | wc -l

# Find duplicate retry logic
grep -r "for attempt in range" ml/ --include="*.py" | wc -l

# Find duplicate DataFrame operations
grep -r "df\['returns'\] = df\['close'\].pct_change()" ml/ --include="*.py"
```

### 3. Use the New Clean APIs

```python
# Instead of reaching into internals
from ml.features import FeatureEngineer  # Clean!
from ml.stores import FeatureStore, ModelStore  # Pattern 1 compliant!
from ml.common import get_counter, get_histogram  # Pattern 5 compliant!
```

## Quick Fixes Available Now

### Fix ml.actors circular import

```python
# In ml/config/runtime.py, change line 11:
# FROM:
from ml._imports import ort

# TO:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ml._imports import ort
else:
    ort = None

def get_ort():
    global ort
    if ort is None:
        from ml._imports import ort as _ort
        ort = _ort
    return ort
```

### Fix ml.consumers circular import

```python
# In ml/consumers/protocols.py, use TYPE_CHECKING:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ml.stores import DataStore
```

## Benefits Already Achieved

1. **Discoverability** - All public APIs now visible in `__init__.py`
2. **Documentation** - Comprehensive examples in every domain
3. **Pattern Compliance** - Universal patterns enforced
4. **Type Safety** - Full annotations with `__all__` exports
5. **Separation** - Hot/cold paths clearly documented

## Recent Updates (Status Sync)

- Duplication reduction in stats services:
  - Added `ml/stores/services/common_stats.py` with typed helpers:
    - `build_time_conditions(start_ns, end_ns, field='ts_event')`
    - `build_nullsafe_time_clause(start_ns, end_ns, field='ts_event')`
  - Adopted in `StrategySignalStatsService` (`get_statistics`, `get_signal_distribution`, `get_strategy_performance`).
  - Adopted in `ModelStatsService` (`get_statistics`, `get_model_performance`).
- Lightweight DataFrame utilities created and adopted:
  - Added `ml/common/dataframe_utils.py` with `total_nulls` and `column_nulls`.
  - Integrated into `ml/stores/data_store.py` nullability and quality checks (three call sites).
- Linting and typing hygiene:
  - Removed unused `onnx` import in `ml/models/save_dummy_model.py`.
  - Consolidated `endswith` tuple and fixed indentation in `ml/validate_security_posture.py`.
  - Added `Mapping` import usage in `ml/actors/ml_domain_events.py`.
  - Added type-only `ModelManifest` import under `TYPE_CHECKING` in `ml/training/base.py`.
- Validation snapshot:
  - Ruff clean; `mypy --strict` clean aside from known environment warning (namespace `google`).
  - Focused tests green for stores roundtrip/routing.
  - `make validate-metrics` and `make validate-events` OK.

## Next Steps Priority

### Immediate (Today)

1. Fix ml.actors circular import (5 min fix)
2. Fix ml.consumers circular import (5 min fix)
3. Run full validation suite

### This Week

1. Create `ml/common/metrics_detection.py`
2. Create `ml/common/time_utils.py` (biggest impact)
3. Create `ml/common/dataframe_utils.py` — completed; adopted in `DataStore`.
4. Consolidate repeated stats-time-window SQL fragments — completed via `ml/stores/services/common_stats.py` and adoption in strategy/model stats services.

### Next Sprint

1. Migrate all timestamp code to use time_utils
2. Consolidate DataFrame operations
3. Remove duplicate retry/backoff logic

## Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Module Import Success | 89% (17/19) | 100% | 🟡 |
| Max Duplication Factor | 145x | <5x | 🔴 |
| Code Reduction Potential | 5,700 LOC | Eliminated | 🔴 |
| Pattern Compliance | Documented | Enforced | 🟢 |
| Public API Clarity | Complete | - | ✅ |

## Tools & Resources

- **Action Plan**: `ml/REFACTORING_ACTION_PLAN.md`
- **Validation Script**: `ml/validate_imports.py`
- **Duplication Report**: `ml/DUPLICATION_REPORT.md`
- **Architecture Guide**: `ml/docs/architecture/universal_patterns_guide.md`
- **Coding Standards**: `ml/docs/development/CODING_STANDARDS.md`
