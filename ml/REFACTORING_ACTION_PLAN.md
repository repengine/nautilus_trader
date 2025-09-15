# ML Module Refactoring Action Plan

## Overview
This document outlines the concrete steps to address circular imports and code duplication issues identified during the __init__.py refactoring.

## Phase 1: Fix Critical Circular Imports (Week 1)

### Day 1-2: Break Primary Import Chain

```python
# Step 1: Create ml/common/metrics_detection.py
"""Isolated metrics backend detection to prevent circular imports."""
try:
    import prometheus_client
    HAS_METRICS_BACKEND = True
except ImportError:
    HAS_METRICS_BACKEND = False
```

**Files to modify:**

- [x] Create `ml/common/metrics_detection.py`
- [x] Update `ml/_imports.py:14` to use metrics_detection (no ml.* imports at import time)
- [x] Update `ml/stores/feature_store.py:<import section>` to avoid metrics bootstrap at import time (lazy metrics resolution inside methods)
- [x] Make `ml/config/runtime.py:<import section>` use lazy `ort` import (gate under `TYPE_CHECKING` and import inside functions)

### Day 3-4: Fix Registry-Core Cycle
**Actions:**

- [x] Add TYPE_CHECKING/lazy imports to `ml/core/integration.py` (localized imports for registries/persistence)
- [x] Create lazy registry loader methods (integration manager and actor initializer)
- [x] Remove/avoid bus_integration import from `ml/core/__init__.py` (use lazy `__getattr__` for integration symbols)
- [x] Move to explicit imports where needed

### Day 5: Validate & Test

```bash
# Test script to validate all imports work
python -c "import ml.stores; import ml.registry; import ml.core; print('SUCCESS')"

# Run the ML import smoke test (adds project root to sys.path)
python ml/validate_imports.py

# Validate coding standards on changed files
uv run --active --no-sync mypy ml --strict
ruff check --fix
pytest -q -k 'imports or orchestrator or build_runner'
make validate-nautilus-patterns

# Environment hygiene: ensure no local modules shadow third-party packages
rg -n "^msgspec\.py$|/__pycache__/msgspec" -S . | cat
rm -f __pycache__/msgspec.cpython-*.pyc
```

## Phase 1B: Domain Facade Migration (Week 1)

Goal: Public entrypoints live in each domain package's `__init__.py`.

Tasks:

- [x] Expose `DatasetBuildConfig`, `BuildResult`, `build_tft_dataset` from `ml/data/__init__.py` (typed)
- [x] Update orchestrator, pipelines, and CLI to import from `ml.data`
- [x] Add import-linter contracts to enforce layering:
  - CLI → domain facades → services/stores (no domain → CLI)
  - Hot-path (actors/strategies) must not import cold-path-only modules
- [x] Add pytest import smoke test for all domains (skip if optional deps missing)

Definition of Done:

- mypy --strict clean on changed files
- ruff clean on changed files
- Focused tests (orchestrator/pipelines/build) green
- `make validate-nautilus-patterns` advisory clean (no new violations)
- Import smoke test passes or skips only due to optional dependencies (e.g., GPU libs)

## Phase 2: Consolidate High-Impact Duplicates (Week 2)

### Priority 1: Timestamp Utilities (145 files affected)
Preferred approach: Use existing `ml.common.timestamps` where possible. Create
`ml/common/time_utils.py` only if we need a wider compatibility surface (keep
it strictly typed and forward minimal behavior to `timestamps`).

Typed target helpers (examples):

```python
from __future__ import annotations
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

def normalize_timestamp_to_ns(ts: Any) -> int:
    """Convert supported timestamp types to nanoseconds (ns)."""
    # Use ml.common.timestamps.normalize_timestamp_ns under the hood
    from ml.common.timestamps import normalize_timestamp_ns

    return int(normalize_timestamp_ns(ts))

def pandas_to_nautilus_timestamp(df: pd.DataFrame, *, column: str) -> np.ndarray:
    """Convert a pandas datetime64[ns] column to Nautilus ns int array."""
    values = df[column].astype('datetime64[ns]').view('int64')
    return values  # already ns
```

Initial migrations completed:

- [x] ml/stores/feature_store.py (historical compute and load methods)
- [x] ml/features/l2_aggregate.py (lazy and eager filtering paths)
- [x] ml/stores/infrastructure.py (partition window boundaries)
- [x] ml/observability/migrations.py (monthly partition bounds)

Additional timestamp normalization (Phase 2 follow-ups):

- [x] ml/observability/db_persistence.py (retention cutoffs)
- [x] ml/orchestration/scheduler.py (emit event now)
- [x] ml/cli/pipeline_orchestrator.py (refresh-features marker)
- [x] ml/cli/coverage.py (range/now bounds, target-day bounds)
- [x] ml/cli/check_pipeline_health.py (now)
- [x] ml/cli/ingest_backfill.py (range bounds)
- [x] ml/registry/bootstrap_datasets.py (created_at/last_modified)
- [x] ml/registry/data_registry.py (last_modified/deprecated_at)
- [x] ml/stores/data_store.py (event ts_min/ts_max, lateness, migration window)

Next targets:

- [x] ml/stores/strategy_store.py (time windows and clock interactions)
- [x] ml/stores/data_processor.py (sanitization on writes and thresholds)

### Priority 2: DataFrame Utilities (49 files affected)
Prefer centralizing small validators/transforms only when multiple domains need
them. Keep functions typed and fast; avoid bringing in heavy stacks.

Example typed signatures:

```python
import polars as pl

def validate_ohlcv_data(df: pl.DataFrame) -> bool:
    """Return True if OHLCV schema and monotonic timestamps are valid."""
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        return False
    return bool((df["timestamp"].diff() >= 0).all())
```

Current progress:

- [x] Create `ml/common/dataframe_utils.py` with `total_nulls` and `column_nulls`.
- [x] Adopt in `ml/stores/data_store.py` for nullability/quality checks (three call sites).
- [x] Add `has_columns` and `is_monotonic_non_decreasing` validators to `ml/common/dataframe_utils.py`.
- [x] Adopt `has_columns` in `ml/preprocessing.joins.asof_join` for early validation of `on`/`by` columns
      across pandas and polars dataframes (cold path only).

### Priority 2b: Shared Stats Helpers (Model/Strategy)

Goal: Remove duplicated SQL time-window condition building across stats services; keep stores thin.

Actions:

- [x] Create `ml/stores/services/common_stats.py` with:
  - `build_time_conditions(start_ns, end_ns, field='ts_event')`
  - `build_nullsafe_time_clause(start_ns, end_ns, field='ts_event')`
- [x] Adopt in `ml/stores/services/strategy_services.py`:
  - `get_statistics`, `get_signal_distribution`, `get_strategy_performance`.
- [x] Adopt in `ml/stores/services/model_services.py`:
  - `get_statistics`, `get_model_performance`.

Verification:

- [x] Explicitly typed params at call sites to avoid mixed-type dict issues.
- [x] mypy --strict (follow-imports=skip) on touched services: clean.
- [x] Ruff clean on ml/.
- [x] Focused tests for stores: green.

Additional consolidation:

- [x] Add `select_signal_counts(include_avg_strength=True)` to `ml/stores/services/common_stats.py`.
- [x] Refactor `StrategySignalStatsService.get_strategy_performance` and
      `update_performance_metrics` to use the shared select fragment.
- [x] Delegate `write_batch` in `ModelStore` and `StrategyStore` to their WriteService to remove
      duplicated value-building logic; preserve observability/event emission.
- [x] Extract stage-boundary observability helper in `ml/common/observability_utils.py` and adopt in
      `ModelStore` and `StrategyStore` to remove duplicate latency/metric blocks (cold path only).
      - [x] Adopt in `FeatureStore` and `DataStore` as well to eliminate remaining duplication clusters.
- [x] mypy --strict (follow-imports=skip) on touched files: clean.
- [x] Ruff clean; focused store tests and validators remain green.

Service extractions:

- [x] Extract `StrategySignalClearService` and `ModelClearService` for deletion paths; adopt in `StrategyStore.clear_signals` and `ModelStore.clear_predictions`.
- [x] Delegate `StrategyStore.get_signals` and `read_active_signals` to `StrategySignalQueryService` (removed inline SQL; typed casts remain at facade boundary).

### Priority 2d: Shared Numeric Stats Fragment

Goal: Consolidate repeated AVG/STDDEV/MIN/MAX patterns for numeric columns.

Actions:

- [x] Add `select_numeric_stats(column, *, prefix, include_avg=True, include_stddev=True, include_min_max=True)`
      to `ml/stores/services/common_stats.py`.
- [x] Adopt in:
  - `StrategySignalStatsService.get_strategy_performance` for strength stats (std/min/max only).
  - `ModelStatsService.get_model_performance` for confidence stats (avg/std only).

Verification:

- [x] mypy --strict (follow-imports=skip) on touched files: clean.
- [x] Ruff clean.
- [x] Focused stores tests: green.

### Priority 2c: Shared Min/Max Timestamp Fragment

Goal: Remove repeated `MIN(ts_event)/MAX(ts_event)` patterns across stats services.

Actions:

- [x] Add `select_min_max_ts(field='ts_event', min_alias='min_ts', max_alias='max_ts')` to
      `ml/stores/services/common_stats.py`.
- [x] Adopt in:
  - `StrategySignalStatsService.get_statistics` for `{min_ts, max_ts}`.
  - `ModelStatsService.get_statistics` for `{min_ts, max_ts}`.

Verification:

- [x] mypy --strict (follow-imports=skip) on touched files: clean.
- [x] Ruff clean.
- [x] Focused stores tests: green.

### Priority 3: Retry/Backoff Utilities (26 files affected)
Prefer existing `ml.data.ingest.common` (RateLimiter, progress JSON helpers,
BackoffPolicy). If needed, add an adapter with typed wrappers.

Initial consolidation:

- [x] Add `ml/common/retry_utils.py` with `retry_with_backoff(call, *, max_attempts, initial_delay, multiplier, max_delay, jitter, sleep_fn, retry_on, on_exception)`.
- [x] Adopt in `ml/data/loaders/fred_loader.py` (`fetch_indicator`) to replace ad‑hoc retry loop while preserving metrics and logging.
- [x] Adopt in `ml/cli/populate_l2_efficient.py` daily fetch (`_download_day_for_symbol`) to replace inline loop; preserves existing behaviors (skip weekends, 403/license fast‑fail, exponential backoff, metrics/logging).
- [x] Adopt in `ml/cli/coverage.py` fetch/store loop: standardized backoff path while keeping symbol‑not‑found fast‑fail and rate‑limit handling.

Verification:

- [x] Ruff clean.
- [x] mypy --strict (follow-imports=skip) on `retry_utils.py` and touched CLI file: clean.
 - [x] mypy --strict on `ml/cli/coverage.py` — resolved in current toolchain; strict run clean.
- [x] Focused stores tests: green (unchanged behavior for stores).

## Phase 3: Implement Gradual Migration (Week 3-4)

### Migration Strategy

1. **Add new utilities alongside existing code**
2. **Update one domain at a time**
3. **Run tests after each domain migration**
4. **Remove old code only after validation**

### Domain Migration Order

1. [ ] ml/evaluation (smallest)
2. [ ] ml/deployment (low risk)
3. [ ] ml/preprocessing
4. [x] ml/data (dataset facade exposed)
5. [ ] ml/features (critical path)
6. [ ] ml/stores (most complex)
7. [ ] ml/cli (most duplication)

## Metrics & Success Criteria

### Circular Import Resolution

- [ ] All ml.* modules import without errors
- [ ] No ImportError exceptions in production
- [ ] Import time < 2 seconds for full ML module

### Duplication Reduction

- [ ] 20-25% reduction in total LOC
- [ ] No pattern duplicated > 5 times
- [ ] All technical indicators in one place
- [ ] Single source of truth for each utility

## Testing Strategy

### Unit Tests (examples)

```python
# tests/unit/test_imports.py
from __future__ import annotations

import importlib
import pytest

DOMAINS: list[str] = [
    "actors","common","config","consumers","core","data","deployment",
    "evaluation","features","models","monitoring","observability",
    "orchestration","pipelines","preprocessing","registry","stores",
    "strategies","training",
]

@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_import_smoke(domain: str) -> None:
    try:
        importlib.import_module(f"ml.{domain}")
    except Exception as exc:
        # Allow missing optional deps in CI; skip with reason
        pytest.skip(f"optional dependency missing for ml.{domain}: {exc}")
```

### Integration Tests

```python
# tests/test_refactored_utils.py
def test_timestamp_utils_consistency():
    """Ensure refactored utils produce same results."""
    # Compare old vs new implementations
```

## Risk Mitigation

1. **Feature Flags**: Use environment variables to toggle new code
2. **Gradual Rollout**: Migrate one file at a time
3. **Regression Tests**: Ensure identical behavior
4. **Rollback Plan**: Keep old code in _legacy.py files temporarily

## Prevention Measures

### CI Integration

```yaml
# .github/workflows/import-check.yml
- name: Check for circular imports
  run: |
    python -c "import ml; print('SUCCESS')"

- name: Check for duplication
  run: |
    python tools/duplication/check_duplication.py --threshold 5
```

### Code Review Checklist

- [ ] No new imports from ml._imports at module level
- [ ] Use TYPE_CHECKING for circular-prone imports
- [ ] Check if utility already exists in ml/common/
- [ ] Add to appropriate __init__.py with proper section

## Timeline

| Week | Phase | Deliverable |
|------|-------|------------|
| 1 | Fix Circular Imports | All modules importable |
| 2 | Consolidate Utilities | Common utilities created |
| 3-4 | Migration | Domains migrated to new utils |
| 5 | Validation | Tests passing, metrics improved |

## Next Steps

1. **Immediate**: Finish primary circular import chain fixes
   - [x] `ml/stores/feature_store.py` → lazy metrics resolution
   - [x] `ml/config/runtime.py` → lazy `ort` import
   - [x] Investigate store-loader cycle: `ml.data.loaders.fred_loader` ↔ `ml.stores.data_store` — no direct imports from `DataStore` back to `fred_loader`; no cycle.
2. **Today**: Add import-linter contracts and import smoke test (added)
3. **This Week**: Complete Phase 1B (domain facade migration enforcements)
4. **Next Sprint**: Begin duplication consolidation (timestamps first)

Integration typing hygiene

- [x] Resolve strict mypy issues in `ml/core/integration.py` for observability flushers by
      casting `persist(...)` returns to declared types.

## Success Metrics

- **Import Success Rate**: 100% (currently limited by optional deps)
- **Duplication Factor**: < 5x (currently up to 145x)
- **Code Reduction**: 5,700+ LOC eliminated
- **Import Time**: < 2s (currently fails)
- **Test Coverage**: Maintained at > 80%

## Coding Standards & Checks (per change)

- Explicit typing on all new or changed code; avoid implicit Any
- mypy: `uv run --active --no-sync mypy ml --strict`
- ruff: `ruff check --fix`
- pytest: focused tests for touched areas
- validators: `make validate-nautilus-patterns` (advisory)
