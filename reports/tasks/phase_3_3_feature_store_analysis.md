# FeatureStore Decomposition Analysis - Phase 3.3

**Date:** 2025-10-13
**Status:** Analysis Complete
**File:** `ml/stores/feature_store.py`

## Current State

- **Total lines:** 1,677
- **Number of methods:** 31
- **Primary responsibilities:** Feature computation, storage, retrieval, and management
- **Dependencies:** SQLAlchemy, FeatureEngineer, DataRegistry, MessageBus, IndicatorManager

## Component Extraction Plan

### Component 1: FeaturePersistence
**Responsibility:** Write/upsert operations to database

**Methods:**
- `_execute_write` (lines 1292-1410) - Core upsert logic with circuit breaker
- `write_features` (lines 1094-1266) - Public write API (batch and single)
- `write_batch` (lines 1436-1505) - Batch write API
- `store_features` (lines 1648-1678) - Backward-compatible alias
- `_store_to_postgres` (lines 392-400) - Placeholder for test monkeypatching

**Estimated lines:** ~350

**Key features:**
- Upsert with conflict resolution
- Circuit breaker integration
- Message bus publishing
- Observability tracking

### Component 2: FeatureRetrieval
**Responsibility:** Query and load features from database

**Methods:**
- `get_training_data` (lines 730-809) - Load features for training
- `get_latest_at_or_before` (lines 1013-1062) - Point-in-time query
- `read_range` (lines 1536-1647) - Time range query
- `_load_bars_from_nautilus` (lines 811-881) - Load bars for computation
- `_features_exist` (lines 882-917) - Check feature existence
- `_execute_query` (lines 1411-1423) - SQL query execution (test hook)
- `_get_connection` (lines 1526-1531) - Connection manager (test hook)

**Estimated lines:** ~300

**Key features:**
- Efficient time-range queries
- Integration with Nautilus bar data
- Point-in-time lookups
- Test-friendly query hooks

### Component 3: FeatureComputation
**Responsibility:** Compute features (batch and realtime)

**Methods:**
- `compute_and_store_historical` (lines 418-543) - Batch computation
- `compute_realtime` (lines 544-729) - Online computation
- `compute_historical_parallel` (lines 252-317) - Parallel batch computation
- `_emit_historical_event` (lines 952-1012) - Event emission after computation

**Estimated lines:** ~450

**Key features:**
- Training/inference parity via FeatureEngineer
- Parallel processing for multiple instruments
- Indicator manager integration
- Event emission for observability

**Dependencies:**
- FeatureEngineer (computation logic)
- IndicatorManager (stateful indicators)
- FeaturePersistence (storage after computation)

### Component 4: FeatureVersioning
**Responsibility:** Feature set identification and versioning

**Methods:**
- `_compute_config_hash` (lines 402-417) - Hash feature configuration
- `_get_feature_set_id` (lines 941-951) - Derive feature set identifier
- `_get_feature_names` (lines 918-927) - Get offline feature names
- `_get_feature_names_online` (lines 928-940) - Get online feature names

**Estimated lines:** ~70

**Key features:**
- Stable hashing for feature set identification
- Pipeline signature integration
- Hot/cold path feature name management

### Component 5: FeatureTableManager
**Responsibility:** Database schema and table management

**Methods:**
- `_setup_tables` (lines 318-381) - Table creation/reflection
- `_normalize_ts_ns` (lines 383-390) - Timestamp normalization (static)
- `clear_features` (lines 1063-1093) - Delete features

**Estimated lines:** ~120

**Key features:**
- Partitioned table support
- Reflection for migrated tables
- Fallback table creation for dev/test

### Supporting Components

**Mixins (Already extracted):**
- `HealthMixin` - Health check implementation
- `BusPublisherMixin` - Message bus integration
- `DataRegistryMixin` - Registry integration

**Methods from mixins:**
- `is_healthy` (lines 1506-1525) - from HealthMixin
- `flush` (lines 1424-1434) - from BusPublisherMixin (no-op currently)
- `_get_data_registry` (lines 233-238) - from DataRegistryMixin
- `set_data_registry` (lines 240-251) - from DataRegistryMixin
- `_record_observability_stage_boundary` (lines 1267-1291) - observability helper

## Component Interactions

```
┌─────────────────────────────────────────────────────────────┐
│                      FeatureStore Facade                     │
│                  (Backward Compatible API)                   │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┬───────────┬──────────────┐
        │           │           │           │              │
┌───────▼──────┐ ┌─▼──────────┐ ┌──────────▼─┐ ┌──────────▼────┐
│  Feature     │ │  Feature    │ │  Feature   │ │  Feature      │
│ Persistence  │ │ Retrieval   │ │ Computation│ │ Versioning    │
└───────┬──────┘ └─────────────┘ └──────┬─────┘ └───────────────┘
        │                               │
        └───────────────┬───────────────┘
                        │
                ┌───────▼────────┐
                │ FeatureTable   │
                │   Manager      │
                └────────────────┘
```

**Data flow:**
1. **Write path:** Computation → Versioning → Persistence → TableManager
2. **Read path:** Facade → Retrieval → TableManager
3. **Training path:** Facade → Computation → Persistence → Retrieval

## Feature Flag Implementation

**Name:** `ML_USE_LEGACY_FEATURE_STORE`
**Default:** "0" (component mode)
**Values:**
- "0" = Component mode (new implementation)
- "1" = Legacy mode (original god class)

**Location:** `ml/stores/__init__.py`

```python
import os

if os.getenv("ML_USE_LEGACY_FEATURE_STORE", "0") == "1":
    from ml.stores.feature_store_legacy import FeatureStoreLegacy as FeatureStore
else:
    from ml.stores.feature_store import FeatureStore

__all__ = [
    "FeatureStore",
    # ... other exports
]
```

## Risk Assessment

**LOW RISK areas:**
- Table management (already abstracted)
- Versioning (pure functions)
- Health checks (mixin-based)

**MEDIUM RISK areas:**
- Persistence (circuit breaker, publishing)
- Retrieval (multiple query patterns)

**HIGH RISK areas:**
- Computation (FeatureEngineer integration, indicator managers)
- Training/inference parity

**Critical dependencies:**
- FeatureEngineer (external, already tested)
- IndicatorManager (stateful, complex)
- DataRegistry (for event emission)

## Lessons from Previous Phases

**From Phase 3.1 (TFTDatasetBuilder):**
- E2E tests found 5 bugs (2 CRITICAL) that 121 unit tests missed
- Column preservation bugs in target generation
- Augmenter integration issues

**From Phase 3.2 (DataRegistry):**
- E2E tests found 1 HIGH severity bug early
- Watermark monotonicity issues

**Apply to Phase 3.3:**
1. Test feature computation parity thoroughly (hot vs cold path)
2. Verify indicator state management across boundaries
3. Test event emission in both computation paths
4. Validate circuit breaker behavior
5. Test message bus integration

## E2E Test Scenarios (Minimum 12)

### Critical Scenarios
1. **Write and read features end-to-end** - Basic round-trip
2. **Historical batch computation** - Compute features for date range
3. **Realtime computation with indicators** - Online feature computation
4. **Training data loading** - Load features for model training
5. **Feature versioning** - Multiple feature sets, correct identification

### Integration Scenarios
6. **Parallel historical computation** - Multiple instruments concurrently
7. **Event emission and watermarks** - Verify DataRegistry integration
8. **Message bus publishing** - Verify bus events published correctly
9. **Circuit breaker activation** - Simulate DB failure, verify fallback
10. **Indicator state management** - Verify indicators update correctly

### Parity Scenarios
11. **Legacy vs component mode parity (write)** - Identical write behavior
12. **Legacy vs component mode parity (read)** - Identical read results
13. **Legacy vs component mode parity (compute)** - Identical features computed

### Edge Cases
14. **Empty result sets** - Handle no features gracefully
15. **Large batch operations** - Performance with 10K+ features
16. **Concurrent writes** - Thread safety

## Success Criteria

### Functional
- ✅ All public methods work identically in both modes
- ✅ Feature computation produces identical results
- ✅ Training/inference parity maintained
- ✅ Event emission works in both modes
- ✅ Circuit breaker functions correctly

### Performance
- ✅ Latency < 10% regression vs legacy
- ✅ Memory usage < 5% increase
- ✅ Throughput maintained or improved

### Quality
- ✅ Ruff check passes (0 violations)
- ✅ MyPy strict mode passes
- ✅ Test coverage ≥ 90% for components
- ✅ All E2E tests pass in both modes
- ✅ Parity verified (<1% difference)

## Implementation Order

1. **Extract FeatureTableManager** (lowest dependency)
2. **Extract FeatureVersioning** (no external dependencies)
3. **Extract FeaturePersistence** (depends on TableManager)
4. **Extract FeatureRetrieval** (depends on TableManager)
5. **Extract FeatureComputation** (depends on Persistence, Versioning)
6. **Create Facade** (coordinates all components)
7. **Implement Feature Flag** (in __init__.py)
8. **Create E2E Tests** (validate both modes)

## Estimated Effort

- Component extraction: 4-5 hours
- Facade creation: 1 hour
- E2E test suite: 3-4 hours
- Validation and fixes: 2-3 hours
- Documentation: 1 hour

**Total:** 11-14 hours (within 12-hour target)

## Next Steps

1. ✅ Create legacy backup (`feature_store_legacy.py`)
2. Extract FeatureTableManager first
3. Extract other components in order
4. Create facade
5. Implement feature flag
6. Write E2E tests
7. Run validations
8. Generate task report
