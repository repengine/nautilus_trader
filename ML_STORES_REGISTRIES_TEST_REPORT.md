# ML Stores and Registries Test Report

## Executive Summary

I tested the claimed "mandatory 4-store + 4-registry integration" in the Nautilus Trader ML system and found **significant gaps between documentation and actual implementation**. While the foundational components exist and some functionality works, there are several critical failures, inconsistencies, and incomplete features.

## Test Methodology

- Created comprehensive test scripts to verify each store and registry
- Tested both PostgreSQL and fallback scenarios  
- Attempted to initialize BaseMLInferenceActor with various configurations
- Tested CRUD operations on each component
- Verified database schema existence
- Tested protocol conformance

## Results Summary

✅ **What Actually Works:**
- Database connection and schema (ml_feature_values, ml_model_predictions, ml_strategy_signals tables exist)
- FeatureStore, ModelStore basic initialization with PostgreSQL
- All 4 registries (FeatureRegistry, ModelRegistry, StrategyRegistry, DataRegistry) initialize
- DataStore initialization with DataRegistry
- DummyStore and DummyRegistry fallback implementations
- ModelStore write_prediction() operations
- Progressive fallback detection in BaseMLInferenceActor

❌ **What's Broken or Missing:**

### 1. MLActorConfig Incompatibility
**Problem:** BaseMLInferenceActor expects `db_connection` and `use_dummy_stores` fields on MLActorConfig, but these **don't exist** in the actual MLActorConfig class.

```python
# BaseMLInferenceActor code expects:
db_connection = getattr(self._config, "db_connection", None)
use_dummy_stores = getattr(self._config, "use_dummy_stores", False)

# But MLActorConfig doesn't have these fields!
```

### 2. FeatureStore SQLAlchemy Bug
**Problem:** Critical bug in `_execute_write()` method with inconsistent column access patterns:

```python
# Line 1185 - BROKEN - uses dict-style access
"values": stmt.excluded["values"],

# Other parts correctly use attribute access:  
"values": stmt.excluded.values,
```

**Error:** `KeyError: 'values'` when attempting to write features.

### 3. StrategyStore API Inconsistency  
**Problem:** The `write_signal()` method signature doesn't match the expected parameters from documentation:

```python
# Expected based on documentation:
store.write_signal(confidence=0.9, ...)

# Actual signature:
def write_signal(self, strategy_id, instrument_id, signal_type, strength, 
                 model_predictions, risk_metrics, execution_params, ts_event, ...)
```

**Error:** `TypeError: StrategyStore.write_signal() got an unexpected keyword argument 'confidence'`

### 4. Protocol Conformance Issues
**Problem:** Stores do not properly conform to their declared protocols:
- `FeatureStore` does not conform to `FeatureStoreProtocol`  
- Protocol checking failed during runtime

### 5. Documentation-Code Mismatch
**Problem:** The CLAUDE.md documentation claims:

> "All ML actors MUST use the four required stores + four registries"
> "These stores and registries are initialized automatically in BaseMLInferenceActor"

**Reality:** 
- MLActorConfig lacks the required fields
- BaseMLInferenceActor cannot be instantiated without manual configuration patching
- No clear way to configure database connections in standard MLActorConfig

## Detailed Test Results

### Store Initialization Results

| Store | PostgreSQL Init | Health Check | Write Operations | Notes |
|-------|---------------|--------------|------------------|--------|
| FeatureStore | ✅ | ✅ | ❌ | SQLAlchemy bug prevents writes |
| ModelStore | ✅ | ✅ | ✅ | Fully functional |
| StrategyStore | ✅ | ✅ | ❌ | API signature mismatch |
| DataStore | ✅ | ❌ (not tested) | ❌ (not tested) | Basic init works |

### Registry Initialization Results

| Registry | File-based | PostgreSQL | Notes |
|----------|------------|------------|-------|
| FeatureRegistry | ✅ | ✅ | Working |
| ModelRegistry | ✅ | ✅ | Working |  
| StrategyRegistry | ✅ | ✅ | Working |
| DataRegistry | ✅ | ✅ | Working |

### Database Schema Status

✅ **Database tables exist:**
- `ml_feature_values`
- `ml_model_predictions`  
- `ml_strategy_signals`
- `ml_strategy_performance`

✅ **Migration system works** - tables are properly created and partitioned.

### Progressive Fallback Status

✅ **Partial Success:**
- DummyStore and DummyRegistry implementations exist
- BaseMLInferenceActor detects PostgreSQL unavailability
- Automatic fallback logic is implemented

❌ **Configuration Gap:** No standard way to configure fallback behavior through MLActorConfig.

## Critical Issues for Production Use

### 1. **Cannot Create Working ML Actors**
The "mandatory 4-store + 4-registry integration" cannot be used in practice because:
- MLActorConfig is incompatible with BaseMLInferenceActor expectations
- No documented way to configure database connections
- Test examples in documentation would fail

### 2. **Data Loss Risk**  
The FeatureStore bug means:
- Features cannot be persisted to database
- Training/inference parity is broken
- Silent failures possible in production

### 3. **API Inconsistencies**
- StrategyStore API doesn't match documented interface
- Different stores have different initialization patterns
- Protocol conformance is not enforced

## Recommendations

### Immediate Fixes Required

1. **Fix FeatureStore SQLAlchemy Bug**
   ```python
   # Change line 1185 from:
   "values": stmt.excluded["values"],
   # To:
   "values": stmt.excluded.values,
   ```

2. **Add Missing MLActorConfig Fields**
   ```python
   class MLActorConfig(NautilusConfig, kw_only=True, frozen=True):
       # ... existing fields ...
       db_connection: str | None = None
       use_dummy_stores: bool = False
   ```

3. **Fix StrategyStore API**
   - Either update `write_signal()` to accept `confidence` parameter
   - Or update documentation to match actual API

### Architectural Improvements

1. **Protocol Enforcement**
   - Add runtime checks to ensure stores conform to protocols
   - Use `isinstance()` checks or formal protocol validation

2. **Configuration Validation**  
   - Add validation in BaseMLInferenceActor `__init__` to check for required config fields
   - Provide clear error messages when configuration is invalid

3. **Integration Testing**
   - Add end-to-end tests that actually instantiate BaseMLInferenceActor
   - Test real workflows rather than individual components

4. **Documentation Updates**
   - Fix examples in CLAUDE.md to use correct configuration
   - Document actual API signatures
   - Provide working code samples

## Conclusion

The ML stores and registries system has a **solid foundation** but is **currently unusable** due to critical implementation gaps. The "mandatory 4-store + 4-registry integration" is more aspirational than functional at present.

**Priority:** HIGH - These issues prevent the ML system from being used in production and represent fundamental integration failures.

**Effort Required:** MEDIUM - Most issues are configuration and API consistency problems that can be fixed with targeted changes rather than major refactoring.

The database schema, fallback mechanisms, and core store logic appear sound. The main issues are in the integration layer between components and configuration management.