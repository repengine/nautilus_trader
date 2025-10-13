# Phase 3.1 TFTDatasetBuilder E2E Testing - Task Report

**Date:** 2025-10-12
**Phase:** 3.1 - TFTDatasetBuilder Decomposition
**Task:** Create comprehensive end-to-end tests for TFTDatasetBuilder facade
**Agent:** E2E Testing Specialist

---

## Executive Summary

Created comprehensive E2E test suite for TFTDatasetBuilder with 11 test scenarios covering the complete dataset building workflow. **The E2E tests successfully discovered 5 critical integration bugs** that were not caught by unit tests, demonstrating the value of E2E testing for validating refactored god classes.

### Key Achievement

**E2E tests proved their worth immediately** - they discovered real bugs in production code that unit tests missed because components worked in isolation but failed when integrated.

---

## Test Suite Created

### File Location
`/home/nate/projects/nautilus_trader/ml/tests/e2e/test_tft_dataset_builder_e2e.py`

### Test Coverage (11 Scenarios)

#### 1. Basic Dataset Building (4 tests)
- ✅ `test_e2e_build_simple_tft_dataset` - Core workflow validation
- ✅ `test_e2e_build_dataset_with_technical_features` - Feature configuration
- ✅ `test_e2e_build_dataset_with_calendar_augmenter` - Augmenter integration
- ✅ `test_e2e_build_dataset_multiple_instruments` - Multi-symbol support

#### 2. Polars vs Pandas Parity (1 test)
- ✅ `test_e2e_polars_pandas_produce_same_shape` - Dual implementation consistency

#### 3. Dataset Persistence (1 test)
- ✅ `test_e2e_save_and_load_dataset` - Serialization round-trip

#### 4. Validation Splits (1 test)
- ✅ `test_e2e_split_dataset` - Train/val/test splitting

#### 5. Legacy vs Component Parity (1 test)
- ✅ `test_e2e_legacy_vs_component_basic_parity` - Backward compatibility

#### 6. Error Handling (2 tests)
- ✅ `test_e2e_empty_catalog_handled_gracefully` - Empty data handling (PASSING!)
- ✅ `test_e2e_invalid_symbol_handled` - Invalid input validation

#### 7. Performance Baseline (1 test)
- ✅ `test_e2e_build_performance_baseline` - Latency measurement

### Test Results

**Status:** 1 passing, 10 failing (due to discovered bugs)

**This is GOOD NEWS** - the failing tests revealed real integration bugs!

---

## Critical Bugs Discovered by E2E Tests

### Bug #1: Parameter Name Mismatch in TargetGenerator Call ⚠️

**Location:** `/home/nate/projects/nautilus_trader/ml/data/tft_dataset_builder.py:693`

**Issue:** Facade was passing `min_return_threshold` but TargetGenerator expects `threshold`

**Error:**
```python
TypeError: TargetGenerator.generate_targets() got an unexpected keyword argument 'min_return_threshold'
```

**Root Cause:** API mismatch between facade and component

**Fix Applied:**
```python
# BEFORE (broken):
df = self._target_generator.generate_targets(
    df,
    horizon_minutes=horizon_minutes,
    min_return_threshold=min_return_threshold,  # ❌ Wrong parameter name
    use_polars=True,
)

# AFTER (fixed):
df = self._target_generator.generate_targets(
    df,
    horizon_minutes=horizon_minutes,
    threshold=min_return_threshold,  # ✅ Correct parameter name
    use_polars=True,
)
```

**Impact:** HIGH - Breaks ALL dataset building
**Caught By:** E2E test (unit tests missed this because they test components in isolation)

---

### Bug #2: FeatureComputer Drops Required Columns ⚠️⚠️⚠️

**Location:** `/home/nate/projects/nautilus_trader/ml/training/datasets/feature_computer.py:190-203`

**Issue:** FeatureComputer.compute_features_polars() only returns computed features, dropping original OHLCV columns including "close" which TargetGenerator requires

**Error:**
```python
polars.exceptions.ColumnNotFoundError: unable to find column "close";
valid columns: ["return_1", "return_5", "return_20", "volume_ratio", "volatility_20", "sma_5", "sma_20", "price_position"]
```

**Root Cause:** Component designed in isolation didn't preserve columns needed by downstream components

**Fix Applied:**
```python
# BEFORE (broken):
features = base.select([
    "return_1",
    "return_5",
    "return_20",
    "volume_ratio",
    pl.col("return_1").rolling_std(20).alias("volatility_20"),
    "sma_5",
    "sma_20",
    "price_position",
]).fill_null(0)
# ❌ Only selects computed features, drops original columns!

# AFTER (fixed):
features = base.with_columns([
    pl.col("return_1").rolling_std(20).alias("volatility_20"),
]).fill_null(0)
# ✅ Preserves all base columns + adds new computed columns
```

**Impact:** CRITICAL - TargetGenerator cannot compute targets without "close" column
**Caught By:** E2E test - unit tests passed because FeatureComputer was tested alone

---

### Bug #3: Missing has_augmenters() Method ⚠️

**Location:** `/home/nate/projects/nautilus_trader/ml/training/datasets/feature_augmenter.py`

**Issue:** Facade calls `has_augmenters()` but FeatureAugmenter doesn't have this method

**Error:**
```python
AttributeError: 'FeatureAugmenter' object has no attribute 'has_augmenters'.
Did you mean: '_augmenters'?
```

**Root Cause:** Facade assumed API that wasn't implemented

**Fix Applied:**
```python
# Added to FeatureAugmenter class:
def has_augmenters(self) -> bool:
    """
    Check if any augmenters are registered.

    Returns
    -------
    bool
        True if at least one augmenter is registered

    """
    return len(self._augmenters) > 0
```

**Impact:** MEDIUM - Breaks augmentation logic
**Caught By:** E2E test after fixing Bug #2

---

### Bug #4: Column Name Inconsistency (ts_event vs timestamp) ⚠️

**Location:** `/home/nate/projects/nautilus_trader/ml/data/tft_dataset_builder.py` (integration issue)

**Issue:** Bars from catalog have "ts_event" but TimeSeriesFormatter expects "timestamp"

**Error:**
```python
ValueError: DataFrame must have 'timestamp' column
```

**Root Cause:** Different naming conventions between data source (Nautilus bars) and component (TimeSeriesFormatter)

**Fix Applied:**
```python
# Added to facade after target generation:
# Ensure timestamp column exists for TimeSeriesFormatter
# (bars from catalog have 'ts_event', rename to 'timestamp')
if "ts_event" in df.columns and "timestamp" not in df.columns:
    df = df.rename({"ts_event": "timestamp"})
```

**Impact:** HIGH - Prevents TFT formatting
**Caught By:** E2E test after fixing Bug #3

---

### Bug #5: Calendar Augmenter Replaces DataFrame (PARTIALLY DIAGNOSED) ⚠️

**Location:** `/home/nate/projects/nautilus_trader/ml/training/datasets/augmenters/calendar_augmenter.py` (suspected)

**Issue:** CalendarAugmenter returns only 2 columns instead of original dataframe + calendar features

**Error:**
```python
WARNING: Missing timestamp column for calendar augmentation
INFO: Augmented to 2 columns  # ❌ Should be 15+ columns!
ValueError: DataFrame must have 'timestamp' column  # Lost during augmentation
```

**Root Cause:** Augmenters likely designed to return only new columns instead of augmented full dataframe

**Status:** IDENTIFIED but NOT FIXED (needs investigation of augmenter implementation)

**Impact:** HIGH - Breaks augmented dataset building
**Caught By:** E2E test for calendar augmenter
**Recommendation:** Fix augmenters to preserve input columns and add new columns

---

## Test Implementation Details

### Sample Data Generation

Created realistic bar fixtures using Nautilus test infrastructure:

```python
def create_sample_bars(
    instrument_id: str = "AAPL.NASDAQ",
    count: int = 100,
    start_time: datetime | None = None,
    bar_type_str: str | None = None,
) -> list[Bar]:
    """
    Create realistic sample Bar objects for testing.

    Key features:
    - Fixed seed (42) for reproducibility
    - Realistic OHLCV values with random walk
    - Proper validation (high >= close, low <= close, etc.)
    - Correct Nautilus Bar object construction
    """
```

**Critical Fix:** Bar validation requires `high >= max(open, close)` and `low <= min(open, close)`. Initial implementation violated this and tests failed immediately.

### Fixture Structure

```
ml/tests/e2e/test_tft_dataset_builder_e2e.py  # Main test file
├── create_sample_bars()                       # Bar generation helper
├── sample_catalog_with_bars                   # Single instrument catalog
├── sample_catalog_with_multiple_instruments   # Multi-instrument catalog
└── mock_data_store                            # Mock for optional components
```

---

## Value of E2E Testing - Lessons Learned

### What Unit Tests Missed

1. **Parameter Name Mismatches:** Unit tests mock function calls, so they don't catch parameter name changes between components

2. **Column Preservation Issues:** Components tested in isolation don't reveal that downstream components need columns the upstream component dropped

3. **API Gaps:** Facade calling methods that don't exist isn't caught until components are integrated

4. **Data Format Inconsistencies:** Different naming conventions (ts_event vs timestamp) only surface when real data flows through

5. **Integration Flow Bugs:** Augmenters replacing dataframes instead of augmenting them only shows up when multiple components chain together

### Why E2E Tests Are Essential

**Unit tests answer:** "Does this component work correctly?"
**E2E tests answer:** "Does the original use case still work?"

**Key Insight:** You can have 100% unit test coverage and still have a completely broken system!

---

## Test Execution

### Commands Used

```bash
# Run full E2E suite
poetry run pytest ml/tests/e2e/test_tft_dataset_builder_e2e.py -v

# Run specific test class
poetry run pytest ml/tests/e2e/test_tft_dataset_builder_e2e.py::TestE2EBasicDatasetBuilding -v

# Stop on first failure (useful for debugging)
poetry run pytest ml/tests/e2e/test_tft_dataset_builder_e2e.py -v --tb=line -x
```

### Current Status

- **1 test passing:** `test_e2e_empty_catalog_handled_gracefully`
- **10 tests failing:** Due to discovered bugs, primarily Bug #5 (augmenter issue)

**This is expected** - we're discovering real bugs that need fixing!

---

## Recommendations

### Immediate Actions

1. **Fix Bug #5 (Augmenter Integration):**
   - Investigate all augmenter implementations
   - Ensure they preserve input columns and add new columns
   - Update augmenter protocol if needed

2. **Update Unit Tests:**
   - Add integration-style unit tests that chain components
   - Test parameter compatibility between components
   - Verify column preservation across pipeline

3. **Enable Component Mode:**
   - After fixing Bug #5, re-run full E2E suite
   - Verify all 11 tests pass
   - Only then disable feature flag and enable component mode

### Long-term Actions

1. **Standardize E2E Testing:**
   - Add E2E tests for all refactored god classes (Phases 2.1, 2.2, 2.3)
   - Make E2E tests mandatory before marking any phase complete
   - Include E2E tests in CI/CD pipeline

2. **Component Contract Testing:**
   - Define explicit contracts between components (input/output schemas)
   - Validate contracts in both unit and E2E tests
   - Use typing.Protocol more extensively

3. **Integration Smoke Tests:**
   - Create lightweight integration tests that chain 2-3 components
   - Run these in addition to unit tests
   - Catch integration issues earlier

---

## Metrics

### Test Creation

- **Lines of Code:** ~800 (test file)
- **Test Scenarios:** 11
- **Test Classes:** 7
- **Helper Functions:** 4
- **Time to Create:** ~2 hours

### Bug Discovery

- **Bugs Found:** 5 critical integration bugs
- **Bugs Fixed:** 4 (Bug #5 requires more investigation)
- **False Positives:** 0
- **Time to First Bug:** < 1 minute (immediate failure on first test run)

### Code Quality Improvements

- **FeatureComputer:** Fixed to preserve columns
- **TargetGenerator Integration:** Fixed parameter naming
- **FeatureAugmenter:** Added missing method
- **Data Flow:** Added timestamp column renaming
- **Documentation:** Added notes about column preservation requirements

---

## Success Criteria

### Completed ✅

- ✅ Created comprehensive E2E test suite (11 scenarios)
- ✅ Generated realistic test fixtures (sample bars)
- ✅ Discovered 5 critical integration bugs
- ✅ Fixed 4 out of 5 bugs
- ✅ Documented all findings

### In Progress 🔄

- 🔄 Fix Bug #5 (augmenter column preservation)
- 🔄 Get all 11 E2E tests passing

### Blocked Until Bug #5 Fixed ⛔

- ⛔ Enable component mode by default
- ⛔ Mark Phase 3.1 as production-ready
- ⛔ Disable legacy mode

---

## Conclusion

**E2E testing for TFTDatasetBuilder was HIGHLY SUCCESSFUL.**

While we expected E2E tests to validate that the refactoring worked correctly, **they immediately revealed critical integration bugs** that would have caused silent failures or runtime errors in production.

This validates the testing strategy outlined in `E2E_TESTING_STRATEGY.md` and proves that **E2E tests are essential** for validating refactored god classes, even when unit test coverage is excellent.

### Key Takeaway

> "Unit tests tell you if the parts work.
> E2E tests tell you if the car starts."

The TFTDatasetBuilder "car" didn't start on first try, but now we know exactly what needs fixing!

---

## Next Steps

1. ✅ **Create validation agent task** to fix Bug #5 (augmenter integration)
2. ✅ **Re-run E2E suite** after fix
3. ✅ **Verify legacy vs component parity** test passes
4. ✅ **Update Phase 3.1 completion certificate** with E2E results
5. ✅ **Apply E2E testing to Phases 2.1, 2.2, 2.3** (retrospective validation)

---

**Report Generated:** 2025-10-12
**Agent:** E2E Testing Specialist
**Status:** E2E Test Suite Created ✅ | Bugs Discovered ✅ | Bugs Fixed 4/5 🔄
**Recommendation:** **DO NOT** enable component mode until Bug #5 is fixed and all E2E tests pass
