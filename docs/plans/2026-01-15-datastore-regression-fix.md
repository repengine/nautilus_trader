# DataStore Regression Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore DataStore write-ingestion routing, event emission/watermark updates, and schema validation metrics to the pre-regression behavior.

**Architecture:** Keep DataWriterComponent responsible for write execution and event emission via EventEmitterComponent, with explicit watermark updates on success. DataStoreFacade remains a thin delegator and restores backward-compatible emit_dataset_event signature.

**Tech Stack:** Python 3.11+, pytest, mypy, ruff, Nautilus ML stores/common components.

### Task 1: Confirm failing routing/event tests (baseline)

**Files:**
- Test: `ml/tests/unit/stores/test_data_store_routing.py`
- Test: `ml/tests/unit/stores/test_data_store_raw_ingestion.py`

**Step 1: Run failing routing test**

```bash
pytest ml/tests/unit/stores/test_data_store_routing.py::test_routing_predictions_to_model_store -q
```
Expected: FAIL with "write_batch called 0 times".

**Step 2: Run raw ingestion partial test**

```bash
pytest ml/tests/unit/stores/test_data_store_raw_ingestion.py::test_raw_write_without_writer_emits_partial_and_no_watermark -q
```
Expected: FAIL due to success status or missing partial event emission.

### Task 2: Restore DataWriterComponent routing + event emission

**Files:**
- Modify: `ml/stores/common/data_writer.py`
- Modify: `ml/stores/data_store_facade.py`

**Step 1: Write failing test (existing)**

Use existing failing tests from Task 1 (no new test needed).

**Step 2: Implement minimal routing + event emission**

```python
# data_writer.py: add EventEmitterProtocol dependency,
# implement _write_to_feature_store/_write_to_model_store/_write_to_strategy_store
# use data_frame_to_* converters, emit_event via event_emitter,
# update watermark on success only.
```

**Step 3: Run the routing test**

```bash
pytest ml/tests/unit/stores/test_data_store_routing.py::test_routing_predictions_to_model_store -q
```
Expected: PASS.

### Task 3: Restore compatibility helpers + metrics

**Files:**
- Modify: `ml/stores/common/schema_validator.py`
- Modify: `ml/stores/feature_store_facade.py`
- Modify: `ml/stores/data_store_facade.py`
- Modify: `ml/core/integration_facade.py`

**Step 1: Write failing test (existing)**

Use existing tests:
- `ml/tests/unit/stores/test_data_store_validation.py::TestPrometheusMetrics::test_validation_metrics_emitted`
- `ml/tests/property/test_store_invariants.py::TestFeatureStoreInvariants::test_timestamp_monotonicity_invariant`
- `ml/tests/unit/core/test_integration_manager_types.py::TestMLIntegrationManagerTypeAnnotations::test_data_store_has_concrete_type_with_optional_none`

**Step 2: Implement minimal fixes**

```python
# schema_validator.py: add quality_score_histogram metric and observe quality_score
# feature_store_facade.py: export EngineManager alias for compatibility
# data_store_facade.py: import time to enable test patching; extend emit_dataset_event signature
# integration_facade.py: narrow DataStoreFacadeLike to DataStoreFacadeProtocol for typing test
```

**Step 3: Run targeted tests**

```bash
pytest ml/tests/unit/stores/test_data_store_validation.py::TestPrometheusMetrics::test_validation_metrics_emitted -q
pytest ml/tests/property/test_store_invariants.py::TestFeatureStoreInvariants::test_timestamp_monotonicity_invariant -q
pytest ml/tests/unit/core/test_integration_manager_types.py::TestMLIntegrationManagerTypeAnnotations::test_data_store_has_concrete_type_with_optional_none -q
```
Expected: PASS.

### Task 4: Update tests for new dependency requirements

**Files:**
- Modify: `ml/tests/unit/stores/test_data_store_event_error_unit.py`
- Modify: `ml/tests/unit/stores/test_data_store_failed_event_normalization_unit.py`
- Modify: `ml/tests/unit/stores/test_data_store_ingestion_json_unit.py`
- Modify: `ml/tests/unit/stores/test_data_store_ingestion_other_json_unit.py`
- Modify: `ml/tests/property/test_data_store_no_duplicate_bus.py`
- Modify: other DataStore tests missing earnings_store

**Step 1: Write failing test (existing)**

Use existing failing tests from the suite.

**Step 2: Implement minimal test adjustments**

```python
# ensure DataStore initialization passes DummyEarningsStore
# patch preflight helper to override DataWriterComponent validator
# remove object.__new__ usage in favor of real DataStore init
# fix indentation errors in property tests
```

**Step 3: Run targeted tests**

```bash
pytest ml/tests/unit/stores/test_data_store_event_error_unit.py -q
pytest ml/tests/unit/stores/test_data_store_failed_event_normalization_unit.py -q
pytest ml/tests/unit/stores/test_data_store_ingestion_json_unit.py -q
pytest ml/tests/unit/stores/test_data_store_ingestion_other_json_unit.py -q
pytest ml/tests/property/test_data_store_no_duplicate_bus.py -q
```
Expected: PASS.

### Task 5: Broad verification

**Files:**
- Test: `ml/tests/unit/stores`

**Step 1: Run shard**

```bash
pytest ml/tests/unit/stores -k data_store -q
```
Expected: PASS (or known environment skips).

**Step 2: Lint/type checks**

```bash
poetry run ruff check ml
poetry run mypy ml --strict
```
Expected: PASS.

**Step 3: Broader shard (per request)**

```bash
pytest ml/tests/unit -q
```
Expected: PASS (or known environment skips).
