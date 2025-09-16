# Test Refactoring Tranche Instructions

## Common Instructions for All Tranches

### Available Resources

1. **Refactoring Guide:** `ml/tests/REFACTORING_GUIDE.md`
2. **Fixtures:** `ml/tests/fixtures/common.py`
3. **Builders:** `ml/tests/builders.py`
4. **Validation Test:** `ml/tests/unit/test_fixture_validation.py` (shows correct usage)

### Key Field Names (CRITICAL - Use Correct Names)

- MLActorConfig: `warm_up_period` (NOT `warmup_bars`)
- MLActorConfig: `batch_size` (NOT `inference_batch_size`)
- MLStrategyConfig: `ml_signal_source` (NOT `signal_source`)
- ModelManifest: `feature_schema` (NOT `features`)
- FeatureManifest: `feature_names` (NOT `features`)

### Definition of Done

1. All previously green tests remain green
2. Duplicated setup code eliminated
3. Tests use fixtures/builders appropriately
4. No behavioral changes to test logic

---

## Tranche 1: Actor Unit Tests
**Files:** `ml/tests/unit/actors/` (21 files)
**Focus:** ML actor tests with heavy config duplication

### Specific Patterns

- Replace `BarType.from_str()` with `default_bar_type` fixture
- Replace `InstrumentId.from_str()` with `default_instrument_id` fixture
- Replace `MLActorConfig()` creation with `base_ml_config` fixture
- Replace `MLSignalActorConfig()` with `base_signal_config` fixture
- Use `MLConfigBuilder` for custom configs

---

## Tranche 2A: Basic Store Tests
**Files:**

```
ml/tests/unit/stores/test_feature_store_basic.py
ml/tests/unit/stores/test_model_store_basic.py
ml/tests/unit/stores/test_strategy_store_basic.py
ml/tests/unit/stores/test_data_store_basic.py
ml/tests/unit/stores/test_*_store_unit.py (all matching)
```

### Specific Patterns

- Replace config setup with fixtures
- Use `test_database` fixture consistently
- Replace mock store creation with `mock_stores_bundle`
- Use `MockBuilder.store_with_data()` for pre-populated stores

---

## Tranche 2B: Store Integration Tests
**Files:**

```
ml/tests/unit/stores/test_*_integration*.py
ml/tests/unit/stores/test_*_routing*.py
ml/tests/unit/stores/test_*_batch*.py
```

### Specific Patterns

- Use database fixtures properly (`test_database`, `database_session`)
- Replace mock registry setup with registry fixtures
- Clean up transaction handling
- Use `clean_postgres_db` fixture where needed

---

## Tranche 2C: Store Edge Cases
**Files:** All remaining files in `ml/tests/unit/stores/` not covered in 2A/2B

### Specific Patterns

- Apply patterns from 2A/2B
- Focus on performance and edge case tests
- Use `DataBuilder` for test data generation

---

## Tranche 3A: Core Registry Tests
**Files:**

```
ml/tests/unit/registry/test_model_registry*.py
ml/tests/unit/registry/test_feature_registry*.py
ml/tests/unit/registry/test_strategy_registry*.py
ml/tests/unit/registry/test_data_registry*.py
```

### Specific Patterns

- Use `RegistryBuilder.model_manifest()` for manifests
- Replace duplicate registry mocks with `mock_model_registry`
- Use `MockBuilder.all_registries()` for complete registry set
- Consolidate metadata creation

---

## Tranche 3B: Registry Contract Tests
**Files:**

```
ml/tests/unit/registry/test_*_contracts.py
ml/tests/unit/registry/test_*_conformance.py
ml/tests/unit/registry/test_unified_registry.py
```

### Specific Patterns

- Apply registry fixtures from 3A
- Focus on behavioral contracts
- Use manifest builders for test data

---

## Tranche 4A: Property Tests
**Files:** `ml/tests/property/` (all 16 files)

### Specific Patterns

- Use fixtures with Hypothesis strategies
- Replace hardcoded test data with `DataBuilder`
- Keep property definitions intact
- Focus on setup reduction

---

## Tranche 4B: Contract Tests
**Files:** `ml/tests/contracts/` (all 9 files)

### Specific Patterns

- Apply schema fixtures
- Use consistent test data from builders
- Focus on contract validation logic

---

## Tranche 5: Integration Tests
**Files:** `ml/tests/integration/` (all 16 files)

### Specific Patterns

- Apply all fixture patterns
- Use `clean_postgres_db_class` for class-scoped cleanup
- Focus on end-to-end test isolation
- Use real database fixtures where appropriate

---

## Tranche 6A: Observability Tests
**Files:** `ml/tests/unit/observability/` (19 files)

### Specific Patterns

- Apply fixture patterns for configs
- Clean up metric mocks
- Use `MockBuilder` for consistent mocks

---

## Tranche 6B: Common & Data Tests
**Files:**

```
ml/tests/unit/common/ (13 files)
ml/tests/unit/data/ (8 files)
```

### Specific Patterns

- Apply fixture patterns
- Use `DataBuilder` extensively for test data
- Focus on utility function tests

---

## Common Refactoring Example

### Before

```python
def test_something():
    # 20+ lines of setup
    bar_type = BarType.from_str("EUR/USD.SIM-1-MINUTE-MID-INTERNAL")
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")

    config = MLActorConfig(
        model_id="test_model",
        model_path="/tmp/model.onnx",
        bar_type=bar_type,
        instrument_id=instrument_id,
        batch_size=1,
        warm_up_period=10,
        prediction_threshold=0.5,
        use_dummy_stores=True,
    )

    mock_registry = MagicMock()
    mock_model_info = MagicMock()
    # ... lots of mock setup

    # Actual test logic (5 lines)
```

### After

```python
def test_something(base_ml_config, mock_model_registry):
    # Actual test logic (5 lines)
```

### For Custom Configs

```python
def test_custom():
    config = MLConfigBuilder.actor_config(
        model_id="custom_model",
        prediction_threshold=0.7
    )
    # Test logic
```

## Validation Command
After refactoring, validate with:

```bash
python -m pytest [test_file_or_directory] -xvs
```
