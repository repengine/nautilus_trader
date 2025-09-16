# ML Test Refactoring Guide

## Overview

This guide explains how to use the new test fixture system to eliminate duplication in ML tests. The fixtures and builders have been designed to match the actual data structures in the codebase.

## Available Fixtures

### Core Type Fixtures

```python
# Import from conftest or use as pytest fixtures
default_venue          # Venue("SIM")
default_instrument_id  # InstrumentId.from_str("EUR/USD.SIM")
default_bar_type      # BarType.from_str("EUR/USD.SIM-1-MINUTE-MID-INTERNAL")
test_component_id     # ComponentId("TEST-001")
```

### Configuration Fixtures

```python
base_ml_config        # MLActorConfig with all required fields
base_signal_config    # MLSignalActorConfig with signal-specific fields
base_feature_config   # MLFeatureConfig with feature engineering settings
model_registry_config # ModelRegistryConfig for registry testing
```

### Model Fixtures

```python
dummy_onnx_model     # Path to valid ONNX model file
dummy_xgboost_model  # Path to valid XGBoost model file
```

### Mock Fixtures

```python
mock_model_registry   # Fully configured model registry mock
mock_feature_registry # Fully configured feature registry mock
mock_data_store      # DataStore mock with common operations
mock_stores_bundle   # Dictionary with all store mocks
```

### Data Fixtures

```python
sample_features       # Dictionary of feature values
sample_predictions    # NumPy array of predictions
test_timestamps      # Tuple of (ts_event, ts_init) in nanoseconds
sample_model_manifest # ModelManifest instance
sample_feature_manifest # FeatureManifest instance
```

## Builder Classes

### MLConfigBuilder

```python
from ml.tests.builders import MLConfigBuilder

# Create configs with defaults and overrides
config = MLConfigBuilder.actor_config(model_id="custom_model")
signal_config = MLConfigBuilder.signal_config(prediction_threshold=0.7)
strategy_config = MLConfigBuilder.strategy_config(max_positions=5)
```

### MockBuilder

```python
from ml.tests.builders import MockBuilder

# Create fully configured mocks
registry = MockBuilder.model_registry(model_id="custom", version="2.0.0")
feature_reg = MockBuilder.feature_registry(feature_names=["custom_feature"])
all_registries = MockBuilder.all_registries()  # Get all 4 registries
```

### DataBuilder

```python
from ml.tests.builders import DataBuilder

# Generate test data
features = DataBuilder.feature_data(n_samples=100, n_features=10)
predictions = DataBuilder.predictions(n_samples=100, bounded=True)
ohlcv = DataBuilder.ohlcv_data(n_bars=100, as_dataframe=True)
signals = DataBuilder.signal_data(n_signals=10)
```

### RegistryBuilder

```python
from ml.tests.builders import RegistryBuilder

# Create manifest objects
model_manifest = RegistryBuilder.model_manifest(model_id="test_model")
feature_manifest = RegistryBuilder.feature_manifest(feature_set_id="test_features")
strategy_manifest = RegistryBuilder.strategy_manifest(strategy_id="test_strategy")
```

## Refactoring Examples

### Before (with duplication)

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
        # ... many more fields
    )

    mock_registry = MagicMock()
    mock_model_info = MagicMock()
    # ... lots of mock setup

    # Actual test logic (5 lines)
```

### After (using fixtures)

```python
def test_something(base_ml_config, mock_model_registry):
    # Actual test logic (5 lines)
    assert base_ml_config.model_id == "test_model"
```

### Using Builders for Custom Configs

```python
def test_custom_config():
    # Create config with specific overrides
    config = MLConfigBuilder.actor_config(
        model_id="custom_model",
        prediction_threshold=0.7
    )

    # All other fields have sensible defaults
    assert config.use_dummy_stores is True  # Safe default for testing
```

## Common Patterns to Replace

### 1. Replace String Parsing

```python
# OLD
bar_type = BarType.from_str("EUR/USD.SIM-1-MINUTE-MID-INTERNAL")
instrument_id = InstrumentId.from_str("EUR/USD.SIM")

# NEW - Use fixtures
def test_foo(default_bar_type, default_instrument_id):
    # Use directly
```

### 2. Replace Config Creation

```python
# OLD - 20+ lines of config
config = MLActorConfig(...)

# NEW - Use fixture or builder
def test_foo(base_ml_config):
    # Use directly

# OR for custom config
config = MLConfigBuilder.actor_config(model_id="custom")
```

### 3. Replace Mock Setup

```python
# OLD - 30+ lines of mock setup
mock_registry = MagicMock()
# ... lots of setup

# NEW - Use fixture
def test_foo(mock_model_registry):
    # Mock is fully configured
```

## Task Agent Instructions

When refactoring test files:

1. **Identify Duplication**
   - Look for `BarType.from_str()` calls
   - Look for `InstrumentId.from_str()` calls
   - Look for `MLActorConfig()` creation
   - Look for `MagicMock()` setup

2. **Replace with Fixtures**
   - Add fixtures to function parameters
   - Remove duplicated setup code
   - Use builders for custom configs

3. **Maintain Test Intent**
   - Keep the actual test logic unchanged
   - Only refactor the setup/boilerplate
   - Ensure tests still pass

4. **Verify Changes**
   - Run tests after refactoring
   - Check that coverage is maintained
   - Ensure no behavioral changes

## Important Notes

- All fixtures use **actual field names** from the real classes
- ModelManifest uses: `role`, `data_requirements`, `feature_schema` (not `features`)
- FeatureManifest uses: `feature_names` (not `features`)
- MLActorConfig uses: `warm_up_period` (not `warmup_bars`)
- MLStrategyConfig uses: `ml_signal_source` (not `signal_source`)

The fixtures have been validated against the actual codebase structures.
