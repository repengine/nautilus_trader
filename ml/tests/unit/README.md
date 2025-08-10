# Unit Tests

This directory contains unit tests for individual ML components, testing each component in isolation with mocked dependencies.

## Purpose

Unit tests verify that individual components work correctly in isolation. They provide fast feedback during development and ensure high code coverage.

## Test Organization

Tests are organized by component type into subdirectories:

### Component Directories

#### `actors/`
ML actor functionality:
- `test_base_actor.py` - Base ML actor functionality
- `test_inference_actor.py` - Actors that perform ML inference
- `test_signal_actor.py` - Actors that generate trading signals

#### `strategies/`
ML trading strategies:
- `test_base_strategy.py` - Base ML strategy functionality
- `test_signal_strategies.py` - Strategies that consume ML signals

#### `features/`
Feature engineering and validation:
- `test_feature_engineering.py` - Feature computation and transformation
- `test_feature_validation.py` - Feature validation and quality checks
- `feature_parity/` - Training/inference feature consistency tests

#### `config/`
Configuration classes and validation:
- `test_config_classes.py` - General ML configuration classes
- `test_ml_actor_config.py` - ML actor-specific configuration

#### `data/`
Data processing and loading:
- `test_data_loader.py` - Data loading functionality
- `test_data_structure.py` - ML-specific data structures

#### `infrastructure/`
Infrastructure components:
- `test_dashboard_factory.py` - Monitoring dashboard creation
- `test_grafana_client.py` - Grafana integration
- `test_optuna_optimizer.py` - Hyperparameter optimization

#### `meta/`
Test infrastructure itself:
- `test_fixtures.py` - Test fixtures and utilities
- `test_init.py` - ML module initialization

### Legacy Specialized Subdirectories

##### `collectors/`
Resource monitoring and data collection components:
- Base collector functionality
- Resource usage tracking
- Performance metrics collection

##### `registry/`
Model registry management:
- `test_registry_statistics.py` - Registry metrics and health
- `test_registry_canary.py` - Registry health checks and monitoring

##### `tracking/`
Experiment and model tracking:
- MLflow integration
- Model versioning
- Experiment metadata

##### `training/`
Training pipeline components:
- Individual trainer unit tests (XGBoost, LightGBM)
- Training configuration validation
- Model serialization/deserialization

## Coverage Requirements

- **Minimum Coverage**: 80% for all Python files
- **ML Module Coverage**: ≥90% (stricter requirement)
- **Critical Path Coverage**: 100% for hot path functions

## When to Add Unit Tests

Add unit tests when:
- Creating new classes or functions
- Adding configuration options
- Implementing data validation
- Building utility functions
- Creating base classes or interfaces

## Unit Test Best Practices

```python
class TestFeatureEngineer:
    def test_compute_features_with_valid_data_returns_expected_shape(self):
        """Test happy path with descriptive naming."""
        # Arrange
        engineer = FeatureEngineer(config=valid_config)
        bar_data = create_test_bars(count=100)
        
        # Act  
        features = engineer.compute_features(bar_data)
        
        # Assert
        assert features.shape == (100, expected_feature_count)
        
    def test_compute_features_with_insufficient_data_raises_value_error(self):
        """Test edge cases and error conditions."""
        engineer = FeatureEngineer(config=valid_config)
        bar_data = create_test_bars(count=5)  # Too few
        
        with pytest.raises(ValueError, match="Insufficient data"):
            engineer.compute_features(bar_data)
```

Unit tests form the foundation of our testing strategy - they should be fast, focused, and comprehensive.