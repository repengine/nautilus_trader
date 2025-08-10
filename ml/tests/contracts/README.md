# Contract Tests

This directory contains behavioral contract tests that verify ML components conform to expected interfaces and behaviors.

## Purpose

Contract tests define **what** components must do, not **how** they do it. They ensure that all implementations of a given interface behave consistently and correctly.

## Test Categories

### Actor Contracts (`test_actor_contracts.py`)
- MLSignal publication on market data
- Model identification in signals
- Multi-instrument handling
- Graceful failure recovery
- Lifecycle management (start, stop, reset)

### Model Contracts (`test_model_contracts.py`)
- Prediction interface compliance
- Model serialization/deserialization
- Feature dimension compatibility
- Error handling for invalid inputs
- Performance guarantees

### Strategy Contracts (`test_strategy_contracts.py`)
- Signal processing behavior
- Position management
- Risk controls activation
- State persistence across restarts

### Training Contracts (`test_training_contracts.py`)
- Data preprocessing pipelines
- Model training workflows
- Validation procedures
- Model persistence formats

### Registry Contracts (`test_registry_contracts.py`)
- Model versioning behavior
- Metadata management
- Hot-swap capabilities
- Rollback procedures

## When to Add Contract Tests

Add contract tests when:
- Defining a new interface that multiple components will implement
- Creating base classes with expected behaviors
- Establishing performance or reliability guarantees
- Ensuring compatibility across different implementations

## Examples

```python
def test_actor_publishes_signal_on_bar():
    """Contract: Actor MUST publish MLSignal when receiving bar data."""
    # Test ensures ANY actor implementation follows this rule
    
def test_model_prediction_dimensions():
    """Contract: Model predictions MUST match expected output shape."""
    # Test ensures ANY model follows the interface
```

These tests protect against breaking changes and ensure system reliability.