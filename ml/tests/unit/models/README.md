# Unit Tests for ML Models

This directory contains unit tests for the ML models package.

## Test Files

### `test_base_model.py`
Unit tests for the BaseModel abstract base class, covering:

- Abstract class enforcement (cannot instantiate directly)
- Subclass implementation requirements (must implement predict method)
- Input validation functionality
- Metadata property access and management
- Type safety enforcement (float32 dtype)
- Edge case handling (NaN, infinity, empty arrays)
- Model ID property derivation
- Error message formatting

The tests focus on behavioral contracts that all ML model implementations must follow.

## Running Tests

```bash
# Run all model unit tests
pytest ml/tests/unit/models/ -v

# Run specific test file
pytest ml/tests/unit/models/test_base_model.py -v

# Run with coverage
pytest ml/tests/unit/models/ --cov=ml.models --cov-report=term-missing
```

## Test Structure

Tests follow the project's testing protocol:
- Focus on behavior, not implementation
- Test edge cases and error conditions
- Use appropriate type annotations
- Follow naming convention: `test_{what}_with_{condition}_{expected_outcome}`
