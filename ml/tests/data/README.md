# Test Data Directory

This directory contains test data for the ML module tests. All test data should be placed here to keep production code clean and maintainable.

## Structure

```
ml/tests/data/
├── model_registry/          # Test model registry data
│   ├── registry.json       # Registry metadata for testing
│   └── models/            # Test model files
│       ├── xgb_v1.json    # XGBoost v1 test model
│       └── xgb_v2.json    # XGBoost v2 test model
└── model_registry_rollout/  # Test data for rollout scenarios
    ├── registry.json       # Rollout registry metadata
    └── models/            # ONNX test models
        ├── prod.onnx      # Production ONNX model
        └── new.onnx       # New ONNX model for testing

```

## Usage

### In Tests

Use the provided fixtures from `ml/tests/conftest.py`:

```python
def test_with_model_registry(model_registry_dir: Path):
    """Test using model registry directory."""
    registry_path = model_registry_dir / "registry.json"
    assert registry_path.exists()
```

### Direct Access

For direct access, use the helper functions from `ml/tests/data/__init__.py`:

```python
from ml.tests.data import get_model_registry_dir

registry_dir = get_model_registry_dir()
```

## Important Notes

1. **Test Data Only**: This directory is for test data only. No production code should reference these paths.

2. **Path Updates**: All registry.json files have been updated to use the new test data paths (ml/tests/data/...).

3. **Fixtures**: Use pytest fixtures for accessing test data in tests. This ensures consistent access patterns.

4. **No Production References**: Production code must never reference test data. Use proper dependency injection or configuration for production models.

## Adding New Test Data

When adding new test data:

1. Create appropriate subdirectory under `ml/tests/data/`
2. Add corresponding fixture in `ml/tests/conftest.py`
3. Document the structure in this README
4. Ensure paths in any JSON/config files use the test data path
