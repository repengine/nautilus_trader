"""Unit tests for mock store fixture factory.

Tests verify:
1. Factory creates all 4 store types correctly
2. Spec parameter works (both True and False)
3. Kwargs properly set attributes
4. Backward compatibility fixtures work
5. Invalid input raises ValueError
6. No circular import issues
"""

from unittest.mock import MagicMock

import pytest

from ml.stores.data_store import DataStore
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore

# Note: Fixtures are auto-discovered by pytest from conftest.py
# No need to import them explicitly here


# ============================================================================
# Factory Creation Tests
# ============================================================================


def test_factory_creates_feature_store_with_spec(mock_store_factory):
    """Verify factory creates FeatureStore mock with correct spec."""
    # Act
    result = mock_store_factory("feature")

    # Assert
    assert isinstance(result, MagicMock)
    assert result._spec_class == FeatureStore


def test_factory_creates_model_store_with_spec(mock_store_factory):
    """Verify factory creates ModelStore mock with correct spec."""
    # Act
    result = mock_store_factory("model")

    # Assert
    assert isinstance(result, MagicMock)
    assert result._spec_class == ModelStore


def test_factory_creates_strategy_store_with_spec(mock_store_factory):
    """Verify factory creates StrategyStore mock with correct spec."""
    # Act
    result = mock_store_factory("strategy")

    # Assert
    assert isinstance(result, MagicMock)
    assert result._spec_class == StrategyStore


def test_factory_creates_data_store_with_spec(mock_store_factory):
    """Verify factory creates DataStore mock with correct spec."""
    # Act
    result = mock_store_factory("data")

    # Assert
    assert isinstance(result, MagicMock)
    assert result._spec_class == DataStore


def test_factory_raises_on_invalid_store_type(mock_store_factory):
    """Verify factory rejects invalid store types."""
    # Act & Assert
    with pytest.raises(ValueError, match="Invalid store_type: invalid"):
        mock_store_factory("invalid")

    with pytest.raises(ValueError, match="Must be one of"):
        mock_store_factory("invalid")


def test_factory_accepts_custom_attributes(mock_store_factory):
    """Verify factory sets custom attributes via kwargs."""
    # Act
    store = mock_store_factory("feature", engine=None, table_name="test")

    # Assert
    assert store.engine is None
    assert store.table_name == "test"


def test_factory_use_spec_false_creates_flexible_mock(mock_store_factory):
    """Verify use_spec=False creates mock without spec restrictions."""
    # Act
    store = mock_store_factory("feature", use_spec=False)

    # Assert
    # Mock without spec doesn't have _spec_class or has None
    assert not hasattr(store, "_spec_class") or store._spec_class is None

    # Can set arbitrary attributes without AttributeError
    store.arbitrary_attribute = "test"
    assert store.arbitrary_attribute == "test"


def test_factory_multiple_calls_create_independent_mocks(mock_store_factory):
    """Verify factory creates independent mock instances."""
    # Act
    store1 = mock_store_factory("feature")
    store2 = mock_store_factory("feature")

    # Assert
    assert store1 is not store2

    # Configuring store1 doesn't affect store2
    store1.custom_value = "test1"
    assert not hasattr(store2, "custom_value")


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


def test_mock_feature_store_fixture_works(mock_feature_store):
    """Verify backward compatibility fixture works."""
    # Assert
    assert isinstance(mock_feature_store, MagicMock)
    assert mock_feature_store._spec_class == FeatureStore

    # Can configure return values (use a method that exists on FeatureStore)
    mock_feature_store.write_features.return_value = True
    assert mock_feature_store.write_features() is True


def test_mock_model_store_fixture_works(mock_model_store):
    """Verify backward compatibility fixture works."""
    # Assert
    assert isinstance(mock_model_store, MagicMock)
    assert mock_model_store._spec_class == ModelStore

    # Can configure return values
    mock_model_store.write_predictions.return_value = True
    assert mock_model_store.write_predictions() is True


def test_mock_strategy_store_fixture_works(mock_strategy_store):
    """Verify backward compatibility fixture works."""
    # Assert
    assert isinstance(mock_strategy_store, MagicMock)
    assert mock_strategy_store._spec_class == StrategyStore

    # Can configure return values
    mock_strategy_store.get_signals.return_value = []
    assert mock_strategy_store.get_signals() == []


def test_mock_data_store_fixture_works(mock_data_store):
    """Verify backward compatibility fixture works."""
    # Assert
    assert isinstance(mock_data_store, MagicMock)
    assert mock_data_store._spec_class == DataStore

    # Can configure return values
    mock_data_store.get_health_status.return_value = {"status": "healthy"}
    assert mock_data_store.get_health_status() == {"status": "healthy"}
