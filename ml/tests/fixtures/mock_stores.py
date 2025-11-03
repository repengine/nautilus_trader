"""Centralized mock store fixture factory."""

from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from ml.stores.data_store import DataStore
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore


@pytest.fixture
def mock_store_factory():
    """Factory for creating mock stores with consistent specs.

    This factory consolidates duplicate mock store fixture definitions across
    the test suite, reducing code duplication and improving maintainability.

    Usage Examples
    --------------
    Basic usage:
        def test_with_feature_store(mock_store_factory):
            store = mock_store_factory("feature")
            store.write_data.return_value = None
            # ... test logic

    With custom attributes:
        def test_with_custom_store(mock_store_factory):
            store = mock_store_factory("model", engine=None, table_name="predictions")
            # ... test logic

    With custom return values:
        def test_with_return_values(mock_store_factory):
            store = mock_store_factory("strategy")
            store.get_signals.return_value = [{"signal": "BUY"}]
            # ... test logic

    Parameters
    ----------
    store_type : str
        Type of store to create. One of: "feature", "model", "strategy", "data"
    use_spec : bool, default=True
        Whether to use the actual store class as spec for the mock.
        Set to False for tests that need fully dynamic mocks.
    **kwargs : Any
        Additional attributes to set on the mock. Can be used to:
        - Set attributes: engine=None, table_name="foo"
        - Configure return values: get_data=MagicMock(return_value=[])

    Returns
    -------
    MagicMock
        Configured mock store with appropriate spec and attributes

    Raises
    ------
    ValueError
        If store_type is not one of the supported types
    """
    def _factory(store_type: str, use_spec: bool = True, **kwargs: Any) -> MagicMock:
        """Create a mock store of the specified type.

        Args:
            store_type: One of "feature", "model", "strategy", "data"
            use_spec: Whether to use actual store class as spec
            **kwargs: Additional attributes to set on the mock

        Returns:
            MagicMock with appropriate spec
        """
        specs = {
            "feature": FeatureStore,
            "model": ModelStore,
            "strategy": StrategyStore,
            "data": DataStore,
        }

        if store_type not in specs:
            raise ValueError(
                f"Invalid store_type: {store_type}. "
                f"Must be one of: {list(specs.keys())}"
            )

        # Create mock with or without spec based on use_spec parameter
        if use_spec:
            mock = MagicMock(spec=specs[store_type])
        else:
            mock = MagicMock()

        # Set any additional attributes or configure return values
        for attr, value in kwargs.items():
            setattr(mock, attr, value)

        return mock

    return _factory


# Convenience fixtures for backward compatibility
# These allow gradual migration and don't require changing all tests at once

@pytest.fixture
def mock_feature_store(mock_store_factory) -> MagicMock:
    """Mock FeatureStore for unit tests.

    DEPRECATED: Use mock_store_factory("feature") directly.
    This fixture exists for backward compatibility during migration.
    """
    return mock_store_factory("feature")


@pytest.fixture
def mock_model_store(mock_store_factory) -> MagicMock:
    """Mock ModelStore for unit tests.

    DEPRECATED: Use mock_store_factory("model") directly.
    This fixture exists for backward compatibility during migration.
    """
    return mock_store_factory("model")


@pytest.fixture
def mock_strategy_store(mock_store_factory) -> MagicMock:
    """Mock StrategyStore for unit tests.

    DEPRECATED: Use mock_store_factory("strategy") directly.
    This fixture exists for backward compatibility during migration.
    """
    return mock_store_factory("strategy")


@pytest.fixture
def mock_data_store(mock_store_factory) -> MagicMock:
    """Mock DataStore for unit tests.

    DEPRECATED: Use mock_store_factory("data") directly.
    This fixture exists for backward compatibility during migration.
    """
    return mock_store_factory("data")


__all__ = [
    "mock_data_store",
    "mock_feature_store",
    "mock_model_store",
    "mock_store_factory",
    "mock_strategy_store",
]
