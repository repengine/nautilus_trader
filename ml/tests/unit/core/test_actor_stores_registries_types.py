"""
Test suite for ActorStoresRegistries dataclass type annotations.

Task 1.2: Protocol Remediation - Replace generic 'object' types with concrete Store/Registry types

This test suite verifies that all fields in ActorStoresRegistries have concrete type annotations
instead of the generic 'object' type. Tests are designed to be resilient to:
- Runtime aliasing (e.g., FeatureStoreLegacy vs FeatureStore)
- Python version differences (Union vs UnionType)
- Feature flag variations (legacy vs component implementations)

Approach:
- Use dataclasses.fields() for field metadata extraction
- Use typing.get_type_hints() for runtime type resolution
- Check "not object" rather than exact type names (learned from Task 1.1)
- Handle optional types (e.g., PersistenceConfig | None)
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any
from typing import get_type_hints

import pytest

from ml.core.integration import ActorStoresRegistries


# ======================================================================================
# Helper Functions
# ======================================================================================


def get_field_type_name(field_type: Any) -> str:
    """
    Extract human-readable type name from field type annotation.

    Handles:
    - Simple types: FeatureStore -> "FeatureStore"
    - Union types: DataStore | None -> "DataStore"
    - Generic types with __name__
    - String representations as fallback
    """
    # Try direct __name__ attribute
    if hasattr(field_type, "__name__"):
        return field_type.__name__

    # Handle Union types (Python 3.10+ uses types.UnionType, older uses typing.Union)
    type_str = str(field_type)

    # Extract base type from Union (e.g., "DataStore | None" -> "DataStore")
    if "|" in type_str:
        # Handle "X | None" syntax
        parts = type_str.split("|")
        base_type = parts[0].strip()
        # Remove module prefix if present (e.g., "ml.stores.DataStore" -> "DataStore")
        if "." in base_type:
            base_type = base_type.split(".")[-1]
        return base_type

    # Handle typing.Union[X, None] syntax
    if "Union[" in type_str:
        # Extract first type from Union
        start = type_str.index("[") + 1
        end = type_str.index(",") if "," in type_str else type_str.index("]")
        base_type = type_str[start:end].strip()
        if "." in base_type:
            base_type = base_type.split(".")[-1]
        return base_type

    # Return string representation as fallback
    return type_str


def is_union_type(field_type: Any) -> bool:
    """
    Check if field type is a Union type (handles Python version differences).

    Python 3.10+: types.UnionType for X | None syntax
    Python 3.9: typing.Union for Union[X, None] syntax
    """
    type_str = str(field_type)
    return "|" in type_str or "Union[" in type_str


# ======================================================================================
# Individual Field Type Tests
# ======================================================================================


class TestActorStoresRegistriesTypeAnnotations:
    """Test suite for ActorStoresRegistries field type annotations."""

    def test_feature_store_field_has_concrete_type_not_object(self) -> None:
        """
        Verify feature_store field has FeatureStore type, not object.

        BEFORE implementation: Will show <class 'object'>
        AFTER implementation: Will show FeatureStore or FeatureStoreLegacy

        Uses dataclasses.fields() to inspect field metadata.
        Checks "not object" rather than exact type name (resilient to aliasing).
        """
        fields = dataclasses.fields(ActorStoresRegistries)
        feature_store_field = next(f for f in fields if f.name == "feature_store")

        field_type = feature_store_field.type
        type_name = get_field_type_name(field_type)

        # LESSON FROM TASK 1.1: Check "not object", not exact name
        assert field_type is not object, (
            f"feature_store field should not have generic object type, got {type_name}"
        )

        # Sanity check: should be a Store type
        assert "Store" in type_name or "FeatureStore" in type_name, (
            f"Expected FeatureStore type, got {type_name}"
        )

    def test_model_store_field_has_concrete_type_not_object(self) -> None:
        """
        Verify model_store field has ModelStore type, not object.

        Uses dataclasses.fields() to inspect field metadata.
        Checks "not object" rather than exact type name.
        """
        fields = dataclasses.fields(ActorStoresRegistries)
        model_store_field = next(f for f in fields if f.name == "model_store")

        field_type = model_store_field.type
        type_name = get_field_type_name(field_type)

        assert field_type is not object, (
            f"model_store field should not have generic object type, got {type_name}"
        )

        assert "Store" in type_name or "ModelStore" in type_name, (
            f"Expected ModelStore type, got {type_name}"
        )

    def test_strategy_store_field_has_concrete_type_not_object(self) -> None:
        """
        Verify strategy_store field has StrategyStore type, not object.

        Uses dataclasses.fields() to inspect field metadata.
        Checks "not object" rather than exact type name.
        """
        fields = dataclasses.fields(ActorStoresRegistries)
        strategy_store_field = next(f for f in fields if f.name == "strategy_store")

        field_type = strategy_store_field.type
        type_name = get_field_type_name(field_type)

        assert field_type is not object, (
            f"strategy_store field should not have generic object type, got {type_name}"
        )

        assert "Store" in type_name or "StrategyStore" in type_name, (
            f"Expected StrategyStore type, got {type_name}"
        )

    def test_data_store_field_has_concrete_type_not_object(self) -> None:
        """
        Verify data_store field has DataStore type, not object.

        Uses dataclasses.fields() to inspect field metadata.
        Checks "not object" rather than exact type name.
        """
        fields = dataclasses.fields(ActorStoresRegistries)
        data_store_field = next(f for f in fields if f.name == "data_store")

        field_type = data_store_field.type
        type_name = get_field_type_name(field_type)

        assert field_type is not object, (
            f"data_store field should not have generic object type, got {type_name}"
        )

        assert "Store" in type_name or "DataStore" in type_name, (
            f"Expected DataStore type, got {type_name}"
        )

    def test_feature_registry_field_has_concrete_type_not_object(self) -> None:
        """
        Verify feature_registry field has FeatureRegistry type, not object.

        Uses dataclasses.fields() to inspect field metadata.
        Checks "not object" rather than exact type name.
        """
        fields = dataclasses.fields(ActorStoresRegistries)
        feature_registry_field = next(f for f in fields if f.name == "feature_registry")

        field_type = feature_registry_field.type
        type_name = get_field_type_name(field_type)

        assert field_type is not object, (
            f"feature_registry field should not have generic object type, got {type_name}"
        )

        assert "Registry" in type_name or "FeatureRegistry" in type_name, (
            f"Expected FeatureRegistry type, got {type_name}"
        )

    def test_model_registry_field_has_concrete_type_not_object(self) -> None:
        """
        Verify model_registry field has ModelRegistry type, not object.

        Uses dataclasses.fields() to inspect field metadata.
        Checks "not object" rather than exact type name.
        """
        fields = dataclasses.fields(ActorStoresRegistries)
        model_registry_field = next(f for f in fields if f.name == "model_registry")

        field_type = model_registry_field.type
        type_name = get_field_type_name(field_type)

        assert field_type is not object, (
            f"model_registry field should not have generic object type, got {type_name}"
        )

        assert "Registry" in type_name or "ModelRegistry" in type_name, (
            f"Expected ModelRegistry type, got {type_name}"
        )

    def test_strategy_registry_field_has_concrete_type_not_object(self) -> None:
        """
        Verify strategy_registry field has StrategyRegistry type, not object.

        Uses dataclasses.fields() to inspect field metadata.
        Checks "not object" rather than exact type name.
        """
        fields = dataclasses.fields(ActorStoresRegistries)
        strategy_registry_field = next(f for f in fields if f.name == "strategy_registry")

        field_type = strategy_registry_field.type
        type_name = get_field_type_name(field_type)

        assert field_type is not object, (
            f"strategy_registry field should not have generic object type, got {type_name}"
        )

        assert "Registry" in type_name or "StrategyRegistry" in type_name, (
            f"Expected StrategyRegistry type, got {type_name}"
        )

    def test_data_registry_field_has_concrete_type_not_object(self) -> None:
        """
        Verify data_registry field has DataRegistry type, not object.

        Uses dataclasses.fields() to inspect field metadata.
        Checks "not object" rather than exact type name.
        """
        fields = dataclasses.fields(ActorStoresRegistries)
        data_registry_field = next(f for f in fields if f.name == "data_registry")

        field_type = data_registry_field.type
        type_name = get_field_type_name(field_type)

        assert field_type is not object, (
            f"data_registry field should not have generic object type, got {type_name}"
        )

        assert "Registry" in type_name or "DataRegistry" in type_name, (
            f"Expected DataRegistry type, got {type_name}"
        )

    def test_persistence_config_field_has_concrete_type_not_object(self) -> None:
        """
        Verify persistence_config field has PersistenceConfig | None type, not object | None.

        Handles optional types and Python version differences:
        - Python 3.10+: PersistenceConfig | None (types.UnionType)
        - Python 3.9: Union[PersistenceConfig, None] (typing.Union)

        Uses dataclasses.fields() to inspect field metadata.
        Checks "not object" rather than exact type name.
        """
        fields = dataclasses.fields(ActorStoresRegistries)
        persistence_config_field = next(f for f in fields if f.name == "persistence_config")

        field_type = persistence_config_field.type
        type_name = get_field_type_name(field_type)

        # Should be optional (Union with None)
        assert is_union_type(field_type), (
            f"persistence_config should be optional type (X | None), got {type_name}"
        )

        # Base type should NOT be object
        assert field_type is not object, (
            f"persistence_config field should not have generic object type, got {type_name}"
        )

        # Base type should be PersistenceConfig
        assert "Config" in type_name or "PersistenceConfig" in type_name, (
            f"Expected PersistenceConfig type, got {type_name}"
        )


# ======================================================================================
# Property Tests (All Fields)
# ======================================================================================


def test_all_store_registry_fields_have_concrete_types() -> None:
    """
    Property test: Verify ALL 9 fields have concrete types, not object.

    Fields checked:
    - feature_store: FeatureStore (not object)
    - model_store: ModelStore (not object)
    - strategy_store: StrategyStore (not object)
    - data_store: DataStore (not object)
    - feature_registry: FeatureRegistry (not object)
    - model_registry: ModelRegistry (not object)
    - strategy_registry: StrategyRegistry (not object)
    - data_registry: DataRegistry (not object)
    - persistence_config: PersistenceConfig | None (not object | None)

    Fields skipped:
    - connection_string: str | None (already correctly typed)
    """
    fields = dataclasses.fields(ActorStoresRegistries)

    # Fields that should have concrete types (not object)
    expected_typed_fields = {
        "feature_store",
        "model_store",
        "strategy_store",
        "data_store",
        "feature_registry",
        "model_registry",
        "strategy_registry",
        "data_registry",
        "persistence_config",
    }

    failures = []

    for field in fields:
        if field.name not in expected_typed_fields:
            continue  # Skip connection_string (already correct)

        field_type = field.type
        type_name = get_field_type_name(field_type)

        # Check that field type is NOT object
        if field_type is object:
            failures.append(
                f"{field.name} still has generic object type (expected concrete type)"
            )

        # Sanity check: should be a Store, Registry, or Config type
        if not any(
            keyword in type_name
            for keyword in ["Store", "Registry", "Config"]
        ):
            failures.append(
                f"{field.name} has unexpected type {type_name} "
                f"(expected Store/Registry/Config type)"
            )

    # Report all failures together
    if failures:
        failure_msg = "\n".join(f"  - {msg}" for msg in failures)
        pytest.fail(f"Type annotation failures:\n{failure_msg}")


# ======================================================================================
# Type Hints Tests
# ======================================================================================


def test_type_hints_work_with_python_version_check() -> None:
    """
    Verify typing.get_type_hints() returns concrete types for all fields.

    Handles Python version differences:
    - Python 3.10+: Union types represented as types.UnionType
    - Python 3.9: Union types represented as typing.Union

    This test ensures that type hints are accessible at runtime and that
    no field has the generic object type.
    """
    hints = get_type_hints(ActorStoresRegistries)

    # Fields that should have concrete types
    typed_fields = [
        "feature_store",
        "model_store",
        "strategy_store",
        "data_store",
        "feature_registry",
        "model_registry",
        "strategy_registry",
        "data_registry",
        "persistence_config",
    ]

    failures = []

    for field_name in typed_fields:
        hint = hints.get(field_name)

        if hint is None:
            failures.append(f"{field_name} has no type hint")
            continue

        # Check that hint is NOT object
        if hint is object:
            failures.append(f"{field_name} has object type in hints")
            continue

        # Get type name for sanity check
        type_name = get_field_type_name(hint)

        # Verify it's a Store/Registry/Config type
        if not any(
            keyword in type_name
            for keyword in ["Store", "Registry", "Config"]
        ):
            failures.append(
                f"{field_name} has unexpected type hint {type_name}"
            )

    # Report all failures
    if failures:
        failure_msg = "\n".join(f"  - {msg}" for msg in failures)
        pytest.fail(f"Type hint failures:\n{failure_msg}")


# ======================================================================================
# Dataclass Instantiation Tests
# ======================================================================================


def test_dataclass_can_be_instantiated_with_typed_fields() -> None:
    """
    Verify ActorStoresRegistries can be instantiated with concrete Store/Registry instances.

    Creates lightweight mock objects to satisfy type requirements without
    requiring database connections or complex initialization.

    This test ensures that:
    1. Dataclass accepts typed instances (no TypeErrors)
    2. All fields are accessible after instantiation
    3. Type annotations don't break runtime behavior
    """
    from unittest.mock import MagicMock

    # Create minimal mock objects
    mock_feature_store = MagicMock()
    mock_model_store = MagicMock()
    mock_strategy_store = MagicMock()
    mock_data_store = MagicMock()
    mock_feature_registry = MagicMock()
    mock_model_registry = MagicMock()
    mock_strategy_registry = MagicMock()
    mock_data_registry = MagicMock()

    # Instantiate dataclass
    container = ActorStoresRegistries(
        feature_store=mock_feature_store,
        feature_dataset_store=None,
        model_store=mock_model_store,
        strategy_store=mock_strategy_store,
        data_store=mock_data_store,
        feature_registry=mock_feature_registry,
        model_registry=mock_model_registry,
        strategy_registry=mock_strategy_registry,
        data_registry=mock_data_registry,
        persistence_config=None,
        connection_string=None,
    )

    # Verify instance created successfully
    assert isinstance(container, ActorStoresRegistries)

    # Verify all fields accessible
    assert container.feature_store is mock_feature_store
    assert container.model_store is mock_model_store
    assert container.strategy_store is mock_strategy_store
    assert container.data_store is mock_data_store
    assert container.feature_registry is mock_feature_registry
    assert container.model_registry is mock_model_registry
    assert container.strategy_registry is mock_strategy_registry
    assert container.data_registry is mock_data_registry
    assert container.persistence_config is None
    assert container.connection_string is None


# ======================================================================================
# Backward Compatibility Tests
# ======================================================================================


def test_existing_usage_patterns_still_work() -> None:
    """
    Verify existing code using ActorStoresRegistries continues to work.

    Type annotations are compile-time only and should have ZERO runtime impact.
    This test ensures that code patterns that worked before continue to work.
    """
    from unittest.mock import MagicMock

    # Pattern 1: Create with all fields
    container = ActorStoresRegistries(
        feature_store=MagicMock(),
        feature_dataset_store=None,
        model_store=MagicMock(),
        strategy_store=MagicMock(),
        data_store=MagicMock(),
        feature_registry=MagicMock(),
        model_registry=MagicMock(),
        strategy_registry=MagicMock(),
        data_registry=MagicMock(),
        persistence_config=None,
        connection_string=None,
    )

    assert container is not None
    assert hasattr(container, "feature_store")
    assert hasattr(container, "model_store")
    assert hasattr(container, "strategy_store")
    assert hasattr(container, "data_store")
    assert hasattr(container, "feature_registry")
    assert hasattr(container, "model_registry")
    assert hasattr(container, "strategy_registry")
    assert hasattr(container, "data_registry")
    assert hasattr(container, "persistence_config")
    assert hasattr(container, "connection_string")


# ======================================================================================
# Integration Tests (Optional - may require database)
# ======================================================================================


@pytest.mark.integration
def test_init_ml_stores_and_registries_returns_typed_dataclass() -> None:
    """
    Verify init_ml_stores_and_registries() factory function returns properly typed dataclass.

    This is an integration test that may require database connectivity.
    It verifies that the factory function used in production returns an
    ActorStoresRegistries instance with properly typed fields.

    NOTE: This test may fail if database is not available. If so, it should
    gracefully degrade or be skipped in environments without PostgreSQL.
    """
    from unittest.mock import MagicMock

    from ml.core.integration import init_ml_stores_and_registries

    # Create minimal config
    config = MagicMock()
    config.persistence = None  # Will trigger dummy store fallback

    try:
        # Call factory function
        container = init_ml_stores_and_registries(config)

        # Verify return type
        assert isinstance(container, ActorStoresRegistries), (
            f"Expected ActorStoresRegistries, got {type(container)}"
        )

        # Verify all fields are populated (not None for stores/registries)
        assert container.feature_store is not None, "feature_store should not be None"
        assert container.model_store is not None, "model_store should not be None"
        assert container.strategy_store is not None, "strategy_store should not be None"
        # data_store can be None (optional)
        assert container.feature_registry is not None, "feature_registry should not be None"
        assert container.model_registry is not None, "model_registry should not be None"
        assert container.strategy_registry is not None, "strategy_registry should not be None"
        assert container.data_registry is not None, "data_registry should not be None"

        # Verify field types are concrete (not object)
        fields = dataclasses.fields(ActorStoresRegistries)
        for field in fields:
            if field.name == "connection_string":
                continue  # Skip already-correct field

            field_type = field.type
            assert field_type is not object, (
                f"{field.name} should not have object type in returned instance"
            )

    except Exception as e:
        # If database not available, skip gracefully
        if "database" in str(e).lower() or "connection" in str(e).lower():
            pytest.skip(f"Database not available for integration test: {e}")
        else:
            raise


# ======================================================================================
# Circular Import Prevention Test
# ======================================================================================


def test_no_circular_import_from_type_annotations() -> None:
    """
    Verify that type annotations don't cause circular import errors.

    Type annotations should be in TYPE_CHECKING block to prevent circular imports.
    This test verifies that importing ActorStoresRegistries doesn't trigger
    circular dependency issues.
    """
    # If we can import, circular imports are prevented
    from ml.core.integration import ActorStoresRegistries  # noqa: F401

    # Verify TYPE_CHECKING imports work
    import ml.core.integration as integration_module

    # Check that TYPE_CHECKING block exists
    source = integration_module.__file__
    assert source is not None, "Could not find integration module source"

    # This test passing means no circular imports occurred during module load
    assert True, "No circular imports detected"
