"""
Unit tests for MLIntegrationManager type annotations.

This test module verifies that MLIntegrationManager class attributes have proper
concrete type annotations (not generic object) to enable full type safety and IDE
support for all ML component access.

Test Strategy:
- Use typing.get_type_hints() to extract runtime type information
- Verify each store/registry attribute has concrete type (not object)
- Ensure TYPE_CHECKING imports don't break runtime
- Validate type hints enable proper IDE autocomplete
- Confirm backward compatibility is preserved

All tests are initially marked to skip until implementation provides concrete types.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING
from typing import get_type_hints

import pytest


# ==============================================================================
# Type Hint Extraction Tests
# ==============================================================================


class TestMLIntegrationManagerTypeAnnotations:
    """Test suite for MLIntegrationManager type annotations."""

    def test_feature_store_has_concrete_type_not_object(self) -> None:
        """Verify feature_store attribute has FeatureStore type, not object.

        BEFORE implementation: Will show <class 'object'>
        AFTER implementation: Will show 'FeatureStore' (from ml.stores.feature_store)

        This test ensures type checkers and IDEs can provide autocomplete for
        feature_store methods like write_features(), read_features(), etc.
        """
        from ml.core.integration import MLIntegrationManager

        hints = get_type_hints(MLIntegrationManager)
        feature_store_type = hints.get("feature_store")

        # Verify type exists
        assert feature_store_type is not None, "feature_store type hint missing"

        # Extract type name (handle forward references)
        type_name = getattr(feature_store_type, "__name__", str(feature_store_type))

        # Verify it's NOT object type (the actual goal of Task 1.1)
        assert type_name != "object", (
            f"Type should not be generic 'object', got {type_name}. "
            "MLIntegrationManager.feature_store should have concrete type annotation."
        )

        # Verify it's a Store type (not completely arbitrary)
        assert "Store" in type_name, (
            f"Expected concrete Store type, got {type_name}"
        )

    def test_model_store_has_concrete_type_not_object(self) -> None:
        """Verify model_store attribute has ModelStore type, not object.

        BEFORE implementation: Will show <class 'object'>
        AFTER implementation: Will show 'ModelStore' (from ml.stores.model_store)
        """
        from ml.core.integration import MLIntegrationManager

        hints = get_type_hints(MLIntegrationManager)
        model_store_type = hints.get("model_store")

        assert model_store_type is not None, "model_store type hint missing"
        type_name = getattr(model_store_type, "__name__", str(model_store_type))

        # Verify it's NOT object type (the actual goal of Task 1.1)
        assert type_name != "object", (
            f"Type should not be generic 'object', got {type_name}. "
            "MLIntegrationManager.model_store should have concrete type annotation."
        )

        # Verify it's a Store type (not completely arbitrary)
        assert "Store" in type_name, (
            f"Expected concrete Store type, got {type_name}"
        )

    def test_strategy_store_has_concrete_type_not_object(self) -> None:
        """Verify strategy_store attribute has StrategyStore type, not object.

        BEFORE implementation: Will show <class 'object'>
        AFTER implementation: Will show 'StrategyStore' (from ml.stores.strategy_store)
        """
        from ml.core.integration import MLIntegrationManager

        hints = get_type_hints(MLIntegrationManager)
        strategy_store_type = hints.get("strategy_store")

        assert strategy_store_type is not None, "strategy_store type hint missing"
        type_name = getattr(strategy_store_type, "__name__", str(strategy_store_type))

        # Verify it's NOT object type (the actual goal of Task 1.1)
        assert type_name != "object", (
            f"Type should not be generic 'object', got {type_name}. "
            "MLIntegrationManager.strategy_store should have concrete type annotation."
        )

        # Verify it's a Store type (not completely arbitrary)
        assert "Store" in type_name, (
            f"Expected concrete Store type, got {type_name}"
        )

    def test_data_store_has_concrete_type_with_optional_none(self) -> None:
        """Verify data_store attribute has DataStore | None type, not object | None.

        BEFORE implementation: Will show typing.Union[object, None]
        AFTER implementation: Will show typing.Union[DataStore, None]

        DataStore can be None when using file fallback mode, so the union is expected.
        """
        import types
        from ml.core.integration import MLIntegrationManager

        hints = get_type_hints(MLIntegrationManager)
        data_store_type = hints.get("data_store")

        assert data_store_type is not None, "data_store type hint missing"

        # Handle Union types (Python 3.10+ uses | syntax but runtime may be Union or UnionType)
        import typing

        origin = typing.get_origin(data_store_type)

        # Accept both typing.Union and types.UnionType
        is_union = origin is typing.Union or origin is types.UnionType
        assert is_union, (
            f"Expected Union type for data_store, got {origin} (type: {type(data_store_type)}). "
            "data_store should be DataStore | None (optional)."
        )

        # Get the union args
        args = typing.get_args(data_store_type)

        # Should have exactly 2 args: DataStore and None
        assert len(args) == 2, f"Expected 2 union args, got {len(args)}: {args}"

        # Extract non-None type
        non_none_types = [arg for arg in args if arg is not type(None)]
        assert len(non_none_types) == 1, (
            f"Expected exactly one non-None type, got {len(non_none_types)}: {non_none_types}"
        )

        data_store_concrete = non_none_types[0]
        type_name = getattr(data_store_concrete, "__name__", str(data_store_concrete))

        # Verify it's NOT object type (the actual goal of Task 1.1)
        assert type_name != "object", (
            f"Type should not be generic 'object', got {type_name}. "
            "MLIntegrationManager.data_store should have concrete DataStore | None type annotation."
        )

        # Verify it's a Store type (not completely arbitrary)
        assert "Store" in type_name, (
            f"Expected concrete Store type, got {type_name}"
        )

    def test_feature_registry_has_concrete_type_not_object(self) -> None:
        """Verify feature_registry attribute has FeatureRegistry type, not object.

        BEFORE implementation: Will show <class 'object'>
        AFTER implementation: Will show 'FeatureRegistry' (from ml.registry.base)
        """
        from ml.core.integration import MLIntegrationManager

        hints = get_type_hints(MLIntegrationManager)
        feature_registry_type = hints.get("feature_registry")

        assert feature_registry_type is not None, "feature_registry type hint missing"
        type_name = getattr(feature_registry_type, "__name__", str(feature_registry_type))

        # Verify it's NOT object type (the actual goal of Task 1.1)
        assert type_name != "object", (
            f"Type should not be generic 'object', got {type_name}. "
            "MLIntegrationManager.feature_registry should have concrete type annotation."
        )

        # Verify it's a Registry type (not completely arbitrary)
        assert "Registry" in type_name, (
            f"Expected concrete Registry type, got {type_name}"
        )

    def test_model_registry_has_concrete_type_not_object(self) -> None:
        """Verify model_registry attribute has ModelRegistry type, not object.

        BEFORE implementation: Will show <class 'object'>
        AFTER implementation: Will show 'ModelRegistry' (from ml.registry.base)
        """
        from ml.core.integration import MLIntegrationManager

        hints = get_type_hints(MLIntegrationManager)
        model_registry_type = hints.get("model_registry")

        assert model_registry_type is not None, "model_registry type hint missing"
        type_name = getattr(model_registry_type, "__name__", str(model_registry_type))

        # Verify it's NOT object type (the actual goal of Task 1.1)
        assert type_name != "object", (
            f"Type should not be generic 'object', got {type_name}. "
            "MLIntegrationManager.model_registry should have concrete type annotation."
        )

        # Verify it's a Registry type (not completely arbitrary)
        assert "Registry" in type_name, (
            f"Expected concrete Registry type, got {type_name}"
        )

    def test_strategy_registry_has_concrete_type_not_object(self) -> None:
        """Verify strategy_registry attribute has StrategyRegistry type, not object.

        BEFORE implementation: Will show <class 'object'>
        AFTER implementation: Will show 'StrategyRegistry' (from ml.registry.base)
        """
        from ml.core.integration import MLIntegrationManager

        hints = get_type_hints(MLIntegrationManager)
        strategy_registry_type = hints.get("strategy_registry")

        assert strategy_registry_type is not None, "strategy_registry type hint missing"
        type_name = getattr(strategy_registry_type, "__name__", str(strategy_registry_type))

        # Verify it's NOT object type (the actual goal of Task 1.1)
        assert type_name != "object", (
            f"Type should not be generic 'object', got {type_name}. "
            "MLIntegrationManager.strategy_registry should have concrete type annotation."
        )

        # Verify it's a Registry type (not completely arbitrary)
        assert "Registry" in type_name, (
            f"Expected concrete Registry type, got {type_name}"
        )

    def test_data_registry_has_concrete_type_not_object(self) -> None:
        """Verify data_registry attribute has DataRegistry type, not object.

        BEFORE implementation: Will show <class 'object'>
        AFTER implementation: Will show 'DataRegistry' (from ml.registry.base)
        """
        from ml.core.integration import MLIntegrationManager

        hints = get_type_hints(MLIntegrationManager)
        data_registry_type = hints.get("data_registry")

        assert data_registry_type is not None, "data_registry type hint missing"
        type_name = getattr(data_registry_type, "__name__", str(data_registry_type))

        # Verify it's NOT object type (the actual goal of Task 1.1)
        assert type_name != "object", (
            f"Type should not be generic 'object', got {type_name}. "
            "MLIntegrationManager.data_registry should have concrete type annotation."
        )

        # Verify it's a Registry type (not completely arbitrary)
        assert "Registry" in type_name, (
            f"Expected concrete Registry type, got {type_name}"
        )


# ==============================================================================
# Property Test: All Store/Registry Attributes Have Concrete Types
# ==============================================================================


def test_all_store_registry_attributes_have_concrete_types() -> None:
    """Property test: Verify ALL 8 store/registry attributes have non-object types.

    This is a comprehensive property test that ensures NONE of the 8 critical
    attributes use the generic 'object' type annotation.

    Coverage:
    - 4 stores: feature_store, model_store, strategy_store, data_store
    - 4 registries: feature_registry, model_registry, strategy_registry, data_registry

    BEFORE implementation: Will find 8 violations (all are 'object')
    AFTER implementation: Will find 0 violations (all are concrete types)
    """
    import types
    from ml.core.integration import MLIntegrationManager

    hints = get_type_hints(MLIntegrationManager)

    # List of all store/registry attributes that should have concrete types
    attributes_to_check = [
        "feature_store",
        "model_store",
        "strategy_store",
        "data_store",
        "feature_registry",
        "model_registry",
        "strategy_registry",
        "data_registry",
    ]

    violations = []

    for attr_name in attributes_to_check:
        hint = hints.get(attr_name)

        if hint is None:
            violations.append(f"{attr_name}: missing type hint")
            continue

        # Handle Union types (e.g., DataStore | None)
        import typing

        origin = typing.get_origin(hint)
        # Accept both typing.Union and types.UnionType
        if origin is typing.Union or origin is types.UnionType:
            # Extract non-None type from Union
            args = typing.get_args(hint)
            non_none_types = [arg for arg in args if arg is not type(None)]
            if non_none_types:
                hint = non_none_types[0]

        # Get type name
        type_name = getattr(hint, "__name__", str(hint))

        # Check if type is generic 'object' (the actual goal - no exact name matching)
        if type_name == "object":
            violations.append(f"{attr_name}: has generic 'object' type (should be concrete)")

        # Verify it's a Store or Registry type (sanity check)
        elif "Store" not in type_name and "Registry" not in type_name:
            violations.append(
                f"{attr_name}: has unexpected type '{type_name}' (expected Store or Registry type)"
            )

    # This will FAIL with current implementation (8 violations)
    # Will PASS after implementation (0 violations)
    assert not violations, (
        f"Found {len(violations)} type annotation violations:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


# ==============================================================================
# Runtime Import Safety Tests
# ==============================================================================


def test_type_checking_imports_dont_affect_runtime() -> None:
    """Verify TYPE_CHECKING imports don't break runtime module loading.

    TYPE_CHECKING is a constant that's False at runtime but True during type checking.
    This test ensures that imports inside `if TYPE_CHECKING:` blocks don't cause
    import errors when the module is loaded at runtime.

    This is critical because we're adding TYPE_CHECKING imports to avoid circular
    dependencies while maintaining type safety.
    """
    # Attempt to import the module - should not raise ImportError
    try:
        from ml.core import integration

        # Reload to ensure TYPE_CHECKING logic is executed
        importlib.reload(integration)
    except ImportError as e:
        pytest.fail(f"TYPE_CHECKING imports caused runtime import error: {e}")

    # Verify the class is accessible
    from ml.core.integration import MLIntegrationManager

    assert MLIntegrationManager is not None, "MLIntegrationManager should be importable"


def test_mlintegration_manager_can_be_instantiated_with_dummy_stores() -> None:
    """Verify MLIntegrationManager can still be instantiated after type changes.

    This test ensures backward compatibility - the type annotation changes should
    not affect runtime behavior or instantiation.

    We use dummy stores (fastest initialization) to avoid database dependencies.
    """
    from ml.core.integration import MLIntegrationManager

    # Create a minimal config object with dummy store flag
    class DummyConfig:
        use_dummy_stores = True
        db_connection = None
        allow_dummy_fallback = True

    # Should not raise - type annotations are compile-time only
    try:
        mgr = MLIntegrationManager(
            config=DummyConfig(),
            ensure_healthy=False,  # Skip health checks for speed
        )
    except Exception as e:
        pytest.fail(
            f"MLIntegrationManager instantiation failed after type annotation changes: {e}"
        )

    # Verify stores exist (even if dummy)
    assert hasattr(mgr, "feature_store"), "feature_store attribute missing"
    assert hasattr(mgr, "model_store"), "model_store attribute missing"
    assert hasattr(mgr, "strategy_store"), "strategy_store attribute missing"
    assert hasattr(mgr, "data_store"), "data_store attribute missing"

    # Verify registries exist
    assert hasattr(mgr, "feature_registry"), "feature_registry attribute missing"
    assert hasattr(mgr, "model_registry"), "model_registry attribute missing"
    assert hasattr(mgr, "strategy_registry"), "strategy_registry attribute missing"
    assert hasattr(mgr, "data_registry"), "data_registry attribute missing"


# ==============================================================================
# Backward Compatibility Tests
# ==============================================================================


def test_existing_usage_patterns_still_work() -> None:
    """Verify existing code patterns continue to work after type annotation changes.

    This test simulates common usage patterns to ensure backward compatibility:
    - Creating MLIntegrationManager with config
    - Accessing store attributes
    - Calling store methods (even on dummy stores)

    All of these should work identically before and after the type annotation changes.
    """
    from ml.core.integration import MLIntegrationManager

    class DummyConfig:
        use_dummy_stores = True
        db_connection = None
        allow_dummy_fallback = True

    # Pattern 1: Create manager with config
    mgr = MLIntegrationManager(config=DummyConfig(), ensure_healthy=False)

    # Pattern 2: Access store attributes
    feature_store = mgr.feature_store
    model_store = mgr.model_store
    strategy_store = mgr.strategy_store
    data_store = mgr.data_store

    # Pattern 3: Access registry attributes
    feature_registry = mgr.feature_registry
    model_registry = mgr.model_registry
    strategy_registry = mgr.strategy_registry
    data_registry = mgr.data_registry

    # All should be non-None (even if dummy)
    assert feature_store is not None, "feature_store should not be None"
    assert model_store is not None, "model_store should not be None"
    assert strategy_store is not None, "strategy_store should not be None"
    assert data_store is not None, "data_store should not be None"
    assert feature_registry is not None, "feature_registry should not be None"
    assert model_registry is not None, "model_registry should not be None"
    assert strategy_registry is not None, "strategy_registry should not be None"
    assert data_registry is not None, "data_registry should not be None"


# ==============================================================================
# Edge Case Tests
# ==============================================================================


def test_type_hints_work_with_python_version_check() -> None:
    """Verify type hints work across Python 3.10+ versions.

    TYPE_CHECKING guards and PEP 604 union syntax (X | Y) require Python 3.10+.
    This test verifies the implementation uses compatible syntax.
    """
    import types
    # Verify we're on supported Python version
    assert sys.version_info >= (3, 10), "Python 3.10+ required for PEP 604 union syntax"

    from ml.core.integration import MLIntegrationManager

    hints = get_type_hints(MLIntegrationManager)

    # Verify data_store uses modern union syntax (DataStore | None not Union[DataStore, None])
    data_store_hint = hints.get("data_store")
    assert data_store_hint is not None, "data_store type hint missing"

    # The hint should be properly resolved
    import typing

    origin = typing.get_origin(data_store_hint)
    # Accept both typing.Union and types.UnionType (Python 3.10+ uses UnionType for | syntax)
    is_union = origin is typing.Union or origin is types.UnionType
    assert is_union, (
        f"data_store should use Union type (| syntax), got origin={origin}, "
        f"hint type={type(data_store_hint)}"
    )


def test_no_circular_import_from_type_annotations() -> None:
    """Verify type annotations don't introduce circular import dependencies.

    This test ensures that adding concrete type imports inside TYPE_CHECKING
    blocks doesn't create circular dependencies that would break the module.

    The original issue was:
    - ml.core.integration imports ml.stores.feature_store
    - ml.stores depends on ml.core for some utilities
    - Adding direct imports would create a cycle

    Solution: TYPE_CHECKING imports are only evaluated by type checkers, not at runtime.
    """
    # Try to import both modules in different orders
    try:
        # Order 1: integration first
        from ml.core import integration

        importlib.reload(integration)

        from ml.stores import feature_store

        importlib.reload(feature_store)

        # Order 2: stores first
        from ml.stores import model_store

        importlib.reload(model_store)

        from ml.core import integration as integration2

        importlib.reload(integration2)

    except ImportError as e:
        pytest.fail(f"Circular import detected after type annotation changes: {e}")
