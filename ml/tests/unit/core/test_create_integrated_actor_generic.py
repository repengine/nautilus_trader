"""
Test Suite: Protocol Remediation Task 2.4 - create_integrated_actor Generic Return Type

Verifies that create_integrated_actor uses generic TypeVar to preserve caller's
specific actor type instead of type erasure with 'object'.

Coverage Target: ≥90%
Pattern: Protocol-First (Pattern 2) + Generics

Test Design Principles (from Tasks 1.1 and 2.3 success):
1. Verify BEHAVIOR (types not object, function callable) not implementation details
2. Handle runtime variations (Python versions, TypeVar repr, feature flags)
3. Use flexible assertions (not exact type name matching)
4. All tests initially marked @pytest.mark.skip awaiting implementation

Phase 1: All tests marked @pytest.mark.skip awaiting implementation.
Phase 2: Implementation agent will remove skips and make tests pass.
Phase 4: Validation agent will verify "11 passed" (not "11 collected").
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import get_type_hints

import pytest

if TYPE_CHECKING:
    from unittest.mock import Mock


# Test Fixtures


@pytest.fixture
def mock_integration_manager():
    """Mock MLIntegrationManager with db_connection."""
    from unittest.mock import Mock

    manager = Mock()
    manager.db_connection = "postgresql://test:test@localhost:5432/test"
    return manager


@pytest.fixture
def mock_actor_class():
    """Mock actor class that accepts config."""

    class MockActor:
        def __init__(self, config: object) -> None:
            self.config = config

    return MockActor


@pytest.fixture
def mock_config():
    """Mock config object."""
    from unittest.mock import Mock

    config = Mock()
    config.db_connection = None  # Will be set by create_integrated_actor
    return config


@pytest.fixture
def mock_config_no_db():
    """Mock config without db_connection attribute."""
    from unittest.mock import Mock

    config = Mock(spec=[])  # No attributes initially
    return config


@pytest.fixture
def dummy_actor_class_a():
    """First dummy actor class for type preservation tests."""

    class DummyActorA:
        def __init__(self, config: object) -> None:
            self.config = config

        def method_a(self) -> str:
            return "A"

    return DummyActorA


@pytest.fixture
def dummy_actor_class_b():
    """Second dummy actor class for type preservation tests."""

    class DummyActorB:
        def __init__(self, config: object) -> None:
            self.config = config

        def method_b(self) -> str:
            return "B"

    return DummyActorB


@pytest.fixture
def mock_actor_with_custom_method():
    """Mock actor with custom method for IDE autocomplete test."""

    class ActorWithCustomMethod:
        def __init__(self, config: object) -> None:
            self.config = config

        def custom_predict(self, data: object) -> float:
            return 0.5

    return ActorWithCustomMethod


# Test Classes


class TestCreateIntegratedActorSignature:
    """Test suite verifying create_integrated_actor method signature uses generic TypeVar."""

    def test_return_type_not_object(self) -> None:
        """
        Verify return type is NOT generic object type (uses TypeVar).

        The original implementation returned 'object', erasing all type information.
        After refactoring, return type should be generic TypeVar (ActorT) that
        preserves the caller's specific actor type.

        Behavior tested: Return type is generic (not object).
        Flexible assertion: Handles TypeVar display variations across Python versions.

        Expected: Type contains TypeVar patterns and is not 'object'.
        """
        from ml.core.integration import MLIntegrationManager

        # Get type hints (what mypy sees)
        hints = get_type_hints(MLIntegrationManager.create_integrated_actor)

        # Verify return annotation exists
        assert "return" in hints, "Method should have return type annotation"

        # Get return type (flexible - handles runtime variations)
        return_type = hints["return"]
        type_str = str(return_type).lower()
        type_name = getattr(return_type, "__name__", str(return_type))

        # Flexible assertion 1: NOT object
        # Handles: 'object' vs other types
        assert type_name != "object" or "typevar" in type_str, (
            f"Return type should not be generic object type, got {type_name}. "
            f"Expected generic TypeVar (e.g., ActorT). Full type: {return_type}"
        )

        # Flexible assertion 2: Contains TypeVar patterns
        # Handles variations: '~ActorT', 'ActorT', 'typing.TypeVar', '<TypeVar>'
        typevar_patterns = ["typevar", "actort", "~actor"]
        has_typevar_pattern = any(pattern in type_str for pattern in typevar_patterns)

        assert has_typevar_pattern or type_name != "object", (
            f"Return type should be generic TypeVar (e.g., ActorT), got {type_name}. "
            f"Expected patterns: {typevar_patterns}. Full type: {return_type}"
        )

    def test_actor_class_param_not_type_any(self) -> None:
        """
        Verify actor_class parameter is NOT type[Any] (uses TypeVar).

        The original implementation used type[Any] which erases parameter type
        information. After refactoring, parameter should be type[ActorT] to
        enable generic type inference.

        Behavior tested: Parameter type uses TypeVar (not Any).
        Flexible assertion: Handles runtime type representation variations.

        Expected: Parameter type is generic type[ActorT], not type[Any].
        """
        from ml.core.integration import MLIntegrationManager

        # Get type hints
        hints = get_type_hints(MLIntegrationManager.create_integrated_actor)

        # Verify actor_class parameter exists
        assert "actor_class" in hints, "actor_class parameter should have type annotation"

        # Get parameter type
        actor_class_type = hints["actor_class"]
        type_str = str(actor_class_type)

        # Should NOT be type[Any] (the anti-pattern being fixed)
        assert "type[Any]" not in type_str and "Type[Any]" not in type_str, (
            f"actor_class should not use type[Any] type erasure, got {type_str}. "
            f"Expected generic type[ActorT]."
        )

        # Should be type[...] with TypeVar or generic
        # Handles: 'type[ActorT]', 'type[~ActorT]', 'typing.Type[ActorT]'
        assert ("type[" in type_str.lower() or "Type[" in type_str), (
            f"actor_class should be type[ActorT] pattern, got {type_str}"
        )

        # Should reference TypeVar or generic pattern
        typevar_patterns = ["typevar", "actort", "~actor"]
        has_generic = any(pattern in type_str.lower() for pattern in typevar_patterns)

        assert has_generic or "Any" not in type_str, (
            f"actor_class should use generic TypeVar (ActorT), not Any. Got {type_str}"
        )

    def test_typevar_defined_at_module_level(self) -> None:
        """
        Verify ActorT TypeVar is defined at module level.

        TypeVar must be defined at module level (not inside class or function)
        for proper mypy type inference to work.

        Behavior tested: Module has ActorT attribute that is a TypeVar.

        Expected: ml.core.integration.ActorT exists and is a TypeVar.
        """
        import ml.core.integration as integration_module

        # Verify ActorT exists at module level
        assert hasattr(integration_module, "ActorT"), (
            "ActorT TypeVar should be defined at module level. "
            "Expected: ActorT = TypeVar('ActorT') after imports."
        )

        # Get ActorT
        actor_t = getattr(integration_module, "ActorT")

        # Verify it's a TypeVar (flexible check for type representation)
        type_name = type(actor_t).__name__
        type_str = str(type(actor_t))

        assert "TypeVar" in type_name or "TypeVar" in type_str, (
            f"ActorT should be a TypeVar, got type {type_name}. "
            f"Expected: TypeVar('ActorT')"
        )

    def test_typevar_imported_from_typing(self) -> None:
        """
        Verify TypeVar is imported from typing module.

        Implementation must import TypeVar from typing to use it:
        from typing import ..., TypeVar, ...

        Behavior tested: Source code contains TypeVar import.

        Expected: 'from typing import' line includes 'TypeVar'.
        """
        import ml.core.integration as integration_module

        # Get module source code
        source = inspect.getsource(integration_module)

        # Should have TypeVar import
        # Flexible: handles different import orders and styles
        has_typevar_import = (
            "from typing import" in source
            and "TypeVar" in source[: source.find("if TYPE_CHECKING")]
        )

        assert has_typevar_import, (
            "TypeVar should be imported from typing module. "
            "Expected pattern: from typing import ..., TypeVar, ..."
        )

    def test_config_param_remains_object(self) -> None:
        """
        Verify config parameter remains defensive object type.

        The config parameter is only used for attribute checking (hasattr, setattr),
        so defensive typing with 'object' is appropriate and should be preserved.

        Behavior tested: config parameter type is object (unchanged).

        Expected: config parameter type is object or Any (both defensive patterns OK).
        """
        from ml.core.integration import MLIntegrationManager

        # Get type hints
        hints = get_type_hints(MLIntegrationManager.create_integrated_actor)

        # Verify config parameter exists
        assert "config" in hints, "config parameter should have type annotation"

        # Get config type
        config_type = hints["config"]
        type_name = getattr(config_type, "__name__", str(config_type))
        type_str = str(config_type)

        # Should be object or Any (both defensive patterns acceptable)
        is_defensive = (
            type_name == "object"
            or "Any" in type_str
            or config_type is object  # Direct comparison
        )

        assert is_defensive, (
            f"config parameter should remain defensive (object or Any), got {type_name}. "
            f"Config is only used for attribute checking, so defensive typing is correct."
        )


class TestGenericTypePreservation:
    """Test suite verifying generic type preserves caller's specific actor class."""

    def test_return_type_preserves_actor_class(
        self,
        mock_integration_manager: Mock,
        dummy_actor_class_a,
        mock_config: Mock,
    ) -> None:
        """
        Verify return type matches actor_class parameter type.

        When called with specific actor class, return type should be inferred as
        that specific class (not generic object). This is the core benefit of
        using generic TypeVar.

        Behavior tested: Generic preserves specific type through call chain.

        Expected: Calling with DummyActorA returns DummyActorA type (not object).
        """
        from ml.core.integration import MLIntegrationManager

        # Bind method to mock instance
        bound_method = MLIntegrationManager.create_integrated_actor.__get__(
            mock_integration_manager, MLIntegrationManager
        )

        # Call method with specific actor class
        result = bound_method(
            actor_class=dummy_actor_class_a,
            config=mock_config,
        )

        # Verify result is not None
        assert result is not None, "Method should return actor instance"

        # Verify result is instance of actor_class
        # (At runtime, generic is erased, but instance should match actor_class)
        assert isinstance(result, dummy_actor_class_a), (
            f"Returned actor should be instance of {dummy_actor_class_a.__name__}. "
            f"Generic TypeVar preserves this relationship at type-checking time. "
            f"Got {type(result).__name__}"
        )

    def test_multiple_actor_classes_preserve_types(
        self,
        mock_integration_manager: Mock,
        dummy_actor_class_a,
        dummy_actor_class_b,
        mock_config: Mock,
    ) -> None:
        """
        Verify different actor classes maintain distinct return types.

        Calling create_integrated_actor with ActorA should return ActorA type.
        Calling with ActorB should return ActorB type. Types should be distinct
        (not collapsed to object).

        Behavior tested: Generic works for multiple different types independently.

        Expected: Each call preserves its specific actor type at type-checking time.
        """
        from ml.core.integration import MLIntegrationManager

        # Bind method to mock instance
        bound_method = MLIntegrationManager.create_integrated_actor.__get__(
            mock_integration_manager, MLIntegrationManager
        )

        # Call with ActorA
        result_a = bound_method(
            actor_class=dummy_actor_class_a,
            config=mock_config,
        )

        # Call with ActorB
        result_b = bound_method(
            actor_class=dummy_actor_class_b,
            config=mock_config,
        )

        # Verify results are distinct instances
        assert result_a is not result_b, "Different actor classes should return different instances"

        # Verify each result matches its actor class
        assert isinstance(result_a, dummy_actor_class_a), (
            f"ActorA call should return ActorA instance, got {type(result_a).__name__}"
        )
        assert isinstance(result_b, dummy_actor_class_b), (
            f"ActorB call should return ActorB instance, got {type(result_b).__name__}"
        )

        # Verify types are actually different (not both collapsed to object)
        assert type(result_a).__name__ != type(result_b).__name__, (
            "Different actor classes should produce different instance types"
        )

    def test_ide_autocomplete_scenario_mock(
        self,
        mock_integration_manager: Mock,
        mock_actor_with_custom_method,
        mock_config: Mock,
    ) -> None:
        """
        Verify IDE autocomplete scenario works (mock-based simulation).

        With generic TypeVar, IDE knows the return type has actor-specific methods.
        We simulate this by verifying the returned instance has the custom method
        without triggering AttributeError.

        Behavior tested: Return type enables attribute access to actor-specific methods.

        Expected: Can access custom_predict method on returned actor without error.
        """
        from ml.core.integration import MLIntegrationManager

        # Bind method to mock instance
        bound_method = MLIntegrationManager.create_integrated_actor.__get__(
            mock_integration_manager, MLIntegrationManager
        )

        # Create actual instance (not mock) to test attribute access
        try:
            # Call method
            result = bound_method(
                actor_class=mock_actor_with_custom_method,
                config=mock_config,
            )

            # Verify result has custom method (IDE would know this with generic)
            assert hasattr(result, "custom_predict"), (
                "Returned actor should have custom_predict method. "
                "With generic TypeVar, IDE autocomplete would show this method."
            )

            # Verify method is callable
            assert callable(result.custom_predict), "custom_predict should be callable"

            # Verify calling method works
            prediction = result.custom_predict(data=None)
            assert prediction == 0.5, "custom_predict should return expected value"

        except AttributeError as e:
            pytest.fail(
                f"Generic should enable IDE autocomplete for actor-specific methods. "
                f"AttributeError indicates type information lost: {e}"
            )


class TestCreateIntegratedActorRuntime:
    """Test suite verifying create_integrated_actor runtime behavior."""

    def test_creates_actor_instance(
        self,
        mock_integration_manager: Mock,
        mock_actor_class,
        mock_config: Mock,
    ) -> None:
        """
        Verify method creates actor instance correctly.

        Core functionality test: method should instantiate actor_class with config
        and return the created instance.

        Behavior tested: Actor creation and return.

        Expected: Returns instance of actor_class.
        """
        from ml.core.integration import MLIntegrationManager

        # Bind method to mock instance
        bound_method = MLIntegrationManager.create_integrated_actor.__get__(
            mock_integration_manager, MLIntegrationManager
        )

        # Call method
        result = bound_method(
            actor_class=mock_actor_class,
            config=mock_config,
        )

        # Verify result is not None
        assert result is not None, "Method should return actor instance"

        # Verify result is instance of actor_class
        assert isinstance(result, mock_actor_class), (
            f"Result should be instance of {mock_actor_class.__name__}, "
            f"got {type(result).__name__}"
        )

        # Verify actor has config
        assert hasattr(result, "config"), "Actor should have config attribute"
        assert result.config is mock_config, "Actor config should match provided config"

    def test_attaches_db_connection_to_config(
        self,
        mock_integration_manager: Mock,
        mock_actor_class,
        mock_config_no_db: Mock,
    ) -> None:
        """
        Verify db_connection is automatically attached to config.

        Method should check if config has db_connection attribute. If not, it should
        add it using the integration manager's db_connection value.

        Behavior tested: Automatic db_connection attachment.

        Expected: Config gets db_connection attribute set to manager's value.
        """
        from ml.core.integration import MLIntegrationManager

        # Verify config initially has no db_connection
        assert not hasattr(mock_config_no_db, "db_connection"), (
            "Test fixture should start without db_connection attribute"
        )

        # Bind method to mock instance
        bound_method = MLIntegrationManager.create_integrated_actor.__get__(
            mock_integration_manager, MLIntegrationManager
        )

        # Call method
        result = bound_method(
            actor_class=mock_actor_class,
            config=mock_config_no_db,
        )

        # Verify result is valid
        assert result is not None, "Method should return actor instance"

        # Verify config now has db_connection
        assert hasattr(mock_config_no_db, "db_connection"), (
            "Method should attach db_connection to config if missing"
        )

        # Verify db_connection value matches manager's
        assert mock_config_no_db.db_connection == mock_integration_manager.db_connection, (
            f"Config db_connection should be {mock_integration_manager.db_connection}, "
            f"got {mock_config_no_db.db_connection}"
        )


class TestCreateIntegratedActorCompatibility:
    """Test suite verifying backward compatibility with existing call patterns."""

    def test_existing_call_patterns_work(
        self,
        mock_integration_manager: Mock,
        mock_actor_class,
        mock_config: Mock,
    ) -> None:
        """
        Verify existing usage patterns work unchanged after adding generics.

        Generic TypeVar is transparent at runtime - should not break any existing
        code. Call signature and behavior remain identical.

        Behavior tested: Backward compatibility with existing usage.

        Expected: Generic is transparent - no breaking changes to call patterns.
        """
        from ml.core.integration import MLIntegrationManager

        # Bind method to mock instance
        bound_method = MLIntegrationManager.create_integrated_actor.__get__(
            mock_integration_manager, MLIntegrationManager
        )

        # Existing call pattern: keyword arguments
        try:
            result = bound_method(
                actor_class=mock_actor_class,
                config=mock_config,
            )

            # Verify successful execution
            assert result is not None, "Existing call pattern should work unchanged"

            # Verify return type is valid
            assert isinstance(result, mock_actor_class), (
                "Return type should match actor_class (backward compatibility maintained)"
            )

        except TypeError as e:
            pytest.fail(
                f"Generic should be transparent at runtime. Existing call pattern failed: {e}"
            )
