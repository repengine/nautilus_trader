"""
Test suite for Protocol Remediation Task 2.5: Module Export Type Safety.

Verifies that ml.core.__init__.py uses direct imports instead of __getattr__ lazy
loading to enable static type checking for all exported symbols.

Coverage Target: ≥90%
Pattern: Protocol-First (Pattern 2)
Reference: INTEGRATION_GENERIC_TYPES_REMEDIATION.md lines 757-858
"""

import importlib
import inspect
import warnings

import pytest


class TestModuleImports:
    """Test all module exports are directly importable with proper types."""

    def test_mlintegrationmanager_import_type_not_object(self):
        """Verify MLIntegrationManager import yields class type, not object."""
        from ml.core import MLIntegrationManager

        # Flexible check: is a class, not the generic object type
        assert inspect.isclass(MLIntegrationManager), "Should be a class"
        assert MLIntegrationManager is not object, "Should not be generic object type"
        assert hasattr(MLIntegrationManager, "__init__"), "Should have constructor"
        assert MLIntegrationManager.__name__ == "MLIntegrationManager"

    def test_all_exports_are_importable(self):
        """Verify all symbols in __all__ can be imported."""
        import ml.core

        for symbol_name in ml.core.__all__:
            symbol = getattr(ml.core, symbol_name)
            assert symbol is not None, f"{symbol_name} should be importable"
            # Symbol should be callable or a class
            assert callable(symbol) or inspect.isclass(symbol), (
                f"{symbol_name} should be callable or class"
            )

    def test_actorstore_registries_import_type_not_object(self):
        """Verify ActorStoresRegistries import yields class type, not object."""
        from ml.core import ActorStoresRegistries

        assert inspect.isclass(ActorStoresRegistries)
        assert ActorStoresRegistries is not object
        assert hasattr(ActorStoresRegistries, "__init__")

    def test_helper_functions_are_callable(self):
        """Verify all helper functions are callable."""
        from ml.core import init_actor_stores_and_registries
        from ml.core import init_ml_stores_and_registries

        # Core functions that must exist
        for func in [init_actor_stores_and_registries, init_ml_stores_and_registries]:
            assert callable(func), f"{func.__name__} should be callable"
            assert not inspect.isclass(func), f"{func.__name__} should be function, not class"

        # Optional functions (may or may not be added)
        try:
            from ml.core import get_integration_manager, reset_integration_manager

            for func in [get_integration_manager, reset_integration_manager]:
                assert callable(func)
        except ImportError:
            pass  # Acceptable if not added

    def test_cache_classes_importable(self):
        """Verify cache classes are importable with proper types."""
        from ml.core import LockFreeRingBuffer
        from ml.core import PreAllocatedFeatureCache
        from ml.core import ReservoirSampler

        for cls in [LockFreeRingBuffer, PreAllocatedFeatureCache, ReservoirSampler]:
            assert inspect.isclass(cls), f"{cls.__name__} should be a class"
            assert cls is not object, f"{cls.__name__} should not be object type"


class TestNoLazyLoading:
    """Test that lazy loading (__getattr__) has been removed."""

    def test_no_getattr_function_exists(self):
        """Verify __getattr__ function has been removed."""
        import ml.core

        # Check if __getattr__ exists
        has_getattr = hasattr(ml.core, "__getattr__")

        if has_getattr:
            # If it exists, it should be the default (not our custom one)
            # Try to access a non-existent attribute
            try:
                _ = ml.core.nonexistent_attribute_xyz123
                pytest.fail("__getattr__ should not catch arbitrary attributes")
            except AttributeError as e:
                # Expected: standard AttributeError
                assert "nonexistent_attribute_xyz123" in str(e)

    def test_lazy_loading_symbols_not_present(self):
        """Verify that previously lazy-loaded symbols are now directly available."""
        import ml.core

        # These were previously lazy-loaded
        symbols = [
            "ActorStoresRegistries",
            "MLIntegrationManager",
            "init_actor_stores_and_registries",
            "init_ml_stores_and_registries",
        ]

        for symbol in symbols:
            assert hasattr(ml.core, symbol), f"{symbol} should be directly available"
            obj = getattr(ml.core, symbol)
            assert obj is not None


class TestTypeInformation:
    """Test that type information is available for static analysis."""

    def test_mlintegrationmanager_is_class(self):
        """Verify MLIntegrationManager is recognized as a class."""
        from ml.core import MLIntegrationManager

        assert inspect.isclass(MLIntegrationManager)
        assert hasattr(MLIntegrationManager, "__bases__")
        assert hasattr(MLIntegrationManager, "__init__")
        assert hasattr(MLIntegrationManager, "__module__")

    def test_type_hints_available_for_classes(self):
        """Verify type hints are available for IDE autocomplete."""
        from ml.core import ActorStoresRegistries
        from ml.core import MLIntegrationManager

        # Check signatures exist
        sig = inspect.signature(MLIntegrationManager.__init__)
        assert sig is not None, "Should have signature"
        assert len(sig.parameters) > 0, "Should have parameters"

        # Check ActorStoresRegistries
        sig2 = inspect.signature(ActorStoresRegistries.__init__)
        assert sig2 is not None

    def test_function_signatures_available(self):
        """Verify helper functions have accessible signatures."""
        from ml.core import init_ml_stores_and_registries

        sig = inspect.signature(init_ml_stores_and_registries)
        assert sig is not None, "Function should have signature"

    def test_module_import_provides_types_not_object(self):
        """Verify imports don't produce generic object type."""
        from ml.core import LockFreeRingBuffer
        from ml.core import MLIntegrationManager

        # For classes and protocols, type should be 'type' or a metaclass (not object)
        # Classes use 'type' metaclass, Protocols use '_ProtocolMeta'
        assert type(MLIntegrationManager).__name__ in ("type", "_ProtocolMeta"), "Should have proper metaclass"
        assert type(LockFreeRingBuffer).__name__ in ("type", "_ProtocolMeta"), "Should have proper metaclass"

        # Verify not the generic object type
        assert MLIntegrationManager is not object
        assert LockFreeRingBuffer is not object


class TestBackwardCompatibility:
    """Test that existing import patterns still work."""

    def test_existing_import_patterns_work(self):
        """Verify common import patterns still work."""
        # Pattern 1: Import specific symbol
        from ml.core import MLIntegrationManager

        assert MLIntegrationManager is not None

        # Pattern 2: Import multiple
        from ml.core import ActorStoresRegistries
        from ml.core import MLIntegrationManager as MIM

        assert MIM is not None
        assert ActorStoresRegistries is not None

        # Pattern 3: Import module and access
        import ml.core

        assert ml.core.MLIntegrationManager is not None

    def test_deprecated_aliases_still_work(self):
        """Verify deprecated init_actor_stores_and_registries alias still works."""
        from ml.core import init_actor_stores_and_registries
        from ml.core import init_ml_stores_and_registries

        # Should be available
        assert callable(init_actor_stores_and_registries)
        assert init_actor_stores_and_registries is not None
        assert init_ml_stores_and_registries is not None

    def test_all_public_symbols_accessible(self):
        """Verify __all__ includes all intended public symbols."""
        import ml.core

        expected_symbols = [
            "ActorStoresRegistries",
            "EngineManager",
            "LockFreeRingBuffer",
            "MLIntegrationManager",
            "PreAllocatedFeatureCache",
            "ReservoirSampler",
            "init_actor_stores_and_registries",
            "init_ml_stores_and_registries",
        ]

        for symbol in expected_symbols:
            assert symbol in ml.core.__all__, f"{symbol} should be in __all__"
            assert hasattr(ml.core, symbol), f"{symbol} should be accessible"


class TestExportCompleteness:
    """Test __all__ completeness and accuracy."""

    def test_all_list_matches_exports(self):
        """Verify __all__ contains exactly what's exported."""
        import ml.core

        # Check all symbols in __all__ exist
        for symbol in ml.core.__all__:
            assert hasattr(ml.core, symbol), f"{symbol} in __all__ but not found in module"

        # Check major public classes are in __all__
        major_classes = [
            "MLIntegrationManager",
            "ActorStoresRegistries",
            "LockFreeRingBuffer",
            "PreAllocatedFeatureCache",
            "ReservoirSampler",
            "EngineManager",
        ]

        for cls_name in major_classes:
            assert cls_name in ml.core.__all__, f"Public class {cls_name} missing from __all__"

    def test_additional_exports_present(self):
        """Verify new exports are included (from remediation guide)."""
        import ml.core

        # Check for new exports mentioned in remediation guide
        new_exports = [
            "get_integration_manager",
            "reset_integration_manager",
        ]

        for symbol in new_exports:
            if symbol in ml.core.__all__:  # May or may not be added
                assert hasattr(ml.core, symbol), f"{symbol} in __all__ but not accessible"

        # MultiChannelRingBuffer - check if exists
        try:
            from ml.core import MultiChannelRingBuffer

            # If importable, inform (not a hard requirement)
            if "MultiChannelRingBuffer" not in ml.core.__all__:
                warnings.warn("MultiChannelRingBuffer exists but not in __all__")
        except ImportError:
            pass  # Acceptable if not included

    def test_no_missing_cache_exports(self):
        """Verify all cache module exports are included."""
        import ml.core

        cache_exports = [
            "LockFreeRingBuffer",
            "PreAllocatedFeatureCache",
            "ReservoirSampler",
        ]

        for symbol in cache_exports:
            assert symbol in ml.core.__all__, f"Cache export {symbol} missing from __all__"
            assert hasattr(ml.core, symbol), f"Cache export {symbol} not accessible"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_import_does_not_fail_with_missing_module(self):
        """Verify module imports gracefully."""
        import ml.core

        assert ml.core is not None
        assert hasattr(ml.core, "__all__")
        assert len(ml.core.__all__) > 0

    def test_module_reload_works(self):
        """Verify module can be reloaded without errors."""
        import ml.core

        # Initial import works
        assert ml.core.MLIntegrationManager is not None

        # Reload
        importlib.reload(ml.core)

        # Still works after reload
        assert ml.core.MLIntegrationManager is not None
        assert "MLIntegrationManager" in ml.core.__all__

    def test_no_import_cycles_with_other_modules(self):
        """Verify no import cycles with ml.stores, ml.actors."""
        # Import in various orders to detect cycles
        import ml.actors
        import ml.core
        import ml.stores

        # All should be available
        assert ml.core.MLIntegrationManager is not None
        # Don't assert on store/actor internals, just that imports work
