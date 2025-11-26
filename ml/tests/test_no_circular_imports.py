"""Test that circular imports are eliminated.

This module verifies Phase 0.1 of the refactoring plan: ensuring that
ml.stores and ml.actors can be imported independently without circular
dependency issues.

The circular dependency that was resolved:
- actors.base imports from ml.stores (for protocols and concrete stores)
- ml.stores should NOT import from ml.actors at runtime
- Any actor examples in docstrings should use TYPE_CHECKING pattern
"""

import importlib
import sys


def test_stores_import_standalone():
    """Stores module can be imported without actors."""
    # Save original module state
    original_modules = sys.modules.copy()

    try:
        # Clean slate - remove any previously imported ml modules
        mods_to_remove = [k for k in sys.modules.keys() if k.startswith("ml.")]
        for mod in mods_to_remove:
            del sys.modules[mod]

        # Import stores first - should not trigger actor imports
        try:
            import ml.stores

            assert ml.stores is not None
            # Verify actors module was NOT imported as side effect
            assert "ml.actors" not in sys.modules, "ml.stores should not import ml.actors at runtime"
            assert (
                "ml.actors.base" not in sys.modules
            ), "ml.stores should not import ml.actors.base at runtime"
        except ImportError as e:
            # Allow databento import errors (optional dependency)
            if "databento" not in str(e):
                raise
    finally:
        # Restore original module state to prevent test pollution
        # Remove any new modules that were loaded
        current_modules = set(sys.modules.keys())
        original_keys = set(original_modules.keys())
        new_modules = current_modules - original_keys
        for mod in new_modules:
            if mod.startswith("ml."):
                del sys.modules[mod]

        # Restore original ml.* modules
        for mod in original_keys:
            if mod.startswith("ml."):
                sys.modules[mod] = original_modules[mod]


def test_actors_import_standalone():
    """Actors module can be imported without stores (tests runtime independence)."""
    # Save original module state
    original_modules = sys.modules.copy()

    try:
        # Clean slate
        mods_to_remove = [k for k in sys.modules.keys() if k.startswith("ml.")]
        for mod in mods_to_remove:
            del sys.modules[mod]

        # Import actors first - will import stores as dependency (this is expected)
        try:
            import ml.actors

            assert ml.actors is not None
            # It's OK for actors to import stores (that's the dependency direction we want)
            # We just need to verify no circular dependency errors occurred
        except ImportError as e:
            # Allow databento import errors (optional dependency)
            if "databento" not in str(e):
                raise
    finally:
        # Restore original module state to prevent test pollution
        current_modules = set(sys.modules.keys())
        original_keys = set(original_modules.keys())
        new_modules = current_modules - original_keys
        for mod in new_modules:
            if mod.startswith("ml."):
                del sys.modules[mod]

        for mod in original_keys:
            if mod.startswith("ml."):
                sys.modules[mod] = original_modules[mod]


def test_import_order_independence():
    """Imports work in either order without circular dependency errors."""
    # Save original module state
    original_modules = sys.modules.copy()

    try:
        # Test order 1: stores then actors
        mods_to_remove = [k for k in sys.modules.keys() if k.startswith("ml.")]
        for mod in mods_to_remove:
            del sys.modules[mod]

        try:
            import ml.stores
            import ml.actors

            assert ml.stores is not None
            assert ml.actors is not None
        except ImportError as e:
            # Allow databento import errors (optional dependency)
            if "databento" not in str(e):
                raise

        # Restore for second test
        current_modules = set(sys.modules.keys())
        original_keys = set(original_modules.keys())
        new_modules = current_modules - original_keys
        for mod in new_modules:
            if mod.startswith("ml."):
                del sys.modules[mod]

        for mod in original_keys:
            if mod.startswith("ml."):
                sys.modules[mod] = original_modules[mod]

        # Test order 2: actors then stores
        mods_to_remove = [k for k in sys.modules.keys() if k.startswith("ml.")]
        for mod in mods_to_remove:
            del sys.modules[mod]

        try:
            import ml.actors
            import ml.stores

            assert ml.actors is not None
            assert ml.stores is not None
        except ImportError as e:
            # Allow databento import errors (optional dependency)
            if "databento" not in str(e):
                raise
    finally:
        # Final restore of original module state
        current_modules = set(sys.modules.keys())
        original_keys = set(original_modules.keys())
        new_modules = current_modules - original_keys
        for mod in new_modules:
            if mod.startswith("ml."):
                del sys.modules[mod]

        for mod in original_keys:
            if mod.startswith("ml."):
                sys.modules[mod] = original_modules[mod]


def test_no_runtime_actor_import_in_stores():
    """Verify ml.stores.__init__.py has no runtime import of BaseMLInferenceActor."""
    import ast
    from pathlib import Path

    stores_init = Path(__file__).parent.parent / "stores" / "__init__.py"
    source = stores_init.read_text()
    tree = ast.parse(source)

    # Get all actual import statements (not in docstrings)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("ml.actors"):
                for alias in node.names:
                    imports.append((node.module, alias.name if hasattr(alias, "name") else str(alias)))

    # Should have NO imports from ml.actors
    assert (
        not imports
    ), f"ml/stores/__init__.py should not import from ml.actors, found: {imports}"


def test_stores_public_api_has_no_actor_types():
    """Verify that ml.stores.__all__ does not export any actor types."""
    try:
        import ml.stores

        # Get the public API
        public_api = getattr(ml.stores, "__all__", [])

        # Check that no actor-related types are exported
        actor_keywords = ["Actor", "BaseMLInferenceActor", "EnhancedMLInferenceActor"]
        exported_actors = [name for name in public_api if any(kw in name for kw in actor_keywords)]

        assert (
            not exported_actors
        ), f"ml.stores should not export actor types, found: {exported_actors}"
    except ImportError as e:
        # Allow databento import errors (optional dependency)
        if "databento" not in str(e):
            raise


def test_stores_not_reexported_from_actors():
    """Store classes should not be importable from actors module at runtime.

    This test verifies Phase 0.3: removal of concrete store re-exports from
    ml/actors/base.py. The TYPE_CHECKING imports should remain for type hints,
    but runtime imports of concrete stores should not be accessible.
    """
    try:
        import ml.actors.base as actors_base

        # These should NOT be accessible at runtime
        assert not hasattr(
            actors_base, "FeatureStore"
        ), "FeatureStore should not be re-exported from actors.base"
        assert not hasattr(
            actors_base, "ModelStore"
        ), "ModelStore should not be re-exported from actors.base"
        assert not hasattr(
            actors_base, "StrategyStore"
        ), "StrategyStore should not be re-exported from actors.base"
        assert not hasattr(
            actors_base, "DataStore"
        ), "DataStore should not be re-exported from actors.base"
    except ImportError as e:
        # Allow databento import errors (optional dependency)
        if "databento" not in str(e):
            raise


def test_stores_available_from_stores_module():
    """Store classes should be imported from ml.stores.

    Verifies that stores are accessible from their proper module location
    after removing re-exports from actors.
    """
    try:
        from ml.stores import DataStore
        from ml.stores import FeatureStore
        from ml.stores import ModelStore
        from ml.stores import StrategyStore

        assert FeatureStore is not None
        assert ModelStore is not None
        assert StrategyStore is not None
        assert DataStore is not None
    except ImportError as e:
        # Allow databento import errors (optional dependency)
        if "databento" not in str(e):
            raise
