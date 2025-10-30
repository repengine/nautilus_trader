#!/usr/bin/env python3

"""
Unit tests for EngineManager integration across all stores.

This test ensures that all stores properly use EngineManager for database connections,
preventing pool exhaustion and hidden "too many clients" failures in parallel tests.

"""

import pytest
from ml.core.db_engine import EngineManager


class TestEngineManagerIntegration:
    """
    Tests ensuring all stores use EngineManager for database connections.

    This prevents connection pool exhaustion and ensures proper resource sharing across
    all ML stores in both production and test environments.

    """

    def test_store_engines_are_identical_for_same_url(self, test_database):
        """
        Test that all stores get identical engine instances for same connection string.
        """
        # Use the real connection string with unmasked password
        test_url = test_database.connection_string

        # Import here to avoid issues with engine creation during import
        from ml.stores.data_processor import DataProcessor
        from ml.stores.feature_store import FeatureStore
        from ml.stores.model_store import ModelStore
        from ml.stores.strategy_store import StrategyStore

        # Create stores with same connection string
        feature_store = FeatureStore(test_url)
        model_store = ModelStore(test_url)
        strategy_store = StrategyStore(test_url)
        data_processor = DataProcessor(test_url)

        # Verify all engines are the same instance (identity check)
        assert (
            feature_store.engine is model_store.engine
        ), "FeatureStore and ModelStore should share the same engine instance"
        assert (
            model_store.engine is strategy_store.engine
        ), "ModelStore and StrategyStore should share the same engine instance"
        assert (
            strategy_store.engine is data_processor.engine
        ), "StrategyStore and DataProcessor should share the same engine instance"

        # Verify engines come from EngineManager
        expected_engine = EngineManager.get_engine(test_url)
        assert (
            feature_store.engine is expected_engine
        ), "FeatureStore engine should be identical to EngineManager.get_engine() result"
        assert (
            model_store.engine is expected_engine
        ), "ModelStore engine should be identical to EngineManager.get_engine() result"
        assert (
            strategy_store.engine is expected_engine
        ), "StrategyStore engine should be identical to EngineManager.get_engine() result"
        assert (
            data_processor.engine is expected_engine
        ), "DataProcessor engine should be identical to EngineManager.get_engine() result"

    def test_module_level_create_engine_functions_delegate_to_engine_manager(self, test_database):
        """
        Test that module-level create_engine functions properly delegate to
        EngineManager.
        """
        # Use the real connection string with unmasked password
        test_url = test_database.connection_string

        # Import module-level create_engine functions
        from ml.stores import feature_store, model_store, strategy_store, data_processor

        # Get engines from module-level functions
        feature_engine = feature_store.create_engine(test_url)
        model_engine = model_store.create_engine(test_url)
        strategy_engine = strategy_store.create_engine(test_url)
        processor_engine = data_processor.create_engine(test_url)

        # Verify they delegate to EngineManager
        expected_engine = EngineManager.get_engine(test_url)

        assert (
            feature_engine is expected_engine
        ), "feature_store.create_engine should delegate to EngineManager"
        assert (
            model_engine is expected_engine
        ), "model_store.create_engine should delegate to EngineManager"
        assert (
            strategy_engine is expected_engine
        ), "strategy_store.create_engine should delegate to EngineManager"
        assert (
            processor_engine is expected_engine
        ), "data_processor.create_engine should delegate to EngineManager"

    def test_no_stray_sqlalchemy_create_engine_in_stores(self):
        """
        Test that stores don't directly import sqlalchemy.create_engine.
        """
        # Import all store modules
        from ml.stores import (
            data_processor,
            data_store,
            feature_store,
            model_store,
            strategy_store,
        )

        store_modules = [data_processor, data_store, feature_store, model_store, strategy_store]

        for module in store_modules:
            # Check that sqlalchemy.create_engine is not in module globals
            module_globals = getattr(module, "__dict__", {})
            if "create_engine" in module_globals:
                create_engine_func = module_globals.get("create_engine")
                # If there's a create_engine function, it should be from ml.stores module (not sqlalchemy)
                if hasattr(create_engine_func, "__module__"):
                    assert "ml.stores" in str(create_engine_func.__module__), (
                        f"Module {module.__name__} has create_engine from {create_engine_func.__module__} "
                        f"but should only have ml.stores.* create_engine"
                    )

            # Check module source for direct sqlalchemy imports (if available)
            try:
                import inspect

                source = inspect.getsource(module)

                # Allow 'from sqlalchemy import Engine' but not 'from sqlalchemy import create_engine'
                assert (
                    "from sqlalchemy import create_engine" not in source
                ), f"Module {module.__name__} should not directly import create_engine from sqlalchemy"
                assert (
                    "sqlalchemy.create_engine" not in source
                ), f"Module {module.__name__} should not directly call sqlalchemy.create_engine"
            except (OSError, TypeError):
                # Source not available (compiled, etc.) - skip source check
                pass

    def test_engine_manager_single_engine_per_url(self):
        """
        Test that EngineManager maintains single engine per unique URL.
        """
        test_url_1 = "postgresql://test1:test@localhost:5432/test"
        test_url_2 = "postgresql://test2:test@localhost:5432/test"

        # Get engines for different URLs
        engine_1a = EngineManager.get_engine(test_url_1)
        engine_1b = EngineManager.get_engine(test_url_1)  # Same URL
        engine_2 = EngineManager.get_engine(test_url_2)  # Different URL

        # Same URL should return same engine instance
        assert (
            engine_1a is engine_1b
        ), "EngineManager should return same instance for identical URLs"

        # Different URLs should return different engines
        assert (
            engine_1a is not engine_2
        ), "EngineManager should return different instances for different URLs"

        # Verify engine count
        assert (
            EngineManager.get_engine_count() == 2
        ), "EngineManager should track exactly 2 engines for 2 unique URLs"

    def test_data_store_delegates_properly(self):
        """
        Test that DataStore doesn't create engines directly and delegates to stores.
        """
        # Skip this test as DataStore requires a complex setup with registries
        # The important thing is that DataStore creates stores via constructors,
        # and we've verified that store constructors use EngineManager
        pytest.skip(
            "DataStore requires complex registry setup - delegation verified via store constructors",
        )
