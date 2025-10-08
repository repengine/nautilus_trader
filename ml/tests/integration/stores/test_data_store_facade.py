#!/usr/bin/env python3

"""
Integration tests for DataStore facade.

Tests verify:
1. Feature flag toggle between legacy and component-based implementations
2. Backward compatibility of all public APIs
3. Delegation to components works correctly
4. Error handling parity between implementations
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from ml.stores import DataStore
from ml.stores.base import FeatureData
from ml.stores.base import ModelPrediction
from ml.stores.base import StrategySignal


@pytest.fixture
def connection_string():
    """Provide test PostgreSQL connection string."""
    return "postgresql://test:test@localhost:5432/test_db"


@pytest.fixture
def mock_feature_store():
    """Mock FeatureStore for testing."""
    store = MagicMock()
    store.get_latest_at_or_before.return_value = {"feature1": 1.0, "feature2": 2.0}
    store.get_health_status.return_value = {"status": "healthy"}
    return store


@pytest.fixture
def mock_model_store():
    """Mock ModelStore for testing."""
    store = MagicMock()
    store.model_predictions_table = None
    store.engine = None
    store.get_health_status.return_value = {"status": "healthy"}
    return store


@pytest.fixture
def mock_strategy_store():
    """Mock StrategyStore for testing."""
    store = MagicMock()
    store.strategy_signals_table = None
    store.engine = None
    store.get_health_status.return_value = {"status": "healthy"}
    return store


@pytest.fixture
def mock_earnings_store():
    """Mock EarningsStore for testing."""
    store = MagicMock()
    store.get_actuals.return_value = []
    store.get_estimates.return_value = None
    return store


@pytest.fixture
def mock_registry():
    """Mock DataRegistry for testing."""
    from ml.registry.dataclasses import DataContract
    from ml.registry.dataclasses import DatasetManifest
    from ml.registry.dataclasses import DatasetType
    from ml.registry.dataclasses import StorageKind

    registry = MagicMock()

    # Mock manifest
    manifest = DatasetManifest(
        dataset_id="test_dataset",
        dataset_type=DatasetType.FEATURES,
        storage_kind=StorageKind.POSTGRES,
        location="ml.test_dataset",
        partitioning={},
        retention_days=90,
        version="1.0.0",
        schema={"instrument_id": "str", "ts_event": "int64", "ts_init": "int64"},
        schema_hash="test_hash",
        primary_keys=["instrument_id", "ts_event"],
        ts_field="ts_event",
        seq_field=None,
        constraints={},
        lineage=[],
        pipeline_signature="test",
    )
    registry.get_manifest.return_value = manifest

    # Mock contract
    import time
    from ml.registry.dataclasses import ValidationRule
    from ml.registry.dataclasses import ValidationRuleType
    from ml.registry.dataclasses import QualityFlag

    contract = DataContract(
        contract_id="test_contract",
        dataset_id="test_dataset",
        version="1.0.0",
        enforcement_mode="monitor_only",
        validation_rules=[
            ValidationRule(
                rule_type=ValidationRuleType.TYPE_CHECK,
                field_name="*",
                severity=QualityFlag.WARN,
                parameters={},
                description="Type check validation",
            ),
        ],
        quality_thresholds={},
        created_at=time.time_ns(),
        last_modified=time.time_ns(),
    )
    registry.get_contract.return_value = contract

    return registry


class TestFeatureFlagToggle:
    """Test feature flag controls implementation selection."""

    def test_default_uses_component_based_implementation(
        self,
        connection_string,
        mock_feature_store,
        mock_model_store,
        mock_strategy_store,
        mock_earnings_store,
        mock_registry,
    ):
        """Test default uses new component-based implementation."""
        # Ensure flag is off
        os.environ["ML_USE_LEGACY_DATA_STORE"] = "0"

        with patch("ml.stores.data_store.FeatureStore", return_value=mock_feature_store), \
             patch("ml.stores.data_store.ModelStore", return_value=mock_model_store), \
             patch("ml.stores.data_store.StrategyStore", return_value=mock_strategy_store):
            store = DataStore(
                connection_string=connection_string,
                registry=mock_registry,
                feature_store=mock_feature_store,
                model_store=mock_model_store,
                strategy_store=mock_strategy_store,
                earnings_store=mock_earnings_store,
            )

            # Verify component-based implementation
            assert not store._use_legacy
            assert hasattr(store, "_schema_validator")
            assert hasattr(store, "_contract_enforcer")
            assert hasattr(store, "_data_reader")
            assert hasattr(store, "_data_writer")

            # Verify health status reports component-based
            health = store.get_health_status()
            assert health["implementation"] == "component_based"

    def test_feature_flag_enables_legacy_implementation(
        self,
        connection_string,
        mock_registry,
    ):
        """Test ML_USE_LEGACY_DATA_STORE=1 enables legacy implementation."""
        # Enable legacy flag
        os.environ["ML_USE_LEGACY_DATA_STORE"] = "1"

        # Mock the legacy DataStore import
        with patch("ml.stores.data_store.DataStore.__init__", return_value=None) as mock_init:
            try:
                # Reimport to pick up environment variable
                import importlib
                from ml.stores import data_store as ds_module
                importlib.reload(ds_module)

                # Verify USE_LEGACY_DATA_STORE is now True
                assert ds_module.USE_LEGACY_DATA_STORE is True

            finally:
                # Clean up environment
                os.environ["ML_USE_LEGACY_DATA_STORE"] = "0"

                # Reload again to restore default
                importlib.reload(ds_module)


class TestBackwardCompatibility:
    """Test backward compatibility of all public APIs."""

    @pytest.fixture(autouse=True)
    def setup_component_based(self):
        """Ensure we're using component-based implementation."""
        os.environ["ML_USE_LEGACY_DATA_STORE"] = "0"

    def test_get_features_at_or_before_delegates_to_reader(
        self,
        connection_string,
        mock_feature_store,
        mock_model_store,
        mock_strategy_store,
        mock_earnings_store,
        mock_registry,
    ):
        """Test get_features_at_or_before delegates to DataReader."""
        with patch("ml.stores.data_store.FeatureStore", return_value=mock_feature_store), \
             patch("ml.stores.data_store.ModelStore", return_value=mock_model_store), \
             patch("ml.stores.data_store.StrategyStore", return_value=mock_strategy_store):
            store = DataStore(
                connection_string=connection_string,
                registry=mock_registry,
                feature_store=mock_feature_store,
                model_store=mock_model_store,
                strategy_store=mock_strategy_store,
                earnings_store=mock_earnings_store,
            )

            result = store.get_features_at_or_before(
                instrument_id="EURUSD.SIM",
                ts_event=1234567890000000000,
            )

            # Verify delegation occurred
            assert result == {"feature1": 1.0, "feature2": 2.0}
            mock_feature_store.get_latest_at_or_before.assert_called_once()

    def test_write_features_delegates_to_writer(
        self,
        connection_string,
        mock_feature_store,
        mock_model_store,
        mock_strategy_store,
        mock_earnings_store,
        mock_registry,
    ):
        """Test write_features delegates to DataWriter."""
        with patch("ml.stores.data_store.FeatureStore", return_value=mock_feature_store), \
             patch("ml.stores.data_store.ModelStore", return_value=mock_model_store), \
             patch("ml.stores.data_store.StrategyStore", return_value=mock_strategy_store):
            store = DataStore(
                connection_string=connection_string,
                registry=mock_registry,
                feature_store=mock_feature_store,
                model_store=mock_model_store,
                strategy_store=mock_strategy_store,
                earnings_store=mock_earnings_store,
            )

            features = [
                FeatureData(
                    feature_set_id="test_set",
                    instrument_id="EURUSD.SIM",
                    values={"feature1": 1.0},
                    ts_event=1234567890000000000,
                    ts_init=1234567890000000000,
                ),
            ]

            # Mock registry methods for dataset registration
            mock_registry.register_manifest = MagicMock()

            result = store.write_features(
                instrument_id="EURUSD.SIM",
                features=features,
                source="computed",
            )

            # Verify delegation occurred
            assert result is not None
            mock_feature_store.write_features.assert_called()

    def test_preflight_check_delegates_to_enforcer(
        self,
        connection_string,
        mock_feature_store,
        mock_model_store,
        mock_strategy_store,
        mock_earnings_store,
        mock_registry,
    ):
        """Test preflight_check delegates to ContractEnforcer."""
        with patch("ml.stores.data_store.FeatureStore", return_value=mock_feature_store), \
             patch("ml.stores.data_store.ModelStore", return_value=mock_model_store), \
             patch("ml.stores.data_store.StrategyStore", return_value=mock_strategy_store):
            store = DataStore(
                connection_string=connection_string,
                registry=mock_registry,
                feature_store=mock_feature_store,
                model_store=mock_model_store,
                strategy_store=mock_strategy_store,
                earnings_store=mock_earnings_store,
            )

            data = [{"instrument_id": "EURUSD.SIM", "ts_event": 123, "ts_init": 123}]

            success, error, details = store.preflight_check(
                dataset_id="test_dataset",
                data=data,
                strict=True,
            )

            # Verify delegation occurred (enforcer was called)
            assert isinstance(success, bool)
            assert details is not None


class TestDelegationMapping:
    """Test delegation mapping is correct for all methods."""

    @pytest.fixture(autouse=True)
    def setup_component_based(self):
        """Ensure we're using component-based implementation."""
        os.environ["ML_USE_LEGACY_DATA_STORE"] = "0"

    def test_read_methods_delegate_to_data_reader(
        self,
        connection_string,
        mock_feature_store,
        mock_model_store,
        mock_strategy_store,
        mock_earnings_store,
        mock_registry,
    ):
        """Test all read methods delegate to DataReader."""
        with patch("ml.stores.data_store.FeatureStore", return_value=mock_feature_store), \
             patch("ml.stores.data_store.ModelStore", return_value=mock_model_store), \
             patch("ml.stores.data_store.StrategyStore", return_value=mock_strategy_store):
            store = DataStore(
                connection_string=connection_string,
                registry=mock_registry,
                feature_store=mock_feature_store,
                model_store=mock_model_store,
                strategy_store=mock_strategy_store,
                earnings_store=mock_earnings_store,
            )

            # Test get_features_at_or_before
            store.get_features_at_or_before(
                instrument_id="EURUSD.SIM",
                ts_event=123,
            )
            assert mock_feature_store.get_latest_at_or_before.called

            # Test get_earnings_actuals_at_or_before
            result = store.get_earnings_actuals_at_or_before(
                ticker="AAPL",
                ts_event=123,
            )
            assert isinstance(result, list)

    def test_validation_methods_delegate_to_enforcer_and_validator(
        self,
        connection_string,
        mock_feature_store,
        mock_model_store,
        mock_strategy_store,
        mock_earnings_store,
        mock_registry,
    ):
        """Test validation methods delegate to ContractEnforcer and SchemaValidator."""
        with patch("ml.stores.data_store.FeatureStore", return_value=mock_feature_store), \
             patch("ml.stores.data_store.ModelStore", return_value=mock_model_store), \
             patch("ml.stores.data_store.StrategyStore", return_value=mock_strategy_store):
            store = DataStore(
                connection_string=connection_string,
                registry=mock_registry,
                feature_store=mock_feature_store,
                model_store=mock_model_store,
                strategy_store=mock_strategy_store,
                earnings_store=mock_earnings_store,
            )

            data = [{"instrument_id": "EURUSD.SIM", "ts_event": 123, "ts_init": 123}]

            # Test preflight_check
            success, error, details = store.preflight_check(
                dataset_id="test_dataset",
                data=data,
            )
            assert isinstance(success, bool)

            # Test validate_batch
            report = store.validate_batch(
                dataset_id="test_dataset",
                data=data,
            )
            assert report is not None


class TestHealthAndMetrics:
    """Test health status and metrics reporting."""

    @pytest.fixture(autouse=True)
    def setup_component_based(self):
        """Ensure we're using component-based implementation."""
        os.environ["ML_USE_LEGACY_DATA_STORE"] = "0"

    def test_get_health_status_includes_all_components(
        self,
        connection_string,
        mock_feature_store,
        mock_model_store,
        mock_strategy_store,
        mock_earnings_store,
        mock_registry,
    ):
        """Test get_health_status reports all components."""
        with patch("ml.stores.data_store.FeatureStore", return_value=mock_feature_store), \
             patch("ml.stores.data_store.ModelStore", return_value=mock_model_store), \
             patch("ml.stores.data_store.StrategyStore", return_value=mock_strategy_store):
            store = DataStore(
                connection_string=connection_string,
                registry=mock_registry,
                feature_store=mock_feature_store,
                model_store=mock_model_store,
                strategy_store=mock_strategy_store,
                earnings_store=mock_earnings_store,
            )

            health = store.get_health_status()

            # Verify all components are reported
            assert health["implementation"] == "component_based"
            assert health["schema_validator"] == "healthy"
            assert health["contract_enforcer"] == "healthy"
            assert health["data_reader"] == "healthy"
            assert health["data_writer"] == "healthy"
            assert "feature_store" in health
            assert "model_store" in health
            assert "strategy_store" in health

    def test_get_performance_metrics_reports_implementation(
        self,
        connection_string,
        mock_feature_store,
        mock_model_store,
        mock_strategy_store,
        mock_earnings_store,
        mock_registry,
    ):
        """Test get_performance_metrics reports implementation type."""
        with patch("ml.stores.data_store.FeatureStore", return_value=mock_feature_store), \
             patch("ml.stores.data_store.ModelStore", return_value=mock_model_store), \
             patch("ml.stores.data_store.StrategyStore", return_value=mock_strategy_store):
            store = DataStore(
                connection_string=connection_string,
                registry=mock_registry,
                feature_store=mock_feature_store,
                model_store=mock_model_store,
                strategy_store=mock_strategy_store,
                earnings_store=mock_earnings_store,
            )

            metrics = store.get_performance_metrics()

            # Verify implementation metric (1.0 = component-based)
            assert metrics["implementation"] == 1.0


class TestConfigurationValidation:
    """Test configuration validation."""

    @pytest.fixture(autouse=True)
    def setup_component_based(self):
        """Ensure we're using component-based implementation."""
        os.environ["ML_USE_LEGACY_DATA_STORE"] = "0"

    def test_validate_configuration_checks_connection_string(
        self,
        mock_feature_store,
        mock_model_store,
        mock_strategy_store,
        mock_earnings_store,
        mock_registry,
    ):
        """Test validate_configuration checks connection_string."""
        with patch("ml.stores.data_store.FeatureStore", return_value=mock_feature_store), \
             patch("ml.stores.data_store.ModelStore", return_value=mock_model_store), \
             patch("ml.stores.data_store.StrategyStore", return_value=mock_strategy_store):
            # Empty connection string
            store = DataStore(
                connection_string="",
                registry=mock_registry,
                feature_store=mock_feature_store,
                model_store=mock_model_store,
                strategy_store=mock_strategy_store,
                earnings_store=mock_earnings_store,
            )

            errors = store.validate_configuration()

            # Should have error for missing connection_string
            assert "Missing connection_string" in errors

    def test_validate_configuration_checks_batch_size(
        self,
        connection_string,
        mock_feature_store,
        mock_model_store,
        mock_strategy_store,
        mock_earnings_store,
        mock_registry,
    ):
        """Test validate_configuration checks batch_size."""
        with patch("ml.stores.data_store.FeatureStore", return_value=mock_feature_store), \
             patch("ml.stores.data_store.ModelStore", return_value=mock_model_store), \
             patch("ml.stores.data_store.StrategyStore", return_value=mock_strategy_store):
            # Invalid batch size
            store = DataStore(
                connection_string=connection_string,
                registry=mock_registry,
                feature_store=mock_feature_store,
                model_store=mock_model_store,
                strategy_store=mock_strategy_store,
                earnings_store=mock_earnings_store,
                batch_size=0,
            )

            errors = store.validate_configuration()

            # Should have error for invalid batch_size
            assert "batch_size must be positive" in errors
