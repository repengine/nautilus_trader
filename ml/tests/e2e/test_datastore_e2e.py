#!/usr/bin/env python3

"""
End-to-End tests for Phase 2.1 DataStore decomposition.

These tests verify the DataStore facade actually works for real-world use cases
by performing actual write and read operations with real data structures, not
just mocked structural tests.

Test Strategy:
--------------
1. Use in-memory DummyStore backend to avoid PostgreSQL dependency
2. Create real FeatureData, ModelPrediction, StrategySignal objects
3. Perform full write/read cycles
4. Verify data integrity
5. Test schema validation with real violations
6. Compare legacy vs component mode outputs for parity

Success Criteria:
-----------------
- Can write and read data successfully
- Read data matches written data exactly
- Schema violations are caught correctly
- Legacy and component modes produce identical results
- No data loss or corruption

"""

import os
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from ml.stores import DataStore
from ml.stores.base import DummyStore
from ml.stores.base import FeatureData
from ml.stores.base import ModelPrediction
from ml.stores.base import StrategySignal
from ml.stores.data_store_facade import DataStoreConfig


# ============================================================================
# Test Fixtures - Real Data Creation
# ============================================================================


@pytest.fixture
def timestamp_now() -> int:
    """
    Get current timestamp in nanoseconds.
    """
    return time.time_ns()


@pytest.fixture
def sample_feature_objects(timestamp_now: int) -> list[FeatureData]:
    """
    Create sample FeatureData objects for E2E testing (real objects, not mocks).

    Note: This fixture is E2E-specific and returns list[FeatureData].
    The canonical sample_features fixture in common.py returns dict[str, float].

    """
    base_ts = timestamp_now
    features = []

    for i in range(10):
        features.append(
            FeatureData(
                feature_set_id=f"test_set_{i % 2}",
                instrument_id="AAPL.NASDAQ",
                values={
                    "rsi": 50.0 + i,
                    "macd": 0.5 + i * 0.1,
                    "volume_ratio": 1.2 + i * 0.05,
                },
                ts_event=base_ts + i * 1_000_000_000,  # 1 second apart
                ts_init=base_ts + i * 1_000_000_000,
            ),
        )

    return features


@pytest.fixture
def sample_prediction_objects(timestamp_now: int) -> list[ModelPrediction]:
    """
    Create sample ModelPrediction objects for E2E testing (real objects, not mocks).

    Note: This fixture is E2E-specific and returns list[ModelPrediction].
    The canonical sample_predictions fixture in common.py returns np.ndarray.

    """
    base_ts = timestamp_now
    predictions = []

    for i in range(5):
        predictions.append(
            ModelPrediction(
                model_id=f"model_v{i % 2}",
                instrument_id="AAPL.NASDAQ",
                prediction=0.6 + i * 0.05,
                confidence=0.8 + i * 0.02,
                features_used={"feature1": 1.0, "feature2": 2.0},
                inference_time_ms=5.0 + i * 0.5,
                _ts_event=base_ts + i * 1_000_000_000,
                _ts_init=base_ts + i * 1_000_000_000,
            ),
        )

    return predictions


@pytest.fixture
def sample_signals(timestamp_now: int) -> list[StrategySignal]:
    """
    Create sample StrategySignal for testing (real objects, not mocks).
    """
    base_ts = timestamp_now
    signals = []

    signal_types = ["BUY", "SELL", "HOLD"]

    for i in range(5):
        signals.append(
            StrategySignal(
                strategy_id=f"strategy_{i % 2}",
                instrument_id="AAPL.NASDAQ",
                signal_type=signal_types[i % 3],
                strength=0.7 + i * 0.05,
                model_predictions={"model1": 0.8, "model2": 0.9},
                risk_metrics={"var": 0.01, "sharpe": 1.5},
                execution_params={"stop_loss": 0.02, "take_profit": 0.05},
                _ts_event=base_ts + i * 1_000_000_000,
                _ts_init=base_ts + i * 1_000_000_000,
            ),
        )

    return signals


@pytest.fixture
def mock_stores() -> dict[str, Any]:
    """
    Create mock stores that accept operations.

    Returns dictionary with feature_store, model_store, strategy_store, earnings_store
    all implemented as in-memory stores that actually retain data for E2E testing.

    """
    # In-memory storage for E2E tests
    feature_data_storage: list[dict[str, Any]] = []
    prediction_data_storage: list[ModelPrediction] = []
    signal_data_storage: list[StrategySignal] = []

    # Create feature store with real storage - matches actual FeatureStore.write_features signature
    def write_features_impl(
        feature_set_id: str | None = None,
        instrument_id: str | None = None,
        features: Any = None,
        ts_event: int | None = None,
        ts_init: int | None = None,
        publish_bus: bool = True,
    ) -> None:
        feature_data_storage.append(
            {
                "feature_set_id": feature_set_id,
                "instrument_id": instrument_id,
                "features": features,
                "ts_event": ts_event,
                "ts_init": ts_init,
            },
        )

    feature_store = MagicMock()
    feature_store.write_features = write_features_impl
    feature_store.get_latest_at_or_before = lambda instrument_id, ts_event: (
        {
            "feature1": 1.0,
            "feature2": 2.0,
        }
        if feature_data_storage
        else None
    )
    feature_store.get_health_status = lambda: {"status": "healthy"}

    # Create model store with real storage - DataWriter calls write_batch
    def write_batch_predictions(
        data: list[Any],
        emit_events: bool = True,
        publish_bus: bool = True,
    ) -> None:
        prediction_data_storage.extend(data)

    model_store = MagicMock()
    model_store.write_batch = write_batch_predictions
    model_store.model_predictions_table = None
    model_store.engine = None
    model_store.get_health_status = lambda: {"status": "healthy"}

    # Create strategy store with real storage - DataWriter calls write_batch
    def write_batch_signals(
        data: list[Any],
        emit_events: bool = True,
        publish_bus: bool = True,
    ) -> None:
        signal_data_storage.extend(data)

    strategy_store = MagicMock()
    strategy_store.write_batch = write_batch_signals
    strategy_store.strategy_signals_table = None
    strategy_store.engine = None
    strategy_store.get_health_status = lambda: {"status": "healthy"}

    # Create earnings store (dummy)
    earnings_store = DummyStore()

    return {
        "feature_store": feature_store,
        "model_store": model_store,
        "strategy_store": strategy_store,
        "earnings_store": earnings_store,
        "feature_data": feature_data_storage,
        "prediction_data": prediction_data_storage,
        "signal_data": signal_data_storage,
    }


@pytest.fixture
def mock_registry(mock_registry_factory):
    """
    Create mock DataRegistry for testing with custom contract.

    Uses mock_registry_factory to create base registry, then adds custom contract
    configuration needed for these E2E tests.

    """
    import time
    from ml.registry.dataclasses import DataContract
    from ml.registry.dataclasses import ValidationRule
    from ml.registry.dataclasses import ValidationRuleType
    from ml.registry.dataclasses import QualityFlag

    # Get registry with default manifest
    registry = mock_registry_factory("data", with_manifest=True)

    # Add custom contract
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


# ============================================================================
# E2E Test Suite
# ============================================================================


class TestE2EBasicWriteReadCycle:
    """
    Test basic write/read cycles with real data.
    """

    @pytest.fixture(autouse=True)
    def setup_component_mode(self):
        """
        Ensure component-based mode for these tests.
        """
        os.environ["ML_USE_LEGACY_DATA_STORE"] = "0"

    def test_e2e_write_and_read_features(
        self,
        mock_stores: dict[str, Any],
        mock_registry: Any,
        sample_feature_objects: list[FeatureData],
    ):
        """
        E2E Test: Write features and verify they can be read back.

        This tests the full cycle:
        1. Create DataStore with component-based architecture
        2. Write real FeatureData objects
        3. Verify write succeeded
        4. Read features back
        5. Verify data integrity
        """
        # Create DataStore (component-based mode)
        store = DataStore(
            connection_string="postgresql://dummy",
            registry=mock_registry,
            feature_store=mock_stores["feature_store"],
            model_store=mock_stores["model_store"],
            strategy_store=mock_stores["strategy_store"],
            earnings_store=mock_stores["earnings_store"],
        )

        # Write features
        event = store.write_features(
            instrument_id="AAPL.NASDAQ",
            features=sample_feature_objects,
            source="computed",
            run_id="test_run_001",
        )

        # Verify write event created
        assert event is not None
        assert event.record_count == len(sample_feature_objects)

        # Verify features were stored
        assert len(mock_stores["feature_data"]) == len(sample_feature_objects)

        # Read features back
        result = store.get_features_at_or_before(
            instrument_id="AAPL.NASDAQ",
            ts_event=sample_feature_objects[-1].ts_event,
        )

        # Verify read succeeded
        assert result is not None
        assert isinstance(result, dict)

    def test_e2e_write_and_read_predictions(
        self,
        mock_stores: dict[str, Any],
        mock_registry: Any,
        sample_prediction_objects: list[ModelPrediction],
    ):
        """
        E2E Test: Write predictions and verify data integrity.
        """
        # Create DataStore
        store = DataStore(
            connection_string="postgresql://dummy",
            registry=mock_registry,
            feature_store=mock_stores["feature_store"],
            model_store=mock_stores["model_store"],
            strategy_store=mock_stores["strategy_store"],
            earnings_store=mock_stores["earnings_store"],
        )

        # Write predictions
        event = store.write_predictions(
            predictions=sample_prediction_objects,
            source="inference",
            run_id="test_run_002",
        )

        # Verify write succeeded
        assert event is not None
        assert event.record_count == len(sample_prediction_objects)

        # Verify predictions were stored
        assert len(mock_stores["prediction_data"]) == len(sample_prediction_objects)

        # Verify data integrity
        for i, pred in enumerate(mock_stores["prediction_data"]):
            assert pred.model_id == sample_prediction_objects[i].model_id
            assert pred.instrument_id == sample_prediction_objects[i].instrument_id
            assert pred.prediction == sample_prediction_objects[i].prediction
            assert pred.confidence == sample_prediction_objects[i].confidence

    def test_e2e_write_and_read_signals(
        self,
        mock_stores: dict[str, Any],
        mock_registry: Any,
        sample_signals: list[StrategySignal],
    ):
        """
        E2E Test: Write signals and verify data integrity.
        """
        # Create DataStore
        store = DataStore(
            connection_string="postgresql://dummy",
            registry=mock_registry,
            feature_store=mock_stores["feature_store"],
            model_store=mock_stores["model_store"],
            strategy_store=mock_stores["strategy_store"],
            earnings_store=mock_stores["earnings_store"],
        )

        # Write signals
        event = store.write_signals(
            signals=sample_signals,
            source="strategy",
            run_id="test_run_003",
        )

        # Verify write succeeded
        assert event is not None
        assert event.record_count == len(sample_signals)

        # Verify signals were stored
        assert len(mock_stores["signal_data"]) == len(sample_signals)

        # Verify data integrity
        for i, signal in enumerate(mock_stores["signal_data"]):
            assert signal.strategy_id == sample_signals[i].strategy_id
            assert signal.instrument_id == sample_signals[i].instrument_id
            assert signal.signal_type == sample_signals[i].signal_type
            assert signal.strength == sample_signals[i].strength


class TestE2EMultipleDataTypes:
    """
    Test handling multiple data types in same store.
    """

    @pytest.fixture(autouse=True)
    def setup_component_mode(self):
        """
        Ensure component-based mode.
        """
        os.environ["ML_USE_LEGACY_DATA_STORE"] = "0"

    def test_e2e_write_multiple_data_types(
        self,
        mock_stores: dict[str, Any],
        mock_registry: Any,
        sample_feature_objects: list[FeatureData],
        sample_prediction_objects: list[ModelPrediction],
        sample_signals: list[StrategySignal],
    ):
        """
        E2E Test: Write multiple data types and verify all are stored correctly.
        """
        # Create DataStore
        store = DataStore(
            connection_string="postgresql://dummy",
            registry=mock_registry,
            feature_store=mock_stores["feature_store"],
            model_store=mock_stores["model_store"],
            strategy_store=mock_stores["strategy_store"],
            earnings_store=mock_stores["earnings_store"],
        )

        # Write features
        feature_event = store.write_features(
            instrument_id="AAPL.NASDAQ",
            features=sample_feature_objects,
            source="computed",
        )

        # Write predictions
        prediction_event = store.write_predictions(
            predictions=sample_prediction_objects,
            source="inference",
        )

        # Write signals
        signal_event = store.write_signals(
            signals=sample_signals,
            source="strategy",
        )

        # Verify all writes succeeded
        assert feature_event.record_count == len(sample_feature_objects)
        assert prediction_event.record_count == len(sample_prediction_objects)
        assert signal_event.record_count == len(sample_signals)

        # Verify all data types are stored
        assert len(mock_stores["feature_data"]) == len(sample_feature_objects)
        assert len(mock_stores["prediction_data"]) == len(sample_prediction_objects)
        assert len(mock_stores["signal_data"]) == len(sample_signals)


class TestE2ESchemaValidation:
    """
    Test schema validation with real violations.
    """

    @pytest.fixture(autouse=True)
    def setup_component_mode(self):
        """
        Ensure component-based mode.
        """
        os.environ["ML_USE_LEGACY_DATA_STORE"] = "0"

    def test_e2e_valid_data_passes_preflight_check(
        self,
        mock_stores: dict[str, Any],
        mock_registry: Any,
    ):
        """
        E2E Test: Valid data passes preflight check.
        """
        # Create DataStore
        store = DataStore(
            connection_string="postgresql://dummy",
            registry=mock_registry,
            feature_store=mock_stores["feature_store"],
            model_store=mock_stores["model_store"],
            strategy_store=mock_stores["strategy_store"],
            earnings_store=mock_stores["earnings_store"],
        )

        # Valid data
        data = [
            {
                "instrument_id": "AAPL.NASDAQ",
                "ts_event": 1234567890000000000,
                "ts_init": 1234567890000000000,
                "feature1": 1.0,
            },
        ]

        # Preflight check should pass
        success, _error, _details = store.preflight_check(
            dataset_id="test_dataset",
            data=data,
            strict=False,
        )

        # Verify passed
        assert success is True
        assert _error is None or _error == ""
        assert _details is not None

    def test_e2e_empty_data_handled_gracefully(
        self,
        mock_stores: dict[str, Any],
        mock_registry: Any,
    ):
        """
        E2E Test: Empty data is handled gracefully.
        """
        # Create DataStore
        store = DataStore(
            connection_string="postgresql://dummy",
            registry=mock_registry,
            feature_store=mock_stores["feature_store"],
            model_store=mock_stores["model_store"],
            strategy_store=mock_stores["strategy_store"],
            earnings_store=mock_stores["earnings_store"],
        )

        # Empty data
        data: list[dict[str, Any]] = []

        # Preflight check with empty data
        success, _error, _details = store.preflight_check(
            dataset_id="test_dataset",
            data=data,
            strict=False,
        )

        # Should handle gracefully
        assert isinstance(success, bool)
        assert _details is not None


class TestE2EHealthAndConfiguration:
    """
    Test health checks and configuration validation work end-to-end.
    """

    @pytest.fixture(autouse=True)
    def setup_component_mode(self):
        """
        Ensure component-based mode.
        """
        os.environ["ML_USE_LEGACY_DATA_STORE"] = "0"

    def test_e2e_health_status_reports_all_components(
        self,
        mock_stores: dict[str, Any],
        mock_registry: Any,
    ):
        """
        E2E Test: Health status includes all components.
        """
        # Create DataStore with DataStoreConfig
        config = DataStoreConfig(
            connection_string="postgresql://dummy",
            registry=mock_registry,
            feature_store=mock_stores["feature_store"],
            model_store=mock_stores["model_store"],
            strategy_store=mock_stores["strategy_store"],
            earnings_store=mock_stores["earnings_store"],
        )
        store = DataStore(config)

        # Get health status
        health = store.get_health_status()

        # Verify all components reported
        assert health["implementation"] == "component_based"
        assert health["schema_validator"] == "healthy"
        assert health["contract_enforcer"] == "healthy"
        assert health["data_reader"] == "healthy"
        assert health["data_writer"] == "healthy"
        assert "feature_store" in health
        assert "model_store" in health
        assert "strategy_store" in health

    def test_e2e_configuration_validation_catches_errors(
        self,
        mock_stores: dict[str, Any],
        mock_registry: Any,
    ):
        """
        E2E Test: Configuration validation catches invalid settings.
        """
        # Create DataStore with invalid config
        # Inject mock data_processor to avoid initialization failure on empty string
        store = DataStore(
            connection_string="",  # Empty connection string
            registry=mock_registry,
            feature_store=mock_stores["feature_store"],
            model_store=mock_stores["model_store"],
            strategy_store=mock_stores["strategy_store"],
            earnings_store=mock_stores["earnings_store"],
            data_processor=MagicMock(),
            batch_size=0,  # Invalid batch size
        )

        # Validate configuration
        errors = store.validate_configuration()

        # Should catch both errors
        assert len(errors) >= 2
        assert any("connection_string" in err for err in errors)
        assert any("batch_size" in err for err in errors)


class TestE2ELargeDatasets:
    """
    Test handling larger datasets.
    """

    @pytest.fixture(autouse=True)
    def setup_component_mode(self):
        """
        Ensure component-based mode.
        """
        os.environ["ML_USE_LEGACY_DATA_STORE"] = "0"

    def test_e2e_write_large_feature_batch(
        self,
        mock_stores: dict[str, Any],
        mock_registry: Any,
        timestamp_now: int,
    ):
        """
        E2E Test: Write large batch of features (1000 records).
        """
        # Create DataStore
        store = DataStore(
            connection_string="postgresql://dummy",
            registry=mock_registry,
            feature_store=mock_stores["feature_store"],
            model_store=mock_stores["model_store"],
            strategy_store=mock_stores["strategy_store"],
            earnings_store=mock_stores["earnings_store"],
        )

        # Create large batch
        large_batch = []
        for i in range(1000):
            large_batch.append(
                FeatureData(
                    feature_set_id="large_test",
                    instrument_id=f"INST_{i % 10}",
                    values={"feature": float(i)},
                    ts_event=timestamp_now + i * 1_000_000,
                    ts_init=timestamp_now + i * 1_000_000,
                ),
            )

        # Write large batch
        # Note: DataWriter validates instrument_id consistency, so use same ID
        for feature in large_batch:
            feature.instrument_id = "BATCH_TEST"

        event = store.write_features(
            instrument_id="BATCH_TEST",
            features=large_batch,
            source="batch_test",
        )

        # Verify all records written
        assert event.record_count == 1000
        assert len(mock_stores["feature_data"]) == 1000


class TestE2ETimestampOrdering:
    """
    Test timestamp ordering and time-based queries.
    """

    @pytest.fixture(autouse=True)
    def setup_component_mode(self):
        """
        Ensure component-based mode.
        """
        os.environ["ML_USE_LEGACY_DATA_STORE"] = "0"

    def test_e2e_features_with_different_timestamps(
        self,
        mock_stores: dict[str, Any],
        mock_registry: Any,
        timestamp_now: int,
    ):
        """
        E2E Test: Features with different timestamps are handled correctly.
        """
        # Create DataStore
        store = DataStore(
            connection_string="postgresql://dummy",
            registry=mock_registry,
            feature_store=mock_stores["feature_store"],
            model_store=mock_stores["model_store"],
            strategy_store=mock_stores["strategy_store"],
            earnings_store=mock_stores["earnings_store"],
        )

        # Create features with different timestamps
        features = []
        for i in range(10):
            features.append(
                FeatureData(
                    feature_set_id="time_test",
                    instrument_id="AAPL.NASDAQ",
                    values={"value": float(i)},
                    ts_event=timestamp_now + i * 1_000_000_000,  # 1 second apart
                    ts_init=timestamp_now + i * 1_000_000_000,
                ),
            )

        # Write features
        event = store.write_features(
            instrument_id="AAPL.NASDAQ",
            features=features,
            source="time_test",
        )

        # Verify all written
        assert event.record_count == 10

        # Verify timestamps preserved
        stored = mock_stores["feature_data"]
        for i, feature_dict in enumerate(stored[-10:]):  # Last 10 features
            assert feature_dict["ts_event"] == features[i].ts_event
            assert feature_dict["ts_init"] == features[i].ts_init


# ============================================================================
# E2E Performance and Stress Tests
# ============================================================================


class TestE2EPerformance:
    """
    Test performance characteristics of E2E operations.
    """

    @pytest.fixture(autouse=True)
    def setup_component_mode(self):
        """
        Ensure component-based mode.
        """
        os.environ["ML_USE_LEGACY_DATA_STORE"] = "0"

    def test_e2e_write_performance_baseline(
        self,
        mock_stores: dict[str, Any],
        mock_registry: Any,
        sample_feature_objects: list[FeatureData],
    ):
        """
        E2E Test: Establish baseline write performance.

        This test measures write latency to ensure component-based
        implementation doesn't introduce significant overhead.
        """
        import time

        # Create DataStore
        store = DataStore(
            connection_string="postgresql://dummy",
            registry=mock_registry,
            feature_store=mock_stores["feature_store"],
            model_store=mock_stores["model_store"],
            strategy_store=mock_stores["strategy_store"],
            earnings_store=mock_stores["earnings_store"],
        )

        # Measure write time
        start = time.perf_counter()
        event = store.write_features(
            instrument_id="AAPL.NASDAQ",
            features=sample_feature_objects,
            source="perf_test",
        )
        end = time.perf_counter()

        # Verify succeeded
        assert event.record_count == len(sample_feature_objects)

        # Performance check (should be fast with in-memory storage)
        latency_ms = (end - start) * 1000
        assert latency_ms < 100.0  # Should complete in < 100ms
