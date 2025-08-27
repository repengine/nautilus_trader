"""
Integration tests for store persistence.

Tests that data is actually persisted to stores and can be retrieved.
"""

import pytest
from sqlalchemy import text

from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore


@pytest.mark.usefixtures("clean_postgres_db")
class TestStorePersistence:
    """
    Test that stores actually persist data with PostgreSQL.
    """

    def test_feature_store_persistence(self, test_database):
        """
        Test that FeatureStore actually persists and retrieves features.
        """
        # Create store
        store = FeatureStore(
            connection_string=test_database.connection_string,
        )

        # Write features
        features = {
            "feature_0": 0.5,
            "feature_1": 0.7,
            "feature_2": -0.3,
        }

        store.write_features(
            feature_set_id="test_set",
            instrument_id="EUR/USD",
            features=features,
            ts_event=1000000000,
            ts_init=1000000001,
        )

        # Flush (should be no-op but test it)
        store.flush()

        # Create new store instance to verify persistence
        store2 = FeatureStore(
            connection_string=test_database.connection_string,
        )

        # Read back features
        with test_database.get_session() as session:
            result = session.execute(
                text("""
                    SELECT * FROM ml_feature_values
                    WHERE feature_set_id = :feature_set_id
                """ ),
                {"feature_set_id": "test_set"}
            ).fetchone()

        assert result is not None
        assert result["instrument_id"] == "EUR/USD"
        assert result["ts_event"] == 1000000000

        # Check health
        assert store2.is_healthy()

    def test_model_store_persistence(self, test_database):
        """
        Test that ModelStore actually persists and retrieves predictions.
        """
        # Create store
        store = ModelStore(
            connection_string=test_database.connection_string,
        )

        # Write predictions
        store.write_prediction(
            model_id="test_model",
            instrument_id="EUR/USD",
            prediction=0.65,
            confidence=0.8,
            features={"feature_0": 0.5},
            inference_time_ms=2.5,
            ts_event=1000000000,
        )

        # Write more to trigger batch
        for i in range(10):
            store.write_prediction(
                model_id="test_model",
                instrument_id="EUR/USD",
                prediction=0.5 + i * 0.01,
                confidence=0.7 + i * 0.01,
                features={"feature_0": 0.5 + i * 0.1},
                inference_time_ms=2.0 + i * 0.1,
                ts_event=1000000000 + i * 1000,
            )

        # Force flush
        store.flush()

        # Create new store instance to verify persistence
        store2 = ModelStore(
            connection_string=test_database.connection_string,
        )

        # Read back predictions
        predictions = store2.get_predictions(
            model_id="test_model",
            start_ns=999999999,
            end_ns=2000000000,
        )

        assert len(predictions) >= 11  # Should have all predictions

        # Check health
        assert store2.is_healthy()

    def test_strategy_store_persistence(self, test_database):
        """
        Test that StrategyStore actually persists and retrieves signals.
        """
        # Create store
        store = StrategyStore(
            connection_string=test_database.connection_string,
        )

        # Write signals
        store.write_signal(
            strategy_id="test_strategy",
            instrument_id="EUR/USD",
            signal_type="buy",
            strength=0.8,
            model_predictions={"model_1": 0.7},
            risk_metrics={"sharpe": 1.5},
            execution_params={"threshold": 0.6},
            ts_event=1000000000,
        )

        # Write more signals
        for i in range(5):
            store.write_signal(
                strategy_id="test_strategy",
                instrument_id="EUR/USD",
                signal_type="sell" if i % 2 else "buy",
                strength=0.5 + i * 0.1,
                model_predictions={"model_1": 0.4 + i * 0.1},
                risk_metrics={"sharpe": 1.0 + i * 0.2},
                execution_params={"threshold": 0.5},
                ts_event=1000000000 + i * 1000,
            )

        # Force flush
        store.flush()

        # Create new store instance to verify persistence
        store2 = StrategyStore(
            connection_string=test_database.connection_string,
        )

        # Read back signals
        signals = store2.get_signals(
            strategy_id="test_strategy",
            start_ns=999999999,
            end_ns=2000000000,
        )

        assert len(signals) >= 6  # Should have all signals

        # Check health
        assert store2.is_healthy()

    def test_store_failure_handling(self):
        """
        Test that stores handle connection failures gracefully.
        """
        # Create store with invalid connection string
        bad_connection = "postgresql://invalid:invalid@nonexistent:5432/none"
        
        # Store should raise on bad connection
        with pytest.raises(Exception):
            ModelStore(connection_string=bad_connection)

    def test_store_is_healthy(self, test_database):
        """
        Test that stores properly report health status with PostgreSQL.
        """
        # Create all stores
        feature_store = FeatureStore(connection_string=test_database.connection_string)
        model_store = ModelStore(connection_string=test_database.connection_string)
        strategy_store = StrategyStore(connection_string=test_database.connection_string)
        
        # All stores should be healthy with valid PostgreSQL connection
        assert feature_store.is_healthy()
        assert model_store.is_healthy()
        assert strategy_store.is_healthy()
        
        # Test writing to each store
        feature_store.write_features(
            feature_set_id="health_test",
            instrument_id="EUR/USD",
            features={"test": 1.0},
            ts_event=1000000000,
            ts_init=1000000001,
        )
        
        model_store.write_prediction(
            model_id="health_test",
            instrument_id="EUR/USD",
            prediction=0.5,
            confidence=0.8,
            features={},
            inference_time_ms=1.0,
            ts_event=1000000000,
        )
        
        strategy_store.write_signal(
            strategy_id="health_test",
            instrument_id="EUR/USD",
            signal_type="buy",
            strength=0.8,
            model_predictions={},
            risk_metrics={},
            execution_params={},
            ts_event=1000000000,
        )
        
        # Flush all stores
        feature_store.flush()
        model_store.flush()
        strategy_store.flush()
        
        # Stores should still be healthy after operations
        assert feature_store.is_healthy()
        assert model_store.is_healthy()
        assert strategy_store.is_healthy()
