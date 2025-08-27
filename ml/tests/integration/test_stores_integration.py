"""
Consolidated Store Integration Tests.

This file consolidates tests from:
- ml/tests/integration/test_stores_integration.py (original)
- ml/tests/test_stores_simple.py (merged and removed)

Consolidation performed on 2025-08-25.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from ml.stores.data_processor import DataProcessor
from ml.stores.data_processor import QualityFlags
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore


@pytest.fixture
def mock_persistence_manager(test_database):
    """
    Create mock persistence manager with real PostgreSQL connection.
    """
    mock = MagicMock()
    mock.connection_string = test_database.connection_string
    mock.session = MagicMock()
    return mock


@pytest.fixture
def feature_store(test_database):
    """
    Create feature store with PostgreSQL connection.
    """
    store = FeatureStore(
        connection_string=test_database.connection_string,
        batch_size=10,
        flush_interval_seconds=1.0,
    )
    yield store
    # Cleanup
    if hasattr(store, "_timer") and store._timer:
        store._timer.cancel()


@pytest.fixture
def model_store(test_database):
    """
    Create model store with PostgreSQL connection.
    """
    store = ModelStore(
        connection_string=test_database.connection_string,
        batch_size=10,
        flush_interval_seconds=1.0,
    )
    yield store
    # Cleanup
    if hasattr(store, "_timer") and store._timer:
        store._timer.cancel()


@pytest.fixture
def strategy_store(test_database):
    """
    Create strategy store with PostgreSQL connection.
    """
    store = StrategyStore(
        connection_string=test_database.connection_string,
        batch_size=10,
        flush_interval_seconds=1.0,
    )
    yield store
    # Cleanup
    if hasattr(store, "_timer") and store._timer:
        store._timer.cancel()


@pytest.fixture
def data_processor(test_database):
    """
    Create data processor with PostgreSQL connection.
    """
    return DataProcessor(
        connection_string=test_database.connection_string,
        outlier_threshold=3.0,
        staleness_threshold_seconds=60,
    )


@pytest.mark.usefixtures("clean_postgres_db")
class TestFeatureStore:
    """
    Test FeatureStore functionality with PostgreSQL.
    """

    def test_write_features(self, feature_store):
        """
        Test writing features.
        """
        ts_event = int(time.time() * 1e9)

        feature_store.write_features(
            feature_set_id="test_features",
            instrument_id="AAPL",
            features={"sma_20": 150.5, "rsi_14": 65.2},
            ts_event=ts_event,
        )

        # Check buffer
        assert len(feature_store._buffer) == 1
        data = feature_store._buffer[0]
        assert data.feature_set_id == "test_features"
        assert data.instrument_id == "AAPL"
        assert data.values["sma_20"] == 150.5
        assert data.values["rsi_14"] == 65.2

    def test_auto_flush(self, feature_store):
        """
        Test automatic buffer flushing.
        """
        # Fill buffer to trigger flush
        for i in range(11):  # Batch size is 10
            feature_store.write_features(
                feature_set_id=f"features_{i}",
                instrument_id="AAPL",
                features={"value": float(i)},
                ts_event=int(time.time() * 1e9) + i,
            )

        # Buffer should be flushed
        assert len(feature_store._buffer) == 1  # Only the 11th item

    def test_read_range(self, feature_store, mock_persistence_manager):
        """
        Test reading features by time range.
        """
        # Mock database response
        mock_persistence_manager.session.execute.return_value.fetchall.return_value = [
            ("test_features", "AAPL", json.dumps({"sma_20": 150.5}), 1000, 1001),
        ]

        result = feature_store.read_range(
            start_ns=900,
            end_ns=1100,
            instrument_id="AAPL",
        )

        assert len(result) == 1
        assert result.iloc[0]["feature_set_id"] == "test_features"


@pytest.mark.usefixtures("clean_postgres_db")
class TestModelStore:
    """
    Test ModelStore functionality with PostgreSQL.
    """

    def test_write_prediction(self, model_store):
        """
        Test writing model predictions.
        """
        ts_event = int(time.time() * 1e9)

        model_store.write_prediction(
            model_id="xgboost_v1",
            instrument_id="AAPL",
            prediction=0.75,
            confidence=0.85,
            features={"sma_20": 150.5},
            inference_time_ms=2.5,
            ts_event=ts_event,
            is_live=True,
        )

        assert len(model_store._buffer) == 1
        data = model_store._buffer[0]
        assert data.model_id == "xgboost_v1"
        assert data.prediction == 0.75
        assert data.confidence == 0.85

    def test_read_latest_predictions(self, model_store, mock_persistence_manager):
        """
        Test reading latest predictions.
        """
        mock_persistence_manager.session.execute.return_value.fetchall.return_value = [
            (
                "xgboost_v1",
                "AAPL",
                0.75,
                0.85,
                json.dumps({"sma_20": 150.5}),
                2.5,
                True,
                1000,
                1001,
            ),
        ]

        result = model_store.read_latest_predictions(
            model_id="xgboost_v1",
            limit=10,
        )

        assert len(result) == 1
        assert result.iloc[0]["prediction"] == 0.75

    def test_get_model_performance(self, model_store, mock_persistence_manager):
        """
        Test getting model performance metrics.
        """
        mock_persistence_manager.session.execute.return_value.fetchone.return_value = (
            100,  # count
            0.75,  # avg_confidence
            2.5,  # avg_inference_time
            5.0,  # max_inference_time
        )

        metrics = model_store.get_model_performance(
            model_id="xgboost_v1",
            hours_back=24,
        )

        assert metrics["prediction_count"] == 100
        assert metrics["avg_confidence"] == 0.75


@pytest.mark.usefixtures("clean_postgres_db")
class TestStrategyStore:
    """
    Test StrategyStore functionality with PostgreSQL.
    """

    def test_write_signal(self, strategy_store):
        """
        Test writing strategy signals.
        """
        ts_event = int(time.time() * 1e9)

        strategy_store.write_signal(
            strategy_id="momentum_v1",
            instrument_id="AAPL",
            signal_type="BUY",
            strength=0.8,
            model_predictions={"xgboost": 0.75},
            risk_metrics={"position_size": 100},
            execution_params={"order_type": "LIMIT"},
            ts_event=ts_event,
        )

        assert len(strategy_store._buffer) == 1
        data = strategy_store._buffer[0]
        assert data.strategy_id == "momentum_v1"
        assert data.signal_type == "BUY"
        assert data.strength == 0.8

    def test_read_active_signals(self, strategy_store, mock_persistence_manager):
        """
        Test reading active signals.
        """
        mock_persistence_manager.session.execute.return_value.fetchall.return_value = [
            (
                "momentum_v1",
                "AAPL",
                "BUY",
                0.8,
                json.dumps({"xgboost": 0.75}),
                json.dumps({"position_size": 100}),
                json.dumps({"order_type": "LIMIT"}),
                1000,
                1001,
            ),
        ]

        result = strategy_store.read_active_signals(
            strategy_id="momentum_v1",
            hours_back=1,
        )

        assert len(result) == 1
        assert result.iloc[0]["signal_type"] == "BUY"


@pytest.mark.usefixtures("clean_postgres_db")
class TestDataProcessor:
    """
    Test DataProcessor functionality with PostgreSQL.
    """

    def test_process_market_data(self, data_processor):
        """
        Test processing market data.
        """
        ts_event = int(time.time() * 1e9)

        data = {
            "bid": 150.0,
            "ask": 150.1,
            "bid_size": 100,
            "ask_size": 200,
            "volume": 10000,
        }

        processed, metrics = data_processor.process_market_data(
            instrument_id="AAPL",
            data=data,
            ts_event=ts_event,
        )

        assert processed["bid"] == 150.0
        assert processed["ask"] == 150.1
        assert processed["quality_score"] >= 0.0
        assert metrics.records_processed == 1

    def test_process_market_data_with_crossed_market(self, data_processor):
        """
        Test processing crossed market.
        """
        ts_event = int(time.time() * 1e9)

        data = {
            "bid": 150.1,  # Bid > Ask (crossed)
            "ask": 150.0,
            "bid_size": 100,
            "ask_size": 200,
        }

        processed, metrics = data_processor.process_market_data(
            instrument_id="AAPL",
            data=data,
            ts_event=ts_event,
        )

        # Should fix the crossed market
        assert processed["bid"] < processed["ask"]
        assert processed["quality_flags"] & QualityFlags.INVALID_RANGE

    def test_process_features_with_nan(self, data_processor):
        """
        Test processing features with NaN values.
        """
        ts_event = int(time.time() * 1e9)

        features = {
            "sma_20": 150.5,
            "rsi_14": np.nan,  # NaN value
            "volume": 10000,
        }

        feature_data, metrics = data_processor.process_features(
            feature_set_id="test_features",
            instrument_id="AAPL",
            features=features,
            ts_event=ts_event,
        )

        # NaN should be imputed
        assert feature_data.values["rsi_14"] == 0.0
        assert metrics.missing_imputed == 1

    def test_process_prediction(self, data_processor):
        """
        Test processing model predictions.
        """
        ts_event = int(time.time() * 1e9)

        pred_data, metrics = data_processor.process_prediction(
            model_id="xgboost_v1",
            instrument_id="AAPL",
            prediction=0.75,
            confidence=0.85,
            features={"sma_20": 150.5},
            inference_time_ms=2.5,
            ts_event=ts_event,
        )

        assert pred_data.prediction == 0.75
        assert pred_data.confidence <= 0.85  # May be adjusted
        assert metrics.records_processed == 1

    def test_process_signal_with_risk_limits(self, data_processor):
        """
        Test processing signals with risk limits.
        """
        ts_event = int(time.time() * 1e9)

        signal_data, metrics = data_processor.process_signal(
            strategy_id="momentum_v1",
            instrument_id="AAPL",
            signal_type="BUY",
            strength=1.0,
            model_predictions={"xgboost": 0.75},
            ts_event=ts_event,
        )

        assert signal_data.strength <= 1.0  # May be adjusted by risk limits
        assert "order_size" in signal_data.execution_params
        assert metrics.records_processed == 1

    def test_quality_score_calculation(self, data_processor):
        """
        Test quality score calculation.
        """
        # Clean data
        score = data_processor._calculate_quality_score(QualityFlags.CLEAN)
        assert score == 1.0

        # Data with issues
        flags = QualityFlags.MISSING_DATA | QualityFlags.OUTLIER_DETECTED
        score = data_processor._calculate_quality_score(flags)
        assert 0 < score < 1.0

        # Severe issues
        flags = QualityFlags.TIMESTAMP_ERROR | QualityFlags.NAN_VALUES | QualityFlags.INF_VALUES
        score = data_processor._calculate_quality_score(flags)
        assert score < 0.5

    def test_batch_processing(self, data_processor):
        """
        Test batch processing.
        """
        batch = [
            {
                "instrument_id": "AAPL",
                "data": {"bid": 150.0, "ask": 150.1},
                "ts_event": int(time.time() * 1e9),
            },
            {
                "instrument_id": "GOOGL",
                "data": {"bid": 2800.0, "ask": 2800.1},
                "ts_event": int(time.time() * 1e9) + 1000,
            },
        ]

        processed, metrics = data_processor.process_batch(
            data_type="market",
            batch=batch,
        )

        assert len(processed) == 2
        assert metrics.records_processed == 2


class TestIntegration:
    """
    Test integration between components.
    """

    def test_end_to_end_flow(self, feature_store, model_store, strategy_store, data_processor):
        """
        Test complete data flow through all stores.
        """
        ts_event = int(time.time() * 1e9)

        # 1. Process market data
        market_data, _ = data_processor.process_market_data(
            instrument_id="AAPL",
            data={"bid": 150.0, "ask": 150.1, "volume": 10000},
            ts_event=ts_event,
        )

        # 2. Process and store features
        features = {"sma_20": 150.5, "rsi_14": 65.2}
        feature_data, _ = data_processor.process_features(
            feature_set_id="test_features",
            instrument_id="AAPL",
            features=features,
            ts_event=ts_event,
        )

        feature_store.write_batch([feature_data])

        # 3. Process and store predictions
        pred_data, _ = data_processor.process_prediction(
            model_id="xgboost_v1",
            instrument_id="AAPL",
            prediction=0.75,
            confidence=0.85,
            features=features,
            inference_time_ms=2.5,
            ts_event=ts_event,
        )

        model_store.write_batch([pred_data])

        # 4. Process and store signals
        signal_data, _ = data_processor.process_signal(
            strategy_id="momentum_v1",
            instrument_id="AAPL",
            signal_type="BUY",
            strength=0.8,
            model_predictions={"xgboost_v1": 0.75},
            ts_event=ts_event,
        )

        strategy_store.write_batch([signal_data])

        # Verify data in stores
        assert len(feature_store._buffer) == 0  # Should be flushed
        assert len(model_store._buffer) == 0  # Should be flushed
        assert len(strategy_store._buffer) == 0  # Should be flushed

    def test_data_quality_propagation(self, data_processor):
        """
        Test that data quality issues propagate through pipeline.
        """
        ts_event = int(time.time() * 1e9)

        # Start with bad market data
        bad_data = {
            "bid": -100.0,  # Invalid negative price
            "ask": 150.1,
            "volume": np.nan,  # Missing volume
        }

        market_data, market_metrics = data_processor.process_market_data(
            instrument_id="AAPL",
            data=bad_data,
            ts_event=ts_event,
        )

        # Quality issues should be flagged
        assert market_data["quality_score"] < 1.0
        assert market_metrics.records_failed > 0

        # Process features based on bad data
        features = {
            "price_based_feature": market_data["bid"],  # Based on bad price
            "volume_based_feature": np.nan,  # Based on missing volume
        }

        feature_data, feature_metrics = data_processor.process_features(
            feature_set_id="test_features",
            instrument_id="AAPL",
            features=features,
            ts_event=ts_event,
        )

        # Feature quality should reflect issues
        assert feature_metrics.missing_imputed > 0


# =================================================================================================
# Tests merged from test_stores_simple.py (without dependencies)
# =================================================================================================


class TestDataProcessorSimple:
    """
    Simple tests for DataProcessor without full dependencies.
    """

    def test_process_market_data_simple(self) -> None:
        """
        Test processing market data.
        """
        # Mock the database engine
        with patch("ml.stores.data_processor.create_engine"):
            processor = DataProcessor(
                connection_string="postgresql://test:test@localhost/test",
                outlier_threshold=3.0,
                staleness_threshold_seconds=60,
            )

            ts_event = int(time.time() * 1e9)

            data = {
                "bid": 150.0,
                "ask": 150.1,
                "bid_size": 100,
                "ask_size": 200,
                "volume": 10000,
            }

            processed, metrics = processor.process_market_data(
                instrument_id="AAPL",
                data=data,
                ts_event=ts_event,
            )

            assert processed["bid"] == 150.0
            assert processed["ask"] == 150.1
            assert processed["quality_score"] >= 0.0
            assert metrics.records_processed == 1

    def test_process_market_data_with_crossed_market(self) -> None:
        """
        Test processing crossed market.
        """
        with patch("ml.stores.data_processor.create_engine"):
            processor = DataProcessor(
                connection_string="postgresql://test:test@localhost/test",
                outlier_threshold=3.0,
                staleness_threshold_seconds=60,
            )

            ts_event = int(time.time() * 1e9)

            data = {
                "bid": 150.1,  # Bid > Ask (crossed)
                "ask": 150.0,
                "bid_size": 100,
                "ask_size": 200,
            }

            processed, metrics = processor.process_market_data(
                instrument_id="AAPL",
                data=data,
                ts_event=ts_event,
            )

            # Should fix the crossed market
            assert processed["bid"] < processed["ask"]
            assert processed["quality_flags"] & QualityFlags.INVALID_RANGE

    def test_process_features_with_nan(self) -> None:
        """
        Test processing features with NaN values.
        """
        with patch("ml.stores.data_processor.create_engine"):
            processor = DataProcessor(
                connection_string="postgresql://test:test@localhost/test",
                outlier_threshold=3.0,
                staleness_threshold_seconds=60,
            )

            ts_event = int(time.time() * 1e9)

            features = {
                "sma_20": 150.5,
                "rsi_14": np.nan,  # NaN value
                "volume": np.inf,  # Inf value
            }

            processed, metrics = processor.process_features(
                feature_set_id="test_features",
                instrument_id="AAPL",
                features=features,
                ts_event=ts_event,
            )

            # Should handle NaN and Inf
            assert not np.isnan(processed.features["rsi_14"])
            assert not np.isinf(processed.features["volume"])
            assert processed.quality_flags & QualityFlags.NAN_VALUES
            assert processed.quality_flags & QualityFlags.INF_VALUES
            assert metrics.missing_imputed == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# mypy: ignore-errors
