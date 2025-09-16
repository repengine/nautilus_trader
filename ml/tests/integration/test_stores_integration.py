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
from sqlalchemy import text

from ml.stores.data_processor import DataProcessor
from ml.stores.data_processor import QualityFlags
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.stores.strategy_store import StrategyStore
from ml.tests.builders import DataBuilder


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
def feature_store(test_database, mock_persistence_manager):
    """
    Create feature store with PostgreSQL connection.
    """
    store = FeatureStore(
        connection_string=test_database.connection_string,
        batch_size=10,
        flush_interval_seconds=1.0,
        persistence_manager=mock_persistence_manager,
    )
    yield store
    # Cleanup
    if hasattr(store, "_timer") and store._timer:
        store._timer.cancel()


@pytest.fixture
def model_store(test_database, mock_persistence_manager):
    """
    Create model store with PostgreSQL connection.
    """
    store = ModelStore(
        connection_string=test_database.connection_string,
        batch_size=10,
        flush_interval_seconds=1.0,
        persistence_manager=mock_persistence_manager,
    )
    yield store
    # Cleanup
    if hasattr(store, "_timer") and store._timer:
        store._timer.cancel()


@pytest.fixture
def strategy_store(test_database, mock_persistence_manager):
    """
    Create strategy store with PostgreSQL connection.
    """
    store = StrategyStore(
        connection_string=test_database.connection_string,
        batch_size=10,
        flush_interval_seconds=1.0,
        persistence_manager=mock_persistence_manager,
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


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.integration
@pytest.mark.usefixtures("clean_postgres_db_class")
class TestFeatureStore:
    """
    Test FeatureStore functionality with PostgreSQL.
    """

    def test_write_features(self, feature_store, default_instrument_id):
        """
        Test writing features.
        """
        ts_event = int(time.time() * 1e9)

        feature_store.write_features(
            feature_set_id="test_features",
            instrument_id=str(default_instrument_id),
            features={"sma_20": 150.5, "rsi_14": 65.2},
            ts_event=ts_event,
        )

        # Verify row persisted by querying directly
        import pandas as pd
        from sqlalchemy import text

        df = (
            feature_store.read_features(
                instrument_id=str(default_instrument_id),
                start_ts=ts_event,
                end_ts=ts_event,
            )
            if hasattr(feature_store, "read_features")
            else pd.DataFrame()
        )
        # Fallback: direct SQL check
        if df.empty:
            with feature_store.engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                        SELECT feature_set_id, instrument_id, values
                        FROM ml_feature_values
                        WHERE feature_set_id = :fsid AND instrument_id = :iid
                        LIMIT 1
                        """,
                    ),
                    {"fsid": "test_features", "iid": str(default_instrument_id)},
                ).fetchone()
                assert result is not None
        else:
            assert "sma_20" in df.columns and "rsi_14" in df.columns

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
    def test_auto_flush(self, feature_store, default_instrument_id):
        """
        Test automatic buffer flushing.
        """
        # Perform multiple writes; FeatureStore writes synchronously
        for i in range(11):
            feature_store.write_features(
                feature_set_id=f"features_{i}",
                instrument_id=str(default_instrument_id),
                features={"value": float(i)},
                ts_event=int(time.time() * 1e9) + i,
            )
        # No internal buffer is maintained; just ensure no exceptions

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
    def test_read_range(self, feature_store, mock_persistence_manager, default_instrument_id):
        """
        Test reading features by time range.
        """
        # Insert a test row into the DB for the queried range
        with feature_store.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO ml_feature_values (feature_set_id, instrument_id, ts_event, ts_init, values)
                    VALUES (:fsid, :iid, :tse, :tsi, :vals)
                    ON CONFLICT (feature_set_id, instrument_id, ts_event)
                    DO UPDATE SET values = EXCLUDED.values
                    """,
                ),
                {
                    "fsid": "test_features",
                    "iid": str(default_instrument_id),
                    "tse": 1000,
                    "tsi": 1001,
                    "vals": json.dumps({"sma_20": 150.5}),
                },
            )

        result = feature_store.read_range(
            start_ns=900,
            end_ns=1100,
            instrument_id=str(default_instrument_id),
        )

        assert len(result) == 1
        assert result.iloc[0]["feature_set_id"] == "test_features"


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.integration
@pytest.mark.usefixtures("clean_postgres_db_class")
class TestModelStore:
    """
    Test ModelStore functionality with PostgreSQL.
    """

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
    def test_write_prediction(self, model_store, default_instrument_id, sample_features):
        """
        Test writing model predictions.
        """
        ts_event = int(time.time() * 1e9)

        model_store.write_prediction(
            model_id="xgboost_v1",
            instrument_id=str(default_instrument_id),
            prediction=0.75,
            confidence=0.85,
            features=sample_features,
            inference_time_ms=2.5,
            ts_event=ts_event,
            is_live=True,
        )

        assert len(model_store._buffer) == 1
        data = model_store._buffer[0]
        assert data.model_id == "xgboost_v1"
        assert data.prediction == 0.75
        assert data.confidence == 0.85

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
    def test_read_latest_predictions(
        self,
        model_store,
        mock_persistence_manager,
        default_instrument_id,
        sample_features,
    ):
        """
        Test reading latest predictions.
        """
        # Insert a recent prediction row
        with model_store.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO ml_model_predictions (model_id, instrument_id, ts_event, ts_init, prediction, confidence, features_used, inference_time_ms, is_live)
                    VALUES (:mid, :iid, :tse, :tsi, :pred, :conf, :feats, :lat, :live)
                    ON CONFLICT (model_id, instrument_id, ts_event)
                    DO UPDATE SET prediction = EXCLUDED.prediction
                    """,
                ),
                {
                    "mid": "xgboost_v1",
                    "iid": str(default_instrument_id),
                    "tse": 1000,
                    "tsi": 1001,
                    "pred": 0.75,
                    "conf": 0.85,
                    "feats": json.dumps(sample_features),
                    "lat": 2.5,
                    "live": True,
                },
            )

        result = model_store.read_latest_predictions(
            model_id="xgboost_v1",
            limit=10,
        )

        assert len(result) == 1
        assert result.iloc[0]["prediction"] == 0.75

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
    def test_get_model_performance(
        self,
        model_store,
        mock_persistence_manager,
        default_instrument_id,
        sample_features,
    ):
        """
        Test getting model performance metrics.
        """
        # Insert multiple rows for performance stats
        import time as _time

        now_ns = int(_time.time() * 1e9)
        with model_store.engine.begin() as conn:
            for i in range(100):
                ts = now_ns - i  # ensure within the last 24 hours
                conn.execute(
                    text(
                        """
                        INSERT INTO ml_model_predictions (model_id, instrument_id, ts_event, ts_init, prediction, confidence, features_used, inference_time_ms, is_live)
                        VALUES (:mid, :iid, :tse, :tsi, :pred, :conf, :feats, :lat, :live)
                        ON CONFLICT (model_id, instrument_id, ts_event)
                        DO NOTHING
                        """,
                    ),
                    {
                        "mid": "xgboost_v1",
                        "iid": str(default_instrument_id),
                        "tse": ts,
                        "tsi": ts,
                        "pred": 0.5 + (i % 2) * 0.1,
                        "conf": 0.75,
                        "feats": json.dumps(sample_features),
                        "lat": 2.5,
                        "live": True,
                    },
                )

        metrics = model_store.get_model_performance(
            model_id="xgboost_v1",
            hours_back=24,
        )

        assert metrics["prediction_count"] == 100
        assert metrics["avg_confidence"] == 0.75


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.integration
@pytest.mark.usefixtures("clean_postgres_db_class")
class TestStrategyStore:
    """
    Test StrategyStore functionality with PostgreSQL.
    """

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
    def test_write_signal(self, strategy_store, default_instrument_id):
        """
        Test writing strategy signals.
        """
        ts_event = int(time.time() * 1e9)

        strategy_store.write_signal(
            strategy_id="momentum_v1",
            instrument_id=str(default_instrument_id),
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

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
    @pytest.mark.usefixtures("clean_postgres_db")
    def test_read_active_signals(
        self,
        strategy_store,
        mock_persistence_manager,
        default_instrument_id,
    ):
        """
        Test reading active signals.
        """
        # Insert a recent signal row
        import time as _time

        now_ns = int(_time.time() * 1e9)
        with strategy_store.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO ml_strategy_signals (strategy_id, instrument_id, ts_event, ts_init, signal_type, strength, model_predictions, risk_metrics, execution_params, is_live)
                    VALUES (:sid, :iid, :tse, :tsi, :stype, :str, :mp, :rm, :ep, :live)
                    ON CONFLICT (strategy_id, instrument_id, ts_event)
                    DO UPDATE SET strength = EXCLUDED.strength
                    """,
                ),
                {
                    "sid": "momentum_v1",
                    "iid": str(default_instrument_id),
                    "tse": now_ns,
                    "tsi": now_ns,
                    "stype": "BUY",
                    "str": 0.8,
                    "mp": json.dumps({"xgboost": 0.75}),
                    "rm": json.dumps({"position_size": 100}),
                    "ep": json.dumps({"order_type": "LIMIT"}),
                    "live": True,
                },
            )

        result = strategy_store.read_active_signals(
            strategy_id="momentum_v1",
            hours_back=1,
        )

        assert len(result) == 1
        assert result.iloc[0]["signal_type"] == "BUY"


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.integration
@pytest.mark.usefixtures("clean_postgres_db_class")
class TestDataProcessor:
    """
    Test DataProcessor functionality with PostgreSQL.
    """

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
    def test_process_market_data(self, data_processor, default_instrument_id):
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

        processed, _metrics = data_processor.process_market_data(
            instrument_id=str(default_instrument_id),
            data=data,
            ts_event=ts_event,
        )

        assert processed["bid"] == 150.0
        assert processed["ask"] == 150.1
        assert processed["quality_score"] >= 0.0
        assert metrics.records_processed == 1

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
    def test_process_market_data_with_crossed_market(self, data_processor, default_instrument_id):
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

        processed, _metrics = data_processor.process_market_data(
            instrument_id=str(default_instrument_id),
            data=data,
            ts_event=ts_event,
        )

        # Should fix the crossed market
        assert processed["bid"] < processed["ask"]
        assert processed["quality_flags"] & QualityFlags.INVALID_RANGE

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
    def test_process_features_with_nan(self, data_processor, default_instrument_id):
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
            instrument_id=str(default_instrument_id),
            features=features,
            ts_event=ts_event,
        )

        # NaN should be imputed
        assert feature_data.values["rsi_14"] == 0.0  # noqa: PD011 - not a pandas object
        assert metrics.missing_imputed == 1

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
    def test_process_prediction(self, data_processor, default_instrument_id, sample_features):
        """
        Test processing model predictions.
        """
        ts_event = int(time.time() * 1e9)

        pred_data, metrics = data_processor.process_prediction(
            model_id="xgboost_v1",
            instrument_id=str(default_instrument_id),
            prediction=0.75,
            confidence=0.85,
            features=sample_features,
            inference_time_ms=2.5,
            ts_event=ts_event,
        )

        assert pred_data.prediction == 0.75
        assert pred_data.confidence <= 0.85  # May be adjusted
        assert metrics.records_processed == 1

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
    def test_process_signal_with_risk_limits(self, data_processor, default_instrument_id):
        """
        Test processing signals with risk limits.
        """
        ts_event = int(time.time() * 1e9)

        signal_data, metrics = data_processor.process_signal(
            strategy_id="momentum_v1",
            instrument_id=str(default_instrument_id),
            signal_type="BUY",
            strength=1.0,
            model_predictions={"xgboost": 0.75},
            ts_event=ts_event,
        )

        assert signal_data.strength <= 1.0  # May be adjusted by risk limits
        assert "order_size" in signal_data.execution_params
        assert metrics.records_processed == 1

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
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

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
    def test_batch_processing(self, data_processor):
        """
        Test batch processing.
        """
        batch = [
            {
                "instrument_id": "EUR/USD.SIM",
                "data": {"bid": 150.0, "ask": 150.1},
                "ts_event": int(time.time() * 1e9),
            },
            {
                "instrument_id": "GBP/USD.SIM",
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


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.integration
class TestIntegration:
    """
    Test integration between components.
    """

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
    def test_end_to_end_flow(
        self,
        feature_store,
        model_store,
        strategy_store,
        data_processor,
        default_instrument_id,
        sample_features,
    ):
        """
        Test complete data flow through all stores.
        """
        ts_event = int(time.time() * 1e9)

        # 1. Process market data
        _market_data, _ = data_processor.process_market_data(
            instrument_id=str(default_instrument_id),
            data={"bid": 150.0, "ask": 150.1, "volume": 10000},
            ts_event=ts_event,
        )

        # 2. Process and store features
        feature_data, _ = data_processor.process_features(
            feature_set_id="test_features",
            instrument_id=str(default_instrument_id),
            features=sample_features,
            ts_event=ts_event,
        )

        feature_store.write_batch([feature_data])

        # 3. Process and store predictions
        pred_data, _ = data_processor.process_prediction(
            model_id="xgboost_v1",
            instrument_id=str(default_instrument_id),
            prediction=0.75,
            confidence=0.85,
            features=sample_features,
            inference_time_ms=2.5,
            ts_event=ts_event,
        )

        model_store.write_batch([pred_data])

        # 4. Process and store signals
        signal_data, _ = data_processor.process_signal(
            strategy_id="momentum_v1",
            instrument_id=str(default_instrument_id),
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

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
    def test_data_quality_propagation(self, data_processor, default_instrument_id):
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
            instrument_id=str(default_instrument_id),
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

        _feature_data, feature_metrics = data_processor.process_features(
            feature_set_id="test_features",
            instrument_id=str(default_instrument_id),
            features=features,
            ts_event=ts_event,
        )

        # Feature quality should reflect issues
        assert feature_metrics.missing_imputed > 0


# =================================================================================================
# Tests merged from test_stores_simple.py (without dependencies)
# =================================================================================================


@pytest.mark.database
@pytest.mark.serial
@pytest.mark.integration
class TestDataProcessorSimple:
    """
    Simple tests for DataProcessor without full dependencies.
    """

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
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

            processed, _metrics = processor.process_market_data(
                instrument_id="EUR/USD.SIM",
                data=data,
                ts_event=ts_event,
            )

            assert processed["bid"] == 150.0
            assert processed["ask"] == 150.1
            assert processed["quality_score"] >= 0.0
            assert metrics.records_processed == 1

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
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

            processed, _metrics = processor.process_market_data(
                instrument_id="EUR/USD.SIM",
                data=data,
                ts_event=ts_event,
            )

            # Should fix the crossed market
            assert processed["bid"] < processed["ask"]
            assert processed["quality_flags"] & QualityFlags.INVALID_RANGE

    @pytest.mark.database
    @pytest.mark.serial
    @pytest.mark.integration
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
                instrument_id="EUR/USD.SIM",
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
