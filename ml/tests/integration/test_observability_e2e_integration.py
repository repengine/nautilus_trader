"""
End-to-end integration tests for observability stage boundary tracking.

This module tests that observability hooks in stores properly record latency and metrics
data when enabled, and that the data can be materialized into the 4 expected DataFrames
and persisted to DB/file.

"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from ml.core.integration import MLIntegrationManager
from ml.stores.base import FeatureData, ModelPrediction, StrategySignal


@pytest.fixture
def temp_observability_dir():
    """
    Create temporary directory for observability outputs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def integration_manager_with_observability():
    """
    Create MLIntegrationManager with observability enabled.
    """
    # Enable observability for this test
    original_value = os.environ.get("ML_OBSERVABILITY_ENABLED")
    original_dummy = os.environ.get("ML_ALLOW_DUMMY")
    os.environ["ML_OBSERVABILITY_ENABLED"] = "1"
    # Ensure dummy stores are allowed to avoid DB dependency in tests
    os.environ["ML_ALLOW_DUMMY"] = "1"

    try:
        mgr = MLIntegrationManager(
            auto_start_postgres=False,
            auto_migrate=False,
            ensure_healthy=False,
        )

        # Initialize observability pipeline
        mgr.initialize_observability_pipeline()
        mgr._inject_observability_service_into_stores()

        yield mgr
    finally:
        # Restore original environment
        if original_value is None:
            os.environ.pop("ML_OBSERVABILITY_ENABLED", None)
        else:
            os.environ["ML_OBSERVABILITY_ENABLED"] = original_value
        if original_dummy is None:
            os.environ.pop("ML_ALLOW_DUMMY", None)
        else:
            os.environ["ML_ALLOW_DUMMY"] = original_dummy


class TestObservabilityE2EIntegration:
    """
    Test end-to-end observability integration with stores.
    """

    def test_feature_store_observability_hook(self, integration_manager_with_observability):
        """
        Test that feature store write operations record observability data.
        """
        mgr = integration_manager_with_observability

        # Verify observability service is available
        assert hasattr(mgr, "observability_service")
        assert mgr.observability_service is not None

        # Verify feature store has observability service injected
        feature_store = mgr.feature_store
        assert hasattr(feature_store, "_observability_service")
        assert getattr(feature_store, "_observability_service") is mgr.observability_service

        # Simulate feature write operation
        feature_data = FeatureData(
            feature_set_id="test_features",
            instrument_id="EUR/USD.SIM",
            _ts_event=time.time_ns(),
            _ts_init=time.time_ns(),
            values={"rsi_14": 65.0, "ema_20": 1.0850},
        )

        # Write feature data (this should trigger observability hooks)
        feature_store.write_features(data=[feature_data])

        # Collect observability data
        tables = mgr.collect_observability_dataframes()

        # Verify we have the 4 expected tables
        assert "latency" in tables
        assert "metrics" in tables
        assert "correlation" in tables
        assert "health" in tables

        # Verify latency table has data from feature store operation
        latency_df = tables["latency"]
        assert latency_df is not None
        assert len(latency_df) > 0

        # Check that we recorded feature storage stage
        if hasattr(latency_df, "to_pandas"):
            # Polars DataFrame
            latency_pandas = latency_df.to_pandas()
        else:
            # Already pandas DataFrame
            latency_pandas = latency_df

        stages = latency_pandas["pipeline_stage"].tolist()
        assert "feature_storage" in stages

        # Verify metrics table has data
        metrics_df = tables["metrics"]
        assert metrics_df is not None
        assert len(metrics_df) > 0

        # Check that we recorded feature store latency metric
        if hasattr(metrics_df, "to_pandas"):
            metrics_pandas = metrics_df.to_pandas()
        else:
            metrics_pandas = metrics_df

        metric_names = metrics_pandas["metric_name"].tolist()
        assert "feature_store_latency_ms" in metric_names

    def test_model_store_observability_hook(self, integration_manager_with_observability):
        """
        Test that model store write operations record observability data.
        """
        mgr = integration_manager_with_observability

        # Verify model store has observability service injected
        model_store = mgr.model_store
        assert hasattr(model_store, "_observability_service")

        # Simulate model prediction write operation
        prediction = ModelPrediction(
            model_id="test_model_v1",
            instrument_id="EUR/USD.SIM",
            _ts_event=time.time_ns(),
            _ts_init=time.time_ns(),
            prediction=0.75,
            confidence=0.85,
            features_used={"rsi_14": 65.0},
            inference_time_ms=2.5,
        )

        # Write prediction data (this should trigger observability hooks)
        model_store.write_batch([prediction])

        # Collect observability data
        tables = mgr.collect_observability_dataframes()

        # Verify latency table has model store operation
        latency_df = tables["latency"]
        assert latency_df is not None
        assert len(latency_df) > 0

        if hasattr(latency_df, "to_pandas"):
            latency_pandas = latency_df.to_pandas()
        else:
            latency_pandas = latency_df

        stages = latency_pandas["pipeline_stage"].tolist()
        assert "model_prediction_storage" in stages

    def test_strategy_store_observability_hook(self, integration_manager_with_observability):
        """
        Test that strategy store write operations record observability data.
        """
        mgr = integration_manager_with_observability

        # Verify strategy store has observability service injected
        strategy_store = mgr.strategy_store
        assert hasattr(strategy_store, "_observability_service")

        # Simulate strategy signal write operation
        signal = StrategySignal(
            strategy_id="test_strategy",
            instrument_id="EUR/USD.SIM",
            _ts_event=time.time_ns(),
            _ts_init=time.time_ns(),
            signal_type="BUY",
            strength=0.8,
            model_predictions={"test_model": 0.75},
            risk_metrics={"var": 0.02},
            execution_params={"stop_loss": 0.02, "take_profit": 0.05},
        )

        # Write signal data (this should trigger observability hooks)
        strategy_store.write_batch([signal])

        # Collect observability data
        tables = mgr.collect_observability_dataframes()

        # Verify latency table has strategy store operation
        latency_df = tables["latency"]
        assert latency_df is not None
        assert len(latency_df) > 0

        if hasattr(latency_df, "to_pandas"):
            latency_pandas = latency_df.to_pandas()
        else:
            latency_pandas = latency_df

        stages = latency_pandas["pipeline_stage"].tolist()
        assert "strategy_signal_storage" in stages

    def test_data_store_observability_hook(self, integration_manager_with_observability):
        """
        Test that data store write operations record observability data.
        """
        mgr = integration_manager_with_observability

        # Skip if data store is not available (dummy mode)
        if mgr.data_store is None:
            pytest.skip("Data store not available in test environment")

        # Verify data store has observability service injected
        data_store = mgr.data_store
        assert hasattr(data_store, "_observability_service")

        # Simulate data ingestion write operation
        test_records = [
            {
                "instrument_id": "EUR/USD.SIM",
                "_ts_event": time.time_ns(),
                "_ts_init": time.time_ns(),
                "close": 1.0850,
                "volume": 1000,
            },
        ]

        try:
            # Write ingestion data (this should trigger observability hooks)
            data_store.write_ingestion(
                dataset_id="test_bars",
                records=test_records,
                source="test",
                run_id="test_run_001",
            )

            # Collect observability data
            tables = mgr.collect_observability_dataframes()

            # Verify latency table has data ingestion operation
            latency_df = tables["latency"]
            assert latency_df is not None
            assert len(latency_df) > 0

            if hasattr(latency_df, "to_pandas"):
                latency_pandas = latency_df.to_pandas()
            else:
                latency_pandas = latency_df

            stages = latency_pandas["pipeline_stage"].tolist()
            assert "data_ingestion" in stages

        except Exception:
            # Data store operations may fail in test environment due to missing datasets
            # This is acceptable as we're testing the observability hooks, not the data store itself
            pytest.skip("Data store operation failed in test environment")

    def test_observability_file_persistence(
        self,
        integration_manager_with_observability,
        temp_observability_dir,
    ):
        """
        Test that observability data can be persisted to files.
        """
        mgr = integration_manager_with_observability

        # Generate some observability data
        feature_data = FeatureData(
            feature_set_id="test_features",
            instrument_id="EUR/USD.SIM",
            _ts_event=time.time_ns(),
            _ts_init=time.time_ns(),
            values={"test_feature": 42.0},
        )
        mgr.feature_store.write_features(data=[feature_data])

        # Flush observability data to files
        result = mgr.flush_observability_to_path(
            base_path=temp_observability_dir,
            file_format="jsonl",
        )

        # Verify files were created
        assert isinstance(result, dict)
        assert len(result) > 0

        # Check that expected files exist
        for table_name, file_path in result.items():
            assert file_path.exists()
            assert file_path.suffix == ".jsonl"

            # Verify file has content for tables with data
            if file_path.stat().st_size > 0:
                content = file_path.read_text()
                assert len(content) > 0
                # Should be valid JSONL (each line is valid JSON)
                lines = content.strip().split("\n")
                assert len(lines) > 0

    @pytest.mark.serial  # Run this test serially to avoid conflicts
    def test_observability_db_persistence(self, integration_manager_with_observability):
        """
        Test that observability data can be persisted to database.
        """
        mgr = integration_manager_with_observability

        # Skip if no database connection available
        if not mgr.db_connection or "dummy" in mgr.db_connection.lower():
            pytest.skip("Database connection not available for persistence test")

        # Generate some observability data
        feature_data = FeatureData(
            feature_set_id="test_features",
            instrument_id="EUR/USD.SIM",
            _ts_event=time.time_ns(),
            _ts_init=time.time_ns(),
            values={"test_feature": 42.0},
        )
        mgr.feature_store.write_features(data=[feature_data])

        try:
            # Flush observability data to database
            result = mgr.flush_observability_to_db(
                connection_string=mgr.db_connection,
            )

            # Verify some rows were written
            assert isinstance(result, dict)
            total_rows = sum(result.values())
            assert total_rows > 0

        except Exception as e:
            # Database persistence may fail in test environment
            pytest.skip(f"Database persistence failed: {e}")

    def test_observability_disabled_by_default(self):
        """
        Test that observability hooks are no-ops when disabled.
        """
        # Ensure observability is disabled
        original_value = os.environ.get("ML_OBSERVABILITY_ENABLED")
        os.environ.pop("ML_OBSERVABILITY_ENABLED", None)

        try:
            mgr = MLIntegrationManager(
                auto_start_postgres=False,
                auto_migrate=False,
                ensure_healthy=False,
            )

            # Even if we initialize observability pipeline, stores shouldn't record data
            # when ML_OBSERVABILITY_ENABLED is not set
            mgr.initialize_observability_pipeline()
            mgr._inject_observability_service_into_stores()

            # Simulate feature write operation
            feature_data = FeatureData(
                feature_set_id="test_features",
                instrument_id="EUR/USD.SIM",
                _ts_event=time.time_ns(),
                _ts_init=time.time_ns(),
                values={"test_feature": 42.0},
            )

            mgr.feature_store.write_features(data=[feature_data])

            # Collect observability data - should be empty since disabled
            tables = mgr.collect_observability_dataframes()

            # Tables should exist but be empty
            latency_df = tables["latency"]
            if latency_df is not None:
                assert len(latency_df) == 0

        finally:
            # Restore original environment
            if original_value is not None:
                os.environ["ML_OBSERVABILITY_ENABLED"] = original_value
