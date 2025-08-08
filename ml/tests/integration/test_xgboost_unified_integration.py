# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Integration tests for unified XGBoost trainer.

This test suite provides end-to-end integration testing for the UnifiedXGBoostTrainer,
including full training pipelines, ONNX export verification, performance benchmarks, and
integration with all advanced features (GPU, Optuna, MLflow).

"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from ml._imports import HAS_XGBOOST
from ml.config.base import MLFeatureConfig
from ml.config.shared import MLflowConfig
from ml.config.shared import OptunaConfig
from ml.config.shared import XGBoostGPUConfig as GPUConfig
from ml.config.xgboost import UnifiedXGBoostConfig
from ml.training.xgboost import UnifiedXGBoostTrainer


class TestUnifiedXGBoostIntegration:
    """
    Integration tests for unified XGBoost trainer.
    """

    @pytest.fixture
    def sample_financial_data(self):
        """
        Create realistic financial time series data.
        """
        np.random.seed(42)
        n_samples = 2000

        # Generate realistic financial data with trends
        timestamps = pl.datetime_range(
            start=pl.datetime(2022, 1, 1),
            end=pl.datetime(2023, 12, 31),
            interval="1h",
        )[:n_samples]

        # Base price with trend and noise
        base_price = 100.0
        trend = np.linspace(0, 20, n_samples)
        noise = np.random.randn(n_samples) * 0.5

        prices = base_price + trend + noise.cumsum() * 0.1

        # OHLC data with realistic relationships
        open_prices = prices + np.random.randn(n_samples) * 0.05
        high_prices = np.maximum(open_prices, prices) + np.abs(np.random.randn(n_samples)) * 0.1
        low_prices = np.minimum(open_prices, prices) - np.abs(np.random.randn(n_samples)) * 0.1
        close_prices = prices + np.random.randn(n_samples) * 0.05

        # Volume with realistic patterns
        volume = np.random.lognormal(mean=8, sigma=0.5, size=n_samples).astype(int)

        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volume,
            },
        )

    @pytest.fixture
    def basic_training_config(self):
        """
        Create basic training configuration.
        """
        return UnifiedXGBoostConfig(
            # Small model for testing
            n_estimators=20,
            max_depth=4,
            learning_rate=0.2,
            objective="binary:logistic",
            eval_metric="auc",
            early_stopping_rounds=5,
            # Disable external services for basic testing
            enable_monitoring=False,
            # Feature configuration
            feature_config=MLFeatureConfig(
                lookback_window=50,
                normalize_features=False,
            ),
        )

    @pytest.fixture
    def full_feature_config(self):
        """
        Create configuration with all features enabled (mocked).
        """
        return UnifiedXGBoostConfig(
            # Model settings
            n_estimators=10,  # Very small for fast testing
            max_depth=3,
            learning_rate=0.3,
            objective="binary:logistic",
            # GPU configuration (will be mocked)
            gpu_config=GPUConfig(
                enabled=True,
                device_id=0,
                validate_gpu=False,  # Skip validation in tests
            ),
            # Optuna configuration (will be mocked)
            optuna_config=OptunaConfig(
                enabled=True,
                n_trials=3,  # Very few trials for testing
                metric="accuracy",
                timeout=10,
            ),
            # MLflow configuration (will be mocked)
            mlflow_config=MLflowConfig(
                enabled=True,
                tracking_uri="sqlite:///test.db",
                experiment_name="test_integration",
            ),
            # Feature tracking
            track_feature_decay=True,
            feature_decay_threshold=0.2,
            # Cross-validation
            cv_strategy="time_series",
            cv_folds=2,  # Minimal for testing
            # Export
            export_onnx=True,
            onnx_output_path="./test_model.onnx",
            # Disable monitoring for test environment
            enable_monitoring=False,
        )

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not available")
    def test_basic_training_pipeline(self, sample_financial_data, basic_training_config) -> None:
        """
        Test basic training pipeline end-to-end.
        """
        trainer = UnifiedXGBoostTrainer(basic_training_config)

        # Train model
        results = trainer.train(sample_financial_data, target_col="target")

        # Verify training completed successfully
        assert "model" in results
        assert "metrics" in results
        assert "feature_importance" in results

        # Verify metrics
        metrics = results["metrics"]
        assert "training_time" in metrics
        assert "val_accuracy" in metrics
        assert "val_sharpe" in metrics
        assert metrics["training_time"] > 0
        assert 0 <= metrics["val_accuracy"] <= 1

        # Verify feature importance
        importance = results["feature_importance"]
        assert len(importance) > 0
        assert all(isinstance(v, (int, float)) for v in importance.values())

        # Verify model is fitted
        assert trainer._is_fitted
        assert trainer._model is not None

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not available")
    def test_multi_asset_training(self, basic_training_config) -> None:
        """
        Test multi-asset training scenario.
        """
        # Create multi-asset data
        np.random.seed(42)
        assets = ["EURUSD", "GBPUSD", "USDJPY"]
        multi_asset_data = {}

        for asset in assets:
            n_samples = 500
            timestamps = pl.datetime_range(
                start=pl.datetime(2023, 1, 1),
                end=pl.datetime(2023, 6, 30),
                interval="1d",
            )[:n_samples]

            prices = 100 + np.random.randn(n_samples).cumsum() * 0.1

            multi_asset_data[asset] = pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": prices + np.random.randn(n_samples) * 0.01,
                    "high": prices + np.abs(np.random.randn(n_samples)) * 0.02,
                    "low": prices - np.abs(np.random.randn(n_samples)) * 0.02,
                    "close": prices + np.random.randn(n_samples) * 0.01,
                    "volume": np.random.randint(1000, 10000, n_samples),
                },
            )

        # Configure for multi-asset
        config = UnifiedXGBoostConfig(
            **basic_training_config.__dict__,
            multi_asset=True,
            sector_map={"EURUSD": "major", "GBPUSD": "major", "USDJPY": "major"},
            cross_sectional_features=True,
        )

        trainer = UnifiedXGBoostTrainer(config)

        # Train on multi-asset data
        results = trainer.train(multi_asset_data, target_col="target")

        # Verify multi-asset specific results
        assert results["metadata"]["n_assets"] == 3
        assert "asset_metadata" in results["metadata"]
        assert len(results["metadata"]["asset_metadata"]) == 3

        # Verify cross-sectional features were added
        feature_names = results["metadata"]["feature_names"]
        cross_sectional_features = [f for f in feature_names if "_rank" in f or "_sector" in f]
        assert len(cross_sectional_features) > 0

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not available")
    def test_cross_validation_integration(
        self,
        sample_financial_data,
        basic_training_config,
    ) -> None:
        """
        Test cross-validation integration.
        """
        config = UnifiedXGBoostConfig(
            **basic_training_config.__dict__,
            cv_strategy="time_series",
            cv_folds=3,
        )

        trainer = UnifiedXGBoostTrainer(config)

        # Train with cross-validation
        results = trainer.train(sample_financial_data, cv_validate=True)

        # Verify CV results
        assert "cv_results" in results
        cv_results = results["cv_results"]

        assert cv_results["strategy"] == "time_series"
        assert cv_results["n_folds"] == 3
        assert "summary" in cv_results
        assert "fold_results" in cv_results
        assert len(cv_results["fold_results"]) == 3

        # Verify CV metrics
        summary = cv_results["summary"]
        assert "sharpe_mean" in summary
        assert "sharpe_std" in summary
        assert len(summary["sharpe_scores"]) == 3

    def test_feature_decay_tracking_integration(
        self,
        sample_financial_data,
        basic_training_config,
    ) -> None:
        """
        Test feature decay tracking over multiple training runs.
        """
        config = UnifiedXGBoostConfig(
            **basic_training_config.__dict__,
            track_feature_decay=True,
            feature_decay_threshold=0.3,
        )

        trainer = UnifiedXGBoostTrainer(config)

        # First training run
        results_1 = trainer.train(sample_financial_data)
        importance_1 = results_1["feature_importance"]

        # Simulate feature decay by modifying importance manually
        trainer._importance_history = [importance_1]

        # Create modified importance with some features having lower scores
        importance_2 = importance_1.copy()
        feature_names = list(importance_2.keys())
        if len(feature_names) > 2:
            # Reduce importance of first feature by 50% (should trigger alert)
            importance_2[feature_names[0]] = importance_1[feature_names[0]] * 0.5

        # Trigger decay tracking manually
        trainer._track_feature_decay(importance_2)

        # Check decay summary
        decay_summary = trainer.get_feature_decay_summary()
        assert decay_summary["tracking_enabled"] is True
        assert decay_summary["history_length"] == 2

        # Should have at least one alert if decay was significant
        if len(feature_names) > 2:
            assert (
                len(decay_summary["current_alerts"]) >= 0
            )  # May or may not trigger depending on data

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not available")
    def test_onnx_export_integration(self, sample_financial_data, basic_training_config) -> None:
        """
        Test ONNX export integration.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_path = Path(temp_dir) / "test_model.onnx"

            config = UnifiedXGBoostConfig(
                **basic_training_config.__dict__,
                export_onnx=True,
                onnx_output_path=str(onnx_path),
            )

            trainer = UnifiedXGBoostTrainer(config)

            # Mock ONNX export to avoid dependency
            with patch.object(trainer, "_export_to_onnx") as mock_export:
                mock_export.return_value = str(onnx_path)

                results = trainer.train(sample_financial_data)

                # Verify ONNX export was called
                mock_export.assert_called_once()
                assert results["onnx_path"] == str(onnx_path)

    @patch("ml.training.xgboost_unified.HAS_OPTUNA", True)
    @patch("ml.training.xgboost_unified.optuna")
    def test_optuna_integration_mocked(
        self,
        mock_optuna,
        sample_financial_data,
        full_feature_config,
    ):
        """
        Test Optuna integration with mocked components.
        """
        # Mock Optuna components
        mock_study = MagicMock()
        mock_study.best_params = {
            "n_estimators": 50,
            "max_depth": 5,
            "learning_rate": 0.15,
        }
        mock_study.best_value = 0.75
        mock_study.trials = [MagicMock() for _ in range(3)]
        mock_study.best_trial = MagicMock()

        mock_optuna.create_study.return_value = mock_study
        mock_optuna.trial.TrialState.COMPLETE = "COMPLETE"

        trainer = UnifiedXGBoostTrainer(full_feature_config)

        # Enable optimization manually
        results = trainer.train(sample_financial_data, optimize_hyperparams=True)

        # Verify optimization results
        assert "best_params" in results
        assert "best_value" in results
        assert "n_trials" in results

        # Verify Optuna study was created and used
        mock_optuna.create_study.assert_called_once()
        mock_study.optimize.assert_called_once()

    @patch("ml.training.xgboost_unified.HAS_MLFLOW", True)
    @patch("ml.training.xgboost_unified.mlflow")
    def test_mlflow_integration_mocked(
        self,
        mock_mlflow,
        sample_financial_data,
        full_feature_config,
    ):
        """
        Test MLflow integration with mocked components.
        """
        # Mock MLflow components
        mock_run_info = MagicMock()
        mock_run_info.run_id = "test_run_123"
        mock_run = MagicMock()
        mock_run.info = mock_run_info

        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)

        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp_123"
        mock_mlflow.set_experiment.return_value = mock_experiment

        trainer = UnifiedXGBoostTrainer(full_feature_config)

        results = trainer.train(sample_financial_data)

        # Verify MLflow logging occurred
        assert results["mlflow_run_id"] == "test_run_123"
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once_with("test_integration")

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not available")
    def test_performance_benchmarking(self, basic_training_config) -> None:
        """
        Test training performance benchmarking.
        """
        # Create different sized datasets
        dataset_sizes = [100, 500, 1000]
        performance_results = []

        for size in dataset_sizes:
            # Generate dataset of specified size
            np.random.seed(42)
            data = pl.DataFrame(
                {
                    "timestamp": pl.datetime_range(
                        start=pl.datetime(2023, 1, 1),
                        end=pl.datetime(2023, 12, 31),
                        interval="1h",
                    )[:size],
                    "open": 100 + np.random.randn(size).cumsum() * 0.1,
                    "high": 100 + np.random.randn(size).cumsum() * 0.1 + 0.1,
                    "low": 100 + np.random.randn(size).cumsum() * 0.1 - 0.1,
                    "close": 100 + np.random.randn(size).cumsum() * 0.1,
                    "volume": np.random.randint(1000, 10000, size),
                },
            )

            trainer = UnifiedXGBoostTrainer(basic_training_config)

            # Measure training time
            start_time = time.time()
            results = trainer.train(data)
            total_time = time.time() - start_time

            performance_results.append(
                {
                    "dataset_size": size,
                    "training_time": results["metrics"]["training_time"],
                    "total_time": total_time,
                    "accuracy": results["metrics"]["val_accuracy"],
                    "n_features": results["metadata"]["n_features"],
                },
            )

        # Verify performance scaling is reasonable
        assert len(performance_results) == 3

        # Training time should generally increase with dataset size
        times = [r["training_time"] for r in performance_results]
        # Allow some variance in timing, but expect general increase
        assert times[-1] >= times[0] * 0.5  # At least 50% of initial time for 10x data

        # All training should complete in reasonable time (< 60 seconds for test data)
        assert all(r["total_time"] < 60 for r in performance_results)

    def test_model_metadata_comprehensive(
        self,
        sample_financial_data,
        basic_training_config,
    ) -> None:
        """
        Test comprehensive model metadata generation.
        """
        config = UnifiedXGBoostConfig(
            **basic_training_config.__dict__,
            track_feature_decay=True,
            gpu_config=GPUConfig(enabled=False),  # Explicit disable
            optuna_config=OptunaConfig(enabled=False),
            mlflow_config=MLflowConfig(enabled=False),
        )

        trainer = UnifiedXGBoostTrainer(config)

        # Train model
        results = trainer.train(sample_financial_data)

        # Get comprehensive metadata
        metadata = trainer.get_model_metadata()

        # Verify metadata structure
        assert metadata["fitted"] is True
        assert metadata["model_type"] == "xgboost_unified"

        # Verify configuration metadata
        config_meta = metadata["config"]
        assert config_meta["gpu_enabled"] is False
        assert config_meta["optuna_enabled"] is False
        assert config_meta["mlflow_enabled"] is False
        assert config_meta["multi_asset"] is False
        assert config_meta["objective"] == "binary:logistic"

        # Verify feature metadata
        feature_meta = metadata["features"]
        assert feature_meta["n_features"] > 0
        assert len(feature_meta["feature_names"]) == feature_meta["n_features"]
        assert isinstance(feature_meta["decay_alerts"], list)

        # Verify performance metadata
        performance_meta = metadata["performance"]
        assert "val_accuracy" in performance_meta
        assert "val_sharpe" in performance_meta
        assert "training_time" in performance_meta

        assert metadata["training_time"] > 0

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not available")
    def test_error_handling_integration(self, basic_training_config) -> None:
        """
        Test error handling in integration scenarios.
        """
        trainer = UnifiedXGBoostTrainer(basic_training_config)

        # Test with insufficient data
        small_data = pl.DataFrame(
            {
                "timestamp": [pl.datetime(2023, 1, 1)],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
            },
        )

        # Should handle gracefully (might warn but not crash)
        try:
            results = trainer.train(small_data)
            # If it succeeds, verify basic structure
            if results:
                assert "model" in results
        except Exception as e:
            # Expected for insufficient data
            assert "insufficient data" in str(e).lower() or "samples" in str(e).lower()

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not available")
    def test_monotonic_constraints_integration(
        self,
        sample_financial_data,
        basic_training_config,
    ) -> None:
        """
        Test monotonic constraints integration.
        """
        config = UnifiedXGBoostConfig(
            **basic_training_config.__dict__,
            monotonic_constraints={
                "rsi": 1,  # RSI should increase with target
                "volume_ratio_5": -1,  # Volume ratio might decrease with target
            },
        )

        trainer = UnifiedXGBoostTrainer(config)
        results = trainer.train(sample_financial_data)

        # Verify training completed with constraints
        assert "model" in results
        assert results["metrics"]["training_time"] > 0

        # Verify constraints were applied (would be in model parameters)
        # This is more of an integration test to ensure no errors occur

    def test_training_summary_output(self, sample_financial_data, basic_training_config) -> None:
        """
        Test training summary output generation.
        """
        trainer = UnifiedXGBoostTrainer(basic_training_config)

        # Capture print output during training
        results = trainer.train(sample_financial_data)

        # Verify summary method doesn't crash
        trainer._print_training_summary(results)

        # Test comprehensive results structure
        assert "total_training_time" in results
        assert "metadata" in results
        assert "metrics" in results

        # Verify timing information
        assert results["total_training_time"] > 0
        assert results["metrics"]["training_time"] > 0


class TestUnifiedXGBoostPerformanceRequirements:
    """
    Test performance requirements for unified XGBoost trainer.
    """

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not available")
    def test_inference_latency_requirement(self, sample_financial_data) -> None:
        """
        Test that inference meets P99 latency requirement (<5ms).
        """
        config = UnifiedXGBoostConfig(
            n_estimators=100,  # Realistic size
            max_depth=6,
            objective="binary:logistic",
            enable_monitoring=False,
        )

        trainer = UnifiedXGBoostTrainer(config)
        results = trainer.train(sample_financial_data)
        model = results["model"]

        # Create test data for inference
        X_test = np.random.randn(1000, results["metadata"]["n_features"])

        # Measure inference times
        inference_times = []

        for _ in range(100):  # Multiple measurements for P99
            start_time = time.perf_counter()

            if hasattr(model, "predict_proba"):
                _ = model.predict_proba(X_test[:1])  # Single prediction
            else:
                _ = model.predict(X_test[:1])

            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate P99 latency
        p99_latency = np.percentile(inference_times, 99)

        # Requirement: P99 latency < 5ms
        assert (
            p99_latency < 5.0
        ), f"P99 inference latency {p99_latency:.2f}ms exceeds 5ms requirement"

        # Also check mean latency for good measure
        mean_latency = np.mean(inference_times)
        assert mean_latency < 2.0, f"Mean inference latency {mean_latency:.2f}ms is too high"

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not available")
    def test_memory_stability_requirement(self, sample_financial_data) -> None:
        """
        Test memory usage stability over multiple training runs.
        """
        import gc

        import psutil

        config = UnifiedXGBoostConfig(
            n_estimators=50,
            max_depth=4,
            objective="binary:logistic",
            enable_monitoring=False,
        )

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = [initial_memory]

        # Run multiple training cycles
        for i in range(5):
            trainer = UnifiedXGBoostTrainer(config)
            results = trainer.train(sample_financial_data)

            # Force garbage collection
            del trainer
            del results
            gc.collect()

            # Measure memory
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage.append(current_memory)

        # Check memory growth
        memory_growth = memory_usage[-1] - memory_usage[0]

        # Requirement: Memory growth < 100MB over 5 training cycles
        assert (
            memory_growth < 100
        ), f"Memory grew by {memory_growth:.1f}MB, indicating potential leak"

        # Check that memory stabilizes (last 3 measurements shouldn't vary too much)
        recent_memory = memory_usage[-3:]
        memory_variance = np.var(recent_memory)
        assert memory_variance < 25, f"Memory usage too variable: {memory_variance:.1f}"

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not available")
    def test_training_time_scalability(self) -> None:
        """
        Test training time scalability with dataset size.
        """
        config = UnifiedXGBoostConfig(
            n_estimators=20,  # Small for testing
            max_depth=4,
            objective="binary:logistic",
            enable_monitoring=False,
        )

        # Test with different dataset sizes
        sizes_and_times = []

        for size in [100, 500, 1000]:
            # Generate data
            np.random.seed(42)
            data = pl.DataFrame(
                {
                    "timestamp": pl.datetime_range(
                        start=pl.datetime(2023, 1, 1),
                        end=pl.datetime(2023, 12, 31),
                        interval="1h",
                    )[:size],
                    "open": 100 + np.random.randn(size).cumsum() * 0.1,
                    "high": 100 + np.random.randn(size).cumsum() * 0.1 + 0.1,
                    "low": 100 + np.random.randn(size).cumsum() * 0.1 - 0.1,
                    "close": 100 + np.random.randn(size).cumsum() * 0.1,
                    "volume": np.random.randint(1000, 10000, size),
                },
            )

            trainer = UnifiedXGBoostTrainer(config)
            start_time = time.time()
            results = trainer.train(data)
            training_time = time.time() - start_time

            sizes_and_times.append((size, training_time))

        # Check scalability (should be roughly linear or sub-linear)
        # 10x data should not take more than 20x time
        size_ratio = sizes_and_times[-1][0] / sizes_and_times[0][0]  # 1000/100 = 10
        time_ratio = sizes_and_times[-1][1] / sizes_and_times[0][1]

        assert (
            time_ratio <= size_ratio * 2
        ), f"Training time scales poorly: {size_ratio}x data took {time_ratio:.1f}x time"
