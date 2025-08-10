
"""
Integration examples for extended ML monitoring.

This module demonstrates how to integrate the extended metrics collectors with existing
ML components like MLDataLoader and FeatureEngineer.

"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ml.monitoring._config import MonitoringConfig
from ml.monitoring.collectors.data import DataQualityCollector
from ml.monitoring.collectors.features import FeatureEngineeringCollector
from ml.monitoring.collectors.model import ModelLifecycleCollector
from ml.monitoring.collectors.registry import MLMetricsRegistry


if TYPE_CHECKING:
    import polars as pl

    from ml.data.loader import MLDataLoader
    from ml.features.engineering import FeatureEngineer


# =============================================================================
# MLDataLoader with Metrics Integration
# =============================================================================

# Configure module logger
logger = logging.getLogger(__name__)


class MonitoredMLDataLoader:
    """
    Example of MLDataLoader with integrated metrics collection.

    This demonstrates how to add comprehensive metrics to the data loading process.

    """

    def __init__(
        self,
        base_loader: MLDataLoader,
        metrics_collector: DataQualityCollector | None = None,
    ):
        """
        Initialize monitored data loader.

        Parameters
        ----------
        base_loader : MLDataLoader
            The base data loader to wrap.
        metrics_collector : DataQualityCollector, optional
            Data quality metrics collector.

        """
        self._loader = base_loader
        self._metrics = metrics_collector
        self._cache_hits = 0
        self._cache_misses = 0

    def load_bars(
        self,
        instrument: str,
        start: Any = None,
        end: Any = None,
    ) -> pl.DataFrame:
        """
        Load bars with comprehensive metrics collection.

        Parameters
        ----------
        instrument : str
            Instrument identifier.
        start : Any, optional
            Start timestamp.
        end : Any, optional
            End timestamp.

        Returns
        -------
        pl.DataFrame
            Loaded bars data.

        """
        start_time = time.perf_counter()

        # Check cache (simulated)
        cache_key = f"{instrument}_{start}_{end}"
        is_cache_hit = self._check_cache(cache_key)

        if is_cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        # Load data
        df = self._loader.load_bars(instrument, start, end)

        # Record metrics if collector is available
        if self._metrics and self._metrics.enabled:
            # Record load latency
            latency = time.perf_counter() - start_time
            if hasattr(self._metrics, "_data_load_latency"):
                self._metrics._data_load_latency.labels(
                    instrument=instrument,
                    data_type="bars",
                    source="cache" if is_cache_hit else "disk",
                ).observe(latency)

            # Calculate data quality metrics
            missing_ratios = self._calculate_missing_ratios(df)
            outlier_count = self._detect_outliers(df)

            # Record data quality
            self._metrics.record_data_quality(
                instrument=instrument,
                data_type="bars",
                missing_ratios=missing_ratios,
                outlier_counts={"total": outlier_count},
            )

            # Record data staleness
            staleness = self._calculate_staleness(df)
            self._metrics.update_data_staleness(
                instrument=instrument,
                data_type="bars",
                last_updated_timestamp=time.time() - staleness,
            )

        return df

    def _check_cache(self, key: str) -> bool:
        """
        Check if data is in cache (simulated).
        """
        # In real implementation, check actual cache
        return np.random.random() > 0.3  # 70% cache hit ratio

    def _calculate_missing_ratios(self, df: pl.DataFrame) -> dict[str, float]:
        """
        Calculate missing value ratios per column.
        """
        ratios = {}
        for col in df.columns:
            null_count = df[col].null_count()
            ratios[col] = null_count / len(df) if len(df) > 0 else 0.0
        return ratios

    def _detect_outliers(self, df: pl.DataFrame) -> int:
        """
        Detect outliers using z-score method.
        """
        # Simplified outlier detection for numeric columns
        outlier_count = 0
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64]]

        for col in numeric_cols:
            values = df[col].to_numpy()
            if len(values) > 0:
                mean = np.mean(values)
                std = np.std(values)
                if std > 0:
                    z_scores = np.abs((values - mean) / std)
                    outlier_count += np.sum(z_scores > 3)

        return outlier_count

    def _calculate_staleness(self, df: pl.DataFrame) -> float:
        """
        Calculate data staleness in seconds.
        """
        # In real implementation, compare with current time
        # For demo, return random staleness
        return np.random.uniform(0, 60)


# =============================================================================
# FeatureEngineer with Metrics Integration
# =============================================================================


class MonitoredFeatureEngineer:
    """
    Example of FeatureEngineer with integrated metrics collection.

    This demonstrates how to add feature engineering metrics.

    """

    def __init__(
        self,
        base_engineer: FeatureEngineer,
        metrics_collector: FeatureEngineeringCollector | None = None,
    ):
        """
        Initialize monitored feature engineer.

        Parameters
        ----------
        base_engineer : FeatureEngineer
            The base feature engineer to wrap.
        metrics_collector : FeatureEngineeringCollector, optional
            Feature engineering metrics collector.

        """
        self._engineer = base_engineer
        self._metrics = metrics_collector
        self._feature_cache: dict[str, pl.DataFrame] = {}
        self._cache_stats = {"hits": 0, "misses": 0}

    def compute_features(
        self,
        bars: pl.DataFrame,
        instrument: str = "UNKNOWN",
    ) -> pl.DataFrame:
        """
        Compute features with comprehensive metrics.

        Parameters
        ----------
        bars : pl.DataFrame
            Input bars data.
        instrument : str, default "UNKNOWN"
            Instrument identifier.

        Returns
        -------
        pl.DataFrame
            Computed features.

        """
        start_time = time.perf_counter()

        # Check cache
        cache_key = self._get_cache_key(bars)
        if cache_key in self._feature_cache:
            self._cache_stats["hits"] += 1
            features = self._feature_cache[cache_key]

            # Record cache hit
            if self._metrics and self._metrics.enabled:
                self._metrics.record_cache_hit(
                    instrument=instrument,
                    cache_level="memory",
                )

            return features

        self._cache_stats["misses"] += 1

        # Compute features
        try:
            features, scaler = self._engineer.calculate_features_batch(bars)

            # Cache results
            self._feature_cache[cache_key] = features

            # Record metrics if collector is available
            if self._metrics and self._metrics.enabled:
                latency = time.perf_counter() - start_time

                # Record computation latency
                if hasattr(self._metrics, "_feature_computation_latency"):
                    self._metrics._feature_computation_latency.labels(
                        feature_set="technical",
                        computation_mode="batch",
                    ).observe(latency)

                # Record cache miss
                self._metrics.record_cache_hit(
                    instrument=instrument,
                    cache_level="memory",
                )

                # Calculate and record feature drift (simplified)
                drift_scores = self._calculate_feature_drift(features)
                for feature, drift in drift_scores.items():
                    self._metrics.record_feature_drift(
                        instrument=instrument,
                        feature=feature,
                        drift_score=drift,
                        reference_window="training",
                    )

                # Record feature importance if available
                if hasattr(self._engineer, "feature_importances_"):
                    self._metrics.record_feature_importance(
                        model="current",
                        feature_importances=self._engineer.feature_importances_,
                    )

            return pl.DataFrame(features) if not isinstance(features, pl.DataFrame) else features

        except Exception as e:
            # Record error
            if self._metrics and self._metrics.enabled:
                if hasattr(self._metrics, "_feature_computation_errors"):
                    self._metrics._feature_computation_errors.labels(
                        instrument=instrument,
                        feature_type="technical",
                        error_type=type(e).__name__,
                    ).inc()
            raise

    def _get_cache_key(self, bars: pl.DataFrame) -> str:
        """
        Generate cache key for bars data.
        """
        # Simplified cache key generation
        return f"{len(bars)}_{bars.columns}"

    def _get_cache_hit_ratio(self) -> float:
        """
        Calculate cache hit ratio.
        """
        total = self._cache_stats["hits"] + self._cache_stats["misses"]
        if total == 0:
            return 0.0
        return self._cache_stats["hits"] / total

    def _calculate_feature_drift(self, features: pl.DataFrame) -> dict[str, float]:
        """
        Calculate feature drift scores (simplified).
        """
        drift_scores = {}
        for col in features.columns[:5]:  # Check first 5 features
            # Simplified drift calculation (in reality, compare with reference distribution)
            drift_scores[col] = np.random.uniform(0, 0.3)
        return drift_scores


# =============================================================================
# Model Training with Metrics Integration
# =============================================================================


class MonitoredModelTrainer:
    """
    Example of model training with lifecycle metrics.

    This demonstrates how to track model training and deployment.

    """

    def __init__(
        self,
        metrics_collector: ModelLifecycleCollector | None = None,
    ):
        """
        Initialize monitored model trainer.

        Parameters
        ----------
        metrics_collector : ModelLifecycleCollector, optional
            Model lifecycle metrics collector.

        """
        self._metrics = metrics_collector
        self._phase_times: dict[str, float] = {}

    def train_model(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        model_name: str = "xgboost_v1",
        instrument: str = "EURUSD",
    ) -> Any:
        """
        Train model with comprehensive metrics.

        Parameters
        ----------
        X : npt.NDArray[np.float64]
            Training features.
        y : npt.NDArray[np.float64]
            Training labels.
        model_name : str, default "xgboost_v1"
            Model identifier.
        instrument : str, default "EURUSD"
            Instrument identifier.

        Returns
        -------
        Any
            Trained model.

        """
        overall_start = time.perf_counter()

        # Phase 1: Data preprocessing
        phase_start = time.perf_counter()
        X_processed = self._preprocess_data(X)
        self._phase_times["preprocessing"] = time.perf_counter() - phase_start

        # Phase 2: Model training
        phase_start = time.perf_counter()
        model = self._train_internal(X_processed, y)
        self._phase_times["training"] = time.perf_counter() - phase_start

        # Phase 3: Validation
        phase_start = time.perf_counter()
        self._validate_model(model, X_processed, y)
        self._phase_times["validation"] = time.perf_counter() - phase_start

        # Record metrics if collector is available
        if self._metrics and self._metrics.enabled:
            total_duration = time.perf_counter() - overall_start

            # Record training completed
            self._metrics.record_model_training(
                model=model_name,
                training_samples=len(X),
                training_duration=total_duration,
            )

            # Record deployment
            self._metrics.record_model_deployment(
                model=model_name,
                version="1.0.0",
                instrument=instrument,
                git_commit="abc123def",
            )

        return model

    def _preprocess_data(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Preprocess training data.
        """
        # Simplified preprocessing
        return X

    def _train_internal(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> Any:
        """
        Train the model (simplified).
        """
        # In reality, train actual model
        return {"model": "dummy", "trained": True}

    def _validate_model(self, model: Any, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        """
        Validate model performance.
        """
        # Simplified validation
        return 0.85

    def _get_model_size(self, model: Any) -> int:
        """
        Get model size in bytes.
        """
        # Simplified size calculation
        return 1024 * 1024  # 1MB


# =============================================================================
# Complete Integration Example
# =============================================================================


def example_complete_integration() -> None:
    """
    Complete example showing all components working together.
    """
    # Initialize monitoring configuration
    config = MonitoringConfig(
        enabled=True,
        metrics_port=8080,
        enable_high_cardinality=False,
        metrics_prefix="nautilus_ml",
    )

    # Create metrics registry
    metrics = MLMetricsRegistry(config)

    # Start metrics server
    metrics.start()

    try:
        # Example 1: Data Loading with Metrics
        logger.info("Loading data with quality metrics...")
        from ml.data.loader import MLDataLoader
        from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

        catalog = ParquetDataCatalog("./data")
        base_loader = MLDataLoader(catalog)

        monitored_loader = MonitoredMLDataLoader(
            base_loader,
            metrics_collector=metrics.data_quality,
        )

        # Load data (will record metrics)
        bars_df = monitored_loader.load_bars(
            "EURUSD",
            start="2024-01-01",
            end="2024-01-31",
        )
        logger.info(f"Loaded {len(bars_df)} bars")

        # Example 2: Feature Engineering with Metrics
        logger.info("\nComputing features with drift detection...")
        from ml.features.engineering import FeatureConfig
        from ml.features.engineering import FeatureEngineer

        feature_config = FeatureConfig()
        base_engineer = FeatureEngineer(feature_config)
        monitored_engineer = MonitoredFeatureEngineer(
            base_engineer,
            metrics_collector=metrics.feature_engineering,
        )

        # Compute features (will record metrics)
        features_df = monitored_engineer.compute_features(
            bars_df,
            instrument="EURUSD",
        )
        logger.info(f"Computed {len(features_df.columns)} features")

        # Example 3: Model Training with Lifecycle Metrics
        logger.info("Training model with lifecycle tracking...")
        trainer = MonitoredModelTrainer(
            metrics_collector=metrics.model_lifecycle,
        )

        # Prepare training data
        X = features_df.to_numpy()
        y = np.random.randint(0, 2, size=len(features_df)).astype(np.float64)

        # Train model (will record metrics)
        trainer.train_model(
            X,
            y,
            model_name="xgboost_demo",
            instrument="EURUSD",
        )
        logger.info("Model trained successfully")

        # Example 4: Real-time Inference with Metrics
        logger.info("\nPerforming inference with latency tracking...")

        # Use existing MLMetricsCollector for predictions
        for i in range(10):
            with metrics.ml_metrics.time_prediction("xgboost_demo", "EURUSD") as timer:
                # Simulate inference
                time.sleep(np.random.uniform(0.001, 0.01))

                # Set prediction details
                timer.set_prediction(
                    prediction_class="buy" if i % 2 == 0 else "sell",
                    confidence=np.random.uniform(0.6, 0.95),
                )

        logger.info("Performed 100 inferences with metrics")

        # Print metrics URL
        logger.info(f"\nMetrics available at: {metrics.server.get_metrics_url()}")
        logger.info(f"Health check at: {metrics.server.get_health_url()}")

        # Simulate running for a bit
        logger.info("\nServer running... Press Ctrl+C to stop")
        time.sleep(5)

    finally:
        # Cleanup
        metrics.stop()
        logger.info("\nMetrics server stopped")


# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == "__main__":
    # Run complete integration example
    example_complete_integration()
