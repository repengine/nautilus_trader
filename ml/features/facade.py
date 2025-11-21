"""
FeatureEngineer Facade - Component-based feature engineering architecture.

This facade integrates 5 extracted components to provide the same API as the legacy
FeatureEngineer while enabling better modularity and testability.

Components:
    1. FeatureStoreAccessor: Feature persistence and retrieval
    2. FeatureRegistryAccessor: Registry read/write operations
    3. FeatureMetricsCollector: Metrics calculation and observability
    4. FeatureCalculator: Core feature computation (HOT PATH - P99 < 0.4ms)
    5. DataExtractor: Data extraction from DataFrames

Architecture Pattern:
    - Pattern 1: Mandatory 4-Store + 4-Registry Integration via accessor components
    - Pattern 2: Protocol-First Interface Design for component boundaries
    - Pattern 3: Hot/Cold Path Separation (FeatureCalculator is hot path optimized)
    - Pattern 4: Progressive Fallback Chains for store/registry unavailability
    - Pattern 5: Centralized Metrics Bootstrap for observability

Performance Requirements:
    - P99 < 5ms for hot path feature computation
    - Overhead < 10% vs legacy implementation
    - Zero memory leaks over 1000 iterations

Backward Compatibility:
    - All public methods from legacy FeatureEngineer preserved
    - Configuration parameters unchanged
    - Return types and signatures identical

Phase: 2.1.6 - Final integration step
Status: Production-ready facade with component delegation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Self, cast, overload

import numpy as np
import numpy.typing as npt
import polars as pl

# Import ML dependencies
# Import extracted components
from ml.features.components.data_extractor import DataExtractor
from ml.features.components.feature_calculator import FeatureCalculator
from ml.features.components.feature_metrics_collector import FeatureMetricsCollector
from ml.features.components.feature_registry_accessor import FeatureRegistryAccessor
from ml.features.components.feature_store_accessor import FeatureStoreAccessor

# Import configuration and indicators
from ml.features.config import FeatureConfig
from ml.features.config import build_pipeline_spec_from_feature_config
from ml.features.indicators import IndicatorManager
from ml.features.pipeline import PipelineSpec
from ml.features.pipeline import _hash_pipeline
from ml.ml_types import DataFrameLike
from ml.ml_types import StandardScaler as StandardScalerT
from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRole
from nautilus_trader.model.data import Bar


if TYPE_CHECKING:
    from typing import Protocol

    from ml.monitoring.collectors.features import FeatureEngineeringCollector
    from ml.stores.protocols import FeatureStoreStrictProtocol

    class ComputeTimerProtocol(Protocol):
        def __enter__(self) -> object: ...
        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            _tb: object | None,
        ) -> bool | None: ...
        def set_computation_result(
            self,
            *,
            features_computed: int,
            cache_hit: bool,
            **kwargs: object,
        ) -> None: ...


logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Facade integrating 5 extracted components for modular feature engineering.

    This facade provides the same API as the legacy FeatureEngineer but delegates
    to specialized components for better separation of concerns. It supports both
    batch (training) and online (inference) feature computation with guaranteed
    mathematical parity.

    Components Integrated:
        - FeatureStoreAccessor: Store read/write operations
        - FeatureRegistryAccessor: Registry read/write operations
        - FeatureMetricsCollector: Metrics and observability
        - FeatureCalculator: Core feature computation (HOT PATH)
        - DataExtractor: Data extraction from DataFrames

    Performance Characteristics:
        - Hot path P99 < 5ms (compute_features with pre-warmed indicators)
        - Facade overhead < 10% vs legacy implementation
        - Zero allocations in hot path after warmup
        - Memory stable over 1000+ iterations

    Parameters
    ----------
    config : FeatureConfig
        Feature engineering configuration specifying which features to compute
    stores : object | None, optional
        Stores container for persistence (Feature/Model/Strategy/Data stores)
    feature_store : FeatureStoreStrictProtocol | None, optional
        Direct feature store instance (alternative to stores container)
    metrics_collector : FeatureEngineeringCollector | None, optional
        Metrics collector for observability
    logger : logging.Logger | None, optional
        Logger instance for diagnostic output

    Attributes
    ----------
    config : FeatureConfig
        Feature configuration
    calculator : FeatureCalculator
        HOT PATH component for feature computation
    data_extractor : DataExtractor
        Data extraction component
    store_accessor : FeatureStoreAccessor
        Feature store operations
    registry_accessor : FeatureRegistryAccessor
        Registry operations
    metrics_collector : FeatureMetricsCollector
        Metrics calculation component

    Examples
    --------
    Basic usage (batch mode):
    >>> import polars as pl
    >>> config = FeatureConfig(
    ...     return_periods=[1, 5, 10],
    ...     rsi_period=14,
    ...     bb_period=20,
    ... )
    >>> engineer = FeatureEngineer(config)
    >>> df = pl.read_parquet("market_data.parquet")
    >>> features_df, scaler = engineer.calculate_features_batch(df, fit_scaler=True)

    Online mode (inference):
    >>> indicator_mgr = IndicatorManager(config)
    >>> # ... update indicators from bars ...
    >>> current_bar = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10000}
    >>> features = engineer.calculate_features_online(current_bar, indicator_mgr, scaler=scaler)

    With stores and registries:
    >>> from ml.common.actor_initialization import init_ml_stores_and_registries
    >>> from ml.config.core import DatabaseConfig
    >>> db_config = DatabaseConfig.from_env()
    >>> stores = init_ml_stores_and_registries(db_config)
    >>> engineer = FeatureEngineer(config, stores=stores)
    >>> # Now can persist features and register schemas

    """

    def __init__(
        self,
        config: FeatureConfig,
        *,
        stores: object | None = None,
        feature_store: FeatureStoreStrictProtocol | None = None,
        metrics_collector: FeatureEngineeringCollector | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the FeatureEngineer facade with all 5 components.

        Parameters
        ----------
        config : FeatureConfig
            Feature engineering configuration
        stores : object | None, optional
            Stores container with feature_store, model_store, etc.
        feature_store : FeatureStoreStrictProtocol | None, optional
            Direct feature store instance (alternative to stores)
        metrics_collector : FeatureEngineeringCollector | None, optional
            Metrics collector for observability
        logger : logging.Logger | None, optional
            Logger instance

        """
        self.config = config
        self._logger = logger if logger is not None else globals()["logger"]

        # Initialize all 5 extracted components
        # Component 1: DataExtractor (stateless data extraction)
        self.data_extractor = DataExtractor()

        # Component 2: FeatureCalculator (HOT PATH - core computation)
        self.calculator = FeatureCalculator(config=config, logger=self._logger)

        # Component 3: FeatureStoreAccessor (persistence operations)
        self.store_accessor = FeatureStoreAccessor(
            feature_store=feature_store,
        )

        # Component 4: FeatureRegistryAccessor (registry operations)
        self.registry_accessor = FeatureRegistryAccessor(stores=stores)

        # Component 5: FeatureMetricsCollector (metrics calculation)
        self.metrics_collector_component = FeatureMetricsCollector(logger=self._logger)

        # Store references for backward compatibility
        self._stores = stores
        self._feature_store = feature_store
        self._metrics = metrics_collector
        self.scaler: StandardScalerT | None = None

    def reset(self) -> None:
        """
        Reset all stateful components to initial state.

        This clears any cached state in components and resets the scaler.

        Examples
        --------
        >>> engineer = FeatureEngineer(config)
        >>> # ... compute features ...
        >>> engineer.reset()  # Clear all state

        """
        self.scaler = None

    # Property accessors for 4-store + 4-registry pattern (via accessor components)

    @property
    def feature_store(self) -> object | None:
        """
        Access feature store via store accessor component.
        """
        return self.store_accessor._feature_store

    @property
    def model_store(self) -> object | None:
        """
        Access model store via registry accessor component.
        """
        return self.registry_accessor.model_registry  # Note: legacy uses registry for store access

    @property
    def strategy_store(self) -> object | None:
        """
        Access strategy store via registry accessor component.
        """
        return self.registry_accessor.strategy_registry

    @property
    def data_store(self) -> object | None:
        """
        Access data store via registry accessor component.
        """
        return self.registry_accessor.data_registry

    @property
    def feature_registry(self) -> object | None:
        """
        Access feature registry via registry accessor component.
        """
        return self.registry_accessor.feature_registry

    @property
    def model_registry(self) -> object | None:
        """
        Access model registry via registry accessor component.
        """
        return self.registry_accessor.model_registry

    @property
    def strategy_registry(self) -> object | None:
        """
        Access strategy registry via registry accessor component.
        """
        return self.registry_accessor.strategy_registry

    @property
    def data_registry(self) -> object | None:
        """
        Access data registry via registry accessor component.
        """
        return self.registry_accessor.data_registry

    # Public API methods - delegate to components

    @property
    def n_features(self) -> int:
        """
        Access total number of features from calculator component.
        """
        return self.calculator.n_features

    @property
    def feature_buffer(self) -> npt.NDArray[np.float32]:
        """
        Access feature buffer from calculator component (hot path optimization).
        """
        return self.calculator.feature_buffer

    def build_pipeline_spec_from_config(self) -> PipelineSpec:
        """
        Build PipelineSpec from feature configuration.

        Returns
        -------
        PipelineSpec
            Pipeline specification for feature computation

        """
        return build_pipeline_spec_from_feature_config(self.config)

    def generate_feature_manifest(
        self,
        name: str,
        version: str,
        role: FeatureRole,
        data_requirements: DataRequirements | None = None,
        pipeline_version: str = "1.0.0",
        capability_flags: dict[str, object] | None = None,
        constraints: dict[str, object] | None = None,
        parity_tolerance: float = 0.0,
        parity_digest: dict[str, object] | None = None,
        perf_digest: dict[str, object] | None = None,
        parent_feature_set_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> FeatureManifest:
        """
        Generate a FeatureManifest for registry registration.

        Parameters
        ----------
        name : str
            Name of the feature set
        version : str
            Semantic version for the feature set
        role : FeatureRole
            Role of the feature set (e.g., PREDICTOR, TARGET)
        data_requirements : DataRequirements | None, optional
            Data requirements for the feature set
        pipeline_version : str, default "1.0.0"
            Version of the pipeline specification
        capability_flags : dict[str, object] | None, optional
            Capability flags for the feature set
        constraints : dict[str, object] | None, optional
            Constraints for the feature set
        parity_tolerance : float, default 0.0
            Tolerance for parity checking
        parity_digest : dict[str, object] | None, optional
            Parity digest for validation
        perf_digest : dict[str, object] | None, optional
            Performance digest
        parent_feature_set_id : str | None, optional
            Parent feature set ID
        metadata : dict[str, object] | None, optional
            Additional metadata

        Returns
        -------
        FeatureManifest
            Manifest ready for registry registration

        """
        spec = self.build_pipeline_spec_from_config()
        feature_names = self.get_feature_names()

        signature = _hash_pipeline(spec.transforms)

        return FeatureManifest(
            feature_set_id="",  # Registry will generate if empty
            name=name,
            version=version,
            role=role,
            data_requirements=data_requirements or self.config.resolved_data_requirements(),
            feature_names=feature_names,
            feature_dtypes=["float32"] * len(feature_names),
            schema_hash="",  # Will be computed by registry
            pipeline_signature=signature,
            pipeline_version=pipeline_version,
            capability_flags=cast(dict[str, bool], capability_flags or {}),
            constraints=cast(dict[str, Any], constraints or {}),
            parity_tolerance=parity_tolerance,
            parity_digest=cast(dict[str, Any], parity_digest or {}),
            perf_digest=cast(dict[str, Any], perf_digest or {}),
            parent_feature_set_id=parent_feature_set_id,
            metadata=metadata or {},
        )

    def validate_feature_quality(self, features_df: DataFrameLike) -> dict[str, dict[str, float]]:
        """
        Validate feature quality metrics.

        Parameters
        ----------
        features_df : DataFrameLike
            DataFrame with computed features

        Returns
        -------
        dict[str, dict[str, float]]
            Quality metrics per feature column

        """
        metrics = {}

        # Detect if Polars or Pandas
        is_polars = hasattr(features_df, "to_numpy") and hasattr(features_df, "schema")

        if is_polars:
            columns = features_df.columns
            total_rows = len(features_df)
            for col in columns:
                col_data = features_df[col]
                metrics[col] = self.metrics_collector_component._calculate_column_metrics(
                    col_data,
                    total_rows,
                )
        else:
            # Convert to Polars for metrics calculation (FeatureMetricsCollector expects Polars)
            # This might be expensive but it's for validation, not hot path
            try:
                pl_df = pl.DataFrame(features_df)
                columns = pl_df.columns
                total_rows = len(pl_df)
                for col in columns:
                    col_data = pl_df[col]
                    metrics[col] = self.metrics_collector_component._calculate_column_metrics(
                        col_data,
                        total_rows,
                    )
            except Exception as e:
                self._logger.warning(f"Failed to convert DataFrame for quality validation: {e}")
                # Return empty metrics or basic Pandas metrics if needed

        return metrics

    def get_feature_names(self) -> list[str]:
        """
        Get list of feature names in order.

        Delegates to config for guaranteed parity.

        Returns
        -------
        list[str]
            List of feature names that will be generated by feature computation

        Examples
        --------
        >>> config = FeatureConfig(return_periods=[1, 5, 10], rsi_period=14)
        >>> engineer = FeatureEngineer(config)
        >>> feature_names = engineer.get_feature_names()
        >>> print(f"Will compute {len(feature_names)} features: {feature_names[:3]}")
        Will compute 15 features: ['return_1', 'return_5', 'return_10']

        """
        return self.config.get_feature_names()

    def compute_features(self, bars: list[Bar]) -> dict[str, float]:
        """
        Compute features from bars (legacy compatibility method).

        Converts Bar objects to DataFrame, computes batch features, and returns
        the latest row as a dict.

        Parameters
        ----------
        bars : list[Bar]
            List of Bar objects with OHLCV data and nanosecond timestamps

        Returns
        -------
        dict[str, float]
            Dictionary mapping feature names to computed values for the latest bar

        Examples
        --------
        >>> from nautilus_trader.model.data import Bar
        >>> config = FeatureConfig(return_periods=[1, 5, 10])
        >>> engineer = FeatureEngineer(config)
        >>> # ... create bars with Bar objects ...
        >>> features = engineer.compute_features(bars)
        >>> assert isinstance(features, dict)
        >>> assert set(features.keys()) == set(engineer.get_feature_names())

        Notes
        -----
        This method is provided for backward compatibility with legacy tests.
        For production use, prefer `calculate_features_batch` or
        `calculate_features_online` which provide more control over scaling
        and indicator state management.

        """
        if not bars:
            return dict.fromkeys(self.get_feature_names(), 0.0)

        # Convert bars to Polars DataFrame
        data = {
            "timestamp": [b.ts_event for b in bars],
            "open": [float(b.open) for b in bars],
            "high": [float(b.high) for b in bars],
            "low": [float(b.low) for b in bars],
            "close": [float(b.close) for b in bars],
            "volume": [float(b.volume) for b in bars],
        }
        bars_df = pl.DataFrame(data)

        # Calculate features (returns Polars DataFrame because input is Polars)
        features_df, _ = self.calculate_features_batch(bars_df, fit_scaler=False)

        # Get last row
        if len(features_df) > 0:
            if hasattr(features_df, "tail"):  # Pandas (fallback guard)
                if hasattr(features_df, "iloc"):
                    last_row = features_df.iloc[-1].to_dict()
                else:
                    # Polars
                    last_row = features_df.tail(1).to_dicts()[0]
            else:
                # Polars
                last_row = features_df.tail(1).to_dicts()[0]
            return {k: float(v) for k, v in last_row.items()}

        return dict.fromkeys(self.get_feature_names(), 0.0)

    @overload
    def calculate_features(
        self: Self,
        data: DataFrameLike,
        *,
        mode: Literal["batch"] = "batch",
        indicator_manager: None = ...,
        fit_scaler: bool = ...,
        scaler_fit_ratio: float = ...,
        scaler: None = ...,
    ) -> tuple[DataFrameLike, StandardScalerT | None]: ...

    @overload
    def calculate_features(
        self: Self,
        data: dict[str, float],
        *,
        mode: Literal["online"],
        indicator_manager: IndicatorManager,
        fit_scaler: bool = ...,
        scaler_fit_ratio: float = ...,
        scaler: StandardScalerT | None = ...,
    ) -> npt.NDArray[np.float32]: ...

    def calculate_features(
        self: Self,
        data: DataFrameLike | dict[str, float],
        mode: Literal["batch", "online"] = "batch",
        indicator_manager: IndicatorManager | None = None,
        fit_scaler: bool = False,
        scaler_fit_ratio: float = 0.7,
        scaler: StandardScalerT | None = None,
    ) -> tuple[DataFrameLike, StandardScalerT | None] | npt.NDArray[np.float32]:
        """
        Unified feature calculation method for both batch and online modes.

        This method ensures perfect feature parity between training (batch) and
        inference (online) by routing to the same underlying computation logic.

        Parameters
        ----------
        data : DataFrameLike | dict[str, float]
            - For batch mode: pl.DataFrame or pd.DataFrame with OHLCV data
            - For online mode: dict with current bar data (open, high, low, close, volume)
        mode : str, default "batch"
            Computation mode - either "batch" or "online"
        indicator_manager : IndicatorManager | None, optional
            Required for online mode. Manages indicator state.
        fit_scaler : bool, default False
            Whether to fit a StandardScaler (batch mode only)
        scaler_fit_ratio : float, default 0.7
            Ratio of data for fitting scaler (batch mode only)
        scaler : StandardScaler | None, optional
            Pre-fitted scaler for scaling features (online mode only)

        Returns
        -------
        tuple[DataFrameLike, StandardScaler | None] | npt.NDArray[np.float32]
            - For batch mode: tuple[DataFrame, StandardScaler or None]
            - For online mode: npt.NDArray[np.float32]

        Raises
        ------
        ValueError
            If mode is not "batch" or "online"
            If online mode is specified without indicator_manager

        Examples
        --------
        Batch mode (training):
        >>> config = FeatureConfig()
        >>> engineer = FeatureEngineer(config)
        >>> features_df, scaler = engineer.calculate_features(
        ...     df, mode="batch", fit_scaler=True
        ... )

        Online mode (inference):
        >>> features = engineer.calculate_features(
        ...     current_bar, mode="online",
        ...     indicator_manager=indicator_mgr,
        ...     scaler=scaler
        ... )

        """
        if mode == "batch":
            return self.calculator.calculate_features(
                data=cast(Any, data),
                mode="batch",
                fit_scaler=fit_scaler,
                scaler_fit_ratio=scaler_fit_ratio,
            )
        elif mode == "online":
            if indicator_manager is None:
                msg = "indicator_manager is required for online mode"
                raise ValueError(msg)
            return self.calculator.calculate_features(
                data=cast(dict[str, float], data),
                mode="online",
                indicator_manager=indicator_manager,
                scaler=scaler,
            )
        else:
            msg = f"Invalid mode: {mode}. Must be 'batch' or 'online'"
            raise ValueError(msg)

    def calculate_features_batch(
        self: Self,
        df: DataFrameLike,
        fit_scaler: bool = False,
        scaler_fit_ratio: float = 0.7,
    ) -> tuple[DataFrameLike, StandardScalerT | None]:
        """
        Calculate features for batch data using Nautilus indicators.

        This method processes historical data sequentially to ensure perfect
        consistency with online calculation. It follows the cold path pattern
        optimized for training data preparation.

        Parameters
        ----------
        df : DataFrameLike
            Input DataFrame with OHLCV data (Polars or Pandas)
        fit_scaler : bool, default False
            Whether to fit a StandardScaler on the data
        scaler_fit_ratio : float, default 0.7
            Ratio of data to use for fitting scaler to prevent look-ahead bias

        Returns
        -------
        tuple[DataFrameLike, StandardScaler | None]
            Tuple of (features DataFrame, fitted scaler or None)

        Examples
        --------
        >>> import polars as pl
        >>> config = FeatureConfig(return_periods=[1, 5, 10])
        >>> engineer = FeatureEngineer(config)
        >>> df = pl.read_parquet("market_data.parquet")
        >>> features_df, scaler = engineer.calculate_features_batch(df, fit_scaler=True)
        >>> print(f"Computed {len(features_df.columns)} features for {len(features_df)} rows")

        """
        return self.calculator.calculate_features(
            data=df,
            mode="batch",
            fit_scaler=fit_scaler,
            scaler_fit_ratio=scaler_fit_ratio,
        )

    def calculate_features_online(
        self,
        current_bar: dict[str, float],
        indicator_manager: IndicatorManager,
        scaler: StandardScalerT | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Calculate features for a single bar in online mode (HOT PATH).

        This method is performance-critical with P99 latency < 5ms requirement.
        It uses pre-allocated buffers and avoids all dynamic allocations.

        Parameters
        ----------
        current_bar : dict[str, float]
            Current bar data with keys: open, high, low, close, volume
        indicator_manager : IndicatorManager
            Pre-warmed indicator manager with historical state
        scaler : StandardScaler | None, optional
            Pre-fitted scaler for feature normalization

        Returns
        -------
        npt.NDArray[np.float32]
            Feature vector of shape (n_features,)

        Examples
        --------
        >>> config = FeatureConfig(return_periods=[1, 5, 10])
        >>> engineer = FeatureEngineer(config)
        >>> indicator_mgr = IndicatorManager(config)
        >>> # ... warm up indicators from historical bars ...
        >>> current_bar = {
        ...     "open": 100.0,
        ...     "high": 101.0,
        ...     "low": 99.0,
        ...     "close": 100.5,
        ...     "volume": 10000.0,
        ... }
        >>> features = engineer.calculate_features_online(
        ...     current_bar, indicator_mgr, scaler=None
        ... )
        >>> assert features.shape == (engineer.config.get_feature_names().__len__(),)

        """
        return self.calculator.calculate_features(
            data=current_bar,
            mode="online",
            indicator_manager=indicator_manager,
            scaler=scaler,
        )

    def __repr__(self) -> str:
        """
        Return string representation of the facade.
        """
        return (
            f"FeatureEngineer(facade, "
            f"config={self.config}, "
            f"n_features={len(self.config.get_feature_names())}, "
            f"components=[DataExtractor, FeatureCalculator, FeatureStoreAccessor, "
            f"FeatureRegistryAccessor, FeatureMetricsCollector])"
        )


__all__ = [
    "FeatureConfig",
    "FeatureEngineer",
    "IndicatorManager",
]
