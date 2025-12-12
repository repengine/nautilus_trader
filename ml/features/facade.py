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
from nautilus_trader.model.data import Bar

from ml.config.feature_flags import use_legacy_feature_engineer
from ml.features.common.data_extractor import DataExtractor
from ml.features.common.feature_calculator import FeatureCalculator
from ml.features.common.feature_metrics_collector import FeatureMetricsCollector
from ml.features.common.feature_registry_accessor import FeatureRegistryAccessor
from ml.features.common.feature_store_accessor import FeatureStoreAccessor
from ml.features.engineering import FeatureConfig as LegacyFeatureConfig
from ml.features.engineering import FeatureConfigLike
from ml.features.engineering import FeatureEngineer as LegacyFeatureEngineer
from ml.features.engineering import IndicatorManager
from ml.features.engineering import IndicatorManagerLike
from ml.features.engineering import build_pipeline_spec_from_feature_config
from ml.features.pipeline import PipelineRunner
from ml.features.pipeline import PipelineSpec
from ml.ml_types import DataFrameLike
from ml.ml_types import StandardScaler as StandardScalerT
from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import compute_schema_hash


if TYPE_CHECKING:
    from typing import Protocol

    import polars as pl

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


# Re-export FeatureConfig from legacy for compatibility
FeatureConfig = LegacyFeatureConfig

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
        config: FeatureConfigLike,
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
        if isinstance(config, LegacyFeatureConfig):
            normalized_config = config
        else:
            raw: dict[str, object]
            try:
                import msgspec

                raw = cast(dict[str, object], msgspec.to_builtins(config))
            except Exception:
                raw = cast(dict[str, object], getattr(config, "__dict__", {}) or {})

            allowed_fields = set(LegacyFeatureConfig.__annotations__.keys())
            filtered: dict[str, object] = {k: v for k, v in raw.items() if k in allowed_fields}

            # Coerce enum-like fields from persisted strings
            data_req = filtered.get("data_requirements")
            if isinstance(data_req, str):
                try:
                    filtered["data_requirements"] = DataRequirements(data_req)
                except Exception:
                    ...

            normalized_config = (
                LegacyFeatureConfig(**cast(dict[str, Any], filtered))
                if filtered
                else LegacyFeatureConfig()
            )

        self.config = normalized_config
        self._logger = logger if logger is not None else globals()["logger"]
        self._use_legacy = use_legacy_feature_engineer()
        self._legacy_impl: LegacyFeatureEngineer | None = None

        # Initialize all 5 extracted components
        # Component 1: DataExtractor (stateless data extraction)
        self.data_extractor = DataExtractor()

        # Component 2: FeatureCalculator (HOT PATH - core computation)
        self.calculator = FeatureCalculator(config=normalized_config, logger=self._logger)

        # Component 3: FeatureStoreAccessor (persistence operations)
        self.store_accessor = FeatureStoreAccessor(
            feature_store=feature_store,
        )

        # Component 4: FeatureRegistryAccessor (registry operations)
        self.registry_accessor = FeatureRegistryAccessor(stores=stores)

        # Component 5: FeatureMetricsCollector (metrics calculation)
        self.metrics_collector_component = FeatureMetricsCollector()

        # Internal indicator manager for convenience kwargs path (no legacy dependency)
        self._indicator_manager = IndicatorManager(normalized_config)

        # Optional legacy implementation for gated fallback
        if self._use_legacy:
            self._legacy_impl = LegacyFeatureEngineer(
                config=normalized_config,
                metrics_collector=metrics_collector,
                feature_store=feature_store,
                stores=stores,
            )

        # Store references for backward compatibility
        self._stores = stores
        self._feature_store = feature_store
        self._metrics = metrics_collector
        self.scaler: StandardScalerT | None = None

    def reset(self) -> None:
        """
        Reset all stateful components to initial state.

        This clears any cached state in the indicator manager and resets the scaler.
        Uses the facade's internal indicator manager (no legacy dependency).

        Examples
        --------
        >>> engineer = FeatureEngineer(config)
        >>> # ... compute features ...
        >>> engineer.reset()  # Clear all state
        """
        # Reset the facade's indicator manager directly - no legacy dependency
        self._indicator_manager.reset()
        self.scaler = None

    # Property accessors for 4-store + 4-registry pattern (via accessor components)

    @property
    def feature_store(self) -> object | None:
        """Access feature store via store accessor component."""
        return self.store_accessor._feature_store

    @property
    def model_store(self) -> object | None:
        """Access model store via registry accessor component."""
        return self.registry_accessor.model_registry  # Note: legacy uses registry for store access

    @property
    def strategy_store(self) -> object | None:
        """Access strategy store via registry accessor component."""
        return self.registry_accessor.strategy_registry

    @property
    def data_store(self) -> object | None:
        """Access data store via registry accessor component."""
        return self.registry_accessor.data_registry

    @property
    def feature_registry(self) -> object | None:
        """Access feature registry via registry accessor component."""
        return self.registry_accessor.feature_registry

    @property
    def model_registry(self) -> object | None:
        """Access model registry via registry accessor component."""
        return self.registry_accessor.model_registry

    @property
    def strategy_registry(self) -> object | None:
        """Access strategy registry via registry accessor component."""
        return self.registry_accessor.strategy_registry

    @property
    def data_registry(self) -> object | None:
        """Access data registry via registry accessor component."""
        return self.registry_accessor.data_registry

    @property
    def feature_buffer(self) -> npt.NDArray[np.float32]:
        """
        Access pre-allocated feature buffer (HOT PATH).

        Returns a view of the internal feature buffer used by calculate_features_online.
        Used for zero-allocation feature computation.

        Returns
        -------
        npt.NDArray[np.float32]
            Pre-allocated buffer for feature values
        """
        if self._use_legacy and self._legacy_impl is not None:
            return self._legacy_impl.feature_buffer
        return self.calculator.feature_buffer

    # Public API methods - delegate to legacy implementation for now
    # Future: Migrate to use extracted components directly

    def build_pipeline_spec_from_config(self) -> PipelineSpec:
        """
        Build PipelineSpec from feature configuration.

        Uses the standalone helper function to build a PipelineSpec that mirrors
        the core feature blocks in engineering and preserves the ordering used
        by get_feature_names().

        Returns
        -------
        PipelineSpec
            Pipeline specification for feature computation

        """
        # Use standalone helper function - no legacy dependency
        return build_pipeline_spec_from_feature_config(self.config)

    def generate_feature_manifest(
        self,
        name: str,
        version: str,
        role: FeatureRole,
        data_requirements: DataRequirements | None = None,
        pipeline_version: str = "1.0.0",
        capability_flags: dict[str, bool] | None = None,
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
        capability_flags : dict[str, bool] | None, optional
            Capability flags for the feature set (e.g., microstructure, trade_flow)
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
        # Build pipeline spec and compute feature names and signature locally
        spec = self.build_pipeline_spec_from_config()
        requirements_value = (
            data_requirements
            if data_requirements is not None
            else getattr(self.config, "data_requirements", DataRequirements.L1_ONLY)
        )
        requirements = cast(DataRequirements, requirements_value)
        runner = PipelineRunner(spec, allowable=requirements)
        names = runner.compute_feature_names()
        dtypes = ["float32"] * len(names)
        signature = runner.compute_signature()
        schema_hash = compute_schema_hash(names, dtypes, signature)
        now = float(np.float64(np.datetime64("now").astype("datetime64[s]").astype(int)))

        return FeatureManifest(
            feature_set_id="",
            name=name,
            version=version,
            role=role,
            data_requirements=requirements,
            feature_names=names,
            feature_dtypes=dtypes,
            schema_hash=schema_hash,
            pipeline_signature=signature,
            pipeline_version=pipeline_version,
            capability_flags=capability_flags or {},
            constraints=constraints or {},
            parity_tolerance=parity_tolerance,
            parity_digest=parity_digest or {},
            perf_digest=perf_digest or {},
            parent_feature_set_id=parent_feature_set_id,
            metadata=metadata or {},
            created_at=now,
            last_modified=now,
        )

    def validate_feature_quality(self, features_df: DataFrameLike) -> dict[str, dict[str, float]]:
        """
        Validate feature quality metrics.

        Uses FeatureMetricsCollector component for column metrics calculation.
        Only runs when config.validate_quality is True.

        Parameters
        ----------
        features_df : DataFrameLike
            DataFrame with computed features

        Returns
        -------
        dict[str, dict[str, float]]
            Quality metrics per feature column

        """
        # Check config flag
        if not getattr(self.config, "validate_quality", False):
            return {}

        # Convert to Polars if needed
        converted = self._convert_to_polars(features_df)
        if converted is None:
            return {}

        # converted is already typed as pl.DataFrame from _convert_to_polars
        pdf_or_pl = converted
        if len(pdf_or_pl) == 0:
            return {}

        quality_metrics: dict[str, dict[str, float]] = {}
        total_rows = len(pdf_or_pl)

        for col in pdf_or_pl.columns:
            if col in ("timestamp", "entity_id", "symbol"):
                continue
            try:
                # Delegate to metrics collector component
                metrics = self.metrics_collector_component._calculate_column_metrics(
                    pdf_or_pl[col], total_rows
                )
                quality_metrics[col] = metrics
            except Exception:
                # Skip non-numeric or problematic columns gracefully
                continue

        return quality_metrics

    def _convert_to_polars(self, features_df: DataFrameLike) -> pl.DataFrame | None:
        """Convert DataFrame to Polars if possible; return None on failure."""
        try:
            import polars as _pl

            # Already Polars?
            if hasattr(features_df, "select") and "polars" in str(type(features_df)):
                return cast("_pl.DataFrame", features_df)
            # Try pandas → polars
            if hasattr(features_df, "__class__") and "pandas" in str(type(features_df)):
                return _pl.from_pandas(features_df)  # type: ignore[arg-type]
        except Exception:
            return None
        return None

    def get_feature_names(self) -> list[str]:
        """
        Get list of feature names in order.

        Delegates to config's get_feature_names() which uses the declarative
        pipeline to compute feature names, ensuring parity with computation logic.

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
        # Delegate to config's get_feature_names() - no legacy dependency
        return self.config.get_feature_names()

    def compute_features(self, bars: list[Bar]) -> dict[str, float]:
        """
        Compute features from bars (legacy compatibility method).

        Delegates to calculator component for feature computation. This is a
        compatibility shim that converts Bar objects to DataFrame, computes
        batch features, and returns the latest row as a dict.

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
        if self._legacy_impl is None:
            # Lazily instantiate legacy for compatibility shim parity/performance
            self._legacy_impl = LegacyFeatureEngineer(
                config=self.config,
                metrics_collector=self._metrics,
                feature_store=self._feature_store,
                stores=self._stores,
            )
        if self._use_legacy:
            return self._legacy_impl.compute_features(bars)
        # Prefer legacy implementation for this compatibility method to preserve parity
        return self._legacy_impl.compute_features(bars)

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
        indicator_manager: IndicatorManagerLike,
        fit_scaler: bool = ...,
        scaler_fit_ratio: float = ...,
        scaler: StandardScalerT | None = ...,
    ) -> npt.NDArray[np.float32]: ...

    def calculate_features(
        self: Self,
        data: DataFrameLike | dict[str, float],
        mode: str = "batch",
        indicator_manager: IndicatorManagerLike | None = None,
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
        indicator_manager : IndicatorManagerLike | None, optional
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
        if mode == "online":
            if indicator_manager is None:
                msg = "indicator_manager is required for calculate_features(mode='online')"
                raise ValueError(msg)
            online_bar = cast(dict[str, float], data)
            online_manager = cast(IndicatorManager, indicator_manager)
            if self._use_legacy:
                if self._legacy_impl is None:
                    raise RuntimeError("Legacy FeatureEngineer not initialized")
                return self._legacy_impl.calculate_features(
                    online_bar,
                    mode="online",
                    indicator_manager=online_manager,
                    fit_scaler=fit_scaler,
                    scaler_fit_ratio=scaler_fit_ratio,
                    scaler=scaler,
                )
            return self.calculator.calculate_features(
                data=online_bar,
                mode="online",
                indicator_manager=online_manager,
                fit_scaler=fit_scaler,
                scaler_fit_ratio=scaler_fit_ratio,
                scaler=scaler,
            )

        batch_data = cast(DataFrameLike, data)
        if self._use_legacy:
            if self._legacy_impl is None:
                raise RuntimeError("Legacy FeatureEngineer not initialized")
            return self._legacy_impl.calculate_features(
                batch_data,
                mode="batch",
                indicator_manager=None,
                fit_scaler=fit_scaler,
                scaler_fit_ratio=scaler_fit_ratio,
                scaler=scaler,
            )
        return self.calculator.calculate_features(
            data=batch_data,
            mode="batch",
            indicator_manager=None,
            fit_scaler=fit_scaler,
            scaler_fit_ratio=scaler_fit_ratio,
            scaler=scaler,
        )

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
        if self._use_legacy:
            if self._legacy_impl is None:
                raise RuntimeError("Legacy FeatureEngineer not initialized")
            return self._legacy_impl.calculate_features_batch(
                df,
                fit_scaler=fit_scaler,
                scaler_fit_ratio=scaler_fit_ratio,
            )
        # Delegate to calculator component (Phase 1.1 wiring)
        # Note: Component uses private _calculate_features_batch() method
        return self.calculator._calculate_features_batch(
            data=df,
            fit_scaler=fit_scaler,
            scaler_fit_ratio=scaler_fit_ratio,
        )

    @overload
    def calculate_features_online(
        self,
        *,
        close_price: float,
        high_price: float,
        low_price: float,
        volume: float,
        scaler: StandardScalerT | None = None,
    ) -> npt.NDArray[np.float32]: ...

    @overload
    def calculate_features_online(
        self,
        current_bar: dict[str, float],
        indicator_manager: IndicatorManagerLike,
        scaler: StandardScalerT | None = None,
    ) -> npt.NDArray[np.float32]: ...

    def calculate_features_online(
        self,
        current_bar: dict[str, float] | None = None,
        indicator_manager: IndicatorManagerLike | None = None,
        scaler: StandardScalerT | None = None,
        *,
        close_price: float | None = None,
        high_price: float | None = None,
        low_price: float | None = None,
        volume: float | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Calculate features for a single bar in online mode (HOT PATH).

        This method is performance-critical with P99 latency < 5ms requirement.
        It uses pre-allocated buffers and avoids all dynamic allocations.

        Supports two calling conventions:
        1. With current_bar dict and indicator_manager (standard)
        2. With individual price kwargs (convenience - uses internal indicator manager)

        Parameters
        ----------
        current_bar : dict[str, float] | None
            Current bar data with keys: open, high, low, close, volume
        indicator_manager : IndicatorManagerLike | None
            Pre-warmed indicator manager with historical state
        scaler : StandardScaler | None, optional
            Pre-fitted scaler for feature normalization
        close_price : float | None
            Current close price if `current_bar` is not provided
        high_price : float | None
            Current high price if `current_bar` is not provided
        low_price : float | None
            Current low price if `current_bar` is not provided
        volume : float | None
            Current trade volume if `current_bar` is not provided

        Returns
        -------
        npt.NDArray[np.float32]
            Feature vector of shape (n_features,)

        Examples
        --------
        Standard usage with dict and indicator manager:
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

        Convenience usage with individual kwargs:
        >>> features = engineer.calculate_features_online(
        ...     close_price=100.5,
        ...     high_price=101.0,
        ...     low_price=99.0,
        ...     volume=10000.0,
        ... )

        """
        resolved_indicator_manager = indicator_manager
        resolved_current_bar = current_bar

        # Handle convenience kwargs (Phase 1.1 - V2 correction)
        if resolved_current_bar is None:
            if close_price is None or high_price is None or low_price is None or volume is None:
                msg = (
                    "calculate_features_online requires either current_bar and indicator_manager, "
                    "or keyword args: close_price, high_price, low_price, volume"
                )
                raise ValueError(msg)
            # Use facade's indicator manager by default - no legacy dependency
            resolved_indicator_manager = self._indicator_manager if indicator_manager is None else indicator_manager
            # Update indicators from raw values
            resolved_indicator_manager.update_from_values(
                close=float(close_price),
                high=float(high_price),
                low=float(low_price),
                volume=float(volume),
            )
            resolved_current_bar = {
                "close": float(close_price),
                "high": float(high_price),
                "low": float(low_price),
                "volume": float(volume),
            }

        # Validate required arguments
        if resolved_indicator_manager is None:
            msg = "indicator_manager is required for calculate_features_online"
            raise ValueError(msg)
        if resolved_current_bar is None:
            msg = "current_bar is required for calculate_features_online"
            raise ValueError(msg)

        if self._use_legacy:
            if self._legacy_impl is None:
                raise RuntimeError("Legacy FeatureEngineer not initialized")
            legacy_manager = cast(IndicatorManager, resolved_indicator_manager)
            return self._legacy_impl.calculate_features_online(
                resolved_current_bar,
                legacy_manager,
                scaler=scaler,
            )

        # Delegate to calculator component (Phase 1.1 wiring)
        # Note: Component uses private _calculate_features_online() method
        typed_indicator_manager = cast(IndicatorManager, resolved_indicator_manager)
        return self.calculator._calculate_features_online(
            current_bar=resolved_current_bar,
            indicator_manager=typed_indicator_manager,
            scaler=scaler,
        )

    def __repr__(self) -> str:
        """Return string representation of the facade."""
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
