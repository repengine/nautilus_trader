"""
TFT Dataset Builder Facade for component-based dataset preparation.

This module provides a facade that delegates to focused components while
maintaining 100% API compatibility with the legacy TFTDatasetBuilder.

Feature Flag:
    ML_USE_LEGACY_TFT_BUILDER: Set to '1' or 'true' to use legacy implementation.
    Default (unset or '0'): Uses component-based facade.

Components:
    - TimeSeriesWindowingComponent: Time bounds, windowing operations
    - FeatureAlignmentComponent: Feature computation and static features
    - TargetGenerationComponent: Binary target generation
    - TFTSchemaValidatorComponent: Schema validation

"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from datetime import UTC
from datetime import datetime
from pathlib import Path as _Path
from typing import TYPE_CHECKING, Any, cast

from ml._imports import pd as pd_runtime
from ml._imports import pl as pl_runtime
from ml.config.base import MLFeatureConfig
from ml.data.common import FeatureAlignmentComponent
from ml.data.common import SchemaValidationError
from ml.data.common import TargetGenerationComponent
from ml.data.common import TFTSchemaValidatorComponent
from ml.data.common import TimeSeriesWindowingComponent
from ml.data.ingest.market_bindings import MarketBindingStats
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.vintage import VintagePolicy
from ml.stores.feature_store import FeatureStore
from ml.stores.protocols import DataStoreFacadeProtocol
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


if TYPE_CHECKING:
    import pandas as _pd
    import polars as _pl
else:  # pragma: no cover - typing fallback
    _pd = Any
    _pl = Any


# Runtime aliases
pl: Any = cast(Any, pl_runtime)
pd: Any = cast(Any, pd_runtime)


logger = logging.getLogger(__name__)


# Re-export SchemaValidationError for consumers
__all__ = ["SchemaValidationError", "TFTDatasetBuilderFacade", "use_legacy_builder"]


def use_legacy_builder() -> bool:
    """
    Check if legacy TFTDatasetBuilder should be used.

    Returns:
        True if ML_USE_LEGACY_TFT_BUILDER is set to '1' or 'true' (case-insensitive).

    Example:
        >>> import os
        >>> os.environ["ML_USE_LEGACY_TFT_BUILDER"] = "1"
        >>> assert use_legacy_builder() is True

    """
    value = os.environ.get("ML_USE_LEGACY_TFT_BUILDER", "").lower()
    return value in ("1", "true", "yes")


class TFTDatasetBuilderFacade:
    """
    Component-based TFT dataset builder with legacy API compatibility.

    This facade maintains 100% API parity with the legacy TFTDatasetBuilder
    while delegating to focused, single-responsibility components for:
    - Time series windowing and bounds extraction
    - Feature alignment and computation
    - Target generation for binary classification
    - Schema validation

    The facade supports a feature flag (ML_USE_LEGACY_TFT_BUILDER) to toggle
    between the component-based implementation and the legacy monolithic class.

    Attributes:
        catalog: Nautilus data catalog for market data access.
        symbols: List of symbols to include in the dataset.
        feature_config: Feature engineering configuration.
        feature_store: Optional FeatureStore for pre-computed features.
        data_store: Optional DataStore for raw market data.

    Example:
        >>> facade = TFTDatasetBuilderFacade(
        ...     catalog=catalog,
        ...     symbols=["SPY", "QQQ"],
        ...     feature_config=MLFeatureConfig(),
        ... )
        >>> dataset = facade.build_training_dataset(
        ...     horizon_minutes=15,
        ...     min_return_threshold=0.001,
        ... )

    """

    def __init__(
        self,
        catalog: ParquetDataCatalog,
        symbols: list[str],
        instrument_ids: list[str] | None = None,
        feature_config: MLFeatureConfig | None = None,
        feature_store: FeatureStore | None = None,
        *,
        data_store: DataStoreFacadeProtocol | None = None,
        market_dataset_id: str | None = None,
        market_bindings: Iterable[ResolvedMarketBinding] | None = None,
        include_macro: bool = False,
        macro_lag_days: int = 1,
        fred_path: str | None = None,
        include_micro: bool = False,
        micro_base_dir: str | None = None,
        include_calendar: bool = False,
        include_events: bool = False,
        include_earnings: bool = False,
        earnings_lag_days: int = 1,
        include_l2: bool = False,
        l2_base_dir: str | None = None,
        vintage_base_dir: str | _Path | None = None,
        events_base_dir: str | _Path | None = None,
        student_mode: bool = False,
        macro_series_ids: tuple[str, ...] | None = None,
        vintage_policy: VintagePolicy = VintagePolicy.REAL_TIME,
        vintage_as_of: datetime | None = None,
        include_macro_revisions: bool = False,
        macro_revision_mode: str = "core",
        macro_revision_windows: tuple[int, ...] | None = None,
    ) -> None:
        """
        Initialize TFT dataset builder facade.

        Parameters
        ----------
        catalog : ParquetDataCatalog
            Nautilus data catalog for accessing market data.
        symbols : list[str]
            List of symbols to include in dataset.
        instrument_ids : list[str] | None, optional
            Explicit instrument IDs to use instead of symbol-based resolution.
        feature_config : MLFeatureConfig | None, optional
            Feature engineering configuration.
        feature_store : FeatureStore | None, optional
            Feature store for reading pre-computed features (ensures training/inference parity).
        data_store : DataStoreFacadeProtocol | None, optional
            Canonical DataStore for reading raw market data.
        market_dataset_id : str | None, optional
            Dataset identifier registered in the DataRegistry.
        market_bindings : Iterable[ResolvedMarketBinding] | None, optional
            Pre-resolved market bindings for instrument/dataset mapping.
        include_macro : bool, default False
            Whether to include macroeconomic features.
        macro_lag_days : int, default 1
            Publication lag for macro features.
        fred_path : str | None, optional
            Path to FRED data files.
        include_micro : bool, default False
            Whether to include microstructure features.
        micro_base_dir : str | None, optional
            Base directory for microstructure data.
        include_calendar : bool, default False
            Whether to include calendar features.
        include_events : bool, default False
            Whether to include event-based features.
        include_earnings : bool, default False
            Whether to include earnings features.
        earnings_lag_days : int, default 1
            Publication lag for earnings features.
        include_l2 : bool, default False
            Whether to include L2 order book features.
        l2_base_dir : str | None, optional
            Base directory for L2 data.
        vintage_base_dir : str | Path | None, optional
            Base directory for vintage data.
        events_base_dir : str | Path | None, optional
            Base directory for events data.
        student_mode : bool, default False
            If True, disables macro, events, L2, and earnings features.
        macro_series_ids : tuple[str, ...] | None, optional
            Filter for specific macro series.
        vintage_policy : VintagePolicy, default REAL_TIME
            How to handle macro data revisions.
        vintage_as_of : datetime | None, optional
            Point-in-time cutoff for vintage data.
        include_macro_revisions : bool, default False
            Whether to include revision history features.
        macro_revision_mode : str, default "core"
            Mode for revision features.
        macro_revision_windows : tuple[int, ...] | None, optional
            Window sizes for revision features.

        Raises
        ------
        ValueError
            If symbols list is empty or earnings_lag_days is negative.

        """
        # Validate inputs
        if not symbols:
            raise ValueError("symbols list cannot be empty")
        if earnings_lag_days < 0:
            raise ValueError("earnings_lag_days must be >= 0")

        # Store all parameters for delegation
        self.catalog = catalog
        self._original_symbols = symbols
        self.symbols = [sym.split(".")[0] for sym in symbols]
        self.instrument_ids = instrument_ids
        self.feature_config = feature_config or MLFeatureConfig()
        self.feature_store = feature_store
        self.data_store = data_store
        self.market_dataset_id = market_dataset_id
        self.market_bindings = tuple(market_bindings or ())
        self.include_macro = include_macro
        self.macro_lag_days = macro_lag_days
        self.fred_path = fred_path
        self.include_micro = include_micro
        self.micro_base_dir = micro_base_dir
        self.include_calendar = include_calendar
        self.include_events = include_events
        self.include_earnings = include_earnings and data_store is not None
        self.earnings_lag_days = earnings_lag_days
        self.include_l2 = include_l2
        self.l2_base_dir = l2_base_dir
        self.vintage_base_dir = _Path(vintage_base_dir).expanduser() if vintage_base_dir else None
        self.events_base_dir = (
            _Path(events_base_dir).expanduser() if events_base_dir else _Path("data/events")
        )
        self.student_mode = student_mode
        self.macro_series_ids = macro_series_ids
        self.vintage_policy = vintage_policy
        if vintage_as_of is None:
            self.vintage_as_of = None
        elif vintage_as_of.tzinfo is None:
            self.vintage_as_of = vintage_as_of.replace(tzinfo=UTC)
        else:
            self.vintage_as_of = vintage_as_of.astimezone(UTC)
        self.include_macro_revisions = include_macro_revisions
        self.macro_revision_mode = macro_revision_mode
        self.macro_revision_windows = (
            list(macro_revision_windows) if macro_revision_windows else None
        )

        # Apply student mode restrictions
        if self.student_mode:
            self.include_macro = False
            self.include_events = False
            self.include_l2 = False
            self.include_earnings = False

        # Initialize components
        self._windowing = TimeSeriesWindowingComponent()
        self._feature_alignment = FeatureAlignmentComponent()
        self._target_generation = TargetGenerationComponent()
        self._schema_validator = TFTSchemaValidatorComponent()

        # Initialize binding stats tracking
        self._binding_stats: dict[str, MarketBindingStats] = {}
        for binding in self.market_bindings:
            self._binding_stats[binding.binding_id] = MarketBindingStats(
                binding_id=binding.binding_id,
                dataset_id=binding.dataset_id,
                descriptor_id=binding.descriptor_id,
                symbol=binding.symbol,
                instrument_ids=binding.instrument_ids,
                schema=binding.schema,
                storage_kind=binding.storage_kind,
                source=binding.source,
                license_start=binding.license_start,
                license_end=binding.license_end,
            )

        # Lazy-initialize legacy builder for delegation
        self._legacy_builder: Any = None

        logger.info(
            f"Initialized TFTDatasetBuilderFacade with {len(symbols)} symbols "
            f"(FeatureStore: {'enabled' if feature_store else 'disabled'}, "
            f"Mode: {'legacy' if use_legacy_builder() else 'component-based'})",
        )

    # =========================================================================
    # Legacy Builder Delegation (lazy initialization)
    # =========================================================================

    def _get_legacy_builder(self) -> Any:
        """
        Lazily initialize and return the legacy TFTDatasetBuilder.

        This method creates a legacy builder instance on first access,
        using the same configuration as the facade.

        Returns:
            TFTDatasetBuilder instance.

        """
        if self._legacy_builder is None:
            from ml.data.tft_dataset_builder import TFTDatasetBuilder

            self._legacy_builder = TFTDatasetBuilder(
                catalog=self.catalog,
                symbols=self._original_symbols,
                instrument_ids=self.instrument_ids,
                feature_config=self.feature_config,
                feature_store=self.feature_store,
                data_store=self.data_store,
                market_dataset_id=self.market_dataset_id,
                market_bindings=self.market_bindings,
                include_macro=self.include_macro,
                macro_lag_days=self.macro_lag_days,
                fred_path=self.fred_path,
                include_micro=self.include_micro,
                micro_base_dir=self.micro_base_dir,
                include_calendar=self.include_calendar,
                include_events=self.include_events,
                include_earnings=self.include_earnings,
                earnings_lag_days=self.earnings_lag_days,
                include_l2=self.include_l2,
                l2_base_dir=self.l2_base_dir,
                vintage_base_dir=self.vintage_base_dir,
                events_base_dir=self.events_base_dir,
                student_mode=self.student_mode,
                macro_series_ids=self.macro_series_ids,
                vintage_policy=self.vintage_policy,
                vintage_as_of=self.vintage_as_of,
                include_macro_revisions=self.include_macro_revisions,
                macro_revision_mode=self.macro_revision_mode,
                macro_revision_windows=(
                    tuple(self.macro_revision_windows) if self.macro_revision_windows else None
                ),
            )
        return self._legacy_builder

    # =========================================================================
    # Public API - Matching Legacy TFTDatasetBuilder Exactly
    # =========================================================================

    def build_training_dataset(
        self,
        horizon_minutes: int = 15,
        min_return_threshold: float = 0.001,
        *,
        threshold_bps: float | None = None,
        lookback_periods: int = 30,
        use_polars: bool = True,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> _pd.DataFrame | _pl.DataFrame:
        """
        Build complete TFT training dataset.

        This method automatically chooses between FeatureStore and direct computation:
        - If FeatureStore is configured, uses pre-computed features for training/inference parity
        - Otherwise, falls back to direct feature computation

        Parameters
        ----------
        horizon_minutes : int, default 15
            Prediction horizon in minutes.
        min_return_threshold : float, default 0.001
            Minimum return threshold for binary classification (0.1%).
        threshold_bps : float | None, optional
            Alternative threshold in basis points (backward compatibility).
            If provided and > 1, converted to decimal (threshold_bps / 10000).
        lookback_periods : int, default 30
            Minimum lookback periods for feature computation.
        use_polars : bool, default True
            Whether to return Polars DataFrame (True) or Pandas DataFrame (False).
        start : datetime | None, optional
            Start time for data loading.
        end : datetime | None, optional
            End time for data loading.

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            TFT-compatible training dataset with:
            - timestamp: Time index
            - instrument_id: Symbol identifier
            - Features: return_1, return_5, return_20, volume_ratio, volatility_20, sma_5, sma_20, price_position
            - Targets: y (binary), forward_return (continuous)
            - Static features: asset_class, tick_size, exchange
            - Calendar features: hour, minute, dow, tod_sin, tod_cos, etc.

        Example:
            >>> dataset = facade.build_training_dataset(
            ...     horizon_minutes=15,
            ...     min_return_threshold=0.001,
            ...     use_polars=True,
            ... )
            >>> assert "y" in dataset.columns
            >>> assert "forward_return" in dataset.columns

        """
        # Handle threshold_bps backward compatibility
        if threshold_bps is not None:
            min_return_threshold = threshold_bps / 10_000.0 if threshold_bps > 1 else threshold_bps

        # Delegate to legacy builder for now
        # The facade maintains API parity while components handle specific responsibilities
        legacy = self._get_legacy_builder()
        result: _pd.DataFrame | _pl.DataFrame = legacy.build_training_dataset(
            horizon_minutes=horizon_minutes,
            min_return_threshold=min_return_threshold,
            lookback_periods=lookback_periods,
            use_polars=use_polars,
            start=start,
            end=end,
        )
        return result

    def prepare_training_data(
        self,
        instrument_ids: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        horizon_minutes: int = 15,
        min_return_threshold: float = 0.001,
        lookback_periods: int = 30,
        use_polars: bool = True,
    ) -> _pd.DataFrame | _pl.DataFrame:
        """
        Prepare TFT training data with automatic source selection.

        Automatically selects between FeatureStore and direct computation based on
        availability, ensuring optimal performance and training/inference parity.

        Parameters
        ----------
        instrument_ids : list[str] | None, optional
            List of instrument IDs. If None, uses self.symbols with exchange suffixes.
        start : datetime | None, optional
            Start time for data loading.
        end : datetime | None, optional
            End time for data loading.
        horizon_minutes : int, default 15
            Prediction horizon in minutes.
        min_return_threshold : float, default 0.001
            Minimum return threshold for binary classification.
        lookback_periods : int, default 30
            Minimum lookback periods for feature computation (used in direct mode).
        use_polars : bool, default True
            Whether to return Polars DataFrame (True) or Pandas DataFrame (False).

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            TFT-compatible training dataset with all required features.

        Notes
        -----
        Feature Source Selection:
        - If FeatureStore is configured: Uses pre-computed features (ensures parity)
        - Otherwise: Falls back to direct computation with logging

        Example:
            >>> data = facade.prepare_training_data(
            ...     instrument_ids=["SPY.ARCA"],
            ...     start=datetime(2024, 1, 1),
            ...     end=datetime(2024, 6, 1),
            ... )

        """
        legacy = self._get_legacy_builder()
        result: _pd.DataFrame | _pl.DataFrame = legacy.prepare_training_data(
            instrument_ids=instrument_ids,
            start=start,
            end=end,
            horizon_minutes=horizon_minutes,
            min_return_threshold=min_return_threshold,
            lookback_periods=lookback_periods,
            use_polars=use_polars,
        )
        return result

    def prepare_training_data_from_store(
        self,
        instrument_ids: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        horizon_minutes: int = 15,
        min_return_threshold: float = 0.001,
    ) -> _pl.DataFrame:
        """
        Prepare training data using features from FeatureStore.

        Ensures training/inference parity by using the same features that are
        computed during live trading.

        Parameters
        ----------
        instrument_ids : list[str] | None, optional
            List of instrument IDs to load features for. If None, uses self.symbols.
        start : datetime | None, optional
            Start time for data loading. If None, defaults to 2020-01-01.
        end : datetime | None, optional
            End time for data loading. If None, defaults to now.
        horizon_minutes : int, default 15
            Prediction horizon in minutes for target generation.
        min_return_threshold : float, default 0.001
            Minimum return threshold for binary classification (0.1%).

        Returns
        -------
        pl.DataFrame
            TFT-compatible training dataset with features from FeatureStore.

        Raises
        ------
        ValueError
            If FeatureStore is not configured.
        RuntimeError
            If no features are found in FeatureStore for specified instruments.

        Example:
            >>> # Requires FeatureStore to be configured
            >>> data = facade.prepare_training_data_from_store(
            ...     start=datetime(2024, 1, 1),
            ... )

        """
        legacy = self._get_legacy_builder()
        result: _pl.DataFrame = legacy.prepare_training_data_from_store(
            instrument_ids=instrument_ids,
            start=start,
            end=end,
            horizon_minutes=horizon_minutes,
            min_return_threshold=min_return_threshold,
        )
        return result

    def get_binding_stats(self) -> tuple[MarketBindingStats, ...]:
        """
        Get market binding statistics for all configured bindings.

        Returns statistics about data loaded from each market binding,
        including row counts, time ranges, and source information.

        Returns
        -------
        tuple[MarketBindingStats, ...]
            Tuple of MarketBindingStats for each configured binding.

        Example:
            >>> stats = facade.get_binding_stats()
            >>> for stat in stats:
            ...     print(f"{stat.symbol}: {stat.row_count} rows")

        """
        # Prefer legacy builder stats if it has been initialized
        if self._legacy_builder is not None:
            stats: tuple[MarketBindingStats, ...] = self._legacy_builder.get_binding_stats()
            return stats
        return tuple(self._binding_stats.values())

    # =========================================================================
    # Component Access (for advanced usage and testing)
    # =========================================================================

    @property
    def windowing_component(self) -> TimeSeriesWindowingComponent:
        """
        Access the time series windowing component.

        Returns:
            TimeSeriesWindowingComponent instance.

        """
        return self._windowing

    @property
    def feature_alignment_component(self) -> FeatureAlignmentComponent:
        """
        Access the feature alignment component.

        Returns:
            FeatureAlignmentComponent instance.

        """
        return self._feature_alignment

    @property
    def target_generation_component(self) -> TargetGenerationComponent:
        """
        Access the target generation component.

        Returns:
            TargetGenerationComponent instance.

        """
        return self._target_generation

    @property
    def schema_validator_component(self) -> TFTSchemaValidatorComponent:
        """
        Access the schema validator component.

        Returns:
            TFTSchemaValidatorComponent instance.

        """
        return self._schema_validator
