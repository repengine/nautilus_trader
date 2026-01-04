"""
TFT Dataset Builder Facade for component-based dataset preparation.

This module provides a facade that delegates to focused components while
maintaining 100% API compatibility with the legacy TFTDatasetBuilder.

Feature Flag:
    ML_USE_LEGACY_TFT_BUILDER: Set to '1' or 'true' to use legacy implementation.
    Default (unset): Uses legacy implementation.
    Set to '0'/'false' to opt into component-based facade.

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
from ml.data.common import KnownFutureFeatureComponent
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
        True when ML_USE_LEGACY_TFT_BUILDER is unset, or set to '1'/'true'/'yes' (case-insensitive).

    Example:
        >>> import os
        >>> os.environ.pop("ML_USE_LEGACY_TFT_BUILDER", None)
        >>> assert use_legacy_builder() is True  # default
        >>> os.environ["ML_USE_LEGACY_TFT_BUILDER"] = "0"
        >>> assert use_legacy_builder() is False

    """
    value = os.environ.get("ML_USE_LEGACY_TFT_BUILDER")
    if value is None:
        return True
    token = value.strip().lower()
    return token in ("1", "true", "yes")


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
        include_macro_deltas: bool = False,
        macro_lag_days: int = 1,
        fred_path: str | None = None,
        include_micro: bool = False,
        micro_base_dir: str | None = None,
        include_calendar: bool = False,
        include_calendar_lags: bool = False,
        include_clustering_tags: bool = False,
        include_context_features: bool = False,
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
        include_macro_deltas : bool, default False
            Whether to include macro delta features.
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
        include_calendar_lags : bool, default False
            Whether to include calendar lag features.
        include_clustering_tags : bool, default False
            Whether to include clustering tag features.
        include_context_features : bool, default False
            Whether to include additional context features.
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
        self.include_macro_deltas = include_macro_deltas
        self.macro_lag_days = macro_lag_days
        self.fred_path = fred_path
        self.include_micro = include_micro
        self.micro_base_dir = micro_base_dir
        self.include_calendar = include_calendar
        self.include_calendar_lags = include_calendar_lags
        self.include_clustering_tags = include_clustering_tags
        self.include_context_features = include_context_features
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
        self._known_future = KnownFutureFeatureComponent(include_calendar=include_calendar)

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
                include_macro_deltas=self.include_macro_deltas,
                macro_lag_days=self.macro_lag_days,
                fred_path=self.fred_path,
                include_micro=self.include_micro,
                micro_base_dir=self.micro_base_dir,
                include_calendar=self.include_calendar,
                include_calendar_lags=self.include_calendar_lags,
                include_clustering_tags=self.include_clustering_tags,
                include_context_features=self.include_context_features,
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

        # Use legacy builder if feature flag is set
        if use_legacy_builder():
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

        # Use component-based implementation
        logger.info("Using component-based TFT dataset builder")

        # Try FeatureStore first if available
        if self.feature_store:
            try:
                return self.prepare_training_data_from_store(
                    instrument_ids=None,
                    start=start,
                    end=end,
                    horizon_minutes=horizon_minutes,
                    min_return_threshold=min_return_threshold,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load from FeatureStore: {e}. Falling back to direct computation",
                )

        # Fall back to direct computation using components
        return self._build_training_dataset_direct(
            horizon_minutes=horizon_minutes,
            min_return_threshold=min_return_threshold,
            lookback_periods=lookback_periods,
            use_polars=use_polars,
            start=start,
            end=end,
        )

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
        # Use legacy builder if feature flag is set
        if use_legacy_builder():
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

        # Use component-based implementation
        # Try FeatureStore first if available
        if self.feature_store:
            try:
                df = self.prepare_training_data_from_store(
                    instrument_ids=instrument_ids,
                    start=start,
                    end=end,
                    horizon_minutes=horizon_minutes,
                    min_return_threshold=min_return_threshold,
                )
                if not use_polars:
                    return df.to_pandas()
                return df
            except Exception as e:
                logger.warning(
                    f"Failed to load from FeatureStore: {e}. Falling back to direct computation",
                )

        # Fall back to direct computation
        return self._build_training_dataset_direct(
            horizon_minutes=horizon_minutes,
            min_return_threshold=min_return_threshold,
            lookback_periods=lookback_periods,
            use_polars=use_polars,
            start=start,
            end=end,
        )

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
        # Use legacy builder if feature flag is set
        if use_legacy_builder():
            legacy = self._get_legacy_builder()
            result: _pl.DataFrame = legacy.prepare_training_data_from_store(
                instrument_ids=instrument_ids,
                start=start,
                end=end,
                horizon_minutes=horizon_minutes,
                min_return_threshold=min_return_threshold,
            )
            return result

        # Component-based implementation
        if not self.feature_store:
            msg = "FeatureStore not configured. Cannot load features from store."
            raise ValueError(msg)

        # Resolve instrument IDs
        resolved_ids = self._resolve_instrument_ids(instrument_ids)
        if not resolved_ids:
            msg = "No instrument identifiers available for feature store load"
            raise ValueError(msg)

        logger.info(
            "Loading features from FeatureStore for %d instruments",
            len(resolved_ids),
        )

        # Collect all feature data
        all_data: list[_pl.DataFrame] = []
        default_start = datetime(2020, 1, 1, tzinfo=UTC)
        default_end = datetime.now(tz=UTC)

        for instrument_id in resolved_ids:
            logger.info(f"Processing {instrument_id} from FeatureStore...")

            try:
                # Load features from FeatureStore
                features, timestamps, feature_names = self.feature_store.get_training_data(
                    instrument_id=instrument_id,
                    start=start or default_start,
                    end=end or default_end,
                    include_bars=False,
                )

                if len(features) == 0:
                    logger.warning(f"No features found for {instrument_id} in FeatureStore")
                    continue

                # Convert to Polars DataFrame
                feature_df = pl.DataFrame(
                    {
                        "ts_event": timestamps,
                        **{name: features[:, i] for i, name in enumerate(feature_names)},
                    },
                )

                # Load corresponding bars for target generation
                bars_df = self._load_bars_dataframe(instrument_id, start, end)

                if bars_df.is_empty():
                    logger.warning(f"No bar data found for {instrument_id}")
                    continue

                # Align column names and join features with bars
                if "timestamp" in bars_df.columns:
                    bars_df = bars_df.rename({"timestamp": "ts_event"})

                combined_df = bars_df.join(feature_df, on="ts_event", how="inner")

                # Add instrument identifier
                combined_df = combined_df.with_columns(
                    pl.lit(instrument_id).alias("instrument_id"),
                )

                # Generate targets using component
                targets = self._target_generation.generate_targets_polars(
                    combined_df,
                    horizon_minutes,
                    min_return_threshold,
                )

                # Add time index for TFT
                combined_df = combined_df.sort("ts_event")
                combined_df = combined_df.with_columns(
                    pl.arange(0, len(combined_df)).alias("time_index"),
                )

                # Combine with targets
                dataset = pl.concat([combined_df, targets], how="horizontal")

                # Add TFT-specific features using components
                dataset = self._feature_alignment.add_static_features_polars(dataset)
                dataset = self._known_future.add_known_future_features_polars(dataset)

                all_data.append(dataset)

            except Exception as e:
                logger.error(f"Failed to load features for {instrument_id}: {e}")
                continue

        if not all_data:
            msg = "No features loaded from FeatureStore for any instrument"
            raise RuntimeError(msg)

        # Combine all instruments
        final_df = pl.concat(all_data, how="vertical")

        # Optionally join macro features (as-of with lag)
        if self.include_macro:
            from ml.data.fred_join import join_fred_asof

            final_df = join_fred_asof(
                final_df,
                timestamp_col="ts_event",
                lag_days=self.macro_lag_days,
                fred_path=self.fred_path,
                vintage_base_dir=self.vintage_base_dir,
                series_filter=None if self.macro_series_ids is None else set(self.macro_series_ids),
                vintage_policy=self.vintage_policy,
                vintage_cutoff=self.vintage_as_of,
            )

        logger.info(
            f"Loaded {len(final_df)} rows from FeatureStore with {len(final_df.columns)} columns",
        )

        # Validate schema using component
        try:
            self._schema_validator.validate(final_df)
        except SchemaValidationError as e:
            logger.warning(f"Schema validation warning: {e}")

        return cast("_pl.DataFrame", final_df)

    def _resolve_instrument_ids(self, override: list[str] | None = None) -> list[str]:
        """
        Resolve instrument IDs from override, config, or symbols.

        Args:
            override: Optional explicit instrument IDs to use.

        Returns:
            List of resolved instrument identifiers.

        """
        if override:
            return override
        if self.instrument_ids:
            return self.instrument_ids
        # Generate candidates from symbols with heuristic exchanges
        candidates: list[str] = []
        heuristic_exchanges = ["NYSE", "NASDAQ", "ARCA", "ARCX", "XNAS", "XNYS"]
        for symbol in self.symbols:
            if "." in symbol:
                candidates.append(symbol)
                continue
            for exchange in heuristic_exchanges:
                candidates.append(f"{symbol}.{exchange}")
        if candidates:
            logger.warning(
                "Instrument IDs not provided; falling back to heuristic venues %s",
                heuristic_exchanges,
            )
        return candidates

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

    @property
    def known_future_component(self) -> KnownFutureFeatureComponent:
        """
        Access the known-future feature component.

        Returns:
            KnownFutureFeatureComponent instance.

        """
        return self._known_future

    # =========================================================================
    # Core Processing Methods (Using Components)
    # =========================================================================

    def _process_symbol_polars(
        self,
        df: _pl.DataFrame,
        symbol: str,
        horizon_minutes: int,
        threshold: float,
        lookback_periods: int,
    ) -> _pl.DataFrame | None:
        """
        Process single symbol using components.

        This method orchestrates the component-based processing pipeline:
        1. Compute technical features (FeatureAlignmentComponent)
        2. Generate targets (TargetGenerationComponent)
        3. Add static features (FeatureAlignmentComponent)
        4. Add known-future features (KnownFutureFeatureComponent)

        Args:
            df: Raw OHLCV DataFrame for the symbol.
            symbol: Symbol identifier.
            horizon_minutes: Prediction horizon in minutes.
            threshold: Minimum return threshold for target classification.
            lookback_periods: Minimum lookback periods for feature computation.

        Returns:
            Processed DataFrame with all features, or None if processing fails.

        """
        # Ensure we have required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns for {symbol}")
            return None

        # Sort by time and create sequential index
        df = df.sort("timestamp" if "timestamp" in df.columns else df.columns[0])
        df = df.with_columns(
            [
                pl.arange(0, len(df)).alias("time_index"),
                pl.lit(symbol).alias("instrument_id"),
            ],
        )

        # 1. Generate features using component
        features = self._feature_alignment.compute_features_polars(df)

        # 2. Generate targets using component
        targets = self._target_generation.generate_targets_polars(df, horizon_minutes, threshold)

        # 3. Combine (retain timestamp for macro joins)
        dataset = pl.concat(
            [
                df.select(["timestamp", "time_index", "instrument_id"]),
                features,
                targets,
            ],
            how="horizontal",
        )

        # 4. Filter for sufficient history
        dataset = dataset.slice(lookback_periods, len(dataset))

        # 5. Add static features using component
        dataset = self._feature_alignment.add_static_features_polars(dataset)

        # 6. Add known-future features using component
        dataset = self._known_future.add_known_future_features_polars(dataset)

        return dataset

    def _process_symbol_pandas(
        self,
        df: _pd.DataFrame,
        symbol: str,
        horizon_minutes: int,
        threshold: float,
        lookback_periods: int,
    ) -> _pd.DataFrame | None:
        """
        Process single symbol using components (Pandas path).

        This method orchestrates the component-based processing pipeline
        using Pandas DataFrames for compatibility.

        Args:
            df: Raw OHLCV DataFrame for the symbol.
            symbol: Symbol identifier.
            horizon_minutes: Prediction horizon in minutes.
            threshold: Minimum return threshold for target classification.
            lookback_periods: Minimum lookback periods for feature computation.

        Returns:
            Processed DataFrame with all features, or None if processing fails.

        """
        # Ensure we have required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns for {symbol}")
            return None

        # Sort by time and create sequential index
        time_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
        df = df.sort_values(time_col).reset_index(drop=True)
        df["time_index"] = range(len(df))
        df["instrument_id"] = symbol

        # 1. Generate features using component
        features = self._feature_alignment.compute_features_pandas(df)

        # 2. Generate targets using component
        targets = self._target_generation.generate_targets_pandas(df, horizon_minutes, threshold)

        # 3. Combine DataFrames
        dataset = pd.concat(
            [df[["timestamp", "time_index", "instrument_id"]], features, targets],
            axis=1,
        )

        # 4. Filter for sufficient history
        dataset = dataset.iloc[lookback_periods:].reset_index(drop=True)

        # 5. Add static features using component
        dataset = self._feature_alignment.add_static_features_pandas(dataset)

        # 6. Add known-future features using component
        dataset = self._known_future.add_known_future_features_pandas(dataset)

        return dataset

    def _build_training_dataset_direct(
        self,
        horizon_minutes: int = 15,
        min_return_threshold: float = 0.001,
        lookback_periods: int = 30,
        use_polars: bool = True,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> _pd.DataFrame | _pl.DataFrame:
        """
        Build training dataset using direct feature computation with components.

        This method orchestrates the dataset building process:
        1. Iterates over configured symbols
        2. Loads bar data from catalog or DataStore
        3. Processes each symbol using components
        4. Combines results into final dataset

        Args:
            horizon_minutes: Prediction horizon in minutes.
            min_return_threshold: Minimum return threshold for binary classification.
            lookback_periods: Minimum lookback periods for feature computation.
            use_polars: Whether to use Polars (True) or Pandas (False).
            start: Start time for data loading.
            end: End time for data loading.

        Returns:
            TFT-compatible training dataset.

        """
        # Collect results separately to keep typing precise
        all_data_pl: list[_pl.DataFrame] = []
        all_data_pd: list[_pd.DataFrame] = []

        # Candidate venues to try per symbol (ETFs frequently ARCA/ARCX)
        candidate_venues = [
            "ARCA",
            "ARCX",
            "NASDAQ",
            "XNAS",
            "NYSE",
            "XNYS",
        ]

        for symbol in self.symbols:
            logger.info(f"Processing {symbol}...")

            # Load data using catalog
            try:
                df = cast("_pl.DataFrame", pl.DataFrame())
                last_err: Exception | None = None

                # Try instrument candidates
                instrument_candidates = self._get_instrument_candidates(symbol, candidate_venues)
                for instrument_id in instrument_candidates:
                    try:
                        df = self._load_bars_dataframe(instrument_id, start, end)
                        if not df.is_empty():
                            break
                    except Exception as e_inner:  # pragma: no cover
                        last_err = e_inner
                        continue

                if df.is_empty():
                    # Try direct parquet fallback
                    df = self._try_parquet_fallback(symbol)
                    if df.is_empty():
                        if last_err is not None:
                            logger.warning(
                                f"Failed to load data for {symbol} (last error: {last_err})"
                            )
                        else:
                            logger.warning(f"No data found for {symbol}")
                        continue

            except Exception as e:
                logger.warning(f"Failed to load data for {symbol}: {e}")
                continue

            if df.is_empty():
                logger.warning(f"No data found for {symbol}, skipping")
                continue

            # Process with Polars or Pandas
            if use_polars:
                processed = self._process_symbol_polars(
                    df,
                    symbol,
                    horizon_minutes,
                    min_return_threshold,
                    lookback_periods,
                )
                if processed is not None:
                    all_data_pl.append(processed)
            else:
                # Convert to pandas for processing
                df_pd = df.to_pandas()
                processed_pd = self._process_symbol_pandas(
                    df_pd,
                    symbol,
                    horizon_minutes,
                    min_return_threshold,
                    lookback_periods,
                )
                if processed_pd is not None:
                    all_data_pd.append(processed_pd)

        # Combine all symbol data
        if use_polars:
            if not all_data_pl:
                logger.warning("No data processed for any symbol")
                return cast("_pl.DataFrame", pl.DataFrame())
            final_df = pl.concat(all_data_pl, how="vertical")

            # Join optional features
            final_df = self._join_optional_features_polars(final_df)

            # Validate schema using component
            try:
                self._schema_validator.validate(final_df)
            except SchemaValidationError as e:
                logger.warning(f"Schema validation warning: {e}")

            return final_df
        else:
            if not all_data_pd:
                logger.warning("No data processed for any symbol")
                return cast("_pd.DataFrame", pd.DataFrame())
            final_df_pd = pd.concat(all_data_pd, ignore_index=True)

            # Join optional features
            final_df_pd = self._join_optional_features_pandas(final_df_pd)

            # Validate schema using component
            try:
                self._schema_validator.validate(final_df_pd)
            except SchemaValidationError as e:
                logger.warning(f"Schema validation warning: {e}")

            return final_df_pd

    def _join_optional_features_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
        """
        Join optional features (macro, micro, L2, earnings) to Polars DataFrame.

        Args:
            df: Base dataset with timestamp column.

        Returns:
            Dataset with optional features joined.

        """
        # Determine timestamp column
        ts_col = "timestamp" if "timestamp" in df.columns else "ts_event"

        # 1. Join macro features (FRED data)
        if self.include_macro:
            try:
                from ml.data.fred_join import join_fred_asof

                before_cols = set(df.columns)
                joined = join_fred_asof(
                    df,
                    timestamp_col=ts_col,
                    lag_days=self.macro_lag_days,
                    fred_path=self.fred_path,
                    vintage_base_dir=self.vintage_base_dir,
                    series_filter=(
                        None if self.macro_series_ids is None else set(self.macro_series_ids)
                    ),
                    vintage_policy=self.vintage_policy,
                    vintage_cutoff=self.vintage_as_of,
                )
                # Handle union return type
                if hasattr(joined, "schema"):
                    df = cast("_pl.DataFrame", joined)
                else:
                    df = pl.from_pandas(joined)
                macro_cols = [c for c in df.columns if c not in before_cols]
                if macro_cols:
                    # Add availability flag BEFORE filling nulls (so it's meaningful)
                    exprs = [pl.col(c).is_not_null() for c in macro_cols]
                    if exprs:
                        any_macro = exprs[0]
                        for ex in exprs[1:]:
                            any_macro = any_macro | ex
                        df = df.with_columns([any_macro.cast(pl.Int32).alias("is_macro_available")])
                    # Fill nulls AFTER computing availability mask
                    fills = [
                        pl.col(c).fill_null(0)
                        for c in macro_cols
                        if df.schema.get(c) is not None and df.schema[c].is_numeric()
                    ]
                    if fills:
                        df = df.with_columns(fills)
            except Exception as e:
                logger.debug(f"Macro feature join skipped: {e}")

        # 2. Join micro features (trade imbalance, VWAP distance)
        if self.include_micro and self.micro_base_dir:
            try:
                from ml.data.loaders.micro import load_micro_features

                micro_df = load_micro_features(
                    self.micro_base_dir,
                    symbols=self.symbols,
                )
                if micro_df is not None and not micro_df.is_empty():
                    df = df.join(micro_df, on=[ts_col, "instrument_id"], how="left")
            except Exception as e:
                logger.debug(f"Micro feature join skipped: {e}")

        # 3. Join L2 features (order book depth, spread)
        if self.include_l2 and self.l2_base_dir:
            try:
                from ml.data.loaders.l2 import load_l2_features

                l2_df = load_l2_features(
                    self.l2_base_dir,
                    symbols=self.symbols,
                )
                if l2_df is not None and not l2_df.is_empty():
                    before_cols = set(df.columns)
                    df = df.join(l2_df, on=[ts_col, "instrument_id"], how="left")
                    l2_cols = [c for c in df.columns if c not in before_cols]
                    if l2_cols:
                        df = df.with_columns(
                            [pl.lit(1).cast(pl.Int32).alias("is_l2_available")]
                        )
            except Exception as e:
                logger.debug(f"L2 feature join skipped: {e}")

        # 4. Join earnings features
        if self.include_earnings and self.data_store:
            try:
                df = self._join_earnings_features_polars(df, ts_col)
            except Exception as e:
                logger.debug(f"Earnings feature join skipped: {e}")

        return df

    def _join_optional_features_pandas(self, df: _pd.DataFrame) -> _pd.DataFrame:
        """
        Join optional features (macro, micro, L2, earnings) to Pandas DataFrame.

        Args:
            df: Base dataset with timestamp column.

        Returns:
            Dataset with optional features joined.

        """
        # Determine timestamp column
        ts_col = "timestamp" if "timestamp" in df.columns else "ts_event"

        # 1. Join macro features (FRED data)
        if self.include_macro:
            try:
                from ml.data.fred_join import join_fred_asof

                joined = join_fred_asof(
                    df,
                    timestamp_col=ts_col,
                    lag_days=self.macro_lag_days,
                    fred_path=self.fred_path,
                    vintage_base_dir=self.vintage_base_dir,
                    series_filter=(
                        None if self.macro_series_ids is None else set(self.macro_series_ids)
                    ),
                    vintage_policy=self.vintage_policy,
                    vintage_cutoff=self.vintage_as_of,
                )
                # Handle union return type - join_fred_asof can return pd or pl
                if isinstance(joined, pl.DataFrame):
                    df = cast("_pd.DataFrame", joined.to_pandas())
                else:
                    df = cast("_pd.DataFrame", joined)
                # Find macro columns and compute availability BEFORE filling nulls
                core = {"timestamp", "ts_event", "time_index", "instrument_id", "y"}
                potential_macro = [c for c in df.columns if c not in core and "__" in c]
                if potential_macro:
                    # Compute availability mask BEFORE filling nulls (so it's meaningful)
                    df["is_macro_available"] = (
                        df[potential_macro].notna().any(axis=1).astype("int32")
                    )
                    # Fill nulls AFTER computing availability
                    df[potential_macro] = df[potential_macro].fillna(0)
            except Exception as e:
                logger.debug(f"Macro feature join skipped: {e}")

        # 2. Join micro features
        if self.include_micro and self.micro_base_dir:
            try:
                from ml.data.loaders.micro import load_micro_features

                micro_df = load_micro_features(
                    self.micro_base_dir,
                    symbols=self.symbols,
                )
                if micro_df is not None and len(micro_df) > 0:
                    if hasattr(micro_df, "to_pandas"):
                        micro_df = micro_df.to_pandas()
                    df = df.merge(micro_df, on=[ts_col, "instrument_id"], how="left")
            except Exception as e:
                logger.debug(f"Micro feature join skipped: {e}")

        # 3. Join L2 features
        if self.include_l2 and self.l2_base_dir:
            try:
                from ml.data.loaders.l2 import load_l2_features

                l2_df = load_l2_features(
                    self.l2_base_dir,
                    symbols=self.symbols,
                )
                if l2_df is not None and len(l2_df) > 0:
                    if hasattr(l2_df, "to_pandas"):
                        l2_df = l2_df.to_pandas()
                    before_cols = set(df.columns)
                    df = df.merge(l2_df, on=[ts_col, "instrument_id"], how="left")
                    if set(df.columns) != before_cols:
                        df["is_l2_available"] = 1
            except Exception as e:
                logger.debug(f"L2 feature join skipped: {e}")

        # 4. Join earnings features
        if self.include_earnings and self.data_store:
            try:
                df = self._join_earnings_features_pandas(df, ts_col)
            except Exception as e:
                logger.debug(f"Earnings feature join skipped: {e}")

        return df

    def _join_earnings_features_polars(
        self,
        df: _pl.DataFrame,
        ts_col: str,
    ) -> _pl.DataFrame:
        """Join earnings features from DataStore to Polars DataFrame."""
        if not self.data_store or "instrument_id" not in df.columns:
            return df

        try:
            # Get unique instruments
            instruments = df.select(pl.col("instrument_id")).unique()["instrument_id"].to_list()

            for instrument_id in instruments:
                ticker = instrument_id.split(".")[0] if "." in instrument_id else instrument_id
                # Get timestamps for this instrument
                mask = df["instrument_id"] == instrument_id
                timestamps = df.filter(mask).select(pl.col(ts_col))

                # Fetch earnings features from legacy builder if available
                legacy = self._get_legacy_builder()
                earnings_df = legacy._fetch_earnings_features(
                    ticker=ticker,
                    timestamps=timestamps[ts_col],
                    as_of_date=self.vintage_as_of,
                )

                if earnings_df is not None and not earnings_df.is_empty():
                    # Rename timestamp column if needed
                    if ts_col not in earnings_df.columns and "ts_event" in earnings_df.columns:
                        earnings_df = earnings_df.rename({"ts_event": ts_col})
                    df = df.join(earnings_df, on=ts_col, how="left")
                    # Add availability flag
                    df = df.with_columns([pl.lit(1).cast(pl.Int32).alias("is_earnings_available")])

        except Exception as e:
            logger.debug(f"Earnings feature join error: {e}")

        return df

    def _join_earnings_features_pandas(
        self,
        df: _pd.DataFrame,
        ts_col: str,
    ) -> _pd.DataFrame:
        """Join earnings features from DataStore to Pandas DataFrame."""
        if not self.data_store or "instrument_id" not in df.columns:
            return df

        try:
            # Convert to Polars, join, convert back
            df_pl = pl.from_pandas(df)
            df_pl = self._join_earnings_features_polars(df_pl, ts_col)
            df = df_pl.to_pandas()
        except Exception as e:
            logger.debug(f"Earnings feature join error: {e}")

        return df

    def _get_instrument_candidates(
        self,
        symbol: str,
        candidate_venues: list[str],
    ) -> list[str]:
        """
        Get list of instrument ID candidates to try for a symbol.

        Args:
            symbol: Base symbol (e.g., "SPY").
            candidate_venues: List of venue suffixes to try.

        Returns:
            List of instrument IDs to attempt loading.

        """
        if self.instrument_ids:
            # Use explicitly provided instrument IDs
            return [iid for iid in self.instrument_ids if iid.startswith(symbol)]
        # Generate candidates from symbol + venue combinations
        return [f"{symbol}.{venue}" for venue in candidate_venues]

    def _load_bars_dataframe(
        self,
        instrument_id: str,
        start: datetime | None,
        end: datetime | None,
    ) -> _pl.DataFrame:
        """
        Load bar data for an instrument from DataStore or catalog.

        Args:
            instrument_id: Instrument identifier (e.g., "SPY.ARCA").
            start: Start time for data loading.
            end: End time for data loading.

        Returns:
            Polars DataFrame with OHLCV data.

        """
        from ml.data.catalog_utils import bars_to_dataframe

        # Try DataStore first if available
        if self.data_store is not None and self.market_dataset_id:
            try:
                start_ns = self._datetime_to_ns(start, fallback=0)
                end_ns = self._datetime_to_ns(end, fallback=self._now_ns())

                raw_result = self.data_store.read_range(
                    dataset_id=self.market_dataset_id,
                    instrument_id=instrument_id,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )

                frame = self._to_polars(raw_result)
                if not frame.is_empty():
                    if "timestamp" not in frame.columns and "ts_event" in frame.columns:
                        frame = frame.rename({"ts_event": "timestamp"})
                    return frame

            except Exception:
                logger.debug(
                    "DataStore read_range failed; falling back to catalog",
                    exc_info=True,
                )

        # Fall back to catalog
        try:
            frame = bars_to_dataframe(
                self.catalog,
                [instrument_id],
                start=start,
                end=end,
            )
            return frame
        except Exception:
            logger.debug("Catalog read failed", exc_info=True)
            return cast("_pl.DataFrame", pl.DataFrame())

    def _try_parquet_fallback(self, symbol: str) -> _pl.DataFrame:
        """
        Try loading data from direct parquet files as fallback.

        Args:
            symbol: Symbol to load data for.

        Returns:
            Polars DataFrame with OHLCV data, or empty DataFrame if not found.

        """
        try:
            base = _Path(self.micro_base_dir or "data/tier1")
            paths = [
                base / symbol / "ohlcv-1m_historical.parquet",
                base / symbol / "ohlcv-1m_recent.parquet",
            ]
            frames: list[_pl.DataFrame] = []

            for p in paths:
                if p.exists():
                    part = pl.read_parquet(str(p))
                    if not part.is_empty():
                        if "timestamp" not in part.columns and "ts_event" in part.columns:
                            part = part.rename({"ts_event": "timestamp"})
                        frames.append(part)

            if frames:
                df = pl.concat(frames, how="vertical")
                if "timestamp" in df.columns:
                    df = df.sort("timestamp")
                return cast("_pl.DataFrame", df)

        except Exception as e:
            logger.debug(f"Parquet fallback failed for {symbol}: {e}")

        return cast("_pl.DataFrame", pl.DataFrame())

    def _datetime_to_ns(self, value: datetime | None, *, fallback: int) -> int:
        """Convert datetime to nanoseconds since epoch using windowing component."""
        return self._windowing.datetime_to_ns(value, fallback=fallback)

    @staticmethod
    def _now_ns() -> int:
        """Get current time in nanoseconds since epoch."""
        return int(datetime.now(tz=UTC).timestamp() * 1_000_000_000)

    def _to_polars(self, data: Any) -> _pl.DataFrame:
        """Convert data to Polars DataFrame."""
        if pl is not None and isinstance(data, pl.DataFrame):
            return cast("_pl.DataFrame", data)
        if pd is not None and isinstance(data, pd.DataFrame):
            return cast("_pl.DataFrame", pl.from_pandas(data))
        # Return empty DataFrame for unsupported types
        return cast("_pl.DataFrame", pl.DataFrame())
