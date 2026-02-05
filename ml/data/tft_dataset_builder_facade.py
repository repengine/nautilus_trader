"""
TFT Dataset Builder Facade for component-based dataset preparation.

This module provides a facade that delegates to focused components while
maintaining API compatibility with the historical TFTDatasetBuilder.

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
from ml.config.feature_cache import FeatureCachePolicy
from ml.config.feature_cache import normalize_feature_cache_policy
from ml.config.targets import TargetSemanticsConfig
from ml.data.common import FeatureAlignmentComponent
from ml.data.common import KnownFutureFeatureComponent
from ml.data.common import SchemaValidationError
from ml.data.common import TargetGenerationComponent
from ml.data.common import TFTSchemaValidatorComponent
from ml.data.common import TimeSeriesWindowingComponent
from ml.data.common.cache_joins import join_l2_cache_pandas
from ml.data.common.cache_joins import join_l2_cache_polars
from ml.data.common.cache_joins import join_micro_cache_pandas
from ml.data.common.cache_joins import join_micro_cache_polars
from ml.data.common.earnings_join import fetch_earnings_features
from ml.data.common.event_provider import build_event_provider
from ml.data.common.feature_config_utils import normalize_feature_config
from ml.data.common.frame_metadata import extract_frame_metadata
from ml.data.common.target_semantics import resolve_target_semantics
from ml.data.ingest.market_bindings import MarketBindingStats
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.ingest.market_bindings import build_binding_index
from ml.data.ingest.market_bindings import resolve_binding
from ml.data.ingest.market_bindings import store_enabled
from ml.data.vintage import VintagePolicy
from ml.stores.feature_store import FeatureStore
from ml.stores.protocols import DataStoreFacadeProtocol
from ml.training.datasets import DatasetSerializer
from ml.training.datasets import ValidationSplitter
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


if TYPE_CHECKING:
    import pandas as _pd
    import polars as _pl

    from ml.data.providers.events import EventScheduleProvider
else:  # pragma: no cover - typing fallback
    _pd = Any
    _pl = Any


# Runtime aliases
pl: Any = cast(Any, pl_runtime)
pd: Any = cast(Any, pd_runtime)


logger = logging.getLogger(__name__)


# Re-export SchemaValidationError for consumers
__all__ = [
    "SchemaValidationError",
    "TFTDatasetBuilder",
    "TFTDatasetBuilderFacade",
]


class TFTDatasetBuilderFacade:
    """
    Component-based TFT dataset builder with API compatibility.

    This facade maintains API parity with the historical TFTDatasetBuilder
    while delegating to focused, single-responsibility components for:
    - Time series windowing and bounds extraction
    - Feature alignment and computation
    - Target generation for binary classification
    - Schema validation

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
        ...     target_semantics=target_semantics,
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
        micro_cache_policy: FeatureCachePolicy = "cache_first",
        include_calendar: bool = False,
        include_calendar_lags: bool = False,
        include_clustering_tags: bool = False,
        include_context_features: bool = False,
        include_events: bool = False,
        include_earnings: bool = False,
        earnings_lag_days: int = 1,
        include_l2: bool = False,
        l2_base_dir: str | None = None,
        l2_cache_policy: FeatureCachePolicy = "cache_first",
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
        micro_cache_policy : FeatureCachePolicy, default "cache_first"
            Cache policy for microstructure features.
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
        l2_cache_policy : FeatureCachePolicy, default "cache_first"
            Cache policy for L2 features.
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
        self.feature_config = normalize_feature_config(feature_config or MLFeatureConfig())
        self.feature_store = feature_store
        self.data_store = data_store
        self.market_dataset_id = market_dataset_id
        self.market_bindings = tuple(market_bindings or ())
        self.include_macro = include_macro
        self.include_macro_deltas = include_macro_deltas
        self.macro_lag_days = macro_lag_days
        self.fred_path = fred_path
        self.include_l2 = include_l2
        self.include_micro = include_micro or include_l2
        self.micro_base_dir = micro_base_dir
        self.micro_cache_policy = normalize_feature_cache_policy(
            micro_cache_policy,
            label="micro_cache_policy",
        )
        self.include_calendar = include_calendar
        self.include_calendar_lags = include_calendar_lags
        self.include_clustering_tags = include_clustering_tags
        self.include_context_features = include_context_features
        self.include_events = include_events
        self.include_event_schedule = (
            self.include_events
            or self.include_calendar_lags
            or self.include_clustering_tags
            or self.include_context_features
        )
        self.include_earnings = include_earnings and data_store is not None
        self.earnings_lag_days = earnings_lag_days
        self.l2_base_dir = l2_base_dir
        self.l2_cache_policy = normalize_feature_cache_policy(
            l2_cache_policy,
            label="l2_cache_policy",
        )
        self.vintage_base_dir = _Path(vintage_base_dir).expanduser() if vintage_base_dir else None
        self.events_base_dir = (
            _Path(events_base_dir).expanduser() if events_base_dir else _Path("data/events")
        )
        self.student_mode = student_mode
        self._allow_parquet_fallback = (
            os.getenv("ML_TFT_ALLOW_PARQUET_FALLBACK", "0") == "1"
        )
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
        self._feature_alignment = FeatureAlignmentComponent(feature_config=self.feature_config)
        self._target_generation = TargetGenerationComponent()
        self._schema_validator = TFTSchemaValidatorComponent()
        self._known_future = KnownFutureFeatureComponent(
            include_calendar=include_calendar,
            include_event_schedule=self.include_event_schedule,
            feature_config=self.feature_config,
            events_base_dir=self.events_base_dir,
        )

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
                provider_dataset_id=binding.provider_dataset_id,
                provider_schema=binding.provider_schema,
            )

        self._binding_index = build_binding_index(self.market_bindings)

        # Shared dataset IO utilities (used by E2E tests and training workflows)
        self._dataset_serializer = DatasetSerializer()
        self._validation_splitter = ValidationSplitter()

        self._event_provider: EventScheduleProvider | None = None

        logger.info(
            f"Initialized TFTDatasetBuilderFacade with {len(symbols)} symbols "
            f"(FeatureStore: {'enabled' if feature_store else 'disabled'})",
        )

    def _get_event_provider(self) -> EventScheduleProvider | None:
        """
        Lazily initialize the event schedule provider.

        Returns:
            EventScheduleProvider instance or None when disabled/unavailable.
        """
        if self._event_provider is not None:
            return self._event_provider
        if not (
            self.include_events
            or self.include_calendar_lags
            or self.include_clustering_tags
            or self.include_context_features
        ):
            return None

        self._event_provider = build_event_provider(self.events_base_dir)
        return self._event_provider

    @staticmethod
    def _resolve_target_semantics(
        *,
        target_semantics: TargetSemanticsConfig | None,
    ) -> TargetSemanticsConfig:
        """
        Resolve target semantics for dataset generation.

        Args:
            target_semantics: Explicit target semantics configuration.

        Returns:
            Resolved TargetSemanticsConfig instance.
        """
        return resolve_target_semantics(
            target_semantics,
            error_message=(
                "target_semantics is required; legacy horizon/threshold defaults are not supported"
            ),
        )

    def _append_macro_delta_features_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
        """
        Append macro delta features using the feature alignment component.

        Args:
            df: Polars DataFrame with macro series columns.

        Returns:
            DataFrame with ``*_delta_1d`` columns appended when enabled.
        """
        return self._feature_alignment.append_macro_delta_features_polars(
            df,
            include_macro=self.include_macro,
            include_macro_deltas=self.include_macro_deltas,
            macro_series_ids=self.macro_series_ids,
        )

    # =========================================================================
    # Public API - Matching Legacy TFTDatasetBuilder Exactly
    # =========================================================================

    def build_training_dataset(
        self,
        *,
        target_semantics: TargetSemanticsConfig,
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
        target_semantics : TargetSemanticsConfig
            Explicit target semantics configuration (required).
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
            - Features: canonical pipeline outputs from `ml/features/pipeline.py`
            - Targets: y (binary), forward_return (continuous)
            - Static features: asset_class, tick_size, exchange
            - Calendar features: canonical calendar/event schedule fields

        Example:
            >>> dataset = facade.build_training_dataset(
            ...     target_semantics=target_semantics,
            ...     use_polars=True,
            ... )
            >>> assert "y" in dataset.columns
            >>> assert "forward_return" in dataset.columns

        """
        resolved_semantics = self._resolve_target_semantics(target_semantics=target_semantics)

        # Use component-based implementation
        logger.info("Using component-based TFT dataset builder")

        # Try FeatureStore first if available
        if self.feature_store:
            try:
                return self.prepare_training_data_from_store(
                    instrument_ids=None,
                    start=start,
                    end=end,
                    target_semantics=resolved_semantics,
                )
            except Exception:
                logger.warning(
                    "Failed to load from FeatureStore; falling back to direct computation",
                    exc_info=True,
                )

        # Fall back to direct computation using components
        return self._build_training_dataset_direct(
            target_semantics=resolved_semantics,
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
        *,
        target_semantics: TargetSemanticsConfig,
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
        target_semantics : TargetSemanticsConfig
            Explicit target semantics configuration (required).
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
            ...     target_semantics=target_semantics,
            ... )

        """
        resolved_semantics = self._resolve_target_semantics(target_semantics=target_semantics)

        # Use component-based implementation
        # Try FeatureStore first if available
        if self.feature_store:
            try:
                df = self.prepare_training_data_from_store(
                    instrument_ids=instrument_ids,
                    start=start,
                    end=end,
                    target_semantics=resolved_semantics,
                )
                if not use_polars:
                    return df.to_pandas()
                return df
            except Exception:
                logger.warning(
                    "Failed to load from FeatureStore; falling back to direct computation",
                    exc_info=True,
                )

        # Fall back to direct computation
        return self._build_training_dataset_direct(
            target_semantics=resolved_semantics,
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
        *,
        target_semantics: TargetSemanticsConfig,
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
        target_semantics : TargetSemanticsConfig
            Explicit target semantics configuration (required).

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
            ...     target_semantics=target_semantics,
            ... )

        """
        resolved_semantics = self._resolve_target_semantics(target_semantics=target_semantics)

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
                    resolved_semantics,
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
                dataset = self._known_future.add_known_future_features_canonical_polars(dataset)

                all_data.append(dataset)

            except Exception:
                logger.error(
                    "Failed to load features for %s",
                    instrument_id,
                    exc_info=True,
                )
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
                include_revisions=self.include_macro_revisions,
                revision_mode=self.macro_revision_mode,
                revision_windows=self.macro_revision_windows,
            )
            final_df = self._append_macro_delta_features_polars(
                cast("_pl.DataFrame", final_df),
            )

        logger.info(
            f"Loaded {len(final_df)} rows from FeatureStore with {len(final_df.columns)} columns",
        )

        # Validate schema using component
        try:
            self._schema_validator.validate(final_df)
        except SchemaValidationError:
            logger.warning("Schema validation warning", exc_info=True)

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

    def _add_known_future_features_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
        """
        Backwards-compatible wrapper for known-future features (Polars).
        """
        return self._known_future.add_known_future_features_canonical_polars(df)

    def _add_known_future_features_pandas(self, df: _pd.DataFrame) -> _pd.DataFrame:
        """
        Backwards-compatible wrapper for known-future features (Pandas).
        """
        return self._known_future.add_known_future_features_canonical_pandas(df)

    # =========================================================================
    # Core Processing Methods (Using Components)
    # =========================================================================

    def _process_symbol_polars(
        self,
        df: _pl.DataFrame,
        symbol: str,
        target_semantics: TargetSemanticsConfig,
        lookback_periods: int,
        end: datetime | None = None,
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
        target_semantics: Target semantics configuration for target generation.
        lookback_periods: Minimum lookback periods for feature computation.
        end: Optional end timestamp for point-in-time joins.

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

        # 1. Generate features using canonical pipeline backend
        features = self._feature_alignment.compute_features_canonical_polars(df)

        # 2. Generate targets using component
        targets = self._target_generation.generate_targets_polars(df, target_semantics)

        # 3. Combine (retain timestamp for macro joins)
        dataset = pl.concat(
            [
                df.select(
                    [
                        "timestamp",
                        "time_index",
                        "instrument_id",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ],
                ),
                features,
                targets,
            ],
            how="horizontal",
        )

        # 4. Filter for sufficient history
        dataset = dataset.slice(lookback_periods, len(dataset))

        # 5. Add static features using component
        dataset = self._feature_alignment.add_static_features_polars(dataset)

        # 6. Add known-future features using canonical pipeline backend
        dataset = self._known_future.add_known_future_features_canonical_polars(
            dataset,
        )

        # Optionally join microstructure features (per-minute)
        if self.include_micro:
            dataset = join_micro_cache_polars(
                dataset,
                symbol=symbol,
                raw_base_dir=_Path(self.micro_base_dir or "data/tier1"),
                cache_dir=_Path("data/features/micro_minute"),
                policy=self.micro_cache_policy,
            )

        # Optionally join L2 features (per-minute)
        if self.include_l2:
            dataset = join_l2_cache_polars(
                dataset,
                symbol=symbol,
                raw_base_dir=_Path(self.l2_base_dir or "data/tier1"),
                cache_dir=_Path("data/features/l2_minute"),
                policy=self.l2_cache_policy,
            )

        if self.include_earnings and "timestamp" in dataset.columns:
            if pl is None:
                logger.debug(
                    "Polars unavailable; skipping earnings features for %s",
                    symbol,
                )
            else:
                try:
                    if dataset["timestamp"].dtype != pl.Datetime("ns", "UTC"):
                        dataset = dataset.with_columns(
                            pl.col("timestamp").cast(pl.Datetime("ns", "UTC")),
                        )
                    ts_series = dataset.get_column("timestamp")
                    assert self.data_store is not None
                    earnings_df = fetch_earnings_features(
                        data_store=self.data_store,
                        ticker=symbol,
                        timestamps=ts_series,
                        earnings_lag_days=self.earnings_lag_days,
                        as_of_date=end,
                    )
                    if earnings_df is not None and not earnings_df.is_empty():
                        if earnings_df["timestamp"].dtype != pl.Datetime:
                            earnings_df = earnings_df.with_columns(
                                pl.col("timestamp").cast(pl.Datetime("ns", "UTC")),
                            )
                        before_cols = set(dataset.columns)
                        dataset = dataset.join(earnings_df, on="timestamp", how="left")
                        new_cols = [c for c in dataset.columns if c not in before_cols]
                        if new_cols:
                            earnings_fills: list[Any] = []
                            earnings_availability: list[Any] = []
                            for col in new_cols:
                                if col == "is_earnings_available":
                                    continue
                                try:
                                    if dataset.schema[col].is_numeric():
                                        earnings_fills.append(pl.col(col).fill_null(0))
                                except Exception:
                                    pass
                                earnings_availability.append(pl.col(col).is_not_null())
                            if earnings_fills:
                                dataset = dataset.with_columns(earnings_fills)
                            if "is_earnings_available" not in dataset.columns:
                                has_any = None
                                for expr in earnings_availability:
                                    has_any = expr if has_any is None else (has_any | expr)
                                if has_any is not None:
                                    dataset = dataset.with_columns(
                                        has_any.cast(pl.Int32).alias("is_earnings_available"),
                                    )
                                else:
                                    dataset = dataset.with_columns(
                                        pl.lit(0).alias("is_earnings_available"),
                                    )
                    else:
                        logger.debug("No earnings features available for %s", symbol)
                except Exception:
                    logger.warning(
                        "Earnings feature join failed for %s",
                        symbol,
                        exc_info=True,
                    )

        # Optionally add event-based known-future features
        if (
            self.include_events
            or self.include_calendar_lags
            or self.include_clustering_tags
            or self.include_context_features
        ):
            provider = self._get_event_provider()
            if provider is not None:
                try:
                    ts_series = dataset.select(pl.col("timestamp").cast(pl.Int64))["timestamp"]
                    ev = provider.compute_features(ts_series, [symbol])
                    if not ev.is_empty():
                        ev = ev.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
                        dataset = dataset.join(ev, on="timestamp", how="left")
                except Exception:
                    logger.debug(
                        "Event feature join failed for %s",
                        symbol,
                        exc_info=True,
                    )

        return dataset

    def _process_symbol_pandas(
        self,
        df: _pd.DataFrame,
        symbol: str,
        target_semantics: TargetSemanticsConfig,
        lookback_periods: int,
    ) -> _pd.DataFrame | None:
        """
        Process single symbol using components (Pandas path).

        This method orchestrates the component-based processing pipeline
        using Pandas DataFrames for compatibility.

        Args:
            df: Raw OHLCV DataFrame for the symbol.
            symbol: Symbol identifier.
        target_semantics: Target semantics configuration for target generation.
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

        # 1. Generate features using canonical pipeline backend
        features = self._feature_alignment.compute_features_canonical_pandas(df)

        # 2. Generate targets using component
        targets = self._target_generation.generate_targets_pandas(df, target_semantics)

        # 3. Combine DataFrames
        dataset = pd.concat(
            [
                df[
                    [
                        "timestamp",
                        "time_index",
                        "instrument_id",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                ],
                features,
                targets,
            ],
            axis=1,
        )

        # 4. Filter for sufficient history
        dataset = dataset.iloc[lookback_periods:].reset_index(drop=True)

        # 5. Add static features using component
        dataset = self._feature_alignment.add_static_features_pandas(dataset)

        # 6. Add known-future features using canonical pipeline backend
        dataset = self._known_future.add_known_future_features_canonical_pandas(
            dataset,
        )

        # Optionally join microstructure features (per-minute)
        if self.include_micro:
            dataset = join_micro_cache_pandas(
                dataset,
                symbol=symbol,
                raw_base_dir=_Path(self.micro_base_dir or "data/tier1"),
                cache_dir=_Path("data/features/micro_minute"),
                policy=self.micro_cache_policy,
            )

        # Optionally join L2 features (per-minute)
        if self.include_l2:
            dataset = join_l2_cache_pandas(
                dataset,
                symbol=symbol,
                raw_base_dir=_Path(self.l2_base_dir or "data/tier1"),
                cache_dir=_Path("data/features/l2_minute"),
                policy=self.l2_cache_policy,
            )

        if self.include_earnings and "timestamp" in dataset.columns:
            if pl is None:
                logger.debug(
                    "Polars unavailable; skipping earnings features for %s",
                    symbol,
                )
            else:
                try:
                    ts_series_pl = pl.Series(
                        "timestamp",
                        dataset["timestamp"].astype("datetime64[ns]"),
                    )
                    assert self.data_store is not None
                    earnings_df_pl = fetch_earnings_features(
                        data_store=self.data_store,
                        ticker=symbol,
                        timestamps=ts_series_pl,
                        earnings_lag_days=self.earnings_lag_days,
                        as_of_date=None,
                    )
                    if earnings_df_pl is not None and not earnings_df_pl.is_empty():
                        earnings_df_pd = earnings_df_pl.to_pandas()
                        dataset = dataset.merge(earnings_df_pd, on="timestamp", how="left")
                        new_cols = [c for c in earnings_df_pd.columns if c != "timestamp"]
                        numeric_cols = [c for c in new_cols if c != "is_earnings_available"]
                        if numeric_cols:
                            dataset[numeric_cols] = dataset[numeric_cols].fillna(0)
                        if "is_earnings_available" in new_cols:
                            dataset["is_earnings_available"] = (
                                dataset["is_earnings_available"].fillna(0).astype("int32")
                            )
                        elif numeric_cols:
                            dataset["is_earnings_available"] = (
                                dataset[numeric_cols].notna().any(axis=1).astype("int32")
                            )
                        else:
                            dataset["is_earnings_available"] = 0
                    else:
                        logger.debug("No earnings features available for %s", symbol)
                except Exception:
                    logger.warning(
                        "Earnings feature join failed for %s",
                        symbol,
                        exc_info=True,
                    )

        # Optionally add event-based known-future features
        if (
            self.include_events
            or self.include_calendar_lags
            or self.include_clustering_tags
            or self.include_context_features
        ):
            if pl is None:
                logger.debug(
                    "Polars unavailable; skipping event features for %s",
                    symbol,
                )
            else:
                provider = self._get_event_provider()
                if provider is not None and "timestamp" in dataset.columns:
                    try:
                        ts_series = pl.Series(
                            "timestamp",
                            dataset["timestamp"].astype("datetime64[ns]"),
                        ).cast(pl.Int64)
                        ev_pl = provider.compute_features(ts_series, [symbol])
                        ev_pl = ev_pl.with_columns(
                            pl.col("timestamp").cast(pl.Datetime("ns", "UTC")),
                        )
                        ev_pd = ev_pl.to_pandas()
                        dataset = dataset.merge(ev_pd, on="timestamp", how="left")
                    except Exception:
                        logger.debug(
                            "Event feature join failed for %s",
                            symbol,
                            exc_info=True,
                        )

        return dataset

    def _build_training_dataset_direct(
        self,
        target_semantics: TargetSemanticsConfig,
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
            target_semantics: Target semantics configuration for target generation.
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

                if df.is_empty() and self._allow_parquet_fallback:
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

            except Exception:
                logger.warning(
                    "Failed to load data for %s",
                    symbol,
                    exc_info=True,
                )
                continue

            if df.is_empty():
                logger.warning(f"No data found for {symbol}, skipping")
                continue

            # Process with Polars or Pandas
            if use_polars:
                processed = self._process_symbol_polars(
                    df,
                    symbol,
                    target_semantics,
                    lookback_periods,
                    end=end,
                )
                if processed is not None:
                    all_data_pl.append(processed)
            else:
                # Convert to pandas for processing
                df_pd = df.to_pandas()
                processed_pd = self._process_symbol_pandas(
                    df_pd,
                    symbol,
                    target_semantics,
                    lookback_periods,
                )
                if processed_pd is not None:
                    if (start is not None or end is not None) and "timestamp" in processed_pd.columns:
                        try:
                            import pandas as _pd

                            s_ts = _pd.Timestamp(start, tz="UTC") if start is not None else None
                            e_ts = _pd.Timestamp(end, tz="UTC") if end is not None else None
                            if s_ts is not None:
                                processed_pd = processed_pd[processed_pd["timestamp"] >= s_ts]
                            if e_ts is not None:
                                processed_pd = processed_pd[processed_pd["timestamp"] < e_ts]
                        except Exception:
                            pass
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
            except SchemaValidationError:
                logger.warning("Schema validation warning", exc_info=True)

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
            except SchemaValidationError:
                logger.warning("Schema validation warning", exc_info=True)

            return final_df_pd

    def _join_optional_features_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
        """
        Join optional macro features to Polars DataFrame.

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
                    include_revisions=self.include_macro_revisions,
                    revision_mode=self.macro_revision_mode,
                    revision_windows=self.macro_revision_windows,
                )
                # Handle union return type
                if hasattr(joined, "schema"):
                    df = cast("_pl.DataFrame", joined)
                else:
                    df = pl.from_pandas(joined)
                if "timestamp_right" in df.columns:
                    df = df.drop("timestamp_right")
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
                else:
                    df = df.with_columns([pl.lit(0).cast(pl.Int32).alias("is_macro_available")])
            except Exception:
                logger.debug("Macro feature join skipped", exc_info=True)

        if self.include_macro_deltas:
            df = self._append_macro_delta_features_polars(df)

        return df

    def _join_optional_features_pandas(self, df: _pd.DataFrame) -> _pd.DataFrame:
        """
        Join optional macro features to Pandas DataFrame.

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
                    include_revisions=self.include_macro_revisions,
                    revision_mode=self.macro_revision_mode,
                    revision_windows=self.macro_revision_windows,
                )
                # Handle union return type - join_fred_asof can return pd or pl
                if isinstance(joined, pl.DataFrame):
                    df = cast("_pd.DataFrame", joined.to_pandas())
                else:
                    df = cast("_pd.DataFrame", joined)
                if "timestamp_right" in df.columns:
                    df = df.drop(columns=["timestamp_right"])
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
                else:
                    df["is_macro_available"] = 0
            except Exception:
                logger.debug("Macro feature join skipped", exc_info=True)

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
                assert self.data_store is not None

                earnings_df = fetch_earnings_features(
                    data_store=self.data_store,
                    ticker=ticker,
                    timestamps=timestamps[ts_col],
                    earnings_lag_days=self.earnings_lag_days,
                    as_of_date=self.vintage_as_of,
                )

                if earnings_df is not None and not earnings_df.is_empty():
                    # Rename timestamp column if needed
                    if ts_col not in earnings_df.columns and "ts_event" in earnings_df.columns:
                        earnings_df = earnings_df.rename({"ts_event": ts_col})
                    earnings_df = earnings_df.with_columns(
                        pl.lit(instrument_id).alias("instrument_id"),
                    )
                    df = df.join(earnings_df, on=[ts_col, "instrument_id"], how="left")

        except Exception:
            logger.debug("Earnings feature join error", exc_info=True)

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
        except Exception:
            logger.debug("Earnings feature join error", exc_info=True)

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

        binding = resolve_binding(self._binding_index, instrument_id)
        binding_dataset_id = binding.dataset_id if binding else self.market_dataset_id
        stats = self._binding_stats.get(binding.binding_id) if binding else None

        store_attempted = False
        if (
            store_enabled(self.data_store, self.market_dataset_id, self._binding_index)
            and binding_dataset_id
        ):
            try:
                start_ns = self._datetime_to_ns(start, fallback=0)
                end_ns = self._datetime_to_ns(end, fallback=self._now_ns())

                assert self.data_store is not None
                store_attempted = True
                raw_result = self.data_store.read_range(
                    dataset_id=binding_dataset_id,
                    instrument_id=instrument_id,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )

                frame = self._to_polars(raw_result)
                if not frame.is_empty():
                    if "timestamp" not in frame.columns and "ts_event" in frame.columns:
                        frame = frame.rename({"ts_event": "timestamp"})
                    keep = [
                        col
                        for col in (
                            "instrument_id",
                            "timestamp",
                            "ts_init",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "bid",
                            "ask",
                            "bid_size",
                            "ask_size",
                            "last",
                            "trade_count",
                            "vwap",
                            "source_dataset",
                        )
                        if col in frame.columns
                    ]
                    if keep:
                        frame = frame.select(keep)
                    if "timestamp" in frame.columns:
                        try:
                            frame = frame.with_columns(
                                pl.col("timestamp").cast(
                                    pl.Datetime("ns"),
                                    strict=False,
                                ),
                            )
                        except Exception:
                            logger.debug(
                                "Failed to cast store timestamp to datetime",
                                exc_info=True,
                                extra={"instrument_id": instrument_id},
                            )
                    if stats is not None:
                        row_count = int(frame.height)
                        ts_min_ns, ts_max_ns = self._windowing.frame_time_bounds(frame)
                        src_dataset, _, _ = self._extract_frame_metadata(frame)
                        stats.record(
                            source="store",
                            row_count=row_count,
                            ts_min_ns=ts_min_ns,
                            ts_max_ns=ts_max_ns,
                            source_dataset=src_dataset,
                        )
                    return frame

            except Exception as exc:
                logger.warning(
                    "DataStore read_range failed; falling back to catalog",
                    exc_info=True,
                    extra={"instrument_id": instrument_id},
                )
                if not self._allow_parquet_fallback:
                    raise RuntimeError(
                        "DataStore read_range failed and parquet fallback is disabled",
                    ) from exc

        if store_attempted and not self._allow_parquet_fallback:
            raise RuntimeError(
                "DataStore returned no bars and parquet fallback is disabled",
            )

        # Fall back to catalog
        try:
            frame = bars_to_dataframe(
                self.catalog,
                [instrument_id],
                start=start,
                end=end,
            )
            if stats is not None and not frame.is_empty():
                row_count = int(frame.height)
                ts_min_ns, ts_max_ns = self._windowing.frame_time_bounds(frame)
                src_dataset, _, _ = self._extract_frame_metadata(frame)
                stats.record(
                    source="catalog",
                    row_count=row_count,
                    ts_min_ns=ts_min_ns,
                    ts_max_ns=ts_max_ns,
                    source_dataset=src_dataset,
                )
            return frame
        except Exception:
            logger.debug(
                "Parquet catalog read failed",
                exc_info=True,
                extra={"instrument_id": instrument_id},
            )
            return cast(
                "_pl.DataFrame",
                pl.DataFrame(
                    {
                        "instrument_id": [],
                        "timestamp": [],
                        "open": [],
                        "high": [],
                        "low": [],
                        "close": [],
                        "volume": [],
                    },
                ),
            )

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

        except Exception:
            logger.debug(
                "Parquet fallback failed for %s",
                symbol,
                exc_info=True,
            )

        return cast("_pl.DataFrame", pl.DataFrame())

    def _datetime_to_ns(self, value: datetime | None, *, fallback: int) -> int:
        """Convert datetime to nanoseconds since epoch using windowing component."""
        return self._windowing.datetime_to_ns(value, fallback=fallback)

    def _restrict_df_to_window(
        self,
        df: _pl.DataFrame,
        *,
        symbol: str,
        start: datetime | None,
        end: datetime | None,
        lookback_periods: int,
        horizon_minutes: int,
    ) -> _pl.DataFrame:
        """
        Restrict a symbol dataframe to a requested [start, end) window with context.

        Args:
            df: Polars DataFrame with a timestamp column.
            symbol: Symbol identifier (unused; kept for legacy parity).
            start: Start datetime for filtering.
            end: End datetime for filtering.
            lookback_periods: Lookback periods to include for feature computation.
            horizon_minutes: Horizon minutes to include for target computation.

        Returns:
            Filtered DataFrame including lookback and horizon buffers.
        """
        del symbol
        return self._windowing.restrict_df_to_window(
            df,
            start=start,
            end=end,
            lookback_periods=lookback_periods,
            horizon_minutes=horizon_minutes,
        )

    @staticmethod
    def _now_ns() -> int:
        """Get current time in nanoseconds since epoch."""
        return int(datetime.now(tz=UTC).timestamp() * 1_000_000_000)

    @staticmethod
    def _extract_frame_metadata(
        frame: _pl.DataFrame,
    ) -> tuple[str | None, None, None]:
        """
        Extract source metadata from a frame when available.

        Args:
            frame: Polars DataFrame that may include a ``source_dataset`` column.

        Returns:
            Tuple of (source_dataset, None, None) for legacy compatibility.
        """
        return extract_frame_metadata(frame)

    def _to_polars(self, data: Any) -> _pl.DataFrame:
        """Convert data to Polars DataFrame."""
        if pl is not None and isinstance(data, pl.DataFrame):
            return cast("_pl.DataFrame", data)
        if pd is not None and isinstance(data, pd.DataFrame):
            return cast("_pl.DataFrame", pl.from_pandas(data))
        # Return empty DataFrame for unsupported types
        return cast("_pl.DataFrame", pl.DataFrame())


# Backwards-compatible alias for docs and legacy import paths
TFTDatasetBuilder = TFTDatasetBuilderFacade
