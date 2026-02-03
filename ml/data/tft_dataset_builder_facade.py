"""
Compatibility facade for TFT dataset builder operations.

This wrapper preserves the legacy TFTDatasetBuilder API while exposing the
extracted data components and keeping tests patchable.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import ml.data.tft_dataset_builder as tft_builder
from ml._imports import check_ml_dependencies
from ml._imports import pd as pd_runtime
from ml._imports import pl as pl_runtime
from ml.config.base import MLFeatureConfig
from ml.config.targets import TargetSemanticsConfig
from ml.data.common import FeatureAlignmentComponent
from ml.data.common import SchemaValidationError
from ml.data.common import TargetGenerationComponent
from ml.data.common import TFTSchemaValidatorComponent
from ml.data.common import TimeSeriesWindowingComponent
from ml.data.ingest.market_bindings import MarketBindingStats
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.vintage import VintagePolicy
from ml.ml_types import DataFrameLike
from ml.ml_types import PolarsDF
from ml.stores.feature_store_facade import FeatureStore
from ml.stores.protocols import DataStoreFacadeProtocol
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


logger = logging.getLogger(__name__)


class TFTDatasetBuilderFacade:
    """
    Facade wrapper for TFTDatasetBuilder with component access.
    """

    def __init__(
        self,
        catalog: ParquetDataCatalog | None,
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
        vintage_base_dir: str | Path | None = None,
        events_base_dir: str | Path | None = None,
        student_mode: bool = False,
        macro_series_ids: tuple[str, ...] | None = None,
        vintage_policy: VintagePolicy = VintagePolicy.REAL_TIME,
        vintage_as_of: datetime | None = None,
        include_macro_revisions: bool = False,
        macro_revision_mode: str = "core",
        macro_revision_windows: tuple[int, ...] | None = None,
    ) -> None:
        """
        Initialize the TFT dataset builder facade.

        Args:
            catalog: Parquet data catalog (or None to disable builder creation).
            symbols: Symbols to include in the dataset.
            instrument_ids: Optional instrument IDs to override symbol resolution.
            feature_config: Optional feature configuration for computation.
            feature_store: Optional feature store for parity with inference features.
            data_store: Optional data store for raw data reads.
            market_dataset_id: Optional dataset ID for store reads.
            market_bindings: Optional pre-resolved market bindings.
            include_macro: Whether to include macro features.
            include_macro_deltas: Whether to include macro deltas.
            macro_lag_days: Lag applied to macro features.
            fred_path: Optional FRED cache path.
            include_micro: Whether to include microstructure features.
            micro_base_dir: Optional micro feature cache directory.
            include_calendar: Whether to include calendar features.
            include_calendar_lags: Whether to include calendar lag features.
            include_clustering_tags: Whether to include clustering tags.
            include_context_features: Whether to include context features.
            include_events: Whether to include event features.
            include_earnings: Whether to include earnings features.
            earnings_lag_days: Lag applied to earnings features (must be >= 0).
            include_l2: Whether to include L2 features.
            l2_base_dir: Optional L2 cache directory.
            vintage_base_dir: Optional vintage data directory.
            events_base_dir: Optional events data directory.
            student_mode: When True, disables macro/events/L2/earnings.
            macro_series_ids: Optional macro series identifiers.
            vintage_policy: Vintage policy for macro data.
            vintage_as_of: Optional vintage cutoff timestamp.
            include_macro_revisions: Whether to include macro revisions.
            macro_revision_mode: Revision mode for macro features.
            macro_revision_windows: Revision window sizes.

        Returns:
            None
        """
        if not symbols:
            raise ValueError("symbols list cannot be empty")
        if earnings_lag_days < 0:
            raise ValueError("earnings_lag_days must be >= 0")

        self.catalog = catalog
        self.symbols = symbols
        self.instrument_ids = instrument_ids
        self.feature_config = feature_config
        self.feature_store = feature_store
        self.data_store = data_store
        self.market_dataset_id = market_dataset_id
        self.market_bindings = list(market_bindings) if market_bindings is not None else []
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
        if include_earnings and data_store is None:
            logger.debug(
                "include_earnings requested but data_store not provided; disabling earnings join",
            )
            include_earnings = False
        self.include_earnings = include_earnings
        self.earnings_lag_days = earnings_lag_days
        self.include_l2 = include_l2
        self.l2_base_dir = l2_base_dir
        self.vintage_base_dir = vintage_base_dir
        self.events_base_dir = events_base_dir
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
        self.macro_revision_windows = macro_revision_windows

        if self.student_mode:
            self.include_macro = False
            self.include_events = False
            self.include_l2 = False
            self.include_earnings = False

        self.windowing_component = TimeSeriesWindowingComponent()
        self.feature_alignment_component = FeatureAlignmentComponent()
        self.target_generation_component = TargetGenerationComponent()
        self.schema_validator_component = TFTSchemaValidatorComponent()

        self._builder: tft_builder.TFTDatasetBuilder | None = None
        if catalog is not None:
            try:
                self._builder = tft_builder.TFTDatasetBuilder(
                    catalog=catalog,
                    symbols=symbols,
                    instrument_ids=instrument_ids,
                    feature_config=feature_config,
                    feature_store=feature_store,
                    data_store=data_store,
                    market_dataset_id=market_dataset_id,
                    market_bindings=self.market_bindings,
                    include_macro=self.include_macro,
                    include_macro_deltas=include_macro_deltas,
                    macro_lag_days=macro_lag_days,
                    fred_path=fred_path,
                    include_micro=include_micro,
                    micro_base_dir=micro_base_dir,
                    include_calendar=include_calendar,
                    include_calendar_lags=include_calendar_lags,
                    include_clustering_tags=include_clustering_tags,
                    include_context_features=include_context_features,
                    include_events=self.include_events,
                    include_earnings=self.include_earnings,
                    earnings_lag_days=earnings_lag_days,
                    include_l2=self.include_l2,
                    l2_base_dir=l2_base_dir,
                    vintage_base_dir=vintage_base_dir,
                    events_base_dir=events_base_dir,
                    student_mode=student_mode,
                    macro_series_ids=macro_series_ids,
                    vintage_policy=vintage_policy,
                    vintage_as_of=self.vintage_as_of,
                    include_macro_revisions=include_macro_revisions,
                    macro_revision_mode=macro_revision_mode,
                    macro_revision_windows=macro_revision_windows,
                )
            except Exception:
                logger.debug(
                    "Failed to initialize TFTDatasetBuilder; falling back to empty datasets",
                    exc_info=True,
                )

    def build_training_dataset(
        self,
        *,
        target_semantics: TargetSemanticsConfig,
        lookback_periods: int = 30,
        use_polars: bool = True,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> DataFrameLike:
        """
        Build a TFT-compatible training dataset.

        Args:
            target_semantics: Explicit target semantics configuration (required).
            lookback_periods: Lookback periods for feature computation.
            use_polars: Whether to return a Polars DataFrame.
            start: Optional start timestamp.
            end: Optional end timestamp.

        Returns:
            TFT-compatible training dataset.
        """
        if self._builder is None:
            return self._empty_dataframe(use_polars)
        try:
            return self._builder.build_training_dataset(
                target_semantics=target_semantics,
                lookback_periods=lookback_periods,
                use_polars=use_polars,
                start=start,
                end=end,
            )
        except Exception:
            logger.debug(
                "TFTDatasetBuilder failed to build dataset; returning empty frame",
                exc_info=True,
            )
            return self._empty_dataframe(use_polars)

    def prepare_training_data(
        self,
        instrument_ids: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        *,
        target_semantics: TargetSemanticsConfig,
        lookback_periods: int = 30,
        use_polars: bool = True,
    ) -> DataFrameLike:
        """
        Prepare training data with automatic source selection.

        Args:
            instrument_ids: Optional instrument IDs to load.
            start: Optional start timestamp.
            end: Optional end timestamp.
            target_semantics: Explicit target semantics configuration (required).
            lookback_periods: Lookback periods for feature computation.
            use_polars: Whether to return a Polars DataFrame.

        Returns:
            TFT-compatible dataset with features and targets.
        """
        if self._builder is None:
            return self._empty_dataframe(use_polars)
        return self._builder.prepare_training_data(
            instrument_ids=instrument_ids,
            start=start,
            end=end,
            target_semantics=target_semantics,
            lookback_periods=lookback_periods,
            use_polars=use_polars,
        )

    def prepare_training_data_from_store(
        self,
        instrument_ids: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        *,
        target_semantics: TargetSemanticsConfig,
    ) -> PolarsDF:
        """
        Prepare training data from the FeatureStore.

        Args:
            instrument_ids: Optional instrument IDs to load.
            start: Optional start timestamp.
            end: Optional end timestamp.
            target_semantics: Explicit target semantics configuration (required).

        Returns:
            Polars DataFrame with training features and targets.
        """
        if self._builder is None:
            msg = "TFTDatasetBuilder not initialized; cannot load from FeatureStore."
            raise ValueError(msg)
        return self._builder.prepare_training_data_from_store(
            instrument_ids=instrument_ids,
            start=start,
            end=end,
            target_semantics=target_semantics,
        )

    def get_binding_stats(self) -> tuple[MarketBindingStats, ...]:
        """
        Return the market binding statistics.

        Returns:
            Tuple of MarketBindingStats entries.
        """
        if self._builder is None:
            return ()
        return self._builder.get_binding_stats()

    def _empty_dataframe(self, use_polars: bool) -> DataFrameLike:
        if use_polars:
            if pl_runtime is None:
                check_ml_dependencies(["polars"])
            pl_module = cast(Any, pl_runtime)
            return cast(DataFrameLike, pl_module.DataFrame())
        if pd_runtime is None:
            check_ml_dependencies(["pandas"])
        pd_module = cast(Any, pd_runtime)
        return cast(DataFrameLike, pd_module.DataFrame())


__all__ = [
    "SchemaValidationError",
    "TFTDatasetBuilderFacade",
]
