"""
TFT Dataset Builder for quick training data preparation.

This module provides a fast path to create TFT-compatible training datasets from
existing collected market data.

"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml._imports import pl
from ml.config.base import MLFeatureConfig
from ml.data.catalog_utils import bars_to_dataframe
from ml.data.providers.utils import cyclic_encode
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


if TYPE_CHECKING:
    from ml.stores.feature_store import FeatureStore


logger = logging.getLogger(__name__)


class TFTDatasetBuilder:
    """
    Fast TFT dataset builder using existing collected data.
    """

    def __init__(
        self,
        catalog: ParquetDataCatalog,
        symbols: list[str],
        feature_config: MLFeatureConfig | None = None,
        feature_store: FeatureStore | None = None,
        *,
        include_macro: bool = False,
        macro_lag_days: int = 1,
        fred_path: str | None = None,
        include_micro: bool = False,
        micro_base_dir: str | None = None,
        include_events: bool = False,
        include_l2: bool = False,
        l2_base_dir: str | None = None,
    ) -> None:
        """
        Initialize TFT dataset builder.

        Parameters
        ----------
        catalog : ParquetDataCatalog
            Nautilus data catalog for accessing market data
        symbols : list[str]
            List of symbols to include in dataset
        feature_config : MLFeatureConfig, optional
            Feature engineering configuration
        feature_store : FeatureStore, optional
            Feature store for reading pre-computed features (ensures training/inference parity)

        """
        self.catalog = catalog
        self.symbols = symbols
        self.feature_config = feature_config or MLFeatureConfig()
        self.feature_store = feature_store
        self.include_macro = include_macro
        self.macro_lag_days = macro_lag_days
        self.fred_path = fred_path
        self.include_micro = include_micro
        self.micro_base_dir = micro_base_dir
        self.include_events = include_events
        self.include_l2 = include_l2
        self.l2_base_dir = l2_base_dir

        logger.info(
            f"Initialized TFTDatasetBuilder with {len(symbols)} symbols "
            f"(FeatureStore: {'enabled' if feature_store else 'disabled'})",
        )

    def prepare_training_data_from_store(
        self,
        instrument_ids: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        horizon_minutes: int = 15,
        min_return_threshold: float = 0.001,
    ) -> pl.DataFrame:
        """
        Prepare training data using features from FeatureStore.

        Ensures training/inference parity by using the same features that are
        computed during live trading.

        Parameters
        ----------
        instrument_ids : list[str], optional
            List of instrument IDs to load features for. If None, uses self.symbols
        start : datetime, optional
            Start time for data loading. If None, loads all available data
        end : datetime, optional
            End time for data loading. If None, loads all available data
        horizon_minutes : int, default 15
            Prediction horizon in minutes for target generation
        min_return_threshold : float, default 0.001
            Minimum return threshold for binary classification (0.1%)

        Returns
        -------
        pl.DataFrame
            TFT-compatible training dataset with features from FeatureStore

        Raises
        ------
        ValueError
            If FeatureStore is not configured
        RuntimeError
            If no features are found in FeatureStore for specified instruments

        """
        if not self.feature_store:
            msg = "FeatureStore not configured. Cannot load features from store."
            raise ValueError(msg)

        if pl is None:
            check_ml_dependencies(["polars"])  # Ensure Polars present when used

        # Use provided instruments or default to configured symbols
        if instrument_ids is None:
            # Convert symbols to instrument_ids with exchange suffix
            instrument_ids = []
            for symbol in self.symbols:
                # Try common exchanges
                for exchange in ["NYSE", "NASDAQ", "ARCA"]:
                    instrument_ids.append(f"{symbol}.{exchange}")

        logger.info(f"Loading features from FeatureStore for {len(instrument_ids)} instruments")

        # Collect all feature data
        all_data: list[pl.DataFrame] = []

        for instrument_id in instrument_ids:
            logger.info(f"Processing {instrument_id} from FeatureStore...")

            try:
                # Load features from FeatureStore
                features, timestamps, feature_names = self.feature_store.get_training_data(
                    instrument_id=instrument_id,
                    start=start or datetime(2020, 1, 1),  # Default start
                    end=end or datetime.now(),  # Default to now
                    include_bars=False,  # We'll load bars separately for targets
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

                # Load corresponding bars for target generation and additional features
                bars_df = bars_to_dataframe(
                    self.catalog,
                    [instrument_id],
                    start=start,
                    end=end,
                )

                if bars_df.is_empty():
                    logger.warning(f"No bar data found for {instrument_id}")
                    continue

                # Align column names and join features with bars
                # bars_to_dataframe provides 'timestamp'; features use 'ts_event'
                bars_df = bars_df.rename({"timestamp": "ts_event"})
                combined_df = bars_df.join(feature_df, on="ts_event", how="inner")

                # Add instrument identifier
                combined_df = combined_df.with_columns(
                    pl.lit(instrument_id).alias("instrument_id"),
                )

                # Generate targets
                targets = self._generate_targets_polars(
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
                dataset = pl.concat(
                    [combined_df, targets],
                    how="horizontal",
                )

                # Add TFT-specific features
                dataset = self._add_static_features_polars(dataset)
                dataset = self._add_known_future_features_polars(dataset)

                # Optionally join microstructure features (per-minute)
                if self.include_micro:
                    try:
                        from pathlib import Path as _Path
                        from ml.features.micro_aggregate import MicrostructureAggregator

                        base_dir = self.micro_base_dir or "data/tier1"
                        agg = MicrostructureAggregator(_Path(base_dir))
                        sym = instrument_id.split(".")[0]
                        micro = agg.compute_for_symbol(sym)
                        if not micro.is_empty():
                            if micro["timestamp"].dtype != pl.Datetime:
                                micro = micro.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
                            dataset = dataset.join(micro, on="timestamp", how="left")
                    except Exception as exc:  # pragma: no cover - best-effort path
                        logger.debug(f"Microstructure join failed for {instrument_id}: {exc}")

                # Optionally join L2 features (per-minute)
                if self.include_l2:
                    try:
                        from pathlib import Path as _Path
                        from ml.features.l2_aggregate import L2Aggregator

                        base_dir = self.l2_base_dir or "data/tier1"
                        agg_l2 = L2Aggregator(_Path(base_dir))
                        sym = instrument_id.split(".")[0]
                        l2 = agg_l2.compute_for_symbol(sym)
                        if not l2.is_empty():
                            if l2["timestamp"].dtype != pl.Datetime:
                                l2 = l2.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
                            dataset = dataset.join(l2, on="timestamp", how="left")
                    except Exception as exc:  # pragma: no cover
                        logger.debug(f"L2 feature join failed for {instrument_id}: {exc}")

                # Optionally add event-based known-future features
                if self.include_events:
                    try:
                        from ml.data.providers.events import EventScheduleProvider
                        from ml.data.sources.events import MockEventSource

                        provider = EventScheduleProvider(MockEventSource())
                        ts_series = dataset.select(pl.col("timestamp").cast(pl.Int64))["timestamp"]
                        ev = provider.compute_features(ts_series, [instrument_id])
                        if not ev.is_empty():
                            ev = ev.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
                            dataset = dataset.join(ev, on="timestamp", how="left")
                    except Exception as exc:  # pragma: no cover - best-effort path
                        logger.debug(f"Event feature join failed for {instrument_id}: {exc}")

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
            from typing import cast

            final_df = cast(
                pl.DataFrame,
                join_fred_asof(
                final_df,
                timestamp_col="ts_event",
                lag_days=self.macro_lag_days,
                fred_path=self.fred_path,
                ),
            )
        logger.info(
            f"Loaded {len(final_df)} rows from FeatureStore with {len(final_df.columns)} columns",
        )

        return final_df

    def prepare_training_data(
        self,
        instrument_ids: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        horizon_minutes: int = 15,
        min_return_threshold: float = 0.001,
        lookback_periods: int = 30,
        use_polars: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Prepare TFT training data with automatic source selection.

        Automatically selects between FeatureStore and direct computation based on
        availability, ensuring optimal performance and training/inference parity.

        Parameters
        ----------
        instrument_ids : list[str], optional
            List of instrument IDs. If None, uses self.symbols with exchange suffixes
        start : datetime, optional
            Start time for data loading
        end : datetime, optional
            End time for data loading
        horizon_minutes : int, default 15
            Prediction horizon in minutes
        min_return_threshold : float, default 0.001
            Minimum return threshold for binary classification
        lookback_periods : int, default 30
            Minimum lookback periods for feature computation (used in direct mode)
        use_polars : bool, default True
            Whether to return Polars DataFrame (True) or pandas DataFrame (False)

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            TFT-compatible training dataset with all required features

        Notes
        -----
        Feature Source Selection:
        - If FeatureStore is configured: Uses pre-computed features (ensures parity)
        - Otherwise: Falls back to direct computation with logging

        The method logs which source was used for monitoring and debugging.

        """
        # Determine which method to use
        if self.feature_store:
            source = "FeatureStore"
            logger.info(
                f"Preparing training data from {source} (ensures training/inference parity)",
            )
            try:
                df = self.prepare_training_data_from_store(
                    instrument_ids=instrument_ids,
                    start=start,
                    end=end,
                    horizon_minutes=horizon_minutes,
                    min_return_threshold=min_return_threshold,
                )

                # Log success with metrics
                logger.info(
                    f"Successfully loaded {len(df)} rows from {source} "
                    f"with {len(df.columns)} features",
                )

                # Convert if needed
                if not use_polars:
                    return df.to_pandas()
                return df

            except Exception as e:
                logger.warning(
                    f"Failed to load from {source}: {e}. " "Falling back to direct computation",
                )
                # Fall through to direct computation

        # Use direct computation
        source = "Direct Computation"
        logger.info(f"Preparing training data using {source}")

        # For direct computation, we use the original method
        direct_df = self._build_training_dataset_direct(
            horizon_minutes=horizon_minutes,
            min_return_threshold=min_return_threshold,
            lookback_periods=lookback_periods,
            use_polars=use_polars,
        )

        # Optionally join macro features
        if self.include_macro:
            from ml.data.fred_join import join_fred_asof

            ts_col = "timestamp" if use_polars else "timestamp"
            direct_df = join_fred_asof(
                direct_df,
                timestamp_col=ts_col,
                lag_days=self.macro_lag_days,
                fred_path=self.fred_path,
            )

        logger.info(
            f"Successfully computed {len(direct_df) if hasattr(direct_df, '__len__') else 'N/A'} rows using {source}",
        )

        return direct_df

    def build_training_dataset(
        self,
        horizon_minutes: int = 15,
        min_return_threshold: float = 0.001,
        lookback_periods: int = 30,
        use_polars: bool = True,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Build complete TFT training dataset.

        This method automatically chooses between FeatureStore and direct computation:
        - If FeatureStore is configured, uses pre-computed features for training/inference parity
        - Otherwise, falls back to direct feature computation

        Parameters
        ----------
        horizon_minutes : int, default 15
            Prediction horizon in minutes
        min_return_threshold : float, default 0.001
            Minimum return threshold for binary classification (0.1%)
        lookback_periods : int, default 30
            Minimum lookback periods for feature computation
        use_polars : bool, default True
            Whether to use Polars for faster processing
        start : datetime, optional
            Start time for data loading
        end : datetime, optional
            End time for data loading

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            TFT-compatible training dataset

        """
        # Check if FeatureStore is available and use it preferentially
        if self.feature_store:
            logger.info("Using FeatureStore for training data preparation (ensures parity)")
            try:
                df = self.prepare_training_data_from_store(
                    instrument_ids=None,  # Will use self.symbols
                    start=start,
                    end=end,
                    horizon_minutes=horizon_minutes,
                    min_return_threshold=min_return_threshold,
                )

                # Convert to pandas if requested
                if not use_polars:
                    return df.to_pandas()
                return df

            except Exception as e:
                logger.warning(
                    f"Failed to load from FeatureStore: {e}. " "Falling back to direct computation",
                )
                # Continue to direct computation below

        # Fall back to direct feature computation
        logger.info("Using direct feature computation for training data")
        return self._build_training_dataset_direct(
            horizon_minutes=horizon_minutes,
            min_return_threshold=min_return_threshold,
            lookback_periods=lookback_periods,
            use_polars=use_polars,
        )

    def _build_training_dataset_direct(
        self,
        horizon_minutes: int = 15,
        min_return_threshold: float = 0.001,
        lookback_periods: int = 30,
        use_polars: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Build training dataset using direct feature computation.

        This is the original implementation, renamed for clarity.

        Parameters
        ----------
        horizon_minutes : int, default 15
            Prediction horizon in minutes
        min_return_threshold : float, default 0.001
            Minimum return threshold for binary classification (0.1%)
        lookback_periods : int, default 30
            Minimum lookback periods for feature computation
        use_polars : bool, default True
            Whether to use Polars for faster processing

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            TFT-compatible training dataset

        """
        # Collect results separately to keep typing precise
        all_data_pl: list[pl.DataFrame] = []
        all_data_pd: list[pd.DataFrame] = []

        for symbol in self.symbols:
            logger.info(f"Processing {symbol}...")

            # Load data using catalog
            try:
                # Assuming symbol needs venue suffix (e.g., NYSE, NASDAQ)
                instrument_id = f"{symbol}.NYSE"  # Default to NYSE, could be configurable
                df = bars_to_dataframe(
                    self.catalog,
                    [instrument_id],
                    start=None,  # Load all available data
                    end=None,
                )

                if df.is_empty():
                    # Try with NASDAQ if NYSE doesn't work
                    instrument_id = f"{symbol}.NASDAQ"
                    df = bars_to_dataframe(
                        self.catalog,
                        [instrument_id],
                        start=None,
                        end=None,
                    )
            except Exception as e:
                logger.warning(f"Failed to load data for {symbol}: {e}")
                continue

            if df.is_empty():
                logger.warning(f"No data found for {symbol}, skipping")
                continue

            # Process with Polars or Pandas
            processed: pl.DataFrame | pd.DataFrame | None = None
            if use_polars:
                if not isinstance(df, pl.DataFrame):
                    logger.warning(f"Expected Polars DataFrame for {symbol}, skipping")
                else:
                    processed = self._process_symbol_polars(
                        df,
                        symbol,
                        horizon_minutes,
                        min_return_threshold,
                        lookback_periods,
                    )
                    if processed is not None:
                        assert isinstance(processed, pl.DataFrame)
                        all_data_pl.append(processed)
            else:
                # Convert to pandas if needed
                if isinstance(df, pl.DataFrame):
                    df_pandas = df.to_pandas()
                else:
                    # Assume already pandas
                    from typing import cast

                    df_pandas = cast(pd.DataFrame, df)
                processed = self._process_symbol_pandas(
                    df_pandas,
                    symbol,
                    horizon_minutes,
                    min_return_threshold,
                    lookback_periods,
                )
                if processed is not None:
                    assert isinstance(processed, pd.DataFrame)
                    all_data_pd.append(processed)

        if (use_polars and not all_data_pl) or (not use_polars and not all_data_pd):
            logger.error("No data processed for any symbol")
            return pd.DataFrame() if not use_polars else pl.DataFrame()

        # Combine all symbols with proper typing
        final_df: pl.DataFrame | pd.DataFrame
        if use_polars:
            # all_data_pl contains Polars DataFrames
            final_df = pl.concat(all_data_pl, how="vertical")
        else:
            # all_data_pd contains Pandas DataFrames
            final_df = pd.concat(all_data_pd, ignore_index=True)

        logger.info(f"Built dataset with shape: {final_df.shape}")

        return final_df

    def _process_symbol_polars(
        self,
        df: pl.DataFrame,
        symbol: str,
        horizon_minutes: int,
        threshold: float,
        lookback_periods: int,
    ) -> pl.DataFrame | None:
        """
        Process single symbol with Polars.
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

        # Generate features
        features = self._compute_features_polars(df)

        # Generate targets
        targets = self._generate_targets_polars(df, horizon_minutes, threshold)

        # Combine (retain timestamp for macro joins)
        dataset = pl.concat(
            [
                df.select(["timestamp", "time_index", "instrument_id"]),
                features,
                targets,
            ],
            how="horizontal",
        )

        # Filter for sufficient history
        dataset = dataset.slice(lookback_periods, len(dataset))

        # Add static and known-future features
        dataset = self._add_static_features_polars(dataset)
        dataset = self._add_known_future_features_polars(dataset)

        # Optionally join microstructure features (per-minute)
        if self.include_micro:
            try:
                from pathlib import Path as _Path
                from ml.features.micro_aggregate import MicrostructureAggregator

                base_dir = self.micro_base_dir or "data/tier1"
                agg = MicrostructureAggregator(_Path(base_dir))
                micro = agg.compute_for_symbol(symbol)
                if not micro.is_empty():
                    if micro["timestamp"].dtype != pl.Datetime:
                        micro = micro.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
                    dataset = dataset.join(micro, on="timestamp", how="left")
            except Exception as exc:  # pragma: no cover
                logger.debug(f"Microstructure join failed for {symbol}: {exc}")

        # Optionally join L2 features (per-minute)
        if self.include_l2:
            try:
                from pathlib import Path as _Path
                from ml.features.l2_aggregate import L2Aggregator

                base_dir = self.l2_base_dir or "data/tier1"
                agg_l2 = L2Aggregator(_Path(base_dir))
                l2 = agg_l2.compute_for_symbol(symbol)
                if not l2.is_empty():
                    if l2["timestamp"].dtype != pl.Datetime:
                        l2 = l2.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
                    dataset = dataset.join(l2, on="timestamp", how="left")
            except Exception as exc:  # pragma: no cover
                logger.debug(f"L2 feature join failed for {symbol}: {exc}")

        # Optionally add event-based known-future features
        if self.include_events:
            try:
                from ml.data.providers.events import EventScheduleProvider
                from ml.data.sources.events import MockEventSource

                provider = EventScheduleProvider(MockEventSource())
                ts_series = dataset.select(pl.col("timestamp").cast(pl.Int64))["timestamp"]
                ev = provider.compute_features(ts_series, [symbol])
                if not ev.is_empty():
                    ev = ev.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
                    dataset = dataset.join(ev, on="timestamp", how="left")
            except Exception as exc:  # pragma: no cover
                logger.debug(f"Event feature join failed for {symbol}: {exc}")

        return dataset

    def _process_symbol_pandas(
        self,
        df: pd.DataFrame,
        symbol: str,
        horizon_minutes: int,
        threshold: float,
        lookback_periods: int,
    ) -> pd.DataFrame | None:
        """
        Process single symbol with Pandas.
        """
        if pd is None:
            check_ml_dependencies(["pandas"])  # Ensure pandas present when used
        # Ensure we have required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns for {symbol}")
            return None

        # Sort by index (timestamps) and create sequential index
        df = df.sort_index().reset_index(drop=False)
        # Rename the index to timestamp if it's called ts_event
        if "ts_event" in df.columns:
            df = df.rename(columns={"ts_event": "timestamp"})
        elif df.index.name == "ts_event":
            df = df.reset_index().rename(columns={"ts_event": "timestamp"})

        df["time_index"] = range(len(df))
        df["instrument_id"] = symbol

        # Generate features
        features = self._compute_features_pandas(df)

        # Generate targets
        targets = self._generate_targets_pandas(df, horizon_minutes, threshold)

        # Combine (retain timestamp for macro joins)
        dataset = pd.concat(
            [
                df[["timestamp", "time_index", "instrument_id"]],
                features,
                targets,
            ],
            axis=1,
        )

        # Filter for sufficient history
        dataset = dataset.iloc[lookback_periods:].copy()

        # Add static and known-future features
        dataset = self._add_static_features_pandas(dataset)
        dataset = self._add_known_future_features_pandas(dataset)

        # Optionally join microstructure features (per-minute)
        if self.include_micro:
            try:
                from pathlib import Path as _Path
                from ml.features.micro_aggregate import MicrostructureAggregator

                base_dir = self.micro_base_dir or "data/tier1"
                agg = MicrostructureAggregator(_Path(base_dir))
                micro_pl = agg.compute_for_symbol(symbol)
                if micro_pl.shape[0] > 0:
                    micro_pd = micro_pl.to_pandas()
                    dataset = dataset.merge(micro_pd, on="timestamp", how="left")
            except Exception as exc:  # pragma: no cover
                logger.debug(f"Microstructure join failed for {symbol}: {exc}")

        # Optionally join L2 features (per-minute)
        if self.include_l2:
            try:
                from pathlib import Path as _Path
                from ml.features.l2_aggregate import L2Aggregator

                base_dir = self.l2_base_dir or "data/tier1"
                agg_l2 = L2Aggregator(_Path(base_dir))
                l2_pl = agg_l2.compute_for_symbol(symbol)
                if l2_pl.shape[0] > 0:
                    l2_pl = l2_pl.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
                    l2_pd = l2_pl.to_pandas()
                    dataset = dataset.merge(l2_pd, on="timestamp", how="left")
            except Exception as exc:  # pragma: no cover
                logger.debug(f"L2 feature join failed for {symbol}: {exc}")

        # Optionally add event-based known-future features
        if self.include_events:
            try:
                from ml.data.providers.events import EventScheduleProvider
                from ml.data.sources.events import MockEventSource

                provider = EventScheduleProvider(MockEventSource())
                if "timestamp" in dataset.columns:
                    ts_series = pl.Series("timestamp", dataset["timestamp"].astype("datetime64[ns]")).cast(pl.Int64)
                    ev_pl = provider.compute_features(ts_series, [symbol])
                    ev_pl = ev_pl.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
                    ev_pd = ev_pl.to_pandas()
                    dataset = dataset.merge(ev_pd, on="timestamp", how="left")
            except Exception as exc:  # pragma: no cover
                logger.debug(f"Event feature join failed for {symbol}: {exc}")

        return dataset

    def _compute_features_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute technical features using Polars.
        """
        base = df.with_columns(
            [
                (pl.col("close") / pl.col("close").shift(1) - 1).alias("return_1"),
                (pl.col("close") / pl.col("close").shift(5) - 1).alias("return_5"),
                (pl.col("close") / pl.col("close").shift(20) - 1).alias("return_20"),
                (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias("volume_ratio"),
                pl.col("close").rolling_mean(5).alias("sma_5"),
                pl.col("close").rolling_mean(20).alias("sma_20"),
                (
                    (pl.col("close") - pl.col("low").rolling_min(20))
                    / (pl.col("high").rolling_max(20) - pl.col("low").rolling_min(20))
                ).alias("price_position"),
            ],
        )
        features = base.select(
            [
                "return_1",
                "return_5",
                "return_20",
                "volume_ratio",
                pl.col("return_1").rolling_std(20).alias("volatility_20"),
                "sma_5",
                "sma_20",
                "price_position",
            ],
        ).fill_null(0)
        return features

    def _compute_features_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical features using Pandas.
        """
        features = pd.DataFrame()

        # Price-based features
        features["return_1"] = df["close"].pct_change(1)
        features["return_5"] = df["close"].pct_change(5)
        features["return_20"] = df["close"].pct_change(20)

        # Volume features
        features["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

        # Volatility
        features["volatility_20"] = features["return_1"].rolling(20).std()

        # Simple moving averages
        features["sma_5"] = df["close"].rolling(5).mean()
        features["sma_20"] = df["close"].rolling(20).mean()

        # Price position
        rolling_min = df["low"].rolling(20).min()
        rolling_max = df["high"].rolling(20).max()
        features["price_position"] = (df["close"] - rolling_min) / (rolling_max - rolling_min)

        # Fill NaN values
        features = features.fillna(0)

        return features

    def _generate_targets_polars(
        self,
        df: pl.DataFrame,
        horizon_minutes: int,
        threshold: float,
    ) -> pl.DataFrame:
        """
        Generate binary targets using Polars.
        """
        # Calculate forward returns
        future_prices = pl.col("close").shift(-horizon_minutes)
        current_prices = pl.col("close")
        forward_returns = (future_prices - current_prices) / current_prices

        # Binary classification
        targets = df.select(
            [
                (forward_returns > threshold).cast(pl.Int32).alias("y"),
            ],
        )

        # Fill NaN at the end
        targets = targets.fill_null(0)

        return targets

    def _generate_targets_pandas(
        self,
        df: pd.DataFrame,
        horizon_minutes: int,
        threshold: float,
    ) -> pd.DataFrame:
        """
        Generate binary targets using Pandas.
        """
        # Calculate forward returns
        future_prices = df["close"].shift(-horizon_minutes)
        current_prices = df["close"]
        forward_returns = (future_prices - current_prices) / current_prices

        # Binary classification
        targets = pd.DataFrame(
            {
                "y": (forward_returns > threshold).astype(int),
            },
        )

        # Fill NaN at the end
        targets = targets.fillna(0)

        return targets

    def _add_static_features_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add static instrument features using Polars.
        """
        # Simple static feature mapping
        static_map = {
            "SPY": {"asset_class": "ETF", "tick_size": 0.01, "exchange": "ARCA"},
            "QQQ": {"asset_class": "ETF", "tick_size": 0.01, "exchange": "NASDAQ"},
            "AAPL": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
            "MSFT": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
            "NVDA": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
            "AMZN": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
            "META": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
            "GOOGL": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
            "TSLA": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
        }

        # Get unique instruments
        instruments = df["instrument_id"].unique().to_list()

        # Add static features for each instrument
        for instrument in instruments:
            static = static_map.get(
                instrument,
                {
                    "asset_class": "STOCK",
                    "tick_size": 0.01,
                    "exchange": "UNKNOWN",
                },
            )

            df = df.with_columns(
                [
                    pl.when(pl.col("instrument_id") == instrument)
                    .then(pl.lit(static["asset_class"]))
                    .otherwise(
                        pl.col("asset_class") if "asset_class" in df.columns else pl.lit("UNKNOWN"),
                    )
                    .alias("asset_class"),
                    pl.when(pl.col("instrument_id") == instrument)
                    .then(pl.lit(static["tick_size"]))
                    .otherwise(pl.col("tick_size") if "tick_size" in df.columns else pl.lit(0.01))
                    .alias("tick_size"),
                    pl.when(pl.col("instrument_id") == instrument)
                    .then(pl.lit(static["exchange"]))
                    .otherwise(
                        pl.col("exchange") if "exchange" in df.columns else pl.lit("UNKNOWN"),
                    )
                    .alias("exchange"),
                ],
            )

        return df

    def _add_static_features_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add static instrument features using Pandas.
        """
        # Simple static feature mapping
        static_map = {
            "SPY": {"asset_class": "ETF", "tick_size": 0.01, "exchange": "ARCA"},
            "QQQ": {"asset_class": "ETF", "tick_size": 0.01, "exchange": "NASDAQ"},
            "AAPL": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
            "MSFT": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
            "NVDA": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
            "AMZN": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
            "META": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
            "GOOGL": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
            "TSLA": {"asset_class": "STOCK", "tick_size": 0.01, "exchange": "NASDAQ"},
        }

        # Add static features
        for col in ["asset_class", "tick_size", "exchange"]:
            df[col] = df["instrument_id"].map(
                lambda x: static_map.get(
                    x,
                    {
                        "asset_class": "STOCK",
                        "tick_size": 0.01,
                        "exchange": "UNKNOWN",
                    },
                ).get(col),
            )

        return df

    def _add_known_future_features_polars(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add known-future time features using Polars.
        """
        # Create hour and minute from time_index (assuming minute bars)
        df = df.with_columns(
            [
                ((pl.col("time_index") // 60) % 24).alias("hour"),
                (pl.col("time_index") % 60).alias("minute"),
            ],
        )

        # Time of day features (cyclical encoding)
        df = df.with_columns(
            [
                (2 * np.pi * (pl.col("hour") * 60 + pl.col("minute")) / (24 * 60))
                .sin()
                .alias("tod_sin"),
                (2 * np.pi * (pl.col("hour") * 60 + pl.col("minute")) / (24 * 60))
                .cos()
                .alias("tod_cos"),
            ],
        )

        # Day of week (simplified - assuming continuous trading for now)
        df = df.with_columns(
            [
                ((pl.col("time_index") // (24 * 60)) % 7).alias("dow"),
            ],
        )

        df = df.with_columns(
            [
                (2 * np.pi * pl.col("dow") / 7).sin().alias("dow_sin"),
                (2 * np.pi * pl.col("dow") / 7).cos().alias("dow_cos"),
            ],
        )

        # Market session flags
        df = df.with_columns(
            [
                ((pl.col("hour") >= 9) & (pl.col("hour") < 16))
                .cast(pl.Int32)
                .alias("is_market_open"),
                ((pl.col("hour") >= 4) & (pl.col("hour") < 9)).cast(pl.Int32).alias("is_premarket"),
                ((pl.col("hour") >= 16) & (pl.col("hour") < 20))
                .cast(pl.Int32)
                .alias("is_aftermarket"),
            ],
        )

        return df

    def _add_known_future_features_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add known-future time features using Pandas.
        """
        # Create hour and minute from time_index (assuming minute bars)
        df["hour"] = (df["time_index"] // 60) % 24
        df["minute"] = df["time_index"] % 60

        # Time of day features (cyclical encoding)
        time_in_minutes = df["hour"] * 60 + df["minute"]
        # Use centralized cyclic_encode for clarity and DRY
        sincos = time_in_minutes.apply(lambda v: cyclic_encode(float(v), 24 * 60))
        df["tod_sin"] = sincos.apply(lambda t: t[0])
        df["tod_cos"] = sincos.apply(lambda t: t[1])

        # Day of week (simplified - assuming continuous trading for now)
        df["dow"] = (df["time_index"] // (24 * 60)) % 7
        # Day-of-week cyclic encoding via centralized helper
        dowsc = df["dow"].apply(lambda d: cyclic_encode(float(d), 7))
        df["dow_sin"] = dowsc.apply(lambda t: t[0])
        df["dow_cos"] = dowsc.apply(lambda t: t[1])

        # Market session flags
        df["is_market_open"] = ((df["hour"] >= 9) & (df["hour"] < 16)).astype(int)
        df["is_premarket"] = ((df["hour"] >= 4) & (df["hour"] < 9)).astype(int)
        df["is_aftermarket"] = ((df["hour"] >= 16) & (df["hour"] < 20)).astype(int)

        return df
