"""
TFT Dataset Builder for quick training data preparation.

This module provides a fast path to create TFT-compatible training datasets from
existing collected market data.

"""

from __future__ import annotations

import logging
from datetime import UTC
from datetime import datetime
from pathlib import Path as _Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ml._imports import check_ml_dependencies
from ml._imports import pd as pd_runtime
from ml._imports import pl as pl_runtime
from ml.config.base import MLFeatureConfig
from ml.data.catalog_utils import bars_to_dataframe
from ml.data.l2_cache import L2MinuteCache
from ml.data.micro_cache import MicroMinuteCache
from ml.data.providers.utils import cyclic_encode
from ml.data.vintage import VintagePolicy
from ml.stores.feature_store import FeatureStore
from ml.stores.protocols import DataStoreFacadeProtocol
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


if TYPE_CHECKING:
    import pandas as _pd
    import polars as _pl
    from pandas import DataFrame as PandasDataFrame
else:  # pragma: no cover - typing fallback
    PandasDataFrame = Any

# Local runtime aliases to avoid Optional[Module] union typing
PL: Any = cast(Any, pl_runtime)
PD: Any = cast(Any, pd_runtime)
# Runtime aliases to maintain existing symbol names for implementations
pl = PL
pd = PD


logger = logging.getLogger(__name__)


class TFTDatasetBuilder:
    """
    Fast TFT dataset builder using existing collected data.
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
        include_macro: bool = False,
        macro_lag_days: int = 1,
        fred_path: str | None = None,
        include_micro: bool = False,
        micro_base_dir: str | None = None,
        include_calendar: bool = False,
        include_events: bool = False,
        include_l2: bool = False,
        l2_base_dir: str | None = None,
        vintage_base_dir: str | _Path | None = None,
        events_base_dir: str | _Path | None = None,
        student_mode: bool = False,
        macro_series_ids: tuple[str, ...] | None = None,
        vintage_policy: VintagePolicy = VintagePolicy.REAL_TIME,
        vintage_as_of: datetime | None = None,
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
        data_store : DataStoreFacadeProtocol, optional
            Canonical DataStore for reading raw market data. When provided, the builder
            reads OHLCV from the store and only falls back to the catalog or on-disk
            parquet files when the store lacks data.
        market_dataset_id : str | None, optional
            Dataset identifier registered in the DataRegistry that represents the raw
            market data. Required when `data_store` is provided.

        """
        self.catalog = catalog
        self._original_symbols = symbols
        self.symbols = [sym.split(".")[0] for sym in symbols]
        self.instrument_ids = instrument_ids
        self._symbol_instrument_map: dict[str, list[str]] = {}
        if instrument_ids:
            for inst in instrument_ids:
                base = inst.split(".")[0]
                self._symbol_instrument_map.setdefault(base, []).append(inst)
                self._symbol_instrument_map.setdefault(inst, []).append(inst)
        self.feature_config = feature_config or MLFeatureConfig()
        self.feature_store = feature_store
        self.data_store = data_store
        self.market_dataset_id = market_dataset_id
        self.include_macro = include_macro
        self.macro_lag_days = macro_lag_days
        self.fred_path = fred_path
        self.include_micro = include_micro
        self.micro_base_dir = micro_base_dir
        self.include_calendar = include_calendar
        self.include_events = include_events
        self.include_l2 = include_l2
        self.l2_base_dir = l2_base_dir
        self.vintage_base_dir = _Path(vintage_base_dir).expanduser() if vintage_base_dir else None
        self.events_base_dir = (
            _Path(events_base_dir).expanduser() if events_base_dir else _Path("data/events")
        )
        self._event_provider: Any | None = None
        self.student_mode = student_mode
        self.macro_series_ids = macro_series_ids
        self.vintage_policy = vintage_policy
        if vintage_as_of is None:
            self.vintage_as_of = None
        elif vintage_as_of.tzinfo is None:
            self.vintage_as_of = vintage_as_of.replace(tzinfo=UTC)
        else:
            self.vintage_as_of = vintage_as_of.astimezone(UTC)

        if self.student_mode:
            self.include_macro = False
            self.include_events = False
            self.include_l2 = False

        logger.info(
            f"Initialized TFTDatasetBuilder with {len(symbols)} symbols "
            f"(FeatureStore: {'enabled' if feature_store else 'disabled'})",
        )

    def _store_enabled(self) -> bool:
        return self.data_store is not None and bool(self.market_dataset_id)

    @staticmethod
    def _datetime_to_ns(value: datetime | None, *, fallback: int) -> int:
        if value is None:
            return fallback
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        else:
            value = value.astimezone(UTC)
        return int(value.timestamp() * 1_000_000_000)

    @staticmethod
    def _now_ns() -> int:
        return int(datetime.now(tz=UTC).timestamp() * 1_000_000_000)

    def _to_polars(self, data: Any) -> _pl.DataFrame:
        if pl is not None and isinstance(data, pl.DataFrame):
            return data
        if pd is not None and isinstance(data, pd.DataFrame):
            return pl.from_pandas(data)
        raise TypeError(f"Unsupported data type for Polars conversion: {type(data)!r}")

    def _load_bars_dataframe(
        self,
        instrument_id: str,
        start: datetime | None,
        end: datetime | None,
    ) -> _pl.DataFrame:
        if self._store_enabled():
            try:
                start_ns = self._datetime_to_ns(start, fallback=0)
                end_ns = self._datetime_to_ns(end, fallback=self._now_ns())
                assert self.data_store is not None
                dataset_id = cast(str, self.market_dataset_id)
                raw = self.data_store.read_range(
                    dataset_id=dataset_id,
                    instrument_id=instrument_id,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )
                frame = self._to_polars(raw)
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
                        )
                        if col in frame.columns
                    ]
                    frame = frame.select(keep)
                    return frame
            except Exception:
                logger.warning(
                    "DataStore read_range failed; falling back to catalog",
                    exc_info=True,
                    extra={"instrument_id": instrument_id},
                )

        try:
            return bars_to_dataframe(
                self.catalog,
                [instrument_id],
                start=start,
                end=end,
            )
        except Exception:  # pragma: no cover - catalog fallback path
            logger.debug(
                "Parquet catalog read failed",
                exc_info=True,
                extra={"instrument_id": instrument_id},
            )
            return pl.DataFrame(
                {
                    "instrument_id": [],
                    "timestamp": [],
                    "open": [],
                    "high": [],
                    "low": [],
                    "close": [],
                    "volume": [],
                },
            )

    def _resolve_instrument_ids(self, override: list[str] | None = None) -> list[str]:
        if override:
            return override
        if self.instrument_ids:
            return self.instrument_ids
        candidates: list[str] = []
        for symbol in self.symbols:
            if "." in symbol:
                candidates.append(symbol)
                continue
            for exchange in ["NYSE", "NASDAQ", "ARCA", "ARCX", "XNAS", "XNYS"]:
                candidates.append(f"{symbol}.{exchange}")
        if candidates:
            logger.warning(
                "Instrument IDs not provided; falling back to heuristic venues %s",
                ["NYSE", "NASDAQ", "ARCA", "ARCX", "XNAS", "XNYS"],
            )
        return candidates

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

        if pl_runtime is None:
            check_ml_dependencies(["polars"])  # Ensure Polars present when used

        # Use provided instruments or default to configured symbols
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

        for instrument_id in resolved_ids:
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
                from typing import cast as _cast

                feature_df = _cast(
                    "_pl.DataFrame",
                    pl.DataFrame(
                        {
                            "ts_event": timestamps,
                            **{name: features[:, i] for i, name in enumerate(feature_names)},
                        },
                    ),
                )

                # Load corresponding bars for target generation and additional features
                bars_df = self._load_bars_dataframe(
                    instrument_id,
                    start,
                    end,
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
                dataset = pl.concat([combined_df, targets], how="horizontal")

                # Add TFT-specific features
                dataset = self._add_static_features_polars(dataset)
                dataset = self._add_known_future_features_polars(dataset)

                # Append dataset for this instrument
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
            from typing import cast

            from ml.data.fred_join import join_fred_asof

            final_df = cast(
                "_pl.DataFrame",
                join_fred_asof(
                    final_df,
                    timestamp_col="ts_event",
                    lag_days=self.macro_lag_days,
                    fred_path=self.fred_path,
                    vintage_base_dir=self.vintage_base_dir,
                    series_filter=None if self.macro_series_ids is None else set(self.macro_series_ids),
                    vintage_policy=self.vintage_policy,
                    vintage_cutoff=self.vintage_as_of,
                ),
            )
        logger.info(
            f"Loaded {len(final_df)} rows from FeatureStore with {len(final_df.columns)} columns",
        )
        from typing import cast as _cast

        return _cast("_pl.DataFrame", final_df)

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

            ts_col = "timestamp"
            if pl is not None and isinstance(direct_df, pl.DataFrame):
                before_cols = set(direct_df.columns)
                direct_df = join_fred_asof(
                    direct_df,
                    timestamp_col=ts_col,
                    lag_days=self.macro_lag_days,
                    fred_path=self.fred_path,
                    vintage_base_dir=self.vintage_base_dir,
                    series_filter=None if self.macro_series_ids is None else set(self.macro_series_ids),
                    vintage_policy=self.vintage_policy,
                    vintage_cutoff=self.vintage_as_of,
                )
                macro_cols = [c for c in direct_df.columns if c not in before_cols]
                if macro_cols:
                    # Fill macro nulls with 0 and add availability mask
                    from typing import Any as _Any

                    fills: list[_Any] = []
                    exprs: list[_Any] = []
                    for c in macro_cols:
                        try:
                            if direct_df.schema[c].is_numeric():
                                fills.append(pl.col(c).fill_null(0))
                        except Exception:
                            pass
                        exprs.append(pl.col(c).is_not_null())
                    if fills:
                        # Narrow type for polars operations
                        from typing import cast as _cast

                        direct_df_pl = _cast("_pl.DataFrame", direct_df)
                        direct_df_pl = direct_df_pl.with_columns(fills)
                        direct_df = direct_df_pl
                    if exprs:
                        any_macro: _Any = exprs[0]
                        for ex in exprs[1:]:
                            any_macro = any_macro | ex
                        from typing import cast as _cast

                        direct_df_pl2 = _cast("_pl.DataFrame", direct_df)
                        direct_df_pl2 = direct_df_pl2.with_columns(
                            [
                                any_macro.cast(pl.Int32).alias("is_macro_available"),
                            ],
                        )
                        direct_df = direct_df_pl2
            else:
                # Pandas path — apply join and compute mask with pandas ops
                direct_df = join_fred_asof(
                    direct_df,
                    timestamp_col=ts_col,
                    lag_days=self.macro_lag_days,
                    fred_path=self.fred_path,
                    vintage_base_dir=self.vintage_base_dir,
                    series_filter=None if self.macro_series_ids is None else set(self.macro_series_ids),
                    vintage_policy=self.vintage_policy,
                    vintage_cutoff=self.vintage_as_of,
                )
                try:  # pragma: no cover
                    import pandas as _pd

                    if isinstance(direct_df, _pd.DataFrame):
                        # Assume all newly added columns after join are macro columns
                        # We compute diff by comparing to columns of a no-op slice (cannot easily capture before)
                        # Fallback: treat all non-core columns except known as potentially macro and fill nulls
                        core = {"timestamp", "time_index", "instrument_id", "y"}
                        macro_cols = [c for c in direct_df.columns if c not in core]
                        direct_df[macro_cols] = direct_df[macro_cols].fillna(0)
                        direct_df["is_macro_available"] = (
                            direct_df[macro_cols].notna().any(axis=1).astype("int32")
                        )
                except Exception:
                    pass

        logger.info(
            f"Successfully computed {len(direct_df) if hasattr(direct_df, '__len__') else 'N/A'} rows using {source}",
        )

        return direct_df

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
        # Backwards-compat for tests: support threshold_bps alias
        if threshold_bps is not None:
            # Convert basis points to decimal if seems large; else use as-is
            min_return_threshold = threshold_bps / 10_000.0 if threshold_bps > 1 else threshold_bps

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
            start=start,
            end=end,
        )

    def _get_event_provider(self) -> Any | None:
        """
        Lazily initialize the event schedule provider.
        """
        if not self.include_events:
            return None
        if self._event_provider is not None:
            return self._event_provider

        try:
            from ml.data.providers.events import EventScheduleProvider
            from ml.data.sources.events import FileEventSource
            from ml.data.sources.events import SimpleEventSource

            events_path = None
            if self.events_base_dir is not None:
                candidate = self.events_base_dir / "events.parquet"
                if candidate.exists():
                    events_path = candidate

            source = FileEventSource(events_path) if events_path else SimpleEventSource()
            self._event_provider = EventScheduleProvider(source)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Event provider initialization failed: %s", exc, exc_info=True)
            self._event_provider = None
        return self._event_provider

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
                from typing import cast as _cast

                df = _cast("_pl.DataFrame", pl.DataFrame())
                last_err: Exception | None = None
                instrument_candidates = self._symbol_instrument_map.get(symbol)
                if not instrument_candidates:
                    instrument_candidates = [f"{symbol}.{venue}" for venue in candidate_venues]
                for instrument_id in instrument_candidates:
                    try:
                        df = self._load_bars_dataframe(instrument_id, start, end)
                        if not df.is_empty():
                            break
                    except Exception as e_inner:  # pragma: no cover - catalog differences
                        last_err = e_inner
                        continue
                if df.is_empty():
                    # Fallback: read OHLCV minute parquet directly under base dir
                    try:
                        base = _Path(self.micro_base_dir or "data/tier1")
                        paths = [
                            base / symbol / "ohlcv-1m_historical.parquet",
                            base / symbol / "ohlcv-1m_recent.parquet",
                        ]
                        frames = []
                        for p in paths:
                            if p.exists():
                                part = pl.read_parquet(str(p))
                                if not part.is_empty():
                                    # Standardize columns to OHLCV schema
                                    if (
                                        "timestamp" not in part.columns
                                        and "ts_event" in part.columns
                                    ):
                                        part = part.rename({"ts_event": "timestamp"})
                                    keep = [
                                        c
                                        for c in [
                                            "timestamp",
                                            "open",
                                            "high",
                                            "low",
                                            "close",
                                            "volume",
                                        ]
                                        if c in part.columns
                                    ]
                                    if keep:
                                        part = part.select(keep)
                                        # Unify timestamp timezone to UTC for concat compatibility
                                        if "timestamp" in part.columns:
                                            try:
                                                part = part.with_columns(
                                                    pl.col("timestamp").dt.replace_time_zone("UTC"),
                                                )
                                            except Exception:
                                                try:
                                                    part = part.with_columns(
                                                        pl.col("timestamp").dt.convert_time_zone(
                                                            "UTC",
                                                        ),
                                                    )
                                                except Exception:
                                                    try:
                                                        part = part.with_columns(
                                                            pl.col("timestamp").cast(
                                                                pl.Datetime("ns", "UTC"),
                                                            ),
                                                        )
                                                    except Exception:
                                                        pass
                                    frames.append(part)
                        # Also support files produced by populate_universe: data/tier1/<SYMBOL>/l0/<SYMBOL>_ohlcv.parquet
                        l0_file = base / symbol / "l0" / f"{symbol}_ohlcv.parquet"
                        if l0_file.exists():
                            part = pl.read_parquet(str(l0_file))
                            if not part.is_empty():
                                # Standardize columns
                                if "timestamp" not in part.columns and "ts_event" in part.columns:
                                    part = part.rename({"ts_event": "timestamp"})
                                keep = [
                                    c
                                    for c in ["timestamp", "open", "high", "low", "close", "volume"]
                                    if c in part.columns
                                ]
                                part = part.select(keep)
                                # Unify timestamp timezone to UTC for concat compatibility
                                if "timestamp" in part.columns:
                                    try:
                                        part = part.with_columns(
                                            pl.col("timestamp").dt.replace_time_zone("UTC"),
                                        )
                                    except Exception:
                                        try:
                                            part = part.with_columns(
                                                pl.col("timestamp").dt.convert_time_zone("UTC"),
                                            )
                                        except Exception:
                                            try:
                                                part = part.with_columns(
                                                    pl.col("timestamp").cast(
                                                        pl.Datetime("ns", "UTC"),
                                                    ),
                                                )
                                            except Exception:
                                                pass
                                frames.append(part)
                        if frames:
                            # Concatenate standardized frames
                            df = pl.concat(frames, how="vertical")
                            # Normalize timestamp column name and type
                            if "timestamp" not in df.columns and "ts_event" in df.columns:
                                df = df.rename({"ts_event": "timestamp"})
                            if "timestamp" in df.columns and df["timestamp"].dtype != pl.Datetime:
                                df = df.with_columns(
                                    pl.col("timestamp").cast(pl.Datetime("ns", "UTC")),
                                )
                            df = df.sort(
                                "timestamp" if "timestamp" in df.columns else df.columns[0],
                            )
                        else:
                            # No files found; log and skip
                            if last_err is not None:
                                logger.warning(
                                    f"Failed to load data for {symbol} (last error: {last_err}); no parquet fallback",
                                )
                            else:
                                logger.warning(
                                    f"No data found for {symbol} across venues {candidate_venues}; parquet fallback missing",
                                )
                            continue
                    except Exception as e_fallback:  # pragma: no cover - environment dependent
                        logger.warning(f"Fallback parquet load failed for {symbol}: {e_fallback}")
                        continue
            except Exception as e:
                logger.warning(f"Failed to load data for {symbol}: {e}")
                continue

            if df.is_empty():
                logger.warning(f"No data found for {symbol}, skipping")
                continue

            # Process with Polars or Pandas
            processed: _pl.DataFrame | _pd.DataFrame | None = None
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
                        start=start,
                        end=end,
                    )
                    if processed is not None:
                        # Optional date filtering
                        if (
                            start is not None or end is not None
                        ) and "timestamp" in processed.columns:
                            ts = pl.col("timestamp").cast(pl.Int64)
                            cond = pl.lit(True)
                            if start is not None:
                                start_ns = int(start.timestamp() * 1_000_000_000)
                                cond = cond & (ts >= start_ns)
                            if end is not None:
                                end_ns = int(end.timestamp() * 1_000_000_000)
                                cond = cond & (ts < end_ns)
                            processed = processed.filter(cond)
                        # processed is a Polars DataFrame here
                        all_data_pl.append(processed)
            else:
                # Convert to pandas if needed
                if isinstance(df, pl.DataFrame):
                    df_pandas = df.to_pandas()
                else:
                    # Assume already pandas
                    from typing import cast

                    df_pandas = cast("_pd.DataFrame", df)
                processed = self._process_symbol_pandas(
                    df_pandas,
                    symbol,
                    horizon_minutes,
                    min_return_threshold,
                    lookback_periods,
                )
                if processed is not None:
                    # Optional date filtering
                    if (start is not None or end is not None) and "timestamp" in processed.columns:
                        try:
                            import pandas as _pd

                            s_ts = _pd.Timestamp(start, tz="UTC") if start is not None else None
                            e_ts = _pd.Timestamp(end, tz="UTC") if end is not None else None
                            if s_ts is not None:
                                processed = processed[processed["timestamp"] >= s_ts]
                            if e_ts is not None:
                                processed = processed[processed["timestamp"] < e_ts]
                        except Exception:
                            pass
                    if processed is not None:
                        all_data_pd.append(processed)

        if (use_polars and not all_data_pl) or (not use_polars and not all_data_pd):
            logger.error("No data processed for any symbol")
            from typing import cast as _cast

            if not use_polars:
                return _cast("_pd.DataFrame", pd.DataFrame())
            return _cast("_pl.DataFrame", pl.DataFrame())

        # Combine all symbols with proper typing
        final_df: _pl.DataFrame | _pd.DataFrame
        if use_polars:
            lazy_frames = [df.lazy() for df in all_data_pl]
            final_df = pl.concat(lazy_frames).collect(streaming=True)
            all_data_pl.clear()

            if self.include_macro:
                from ml.data.fred_join import join_fred_asof

                base_cols = set(final_df.columns)
                joined = _cast(
                    "_pl.DataFrame",
                    join_fred_asof(
                        final_df,
                        timestamp_col="timestamp",
                        lag_days=self.macro_lag_days,
                        fred_path=self.fred_path,
                        vintage_base_dir=self.vintage_base_dir,
                        series_filter=None if self.macro_series_ids is None else set(self.macro_series_ids),
                        vintage_policy=self.vintage_policy,
                        vintage_cutoff=self.vintage_as_of,
                    ),
                )
                macro_cols = [
                    col
                    for col in joined.columns
                    if col not in base_cols and col != "timestamp_right"
                ]
                if "timestamp_right" in joined.columns:
                    joined = joined.drop("timestamp_right")
                if macro_cols:
                    df_pd = joined.to_pandas()
                    df_pd = df_pd.fillna(dict.fromkeys(macro_cols, 0.0))
                    df_pd["is_macro_available"] = (
                        df_pd[macro_cols].notna().any(axis=1).astype("int32")
                    )
                    joined = pl.from_pandas(df_pd)
                final_df = joined
        else:
            # all_data_pd contains Pandas DataFrames
            final_df = pd.concat(all_data_pd, ignore_index=True)
            if self.include_macro:
                from ml.data.fred_join import join_fred_asof

                base_cols = set(final_df.columns)
                final_df_pd: PandasDataFrame = cast(PandasDataFrame, final_df)
                final_df_pd = cast(
                    PandasDataFrame,
                    join_fred_asof(
                        final_df_pd,
                        timestamp_col="timestamp",
                        lag_days=self.macro_lag_days,
                        fred_path=self.fred_path,
                        vintage_base_dir=self.vintage_base_dir,
                        series_filter=None if self.macro_series_ids is None else set(self.macro_series_ids),
                        vintage_policy=self.vintage_policy,
                        vintage_cutoff=self.vintage_as_of,
                    ),
                )
                macro_cols = [
                    col
                    for col in final_df_pd.columns
                    if col not in base_cols and col != "timestamp_right"
                ]
                if "timestamp_right" in final_df_pd.columns:
                    final_df_pd = final_df_pd.drop(columns=["timestamp_right"])
                if macro_cols:
                    final_df_pd = final_df_pd.fillna(dict.fromkeys(macro_cols, 0.0))
                    final_df_pd["is_macro_available"] = (
                        final_df_pd[macro_cols].notna().any(axis=1).astype("int32")
                    )
                final_df = final_df_pd

        logger.info(f"Built dataset with shape: {final_df.shape}")

        return final_df

    def _process_symbol_polars(
        self,
        df: _pl.DataFrame,
        symbol: str,
        horizon_minutes: int,
        threshold: float,
        lookback_periods: int,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> _pl.DataFrame | None:
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
            # Prefer aggregator (matches unit test monkeypatch), then fallback to cache
            try:
                from ml.features.micro_aggregate import MicrostructureAggregator

                base_dir = self.micro_base_dir or "data/tier1"
                agg = MicrostructureAggregator(_Path(base_dir))
                micro = agg.compute_for_symbol(symbol)
                if micro.shape[0] > 0:
                    if micro["timestamp"].dtype != pl.Datetime:
                        micro = micro.with_columns(
                            pl.col("timestamp").cast(pl.Datetime("ns", "UTC")),
                        )
                    before_cols = set(dataset.columns)
                    dataset = dataset.join(micro, on="timestamp", how="left")
                    # Fill newly added numeric micro columns with 0 for stability
                    micro_cols = [c for c in dataset.columns if c not in before_cols]
                    if micro_cols:
                        fills = []
                        for c in micro_cols:
                            try:
                                if dataset.schema[c].is_numeric():
                                    fills.append(pl.col(c).fill_null(0))
                            except Exception:
                                pass
                        if fills:
                            dataset = dataset.with_columns(fills)
            except Exception as exc:
                logger.debug(f"Microstructure aggregator join failed for {symbol}: {exc}")
                try:  # fallback to cache-based join
                    base_dir_path = _Path(self.micro_base_dir or "data/tier1")
                    micro_cache = MicroMinuteCache(_Path("data/features/micro_minute"))
                    ts_min = dataset.select(pl.col("timestamp").min())[0, 0]
                    ts_max = dataset.select(pl.col("timestamp").max())[0, 0]

                    if ts_min is not None and ts_max is not None:
                        start_dt = (
                            ts_min.to_pydatetime() if hasattr(ts_min, "to_pydatetime") else ts_min
                        ).replace(tzinfo=UTC)
                        end_dt = (
                            ts_max.to_pydatetime() if hasattr(ts_max, "to_pydatetime") else ts_max
                        ).replace(tzinfo=UTC)
                        micro = micro_cache.get_range(symbol, start_dt, end_dt, base_dir_path)
                        if not micro.is_empty():
                            if micro["timestamp"].dtype != pl.Datetime:
                                micro = micro.with_columns(
                                    pl.col("timestamp").cast(pl.Datetime("ns", "UTC")),
                                )
                            before_cols = set(dataset.columns)
                            dataset = dataset.join(micro, on="timestamp", how="left")
                            micro_cols = [c for c in dataset.columns if c not in before_cols]
                            if micro_cols:
                                fills = []
                                for c in micro_cols:
                                    try:
                                        if dataset.schema[c].is_numeric():
                                            fills.append(pl.col(c).fill_null(0))
                                    except Exception:
                                        pass
                                if fills:
                                    dataset = dataset.with_columns(fills)
                except Exception as exc2:  # pragma: no cover
                    logger.debug(f"Microstructure cache join failed for {symbol}: {exc2}")

        # Optionally join L2 features (per-minute) via cache
        if self.include_l2:
            try:
                base_dir_path = _Path(self.l2_base_dir or "data/tier1")
                l2_cache = L2MinuteCache(_Path("data/features/l2_minute"))
                ts_min = dataset.select(pl.col("timestamp").min())[0, 0]
                ts_max = dataset.select(pl.col("timestamp").max())[0, 0]

                if ts_min is not None and ts_max is not None:
                    start_dt = (
                        ts_min.to_pydatetime() if hasattr(ts_min, "to_pydatetime") else ts_min
                    ).replace(tzinfo=UTC)
                    end_dt = (
                        ts_max.to_pydatetime() if hasattr(ts_max, "to_pydatetime") else ts_max
                    ).replace(tzinfo=UTC)
                    l2 = l2_cache.get_range(symbol, start_dt, end_dt, base_dir_path)
                    if not l2.is_empty():
                        if l2["timestamp"].dtype != pl.Datetime:
                            l2 = l2.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
                        before_cols = set(dataset.columns)
                        dataset = dataset.join(l2, on="timestamp", how="left")
                        l2_cols = [c for c in dataset.columns if c not in before_cols]
                        # Add availability mask for L2 and fill nulls with 0 for numeric L2 fields
                        if l2_cols:
                            has_any = None
                            fills = []
                            for c in l2_cols:
                                try:
                                    if dataset.schema[c].is_numeric():
                                        fills.append(pl.col(c).fill_null(0))
                                except Exception:
                                    pass
                                expr = pl.col(c).is_not_null()
                                has_any = expr if has_any is None else (has_any | expr)
                            if has_any is not None:
                                dataset = dataset.with_columns(
                                    [
                                        (has_any.cast(pl.Int32)).alias("is_l2_available"),
                                    ],
                                )
                            if fills:
                                dataset = dataset.with_columns(fills)
                        # Compute minimal derived L2 microstructure features
                        # - pressure_accel_top{k}: minute-over-minute change in depth_imbalance_top{k}
                        # - liquidity_gradient_top{k}: ask_slope_top{k} - bid_slope_top{k}
                        # - session_rel_spread: spread_bps normalized by daily median
                        try:
                            topks = [1, 3, 5, 10]
                            derived = []
                            for k in topks:
                                di = f"depth_imbalance_top{k}"
                                if di in dataset.columns:
                                    derived.append(
                                        (pl.col(di) - pl.col(di).shift(1)).alias(
                                            f"pressure_accel_top{k}",
                                        ),
                                    )
                                bs = f"bid_slope_top{k}"
                                aS = f"ask_slope_top{k}"
                                if bs in dataset.columns and aS in dataset.columns:
                                    derived.append(
                                        (pl.col(aS) - pl.col(bs)).alias(
                                            f"liquidity_gradient_top{k}",
                                        ),
                                    )
                            if derived:
                                dataset = dataset.with_columns(derived)
                            if "spread_bps" in dataset.columns:
                                # daily median per instrument_id
                                if "instrument_id" in dataset.columns:
                                    dataset = dataset.with_columns(
                                        [pl.col("timestamp").dt.date().alias("_day")],
                                    )
                                    med = dataset.group_by(["instrument_id", "_day"]).agg(
                                        pl.col("spread_bps").median().alias("_med_spread"),
                                    )
                                    dataset = dataset.join(
                                        med,
                                        on=["instrument_id", "_day"],
                                        how="left",
                                    )
                                    dataset = dataset.with_columns(
                                        [
                                            pl.when(pl.col("_med_spread") > 0)
                                            .then(pl.col("spread_bps") / pl.col("_med_spread"))
                                            .otherwise(1.0)
                                            .alias("session_rel_spread"),
                                        ],
                                    ).drop(["_day", "_med_spread"], strict=False)
                        except Exception:
                            pass
            except Exception as exc:  # pragma: no cover
                logger.debug(f"L2 cache join failed for {symbol}: {exc}")

        # Optionally add event-based known-future features
        if self.include_events:
            provider = self._get_event_provider()
            if provider is not None:
                try:
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
        df: _pd.DataFrame,
        symbol: str,
        horizon_minutes: int,
        threshold: float,
        lookback_periods: int,
    ) -> _pd.DataFrame | None:
        """
        Process single symbol with Pandas.
        """
        if pd_runtime is None:
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
                from ml.data.sources.events import SimpleEventSource

                provider = EventScheduleProvider(SimpleEventSource())
                if "timestamp" in dataset.columns:
                    ts_series = pl.Series(
                        "timestamp",
                        dataset["timestamp"].astype("datetime64[ns]"),
                    ).cast(pl.Int64)
                    ev_pl = provider.compute_features(ts_series, [symbol])
                    ev_pl = ev_pl.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
                    ev_pd = ev_pl.to_pandas()
                    dataset = dataset.merge(ev_pd, on="timestamp", how="left")
            except Exception as exc:  # pragma: no cover
                logger.debug(f"Event feature join failed for {symbol}: {exc}")

        return dataset

    def _compute_features_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
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

    def _compute_features_pandas(self, df: _pd.DataFrame) -> _pd.DataFrame:
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

        from typing import cast as _cast

        return _cast("_pd.DataFrame", features)

    def _generate_targets_polars(
        self,
        df: _pl.DataFrame,
        horizon_minutes: int,
        threshold: float,
    ) -> _pl.DataFrame:
        """
        Generate binary targets using Polars.
        """
        # Calculate forward returns
        future_prices = pl.col("close").shift(-horizon_minutes)
        current_prices = pl.col("close")
        forward_returns = (future_prices - current_prices) / current_prices

        # Binary classification + forward return sidecar for downstream Sharpe metrics
        targets = df.select(
            [
                (forward_returns > threshold).cast(pl.Int32).alias("y"),
                forward_returns.cast(pl.Float32).alias("forward_return"),
            ],
        )

        # Fill trailing NaNs introduced by the horizon shift
        targets = targets.with_columns(
            [
                pl.col("y").fill_null(0),
                pl.col("forward_return").fill_null(0.0),
            ],
        )

        return targets

    def _generate_targets_pandas(
        self,
        df: _pd.DataFrame,
        horizon_minutes: int,
        threshold: float,
    ) -> _pd.DataFrame:
        """
        Generate binary targets using Pandas.
        """
        # Calculate forward returns
        future_prices = df["close"].shift(-horizon_minutes)
        current_prices = df["close"]
        forward_returns = (future_prices - current_prices) / current_prices

        # Binary classification + forward return sidecar for downstream Sharpe metrics
        targets = pd.DataFrame(
            {
                "y": (forward_returns > threshold).astype(int),
                "forward_return": forward_returns.astype(float),
            },
        )

        # Fill trailing NaNs introduced by the horizon shift
        targets = targets.fillna({"y": 0, "forward_return": 0.0})

        from typing import cast as _cast

        return _cast("_pd.DataFrame", targets)

    def _add_static_features_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
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

    def _add_static_features_pandas(self, df: _pd.DataFrame) -> _pd.DataFrame:
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

    def _add_known_future_features_polars(self, df: _pl.DataFrame) -> _pl.DataFrame:
        """
        Add known-future time and calendar features using Polars.
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

        # Optional: precise market calendar features (known-future)
        if self.include_calendar:
            try:
                from ml.data.providers.calendar import MarketCalendarProvider
                from ml.data.sources.calendar import PandasCalendarSource

                # Determine instrument(s) for this frame; expect single instrument per symbol
                instruments = (
                    df.select(pl.col("instrument_id")).unique()["instrument_id"].to_list()
                    if "instrument_id" in df.columns
                    else ["GLOBAL"]
                )
                provider = MarketCalendarProvider(PandasCalendarSource())
                ts_series = df.select(pl.col("timestamp").cast(pl.Int64))["timestamp"]
                cal = provider.load_timeseries(instruments, ts_series)
                if not cal.is_empty():
                    cal = cal.with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC")))
                    join_keys: list[str] = ["timestamp"]
                    if "instrument_id" in df.columns and "instrument_id" in cal.columns:
                        join_keys.append("instrument_id")
                    df = df.join(cal, on=join_keys, how="left")
            except Exception as exc:  # pragma: no cover
                logger.debug(f"Calendar feature join skipped: {exc}")

        return df

    def _add_known_future_features_pandas(self, df: _pd.DataFrame) -> _pd.DataFrame:
        """
        Add known-future time and calendar features using Pandas.
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

        # Optional: precise market calendar features via provider (converted to pandas)
        if self.include_calendar:
            try:
                import polars as _pl

                from ml.data.providers.calendar import MarketCalendarProvider
                from ml.data.sources.calendar import PandasCalendarSource

                provider = MarketCalendarProvider(PandasCalendarSource())
                ts_series = _pl.Series(df["timestamp"].astype("int64").to_numpy())
                instruments = (
                    list({str(v) for v in df["instrument_id"].astype(str).tolist()})
                    if "instrument_id" in df.columns
                    else ["GLOBAL"]
                )
                cal_pl = provider.load_timeseries(instruments, ts_series)
                if cal_pl.shape[0] > 0:
                    cal_pl = cal_pl.with_columns(
                        _pl.col("timestamp").cast(_pl.Datetime("ns", "UTC")),
                    )
                    cal_pd = cal_pl.to_pandas()
                    join_cols = ["timestamp"] + (
                        ["instrument_id"] if "instrument_id" in df.columns else []
                    )
                    df = df.merge(cal_pd, on=join_cols, how="left")
            except Exception as exc:  # pragma: no cover
                logger.debug(f"Calendar feature join skipped: {exc}")

        return df
