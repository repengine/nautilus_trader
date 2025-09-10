# ML Pipeline Implementation Plan: Data Pipeline & Domain Bookkeeping

## Executive Summary

This document provides comprehensive implementation plans for:

1. **ML Data Pipeline**: End-to-end data pipeline from Databento market data ingestion through feature computation to TFT dataset creation
2. **Domain Bookkeeping & Unified Observability**: Enterprise-grade observability and intelligent automation systems

**Current State**:

- Data Pipeline: 40% complete (components 80-90% ready, integration 10% working)
- Domain Bookkeeping: 75% complete (infrastructure ready, integration needed)

**Target State**: Production-ready pipeline with automated daily updates and enterprise observability
**Key Principle**: Leverage existing Nautilus components rather than building custom solutions

---

# PART I: ML DATA PIPELINE

## Current State Assessment

**Data Pipeline Status**: 40% complete (components 80-90% ready, integration 10% working)
**Target State**: Production-ready pipeline with automated daily updates
**Estimated Timeline**: 8-9 days with one developer

## Architecture Decisions

### 1. Data Storage Strategy
**Decision**: Use ParquetDataCatalog as primary storage

- Historical data (>30 days): ParquetDataCatalog
- Recent data (<30 days): ParquetDataCatalog with potential PostgreSQL cache
- Rationale: Minimal overhead, native Nautilus integration, no additional storage costs

### 2. Instrument ID Mapping
**Decision**: Dynamic resolution with fallback mapping

- Primary: Query Databento metadata API for venue information
- Fallback: Hardcoded mapping for known instruments
- Default: Assume XNAS for equities if unknown

### 3. Feature Computation
**Decision**: Mandatory FeatureStore for all environments

- Ensures training/inference parity
- Minimal performance impact (~2ms for optional async writes)
- Hot path remains <5ms as required

### 4. Data Provider Priority

1. Calendar: `pandas_market_calendars`
2. Metadata: Databento metadata API
3. Events: Static CSV initially, FRED API later

### 5. Data Coverage Targets (explicit)
**Decision**: Enforce concrete horizons and rolling updates

- L0/L1 (minute bars/trades): 7 years for priority symbols
- L1 breadth (quotes/TBBO): 1 year for selected symbols (where needed)
- L2/L3 (mbp-1 depth, trade flow): 30 days rolling for top 10–50 symbols
- Daily updates: ingest prior trading day, update features, append TFT partitions

Rationale: Matches Databento availability and storage constraints while enabling
production-grade TFT training on comprehensive history and microstructure windows.

## Implementation Phases

### Phase 1: Foundation Fixes (Day 1)

#### Task 1.1: (Removed — obsolete)
The prior “EnhancedDataCollector” rename is no longer applicable; the collector is `DataCollector`.

#### Task 1.2: Create Centralized Schema Module

- [ ] **Create**: `ml/schema/polars_schemas.py`
- [ ] **Define**: Polars schemas for all data types
- [ ] **Import**: Add to `ml/_imports.py` for centralized access
- [ ] **Test**: Validate schemas with sample data

```python
"""
Centralized Polars schemas for ML data pipeline.

Ensures consistent data validation across all components.
"""
from __future__ import annotations

import polars as pl


# Market data schema (Nautilus standard)
MARKET_DATA_SCHEMA = pl.Schema({
    "instrument_id": pl.Utf8,
    "ts_event": pl.Int64,      # Nanoseconds since epoch
    "ts_init": pl.Int64,       # Nanoseconds since epoch
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
})

# Quote data schema
QUOTE_SCHEMA = pl.Schema({
    "instrument_id": pl.Utf8,
    "ts_event": pl.Int64,
    "ts_init": pl.Int64,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "bid_size": pl.Float64,
    "ask_size": pl.Float64,
})

# Trade data schema
TRADE_SCHEMA = pl.Schema({
    "instrument_id": pl.Utf8,
    "ts_event": pl.Int64,
    "ts_init": pl.Int64,
    "price": pl.Float64,
    "size": pl.Float64,
    "aggressor_side": pl.Utf8,
})

# Feature data schema
FEATURE_SCHEMA = pl.Schema({
    "instrument_id": pl.Utf8,
    "ts_event": pl.Int64,
    "feature_set_id": pl.Utf8,
    "values": pl.List(pl.Float32),  # Feature vector
})

# TFT dataset schema
TFT_SCHEMA = pl.Schema({
    "instrument_id": pl.Utf8,
    "timestamp": pl.Datetime,
    "target": pl.Float32,
    # Static features
    "asset_class": pl.Categorical,
    "exchange": pl.Categorical,
    # Known future features
    "hour_sin": pl.Float32,
    "hour_cos": pl.Float32,
    "day_sin": pl.Float32,
    "day_cos": pl.Float32,
    "is_market_hours": pl.Boolean,
    # Add feature columns dynamically
})


def validate_schema(df: pl.DataFrame, schema: pl.Schema, name: str) -> None:
    """
    Validate DataFrame against schema.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to validate
    schema : pl.Schema
        Expected schema
    name : str
        Name for error messages

    Raises
    ------
    ValueError
        If schema validation fails
    """
    try:
        # Check column presence
        missing_cols = set(schema.names) - set(df.columns)
        if missing_cols:
            raise ValueError(f"{name}: Missing columns {missing_cols}")

        # Check data types
        for col, expected_type in schema.items():
            actual_type = df[col].dtype
            if actual_type != expected_type:
                raise ValueError(
                    f"{name}: Column '{col}' has type {actual_type}, "
                    f"expected {expected_type}"
                )
    except Exception as e:
        raise ValueError(f"Schema validation failed for {name}: {e}")
```

#### Task 1.3: Consolidate Docker Compose Files

- [x] Compare: `ml/docker-compose.yml` vs `ml/deployment/docker-compose.yml`
- [x] Merge: unified `ml/deployment/docker-compose.yml` (canonical)
- [x] Archive: dev overrides moved to `ml/docker-compose.dev.yml`
- [x] Test: `docker compose -f ml/deployment/docker-compose.yml config`

```yaml
# ml/deployment/docker-compose.yml - Consolidated version
# Merge strategy:
# - Use ml/deployment version as base (has ML actors)
# - Add pgadmin from ml/ version
# - Use explicit migration mounting from ml/ version
# - Keep resource limits from deployment version
```

### Phase 2: Data Bridge Implementation (Days 2-3)

#### Task 2.1: Create Instrument Resolver

- [ ] **Create**: `ml/data/instrument_resolver.py`
- [ ] **Implement**: Dynamic resolution with fallback
- [ ] **Cache**: Add caching to reduce API calls
- [ ] **Test**: Unit tests with hypothesis for edge cases
- [ ] **Type hints**: Full annotations, mypy strict compliance

```python
"""
Instrument ID resolution for mapping symbols to Nautilus InstrumentId objects.

Provides dynamic resolution via Databento metadata with fallback mappings.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Venue

if TYPE_CHECKING:
    import databento


logger = logging.getLogger(__name__)

# Default fallback mappings for common instruments
DEFAULT_INSTRUMENT_MAP = {
    # Equities
    "SPY": "SPY.XNAS",
    "QQQ": "QQQ.XNAS",
    "IWM": "IWM.XNAS",
    "AAPL": "AAPL.XNAS",
    "MSFT": "MSFT.XNAS",
    # Futures
    "ES": "ES.XCME",
    "NQ": "NQ.XCME",
    "CL": "CL.XNYM",
    "GC": "GC.XCME",
}

# Exchange to Nautilus Venue mapping
EXCHANGE_VENUE_MAP = {
    "NASDAQ": "XNAS",
    "NYSE": "XNYS",
    "CME": "XCME",
    "NYMEX": "XNYM",
    "COMEX": "XCEC",
    "CBOT": "XCBT",
}


class InstrumentResolver:
    """
    Resolves symbol strings to Nautilus InstrumentId objects.

    Uses Databento metadata API with fallback to predefined mappings.
    Caches resolutions to minimize API calls.
    """

    def __init__(
        self,
        databento_client: databento.Historical | None = None,
        fallback_map: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize instrument resolver.

        Parameters
        ----------
        databento_client : databento.Historical, optional
            Databento client for metadata queries
        fallback_map : dict[str, str], optional
            Symbol to InstrumentId string mappings
        """
        self.client = databento_client
        self.fallback_map = fallback_map or DEFAULT_INSTRUMENT_MAP
        self._cache: dict[str, InstrumentId] = {}

    def resolve(self, symbol: str) -> InstrumentId:
        """
        Resolve symbol to InstrumentId.

        Parameters
        ----------
        symbol : str
            Symbol to resolve (e.g., "SPY", "ES")

        Returns
        -------
        InstrumentId
            Resolved instrument ID

        Raises
        ------
        ValueError
            If symbol cannot be resolved
        """
        # Check cache first
        if symbol in self._cache:
            return self._cache[symbol]

        instrument_id = self._resolve_uncached(symbol)
        self._cache[symbol] = instrument_id
        return instrument_id

    def _resolve_uncached(self, symbol: str) -> InstrumentId:
        """Resolve symbol without cache."""
        # Try Databento metadata if client available
        if self.client is not None:
            try:
                metadata = self.client.metadata.get_dataset_condition(
                    dataset="EQUS.MINI",
                    symbols=[symbol],
                )
                if metadata and len(metadata) > 0:
                    exchange = metadata[0].get("exchange", "")
                    venue = EXCHANGE_VENUE_MAP.get(exchange, "XNAS")
                    instrument_str = f"{symbol}.{venue}"
                    logger.info(f"Resolved {symbol} → {instrument_str} via metadata")
                    return InstrumentId.from_str(instrument_str)
            except Exception as e:
                logger.warning(f"Metadata lookup failed for {symbol}: {e}")

        # Try fallback map
        if symbol in self.fallback_map:
            instrument_str = self.fallback_map[symbol]
            logger.info(f"Resolved {symbol} → {instrument_str} via fallback")
            return InstrumentId.from_str(instrument_str)

        # Default to XNAS for unknown equities
        if symbol.isalpha() and len(symbol) <= 5:
            instrument_str = f"{symbol}.XNAS"
            logger.warning(f"Defaulting {symbol} → {instrument_str}")
            return InstrumentId.from_str(instrument_str)

        raise ValueError(f"Cannot resolve symbol: {symbol}")

    def bulk_resolve(self, symbols: list[str]) -> dict[str, InstrumentId]:
        """
        Resolve multiple symbols efficiently.

        Parameters
        ----------
        symbols : list[str]
            Symbols to resolve

        Returns
        -------
        dict[str, InstrumentId]
            Mapping of symbols to instrument IDs
        """
        results = {}
        uncached = []

        # Get cached results
        for symbol in symbols:
            if symbol in self._cache:
                results[symbol] = self._cache[symbol]
            else:
                uncached.append(symbol)

        # Resolve uncached symbols
        if uncached and self.client is not None:
            # Try bulk metadata query
            try:
                metadata_list = self.client.metadata.get_dataset_condition(
                    dataset="EQUS.MINI",
                    symbols=uncached,
                )
                # Process metadata results
                # ... implementation ...
            except Exception as e:
                logger.warning(f"Bulk metadata lookup failed: {e}")

        # Resolve remaining individually
        for symbol in uncached:
            if symbol not in results:
                results[symbol] = self.resolve(symbol)

        return results
```

#### Task 2.2: Wire DataCollector to ParquetDataCatalog

- [ ] **Update**: `ml/data/collector.py`
- [ ] **Import**: Add Nautilus Databento data client integration
- [ ] **Integrate**: Replace raw parquet writes with catalog writes
- [ ] **Test**: Verify data flows to catalog correctly
- [ ] **Validate**: Check Nautilus object conversion

```python
# ml/data/collector.py - Updated sections

from nautilus_trader.adapters.databento.config import DatabentoDataClientConfig  # configuration only
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.model.data import BarSpecification, BarAggregation
from ml.data.instrument_resolver import InstrumentResolver

class DataCollector:
    """
    Enhanced data collector using Nautilus components.
    """

    def __init__(
        self,
        storage_limit_gb: float = 500.0,
        catalog_path: str = "./data/catalog",
    ):
        """Initialize with Nautilus integration."""
        self.api_key = os.getenv("DATABENTO_API_KEY")
        if not self.api_key:
            raise ValueError("DATABENTO_API_KEY not set")

        # Initialize Databento client (existing)
        import databento as db
        self.client = db.Historical(self.api_key)

        # Add Nautilus components (runtime node wires the Databento client; see deployment service)
        self.catalog = ParquetDataCatalog(catalog_path)
        self.instrument_resolver = InstrumentResolver(
            databento_client=self.client
        )

        # ... rest of existing __init__ ...

    def collect_minute_bars(
        self,
        symbols: list[str] | None = None,
        days: int = 365,
    ) -> None:
        """
        Collect minute bars and store in ParquetDataCatalog.

        Updated to use Nautilus data structures.
        """
        # ... existing code until line 534 ...

        # After getting df from Databento (line 532)
        if not df.empty:
            # Resolve instrument ID
            try:
                instrument_id = self.instrument_resolver.resolve(symbol)
            except ValueError as e:
                logger.error(f"Cannot resolve {symbol}: {e}")
                continue

            # Convert to Nautilus Bar objects
            try:
                bars = self.nautilus_loader.parse_bars(
                    data=df,
                    instrument_id=instrument_id,
                    bar_spec=BarSpecification(
                        step=1,
                        aggregation=BarAggregation.MINUTE,
                    ),
                    ts_init_delta=0,  # Use ts_event as ts_init
                )

                # Write to catalog
                self.catalog.write_data(bars)

                # Also save raw parquet for debugging (optional)
                if self.save_raw_files:
                    df.to_parquet(bars_file)

                logger.info(
                    f"✓ {symbol}: {len(bars)} bars written to catalog"
                )

            except Exception as e:
                logger.error(f"Failed to convert {symbol} data: {e}")
                # Save raw file for debugging
                df.to_parquet(bars_file.with_suffix(".error.parquet"))
```

### Phase 2B: L2/L3 Microstructure Ingestion and Features (Days 3–4)

#### Task 2B.1: Ingest L2/L3 Windows into Catalog

- [ ] **Create**: `ml/data/ingest/microstructure_backfill.py`
- [ ] **Ingest**: L2 `mbp-1` depth (top N levels) for 30 days rolling
- [ ] **Ingest**: L3 trade prints (with aggressor side) for 30 days rolling
- [ ] **Store**: Write to `ParquetDataCatalog` under separate namespaces/partitions
- [ ] **Batching**: Symbols in batches of 5–10, with storage budget checks

#### Task 2B.2: Compute Microstructure Features

- [ ] **Use**: `ml/features/microstructure.py` (spread, imbalance, depth shape, intensity, impact)
- [ ] **Integrate**: Store outputs via `FeatureStore` using `feature_set_id="microstructure_v1"`
- [ ] **Validate**: Polars schema + manifest hash; backfill for 30 days window

#### Task 2B.3: Rolling Daily Update

- [ ] **Scheduler**: Add a daily job to refresh yesterday’s L2/L3, age out >30d partitions
- [ ] **Join**: Align microstructure features to bars via `asof_join` (point-in-time)

### Phase 3: Complete Scheduler Implementation (Day 4)

#### Task 3.1: Wire Scheduler to Actual Operations

- [ ] **Update**: `ml/data/scheduler.py`
- [ ] **Implement**: `_collect_latest_data()` method
- [ ] **Implement**: `_compute_features()` method
- [ ] **Implement**: `_clean_old_data()` method
- [ ] **Test**: End-to-end scheduler execution
- [ ] **Add**: APScheduler integration

```python
# ml/data/scheduler.py - Complete implementation

from ml._imports import HAS_APSCHEDULER, check_ml_dependencies
from ml.data.catalog_utils import bars_to_dataframe
from ml.stores.feature_store import FeatureStore

if HAS_APSCHEDULER:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

class DataScheduler:
    """Production-ready scheduler implementation."""

    def __init__(
        self,
        catalog: ParquetDataCatalog,
        collector: DataCollector | None = None,
        feature_store: FeatureStore | None = None,
        retention_days: int = 90,
    ) -> None:
        """Initialize with actual components."""
        self.catalog = catalog
        self.collector = collector or DataCollector(catalog_path=str(catalog.path))
        self.feature_store = feature_store
        self.retention_days = retention_days

        # Initialize scheduler
        self.scheduler = None
        if HAS_APSCHEDULER:
            self.scheduler = BackgroundScheduler()

        # Track universe
        self._universe_symbols = self._load_universe()

    def _load_universe(self) -> list[str]:
        """Load trading universe from configuration."""
        # TODO: Load from config file
        return [
            "SPY", "QQQ", "IWM", "DIA",
            "AAPL", "MSFT", "NVDA", "AMZN",
            "META", "GOOGL", "TSLA",
        ]

    def _collect_latest_data(self) -> None:
        """
        Collect latest data from Databento.

        Actually implemented now!
        """
        # Calculate target date (previous trading day)
        today = datetime.now()
        if today.weekday() == 0:  # Monday
            target_date = today - timedelta(days=3)  # Friday
        elif today.weekday() == 6:  # Sunday
            target_date = today - timedelta(days=2)  # Friday
        else:
            target_date = today - timedelta(days=1)

        start_date = target_date.replace(hour=0, minute=0, second=0)
        end_date = target_date.replace(hour=23, minute=59, second=59)

        logger.info(f"Collecting data for {start_date.date()}")

        # Collect minute bars for universe
        try:
            self.collector.collect_minute_bars(
                symbols=self._universe_symbols,
                days=1,  # Just previous day
            )

            # Verify data was written
            instrument_ids = [
                self.collector.instrument_resolver.resolve(s)
                for s in self._universe_symbols
            ]

            bars = self.catalog.bars(
                instrument_ids=instrument_ids,
                start=start_date,
                end=end_date,
            )

            logger.info(f"Collected {len(bars)} bars for {len(self._universe_symbols)} symbols")

        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            # Send alert/notification
            raise

    def _compute_features(self) -> None:
        """
        Compute features for newly collected data.

        Uses FeatureStore for guaranteed parity.
        """
        if self.feature_store is None:
            logger.warning("No FeatureStore configured, skipping features")
            return

        logger.info("Computing features for new data...")

        # Compute for last 30 days to update indicators
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        success_count = 0
        error_count = 0

        for symbol in self._universe_symbols:
            try:
                instrument_id = self.collector.instrument_resolver.resolve(symbol)

                # Compute and store features
                feature_count = self.feature_store.compute_and_store_historical(
                    instrument_id=str(instrument_id),
                    start=start_date,
                    end=end_date,
                )

                if feature_count > 0:
                    success_count += 1
                    logger.debug(f"Computed {feature_count} features for {symbol}")

            except Exception as e:
                error_count += 1
                logger.error(f"Feature computation failed for {symbol}: {e}")

        logger.info(
            f"Feature computation complete: "
            f"{success_count} success, {error_count} errors"
        )

    def _clean_old_data(self) -> None:
        """
        Clean data older than retention period.

        Removes old partitions and catalog files.
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        logger.info(f"Cleaning data older than {cutoff_date.date()}")

        # Clean catalog files
        cleaned_files = 0
        total_size_mb = 0

        catalog_path = Path(self.catalog.path)
        for parquet_file in catalog_path.rglob("*.parquet"):
            # Parse date from filename (assumes YYYY-MM-DD format)
            try:
                file_stat = parquet_file.stat()
                file_mtime = datetime.fromtimestamp(file_stat.st_mtime)

                if file_mtime < cutoff_date:
                    size_mb = file_stat.st_size / (1024 * 1024)
                    parquet_file.unlink()
                    cleaned_files += 1
                    total_size_mb += size_mb

            except Exception as e:
                logger.warning(f"Could not clean {parquet_file}: {e}")

        logger.info(
            f"Cleaned {cleaned_files} files, "
            f"freed {total_size_mb:.1f} MB"
        )

    def schedule_updates(self, cron_expression: str | None = None) -> None:
        """
        Schedule automated daily updates.

        Actually schedules jobs now!
        """
        if not HAS_APSCHEDULER:
            check_ml_dependencies(["apscheduler"])

        if cron_expression is None:
            # Default: Daily at 4 AM UTC
            cron_expression = "0 4 * * *"

        logger.info(f"Scheduling updates with cron: {cron_expression}")

        # Schedule the job
        self.scheduler.add_job(
            func=self.run_daily_update,
            trigger=CronTrigger.from_crontab(cron_expression),
            id="daily_data_update",
            name="Daily Data Update",
            replace_existing=True,
            max_instances=1,  # Prevent overlapping runs
        )

        # Start scheduler
        self.scheduler.start()
        logger.info("Scheduler started successfully")

    def shutdown(self) -> None:
        """Shutdown scheduler gracefully."""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("Scheduler shutdown complete")
```

### Phase 4: Fix TFT Dataset Builder (Day 5)

#### Task 4.1: Integrate with FeatureStore

- [ ] **Update**: `ml/data/tft_dataset_builder.py`
- [ ] **Remove**: Hardcoded feature computation (lines 255-315)
- [ ] **Add**: FeatureStore integration
- [ ] **Test**: Verify feature parity
- [ ] **Validate**: Check TFT schema compliance

```python
# ml/data/tft_dataset_builder.py - Updated to use FeatureStore

from ml.stores.feature_store import FeatureStore
from ml.preprocessing.stationarity import StationarityTransformer
from ml.preprocessing.joins import asof_join, create_lag_features

class TFTDatasetBuilder:
    """
    TFT dataset builder with proper FeatureStore integration.
    """

    def __init__(
        self,
        catalog: ParquetDataCatalog,
        symbols: list[str],
        feature_config: MLFeatureConfig | None = None,
        feature_store: FeatureStore | None = None,  # ADDED
        instrument_resolver: InstrumentResolver | None = None,  # ADDED
    ) -> None:
        """Initialize with mandatory FeatureStore."""
        self.catalog = catalog
        self.symbols = symbols
        self.feature_config = feature_config or MLFeatureConfig()

        # Initialize resolver
        self.instrument_resolver = instrument_resolver or InstrumentResolver()

        # Initialize or create FeatureStore (MANDATORY)
        if feature_store is None:
            db_connection = os.getenv(
                "DB_CONNECTION",
                "postgresql://postgres:postgres@localhost:5432/nautilus"
            )
            self.feature_store = FeatureStore(
                connection_string=db_connection,
                feature_config=self.feature_config,
            )
        else:
            self.feature_store = feature_store

        # Add preprocessing components
        self.stationarity_transformer = StationarityTransformer()

        logger.info(
            f"Initialized TFTDatasetBuilder with FeatureStore "
            f"for {len(symbols)} symbols"
        )

    def _compute_features(
        self,
        symbol: str,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Compute features using FeatureStore for parity.

        Replaces hardcoded feature computation.
        """
        # Resolve instrument ID
        instrument_id = self.instrument_resolver.resolve(symbol)

        # Convert DataFrame to format expected by FeatureStore
        bars_df = df.select([
            pl.col("timestamp").alias("ts_event"),
            pl.col("timestamp").alias("ts_init"),  # Same as ts_event for historical
            pl.col("open"),
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            pl.col("volume"),
        ])

        # Compute features using FeatureStore
        # This ensures perfect parity with live trading
        features_df, scaler = self.feature_store.feature_engineer.calculate_features_batch(
            bars_df.to_pandas()  # FeatureEngineer expects pandas
        )

        # Convert back to Polars
        features_pl = pl.from_pandas(features_df)

        # Join features with original data
        result = df.hstack(features_pl)

        # Apply stationarity transformation if configured
        if self.feature_config.apply_stationarity:
            numeric_cols = [c for c in result.columns if c not in ["timestamp", "instrument_id"]]
            result = self.stationarity_transformer.fit_transform(
                result,
                columns=numeric_cols,
            )

        return result

    def _add_known_future_features(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Add known future covariates for TFT.

        Enhanced with real calendar data.
        """
        # Time-based features (existing)
        df = df.with_columns([
            # Cyclical encoding for hour
            (2 * np.pi * pl.col("timestamp").dt.hour() / 24).sin().alias("hour_sin"),
            (2 * np.pi * pl.col("timestamp").dt.hour() / 24).cos().alias("hour_cos"),

            # Cyclical encoding for day of week
            (2 * np.pi * pl.col("timestamp").dt.weekday() / 7).sin().alias("day_sin"),
            (2 * np.pi * pl.col("timestamp").dt.weekday() / 7).cos().alias("day_cos"),

            # Market session flags
            ((pl.col("timestamp").dt.hour() >= 9) &
             (pl.col("timestamp").dt.hour() < 16)).alias("is_market_hours"),

            ((pl.col("timestamp").dt.hour() >= 4) &
             (pl.col("timestamp").dt.hour() < 9)).alias("is_premarket"),

            ((pl.col("timestamp").dt.hour() >= 16) &
             (pl.col("timestamp").dt.hour() < 20)).alias("is_aftermarket"),
        ])

        # Add calendar features if provider available
        if hasattr(self, "calendar_provider"):
            calendar_df = self.calendar_provider.get_calendar_features(
                start=df["timestamp"].min(),
                end=df["timestamp"].max(),
            )
            df = df.join(calendar_df, on="timestamp", how="left")

        return df

    def build_training_dataset(
        self,
        horizon_minutes: int = 15,
        min_return_threshold: float = 0.001,
        lookback_periods: int = 30,
        use_polars: bool = True,
        compute_missing_features: bool = True,
    ) -> pl.DataFrame | pd.DataFrame:
        """
        Build complete TFT training dataset.

        Now with proper feature engineering and preprocessing.
        """
        all_data = []

        for symbol in self.symbols:
            logger.info(f"Processing {symbol}...")

            # Load data from catalog
            try:
                instrument_id = self.instrument_resolver.resolve(symbol)
                df = bars_to_dataframe(
                    self.catalog,
                    [str(instrument_id)],
                    start=None,
                    end=None,
                )

                if df.is_empty():
                    logger.warning(f"No data for {symbol}")
                    continue

            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                continue

            # Check if features need computation
            if compute_missing_features:
                try:
                    # This will compute and store if missing
                    self.feature_store.compute_and_store_historical(
                        instrument_id=str(instrument_id),
                        start=df["timestamp"].min(),
                        end=df["timestamp"].max(),
                    )
                except Exception as e:
                    logger.error(f"Feature computation failed for {symbol}: {e}")

            # Compute features using FeatureStore
            df_with_features = self._compute_features(symbol, df)

            # Add known future features
            df_with_features = self._add_known_future_features(df_with_features)

            # Add static features
            df_with_features = self._add_static_features(df_with_features, symbol)

            # Create target variable
            df_with_features = self._create_target(
                df_with_features,
                horizon_minutes,
                min_return_threshold,
            )

            # Apply time series preprocessing
            df_with_features = asof_join(
                df_with_features,
                df_with_features,  # Self-join for lagged features
                on="timestamp",
                by="instrument_id",
            )

            # Add lagged features
            feature_cols = [c for c in df_with_features.columns
                            if c.startswith("feature_")]
            df_with_features = create_lag_features(
                df_with_features,
                columns=feature_cols,
                lags=[1, 5, 10, 20],
                group_by="instrument_id",
                timestamp_col="timestamp",
            )

            # Filter to minimum lookback
            df_with_features = df_with_features.filter(
                pl.col("timestamp").is_not_null()
            ).slice(lookback_periods)

            all_data.append(df_with_features)

        # Combine all symbols
        if all_data:
            final_df = pl.concat(all_data)

            # Validate against schema
            from ml.schema.polars_schemas import validate_schema, TFT_SCHEMA
            # Note: TFT_SCHEMA is partial, features are dynamic

            # Sort by time for proper train/test splits
            final_df = final_df.sort("timestamp")

            if use_polars:
                return final_df
            else:
                return final_df.to_pandas()
        else:
            logger.error("No data processed successfully")
            return pl.DataFrame() if use_polars else pd.DataFrame()
```

### Phase 5: Implement Real Data Sources (Days 6-7)

#### Task 5.1: Calendar Data Source

- [ ] **Create**: `ml/data/sources/calendar_real.py`
- [ ] **Implement**: MarketCalendarSource using pandas_market_calendars
- [ ] **Test**: Validate calendar data
- [ ] **Wire**: Update factory to use real source

```python
# ml/data/sources/calendar_real.py

import pandas_market_calendars as mcal
import polars as pl
from datetime import datetime
from typing import TYPE_CHECKING

from ml.data.sources.calendar import CalendarSource

if TYPE_CHECKING:
    from pandas import DataFrame as PandasDF


class MarketCalendarSource(CalendarSource):
    """
    Real market calendar data source using pandas_market_calendars.

    Provides actual trading calendars for various exchanges.
    """

    def __init__(self, exchange: str = "NYSE") -> None:
        """
        Initialize with specific exchange calendar.

        Parameters
        ----------
        exchange : str
            Exchange name (NYSE, NASDAQ, CME, etc.)
        """
        self.exchange = exchange
        try:
            self.calendar = mcal.get_calendar(exchange)
        except Exception as e:
            logger.error(f"Failed to load {exchange} calendar: {e}")
            # Fallback to NYSE
            self.calendar = mcal.get_calendar("NYSE")

    def fetch_calendar(
        self,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Fetch calendar data for date range.

        Parameters
        ----------
        start : datetime
            Start date
        end : datetime
            End date

        Returns
        -------
        pl.DataFrame
            Calendar data with trading days and sessions
        """
        try:
            # Get trading schedule
            schedule = self.calendar.schedule(
                start_date=start.date(),
                end_date=end.date(),
            )

            # Get valid trading days
            valid_days = self.calendar.valid_days(
                start_date=start.date(),
                end_date=end.date(),
            )

            # Convert to Polars
            if len(schedule) > 0:
                return pl.DataFrame({
                    "date": schedule.index.to_list(),
                    "market_open": schedule["market_open"].to_list(),
                    "market_close": schedule["market_close"].to_list(),
                    "is_trading_day": True,
                    "session": "regular",
                })
            else:
                # Return empty DataFrame with schema
                return pl.DataFrame({
                    "date": [],
                    "market_open": [],
                    "market_close": [],
                    "is_trading_day": [],
                    "session": [],
                })

        except Exception as e:
            logger.error(f"Calendar fetch failed: {e}")
            # Return mock data as fallback
            from ml.data.sources.calendar import MockCalendarSource
            return MockCalendarSource().fetch_calendar(start, end)

    def get_holidays(
        self,
        start: datetime,
        end: datetime,
    ) -> list[datetime]:
        """
        Get market holidays in date range.

        Parameters
        ----------
        start : datetime
            Start date
        end : datetime
            End date

        Returns
        -------
        list[datetime]
            List of holiday dates
        """
        try:
            # Get all days
            all_days = pd.date_range(start=start, end=end, freq='B')  # Business days

            # Get valid trading days
            valid_days = self.calendar.valid_days(
                start_date=start.date(),
                end_date=end.date(),
            )

            # Find holidays (business days that aren't trading days)
            holidays = []
            for day in all_days:
                if day not in valid_days:
                    holidays.append(day.to_pydatetime())

            return holidays

        except Exception as e:
            logger.error(f"Holiday fetch failed: {e}")
            return []
```

#### Task 5.2: Update Provider Factory

- [ ] **Update**: `ml/data/providers/factory.py`
- [ ] **Add**: Configuration for real vs mock sources
- [ ] **Test**: Verify provider creation with real sources
- [ ] **Document**: Add usage examples

```python
# ml/data/providers/factory.py - Updated

from ml.data.sources.calendar_real import MarketCalendarSource
from ml.data.sources.databento_metadata import DatabentoMetadataSource

class ProviderFactory:
    """Enhanced factory with real data source support."""

    def __init__(
        self,
        use_real_sources: bool = False,
        databento_api_key: str | None = None,
        metadata_source: MetadataSource | None = None,
        calendar_source: CalendarSource | None = None,
        event_source: EventSource | None = None,
    ) -> None:
        """
        Initialize with configurable real/mock sources.

        Parameters
        ----------
        use_real_sources : bool
            Use real data sources instead of mocks
        databento_api_key : str, optional
            API key for Databento metadata
        """
        if use_real_sources:
            # Use real sources where available
            self._calendar_source = calendar_source or MarketCalendarSource()

            if databento_api_key:
                self._metadata_source = metadata_source or DatabentoMetadataSource(
                    api_key=databento_api_key
                )
            else:
                logger.warning("No Databento API key, using mock metadata")
                self._metadata_source = metadata_source or MockMetadataSource()

            # Events still mock for now (Phase 2 implementation)
            self._event_source = event_source or MockEventSource()

            logger.info("Initialized with REAL data sources")

        else:
            # Use all mocks (existing behavior)
            self._metadata_source = metadata_source or MockMetadataSource()
            self._calendar_source = calendar_source or MockCalendarSource()
            self._event_source = event_source or MockEventSource()

            logger.info("Initialized with MOCK data sources")

        # Provider cache (singleton pattern)
        self._providers: dict[str, DataProvider] = {}
```

#### Task 5.3: Wire TransformProviderAdapter for Known‑Future/Static Features

- [ ] **Update**: Ensure mappings for `calendar`, `event_schedule`, `static_covariates`
- [ ] **Integrate**: Use adapter in the feature pipeline to fetch provider data by transform
- [ ] **Test**: Validate returned Polars DataFrames pass schema checks and align via timestamps

```python
# Example usage in a pipeline step
from ml.data.providers.factory import ProviderFactory, TransformProviderAdapter
from ml.features.pipeline import TransformSpec

factory = ProviderFactory(use_real_sources=True, databento_api_key=os.getenv("DATABENTO_API_KEY"))
adapter = TransformProviderAdapter(factory)

# Known future calendar features for timestamps
calendar_spec = TransformSpec(name="calendar", params={"encoding": "cyclic", "granularity": "hour"})
calendar_df = adapter.load_transform_data(
    transform=calendar_spec,
    timestamps=df["timestamp"],
    instruments=[symbol],
)
```

### Phase 5B: Registries Integration (Day 6)

#### Task 5B.1: FeatureRegistry Manifests

- [ ] **Create**: `FeatureManifest` for each feature_set_id (names, dtypes, pipeline signature)
- [ ] **Persist**: Save manifests via `FeatureRegistry` (JSON or Postgres backend)
- [ ] **Validate**: Compute schema hash with `compute_schema_hash` and enforce match at IO boundaries

#### Task 5B.2: ModelRegistry Hooks

- [ ] **Integrate**: Record trained teacher/student models (versions, roles, deployment status)
- [ ] **Link**: Associate models with `feature_set_id` and manifest schema hash
- [ ] **Populate**: Any A/B testing and canary configs as needed

### Phase 6: Integration Testing (Day 8)

#### Task 6.1: Create End-to-End Test

- [ ] **Create**: `ml/tests/integration/test_complete_pipeline.py`
- [ ] **Test**: Full data flow from Databento to TFT
- [ ] **Coverage**: Achieve >80% test coverage
- [ ] **Hypothesis**: Add property-based tests

```python
# ml/tests/integration/test_complete_pipeline.py

import pytest
from hypothesis import given, strategies as st
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from ml.data.collector import DataCollector
from ml.stores.feature_store import FeatureStore
from ml.data.tft_dataset_builder import TFTDatasetBuilder


class TestCompletePipeline:
    """
    End-to-end integration tests for ML data pipeline.

    Tests complete flow: Databento → Catalog → Features → TFT
    """

    @pytest.fixture
    def temp_catalog(self):
        """Create temporary catalog for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ParquetDataCatalog(tmpdir)
            yield catalog

    @pytest.fixture
    def test_db_connection(self):
        """Test database connection string."""
        return "postgresql://postgres:postgres@localhost:5432/test_nautilus"

    def test_complete_data_pipeline(self, temp_catalog, test_db_connection):
        """
        Test complete data flow from collection to TFT dataset.

        This is the critical integration test.
        """
        # 1. Initialize components
        collector = DataCollector(catalog_path=str(temp_catalog.path))

        feature_store = FeatureStore(
            connection_string=test_db_connection,
            feature_config=MLFeatureConfig(),
        )

        # 2. Collect data (using small test dataset)
        test_symbols = ["SPY"]
        collector.collect_minute_bars(
            symbols=test_symbols,
            days=5,  # Just 5 days for testing
        )

        # 3. Verify catalog has data
        instrument_id = collector.instrument_resolver.resolve("SPY")
        bars = temp_catalog.bars(
            instrument_ids=[instrument_id],
            start=datetime.now() - timedelta(days=5),
            end=datetime.now(),
        )

        assert len(bars) > 0, "No bars in catalog"

        # 4. Compute features
        feature_count = feature_store.compute_and_store_historical(
            instrument_id=str(instrument_id),
            start=datetime.now() - timedelta(days=5),
            end=datetime.now(),
        )

        assert feature_count > 0, "No features computed"

        # 5. Build TFT dataset
        builder = TFTDatasetBuilder(
            catalog=temp_catalog,
            symbols=test_symbols,
            feature_store=feature_store,
        )

        dataset = builder.build_training_dataset(
            horizon_minutes=15,
            lookback_periods=10,
        )

        assert not dataset.is_empty(), "TFT dataset is empty"
        assert "target" in dataset.columns, "No target variable"
        assert any(c.startswith("feature_") for c in dataset.columns), "No features"

        # 6. Validate data quality
        assert dataset["target"].null_count() == 0, "Target has nulls"

        # Check feature columns have no NaNs (after lookback)
        feature_cols = [c for c in dataset.columns if c.startswith("feature_")]
        for col in feature_cols:
            null_pct = dataset[col].null_count() / len(dataset)
            assert null_pct < 0.01, f"Feature {col} has {null_pct:.1%} nulls"

    @given(
        symbols=st.lists(
            st.sampled_from(["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]),
            min_size=1,
            max_size=3,
        ),
        days=st.integers(min_value=1, max_value=10),
    )
    def test_pipeline_with_hypothesis(
        self,
        temp_catalog,
        test_db_connection,
        symbols,
        days,
    ):
        """
        Property-based testing for pipeline robustness.

        Tests with various symbol combinations and date ranges.
        """
        # Initialize components
        collector = DataCollector(catalog_path=str(temp_catalog.path))

        # Collect data
        collector.collect_minute_bars(symbols=symbols, days=days)

        # Basic invariants that should always hold
        for symbol in symbols:
            try:
                instrument_id = collector.instrument_resolver.resolve(symbol)

                # Should either have data or have logged an error
                bars = temp_catalog.bars(
                    instrument_ids=[instrument_id],
                    start=datetime.now() - timedelta(days=days),
                )

                # If we got bars, they should be valid
                if bars:
                    assert all(bar.open <= bar.high for bar in bars)
                    assert all(bar.low <= bar.close for bar in bars)
                    assert all(bar.volume >= 0 for bar in bars)

            except ValueError:
                # Symbol resolution failure is acceptable
                pass

    def test_scheduler_integration(self, temp_catalog, test_db_connection):
        """Test scheduler actually runs pipeline."""
        from ml.data.scheduler import DataScheduler

        # Initialize scheduler
        scheduler = DataScheduler(
            catalog=temp_catalog,
            retention_days=30,
        )

        # Run update
        scheduler.run_daily_update()

        # Verify something happened
        # (Would need to mock Databento API for full test)

    def test_feature_parity(self, temp_catalog, test_db_connection):
        """
        Critical test: Verify batch and online features match.

        This ensures training/inference parity.
        """
        from ml.features.validation import FeatureParityValidator

        # Create sample data
        # ... test implementation ...

        validator = FeatureParityValidator()
        is_valid, report = validator.validate_parity(
            batch_features=batch_result,
            online_features=online_result,
            tolerance=1e-10,
        )

        assert is_valid, f"Parity validation failed: {report}"
```

### Phase 7: Production Deployment (Day 9)

#### Task 7.1: Create Production Scripts

- [ ] **Create**: `ml/scripts/run_daily_pipeline.py`
- [ ] **Create**: `ml/scripts/run_backfill.py`
- [ ] **Create**: `ml/scripts/run_feature_compute.py`
- [ ] **Test**: Scripts execute without errors
- [ ] **Document**: Add usage instructions

```python
# ml/scripts/run_daily_pipeline.py

#!/usr/bin/env python
"""
Production script for daily ML data pipeline.

Usage:
    python ml/scripts/run_daily_pipeline.py [--dry-run]

Environment Variables:
    DATABENTO_API_KEY: API key for market data
    DB_CONNECTION: PostgreSQL connection string
    CATALOG_PATH: Path to ParquetDataCatalog
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from ml.data.collector import DataCollector
from ml.data.scheduler import DataScheduler
from ml.stores.feature_store import FeatureStore
from ml.config.base import MLFeatureConfig


def setup_logging() -> None:
    """Configure production logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("/var/log/nautilus/ml_pipeline.log"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    """Run daily pipeline update."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Validate environment
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        logger.error("DATABENTO_API_KEY not set")
        sys.exit(1)

    db_connection = os.getenv(
        "DB_CONNECTION",
        "postgresql://postgres:postgres@localhost:5432/nautilus"
    )

    catalog_path = os.getenv("CATALOG_PATH", "./data/catalog")

    try:
        # Initialize components
        logger.info("Initializing ML pipeline components...")

        catalog = ParquetDataCatalog(catalog_path)

        feature_store = FeatureStore(
            connection_string=db_connection,
            feature_config=MLFeatureConfig(),
        )

        collector = DataCollector(
            catalog_path=catalog_path,
            storage_limit_gb=500.0,
        )

        # Create scheduler
        scheduler = DataScheduler(
            catalog=catalog,
            collector=collector,
            feature_store=feature_store,
            retention_days=90,
        )

        # Run daily update
        logger.info("Starting daily pipeline update...")
        scheduler.run_daily_update()

        logger.info("Daily pipeline update completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

#### Task 7.2: Update Docker Services

- [ ] **Create**: `ml/deployment/Dockerfile.pipeline`
- [ ] **Update**: `ml/deployment/docker-compose.yml`
- [ ] **Add**: Pipeline services to compose
- [ ] **Test**: Services start correctly

```dockerfile
# ml/deployment/Dockerfile.pipeline

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML code
COPY nautilus_trader nautilus_trader/
COPY ml ml/

# Set Python path
ENV PYTHONPATH=/app

# Default command (override in docker-compose)
CMD ["python", "ml/scripts/run_daily_pipeline.py"]
```

#### Task 7.3: Compose Orchestration (Services & Volumes)

- [ ] **Services**: Add one‑shot + long‑running services
  - `databento_backfill`: batch backfill to ParquetDataCatalog (7y L0/L1; 30d L2/L3)
  - `feature_compute`: compute/store indicator + engineered features via FeatureStore
  - `tft_prep`: join features/targets/known‑future + preprocessing → Parquet datasets
  - `scheduler`: daily updates (ingest → features → datasets)
- [ ] **Volumes**: Share data between services
  - `catalog_data`: ParquetDataCatalog root
  - `datasets`: TFT output datasets
  - `postgres_data`: DB storage (mounted migrations from `ml/stores/migrations`)

```yaml
services:
  databento_backfill:
    build:
      context: ../..
      dockerfile: ml/deployment/Dockerfile.pipeline
    environment:
      DATABENTO_API_KEY: ${DATABENTO_API_KEY}
    volumes:
      - catalog_data:/app/data/catalog
    command: ["python", "ml/deployment/run_backfill.py", "--l0l1", "--l2l3"]

  feature_compute:
    build:
      context: ../..
      dockerfile: ml/deployment/Dockerfile.pipeline
    environment:
      DB_CONNECTION: postgresql://postgres:postgres@postgres:5432/nautilus
    volumes:
      - catalog_data:/app/data/catalog
    depends_on:
      - postgres
    command: ["python", "ml/deployment/run_feature_build.py"]

  tft_prep:
    build:
      context: ../..
      dockerfile: ml/deployment/Dockerfile.pipeline
    environment:
      DB_CONNECTION: postgresql://postgres:postgres@postgres:5432/nautilus
    volumes:
      - catalog_data:/app/data/catalog
      - datasets:/app/data/datasets
    depends_on:
      - postgres
    command: ["python", "ml/deployment/run_tft_prep.py"]

  scheduler:
    build:
      context: ../..
      dockerfile: ml/deployment/Dockerfile.pipeline
    environment:
      DB_CONNECTION: postgresql://postgres:postgres@postgres:5432/nautilus
      DATABENTO_API_KEY: ${DATABENTO_API_KEY}
    volumes:
      - catalog_data:/app/data/catalog
      - datasets:/app/data/datasets
    depends_on:
      - postgres
    command: ["python", "ml/deployment/run_scheduler.py"]

volumes:
  catalog_data:
  datasets:
```

### Phase 8: Final Validation

#### Task 8.1: Quality Gates

- [ ] **Run**: `mypy ml --strict` - Must have 0 violations
- [ ] **Run**: `ruff check ml` - Must have 0 violations
- [ ] **Run**: `pytest ml -q --cov=ml` - Must have >80% coverage
- [ ] **Run**: Full integration test suite
- [ ] **Document**: Update README with setup instructions

#### Task 8.2: Performance Validation

#### Task 8.3: Schema & Parity Gates (CI)

- [ ] **Schema**: Validate Polars schemas at all IO boundaries (catalog → features → datasets)
- [ ] **Manifest**: Compute and check `schema_hash` equality with registered FeatureManifest
- [ ] **Parity**: Run `FeatureParityValidator` on sampled windows (batch vs online) and fail if tolerance exceeded
- [ ] **Point-in-time**: Enforce `asof_join` joins and `embargo_window` in TFT prep; add checks to prevent lookahead
- [ ] **Test**: Feature computation <5ms (hot path)
- [ ] **Test**: End-to-end pipeline performance
- [ ] **Verify**: Memory usage within bounds
- [ ] **Check**: No memory leaks in long-running processes

## Success Criteria

The implementation is complete when:

1. ✅ Data flows from Databento → ParquetDataCatalog → FeatureStore → TFT Dataset
2. ✅ All tests pass with >80% coverage
3. ✅ MyPy strict mode has 0 violations
4. ✅ Ruff has 0 violations
5. ✅ Docker services run without errors
6. ✅ Daily scheduler successfully updates data
7. ✅ Feature parity between training and inference is validated
8. ✅ Production scripts are documented and tested

## Risk Mitigation

### Technical Risks

1. **Instrument ID resolution complexity**
   - Mitigation: Comprehensive fallback mappings
   - Test with real Databento data early

2. **API rate limits**
   - Mitigation: Implement rate limiting in collector
   - Cache metadata aggressively

3. **Storage growth**
   - Mitigation: Automated retention policies
   - Monitor disk usage with alerts

### Process Risks

1. **Integration complexity**
   - Mitigation: Test each integration point separately
   - Use existing Nautilus components

2. **Performance requirements**
   - Mitigation: Profile critical paths early
   - Pre-allocate buffers in hot path

## Maintenance Notes

### Daily Operations

- Monitor `/var/log/nautilus/ml_pipeline.log` for errors
- Check Prometheus metrics for pipeline health
- Verify partition creation for new months

### Monthly Tasks

- Review storage usage and adjust retention
- Update instrument mappings for new listings
- Performance profiling of hot paths

### Troubleshooting

- If pipeline fails: Check API keys first
- If features drift: Verify FeatureStore parity
- If storage full: Run cleanup manually

## References

### Key Files

- [`ml/data/collector.py`](../data/collector.py) - Data collection
- [`ml/data/scheduler.py`](../data/scheduler.py) - Pipeline orchestration
- [`ml/stores/feature_store.py`](../stores/feature_store.py) - Feature computation
- [`ml/data/tft_dataset_builder.py`](../data/tft_dataset_builder.py) - Dataset creation

### Documentation

- [Nautilus Databento Integration](https://docs.nautilustrader.io/latest/adapters/databento.html)
- [ML Architecture](./ML_GENERAL.md)
- [Feature Store Design](./Store_Triad.md)

### External Dependencies

- `databento`: Market data API
- `pandas_market_calendars`: Trading calendars
- `nautilus_trader`: Core trading framework

---
