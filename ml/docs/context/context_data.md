# ML Data Module Context Document

## Executive Summary

The `ml/data/` module provides a comprehensive data pipeline infrastructure for machine learning workflows within Nautilus Trader. The system integrates with Nautilus native components while providing ML-specific capabilities for data collection, processing, and feature engineering.

Operational notes:

- Timestamps: All pipeline timestamps are UNIX nanoseconds. Store write paths perform defensive normalization from seconds/ms/us to ns and log a warning if triggered. See `context_stores.md` → "Timestamp Policy & Normalization".
- DB readiness: Apply canonical migrations and run a DB preflight before running ingestion/backfills to ensure required functions and partitions exist. See `context_deployment.md`.

**Key Components:**

- **Data Collection**: Enhanced `DataCollector` for Databento API integration with intelligent multi-tier collection strategy and storage management
- **Data Utilities**: Helper functions working directly with `ParquetDataCatalog` for seamless Nautilus integration
- **Scheduling**: Production-ready automated data collection and processing via `DataScheduler` with comprehensive DataRegistry integration
- **TFT Dataset Building**: Training data preparation with dual-source architecture (FeatureStore priority, direct computation fallback)
- **Provider Architecture**: Extensible SOLID-principle data provider system for static and time-series features
- **Alternative Data Loaders**: FRED economic indicators loader with caching, rate limiting, and DataStore integration
- **Feature Caching**: L2 and microstructure per-minute feature caches for efficient training dataset building
- **Build Infrastructure**: Pipeline orchestration with `build_runner.py` for parallel dataset construction
- **Observability Integration**: Comprehensive event tracking, correlation IDs, and 15+ Prometheus metrics
- **Message Bus Support**: Event publication to external systems via configurable publisher protocols

**Implementation Status**: 100% complete, production-ready with comprehensive metrics, monitoring, and operational resilience

## Quick Imports (Public API)

For common workflows, import directly from `ml.data` to avoid hunting files:

```
from ml.data import (
    DataCollector, DataScheduler, IngestionOrchestrator, TFTDatasetBuilder,
    InstrumentMetadataProvider, MarketCalendarProvider, EventScheduleProvider,
    MockCalendarSource, SimpleCalendarSource, PandasCalendarSource,
    DatabentoMetadataSource, NautilusMetadataSource,
    L2MinuteCache, MicroMinuteCache,
    bars_to_dataframe, quotes_to_dataframe, trades_to_dataframe,
)
```

This curated surface keeps the module easy to navigate.

## Recent Data Progress (Sept 2025)

This section documents the latest backfills, gap fills, macro refresh, and dataset builds performed to stabilize training and improve micro/L2 signal quality.

- L0 minute bars (Tier‑1)
  - Safe, targeted recent backfill added via `ml/scripts/backfill_ohlcv_recent.py`.
  - Writes to `data/tier1/<SYMBOL>/l0/<SYMBOL>_ohlcv.parquet`.
  - `TFTDatasetBuilder` now recognizes this `l0` path as a fallback source alongside `ohlcv‑1m_{historical,recent}.parquet`.

- L2 (depth) gap fill (Tier‑1)
  - Enhanced `ml/scripts/populate_l2_efficient.py` to be resource‑safe:
    - Streaming merges using PyArrow row‑groups (no full‑file in RAM).
    - Resume/progress file at `data/tier1/.l2_progress.json` (per‑day completion).
    - Sharding flags: `--max-symbols`, `--symbol-offset`, `--shuffle`.
    - Throttling: `--rate-limit` and `--sleep-between-symbols`.
    - Signal‑safe flush on SIGINT/SIGTERM.
  - Outputs final files `data/tier1/<SYMBOL>/l2/<SYMBOL>_mbp-10.parquet`.
  - Operational caps that avoided kills on shared hosts:
    - `POLARS_MAX_THREADS=1 PYARROW_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

- FRED macro refresh (90d)
  - Refreshed via `FREDDataLoader` and saved both wide and ML‑format:
    - `data/fred/fred_indicators_updated.parquet` (wide)
    - `data/fred/fred_indicators_ml_format.parquet` (long; for `join_fred_asof`).
  - Publication lag handled via `join_fred_asof(..., lag_days=N)`.

- Dataset builds
  - 60d dataset rebuilt for Tier‑1 with macro+micro+L2.
    - Per‑symbol: `/tmp/tft_universe_60d/<SYMBOL>/dataset.(parquet|csv)`
    - Feature sets registered in `~/.nautilus/ml/features`.
  - 90d dataset merged for Tier‑1 (macro+micro+L2), ready for HPO:
    - Merged: `/tmp/tft_universe_90d/merged/dataset.(parquet|csv)`.

### Runbook snippets

- L0 backfill (recent minutes):
  - `python -m ml.scripts.backfill_ohlcv_recent --tier 1 --days 14`

- L2 gap fill (resume‑safe shards):
  - Environment caps: `export POLARS_MAX_THREADS=1 PYARROW_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
  - Shards of 5 symbols (example):
    - `for off in 0 5 10 15 20; do python ml/scripts/populate_l2_efficient.py --tier 1 --days 14 --check-gaps --max-symbols 5 --symbol-offset $off --rate-limit 10 --sleep-between-symbols 1; done`

- FRED refresh (90d):
  - Set `FRED_API_KEY` then refresh via `FREDDataLoader` or `ml/scripts/fred_integration_bridge.py`.

- Builds:
  - 60d: `python -m ml.pipelines.build_runner --config ml/config/build_universe_60d.json`
  - 90d: `python -m ml.pipelines.build_runner --config ml/config/build_universe_90d.json`

## Per‑Minute Feature Caches (Cold Path)

Caches avoid recomputing expensive aggregates and live strictly off the hot path.

- Layout
  - L2: `data/features/l2_minute/<SYMBOL>/year=YYYY/month=MM/day=DD.parquet`
  - Micro: `data/features/micro_minute/<SYMBOL>/year=YYYY/month=MM/day=DD.parquet`
- Usage
  - L2 cache

    ```python
    from pathlib import Path
    from datetime import datetime, timezone
    from ml.data import L2MinuteCache

    cache = L2MinuteCache(Path("data/features/l2_minute"))
    start = datetime(2025, 8, 11, tzinfo=timezone.utc)
    end = datetime(2025, 8, 18, tzinfo=timezone.utc)
    df = cache.get_range("SPY", start, end, raw_base_dir=Path("data/tier1"))
    ```

  - Micro cache (mirrors L2)

    ```python
    from ml.data import MicroMinuteCache
    cache = MicroMinuteCache(Path("data/features/micro_minute"))
    df = cache.get_range("SPY", start, end, raw_base_dir=Path("data/tier1"))
    ```

- Semantics
  - Timestamp filter is half‑open `[start, end)`; results are sorted and cast to `Datetime[ns, UTC]`.
  - Missing partitions compute on demand and persist before returning.

## Sources vs Providers (Layering)

- Sources (`ml/data/sources/*`): access/raw domain models and real connectors
  - Examples: `PandasCalendarSource`, `DatabentoMetadataSource`, `MockEventSource`
- Providers (`ml/data/providers/*`): ML‑ready features derived from sources
  - Examples: `MarketCalendarProvider`, `InstrumentMetadataProvider`, `EventScheduleProvider`
- Pattern: providers depend on sources; dataset builders and schedulers depend on providers.

## Canonical Defaults (Metadata)

To avoid drift, default instrument metadata is defined once and reused:

- Function: `ml.data.sources.metadata.default_metadata(symbol) -> dict[str, Any]`
- Used by: Databento and Nautilus metadata sources and the provider’s empty‑frame builder.

## Environment Variables (Data + Scheduling)

These enable external APIs and DB access for schedulers/registry:

- Data APIs
  - `DATABENTO_API_KEY` — Databento client (optional for local dev; required for real API tests)
  - `FRED_API_KEY` — FRED economic data loader (required to fetch real data)
- Database (tests and registry helpers)
  - `DATABASE_URL` — primary Postgres URL
  - `ML_DATABASE_URL` — alias used by tools (mirrors `DATABASE_URL`)
  - `NAUTILUS_REGISTRY_DB_URL` — alias for registry helpers

## Unified Ingestion (Dual-Write)

To guarantee that the dataset builder can always read from a ParquetDataCatalog while
preserving SQL coverage and watermarks, the ingestion control path supports dual-write:

- Backfill (CLI):
  - `python -m ml.cli.ingest_backfill ... --also-write-catalog --catalog-path $CATALOG_PATH`
  - Writes SQL (`market_data`) and domain objects to `ParquetDataCatalog`.

- Daily (Scheduler):
  - `DataScheduler(..., use_orchestrator=True, dual_write=True).run_daily_update()`
  - Uses `IngestionOrchestrator` under the hood for consistent behavior.

IntegrationManager can pass `ALSO_WRITE_CATALOG=1` to its boot-time backfill command so
containers start with a populated catalog.

## Macro (FRED) ML-Format Parquet

Dataset builder performs an as-of join against an ML-format parquet file with columns
`timestamp`, `series_id`, `value`.

- Export via loader: `FREDDataLoader.export_ml_parquet(out_path="data/fred/fred_indicators_ml_format.parquet")`
- Or use CLI: `python -m ml.cli.fred_export_ml_parquet --out data/fred/fred_indicators_ml_format.parquet`

Then enable macro in the builder (`include_macro=True`) or pass an explicit path in
`join_fred_asof(..., fred_path=...)`.

## Architecture Overview

The data pipeline follows a layered architecture with clear separation of concerns:

```
Raw Data Layer
├── DataCollector: Multi-tier collection strategy
├── ParquetDataCatalog: Nautilus native storage
└── Databento Integration: Real-time market data

Processing Layer
├── DataScheduler: Automated orchestration
├── FeatureEngineer: On-demand computation
├── Feature Caches: L2/Micro pre-computed features
└── TFTDatasetBuilder: Training data preparation

Integration Layer
├── DataStore: Unified persistence facade
├── DataRegistry: Event emission and tracking
├── FeatureStore: Training/inference parity
└── FRED Loader: Economic indicators

Provider Layer
├── Base Providers: SOLID architecture
├── Calendar Sources: Market schedule data
├── Metadata Sources: Instrument information
└── Event Sources: Corporate actions/earnings
```

## Component Details

### 1. Data Collection (`collector.py`)

**Purpose**: Enhanced data collector optimizing Databento subscription value with intelligent multi-tier collection strategy.

**Class**: `DataCollector`

**I/O Specifications**:

- **Input**:
  - `storage_limit_gb`: Maximum storage budget (default: 1000GB, configurable)
  - `data_dir`: Output directory for collected data
  - `config`: DataCollectorConfig with collection parameters
  - `end_date`: Last available data date (configurable for backtesting)
  - Databento API key via `DATABENTO_API_KEY` environment variable
- **Output**: Raw market data files in Parquet format with comprehensive metadata
- **Storage**: Hierarchically organized by symbol and data type under configurable base directory

**Key Features**:

- **Intelligent Collection Strategy**:
  - L2 market depth (mbp-1): 30 days for top 50 most liquid symbols
  - L1 trades: Multi-year historical data (1-7 years) for priority symbols
  - TBBO quotes: 30 days for spread dynamics analysis
  - Minute bars: 1 year coverage for all symbols with minimal storage footprint
  - Extended L1 trades: Additional symbols based on available storage capacity
- **Smart Storage Management**:
  - Real-time storage usage tracking and projection
  - Automatic phase adjustment and scope reduction based on available space
  - Liquidity-based data size estimation with symbol-specific adjustments
  - Collection metadata persistence with comprehensive statistics
- **Priority Symbols**: Tiered approach with 20+ core symbols (SPY, QQQ, IWM, AAPL, MSFT, NVDA, etc.) plus extended coverage
- **Operational Features**:
  - Graceful degradation when API key unavailable (test environments)
  - Rate limiting compliance with Databento API constraints
  - Progress tracking and resumable collection sessions

**Current Status**: ✅ Complete production implementation

### 2. Data Utilities (`catalog_utils.py`)

**Purpose**: Helper functions for seamless ParquetDataCatalog integration.

**Key Functions**:

- `bars_to_dataframe()`: Converts Nautilus bars to ML-friendly DataFrames
- `quotes_to_dataframe()`: Converts quote ticks to DataFrames
- `trades_to_dataframe()`: Converts trade ticks to DataFrames
- `resolve_instrument_id()`: Multi-venue symbol resolution

**Features**:

- Native Nautilus type handling with automatic conversion
- Memory-efficient data loading with chunking support
- Multi-venue instrument ID resolution (ARCA, NASDAQ, NYSE, etc.)
- Comprehensive error handling with per-instrument failure isolation

**Current Status**: ✅ Complete implementation with Nautilus integration

### 3. Data Scheduling (`scheduler.py`)

**Purpose**: Production-ready automated data collection and processing with comprehensive monitoring and resilience.

**Class**: `DataScheduler`

**I/O Specifications**:

- **Input**:
  - `catalog`: ParquetDataCatalog for data storage
  - `config`: SchedulerConfig with Databento settings and feature store configuration
  - `collector`: DataCollector instance (optional, creates default if None)
  - `feature_engineer`: FeatureEngineer for feature computation (optional)
  - `metrics_port`: Port for Prometheus metrics server (default: 8000)
  - `connection`: Database connection string for feature store (optional)
- **Output**:
  - Updated ParquetDataCatalog with new market data and proper Nautilus types
  - Computed features persisted to FeatureStore for training/inference parity
  - DataRegistry events with SHA256 correlation IDs for end-to-end tracking
  - Comprehensive Prometheus metrics for operational monitoring
  - Watermark updates for data freshness and completeness tracking

**Current Status**: ✅ Complete production implementation with enterprise-grade resilience

**Key Features**:

- **DataRegistry Integration**:
  - Emits CATALOG_WRITTEN events for data lineage with SHA256-based correlation IDs
  - Updates watermarks for dataset freshness and completeness tracking
  - Progressive fallback: PostgreSQL backend → JSON backend for development
  - Cross-domain event correlation for end-to-end pipeline tracing and analytics
  - Event emission with comprehensive error handling and retry logic
- **Comprehensive Metrics** (15+ Prometheus metrics):
  - `data_collection_latency`: Collection time by instrument and data type
  - `pipeline_stage_latency`: Stage execution times with percentile tracking
  - `catalog_write_operations_total`: Catalog write success/failure counters
  - `feature_store_operations_total`: Feature store operations with operation type labels
  - `data_staleness_seconds`: Data freshness tracking per instrument
  - `api_rate_limit_hits`: API rate limit monitoring with endpoint labels
  - `active_collection_tasks`: Real-time task monitoring
  - `data_retention_cleanup_total`: Cleanup operation tracking
  - `features_computed_total`: Feature computation counters by instrument and type

### 4. TFT Dataset Builder (`tft_dataset_builder.py`)

**Purpose**: Advanced training dataset preparation with dual-source architecture and comprehensive feature integration.

**Class**: `TFTDatasetBuilder`

**I/O Specifications**:

- **Input**:
  - `catalog`: ParquetDataCatalog for raw data access
  - `symbols`: List of symbols to include in dataset
  - `feature_config`: MLFeatureConfig for feature engineering parameters (optional)
  - `feature_store`: FeatureStore for pre-computed features (optional, enables training/inference parity)
  - `include_macro`: Boolean flag for FRED economic indicators integration
  - `include_micro`: Boolean flag for microstructure features via caching
  - `include_l2`: Boolean flag for L2 order book features via caching
  - `include_events`: Boolean flag for event-based known-future features
- **Output**: TFT-compatible DataFrame (Pandas or Polars) with comprehensive feature set

**Key Methods**:

- `prepare_training_data_from_store()`: Load features from FeatureStore
  - Ensures strict training/inference parity using identical feature computation paths
  - Combines FeatureStore features with bar data for target generation
  - Adds TFT-specific features (static covariates, known-future features)
  - Comprehensive error handling with fallback to direct computation

- `prepare_training_data()`: Intelligent source selection with automatic failover
  - Priority: FeatureStore (for parity) → Direct computation (with logging)
  - Supports both Polars and Pandas output formats
  - Comprehensive logging for monitoring source selection decisions

- `_build_training_dataset_direct()`: Direct feature computation with venue fallback
  - Multi-venue symbol resolution (ARCA, NASDAQ, NYSE, etc.)
  - Parquet file fallback for missing catalog entries
  - Robust error handling with per-symbol failure isolation

- `build_training_dataset()`: Legacy compatibility wrapper
  - Maintains backward compatibility while using new dual-source architecture
  - Automatic source selection with comprehensive error recovery

**Current Status**: ✅ Complete with advanced dual-source architecture and comprehensive feature integration

- **FeatureStore Integration**: Priority source ensuring strict training/inference parity with feature validation
- **Automatic Fallback**: Graceful degradation to direct computation with comprehensive logging and error recovery
- **Performance**: Optimized batch processing with parallel symbol processing and memory-efficient data handling
- **Format Support**: Native support for both Polars and Pandas DataFrames with format-specific optimizations
- **Advanced Features**:
  - FRED economic indicators integration via `fred_join.py` with configurable publication lag
  - L2 and microstructure feature caching via `l2_cache.py` and `micro_cache.py`
  - Event-based known-future features via provider integration
  - Comprehensive venue mapping and instrument ID resolution
  - Date range filtering and time-based dataset slicing

### 5. Data Provider Architecture (`providers/`)

**Purpose**: Extensible SOLID-principle provider system for heterogeneous data sources.

**Base Classes**:

- `BaseDataProvider`: Common functionality (logging, metrics, validation)
- `CachedDataProvider`: Template method with caching logic
- `BaseStaticProvider`: Static data with indefinite caching
- `BaseTimeSeriesProvider`: Time-varying data with validation

**Concrete Implementations**:

- `InstrumentMetadataProvider`: Static instrument information
- `MarketCalendarProvider`: Calendar features (trading hours, holidays)
- `EventScheduleProvider`: Corporate actions and earnings

**Key Features**:

- **Protocol-First Design**: Structural typing without implementation coupling
- **Factory Pattern**: Singleton provider instances with dependency injection
- **Transform Adapter**: Maps feature transforms to appropriate providers
- **Progressive Fallback**: Mock providers when real sources unavailable

**Current Status**: ✅ Complete implementation with factory and adapter patterns

### 6. Data Sources

#### Calendar Sources (`sources/calendar.py`)

**Purpose**: Comprehensive market calendar data sources with exchange-specific trading hours and holiday schedules.

**Implementations**:

- `MockCalendarSource`: Testing with realistic market schedules and holiday simulation
- `SimpleCalendarSource`: Basic NYSE schedule with fixed hours and weekend detection
- `PandasCalendarSource`: Production-ready real market calendar integration

**Current Status**: ✅ Complete implementation with production-grade calendar provider

**PandasCalendarSource Features**:

- **Exchange Coverage**: NYSE, NASDAQ, CME, CBOT, ICE, CBOE, LSE, EUREX, JPX, HKEX, ASX, and 20+ major global exchanges
- **Holiday Integration**: Accurate holiday calendars with early close detection
- **Extended Hours**: Pre-market (4:00-9:30 AM) and after-hours (4:00-8:00 PM) session tracking
- **24/7 Markets**: Special handling for cryptocurrency exchanges (BINANCE, COINBASE)
- **Caching**: Intelligent schedule caching with configurable TTL to minimize API calls
- **Fallback Logic**: Automatic fallback to SimpleCalendarSource when pandas_market_calendars unavailable

## Deterministic Data Fixtures (Testing)

For provider-agnostic ingestion and contract testing, deterministic, lightweight fixtures are provided under `ml/data/fixtures/`:

- `make_tbbo_fixture` (L1 TBBO), `make_trades_fixture` (trades), `make_mbp10_fixture` (L2 snapshots)
- Each returns `(DataFrame, FixtureManifest)` where `FixtureManifest` records `schema_hash` and `content_sha256` for reproducibility.
- Property/contract tests validate ordering, idempotent replay deduplication, and schema stability.

Usage example:

```python
from ml.data.fixtures import make_tbbo_fixture

df, manifest = make_tbbo_fixture(instrument_id="EURUSD.SIM", rows=60)
assert len(df) == 60
```

See:

- Contracts: `ml/tests/contracts/test_databento_fixtures_contracts.py`
- Property: `ml/tests/property/test_ingestion_watermark_properties.py`
- Performance: `ml/tests/performance/test_ingestion_microbench.py`

## Ingestion Resume & Backoff

The ingestion helper `ml.data.ingest.resume.DatabentoIngestor` provides robust, provider‑agnostic ingestion with:

- Resume from last timestamp (stateful per instrument)
- Retry/backoff on transient errors (configurable policy; injectable sleep function for tests)
- Daily window planning across time zones (DST‑aware via `zoneinfo`)
- Metrics emission through `ml.data.ingest.metrics`

Usage outline:

```python
from datetime import date
from ml.data.ingest.resume import DatabentoIngestor, IngestState, BackoffPolicy

ingestor = DatabentoIngestor(client=databento_like_client, policy=BackoffPolicy())
state = IngestState()

# Plan DST‑aware daily windows
windows = ingestor.plan_daily_windows(start_date=date(2021,3,13), end_date=date(2021,3,16), tz="America/New_York")

for start_ns, end_ns in windows:
    df = ingestor.ingest_time_window(
        dataset="GLBX.MDP3",
        schema="tbbo",
        instrument="ES-USD-FUT.CME",
        start_ns=start_ns,
        end_ns=end_ns,
        source="historical",
        state=state,
    )
    # Process df...
```

Tests:

- Backoff + resume: `ml/tests/unit/ingest/test_resume_backoff.py`
- DST window planning: `ml/tests/property/test_window_planner_dst.py`

## Streaming + Gap Backfill Orchestration

At process startup, orchestrate two complementary paths:

- Live streaming: attach a streaming client that writes incoming records through a `MarketDataWriterProtocol` to canonical storage.
- Gap backfill: detect missing UTC day buckets via a `CoverageProviderProtocol` and backfill each window using `DatabentoIngestor`.

The orchestrator (`ml.data.ingest.orchestrator.IngestionOrchestrator`) wires these pieces together and integrates with the registry for events + watermarks:

```python
from pathlib import Path
from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.data.ingest.resume import DatabentoIngestor, IngestState
from ml.stores.providers import SqlCoverageProvider, SqlMarketDataWriter
from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import PersistenceConfig, BackendType

DB_URL = "postgresql://postgres:postgres@localhost:5433/nautilus"  # or $DB_CONNECTION

coverage = SqlCoverageProvider(connection_string=DB_URL, table_name="market_data", ts_field="ts_event")
writer = SqlMarketDataWriter(connection_string=DB_URL, table_name="market_data")
registry = DataRegistry(
    registry_path=Path("/tmp/registry"),
    persistence_config=PersistenceConfig(backend=BackendType.POSTGRES, connection_string=DB_URL),
)

ingestor = DatabentoIngestor(client=databento_like_client)
orch = IngestionOrchestrator(coverage=coverage, writer=writer, registry=registry, ingestor=ingestor)
state = IngestState()

# Backfill gaps in the last 7 days for an instrument
gaps = orch.backfill_gaps(
    dataset_id="GLBX.MDP3", schema="tbbo", instrument_id="ES-USD-FUT.CME", lookback_days=7, state=state
)

# Live path integration hook (attach streaming client that writes via `writer`)
orch.start_live()
```

Tests:

- Gap detection + backfill: `ml/tests/unit/ingest/test_orchestrator_backfill.py`
- Retry/resume: `ml/tests/unit/ingest/test_resume_backoff.py`
- DST window planning: `ml/tests/property/test_window_planner_dst.py`

SQL implementations for Postgres are provided via `ml.stores.providers` and use the canonical `market_data` table created by migration `ml/stores/migrations/003_market_data.sql`. If your raw layer is a Nautilus `ParquetDataCatalog`, you can use `ml.stores.providers.CatalogCoverageProvider` to derive gap coverage from catalog file intervals.

### 7. FRED Economic Data Integration (`loaders/fred_loader.py`)

**Purpose**: Complete economic indicator integration with comprehensive observability.

**Class**: `FREDDataLoader`

**Key Features**:

- **Comprehensive Indicators**: 20+ economic indicators including rates, volatility, GDP, CPI, employment
- **Caching**: Configurable TTL with metadata persistence
- **Rate Limiting**: FRED API compliance (120 calls/minute)
- **DataStore Integration**: Proper instrument ID generation and storage
- **DataRegistry Events**: Manifest registration with schema validation
- **Observability**: Dedicated Prometheus metrics for monitoring

**Current Status**: ✅ Complete implementation with production observability

### 8. Feature Caching Infrastructure

#### L2 Minute Cache (`l2_cache.py`)

**Purpose**: Efficient caching of L2 order book per-minute aggregates.

**Features**:

- Day-partitioned storage (`data/features/l2_minute/<SYMBOL>/year=YYYY/month=MM/day=DD.parquet`)
- On-demand computation using `L2Aggregator`
- UTC timestamp handling with nanosecond precision
- Automatic cache directory creation and management

#### Microstructure Cache (`micro_cache.py`)

**Purpose**: Caching of L1/L0-derived per-minute microstructure features.

**Features**:

- Same partitioning scheme as L2 cache
- Integration with `MicrostructureAggregator`
- Efficient date range queries with [start, end) semantics
- Memory-efficient concatenation of day partitions

**Current Status**: ✅ Complete implementation with comprehensive caching

### 9. Build Pipeline Orchestration (`pipelines/build_runner.py`)

**Purpose**: Orchestrate parallel dataset building with progress tracking.

**Key Features**:

- **Configuration**: JSON/TOML config support with comprehensive parameter validation
- **Parallel Execution**: Configurable worker pool for symbol-level parallelism
- **Progress Tracking**: JSONL progress logs with resumable execution
- **Error Recovery**: Per-symbol error isolation with detailed logging
- **Metrics Integration**: Prometheus metrics for build task monitoring
- **Flexible Windows**: Date range, days-back, or open-ended collection

**Current Status**: ✅ Complete implementation with production orchestration

### 10. Feature Pipeline Infrastructure (`features/pipeline.py`)

**Purpose**: Declarative feature pipeline with transform catalog and capability gating.

**Key Components**:

- **Transform Protocols**: Extensible feature transform interface with data requirements
- **Pipeline Specification**: Configuration-driven feature computation
- **Signature Hashing**: Deterministic pipeline versioning
- **Capability Gating**: Data requirement validation (L1_ONLY, L1_L2, L1_L2_L3)

**Built-in Transforms**:

- Returns, momentum, volatility transforms
- Volume ratio and core technical indicators
- Static covariates and calendar features

**Current Status**: ✅ Complete implementation with extensible transform system

## Status Summary

### Recently Completed 🎯

- **Feature Caching Infrastructure**: L2MinuteCache and MicroMinuteCache for efficient training dataset building
- **Build Pipeline Orchestration**: build_runner.py with parallel execution, progress tracking, and resumable builds
- **Advanced TFT Integration**: Comprehensive feature integration (macro, micro, L2, events) with intelligent source selection
- **FRED Economic Data**: Complete integration with caching, rate limiting, and DataStore persistence
- **Enhanced Calendar Sources**: Production-ready PandasCalendarSource with global exchange support
- **DataRegistry Event Correlation**: SHA256-based correlation IDs for end-to-end pipeline tracing
- **Comprehensive Metrics**: 15+ Prometheus metrics with operational dashboards and alerting
- **Production Resilience**: Progressive fallback chains and graceful degradation across all components

### Integration Points

- **Nautilus Core**: Uses ParquetDataCatalog for data storage, native Nautilus types
- **ML Stores**: Integrates with FeatureStore, DataStore, ModelStore for persistence
- **Registry System**: Full DataRegistry integration with event emission and tracking
- **Observability**: Comprehensive metrics, logging, and event correlation
- **Configuration**: msgspec-based configs with validation and freezing

## API Reference

### Core Functions

```python
# Data loading utilities (seamless Nautilus integration)
from ml.data import bars_to_dataframe, quotes_to_dataframe, trades_to_dataframe

# Load bars data with native Nautilus types
bars_df = bars_to_dataframe(
    catalog=catalog,
    instrument_ids=["SPY.NYSE", "AAPL.NASDAQ"],
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31)
)

# Advanced dataset building with comprehensive feature integration
from ml.data.tft_dataset_builder import TFTDatasetBuilder
from ml.stores.feature_store import FeatureStore

feature_store = FeatureStore(connection_string="postgresql://...")
builder = TFTDatasetBuilder(
    catalog=catalog,
    symbols=["SPY", "AAPL"],
    feature_store=feature_store,  # Enables training/inference parity
    include_macro=True,  # FRED economic indicators
    include_micro=True,  # Microstructure features
    include_l2=True,     # L2 order book features
    include_events=True, # Event-based features
)

# Intelligent source selection with comprehensive feature set
dataset = builder.prepare_training_data(
    start=datetime(2024, 1, 1),
    horizon_minutes=15,
    use_polars=True
)
```

### Data Collection and Scheduling

```python
# Automated data collection with monitoring
from ml.data.collector import DataCollector
from ml.data.scheduler import DataScheduler

collector = DataCollector(
    storage_limit_gb=1000,
    data_dir="data/tier1"
)

scheduler = DataScheduler(
    catalog=catalog,
    collector=collector,
    metrics_port=8000
)

# Run collection with comprehensive observability
scheduler.run_once()
```

### FRED Economic Data Integration

```python
# FRED economic data integration with comprehensive caching
from ml.data.loaders.fred_loader import FREDConfig, FREDDataLoader

fred_config = FREDConfig(
    cache_ttl_hours=24,
    backfill_years=10,
    rate_limit_calls=120  # Compliant with FRED API limits
)
fred_loader = FREDDataLoader(fred_config)

# Store with DataRegistry integration and correlation tracking
fred_loader.store_indicators(data_store, data_registry)

# Real-time updates
fred_loader.update_realtime(data_store, data_registry)

# FRED data joining with configurable publication lag
from ml.data.fred_join import join_fred_asof

# Join macro features with proper as-of semantics
dataset_with_macro = join_fred_asof(
    dataset,
    timestamp_col="timestamp",
    lag_days=1,  # Publication lag
    fred_path="data/fred/fred_indicators_ml_format.parquet"
)
```

## Advanced Features and Caching

### L2 and Microstructure Feature Caching

```python
# L2 per-minute feature caching for efficient dataset building
from ml.data.l2_cache import L2MinuteCache
from datetime import datetime, timezone

l2_cache = L2MinuteCache(cache_dir=Path("data/features/l2_minute"))
start = datetime(2025, 8, 11, tzinfo=timezone.utc)
end = datetime(2025, 8, 18, tzinfo=timezone.utc)

# Cached L2 features with on-demand computation
l2_features = l2_cache.get_range(
    symbol="SPY",
    start=start,
    end=end,
    raw_base_dir=Path("data/tier1")
)

# Microstructure feature caching
from ml.data.micro_cache import MicroMinuteCache

micro_cache = MicroMinuteCache(cache_dir=Path("data/features/micro_minute"))
micro_features = micro_cache.get_range("SPY", start, end, raw_base_dir)
```

### Build Pipeline Orchestration

```python
# Parallel dataset building with build_runner
from ml.pipelines.build_runner import BuildConfig, execute
from pathlib import Path

# Configuration for large-scale dataset building
config = BuildConfig(
    data_dir=Path("data/tier1"),
    out_dir=Path("./tft_datasets"),
    symbols=["SPY", "QQQ", "AAPL", "MSFT", "NVDA"],  # Can handle 100+ symbols
    workers=4,  # Parallel processing
    include_macro=True,
    include_micro=True,
    include_l2=True,
    horizon_minutes=15,
    lookback_periods=60,
    register_features=True
)

# Execute with progress tracking and metrics
results = execute(config)
print(f"Success: {results['succeeded']}, Failed: {results['failed']}")
```

### Feature Manifest Export

```python
# Export feature manifests for registry integration
from ml.data.feature_manifest_export import export_feature_manifest, FeatureExportConfig
from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureRole

config = FeatureExportConfig(
    registry_path=Path("~/.nautilus/ml/features"),
    role=FeatureRole.TEACHER,
    data_requirements=DataRequirements.L1_ONLY,
    version="2.0.0"
)

# Register computed features with hash-based versioning
feature_set_id = export_feature_manifest(
    feature_names=feature_columns,
    feature_dtypes=["float32"] * len(feature_columns),
    flags={"include_macro": True, "lookback_periods": 60},
    cfg=config
)
```

---

## Universal ML Architecture Pattern Compliance Analysis

**Analysis Date**: 2025-09-16
**Scope**: Comprehensive validation of Universal ML Architecture Pattern adherence

### Pattern Compliance Assessment

**Pattern 1: Mandatory 4-Store + 4-Registry Integration**

✅ **APPROPRIATELY EXEMPT**: Data layer components are cold path utilities, not ML actors
- **Rationale**: ml/data provides foundational data utilities that serve ML actors, not actors themselves
- **Partial Integration**: FeatureStore integration in TFTDatasetBuilder, DataStore in FRED loader
- **Proper Scope**: Components focus on data collection, processing, and preparation for ML actors

**Pattern 2: Protocol-First Interface Design**

✅ **COMPLIANT**: Strong protocol-based architecture in providers
- **Implementation**: `DataProvider`, `StaticDataProvider`, `TimeSeriesProvider` protocols
- **Location**: `/ml/data/providers/base.py` with runtime-checkable protocols
- **Benefits**: Enables structural typing and duck typing for testing
- **Enhancement Opportunity**: Could expand protocol usage to additional components

**Pattern 3: Hot/Cold Path Separation**

✅ **FULLY COMPLIANT**: Exclusively cold path operations
- **Design Intent**: Data collection, processing, and caching are inherently cold path
- **No Hot Path Constraints**: Appropriate use of DataFrames, file I/O, heavy computation
- **Clear Documentation**: Module docstring explicitly states cold path focus
- **Performance Appropriate**: Uses efficient processing (Polars) without hot path constraints

**Pattern 4: Progressive Fallback Chains**

✅ **APPROPRIATELY IMPLEMENTED**: Graceful degradation where applicable
- **DataRegistry Fallback**: PostgreSQL → JSON backend fallback implemented
- **Dependency Handling**: Optional dependency checks with clear error messages
- **Scope-Appropriate**: Circuit breakers not required for cold path data utilities
- **Progressive Degradation**: Components handle missing dependencies gracefully

**Pattern 5: Centralized Metrics Bootstrap**

✅ **MOSTLY COMPLIANT**: Consistent use of metrics_bootstrap
- **Primary Components**: DataScheduler and FRED loader use `ml.common.metrics_bootstrap`
- **No Direct Imports**: No direct prometheus_client usage found
- ⚠️ **Minor Inconsistency**: build_runner.py uses MetricsManager instead of metrics_bootstrap
- **Overall Pattern**: Strong adherence to centralized metrics approach

### Implementation Quality Assessment

#### Core Strengths

**Data Collection & Processing**
- ✅ **DataCollector**: Configurable storage management with Databento integration
- ✅ **TFTDatasetBuilder**: Dual-source architecture (FeatureStore + direct computation)
- ✅ **FRED Integration**: Complete economic data loader with caching and rate limiting
- ✅ **Feature Caching**: Efficient L2 and microstructure per-minute caches

**Architecture & Design**
- ✅ **Protocol-First Design**: Well-implemented provider architecture
- ✅ **Public API**: Clean `__init__.py` with proper exports and lazy imports
- ✅ **Separation of Concerns**: Clear distinction between sources, providers, and builders
- ✅ **Nautilus Integration**: Effective use of ParquetDataCatalog and native types

**Observability & Operations**
- ✅ **Metrics Integration**: Proper use of metrics_bootstrap in key components
- ✅ **Structured Logging**: Appropriate logging with context information
- ✅ **Error Handling**: Per-symbol error isolation with detailed logging
- ✅ **Progress Tracking**: JSONL progress logs with resumable execution

#### Areas for Enhancement

**Consistency Improvements**
- ⚠️ **Metrics Standardization**: Ensure all components use metrics_bootstrap uniformly
- ⚠️ **Protocol Expansion**: Could extend Protocol usage to additional components

**Documentation Clarity**
- ⚠️ **Pattern Exemption**: Better document why data utilities are exempt from ML actor patterns
- ⚠️ **Scope Emphasis**: More clearly emphasize cold path nature and appropriate scope

### Architectural Assessment

**Design Excellence:**
- Clean separation between data utilities and ML actors
- Appropriate cold path patterns with efficient processing
- Well-designed provider architecture with Protocol-based interfaces
- Effective integration with Nautilus native components

**Production Readiness:**
- Functional data collection and processing capabilities
- Proper error handling and observability integration
- Progressive fallback strategies where applicable
- Comprehensive feature integration (macro, micro, L2, events)

**Universal Pattern Alignment:**
- Appropriate exemptions for data utilities vs ML actors
- Strong compliance with applicable patterns
- Good architectural foundations for ML workflow support

### Final Assessment

**Pattern Compliance**: 90/100 (appropriate exemptions with strong compliance where applicable)
**Implementation Quality**: 85/100 (solid functionality with good architectural patterns)
**Documentation Accuracy**: 85/100 (accurate representation of capabilities and scope)
**Scope Appropriateness**: 95/100 (well-designed for cold path data pipeline role)

The ml/data module provides a solid, well-architected foundation for ML data workflows. It appropriately follows Universal ML Architecture Patterns where applicable while correctly operating as cold path utilities that serve ML actors rather than being ML actors themselves.

---

**Document Version**: 4.0
**Last Updated**: 2025-09-16
**Maintainer**: ML Data Pipeline Team
**Status**: Implementation Complete - Cold Path Data Utilities
**Changes**: Comprehensive accuracy review and Universal ML Architecture Pattern compliance analysis. Corrected documentation to accurately reflect current implementation state, clarified appropriate pattern exemptions for data utilities, and aligned with cold path design patterns.


### Canonical Ingestion Path and Dual-Write Guidance

- Prefer a single canonical ingestion writer for raw datasets per run: either SQL via `SqlMarketDataWriter` (canonical `market_data`)
  or Parquet via `ParquetCatalogRawWriter` (for cold-path training convenience), orchestrated by DataStore for validation + eventing.
- Avoid configuring multiple raw writers to the same dataset concurrently to prevent duplicated artifacts.
- When DataStore is configured with a raw writer, it emits events and updates watermarks only after successful writes.
