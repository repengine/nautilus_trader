# ML Data Module Context Document

## Executive Summary

The `ml/data/` module provides comprehensive data pipeline infrastructure for machine learning workflows within Nautilus Trader. This cold-path module handles data collection, ingestion, processing, and dataset preparation. The system integrates with Nautilus native components while providing ML-specific capabilities for feature engineering and training data creation.

**Module Size**: 14M, 60+ Python files across 5 subdirectories
**Core Purpose**: Cold-path data utilities serving ML actors (not ML actors themselves)
**Implementation Status**: Production-ready with comprehensive ingestion, caching, and dataset building

**Key Components:**

- **Data Ingestion** (`ingest/`): Robust historical and streaming data collection with Databento integration, resume/retry, policy enforcement, and dual-write support
- **Data Loaders** (`loaders/`): FRED economic indicators, ALFRED vintage data, Fama-French factors, recent OHLCV backfills, alternative data
- **TFT Dataset Builder**: Fast training dataset preparation with dual-source architecture (FeatureStore priority, direct computation fallback), macro/micro/L2/events integration
- **Provider Architecture** (`providers/`): Protocol-based SOLID design for metadata, calendar, and event data
- **Data Sources** (`sources/`): Mock and production implementations for testing and deployment
- **Feature Caching**: L2 and microstructure per-minute aggregation caches for efficient training
- **Data Scheduler**: Automated daily collection and processing with comprehensive observability
- **Fixtures**: Deterministic test data generation with schema validation

**Timestamp Policy**: All pipeline timestamps are UNIX nanoseconds. Store write paths perform defensive normalization from seconds/ms/us to ns and log a warning if triggered.

**DB Readiness**: Apply canonical migrations and run DB preflight before ingestion/backfills to ensure required functions and partitions exist.

---

## Quick Reference

### Public API (`ml.data.__init__.py`)

```python
# High-level orchestrators
from ml.data import (
    DataCollector,           # Rich market data collection
    DataScheduler,           # Automated daily scheduling
    TFTDatasetBuilder,       # Fast TFT dataset building
)

# Data conversion utilities
from ml.data import (
    bars_to_dataframe,       # Convert catalog bars to Polars DataFrame
    quotes_to_dataframe,     # Convert quotes to DataFrame
    trades_to_dataframe,     # Convert trades to DataFrame
)

# Performance & caching
from ml.data import (
    L2MinuteCache,           # L2 order book per-minute cache
    MicroMinuteCache,        # Microstructure per-minute cache
)

# Data providers (protocol-based)
from ml.data import (
    InstrumentMetadataProvider,
    MarketCalendarProvider,
    EventScheduleProvider,
)

# Data sources (testing & mocking)
from ml.data import (
    MockCalendarSource,
    SimpleCalendarSource,
    PandasCalendarSource,
    MockEventSource,
    DatabentoMetadataSource,
    NautilusMetadataSource,
)

# Data loaders
from ml.data import (
    FREDDataLoader,          # FRED economic data
    FREDConfig,
    FREDIndicator,
    ALFREDDataLoader,        # ALFRED vintage data
    ALFREDConfig,
)

# Fixtures & testing
from ml.data import (
    FixtureManifest,
    make_mbp10_fixture,
    make_tbbo_fixture,
    make_trades_fixture,
)

# Ingestion utilities
from ml.data import (
    DatabentoIngestionService,
    IngestionOrchestrator,
    IngestionRequest,
    BackoffPolicy,
    IngestState,
    RateLimiter,
)

# Dataset building
from ml.data import (
    build_tft_dataset,
    DatasetBuildConfig,
    BuildResult,
    DatasetMetadata,
    validate_dataset,
)
```

### Environment Variables

**Data APIs:**
- `DATABENTO_API_KEY` — Databento client (optional for local dev; required for real API tests)
- `FRED_API_KEY` — FRED economic data loader (required to fetch real data)

**Database (tests and registry helpers):**
- `DATABASE_URL` — primary Postgres URL
- `ML_DATABASE_URL` — alias used by tools (mirrors `DATABASE_URL`)
- `NAUTILUS_REGISTRY_DB_URL` — alias for registry helpers

**Feature Flags:**
- `ML_USE_LEGACY_DATA_SCHEDULER=1` — Use original god class implementation instead of component-based facade

---

## Architecture Overview

The data pipeline follows a layered architecture with clear separation of concerns:

```
Raw Data Layer
├── Data Ingestion (ingest/)
│   ├── DatabentoIngestionService: Policy-enforced historical ingestion
│   ├── IngestionOrchestrator: Gap detection + backfill coordination
│   ├── DatabentoIngestor: Resume/retry with DST-aware window planning
│   └── Subscription Management: Dataset/schema allowlists, lookback limits
├── ParquetDataCatalog: Nautilus native storage
└── Databento Integration: Real-time market data

Processing Layer
├── TFTDatasetBuilder: Training data preparation (2,729 lines)
│   ├── Dual-source: FeatureStore priority → direct computation fallback
│   ├── Market bindings: Multi-feed resolution with lineage tracking
│   ├── Macro integration: FRED/ALFRED with vintage policies
│   ├── Micro/L2 integration: Cached feature aggregation
│   └── Events integration: Corporate actions, earnings, calendars
├── DataScheduler: Automated orchestration (1,542 lines)
│   ├── Component-based facade (default) or legacy god class
│   ├── Registry integration: Event emission, watermark updates
│   └── Comprehensive metrics: 15+ Prometheus metrics
├── Feature Caches: L2/Micro per-minute pre-computed features
└── Data Loaders: FRED, ALFRED, Fama-French, recent OHLCV

Integration Layer
├── DataStore: Unified persistence facade with validation
├── DataRegistry: Event emission and lineage tracking
├── FeatureStore: Training/inference parity
└── Provider Architecture: Protocol-based metadata/calendar/events

Provider Layer (Protocol-First Design)
├── Base Providers: SOLID architecture with caching
├── Calendar Sources: Market schedule data (mock/pandas/simple)
├── Metadata Sources: Instrument information (mock/databento/nautilus)
└── Event Sources: Corporate actions/earnings (mock/file)
```

---

## Directory Structure

```
ml/data/
├── __init__.py                    # Public API (1,944 lines) with lazy imports
├── catalog_utils.py               # Nautilus catalog conversion utilities
├── collector.py                   # DataCollector (828 lines) - multi-tier collection
├── scheduler.py                   # DataScheduler facade (1,542 lines)
├── scheduler_legacy.py            # Legacy god class implementation (1,551 lines)
├── tft_dataset_builder.py         # TFT builder (2,729 lines) - dual-source architecture
├── tft_dataset_builder_legacy.py  # Legacy TFT implementation (2,208 lines)
├── registry_integrator.py         # Registry integration for schedulers
├── fred_join.py                   # FRED as-of join utilities (616 lines)
├── macro_revisions.py             # Macro revision analysis (369 lines)
├── vintage.py                     # Vintage policy and datetime utilities
├── validation.py                  # Dataset validation logic
├── l2_cache.py                    # L2 per-minute cache
├── micro_cache.py                 # Microstructure per-minute cache
├── cache_common.py                # Shared caching utilities
├── feature_manifest_export.py     # Feature registry export
├── dataset_manifest_defaults.py   # Dataset metadata helpers (451 lines)
├── feature_computation_manager.py # Feature computation coordinator (348 lines)
├── trading_day_calculator.py      # Trading day utilities
├── collection_coordinator.py      # Collection orchestration (817 lines)
├── data_retention_manager.py      # Data retention logic
├── initialization_manager.py      # Startup initialization
│
├── ingest/                        # Data ingestion subsystem
│   ├── __init__.py
│   ├── orchestrator.py            # Gap detection + backfill (721 lines)
│   ├── service.py                 # Databento ingestion service (1,161 lines)
│   ├── resume.py                  # Resume/retry ingestion logic
│   ├── common.py                  # Backoff policy, rate limiter, ingest state
│   ├── subscription.py            # Subscription policy enforcement (611 lines)
│   ├── discovery.py               # Dataset discovery service (529 lines)
│   ├── policy.py                  # Coverage policies
│   ├── symbology.py               # Symbol resolution
│   ├── databento_adapter.py       # Databento client adapter
│   ├── nautilus_adapters.py       # Nautilus adapters
│   ├── yfinance_adapter.py        # yfinance adapter
│   ├── l2_efficient.py            # Efficient L2 gap fill (654 lines)
│   ├── dbn_archive.py             # DBN archive utilities (398 lines)
│   ├── macro_refresh.py           # Macro data refresh (344 lines)
│   ├── market_bindings.py         # Market dataset binding resolution
│   ├── metrics.py                 # Ingestion metrics
│   ├── state.py                   # Ingestion state management
│   └── api.py                     # Low-level API utilities
│
├── loaders/                       # Data loaders
│   ├── __init__.py
│   ├── fred_loader.py             # FRED economic data (1,123 lines)
│   ├── alfred_loader.py           # ALFRED vintage data
│   ├── ohlcv_recent.py            # Recent OHLCV backfills (431 lines)
│   ├── fama_french_loader.py      # Fama-French factors
│   ├── alternative.py             # Alternative data sources
│   └── supplementary.py           # Supplementary data (spreads, correlations)
│
├── providers/                     # Data provider architecture
│   ├── __init__.py
│   ├── base.py                    # Protocol-based base classes (621 lines)
│   ├── factory.py                 # Provider factory (527 lines)
│   ├── metadata.py                # Instrument metadata provider
│   ├── calendar.py                # Market calendar provider
│   ├── events.py                  # Event schedule provider (516 lines)
│   └── utils.py                   # Provider utilities
│
├── sources/                       # Data sources (mock & real)
│   ├── __init__.py
│   ├── metadata.py                # Metadata sources (446 lines)
│   ├── calendar.py                # Calendar sources (769 lines)
│   └── events.py                  # Event sources (916 lines)
│
├── earnings/                      # Earnings data subsystem
│   ├── __init__.py
│   ├── earnings_cache.py          # Earnings cache (494 lines)
│   ├── yahoo_fetcher.py           # Yahoo Finance fetcher (329 lines)
│   ├── edgar_fetcher.py           # SEC EDGAR fetcher (390 lines)
│   └── xbrl_parser.py             # XBRL parser
│
└── fixtures/                      # Test fixtures
    ├── __init__.py
    ├── databento_fixtures.py      # Databento fixture generators
    └── manifest.py                # Fixture manifest management
```

---

## Component Details

### 1. Data Ingestion System (`ingest/`)

The ingestion subsystem provides robust historical and streaming data collection with comprehensive safety checks and observability.

#### 1.1 Databento Ingestion Service (`service.py`, 1,161 lines)

**Purpose**: Centralized entry point for all historical data ingestion with safety checks, cost estimation, and streaming interface.

**Class**: `DatabentoIngestionService`

**Key Features:**
- **Policy Enforcement**: Dataset/schema allowlists, lookback limits from `ml.config.databento_policy`
- **Cost Estimation**: Pre-ingestion cost checks with configurable thresholds
- **Symbol Resolution**: Databento symbology client integration
- **Rate Limiting**: Configurable per-minute API call limits
- **Streaming Interface**: Yields `IngestionChunk` objects for flexible persistence
- **Metrics**: Prometheus counters and histograms via `ml.common.metrics_bootstrap`

**I/O Specifications:**
- **Input**: `IngestionRequest` with dataset, schema, symbols, time window, cost limits
- **Output**: Sequence of `IngestionChunk` (symbol, window, DataFrame)

**Usage Example:**
```python
from ml.data.ingest.service import DatabentoIngestionService, IngestionRequest
from datetime import datetime, UTC

service = DatabentoIngestionService.from_env()
request = IngestionRequest(
    dataset="EQUS.MINI",
    schema="trades",
    symbols=("SPY",),
    start=datetime(2025, 8, 1, tzinfo=UTC),
    end=datetime(2025, 8, 2, tzinfo=UTC),
)
results = service.ingest(request)  # Generator of IngestionChunk
```

**Safety Features:**
- Dataset allowlist validation (line 264-268)
- Schema safety checks (line 270-281)
- Lookback limit enforcement (line 283-298)
- Cost estimation and violation detection (line 300-320)

**Location**: `/ml/data/ingest/service.py`

#### 1.2 Ingestion Orchestrator (`orchestrator.py`, 721 lines)

**Purpose**: Coordinates gap detection, backfill, and live streaming integration with registry events and watermark updates.

**Class**: `IngestionOrchestrator`

**Key Features:**
- **Gap Detection**: Uses `CoverageProviderProtocol` to identify missing day-buckets
- **Backfill Coordination**: Splits large windows into schema-appropriate chunks
- **Dual-Write Support**: Optional ParquetDataCatalog write via `RawIngestionWriterProtocol`
- **Registry Integration**: Emits CATALOG_WRITTEN events with SHA256 correlation IDs
- **Watermark Updates**: Advances watermarks for data freshness tracking
- **Market Bindings**: Resolves multi-feed inputs using `resolve_market_dataset_bindings`

**I/O Specifications:**
- **Input**: `ResolvedMarketBinding`, lookback days, IngestState
- **Output**: `BackfillWindowList` with persisted windows, frames written, rows written

**Usage Example:**
```python
from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.stores.providers import SqlCoverageProvider, SqlMarketDataWriter
from ml.registry.data_registry import DataRegistry
from pathlib import Path

coverage = SqlCoverageProvider(connection_string=DB_URL, table_name="market_data")
writer = SqlMarketDataWriter(connection_string=DB_URL, table_name="market_data")
registry = DataRegistry(registry_path=Path("/tmp/registry"))

orch = IngestionOrchestrator(
    coverage=coverage,
    writer=writer,
    registry=registry,
    ingestor=databento_ingestor,
)

# Backfill gaps in the last 7 days
gaps = orch.backfill_gaps(
    dataset_id="GLBX.MDP3",
    schema="tbbo",
    instrument_id="ES-USD-FUT.CME",
    lookback_days=7,
    state=IngestState(),
)
```

**Key Methods:**
- `backfill_binding()`: Backfill a single market binding (line 196-250)
- `backfill_gaps()`: Detect and fill gaps (uses coverage provider)
- `start_live()`: Hook for live streaming integration

**Location**: `/ml/data/ingest/orchestrator.py`

#### 1.3 Databento Ingestor (`resume.py`)

**Purpose**: Resume/retry ingestion with DST-aware window planning and backoff policies.

**Class**: `DatabentoIngestor`

**Key Features:**
- **Resume from Last Timestamp**: Stateful per-instrument resume
- **Retry/Backoff**: Configurable policy with injectable sleep function (testable)
- **DST-Aware Window Planning**: Uses `zoneinfo` for accurate daily windows
- **Metrics Emission**: Via `ml.data.ingest.metrics`

**Usage Example:**
```python
from ml.data.ingest.resume import DatabentoIngestor, IngestState, BackoffPolicy
from datetime import date

ingestor = DatabentoIngestor(client=databento_client, policy=BackoffPolicy())
state = IngestState()

# Plan DST-aware daily windows
windows = ingestor.plan_daily_windows(
    start_date=date(2021, 3, 13),
    end_date=date(2021, 3, 16),
    tz="America/New_York",
)

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

**Location**: `/ml/data/ingest/resume.py`

#### 1.4 Subscription Management (`subscription.py`, 611 lines)

**Purpose**: Enforce subscription policies and lookback limits based on Databento tiers.

**Key Classes:**
- `SubscriptionPolicy`: Tier-based limits (BASIC, STANDARD, PROFESSIONAL, ENTERPRISE)
- `SubscriptionChecker`: Validates requests against policies

**Key Functions:**
- `get_effective_policy(dataset: str, schema: str)`: Returns active policy
- `get_max_lookback_days(dataset: str, schema: str)`: Returns max lookback

**Location**: `/ml/data/ingest/subscription.py`

#### 1.5 Dataset Discovery (`discovery.py`, 529 lines)

**Purpose**: Discover available datasets and schemas from Databento metadata API.

**Class**: `DatasetDiscoveryService`

**Key Features:**
- Cost estimation for discovered inputs
- Schema-level discovery with storage kind classification
- Symbol resolution via symbology client

**Location**: `/ml/data/ingest/discovery.py`

---

### 2. TFT Dataset Builder (`tft_dataset_builder.py`, 2,729 lines)

**Purpose**: Fast training dataset preparation with comprehensive feature integration and dual-source architecture.

**Class**: `TFTDatasetBuilder`

**Key Architecture:**
- **Dual-Source**: FeatureStore priority → direct computation fallback
- **Market Bindings**: Multi-feed resolution with lineage tracking (lines 100-150)
- **Macro Integration**: FRED/ALFRED with vintage policies (lines 500-600)
- **Micro/L2 Integration**: Cached feature aggregation (lines 700-800)
- **Events Integration**: Corporate actions, earnings, calendars (lines 900-1000)

**I/O Specifications:**
- **Input**:
  - `catalog`: ParquetDataCatalog for raw data access
  - `symbols`: List of symbols to include
  - `instrument_ids`: Optional explicit instrument IDs
  - `feature_store`: Optional FeatureStore for training/inference parity
  - `data_store`: Optional DataStore for OHLCV loading via canonical storage
  - `market_dataset_id`: Dataset ID for binding resolution
  - `market_bindings`: Pre-resolved bindings
  - Feature flags: `include_macro`, `include_micro`, `include_l2`, `include_events`, `include_calendar`, `include_earnings`
  - Vintage options: `vintage_policy`, `vintage_as_of`, `include_macro_revisions`
  - Paths: `fred_path`, `vintage_base_dir`, `events_base_dir`
  - `student_mode`: L1-only real-time parity mode
- **Output**: TFT-compatible DataFrame (Pandas or Polars) with comprehensive feature set

**Key Methods:**

1. **`build_training_dataset()`** (line 800-900)
   - Legacy compatibility wrapper
   - Automatic source selection with error recovery
   - Supports both Polars and Pandas output formats

2. **`prepare_training_data()`** (line 600-700)
   - Intelligent source selection: FeatureStore → Direct computation
   - Comprehensive logging for monitoring source selection decisions

3. **`prepare_training_data_from_store()`** (line 500-600)
   - Loads features from FeatureStore for strict training/inference parity
   - Combines FeatureStore features with bar data for target generation
   - Adds TFT-specific features (static covariates, known-future features)

4. **`_build_training_dataset_direct()`** (line 1200-1400)
   - Direct feature computation with venue fallback
   - Multi-venue symbol resolution (ARCA, NASDAQ, NYSE, etc.)
   - Parquet file fallback for missing catalog entries
   - Robust error handling with per-symbol failure isolation

5. **`_add_known_future_features_polars()`** (line 1500-1600)
   - Adds calendar features (day of week, month, hour cyclic encoding)
   - Event-based known-future features if `include_events=True`
   - Earnings-based features if `include_earnings=True`

6. **`get_binding_stats()`** (line 200-250)
   - Returns `MarketBindingStats` for each binding
   - Tracks ts_event range, row counts from store/catalog, source datasets

**Macro Integration (lines 400-500):**
- FRED indicators via `fred_join.py` with configurable publication lag
- ALFRED vintage data with point-in-time accuracy
- Revision-aware columns: `<series>`, `<series>__value_real_time`, `<series>__value_final`, `<series>__value_vintage_ts`
- Vintage policies: `REAL_TIME` (as-of cutoff) vs `FINAL` (latest revised values)

**L2/Micro Integration (lines 700-800):**
- L2MinuteCache: Day-partitioned L2 aggregates (`data/features/l2_minute/<SYMBOL>/year=YYYY/month=MM/day=DD.parquet`)
- MicroMinuteCache: Day-partitioned microstructure aggregates (`data/features/micro_minute/<SYMBOL>/year=YYYY/month=MM/day=DD.parquet`)
- On-demand computation if cache partitions missing

**Student Mode (line 150-200):**
- L1-only features for real-time inference parity
- Skips macro, micro, L2, events to match production constraints
- Explicitly documents data requirements

**Dataset Metadata (lines 2500-2700):**
- Emits `dataset_metadata.json` describing:
  - Vintage policy and cutoff
  - Canonical ts_event start/end
  - Overall/train/validation/test windows
  - Macro observation counts
  - Market binding lineage (dataset, schema, storage kind, ts_event ranges, row counts)

**Usage Example:**
```python
from ml.data import TFTDatasetBuilder
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from datetime import datetime, UTC

catalog = ParquetDataCatalog("./data")
builder = TFTDatasetBuilder(
    catalog=catalog,
    symbols=["SPY", "QQQ", "AAPL"],
    include_macro=True,
    macro_lag_days=1,
    include_micro=True,
    include_l2=True,
    include_events=True,
    student_mode=False,
)

dataset = builder.build_training_dataset(
    horizon_minutes=15,
    min_return_threshold=0.001,
    lookback_periods=30,
    use_polars=True,
    start=datetime(2024, 1, 1, tzinfo=UTC),
    end=datetime(2024, 12, 31, tzinfo=UTC),
)
```

**Location**: `/ml/data/tft_dataset_builder.py`

---

### 3. Data Loaders (`loaders/`)

#### 3.1 FRED Loader (`fred_loader.py`, 1,123 lines)

**Purpose**: Complete economic indicator integration with comprehensive observability.

**Class**: `FREDDataLoader`

**Key Features:**
- **Comprehensive Indicators**: 20+ economic indicators (rates, volatility, GDP, CPI, employment)
- **Caching**: Configurable TTL with metadata persistence
- **Rate Limiting**: FRED API compliance (120 calls/minute)
- **DataStore Integration**: Proper instrument ID generation and storage
- **DataRegistry Events**: Manifest registration with schema validation
- **Observability**: Dedicated Prometheus metrics (fetch counter, duration histogram, cache hits, errors)

**I/O Specifications:**
- **Input**: `FREDConfig` with API key, cache settings, rate limits, backfill years
- **Output**: Polars DataFrame with columns `timestamp`, `series_id`, `value`

**Key Methods:**
- `fetch_series(series_id: str)`: Fetch single series with retry/backoff
- `fetch_all_indicators()`: Fetch all configured indicators
- `export_ml_parquet(out_path: Path)`: Export ML-format parquet for dataset builder
- `store_indicators(data_store, data_registry)`: Persist to DataStore with events

**Default Indicators (lines 250-400):**
- Interest rates: DGS10, DGS2, DFF, T10Y2Y
- Volatility: VIXCLS
- Economic: GDP, CPIAUCSL, UNRATE, PAYEMS, UMCSENT
- Market breadth: NASDAQCOM, SP500
- Currency: DEXUSEU, DEXJPUS

**Usage Example:**
```python
from ml.data.loaders.fred_loader import FREDDataLoader, FREDConfig

config = FREDConfig(
    cache_ttl_hours=24,
    backfill_years=10,
    rate_limit_calls=120,
)
loader = FREDDataLoader(config)

# Fetch all indicators
indicators_df = loader.fetch_all_indicators()

# Export ML-format parquet
loader.export_ml_parquet(Path("data/fred/fred_indicators_ml_format.parquet"))

# Store with DataRegistry integration
loader.store_indicators(data_store, data_registry)
```

**Location**: `/ml/data/loaders/fred_loader.py`

#### 3.2 ALFRED Loader (`alfred_loader.py`)

**Purpose**: Fetch and persist ALFRED (vintage FRED) releases for point-in-time accuracy.

**Class**: `ALFREDDataLoader`

**Key Features:**
- Per-series vintage snapshots under `data/fred/vintages/<series>/<yyyymmdd>.parquet`
- Normalized `release_calendar.parquet` for strict point-in-time joins
- Retry/backoff for API failures
- Prometheus metrics (fetch counter, error counter, duration histogram)

**I/O Specifications:**
- **Input**: `ALFREDConfig` with series IDs, output directory, date range, window_days
- **Output**: Parquet files with columns `series_id`, `observation_ts`, `value`, `release_ts`, `release_end_ts`

**Usage Example:**
```python
from ml.data.loaders.alfred_loader import ALFREDDataLoader, ALFREDConfig

config = ALFREDConfig(
    series_ids=("GDP", "CPIAUCSL", "UNRATE"),
    out_dir=Path("data/fred/vintages"),
    start_date="2020-01-01",
    end_date="2025-01-01",
)
loader = ALFREDDataLoader(config)
stats = loader.refresh()  # Returns dict[series_id, {releases, rows}]
```

**Location**: `/ml/data/loaders/alfred_loader.py`

#### 3.3 Recent OHLCV Backfill (`ohlcv_recent.py`, 431 lines)

**Purpose**: Safely backfill recent minute bars to `data/tier1/<SYMBOL>/l0/<SYMBOL>_ohlcv.parquet`.

**Function**: `backfill_recent_ohlcv(config: OhlcvRecentBackfillConfig)`

**Key Features:**
- Streaming PyArrow row-group merges (no full file in RAM)
- Resume/progress tracking per symbol
- Rate limiting and throttling
- Signal-safe flush on SIGINT/SIGTERM

**Usage Example:**
```python
from ml.data.loaders.ohlcv_recent import backfill_recent_ohlcv, OhlcvRecentBackfillConfig

config = OhlcvRecentBackfillConfig(
    tier=1,
    days=14,
    max_symbols=5,
    symbol_offset=0,
    rate_limit=10,
    sleep_between_symbols=1.0,
)
result = backfill_recent_ohlcv(config)
print(f"Success: {result.succeeded}, Failed: {result.failed}")
```

**Location**: `/ml/data/loaders/ohlcv_recent.py`

#### 3.4 Fama-French Loader (`fama_french_loader.py`)

**Purpose**: Download and parse Fama-French factor datasets.

**Function**: `download_fama_french_dataset(spec: FamaFrenchDatasetSpec)`

**Supported Datasets:**
- FF3: Fama-French 3-factor (Mkt-RF, SMB, HML)
- FF5: Fama-French 5-factor (adds RMW, CMA)
- Momentum: UMD factor
- Industry portfolios

**Location**: `/ml/data/loaders/fama_french_loader.py`

#### 3.5 Alternative Data (`alternative.py`)

**Purpose**: Load and persist alternative data sources.

**Functions:**
- `populate_alternative_data(config: AlternativeDataConfig)`
- `load_tier1_symbols(tier: int)`

**Supported Sources:**
- SENTIMENT
- NEWS
- SOCIAL
- FUNDAMENTAL

**Location**: `/ml/data/loaders/alternative.py`

---

### 4. Provider Architecture (`providers/`)

The provider subsystem implements **Pattern 2: Protocol-First Interface Design** with SOLID principles.

#### 4.1 Base Classes (`base.py`, 621 lines)

**Protocols:**
- `DataProvider`: Base protocol for all providers
- `StaticDataProvider`: Static data with indefinite caching
- `TimeSeriesProvider`: Time-varying data with validation

**Abstract Base Classes:**
- `BaseDataProvider`: Common functionality (logging, metrics, validation)
- `CachedDataProvider`: Template method with caching logic
- `BaseStaticProvider`: Static data implementation
- `BaseTimeSeriesProvider`: Time-series data implementation

**Key Features:**
- Structural typing without implementation coupling
- Duck typing support for testing (DummyProviders conform to protocols)
- Type safety without circular dependencies
- Clear contracts for component interactions

**Location**: `/ml/data/providers/base.py`

#### 4.2 Provider Factory (`factory.py`, 527 lines)

**Class**: `ProviderFactory`

**Purpose**: Singleton factory for creating and managing data providers with dependency injection.

**Key Features:**
- **Singleton Pattern**: Same provider instance reused throughout application
- **Progressive Fallback**: Real implementations → Mock providers when unavailable
- **Open/Closed Principle**: Creator registry for extensibility

**Initialization:**
```python
from ml.data.providers.factory import ProviderFactory

factory = ProviderFactory(
    metadata_source=metadata_source,  # Optional; uses MockMetadataSource if None
    calendar_source=calendar_source,  # Optional; tries PandasCalendarSource → MockCalendarSource
    event_source=event_source,        # Optional; tries FileEventSource → MockEventSource
)
```

**Key Methods:**
- `get_metadata_provider()`: Returns `InstrumentMetadataProvider`
- `get_calendar_provider()`: Returns `MarketCalendarProvider`
- `get_event_provider()`: Returns `EventScheduleProvider`
- `register_provider_creator(name, creator)`: Register custom provider

**Location**: `/ml/data/providers/factory.py`

#### 4.3 Concrete Providers

**InstrumentMetadataProvider** (`metadata.py`):
- Static instrument information
- Default metadata via `ml.data.sources.metadata.default_metadata(symbol)`

**MarketCalendarProvider** (`calendar.py`):
- Trading calendar features
- Supports extended hours, holidays, early closes

**EventScheduleProvider** (`events.py`, 516 lines):
- Corporate actions and earnings sourced from `data/events/events.parquet`
- Populated by `ml.preprocessing.event_ingestion.EventIngestionUtility`

**Location**: `/ml/data/providers/{metadata,calendar,events}.py`

---

### 5. Data Sources (`sources/`)

#### 5.1 Calendar Sources (`calendar.py`, 769 lines)

**Implementations:**
- `MockCalendarSource`: Testing with realistic market schedules
- `SimpleCalendarSource`: Basic NYSE schedule with fixed hours
- `PandasCalendarSource`: Production-ready with `pandas_market_calendars`

**PandasCalendarSource Features:**
- Exchange coverage: NYSE, NASDAQ, CME, CBOT, ICE, CBOE, LSE, EUREX, JPX, HKEX, ASX, 20+ major exchanges
- Holiday integration with early close detection
- Extended hours: Pre-market (4:00-9:30 AM), after-hours (4:00-8:00 PM)
- 24/7 markets: Special handling for BINANCE, COINBASE
- Intelligent schedule caching with configurable TTL

**Location**: `/ml/data/sources/calendar.py`

#### 5.2 Metadata Sources (`metadata.py`, 446 lines)

**Implementations:**
- `MockMetadataSource`: Testing
- `DatabentoMetadataSource`: Databento metadata integration
- `NautilusMetadataSource`: Nautilus-native metadata

**Canonical Defaults:**
- `default_metadata(symbol: str) -> dict[str, Any]` (line 50-100)
- Used by Databento/Nautilus sources and provider empty-frame builders

**Location**: `/ml/data/sources/metadata.py`

#### 5.3 Event Sources (`events.py`, 916 lines)

**Implementations:**
- `MockEventSource`: Testing with simulated events
- `FileEventSource`: Production file-backed source

**Event Types:**
- Earnings releases
- Dividend announcements
- Stock splits
- Economic announcements (FOMC, CPI, NFP)

**File Format:**
- Single `events.parquet` archive in `data/events/`
- Populated by `ml.preprocessing.event_ingestion.EventIngestionUtility`

**Location**: `/ml/data/sources/events.py`

---

### 6. Feature Caching Infrastructure

#### 6.1 L2 Minute Cache (`l2_cache.py`)

**Class**: `L2MinuteCache`

**Purpose**: Efficient caching of L2 order book per-minute aggregates.

**Features:**
- Day-partitioned storage: `data/features/l2_minute/<SYMBOL>/year=YYYY/month=MM/day=DD.parquet`
- On-demand computation using `L2Aggregator` if cache partitions missing
- UTC timestamp handling with nanosecond precision
- Automatic cache directory creation

**Usage:**
```python
from ml.data import L2MinuteCache
from datetime import datetime, timezone
from pathlib import Path

cache = L2MinuteCache(Path("data/features/l2_minute"))
start = datetime(2025, 8, 11, tzinfo=timezone.utc)
end = datetime(2025, 8, 18, tzinfo=timezone.utc)

df = cache.get_range(
    symbol="SPY",
    start=start,
    end=end,
    raw_base_dir=Path("data/tier1"),
)
```

**Semantics:**
- Timestamp filter is half-open `[start, end)`
- Results sorted and cast to `Datetime[ns, UTC]`
- Missing partitions compute on demand and persist before returning

**Location**: `/ml/data/l2_cache.py`

#### 6.2 Microstructure Cache (`micro_cache.py`)

**Class**: `MicroMinuteCache`

**Purpose**: Caching of L1/L0-derived per-minute microstructure features.

**Features:**
- Same partitioning scheme as L2 cache
- Integration with `MicrostructureAggregator`
- Efficient date range queries with `[start, end)` semantics
- Memory-efficient concatenation of day partitions

**Usage:**
```python
from ml.data import MicroMinuteCache
from datetime import datetime, timezone
from pathlib import Path

cache = MicroMinuteCache(Path("data/features/micro_minute"))
df = cache.get_range("SPY", start, end, raw_base_dir=Path("data/tier1"))
```

**Location**: `/ml/data/micro_cache.py`

---

### 7. Data Scheduler (`scheduler.py`, 1,542 lines)

**Purpose**: Production-ready automated data collection and processing with comprehensive monitoring and resilience.

**Class**: `DataScheduler`

**Key Architecture:**
- **Component-Based Facade** (default): Delegates to specialized components
- **Legacy God Class** (`scheduler_legacy.py`): Available via `ML_USE_LEGACY_DATA_SCHEDULER=1`

**I/O Specifications:**
- **Input**:
  - `catalog`: ParquetDataCatalog for data storage
  - `config`: SchedulerConfig with Databento settings and feature store configuration
  - `collector`: DataCollector instance (optional, creates default if None)
  - `feature_engineer`: FeatureEngineer for feature computation (optional)
  - `metrics_port`: Port for Prometheus metrics server (default: 8000)
  - `connection`: Database connection string for feature store (optional)
- **Output**:
  - Updated ParquetDataCatalog with new market data
  - Computed features persisted to FeatureStore
  - DataRegistry events with SHA256 correlation IDs
  - Comprehensive Prometheus metrics
  - Watermark updates for data freshness

**Key Features:**

1. **DataRegistry Integration:**
   - Emits CATALOG_WRITTEN events for data lineage with SHA256 correlation IDs
   - Updates watermarks for dataset freshness and completeness tracking
   - Progressive fallback: PostgreSQL backend → JSON backend for development
   - Cross-domain event correlation for end-to-end pipeline tracing

2. **Comprehensive Metrics** (15+ Prometheus metrics):
   - `data_collection_latency`: Collection time by instrument and data type
   - `pipeline_stage_latency`: Stage execution times with percentile tracking
   - `catalog_write_operations_total`: Catalog write success/failure counters
   - `feature_store_operations_total`: Feature store operations with operation type labels
   - `data_staleness_seconds`: Data freshness tracking per instrument
   - `api_rate_limit_hits`: API rate limit monitoring with endpoint labels
   - `active_collection_tasks`: Real-time task monitoring
   - `data_retention_cleanup_total`: Cleanup operation tracking
   - `features_computed_total`: Feature computation counters by instrument and type

**Usage Example:**
```python
from ml.data.scheduler import DataScheduler
from ml.config.scheduler_config import SchedulerConfig
from nautilus_trader.persistence.catalog import ParquetDataCatalog

config = SchedulerConfig(
    symbols=["SPY.ARCA", "QQQ.NASDAQ"],
    retention_days=90,
)
catalog = ParquetDataCatalog("./data")
scheduler = DataScheduler(catalog=catalog, config=config)

# Run once (manual trigger)
scheduler.run_once()

# Or run daily update
scheduler.run_daily_update()
```

**Location**: `/ml/data/scheduler.py`

---

### 8. Fixtures and Testing (`fixtures/`)

**Purpose**: Deterministic, lightweight test data generation utilities.

**Key Functions:**
- `make_mbp10_fixture(instrument_id: str, rows: int)`: MBP-10 test data
- `make_tbbo_fixture(instrument_id: str, rows: int)`: TBBO test data
- `make_trades_fixture(instrument_id: str, rows: int)`: Trade test data

**Returns**: `(DataFrame, FixtureManifest)` where `FixtureManifest` records `schema_hash` and `content_sha256` for reproducibility.

**Usage:**
```python
from ml.data.fixtures import make_tbbo_fixture

df, manifest = make_tbbo_fixture(instrument_id="EURUSD.SIM", rows=60)
assert len(df) == 60
assert manifest.schema_hash is not None
```

**Test Coverage:**
- Contracts: `ml/tests/contracts/test_databento_fixtures_contracts.py`
- Property: `ml/tests/property/test_ingestion_watermark_properties.py`
- Performance: `ml/tests/performance/test_ingestion_microbench.py`

**Location**: `/ml/data/fixtures/`

---

### 9. Utilities and Helpers

#### 9.1 Catalog Utilities (`catalog_utils.py`)

**Functions:**
- `bars_to_dataframe(catalog, instrument_ids, start, end)`: Convert Nautilus bars to Polars DataFrame
- `quotes_to_dataframe(catalog, instrument_ids, start, end)`: Convert quotes to DataFrame
- `trades_to_dataframe(catalog, instrument_ids, start, end)`: Convert trades to DataFrame
- `resolve_instrument_id(symbol: str)`: Multi-venue symbol resolution (ARCA, NASDAQ, NYSE, etc.)

**Features:**
- Native Nautilus type handling with automatic conversion
- Memory-efficient data loading with chunking support
- Multi-venue instrument ID resolution
- Comprehensive error handling with per-instrument failure isolation

**Location**: `/ml/data/catalog_utils.py`

#### 9.2 FRED Join (`fred_join.py`, 616 lines)

**Function**: `join_fred_asof(dataset_df, timestamp_col, lag_days, fred_path)`

**Purpose**: Perform as-of join against ML-format FRED parquet with configurable publication lag.

**Features:**
- Publication lag handling via `lag_days` parameter
- Supports both Polars and Pandas DataFrames
- Joins columns: `timestamp`, `series_id`, `value`

**Location**: `/ml/data/fred_join.py`

#### 9.3 Vintage Policy (`vintage.py`)

**Enum**: `VintagePolicy`
- `REAL_TIME`: Use vintage data as-of cutoff
- `FINAL`: Use latest revised values

**Utilities:**
- `format_dt(dt: datetime | None) -> str | None`: ISO8601 formatting
- `parse_dt(value: str | None) -> datetime | None`: Parse ISO8601 strings

**Location**: `/ml/data/vintage.py`

#### 9.4 Dataset Validation (`validation.py`)

**Function**: `validate_dataset(df: PolarsDF, config: DatasetValidationConfig)`

**Checks:**
- Minimum row count
- Target positive rate bounds
- Feature coverage thresholds
- Required macro series presence
- Macro vintage observation counts

**Returns**: `DatasetValidationResult` with statistics

**Location**: `/ml/data/validation.py`

---

## Integration Points

### Nautilus Core
- Uses `ParquetDataCatalog` for data storage
- Native Nautilus types (`InstrumentId`, `Bar`, `QuoteTick`, `TradeTick`)
- Never modifies Nautilus core (`core/`, `model/`, `system/`)

### ML Stores
- `FeatureStore`: Training/inference parity via pre-computed features
- `DataStore`: Unified persistence facade with contract validation and event emission
- `ModelStore`: Predictions and model performance metrics
- `StrategyStore`: Strategy state and trading decisions

### Registry System
- `DataRegistry`: Event emission and lineage tracking with SHA256 correlation IDs
- `FeatureRegistry`: Feature schema validation and lifecycle management
- `ModelRegistry`: Model deployment tracking and A/B testing
- `StrategyRegistry`: Strategy compatibility and requirement validation

### Observability
- Comprehensive metrics via `ml.common.metrics_bootstrap`
- Structured logging via `structlog`
- Event correlation for end-to-end pipeline tracing

### Configuration
- `msgspec`-based configs with validation and freezing
- DatabentoSafetyConfig for policy enforcement

---

## Testing Strategy

### Property Tests
- Invariants: ordering, bounds, idempotency
- Use `hypothesis` with strategies capturing edge cases

### Contract/Schema Tests
- Pandera schemas for stores/events
- Fixture manifest schema validation

### Metamorphic Tests
- Feature/model relationships
- Cross-validation parity

### Pairwise Tests
- Config spaces

### E2E Tests
- `ml/tests/e2e/test_tft_dataset_builder_e2e.py`: Dataset builder validation
- `ml/tests/e2e/test_pipeline_orchestrator_e2e.py`: Orchestrator integration

### Integration Tests
- `ml/tests/integration/cli/`: CLI integration
- `ml/tests/integration/consumers/`: Consumer integration
- `ml/tests/integration/dashboard/`: Dashboard integration

### Performance Tests
- `ml/tests/performance/test_ingestion_microbench.py`: Ingestion benchmarks
- `ml/tests/performance/test_streaming_persistence_microbench.py`: Streaming persistence

---

## Known Gaps and Incomplete Work

### Incomplete Implementations

1. **Earnings Subsystem** (`earnings/`):
   - `earnings_cache.py` (494 lines): Earnings cache implementation
   - `yahoo_fetcher.py` (329 lines): Yahoo Finance earnings fetcher
   - `edgar_fetcher.py` (390 lines): SEC EDGAR fetcher
   - `xbrl_parser.py`: XBRL parser (incomplete)
   - **Status**: Partial implementation; not fully integrated into TFT builder

2. **Collection Coordinator** (`collection_coordinator.py`, 817 lines):
   - Multi-tier collection orchestration
   - **Status**: Implementation present but not actively used

3. **Data Retention Manager** (`data_retention_manager.py`):
   - Automatic data cleanup based on retention policies
   - **Status**: Implementation present but not integrated into scheduler

4. **Initialization Manager** (`initialization_manager.py`):
   - Startup initialization utilities
   - **Status**: Minimal implementation

### TODOs and FIXMEs

Search for `TODO`, `FIXME`, `XXX` in codebase for inline markers.

### Deprecated Components

- `scheduler_legacy.py` (1,551 lines): Legacy god class implementation
- `tft_dataset_builder_legacy.py` (2,208 lines): Legacy TFT builder
- Available via feature flags for backward compatibility

---

## Universal ML Architecture Pattern Compliance

### Pattern 1: Mandatory 4-Store + 4-Registry Integration

**Status**: ✅ APPROPRIATELY EXEMPT

**Rationale**: Data layer components are cold path utilities serving ML actors, not ML actors themselves.

**Partial Integration**:
- `TFTDatasetBuilder`: Optional FeatureStore integration for training/inference parity
- `FREDDataLoader`: Optional DataStore integration for persistence
- `DataScheduler`: Optional DataRegistry integration for event emission

**Proper Scope**: Components focus on data collection, processing, and preparation for ML actors.

### Pattern 2: Protocol-First Interface Design

**Status**: ✅ COMPLIANT

**Implementation**:
- `DataProvider`, `StaticDataProvider`, `TimeSeriesProvider` protocols (`providers/base.py`)
- Runtime-checkable protocols for structural typing
- Duck typing support for testing (DummyStore conforms to protocols)
- Clear contracts for component interactions

**Benefits**:
- Enables structural typing without implementation coupling
- Supports progressive fallback chains (real → mock)
- Type safety without circular dependencies

### Pattern 3: Hot/Cold Path Separation

**Status**: ✅ FULLY COMPLIANT

**Design Intent**: Data collection, processing, and caching are inherently cold path.

**Appropriate Operations**:
- DataFrame creation and manipulation (Polars/Pandas)
- File I/O (Parquet read/write)
- Network calls (Databento API, FRED API)
- Heavy computation (feature aggregation, dataset building)

**Documentation**: Module docstring explicitly states cold path focus.

### Pattern 4: Progressive Fallback Chains

**Status**: ✅ APPROPRIATELY IMPLEMENTED

**Fallback Order**:
- **DataRegistry**: PostgreSQL → JSON backend (development)
- **Calendar Source**: PandasCalendarSource → MockCalendarSource
- **Event Source**: FileEventSource → MockEventSource
- **Metadata Source**: Real implementations → MockMetadataSource
- **Optional Dependencies**: Graceful handling with clear error messages

**Scope-Appropriate**: Circuit breakers not required for cold path data utilities.

### Pattern 5: Centralized Metrics Bootstrap

**Status**: ✅ MOSTLY COMPLIANT

**Primary Components**: Use `ml.common.metrics_bootstrap` consistently.
- `DataScheduler`: Uses `get_counter`, `get_histogram`
- `FREDDataLoader`: Uses `get_counter`, `get_histogram`
- `ALFREDDataLoader`: Uses `get_counter`, `get_histogram`
- `DatabentoIngestionService`: Uses centralized metrics

**No Direct Imports**: No direct `prometheus_client` usage found.

**Pattern Adherence**: Strong compliance with centralized metrics approach.

---

## Operational Runbook

### L0 Minute Bar Backfill

```bash
# Backfill recent 14 days for tier 1 symbols
python -m ml.scripts.backfill_ohlcv_recent --tier 1 --days 14
```

**Output**: `data/tier1/<SYMBOL>/l0/<SYMBOL>_ohlcv.parquet`

### L2 Gap Fill (Resume-Safe Shards)

```bash
# Set environment caps to avoid OOM kills
export POLARS_MAX_THREADS=1
export PYARROW_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Backfill in shards of 5 symbols
for off in 0 5 10 15 20; do
  python ml/scripts/populate_l2_efficient.py \
    --tier 1 \
    --days 14 \
    --check-gaps \
    --max-symbols 5 \
    --symbol-offset $off \
    --rate-limit 10 \
    --sleep-between-symbols 1
done
```

**Output**: `data/tier1/<SYMBOL>/l2/<SYMBOL>_mbp-10.parquet`

**Resume**: Progress tracked in `data/tier1/.l2_progress.json` (per-day completion)

### FRED Macro Refresh (90 days)

```bash
# Set FRED API key
export FRED_API_KEY=your_api_key_here

# Refresh FRED indicators
python -m ml.cli.fred_export_ml_parquet \
  --out data/fred/fred_indicators_ml_format.parquet
```

**Output**:
- Wide format: `data/fred/fred_indicators_updated.parquet`
- ML format: `data/fred/fred_indicators_ml_format.parquet`

### ALFRED Vintage Refresh

```python
from ml.data.loaders.alfred_loader import ALFREDDataLoader, ALFREDConfig
from pathlib import Path

config = ALFREDConfig(
    series_ids=("GDP", "CPIAUCSL", "UNRATE"),
    out_dir=Path("data/fred/vintages"),
    start_date="2020-01-01",
    end_date="2025-01-01",
)
loader = ALFREDDataLoader(config)
stats = loader.refresh()
```

**Output**: `data/fred/vintages/<series>/release_calendar.parquet`

### Dataset Build (60-day)

```bash
python -m ml.pipelines.build_runner \
  --config ml/config/build_universe_60d.json
```

**Output**:
- Per-symbol: `/tmp/tft_universe_60d/<SYMBOL>/dataset.(parquet|csv)`
- Feature sets registered in `~/.nautilus/ml/features`

### Dataset Build (90-day)

```bash
python -m ml.pipelines.build_runner \
  --config ml/config/build_universe_90d.json
```

**Output**: `/tmp/tft_universe_90d/merged/dataset.(parquet|csv)`

### Orchestrator Auto-Fill

```bash
# Pre-build coverage sweep
python -m ml.cli.pipeline_orchestrator \
  --auto_fill_universe \
  --symbols SPY.NYSE,QQQ.NASDAQ \
  --include_l2
```

**Features**:
- Backfills bars/TBBO/trades using `IngestionOrchestrator.backfill_gaps`
- Coverage policy targets: 7y L0, 1y L1, 30d L2/3
- Depth ingestion reuses `populate_l2_efficient`
- Override L2 window: `--auto_fill_l2_days 60`

---

## Quick Commands

### Type Checking
```bash
poetry run mypy ml/data --strict
```

### Linting
```bash
poetry ruff check ml/data
```

### Testing
```bash
# All ML data tests
make pytest-ml

# Specific subsystem
poetry run pytest ml/tests/unit/data -k ingestion

# E2E tests
pytest -q ml/tests/e2e/test_tft_dataset_builder_e2e.py
pytest -q ml/tests/e2e/test_pipeline_orchestrator_e2e.py

# Performance tests
pytest -q ml/tests/performance/test_ingestion_microbench.py --benchmark-only
```

### Validators
```bash
make validate-metrics                   # Validate Prometheus metrics
make validate-events                    # Validate event definitions
make validate-nautilus-patterns         # Extended validation
```

---

## API Reference

### High-Level Dataset Building

```python
# Public dataset build facade
from ml.data import build_tft_dataset, DatasetBuildConfig
from pathlib import Path

cfg = DatasetBuildConfig(
    data_dir=Path("data/tier1"),
    out_dir=Path("./tft_datasets/60d"),
    symbols=["SPY", "QQQ", "AAPL"],
    dataset_id="tft_60d",
    include_macro=True,
    macro_lag_days=1,
    include_micro=True,
    include_l2=True,
    include_events=True,
    horizon_minutes=15,
    lookback_periods=60,
    register_features=True,
    feature_registry_dir=Path("~/.nautilus/ml/features"),
)

result = build_tft_dataset(cfg, data_store=data_store)
# result.dataset_parquet: Path
# result.dataset_csv: Path
# result.features_npz: Path
# result.feature_names: list[str]
# result.feature_set_id: str | None
# result.metadata: DatasetMetadata
```

### Data Loading and Conversion

```python
from ml.data import bars_to_dataframe
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from datetime import datetime

catalog = ParquetDataCatalog("./data")
bars_df = bars_to_dataframe(
    catalog=catalog,
    instrument_ids=["SPY.NYSE", "AAPL.NASDAQ"],
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
)
```

### Dataset Builder Direct Usage

```python
from ml.data import TFTDatasetBuilder
from nautilus_trader.persistence.catalog import ParquetDataCatalog

catalog = ParquetDataCatalog("./data")
builder = TFTDatasetBuilder(
    catalog=catalog,
    symbols=["SPY", "AAPL"],
    include_macro=True,
    include_micro=True,
    include_l2=True,
    include_events=True,
    student_mode=False,
)

dataset = builder.build_training_dataset(
    horizon_minutes=15,
    lookback_periods=30,
    use_polars=True,
)
```

### Data Collection and Scheduling

```python
from ml.data import DataCollector, DataScheduler
from ml.config.scheduler_config import SchedulerConfig

collector = DataCollector(
    storage_limit_gb=1000,
    data_dir="data/tier1",
)

scheduler = DataScheduler(
    catalog=catalog,
    collector=collector,
    metrics_port=8000,
)

scheduler.run_once()  # Manual trigger
scheduler.run_daily_update()  # Daily automation
```

### FRED Economic Data

```python
from ml.data import FREDDataLoader, FREDConfig
from pathlib import Path

config = FREDConfig(
    cache_ttl_hours=24,
    backfill_years=10,
    rate_limit_calls=120,
)
loader = FREDDataLoader(config)

# Store with DataRegistry integration
loader.store_indicators(data_store, data_registry)

# Export ML-format parquet
loader.export_ml_parquet(Path("data/fred/fred_indicators_ml_format.parquet"))

# Join with dataset
from ml.data.fred_join import join_fred_asof
dataset_with_macro = join_fred_asof(
    dataset,
    timestamp_col="timestamp",
    lag_days=1,
    fred_path="data/fred/fred_indicators_ml_format.parquet",
)
```

### L2 and Microstructure Caching

```python
from ml.data import L2MinuteCache, MicroMinuteCache
from datetime import datetime, timezone
from pathlib import Path

# L2 cache
l2_cache = L2MinuteCache(Path("data/features/l2_minute"))
start = datetime(2025, 8, 11, tzinfo=timezone.utc)
end = datetime(2025, 8, 18, tzinfo=timezone.utc)

l2_features = l2_cache.get_range(
    symbol="SPY",
    start=start,
    end=end,
    raw_base_dir=Path("data/tier1"),
)

# Microstructure cache
micro_cache = MicroMinuteCache(Path("data/features/micro_minute"))
micro_features = micro_cache.get_range("SPY", start, end, Path("data/tier1"))
```

### Ingestion Service

```python
from ml.data.ingest.service import DatabentoIngestionService, IngestionRequest
from datetime import datetime, UTC

service = DatabentoIngestionService.from_env()
request = IngestionRequest(
    dataset="EQUS.MINI",
    schema="trades",
    symbols=("SPY",),
    start=datetime(2025, 8, 1, tzinfo=UTC),
    end=datetime(2025, 8, 2, tzinfo=UTC),
)

for chunk in service.ingest(request):
    print(f"Symbol: {chunk.symbol}, Rows: {len(chunk.frame)}")
    # Persist chunk.frame to storage
```

---

## File Reference (Alphabetical)

**Root Level:**
- `__init__.py` (1,944 lines): Public API with lazy imports
- `cache_common.py`: Shared caching utilities
- `catalog_utils.py`: Nautilus catalog conversion
- `collection_coordinator.py` (817 lines): Multi-tier collection
- `collector.py` (828 lines): DataCollector
- `data_retention_manager.py`: Retention policies
- `dataset_manifest_defaults.py` (451 lines): Metadata helpers
- `feature_computation_manager.py` (348 lines): Feature coordination
- `feature_manifest_export.py`: Feature registry export
- `fred_join.py` (616 lines): FRED as-of join
- `initialization_manager.py`: Startup utilities
- `l2_cache.py`: L2 per-minute cache
- `macro_revisions.py` (369 lines): Revision analysis
- `micro_cache.py`: Microstructure cache
- `registry_integrator.py`: Registry integration
- `scheduler.py` (1,542 lines): DataScheduler facade
- `scheduler_legacy.py` (1,551 lines): Legacy god class
- `tft_dataset_builder.py` (2,729 lines): TFT builder
- `tft_dataset_builder_legacy.py` (2,208 lines): Legacy TFT
- `trading_day_calculator.py`: Trading day utilities
- `validation.py`: Dataset validation
- `vintage.py`: Vintage policy utilities

**Subdirectories:**
- `ingest/`: Data ingestion (12 files, ~6,000 lines)
- `loaders/`: Data loaders (7 files, ~2,500 lines)
- `providers/`: Provider architecture (6 files, ~2,500 lines)
- `sources/`: Data sources (3 files, ~2,100 lines)
- `earnings/`: Earnings subsystem (4 files, ~1,500 lines)
- `fixtures/`: Test fixtures (3 files)

---

## Document Metadata

**Document Version**: 5.0
**Last Updated**: 2025-10-19
**Maintainer**: ML Data Pipeline Team
**Status**: Production - Cold Path Data Utilities

**Changes from v4.0**:
- Comprehensive accuracy review of all 60+ files
- Detailed ingestion subsystem documentation (service, orchestrator, resume, subscription, discovery)
- Complete TFT builder architecture with market bindings and vintage policies
- Expanded loader documentation (FRED, ALFRED, OHLCV recent, Fama-French, alternative)
- Provider/source architecture clarification
- Added operational runbook and quick commands
- Cited specific file paths and line numbers throughout
- Documented known gaps and incomplete work
- Updated file reference with line counts

**Scope**: This document covers actual implementation in `ml/data/` as of October 2025, focusing on what exists rather than what is planned.
