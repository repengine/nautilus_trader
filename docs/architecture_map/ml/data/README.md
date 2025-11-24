# ML Data Architecture

**Status:** Living Document
**Root:** `ml/data/`
**Primary Dependency:** `nautilus_trader.persistence.catalog.parquet`

## 1. System Overview

The `ml/data` module manages the **Cold Path** data lifecycle: ingestion, storage, and transformation into training datasets.

**Key Design Decisions:**

-   **No Abstraction Layers:** It interacts *directly* with Nautilus `ParquetDataCatalog`. The old `MLDataLoader` abstraction has been removed (Aug 2025) to reduce overhead.
-   **Polars First:** All heavy lifting is done via Polars for performance.
-   **Lazy Loading:** Dependencies (Polars, Databento) are imported lazily to prevent import-time crashes in lean environments.

## 2. Core Components

### A. Public API (`__init__.py`)

-   **`DatasetBuildConfig`**: The central configuration object for building datasets.
-   **`build_tft_dataset()`**: The main entry point for generating training artifacts.
-   **`DatasetMetadata`**: Tracks lineage, vintage policies, and market bindings.

### B. Ingestion & Collection

-   **`collector.py`**: The `DataCollector` class.
-   **Responsibility:** Fetch raw data (L2, Trades, Bars) from external APIs (like Databento).
-   **Constraint:** Writes directly to Catalog. Does *not* return DataFrames.
-   **`ingest/`**: detailed implementation of ingestion pipelines.
-   `service.py`: The `DatabentoIngestionService`.

### C. Dataset Construction

-   **`tft_dataset_builder.py`**: Specialized builder for Temporal Fusion Transformer (TFT) models.
-   **Output:** Parquet files + `.npz` feature matrices.
-   **Features:** Handles vintage data (Macro revisions), earnings announcements, and calendar events.
-   **`catalog_utils.py`**: Low-level helpers to convert Catalog binary formats into Polars DataFrames.

### D. Automation

-   **`scheduler.py`**: The `DataScheduler`.
-   **Role:** Runs daily cron jobs to fetch new data and trigger feature computation.
-   **Flag:** Controlled by `ML_USE_LEGACY_DATA_SCHEDULER` env var.

### E. Providers (Covariates)
Located in `providers/` and `sources/`:

-   **`MarketCalendarProvider`**: Trading hours/holidays.
-   **`EventScheduleProvider`**: Economic events (FOMC, CPI).
-   **`InstrumentMetadataProvider`**: Symbol details.

## 3. Data Flow

```mermaid
graph TD
    A[External API (Databento)] -->|DataCollector| B[ParquetDataCatalog]
    C[External API (FRED)] -->|FREDDataLoader| B
    B -->|catalog_utils| D[Polars DataFrame]
    D -->|TFTDatasetBuilder| E[Training Dataset (Parquet/NPZ)]
    E --> F[Training Pipeline]
```

## 4. Important Files

-   `ml/data/__init__.py`: **[HEAVY]** Contains `DatasetBuildConfig`, `DatasetMetadata`, and the `build_tft_dataset` facade.
-   `ml/data/vintage.py`: Handling of "Point-in-Time" (Vintage) data to prevent look-ahead bias in backtests.

## 5. Known Constraints

1.  **Strict Types:** 100% MyPy coverage required.
2.  **Native Types:** `DataCollector` must return Nautilus native objects (`Bar`, `QuoteTick`), not dicts or DataFrames.

## 6. Code Audit Findings (2025-11-19)

### A. Calendar Stubs (`tft_dataset_builder.py`)

-   **Severity:** **MAJOR**
-   **Location:** `class _TradingDayCalendar` (Line ~230)
-   **Issue:** `is_trading_day` returns `True` for every timestamp.
-   **Impact:** Calendar features (e.g. "days since last close") are incorrect on weekends/holidays.

### B. N+1 Query Pattern (`tft_dataset_builder.py`)

-   **Severity:** **MODERATE**
-   **Location:** `_fetch_earnings_features` (Line ~1080)
-   **Issue:** Iterates through `actuals` list and calls `self.data_store.get_earnings_estimate...` for every single record.
-   **Impact:** Significant slowdown when processing instruments with long history. Should be replaced with a batched store query.

### C. Spaghetti Logic (`tft_dataset_builder.py`)

-   **Severity:** **MINOR**
-   **Location:** `_build_training_dataset_direct` (Line ~1450)
-   **Issue:** Deeply nested loops and try/except blocks for handling fallback paths (Catalog -> Parquet Files -> Manual Paths).
-   **Impact:** High maintenance burden; difficult to verify coverage.
