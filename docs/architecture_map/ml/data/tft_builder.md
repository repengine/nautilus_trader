# Dataset Building Architecture

**Status:** Living Document
**Root:** `ml/data/tft_dataset_builder.py`
**Key Class:** `TFTDatasetBuilder`

## 1. System Overview

The `TFTDatasetBuilder` is the **Assembly Line** of the ML system. It takes raw market data (L1/L2) and auxiliary data (Macro, Earnings, Calendar) and fuses them into a single, time-aligned `Polars` DataFrame suitable for training Temporal Fusion Transformers (TFT) or gradient boosters.

**Core Responsibility:**
Ensure that at any time step $t$, the feature vector $X_t$ contains *only* information available at or before $t$, preserving strict causal ordering (no look-ahead bias).

## 2. Key Capabilities

### A. Multi-Source Fusion
The builder stitches together data from disparate sources:

1.  **L1 Data:** OHLCV bars from `ParquetDataCatalog` or `DataStore`.
2.  **L2 Data:** Order book imbalance and liquidity metrics (`L2MinuteCache`).
3.  **Macro Data:** Economic indicators from FRED/ALFRED (`vintage.py`), handling Point-in-Time revision history.
4.  **Earnings:** EPS estimates and actuals, aligned by publication date (`earnings_lag_days`).
5.  **Calendar:** Trading days, time-of-day, day-of-week embeddings.

### B. Target Generation
It automatically computes regression/classification targets:

-   **Horizon:** `horizon_minutes` (e.g., 15m into the future).
-   **Return:** Log-returns or simple returns.
-   **Binary:** `y > threshold` (for classification).

### C. Dual-Path Execution

-   **Direct:** Computes features on-the-fly from raw data (Legacy/Research).
-   **FeatureStore:** Loads pre-computed features from `FeatureStore` (Production/Parity).
-   *Note:* The builder prefers the `FeatureStore` path to guarantee that training data matches what the Actor sees.

## 3. Critical Workflows

### The "Build" Loop

1.  **Load Bars:** Fetch OHLCV for all symbols.
2.  **Compute Technicals:** Apply `FeatureEngineer` (if not using Store).
3.  **Generate Targets:** Compute forward-looking returns.
4.  **Join Auxiliaries:**
    -   Join Macro (Asof join on `ts_event` <= `publication_time`).
    -   Join Earnings (Asof join on `timestamp` <= `filing_date` + lag).
    -   Join Microstructure (L2 cache).
5.  **Add Known Inputs:** Add calendar features (Hour, Day, etc.) which are known in the future (for TFT).
6.  **Concat & Sort:** Combine all instruments into a single large table.

## 4. Constraints & Invariants

-   **Polars-First:** All heavy joins use `polars` for speed and memory efficiency.
-   **Timestamp Alignment:** All sources must be aligned to `ts_event` (nanosecond int64).
-   **No Look-Ahead:**
-   Macro data uses `vintage.py` to select the value *as known* at time $t$.
-   Earnings data applies `earnings_lag_days` to simulate processing delay.

## 5. Data Flow

```mermaid
graph TD
    subgraph "Sources"
        A[Data Catalog (L1)]
        B[L2 Cache]
        C[Macro (FRED)]
        D[Earnings Store]
    end

    subgraph "Builder"
        E[Load Bars] --> F[Feature Engineer]
        F --> G[Join L2]
        G --> H[Join Macro (Vintage)]
        H --> I[Join Earnings]
        I --> J[Add Calendar]
        J --> K[Compute Targets]
    end

    A --> E
    B --> G
    C --> H
    D --> I
    K --> L[Training Dataset]
```
