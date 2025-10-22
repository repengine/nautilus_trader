# Context: Preprocessing Module

**Last Updated:** 2025-10-19 (replaces 2024-09-09 version)
**Module Size:** 2,170 lines across 5 files
**Test Coverage:** 3 unit test files (event_ingestion, joins, vintage_age)

## Overview

The `ml/preprocessing/` directory implements advanced preprocessing and transformation utilities for financial time series data, focusing on three critical areas: (1) **stationarity transformations** and cross-validation using López de Prado's techniques, (2) **point-in-time correct joins** to prevent lookahead bias, and (3) **specialized data transformations** including vintage age conversion and scheduled event ingestion for macro and market data. All components are strictly **cold-path only** and designed for batch dataset preparation, never for use in hot paths (actors, on_* handlers).

The preprocessing module serves as the statistical foundation for robust financial ML applications, ensuring temporal integrity, preventing information leakage, and applying academically rigorous transformations to maintain stationarity while preserving memory in financial time series.

## Architecture

### Structural Organization

```
ml/preprocessing/
├── __init__.py              # Lazy-loading public API with circular import protection (241 lines)
├── stationarity.py          # Stationarity transformations, CV, microstructure features (827 lines)
├── joins.py                 # Point-in-time correct joins and temporal utilities (470 lines)
├── event_ingestion.py       # Scheduled event ingestion for macro/market calendars (399 lines)
└── vintage_age.py           # Vintage timestamp to age feature conversion (233 lines)
```

**Total:** 2,170 lines of cold-path preprocessing code implementing academic ML techniques.

### Design Principles

1. **Point-in-Time Correctness**: All temporal operations ensure no future information leakage
2. **Dual DataFrame Support**: Native support for both Polars (preferred) and Pandas with automatic routing
3. **Academic Rigor**: Implementation of proven techniques from financial ML literature (López de Prado)
4. **Performance Optimization**: JIT compilation via Numba where available, efficient vectorized operations
5. **Nautilus Integration**: Timestamp handling in nanoseconds, instrument_id grouping support
6. **Cold-Path Only**: No preprocessing functions are safe for hot paths; all involve heavy DataFrame operations, statistical computations, or file I/O

### Lazy Loading and Circular Import Protection

**File:** `ml/preprocessing/__init__.py` (lines 1-242)

The module uses a sophisticated `__getattr__` lazy loading mechanism to avoid circular imports in the broader ML dependency chain (`ml._imports` ↔ `ml.common.metrics_bootstrap`). This allows clean public API usage while deferring actual imports until needed.

**Key patterns:**
- **Stationarity components** (StationarityTransformer, DataNormalizer, etc.) can be imported directly via `from ml.preprocessing import ...`
- **Join utilities** (asof_join, embargo_window, etc.) should be imported directly to avoid circular import issues: `from ml.preprocessing.joins import asof_join`
- Informative `ImportError` messages guide users to the correct import pattern when circular imports are detected (lines 164-201)

**Public API** (`__all__`, line 220):
- DataNormalizer
- EventIngestionConfig, EventIngestionUtility
- FeatureLagGenerator
- MarketMicrostructureFeatures
- PurgedCrossValidator
- StationarityTransformer
- asof_join, create_lag_features, embargo_window, validate_no_lookahead

**Module metadata** (lines 234-241):
- `__module_type__ = "cold_path"`
- `__performance_budget__ = "unlimited"` (cold path operations)
- Dependencies: numpy, scipy, statsmodels, numba (optional), polars (optional), pandas (optional)

## Core Components

### 1. Stationarity Transformations (`stationarity.py`, 827 lines)

Implements advanced stationarity techniques and cross-validation for financial time series, drawing from "Advances in Financial Machine Learning" by López de Prado (2018).

#### StationarityTransformer (lines 48-294)

Advanced transformer implementing fractional differencing to achieve stationarity while preserving memory.

**Key attributes:**
- `method: str` - Transformation method: 'fractional', 'standard', or 'auto'
- `d: float` - Differencing order for fractional method (default 0.5)
- `threshold: float` - Minimum weight magnitude to retain (default 1e-3)
- `max_lags: int | None` - Maximum number of lags for weights
- `_weights: NDArray[float64] | None` - Cached fractional weights for inverse transform
- `_optimal_d: float | None` - Optimal d found via ADF test when using auto mode

**Key methods:**

1. **`_compute_weights_numba(d: float, size: int)`** (lines 91-99, JIT-compiled)
   - Computes fractional differencing weights using binomial expansion
   - JIT-compiled with Numba for performance (degrades to pure Python if Numba unavailable)
   - Uses iterative formula: `w[k] = -w[k-1] * (d - k + 1) / k`
   - Returns reversed weights for convolution application

2. **`fractional_difference(series, d=None)`** (lines 120-167)
   - Applies fractional differencing with configurable order d
   - Drops small weights below threshold for computational efficiency and numerical stability
   - Uses vectorized convolution for efficient computation
   - Stores weights in `_weights` for potential inverse transformation
   - Returns differenced series with initial NaN values

3. **`find_optimal_d(series, adf_threshold=0.05, ...)`** (lines 169-226)
   - Uses Augmented Dickey-Fuller test to find minimum d achieving stationarity
   - Searches from min_d to max_d with configurable step size (default 0.01)
   - Returns 0.0 if series already stationary
   - Stores result in `_optimal_d` attribute
   - Requires statsmodels for ADF test

4. **`fit_transform(series, auto_d=False)`** (lines 228-261)
   - Combined fitting and transformation
   - Stores original mean and std for potential inverse operations
   - If auto_d=True or method='auto', calls `find_optimal_d()`
   - Supports fractional, standard (first difference), and auto methods

5. **`inverse_transform(series)`** (lines 263-293)
   - Approximate reconstruction of original series
   - For standard differencing: uses cumulative sum
   - For fractional: applies fractional differencing with -d
   - Not exact for fractional due to truncation and threshold filtering

**Academic basis:** López de Prado (2018), Chapter 5 - Fractional Differentiation

#### MarketMicrostructureFeatures (lines 296-440)

Implements market microstructure analytics for L1/L2/L3 order book data.

**Static methods:**

1. **`roll_spread(prices)`** (lines 309-332)
   - Estimates bid-ask spread from price covariance (Roll 1984)
   - Formula: `spread = 2 * sqrt(-cov(r[t], r[t+1]))` if cov < 0, else 0
   - Handles edge case where covariance is positive (no spread estimate)

2. **`kyle_lambda(prices, volumes)`** (lines 334-362)
   - Measures price impact using Kyle's lambda (Kyle 1985)
   - Regresses absolute returns on signed volumes
   - Returns coefficient from linear regression
   - Handles zero volume and insufficient data cases

3. **`amihud_illiquidity(returns, volumes)`** (lines 364-389)
   - Computes Amihud illiquidity ratio (Amihud 2002)
   - Formula: `mean(|returns| / volumes)`
   - Avoids division by zero with safe volume clamping

4. **`vpin(prices, volumes, bucket_size=50)`** (lines 391-439)
   - Volume-synchronized Probability of Informed Trading (Easley et al. 2012)
   - Classifies volumes as buy/sell based on price direction
   - Aggregates into volume buckets (default 50 trades per bucket)
   - Returns mean VPIN across buckets

**Use case:** Feature engineering for L2/L3 order book data; falls back to simplified calculations when only OHLCV available.

#### FeatureLagGenerator (lines 442-543)

Comprehensive lagged feature creation system for time series models.

**Initialization parameters:**
- `lag_periods: list[int]` - Simple lag periods (default: [1, 2, 3, 5, 10, 20])
- `rolling_windows: list[int]` - Rolling window sizes (default: [5, 10, 20, 50])
- `ewm_spans: list[int]` - Exponential weighted spans (default: [5, 10, 20])

**Method: `create_lagged_features(series, include_rolling=True, include_ewm=True)`** (lines 476-542)

Returns dictionary of feature arrays:

1. **Simple lags:** `lag_1`, `lag_2`, ..., `lag_20`
   - Uses `np.roll()` with proper NaN filling for initial periods
   - Only creates lags that fit within series length

2. **Rolling statistics:**
   - `rolling_mean_{window}`: Vectorized convolution for efficiency (lines 513-518)
   - `rolling_std_{window}`: Expanding window standard deviation (lines 521-527)

3. **Exponentially weighted features:**
   - `ewm_{span}`: Manual EWM computation using iterative formula (lines 532-539)
   - Alpha calculated as `2 / (span + 1)`

**Performance:** Uses NumPy vectorization throughout; pre-allocates arrays where possible.

#### DataNormalizer (lines 545-685)

Advanced normalization techniques resistant to financial data characteristics (outliers, non-normality).

**Initialization:**
- `method: str` - Normalization method: 'robust', 'rank', 'boxcox', or 'standard'
- `_params: dict[str, Any]` - Cached parameters for inverse transformation

**Method: `fit_transform(data)`** (lines 569-638)

Implements four normalization strategies:

1. **Robust scaling** (lines 587-595)
   - Uses median and MAD (Median Absolute Deviation) instead of mean/std
   - Highly resistant to outliers
   - Formula: `(x - median) / MAD`

2. **Rank transformation** (lines 597-610)
   - Converts to uniform distribution via ranks, then to normal via inverse CDF
   - Non-parametric transformation resistant to extreme values
   - Uses scipy.stats.rankdata and norm.ppf

3. **Box-Cox transformation** (lines 612-628)
   - Power transformation with automatic lambda estimation
   - Requires positive values (adds shift if necessary)
   - Stores lambda and shift for inverse

4. **Standard normalization** (fallback, lines 630-638)
   - Traditional z-score: `(x - mean) / std`

**Method: `inverse_transform(data)`** (lines 640-684)

Provides approximate inverse for each method. Note: rank transformation inverse is approximate since it doesn't store original data distribution.

#### PurgedCrossValidator (lines 687-828)

Industry-standard purged walk-forward cross-validation for financial time series, implementing López de Prado (2018) Chapter 7.

**Initialization parameters:**
- `n_splits: int` - Number of CV splits (default 5)
- `purge_gap: int` - Number of samples to exclude between train/test to prevent leakage (default 0)
- `embargo_pct: float` - Percentage of total samples to embargo after each test set (default 0.0)

**Validation** (lines 729-741):
- Ensures n_splits >= 2
- Ensures purge_gap >= 0
- Ensures embargo_pct in [0, 1)

**Method: `split(X, y=None, groups=None)`** (lines 743-801)

Generates indices for train/test splits with purging and embargo:

1. **Calculate test size:** `n_samples // n_splits` for each fold
2. **Define test set:** Contiguous block for fold i
3. **Apply purge gap:** Remove `purge_gap` samples before and after test set from training
4. **Apply embargo:** Remove `embargo_pct * n_samples` samples after test set (lines 791-793)
5. **Combine train indices:** Concatenate before-purge and after-purge training samples

**sklearn compatibility:** Implements `get_n_splits()` method for sklearn integration (lines 803-827)

**Academic basis:** López de Prado (2018), Chapter 7 - Cross-Validation in Finance

### 2. Point-in-Time Joins (`joins.py`, 470 lines)

Provides dual-implementation (Polars/Pandas) utilities for point-in-time correct joins to prevent lookahead bias.

#### asof_join() (lines 40-113)

Performs point-in-time correct as-of join between two dataframes, ensuring only information available at or before the timestamp is used.

**Parameters:**
- `left: DataFrameLike` - Left dataframe with timestamps to join on
- `right: DataFrameLike` - Right dataframe with reference data
- `on: str | list[str]` - Column(s) to perform temporal join on (typically timestamp)
- `by: str | list[str] | None` - Column(s) to group by before joining (e.g., instrument_id)
- `tolerance: str | None` - Maximum time tolerance for matches (e.g., "1h", "5m")
- `direction: DirectionType` - Join direction: "backward" (past data only), "forward", or "nearest"

**Automatic routing** (lines 102-112):
- Detects dataframe type (Polars vs Pandas) via isinstance check
- Routes to `_asof_join_polars()` or `_asof_join_pandas()` implementation
- Ensures appropriate dependencies available via `check_ml_dependencies()`

**Polars implementation: `_asof_join_polars()`** (lines 115-150)
- Ensures single column for `on` (Polars limitation, line 127-130)
- Sorts both dataframes by temporal column
- Uses `left.join_asof(other=right, on=on, strategy=direction, by=by, tolerance=tolerance)`

**Pandas implementation: `_asof_join_pandas()`** (lines 153-200)
- Converts direction naming to Pandas convention
- Ensures single column for `on` (line 172-175)
- Sorts both dataframes by temporal column
- Converts string tolerance to `pd.Timedelta` if needed
- Uses `pd.merge_asof(left, right, on=on, by=by, tolerance=tolerance, direction=direction)`

**Use case:** Join market data with corporate events, economic releases, or news without lookahead bias.

#### embargo_window() (lines 203-267)

Applies embargo windows around significant events to prevent information leakage during training.

**Parameters:**
- `df: DataFrameLike` - Dataframe with timestamp column
- `event_timestamps: list[int] | NDArray[int64]` - Timestamps of events requiring embargo (nanoseconds)
- `embargo_before_ns: int` - Embargo window before event in nanoseconds (default 1 hour)
- `embargo_after_ns: int` - Embargo window after event in nanoseconds (default 1 hour)
- `timestamp_col: str` - Name of timestamp column (default "ts_event")

**Returns:** Original dataframe with added 'embargo' boolean column marking affected periods.

**Polars implementation: `_embargo_window_polars()`** (lines 269-292)
- Initializes embargo mask as `pl.lit(False)`
- For each event timestamp, marks rows within `[event_ts - before, event_ts + after]`
- Combines masks with logical OR (`embargo_mask | event_embargo`)
- Adds embargo column via `df.with_columns(embargo_mask.alias("embargo"))`

**Pandas implementation: `_embargo_window_pandas()`** (lines 295-324)
- Creates DataFrame copy to avoid mutations
- Converts timestamp column to numpy array for vectorized operations
- Initializes boolean embargo_mask array
- For each event, applies vectorized check: `(timestamps >= start) & (timestamps <= end)`
- Combines with bitwise OR and adds to DataFrame

**Use case:** Exclude data around earnings releases, FOMC meetings, or other market-moving events from training sets.

#### validate_no_lookahead() (lines 327-383)

Validates that features don't contain future information relative to targets.

**Parameters:**
- `features_df: DataFrameLike` - Features dataframe with timestamps
- `targets_df: DataFrameLike` - Targets dataframe with timestamps
- `feature_timestamp_col: str` - Timestamp column in features (default "ts_event")
- `target_timestamp_col: str` - Timestamp column in targets (default "ts_event")

**Returns:** `True` if no lookahead bias detected.

**Raises:** `ValueError` if `max(feature_timestamps) > min(target_timestamps)` (lines 375-381).

**Implementation:**
- Works with both Polars and Pandas via duck typing
- Extracts max feature timestamp and min target timestamp
- Handles empty dataframes (returns True)
- Provides clear error message with actual timestamp values when lookahead detected

#### create_lag_features() (lines 386-420)

Creates lagged features ensuring point-in-time correctness.

**Parameters:**
- `df: DataFrameLike` - Input dataframe
- `columns: list[str]` - Columns to create lags for
- `lags: list[int]` - Number of periods to lag (positive = past values)
- `group_by: str | list[str] | None` - Columns to group by (e.g., instrument_id)
- `timestamp_col: str` - Timestamp column for ordering (default "ts_event")

**Polars implementation: `_create_lag_features_polars()`** (lines 422-445)
- Sorts dataframe by timestamp_col
- Creates lag expressions for each column/lag combination
- If group_by specified, applies `pl.col(col).shift(lag).over(group_by)`
- Otherwise, uses `pl.col(col).shift(lag)`
- Returns dataframe with all lag columns added via `df.with_columns(lag_exprs)`

**Pandas implementation: `_create_lag_features_pandas()`** (lines 448-470)
- Sorts dataframe by timestamp_col and creates copy
- For each column/lag combination:
  - If group_by specified: `df.groupby(group_by)[col].shift(lag)`
  - Otherwise: `df[col].shift(lag)`
- Adds lag columns directly to DataFrame with naming convention `{col}_lag_{lag}`

**Use case:** Create lagged price, volume, or feature columns for time series models while maintaining proper temporal ordering.

### 3. Event Ingestion (`event_ingestion.py`, 399 lines)

Utilities for ingesting scheduled events (FOMC meetings, earnings, economic releases, options expiry, holidays) into normalized Polars datasets for use in ML pipelines.

#### EventIngestionConfig (lines 83-103)

Frozen dataclass configuration for EventIngestionUtility.

**Attributes:**
- `start: datetime` - Ingestion start date
- `end: datetime` - Ingestion end date
- `out_dir: Path` - Output directory for events.parquet (default: "data/events")
- `alfred_vintage_dir: Path | None` - Directory containing ALFRED vintage data
- `economic_series: tuple[str, ...]` - Economic series to ingest (default: ("CPI",))
- `economic_stub_path: Path | None` - CSV file containing economic event stubs
- `corporate_source_path: Path | None` - CSV file containing corporate events
- `calendar_code: str` - Exchange calendar code (default: "XNYS")
- `include_options_expiry: bool` - Whether to include monthly options expiry (default: True)

**Validation** (`__post_init__`, lines 99-102): Ensures `end > start`.

#### EventIngestionUtility (lines 105-399)

Ingests scheduled events into a normalized Polars dataset with consistent schema.

**Schema** (lines 143-151):
- `event_timestamp: Datetime("ns")` - Event timestamp in nanoseconds
- `event_type: Utf8` - Event type: "fed_meeting", "economic_release", "earnings", "options_expiry", "holiday"
- `name: Utf8` - Event name/description
- `instrument_id: Utf8` - Associated instrument (empty string if not applicable)
- `importance: Utf8` - Importance level: "HIGH", "MEDIUM", "LOW"
- `source: Utf8` - Data source: "federal_reserve", "alfred", "exchange", "stub", "calendar_stub", etc.
- `metadata: Utf8` - JSON-encoded metadata dictionary

**Method: `ingest()`** (lines 117-126)

Main entry point:
1. Calls `_collect_events()` to gather all events
2. Creates output directory if needed
3. Writes events DataFrame to `out_dir/events.parquet`
4. Returns path to written file

**Event collection methods:**

1. **`_generate_fomc_events()`** (lines 166-192)
   - Hard-coded 2024 FOMC meeting dates (lines 167-176)
   - Filters to start/end range
   - Sets importance="HIGH", source="federal_reserve"
   - Metadata includes series="FOMC"

2. **`_generate_options_expiry_events()`** (lines 194-215)
   - Computes third Friday of each month using `_third_friday()` helper (lines 62-73)
   - Marks March/June/September/December as "Triple Witching" (line 202)
   - Sets importance="MEDIUM", source="exchange"

3. **`_generate_quarterly_earnings_stub()`** (lines 217-241)
   - Generates placeholder earnings events for Jan/Apr/Jul/Oct (Q1/Q2/Q3/Q4)
   - Assumes earnings reported on 20th of month at 21:30 UTC
   - Sets importance="MEDIUM", source="stub"
   - Metadata includes quarter and year

4. **`_load_economic_stub()`** (lines 243-280)
   - Loads economic events from CSV file if `economic_stub_path` configured
   - Reads with Polars, filters to date range
   - Preserves all columns as metadata (lines 263-276)

5. **`_load_alfred_vintages()`** (lines 282-320)
   - Loads FRED vintage release calendars from `alfred_vintage_dir/{series}/release_calendar.parquet`
   - For each series in `economic_series`, reads release_ts as event timestamp
   - Sets event_type="economic_release", importance="MEDIUM", source="alfred"
   - Metadata includes series_id and observation_ts

6. **`_load_corporate_events()`** (lines 322-359)
   - Loads corporate events (earnings, dividends, etc.) from CSV if `corporate_source_path` configured
   - Reads with Polars, filters to date range
   - Preserves instrument_id and all extra columns as metadata

7. **`_generate_holiday_events()`** (lines 361-399)
   - Generates US exchange holidays: New Year's Day, Independence Day, Thanksgiving (4th Thursday of November), Christmas
   - Sets importance="LOW", source="calendar_stub"
   - Metadata includes calendar code (e.g., "XNYS")

**Helper functions:**

- `_normalize_datetime(dt)` (lines 38-44): Converts timezone-aware datetime to UTC, removes tzinfo for consistent persistence
- `_iter_months(start, end)` (lines 47-59): Yields (year, month) pairs covering inclusive range
- `_third_friday(year, month)` (lines 62-73): Computes third Friday at 16:00 UTC
- `_quarterly_months()` (lines 76-80): Returns (1, 4, 7, 10) for earnings quarters

**Use case:** Generate normalized event calendar for use in training pipelines, particularly for embargo windows and event-driven features. Integrates with `MLIntegrationManager.ingest_events()` (ml/core/integration.py, line 238).

### 4. Vintage Age Conversion (`vintage_age.py`, 233 lines)

Batch-aware utilities for converting macro vintage timestamp columns (e.g., `FRED_GDP__value_vintage_ts`) into numeric age-in-minutes features (e.g., `FRED_GDP__vintage_age_minutes`) using PyArrow streaming to keep memory usage bounded.

**Problem:** Macro economic data often includes "vintage" timestamps indicating when a value was released or revised. These timestamps are not directly usable by neural network models. Converting to "age" (time elapsed between event and release) creates a numeric feature suitable for TFT and other models.

#### VintageConversionResult (lines 40-46)

Frozen dataclass summarizing conversion:
- `vintage_columns: tuple[str, ...]` - Original timestamp column names
- `age_columns: tuple[str, ...]` - Replacement age feature column names

#### convert_vintage_timestamps_to_age() (lines 84-166)

Main conversion function: streams parquet file, replaces `*__value_vintage_ts` columns with `*__vintage_age_minutes`.

**Parameters:**
- `source: Path` - Input parquet file containing vintage timestamp columns
- `destination: Path` - Output parquet file (must differ from source)
- `timestamp_column: str` - Column providing event timestamps (int64 nanoseconds, default "timestamp")
- `batch_size: int` - Maximum rows per batch during streaming (default 32,768)
- `compression: str` - Compression codec for output (default "snappy")
- `vintage_suffix: str` - Suffix identifying vintage columns (default "__value_vintage_ts")
- `age_suffix: str` - Suffix for age feature columns (default "__vintage_age_minutes")

**Algorithm:**
1. Validate source file exists and is .parquet (lines 121-123)
2. Ensure destination != source (lines 122-123)
3. Open parquet file, validate timestamp column type (int64 or timestamp, lines 125-126)
4. Derive vintage columns using `_derive_vintage_columns()` (line 127)
5. Raise ValueError if no vintage columns found (lines 128-130)
6. Stream batches via `parquet.iter_batches(batch_size=batch_size)` (line 137)
7. For each batch:
   - Convert vintage timestamps to age in minutes using `_compute_age_minutes_array()` (lines 140-145)
   - Replace vintage columns with age columns in schema (lines 147-156)
   - Write transformed batch to ParquetWriter (lines 159-161)
8. Close writer and return VintageConversionResult (lines 162-166)

**Helper: `_compute_age_minutes_array()`** (lines 71-81)
- Casts timestamp and vintage to int64
- Computes delta: `timestamp_ns - vintage_ts`
- Divides by 60_000_000_000 (nanoseconds per minute)
- Casts to float32 for reduced memory footprint
- Uses PyArrow compute functions (`pc.cast`, `pc.subtract`, `pc.divide`)

**Memory efficiency:** Processes data in batches rather than loading entire file, enabling processing of multi-GB datasets.

#### update_metadata_with_vintage_age() (lines 169-219)

Updates dataset metadata JSON to reflect vintage age conversion.

**Parameters:**
- `metadata: dict[str, object]` - Parsed dataset metadata to update
- `vintage_columns: Sequence[str]` - Original timestamp column names
- `age_columns: Sequence[str]` - Replacement age feature column names

**Returns:** Deep copy of metadata with updated column listings (ensures immutability).

**Updates:**
1. **time_varying_known_reals** (lines 197-202):
   - Removes vintage timestamp columns
   - Adds age feature columns
2. **drop_columns** (lines 204-207):
   - Adds vintage timestamp columns to drop list (no longer in dataset)
3. **vintage_handling metadata** (lines 211-216):
   - Adds `vintage_handling.strategy = "age_features"`
   - Records reason for conversion
   - Lists original and replacement columns

**Validation:** Raises ValueError if `column_info` missing from metadata (lines 193-195).

#### write_metadata() (lines 222-225)

Persists metadata dictionary to disk as JSON with newline termination.

**Use case:** Update dataset metadata after converting parquet file to maintain consistency.

#### Integration Points

**CLI:** `ml/cli/convert_vintage_age.py` (142 lines)
- Command-line interface for batch conversion
- Loads metadata, runs conversion, updates metadata, writes back
- Example: `python -m ml.cli.convert_vintage_age --source dataset.parquet --metadata dataset_metadata.json`

**Dataset builders:**
- `ml/orchestration/dataset_builder.py` (line 33-35): Imports vintage_age utilities
- `ml/pipelines/build_runner.py` (line 36-38): Imports vintage_age utilities
- Used in post-processing step after TFT dataset creation to convert FRED vintage timestamps

**Tests:** `ml/tests/unit/preprocessing/test_vintage_age.py` (3,521 bytes)
- Tests conversion with sample dataset containing `foo__value_vintage_ts`
- Validates age computation (5 minutes delta between timestamps)
- Ensures metadata immutability and correct updates

## Dependencies

### Internal Dependencies

- **ml._imports**: Lazy loading system for Polars (`HAS_POLARS`, `pl`), Pandas (`HAS_PANDAS`, `pd`)
- **ml.ml_types**: `DataFrameLike`, `PolarsDF` type aliases for cross-framework compatibility
- **nautilus_trader.core.data**: Nanosecond timestamp handling standards (implicit via schema requirements)

### External Dependencies

**Required:**
- **numpy**: Core numerical computations and array operations (all modules)
- **pyarrow**: Zero-copy columnar data processing for vintage_age streaming (vintage_age.py)

**Optional (lazy-loaded):**
- **polars**: High-performance dataframe operations (joins.py, event_ingestion.py)
- **pandas**: Traditional dataframe operations for compatibility (joins.py)
- **scipy.stats**: Statistical functions (ADF test, Box-Cox, normal distribution) (stationarity.py)
- **statsmodels**: Augmented Dickey-Fuller test for stationarity detection (stationarity.py)
- **numba**: JIT compilation for performance-critical operations (stationarity.py, optional)

## Usage Patterns

### Stationarity Transformation

```python
from ml.preprocessing import StationarityTransformer

# Fractional differencing with auto d-selection
transformer = StationarityTransformer(method="auto")
stationary_series = transformer.fit_transform(price_series, auto_d=True)
print(f"Optimal d: {transformer._optimal_d}")

# Manual fractional differencing (preserve memory while achieving stationarity)
transformer = StationarityTransformer(method="fractional", d=0.4, threshold=1e-3)
stationary_series = transformer.fractional_difference(price_series)
```

### Point-in-Time Joins

```python
from ml.preprocessing.joins import asof_join, embargo_window

# Join market data with corporate events (no lookahead bias)
joined_data = asof_join(
    market_df, events_df,
    on="ts_event",
    by="instrument_id",
    direction="backward"  # only past events
)

# Apply embargo around earnings releases
embargoed_data = embargo_window(
    df, earnings_timestamps,
    embargo_before_ns=3600_000_000_000,  # 1 hour before
    embargo_after_ns=7200_000_000_000    # 2 hours after
)

# Filter out embargoed periods from training set
training_data = embargoed_data.filter(~pl.col("embargo"))  # Polars
# OR: training_data = embargoed_data[~embargoed_data["embargo"]]  # Pandas
```

### Purged Cross-Validation

```python
from ml.preprocessing import PurgedCrossValidator

# Purged walk-forward CV with embargo
cv = PurgedCrossValidator(
    n_splits=5,
    purge_gap=10,      # 10 samples gap to prevent leakage
    embargo_pct=0.1    # 10% embargo after test set
)

for train_idx, test_idx in cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

### Market Microstructure Features

```python
from ml.preprocessing import MarketMicrostructureFeatures

features = MarketMicrostructureFeatures()

# Estimate bid-ask spread from prices
spread = features.roll_spread(prices)

# Measure price impact
lambda_param = features.kyle_lambda(prices, volumes)

# Compute illiquidity ratio
illiquidity = features.amihud_illiquidity(returns, volumes)

# Calculate volume-synchronized informed trading probability
vpin = features.vpin(prices, volumes, bucket_size=50)
```

### Event Ingestion

```python
from datetime import datetime, UTC
from pathlib import Path
from ml.preprocessing import EventIngestionConfig, EventIngestionUtility

cfg = EventIngestionConfig(
    start=datetime(2024, 1, 1, tzinfo=UTC),
    end=datetime(2024, 12, 31, tzinfo=UTC),
    out_dir=Path("data/events"),
    alfred_vintage_dir=Path("data/alfred"),
    economic_series=("CPI", "GDP", "UNEMPLOYMENT"),
    economic_stub_path=Path("stubs/economic_events.csv"),
    corporate_source_path=Path("stubs/earnings.csv"),
    include_options_expiry=True
)

utility = EventIngestionUtility(cfg)
events_path = utility.ingest()  # Returns Path to data/events/events.parquet

# Load and use events
import polars as pl
events_df = pl.read_parquet(events_path)
print(events_df.select(["event_timestamp", "event_type", "name", "importance"]))
```

### Vintage Age Conversion

```python
from pathlib import Path
from ml.preprocessing.vintage_age import convert_vintage_timestamps_to_age
from ml.preprocessing.vintage_age import update_metadata_with_vintage_age
from ml.preprocessing.vintage_age import write_metadata
import json

# Convert vintage timestamps to age features
src = Path("ml_out/full_tft_95/dataset.parquet")
dst = src.with_name("dataset_with_vintage_age.parquet")

result = convert_vintage_timestamps_to_age(
    src, dst,
    timestamp_column="timestamp",
    batch_size=32_768,
    compression="snappy"
)

print(f"Converted columns: {result.vintage_columns} -> {result.age_columns}")
# Example output:
# Converted columns: ('CPI__value_vintage_ts', 'GDP__value_vintage_ts')
#                 -> ('CPI__vintage_age_minutes', 'GDP__vintage_age_minutes')

# Update metadata
metadata_path = src.parent / "dataset_metadata.json"
with metadata_path.open("r") as f:
    metadata = json.load(f)

updated = update_metadata_with_vintage_age(
    metadata,
    vintage_columns=result.vintage_columns,
    age_columns=result.age_columns
)

write_metadata(metadata_path, updated)
```

## Integration Points

### Feature Pipeline Integration

**Stationarity in features:**
- StationarityTransformer is used in cold-path feature engineering pipelines
- Not directly referenced in ml/features/, but available for use in custom preprocessing
- Typically applied during dataset construction rather than online feature computation

**Batch/online parity:**
- Preprocessing transformations are **batch-only** (cold path)
- Online inference uses pre-computed features from FeatureStore, not raw preprocessing

### ML Actor Integration

**Actors do NOT use preprocessing directly:**
- All preprocessing happens in batch dataset construction (cold path)
- Actors consume preprocessed features from FeatureStore
- Hot path constraint: P99 < 5ms rules out DataFrame operations, statistical tests, file I/O

### Data Store Integration

**Temporal validation:**
- asof_join ensures proper ts_event/ts_init handling
- embargo_window preserves Nautilus timestamp schema
- All operations maintain required columns: instrument_id, ts_event, ts_init

**Schema compliance:**
- event_ingestion.py produces normalized schema with ts_event equivalent (event_timestamp)
- vintage_age.py preserves timestamp column (int64 nanoseconds)
- joins.py works with any timestamp column name but defaults to ts_event

### Training Pipeline Integration

**Cross-validation:**
- `ml/training/base.py` imports PurgedCrossValidator (lines 989-1028)
- Used in `BaseTrainer._create_cross_validator()` when `use_purged_cv=True`
- Falls back to sklearn's TimeSeriesSplit if PurgedCrossValidator unavailable

**Dataset preparation:**
- `ml/orchestration/dataset_builder.py` uses vintage_age utilities for post-processing
- `ml/pipelines/build_runner.py` integrates vintage_age conversion step
- event_ingestion.py not directly called by training, but feeds into dataset builders

### CLI Integration

**Standalone CLIs:**
- `ml/cli/convert_vintage_age.py` (142 lines): Batch conversion CLI for vintage timestamps
- `ml/scripts/convert_vintage_age.py` (12 lines): Compatibility wrapper

**Example usage:**
```bash
python -m ml.cli.convert_vintage_age \
    --source ml_out/full_tft_95/dataset.parquet \
    --metadata ml_out/full_tft_95/dataset_metadata.json \
    --batch-size 65536 \
    --overwrite
```

### MLIntegrationManager Integration

**Event ingestion:**
- `ml/core/integration.py` defines `ingest_events(config: EventIngestionConfig)` method (line 238)
- Wraps EventIngestionUtility for centralized event ingestion
- Returns Path to normalized events.parquet

## Implementation Notes

### Performance Optimizations

**JIT compilation:**
- Numba acceleration for fractional weight computation (`_compute_weights_numba`, stationarity.py line 91)
- Degrades gracefully to pure Python if Numba unavailable via `jit_typed` decorator (lines 26-41)

**Vectorization:**
- All operations use vectorized numpy/polars operations
- embargo_window uses boolean array operations instead of loops (joins.py lines 313-321)
- create_lag_features uses Polars/Pandas shift operations, not Python loops

**Memory efficiency:**
- vintage_age.py streams parquet in batches (default 32,768 rows) to avoid loading full file
- In-place operations and pre-allocated arrays where possible
- Polars lazy evaluation patterns for large dataset processing (where used)

**PyArrow streaming:**
- vintage_age uses PyArrow's `iter_batches()` for bounded memory consumption
- ParquetWriter appends batches without intermediate concatenation

### Numerical Stability

**Weight thresholding:**
- Fractional differencing drops weights below `threshold` (default 1e-3) for stability and efficiency (stationarity.py line 156)

**Division by zero:**
- Amihud illiquidity clamps volumes to minimum 1.0 (stationarity.py line 386)
- Kyle lambda checks for non-zero std before regression (line 359)
- Roll spread returns 0.0 if covariance is positive (line 330)

**NaN propagation:**
- Consistent NaN handling across all transformations
- Lagged features use NaN for initial periods without data
- Box-Cox transformation adds shift for negative values (stationarity.py lines 617-621)

**Precision:**
- Uses float64 throughout for numerical accuracy
- vintage_age casts final result to float32 for memory efficiency (vintage_age.py line 81)

### Temporal Correctness

**Strict ordering:**
- asof_join sorts both dataframes before joining (joins.py lines 133-134, 178-179)
- create_lag_features sorts by timestamp_col before applying lags (lines 433, 459)
- All operations maintain proper temporal ordering

**No lookahead:**
- asof_join defaults to "backward" direction (past data only)
- embargo_window explicitly marks periods to exclude
- validate_no_lookahead provides validation layer (lines 327-383)
- PurgedCrossValidator removes samples between train/test sets

**Point-in-time joins:**
- Guaranteed historical data usage only via as-of semantics
- Tolerance parameter prevents matching distant events

**Embargo enforcement:**
- Event-based embargo windows for information quarantine (earnings, FOMC, etc.)
- Configurable before/after periods in nanoseconds

### Error Handling and Validation

**Input validation:**
- PurgedCrossValidator validates n_splits >= 2, purge_gap >= 0, embargo_pct in [0, 1) (stationarity.py lines 729-737)
- EventIngestionConfig validates end > start (event_ingestion.py lines 99-102)
- convert_vintage_timestamps_to_age validates source exists, is .parquet, destination != source (vintage_age.py lines 48-54, 121-123)

**Data quality checks:**
- find_optimal_d checks for sufficient data (>10 samples) before ADF test (stationarity.py line 218)
- vpin handles edge case of insufficient data by returning empty list (line 437-439)
- asof_join validates on/by parameters before backend dispatch (joins.py lines 97-99)

**Framework availability:**
- Graceful handling when Polars/Pandas unavailable via HAS_POLARS, HAS_PANDAS flags
- check_ml_dependencies() raises informative errors with installation instructions

**Statistical validity:**
- ADF test validation for stationarity assessment (stationarity.py lines 202-222)
- Box-Cox requires positive values (handles via shift if needed)

### Academic Compliance

**López de Prado methods:**
- Fractional differencing: Chapter 5 of "Advances in Financial Machine Learning"
- PurgedCrossValidator: Chapter 7 of AFML
- Faithful implementation of algorithms with proper citations in docstrings

**Statistical rigor:**
- ADF test for stationarity detection via statsmodels
- Proper statistical tests and validation procedures

**Literature references:**
- Roll (1984): Bid-ask spread estimation
- Kyle (1985): Lambda price impact measure
- Amihud (2002): Illiquidity ratio
- Easley et al. (2012): VPIN calculation

**Best practices:**
- Incorporates industry best practices for financial ML preprocessing
- Avoids common pitfalls: lookahead bias, data leakage, overfitting

## Testing and Validation

### Test Coverage

**Unit tests:**
- `ml/tests/unit/preprocessing/test_joins.py` (2,623 bytes): Tests asof_join, embargo_window with both Polars and Pandas
- `ml/tests/unit/preprocessing/test_event_ingestion.py` (2,738 bytes): Tests EventIngestionUtility with stubs and ALFRED data
- `ml/tests/unit/preprocessing/test_vintage_age.py` (3,521 bytes): Tests conversion, metadata updates, immutability

**Parametrized testing:**
- test_joins.py uses `@pytest.mark.parametrize("use_polars", [True, False])` for dual-backend testing

**Integration tests:**
- `ml/tests/unit/core/test_integration_event_ingestion.py`: Tests MLIntegrationManager.ingest_events()

**Property tests:**
- No hypothesis-based property tests currently in preprocessing module
- Could benefit from property tests for:
  - Fractional differencing invariants (stationarity, invertibility)
  - asof_join temporal ordering preservation
  - embargo_window coverage properties

### Validation Reports

**Overfitting prevention:**
- `ml/tests/validation_reports/OVERFITTING_PREVENTION_ANALYSIS.md` references PurgedCrossValidator
- Validates purged CV prevents leakage in financial time series

**Test redundancy:**
- `ml/tests/docs/TEST_REDUNDANCY_REPORT.md` mentions preprocessing tests
- No redundant tests identified

## Known Gaps and Future Work

### Missing Functionality

**No stationarity tests in tests/:**
- No unit tests for StationarityTransformer, DataNormalizer, FeatureLagGenerator, MarketMicrostructureFeatures
- Only joins, event_ingestion, and vintage_age have test coverage
- **Gap:** Core stationarity functionality untested despite being 827 lines

**No property tests:**
- No hypothesis-based property tests for fractional differencing invariants
- No metamorphic tests for stationarity transformations
- **Gap:** Academic algorithms lack rigorous validation

**No benchmark tests:**
- No performance tests for JIT-compiled fractional differencing
- No microbenchmarks for large dataset processing
- **Gap:** Performance claims unvalidated

### Integration Gaps

**Limited feature pipeline integration:**
- StationarityTransformer not used in ml/features/ (grep shows zero matches)
- MarketMicrostructureFeatures not referenced in feature engineering code
- **Gap:** Academic features implemented but not integrated

**No online stationarity:**
- All preprocessing is cold-path batch only
- No incremental fractional differencing for streaming data
- **Gap:** Cannot apply stationarity transformations in online inference

**No actor usage:**
- Actors cannot use preprocessing due to hot-path constraints
- All preprocessing happens during dataset construction
- **Gap:** Preprocessing divorced from inference; batch/online parity risk

### Documentation Gaps

**Missing usage examples:**
- No comprehensive examples showing full pipeline (stationarity -> CV -> training)
- Event ingestion examples missing corporate event format specification
- Vintage age examples don't show TFT integration

**Missing theory:**
- No explanation of why fractional differencing preserves memory
- No discussion of optimal d selection trade-offs
- No guidance on when to use robust vs rank normalization

### Configuration Gaps

**Hard-coded constants:**
- FOMC dates hard-coded in event_ingestion.py (lines 167-176) - only 2024
- Holiday logic US-only (lines 361-399)
- No support for non-US exchanges beyond calendar_code parameter

**No preprocessing config:**
- No centralized PreprocessingConfig dataclass
- Each component has ad-hoc parameters
- **Gap:** Inconsistent configuration patterns

### Future Enhancements

**Streaming stationarity:**
- Implement incremental fractional differencing for online use
- Requires buffering past values and updating weights online

**Expanded events:**
- Extend FOMC dates beyond 2024
- Support international holidays and exchange calendars
- Add dividend events, splits, M&A announcements

**Property tests:**
- Add hypothesis tests for stationarity invariants
- Validate asof_join preserves temporal ordering
- Test embargo_window coverage properties

**Performance benchmarks:**
- Add microbenchmarks for vintage_age streaming
- Benchmark Numba vs pure Python fractional differencing
- Validate PyArrow streaming memory bounds

**Type safety:**
- Add runtime type validation for DataFrameLike
- Stricter timestamp column type checking
- Better error messages for schema mismatches

## References

### Academic Literature

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  - Chapter 5: Fractional Differentiation
  - Chapter 7: Cross-Validation in Finance

- Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market." *Journal of Finance*, 39(4), 1127-1139.

- Kyle, A. S. (1985). "Continuous Auctions and Insider Trading." *Econometrica*, 53(6), 1315-1335.

- Amihud, Y. (2002). "Illiquidity and Stock Returns: Cross-Section and Time-Series Effects." *Journal of Financial Markets*, 5(1), 31-56.

- Easley, D., López de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High-Frequency World." *Review of Financial Studies*, 25(5), 1457-1493.

### Related Documentation

- `ml/docs/ROADMAP.md`: Future preprocessing enhancements
- `ml/docs/development/CODING_STANDARDS.md`: Type safety and testing requirements
- `ml/docs/context/context_data.md`: Data loading and schema standards
- `ml/docs/context/context_training.md`: Training pipeline integration
- `ml/docs/context/context_tests.md`: Testing strategy and coverage requirements

### Code References

**Integration points:**
- `ml/core/integration.py` (line 238): MLIntegrationManager.ingest_events()
- `ml/training/base.py` (lines 989-1028): PurgedCrossValidator usage
- `ml/orchestration/dataset_builder.py` (lines 33-35): Vintage age imports
- `ml/pipelines/build_runner.py` (lines 36-38): Vintage age imports

**CLI tools:**
- `ml/cli/convert_vintage_age.py`: Batch vintage conversion CLI
- `ml/scripts/convert_vintage_age.py`: CLI compatibility wrapper

---

**Module Health:**
- ✅ Type annotations complete (stationarity, joins, vintage_age, event_ingestion)
- ✅ No TODO/FIXME/XXX comments
- ✅ No NotImplementedError stubs
- ⚠️  Missing tests for stationarity.py (827 lines untested)
- ⚠️  No property tests or benchmarks
- ✅ Academic compliance (López de Prado, Roll, Kyle, Amihud, Easley)
- ✅ Cold-path only (no hot-path violations)
- ✅ Dual dataframe support (Polars/Pandas)
- ✅ Memory-efficient streaming (vintage_age PyArrow)

**Last audit:** 2025-10-19
**Next recommended audit:** After stationarity tests added
