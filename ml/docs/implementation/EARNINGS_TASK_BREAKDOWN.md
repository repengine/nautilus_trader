# Earnings Integration - Task Breakdown for Parallel Agent Execution

## Overview

Based on the comprehensive refactor review, we have **10 discrete tasks** that can be executed by separate agents with **no file overlap**. Each agent will read the design documentation and coding standards before implementing their task.

**Current Progress: ~60% Complete**
**Remaining Work: 20-29 hours (3-4 days)**

---

## Task Assignment Strategy

### Principle 1: One Agent Per File (No Overlap)
Each agent is assigned to modify exactly ONE file to prevent merge conflicts.

### Principle 2: Read Before Write
Every agent must:
1. Read design documentation
2. Read coding standards and architectural patterns
3. Plan their implementation
4. Execute the implementation
5. Report completion with file references

### Principle 3: Validation After Implementation
After each task completes, a separate validation agent will:
- Run mypy --strict on the modified file
- Run ruff check on the modified file
- Run relevant pytest tests
- Verify protocol compliance
- Report issues or approve

---

## Critical Path Tasks (Must Complete for Architecture Compliance)

### Task 1: FileEarningsStore Implementation
**Priority:** HIGH
**Complexity:** SIMPLE
**Estimated Time:** 2-4 hours
**File:** `/home/nate/projects/nautilus_trader/ml/stores/file_backed.py`

**Agent Instructions:**

**Step 1: Read Documentation**
- Read `playground/EARNINGS_ARCHITECTURE_DESIGN.md` (full document)
- Read `playground/EARNINGS_ARCHITECTURE_DIAGRAM.md` (progressive fallback section)
- Read `ml/docs/development/CODING_STANDARDS.md` (full document)
- Read `ml/docs/architecture/universal_patterns_guide.md` (Pattern #4: Progressive Fallback)
- Read `ml/stores/protocols.py` lines 498-650 (`EarningsStoreProtocol` definition)
- Read `ml/stores/earnings_store.py` lines 53-409 (reference PostgreSQL implementation)

**Step 2: Plan Implementation**
Create a plan that includes:
1. Class name: `FileEarningsStore`
2. Inheritance: Implement `EarningsStoreProtocol`
3. Storage mechanism: Parquet files in `{base_path}/earnings/`
   - `actuals.parquet` for earnings actuals
   - `estimates.parquet` for earnings estimates
4. Methods to implement:
   - `__init__(base_path: Path)`
   - `write_actuals(ticker, period_end, filing_date, eps_diluted, revenue, ts_event, ts_init, **kwargs)`
   - `write_estimates(ticker, estimate_date, period_end, eps_consensus, ts_event, ts_init, **kwargs)`
   - `get_actuals(ticker, start_date=None, end_date=None, as_of_ts=None) -> list[dict]`
   - `get_estimates(ticker, period_end, as_of_ts=None) -> dict | None`
   - `flush()`
5. Point-in-time filtering: Use `as_of_ts` to filter results where `ts_event < as_of_ts`
6. Follow patterns from `FileFeatureStore` in same file

**Step 3: Implementation Requirements**
- Add class AFTER `FileStrategyStore` class (around line 530)
- Use `polars` for parquet read/write operations
- Include comprehensive docstrings (Google style)
- Add type hints for all parameters
- Include error handling with logging
- Preserve point-in-time correctness in queries

**Step 4: Code Template**
```python
class FileEarningsStore:
    """
    File-based earnings store using Parquet backend.

    Provides progressive fallback when PostgreSQL is unavailable.
    Stores earnings actuals and estimates in separate parquet files.

    Parameters
    ----------
    base_path : Path
        Base directory for earnings parquet files
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._actuals_path = base_path / "actuals.parquet"
        self._estimates_path = base_path / "estimates.parquet"
        base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileEarningsStore initialized at {base_path}")

    def write_actuals(self, ...) -> None:
        # Read existing parquet, append new row, write back
        ...

    def get_actuals(self, ticker: str, ..., as_of_ts: int | None = None) -> list[dict]:
        # Read parquet, filter by ticker and as_of_ts, return sorted by period_end
        ...
```

**Step 5: Validation Criteria**
- Implements all methods from `EarningsStoreProtocol`
- Passes mypy --strict type checking
- Passes ruff check with zero violations
- Includes docstrings for all public methods
- Point-in-time queries filter correctly with `ts_event < as_of_ts`

---

### Task 2: FileDataStore Earnings Methods
**Priority:** HIGH
**Complexity:** SIMPLE
**Estimated Time:** 1-2 hours
**File:** `/home/nate/projects/nautilus_trader/ml/stores/file_backed.py`

**Agent Instructions:**

**Step 1: Read Documentation**
- Read `playground/EARNINGS_ARCHITECTURE_DESIGN.md` lines 130-198 (DataStore facade section)
- Read `ml/docs/architecture/universal_patterns_guide.md` (Pattern #2: Protocol-First Design)
- Read `ml/stores/protocols.py` lines 221-318 (`DataStoreFacadeProtocol` earnings methods)
- Read `ml/stores/data_store.py` lines 621-670 (reference DataStore implementation)

**Step 2: Plan Implementation**
1. Locate `FileDataStore` class (currently at line 496)
2. Add `_earnings_store` initialization in `__init__`
3. Add 4 earnings methods:
   - `write_earnings_actual(...)`
   - `write_earnings_estimate(...)`
   - `get_earnings_actuals_at_or_before(...)`
   - `get_earnings_estimate_at_or_before(...)`
4. Delegate to `self._earnings_store` (FileEarningsStore instance)
5. Return `DataEvent` from write methods

**Step 3: Implementation Requirements**
- Modify `FileDataStore.__init__()` to accept optional `earnings_store` parameter
- If not provided, create `FileEarningsStore(base_path / "earnings")`
- Add earnings methods that delegate to `_earnings_store`
- Return `DataEvent` objects from write methods (follow pattern from existing methods)
- Use same error handling and logging patterns as existing DataStore methods

**Step 4: Code Template**
```python
class FileDataStore:
    def __init__(
        self,
        *,
        base_path: Path,
        earnings_store: EarningsStoreProtocol | None = None,
        # ... existing params
    ) -> None:
        # ... existing init ...
        self._earnings_store = earnings_store or FileEarningsStore(base_path / "earnings")

    def write_earnings_actual(
        self,
        *,
        ticker: str,
        period_end: str,
        filing_date: str,
        eps_diluted: float | None,
        revenue: float | None,
        ts_event: int,
        ts_init: int,
        **kwargs: Any,
    ) -> DataEvent:
        """Write earnings actual to file store."""
        self._earnings_store.write_actuals(
            ticker=ticker,
            period_end=period_end,
            filing_date=filing_date,
            eps_diluted=eps_diluted,
            revenue=revenue,
            ts_event=ts_event,
            ts_init=ts_init,
            **kwargs,
        )

        return DataEvent(
            event_id=str(uuid.uuid4()),
            dataset_id="earnings_actuals",
            instrument_id=ticker,
            operation="write_earnings_actual",
            source="file",
            run_id=str(uuid.uuid4()),
            ts_min=ts_event,
            ts_max=ts_event,
            record_count=1,
            status="success",
            error_message=None,
            created_at=ts_init,
            metadata={"filing_date": filing_date},
        )

    def get_earnings_actuals_at_or_before(...) -> list[dict]:
        # Delegate to _earnings_store
        ...
```

**Step 5: Validation Criteria**
- `FileDataStore` conforms to `DataStoreFacadeProtocol` (check with isinstance)
- All 4 earnings methods implemented
- Passes mypy --strict
- Passes ruff check
- Returns proper `DataEvent` objects from write methods

**⚠️ IMPORTANT:** This task depends on Task 1 completing first (FileEarningsStore must exist).

---

### Task 3: Integration Manager File Fallback Wiring
**Priority:** HIGH
**Complexity:** SIMPLE
**Estimated Time:** 1 hour
**File:** `/home/nate/projects/nautilus_trader/ml/core/integration.py`

**Agent Instructions:**

**Step 1: Read Documentation**
- Read `playground/EARNINGS_ARCHITECTURE_DESIGN.md` lines 651-707 (Progressive Fallback section)
- Read `playground/EARNINGS_ARCHITECTURE_DIAGRAM.md` (Progressive Fallback Flow diagram)
- Read `ml/docs/architecture/universal_patterns_guide.md` (Pattern #4: Progressive Fallback)

**Step 2: Plan Implementation**
1. Locate `_init_stores()` method in `MLIntegrationManager` class
2. Find the file fallback section (where `self._file_fallback == True`)
3. Add earnings store initialization:
   - Create `FileEarningsStore` instance
   - Pass to `FileDataStore` constructor
4. Ensure earnings store is also initialized in PostgreSQL path (already done via DataStore)

**Step 3: Implementation Requirements**
- Modify the file fallback block (around line 356-361)
- Import `FileEarningsStore` at top of file
- Create earnings store with `file_root / "earnings"` path
- Pass to `FileDataStore` constructor as `earnings_store=...` parameter
- Add log message confirming earnings store initialization

**Step 4: Code Template**
```python
# In _init_stores() method, file fallback section:
if self._file_fallback:
    file_root = self._file_store_path
    self.feature_store = FileFeatureStore(base_path=file_root / "features")
    self.model_store = FileModelStore(base_path=file_root / "models")
    self.strategy_store = FileStrategyStore(base_path=file_root / "strategies")

    # NEW: Initialize earnings store for file fallback
    from ml.stores.file_backed import FileEarningsStore
    earnings_store = FileEarningsStore(base_path=file_root / "earnings")
    logger.info(f"FileEarningsStore initialized at {file_root / 'earnings'}")

    self.data_store = FileDataStore(
        base_path=file_root / "datastore",
        earnings_store=earnings_store,  # Add this parameter
    )
```

**Step 5: Validation Criteria**
- Import statement added for `FileEarningsStore`
- Earnings store created in file fallback path
- Passed to `FileDataStore` constructor
- Log message confirms initialization
- Passes mypy --strict
- Passes ruff check

**⚠️ IMPORTANT:** This task depends on Tasks 1 & 2 completing first.

---

### Task 4: TFT Builder DataStore Integration
**Priority:** HIGH
**Complexity:** MEDIUM
**Estimated Time:** 4-6 hours
**File:** `/home/nate/projects/nautilus_trader/ml/data/tft_dataset_builder.py`

**Agent Instructions:**

**Step 1: Read Documentation**
- Read `playground/EARNINGS_ARCHITECTURE_DESIGN.md` lines 453-595 (TFT Builder Integration section)
- Read `playground/EARNINGS_ARCHITECTURE_DIAGRAM.md` (Data Flow: End-to-End Example)
- Read `playground/EARNINGS_INTEGRATION_PLAN.md` lines 200-350 (TFT Builder implementation details)
- Read `ml/docs/development/CODING_STANDARDS.md` (Schema adherence, Type annotations)
- Read `ml/stores/protocols.py` lines 221-318 (DataStoreFacadeProtocol earnings methods)

**Step 2: Plan Implementation**
1. Add `include_earnings: bool = False` parameter to `__init__`
2. Add `earnings_lag_days: int = 1` parameter to `__init__`
3. Store `self.include_earnings` and `self.earnings_lag_days` as instance variables
4. Implement `_fetch_earnings_features(ticker, timestamps, as_of_date)` method:
   - Query `self.data_store.get_earnings_actuals_at_or_before()`
   - Fetch estimates for surprise calculation
   - Compute features using `ml.features.earnings` functions
   - Build quarterly DataFrame with features
   - Add publication lag adjustment
   - Perform ASOF join to expand to bar-level
   - Return Polars DataFrame with feature columns

5. Integrate in `_process_symbol_polars()` method:
   - Add earnings join after L2 features (around line 1442 per investigation report)
   - Handle null filling and availability mask

6. Integrate in `_process_symbol_pandas()` method:
   - Add earnings join after L2 features (around line 1542 per investigation report)
   - Convert Polars to Pandas for merge

7. Add to student_mode exclusions (around line 157)

**Step 3: Implementation Requirements**
- Follow exact pattern from macro features integration (reference implementation in same file)
- Use Polars for data manipulation in `_fetch_earnings_features()`
- Import earnings feature computation functions from `ml.features.earnings`
- Add proper error handling with try/except blocks
- Add debug logging at key points
- Fill nulls with 0 for numeric columns
- Add `is_earnings_available` mask column
- Validate ticker normalization (strip venue suffixes like `.NASDAQ`)

**Step 4: Code Template**
```python
# In __init__:
self.include_earnings = include_earnings and data_store is not None
self.earnings_lag_days = earnings_lag_days

# New helper method:
def _fetch_earnings_features(
    self,
    ticker: str,
    timestamps: pl.Series,
    as_of_date: datetime | None = None,
) -> pl.DataFrame | None:
    """
    Fetch earnings features for a ticker at specified timestamps.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL')
    timestamps : pl.Series
        Polars Series of bar timestamps (datetime64[ns])
    as_of_date : datetime | None
        Backtest cutoff date (only use data filed before this date)

    Returns
    -------
    pl.DataFrame | None
        DataFrame with earnings feature columns, or None if no data available
    """
    if not self.data_store or not self.include_earnings:
        return None

    # Normalize ticker (strip venue suffix)
    base_ticker = ticker.split('.')[0]

    # Convert as_of_date to nanoseconds
    max_ts_ns = int(timestamps.max().timestamp() * 1e9)
    as_of_ts = max_ts_ns
    if as_of_date:
        if as_of_date.tzinfo is None:
            as_of_date = as_of_date.replace(tzinfo=timezone.utc)
        as_of_ts = int(as_of_date.timestamp() * 1_000_000_000)

    # Fetch actuals (last 5 quarters for YoY growth)
    actuals = self.data_store.get_earnings_actuals_at_or_before(
        ticker=base_ticker,
        ts_event=as_of_ts,
        limit=5,
    )

    if not actuals or len(actuals) < 1:
        logger.debug(f"No earnings actuals found for {base_ticker}")
        return None

    # Build EPS series
    eps_series = np.array([a.get('eps_diluted') or 0.0 for a in actuals], dtype=np.float64)

    # Fetch estimates for surprise calculation
    estimates = []
    for actual in actuals:
        estimate = self.data_store.get_earnings_estimate_at_or_before(
            ticker=base_ticker,
            period_end=actual['period_end'],
            ts_event=actual['ts_event'],
        )
        est_value = estimate.get('eps_consensus') if estimate else actual.get('eps_diluted')
        estimates.append(est_value or 0.0)

    estimates_array = np.array(estimates, dtype=np.float64)

    # Compute features in batch
    from ml.features.earnings import (
        compute_earnings_surprise_batch,
        compute_earnings_growth_batch,
        compute_earnings_momentum_batch,
    )

    surprise_features = compute_earnings_surprise_batch(eps_series, estimates_array)
    growth_features = compute_earnings_growth_batch(eps_series)
    momentum_features = compute_earnings_momentum_batch(
        surprise_features['eps_surprise_q0'],
        eps_series
    )

    # Build quarterly feature DataFrame
    quarterly_data = {
        'period_end': [a['period_end'] for a in actuals],
        'filing_date': [a['filing_date'] for a in actuals],
        f'eps_surprise_q0_{base_ticker}': surprise_features['eps_surprise_q0'],
        f'eps_surprise_pct_q0_{base_ticker}': surprise_features['eps_surprise_pct_q0'],
        f'eps_growth_yoy_{base_ticker}': growth_features['eps_growth_yoy'],
        f'eps_growth_qoq_{base_ticker}': growth_features['eps_growth_qoq'],
        f'earnings_beat_streak_{base_ticker}': momentum_features['earnings_beat_streak'],
        f'eps_volatility_4q_{base_ticker}': momentum_features['eps_volatility_4q'],
    }

    quarterly_df = pl.DataFrame(quarterly_data)

    # Convert filing_date to datetime
    quarterly_df = quarterly_df.with_columns([
        pl.col('filing_date').str.strptime(pl.Date, "%Y-%m-%d"),
    ])

    # Add publication lag
    lag_duration = pl.duration(days=self.earnings_lag_days)
    quarterly_df = quarterly_df.with_columns([
        (pl.col('filing_date').cast(pl.Datetime('ns')) + lag_duration).alias('effective_date')
    ])

    # Expand to minute-level via ASOF join
    bar_df = pl.DataFrame({'timestamp': timestamps})

    result = bar_df.join_asof(
        quarterly_df,
        left_on='timestamp',
        right_on='effective_date',
        strategy='backward',
    )

    # Select output columns (drop internal columns)
    feature_cols = [col for col in result.columns if col.startswith((
        'eps_surprise', 'eps_growth', 'earnings_beat', 'eps_volatility'
    ))]

    result = result.select(['timestamp'] + feature_cols)

    logger.debug(
        f"Fetched earnings features for {base_ticker}: {len(result)} rows, "
        f"{len(feature_cols)} features from {len(actuals)} quarters"
    )

    return result

# In _process_symbol_polars (after L2 join, around line 1442):
if self.include_earnings:
    try:
        ts_series = dataset.select(pl.col("timestamp"))["timestamp"]

        earnings_df = self._fetch_earnings_features(
            ticker=symbol,
            timestamps=ts_series,
            as_of_date=end,
        )

        if earnings_df is not None and not earnings_df.is_empty():
            # Cast timestamp to match dataset
            if earnings_df["timestamp"].dtype != pl.Datetime:
                earnings_df = earnings_df.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("ns", "UTC"))
                )

            # Join earnings features
            before_cols = set(dataset.columns)
            dataset = dataset.join(earnings_df, on="timestamp", how="left")

            # Fill nulls for numeric earnings columns
            earnings_cols = [c for c in dataset.columns if c not in before_cols]
            if earnings_cols:
                fills = []
                for c in earnings_cols:
                    try:
                        if dataset.schema[c].is_numeric():
                            fills.append(pl.col(c).fill_null(0))
                    except Exception:
                        pass
                if fills:
                    dataset = dataset.with_columns(fills)

                # Add availability mask
                has_any = None
                for c in earnings_cols:
                    expr = pl.col(c).is_not_null()
                    has_any = expr if has_any is None else (has_any | expr)

                if has_any is not None:
                    dataset = dataset.with_columns([
                        (has_any.cast(pl.Int32)).alias("is_earnings_available"),
                    ])
        else:
            logger.debug(f"No earnings features available for {symbol}")

    except Exception as exc:
        logger.warning(f"Earnings feature join failed for {symbol}: {exc}", exc_info=True)

# In _process_symbol_pandas (after L2 join, around line 1542):
if self.include_earnings:
    try:
        ts_series_pl = pl.Series("timestamp", dataset["timestamp"].astype("datetime64[ns]"))

        earnings_df_pl = self._fetch_earnings_features(
            ticker=symbol,
            timestamps=ts_series_pl,
            as_of_date=None,
        )

        if earnings_df_pl is not None and not earnings_df_pl.is_empty():
            earnings_df_pd = earnings_df_pl.to_pandas()

            dataset = dataset.merge(earnings_df_pd, on="timestamp", how="left")

            earnings_cols = [c for c in earnings_df_pd.columns if c != "timestamp"]
            dataset[earnings_cols] = dataset[earnings_cols].fillna(0)

            dataset["is_earnings_available"] = (
                dataset[earnings_cols].notna().any(axis=1).astype("int32")
            )
    except Exception as exc:
        logger.warning(f"Earnings feature join failed for {symbol}: {exc}", exc_info=True)

# In student_mode section (around line 157):
if self.student_mode:
    self.include_macro = False
    self.include_events = False
    self.include_l2 = False
    self.include_earnings = False  # Add this line
```

**Step 5: Validation Criteria**
- Constructor accepts `include_earnings` and `earnings_lag_days` parameters
- `_fetch_earnings_features()` method implemented correctly
- Integration in both Polars and Pandas paths
- ASOF join logic correct (backward strategy)
- Null filling and availability mask added
- Student mode disables earnings features
- Passes mypy --strict
- Passes ruff check
- Existing tests don't break

---

### Task 5: DataRegistry Earnings Contracts
**Priority:** MEDIUM
**Complexity:** MEDIUM
**Estimated Time:** 2-3 hours
**File:** `/home/nate/projects/nautilus_trader/ml/registry/data_registry.py`

**Agent Instructions:**

**Step 1: Read Documentation**
- Read `playground/EARNINGS_ARCHITECTURE_DESIGN.md` lines 313-343 (DataRegistry Contracts section)
- Read `ml/docs/architecture/universal_patterns_guide.md` (Pattern #2: Protocol-First Design)
- Read `ml/registry/dataclasses.py` to understand `DataContract`, `ValidationRule` structures

**Step 2: Plan Implementation**
1. Define `EARNINGS_ACTUALS_CONTRACT` constant
2. Define `EARNINGS_ESTIMATES_CONTRACT` constant
3. Register contracts in `DataRegistry.__init__()` or separate registration method
4. Add validation rules:
   - NOT_NULL for required fields
   - RANGE checks for timestamps
   - TYPE checks for numeric fields

**Step 3: Implementation Requirements**
- Add contract definitions at module level (after imports)
- Follow pattern from existing contracts in the file
- Include comprehensive field validation
- Register contracts during DataRegistry initialization
- Add docstrings explaining the contracts

**Step 4: Code Template**
```python
# Add after imports at module level:
from ml.registry.dataclasses import (
    DataContract,
    ValidationRule,
    ValidationRuleType,
)

EARNINGS_ACTUALS_CONTRACT = DataContract(
    dataset_id="earnings_actuals",
    schema_version="1.0.0",
    required_fields=["ticker", "period_end", "filing_date", "ts_event", "ts_init"],
    optional_fields=["eps_basic", "eps_diluted", "revenue", "net_income", "operating_income",
                     "shares_outstanding", "filing_type", "fiscal_year", "fiscal_quarter"],
    field_types={
        "ticker": "string",
        "period_end": "date",
        "filing_date": "date",
        "ts_event": "int64",
        "ts_init": "int64",
        "eps_basic": "float64",
        "eps_diluted": "float64",
        "revenue": "float64",
        "net_income": "float64",
        "operating_income": "float64",
        "shares_outstanding": "int64",
        "filing_type": "string",
        "fiscal_year": "int32",
        "fiscal_quarter": "int32",
    },
    validation_rules=[
        ValidationRule(
            rule_type=ValidationRuleType.NOT_NULL,
            field="ticker",
        ),
        ValidationRule(
            rule_type=ValidationRuleType.NOT_NULL,
            field="period_end",
        ),
        ValidationRule(
            rule_type=ValidationRuleType.NOT_NULL,
            field="filing_date",
        ),
        ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field="ts_event",
            params={"min": 0, "max": 9999999999999999999},
        ),
        ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field="ts_init",
            params={"min": 0, "max": 9999999999999999999},
        ),
    ],
)

EARNINGS_ESTIMATES_CONTRACT = DataContract(
    dataset_id="earnings_estimates",
    schema_version="1.0.0",
    required_fields=["ticker", "estimate_date", "period_end", "ts_event", "ts_init"],
    optional_fields=["eps_consensus", "revenue_consensus", "num_analysts"],
    field_types={
        "ticker": "string",
        "estimate_date": "date",
        "period_end": "date",
        "ts_event": "int64",
        "ts_init": "int64",
        "eps_consensus": "float64",
        "revenue_consensus": "float64",
        "num_analysts": "int32",
    },
    validation_rules=[
        ValidationRule(
            rule_type=ValidationRuleType.NOT_NULL,
            field="ticker",
        ),
        ValidationRule(
            rule_type=ValidationRuleType.NOT_NULL,
            field="estimate_date",
        ),
        ValidationRule(
            rule_type=ValidationRuleType.NOT_NULL,
            field="period_end",
        ),
        ValidationRule(
            rule_type=ValidationRuleType.RANGE,
            field="ts_event",
            params={"min": 0, "max": 9999999999999999999},
        ),
    ],
)

# In DataRegistry class (find appropriate location in __init__ or create registration method):
def _register_default_contracts(self) -> None:
    """Register default data contracts for core datasets."""
    # Register earnings contracts
    self.register_contract(EARNINGS_ACTUALS_CONTRACT)
    self.register_contract(EARNINGS_ESTIMATES_CONTRACT)
    logger.info("Registered earnings data contracts")

    # ... existing contract registrations
```

**Step 5: Validation Criteria**
- Both contracts defined with all required fields
- Validation rules include NOT_NULL and RANGE checks
- Contracts registered during DataRegistry initialization
- Passes mypy --strict
- Passes ruff check
- No breaking changes to existing functionality

---

### Task 6: DataStore Earnings Tests
**Priority:** MEDIUM
**Complexity:** MEDIUM
**Estimated Time:** 4-6 hours
**Files:**
- Create `/home/nate/projects/nautilus_trader/ml/tests/unit/stores/test_data_store_earnings.py`
- Create `/home/nate/projects/nautilus_trader/ml/tests/integration/test_earnings_datastore_integration.py`

**Agent Instructions:**

**Step 1: Read Documentation**
- Read `playground/EARNINGS_INTEGRATION_SUMMARY.md` lines 450-650 (Testing Strategy section)
- Read `ml/docs/development/CODING_STANDARDS.md` (Testing and coverage section)
- Read `ml/tests/conftest.py` to understand test fixtures
- Read `ml/tests/unit/stores/test_feature_store.py` as reference pattern

**Step 2: Plan Implementation**

**Unit Tests File (`test_data_store_earnings.py`):**
1. Test `write_earnings_actual()` and read back
2. Test `write_earnings_estimate()` and read back
3. Test point-in-time correctness (future filings not visible)
4. Test progressive fallback to DummyEarningsStore
5. Test contract validation (valid and invalid data)
6. Test null handling
7. Test limit parameter in queries

**Integration Tests File (`test_earnings_datastore_integration.py`):**
1. End-to-end with real PostgreSQL
2. TFT builder integration with DataStore
3. Actor earnings access via `self.data_store`
4. File fallback mode (FileEarningsStore)
5. Performance benchmarks (P99 < 10ms)

**Step 3: Implementation Requirements**
- Use pytest fixtures for test setup/teardown
- Use test database URL from environment or fallback
- Include parametrized tests for different scenarios
- Add performance benchmarks with timing assertions
- Follow naming convention: `test_{function}_when_{condition}_returns_{expected}`
- Include both positive and negative test cases
- Achieve ≥90% coverage on new code

**Step 4: Code Template**

**Unit Tests:**
```python
"""Unit tests for DataStore earnings methods."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pytest

from ml.stores.data_store import DataStore
from ml.stores.earnings_store import DummyEarningsStore
from ml.registry import DataRegistry


@pytest.fixture
def test_data_store(tmp_path):
    """Create DataStore with test database."""
    registry = DataRegistry(registry_path=tmp_path / "registry")
    store = DataStore(
        connection_string="postgresql://postgres:postgres@localhost:5432/test",
        registry=registry,
        schema="test",
    )
    yield store
    # Cleanup if needed


def test_write_and_read_earnings_actual(test_data_store):
    """Test writing and reading earnings actual via DataStore."""
    filing_ts = int(datetime(2024, 4, 1, tzinfo=timezone.utc).timestamp() * 1e9)

    # Write earnings actual
    test_data_store.write_earnings_actual(
        ticker="AAPL",
        period_end="2024-03-31",
        filing_date="2024-04-01",
        eps_diluted=2.52,
        revenue=90753000000,
        ts_event=filing_ts,
        ts_init=int(time.time() * 1e9),
    )

    # Read back
    actuals = test_data_store.get_earnings_actuals_at_or_before(
        ticker="AAPL",
        ts_event=filing_ts + 1,  # Just after filing
        limit=5,
    )

    assert len(actuals) == 1
    assert actuals[0]['eps_diluted'] == 2.52
    assert actuals[0]['ticker'] == 'AAPL'


def test_point_in_time_correctness(test_data_store):
    """Test that future filings are not visible in backtest."""
    filing_ts = int(datetime(2024, 4, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    before_ts = int(datetime(2024, 3, 31, tzinfo=timezone.utc).timestamp() * 1e9)
    after_ts = int(datetime(2024, 4, 2, tzinfo=timezone.utc).timestamp() * 1e9)

    # Write filing on 2024-04-01
    test_data_store.write_earnings_actual(
        ticker="AAPL",
        period_end="2024-03-31",
        filing_date="2024-04-01",
        eps_diluted=2.52,
        ts_event=filing_ts,
        ts_init=int(time.time() * 1e9),
    )

    # Query before filing
    actuals_before = test_data_store.get_earnings_actuals_at_or_before(
        ticker="AAPL",
        ts_event=before_ts,
        limit=5,
    )

    # Query after filing
    actuals_after = test_data_store.get_earnings_actuals_at_or_before(
        ticker="AAPL",
        ts_event=after_ts,
        limit=5,
    )

    assert len(actuals_before) == 0  # Not visible before ts_event
    assert len(actuals_after) == 1   # Visible after ts_event


def test_progressive_fallback_to_dummy_store():
    """Test fallback to DummyEarningsStore when PostgreSQL unavailable."""
    # Intentionally bad connection string
    registry = DataRegistry(registry_path="/tmp/test_registry")
    store = DataStore(
        connection_string="postgresql://invalid:5432/db",
        registry=registry,
    )

    # Should fall back to DummyEarningsStore
    filing_ts = int(datetime(2024, 4, 1, tzinfo=timezone.utc).timestamp() * 1e9)

    store.write_earnings_actual(
        ticker="AAPL",
        period_end="2024-03-31",
        filing_date="2024-04-01",
        eps_diluted=2.52,
        ts_event=filing_ts,
        ts_init=int(time.time() * 1e9),
    )

    actuals = store.get_earnings_actuals_at_or_before(
        ticker="AAPL",
        ts_event=filing_ts + 1,
        limit=5,
    )

    assert len(actuals) == 1
    assert actuals[0]['eps_diluted'] == 2.52
    # Data stored in memory (not persisted)


def test_earnings_query_performance(test_data_store):
    """Ensure earnings queries meet SLA (P99 < 10ms)."""
    # Populate with 100 quarters
    base_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    for i in range(100):
        test_data_store.write_earnings_actual(
            ticker="AAPL",
            period_end=f"2024-{(i%12)+1:02d}-01",
            filing_date=f"2024-{(i%12)+1:02d}-01",
            eps_diluted=2.0 + (i * 0.01),
            ts_event=base_ts + (i * 86400_000_000_000),
            ts_init=int(time.time() * 1e9),
        )

    # Benchmark queries
    latencies = []
    for _ in range(1000):
        start = time.perf_counter()
        actuals = test_data_store.get_earnings_actuals_at_or_before(
            ticker="AAPL",
            ts_event=base_ts + (50 * 86400_000_000_000),
            limit=5,
        )
        latencies.append(time.perf_counter() - start)

    p99_latency_ms = np.percentile(latencies, 99) * 1000

    assert p99_latency_ms < 10.0  # P99 < 10ms
    assert len(actuals) == 5
```

**Integration Tests:**
```python
"""Integration tests for earnings DataStore with TFT builder and actors."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ml.data.tft_dataset_builder import TFTDatasetBuilder
from ml.stores.data_store import DataStore
from ml.registry import DataRegistry


@pytest.fixture
def integrated_data_store(test_db_url):
    """Create DataStore with real PostgreSQL."""
    registry = DataRegistry(registry_path="/tmp/test_registry")
    store = DataStore(
        connection_string=test_db_url,
        registry=registry,
    )
    yield store


def test_tft_builder_with_earnings(integrated_data_store, parquet_catalog):
    """Test TFT builder with earnings features end-to-end."""
    # Ingest earnings data
    filing_ts = int(datetime(2024, 4, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    integrated_data_store.write_earnings_actual(
        ticker="AAPL",
        period_end="2024-03-31",
        filing_date="2024-04-01",
        eps_diluted=2.52,
        revenue=90753000000,
        ts_event=filing_ts,
        ts_init=int(time.time() * 1e9),
    )

    # Build TFT dataset
    builder = TFTDatasetBuilder(
        catalog=parquet_catalog,
        symbols=["AAPL"],
        data_store=integrated_data_store,
        include_earnings=True,
    )

    df = builder.build_training_dataset(
        start=datetime(2024, 4, 1),
        end=datetime(2024, 4, 2),
    )

    # Verify earnings features
    assert "eps_surprise_q0_AAPL" in df.columns
    assert "eps_growth_yoy_AAPL" in df.columns
    assert "is_earnings_available" in df.columns
    assert df["eps_surprise_q0_AAPL"].notna().sum() > 0
```

**Step 5: Validation Criteria**
- All unit tests pass
- All integration tests pass
- Coverage ≥90% on DataStore earnings methods
- Performance benchmarks meet SLA
- Tests work with both PostgreSQL and DummyStore
- Passes mypy --strict
- Passes ruff check

---

## Documentation Tasks (Low Priority)

### Task 7: Earnings Integration Guide
**Priority:** LOW
**Complexity:** SIMPLE
**Estimated Time:** 2-3 hours
**File:** Create `/home/nate/projects/nautilus_trader/ml/docs/guides/earnings_integration_guide.md`

**Agent Instructions:**

**Step 1: Read Documentation**
- Read `playground/EARNINGS_ARCHITECTURE_DESIGN.md` (full document)
- Read `playground/EARNINGS_INTEGRATION_SUMMARY.md` (code examples sections)
- Read `ml/docs/README.md` to understand documentation structure

**Step 2: Plan Documentation**
1. Overview section explaining earnings data flow
2. Ingestion guide (how to write earnings via DataStore)
3. Query guide (how to read earnings in actors)
4. TFT builder guide (how to include earnings features)
5. Code examples for each use case
6. Troubleshooting section

**Step 3: Content Template**
```markdown
# Earnings Data Integration Guide

## Overview

Earnings data (actuals and estimates) flows through the Universal ML Architecture's
4-store + 4-registry pattern, ensuring contract validation, lineage tracking, and
progressive fallback.

## Data Flow

Raw earnings data → DataStore → EarningsStore → PostgreSQL
Computed features → FeatureStore → FeatureRegistry

## Ingesting Earnings Data

### Writing Earnings Actuals

```python
from ml.stores.data_store import DataStore
from datetime import datetime, timezone

data_store = DataStore(connection_string="postgresql://...")

filing_date = datetime(2024, 4, 1, tzinfo=timezone.utc)
ts_event = int(filing_date.timestamp() * 1e9)

data_store.write_earnings_actual(
    ticker="AAPL",
    period_end="2024-03-31",
    filing_date="2024-04-01",
    eps_diluted=2.52,
    revenue=90_753_000_000,
    ts_event=ts_event,
    ts_init=int(datetime.now(timezone.utc).timestamp() * 1e9),
)
```

[... continue with more sections ...]
```

**Step 4: Validation Criteria**
- Covers all use cases from design document
- Includes working code examples
- Links to API reference documentation
- Troubleshooting section addresses common issues
- Follows markdown formatting standards

---

### Task 8: Add Deprecation Warnings
**Priority:** LOW
**Complexity:** SIMPLE
**Estimated Time:** 1 hour
**File:** `/home/nate/projects/nautilus_trader/ml/stores/earnings_store.py`

**Agent Instructions:**

**Step 1: Read Documentation**
- Read `playground/EARNINGS_INTEGRATION_SUMMARY.md` lines 360-380 (Migration Path section)
- Read Python warnings module documentation

**Step 2: Plan Implementation**
1. Add deprecation warning to `EarningsStore.__init__()`
2. Warning message should direct users to use `DataStore.get_earnings_*()` methods
3. Use `DeprecationWarning` category
4. Set `stacklevel=2` to show caller location

**Step 3: Code Template**
```python
import warnings

class EarningsStore:
    def __init__(self, connection_string: str, schema: str = "ml") -> None:
        warnings.warn(
            "Direct EarningsStore usage is deprecated and will be removed in v2.0. "
            "Use DataStore.get_earnings_*() methods instead. "
            "See migration guide: ml/docs/migrations/earnings_datastore_migration.md",
            DeprecationWarning,
            stacklevel=2,
        )
        # ... rest of init
```

**Step 4: Validation Criteria**
- Warning is emitted when EarningsStore is instantiated directly
- Warning message is clear and actionable
- Stacklevel shows caller location
- Passes ruff check

---

### Task 9: Create Migration Guide
**Priority:** LOW
**Complexity:** SIMPLE
**Estimated Time:** 1-2 hours
**File:** Create `/home/nate/projects/nautilus_trader/ml/docs/migrations/earnings_datastore_migration.md`

**Agent Instructions:**

**Step 1: Read Documentation**
- Read `playground/EARNINGS_INTEGRATION_SUMMARY.md` lines 315-415 (Migration Path section)
- Read existing migration guides in `ml/docs/migrations/` (if any)

**Step 2: Plan Content**
1. Why migration is needed (architectural compliance)
2. Before/after code examples
3. Step-by-step migration instructions
4. Breaking changes timeline
5. FAQ section

**Step 3: Content Template**
```markdown
# Earnings DataStore Migration Guide

## Why Migrate?

Direct `EarningsStore` usage bypasses the Universal ML Architecture's 4-store + 4-registry
pattern, preventing contract validation, lineage tracking, and progressive fallback.

## Timeline

- **Now:** Old APIs work with deprecation warnings
- **v1.5 (Month 6):** Deprecation warnings become errors in strict mode
- **v2.0 (Month 12):** Old APIs removed (breaking change)

## Migration Steps

### Before (Deprecated)
```python
from ml.stores.earnings_store import EarningsStore

earnings_store = EarningsStore(connection_string)
actuals = earnings_store.get_actuals(ticker, as_of_ts)
```

### After (Recommended)
```python
from ml.stores.data_store import DataStore

data_store = DataStore(connection_string, registry)
actuals = data_store.get_earnings_actuals_at_or_before(ticker, ts_event)
```

[... continue with more sections ...]
```

**Step 4: Validation Criteria**
- Clear migration path documented
- Before/after examples for all use cases
- Breaking changes timeline included
- FAQ addresses common questions

---

### Task 10: Update Architecture Documentation
**Priority:** LOW
**Complexity:** SIMPLE
**Estimated Time:** 1 hour
**Files:**
- `/home/nate/projects/nautilus_trader/ml/docs/architecture/universal_patterns_guide.md`
- `/home/nate/projects/nautilus_trader/ml/docs/development/CODING_STANDARDS.md`

**Agent Instructions:**

**Step 1: Read Documentation**
- Read both files to understand current structure
- Read `playground/EARNINGS_ARCHITECTURE_DESIGN.md` for content to add

**Step 2: Plan Updates**

**universal_patterns_guide.md:**
- Add earnings example to Pattern #1 (4-store + 4-registry)
- Add earnings example to Pattern #4 (Progressive Fallback)

**CODING_STANDARDS.md:**
- Add earnings to "Data Storage" section
- Add example of earnings usage following standards

**Step 3: Content to Add**

**universal_patterns_guide.md (Pattern #1 section):**
```markdown
### Example: Earnings Data Flow

Raw earnings data flows through DataStore:
```python
# Write earnings actual
data_store.write_earnings_actual(
    ticker="AAPL",
    eps_diluted=2.52,
    ts_event=filing_ts,
    ts_init=now_ts,
)

# Query with point-in-time correctness
actuals = data_store.get_earnings_actuals_at_or_before(
    ticker="AAPL",
    ts_event=backtest_ts,
)
```

Computed earnings features flow through FeatureStore:
```python
from ml.features.earnings import compute_earnings_surprise_batch

surprise = compute_earnings_surprise_batch(actuals, estimates)
feature_store.write_features(features=surprise, ...)
```
```

**Step 4: Validation Criteria**
- Earnings examples added to relevant sections
- Examples follow coding standards
- No breaking changes to existing content
- Links to earnings integration guide included

---

## Validation Tasks

### Validation Agent Instructions

After EACH task completes, assign a validation agent to:

1. **Run Static Analysis**
   ```bash
   mypy --strict <modified_file>
   ruff check <modified_file>
   ```

2. **Run Tests**
   ```bash
   pytest <related_test_files> -v
   pytest ml/tests/unit/ -k earnings  # For earnings-related tests
   pytest ml/tests/integration/ -k earnings
   ```

3. **Check Protocol Compliance**
   - Verify class implements required protocols using `isinstance()` checks
   - Verify all protocol methods have correct signatures

4. **Check Performance** (if applicable)
   - Run performance benchmarks
   - Verify P99 latency < 10ms for data queries

5. **Generate Validation Report**
   ```markdown
   # Validation Report: Task X

   ## Static Analysis
   - ✅ mypy --strict: PASSED
   - ✅ ruff check: PASSED

   ## Tests
   - ✅ Unit tests: X/X passed
   - ✅ Integration tests: X/X passed

   ## Protocol Compliance
   - ✅ Implements XProtocol correctly

   ## Performance
   - ✅ P99 latency: X.Xms (target: <10ms)

   ## Issues Found
   - None

   ## Approval
   ✅ APPROVED - Ready for merge
   ```

---

## Task Dependency Graph

```
Task 1 (FileEarningsStore)
   ↓
Task 2 (FileDataStore earnings methods)
   ↓
Task 3 (Integration manager wiring)
   ↓
Task 4 (TFT builder integration) ←─── Can run in parallel ───→ Task 5 (DataRegistry contracts)
   ↓                                                                      ↓
Task 6 (DataStore tests) ←─────────────────────────────────────────────┘
   ↓
Tasks 7-10 (Documentation) - All can run in parallel
```

## Execution Plan

### Phase 1: Critical Path (Tasks 1-6)
**Execute in sequence due to dependencies**

1. Agent 1: Task 1 (FileEarningsStore) → Validation Agent
2. Agent 2: Task 2 (FileDataStore) → Validation Agent
3. Agent 3: Task 3 (Integration manager) → Validation Agent
4. **Parallel execution:**
   - Agent 4: Task 4 (TFT builder) → Validation Agent
   - Agent 5: Task 5 (DataRegistry) → Validation Agent
5. Agent 6: Task 6 (Tests) → Validation Agent

### Phase 2: Documentation (Tasks 7-10)
**Execute in parallel (no dependencies)**

- Agent 7: Task 7 (Integration guide) → Validation Agent
- Agent 8: Task 8 (Deprecation warnings) → Validation Agent
- Agent 9: Task 9 (Migration guide) → Validation Agent
- Agent 10: Task 10 (Arch docs) → Validation Agent

## Success Criteria

✅ All tasks completed
✅ All validation agents approve
✅ Zero mypy errors
✅ Zero ruff violations
✅ All tests pass
✅ Performance benchmarks meet SLA
✅ Documentation complete
✅ Architecture compliance verified
