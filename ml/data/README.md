# ML Data Module

This module provides data collection and management utilities for machine learning workflows in Nautilus Trader.

## Overview

The ML data module uses Nautilus Trader's native `ParquetDataCatalog` directly, eliminating unnecessary abstraction layers for improved performance and maintainability.

## Architecture

```
Databento API
    ↓
DataCollector (fetch only)
    ↓
ParquetDataCatalog (Nautilus native)
    ↓
catalog_utils (helper functions)
    ↓
FeatureStore / ML Models
```

## Components

### catalog_utils.py
Helper functions for working directly with ParquetDataCatalog:

- `bars_to_dataframe()` - Convert bars to Polars DataFrame
- `quotes_to_dataframe()` - Convert quotes to Polars DataFrame
- `trades_to_dataframe()` - Convert trades to Polars DataFrame

### collector.py
`DataCollector` class for fetching data from Databento API:

- Collects L2 depth, L1 trades, TBBO quotes, and minute bars
- Outputs native Nautilus types (Bar, QuoteTick, TradeTick)
- No caching or DataFrame conversion (separation of concerns)

### scheduler.py
`DataScheduler` class for automated daily updates:

- Daily collection from Databento
- Feature computation triggers
- Configurable retention policies
- Error handling and logging

### tft_dataset_builder.py
`TFTDatasetBuilder` for creating TFT-compatible datasets:

- Uses ParquetDataCatalog directly
- Builds training datasets with configurable horizons
- Supports multiple symbols

### providers/
Data providers for static and known covariates:

- Calendar data (trading days, holidays)
- Economic events (FOMC, earnings)
- Symbol metadata

## Usage

### Loading Data

```python
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from ml.data.catalog_utils import bars_to_dataframe

# Initialize catalog
catalog = ParquetDataCatalog("./data")

# Load bars as DataFrame
bars_df = bars_to_dataframe(
    catalog,
    ["EURUSD.SIM"],
    start="2023-01-01",
    end="2023-12-31"
)
```

### Public API (quick imports)

For common workflows, import directly from `ml.data`:

```python
from ml.data import (
    DataCollector,
    DataScheduler,
    TFTDatasetBuilder,
    InstrumentMetadataProvider,
    MarketCalendarProvider,
    EventScheduleProvider,
    MockCalendarSource,
    SimpleCalendarSource,
    PandasCalendarSource,
    DatabentoMetadataSource,
    NautilusMetadataSource,
    L2MinuteCache,
    MicroMinuteCache,
    bars_to_dataframe,
    quotes_to_dataframe,
    trades_to_dataframe,
)
```

### Collecting New Data

```python
from ml.data.collector import DataCollector

# Initialize collector
collector = DataCollector()

# Fetch data (returns native Nautilus types)
# In production, this calls Databento API
```

### Scheduling Updates

```python
from ml.data.scheduler import DataScheduler

# Create scheduler
scheduler = DataScheduler(
    catalog=catalog,
    retention_days=90
)

# Run daily update
scheduler.run_daily_update()

# Schedule automated updates
scheduler.schedule_updates("0 4 * * *")  # 4 AM UTC daily
```

## Code Quality Standards

This module adheres to strict quality standards:

- ✅ **Type Safety**: Full mypy strict compliance (0 errors)
- ✅ **Linting**: Ruff compliant (0 violations)
- ✅ **Testing**: Property-based tests with Hypothesis
- ✅ **DRY**: No duplicate code
- ✅ **SOLID**: Single responsibility principle

## Testing

Run tests:

```bash
# Unit tests
pytest ml/tests/unit/data/ -xvs

# With coverage
pytest ml/tests/unit/data/ --cov=ml/data --cov-report=term-missing
```

Quality checks:

```bash
# Type checking
mypy --strict ml/data/

# Linting
ruff check ml/data/
```

## Migration from MLDataLoader

If upgrading from the old MLDataLoader:

**Before:**

```python
from ml.data.loader import MLDataLoader
loader = MLDataLoader(catalog)
bars_df = loader.load_bars("EURUSD.SIM")
```

**After:**

```python
from ml.data.catalog_utils import bars_to_dataframe
bars_df = bars_to_dataframe(catalog, ["EURUSD.SIM"])
```

## Dependencies

- `nautilus_trader`: Core trading framework
- `polars`: DataFrame operations
- `databento`: Market data API (for collection)
- `hypothesis`: Property-based testing

## Recent Changes

- **2025-08-19**: Major refactoring to use Nautilus components directly
  - Removed MLDataLoader, DatabentoSource, ProductionDataCollector
  - Created catalog_utils for direct ParquetDataCatalog usage
  - Added DataScheduler for automated updates
  - Full compliance with strict type checking and linting
