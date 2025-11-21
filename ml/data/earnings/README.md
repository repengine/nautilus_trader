# Earnings Data Sources - EDGAR & Yahoo Finance

## Overview

This module provides data fetching and storage for corporate earnings, combining:

1. **SEC EDGAR** (Primary): Actual earnings from 10-Q/10-K filings - 100% US coverage, free
2. **Yahoo Finance** (Secondary): Consensus estimates from analysts - ~70% coverage, free

All data is stored in PostgreSQL with point-in-time correctness for backtesting.

---

## Data Sources

### SEC EDGAR (Actuals)

**Purpose**: Fetch actual reported earnings from SEC filings

**Library**: [`edgartools`](https://github.com/dgunning/edgartools) (v2.0+)

**Coverage**:
- 100% of US public companies
- Historical data back to 2000s
- Update frequency: T+1 day after filing

**Data Extracted**:
- EPS (basic and diluted)
- Revenue
- Net income
- Operating income
- Shares outstanding
- Filing metadata (10-Q, 10-K, 8-K)

**Installation**:
```bash
pip install edgartools
```

**API Rate Limits**:
- No official limit, but be respectful (~1 request/second)
- Use exponential backoff if rate limited

### Yahoo Finance (Estimates)

**Purpose**: Fetch analyst consensus estimates

**Library**: [`yfinance`](https://github.com/ranaroussi/yfinance) (v0.2.40+)

**Coverage**:
- ~70% of US equities (major stocks only)
- Limited historical consensus (last 2-4 quarters)
- Update frequency: Daily

**Data Extracted**:
- Consensus EPS estimate
- Consensus revenue estimate
- Number of analysts
- Earnings calendar (next announcement date)

**Installation**:
```bash
pip install yfinance
```

**API Rate Limits**:
- ~100 requests/hour per IP (free tier)
- Use `time.sleep(0.5)` between requests to avoid throttling

---

## Fetcher Usage

### EdgarFetcher

Fetches actual earnings from SEC EDGAR filings.

```python
from ml.data.earnings import EdgarFetcher

fetcher = EdgarFetcher()

# Fetch last 8 quarters of 10-Q filings for AAPL
actuals = fetcher.fetch_earnings(
    ticker="AAPL",
    quarters=8,
    form="10-Q",  # Can also use "10-K" for annual
)

# Result: List[EarningsActual]
for actual in actuals:
    print(f"{actual.period_end}: EPS=${actual.eps_diluted:.2f}, Revenue=${actual.revenue/1e9:.1f}B")

# Output:
# 2024-09-30: EPS=$1.64, Revenue=$94.9B
# 2024-06-30: EPS=$1.40, Revenue=$85.8B
# ...
```

**EarningsActual Dataclass**:
```python
@dataclass
class EarningsActual:
    ticker: str
    period_end: date
    filing_date: date
    eps_basic: float | None
    eps_diluted: float
    revenue: float
    net_income: float | None
    operating_income: float | None
    shares_outstanding: int | None
    filing_type: str  # "10-Q" or "10-K"
    fiscal_year: int
    fiscal_quarter: int  # 1, 2, 3, 4
    ts_event: int  # Filing date in nanoseconds
    ts_init: int  # Record creation time in nanoseconds
```

**Error Handling**:
```python
# Graceful handling of missing data
try:
    actuals = fetcher.fetch_earnings("INVALID_TICKER", quarters=1)
except Exception as e:
    print(f"Error: {e}")
    actuals = []  # Empty list on failure

# Or use the built-in graceful handling
actuals = fetcher.fetch_earnings("INVALID_TICKER", quarters=1)
# Returns: [] (empty list, no exception)
```

### YahooFetcher

Fetches consensus estimates and earnings calendar.

```python
from ml.data.earnings import YahooFetcher

fetcher = YahooFetcher()

# Fetch consensus estimate for AAPL
consensus = fetcher.fetch_consensus(ticker="AAPL")

# Result: EarningsConsensus
print(f"Next earnings: {consensus.next_earnings_date}")
print(f"Consensus EPS: ${consensus.eps_consensus:.2f}")
print(f"Number of analysts: {consensus.num_analysts}")

# Output:
# Next earnings: 2025-01-30
# Consensus EPS: $2.10
# Number of analysts: 35
```

**EarningsConsensus Dataclass**:
```python
@dataclass
class EarningsConsensus:
    ticker: str
    estimate_date: date
    period_end: date
    eps_consensus: float
    revenue_consensus: float | None
    num_analysts: int | None
    next_earnings_date: date | None
    ts_event: int  # Estimate date in nanoseconds
    ts_init: int  # Record creation time in nanoseconds
```

**Rate Limiting**:
```python
# Add delay between requests to avoid throttling
fetcher = YahooFetcher(rate_limit_delay=0.5)  # 500ms between requests

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
for ticker in tickers:
    consensus = fetcher.fetch_consensus(ticker)
    # Automatic 500ms delay
```

---

## Storage (DataStore Facade)

All earnings data is stored in PostgreSQL with point-in-time correctness.

### Schema

**3 Tables**:
1. `ml.earnings_actuals` - Actual reported earnings from EDGAR
2. `ml.earnings_estimates` - Consensus estimates from Yahoo
3. `ml.earnings_calendar` - Upcoming earnings announcements

**View**:
- `ml.earnings_combined` - Joins actuals with estimates for surprise calculation

See `/home/nate/projects/nautilus_trader/ml/schema/earnings.sql` for full schema definition.

### DataStore Usage

```python
import time

from ml.stores.data_store import DataStore

# Initialize facade (falls back to DummyEarningsStore if PostgreSQL unavailable)
store = DataStore(
    connection_string="postgresql://user:pass@localhost/nautilus_trader",
    registry=my_registry,
)

# Write actuals
store.write_earnings_actual(
    ticker="AAPL",
    period_end="2024-09-30",
    filing_date="2024-10-31",
    eps_diluted=1.64,
    revenue=94.9e9,
    ts_event=1730332800000000000,  # Filing date in nanoseconds
    ts_init=int(time.time_ns()),
    filing_type="10-Q",
    fiscal_year=2024,
    fiscal_quarter=3,
)

# Write estimates
store.write_earnings_estimate(
    ticker="AAPL",
    estimate_date="2024-09-20",
    period_end="2024-09-30",
    eps_consensus=1.60,
    revenue_consensus=92.0e9,
    ts_event=1726790400000000000,  # Estimate date in nanoseconds
    ts_init=int(time.time_ns()),
    num_analysts=35,
)

# Read actuals (point-in-time)
actuals = store.get_earnings_actuals_at_or_before(
    ticker="AAPL",
    ts_event=1730332800000000000,  # Only data filed before this timestamp
    limit=8,
)

# Read estimates
estimate = store.get_earnings_estimate_at_or_before(
    ticker="AAPL",
    period_end="2024-09-30",
    ts_event=1730332800000000000,  # Most recent estimate before this timestamp
)
```

### DummyEarningsStore (Fallback)

`DataStore` automatically falls back to the in-memory `DummyEarningsStore` when PostgreSQL is
unavailable. This keeps reads/writes in memory while preserving the same facade.

```python
from ml.stores.data_store import DataStore

store = DataStore(connection_string="postgresql://localhost:15432/unavailable")
# If the connection fails, the facade logs the fallback and continues with DummyEarningsStore.
store.write_earnings_actual(...)
actuals = store.get_earnings_actuals_at_or_before(...)
```

---

## Caching (EarningsCache)

Point-in-time cache for fast backtesting with temporal correctness.

```python
from ml.data.earnings import EarningsCache
from ml.stores.adapters import DataStoreEarningsAdapter
from ml.stores.data_store import DataStore

# Initialize cache with DataStore facade
store = DataStore(connection_string="postgresql://...")
cache = EarningsCache(DataStoreEarningsAdapter(store), maxsize=1024)
cache = EarningsCache(store=store)

# Warm cache for AAPL (loads all historical data)
cache.warm_cache(ticker="AAPL")

# Fast point-in-time lookups (<1ms)
as_of_date = "2024-01-01"
actuals = cache.get_actuals_cached(ticker="AAPL", as_of_date=as_of_date)
estimate = cache.get_estimate_cached(ticker="AAPL", period_end="2023-12-31", as_of_date=as_of_date)

# Cache hit rate
print(f"Cache hit rate: {cache.get_hit_rate():.2%}")
# Output: Cache hit rate: 95.3%
```

**Cache Invalidation**:
```python
# Clear cache for specific ticker
cache.clear_ticker("AAPL")

# Clear entire cache
cache.clear_all()

# Automatic invalidation after 1 hour (configurable)
cache = EarningsCache(store=store, ttl_seconds=3600)
```

---

## Pipeline Integration

### End-to-End Example

```python
from ml.data.earnings import EdgarFetcher, YahooFetcher
from ml.stores.data_store import DataStore
from ml.features.earnings import compute_earnings_surprise_batch
import time

# Step 1: Fetch actuals from EDGAR
edgar_fetcher = EdgarFetcher()
actuals = edgar_fetcher.fetch_earnings(ticker="AAPL", quarters=8)

# Step 2: Fetch estimates from Yahoo
yahoo_fetcher = YahooFetcher()
consensus = yahoo_fetcher.fetch_consensus(ticker="AAPL")

# Step 3: Store via DataStore facade (handles fallback + validation)
store = DataStore(connection_string="postgresql://...")

for actual in actuals:
    store.write_earnings_actual(
        ticker=actual.ticker,
        period_end=str(actual.period_end),
        filing_date=str(actual.filing_date),
        eps_diluted=actual.eps_diluted,
        revenue=actual.revenue,
        ts_event=actual.ts_event,
        ts_init=actual.ts_init,
        filing_type=actual.filing_type,
        fiscal_year=actual.fiscal_year,
        fiscal_quarter=actual.fiscal_quarter,
    )

store.write_earnings_estimate(
    ticker=consensus.ticker,
    estimate_date=str(consensus.estimate_date),
    period_end=str(consensus.period_end),
    eps_consensus=consensus.eps_consensus,
    revenue_consensus=consensus.revenue_consensus,
    ts_event=consensus.ts_event,
    ts_init=consensus.ts_init,
    num_analysts=consensus.num_analysts,
)

# Step 4: Compute earnings surprise
actual_eps = actuals[0].eps_diluted
consensus_eps = consensus.eps_consensus

surprise = compute_earnings_surprise_batch([actual_eps], [consensus_eps])
print(f"EPS Surprise: ${surprise['eps_surprise_q0']:.2f} ({surprise['eps_surprise_pct_q0']:.2f}%)")
```

---

## XBRL Parsing

### XBRLParser Utilities

The `XBRLParser` provides utilities for extracting structured financial data from SEC filings.

```python
from ml.data.earnings import XBRLParser

parser = XBRLParser()

# Common XBRL tags for earnings data
tags = {
    "eps_diluted": "us-gaap:EarningsPerShareDiluted",
    "revenue": "us-gaap:Revenues",
    "net_income": "us-gaap:NetIncomeLoss",
    "operating_income": "us-gaap:OperatingIncomeLoss",
}

# Parse XBRL document
filing_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240930.htm"
data = parser.parse(filing_url, tags=tags)

print(f"EPS: ${data['eps_diluted']:.2f}")
print(f"Revenue: ${data['revenue']/1e9:.1f}B")
```

**Handling Non-Standard Tags**:
```python
# Some companies use custom tags
custom_tags = {
    "eps_diluted": [
        "us-gaap:EarningsPerShareDiluted",
        "aapl:CustomEPSDiluted",  # Company-specific tag
    ],
}

# Parser tries each tag in order
data = parser.parse(filing_url, tags=custom_tags)
```

---

## Troubleshooting

### Issue 1: EdgarFetcher Returns Empty List

**Symptoms**: `fetch_earnings()` returns `[]` for valid ticker.

**Causes**:
1. Ticker doesn't exist or hasn't filed yet
2. EDGAR API is down or rate-limited
3. Non-standard XBRL tags (company uses custom tags)

**Solutions**:
```python
# Check ticker exists
actuals = fetcher.fetch_earnings("AAPL", quarters=1)
if not actuals:
    print("No filings found - check ticker spelling")

# Try alternative forms
actuals_10k = fetcher.fetch_earnings("AAPL", quarters=1, form="10-K")  # Annual instead of quarterly

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
actuals = fetcher.fetch_earnings("AAPL", quarters=1)
```

### Issue 2: YahooFetcher Rate Limited

**Symptoms**: `HTTPError: 429 Too Many Requests`

**Cause**: Exceeded Yahoo Finance rate limit (~100 requests/hour).

**Solutions**:
```python
# Add rate limiting delay
fetcher = YahooFetcher(rate_limit_delay=1.0)  # 1 second between requests

# Use exponential backoff
import time

for attempt in range(3):
    try:
        consensus = fetcher.fetch_consensus("AAPL")
        break
    except Exception as e:
        if "429" in str(e):
            delay = 2 ** attempt  # 1s, 2s, 4s
            print(f"Rate limited, retrying in {delay}s...")
            time.sleep(delay)
        else:
            raise
```

### Issue 3: PostgreSQL Connection Failed

**Symptoms**: `OperationalError: could not connect to server`

**Cause**: PostgreSQL not running or wrong connection string.

**Solutions**:
```python
# Check PostgreSQL is running
import subprocess
result = subprocess.run(["pg_isready"], capture_output=True)
if result.returncode != 0:
    print("PostgreSQL is not running")

# Fallback handled by DataStore facade
store = DataStore(connection_string="postgresql://...")
# If connection fails, DataStore logs a warning and switches to DummyEarningsStore automatically.
```

### Issue 4: Point-in-Time Query Returns No Data

**Symptoms**: `get_actuals(as_of_ts=...)` returns `[]`.

**Cause**: `as_of_ts` is before any filings.

**Solutions**:
```python
# Check earliest filing
current_ts = int(datetime.utcnow().timestamp() * 1_000_000_000)
all_actuals = store.get_earnings_actuals_at_or_before("AAPL", ts_event=current_ts)
if all_actuals:
    earliest_ts = min(a["ts_event"] for a in all_actuals)
    earliest_date = datetime.fromtimestamp(earliest_ts / 1e9)
    print(f"Earliest filing: {earliest_date}")

# Query after earliest filing
as_of_ts = earliest_ts + 1
actuals = store.get_earnings_actuals_at_or_before("AAPL", ts_event=as_of_ts)
```

---

## Performance Benchmarks

### Fetcher Performance

| Operation | Target | Measured |
|-----------|--------|----------|
| EDGAR fetch (single 10-Q) | <2s | ~1.2s |
| Yahoo fetch (consensus) | <500ms | ~300ms |
| XBRL parsing | <100ms | ~80ms |

### Store Performance

| Operation | Target | Measured |
|-----------|--------|----------|
| Write actuals (100 records) | <100ms | ~65ms |
| Read actuals (point-in-time) | <1ms | ~0.4ms |
| Cache lookup | <1ms P99 | ~0.3ms |

### Validation

Run performance tests:
```bash
poetry run pytest ml/tests/performance/test_earnings_performance.py -v
```

---

## Data Quality

### Validation Rules

The pipeline enforces strict data quality:

1. **Required Fields**: ticker, period_end, filing_date, eps_diluted, ts_event, ts_init
2. **Value Ranges**:
   - EPS: -1000 < eps < 1000
   - Revenue: > 0
   - Shares outstanding: > 0
3. **Temporal Consistency**:
   - filing_date > period_end
   - ts_event matches filing_date
4. **Cross-Validation**:
   - EPS ≈ net_income / shares_outstanding (within tolerance)

### Outlier Detection

Statistical outliers are flagged (but not rejected):

```python
# Z-score based outlier detection
import numpy as np

eps_history = [2.0, 2.1, 2.05, 2.15, 10.0]  # 10.0 is outlier
mean = np.mean(eps_history[:-1])  # Exclude outlier
std = np.std(eps_history[:-1])

for eps in eps_history:
    z_score = abs((eps - mean) / std)
    if z_score > 3:
        print(f"Outlier detected: EPS=${eps:.2f}, Z-score={z_score:.2f}")

# Output: Outlier detected: EPS=$10.00, Z-score=38.74
```

---

## API Reference

### EdgarFetcher

```python
class EdgarFetcher:
    def __init__(self, user_agent: str = "NautilusTrader/1.0"):
        """Initialize EDGAR fetcher with user agent."""

    def fetch_earnings(
        self,
        ticker: str,
        quarters: int = 4,
        form: str = "10-Q",
    ) -> list[EarningsActual]:
        """
        Fetch actual earnings from SEC EDGAR.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g., "AAPL")
        quarters : int
            Number of quarters to fetch (default: 4)
        form : str
            Filing form type: "10-Q" (quarterly) or "10-K" (annual)

        Returns
        -------
        list[EarningsActual]
            List of earnings actuals, sorted by filing_date descending
        """
```

### YahooFetcher

```python
class YahooFetcher:
    def __init__(self, rate_limit_delay: float = 0.0):
        """
        Initialize Yahoo Finance fetcher.

        Parameters
        ----------
        rate_limit_delay : float
            Delay in seconds between requests (default: 0.0)
        """

    def fetch_consensus(self, ticker: str) -> EarningsConsensus:
        """
        Fetch consensus estimate for ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol

        Returns
        -------
        EarningsConsensus
            Consensus estimate with next earnings date
        """
```

### EarningsStore

```python
class EarningsStore:
    def __init__(
        self,
        connection_string: str,
        schema: str = "ml",
    ):
        """Initialize PostgreSQL earnings store."""

    def write_actuals(
        self,
        ticker: str,
        period_end: str,
        filing_date: str,
        eps_diluted: float | None,
        revenue: float | None,
        ts_event: int,
        ts_init: int,
        **kwargs,
    ) -> None:
        """Write actual earnings (upsert on conflict)."""

    def write_estimates(
        self,
        ticker: str,
        estimate_date: str,
        period_end: str,
        eps_consensus: float | None,
        ts_event: int,
        ts_init: int,
        **kwargs,
    ) -> None:
        """Write consensus estimate (upsert on conflict)."""

    def get_actuals(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        as_of_ts: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get actuals with optional point-in-time filtering."""

    def get_estimates(
        self,
        ticker: str,
        period_end: str,
        as_of_ts: int | None = None,
    ) -> dict[str, Any] | None:
        """Get most recent estimate for period."""
```

### EarningsCache

```python
class EarningsCache:
    def __init__(
        self,
        store: EarningsStore | DummyEarningsStore,
        ttl_seconds: int = 3600,
    ):
        """
        Initialize cache with store and TTL.

        Parameters
        ----------
        store : EarningsStore | DummyEarningsStore
            Underlying store
        ttl_seconds : int
            Cache time-to-live in seconds (default: 3600 = 1 hour)
        """

    def warm_cache(self, ticker: str) -> None:
        """Pre-load all data for ticker into cache."""

    def get_actuals_cached(
        self,
        ticker: str,
        as_of_date: str,
    ) -> list[dict[str, Any]]:
        """Get actuals from cache (point-in-time)."""

    def get_estimate_cached(
        self,
        ticker: str,
        period_end: str,
        as_of_date: str,
    ) -> dict[str, Any] | None:
        """Get estimate from cache (point-in-time)."""

    def get_hit_rate(self) -> float:
        """Return cache hit rate (0.0 to 1.0)."""

    def clear_ticker(self, ticker: str) -> None:
        """Clear cache for specific ticker."""

    def clear_all(self) -> None:
        """Clear entire cache."""
```

---

## Migration Guide

### Automated Ingestion

Earnings ingestion can be re-run via CLI/Makefile:

```bash
# Mirror to Postgres + Parquet (ml_out/earnings_raw)
poetry run python ml/cli/ingest_earnings.py --dsn "$NAUTILUS_DB"

# Or via Make target (honors DSN/PARQUET_ROOT environment variables)
make earnings-ingest DSN="$NAUTILUS_DB"
```

Flags of interest:

- `--symbol TICKER` to constrain ingestion (repeatable)
- `--skip-actuals ETF1 --skip-actuals ETF2` to extend the ETF skip list
- `--no-yahoo` to disable consensus ingestion
- `--sec-identity "Your Org <ops@example.com>"` to satisfy SEC API requirements

### From No Earnings Data → Earnings Integration

**Step 1**: Install dependencies
```bash
pip install edgartools yfinance
```

**Step 2**: Create PostgreSQL schema
```bash
psql -h localhost -U postgres -d nautilus_trader -f ml/schema/earnings.sql
```

**Step 3**: Backfill historical data
```python
from ml.data.earnings import EdgarFetcher, YahooFetcher
from ml.stores.data_store import DataStore

# Initialize facade-backed integration (PostgreSQL → file → dummy)
edgar = EdgarFetcher()
yahoo = YahooFetcher(rate_limit_delay=0.5)
data_store = DataStore(connection_string="postgresql://...")

# Backfill for list of tickers
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

for ticker in tickers:
    print(f"Backfilling {ticker}...")

    # Fetch actuals (last 8 quarters)
    actuals = edgar.fetch_earnings(ticker, quarters=8)
    for actual in actuals:
        data_store.write_earnings_actual(
            ticker=actual.ticker,
            period_end=str(actual.period_end),
            filing_date=str(actual.filing_date),
            eps_diluted=actual.eps_diluted,
            revenue=actual.revenue,
            ts_event=actual.ts_event,
            ts_init=actual.ts_init,
            filing_type=actual.filing_type,
            fiscal_year=actual.fiscal_year,
            fiscal_quarter=actual.fiscal_quarter,
        )

    # Fetch consensus
    consensus = yahoo.fetch_consensus(ticker)
    if consensus:
        data_store.write_earnings_estimate(
            ticker=consensus.ticker,
            estimate_date=str(consensus.estimate_date),
            period_end=str(consensus.period_end),
            eps_consensus=consensus.eps_consensus,
            revenue_consensus=consensus.revenue_consensus,
            ts_event=consensus.ts_event,
            ts_init=consensus.ts_init,
            num_analysts=consensus.num_analysts,
        )

    print(f"  Stored {len(actuals)} actuals + 1 estimate")
```

The facade transparently records fallback activations via the
`ml_fallback_activations_total` metric. Set `ML_FILE_STORE_PATH` to point at a
shared directory when you expect the PostgreSQL primary to be unavailable so
the `FileEarningsStore` stage can hydrate on-disk parquet snapshots before the
`DummyEarningsStore` safety net is considered.

**Step 4**: Add earnings features to pipeline
```python
from ml.features.earnings import (
    EarningsSurpriseTransformSpec,
    EarningsGrowthTransformSpec,
)

# Add to existing pipeline
pipeline.transforms.extend([
    EarningsSurpriseTransformSpec(ticker="AAPL"),
    EarningsGrowthTransformSpec(ticker="AAPL"),
])
```

**Step 5**: Seed registry contracts + validate instrumentation
```bash
# Seed dataset manifests/contracts (JSON example)
uv run --active --no-sync python -m ml.registry.bootstrap_datasets --backend json --registry-path ml/tests/fixtures/registry

# Or PostgreSQL-backed registry
NAUTILUS_REGISTRY_DB_URL=postgresql://... \
  uv run --active --no-sync python -m ml.registry.bootstrap_datasets --backend postgres

# Validate facade-usage patterns and canonical events
make validate-metrics
make validate-events
```

---

## Testing

```bash
# Unit tests for data fetchers
poetry run pytest ml/tests/unit/data/earnings -v

# Integration tests (full pipeline)
poetry run pytest ml/tests/integration/earnings -v -m integration

# Data quality validation
poetry run pytest ml/tests/integration/earnings/test_data_quality.py -v
```

---

## Support

For questions or issues:

- Features: `ml/features/README_EARNINGS.md`
- Architecture: `ml/docs/architecture/universal_patterns_guide.md`
- Schema: `ml/schema/earnings.sql`
