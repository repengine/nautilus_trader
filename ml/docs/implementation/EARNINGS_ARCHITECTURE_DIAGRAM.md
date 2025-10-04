# Earnings Data Flow - Architecture Diagram

## Current State (Isolated - WRONG ❌)

```
┌─────────────────────────────────────────────────────────────┐
│                     TFT Dataset Builder                     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                Direct EarningsStore Access            │  │
│  │                  (BYPASSES FACADE)                    │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       │                                      │
└───────────────────────┼──────────────────────────────────────┘
                        ▼
          ┌─────────────────────────┐
          │    EarningsStore        │  ← ISOLATED (not in 4-store pattern)
          │  (Standalone Service)   │
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │      PostgreSQL         │
          │  earnings_actuals       │
          │  earnings_estimates     │
          └─────────────────────────┘

Problems:
  ❌ No contract validation
  ❌ No lineage tracking
  ❌ No event emission
  ❌ Not accessible to actors
  ❌ No progressive fallback
  ❌ Separate caching layer
```

---

## Target State (Integrated - CORRECT ✓)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Universal ML Architecture                        │
│                        (4 Stores + 4 Registries)                         │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          Data Ingestion Layer                           │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │   EDGAR      │  │ Yahoo Finance│  │  Databento   │                  │
│  │  (Actuals)   │  │ (Estimates)  │  │   (Bars)     │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                  │                  │                          │
│         └──────────────────┼──────────────────┘                          │
│                            ▼                                             │
│                 ┌──────────────────────┐                                │
│                 │   Ingestion Adapters │                                │
│                 └──────────┬───────────┘                                │
└────────────────────────────┼────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Layer 1: Raw Data Storage                          │
│                    (DataStore + DataRegistry)                           │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                        DataStore Facade                        │    │
│  │                 (DataStoreFacadeProtocol)                      │    │
│  │                                                                 │    │
│  │  Methods:                                                       │    │
│  │  • write_earnings_actual(ticker, period_end, eps, ts_event)    │    │
│  │  • write_earnings_estimate(ticker, period_end, consensus, ...)  │    │
│  │  • get_earnings_actuals_at_or_before(ticker, ts_event, limit)  │    │
│  │  • get_earnings_estimate_at_or_before(ticker, period, ts)      │    │
│  │                                                                 │    │
│  │  Responsibilities:                                              │    │
│  │  ✓ Contract validation (via DataRegistry)                      │    │
│  │  ✓ Event emission (observability)                              │    │
│  │  ✓ Watermark tracking                                          │    │
│  │  ✓ Progressive fallback (PostgreSQL → File → Dummy)            │    │
│  └────────────────────────┬───────────────────────────────────────┘    │
│                           │                                             │
│         ┌─────────────────┼─────────────────┐                           │
│         ▼                 ▼                 ▼                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐                │
│  │  DataRegistry│  │ EarningsStore│  │  Event Emitter  │                │
│  │              │  │  (Internal)  │  │                 │                │
│  │ • Contracts  │  │              │  │ • Prometheus    │                │
│  │ • Manifests  │  │ PostgreSQL   │  │ • Message Bus   │                │
│  │ • Lineage    │  │   or         │  │ • Watermarks    │                │
│  │ • Versioning │  │ DummyStore   │  │                 │                │
│  └─────────────┘  └──────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  Layer 2: Feature Computation                           │
│                (FeatureStore + FeatureRegistry)                         │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                   Feature Computation Pipeline                  │    │
│  │                                                                 │    │
│  │  Raw Earnings → TransformSpec → Computed Features              │    │
│  │                                                                 │    │
│  │  TransformSpecs:                                                │    │
│  │  • EarningsSurpriseTransformSpec   (eps_surprise, %)           │    │
│  │  • EarningsGrowthTransformSpec     (YoY, QoQ growth)           │    │
│  │  • EarningsMomentumTransformSpec   (beat streak, volatility)   │    │
│  │  • EarningsCalendarTransformSpec   (days to earnings)          │    │
│  │                                                                 │    │
│  │  Computation Functions:                                         │    │
│  │  • compute_earnings_surprise_batch(actuals, estimates)         │    │
│  │  • compute_earnings_growth_batch(eps_series)                   │    │
│  │  • compute_earnings_momentum_batch(surprises, eps_series)      │    │
│  │  • compute_calendar_features_batch(dates, current_dates)       │    │
│  └────────────────────────┬───────────────────────────────────────┘    │
│                           │                                             │
│         ┌─────────────────┼─────────────────┐                           │
│         ▼                 ▼                 ▼                           │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐               │
│  │FeatureStore  │  │Feature      │  │  Hot/Cold Path   │               │
│  │              │  │Registry     │  │  Separation      │               │
│  │• Write       │  │             │  │                  │               │
│  │  features    │  │• Schemas    │  │• Hot: O(1)       │               │
│  │• Read        │  │• Validation │  │  incremental     │               │
│  │  features    │  │• Versioning │  │• Cold: Batch     │               │
│  │• Batch ops   │  │• Hash check │  │  vectorized      │               │
│  └──────────────┘  └─────────────┘  └──────────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Consumer Layer (Actors/TFT)                        │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                   BaseMLInferenceActor                          │    │
│  │                                                                 │    │
│  │  Properties (Auto-initialized):                                │    │
│  │  • self.data_store       → Access to earnings via facade       │    │
│  │  • self.feature_store    → Access to computed features         │    │
│  │  • self.model_store      → Access to predictions              │    │
│  │  • self.strategy_store   → Access to signals                  │    │
│  │                                                                 │    │
│  │  Example Usage:                                                │    │
│  │    actuals = self.data_store.get_earnings_actuals_at_or_before(│    │
│  │        ticker="AAPL", ts_event=bar.ts_event, limit=5           │    │
│  │    )                                                            │    │
│  │    surprise = compute_earnings_surprise_incremental(           │    │
│  │        actual=actuals[0]['eps_diluted'],                       │    │
│  │        estimate=actuals[0]['eps_consensus']                    │    │
│  │    )                                                            │    │
│  │    self.feature_store.write_features(...)                      │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    TFTDatasetBuilder                            │    │
│  │                                                                 │    │
│  │  Constructor:                                                   │    │
│  │    def __init__(                                                │    │
│  │        self, catalog, symbols,                                  │    │
│  │        data_store: DataStoreFacadeProtocol,  # ← Unified!      │    │
│  │        include_earnings=True                                    │    │
│  │    )                                                            │    │
│  │                                                                 │    │
│  │  Feature Fetching:                                              │    │
│  │    actuals = self.data_store.get_earnings_actuals_at_or_before(│    │
│  │        ticker=symbol, ts_event=max_ts, limit=5                  │    │
│  │    )                                                            │    │
│  │    features = compute_earnings_surprise_batch(actuals, ...)    │    │
│  │    quarterly_df = polars.DataFrame(features)                   │    │
│  │    result = bar_df.join_asof(quarterly_df, ...)  # ASOF join   │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘

```

---

## Progressive Fallback Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                   DataStore Initialization                          │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  PostgreSQL         │
                  │  Available?         │
                  └──────────┬──────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼ YES                         ▼ NO
    ┌─────────────────────┐       ┌─────────────────────┐
    │  EarningsStore      │       │  File Backend       │
    │  (PostgreSQL)       │       │  Available?         │
    │                     │       └──────────┬──────────┘
    │  Tables:            │                  │
    │  • earnings_actuals │       ┌──────────┴──────────┐
    │  • earnings_estimates│       ▼ YES                ▼ NO
    │                     │  ┌──────────────┐  ┌────────────────┐
    │  Features:          │  │FileEarnings  │  │DummyEarnings   │
    │  ✓ Persistence      │  │Store         │  │Store           │
    │  ✓ Transactions     │  │              │  │                │
    │  ✓ Indexing         │  │ Parquet files│  │ In-memory dict │
    │  ✓ Point-in-time    │  │ in ~/.nautilus│  │                │
    │  ✓ High performance │  │              │  │ ⚠️  No persist  │
    │                     │  │ ✓ Persistence│  │ ⚠️  Warnings    │
    └─────────────────────┘  │ ⚠️  Slower   │  │                │
                             └──────────────┘  └────────────────┘
                                    │                   │
                                    └─────────┬─────────┘
                                              ▼
                            ┌─────────────────────────────┐
                            │   DataStore Facade          │
                            │   (Unified Interface)       │
                            │                             │
                            │ All implementations         │
                            │ conform to protocol:        │
                            │ • get_earnings_actuals_*()  │
                            │ • get_earnings_estimate_*() │
                            │ • write_earnings_*()        │
                            └─────────────────────────────┘
```

---

## Data Flow: End-to-End Example

### Scenario: Process AAPL earnings and generate features

```
Step 1: Ingest Raw Earnings Data
─────────────────────────────────
┌──────────────┐
│ EDGAR API    │  ← Fetch AAPL 10-Q filing (Q1 2024)
└──────┬───────┘
       │ eps_diluted=2.52, filing_date=2024-04-01
       ▼
┌──────────────────────────────────────────────┐
│ DataStore.write_earnings_actual(            │
│   ticker="AAPL",                             │
│   period_end="2024-03-31",                   │
│   filing_date="2024-04-01",                  │
│   eps_diluted=2.52,                          │
│   ts_event=1711929600000000000,  # 2024-04-01│
│   ts_init=<now>                              │
│ )                                            │
└──────────────┬───────────────────────────────┘
               │
               ├─→ Contract validation (DataRegistry)
               ├─→ Write to EarningsStore (PostgreSQL)
               ├─→ Emit event (observability)
               └─→ Update watermark

Step 2: TFT Builder Fetches Earnings
─────────────────────────────────────
┌──────────────────────────────────────────────┐
│ TFTDatasetBuilder._fetch_earnings_features() │
└──────────────┬───────────────────────────────┘
               ▼
┌──────────────────────────────────────────────┐
│ actuals = data_store                         │
│   .get_earnings_actuals_at_or_before(        │
│     ticker="AAPL",                           │
│     ts_event=bar.ts_event,  # Point-in-time! │
│     limit=5                                  │
│   )                                          │
└──────────────┬───────────────────────────────┘
               │ Returns: [Q1'24, Q4'23, Q3'23, Q2'23, Q1'23]
               ▼
Step 3: Compute Features (Cold Path)
─────────────────────────────────────
┌──────────────────────────────────────────────┐
│ from ml.features.earnings import             │
│   compute_earnings_surprise_batch,           │
│   compute_earnings_growth_batch              │
│                                              │
│ eps_series = [2.52, 2.45, 2.38, 2.30, 2.24]  │
│ estimates  = [2.45, 2.40, 2.35, 2.28, 2.20]  │
│                                              │
│ surprise = compute_earnings_surprise_batch(  │
│   eps_series, estimates                      │
│ )  # {eps_surprise_q0: 0.07, ...}            │
│                                              │
│ growth = compute_earnings_growth_batch(      │
│   eps_series                                 │
│ )  # {eps_growth_yoy: 12.5, ...}             │
└──────────────┬───────────────────────────────┘
               ▼
Step 4: Expand to Bar-Level via ASOF Join
──────────────────────────────────────────
┌──────────────────────────────────────────────┐
│ quarterly_df = pl.DataFrame({                │
│   'filing_date': ['2024-04-01', '2024-01-01'],│
│   'eps_surprise_q0_AAPL': [0.07, 0.05],      │
│   'eps_growth_yoy_AAPL': [12.5, 10.2],       │
│ })                                           │
│                                              │
│ quarterly_df = quarterly_df.with_columns([   │
│   (filing_date + lag_days).alias('effective')│
│ ])                                           │
│                                              │
│ bar_df = pl.DataFrame({                      │
│   'timestamp': [minute timestamps...]        │
│ })                                           │
│                                              │
│ result = bar_df.join_asof(                   │
│   quarterly_df,                              │
│   left_on='timestamp',                       │
│   right_on='effective_date',                 │
│   strategy='backward'  # Use last earnings   │
│ )                                            │
└──────────────┬───────────────────────────────┘
               ▼
Step 5: Return Features to TFT Builder
───────────────────────────────────────
┌──────────────────────────────────────────────┐
│ Result DataFrame (1 row per bar):            │
│                                              │
│ timestamp            | eps_surprise_q0_AAPL  │
│ 2024-04-01 09:30:00 | 0.07                  │
│ 2024-04-01 09:31:00 | 0.07                  │
│ 2024-04-01 09:32:00 | 0.07                  │
│ ...                                          │
│ 2024-04-01 16:00:00 | 0.07                  │
└──────────────────────────────────────────────┘

Step 6: Store Features in FeatureStore
───────────────────────────────────────
┌──────────────────────────────────────────────┐
│ feature_store.write_features(                │
│   feature_set_id="earnings_features_v1",     │
│   instrument_id="AAPL.NASDAQ",               │
│   features={                                 │
│     'eps_surprise_q0_AAPL': 0.07,            │
│     'eps_growth_yoy_AAPL': 12.5,             │
│   },                                         │
│   ts_event=bar.ts_event,                     │
│   ts_init=bar.ts_init                        │
│ )                                            │
└──────────────┬───────────────────────────────┘
               │
               ├─→ Schema validation (FeatureRegistry)
               ├─→ Write to PostgreSQL (ml_feature_values)
               ├─→ Emit event (observability)
               └─→ Training/inference parity guaranteed

Step 7: Actor Uses Features in Real-Time
─────────────────────────────────────────
┌──────────────────────────────────────────────┐
│ class MyActor(BaseMLInferenceActor):         │
│     def on_bar(self, bar):                   │
│         # Get latest earnings                │
│         actuals = self.data_store            │
│           .get_earnings_actuals_at_or_before(│
│             ticker="AAPL",                   │
│             ts_event=bar.ts_event            │
│           )                                  │
│                                              │
│         # Hot path incremental update        │
│         surprise = compute_earnings_surprise_│
│           incremental(                       │
│             actual=actuals[0]['eps_diluted'],│
│             estimate=actuals[0]['consensus'] │
│           )                                  │
│                                              │
│         # Use in signal generation           │
│         if surprise['eps_surprise_q0'] > 0.1:│
│             self.publish_signal(...)         │
└──────────────────────────────────────────────┘
```

---

## Key Architectural Decisions

### ✓ Decision 1: Raw Data → DataStore (Not FeatureStore)

**Rationale:**
- Earnings actuals/estimates are **raw data inputs**, not computed features
- DataStore handles ingestion, validation, lineage tracking
- FeatureStore handles computed features derived from raw data

```
✓ CORRECT:
  EDGAR → DataStore (earnings_actuals) → Feature Computation → FeatureStore

✗ WRONG:
  EDGAR → FeatureStore (would mix raw data with computed features)
```

### ✓ Decision 2: Computed Features → FeatureStore (Not DataStore)

**Rationale:**
- Earnings surprise, growth, momentum are **computed features**
- FeatureStore ensures training/inference parity
- FeatureRegistry validates feature schemas

```
✓ CORRECT:
  Raw Earnings + TransformSpec → FeatureStore (earnings_features)

✗ WRONG:
  Raw Earnings → DataStore (earnings_surprise)  ← Should be in FeatureStore
```

### ✓ Decision 3: Internal EarningsStore (Not Public API)

**Rationale:**
- EarningsStore becomes **internal implementation detail** of DataStore
- Public API is `DataStoreFacadeProtocol.get_earnings_*()`
- Progressive fallback hidden from consumers

```
✓ CORRECT:
  Actor → DataStore.get_earnings_*() → (internal) EarningsStore → PostgreSQL

✗ WRONG:
  Actor → EarningsStore (direct) → PostgreSQL  ← Bypasses facade
```

### ✓ Decision 4: Point-in-Time Correctness in DataStore

**Rationale:**
- `as_of_ts` filtering must happen at **data retrieval layer**
- Ensures no look-ahead bias in backtesting
- Feature computation uses already-filtered data

```
✓ CORRECT:
  DataStore.get_earnings_actuals_at_or_before(ts_event=backtest_time)
  → Only returns filings with ts_event < backtest_time

✗ WRONG:
  DataStore.get_all_earnings() → Feature computation filters
  ← Too late, risk of look-ahead bias
```

---

## Summary

This architecture ensures:

1. ✅ **Raw earnings data** flows through **DataStore + DataRegistry**
2. ✅ **Computed features** flow through **FeatureStore + FeatureRegistry**
3. ✅ **Actors and TFT builder** use unified `DataStoreFacadeProtocol`
4. ✅ **Progressive fallback** (PostgreSQL → File → Dummy)
5. ✅ **Point-in-time correctness** at data retrieval layer
6. ✅ **Event-driven observability** with metrics, events, watermarks
7. ✅ **No architectural violations** - follows Universal ML Architecture Pattern #1

The result is a **clean separation of concerns** where:
- **DataStore** = Raw data ingestion, validation, lineage
- **FeatureStore** = Computed features, training/inference parity
- **EarningsStore** = Internal PostgreSQL implementation (hidden from consumers)
