# TFT Dataset Structure After Feature Expansion

## Overview

After implementing Macro Factor Depth, Instrument-Factor Mapping, and Cross-Asset Relationships, the TFT training dataset now has **4 distinct feature groups** with rich temporal and static covariates.

---

## Dataset Schema

### Core Dimensions
- **Time dimension**: `ts_event` (nanoseconds since epoch)
- **Entity dimension**: `instrument_id` (95 instruments)
- **Lookback window**: 60-252 bars (configurable)
- **Forecast horizon**: 1-20 bars ahead (configurable)

### Total Feature Count
- **Time-varying known reals**: ~50-100 features
- **Time-varying unknown reals**: ~150-200 features
- **Static categoricals**: 5 features
- **Static reals**: 0-10 features (optional)

**Total**: ~200-300+ features per instrument-timestep

---

## 1. Market Data (Base Layer)

### L1 OHLCV Features (5 base)
```
instrument_id  | ts_event           | open    | high    | low     | close   | volume
ES            | 1704067200000000000| 4750.25 | 4752.50 | 4748.00 | 4751.75 | 125000
NQ            | 1704067200000000000| 16500.0 | 16520.0 | 16490.0 | 16515.0 | 85000
```

### Technical Indicators (from FeatureEngineer - ~40 features)
```
instrument_id | ts_event           | return_1 | return_5 | volatility_20 | rsi_14 | macd | bb_upper | atr_14
ES            | 1704067200000000000| 0.0002   | 0.0015   | 0.0125        | 55.2   | 0.3  | 4760.0   | 12.5
```

**Feature categories**:
- Returns: 1, 5, 10, 20 bar periods
- Volatility: 20, 60 bar windows
- Momentum: RSI, MACD, Stochastic
- Trend: Bollinger Bands, EMA, SMA
- Volume: Volume ratios, VWAP deviation

---

## 2. Macro Features (New - 36 composites)

### Credit/Risk Dimension (9 features)
```
instrument_id | ts_event           | credit_spread_ig | credit_spread_hy | credit_spread_hy_ig | credit_distress_index | ...
ES            | 1704067200000000000| 1.25             | 3.45             | 2.20                | 0.35                  | ...
```

**All credit features**:
- `credit_spread_ig`: Investment grade corporate spread
- `credit_spread_hy`: High yield corporate spread
- `credit_spread_hy_ig`: HY-IG quality premium
- `credit_spread_bbb_a`: BBB-A quality spread
- `credit_spread_ig_momentum`: 3-month IG spread change
- `credit_spread_hy_momentum`: 3-month HY spread change
- `credit_distress_index`: Multi-indicator distress measure
- `ted_spread`: TED spread (bank funding stress)
- `credit_risk_index`: Composite risk index

### Duration/Term Dimension (8 features)
```
instrument_id | ts_event           | term_spread | term_spread_5s30s | curve_curvature | real_term_premium | ...
ES            | 1704067200000000000| 0.45        | 1.25              | -0.05           | 0.35              | ...
```

**All duration features**:
- `term_spread`: 10y-2y Treasury spread
- `term_spread_5s30s`: 30y-5y spread (long-end slope)
- `term_spread_2s30s`: 30y-2y spread (full curve)
- `curve_curvature`: Butterfly spread (2s-5s-10s)
- `real_yield_10y`: 10y TIPS real yield
- `real_term_premium`: Nominal - real curve differential
- `yield_curve_slope`: (10y-2y)/2y normalized
- `fed_policy_stance`: Fed funds - 10y spread

### Liquidity/Funding Dimension (6 features)
```
instrument_id | ts_event           | sofr_obfr_spread | financial_stress_composite | liquidity_index | qe_intensity | ...
ES            | 1704067200000000000| 0.05             | 0.25                       | 0.65            | 0.42         | ...
```

**All liquidity features**:
- `sofr_obfr_spread`: Repo market stress indicator
- `financial_stress_composite`: Multi-indicator stress measure
- `liquidity_index`: Fed balance sheet + bank credit composite
- `fed_balance_sheet`: Fed total assets
- `qe_intensity`: Balance sheet growth rate
- `bank_credit_growth_3m`: 3-month credit expansion

### Growth/Inflation Dimension (9 features)
```
instrument_id | ts_event           | growth_momentum | inflation_momentum | stagflation_risk | goldilocks_score | ...
ES            | 1704067200000000000| 0.35            | 0.15               | 0.05             | 0.75             | ...
```

**All growth/inflation features**:
- `payems_mom`: Employment month-over-month growth
- `indpro_mom`: Industrial production MoM growth
- `growth_momentum`: Composite growth indicator
- `cpi_yoy`: CPI year-over-year change
- `pce_yoy`: PCE YoY change
- `ppi_yoy`: PPI YoY change
- `inflation_momentum`: Composite inflation indicator
- `stagflation_risk`: High inflation + weak growth score
- `goldilocks_score`: Moderate growth + low inflation score

### FX Dimension (4 features)
```
instrument_id | ts_event           | dollar_strength | dollar_momentum_3m | fx_volatility_composite | fx_stress | ...
ES            | 1704067200000000000| 102.5           | 0.02               | 0.08                    | 0.15      | ...
```

**All FX features**:
- `dollar_strength`: Broad dollar index (DTWEXBGS)
- `dollar_momentum_3m`: 3-month dollar trend
- `fx_volatility_composite`: Cross-pair realized volatility
- `fx_stress`: Cross-currency volatility measure

---

## 3. Cross-Asset Features (New - 3 core features × N pairs)

### EWMA Beta (Instrument vs Macro Factors)
```
instrument_id | ts_event           | ewma_beta_FEDFUNDS | ewma_beta_DGS10 | ewma_beta_VIXCLS | ...
ES            | 1704067200000000000| -0.35              | 0.52            | -0.85            | ...
NQ            | 1704067200000000000| -0.28              | 0.48            | -0.92            | ...
ZN            | 1704067200000000000| 0.15               | 0.88            | -0.35            | ...
```

**Configuration**:
- Top-5 most correlated macro factors per instrument
- Half-life: 30 days (alpha=0.94)
- **Total features**: 95 instruments × 5 factors = **475 beta features**

### Z-Scored Spreads (Instrument Pairs)
```
instrument_id_1 | instrument_id_2 | ts_event           | zscore_spread_ES_NQ | zscore_spread_ZN_ZB | ...
ES              | NQ              | 1704067200000000000| 1.25                | N/A                 | ...
ZN              | ZB              | 1704067200000000000| N/A                 | -0.85               | ...
```

**Configuration**:
- 50-100 selected pairs (factor-based grouping)
- Window: 60 bars rolling
- **Total features**: ~50-100 spread features (selective computation)

### Rolling Correlation (Instrument Pairs)
```
instrument_id_1 | instrument_id_2 | ts_event           | correlation_ES_NQ_60d | correlation_ZN_ZB_60d | ...
ES              | NQ              | 1704067200000000000| 0.85                  | N/A                   | ...
ZN              | ZB              | 1704067200000000000| N/A                   | 0.92                  | ...
```

**Configuration**:
- Same 50-100 pairs as spreads
- Window: 60 bars rolling
- **Total features**: ~50-100 correlation features

**Total cross-asset features**: ~600-650 features (475 betas + 50-100 spreads + 50-100 correlations)

---

## 4. Instrument Metadata (New - Static Covariates)

### Static Categorical Features (5 features)
```
instrument_id | duration_bucket | issuer_type | credit_rating | security_type | liquidity_tier
ES            | 0               | 0           | 0             | 3             | 1
ZN            | 2               | 0           | 0             | 0             | 1
AAPL.NASDAQ   | 1               | 2           | 1             | 0             | 1
```

**Feature definitions**:
- `duration_bucket`: 0=Short (<2y), 1=Medium (2-7y), 2=Long (>7y)
- `issuer_type`: 0=SOVEREIGN, 1=QUASI_SOVEREIGN, 2=CORPORATE_IG, 3=CORPORATE_HY
- `credit_rating`: 0=AAA, 1=AA, 2=A, 3=BBB, 4=BB, 5=B, 6=CCC_BELOW
- `security_type`: 0=SENIOR_UNSECURED, 1=SENIOR_SECURED, 2=SUBORDINATED, 3=STRUCTURED, 4=INFLATION_LINKED
- `liquidity_tier`: 1=High, 2=Medium, 3=Low

**TFT Usage**: These become **static categorical embeddings** that condition the model's temporal attention mechanism.

---

## Complete TFT Dataset Example

### Single Timestep for ES (S&P 500 E-mini)

```python
{
    # Identifiers
    "instrument_id": "ES",
    "ts_event": 1704067200000000000,
    "ts_init": 1704067200000000000,

    # Market data (5)
    "open": 4750.25,
    "high": 4752.50,
    "low": 4748.00,
    "close": 4751.75,
    "volume": 125000,

    # Technical indicators (40)
    "return_1": 0.0002,
    "return_5": 0.0015,
    "return_10": 0.0025,
    "return_20": 0.0045,
    "volatility_20": 0.0125,
    "volatility_60": 0.0105,
    "rsi_14": 55.2,
    "macd": 0.3,
    "bb_upper": 4760.0,
    "bb_lower": 4740.0,
    "atr_14": 12.5,
    # ... (30 more technical features)

    # Macro features (36)
    "credit_spread_ig": 1.25,
    "credit_spread_hy": 3.45,
    "credit_spread_hy_ig": 2.20,
    "term_spread": 0.45,
    "term_spread_5s30s": 1.25,
    "curve_curvature": -0.05,
    "sofr_obfr_spread": 0.05,
    "financial_stress_composite": 0.25,
    "growth_momentum": 0.35,
    "inflation_momentum": 0.15,
    "dollar_strength": 102.5,
    "fx_volatility_composite": 0.08,
    # ... (24 more macro features)

    # Cross-asset features (15 for this instrument)
    "ewma_beta_FEDFUNDS": -0.35,
    "ewma_beta_DGS10": 0.52,
    "ewma_beta_VIXCLS": -0.85,
    "ewma_beta_credit_spread_hy": 0.42,
    "ewma_beta_term_spread": 0.28,
    "zscore_spread_ES_NQ": 1.25,
    "zscore_spread_ES_RTY": -0.85,
    "correlation_ES_NQ_60d": 0.85,
    "correlation_ES_RTY_60d": 0.72,
    # ... (6 more cross-asset features)

    # Static metadata (5) - same for all timesteps
    "duration_bucket": 0,        # Short duration (equity index futures)
    "issuer_type": 0,            # Sovereign (US Treasury underlying)
    "credit_rating": 0,          # AAA (implicit)
    "security_type": 3,          # Structured (derivative)
    "liquidity_tier": 1,         # High liquidity
}
```

**Total features for ES**: ~96 features (5 market + 40 technical + 36 macro + 15 cross-asset + 5 static metadata)

---

## TFT Configuration

### Time-Varying Known Reals (can be observed at prediction time)
```python
time_varying_known_reals = [
    # Macro features (all 36) - known in advance from FRED releases
    "credit_spread_ig", "credit_spread_hy", "term_spread", "sofr_obfr_spread",
    "financial_stress_composite", "growth_momentum", "dollar_strength",
    # ... (all 36 macro composites)
]
```

### Time-Varying Unknown Reals (cannot be observed at prediction time)
```python
time_varying_unknown_reals = [
    # Market data (5)
    "open", "high", "low", "close", "volume",

    # Technical indicators (40)
    "return_1", "return_5", "volatility_20", "rsi_14", "macd",
    # ... (all 40 technical features)

    # Cross-asset features (~15 per instrument)
    "ewma_beta_FEDFUNDS", "ewma_beta_DGS10", "zscore_spread_ES_NQ",
    "correlation_ES_NQ_60d",
    # ... (all cross-asset features)
]
```

### Static Categoricals (fixed per instrument)
```python
static_categoricals = [
    "duration_bucket",    # Will create 3-class embedding
    "issuer_type",        # Will create 4-class embedding
    "credit_rating",      # Will create 7-class embedding
    "security_type",      # Will create 5-class embedding
    "liquidity_tier",     # Will create 3-class embedding
]
```

### Target Variables
```python
target = "return_1"  # Predict next-bar return
# OR
target = "close"     # Predict next-bar price
```

---

## Dataset Statistics

### Dimensionality
- **Instruments**: 95
- **Lookback window**: 60 bars (typical)
- **Features per timestep**: ~96 per instrument
- **Total features in window**: 60 × 96 = 5,760 features per instrument
- **Batch size**: 32 instruments → 184,320 input features per batch

### Temporal Structure
```
Time axis (lookback):  t-59, t-58, ..., t-1, t
                        └──────────────────────┘
                         Historical context (60 bars)

Prediction target:      t+1, t+2, ..., t+20
                        └────────────────────┘
                         Forecast horizon
```

### Memory Footprint (Rough Estimate)
```
Single instrument, 60-bar window:
60 bars × 96 features × 4 bytes (float32) = 23 KB

95 instruments, 60-bar window:
95 × 23 KB = 2.2 MB per batch

250 days of data (97,500 bars total):
97,500 bars × 96 features × 4 bytes × 95 instruments = 3.6 GB
```

### Feature Importance Distribution (Expected)
Based on typical TFT variable selection:
- **Market data**: 30-40% (price, volume, technical indicators)
- **Macro features**: 20-30% (regime signals, risk premia)
- **Cross-asset features**: 15-25% (correlation breakdowns, beta shifts)
- **Static metadata**: 10-15% (duration, liquidity, sector interactions)

---

## Data Pipeline Flow

### Training Data Assembly
```python
from ml.data import TFTDatasetBuilder
from ml.features import FeatureEngineer, compute_macro_composites_pl
from ml.features.cross_asset import compute_ewma_beta_batch
from ml.stores import InstrumentMetadataStore

# 1. Load market data (OHLCV)
market_data = load_bars(instruments=["ES", "NQ", ...], start_date="2024-01-01")

# 2. Compute technical indicators (FeatureEngineer)
feature_config = FeatureConfig(enable_returns=True, enable_volatility=True, ...)
engineer = FeatureEngineer(feature_config)
technical_features = engineer.calculate_features_batch(market_data)

# 3. Join macro composites
macro_features = compute_macro_composites_pl(technical_features)

# 4. Compute cross-asset features
cross_asset_features = compute_cross_asset_features(
    market_data,
    pairs=[("ES", "NQ"), ("ZN", "ZB"), ...],
    factors=["FEDFUNDS", "DGS10", "VIXCLS", ...]
)

# 5. Load instrument metadata
metadata_store = InstrumentMetadataStore(connection_string)
instrument_metadata = metadata_store.get_metadata_for_instruments(["ES", "NQ", ...])

# 6. Combine all features
full_dataset = (
    market_data
    .join(technical_features, on=["instrument_id", "ts_event"])
    .join(macro_features, on=["ts_event"])
    .join(cross_asset_features, on=["instrument_id", "ts_event"])
    .join(instrument_metadata, on=["instrument_id"])
)

# 7. Build TFT dataset
tft_dataset = TFTDatasetBuilder().build_dataset(
    df=full_dataset,
    time_idx="ts_event",
    target="return_1",
    group_ids=["instrument_id"],
    static_categoricals=["duration_bucket", "issuer_type", "liquidity_tier"],
    time_varying_known_reals=macro_feature_names,
    time_varying_unknown_reals=technical_feature_names + cross_asset_feature_names,
)
```

---

## Example Training Configuration

```python
from pytorch_forecasting import TemporalFusionTransformer

tft = TemporalFusionTransformer.from_dataset(
    tft_dataset,

    # Model architecture
    hidden_size=128,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,

    # Static embeddings (instrument metadata)
    static_categoricals=["duration_bucket", "issuer_type", "liquidity_tier"],
    embedding_sizes={
        "duration_bucket": (3, 2),    # 3 classes → 2D embedding
        "issuer_type": (4, 3),         # 4 classes → 3D embedding
        "liquidity_tier": (3, 2),      # 3 classes → 2D embedding
    },

    # Temporal features
    time_varying_known_reals=[...],   # 36 macro features
    time_varying_unknown_reals=[...], # ~150 market + cross-asset features

    # Training config
    learning_rate=1e-3,
    loss=QuantileLoss(),
    reduce_on_plateau_patience=3,
)
```

---

## Key Insights for TFT

### 1. Regime-Aware Predictions
The **36 macro composites** allow TFT to learn regime-dependent patterns:
- **High credit stress** (credit_distress_index > 0.7) → Reduce equity exposure
- **Steepening curve** (term_spread increasing) → Favor long-duration bonds
- **Dollar strength** (dollar_momentum_3m > 0.5) → Adjust FX positioning

### 2. Instrument-Specific Dynamics
The **static metadata** enables TFT to learn that:
- **Long-duration instruments** (duration_bucket=2) are more sensitive to term_spread
- **High-liquidity instruments** (liquidity_tier=1) respond faster to macro shocks
- **Corporate bonds** (issuer_type=2) are more sensitive to credit_spread_hy

### 3. Cross-Asset Signals
The **cross-asset features** capture:
- **Beta breakdowns**: When ewma_beta_VIXCLS spikes → Flight to quality
- **Spread widening**: When zscore_spread_ES_NQ > 2σ → Mean reversion opportunity
- **Correlation breakdowns**: When correlation drops from 0.9 → 0.5 → Diversification opportunity

### 4. Temporal Attention
TFT's variable selection network will automatically:
- Up-weight macro features during regime transitions
- Down-weight stale cross-asset features (low recent variance)
- Focus on instrument-specific technical indicators for short-term predictions

---

## Production Deployment

### Real-Time Inference Pipeline
```python
from ml.actors import BaseMLInferenceActor
from ml.features.cross_asset import compute_ewma_beta_incremental

class TFTPredictionActor(BaseMLInferenceActor):
    def on_bar(self, bar: Bar):
        # 1. Update technical indicators (hot path)
        self.indicator_manager.update_from_bar(bar)
        technical_features = self.feature_engineer.calculate_features_online(bar)

        # 2. Get latest macro features (cache lookup)
        macro_features = self.macro_cache.get_all_features()

        # 3. Update cross-asset features (incremental state)
        beta_state, beta = compute_ewma_beta_incremental(
            self.beta_states[bar.instrument_id],
            bar.close.as_double(),
            self.factor_values["DGS10"]
        )
        self.beta_states[bar.instrument_id] = beta_state

        # 4. Get instrument metadata (static, cached)
        metadata = self.metadata_cache[bar.instrument_id]

        # 5. Combine features
        input_features = {
            **technical_features,
            **macro_features,
            "ewma_beta_DGS10": beta,
            **metadata
        }

        # 6. TFT prediction
        prediction = self.tft_model.predict(input_features)

        # 7. Generate signal
        if prediction > self.threshold:
            self.emit_signal(bar.instrument_id, SignalType.LONG)
```

**Hot path budget**: <5ms for steps 1-6, TFT inference <10ms → **Total <15ms P99**

---

## Summary

The TFT training dataset now includes:

1. **Market Data Layer** (45 features): OHLCV + 40 technical indicators
2. **Macro Layer** (36 features): Credit, duration, liquidity, growth/inflation, FX composites
3. **Cross-Asset Layer** (~600 features): Betas, spreads, correlations across 95 instruments
4. **Metadata Layer** (5 features): Static instrument characteristics as categorical embeddings

**Total**: ~680 features per instrument-timestep, providing TFT with:
- Rich temporal context (60-bar lookback × 680 features)
- Regime awareness (36 macro composites)
- Cross-instrument signals (600 interaction features)
- Instrument-specific conditioning (5 static embeddings)

This enables TFT to learn complex, regime-dependent, multi-instrument trading strategies with unprecedented feature richness.
