# Additional Data Sources for TFT Training

## Currently Implemented ✅

### 1. Market Data (Databento)
- **L0**: 7 years OHLCV bars 
- **L1**: 1 year quotes/trades
- **L2**: 30 days market depth (in progress)
- **Cost**: $0 with EQUS.MINI subscription

### 2. Economic Indicators (FRED)
- **24 indicators** including:
  - Interest rates (Fed funds, Treasury yields, SOFR)
  - Economic data (GDP, CPI, unemployment, retail sales)
  - Market indicators (VIX, credit spreads)
  - Currency indices
- **Cost**: Free
- **Update**: Daily/Monthly depending on series
- **Script**: `ml/scripts/populate_fred_data.py`

## High-Value Additions 🎯

### 3. Supplementary Market Data (Yahoo Finance)
**Why valuable**: Provides regime context and cross-asset correlations

- **Sector ETFs** (XLK, XLF, XLV, etc.) - sector rotation signals
- **Factor ETFs** (IWF, IWD, MTUM) - style regime detection  
- **International** (EWJ, FXI, EEM) - global risk sentiment
- **Commodities** (GLD, USO, DBA) - inflation/growth signals
- **Bonds** (TLT, HYG, LQD) - yield curve, credit conditions
- **Currencies** (FXY, FXA) - risk-on/risk-off
- **Volatility** (VXX, UVXY) - term structure

**Cost**: Free
**Script**: `ml/scripts/populate_yahoo_data.py`

### 4. Options Flow (CBOE)
**Why valuable**: Forward-looking sentiment from options market

- **Put/Call Ratios** (total, equity, index)
- **VIX Term Structure** (contango/backwardation)
- **Options Volume** by strike/expiry
- **Skew Index** (tail risk)

**Cost**: Free (delayed data)
**Frequency**: Daily EOD

### 5. Positioning Data (CFTC)
**Why valuable**: Smart money positioning

- **COT Reports** for futures
- Commercial vs Non-commercial positioning
- Focus on: ES (S&P), NQ (Nasdaq), VX (VIX), DX (Dollar)

**Cost**: Free
**Frequency**: Weekly (Tuesdays)

### 6. Short Interest (FINRA)
**Why valuable**: Crowded shorts = squeeze potential

- Short interest by symbol
- Days to cover
- Short % of float

**Cost**: Free (bi-monthly, delayed)

### 7. Market Microstructure (Calculated)
**Why valuable**: Execution quality and toxicity metrics

Calculate from L1/L2 data:
- **Effective/Realized Spread**
- **Kyle's Lambda** (price impact)
- **Amihud Illiquidity**
- **VPIN** (order flow toxicity)
- **Trade Imbalance**

**Cost**: Free (compute from existing data)

## Nice-to-Have Additions 📊

### 8. Alternative Data
- **News Sentiment** (NewsAPI free tier: 100 req/day)
- **Social Sentiment** (Reddit API - WSB mentions)
- **Earnings Calendar** (Yahoo Finance)
- **Analyst Estimates** (limited free from Yahoo)
- **Sector/Industry Maps** (for factor exposures)

### 9. Macro Indicators
- **AAII Sentiment Survey** (weekly)
- **Michigan Consumer Sentiment** (via FRED)
- **Baltic Dry Index** (global trade)
- **Copper/Gold Ratio** (growth expectations)

## Implementation Priority 📋

1. **Yahoo supplementary data** ← Start here (easy, high value)
   ```bash
   pip install yfinance
   python ml/scripts/populate_yahoo_data.py --all --years 2
   ```

2. **FRED economic data** ← Already done
   ```bash
   python ml/scripts/populate_fred_data.py --backfill --years 10
   ```

3. **Market microstructure** ← After L2 completes
   ```bash
   python ml/scripts/populate_alternative_data.py --source micro
   ```

4. **Options flow** ← Requires API setup
5. **COT reports** ← Weekly update process
6. **Short interest** ← Bi-monthly updates

## Data Schema for TFT

All data should be aligned to daily frequency with these columns:
- `timestamp` (datetime)
- `symbol` (str) 
- `feature_name` (str)
- `value` (float)
- `timestamp_ns` (int64)

This allows joining with the main OHLCV data on (timestamp, symbol).

## Storage Estimates

- **Yahoo supplementary**: ~500MB for 2 years, all symbols
- **FRED indicators**: ~50MB for 10 years
- **Options flow**: ~100MB per year
- **COT reports**: ~10MB per year
- **Microstructure**: ~1GB (calculated from L2)

**Total additional**: ~2GB (compared to 85GB for L2 alone)

## Next Steps

1. Run Yahoo data population:
   ```bash
   python ml/scripts/populate_yahoo_data.py --all --years 2
   ```

2. Verify FRED data is complete:
   ```bash
   python ml/scripts/populate_fred_data.py --status
   ```

3. Once L2 completes, calculate microstructure:
   ```bash
   python ml/scripts/populate_alternative_data.py --source micro
   ```

These additional sources will significantly enhance the TFT model's ability to:
- Detect regime changes
- Understand cross-asset correlations
- Anticipate volatility shifts
- Identify sector rotations
- Gauge market sentiment