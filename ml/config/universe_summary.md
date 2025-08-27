# Trading Universe Design Summary

## Overview

This document outlines the comprehensive tiered trading universe designed for Nautilus Trader's ML pipeline. The universe is structured in three tiers to balance breadth, liquidity, and computational feasibility while maximizing opportunities for various ML strategies.

## Current Status Analysis

**Previous Universe**: 15-20 core symbols focused on major indices and mega-cap tech stocks
- Core holdings: SPY, QQQ, IWM, AAPL, MSFT, NVDA, AMZN, META, TSLA
- Limited sector diversity and cross-asset coverage
- Databento subscription allows up to 750+ symbols at no extra cost

## Tiered Universe Structure

### Tier 1: Core Universe (150 symbols)
**File**: `universe_tier1.json`

**Criteria**:
- Minimum average daily volume: $50M
- Minimum market cap: $10B
- Data requirement: 7 years clean data
- Update frequency: Quarterly

**Composition**:
- 14 Essential ETF proxies (SPY, QQQ, IWM, sector ETFs, TLT, GLD, etc.)
- 28 Mega-cap stocks (AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, etc.)
- 60+ High-volume leaders across all sectors
- 20+ Sector leaders and defensive plays
- 15+ International exposure (EFA, EEM, VEA, VWO)
- 10+ Currency/commodity ETFs (UUP, FXE, USO, UNG)
- 10+ High-beta momentum plays and crypto exposure

**Use Cases**:
- High-frequency trading
- Cross-sectional momentum
- Market regime detection
- Real-time inference

### Tier 2: Extended Universe (300 symbols)
**File**: `universe_tier2.json`

**Criteria**:
- Minimum average daily volume: $10M
- Minimum market cap: $5B
- Data requirement: 5 years clean data
- Update frequency: Monthly

**Composition**:
- All Tier 1 symbols plus:
- Expanded S&P 500 coverage
- Additional sector ETFs (XLP, XLY, XLU, XLB, XLRE)
- Style factor ETFs (IWF, IWD, IJH, IJR)
- International ADRs (TSM, ASML, NVO, BABA, TM)
- Healthcare/biotech expansion
- Industrial and energy sectors
- Fixed income expansion (IEF, SHY, LQD, HYG, TIP)
- Commodities expansion (PDBC, DBA, PALL, PPLT)

**Use Cases**:
- Sector rotation strategies
- Cross-sectional factor models
- Pairs trading identification
- Regional/style factor exposure

### Tier 3: Full Universe (750 symbols)
**File**: `universe_tier3.json`

**Criteria**:
- Minimum average daily volume: $5M
- Minimum market cap: $1B
- Data requirement: 3 years clean data
- Update frequency: Quarterly

**Composition**:
- All Tier 1 & 2 symbols plus:
- Remaining S&P 500 members
- Mid-cap leaders and growth stocks
- Specialty/thematic ETFs (ARKK, ICLN, HACK, BOTZ)
- Volatility products (UVXY, SVXY, VXX)
- Leveraged/inverse ETFs (TQQQ, SQQQ, UPRO, SH, PSQ)
- Extended international coverage
- Small-cap and factor ETFs
- Gaming, entertainment, and emerging sectors

**Use Cases**:
- Global macro strategies
- Market structure analysis
- Specialty factor strategies
- Complete cross-sectional universe

### Proxy Universe (150 symbols)
**File**: `universe_proxies.json`

**Comprehensive ETF proxy collection organized by**:
- Equity indices (US broad market, sectors, international)
- Fixed income (treasuries, corporate, municipal, international)
- Commodities (precious metals, energy, agriculture, diversified)
- Currencies (USD, EUR, JPY, GBP, CAD, AUD)
- Real estate and volatility products
- Style factors (growth/value, momentum, quality, dividend)
- Thematic exposures (technology, clean energy)
- Leveraged and inverse products

## Strategy Recommendations

### Cross-Sectional Momentum
**Recommended Symbols**: SPY, QQQ, XLK, XLF, XLE, XLV, IWM, EFA, EEM
- Focus on liquid ETFs with distinct factor exposures
- Use Tier 1 for real-time signals
- Expand to Tier 2 for broader opportunity set

### Sector Rotation Models
**Recommended Symbols**: All XL* sector ETFs, style factors, defensive proxies
- Monitor relative performance across 11 GICS sectors
- Include international exposure for global rotation
- Use factor ETFs to identify style preferences

### Pairs Trading
**Recommended Symbols**: SPY/QQQ, TLT/HYG, GLD/USO, EFA/EEM
- Focus on liquid pairs with economic relationships
- Cross-asset pairs for regime changes
- Use correlation analysis across tiers

### Market Regime Detection
**Recommended Symbols**: VIX, UVXY, TLT, GLD, UUP, HYG, LQD, EEM
- Volatility products for fear/greed measurement
- Flight-to-quality assets (TLT, GLD)
- Risk-on/risk-off indicators (HYG vs TLT)
- Currency strength (UUP) and EM performance (EEM)

### Global Macro Strategies
**Recommended Symbols**: SPY, EFA, EEM, TLT, GLD, USO, UUP, FXE, FXY, VIX
- Multi-asset approach across equities, bonds, commodities, currencies
- Regional equity exposure through ETFs
- Commodity and currency proxies for macro themes

## Implementation Guidelines

### Data Collection Priority
1. **Phase 1**: Collect Tier 1 (150 symbols) first
   - 7 years of L0 (OHLCV) data
   - 1 year of L1 (quotes/trades) data
   - 30 days of L2/L3 (market depth) for top 20 symbols

2. **Phase 2**: Add Tier 2 symbols (additional 150)
   - 5 years of L0 data minimum
   - L1 data for top 50 by volume
   - L2 data for sector leaders

3. **Phase 3**: Complete with Tier 3 (additional 300)
   - 3 years of L0 data minimum
   - Selective L1/L2 based on strategy needs

### Computational Considerations
- **Real-time inference**: Use Tier 1 only (150 symbols)
- **Feature engineering**: Tier 1 + selective Tier 2 (200-250 symbols)
- **Research and backtesting**: Full universe as needed
- **Cross-sectional models**: Focus on liquid subset within each tier

### Update Schedule
- **Tier 1**: Quarterly review, annual major updates
- **Tier 2**: Monthly monitoring, quarterly updates
- **Tier 3**: Quarterly review for additions/removals
- **Proxies**: Semi-annual review for new products

## Risk Management

### Liquidity Risk
- All Tier 1 symbols exceed $50M daily volume
- Tier 2/3 symbols monitored for liquidity deterioration
- Alternative proxies identified for each major exposure

### Data Quality
- Minimum data history requirements enforced
- Regular quality checks for gaps, outliers, corporate actions
- Backup data sources identified for critical symbols

### Concentration Risk
- Maximum 10% allocation to any single sector in portfolios
- Geographic diversification through international ETFs
- Asset class diversification through fixed income and commodity proxies

## Future Enhancements

### Potential Additions
- Cryptocurrency ETFs (if approved and liquid)
- Additional international single-country ETFs
- ESG-focused factor ETFs
- Options-based strategy ETFs

### Dynamic Universe Management
- Automated liquidity monitoring
- Momentum-based inclusions/exclusions
- Regime-dependent universe selection
- Machine learning-driven symbol selection

## File Locations

All universe configuration files are located in `/home/nate/projects/nautilus_trader/ml/config/`:

- `universe_tier1.json` - Core 150 symbols for high-frequency and real-time strategies
- `universe_tier2.json` - Extended 300 symbols for sector and factor strategies
- `universe_tier3.json` - Full 750 symbols for comprehensive analysis
- `universe_proxies.json` - 150 ETF proxies organized by asset class and strategy
- `universe_summary.md` - This summary document

Each JSON file includes detailed metadata, selection criteria, symbol classifications, and priority rankings for systematic implementation.
