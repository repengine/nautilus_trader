#!/usr/bin/env python3
"""
Populate additional data from Yahoo Finance.

Free data that adds value for TFT:
1. Sector/Industry ETF prices (for regime detection)
2. Commodity ETFs (correlations and macro trends)
3. Currency ETFs (risk-on/risk-off indicators)
4. Bond ETFs (yield curve dynamics)
5. International indices (global market conditions)
6. Volatility products (beyond VIX)

Usage:
    python ml/scripts/populate_yahoo_data.py --all
    python ml/scripts/populate_yahoo_data.py --category sectors
"""

import argparse
import logging
import sys
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Define universe of supplementary symbols
SUPPLEMENTARY_UNIVERSE = {
    # Sector ETFs (for sector rotation signals)
    "sectors": [
        "XLK",  # Technology
        "XLF",  # Financials
        "XLV",  # Healthcare
        "XLE",  # Energy
        "XLI",  # Industrials
        "XLY",  # Consumer Discretionary
        "XLP",  # Consumer Staples
        "XLB",  # Materials
        "XLRE", # Real Estate
        "XLU",  # Utilities
        "XLC",  # Communication Services
    ],

    # Style/Factor ETFs
    "factors": [
        "IWF",  # Growth
        "IWD",  # Value
        "IWM",  # Small Cap
        "IWB",  # Large Cap
        "MTUM", # Momentum
        "QUAL", # Quality
        "USMV", # Low Volatility
        "SIZE", # Size Factor
    ],

    # International Markets (for global regime)
    "international": [
        "EWJ",  # Japan
        "EWG",  # Germany
        "EWU",  # UK
        "FXI",  # China
        "EWZ",  # Brazil
        "EWA",  # Australia
        "EWC",  # Canada
        "INDA", # India
        "EEM",  # Emerging Markets
        "EFA",  # Developed ex-US
    ],

    # Commodities (inflation/growth signals)
    "commodities": [
        "GLD",  # Gold
        "SLV",  # Silver
        "USO",  # Oil
        "UNG",  # Natural Gas
        "DBA",  # Agriculture
        "DBB",  # Base Metals
        "DBC",  # Broad Commodities
        "COPX", # Copper Miners (copper = economic indicator)
    ],

    # Bonds/Rates (yield curve, risk-off)
    "bonds": [
        "SHY",  # 1-3 Year Treasury
        "IEF",  # 7-10 Year Treasury
        "TLT",  # 20+ Year Treasury
        "TIP",  # TIPS (inflation protected)
        "LQD",  # Investment Grade Corporate
        "HYG",  # High Yield Corporate
        "EMB",  # Emerging Market Bonds
        "AGG",  # Aggregate Bond
    ],

    # Currencies (risk sentiment)
    "currencies": [
        "UUP",  # US Dollar
        "FXE",  # Euro
        "FXY",  # Yen (safe haven)
        "FXB",  # British Pound
        "FXC",  # Canadian Dollar
        "FXA",  # Australian Dollar (risk-on)
        "FXF",  # Swiss Franc (safe haven)
        "UDN",  # Dollar Bear
    ],

    # Volatility (beyond VIX)
    "volatility": [
        "VXX",  # Short-term VIX futures
        "VIXY", # VIX ETF
        "VXZ",  # Mid-term VIX futures
        "SVXY", # Inverse VIX
        "UVXY", # Ultra VIX
        "VIXM", # VIX mid-term
        "VIIX", # VIX Index ETN
    ],

    # Thematic/Sentiment
    "thematic": [
        "ARKK", # Innovation (risk appetite)
        "ICLN", # Clean Energy
        "JETS", # Airlines (recovery play)
        "XRT",  # Retail (consumer strength)
        "XHB",  # Homebuilders (housing)
        "KRE",  # Regional Banks
        "XME",  # Metals & Mining
        "GDXJ", # Junior Gold Miners (speculation)
    ]
}


class YahooDataLoader:
    """Load supplementary data from Yahoo Finance."""

    def __init__(self):
        self.symbols_cache = {}

    def fetch_historical(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch historical data for symbols."""
        logger.info(f"Fetching Yahoo data for {len(symbols)} symbols...")

        all_data = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)

                # Get historical data
                hist = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval="1d"
                )

                if hist.empty:
                    logger.warning(f"No data for {symbol}")
                    continue

                # Add symbol column
                hist["symbol"] = symbol

                # Add additional metrics
                hist["returns"] = hist["Close"].pct_change()
                hist["log_returns"] = pd.np.log(hist["Close"] / hist["Close"].shift(1))
                hist["volatility_20d"] = hist["returns"].rolling(20).std()
                hist["volume_ma_20d"] = hist["Volume"].rolling(20).mean()

                # Add relative strength
                hist["rs_5d"] = hist["Close"] / hist["Close"].shift(5) - 1
                hist["rs_20d"] = hist["Close"] / hist["Close"].shift(20) - 1
                hist["rs_60d"] = hist["Close"] / hist["Close"].shift(60) - 1

                all_data.append(hist)
                logger.info(f"  ✓ {symbol}: {len(hist)} days")

            except Exception as e:
                logger.error(f"  ✗ {symbol}: {e}")

        if all_data:
            combined = pd.concat(all_data)
            return combined

        return pd.DataFrame()

    def fetch_info(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch symbol info and fundamentals."""
        logger.info(f"Fetching symbol info for {len(symbols)} symbols...")

        info_data = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                # Extract relevant fields
                symbol_info = {
                    "symbol": symbol,
                    "longName": info.get("longName", ""),
                    "sector": info.get("sector", ""),
                    "industry": info.get("industry", ""),
                    "marketCap": info.get("marketCap", 0),
                    "beta": info.get("beta", 1.0),
                    "trailingPE": info.get("trailingPE", 0),
                    "dividendYield": info.get("dividendYield", 0),
                    "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", 0),
                    "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", 0),
                    "averageVolume": info.get("averageVolume", 0),
                }

                info_data.append(symbol_info)

            except Exception as e:
                logger.error(f"Failed to get info for {symbol}: {e}")

        return pd.DataFrame(info_data)

    def calculate_correlations(
        self,
        data: pd.DataFrame,
        base_symbols: list[str],
        window: int = 60
    ) -> pd.DataFrame:
        """Calculate rolling correlations with base symbols."""
        logger.info("Calculating rolling correlations...")

        # Pivot data to have symbols as columns
        pivot = data.pivot_table(
            index="Date",
            columns="symbol",
            values="returns"
        )

        correlations = []

        for base in base_symbols:
            if base not in pivot.columns:
                continue

            for symbol in pivot.columns:
                if symbol == base:
                    continue

                # Calculate rolling correlation
                corr = pivot[base].rolling(window).corr(pivot[symbol])

                corr_df = pd.DataFrame({
                    "date": corr.index,
                    "base_symbol": base,
                    "corr_symbol": symbol,
                    "correlation": corr.to_numpy(),
                    f"corr_{window}d": corr.to_numpy(),
                })

                correlations.append(corr_df)

        if correlations:
            return pd.concat(correlations)

        return pd.DataFrame()

    def calculate_spreads(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate important spreads and ratios."""
        logger.info("Calculating spreads and ratios...")

        # Pivot data
        pivot = data.pivot_table(
            index="Date",
            columns="symbol",
            values="Close"
        )

        spreads = pd.DataFrame(index=pivot.index)

        # Define spread pairs
        spread_pairs = [
            ("TLT", "IEF", "yield_curve"),  # Long vs short duration
            ("HYG", "LQD", "credit_spread"),  # High yield vs IG
            ("XLK", "XLU", "risk_on_off"),  # Tech vs Utilities
            ("IWF", "IWD", "growth_value"),  # Growth vs Value
            ("EEM", "EFA", "em_dm_spread"),  # EM vs DM
            ("GLD", "TLT", "gold_bonds"),  # Gold vs Bonds
            ("FXY", "FXA", "safe_risk_fx"),  # Yen vs Aussie
        ]

        for long_sym, short_sym, spread_name in spread_pairs:
            if long_sym in pivot.columns and short_sym in pivot.columns:
                spreads[spread_name] = pivot[long_sym] / pivot[short_sym]
                spreads[f"{spread_name}_ma20"] = spreads[spread_name].rolling(20).mean()
                spreads[f"{spread_name}_zscore"] = (
                    (spreads[spread_name] - spreads[spread_name].rolling(60).mean()) /
                    spreads[spread_name].rolling(60).std()
                )

        return spreads


def main():
    parser = argparse.ArgumentParser(description="Populate Yahoo Finance data")

    parser.add_argument("--all", action="store_true",
                       help="Download all categories")
    parser.add_argument("--category", choices=list(SUPPLEMENTARY_UNIVERSE.keys()),
                       help="Specific category to download")
    parser.add_argument("--years", type=int, default=2,
                       help="Years of history to download")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("data/supplementary"),
                       help="Output directory")

    args = parser.parse_args()

    # Determine symbols to download
    if args.all:
        symbols = []
        for category_symbols in SUPPLEMENTARY_UNIVERSE.values():
            symbols.extend(category_symbols)
        symbols = list(set(symbols))  # Remove duplicates
    elif args.category:
        symbols = SUPPLEMENTARY_UNIVERSE[args.category]
    else:
        logger.error("Specify --all or --category")
        return 1

    logger.info(f"Downloading data for {len(symbols)} symbols")

    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.years * 365)

    # Initialize loader
    loader = YahooDataLoader()

    # Fetch historical data
    hist_data = loader.fetch_historical(symbols, start_date, end_date)

    if hist_data.empty:
        logger.error("No data fetched")
        return 1

    # Save historical data
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_file = args.output_dir / "yahoo_historical.parquet"
    hist_data.to_parquet(output_file)
    logger.info(f"Saved historical data to {output_file}")

    # Fetch symbol info
    info_data = loader.fetch_info(symbols)
    if not info_data.empty:
        info_file = args.output_dir / "yahoo_info.parquet"
        info_data.to_parquet(info_file)
        logger.info(f"Saved symbol info to {info_file}")

    # Calculate correlations with major indices
    base_symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD", "VIX"]
    correlations = loader.calculate_correlations(hist_data, base_symbols)
    if not correlations.empty:
        corr_file = args.output_dir / "correlations.parquet"
        correlations.to_parquet(corr_file)
        logger.info(f"Saved correlations to {corr_file}")

    # Calculate spreads
    spreads = loader.calculate_spreads(hist_data)
    if not spreads.empty:
        spreads_file = args.output_dir / "spreads.parquet"
        spreads.to_parquet(spreads_file)
        logger.info(f"Saved spreads to {spreads_file}")

    logger.info("Yahoo data population complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
