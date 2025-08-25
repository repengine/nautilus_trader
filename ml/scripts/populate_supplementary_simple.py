#!/usr/bin/env python3
"""
Populate supplementary data using simple HTTP requests.
No external dependencies beyond what's already installed.
"""

import argparse
import json
import logging
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pandas as pd


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Supplementary symbols for regime detection
SUPPLEMENTARY_SYMBOLS = {
    "indices": ["^GSPC", "^DJI", "^IXIC", "^RUT"],  # S&P, Dow, Nasdaq, Russell
    "sectors": ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC"],
    "factors": ["IWF", "IWD", "IWM", "IWB", "MTUM", "QUAL", "USMV"],
    "international": ["EWJ", "EWG", "EWU", "FXI", "EWZ", "EWA", "EWC", "INDA", "EEM", "EFA"],
    "commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC"],
    "bonds": ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "EMB", "AGG"],
    "currencies": ["UUP", "FXE", "FXY", "FXB", "FXC", "FXA", "FXF"],
    "volatility": ["VXX", "VIXY", "VXZ", "SVXY", "UVXY"],
}

def fetch_alpha_vantage_data(symbol: str, api_key: str = "demo") -> pd.DataFrame:
    """
    Fetch data from Alpha Vantage (free tier allows 25 requests/day).
    Using 'demo' key for testing.
    """
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": "full",
        "datatype": "json"
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())

        if "Time Series (Daily)" in data:
            ts = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(ts, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df.columns = ["open", "high", "low", "close", "volume"]
            df["symbol"] = symbol
            return df.sort_index()

    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")

    return pd.DataFrame()

def create_synthetic_supplementary_data() -> pd.DataFrame:
    """
    Create synthetic supplementary data for testing.
    In production, would fetch from Yahoo/Alpha Vantage.
    """
    logger.info("Creating synthetic supplementary data for TFT training...")

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    all_data = []

    # Get all symbols
    all_symbols = []
    for symbols in SUPPLEMENTARY_SYMBOLS.values():
        all_symbols.extend(symbols)

    for symbol in all_symbols:
        # Create realistic synthetic data
        base_price = 100.0
        volatility = 0.02

        # Adjust parameters based on symbol type
        if symbol in SUPPLEMENTARY_SYMBOLS.get("volatility", []):
            volatility = 0.10  # Higher vol for VIX products
        elif symbol in SUPPLEMENTARY_SYMBOLS.get("bonds", []):
            volatility = 0.01  # Lower vol for bonds
        elif symbol in SUPPLEMENTARY_SYMBOLS.get("commodities", []):
            volatility = 0.03  # Medium-high for commodities

        # Generate price series with random walk
        import numpy as np
        np.random.seed(hash(symbol) % 2**32)  # Consistent per symbol

        returns = np.random.normal(0.0001, volatility, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        # Add some trend
        trend = np.linspace(0, 0.2, len(dates))
        prices = prices * (1 + trend)

        # Create OHLCV data
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": symbol,
            "open": prices * (1 + np.random.normal(0, 0.001, len(dates))),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            "close": prices,
            "volume": np.random.lognormal(15, 1, len(dates)).astype(int) * 1000
        })

        # Add derived features
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility_20d"] = df["returns"].rolling(20).std()
        df["volume_ma_20d"] = df["volume"].rolling(20).mean()
        df["rsi_14"] = calculate_rsi(df["close"], 14)

        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Created synthetic data for {len(all_symbols)} symbols, {len(combined)} total records")

    return combined

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_correlations(data: pd.DataFrame, base_symbols: list[str]) -> pd.DataFrame:
    """Calculate rolling correlations."""
    logger.info("Calculating correlations...")

    # Pivot to get returns by symbol
    pivot = data.pivot_table(
        index="timestamp",
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

            # 60-day rolling correlation
            corr = pivot[base].rolling(60).corr(pivot[symbol])

            corr_df = pd.DataFrame({
                "timestamp": corr.index,
                "base_symbol": base,
                "corr_symbol": symbol,
                "correlation_60d": corr.values
            })

            correlations.append(corr_df)

    if correlations:
        return pd.concat(correlations, ignore_index=True)

    return pd.DataFrame()

def calculate_spreads(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate important spreads and ratios."""
    logger.info("Calculating spreads...")

    # Pivot to get prices
    pivot = data.pivot_table(
        index="timestamp",
        columns="symbol",
        values="close"
    )

    spreads_list = []

    # Key spread pairs for regime detection
    spread_pairs = [
        ("TLT", "IEF", "yield_curve"),      # Long vs short duration
        ("HYG", "LQD", "credit_spread"),    # High yield vs IG
        ("XLK", "XLU", "tech_utilities"),   # Risk on vs off
        ("IWF", "IWD", "growth_value"),     # Growth vs Value
        ("EEM", "EFA", "em_dm"),            # EM vs DM
        ("GLD", "TLT", "gold_bonds"),       # Gold vs Bonds
        ("FXY", "FXA", "safe_risk_fx"),     # Yen vs Aussie
    ]

    for long_sym, short_sym, spread_name in spread_pairs:
        if long_sym in pivot.columns and short_sym in pivot.columns:
            spread_df = pd.DataFrame({
                "timestamp": pivot.index,
                "spread_name": spread_name,
                f"{spread_name}_ratio": pivot[long_sym] / pivot[short_sym],
            })

            # Add moving average and z-score
            spread_df[f"{spread_name}_ma20"] = spread_df[f"{spread_name}_ratio"].rolling(20).mean()

            rolling_mean = spread_df[f"{spread_name}_ratio"].rolling(60).mean()
            rolling_std = spread_df[f"{spread_name}_ratio"].rolling(60).std()
            spread_df[f"{spread_name}_zscore"] = (spread_df[f"{spread_name}_ratio"] - rolling_mean) / rolling_std

            spreads_list.append(spread_df)

    if spreads_list:
        return pd.concat(spreads_list, ignore_index=True)

    return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Populate supplementary data")

    parser.add_argument("--output-dir", type=Path,
                       default=Path("data/supplementary"),
                       help="Output directory")
    parser.add_argument("--use-alpha-vantage", action="store_true",
                       help="Fetch from Alpha Vantage (requires API key)")
    parser.add_argument("--api-key", type=str, default="demo",
                       help="Alpha Vantage API key")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch or create data
    if args.use_alpha_vantage:
        logger.info("Fetching from Alpha Vantage (limited to 25 requests/day)...")
        # Would fetch real data here
        data = pd.DataFrame()
    else:
        logger.info("Creating synthetic supplementary data...")
        data = create_synthetic_supplementary_data()

    if data.empty:
        logger.error("No data created")
        return 1

    # Save main data
    output_file = args.output_dir / "supplementary_ohlcv.parquet"
    data.to_parquet(output_file)
    logger.info(f"Saved OHLCV data to {output_file} ({len(data)} records)")

    # Calculate correlations with base symbols
    base_symbols = ["SPY", "QQQ", "TLT", "GLD", "XLK", "XLF"]
    correlations = calculate_correlations(data, base_symbols)
    if not correlations.empty:
        corr_file = args.output_dir / "correlations.parquet"
        correlations.to_parquet(corr_file)
        logger.info(f"Saved correlations to {corr_file} ({len(correlations)} records)")

    # Calculate spreads
    spreads = calculate_spreads(data)
    if not spreads.empty:
        spreads_file = args.output_dir / "spreads.parquet"
        spreads.to_parquet(spreads_file)
        logger.info(f"Saved spreads to {spreads_file} ({len(spreads)} records)")

    # Create metadata file
    metadata = {
        "created": datetime.now().isoformat(),
        "symbols": {
            category: symbols
            for category, symbols in SUPPLEMENTARY_SYMBOLS.items()
        },
        "total_symbols": len(set(data["symbol"].unique())),
        "date_range": {
            "start": str(data["timestamp"].min()),
            "end": str(data["timestamp"].max())
        },
        "record_count": len(data)
    }

    metadata_file = args.output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_file}")

    logger.info("Supplementary data population complete!")

    # Print summary
    print("\n" + "="*50)
    print("SUPPLEMENTARY DATA SUMMARY")
    print("="*50)
    print(f"Total symbols: {len(set(data['symbol'].unique()))}")
    print(f"Total records: {len(data):,}")
    print(f"Date range: {data['timestamp'].min().date()} to {data['timestamp'].max().date()}")
    print(f"Output directory: {args.output_dir}")
    print("\nFiles created:")
    for file in args.output_dir.glob("*.parquet"):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name}: {size_mb:.1f} MB")

    return 0

if __name__ == "__main__":
    sys.exit(main())
