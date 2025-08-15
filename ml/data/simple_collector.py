#!/usr/bin/env python
"""
Simplified Historical Data Collection System.
Collects data within license limits and stores with features.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import databento as db
from sqlalchemy import create_engine, text

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class SimpleHistoricalCollector:
    """
    Simplified collector for historical market data.
    """
    
    def __init__(self):
        """Initialize the collector."""
        self.databento_key = os.getenv("DATABENTO_API_KEY")
        if not self.databento_key:
            raise ValueError("DATABENTO_API_KEY not set")
            
        self.client = db.Historical(self.databento_key)
        self.db_connection = os.getenv(
            "DB_CONNECTION",
            "postgresql://postgres:postgres@localhost:5432/nautilus"
        )
        
        print("Historical Data Collector initialized")
        
    def check_availability(self, symbol: str = "SPY") -> Dict:
        """Check what data is available for free."""
        results = {}
        
        # Test configurations based on your subscription
        tests = [
            ("1min_bars_1y", "XNAS.BASIC", "ohlcv-1m", 365),
            ("1hour_bars_2y", "XNAS.BASIC", "ohlcv-1h", 730),
            ("daily_bars_5y", "XNAS.BASIC", "ohlcv-1d", 1825),
            ("trades_1y", "XNAS.BASIC", "trades", 365),
            ("quotes_1y", "EQUS.MINI", "tbbo", 365),
            ("depth_30d", "EQUS.MINI", "mbp-1", 30),
        ]
        
        for name, dataset, schema, days in tests:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                cost = self.client.metadata.get_cost(
                    dataset=dataset,
                    symbols=[symbol],
                    schema=schema,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                )
                
                results[name] = {
                    "dataset": dataset,
                    "schema": schema,
                    "days": days,
                    "cost": cost,
                    "free": cost == 0,
                    "available": True,
                }
                
            except Exception as e:
                results[name] = {
                    "dataset": dataset,
                    "schema": schema,
                    "days": days,
                    "available": False,
                    "error": str(e)[:50],
                }
                
        return results
    
    def collect_data(
        self,
        symbol: str,
        dataset: str = "XNAS.BASIC",
        schema: str = "ohlcv-1m",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Collect historical data for a symbol.
        
        Returns
        -------
        pd.DataFrame
            The collected data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"Collecting {schema} for {symbol} ({days} days)...")
        
        try:
            # Download data
            data = self.client.timeseries.get_range(
                dataset=dataset,
                symbols=[symbol],
                schema=schema,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )
            
            # Convert to DataFrame
            df = data.to_df()
            print(f"  Collected {len(df):,} records")
            
            return df
            
        except Exception as e:
            print(f"  Error: {e}")
            return pd.DataFrame()
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic features from OHLCV data.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with features
        """
        if df.empty or "close" not in df.columns:
            return df
            
        # Calculate basic technical indicators
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features["close"] = df["close"]
        features["volume"] = df["volume"] if "volume" in df.columns else 0
        
        # Simple Moving Averages
        features["sma_10"] = df["close"].rolling(10).mean()
        features["sma_20"] = df["close"].rolling(20).mean()
        features["sma_50"] = df["close"].rolling(50).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features["rsi"] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        features["bb_upper"] = sma + (std * 2)
        features["bb_lower"] = sma - (std * 2)
        features["bb_width"] = features["bb_upper"] - features["bb_lower"]
        
        # Volume indicators
        if "volume" in df.columns:
            features["volume_sma"] = df["volume"].rolling(20).mean()
            features["volume_ratio"] = df["volume"] / features["volume_sma"]
        
        # Price change features
        features["returns_1d"] = df["close"].pct_change()
        features["returns_5d"] = df["close"].pct_change(5)
        features["returns_20d"] = df["close"].pct_change(20)
        
        # Volatility
        features["volatility_20d"] = features["returns_1d"].rolling(20).std()
        
        # Drop NaN rows from initialization
        features = features.dropna()
        
        print(f"  Calculated {len(features.columns)} features for {len(features)} rows")
        
        return features
    
    def store_to_database(self, df: pd.DataFrame, table_name: str, symbol: str):
        """
        Store data to PostgreSQL database.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to store
        table_name : str
            Target table name
        symbol : str
            Symbol identifier
        """
        if df.empty:
            return
            
        try:
            engine = create_engine(self.db_connection)
            
            # Add metadata columns
            df["symbol"] = symbol
            df["collected_at"] = datetime.now()
            
            # Store to database
            df.to_sql(
                table_name,
                engine,
                schema="ml",
                if_exists="append",
                index=True,
                method="multi",
            )
            
            print(f"  Stored {len(df)} rows to ml.{table_name}")
            
        except Exception as e:
            print(f"  Database storage error: {e}")
            # Save to local file as backup
            backup_path = f"data/{symbol}_{table_name}_{datetime.now().strftime('%Y%m%d')}.parquet"
            Path("data").mkdir(exist_ok=True)
            df.to_parquet(backup_path)
            print(f"  Saved backup to {backup_path}")
    
    def run_collection(
        self,
        symbols: List[str] = None,
        days_back: int = 30,
        only_free: bool = True,
    ):
        """
        Run full collection pipeline.
        
        Parameters
        ----------
        symbols : List[str]
            Symbols to collect
        days_back : int
            Number of days to collect
        only_free : bool
            Only collect free data
        """
        if symbols is None:
            symbols = ["SPY", "QQQ", "IWM"]  # Major ETFs
            
        print("\n" + "=" * 80)
        print("HISTORICAL DATA COLLECTION")
        print("=" * 80)
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Days: {days_back}")
        print(f"Free only: {only_free}")
        print()
        
        # Check availability for first symbol
        print("Checking data availability...")
        availability = self.check_availability(symbols[0])
        
        print("\nAvailable datasets:")
        for name, info in availability.items():
            if info.get("available"):
                status = "FREE" if info.get("free") else f"${info.get('cost', 0):.2f}"
                print(f"  {name:15} | {info['days']:4} days | {status:8} | {info['dataset']}")
        
        # Collect data for each symbol
        print("\n" + "-" * 80)
        
        for symbol in symbols:
            print(f"\nProcessing {symbol}...")
            
            # Collect 1-minute bars (free)
            df = self.collect_data(
                symbol=symbol,
                dataset="XNAS.BASIC",
                schema="ohlcv-1m",
                days=min(days_back, 365),  # Max 1 year of free data
            )
            
            if not df.empty:
                # Calculate features
                features = self.calculate_features(df)
                
                # Store raw data
                self.store_to_database(df, "historical_bars", symbol)
                
                # Store features
                if not features.empty:
                    self.store_to_database(features, "historical_features", symbol)
            
            # Also collect daily bars for longer history
            df_daily = self.collect_data(
                symbol=symbol,
                dataset="XNAS.BASIC",
                schema="ohlcv-1d",
                days=365 * 5,  # 5 years of daily data
            )
            
            if not df_daily.empty:
                features_daily = self.calculate_features(df_daily)
                self.store_to_database(df_daily, "historical_bars_daily", symbol)
                if not features_daily.empty:
                    self.store_to_database(features_daily, "historical_features_daily", symbol)
        
        print("\n" + "=" * 80)
        print("COLLECTION COMPLETE")
        print("=" * 80)
        
        # Summary
        try:
            engine = create_engine(self.db_connection)
            with engine.connect() as conn:
                for table in ["historical_bars", "historical_features"]:
                    result = conn.execute(
                        text(f"SELECT COUNT(*) FROM ml.{table}")
                    ).scalar()
                    print(f"{table}: {result:,} total rows")
        except:
            print("Check database for stored data")


def main():
    """Main entry point."""
    collector = SimpleHistoricalCollector()
    
    # Run collection for major ETFs
    collector.run_collection(
        symbols=["SPY", "QQQ", "IWM", "DIA", "VTI"],
        days_back=30,  # Last 30 days
        only_free=True,
    )


if __name__ == "__main__":
    main()