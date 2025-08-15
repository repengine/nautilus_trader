#!/usr/bin/env python
"""
Automated Historical Data Collection and Feature Engineering System.

This module collects historical data up to license limits, calculates features,
registers them, and stores everything in the database.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import databento as db
from sqlalchemy import create_engine

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from nautilus_trader.adapters.databento import DatabentoDataLoader
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.core.datetime import dt_to_unix_nanos

from ml.features.engineering import FeatureEngineer
from ml.config.base import MLFeatureConfig
from ml.stores.feature_store import FeatureStore
from ml.stores.model_store import ModelStore
from ml.registry.feature_registry import FeatureRegistry, FeatureManifest


class HistoricalDataCollector:
    """
    Automated system for collecting historical data and engineering features.
    """
    
    def __init__(
        self,
        db_connection: str = None,
        databento_key: str = None,
    ):
        """
        Initialize the collector.
        
        Parameters
        ----------
        db_connection : str
            PostgreSQL connection string
        databento_key : str
            Databento API key
        """
        self.db_connection = db_connection or os.getenv(
            "DB_CONNECTION",
            "postgresql://postgres:postgres@localhost:5432/nautilus"
        )
        self.databento_key = databento_key or os.getenv("DATABENTO_API_KEY")
        
        if not self.databento_key:
            raise ValueError("DATABENTO_API_KEY not set")
        
        # Initialize clients
        self.historical_client = db.Historical(self.databento_key)
        self.loader = DatabentoDataLoader()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize stores and registries
        self._init_stores()
        
    def _init_stores(self):
        """Initialize feature store and registry."""
        # Feature configuration
        self.feature_config = MLFeatureConfig(
            lookback_window=20,
            indicators={
                "sma": {"period": 10},
                "rsi": {"period": 14},
                "bbands": {"period": 20, "std": 2},
                "ema": {"period": 12},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "volume_profile": {"bins": 10},
            },
            normalize_features=True,
        )
        
        # Initialize feature store
        self.feature_store = FeatureStore(
            connection_string=self.db_connection,
            feature_config=self.feature_config,
        )
        
        # Initialize feature registry
        self.feature_registry = FeatureRegistry(
            backend_type="postgres",
            connection_string=self.db_connection,
        )
        
        # Initialize feature engineer
        # Note: FeatureEngineer will be created by feature_store
        
        self.logger.info("Stores and registries initialized")
        
    def check_data_availability(self) -> Dict[str, Dict]:
        """
        Check available data based on subscription limits.
        
        Returns
        -------
        Dict[str, Dict]
            Available data configurations
        """
        availability = {}
        
        # Define what to check based on your subscription
        configs = [
            # L0 Core - Entire history of trades
            {
                "name": "trades_full",
                "dataset": "XNAS.BASIC",
                "schema": "trades",
                "days": 365 * 2,  # 2 years
                "description": "Full trade history"
            },
            # L0 - OHLCV bars (entire history)
            {
                "name": "bars_1m",
                "dataset": "XNAS.BASIC", 
                "schema": "ohlcv-1m",
                "days": 365,  # 1 year
                "description": "1-minute bars"
            },
            {
                "name": "bars_1h",
                "dataset": "XNAS.BASIC",
                "schema": "ohlcv-1h",
                "days": 365 * 2,  # 2 years
                "description": "Hourly bars"
            },
            {
                "name": "bars_1d",
                "dataset": "XNAS.BASIC",
                "schema": "ohlcv-1d",
                "days": 365 * 5,  # 5 years
                "description": "Daily bars"
            },
            # L1 - Top of book (12 months)
            {
                "name": "quotes",
                "dataset": "EQUS.MINI",
                "schema": "tbbo",
                "days": 365,  # 12 months
                "description": "Best bid/offer"
            },
            # L2 - Market depth (1 month)
            {
                "name": "depth",
                "dataset": "EQUS.MINI",
                "schema": "mbp-1",
                "days": 30,  # 1 month
                "description": "Market depth"
            },
        ]
        
        for config in configs:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=config["days"])
                
                cost = self.historical_client.metadata.get_cost(
                    dataset=config["dataset"],
                    symbols=["SPY"],
                    schema=config["schema"],
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                )
                
                availability[config["name"]] = {
                    **config,
                    "available": True,
                    "cost": cost,
                    "start_date": start_date,
                    "end_date": end_date,
                }
                
                self.logger.info(
                    f"✓ {config['description']:25} | "
                    f"{config['days']:4} days | "
                    f"Cost: ${cost:.2f}"
                )
                
            except Exception as e:
                availability[config["name"]] = {
                    **config,
                    "available": False,
                    "error": str(e),
                }
                self.logger.warning(f"✗ {config['description']:25} | Not available")
                
        return availability
    
    def collect_historical_data(
        self,
        symbols: List[str],
        dataset: str = "XNAS.BASIC",
        schema: str = "ohlcv-1m",
        days_back: int = 30,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Collect historical data for specified symbols.
        
        Parameters
        ----------
        symbols : List[str]
            List of symbols to collect
        dataset : str
            Databento dataset
        schema : str
            Data schema (ohlcv-1m, trades, etc.)
        days_back : int
            Number of days to collect
        save_path : str, optional
            Path to save the data
            
        Returns
        -------
        pd.DataFrame
            Collected data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        self.logger.info(
            f"Collecting {schema} data for {symbols} "
            f"from {start_date.date()} to {end_date.date()}"
        )
        
        try:
            # Download data
            data = self.historical_client.timeseries.get_range(
                dataset=dataset,
                symbols=symbols,
                schema=schema,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )
            
            # Convert to DataFrame
            df = data.to_df()
            
            self.logger.info(f"Collected {len(df)} records")
            
            # Save if requested
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(save_path)
                self.logger.info(f"Saved to {save_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to collect data: {e}")
            raise
    
    def calculate_and_store_features(
        self,
        df: pd.DataFrame,
        instrument_id: str,
        is_live: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate features from data and store them.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data (OHLCV or trades)
        instrument_id : str
            Instrument identifier
        is_live : bool
            Whether this is live data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with calculated features
        """
        self.logger.info(f"Calculating features for {instrument_id}")
        
        # Ensure proper column names
        if "open" in df.columns:
            # OHLCV data
            feature_df = self.feature_store.feature_engineer.compute_features_batch(
                bars_df=df,
                instrument_id=instrument_id,
            )
        else:
            # Convert trades to OHLCV if needed
            feature_df = self._trades_to_features(df, instrument_id)
        
        # Store features
        self.logger.info(f"Storing {len(feature_df)} feature rows")
        
        for _, row in feature_df.iterrows():
            self.feature_store.store_features(
                instrument_id=instrument_id,
                features=row.to_dict(),
                ts_event=int(row.name.timestamp() * 1e9) if hasattr(row.name, 'timestamp') else row.name,
                is_live=is_live,
            )
        
        # Register feature manifest
        self._register_features(instrument_id, feature_df)
        
        return feature_df
    
    def _trades_to_features(self, trades_df: pd.DataFrame, instrument_id: str) -> pd.DataFrame:
        """
        Convert trades to OHLCV and calculate features.
        """
        # Aggregate trades to 1-minute bars
        ohlcv = trades_df.resample("1min").agg({
            "price": ["first", "max", "min", "last"],
            "size": "sum",
        })
        
        ohlcv.columns = ["open", "high", "low", "close", "volume"]
        ohlcv = ohlcv.dropna()
        
        # Calculate features
        return self.feature_store.feature_engineer.compute_features_batch(
            bars_df=ohlcv,
            instrument_id=instrument_id,
        )
    
    def _register_features(self, instrument_id: str, feature_df: pd.DataFrame):
        """
        Register features in the feature registry.
        """
        manifest = FeatureManifest(
            feature_id=f"{instrument_id}_historical_{datetime.now().strftime('%Y%m%d')}",
            feature_names=list(feature_df.columns),
            schema_version="1.0.0",
            description=f"Historical features for {instrument_id}",
            tags=["historical", instrument_id],
            metadata={
                "instrument_id": instrument_id,
                "date_range": {
                    "start": str(feature_df.index[0]),
                    "end": str(feature_df.index[-1]),
                },
                "row_count": len(feature_df),
                "config": str(self.feature_config),
            }
        )
        
        self.feature_registry.register(manifest)
        self.logger.info(f"Registered feature manifest: {manifest.feature_id}")
    
    def run_full_collection(
        self,
        symbols: List[str] = None,
        max_cost: float = 0.0,  # Only free data by default
    ):
        """
        Run full historical data collection pipeline.
        
        Parameters
        ----------
        symbols : List[str]
            Symbols to collect (default: major ETFs)
        max_cost : float
            Maximum cost allowed (0 = free only)
        """
        if symbols is None:
            symbols = ["SPY", "QQQ", "IWM", "DIA", "VTI"]  # Major ETFs
        
        self.logger.info("=" * 80)
        self.logger.info("STARTING FULL HISTORICAL DATA COLLECTION")
        self.logger.info("=" * 80)
        
        # Check availability
        availability = self.check_data_availability()
        
        # Collect data for each configuration
        results = {}
        
        for config_name, config in availability.items():
            if not config["available"]:
                continue
                
            if config["cost"] > max_cost:
                self.logger.info(f"Skipping {config_name} (cost ${config['cost']:.2f} > ${max_cost})")
                continue
            
            try:
                # Collect data
                for symbol in symbols:
                    self.logger.info(f"\nProcessing {symbol} - {config['description']}")
                    
                    # Collect data
                    df = self.collect_historical_data(
                        symbols=[symbol],
                        dataset=config["dataset"],
                        schema=config["schema"],
                        days_back=config["days"],
                        save_path=f"data/historical/{symbol}_{config['schema']}_{config['days']}d.parquet",
                    )
                    
                    # Calculate and store features (only for OHLCV data)
                    if "ohlcv" in config["schema"]:
                        feature_df = self.calculate_and_store_features(
                            df=df,
                            instrument_id=f"{symbol}.XNAS",
                            is_live=False,
                        )
                        
                        results[f"{symbol}_{config_name}"] = {
                            "rows": len(df),
                            "features": len(feature_df) if "ohlcv" in config["schema"] else 0,
                            "cost": config["cost"],
                        }
                        
            except Exception as e:
                self.logger.error(f"Failed to process {config_name}: {e}")
                continue
        
        # Print summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COLLECTION SUMMARY")
        self.logger.info("=" * 80)
        
        total_rows = sum(r["rows"] for r in results.values())
        total_features = sum(r["features"] for r in results.values())
        total_cost = sum(r["cost"] for r in results.values())
        
        for name, result in results.items():
            self.logger.info(
                f"{name:30} | Rows: {result['rows']:8,} | "
                f"Features: {result['features']:8,} | Cost: ${result['cost']:.2f}"
            )
        
        self.logger.info("-" * 80)
        self.logger.info(
            f"{'TOTAL':30} | Rows: {total_rows:8,} | "
            f"Features: {total_features:8,} | Cost: ${total_cost:.2f}"
        )
        
        return results


def main():
    """
    Main entry point for historical data collection.
    """
    collector = HistoricalDataCollector()
    
    # Run full collection (free data only)
    results = collector.run_full_collection(
        symbols=["SPY", "QQQ", "IWM"],  # Top 3 ETFs
        max_cost=0.0,  # Free data only
    )
    
    print("\nData collection complete!")
    print(f"Check PostgreSQL for stored features:")
    print("  psql -U postgres -d nautilus")
    print("  SELECT COUNT(*) FROM ml.ml_features;")


if __name__ == "__main__":
    main()