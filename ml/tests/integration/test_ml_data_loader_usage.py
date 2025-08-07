# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Real-world usage example of MLDataLoader for ML training workflows.

This example demonstrates practical usage patterns for loading and preparing data for
machine learning models.

"""

import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

from ml._imports import HAS_POLARS
from ml._imports import pl
from ml.data.loader import MLDataLoader
from ml.data.loader import load_ml_data
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarSpecification
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AggregationSource
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog


def example_ml_training_workflow():
    """
    Example ML training workflow using MLDataLoader.

    This demonstrates:
    1. Loading historical data from ParquetDataCatalog
    2. Preparing features for ML model training
    3. Efficient data handling with caching
    4. Multiple instrument loading

    """
    if not HAS_POLARS:
        print("Polars is required for ML data loading. Install with: pip install polars")
        return

    # Setup temporary catalog for demo
    with tempfile.TemporaryDirectory(prefix="ml_demo_") as temp_dir:
        catalog = ParquetDataCatalog(temp_dir)

        # Generate and write sample data
        print("Generating sample data...")
        instruments = ["EURUSD.SIM", "GBPUSD.SIM", "USDJPY.SIM"]

        for instrument_str in instruments:
            instrument_id = InstrumentId.from_str(instrument_str)
            bars = generate_sample_bars(instrument_id, days=30)
            catalog.write_data(bars)
            print(f"  - Generated {len(bars)} bars for {instrument_str}")

        # Initialize MLDataLoader
        print("\nInitializing MLDataLoader...")
        loader = MLDataLoader(catalog, cache_size=100, enable_cache=True)

        # Example 1: Load single instrument with date range
        print("\n1. Loading single instrument with date range:")
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 7)
        eurusd_df = loader.load_bars("EURUSD.SIM", start=start, end=end)
        print(f"   - Loaded {eurusd_df.shape[0]} bars for EURUSD")
        print(f"   - Columns: {eurusd_df.columns}")
        print(f"   - Date range: {eurusd_df['timestamp'].min()} to {eurusd_df['timestamp'].max()}")

        # Example 2: Feature engineering with Polars
        print("\n2. Feature engineering example:")
        features_df = engineer_features(eurusd_df)
        print(f"   - Generated {len(features_df.columns)} features")
        print(
            f"   - New features: {[col for col in features_df.columns if col not in eurusd_df.columns]}",
        )

        # Example 3: Load multiple instruments for portfolio analysis
        print("\n3. Loading multiple instruments:")
        portfolio_data = loader.load_multiple(
            instruments,
            data_type="bars",
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 10),
        )
        print(f"   - Loaded data for {len(portfolio_data)} instruments")
        for inst, df in portfolio_data.items():
            print(f"   - {inst}: {df.shape[0]} bars")

        # Example 4: Cache performance demonstration
        print("\n4. Cache performance test:")
        import time

        # First load (uncached)
        start_time = time.time()
        df1 = loader.load_bars("EURUSD.SIM")
        first_load = time.time() - start_time

        # Second load (cached)
        start_time = time.time()
        df2 = loader.load_bars("EURUSD.SIM")
        cached_load = time.time() - start_time

        print(f"   - First load: {first_load:.4f}s")
        print(f"   - Cached load: {cached_load:.4f}s")
        print(f"   - Speedup: {first_load/cached_load:.1f}x")

        # Example 5: Memory management
        print("\n5. Cache management:")
        stats = loader.get_cache_stats()
        print(f"   - Cache size: {stats['size']} / {stats['max_size']}")
        print(f"   - Cache enabled: {stats['enabled']}")

        # Clear cache when done
        loader.clear_cache()
        print("   - Cache cleared")

        # Example 6: Using convenience function
        print("\n6. Using convenience function:")
        all_data = load_ml_data(
            instrument_ids=instruments,
            catalog=catalog,
            data_type="bars",
            start="2023-01-01",
            end="2023-01-05",
        )
        print(f"   - Loaded {len(all_data)} instruments via convenience function")

        # Example 7: Prepare data for ML model
        print("\n7. Preparing data for ML training:")
        ml_ready_df = prepare_ml_dataset(portfolio_data)
        print(f"   - Final dataset shape: {ml_ready_df.shape}")
        print(f"   - Features ready for training: {ml_ready_df.columns[:5]}...")

        return ml_ready_df


def generate_sample_bars(instrument_id: InstrumentId, days: int = 30) -> list[Bar]:
    """
    Generate realistic sample bar data.
    """
    bars = []
    base_time = pd.Timestamp("2023-01-01", tz="UTC").value
    base_price = (
        1.1000 if "EUR" in str(instrument_id) else 1.3000 if "GBP" in str(instrument_id) else 110.00
    )

    # Generate 1-minute bars
    bars_per_day = 1440  # 24 hours * 60 minutes
    total_bars = days * bars_per_day

    for i in range(total_bars):
        # Add some realistic price movement
        random_walk = np.random.randn() * 0.0001
        base_price += random_walk

        bar = Bar(
            bar_type=BarType(
                instrument_id=instrument_id,
                bar_spec=BarSpecification(
                    step=1,
                    aggregation=BarAggregation.MINUTE,
                    price_type=PriceType.LAST,
                ),
                aggregation_source=AggregationSource.EXTERNAL,
            ),
            open=Price.from_str(f"{base_price:.5f}"),
            high=Price.from_str(f"{base_price + abs(random_walk):.5f}"),
            low=Price.from_str(f"{base_price - abs(random_walk):.5f}"),
            close=Price.from_str(f"{base_price:.5f}"),
            volume=Quantity.from_int(np.random.randint(100, 10000)),
            ts_event=base_time + i * 60_000_000_000,  # 1 minute in nanoseconds
            ts_init=base_time + i * 60_000_000_000,
        )
        bars.append(bar)

    return bars


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Engineer features for ML training using Polars.

    This demonstrates how to create technical indicators and features efficiently using
    Polars expressions.

    """
    return df.with_columns(
        [
            # Price-based features
            (pl.col("close") - pl.col("open")).alias("body"),
            (pl.col("high") - pl.col("low")).alias("range"),
            ((pl.col("close") - pl.col("open")) / pl.col("open") * 100).alias("return_pct"),
            # Moving averages
            pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
            pl.col("close").rolling_mean(window_size=50).alias("sma_50"),
            # Volatility
            pl.col("close").rolling_std(window_size=20).alias("volatility_20"),
            # Volume features
            pl.col("volume").rolling_mean(window_size=20).alias("volume_ma_20"),
            (pl.col("volume") / pl.col("volume").rolling_mean(window_size=20)).alias(
                "volume_ratio",
            ),
            # Lag features
            pl.col("close").shift(1).alias("close_lag_1"),
            pl.col("close").shift(5).alias("close_lag_5"),
        ],
    )


def prepare_ml_dataset(portfolio_data: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Prepare a combined dataset for ML training from multiple instruments.

    This demonstrates how to combine data from multiple instruments into a single
    dataset suitable for training.

    """
    prepared_dfs = []

    for instrument, df in portfolio_data.items():
        # Add instrument identifier
        df = df.with_columns(pl.lit(instrument).alias("instrument"))

        # Engineer features
        df = engineer_features(df)

        # Add to list
        prepared_dfs.append(df)

    # Combine all DataFrames
    combined_df = pl.concat(prepared_dfs)

    # Sort by timestamp
    combined_df = combined_df.sort("timestamp")

    # Remove any rows with null values (from rolling calculations)
    combined_df = combined_df.drop_nulls()

    return combined_df


if __name__ == "__main__":
    print("=" * 60)
    print("MLDataLoader Usage Example")
    print("=" * 60)

    # Run the example workflow
    result_df = example_ml_training_workflow()

    if result_df is not None:
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)
        print("\nThe MLDataLoader is ready for production use in ML workflows.")
        print("It provides:")
        print("  ✓ Efficient data loading from ParquetDataCatalog")
        print("  ✓ Built-in caching for performance")
        print("  ✓ Polars DataFrames for fast feature engineering")
        print("  ✓ Support for multiple instruments and data types")
        print("  ✓ Memory-efficient vectorized operations")
        print("  ✓ Date range filtering capabilities")
