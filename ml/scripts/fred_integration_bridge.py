#!/usr/bin/env python3
"""
Integration bridge between simple FRED updater and ML pipeline.

This script converts the simple_fred_updater.py output into the format 
expected by the ML pipeline's FRED loader and DataStore system.
"""
import os
import warnings
from datetime import datetime

import pandas as pd
import polars as pl


warnings.filterwarnings("ignore")

def convert_simple_to_ml_format():
    """Convert simple FRED updater format to ML pipeline format."""
    print("🔗 FRED Integration Bridge")
    print("=" * 40)

    # Check if updated data exists
    updated_file = "data/fred/fred_indicators_updated.parquet"
    if not os.path.exists(updated_file):
        print(f"❌ Updated FRED data not found: {updated_file}")
        print("Run: python simple_fred_updater.py first")
        return False

    # Load updated data
    df = pd.read_parquet(updated_file)
    print(f"📊 Loaded {len(df)} rows from updated FRED data")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Indicators: {[col for col in df.columns if col not in ['date', 'timestamp_ns']]}")

    # Convert to ML pipeline format (long format)
    ml_data = []

    for _, row in df.iterrows():
        timestamp = row["date"]
        timestamp_ns = row["timestamp_ns"]

        # Add each indicator as a separate row
        for col in df.columns:
            if col in ["date", "timestamp_ns"]:
                continue

            value = row[col]
            if pd.notna(value):  # Only include non-null values
                ml_data.append({
                    "timestamp": timestamp,
                    "timestamp_ns": timestamp_ns,
                    "series_id": col,
                    "value": float(value)
                })

    # Create ML-format DataFrame
    ml_df = pd.DataFrame(ml_data)
    print(f"✅ Converted to ML format: {len(ml_df)} indicator-date pairs")

    # Convert to Polars for ML pipeline compatibility
    pl_df = pl.from_pandas(ml_df)

    # Save in ML-compatible format
    ml_output_file = "data/fred/fred_indicators_ml_format.parquet"
    pl_df.write_parquet(ml_output_file)
    print(f"💾 Saved ML-compatible format: {ml_output_file}")

    # Update the original fred_indicators.parquet with recent data
    original_file = "data/fred/fred_indicators.parquet"
    if os.path.exists(original_file):
        try:
            # Load original data
            original_df = pd.read_parquet(original_file)
            print(f"📚 Original data: {len(original_df)} rows")

            # Find cutoff date (where to merge new data)
            if "timestamp" in original_df.columns:
                original_df["date"] = pd.to_datetime(original_df["timestamp"])
            elif "date" not in original_df.columns and "timestamp" not in original_df.columns:
                # Handle different date column names
                date_cols = [col for col in original_df.columns if "date" in col.lower() or "time" in col.lower()]
                if date_cols:
                    original_df["date"] = pd.to_datetime(original_df[date_cols[0]])

            if "date" in original_df.columns:
                latest_original_date = original_df["date"].max()
                new_data_start = df["date"].min()

                print(f"   Latest original date: {latest_original_date}")
                print(f"   New data starts: {new_data_start}")

                # Merge if there's a reasonable gap (less than 30 days)
                gap_days = (new_data_start - latest_original_date).days
                if 0 < gap_days < 30:
                    # Simple append approach - convert new data to same format as original
                    if "series_id" in original_df.columns:
                        # Original is in long format, append our ML format
                        combined_df = pd.concat([original_df, ml_df.drop("timestamp_ns", axis=1, errors="ignore")], ignore_index=True)
                    else:
                        # Original is in wide format, append our wide format
                        original_wide = original_df.copy()
                        new_wide = df.drop("timestamp_ns", axis=1, errors="ignore")

                        # Align columns
                        all_cols = set(original_wide.columns) | set(new_wide.columns)
                        for col in all_cols:
                            if col not in original_wide.columns:
                                original_wide[col] = None
                            if col not in new_wide.columns:
                                new_wide[col] = None

                        combined_df = pd.concat([original_wide, new_wide], ignore_index=True)

                    # Save merged data
                    backup_file = f"{original_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.rename(original_file, backup_file)
                    print(f"📂 Backed up original to: {backup_file}")

                    combined_df.to_parquet(original_file)
                    print(f"🔄 Updated original file with {len(combined_df) - len(original_df)} new rows")
                else:
                    print(f"⚠️  Gap of {gap_days} days - not merging automatically")
            else:
                print("⚠️  Could not determine date format in original data")

        except Exception as e:
            print(f"⚠️  Could not merge with original data: {e}")

    # Create feature summary
    print("\n📈 FRED Feature Summary:")
    print("-" * 30)

    # Group by series and show latest values
    latest_values = ml_df.loc[ml_df.groupby("series_id")["timestamp"].idxmax()]

    # Map series to friendly names
    series_names = {
        "VIXCLS": "VIX Volatility Index",
        "DGS1": "1-Year Treasury Rate",
        "DGS2": "2-Year Treasury Rate",
        "DGS10": "10-Year Treasury Rate",
        "DGS30": "30-Year Treasury Rate",
        "SOFR": "SOFR Rate",
        "DTWEXBGS": "Dollar Index (DXY)",
        "DEXUSEU": "USD/EUR Exchange Rate",
        "BAMLH0A0HYM2": "High Yield Credit Spread",
        "BAMLC0A0CM": "Investment Grade Spread",
        "MORTGAGE30US": "30-Year Mortgage Rate"
    }

    for _, row in latest_values.iterrows():
        name = series_names.get(row["series_id"], row["series_id"])
        print(f"  {name}: {row['value']:.2f}")

    print("\n✅ Integration complete!")
    print(f"   ML format file: {ml_output_file}")
    print(f"   {len(latest_values)} indicators ready for ML pipeline")

    return True

def test_ml_integration():
    """Test integration with ML pipeline components."""
    print("\n🧪 Testing ML Integration")
    print("=" * 30)

    try:
        # Test importing ML components
        from ml.data.loaders.fred_loader import FREDConfig
        from ml.data.loaders.fred_loader import FREDDataLoader
        from ml.registry.data_registry import DataRegistry
        from ml.stores.data_store import DataStore

        print("✅ ML imports successful")

        # Test loading our converted data
        ml_file = "data/fred/fred_indicators_ml_format.parquet"
        if os.path.exists(ml_file):
            df = pl.read_parquet(ml_file)
            print(f"✅ ML format data loads: {len(df)} rows, {len(df['series_id'].unique())} series")

        print("✅ Integration test passed")

    except ImportError as e:
        print(f"⚠️  ML imports failed: {e}")
        print("   This is expected if ML dependencies aren't fully installed")
        return False
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

    return True

def create_incremental_update_schedule():
    """Create a simple cron-style scheduler for incremental updates."""
    print("\n⏰ Setting up Incremental Updates")
    print("=" * 35)

    schedule_script = """#!/bin/bash
# FRED Data Incremental Update Schedule
# Add to crontab with: crontab -e
# Then add: 0 9,15 * * 1-5 /path/to/this/script

cd /home/nate/projects/nautilus_trader
source .venv/bin/activate

echo "$(date): Starting FRED data update"

# Update FRED data
python simple_fred_updater.py

# Integrate with ML pipeline
python fred_integration_bridge.py

echo "$(date): FRED data update complete"
"""

    script_path = "scripts/update_fred_data.sh"
    os.makedirs("scripts", exist_ok=True)

    with open(script_path, "w") as f:
        f.write(schedule_script)

    # Make executable
    os.chmod(script_path, 0o755)

    print(f"📝 Created update script: {script_path}")
    print("   To schedule automatic updates:")
    print("   1. crontab -e")
    print(f"   2. Add: 0 9,15 * * 1-5 {os.path.abspath(script_path)}")
    print("   (Updates twice daily on weekdays)")

    return script_path

def main():
    """Main integration function."""
    print("🏦 FRED-ML Integration Bridge")
    print("=" * 50)

    success = convert_simple_to_ml_format()
    if not success:
        return

    test_ml_integration()
    create_incremental_update_schedule()

    print("\n" + "=" * 50)
    print("🎉 FRED Integration Complete!")
    print("\nNext steps:")
    print("1. Use ML-format data in feature engineering:")
    print("   python -c \"import polars as pl; df=pl.read_parquet('data/fred/fred_indicators_ml_format.parquet'); print(df.head())\"")
    print("2. Integrate with TFT training pipeline")
    print("3. Set up automatic daily updates with cron")
    print("4. Add FRED features to trading strategies")

if __name__ == "__main__":
    main()
