#!/usr/bin/env python3
"""
Comprehensive Databento data downloader that maximizes subscription value.

Downloads everything you're paying for:
- Core Schema (OHLCV): 7 years
- L1 (BBO/Trades): 1 year  
- L2 (MBP-10): 30 days
- L3 (MBP-1): 30 days (if needed)

NO DATA LOSS - Incremental updates preserve all historical data.
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import databento as db
import pandas as pd

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_subscription_date_ranges() -> dict[str, tuple[datetime, datetime]]:
    """Get optimal date ranges for each data type based on subscription."""
    today = datetime.now().date()
    yesterday = datetime.combine(today - timedelta(days=1), datetime.min.time())
    
    # Ensure timezone-naive datetimes for consistent comparison
    yesterday = yesterday.replace(tzinfo=None)
    
    return {
        "core": (
            yesterday - timedelta(days=365 * 7),  # 7 years
            yesterday
        ),
        "l1": (
            yesterday - timedelta(days=365),       # 1 year
            yesterday
        ),
        "l2": (
            yesterday - timedelta(days=30),        # 30 days
            yesterday
        ),
        "l3": (
            yesterday - timedelta(days=30),        # 30 days  
            yesterday
        )
    }


def get_existing_date_range(file_path: Path) -> Optional[tuple[datetime, datetime]]:
    """Get date range from existing parquet file, handling both formats."""
    if not file_path.exists():
        return None
        
    try:
        # First try reading with pandas to handle both new and old formats
        df = pd.read_parquet(file_path)
        if len(df) == 0:
            return None
            
        # Handle different timestamp formats
        if 'ts_event' in df.columns:
            # New format: ts_event is a column (nanoseconds since epoch)
            min_ts = df['ts_event'].min()
            max_ts = df['ts_event'].max()
            min_date = pd.to_datetime(min_ts, unit='ns')
            max_date = pd.to_datetime(max_ts, unit='ns')
        elif df.index.name == 'ts_event' or 'ts_event' in str(df.index.name):
            # Old format: ts_event is the index (datetime)
            min_date = df.index.min()
            max_date = df.index.max()
            # Convert to pandas datetime if needed
            if not isinstance(min_date, pd.Timestamp):
                min_date = pd.to_datetime(min_date)
                max_date = pd.to_datetime(max_date)
        else:
            logger.warning(f"No ts_event found in {file_path}")
            return None
        
        # Ensure timezone-naive datetimes for consistent comparison
        if hasattr(min_date, 'tz_localize'):
            min_date = min_date.tz_localize(None) if min_date.tz is not None else min_date
            max_date = max_date.tz_localize(None) if max_date.tz is not None else max_date
        
        # Convert to python datetime
        if hasattr(min_date, 'to_pydatetime'):
            min_date = min_date.to_pydatetime()
            max_date = max_date.to_pydatetime()
        
        # Remove timezone info if present
        if min_date.tzinfo is not None:
            min_date = min_date.replace(tzinfo=None)
            max_date = max_date.replace(tzinfo=None)
            
        return min_date, max_date
        
    except Exception as e:
        logger.warning(f"Failed to read existing file {file_path}: {e}")
        return None


def calculate_missing_ranges(
    target_range: tuple[datetime, datetime],
    existing_range: Optional[tuple[datetime, datetime]]
) -> list[tuple[datetime, datetime]]:
    """Calculate date ranges that need to be downloaded."""
    target_start, target_end = target_range
    
    if existing_range is None:
        # No existing data - download everything
        return [(target_start, target_end)]
    
    existing_start, existing_end = existing_range
    missing_ranges = []
    
    # Check for gap at the beginning
    if target_start < existing_start:
        missing_ranges.append((target_start, existing_start - timedelta(days=1)))
    
    # Check for gap at the end  
    if existing_end < target_end:
        missing_ranges.append((existing_end + timedelta(days=1), target_end))
    
    return missing_ranges


def download_data_range(
    client: db.Historical,
    symbol: str,
    schema: str,
    dataset: str,
    start_date: datetime,
    end_date: datetime,
    output_file: Path
) -> bool:
    """Download data for a date range with intelligent batching."""
    
    logger.info(f"Downloading {schema} for {symbol}: {start_date.date()} to {end_date.date()}")
    
    # Calculate optimal batch size based on schema type and date range
    total_days = (end_date - start_date).days
    
    # Batch size heuristics to stay under 1GB per request
    if schema.startswith("ohlcv"):
        batch_days = min(365, total_days)  # 1 year max for OHLCV
    elif schema in ["tbbo", "trades"]:
        batch_days = min(90, total_days)   # 3 months max for L1 data
    elif schema.startswith("mbp"):
        batch_days = min(30, total_days)   # 1 month max for L2 data
    else:
        batch_days = min(180, total_days)  # 6 months default
    
    if total_days <= batch_days:
        # Single request
        return _download_single_batch(client, symbol, schema, dataset, start_date, end_date, output_file)
    else:
        # Multiple batches
        logger.info(f"Large request - splitting into {batch_days}-day batches")
        return _download_multiple_batches(client, symbol, schema, dataset, start_date, end_date, output_file, batch_days)


def _download_single_batch(
    client: db.Historical,
    symbol: str,
    schema: str,
    dataset: str,
    start_date: datetime,
    end_date: datetime,
    output_file: Path
) -> bool:
    """Download a single batch of data."""
    try:
        df = client.timeseries.get_range(
            dataset=dataset,
            symbols=[symbol],
            schema=schema,
            start=start_date,
            end=end_date,
        ).to_df()
        
        if df.empty:
            logger.warning(f"No data returned for {symbol} {schema}")
            return False
        
        # Save to temporary file first
        temp_file = output_file.with_suffix('.tmp')
        df.to_parquet(temp_file)
        
        # Atomic move
        temp_file.rename(output_file)
        
        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {len(df):,} records ({size_mb:.1f} MB) to {output_file.name}")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        
        # Handle known acceptable errors
        if any(code in error_msg for code in ("403", "license", "unauthorized")):
            logger.warning(f"License restriction for {symbol} {schema}: {error_msg}")
            return True  # Not an error - just not licensed
        
        logger.error(f"Failed to download {symbol} {schema}: {error_msg}")
        return False


def _download_multiple_batches(
    client: db.Historical,
    symbol: str,
    schema: str,
    dataset: str,
    start_date: datetime,
    end_date: datetime,
    output_file: Path,
    batch_days: int
) -> bool:
    """Download data in multiple batches and combine."""
    
    batch_files = []
    current_start = start_date
    batch_num = 0
    
    while current_start < end_date:
        batch_end = min(current_start + timedelta(days=batch_days), end_date)
        batch_file = output_file.parent / f"{output_file.stem}_batch_{batch_num}.parquet"
        
        logger.info(f"  Batch {batch_num + 1}: {current_start.date()} to {batch_end.date()}")
        
        if _download_single_batch(client, symbol, schema, dataset, current_start, batch_end, batch_file):
            batch_files.append(batch_file)
        else:
            logger.warning(f"  Batch {batch_num + 1} failed - continuing with remaining batches")
        
        current_start = batch_end + timedelta(days=1)
        batch_num += 1
    
    if not batch_files:
        logger.error("All batches failed")
        return False
    
    # Combine all batch files
    logger.info(f"Combining {len(batch_files)} batches...")
    
    try:
        # Load and combine all batches
        dfs = []
        total_records = 0
        
        for batch_file in batch_files:
            if HAS_POLARS:
                df = pl.read_parquet(batch_file)
            else:
                df = pd.read_parquet(batch_file)
            dfs.append(df)
            total_records += len(df)
            logger.info(f"  Loaded batch: {len(df):,} records from {batch_file.name}")
        
        # Concatenate and sort
        if HAS_POLARS:
            combined_df = pl.concat(dfs)
            final_df = combined_df.sort("ts_event").unique(subset=["symbol", "ts_event"], keep="last")
        else:
            combined_df = pd.concat(dfs, ignore_index=True)
            final_df = combined_df.sort_values("ts_event").drop_duplicates(subset=["symbol", "ts_event"], keep="last")
        
        # Save final result
        temp_file = output_file.with_suffix('.tmp')
        if HAS_POLARS:
            final_df.write_parquet(temp_file)
        else:
            final_df.to_parquet(temp_file)
        temp_file.rename(output_file)
        
        # Clean up batch files
        for batch_file in batch_files:
            batch_file.unlink()
        
        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"Combined result: {len(final_df):,} records ({size_mb:.1f} MB)")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to combine batches: {e}")
        
        # Clean up batch files on failure
        for batch_file in batch_files:
            if batch_file.exists():
                batch_file.unlink()
        
        return False


def merge_data_files(base_file: Path, new_files: list[Path], output_file: Path) -> bool:
    """Merge existing and new data files, handling overlaps."""
    
    try:
        dfs = []
        
        # Load base file if it exists
        if base_file.exists():
            logger.info(f"Loading existing data from {base_file.name}")
            if HAS_POLARS:
                base_df = pl.read_parquet(base_file)
            else:
                base_df = pd.read_parquet(base_file)
            dfs.append(base_df)
        
        # Load new data files
        for new_file in new_files:
            if new_file.exists():
                logger.info(f"Loading new data from {new_file.name}")
                if HAS_POLARS:
                    new_df = pl.read_parquet(new_file)
                else:
                    new_df = pd.read_parquet(new_file)
                dfs.append(new_df)
        
        if not dfs:
            logger.warning("No data files to merge")
            return False
        
        # Concatenate and deduplicate
        logger.info("Merging and deduplicating data...")
        if HAS_POLARS:
            combined_df = pl.concat(dfs)
            
            # Sort by timestamp and remove duplicates
            final_df = (combined_df
                       .sort("ts_event")
                       .unique(subset=["symbol", "ts_event"], keep="last"))
        else:
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Sort by timestamp and remove duplicates
            final_df = combined_df.sort_values("ts_event").drop_duplicates(subset=["symbol", "ts_event"], keep="last")
        
        # Save merged result
        temp_file = output_file.with_suffix('.tmp')
        if HAS_POLARS:
            final_df.write_parquet(temp_file)
        else:
            final_df.to_parquet(temp_file)
        temp_file.rename(output_file)
        
        # Clean up temporary files
        for temp_file in new_files:
            if temp_file.exists():
                temp_file.unlink()
        
        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"Merged result: {len(final_df):,} records ({size_mb:.1f} MB)")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to merge data files: {e}")
        return False


def process_symbol_schema(
    client: db.Historical,
    symbol: str,
    schema: str,
    dataset: str,
    target_range: tuple[datetime, datetime],
    output_dir: Path
) -> bool:
    """Process a single symbol-schema combination with incremental updates."""
    
    # Determine output file path
    if schema.startswith("ohlcv"):
        schema_short = "bars"
    elif schema == "tbbo":
        schema_short = "bbo"
    elif schema.startswith("mbp"):
        depth = schema.split("-")[1] if "-" in schema else "1"
        schema_short = f"mbp-{depth}"
    else:
        schema_short = schema
    
    output_file = output_dir / f"{symbol}_{schema_short}.parquet"
    
    # Check existing data
    existing_range = get_existing_date_range(output_file)
    
    if existing_range:
        logger.info(f"Existing data: {existing_range[0].date()} to {existing_range[1].date()}")
    else:
        logger.info("No existing data found")
    
    # Calculate missing ranges
    missing_ranges = calculate_missing_ranges(target_range, existing_range)
    
    if not missing_ranges:
        logger.info("✅ Data is already complete - no download needed")
        return True
    
    logger.info(f"Need to download {len(missing_ranges)} missing range(s)")
    
    # Download missing data
    temp_files = []
    success_count = 0
    
    for i, (start, end) in enumerate(missing_ranges):
        temp_file = output_dir / f"{symbol}_{schema_short}_temp_{i}.parquet"
        
        if download_data_range(client, symbol, schema, dataset, start, end, temp_file):
            temp_files.append(temp_file)
            success_count += 1
    
    if success_count == 0:
        logger.warning("No new data downloaded")
        return False
    
    # Merge with existing data
    if temp_files:
        return merge_data_files(output_file, temp_files, output_file)
    
    return True


def get_tier1_symbols() -> list[str]:
    """Get comprehensive list of Tier 1 symbols."""
    return [
        # Major ETFs
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VEA", "VWO", "EFA", "EEM",
        
        # Sector ETFs
        "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC",
        
        # Bond ETFs
        "TLT", "IEF", "SHY", "AGG", "LQD", "HYG", "TIP", "EMB",
        
        # Commodity ETFs
        "GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC", "VNQ",
        
        # Currency ETFs
        "UUP", "FXE", "FXY", "FXB", "FXC", "FXA", "FXF",
        
        # Volatility ETFs  
        "VXX", "VIXY", "VXZ", "SVXY", "UVXY",
        
        # Top Individual Stocks
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK.B",
        "AMD", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "BAC",
        "ADBE", "CRM", "NFLX", "KO", "PEP", "TMO", "ABBV", "CVX", "WMT",
        "MRK", "LLY", "AVGO", "NKE", "ORCL", "ACN", "COST", "MCD", "ABT",
        "TXN", "GS", "MS", "WFC", "C", "XOM", "COP", "CAT", "BA", "GE",
        "MMM", "VZ", "T", 
        
        # Newer high-interest stocks
        "PLTR", "SOFI", "RIVN", "LCID", "COIN", "MSTR"
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Databento downloader - gets everything you're paying for!"
    )
    
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to download")
    parser.add_argument("--tier", type=int, choices=[1], help="Use Tier 1 symbol list")
    parser.add_argument("--data-dir", type=Path, default=Path("data/comprehensive"), 
                       help="Output data directory")
    parser.add_argument("--schemas", nargs="+", 
                       choices=["core", "l1", "l2", "l3", "all"],
                       default=["all"],
                       help="Which data types to download")
    parser.add_argument("--force", action="store_true", 
                       help="Force re-download (ignores existing data)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be downloaded without actually downloading")
    
    args = parser.parse_args()
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
    elif args.tier == 1:
        symbols = get_tier1_symbols()
    else:
        logger.error("Specify either --symbols or --tier")
        return 1
    
    # Get schemas to process
    if "all" in args.schemas:
        schemas_to_process = ["core", "l1", "l2"]  # Skip L3 unless specifically requested
    else:
        schemas_to_process = args.schemas
    
    # Get date ranges
    date_ranges = get_subscription_date_ranges()
    
    # Schema configuration
    schema_configs = {
        "core": [
            {"schema": "ohlcv-1m", "dataset": "XNAS.ITCH"},
            {"schema": "ohlcv-1d", "dataset": "XNAS.ITCH"},
        ],
        "l1": [
            {"schema": "tbbo", "dataset": "XNAS.ITCH"},
            {"schema": "trades", "dataset": "XNAS.ITCH"},
        ],
        "l2": [
            {"schema": "mbp-10", "dataset": "XNAS.ITCH"},
        ],
        "l3": [
            {"schema": "mbp-1", "dataset": "XNAS.ITCH"},
        ]
    }
    
    logger.info(f"Processing {len(symbols)} symbols")
    logger.info(f"Data types: {schemas_to_process}")
    logger.info(f"Output directory: {args.data_dir}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - showing what would be downloaded")
    
    # Initialize client
    client = None
    if not args.dry_run:
        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            logger.error("DATABENTO_API_KEY not set")
            return 1
        client = db.Historical(api_key)
    
    # Process each symbol
    total_success = 0
    total_attempted = 0
    
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
        
        # Create symbol directory
        symbol_dir = args.data_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        symbol_success = 0
        symbol_attempted = 0
        
        # Process each schema type
        for schema_type in schemas_to_process:
            if schema_type not in schema_configs:
                logger.warning(f"Unknown schema type: {schema_type}")
                continue
                
            target_range = date_ranges[schema_type]
            logger.info(f"  {schema_type.upper()}: {target_range[0].date()} to {target_range[1].date()}")
            
            # Process each schema variant
            for config in schema_configs[schema_type]:
                schema = config["schema"]
                dataset = config["dataset"]
                
                symbol_attempted += 1
                total_attempted += 1
                
                if args.dry_run:
                    output_file = symbol_dir / f"{symbol}_{schema}.parquet"
                    existing_range = get_existing_date_range(output_file)
                    missing_ranges = calculate_missing_ranges(target_range, existing_range)
                    
                    if missing_ranges:
                        logger.info(f"    Would download {schema}: {len(missing_ranges)} range(s)")
                    else:
                        logger.info(f"    {schema}: ✅ Already complete")
                    continue
                
                # Actual download
                try:
                    success = process_symbol_schema(
                        client, symbol, schema, dataset, target_range, symbol_dir
                    )
                    if success:
                        symbol_success += 1
                        total_success += 1
                        logger.info(f"    ✅ {schema}")
                    else:
                        logger.warning(f"    ❌ {schema}")
                        
                except Exception as e:
                    logger.error(f"    ❌ {schema}: {e}")
        
        # Symbol summary
        if not args.dry_run:
            logger.info(f"  Symbol result: {symbol_success}/{symbol_attempted} schemas successful")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("DRY RUN COMPLETED - no data was downloaded")
    else:
        success_rate = (total_success / total_attempted) * 100 if total_attempted > 0 else 0
        logger.info(f"Overall success rate: {total_success}/{total_attempted} ({success_rate:.1f}%)")
        
        if total_success == total_attempted:
            logger.info("🎉 ALL DOWNLOADS SUCCESSFUL!")
        elif total_success > 0:
            logger.info("⚠️  Some downloads failed - check logs above")
        else:
            logger.error("❌ ALL DOWNLOADS FAILED")
            return 1
    
    logger.info("\nNext steps:")
    logger.info("1. Verify data completeness with: python comprehensive_data_downloader.py --dry-run")
    logger.info("2. Set up incremental updates (daily)")
    logger.info("3. Configure TFT training with comprehensive dataset")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())