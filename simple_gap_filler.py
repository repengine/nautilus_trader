#!/usr/bin/env python3
"""
Simple Gap Filler - Fast gap detection and filling for your Databento subscription.

This script efficiently identifies and downloads missing data without the noise.
"""
import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import databento as db
import warnings
warnings.filterwarnings("ignore")


def get_subscription_ranges():
    """What you should have based on Databento subscription."""
    today = datetime.now().date()
    yesterday = datetime.combine(today - timedelta(days=1), datetime.min.time())
    
    return {
        "core": (yesterday - timedelta(days=365 * 7), yesterday),    # 7 years OHLCV
        "l1": (yesterday - timedelta(days=365), yesterday),         # 1 year L1 
        "l2": (yesterday - timedelta(days=30), yesterday),          # 30 days L2
    }


def analyze_symbol(symbol_dir: Path, subscription_ranges: dict) -> dict:
    """Fast analysis of what's missing for a symbol."""
    symbol = symbol_dir.name
    gaps = []
    
    # Check for OHLCV files (various naming patterns)
    ohlcv_daily_files = list(symbol_dir.glob("*daily*.parquet"))
    ohlcv_hourly_files = list(symbol_dir.glob("*hourly*.parquet"))
    l1_tbbo_files = list((symbol_dir / "l1").glob("*bbo*"))
    l1_trades_files = list((symbol_dir / "l1").glob("*trades*"))  
    l2_files = list((symbol_dir / "l2").glob("*"))
    
    # Check each data type
    checks = [
        (ohlcv_daily_files, "ohlcv-1d", "core", "daily OHLCV"),
        (ohlcv_hourly_files, "ohlcv-1m", "core", "hourly/1m OHLCV"),
        (l1_tbbo_files, "tbbo", "l1", "L1 BBO data"),
        (l1_trades_files, "trades", "l1", "L1 trades data"),
        (l2_files, "mbp-10", "l2", "L2 MBP data")
    ]
    
    for files, schema, data_type, description in checks:
        if not files:
            # No files found - need full range
            if data_type in subscription_ranges:
                start_date, end_date = subscription_ranges[data_type]
                output_file = symbol_dir / f"{schema}.parquet"
                if data_type == "l1":
                    output_file = symbol_dir / "l1" / f"{schema}.parquet"
                elif data_type == "l2":
                    output_file = symbol_dir / "l2" / f"{schema}.parquet"
                    
                gaps.append({
                    "symbol": symbol,
                    "schema": schema,
                    "start": start_date,
                    "end": end_date,
                    "days": (end_date - start_date).days,
                    "reason": f"Missing {description}",
                    "output_file": output_file
                })
        else:
            # Files exist - check if they're comprehensive enough
            if data_type == "core":
                file_path = files[0]  # Use first found file
                target_start, target_end = subscription_ranges[data_type]
                
                # Simple heuristic: if file is named "1y", we likely need more historical data
                if "1y" in file_path.name:
                    # Need more historical data (7 years vs 1 year)
                    gaps.append({
                        "symbol": symbol,
                        "schema": schema,
                        "start": target_start,
                        "end": target_start + timedelta(days=365*6),  # 6 more years
                        "days": 365*6,
                        "reason": f"Need historical {description} (currently only 1 year)",
                        "output_file": symbol_dir / f"{schema}_historical.parquet"
                    })
                
                # Also check for recent data gaps
                file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.days > 7:
                    recent_start = target_end - timedelta(days=7)
                    gaps.append({
                        "symbol": symbol,
                        "schema": schema,
                        "start": recent_start,
                        "end": target_end,
                        "days": 7,
                        "reason": f"Recent {description} updates needed",
                        "output_file": symbol_dir / f"{schema}_recent.parquet"
                    })
    
    return {"symbol": symbol, "gaps": gaps}


def download_gap(client: db.Historical, gap: dict, dry_run: bool = False) -> bool:
    """Download a specific gap."""
    symbol = gap["symbol"]
    schema = gap["schema"]
    start_date = gap["start"]
    end_date = gap["end"]
    days = gap["days"]
    
    if dry_run:
        print(f"  [DRY RUN] Would download {schema}: {start_date.date()} to {end_date.date()} ({days} days)")
        return True
    
    try:
        print(f"  Downloading {schema}: {start_date.date()} to {end_date.date()} ({days} days)")
        
        # Download with intelligent batching
        total_days = (end_date - start_date).days
        
        # Batch size based on schema type
        if schema.startswith("ohlcv"):
            batch_days = min(365, total_days)  # 1 year max
        elif schema in ["tbbo", "trades"]:
            batch_days = min(90, total_days)   # 3 months max  
        else:
            batch_days = min(30, total_days)   # 1 month max
        
        if total_days <= batch_days:
            # Single download
            df = client.timeseries.get_range(
                dataset="XNAS.ITCH",
                symbols=[symbol],
                schema=schema,
                start=start_date,
                end=end_date,
            ).to_df()
            
            if df.empty:
                print(f"    ⚠️  No data returned")
                return False
            
            # Save file
            gap["output_file"].parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(gap["output_file"])
            
            size_mb = gap["output_file"].stat().st_size / (1024 * 1024)
            print(f"    ✅ Saved {len(df):,} records ({size_mb:.1f} MB)")
            
        else:
            # Multi-batch download
            print(f"    Large request - splitting into {batch_days}-day batches")
            current_start = start_date
            all_dfs = []
            
            while current_start < end_date:
                batch_end = min(current_start + timedelta(days=batch_days), end_date)
                
                df = client.timeseries.get_range(
                    dataset="XNAS.ITCH",
                    symbols=[symbol],
                    schema=schema,
                    start=current_start,
                    end=batch_end,
                ).to_df()
                
                if not df.empty:
                    all_dfs.append(df)
                
                current_start = batch_end + timedelta(days=1)
            
            if all_dfs:
                # Combine all batches
                import pandas as pd
                combined_df = pd.concat(all_dfs)  # Keep the index (ts_event)
                
                # Sort and deduplicate by ts_event index
                combined_df = combined_df.sort_index()
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                
                gap["output_file"].parent.mkdir(parents=True, exist_ok=True)
                combined_df.to_parquet(gap["output_file"])
                
                size_mb = gap["output_file"].stat().st_size / (1024 * 1024)
                print(f"    ✅ Combined {len(combined_df):,} records ({size_mb:.1f} MB)")
            else:
                print(f"    ⚠️  No data returned")
                return False
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "license" in error_msg.lower() or "403" in error_msg:
            print(f"    ⚠️  License restriction")
            return True  # Not an error
        
        print(f"    ❌ Failed: {error_msg}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Simple Gap Filler - Fast and quiet")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tier1"))
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    parser.add_argument("--max-symbols", type=int, help="Limit to first N symbols")
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"❌ Data directory {args.data_dir} does not exist")
        return
    
    # Get subscription ranges
    subscription_ranges = get_subscription_ranges()
    print("📊 Databento Subscription Coverage:")
    print(f"  Core (OHLCV): 7 years")
    print(f"  L1 (BBO/Trades): 1 year") 
    print(f"  L2 (MBP): 30 days")
    print()
    
    # Get symbol directories
    symbol_dirs = [d for d in args.data_dir.iterdir() if d.is_dir()]
    symbol_dirs.sort()
    
    if args.max_symbols:
        symbol_dirs = symbol_dirs[:args.max_symbols]
    
    if not symbol_dirs:
        print("❌ No symbol directories found")
        return
    
    # Initialize client if not dry run
    client = None
    if not args.dry_run:
        try:
            client = db.Historical()
            print("✅ Connected to Databento")
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return
        print()
    
    # Analyze symbols
    print(f"🔍 Analyzing {len(symbol_dirs)} symbols...")
    print()
    
    all_gaps = []
    symbols_with_gaps = 0
    
    for i, symbol_dir in enumerate(symbol_dirs, 1):
        analysis = analyze_symbol(symbol_dir, subscription_ranges)
        symbol = analysis["symbol"]
        gaps = analysis["gaps"]
        
        print(f"[{i:3d}/{len(symbol_dirs)}] {symbol}", end="")
        
        if gaps:
            print(f" - {len(gaps)} gaps found")
            for gap in gaps:
                print(f"    {gap['reason']}: {gap['days']} days")
            symbols_with_gaps += 1
            all_gaps.extend(gaps)
        else:
            print(" - Complete")
    
    print()
    print("=" * 60)
    print("📈 SUMMARY")
    print("=" * 60)
    print(f"Symbols analyzed: {len(symbol_dirs)}")
    print(f"Symbols with gaps: {symbols_with_gaps}")
    print(f"Total downloads needed: {len(all_gaps)}")
    
    if not all_gaps:
        print("🎉 No gaps found!")
        return
    
    if args.dry_run:
        print()
        print("🔍 DRY RUN - What would be downloaded:")
        for gap in all_gaps:
            print(f"  {gap['symbol']} {gap['schema']}: {gap['days']} days ({gap['reason']})")
        return
    
    # Execute downloads
    print()
    print("🚀 Starting downloads...")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for i, gap in enumerate(all_gaps, 1):
        print(f"[{i:2d}/{len(all_gaps)}] {gap['symbol']}")
        if download_gap(client, gap):
            successful += 1
        else:
            failed += 1
        print()
    
    print("=" * 60)
    print("🏁 DOWNLOAD COMPLETE!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success rate: {successful/(successful+failed)*100:.1f}%")


if __name__ == "__main__":
    main()