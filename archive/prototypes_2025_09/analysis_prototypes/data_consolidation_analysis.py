#!/usr/bin/env python3
"""
Analyze and consolidate the tier1/enhanced data overlap.

This script helps determine the optimal data organization strategy.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import polars as pl


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parquet_date_range(file_path: Path) -> Optional[Tuple[datetime, datetime, int]]:
    """Get date range and record count from parquet file."""
    if not file_path.exists():
        return None
        
    try:
        df = pl.scan_parquet(file_path)
        
        # Get metadata efficiently
        sample = df.head(1000).select('ts_event').collect()
        tail = df.tail(1000).select('ts_event').collect()
        
        if len(sample) == 0:
            return None
            
        min_ts = min(sample['ts_event'].min(), tail['ts_event'].min())
        max_ts = max(sample['ts_event'].max(), tail['ts_event'].max())
        
        # Get approximate count (faster than exact count for large files)
        count = df.select(pl.count()).collect().item()
        
        min_date = pd.to_datetime(min_ts, unit='ns')
        max_date = pd.to_datetime(max_ts, unit='ns')
        
        return min_date.to_pydatetime(), max_date.to_pydatetime(), count
        
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return None


def analyze_symbol_data(symbol: str, base_path: Path) -> Dict:
    """Analyze all data files for a symbol across tier1/enhanced."""
    
    results = {
        'symbol': symbol,
        'tier1': {},
        'enhanced': {},
        'recommendations': []
    }
    
    # Analyze tier1 data
    tier1_dir = base_path / "tier1" / symbol
    if tier1_dir.exists():
        
        # L1 data (BBO/Trades)
        l1_dir = tier1_dir / "l1"
        if l1_dir.exists():
            for file_path in l1_dir.glob("*.parquet"):
                data_type = "bbo" if "bbo" in file_path.name else "trades"
                info = get_parquet_date_range(file_path)
                if info:
                    results['tier1'][f'l1_{data_type}'] = {
                        'file': file_path,
                        'date_range': f"{info[0].date()} to {info[1].date()}",
                        'days': (info[1] - info[0]).days + 1,
                        'records': info[2],
                        'size_mb': file_path.stat().st_size / (1024 * 1024)
                    }
        
        # L2 data (MBP)
        l2_dir = tier1_dir / "l2"
        if l2_dir.exists():
            for file_path in l2_dir.glob("*.parquet"):
                info = get_parquet_date_range(file_path)
                if info:
                    results['tier1']['l2_mbp'] = {
                        'file': file_path,
                        'date_range': f"{info[0].date()} to {info[1].date()}",
                        'days': (info[1] - info[0]).days + 1,
                        'records': info[2],
                        'size_mb': file_path.stat().st_size / (1024 * 1024)
                    }
        
        # Aggregated data (daily/hourly)
        for file_path in tier1_dir.glob("*.parquet"):
            if "daily" in file_path.name or "hourly" in file_path.name:
                info = get_parquet_date_range(file_path)
                if info:
                    data_type = "daily" if "daily" in file_path.name else "hourly"
                    results['tier1'][f'aggregated_{data_type}'] = {
                        'file': file_path,
                        'date_range': f"{info[0].date()} to {info[1].date()}",
                        'days': (info[1] - info[0]).days + 1,
                        'records': info[2],
                        'size_mb': file_path.stat().st_size / (1024 * 1024)
                    }
    
    # Analyze enhanced data
    enhanced_dir = base_path / "enhanced" / symbol
    if enhanced_dir.exists():
        for file_path in enhanced_dir.glob("*.parquet"):
            if "trades_" in file_path.name:
                year = file_path.name.split("_")[1].split(".")[0]
                data_key = f"trades_{year}"
            elif "l2_depth" in file_path.name:
                data_key = "l2_depth"
            elif "tbbo" in file_path.name:
                data_key = "tbbo"
            elif "bars_1m" in file_path.name:
                data_key = "bars_1m"
            else:
                continue
                
            info = get_parquet_date_range(file_path)
            if info:
                results['enhanced'][data_key] = {
                    'file': file_path,
                    'date_range': f"{info[0].date()} to {info[1].date()}",
                    'days': (info[1] - info[0]).days + 1,
                    'records': info[2],
                    'size_mb': file_path.stat().st_size / (1024 * 1024)
                }
    
    # Generate recommendations
    recommendations = []
    
    # Check for overlaps
    if results['tier1'] and results['enhanced']:
        recommendations.append("⚠️  Data exists in both tier1 and enhanced")
        
        # Compare L2 data
        if 'l2_mbp' in results['tier1'] and 'l2_depth' in results['enhanced']:
            tier1_days = results['tier1']['l2_mbp']['days']
            enhanced_days = results['enhanced']['l2_depth']['days']
            
            if tier1_days > enhanced_days:
                recommendations.append(f"✅ Keep tier1 L2 data ({tier1_days} days vs {enhanced_days} days)")
            else:
                recommendations.append(f"⚠️  Enhanced L2 may be more complete ({enhanced_days} days vs {tier1_days} days)")
        
        # Compare trade data
        tier1_trades = results['tier1'].get('l1_trades', {})
        enhanced_trades_years = [k for k in results['enhanced'].keys() if k.startswith('trades_')]
        
        if tier1_trades and enhanced_trades_years:
            tier1_days = tier1_trades.get('days', 0)
            enhanced_total_records = sum(results['enhanced'][k]['records'] for k in enhanced_trades_years)
            
            recommendations.append(f"🔍 Compare: tier1 trades ({tier1_days} days, {tier1_trades.get('records', 0):,} records) vs enhanced trades ({len(enhanced_trades_years)} years, {enhanced_total_records:,} records)")
    
    elif results['tier1']:
        recommendations.append("✅ Only tier1 data exists")
    elif results['enhanced']:
        recommendations.append("✅ Only enhanced data exists")
    else:
        recommendations.append("❌ No data found")
    
    results['recommendations'] = recommendations
    return results


def main():
    base_path = Path("data")
    
    # Get all symbols
    tier1_symbols = set()
    enhanced_symbols = set()
    
    tier1_path = base_path / "tier1"
    if tier1_path.exists():
        tier1_symbols = {d.name for d in tier1_path.iterdir() if d.is_dir()}
    
    enhanced_path = base_path / "enhanced"  
    if enhanced_path.exists():
        enhanced_symbols = {d.name for d in enhanced_path.iterdir() if d.is_dir()}
    
    all_symbols = sorted(tier1_symbols | enhanced_symbols)
    
    logger.info(f"Found {len(all_symbols)} symbols to analyze")
    logger.info(f"  Tier1: {len(tier1_symbols)} symbols")
    logger.info(f"  Enhanced: {len(enhanced_symbols)} symbols")
    logger.info(f"  Overlap: {len(tier1_symbols & enhanced_symbols)} symbols")
    
    print("\n" + "=" * 80)
    print("TIER1 vs ENHANCED DATA ANALYSIS")
    print("=" * 80)
    
    # Analyze each symbol
    overlap_symbols = []
    tier1_only = []
    enhanced_only = []
    
    for symbol in all_symbols:
        results = analyze_symbol_data(symbol, base_path)
        
        has_tier1 = bool(results['tier1'])
        has_enhanced = bool(results['enhanced'])
        
        if has_tier1 and has_enhanced:
            overlap_symbols.append((symbol, results))
        elif has_tier1:
            tier1_only.append((symbol, results))
        else:
            enhanced_only.append((symbol, results))
    
    # Report overlap symbols (most important)
    if overlap_symbols:
        print(f"\n📊 OVERLAP ANALYSIS ({len(overlap_symbols)} symbols)")
        print("-" * 50)
        
        for symbol, results in overlap_symbols:
            print(f"\n{symbol}:")
            
            # Tier1 summary
            if results['tier1']:
                total_tier1_mb = sum(v.get('size_mb', 0) for v in results['tier1'].values())
                print(f"  Tier1: {len(results['tier1'])} files, {total_tier1_mb:.1f} MB")
                
                for data_type, info in results['tier1'].items():
                    print(f"    {data_type}: {info['date_range']} ({info['records']:,} records, {info['size_mb']:.1f} MB)")
            
            # Enhanced summary
            if results['enhanced']:
                total_enhanced_mb = sum(v.get('size_mb', 0) for v in results['enhanced'].values())
                print(f"  Enhanced: {len(results['enhanced'])} files, {total_enhanced_mb:.1f} MB")
                
                for data_type, info in results['enhanced'].items():
                    print(f"    {data_type}: {info['date_range']} ({info['records']:,} records, {info['size_mb']:.1f} MB)")
            
            # Recommendations
            for rec in results['recommendations']:
                print(f"  {rec}")
    
    # Summary statistics
    print(f"\n📈 CONSOLIDATION SUMMARY")
    print("-" * 50)
    print(f"Symbols with data overlap: {len(overlap_symbols)}")
    print(f"Symbols only in tier1: {len(tier1_only)}")
    print(f"Symbols only in enhanced: {len(enhanced_only)}")
    
    # Calculate potential savings
    total_overlap_mb = 0
    for symbol, results in overlap_symbols:
        tier1_mb = sum(v.get('size_mb', 0) for v in results['tier1'].values())
        enhanced_mb = sum(v.get('size_mb', 0) for v in results['enhanced'].values())
        # Estimate overlap (conservative)
        overlap_mb = min(tier1_mb, enhanced_mb) * 0.5  # Assume 50% overlap
        total_overlap_mb += overlap_mb
    
    print(f"\nEstimated duplicate data: {total_overlap_mb / 1024:.1f} GB")
    
    print(f"\n🎯 RECOMMENDATIONS")
    print("-" * 50)
    print("1. Keep tier1 as primary dataset (better organized, more recent)")
    print("2. Archive enhanced data for symbols already in tier1")
    print("3. Keep enhanced data for symbols NOT in tier1")
    print("4. Use comprehensive downloader to fill gaps with subscription data")
    print("5. Consolidate into single organized structure")


if __name__ == "__main__":
    main()