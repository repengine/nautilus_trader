#!/usr/bin/env python3
"""
Analyze existing data vs Databento subscription to show exactly what gaps need filling.
"""
import logging
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_subscription_coverage():
    """What you should have based on your Databento subscription."""
    today = datetime.now().date()
    yesterday = datetime.combine(today - timedelta(days=1), datetime.min.time())

    return {
        "ohlcv_daily": (yesterday - timedelta(days=365 * 7), yesterday),  # 7 years
        "ohlcv_1m": (yesterday - timedelta(days=365 * 7), yesterday),     # 7 years
        "l1_bbo": (yesterday - timedelta(days=365), yesterday),           # 1 year
        "l1_trades": (yesterday - timedelta(days=365), yesterday),        # 1 year
        "l2_mbp": (yesterday - timedelta(days=30), yesterday),            # 30 days
    }

def analyze_symbol(symbol_dir: Path, target_coverage: dict):
    """Analyze what data exists vs what you should have."""
    symbol = symbol_dir.name
    results = {
        "symbol": symbol,
        "coverage": {},
        "gaps": {},
        "recommendations": []
    }

    # Check daily data
    daily_file = symbol_dir / "daily_7y.parquet"
    if daily_file.exists():
        try:
            df = pd.read_parquet(daily_file)
            if len(df) > 0:
                # ts_event is the index in your format
                start_date = pd.to_datetime(df.index.min()).to_pydatetime()
                end_date = pd.to_datetime(df.index.max()).to_pydatetime()

                results["coverage"]["ohlcv_daily"] = {
                    "start": start_date,
                    "end": end_date,
                    "days": (end_date - start_date).days,
                    "records": len(df),
                    "size_mb": daily_file.stat().st_size / (1024 * 1024)
                }

                target_start, target_end = target_coverage["ohlcv_daily"]
                if start_date > target_start:
                    gap_days = (start_date - target_start).days
                    results["gaps"]["ohlcv_daily_historical"] = f"Missing {gap_days} days of historical data ({target_start.date()} to {start_date.date()})"

                if end_date < target_end:
                    gap_days = (target_end - end_date).days
                    results["gaps"]["ohlcv_daily_recent"] = f"Missing {gap_days} days of recent data ({end_date.date()} to {target_end.date()})"

        except Exception as e:
            logger.warning(f"Failed to read {daily_file}: {e}")
    else:
        results["gaps"]["ohlcv_daily_all"] = "Missing all daily OHLCV data (7 years)"

    # Check hourly/1m data
    hourly_file = symbol_dir / "hourly_7y.parquet"
    if hourly_file.exists():
        try:
            df = pd.read_parquet(hourly_file)
            if len(df) > 0:
                start_date = pd.to_datetime(df.index.min()).to_pydatetime()
                end_date = pd.to_datetime(df.index.max()).to_pydatetime()

                results["coverage"]["ohlcv_1m"] = {
                    "start": start_date,
                    "end": end_date,
                    "days": (end_date - start_date).days,
                    "records": len(df),
                    "size_mb": hourly_file.stat().st_size / (1024 * 1024),
                    "note": "Currently hourly, could get 1-minute"
                }

                target_start, target_end = target_coverage["ohlcv_1m"]
                if start_date > target_start:
                    gap_days = (start_date - target_start).days
                    results["gaps"]["ohlcv_1m_historical"] = f"Missing {gap_days} days of historical 1m data ({target_start.date()} to {start_date.date()})"

                if end_date < target_end:
                    gap_days = (target_end - end_date).days
                    results["gaps"]["ohlcv_1m_recent"] = f"Missing {gap_days} days of recent 1m data ({end_date.date()} to {target_end.date()})"

                results["recommendations"].append("💡 Could upgrade hourly to 1-minute data")
        except Exception as e:
            logger.warning(f"Failed to read {hourly_file}: {e}")
    else:
        results["gaps"]["ohlcv_1m_all"] = "Missing all 1-minute OHLCV data (7 years)"

    # Check L1 data
    l1_dir = symbol_dir / "l1"
    if l1_dir.exists():
        # BBO data
        bbo_files = list(l1_dir.glob("*bbo*"))
        if bbo_files:
            try:
                df = pd.read_parquet(bbo_files[0])
                if "ts_event" in df.columns:
                    start_ts = df["ts_event"].min()
                    end_ts = df["ts_event"].max()
                    start_date = pd.to_datetime(start_ts, unit="ns").to_pydatetime()
                    end_date = pd.to_datetime(end_ts, unit="ns").to_pydatetime()

                    results["coverage"]["l1_bbo"] = {
                        "start": start_date,
                        "end": end_date,
                        "days": (end_date - start_date).days,
                        "records": len(df),
                        "size_mb": bbo_files[0].stat().st_size / (1024 * 1024)
                    }

                    target_start, target_end = target_coverage["l1_bbo"]
                    if start_date > target_start:
                        gap_days = (start_date - target_start).days
                        results["gaps"]["l1_bbo_historical"] = f"Missing {gap_days} days of historical L1 BBO ({target_start.date()} to {start_date.date()})"

            except Exception as e:
                logger.warning(f"Failed to read BBO file: {e}")
        else:
            results["gaps"]["l1_bbo_all"] = "Missing all L1 BBO data (1 year)"

        # Trades data
        trades_files = list(l1_dir.glob("*trades*"))
        if trades_files:
            try:
                df = pd.read_parquet(trades_files[0])
                if "ts_event" in df.columns:
                    start_ts = df["ts_event"].min()
                    end_ts = df["ts_event"].max()
                    start_date = pd.to_datetime(start_ts, unit="ns").to_pydatetime()
                    end_date = pd.to_datetime(end_ts, unit="ns").to_pydatetime()

                    results["coverage"]["l1_trades"] = {
                        "start": start_date,
                        "end": end_date,
                        "days": (end_date - start_date).days,
                        "records": len(df),
                        "size_mb": trades_files[0].stat().st_size / (1024 * 1024)
                    }

            except Exception as e:
                logger.warning(f"Failed to read trades file: {e}")
        else:
            results["gaps"]["l1_trades_all"] = "Missing all L1 trades data (1 year)"
    else:
        results["gaps"]["l1_all"] = "Missing all L1 data (1 year BBO + trades)"

    # Check L2 data
    l2_dir = symbol_dir / "l2"
    if l2_dir.exists():
        mbp_files = list(l2_dir.glob("*"))
        if mbp_files:
            try:
                df = pd.read_parquet(mbp_files[0])
                if "ts_event" in df.columns:
                    start_ts = df["ts_event"].min()
                    end_ts = df["ts_event"].max()
                    start_date = pd.to_datetime(start_ts, unit="ns").to_pydatetime()
                    end_date = pd.to_datetime(end_ts, unit="ns").to_pydatetime()

                    results["coverage"]["l2_mbp"] = {
                        "start": start_date,
                        "end": end_date,
                        "days": (end_date - start_date).days,
                        "records": len(df),
                        "size_mb": mbp_files[0].stat().st_size / (1024 * 1024)
                    }

            except Exception as e:
                logger.warning(f"Failed to read L2 file: {e}")
        else:
            results["gaps"]["l2_all"] = "Missing all L2 data (30 days)"
    else:
        results["gaps"]["l2_all"] = "Missing all L2 data (30 days)"

    return results

def main():
    tier1_dir = Path("data/tier1")
    if not tier1_dir.exists():
        logger.error("data/tier1 directory not found")
        return

    target_coverage = get_subscription_coverage()

    print("🔍 ANALYZING EXISTING DATA vs DATABENTO SUBSCRIPTION")
    print("=" * 80)
    print("Subscription entitlements:")
    print(f"  📊 OHLCV Daily/1m: 7 years ({target_coverage['ohlcv_daily'][0].date()} to {target_coverage['ohlcv_daily'][1].date()})")
    print(f"  📈 L1 BBO/Trades: 1 year ({target_coverage['l1_bbo'][0].date()} to {target_coverage['l1_bbo'][1].date()})")
    print(f"  📊 L2 MBP: 30 days ({target_coverage['l2_mbp'][0].date()} to {target_coverage['l2_mbp'][1].date()})")
    print()

    symbol_dirs = [d for d in tier1_dir.iterdir() if d.is_dir()]
    symbol_dirs.sort()

    total_gaps = 0
    symbols_with_gaps = 0

    for i, symbol_dir in enumerate(symbol_dirs[:10], 1):  # Show first 10 symbols
        results = analyze_symbol(symbol_dir, target_coverage)
        symbol = results["symbol"]

        print(f"[{i:2d}] {symbol}")

        # Show coverage
        if results["coverage"]:
            print("  ✅ Current coverage:")
            for data_type, info in results["coverage"].items():
                days = info["days"]
                records = info["records"]
                size_mb = info["size_mb"]
                note = info.get("note", "")
                print(f"    {data_type}: {days} days, {records:,} records, {size_mb:.1f}MB {note}")

        # Show gaps
        if results["gaps"]:
            print("  ⚠️  Missing data:")
            for gap_type, gap_desc in results["gaps"].items():
                print(f"    {gap_desc}")
            total_gaps += len(results["gaps"])
            symbols_with_gaps += 1
        else:
            print("  🎉 Complete coverage!")

        # Show recommendations
        if results["recommendations"]:
            for rec in results["recommendations"]:
                print(f"  {rec}")

        print()

    if len(symbol_dirs) > 10:
        print(f"... and {len(symbol_dirs) - 10} more symbols")
        print()

    print("📈 SUMMARY")
    print("-" * 40)
    print(f"Total symbols: {len(symbol_dirs)}")
    print(f"Symbols with gaps: {symbols_with_gaps}")
    print(f"Total gaps identified: {total_gaps}")
    print()
    print("💡 RECOMMENDATION:")
    print("Use comprehensive_data_downloader.py to fill gaps and get full subscription value!")

if __name__ == "__main__":
    main()
