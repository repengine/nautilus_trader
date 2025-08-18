#!/usr/bin/env python
"""Smoke script for enhanced collector on a single symbol."""


from ml.data.enhanced_collector import EnhancedDataCollector


# Test on SPY - most liquid symbol
test_symbol = ["SPY"]

print("Testing Enhanced Collector")
print("=" * 50)
print(f"Test symbol: {test_symbol[0]}")
print()

collector = EnhancedDataCollector(storage_limit_gb=1.0)  # 1GB limit for test

# Test each data type
print("1. Testing L2 Depth (mbp-1) for 1 day...")
collector.collect_l2_depth(symbols=test_symbol, days=1)

print("\n2. Testing L1 Trades for 1 year...")
collector.collect_l1_trades(symbols=test_symbol, years=1)

print("\n3. Testing TBBO Quotes for 1 day...")
collector.collect_tbbo_quotes(symbols=test_symbol, days=1)

print("\n4. Testing Minute Bars for 7 days...")
collector.collect_minute_bars(symbols=test_symbol, days=7)

print("\n" + "=" * 50)
print("Test Complete!")
print("Check /data/enhanced/SPY/ for collected files")
