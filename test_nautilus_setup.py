#!/usr/bin/env python
"""
Test script to verify Nautilus Trader installation and basic functionality.
"""

import sys
from datetime import datetime

print("Testing Nautilus Trader setup...")
print("-" * 50)

# Test imports
try:
    import nautilus_trader
    print(f"✓ Nautilus Trader version: {nautilus_trader.__version__}")
except ImportError as e:
    print(f"✗ Failed to import nautilus_trader: {e}")
    sys.exit(1)

# Test core components
try:
    from nautilus_trader.core.data import Data
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
    from nautilus_trader.model.instruments import Instrument
    from nautilus_trader.model.data import Bar, BarType, BarSpecification
    from nautilus_trader.model.enums import AggregationSource
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.config import BacktestEngineConfig
    from nautilus_trader.model.enums import BarAggregation
    print("✓ Core components imported successfully")
except ImportError as e:
    print(f"✗ Failed to import core components: {e}")
    sys.exit(1)

# Test data types
try:
    # Create identifiers
    venue = Venue("BINANCE")
    symbol = Symbol("BTCUSDT")
    instrument_id = InstrumentId(symbol=symbol, venue=venue)
    print(f"✓ Created InstrumentId: {instrument_id}")
    
    # Create bar specification
    from nautilus_trader.model.enums import PriceType
    bar_spec = BarSpecification(
        step=1,
        aggregation=BarAggregation.MINUTE,
        price_type=PriceType.LAST
    )
    bar_type = BarType(
        instrument_id=instrument_id,
        bar_spec=bar_spec,
        aggregation_source=AggregationSource.EXTERNAL
    )
    print(f"✓ Created BarType: {bar_type}")
    
except Exception as e:
    print(f"✗ Failed to create data types: {e}")
    sys.exit(1)

# Test backtest engine configuration
try:
    config = BacktestEngineConfig()
    engine = BacktestEngine(config=config)
    print("✓ BacktestEngine created successfully")
except Exception as e:
    print(f"✗ Failed to create BacktestEngine: {e}")
    sys.exit(1)

# Test common ML-related dependencies
ml_deps = []
try:
    import pandas as pd
    ml_deps.append(f"pandas=={pd.__version__}")
except ImportError:
    ml_deps.append("pandas (NOT INSTALLED)")

try:
    import numpy as np
    ml_deps.append(f"numpy=={np.__version__}")
except ImportError:
    ml_deps.append("numpy (NOT INSTALLED)")

try:
    import sklearn
    ml_deps.append(f"scikit-learn=={sklearn.__version__}")
except ImportError:
    ml_deps.append("scikit-learn (NOT INSTALLED - optional)")

print("\nML-related dependencies:")
for dep in ml_deps:
    print(f"  - {dep}")

print("\n" + "-" * 50)
print("✓ Nautilus Trader is set up correctly!")
print("\nNext steps for ML integration:")
print("1. Install ML dependencies: pip install scikit-learn torch tensorflow (as needed)")
print("2. Set up data pipelines to extract features from Nautilus data")
print("3. Create custom strategies that use ML models for portfolio optimization")
print("4. Use BacktestEngine to validate ML-based strategies")