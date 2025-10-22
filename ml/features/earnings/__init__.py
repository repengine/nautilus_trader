"""
Earnings Features Module - Corporate Fundamentals Integration.

This module provides earnings-based fundamental features with guaranteed hot/cold path
parity following Universal ML Architecture Patterns.

## Features Provided

### Earnings Surprise Features
- **EPS Surprise**: Dollar and percentage surprise calculations
  - Hot path: O(1) incremental computation
  - Cold path: Vectorized batch computation
  - Parity: Validated to rtol=1e-10

### Earnings Growth Features
- **YoY/QoQ Growth**: Year-over-year and quarter-over-quarter EPS growth
  - Hot path: O(1) with circular buffer lookback
  - Cold path: Batch windowed computation
  - Parity: Validated to rtol=1e-10

### Earnings Momentum Features
- **Beat Streak**: Consecutive quarters beating consensus
- **EPS Volatility**: 4-quarter EPS volatility (coefficient of variation)
  - Hot path: Incremental updates with rolling windows
  - Cold path: Batch statistical computation
  - Parity: Validated to rtol=1e-10

### Earnings Calendar Features
- **Days to Earnings**: Days until next earnings announcement
  - Hot path: Simple date arithmetic
  - Cold path: Batch date computation
  - Parity: Exact (integer arithmetic)

## Hot Path Performance

All hot path components meet strict SLA requirements:
- P99 latency < 5ms
- O(1) computational complexity
- Zero allocations after warmup
- Pre-allocated arrays for all computations

## Usage Examples

### Hot Path (Real-time)
```python
from ml.features.earnings import compute_earnings_surprise_incremental

# Update incrementally (hot path - O(1))
surprise = compute_earnings_surprise_incremental(
    actual=2.52,
    estimate=2.45
)
print(f"EPS Surprise: ${surprise['eps_surprise_q0']:.2f}")
```

### Cold Path (Batch)
```python
from ml.features.earnings import compute_earnings_surprise_batch
import numpy as np

# Batch computation (cold path)
actuals = np.array([2.52, 2.45, 2.38, 2.30])
estimates = np.array([2.45, 2.40, 2.35, 2.28])
surprises = compute_earnings_surprise_batch(actuals, estimates)
```

## Pattern Compliance

This module follows Universal ML Architecture Patterns:
- **Pattern 2**: Protocol-First Interface Design
- **Pattern 3**: Hot/Cold Path Separation with strict performance SLAs
- **Pattern 5**: Centralized Metrics Bootstrap for monitoring
"""

from __future__ import annotations


__all__ = [
    "EarningsCalendarTransformSpec",
    "EarningsGrowthTransformSpec",
    "EarningsMomentumTransformSpec",
    "EarningsSurpriseTransformSpec",
    "compute_calendar_features_batch",
    "compute_calendar_features_incremental",
    "compute_earnings_growth_batch",
    "compute_earnings_growth_incremental",
    "compute_earnings_momentum_batch",
    "compute_earnings_momentum_incremental",
    "compute_earnings_surprise_batch",
    "compute_earnings_surprise_incremental",
]

# Direct imports for pickle compatibility
# (lazy imports via __getattr__ break pickle identity checks)
from ml.features.earnings.earnings_features import compute_calendar_features_batch
from ml.features.earnings.earnings_features import compute_calendar_features_incremental
from ml.features.earnings.earnings_features import compute_earnings_growth_batch
from ml.features.earnings.earnings_features import compute_earnings_growth_incremental
from ml.features.earnings.earnings_features import compute_earnings_momentum_batch
from ml.features.earnings.earnings_features import compute_earnings_momentum_incremental
from ml.features.earnings.earnings_features import compute_earnings_surprise_batch
from ml.features.earnings.earnings_features import compute_earnings_surprise_incremental
from ml.features.earnings.earnings_transforms import EarningsCalendarTransformSpec
from ml.features.earnings.earnings_transforms import EarningsGrowthTransformSpec
from ml.features.earnings.earnings_transforms import EarningsMomentumTransformSpec
from ml.features.earnings.earnings_transforms import EarningsSurpriseTransformSpec
