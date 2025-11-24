"""
Cross-Asset Relationship Features Module.

This module provides cross-asset relationship features with guaranteed hot/cold path
parity following Universal ML Architecture Patterns.

## Features Provided

### Beta Computation
- **EWMA Beta**: Exponentially weighted moving average beta computation
  - Hot path: O(1) incremental updates using Welford's algorithm
  - Cold path: Vectorized batch computation
  - Parity: Validated to rtol=1e-10

### Spread Features
- **Z-Scored Spreads**: Statistical spread analysis between asset pairs
  - Hot path: Incremental mean/std updates with Welford's algorithm
  - Cold path: Batch z-score computation
  - Parity: Validated to rtol=1e-10

### State Management
- Serializable dataclasses for cross-asset state persistence
- Supports parity validation between hot and cold paths

## Hot Path Performance

All hot path components meet strict SLA requirements:
- P99 latency < 5ms
- O(1) computational complexity
- Zero allocations after warmup
- Pre-allocated arrays for all computations

## Usage Examples

### Hot Path (Real-time)
```python
from ml.features.cross_asset import EWMABetaState, compute_ewma_beta_incremental

# Initialize state
state = EWMABetaState(alpha=0.94)

# Update incrementally (hot path - O(1))
for asset_return, market_return in zip(asset_returns, market_returns):
    beta = compute_ewma_beta_incremental(
        state, asset_return, market_return
    )
```

### Cold Path (Batch)
```python
from ml.features.cross_asset import compute_ewma_beta_batch
import numpy as np

# Batch computation (cold path)
asset_returns = np.array([...])
market_returns = np.array([...])
betas = compute_ewma_beta_batch(
    asset_returns, market_returns, alpha=0.94
)
```

## Pattern Compliance

This module follows Universal ML Architecture Patterns:
- **Pattern 2**: Protocol-First Interface Design
- **Pattern 3**: Hot/Cold Path Separation with strict performance SLAs
- **Pattern 5**: Centralized Metrics Bootstrap for monitoring
"""

from __future__ import annotations


__all__ = [
    "CorrelationState",
    "EWMABetaState",
    "ZScoreSpreadState",
    "compute_correlation_batch",
    "compute_correlation_incremental",
    "compute_ewma_beta_batch",
    "compute_ewma_beta_incremental",
    "compute_zscore_spread_batch",
    "compute_zscore_spread_incremental",
]


def __getattr__(name: str) -> object:
    """Lazy import implementation to avoid circular imports."""
    if name == "CorrelationState":
        from ml.features.cross_asset.state import CorrelationState

        return CorrelationState
    elif name == "EWMABetaState":
        from ml.features.cross_asset.state import EWMABetaState

        return EWMABetaState
    elif name == "ZScoreSpreadState":
        from ml.features.cross_asset.state import ZScoreSpreadState

        return ZScoreSpreadState
    elif name == "compute_correlation_incremental":
        from ml.features.cross_asset.correlation import compute_correlation_incremental

        return compute_correlation_incremental
    elif name == "compute_correlation_batch":
        from ml.features.cross_asset.correlation import compute_correlation_batch

        return compute_correlation_batch
    elif name == "compute_ewma_beta_incremental":
        from ml.features.cross_asset.beta import compute_ewma_beta_incremental

        return compute_ewma_beta_incremental
    elif name == "compute_ewma_beta_batch":
        from ml.features.cross_asset.beta import compute_ewma_beta_batch

        return compute_ewma_beta_batch
    elif name == "compute_zscore_spread_incremental":
        from ml.features.cross_asset.spreads import compute_zscore_spread_incremental

        return compute_zscore_spread_incremental
    elif name == "compute_zscore_spread_batch":
        from ml.features.cross_asset.spreads import compute_zscore_spread_batch

        return compute_zscore_spread_batch
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
