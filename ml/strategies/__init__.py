"""
ML-driven trading strategies for Nautilus Trader.

This module provides production-ready ML trading strategies that consume signals from
ML actors and execute trades based on machine learning predictions. All strategies
follow Nautilus Trader's architecture patterns and universal ML patterns.

## Strategy Hierarchy

### Base Classes
- `BaseMLStrategy`: Abstract base class providing common ML strategy functionality
- `SimpleMLStrategy`: Basic concrete implementation for single-model strategies

### Production Strategies
- `MLTradingStrategy`: Full-featured strategy with multi-model support
- `MultiModelMLStrategy`: Advanced strategy with dynamic model weighting

## Hot Path Performance

All ML strategies are designed for real-time trading with <5ms P99 latency requirements:
- Pre-allocated arrays for feature computation
- Zero allocations in hot path methods (on_bar, on_data, etc.)
- Circuit breaker patterns for external dependency failures
- Progressive fallback chains for resilience

## Configuration

Strategies use `MLStrategyConfig` and its variants for type-safe configuration:
- Risk management parameters (position sizing, stop loss, take profit)
- Model filtering and aggregation settings
- Strategy store persistence options
- Dry run execution controls

## Usage Examples

### Single Model Strategy
```python
from ml.strategies import MLTradingStrategy
from ml.config.base import MLStrategyConfig

config = MLStrategyConfig(
    strategy_id="MLStrategy-001",
    instrument_id=InstrumentId.from_str("EUR/USD.SIM"),
    ml_signal_source="MLSignalActor-001",
    position_size_pct=0.02,
    min_confidence=0.7,
    execute_trades=True,  # Set to False for dry run
)

strategy = MLTradingStrategy(config)
```

### Multi-Model Strategy with Aggregation
```python
from ml.strategies import MultiModelMLStrategy
from ml.config.base import MultiModelStrategyConfig

config = MultiModelStrategyConfig(
    strategy_id="MultiModel-001",
    instrument_id=InstrumentId.from_str("BTC/USD.SIM"),
    ml_signal_source="MLSignalActor-001",
    target_model_ids=["model_a", "model_b", "model_c"],
    aggregation_mode="weighted_average",
    model_weights={"model_a": 0.5, "model_b": 0.3, "model_c": 0.2},
    required_models=2,
    execute_trades=False,  # Dry run mode
)

strategy = MultiModelMLStrategy(config)
```

### Simple Strategy for Testing
```python
from ml.strategies import SimpleMLStrategy
from ml.config.base import MLStrategyConfig

config = MLStrategyConfig(
    strategy_id="SimpleStrategy-001",
    instrument_id=InstrumentId.from_str("GBP/USD.SIM"),
    ml_signal_source="MLSignalActor-001",
    position_size_pct=0.01,
    min_confidence=0.5,
)

strategy = SimpleMLStrategy(config)
```

## Integration with ML Pipeline

### Signal Consumption
All strategies consume `MLSignal` objects from ML actors:
- Automatic filtering by model_id if configured
- Confidence threshold validation
- Multi-model aggregation support
- Signal history tracking for analysis

### Store Integration
Strategies automatically integrate with ML stores when configured:
- **StrategyStore**: Persists trading decisions and risk metrics
- **FeatureStore**: Access to feature values for analysis
- **ModelStore**: Model performance tracking
- **DataStore**: Unified data access facade

### Metrics and Monitoring
All strategies expose Prometheus metrics for monitoring:
- `ml_signals_received_total`: Total signals processed
- `ml_trades_executed_total`: Total trades executed
- `ml_signal_to_trade_latency_seconds`: Signal processing latency
- `ml_position_count`: Current position count
- `ml_strategy_decisions_persisted_total`: Decisions stored

## Architecture Compliance

All strategies in this module follow the Universal ML Architecture Patterns:

1. **Pattern 1**: Automatic 4-store + 4-registry integration via `BaseMLInferenceActor`
2. **Pattern 2**: Protocol-first interfaces for type safety and testing
3. **Pattern 3**: Hot/cold path separation with performance budgets
4. **Pattern 4**: Progressive fallback chains for resilience
5. **Pattern 5**: Centralized metrics bootstrap (no direct prometheus imports)

## Safety Features

### Risk Management
- Position sizing based on account balance percentages
- Configurable stop loss and take profit levels
- Maximum position limits
- Circuit breakers for external service failures

### Execution Controls
- `execute_trades` flag for dry run testing
- Signal confidence thresholds
- Model filtering and validation
- Order execution with proper reduce-only flags

### Data Validation
- Instrument ID validation
- Timestamp validation for signal freshness
- Model prediction range validation
- Configuration validation on startup

## Testing and Development

### Dry Run Mode
Set `execute_trades=False` to run strategies without placing actual orders:
- All signal processing and decision logic executes normally
- Risk metrics and performance tracking remain active
- Strategy decisions are persisted to stores
- Metrics are updated for monitoring
- No actual orders submitted to brokers

### Dummy Stores
Use `use_dummy_stores=True` for testing without PostgreSQL:
- In-memory store implementations
- No persistence between runs
- Suitable for unit tests and local development
- Automatic fallback when database unavailable

### Performance Testing
All strategies include built-in performance monitoring:
- Hot path latency tracking
- Memory allocation monitoring
- Signal processing throughput metrics
- Store operation timing

## Migration from Legacy Code

When upgrading from older ML strategy implementations:

1. Update imports: `from ml.strategies import MLTradingStrategy`
2. Use new configuration classes: `MLStrategyConfig` instead of dict configs
3. Enable store integration: `use_strategy_store=True`
4. Add dry run controls: `execute_trades=False` for testing
5. Update metrics bootstrap: Remove direct prometheus imports

## See Also

- `ml.actors`: ML signal generation and inference
- `ml.config.base`: Configuration classes and validation
- `ml.stores`: Data persistence and retrieval
- `ml.common.protocols`: Component interfaces and protocols

"""

# Position sizing and risk management
from ml.strategies.analytics import AnalyticsConfig
from ml.strategies.analytics import PerformanceTracker
from ml.strategies.analytics import SignalRecord
from ml.strategies.base import BaseMLStrategy
from ml.strategies.base import SimpleMLStrategy
from ml.strategies.execution import ExecutionConfig
from ml.strategies.execution import OrderExecutor
from ml.strategies.ml_strategy import MLTradingStrategy
from ml.strategies.ml_strategy import MultiModelMLStrategy
from ml.strategies.portfolio import PortfolioConfig
from ml.strategies.portfolio import PortfolioManager

# Type protocols
from ml.strategies.protocols import OrderExecutorProtocol
from ml.strategies.protocols import PerformanceTrackerProtocol
from ml.strategies.protocols import PortfolioManagerProtocol
from ml.strategies.protocols import PositionSizerProtocol
from ml.strategies.protocols import RiskManagerProtocol
from ml.strategies.risk import RiskConfig
from ml.strategies.risk import RiskManager
from ml.strategies.sizing import CompositeSizer
from ml.strategies.sizing import KellySizer
from ml.strategies.sizing import SizingConfig
from ml.strategies.sizing import VolatilitySizer


__all__ = [
    "AnalyticsConfig",
    "BaseMLStrategy",
    "CompositeSizer",
    "ExecutionConfig",
    "KellySizer",
    "MLTradingStrategy",
    "MultiModelMLStrategy",
    "OrderExecutor",
    "OrderExecutorProtocol",
    "PerformanceTracker",
    "PerformanceTrackerProtocol",
    "PortfolioConfig",
    "PortfolioManager",
    "PortfolioManagerProtocol",
    "PositionSizerProtocol",
    "RiskConfig",
    "RiskManager",
    "RiskManagerProtocol",
    "SignalRecord",
    "SimpleMLStrategy",
    "SizingConfig",
    "VolatilitySizer",
]
