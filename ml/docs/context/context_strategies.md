# ML Strategies Context Documentation

## Executive Summary

The ml/strategies/ directory implements the core ML-driven trading strategy framework for Nautilus Trader. This framework provides a sophisticated architecture for integrating machine learning signals into trading execution while maintaining hot path performance requirements. The system supports both single-model and multi-model strategies with advanced aggregation capabilities, comprehensive performance tracking, and seamless integration with Nautilus Trader's execution engine.

## Directory Structure

```
ml/strategies/
├── __init__.py                    # Public API exports
├── base.py                        # BaseMLStrategy and SimpleMLStrategy
├── ml_strategy.py                 # MLTradingStrategy and MultiModelMLStrategy
└── META_LEARNING_ARCHITECTURE.md  # Future meta-learning design document
```

## Base Class Hierarchy

### BaseMLStrategy (Abstract Base Class)

**Location**: `/home/nate/projects/nautilus_trader/ml/strategies/base.py`

The foundational abstract class that all ML strategies inherit from. Extends Nautilus Trader's `Strategy` class with ML-specific capabilities.

**Key Features**:

- **Signal Subscription**: Automatically subscribes to MLSignal data types
- **Signal Filtering**: Filters signals by model_id, confidence thresholds, and instrument
- **Signal Aggregation**: Supports multiple models with voting and weighted average modes
- **Position Management**: Handles position sizing, stop loss, and take profit
- **Performance Tracking**: Comprehensive metrics via Prometheus
- **Store Integration**: Mandatory integration with three stores (FeatureStore, ModelStore, StrategyStore)
- **Hot Path Optimization**: Pre-allocated buffers and minimal dynamic allocations

**Configuration Support**:

- Target model filtering (`target_model_ids`)
- Aggregation modes (`aggregation_mode`: "voting", "weighted_average")
- Signal timing windows (`time_window_ms`)
- Model performance tracking (`track_performance`)
- Dynamic model weighting (`model_weights`)

**Store Integration**:

```python
# Mandatory stores initialized automatically
self.strategy_store: StrategyStore | None = None
if self._config.use_strategy_store:
    self.strategy_store = StrategyStore(...)
```

**Prometheus Metrics**:

- `nautilus_ml_signals_received_total`: Count of signals received by strategy
- `nautilus_ml_trades_executed_total`: Count of trades executed by strategy
- `nautilus_ml_signal_to_trade_latency_seconds`: Latency from signal to execution
- `nautilus_ml_position_count`: Current number of open positions
- `nautilus_ml_strategy_decisions_persisted_total`: Decisions written to StrategyStore
- `nautilus_ml_strategy_store_write_latency_seconds`: StrategyStore write latency
- `nautilus_ml_strategy_store_batch_size`: Current buffer size in StrategyStore

### SimpleMLStrategy (Concrete Implementation)

**Location**: `/home/nate/projects/nautilus_trader/ml/strategies/base.py` (lines 883-968)

A basic implementation demonstrating binary signal trading logic.

**Trading Logic**:

- Long position: `signal.prediction > 0.5`
- Short position: `signal.prediction <= 0.5`
- Position reversal on opposite signals
- Basic position management (one position at a time)

**Usage**: Ideal for single-model strategies with straightforward binary classification signals.

### MLTradingStrategy (Production Implementation)

**Location**: `/home/nate/projects/nautilus_trader/ml/strategies/ml_strategy.py`

Production-ready strategy with advanced multi-model support and comprehensive decision persistence.

**Enhanced Features**:

- **Decision Persistence**: All trading decisions persisted to StrategyStore
- **Dry Run Mode**: `execute_trades=False` for risk-free testing in production
- **Risk Management**: Advanced risk metrics calculation and tracking
- **Performance Attribution**: Per-model performance tracking and attribution
- **Position Reversal Logic**: Sophisticated position management with reversal detection

**Decision Types**:

- `BUY`: Enter long position
- `SELL`: Enter short position
- `HOLD`: Maintain current position (optionally persisted)

**Execution Flow**:

1. Signal reception and validation
2. Position analysis and decision calculation
3. Risk metrics computation
4. Decision persistence to StrategyStore
5. Trade execution (if `execute_trades=True`)
6. Performance tracking and metrics update

### MultiModelMLStrategy (Advanced Multi-Model)

**Location**: `/home/nate/projects/nautilus_trader/ml/strategies/ml_strategy.py` (lines 347-426)

Extends MLTradingStrategy with dynamic model weighting and performance-based adaptation.

**Advanced Capabilities**:

- **Dynamic Weighting**: Model weights adapt based on historical performance
- **Performance-Based Allocation**: Automatically adjusts model influence
- **Conflict Resolution**: Sophisticated handling of conflicting signals
- **Meta-Learning Ready**: Foundation for future meta-learning implementations

## Signal Generation Architecture

### MLSignal Data Structure

**Source**: `ml.actors.base.MLSignal`

Core data structure for ML predictions:

```python
class MLSignal(Data):
    instrument_id: InstrumentId
    model_id: str
    prediction: float      # Model prediction (0.0 to 1.0)
    confidence: float      # Confidence score (0.0 to 1.0)
    metadata: dict[str, Any]
    ts_event: int         # Event timestamp (nanoseconds)
    ts_init: int          # Initialization timestamp
```

### Signal Flow Pipeline

```
ML Actor (Model Inference)
    ↓
MLSignal Publication
    ↓
Strategy Signal Reception (`on_data`)
    ↓
Signal Filtering (model_id, confidence, instrument)
    ↓
Signal Aggregation (if multi-model)
    ↓
Trading Decision Logic (`_process_ml_signal`)
    ↓
Decision Persistence (StrategyStore)
    ↓
Trade Execution (if enabled)
    ↓
Performance Tracking & Metrics
```

### Signal Aggregation Methods

**Voting Aggregation**:

```python
# Simple majority vote
bullish = sum(1 for s in signals if s.prediction > 0.5)
bearish = len(signals) - bullish
action = "BUY" if bullish > bearish else "SELL"
```

**Weighted Average Aggregation**:

```python
# Model-weighted prediction combination
weighted_sum = sum(weight * signal.prediction
                  for model_id, signal in signals.items()
                  for weight in [model_weights.get(model_id, 1.0)])
prediction = weighted_sum / total_weight
```

## Hot Path Performance Requirements

### Performance Constraints

- **Maximum Latency**: 5ms P99 for signal processing
- **Memory Allocation**: Minimal dynamic allocations in hot path
- **Pre-allocation**: Signal buffers and feature arrays pre-allocated
- **No Blocking I/O**: Database writes are batched and flushed periodically (or on stop)

### Hot Path Optimizations

**Signal Processing**:

```python
# Pre-allocated signal history
self._signal_history: deque[MLSignal] = deque(maxlen=config.history_size)

# Model signal buffer for aggregation
self._signal_buffer: dict[str, MLSignal] = {}
self._model_signals: dict[str, MLSignal] = {}
```

**Position Sizing Cache**:

```python
# Cache instrument and account data to avoid repeated lookups
instrument = self.cache.instrument(self._config.instrument_id)
account = self.cache.account_for_venue(instrument.venue)
```

**Metrics Update**:

```python
# Prometheus metrics are singleton instances to avoid registry overhead
if self._signals_received_metric:
    self._signals_received_metric.labels(
        strategy_id=str(self.id),
        signal_source=signal_source
    ).inc()
```

## Store Integration Architecture

All ML strategies MUST integrate with the three mandatory stores to ensure proper data persistence and system consistency.

### StrategyStore Integration

**Purpose**: Persist all trading decisions for audit, analysis, and model improvement.

**Persistence Logic**:

```python
def _persist_strategy_decision(
    self,
    signal: MLSignal,
    decision_type: str,  # "BUY", "SELL", "HOLD"
    position_size: Quantity | None = None,
    risk_metrics: dict[str, float] | None = None,
    execution_params: dict[str, Any] | None = None,
) -> None:
    """Persist comprehensive decision context."""

    self.strategy_store.write_signal(
        strategy_id=str(self.id),
        instrument_id=str(signal.instrument_id),
        signal_type=decision_type,
        strength=float(signal.confidence),
        model_predictions={model_id: prediction, ...},
        risk_metrics=risk_metrics,
        execution_params=execution_params,
        ts_event=signal.ts_event,
        is_live=not self.cache.is_backtesting,
    )
```

**Stored Data**:

- Strategy decisions (BUY/SELL/HOLD)
- Model predictions and confidence scores
- Risk metrics (account balance, position count, etc.)
- Execution parameters (stop loss, take profit, position size)
- Market context and timing information

### Automatic Store Initialization

**Base Class Initialization**:

```python
# Store initialization in BaseMLStrategy.__init__
if self._config.use_strategy_store:
    store_config = self._config.strategy_store_config or {}
    self.strategy_store = StrategyStore(
        connection_string=store_config.get("connection_string", ...),
        batch_size=store_config.get("batch_size", 100),
        flush_interval_ms=store_config.get("flush_interval_ms", 1000),
        clock=self.clock,
    )
```

## Backtesting Integration

### Backtesting Compatibility

All strategies are fully compatible with Nautilus Trader's backtesting engine:

**Data Replay**:

- Strategies receive historical MLSignal data during backtest
- All signal processing logic remains identical to live trading
- Store operations work with historical timestamps

**Performance Attribution**:

- Per-model performance tracking works in backtest mode
- Historical decisions can be analyzed post-backtest
- Risk metrics are calculated using simulated account state

### Backtest Configuration

```python
# Example backtest configuration
config = MLStrategyConfig(
    instrument_id=InstrumentId.from_str("EURUSD.SIM"),
    ml_signal_source="ml_signal_actor",
    position_size_pct=0.1,
    min_confidence=0.7,
    execute_trades=True,  # Full execution in backtest
    use_strategy_store=True,
)
```

## Current Implementation Status

### ✅ Completed Features

1. **Base Strategy Framework**
   - Complete abstract base class with ML signal handling
   - Production-ready simple strategy implementation
   - Comprehensive configuration system

2. **Multi-Model Support**
   - Signal aggregation (voting, weighted average)
   - Model filtering and routing
   - Performance-based dynamic weighting

3. **Store Integration**
   - Mandatory StrategyStore integration
   - Comprehensive decision persistence
   - Async batched writes for performance

4. **Performance Monitoring**
   - Complete Prometheus metrics suite
   - Hot path latency tracking
   - Model performance attribution

5. **Production Features**
   - Dry run mode for risk-free testing
   - Circuit breaker patterns
   - Health monitoring integration

### 🔄 In Progress Features

1. **Meta-Learning Architecture**
   - Design document completed (`META_LEARNING_ARCHITECTURE.md`)
   - Implementation planned for future releases
   - Framework ready for meta-model integration

### 📋 Planned Features

1. **Advanced Aggregation**
   - Reinforcement learning-based orchestration
   - Bayesian model combination
   - Market regime-aware weighting

2. **Risk Management Enhancements**
   - Dynamic position sizing based on model agreement
   - Portfolio-level risk management
   - Cross-asset correlation analysis

3. **Online Learning**
   - Continuous model weight adaptation
   - Performance-based model selection
   - Real-time regime detection

## Integration with Nautilus Execution Engine

### Order Management

**Market Orders**:

```python
def _place_market_order(
    self,
    side: OrderSide,
    quantity: Quantity,
    reduce_only: bool = False,
) -> ClientOrderId:
    """Place market order with Nautilus execution engine."""

    order = MarketOrder(
        trader_id=self.trader_id,
        strategy_id=self.id,
        instrument_id=self._config.instrument_id,
        client_order_id=self.cache.client_order_id(),
        order_side=side,
        quantity=quantity,
        # ... additional parameters
    )

    self.submit_order(order)
    return order.client_order_id
```

**Position Management**:

- Automatic position tracking via Nautilus cache
- Support for position reversal and scaling
- Integration with Nautilus risk management

### Event Handling

**Order Fill Processing**:

```python
def on_order_filled(self, event: OrderFilled) -> None:
    """Handle order fills and update strategy state."""
    super().on_order_filled(event)

    # Update position tracking
    self._active_positions = len(self.cache.positions_open())

    # Track model performance if attribution enabled
    if self.track_performance:
        self._update_model_performance(model_id, realized_pnl)
```

## Critical Implementation Details

### Thread Safety

**Signal Buffer Management**:

```python
# Thread-safe signal aggregation
def _aggregate_signal(self, signal: MLSignal) -> None:
    model_id = getattr(signal, "model_id", None) or signal.metadata.get("model_id")
    if model_id:
        self._model_signals[model_id] = signal  # Atomic update
```

### Memory Management

**Buffer Size Limits**:

```python
# Bounded signal history to prevent memory leaks
self._signal_history: deque[MLSignal] = deque(
    maxlen=config.history_size if hasattr(config, "history_size") else 100
)
```

### Error Handling

**Graceful Degradation**:

```python
def _calculate_position_size(self) -> Quantity | None:
    """Calculate position size with comprehensive error handling."""

    instrument = self.cache.instrument(self._config.instrument_id)
    if instrument is None:
        self.log.error(f"Instrument {self._config.instrument_id} not found")
        return None

    # Continue with calculation...
```

### Configuration Validation

**Startup Validation**:

```python
def __post_init__(self) -> None:
    """Validate configuration on initialization."""
    if not self.model_path and not self.model_id:
        raise ValidationError("Either model_path or model_id must be provided")
```

## Future Architecture Evolution

The strategy framework is designed for evolution toward meta-learning and advanced multi-model orchestration. The `META_LEARNING_ARCHITECTURE.md` document outlines plans for:

1. **Dynamic Model Orchestration**: ML-driven model weight allocation
2. **Market Regime Detection**: Context-aware model selection
3. **Reinforcement Learning**: RL-based trading policy optimization
4. **Bayesian Ensemble Methods**: Probabilistic model combination

The current implementation provides a solid foundation for these advanced capabilities while maintaining production stability and performance requirements.

## Conclusion

The ml/strategies/ directory provides a comprehensive, production-ready framework for ML-driven trading strategies. The architecture successfully balances sophistication with performance, providing powerful multi-model capabilities while maintaining the sub-5ms latency requirements of high-frequency trading. The mandatory store integrations ensure complete audit trails and enable sophisticated post-trade analysis and model improvement workflows.
## Cross-Module References

- **Data Pipeline**: See `context_data.md` for data ingestion and collection
- **Feature Engineering**: See `context_features.md` for feature computation
- **Stores**: See `context_stores.md` for persistence layer
- **Training**: See `context_training.md` for model training pipelines
- **Registry**: See `context_registry.md` for lifecycle management
- **Strategies**: See `context_strategies.md` for trading strategy framework
- **Deployment**: See `context_deployment.md` for containerization
- **Monitoring**: See `context_monitoring.md` for observability
- **Actors**: See `context_actors.md` for inference actors
- **Models**: See `context_models.md` for model implementations
