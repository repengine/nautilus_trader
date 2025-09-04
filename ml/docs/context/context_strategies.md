# ML Strategies Context Documentation

## Executive Summary

The ml/strategies/ directory implements the core ML-driven trading strategy framework for Nautilus Trader. This framework provides a sophisticated architecture for integrating machine learning signals into trading execution while maintaining hot path performance requirements. The system supports both single-model and multi-model strategies with advanced aggregation capabilities, comprehensive performance tracking, and seamless integration with Nautilus Trader's execution engine.

Operational notes:

- Persistence: Strategy signals are stored with UNIX nanosecond timestamps; the `StrategyStore` will normalize smaller units to ns with a warning. See `context_stores.md` → "Timestamp Policy & Normalization".
- DB readiness: Ensure canonical migrations are applied and run DB preflight checks prior to deployment. See `context_deployment.md`.

## Directory Structure

```
ml/strategies/
├── __init__.py                    # Public API exports
├── base.py                        # BaseMLStrategy and SimpleMLStrategy
├── ml_strategy.py                 # MLTradingStrategy and MultiModelMLStrategy
└── META_LEARNING_ARCHITECTURE.md  # Future meta-learning design document
```

## Public API

The following classes are exported from `ml.strategies`:

- **`BaseMLStrategy`**: Abstract base class for all ML strategies
- **`SimpleMLStrategy`**: Basic binary signal trading implementation
- **`MLTradingStrategy`**: Production-ready strategy with full features
- **`MultiModelMLStrategy`**: Advanced multi-model ensemble strategy

All strategies inherit from `BaseMLStrategy` and extend Nautilus Trader's `Strategy` class.

## Base Class Hierarchy

### BaseMLStrategy (Abstract Base Class)

**Location**: `/home/nate/projects/nautilus_trader/ml/strategies/base.py`

The foundational abstract class that all ML strategies inherit from. Extends Nautilus Trader's `Strategy` class with ML-specific capabilities.

**Enhanced Production Features:**

- **✨ ENHANCEMENT:** DRY RUN MODE by default with execute_trades=False for safety
- **📝 ADDITION:** Signal consumption from MLSignalActor via ml_signal_source
- **📝 ADDITION:** Risk management parameters via environment variables
- **📝 ADDITION:** Strategy store persistence with configurable batch sizes
- **📝 ADDITION:** Final statistics reporting on shutdown

**Key Features**:

- **Signal Subscription**: Automatically subscribes to MLSignal data types (with optional client_id filtering)
- **Signal Filtering**: Filters signals by model_id, confidence thresholds, and instrument
- **Signal Aggregation**: Supports multiple models with voting and weighted average modes
- **Position Management**: Handles position sizing, stop loss, and take profit
- **Performance Tracking**: Comprehensive metrics via Prometheus
- **Store Integration**: Optional StrategyStore integration (configured via `use_strategy_store`)
- **Hot Path Optimization**: Pre-allocated buffers and minimal dynamic allocations
- **Dry Run Mode**: Supports `execute_trades=False` for risk-free testing

**Configuration Support**:

- Target model filtering (`target_model_ids`)
- Aggregation modes (`aggregation_mode`: "voting", "weighted_average")
- Signal timing windows (`time_window_ms`)
- Model performance tracking (`track_performance`)
- Dynamic model weighting (`model_weights`)
- Client ID filtering (`signal_client_id`)
- Minimum confidence threshold (`min_confidence`)
- Position size percentage (`position_size_pct`)
- Maximum positions (`max_positions`)
- Stop loss percentage (`stop_loss_pct`)
- Take profit percentage (`take_profit_pct`)
- Dry run mode (`execute_trades`)
- Signal persistence control (`persist_all_signals`)

**Store Integration**:

```python
# Optional StrategyStore initialized based on configuration
self.strategy_store: StrategyStore | None = None
if self._config.use_strategy_store:
    store_config = self._config.strategy_store_config or {}
    self.strategy_store = StrategyStore(
        connection_string=store_config.get("connection_string", ...),
        batch_size=store_config.get("batch_size", 100),
        flush_interval_ms=store_config.get("flush_interval_ms", 1000),
        clock=self.clock,
    )
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
- Implements `on_order_filled` for position tracking
- **⚠️ CORRECTION:** Position alignment check prevents unnecessary trades
- **✨ ENHANCEMENT:** Intelligent position reversal with error handling

**Key Methods**:

- `_process_ml_signal()`: Implements simple binary trading logic
- `on_order_filled()`: Updates position and pending order counts

**Usage**: Ideal for single-model strategies with straightforward binary classification signals.

### MLTradingStrategy (Production Implementation)

**Location**: `/home/nate/projects/nautilus_trader/ml/strategies/ml_strategy.py` (lines 23-345)

Production-ready strategy with advanced multi-model support and comprehensive decision persistence.

**Enhanced Features**:

- **Decision Persistence**: All trading decisions persisted to StrategyStore with comprehensive context
- **Dry Run Mode**: `execute_trades=False` for risk-free testing in production
- **Risk Management**: Advanced risk metrics calculation and tracking
- **Performance Attribution**: Per-model performance tracking via `_order_to_model` mapping
- **Position Reversal Logic**: Sophisticated position management with reversal detection
- **✨ ENHANCEMENT:** DRY RUN MODE by default with execute_trades=False for safety
- **📝 ADDITION:** Risk management parameters via environment variables
- **📝 ADDITION:** Strategy store persistence with configurable batch sizes
- **📝 ADDITION:** Test-safe initialization with graceful error handling

**Decision Types**:

- `BUY`: Enter long position
- `SELL`: Enter short position
- `HOLD`: Maintain current position (optionally persisted based on `persist_all_signals`)

**Key Methods**:

- `_process_ml_signal()`: Main signal processing with decision persistence
- `_enter_position()`: Enter new position with dry-run support
- `_reverse_position()`: Handle position reversal with proper closing/opening
- `_should_reverse_position()`: Logic to determine if reversal needed
- `_track_trade_entry()`: Map orders to models for performance tracking
- `on_order_filled()`: Track model performance on order fills

**Execution Flow**:

1. Signal reception and validation
2. Position analysis and decision calculation
3. Risk metrics computation (confidence, prediction, position counts)
4. Decision persistence to StrategyStore
5. Trade execution (if `execute_trades=True`)
6. Performance tracking and metrics update

### MultiModelMLStrategy (Advanced Multi-Model)

**Location**: `/home/nate/projects/nautilus_trader/ml/strategies/ml_strategy.py` (lines 347-426)

Extends MLTradingStrategy with dynamic model weighting and performance-based adaptation.

**Advanced Capabilities**:

- **Dynamic Weighting**: Model weights adapt based on historical performance
- **Performance-Based Allocation**: Automatically adjusts model influence using accuracy and profit metrics
- **Automatic Performance Tracking**: Enables `track_performance=True` by default
- **Meta-Learning Ready**: Foundation for future meta-learning implementations

**Key Methods**:

- `_get_dynamic_model_weights()`: Calculates weights based on model accuracy and profit per trade
- `_aggregate_signal()`: Overrides parent to use dynamic weights when `use_dynamic_weights=True`

**Weight Calculation Formula**:

```python
weight = accuracy * (1.0 + np.tanh(profit_per_trade / 100.0))
```

- Combines accuracy with normalized profit per trade
- Minimum weight of 0.1 to prevent complete exclusion
- Weights are normalized to sum to 1.0

**Dynamic Weighting Enhancements:**

- **✨ ENHANCEMENT:** Performance-Based Weighting: Automatic model weight adjustment based on historical performance
- **📝 ADDITION:** Adaptive Learning: Model weights evolve with observed accuracy and profitability
- **📝 ADDITION:** Minimum Weight Constraints: Prevents complete model exclusion (0.1 minimum weight)
- **📝 ADDITION:** Normalized Weighting: Ensures total weights sum to 1.0 for proper aggregation

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

The `_aggregate_signal()` method in BaseMLStrategy handles multi-model signal aggregation with time-window validation.

**Voting Aggregation** (default):

```python
# Simple majority vote
bullish = sum(1 for s in self._model_signals.values() if s.prediction > 0.5)
bearish = len(self._model_signals) - bullish
action = "BUY" if bullish > bearish else "SELL"
confidence = max(s.confidence for s in self._model_signals.values())
# Creates aggregated signal with prediction 0.8 for BUY, 0.2 for SELL
```

**Weighted Average Aggregation** (`conflict_resolution="weighted_average"`):

```python
# Model-weighted prediction combination
for mid, sig in self._model_signals.items():
    weight = self.model_weights.get(mid, 1.0)  # Default weight 1.0
    weighted_sum += weight * sig.prediction
    total_weight += weight

weighted_pred = weighted_sum / total_weight
avg_confidence = np.mean([s.confidence for s in self._model_signals.values()])
```

**Key Features**:

- Requires minimum number of models (`required_models` parameter)
- Validates signals are within time window (`time_window_ms` parameter)
- Automatically clears old signals outside time window
- Creates aggregated MLSignal with metadata tracking source models
- Calls both stub methods (`_make_decision`, `_execute_trade`) and `_process_ml_signal`

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

ML strategies optionally integrate with StrategyStore for decision persistence and analysis.

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

### Store Configuration

**Store Initialization**:

```python
# Store initialization in BaseMLStrategy.__init__
self.strategy_store: StrategyStore | None = None
if self._config.use_strategy_store:
    store_config = self._config.strategy_store_config or {}
    self.strategy_store = StrategyStore(
        connection_string=store_config.get(
            "connection_string",
            "postgresql://postgres:postgres@localhost:5432/nautilus"
        ),
        batch_size=store_config.get("batch_size", 100),
        flush_interval_ms=store_config.get("flush_interval_ms", 1000),
        clock=self.clock,
    )
```

**Flush on Stop**:

- StrategyStore is automatically flushed when strategy stops
- Ensures all pending decisions are persisted
- Handles exceptions gracefully to prevent data loss

## Common Usage Patterns

### Single Model Strategy

```python
from ml.strategies import SimpleMLStrategy
from ml.config.base import MLStrategyConfig

# Basic single-model configuration
config = MLStrategyConfig(
    instrument_id=InstrumentId.from_str("EURUSD.SIM"),
    ml_signal_source="ml_signal_actor",
    position_size_pct=0.1,  # 10% of account
    min_confidence=0.7,
    max_positions=1,
    stop_loss_pct=0.02,  # 2% stop loss
    take_profit_pct=0.04,  # 4% take profit
    execute_trades=True,
)

strategy = SimpleMLStrategy(config)
```

### Multi-Model Ensemble

```python
from ml.strategies import MultiModelMLStrategy

# Multi-model configuration with aggregation
config = MLStrategyConfig(
    instrument_id=InstrumentId.from_str("BTCUSDT.BINANCE"),
    target_model_ids=["momentum", "mean_revert", "microstructure"],
    aggregation_mode="weighted_average",
    required_models=2,  # Need at least 2 models
    time_window_ms=500,  # 500ms aggregation window
    model_weights={
        "momentum": 0.4,
        "mean_revert": 0.3,
        "microstructure": 0.3,
    },
    use_dynamic_weights=True,  # Enable adaptive weighting
    track_performance=True,
    use_strategy_store=True,
    execute_trades=True,
)

strategy = MultiModelMLStrategy(config)
```

### Dry-Run Testing

```python
# Production dry-run configuration
config = MLStrategyConfig(
    instrument_id=InstrumentId.from_str("EURUSD.SIM"),
    ml_signal_source="ml_signal_actor",
    position_size_pct=0.05,
    min_confidence=0.8,  # Higher threshold for testing
    execute_trades=False,  # DRY RUN MODE
    use_strategy_store=True,  # Still persist decisions
    persist_all_signals=True,  # Persist even HOLD decisions
    strategy_store_config={
        "batch_size": 50,
        "flush_interval_ms": 2000,
    },
)

strategy = MLTradingStrategy(config)
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
   - Complete abstract base class (BaseMLStrategy) with ML signal handling
   - Production-ready simple strategy implementation (SimpleMLStrategy)
   - Production ML trading strategy (MLTradingStrategy)
   - Multi-model strategy with dynamic weighting (MultiModelMLStrategy)
   - Comprehensive configuration system via MLStrategyConfig

2. **Multi-Model Support**
   - Signal aggregation (voting, weighted average)
   - Model filtering and routing via target_model_ids
   - Performance-based dynamic weighting
   - Time-window based signal aggregation
   - Model performance tracking per model_id

3. **Store Integration**
   - Optional StrategyStore integration (configurable)
   - Comprehensive decision persistence with risk metrics
   - Batched writes for performance
   - Automatic flush on strategy stop

4. **Performance Monitoring**
   - Complete Prometheus metrics suite (7 metrics)
   - Hot path latency tracking
   - Model performance attribution via order tracking
   - Strategy decision persistence metrics

5. **Production Features**
   - Dry run mode (`execute_trades=False`) for risk-free testing
   - Position sizing based on account balance
   - Stop loss and take profit support
   - Position reversal logic
   - Client ID filtering for signal sources

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

## Helper Methods in BaseMLStrategy

### Trading Utilities

**`_place_market_order()`**: Places market orders with proper initialization

- Supports reduce_only flag for position closing
- Increments pending_orders and trades_executed counters
- Returns ClientOrderId for tracking

**`_place_stop_loss()`**: Places stop-loss orders

- Automatically sets reduce_only=True
- Uses StopMarketOrder with configurable trigger price
- Logs order placement for audit trail

**`_get_current_position()`**: Retrieves current open position

- Searches across all venues for the configured instrument
- Returns first open position or None
- Used for position management decisions

### Performance Tracking

**`_update_model_performance()`**: Tracks per-model trading performance

- Records total trades, profit, wins/losses
- Calculates running accuracy percentage
- Used by MultiModelMLStrategy for dynamic weighting

### Stub Methods for Compatibility

**`_process_signal()`**, **`_make_decision()`**, **`_execute_trade()`**: Empty stubs

- Maintained for backward compatibility with tests
- Called during aggregation for extensibility
- Can be overridden in subclasses if needed

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

# Pre-allocated buffers for aggregation
self._signal_buffer: dict[str, MLSignal] = {}
self._model_signals: dict[str, MLSignal] = {}
self._model_performance: dict[str, dict[str, Any]] = {}
```

### Error Handling

**Comprehensive Position Sizing**:

```python
def _calculate_position_size(self) -> Quantity | None:
    """Calculate position size with comprehensive error handling."""

    # Validate instrument exists
    instrument = self.cache.instrument(self._config.instrument_id)
    if instrument is None:
        self.log.error(f"Cannot calculate position size: Instrument {self._config.instrument_id} not found")
        return None

    # Validate account exists
    account = self.cache.account_for_venue(instrument.venue)
    if account is None:
        self.log.error(f"Cannot calculate position size: No account found for venue {instrument.venue}")
        return None

    # Get price data with fallbacks
    last_tick = self.cache.trade_tick(self._config.instrument_id)
    if last_tick is not None:
        current_price = float(last_tick.price.as_double())
    else:
        quote_tick = self.cache.quote_tick(self._config.instrument_id)
        if quote_tick is not None:
            current_price = (float(quote_tick.bid_price.as_double()) +
                           float(quote_tick.ask_price.as_double())) / 2.0
        else:
            self.log.error("Cannot calculate position size: No price data available")
            return None

    # Calculate with proper rounding and minimum size enforcement
    return Quantity.from_str(str(quantity_value))
```

### Metrics Initialization

**Singleton Pattern for Prometheus Metrics**:

```python
# Module-level singleton metrics to avoid registry collisions
_metrics_initialized = False

def _initialize_metrics() -> None:
    global _metrics_initialized, ml_signals_received, ...

    if _metrics_initialized:
        return

    if HAS_PROMETHEUS:
        # Check if metrics already exist in registry
        existing_names = set(REGISTRY._names_to_collectors.keys())

        if METRIC_SIGNALS_RECEIVED_TOTAL not in existing_names:
            ml_signals_received = Counter(...)
        else:
            ml_signals_received = cast(Counter, REGISTRY._names_to_collectors[...])

    _metrics_initialized = True
```

## Future Architecture Evolution

The strategy framework is designed for evolution toward meta-learning and advanced multi-model orchestration. The `META_LEARNING_ARCHITECTURE.md` document outlines plans for:

1. **Dynamic Model Orchestration**: ML-driven model weight allocation
2. **Market Regime Detection**: Context-aware model selection
3. **Reinforcement Learning**: RL-based trading policy optimization
4. **Bayesian Ensemble Methods**: Probabilistic model combination

The current implementation provides a solid foundation for these advanced capabilities while maintaining production stability and performance requirements.

## Strategy Selection Guide

### When to Use Each Strategy

**SimpleMLStrategy**:

- Single model deployments
- Binary classification signals
- Basic position management needs
- Proof of concept implementations

**MLTradingStrategy**:

- Production deployments
- Need for comprehensive decision persistence
- Dry-run testing requirements
- Model performance tracking needed
- Position reversal support required

**MultiModelMLStrategy**:

- Multiple ML models in ensemble
- Dynamic weight adjustment needed
- Performance-based model selection
- Advanced multi-model orchestration

## Key Implementation Differences

| Feature | SimpleMLStrategy | MLTradingStrategy | MultiModelMLStrategy |
|---------|-----------------|-------------------|---------------------|
| **Signal Processing** | Basic binary | Advanced with persistence | Dynamic weighting |
| **Decision Persistence** | No | Yes (comprehensive) | Yes (enhanced) |
| **Dry Run Support** | No | Yes | Yes |
| **Position Reversal** | Basic | Advanced logic | Advanced logic |
| **Performance Tracking** | No | Yes (per-model) | Yes (with dynamic weights) |
| **Model Aggregation** | No | Via base class | Enhanced with adaptation |
| **Production Ready** | Development | Yes | Yes |

## Conclusion

The ml/strategies/ directory provides a comprehensive, production-ready framework for ML-driven trading strategies. The architecture successfully balances sophistication with performance, providing powerful multi-model capabilities while maintaining the sub-5ms latency requirements of high-frequency trading. The optional store integrations ensure complete audit trails and enable sophisticated post-trade analysis and model improvement workflows.

Key strengths:

- Flexible architecture supporting single to multi-model deployments
- Production-grade features (dry-run, persistence, monitoring)
- Performance optimized for hot-path execution
- Extensible design ready for meta-learning evolution
- Comprehensive error handling and logging

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
