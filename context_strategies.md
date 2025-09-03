# ML Strategies Module Context

## Overview
The `ml/strategies/` module provides production-ready ML-driven trading strategies that consume signals from ML actors and execute trades based on configurable risk management rules and multi-model aggregation patterns.

## Core Components

### Base Strategy Framework

**base.py** - Base ML Strategy Infrastructure
```python
class BaseMLStrategy(Strategy, ABC):
    """Base class for ML-driven trading strategies."""
```

**Key Features:**
- **📝 ADDITION:** Complete Nautilus Trader Strategy inheritance with ML-specific extensions
- **✨ ENHANCEMENT:** MLSignal subscription system with client-based filtering
- **📝 ADDITION:** Configurable position sizing based on account balance percentage
- **📝 ADDITION:** Comprehensive risk management with stop loss/take profit
- **📝 ADDITION:** Multi-model signal aggregation with time-window correlation
- **📝 ADDITION:** Strategy decision persistence to StrategyStore for backtesting correlation

**Signal Processing Pipeline:**
1. **MLSignal Reception**: DataType subscription with client filtering
2. **Model Filtering**: target_model_ids whitelist support
3. **Confidence Thresholding**: min_confidence filtering
4. **Aggregation Logic**: Optional multi-model consensus
5. **Risk Validation**: Position limits and account balance checks
6. **Trade Execution**: Market orders with optional risk management

### Simple Strategy Implementation

**SimpleMLStrategy** (base.py) - Basic Binary Strategy
```python
class SimpleMLStrategy(BaseMLStrategy):
    """Simple ML strategy that trades based on binary ML signals."""
```

**Trading Logic:**
- **Binary Classification**: prediction > 0.5 → BUY, prediction < 0.5 → SELL
- **Position Management**: Single position per instrument
- **Reversal Strategy**: Switches position direction on opposing signals
- **⚠️ CORRECTION:** Position alignment check prevents unnecessary trades

### Production Strategy Implementation

**ml_strategy.py** - Production ML Trading Strategy
```python
class MLTradingStrategy(BaseMLStrategy):
    """Production ML trading strategy with multi-model support."""
```

**Advanced Features:**
- **Multi-Model Support**: Filter and aggregate signals from specified models
- **Decision Persistence**: All trading decisions stored in StrategyStore with risk metrics
- **Dry Run Mode**: execute_trades=False for safe strategy testing
- **Performance Tracking**: Per-model performance analytics
- **Position Reversal**: Intelligent position direction changes
- **Risk Metrics Integration**: Comprehensive risk data collection

**Configuration Options:**
```python
# Model Selection and Filtering
target_model_ids: list[str] | None = None  # Specific models to listen to
required_models: int = 1  # Minimum models needed for aggregation

# Signal Aggregation
aggregation_mode: str | None = None  # "voting", "weighted_average"
time_window_ms: int = 1000  # Signal correlation window
conflict_resolution: str | None = None  # How to handle conflicts
model_weights: dict[str, float] = {}  # Model-specific weights

# Performance and Tracking  
track_performance: bool = False  # Enable per-model performance tracking
```

### Advanced Multi-Model Strategy

**MultiModelMLStrategy** (ml_strategy.py) - Dynamic Weighting Strategy
```python
class MultiModelMLStrategy(MLTradingStrategy):
    """Extended ML strategy specifically designed for multi-model aggregation."""
```

**Dynamic Weighting Features:**
- **Performance-Based Weighting**: Automatic model weight adjustment based on historical performance
- **Adaptive Learning**: Model weights evolve with observed accuracy and profitability
- **Minimum Weight Constraints**: Prevents complete model exclusion (0.1 minimum weight)
- **Normalized Weighting**: Ensures total weights sum to 1.0 for proper aggregation

**Weight Calculation Algorithm:**
```python
weight = accuracy * (1.0 + np.tanh(profit_per_trade / 100.0))
```

## Architecture Patterns

### Signal Aggregation Framework
**📝 ADDITION:** Multi-model consensus mechanisms:

**Voting Aggregation:**
```python
bullish_votes = sum(1 for s in signals if s.prediction > 0.5)
action = "BUY" if bullish_votes > bearish_votes else "SELL"
```

**Weighted Average Aggregation:**
```python
weighted_prediction = sum(weight * signal.prediction for model_id, signal in signals) / total_weight
```

**Time Window Correlation:**
```python
time_diff_ms = (latest_signal_time - earliest_signal_time) / 1_000_000
valid_aggregation = time_diff_ms <= time_window_ms
```

### Strategy Store Integration
**✨ ENHANCEMENT:** Comprehensive decision tracking:
```python
def _persist_strategy_decision(
    self,
    signal: MLSignal,
    decision_type: str,  # "BUY", "SELL", "HOLD"
    risk_metrics: dict[str, float],
    execution_params: dict[str, Any]
) -> None:
```

**Tracked Decision Data:**
- Signal metadata (model_id, confidence, prediction)
- Risk metrics (account balance, active positions, confidence)
- Execution parameters (position size, stop loss, take profit)
- Model performance attribution
- Live/backtest environment detection

### Prometheus Metrics Integration
**📝 ADDITION:** Comprehensive strategy monitoring:
```python
# Module-level metrics initialization
ml_signals_received = get_counter(METRIC_SIGNALS_RECEIVED_TOTAL)
ml_trades_executed = get_counter(METRIC_TRADES_EXECUTED_TOTAL) 
ml_signal_to_trade_latency = get_histogram(METRIC_SIGNAL_TO_TRADE_LATENCY_SECONDS)
ml_position_count = get_gauge(METRIC_POSITION_COUNT)
```

**Key Metrics:**
- `nautilus_ml_signals_received_total{strategy_id, signal_source}`
- `nautilus_ml_trades_executed_total{strategy_id, order_side}`
- `nautilus_ml_signal_to_trade_latency_seconds{strategy_id}` 
- `nautilus_ml_position_count{strategy_id, instrument}`
- `nautilus_ml_strategy_decisions_persisted_total{strategy_id}`

### Dry Run Mode Implementation
**✨ ENHANCEMENT:** Safe strategy testing without real trades:
```python
if self._config.execute_trades:
    self._place_market_order(target_side, quantity)
else:
    self._dry_run_trades += 1
    self.log.info("[DRY RUN] Would place order...")
```

**Dry Run Features:**
- **Complete Signal Processing**: Full strategy logic execution
- **Risk Validation**: Position sizing and account checks
- **Decision Persistence**: StrategyStore recording for analysis
- **Metrics Collection**: Prometheus metrics for monitoring
- **Performance Simulation**: Trade tracking without execution

## Position Management

### Position Sizing Algorithm
**📝 ADDITION:** Account balance-based position sizing:
```python
def _calculate_position_size(self) -> Quantity | None:
    account_balance = float(account.balance_total().as_double())
    position_value = account_balance * self._config.position_size_pct
    raw_quantity = position_value / current_price
    quantity_value = max(round(raw_quantity, precision), min_quantity)
```

**Size Calculation Components:**
- **Account Balance Integration**: Uses actual account balance for sizing
- **Percentage-Based Sizing**: Configurable percentage of total balance
- **Price Discovery**: Last trade or mid-price for sizing calculations
- **Instrument Precision**: Respects instrument size precision requirements
- **Minimum Size Validation**: Ensures minimum quantity requirements met

### Risk Management
**📝 ADDITION:** Comprehensive risk controls:
```python
# Position Limits
max_positions: int = 1  # Maximum concurrent positions
position_size_pct: float = 0.02  # 2% of account per position

# Risk Parameters  
stop_loss_pct: float = 0.02  # 2% stop loss
take_profit_pct: float = 0.04  # 4% take profit
min_confidence: float = 0.6  # Minimum signal confidence
```

### Order Management
**📝 ADDITION:** Complete order lifecycle management:
- **Market Orders**: Immediate execution for signal responsiveness
- **Stop Loss Orders**: Automatic risk management on position entry
- **Position Reversal**: Efficient close-and-reverse logic
- **Reduce-Only Orders**: Safe position closing without over-leveraging
- **Order Tracking**: Comprehensive order state monitoring

## Multi-Model Features

### Model Performance Tracking
**📝 ADDITION:** Per-model analytics:
```python
def _update_model_performance(self, model_id: str, profit: float) -> None:
    performance = self._model_performance[model_id]
    performance["total_trades"] += 1
    performance["total_profit"] += profit
    performance["accuracy"] = wins / total_trades
```

**Performance Metrics:**
- **Total Trades**: Number of trades attributed to each model
- **Total Profit**: Cumulative profit/loss per model
- **Win/Loss Ratio**: Success rate tracking
- **Accuracy**: Percentage of profitable trades
- **Profit Per Trade**: Average profitability metric

### Signal Aggregation Modes
**📝 ADDITION:** Multiple consensus mechanisms:

**Simple Voting:**
- Count bullish vs bearish signals
- Majority wins approach
- Equal weight per model

**Weighted Average:**
- Model-specific weights
- Confidence-weighted predictions
- Dynamic weight adjustment

**Conflict Resolution:**
- Highest confidence signal priority
- Most recent signal priority
- Performance-weighted priority

### Dynamic Weight Adjustment
**✨ ENHANCEMENT:** Adaptive model weighting:
```python
def _get_dynamic_model_weights(self) -> dict[str, float]:
    weight = accuracy * (1.0 + np.tanh(profit_per_trade / 100.0))
    return normalized_weights
```

**Weighting Factors:**
- **Historical Accuracy**: Model prediction success rate
- **Profit Performance**: Actual profitability contribution
- **Trade Frequency**: Model signal generation rate
- **Confidence Calibration**: How well confidence correlates with success

## Integration Points

### ML Actor Communication
**📝 ADDITION:** Seamless signal consumption:
- **DataType Subscription**: MLSignal as first-class Nautilus data type
- **Client Filtering**: Optional signal source filtering
- **Metadata Propagation**: Model information and custom data preservation
- **Timestamp Validation**: Event time vs system time tracking

### Store Integration
**✨ ENHANCEMENT:** Full 4-store pattern compliance:
- **StrategyStore**: Decision persistence and analysis
- **FeatureStore**: Access to feature data for validation
- **ModelStore**: Model performance correlation
- **DataStore**: Market data access for position sizing

### Risk Management Integration
**📝 ADDITION:** Comprehensive risk controls:
- **Account Validation**: Real-time account balance checks
- **Position Limits**: Per-strategy and global position constraints
- **Confidence Thresholds**: Signal quality filtering
- **Market Hours**: Trading session validation
- **Instrument Availability**: Market data quality checks

## Configuration Management

### Strategy Configuration
**📝 ADDITION:** Comprehensive MLStrategyConfig:
```python
class MLStrategyConfig(StrategyConfig):
    instrument_id: InstrumentId
    ml_signal_source: str
    position_size_pct: float = 0.02
    min_confidence: float = 0.6
    max_positions: int = 1
    execute_trades: bool = True  # Dry run control
    use_strategy_store: bool = False
    persist_all_signals: bool = False
```

### Multi-Model Configuration
**✨ ENHANCEMENT:** Advanced aggregation settings:
```python
# Model Selection
target_model_ids: list[str] | None = None
signal_client_id: ClientId | None = None

# Aggregation Parameters
aggregation_mode: str | None = "weighted_average"  
required_models: int = 2
time_window_ms: int = 1000
conflict_resolution: str = "highest_confidence"

# Performance Tracking
track_performance: bool = True
use_dynamic_weights: bool = True
```

### Store Configuration
**📝 ADDITION:** StrategyStore integration:
```python
strategy_store_config = {
    "connection_string": "postgresql://...",
    "batch_size": 100,
    "flush_interval_ms": 1000
}
```

## Performance Optimization

### Signal Processing Performance
**Hot Path Requirements:**
- **Signal Validation**: <1ms for filtering and threshold checks
- **Position Sizing**: <2ms for balance and price calculations  
- **Order Placement**: <5ms total latency from signal to order
- **Metrics Recording**: Zero-allocation metric updates

### Memory Management
**📝 ADDITION:** Efficient data structures:
- **Signal History**: Fixed-size deque with configurable history_size
- **Model Signals**: Dict-based current signal tracking
- **Performance Data**: Lazy-loaded per-model statistics
- **Order Tracking**: Minimal metadata for performance correlation

### Aggregation Efficiency
**📝 ADDITION:** Optimized multi-model processing:
- **Time Window Checks**: Early filtering of stale signals
- **Weight Calculation**: Cached dynamic weights with lazy updates
- **Conflict Resolution**: Fast decision algorithms
- **Signal Buffering**: Efficient aggregation data structures

## Testing and Validation

### Strategy Testing Framework
**📝 ADDITION:** Comprehensive test coverage:
- **Unit Tests**: Individual method testing with mock signals
- **Integration Tests**: End-to-end signal processing workflows
- **Performance Tests**: Latency and throughput benchmarks
- **Risk Tests**: Position sizing and risk management validation

### Backtesting Support
**✨ ENHANCEMENT:** Historical strategy validation:
- **Signal Replay**: Historical MLSignal data processing
- **Decision Tracking**: StrategyStore integration for analysis
- **Performance Attribution**: Per-model historical performance
- **Risk Analysis**: Historical risk metric validation

### Dry Run Validation
**📝 ADDITION:** Safe production testing:
- **Signal Processing Validation**: Complete logic without execution
- **Risk Management Testing**: Position sizing and validation
- **Metrics Collection**: Full monitoring without trades
- **Decision Persistence**: Strategy analysis data collection

## Best Practices

### Strategy Development Guidelines
- **Signal Validation**: Always validate signal quality and instrument matching
- **Risk First**: Implement risk checks before any trade execution
- **Metrics Integration**: Add Prometheus metrics for all critical operations
- **Store Integration**: Persist decisions for analysis and debugging
- **Error Handling**: Graceful degradation when external systems unavailable

### Multi-Model Strategy Design
- **Model Selection**: Choose complementary models with different strengths
- **Aggregation Logic**: Match aggregation method to model characteristics
- **Performance Monitoring**: Track individual model contributions
- **Dynamic Adaptation**: Allow strategy to learn from model performance
- **Conflict Resolution**: Design clear rules for opposing signals

### Production Deployment
- **Dry Run First**: Always test strategies in dry run mode before live deployment
- **Gradual Rollout**: Start with small position sizes and single instruments
- **Monitoring**: Set up comprehensive alerting for strategy performance
- **Rollback Capability**: Maintain ability to quickly disable strategies
- **Risk Limits**: Implement hard stops for maximum loss scenarios

This strategies module provides the foundation for sophisticated ML-driven trading systems while maintaining safety, performance, and observability requirements for production deployment.