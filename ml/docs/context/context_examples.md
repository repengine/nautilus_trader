# Context: Examples Module

## Overview

The `ml/examples/` directory provides comprehensive educational and demonstration scripts showcasing the Nautilus Trader ML system's capabilities. This collection covers everything from basic actor implementation patterns to advanced production deployment scenarios. The examples serve as both learning resources and reference implementations for building ML trading systems, with emphasis on proper configuration, testing methodologies, and integration patterns.

Each example is designed to be self-contained and executable, demonstrating specific ML system features while following the mandatory architectural patterns established in CLAUDE.md. The examples range from simple model creation utilities to complete end-to-end ML trading pipelines with proper error handling, metrics collection, and production-ready configurations.

## Architecture

### Example Categories

```
ml/examples/
├── Core ML Patterns
│   ├── simple_ml_actor.py          # Basic ML actor implementation with technical indicators
│   ├── mandatory_stores_example.py  # Demonstrates mandatory 4-store + 4-registry pattern
│   └── feature_store_example.py     # Training/inference parity with FeatureStore
│
├── Registry & Backend Patterns
│   ├── strategy_registry_demo.py    # Strategy lifecycle and performance tracking
│   ├── postgres_registry_demo.py    # PostgreSQL vs JSON backend comparison
│   └── test_registry_backends.py    # Backend functionality testing
│
├── Production Deployment
│   ├── dry_run_example.py          # Complete dry-run testing framework
│   ├── scheduler_with_features.py   # DataScheduler with FeatureStore integration
│   └── scheduler_with_metrics.py    # Prometheus metrics and monitoring
│
├── Data Sources & Integration
│   ├── calendar_provider_demo.py    # Market calendar and trading hours
│   └── tft_with_feature_store.py    # TFT Dataset Builder with feature parity
│
└── Testing & Development Utilities
    ├── create_dummy_model.py        # Dummy model creation for testing
    └── Example Test Infrastructure   # Mocking, fixtures, and test patterns
```

### Educational Learning Path

1. **Foundation**: `simple_ml_actor.py` → Basic actor patterns and feature computation
2. **Data Management**: `mandatory_stores_example.py` → Store integration and persistence
3. **Feature Engineering**: `feature_store_example.py` → Training/inference parity
4. **Registry Systems**: `strategy_registry_demo.py` → Component lifecycle management
5. **Production Deployment**: `dry_run_example.py` → End-to-end system testing
6. **Monitoring**: `scheduler_with_metrics.py` → Observability and metrics
7. **Advanced Integration**: `tft_with_feature_store.py` → Complex model training workflows

## Key Components

### Core ML Actor Patterns

#### Simple ML Actor (`simple_ml_actor.py`)
Demonstrates fundamental ML actor implementation with technical indicators:

- **BaseMLInferenceActor** inheritance with proper configuration handling
- **Feature Computation**: Technical indicators (SMA, RSI, EMA) as feature engineering
- **Model Loading**: Flexible model loading with fallback to dummy models
- **Type Safety**: Complete type annotations and numpy array management
- **Hot Path Optimization**: Pre-allocated feature buffers and zero-allocation patterns

```python
class SimpleMLActor(BaseMLInferenceActor):
    def _compute_features(self, bar: Bar) -> npt.NDArray[np.float32] | None:
        # Pre-allocated buffer for zero allocations
        self._feature_buffer[0] = close_price / sma_fast_val
        self._feature_buffer[1] = close_price / sma_slow_val
        # ... more features
        return self._feature_buffer.copy()
```

#### Mandatory Stores Example (`mandatory_stores_example.py`)
Showcases the mandatory 4-store + 4-registry integration pattern:

- **Automatic Store Initialization**: No configuration needed, stores work automatically
- **Production vs Testing**: Automatic fallback from PostgreSQL to DummyStore
- **Data Persistence**: Every feature, prediction, and signal automatically stored
- **Audit Trail**: Complete history for compliance and debugging
- **Performance Benefits**: A/B testing, model drift detection, replay capabilities

### Registry and Backend Management

#### Strategy Registry Demo (`strategy_registry_demo.py`)
Comprehensive strategy lifecycle management example:

- **Strategy Manifests**: Complete metadata including performance metrics, requirements
- **Market Regime Matching**: Query strategies by market conditions and instrument types
- **Performance Ranking**: Sort strategies by Sharpe ratio, return, or custom metrics
- **Compatibility Checking**: Validate strategy requirements against available resources
- **Strategy Lineage**: Track strategy evolution and parent-child relationships
- **Live Performance Updates**: Update metrics from live trading results

#### PostgreSQL Registry Demo (`postgres_registry_demo.py`)
Backend comparison and migration patterns:

- **Backend Types**: JSON file-based vs PostgreSQL database persistence
- **Teacher-Student Models**: Demonstrates knowledge distillation workflows
- **Model Deployment**: Track deployment status and target environments
- **ACID Compliance**: Transactional operations and concurrent access support
- **Migration Utilities**: Tools for moving from JSON to PostgreSQL backends

### Production Deployment Patterns

#### Dry Run Example (`dry_run_example.py`)
Complete testing framework for ML trading systems:

- **Risk-Free Testing**: Execute trades disabled, signal generation only
- **Backtest Engine Integration**: Historical data simulation without live connections
- **Configuration Validation**: Verify all components work together correctly
- **Error Simulation**: Test failure scenarios and recovery mechanisms
- **Deployment Preparation**: Step-by-step production readiness checklist

#### Data Scheduler with Features (`scheduler_with_features.py`)
Automated data collection with feature computation:

- **Databento Integration**: Real-time market data ingestion configuration
- **FeatureStore Integration**: Automatic feature computation and storage
- **Symbol Universe Management**: Multi-asset data collection configuration
- **Retention Policies**: Automatic data cleanup and archival
- **Error Handling**: Retry logic and failure recovery mechanisms

### Advanced Integration Examples

#### TFT with FeatureStore (`tft_with_feature_store.py`)
Advanced model training with guaranteed feature parity:

- **Training/Inference Parity**: Identical feature computation for both modes
- **Dataset Builder Integration**: Polars-based efficient data preparation
- **PostgreSQL Integration**: Direct access to Nautilus data warehouse
- **Feature Source Validation**: Verify features come from same computation engine
- **Batch vs Online Features**: Demonstrate parity validation between methods

#### Calendar Provider Demo (`calendar_provider_demo.py`)
Market schedule and trading hours integration:

- **Multi-Exchange Support**: NYSE, NASDAQ, LSE, JPX, crypto markets
- **Holiday Detection**: Automatic market closure identification
- **Trading Hours**: Pre-market, regular hours, after-hours detection
- **Time Zone Handling**: Proper UTC and local time conversions
- **Schedule Queries**: Market open/close times and remaining trading time

## Dependencies

### Internal Dependencies

- **ml.actors.base**: BaseMLInferenceActor and configuration classes
- **ml.actors.signal**: MLSignalActor and signal generation strategies
- **ml.stores**: All 4 mandatory stores (Feature, Model, Strategy, Data)
- **ml.registry**: All 4 mandatory registries (Feature, Model, Strategy, Data)
- **ml.data.scheduler**: DataScheduler for automated data collection
- **ml.features.engineering**: FeatureEngineer and FeatureConfig classes
- **ml.training.base**: BaseMLTrainer and training configuration

### External Dependencies

- **nautilus_trader**: Core trading framework, data types, backtest engine
- **numpy/polars**: Data processing and numerical computations
- **requests**: HTTP client for metrics endpoint checking
- **prometheus_client**: Metrics collection and monitoring (lazy-loaded)
- **pandas_market_calendars**: Market schedule information (optional)
- **databento**: Real-time market data integration (optional)

### Framework Dependencies (Lazy-Loaded)

- **onnxruntime**: ONNX model inference (for production models)
- **xgboost/lightgbm**: ML framework models (for training examples)
- **pickle**: Model serialization (DEPRECATED - security risk)

## Usage Patterns

### Basic Actor Implementation
Start with `simple_ml_actor.py` to understand fundamental patterns:

```python
# 1. Create a simple ML actor
from ml.examples.simple_ml_actor import SimpleMLActor
from ml.config.base import MLActorConfig

config = MLActorConfig(
    model_path="path/to/model.onnx",
    bar_type=bar_type,
    instrument_id=instrument_id,
)

actor = SimpleMLActor(config)
```

### Store Integration Testing
Use `mandatory_stores_example.py` for store integration validation:

```python
# Demonstrates automatic store initialization
from ml.examples.mandatory_stores_example import ProductionMLActor

# Stores are automatically initialized - no configuration needed
actor = ProductionMLActor(config)
# Features, predictions, signals automatically persisted
```

### Registry-Based Development
Follow `strategy_registry_demo.py` for component lifecycle management:

```python
# Query strategies by market conditions
trending_strategies = registry.get_strategies_for_regime(MarketRegime.TRENDING_UP)

# Validate requirements before deployment
can_run = registry.validate_requirements(
    strategy_id="trend_follow_ma_cross",
    available_models=["lgb_directional_v1"],
    available_features=["sma_20", "rsi_14"]
)
```

### Production Testing
Use `dry_run_example.py` for comprehensive system validation:

```bash
# Backtest mode (default)
python ml/examples/dry_run_example.py

# Live dry run mode
python ml/examples/dry_run_example.py --live
```

### Model Creation for Testing
Use `create_dummy_model.py` to generate test models:

```python
# Creates bullish, bearish, and neutral test models
from ml.examples.create_dummy_model import create_dummy_models

models_dir = create_dummy_models()
# Models available at: ml/models/dummy_*.pkl
```

## Integration Points

### Nautilus Trader Integration

- **Actor Framework**: Seamless integration with Nautilus actor lifecycle
- **Data Types**: Proper use of InstrumentId, BarType, ComponentId
- **Event System**: Message bus integration for signal publishing
- **Backtest Engine**: Complete backtesting framework integration
- **Configuration System**: msgspec-based configuration with validation

### ML Pipeline Integration

- **Store System**: Automatic integration with all 4 mandatory stores
- **Registry System**: Component lifecycle and schema management
- **Feature Engineering**: FeatureEngineer integration with parity validation
- **Model Loading**: Multi-format model support with security validation
- **Signal Generation**: Multiple signal strategies with confidence thresholds

### Data Source Integration

- **Market Data**: Real-time and historical data ingestion
- **External APIs**: Databento integration for professional market data
- **Calendar Systems**: Trading schedule and holiday detection
- **Database Systems**: PostgreSQL and SQLite integration

### Monitoring and Observability

- **Prometheus Metrics**: Comprehensive metrics collection and export
- **Health Monitoring**: Circuit breakers and health status tracking
- **Performance Tracking**: Latency monitoring and SLA enforcement
- **Audit Logging**: Complete operation trail for compliance

### Testing Framework Integration

- **Mock Support**: DummyStore and DummyRegistry for isolated testing
- **Fixture Generation**: Dummy model creation and test data generation
- **Error Simulation**: Failure injection and recovery testing
- **Performance Validation**: Latency and throughput benchmarking

## Implementation Notes

### Security and Safety

- **Model Format Validation**: ONNX preferred, pickle deprecated (security risk)
- **Input Sanitization**: Comprehensive validation of all user inputs
- **Error Handling**: Graceful degradation and recovery mechanisms
- **Audit Trail**: Complete logging of all operations for security compliance

### Performance Optimization

- **Zero-Allocation Patterns**: Pre-allocated buffers in hot path
- **Lazy Loading**: Framework dependencies loaded only when needed
- **Efficient Data Structures**: Polars for large dataset operations
- **Memory Management**: Bounded memory usage with reservoir sampling

### Configuration Management

- **Type Safety**: Complete type annotations and validation
- **Environment Integration**: Environment variable override support
- **Layered Configuration**: File, environment, CLI override hierarchy
- **Framework Detection**: Automatic adaptation based on available dependencies

### Error Handling and Resilience

- **Circuit Breakers**: Prevent cascade failures in production
- **Retry Logic**: Configurable retry policies for transient failures
- **Fallback Mechanisms**: Graceful degradation when components unavailable
- **Health Monitoring**: Continuous health assessment and alerting

### Development Workflow

- **Progressive Complexity**: Examples ordered by difficulty and features
- **Self-Contained**: Each example can run independently
- **Comprehensive Documentation**: Detailed comments and docstrings
- **Production Patterns**: Real-world deployment scenarios and best practices

The examples module serves as the primary educational resource for ML system development in Nautilus Trader, providing practical demonstrations of all major features while following production-ready patterns and best practices. Each example is designed to be both a learning tool and a reference implementation for specific use cases.

## Cross-Module References

- **Actors**: See `context_actors.md` for ML actor architecture details
- **Stores**: See `context_stores.md` for persistence layer implementation
- **Registry**: See `context_registry.md` for component lifecycle management
- **Features**: See `context_features.md` for feature engineering pipelines
- **Config**: See `context_config.md` for configuration system details
- **Training**: See `context_training.md` for model training workflows
- **Data**: See `context_data.md` for data collection and processing
- **Monitoring**: See `context_monitoring.md` for observability systems

## Universal Pattern Compliance

All examples demonstrate compliance with the 5 universal ML architecture patterns:

### ✅ Pattern 1: Mandatory 4-Store + 4-Registry Integration

- Examples show automatic store initialization without configuration
- Progressive fallback demonstrated from PostgreSQL to DummyStore
- Property accessors showcase clean component interface

### ✅ Pattern 2: Protocol-First Interface Design

- Store protocols enable structural typing in examples
- DummyStore compatibility demonstrated for testing scenarios
- Clear separation between interface and implementation

### ✅ Pattern 3: Hot/Cold Path Separation

- Performance budgets enforced in actor examples (<5ms P99)
- Model loading relegated to cold path during initialization
- Feature computation optimized for zero-allocation hot path

### ✅ Pattern 4: Progressive Fallback Chains

- PostgreSQL → DummyStore fallback demonstrated with warnings
- Registry loading → Direct file loading fallback patterns
- Graceful degradation examples for missing dependencies

### ✅ Pattern 5: Centralized Metrics Bootstrap

- All examples use `ml.common.metrics_bootstrap` for Prometheus metrics
- No direct prometheus_client imports in example code
- Safe metric registration patterns across module reloads
