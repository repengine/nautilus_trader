# ML Deployment Module Context

## Overview
The `ml/deployment/` module provides production-ready entry points and utilities for deploying ML trading systems in containerized environments. It offers multiple deployment modes including signal actors, trading strategies, and data pipelines.

## Core Components

### Container Entry Points

**entrypoint_actor.py** - ML Signal Actor Container
```python
class MLSignalActorNode:
    """Container-ready ML Signal Actor node."""
```

**Key Features:**
- Environment-based configuration (DB_CONNECTION, DATABENTO_API_KEY, MODEL_PATH)
- Support for both PostgreSQL and dummy stores via USE_DUMMY_STORES
- Automatic model validation and error handling
- Graceful shutdown with signal handlers (SIGTERM, SIGINT)
- Health monitoring integration with enable_health_monitoring=True
- **⚠️ CORRECTION:** No longer supports dummy pickle models - requires ONNX or framework-native models

**Configuration Environment Variables:**
- `DB_CONNECTION`: PostgreSQL connection string 
- `DATABENTO_API_KEY`: Required for market data access
- `MODEL_PATH`: Path to model file (default: /app/models/model.pkl)
- `INSTRUMENT_ID`: Trading instrument (default: BTC-USDT.DATABENTO)
- `BAR_TYPE`: Bar type specification (default: BTC-USDT.DATABENTO-1-MINUTE)
- `ACTOR_ID`: Component identifier (default: MLSignalActor-001)
- `USE_DUMMY_STORES`: Enable dummy stores for testing (default: false)

**entrypoint_strategy.py** - ML Trading Strategy Container
```python
class MLStrategyNode:
    """Container-ready ML Trading Strategy node."""
```

**Key Features:**
- **✨ ENHANCEMENT:** DRY RUN MODE by default with execute_trades=False for safety
- Risk management parameters via environment variables
- Signal consumption from MLSignalActor via ml_signal_source
- Strategy store persistence with configurable batch sizes
- Final statistics reporting on shutdown
- Test-safe initialization with graceful error handling

**Risk Management Configuration:**
- `POSITION_SIZE_PCT`: Position sizing (default: 0.02 = 2%)
- `MIN_CONFIDENCE`: Minimum signal confidence (default: 0.6)
- `MAX_POSITIONS`: Maximum concurrent positions (default: 1)
- `STOP_LOSS_PCT`: Stop loss percentage (default: 0.02 = 2%)
- `TAKE_PROFIT_PCT`: Take profit percentage (default: 0.04 = 4%)

**entrypoint_pipeline.py** - ML Data Pipeline Container
```python
class PipelineRunner:
    """ML Pipeline runner for Docker deployment."""
```

**Pipeline Modes:**
- **backfill**: Historical data collection (single daily update for simplicity)
- **daily**: Scheduled updates with cron configuration
- **realtime**: Continuous updates with configurable intervals (default: 5 minutes)

**Key Features:**
- Flask health check endpoint on /health (port 8080)
- Universe symbol expansion via UniverseConfig
- Databento integration with schema/dataset configuration
- Feature store and model store initialization
- **📝 ADDITION:** PipelineStatus tracking with error collection
- **📝 ADDITION:** Thread-safe shutdown handling with signal handlers

### Testing and Validation Tools

**run_local_dry_run.py** - Local Development Testing
- **✨ ENHANCEMENT:** Prerequisites validation including PostgreSQL connectivity
- Automatic fallback to SQLite when PostgreSQL unavailable
- **🔄 UPDATE:** Expects ONNX models (dummy_bullish_model.onnx) instead of pickle
- Real Databento data feed integration for realistic testing
- Comprehensive final statistics reporting

**run_backtest_dry_run.py** - Historical Data Backtesting  
- BacktestEngine integration with synthetic data generation
- **📝 ADDITION:** 1000 bars of realistic price movement simulation
- Support for multiple venue types (XNAS equity testing)
- Performance metrics collection and reporting

**check_health.py** - Service Health Monitoring
- Multi-service health checks: Docker Compose, PostgreSQL, Redis, ML Pipeline, Prometheus, Grafana
- HTTP endpoint validation with timeout handling
- **📝 ADDITION:** JSON-based docker-compose service status parsing
- Comprehensive error reporting with actionable debugging suggestions

## Architecture Patterns

### Progressive Fallback Implementation
All deployment scripts implement the mandatory Progressive Fallback pattern:
```python
# PostgreSQL → DummyStore fallback
use_dummy_stores = "sqlite" in self.db_connection or os.getenv("USE_DUMMY_STORES", "false").lower() == "true"

# Configuration fallback chain
db_connection = os.getenv("DB_CONNECTION", 
                         os.getenv("DATABASE_URL", 
                                  "postgresql://postgres:postgres@localhost:5432/nautilus"))
```

### Container-Ready Configuration
- **✨ ENHANCEMENT:** All entry points use environment variable configuration for containerization
- Consistent error handling with descriptive messages
- **📝 ADDITION:** Support for both development and production deployment modes
- Health monitoring integration across all components

### Hot/Cold Path Separation
- **Entry points handle cold path operations**: Configuration, initialization, model loading
- **Signal processing occurs in hot path**: Sub-5ms inference requirements maintained
- Model loading occurs once at startup, never during inference loops

## Integration Points

### Store Integration
All deployment entry points use the mandatory 4-store pattern:
- **FeatureStore**: Feature persistence and retrieval
- **ModelStore**: Model performance tracking  
- **StrategyStore**: Strategy state persistence
- **DataStore**: Unified data access layer

### Registry Integration  
- **ModelRegistry**: Model version management and deployment tracking
- **FeatureRegistry**: Feature schema validation
- **StrategyRegistry**: Strategy compatibility validation
- **DataRegistry**: Dataset manifest management

### Prometheus Metrics
**📝 ADDITION:** Health monitoring integration in all entry points:
- Service availability metrics
- Pipeline execution status
- Error rate tracking
- Performance monitoring (latency, throughput)

## Configuration Management

### Environment Variables
**Database Configuration:**
```bash
DB_CONNECTION="postgresql://user:pass@host:port/db"
DATABASE_URL="postgresql://user:pass@host:port/db"  # Fallback
```

**Market Data Configuration:**
```bash
DATABENTO_API_KEY="your_api_key"
DATABENTO_DATASET="EQUS.MINI"  
DATABENTO_SCHEMA="ohlcv-1m"
```

**Pipeline Configuration:**
```bash
PIPELINE_MODE="daily|backfill|realtime"
PIPELINE_SCHEDULE="0 17 * * *"  # Daily at 5 PM
REALTIME_INTERVAL="300"  # 5 minutes
```

### **🔄 UPDATE:** Model Path Configuration
- **DEPRECATED**: Pickle model support removed for security
- **CURRENT**: ONNX and framework-native models only
- **PATH**: /app/models/model.onnx (containerized) or ml/models/*.onnx (local)

## Deployment Modes

### Development Mode (run_local_dry_run.py)
- Local execution with real data feeds
- PostgreSQL with SQLite fallback
- Comprehensive prerequisite validation
- **⚠️ CORRECTION:** DRY RUN only - no live trading execution

### Container Mode (entrypoint_*.py)  
- Docker-ready with environment configuration
- Health check endpoints for orchestration
- Graceful shutdown handling
- Production logging and metrics

### Backtest Mode (run_backtest_dry_run.py)
- Historical data simulation
- Synthetic market data generation
- Performance validation and reporting
- **📝 ADDITION:** 5-day historical window with 1-minute granularity

## Error Handling and Resilience

### Graceful Degradation
- **PostgreSQL unavailable**: Falls back to SQLite or DummyStores
- **Databento connection issues**: Error logging with retry mechanisms  
- **Model loading failures**: Descriptive error messages with path validation
- **Missing dependencies**: Clear installation instructions

### Health Monitoring
- **/health endpoints** for all services
- **Service dependency validation** (PostgreSQL, Redis, external APIs)
- **Error collection and reporting** via pipeline_status tracking
- **Timeout handling** for external service checks

## Best Practices

### Security
- **⚠️ CORRECTION:** No support for pickle models (security risk)
- Environment variable validation for sensitive configuration
- **📝 ADDITION:** API key presence validation with clear error messages

### Performance  
- **Model loading at startup only**: No hot-path model operations
- **Connection pooling**: Reuse database connections across components
- **Batch processing**: Configurable batch sizes for store operations

### Monitoring
- **Prometheus metrics**: Integrated in all deployment modes
- **Structured logging**: Consistent format across all entry points  
- **Final statistics**: Comprehensive reporting on shutdown

## Testing Integration

### Unit Test Support
- **📝 ADDITION:** PYTEST_CURRENT_TEST environment detection
- Mock-friendly initialization with graceful error handling
- **Test-safe configurations**: Dummy stores and simplified setups

### Integration Testing
- **Health check validation**: Automated service dependency testing
- **End-to-end workflows**: Full pipeline testing with synthetic data
- **Performance benchmarking**: Latency and throughput validation

This deployment module ensures production-ready ML trading system deployment with comprehensive configuration management, error handling, and monitoring capabilities following all Nautilus ML architectural patterns.