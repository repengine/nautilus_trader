# ML Deployment Architecture Context

## Executive Summary

The ML deployment architecture for Nautilus Trader provides a comprehensive containerized production environment for real-time algorithmic trading with machine learning components. The system is built around Docker Compose orchestration and supports multiple deployment modes including local development, Docker containers, backtest simulations, and production deployment with full monitoring, persistence, and safety controls.

## Architecture Overview

```
┌─────────────────────┐
│   Databento Feed    │
│   (Real Market)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ML Pipeline        │
│  - Data collection  │
│  - Feature compute  │
│  - Model training   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ML Signal Actor    │
│  - Load model       │
│  - Calculate features│
│  - Generate signals │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ML Trading Strategy│
│  - Receive signals  │
│  - Make decisions   │
│  - [DRY RUN MODE]   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    PostgreSQL       │
│  - Store features   │
│  - Store signals    │
│  - Store decisions  │
└─────────────────────┘
```

## Service Breakdown

### Data Pipeline Services

#### ML Pipeline Container (`ml_pipeline`)

- **Image**: Custom built from `Dockerfile.pipeline`
- **Purpose**: Automated data collection and feature computation
- **Resources**: 4GB memory, 2 CPU cores
- **Key Components**:
  - DataScheduler with Databento integration
  - FeatureEngineer for feature computation
  - FeatureStore persistence
  - Health monitoring endpoint
  - Prometheus metrics exposure

**Environment Variables**:

```yaml
DB_CONNECTION: postgresql://postgres:postgres@postgres:5432/nautilus
DATABENTO_API_KEY: "${DATABENTO_API_KEY}"
PIPELINE_MODE: "${PIPELINE_MODE:-daily}"  # daily, backfill, or realtime
SCHEDULE_CRON: "0 4 * * *"  # Daily at 4 AM UTC
HEALTH_PORT: 8080
METRICS_PORT: 8000
```

**Entrypoint**: `ml/deployment/entrypoint_pipeline.py`

- Three operational modes:
  - **daily**: Scheduled updates using cron expression (default: 5 PM daily)
  - **backfill**: Historical data collection for specified date ranges
  - **realtime**: Continuous data streaming with configurable intervals
- Flask-based health check endpoint on port 8080
- Pipeline status tracking with error reporting
- Graceful shutdown handling via signal handlers

### Core Trading Services

#### 1. ML Signal Actor Container (`ml_signal_actor`)

- **Image**: Custom built from `Dockerfile.actor`
- **Purpose**: Real-time ML inference for signal generation
- **Resources**: 8GB memory, 2 CPU cores
- **Key Components**:
  - Databento data client for market data ingestion
  - Model loading and inference pipeline
  - Feature engineering pipeline
  - PostgreSQL persistence via FeatureStore
  - Prometheus metrics exposure

**Environment Variables**:

```yaml
DB_CONNECTION: postgresql://postgres:postgres@postgres:5432/nautilus
DATABENTO_API_KEY: "${DATABENTO_API_KEY}"
DATABENTO_DATASET: "${DATABENTO_DATASET:-EQUS.MINI}"
MODEL_PATH: /app/models/model.onnx
INSTRUMENT_ID: SPY.XNAS
BAR_TYPE: SPY.XNAS-1-MINUTE-LAST-EXTERNAL
ACTOR_ID: MLSignalActor-001
USE_DUMMY_STORES: "false"
```

**Entrypoint**: `ml/deployment/entrypoint_actor.py`

- Configures ML Signal Actor with Databento data client
- Validates model file existence at specified path
- Sets up feature engineering with configurable indicators (SMA, RSI, Bollinger Bands)
- Implements graceful shutdown with SIGTERM and SIGINT handlers
- Supports both real stores and dummy stores for testing
- No longer supports pickle models - requires ONNX or framework-native models

#### 2. ML Trading Strategy Container (`ml_strategy`)

- **Image**: Custom built from `Dockerfile.strategy`
- **Purpose**: Trading decision execution based on ML signals
- **Resources**: 8GB memory, 2 CPU cores
- **Key Components**:
  - Signal consumption and processing
  - Risk management and position sizing
  - Strategy state persistence
  - Dry run mode enforcement

**Environment Variables**:

```yaml
DB_CONNECTION: postgresql://postgres:postgres@postgres:5432/nautilus
STRATEGY_ID: MLStrategy-DRY-001
ML_SIGNAL_SOURCE: MLSignalActor-001
INSTRUMENT_ID: SPY.XNAS
EXECUTE_TRADES: "false"  # DRY RUN CONTROL
POSITION_SIZE_PCT: "0.02"
MIN_CONFIDENCE: "0.6"
MAX_POSITIONS: "1"
STOP_LOSS_PCT: "0.02"
TAKE_PROFIT_PCT: "0.04"
USE_STRATEGY_STORE: "true"
PERSIST_ALL_SIGNALS: "true"
```

**Entrypoint**: `ml/deployment/entrypoint_strategy.py`

- Configures ML Trading Strategy to consume signals from ML Signal Actor
- Enforces dry run mode by default with clear warning messages
- Provides comprehensive trading statistics on shutdown
- Supports optional Databento data client for market data
- Configurable risk parameters (position sizing, stop loss, take profit)
- Persists all signals and decisions to PostgreSQL via StrategyStore

### Data Persistence Services

#### 3. PostgreSQL Database (`postgres`)

- **Image**: `postgres:15-alpine`
- **Purpose**: Primary data persistence for all ML components
- **Port**: 5432
- **Database**: `nautilus`
- **Credentials**: postgres/postgres

**Schema Auto-initialization**:

- `ml/schema/00_init.sql` - Base schema setup
- `ml/schema/features.sql` - Feature storage tables
- `ml/schema/models.sql` - Model predictions and metrics
- `ml/schema/strategies.sql` - Strategy signals and decisions
- `ml/schema/pipeline_health.sql` - Pipeline monitoring tables

**Health Check**: `pg_isready -U postgres`

#### 4. Redis Message Bus (`redis`)

- **Image**: `redis:7-alpine`
- **Purpose**: Inter-service communication and caching
- **Port**: 6379
- **Health Check**: `redis-cli ping`

### Monitoring Stack

#### 5. Prometheus (`prometheus`)

- **Image**: `prom/prometheus:latest`
- **Purpose**: Metrics collection and monitoring
- **Port**: 9090
- **Configuration**: `/home/nate/projects/nautilus_trader/ml/deployment/prometheus.yml`

**Scrape Targets**:

```yaml
- job_name: 'ml_signal_actor'
  targets: ['ml_signal_actor:8000']
- job_name: 'ml_strategy'
  targets: ['ml_strategy:8001']
- job_name: 'postgres'
  targets: ['postgres_exporter:9187']
- job_name: 'node'
  targets: ['node_exporter:9100']
```

#### 6. Grafana (`grafana`)

- **Image**: `grafana/grafana:latest`
- **Purpose**: Visualization and dashboards
- **Port**: 3000
- **Credentials**: admin/admin
- **Plugins**: grafana-piechart-panel

## Container Orchestration Details

### Docker Compose Configuration

**Main Compose File**: `ml/deployment/docker-compose.yml`

**Network**: `nautilus_network` (bridge driver)

**Volumes**:

- `postgres_data`: PostgreSQL data persistence
- `prometheus_data`: Metrics storage
- `grafana_data`: Dashboard and configuration storage

**Service Dependencies**:

```yaml
ml_pipeline:
  depends_on:
    postgres:
      condition: service_healthy
ml_signal_actor:
  depends_on: [postgres, redis]
ml_strategy:
  depends_on: [postgres, redis, ml_signal_actor]
grafana:
  depends_on: [prometheus]
```

### Dockerfile Specifications

#### ML Pipeline Dockerfile (`Dockerfile.pipeline`)

```dockerfile
FROM python:3.11-slim
- System deps: gcc, g++, postgresql-client, curl
- Poetry installation with --no-dev flag
- Flask and requests for health check endpoint
- Creates directories: /app/data/catalog, /app/models, /app/logs, /app/configs
- Health check: Python import verification
- Entrypoint: /app/entrypoint.py
```

#### ML Signal Actor Dockerfile (`Dockerfile.actor`)

```dockerfile
FROM python:3.11-slim
- System deps: gcc, g++, postgresql-client
- Poetry installation with --no-dev flag
- Creates models directory: /app/models
- Health check: Python sys.exit(0) test
- Entrypoint: /app/entrypoint.py
```

#### ML Trading Strategy Dockerfile (`Dockerfile.strategy`)

```dockerfile
FROM python:3.11-slim
- System deps: gcc, g++, postgresql-client
- Poetry installation with --no-dev flag
- No models directory (consumes signals only)
- Health check: Python sys.exit(0) test
- Entrypoint: /app/entrypoint.py
```

## Configuration Management

### Environment-Driven Configuration

**Required Environment Variables**:

- `DATABENTO_API_KEY`: Market data access
- `DB_CONNECTION`: PostgreSQL connection string
- `EXECUTE_TRADES`: Dry run mode control (must be "false")

**Optional Configuration**:

- `DATABENTO_DATASET`: Data source selection
- `INSTRUMENT_ID`: Trading instrument
- `BAR_TYPE`: Market data granularity
- Risk parameters (position sizing, stop loss, etc.)

### Configuration Files

**Prometheus Config**: `ml/deployment/prometheus.yml`

- 15-second scrape intervals
- Service endpoints:
  - ml_signal_actor:8000/metrics
  - ml_strategy:8001/metrics
  - postgres_exporter:9187 (optional)
  - node_exporter:9100 (system metrics)

**Grafana Dashboard**: `ml/deployment/grafana/ml_pipeline_health.json`

- Comprehensive ML pipeline health monitoring
- Pipeline overview with health score gauge
- Data freshness and staleness tracking
- Feature computation metrics
- Error tracking and recent errors table
- Instrument data freshness heatmap
- Configurable time intervals and filters

## Deployment Modes

### 1. Docker Compose Deployment (Production-like)
**Script**: `ml/deployment/run_dry_run.sh`

**Features**:

- Full containerized environment with health monitoring
- Automated service startup with dependency management
- Database schema initialization from SQL files
- Real-time log aggregation
- Graceful shutdown with cleanup trap
- Color-coded output for better visibility

**Execution Steps**:

1. Validate DATABENTO_API_KEY environment variable
2. Start PostgreSQL and wait for healthy status
3. Check/start Redis service
4. Initialize database schemas (00_init.sql, features.sql, models.sql, strategies.sql)
5. Launch ML Signal Actor container
6. Launch ML Trading Strategy container
7. Start Prometheus and Grafana monitoring
8. Display service URLs and tail logs

### 2. Local Development Deployment
**Script**: `ml/deployment/run_local_dry_run.py`

**Features**:

- Local process execution without containers
- PostgreSQL connection testing with automatic SQLite fallback
- Automatic dummy model creation if needed
- Real Databento market data integration
- Support for US equities (SPY on NASDAQ by default)
- Configurable feature engineering (SMA, RSI, Bollinger Bands)
- Comprehensive statistics reporting on shutdown

**Prerequisites Check**:
- Validates Databento API key
- Tests PostgreSQL connection
- Creates dummy model if missing
- Falls back to SQLite if PostgreSQL unavailable

### 3. Backtest Deployment
**Script**: `ml/deployment/run_backtest_dry_run.py`

**Features**:

- Historical data replay with BacktestEngine
- Synthetic bar data generation for testing
- Support for SPY.XNAS with 1-minute bars
- Netting OMS with cash account type
- ML Signal Actor and Strategy integration
- Configurable feature engineering
- Performance statistics reporting

**Data Generation**:
- Creates 1000 synthetic bars over 5 days
- Realistic price movements with continuity
- Random volume generation
- Nanosecond timestamp precision

## Monitoring and Logging Setup

### Metrics Collection

**ML Signal Actor Metrics**:

- `ml_signals_generated_total`: Signal generation count
- `ml_signal_generation_seconds`: Latency measurements
- `ml_feature_computation_seconds`: Feature calculation time
- `ml_inference_latency_seconds`: Model inference performance

**ML Strategy Metrics**:

- `ml_strategy_dry_run_trades_total`: Dry run trade counter
- Strategy decision latencies
- Risk management metrics

### Log Management

**Container Logs**:

```bash
# View all services
docker-compose logs -f

# Specific services
docker-compose logs -f ml_signal_actor
docker-compose logs -f ml_strategy
```

**Database Monitoring**:

```sql
-- Signal monitoring
SELECT * FROM ml.strategy_signals ORDER BY ts_event DESC LIMIT 10;

-- Feature monitoring
SELECT * FROM ml.features ORDER BY ts_event DESC LIMIT 10;

-- Model prediction monitoring
SELECT * FROM ml.model_predictions ORDER BY ts_event DESC LIMIT 10;
```

### Health Checks

**Service Health Endpoints**:

- PostgreSQL: `pg_isready` command
- Redis: `redis-cli ping`
- Prometheus: `http://localhost:9090/-/healthy`
- Grafana: `http://localhost:3000/api/health`

## Port Mappings and Networking

### External Port Mappings

```yaml
3000: Grafana dashboards
5432: PostgreSQL database
6379: Redis cache
8080: ML Pipeline health check endpoint
9090: Prometheus metrics
```

### Internal Port Mappings

```yaml
8000: ML Signal Actor metrics endpoint
8001: ML Strategy metrics endpoint
9100: Node exporter (system metrics)
9187: PostgreSQL exporter (optional)
```

### Internal Network Communication

- **Network**: `nautilus_network` (172.18.0.0/16)
- **Service Discovery**: Docker DNS resolution
- **Communication**: HTTP/gRPC for metrics, PostgreSQL protocol for persistence

## Deployment Tools and Scripts

### Quick Start Script
**Script**: `ml/deployment/quick_start.sh`

Automated setup script for first-time users:
- Creates .env from .env.example if missing
- Validates Databento API key configuration
- Builds Docker images
- Starts all services
- Runs health checks
- Displays useful commands and monitoring URLs

### Makefile Commands
**File**: `ml/deployment/Makefile`

Convenient development and deployment commands:
- `make build` - Build all Docker images
- `make up` - Start all services
- `make down` - Stop all services
- `make logs [SERVICE=name]` - View logs
- `make clean` - Clean up volumes and images
- `make test` - Run integration tests
- `make health` - Check service health
- `make pipeline` - Start only ML pipeline
- `make backfill START=date END=date` - Run backfill mode
- `make realtime` - Run realtime mode
- `make dev` - Development mode with live code mounting
- `make deploy` - Production deployment
- `make scale WORKERS=n` - Scale pipeline workers
- `make backup` - Backup PostgreSQL database
- `make restore FILE=backup.sql` - Restore database

### Test Script
**Script**: `ml/deployment/test_docker_setup.sh`

Validates deployment prerequisites:
- Checks Docker and Docker Compose installation
- Validates environment configuration
- Verifies Docker Compose configuration syntax
- Checks required files existence
- Tests Docker build (dry run)
- Validates Python imports in container
- Provides colored output for test results

### Health Check Script
**Script**: `ml/deployment/check_health.py`

Comprehensive health monitoring:
- Checks Docker Compose services status
- Validates PostgreSQL connectivity
- Tests Redis connection
- Verifies ML Pipeline HTTP endpoint
- Checks Prometheus and Grafana health
- Returns aggregated health status

### Development Override
**File**: `ml/deployment/docker-compose.override.yml.example`

Development-specific configurations:
- Source code live mounting for hot reload
- Debug logging levels
- Reduced batch sizes for testing
- PostgreSQL port exposure for debugging
- Optional pgAdmin service for database inspection

## Production Deployment Steps

### Prerequisites

1. **API Access**: Valid Databento API key
2. **Infrastructure**: Docker and Docker Compose v2 installed
3. **Database**: PostgreSQL 15+ server (local or remote)
4. **Environment**: Environment variables configured

### Deployment Procedure

1. **Environment Setup**:

   ```bash
   export DATABENTO_API_KEY=your_key_here
   export DB_CONNECTION=postgresql://postgres:postgres@localhost:5432/nautilus
   ```

2. **Database Initialization**:

   ```bash
   createdb nautilus
   psql nautilus < ml/schema/features.sql
   psql nautilus < ml/schema/models.sql
   psql nautilus < ml/schema/strategies.sql
   ```

3. **Quick Start Deployment**:

   ```bash
   cd ml/deployment
   chmod +x quick_start.sh
   ./quick_start.sh
   ```

   Or manual deployment:

   ```bash
   chmod +x ml/deployment/run_dry_run.sh
   ./ml/deployment/run_dry_run.sh
   ```

4. **Verification**:
   - Check service health: `docker-compose ps`
   - Monitor logs: `docker-compose logs -f`
   - Verify metrics: `http://localhost:9090`
   - Check dashboards: `http://localhost:3000`

### Live Trading Transition

**Safety Checklist**:

1. Verify dry run mode: Check `EXECUTE_TRADES=false`
2. Test with small data: Run for 1-hour periods
3. Validate model predictions: Ensure reasonable outputs
4. Monitor resource usage: CPU < 50%, stable memory

**Live Mode Activation**:

1. Configure execution client
2. Set `EXECUTE_TRADES=true`
3. Start with minimal position sizes
4. Implement kill switch procedures

## Current Implementation Status

### Completed Components
✅ **Docker Compose orchestration** - Full multi-service setup with Docker Compose v2
✅ **ML Pipeline service** - Automated data collection with three operational modes
✅ **ML Signal Actor** - Real-time inference with Databento integration
✅ **ML Trading Strategy** - Signal consumption and dry-run trading
✅ **Container specifications** - Optimized Dockerfiles for all services
✅ **Environment configuration** - Comprehensive environment variable support
✅ **Service dependencies** - Health-based startup ordering
✅ **Health monitoring** - Flask-based health endpoints and Docker health checks
✅ **Database integration** - PostgreSQL with schema auto-initialization
✅ **Monitoring stack** - Prometheus metrics and Grafana dashboards
✅ **Deployment scripts** - Shell and Python scripts for various deployment modes
✅ **Development tools** - Makefile, quick start, and test scripts
✅ **Safety controls** - Mandatory dry run mode with explicit overrides
✅ **Graceful shutdown** - Signal handling in all services
✅ **Logging** - Centralized logging with configurable levels

### Architecture Strengths

- **Separation of concerns**: Distinct containers for pipeline, signal generation, and trading
- **Data persistence**: PostgreSQL with automated schema management
- **Pipeline flexibility**: Three operational modes (daily, backfill, realtime)
- **Safety first**: Dry run mode by default with clear warnings
- **Comprehensive monitoring**: Prometheus metrics with custom Grafana dashboards
- **Developer friendly**: Multiple deployment modes and convenience scripts
- **Production ready**: Health checks, graceful shutdown, and error handling
- **Easy onboarding**: Quick start script and test utilities

### Current Limitations
⚠️ **No Kubernetes support** - Docker Compose only, no Helm charts or K8s manifests
⚠️ **Limited execution clients** - No production broker integrations configured
⚠️ **Manual scaling** - No auto-scaling based on load metrics
⚠️ **Single region** - No multi-region or failover support
⚠️ **No secrets management** - API keys via environment variables only

## Critical Operational Notes

### Safety Protocols

1. **Dry Run Mode**: `EXECUTE_TRADES=false` is enforced by default
2. **Signal Validation**: All signals logged and persisted before execution
3. **Resource Limits**: Memory and CPU constraints prevent resource exhaustion
4. **Database Persistence**: All decisions stored for audit and analysis
5. **Graceful Shutdown**: Proper signal handling for clean service termination

### Performance Considerations

- **Memory Allocation**:
  - ML Pipeline: 16GB (data processing intensive)
  - ML Signal Actor: 8GB (model inference)
  - ML Trading Strategy: 8GB (signal processing)
- **CPU Allocation**:
  - ML Pipeline: 4 cores (parallel processing)
  - Other services: 2 cores each
- **Database Connections**: Connection pooling with proper cleanup
- **Metrics Collection**: 15-second scrape intervals for minimal overhead

### Security Measures

- **API Key Protection**: Environment variable injection only
- **Database Security**: Isolated network with credential management
- **Container Isolation**: Minimal base images with only required dependencies
- **Network Segmentation**: Bridge network with controlled inter-service communication

### Disaster Recovery

- **Data Persistence**: Volumes for critical data survival across restarts
- **Configuration Backup**: All configurations version controlled
- **State Recovery**: Database-backed state for service restart resilience
- **Monitoring Alerts**: Prometheus alerting for service failure detection

## Summary

The ML deployment architecture provides a comprehensive, production-ready foundation for ML-driven algorithmic trading. The system features:

- **Multiple deployment modes**: Local development, Docker containers, and backtest simulations
- **Modular architecture**: Separate containers for data pipeline, signal generation, and trading
- **Robust monitoring**: Prometheus metrics with custom Grafana dashboards
- **Safety by design**: Dry run mode by default with explicit overrides
- **Developer tools**: Quick start scripts, Makefile commands, and health checks
- **Production readiness**: Graceful shutdown, error handling, and comprehensive logging

The architecture emphasizes safety, observability, and ease of deployment while maintaining flexibility for different operational scenarios.
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
