# ML Deployment Architecture Context

## Executive Summary

The ML deployment architecture for Nautilus Trader provides a comprehensive containerized production environment for real-time algorithmic trading with machine learning components. The system is built around Docker Compose orchestration and supports both dry-run testing and live trading modes with full monitoring, persistence, and safety controls.

## Architecture Overview

```
┌─────────────────────┐
│   Databento Feed    │
│   (Real Market)     │
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

**Entrypoint**: `/home/nate/projects/nautilus_trader/ml/pipeline/entrypoint_pipeline.py`

- Three operational modes:
  - **daily**: Scheduled daily updates with APScheduler
  - **backfill**: Historical data collection for date ranges
  - **realtime**: Continuous real-time data streaming
- Health check endpoint on port 8080
- Metrics server on port 8000
- Graceful shutdown handling

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

**Entrypoint**: `/home/nate/projects/nautilus_trader/ml/deployment/entrypoint_actor.py`

- Configures ML Signal Actor with Databento feed
- Sets up feature engineering pipeline
- Handles graceful shutdown with signal handlers
- Provides health monitoring and metrics

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

**Entrypoint**: `/home/nate/projects/nautilus_trader/ml/deployment/entrypoint_strategy.py`

- Configures ML Trading Strategy with signal subscriptions
- Enforces dry run mode safeguards
- Provides comprehensive trading statistics

### Data Persistence Services

#### 3. PostgreSQL Database (`postgres`)

- **Image**: `postgres:15-alpine`
- **Purpose**: Primary data persistence for all ML components
- **Port**: 5432
- **Database**: `nautilus`
- **Credentials**: postgres/postgres

**Schema Auto-initialization**:

- `/home/nate/projects/nautilus_trader/ml/schema/features.sql`
- `/home/nate/projects/nautilus_trader/ml/schema/models.sql`
- `/home/nate/projects/nautilus_trader/ml/schema/strategies.sql`

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

**Main Compose File**: `/home/nate/projects/nautilus_trader/ml/deployment/docker-compose.yml`

**Network**: `nautilus_network` (bridge driver)

**Volumes**:

- `postgres_data`: PostgreSQL data persistence
- `prometheus_data`: Metrics storage
- `grafana_data`: Dashboard and configuration storage

**Service Dependencies**:

```yaml
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
- Poetry installation with production dependencies
- Health check: HTTP endpoint check on port 8080
- Configuration directory: /app/config
- Supports three operational modes via PIPELINE_MODE env var
```

#### ML Signal Actor Dockerfile (`Dockerfile.actor`)

```dockerfile
FROM python:3.11-slim
- System deps: gcc, g++, postgresql-client
- Poetry installation with --no-dev
- Health check: Basic Python import test
- Models directory: /app/models
```

#### ML Trading Strategy Dockerfile (`Dockerfile.strategy`)

```dockerfile
FROM python:3.11-slim
- Similar base configuration as actor
- No models directory (consumes signals)
- Same health check pattern
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

**Prometheus Config**: `/home/nate/projects/nautilus_trader/ml/deployment/prometheus.yml`

- 15-second scrape intervals
- ML service discovery
- System metrics collection

## Deployment Modes

### 1. Docker Compose Deployment (Production-like)
**Script**: `/home/nate/projects/nautilus_trader/ml/deployment/run_dry_run.sh`

**Features**:

- Full containerized environment
- Service health checks
- Automated schema initialization
- Log aggregation
- Graceful shutdown handling

**Execution Steps**:

1. Validate API key
2. Start PostgreSQL with health checks
3. Initialize Redis
4. Create database schemas
5. Launch ML Signal Actor
6. Launch ML Trading Strategy
7. Start monitoring services
8. Display service URLs and monitoring commands

### 2. Local Development Deployment
**Script**: `/home/nate/projects/nautilus_trader/ml/deployment/run_local_dry_run.py`

**Features**:

- Local process execution
- PostgreSQL connection testing
- SQLite fallback for development
- Synthetic model generation
- Real-time market data integration

### 3. Backtest Deployment
**Script**: `/home/nate/projects/nautilus_trader/ml/deployment/run_backtest_dry_run.py`

**Features**:

- Historical data replay
- Synthetic data generation
- Backtest engine integration
- Performance analysis

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
5432: PostgreSQL database
6379: Redis cache
9090: Prometheus metrics
3000: Grafana dashboards
8000: ML Pipeline metrics server
8001: ML Strategy metrics (internal)
8080: ML Pipeline health check endpoint
```

### Internal Network Communication

- **Network**: `nautilus_network` (172.18.0.0/16)
- **Service Discovery**: Docker DNS resolution
- **Communication**: HTTP/gRPC for metrics, PostgreSQL protocol for persistence

## Production Deployment Steps

### Prerequisites

1. **API Access**: Valid Databento API key
2. **Infrastructure**: Docker and Docker Compose installed
3. **Database**: PostgreSQL server (local or remote)
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

3. **Service Deployment**:

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
✅ **Docker Compose orchestration** - Full multi-service setup
✅ **ML Pipeline service** - Automated data collection and feature computation
✅ **Container specifications** - Optimized Dockerfiles including pipeline
✅ **Environment configuration** - Comprehensive env var support
✅ **Service dependencies** - Proper startup ordering
✅ **Health checks** - Service health monitoring with dedicated endpoints
✅ **Database integration** - PostgreSQL with auto-schema init and FeatureStore
✅ **Monitoring stack** - Prometheus/Grafana integration with pipeline metrics
✅ **Deployment scripts** - Multiple deployment modes (daily, backfill, realtime)
✅ **Safety controls** - Dry run mode enforcement
✅ **Graceful shutdown** - Signal handling and cleanup
✅ **Quick start script** - `quick_start.sh` for easy setup
✅ **Makefile commands** - Convenient development commands

### Architecture Strengths

- **Separation of concerns**: Distinct pipeline, signal generation, and trading strategy containers
- **Data persistence**: PostgreSQL integration with store triad and FeatureStore
- **Automated pipeline**: Daily data collection and feature computation
- **Safety first**: Mandatory dry run mode with explicit trade execution control
- **Observability**: Comprehensive metrics, logging, and health monitoring
- **Scalability**: Resource limits and horizontal scaling support
- **Development workflow**: Local, Docker, and backtest deployment options
- **Easy setup**: Quick start script and Makefile for rapid deployment

### Current Limitations
⚠️ **No Kubernetes support** - Only Docker Compose orchestration
⚠️ **Limited execution clients** - No production execution client configurations
⚠️ **Basic health checks** - Could be more sophisticated
⚠️ **Manual scaling** - No auto-scaling configuration
⚠️ **Single region** - No multi-region deployment support

## Critical Operational Notes

### Safety Protocols

1. **Dry Run Mode**: `EXECUTE_TRADES=false` is enforced by default
2. **Signal Validation**: All signals logged and persisted before execution
3. **Resource Limits**: Memory and CPU constraints prevent resource exhaustion
4. **Database Persistence**: All decisions stored for audit and analysis
5. **Graceful Shutdown**: Proper signal handling for clean service termination

### Performance Considerations

- **Memory Allocation**: 8GB per ML service for model loading and feature computation
- **CPU Allocation**: 2 cores per service for parallel processing
- **Database Connections**: Connection pooling and proper cleanup
- **Metrics Overhead**: Prometheus scraping configured for minimal performance impact

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

The deployment architecture provides a robust, production-ready foundation for ML-driven algorithmic trading with comprehensive safety controls, monitoring, and operational excellence built-in.
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
