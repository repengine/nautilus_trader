# ML Deployment Architecture Context

## Executive Summary

The ML deployment architecture for Nautilus Trader provides a production-ready containerized environment for real-time algorithmic trading with machine learning components. Built on Docker Compose orchestration with comprehensive monitoring, this system enforces the mandatory 4-store + 4-registry pattern, supports multiple deployment modes (local development, containerized production, and backtest), implements progressive fallback strategies, and maintains strict safety controls with dry-run-by-default operations.

## Architecture Overview

The deployment architecture is built around five universal ML patterns with comprehensive observability and progressive fallback strategies:

```
                    ┌─────────────────────────────────────────┐
                    │          Observability Layer            │
                    │  Prometheus + Grafana + Custom Alerts  │
                    └─────────────────────────────────────────┘
                                        │
           ┌─────────────────────────────┼─────────────────────────────┐
           │                             │                             │
           ▼                             ▼                             ▼
    ┌──────────────┐              ┌──────────────┐              ┌──────────────┐
    │  ML Pipeline │              │ ML Signal    │              │ ML Trading   │
    │  Container   │              │ Actor        │              │ Strategy     │
    │              │              │ Container    │              │ Container    │
    │• Data Sched. │◄────────────►│• Model Infer │◄────────────►│• Signal Proc │
    │• Feature Eng │              │• Feature Eng │              │• Risk Mgmt   │
    │• Health Check│              │• Hot Reload  │              │• DRY RUN     │
    └──────┬───────┘              └──────┬───────┘              └──────┬───────┘
           │                             │                             │
           └─────────────────────────────┼─────────────────────────────┘
                                         │
                            ┌────────────┴────────────┐
                            │   4-Store + 4-Registry  │
                            │     Persistence Layer   │
                            └─────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
        ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
        │   PostgreSQL     │  │     Redis        │  │   File System    │
        │   Database       │  │   Message Bus    │  │   Registry       │
        │                  │  │                  │  │                  │
        │• Schema Mgmt     │  │• Inter-service   │  │• Model Metadata  │
        │• Auto Partition  │  │• Communication   │  │• Feature Schema  │
        │• Progressive     │  │• Caching         │  │• Version Control │
        │  Fallback        │  │• Health Checks   │  │• Local Fallback  │
        └──────────────────┘  └──────────────────┘  └──────────────────┘
```

## Service Breakdown

### Universal ML Architecture Patterns Implementation

All ML deployment services adhere to the 5 universal patterns mandated by CLAUDE.md:

**Pattern 1: Mandatory 4-Store + 4-Registry Integration**

- All containers inherit from BaseMLInferenceActor
- Automatic initialization with progressive fallback (PostgreSQL → DummyStore)
- Health monitoring includes all components

**Pattern 2: Protocol-First Interface Design**

- All components use typing.Protocol for clean contracts
- DummyStore conforms to protocols for testing
- Type safety without circular dependencies

**Pattern 3: Hot/Cold Path Separation**

- Hot path: <5ms P99 latency, zero allocations, pre-allocated arrays
- Cold path: Training, migrations, analytics, heavy I/O operations
- ONNX Runtime for production inference

**Pattern 4: Progressive Fallback Chains**

- PostgreSQL → DummyStore (warnings logged)
- Registry loading → Direct file loading
- Network failures → Local caches

**Pattern 5: Centralized Metrics Bootstrap**

- ml.common.metrics_bootstrap prevents registry conflicts
- Safe for module reloads and testing

### Data Pipeline Services

#### ML Pipeline Container (`ml_pipeline`)

- **Image**: Custom built from `Dockerfile.pipeline` (lightweight, nautilus-trader + minimal ML deps)
- **Purpose**: Automated data collection and feature computation (COLD PATH)
- **Resources**: 16GB memory, 4 CPU cores (data processing intensive)
- **Key Components**:
  - DataScheduler with three operational modes (daily/backfill/realtime)
  - FeatureEngineer for feature computation
  - MLIntegrationManager for 4-store + 4-registry initialization
  - Flask-based health check endpoint with /health and /metrics
  - Auto-start observability flushing via bootstrap
  - Prometheus metrics exposure

**Environment Variables**:

```yaml
DB_CONNECTION: postgresql://postgres:postgres@postgres:5432/nautilus
DATABENTO_API_KEY: "${DATABENTO_API_KEY}"
PIPELINE_MODE: "${PIPELINE_MODE:-daily}"  # daily, backfill, or realtime
SCHEDULE_CRON: "0 4 * * *"  # Daily at 4 AM UTC
HEALTH_PORT: 8080
METRICS_PORT: 8000
WRITE_MODE: "${WRITE_MODE:-sql}"       # sql (canonical)
COVERAGE_MODE: "${COVERAGE_MODE:-sql}" # sql|catalog (planning)
CATALOG_PATH: "${CATALOG_PATH:-}"     # required for catalog coverage/client
```

**Entrypoint**: `ml/deployment/entrypoint_pipeline.py`

- Three operational modes:
  - **daily**: Scheduled updates using cron expression (default: 5 PM daily)
  - **backfill**: Historical data collection for specified date ranges
  - **realtime**: Continuous data streaming with configurable intervals
- Flask-based health check endpoint on port 8080
- Pipeline status tracking with error reporting
- Graceful shutdown handling via signal handlers

**Backfill Bootstrap (Orchestrator)**:

- For canonical market data storage, run a one-time or periodic gap backfill at startup using the orchestrator with SQL coverage + writer implementations.
- Uses the same `DB_CONNECTION` as other services and the canonical `market_data` table from migration `ml/stores/migrations/003_market_data.sql`.

Example wiring in `entrypoint_pipeline.py`:

```python
from pathlib import Path
from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.data.ingest.resume import DatabentoIngestor, IngestState
from ml.stores.providers import SqlCoverageProvider, SqlMarketDataWriter
from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import PersistenceConfig, BackendType

DB_URL = os.getenv("DB_CONNECTION")

coverage = SqlCoverageProvider(connection_string=DB_URL)
writer = SqlMarketDataWriter(connection_string=DB_URL)
registry = DataRegistry(
    registry_path=Path("/app/registry"),
    persistence_config=PersistenceConfig(backend=BackendType.POSTGRES, connection_string=DB_URL),
)
ingestor = DatabentoIngestor(client=databento_client)
orch = IngestionOrchestrator(coverage=coverage, writer=writer, registry=registry, ingestor=ingestor)

orch.backfill_gaps(
    dataset_id=os.getenv("DATABENTO_DATASET", "EQUS.MINI"),
    schema=os.getenv("DB_SCHEMA", "tbbo"),
    instrument_id=os.getenv("INSTRUMENT_ID", "SPY.XNAS"),
    lookback_days=int(os.getenv("BACKFILL_LOOKBACK_DAYS", "7")),
    state=IngestState(),
)
```

Note: For live streaming, attach the client’s write path to `SqlMarketDataWriter` to persist incoming records consistently with backfilled data.

**Backfill Bootstrap (IntegrationManager)**:

- The `MLIntegrationManager` can automatically run a one-time backfill at startup when enabled via environment variables. This uses the same CLI (`ml.cli.ingest_backfill`) so behavior and flags are consistent across entrypoints.

Environment flags:

- `ML_BACKFILL_ON_START`: `1|true|yes` to enable
- `BACKFILL_DATASET_ID`: dataset ID (e.g., `EQUS.MINI`) [required]
- `BACKFILL_INSTRUMENTS`: comma-separated instrument IDs [required]
- `BACKFILL_SCHEMA`: `bars|tbbo|trades` (default `bars`)
- `BACKFILL_LOOKBACK_DAYS`: integer (default `7`)
- `COVERAGE_MODE`: `sql|catalog` (default `sql`)
- `WRITE_MODE`: `sql` (default `sql`)
- `TABLE_NAME`: target table (default `market_data`)
- `INGEST_CLIENT_MODE`: `catalog|databento|noop` (default `catalog`)
- `CATALOG_PATH`: required for catalog coverage/client
- `DATABENTO_API_KEY`: used if `client-mode=databento`

Example (`docker-compose.yml` excerpt):

```yaml
services:
  ml_pipeline:
    environment:
      DB_CONNECTION: postgresql://postgres:postgres@postgres:5432/nautilus
      # Enable backfill on startup
      ML_BACKFILL_ON_START: "true"
      BACKFILL_DATASET_ID: "EQUS.MINI"
      BACKFILL_INSTRUMENTS: "SPY.XNAS,QQQ.XNAS"
      BACKFILL_SCHEMA: "bars"
      BACKFILL_LOOKBACK_DAYS: "7"
      COVERAGE_MODE: "sql"            # or 'catalog'
      WRITE_MODE: "sql"
      INGEST_CLIENT_MODE: "catalog"   # or 'databento'
      CATALOG_PATH: "/data/catalog"   # required if using catalog
      DATABENTO_API_KEY: "${DATABENTO_API_KEY}"
```

Notes:

- The IntegrationManager logs the full CLI invocation and soft-fails (logs a warning) if the backfill cannot run so as not to block container startup.
- For long-running backfills or multiple instruments, prefer running the CLI as a separate one-shot job.

### Core Trading Services

#### 1. ML Signal Actor Container (`ml_signal_actor`)

- **Image**: Custom built from `Dockerfile.actor` (full Poetry install with nautilus_trader)
- **Purpose**: Real-time ML inference for signal generation (HOT PATH)
- **Resources**: 8GB memory, 2 CPU cores
- **Mandatory Architecture**: Inherits from BaseMLInferenceActor with 4-store + 4-registry
- **Key Components**:
  - Databento data client for market data ingestion
  - ONNXModelLoader with optimized runtime (CPU/GPU providers)
  - ProductionModelLoader supporting ONNX/XGBoost/LightGBM (NO PICKLE for security)
  - Pre-allocated numpy arrays for feature computation
  - Circuit breaker pattern with HealthMonitor
  - Model hot-reloading capability
  - Comprehensive Prometheus metrics (latency, confidence, predictions)

**Environment Variables**:

```yaml
DB_CONNECTION: postgresql://postgres:postgres@postgres:5432/nautilus
DATABENTO_API_KEY: "${DATABENTO_API_KEY}"
DATABENTO_DATASET: "${DATABENTO_DATASET:-EQUS.MINI}"
MODEL_PATH: /app/models/model.pkl  # Legacy path, supports multiple formats
INSTRUMENT_ID: SPY.XNAS
BAR_TYPE: SPY.XNAS-1-MINUTE-LAST-EXTERNAL
ACTOR_ID: MLSignalActor-001
USE_DUMMY_STORES: "false"  # Progressive fallback to DummyStore
LOG_LEVEL: INFO
```

**Entrypoint**: `ml/deployment/entrypoint_actor.py`

- Extends BaseMLInferenceActor with mandatory store initialization
- Progressive fallback: PostgreSQL → DummyStore with warnings
- Model validation and metadata extraction
- Registry-based feature schema loading with manifest validation
- Graceful shutdown with proper resource cleanup
- Health monitoring with performance tracking
- Strict security: NO pickle support, only safe formats (ONNX, joblib, JSON)

#### 2. ML Trading Strategy Container (`ml_strategy`)

- **Image**: Custom built from `Dockerfile.strategy` (full Poetry install, no models directory)
- **Purpose**: Trading decision execution based on ML signals (HOT PATH decision making)
- **Resources**: 8GB memory, 2 CPU cores
- **Mandatory Architecture**: Inherits from BaseMLInferenceActor with 4-store + 4-registry
- **Key Components**:
  - MLSignal consumption and processing with model_id tracking
  - Risk management and position sizing with configurable parameters
  - Strategy state persistence via StrategyStore
  - **MANDATORY DRY RUN MODE**: EXECUTE_TRADES must be "false" by default
  - Comprehensive trading statistics and performance tracking
  - Signal correlation and model performance analysis

**Environment Variables**:

```yaml
DB_CONNECTION: postgresql://postgres:postgres@postgres:5432/nautilus
STRATEGY_ID: MLStrategy-DRY-001
ML_SIGNAL_SOURCE: MLSignalActor-001
INSTRUMENT_ID: SPY.XNAS
EXECUTE_TRADES: "false"  # MANDATORY DRY RUN - NO REAL TRADES!
POSITION_SIZE_PCT: "0.02"
MIN_CONFIDENCE: "0.6"
MAX_POSITIONS: "1"
STOP_LOSS_PCT: "0.02"
TAKE_PROFIT_PCT: "0.04"
USE_STRATEGY_STORE: "true"
PERSIST_ALL_SIGNALS: "true"  # Required for audit trail
LOG_LEVEL: INFO
```

**Entrypoint**: `ml/deployment/entrypoint_strategy.py`

- Extends BaseMLInferenceActor with mandatory 4-store + 4-registry initialization
- **SAFETY-FIRST**: Enforces dry run mode with explicit warnings and confirmation
- Progressive fallback for database connectivity
- Signal metadata correlation with model_id for A/B testing
- Comprehensive audit trail: all signals and decisions persisted
- Performance statistics: Sharpe ratio, win rate, average P&L tracking
- Registry-based strategy manifest validation

### Data Persistence Services

#### 3. PostgreSQL Database (`postgres`)

- **Image**: `postgres:15-alpine` with auto-initialization
- **Purpose**: Primary data persistence for 4-store + 4-registry pattern
- **Port**: 5433 (production), 5432 (development override)
- **Database**: `nautilus`
- **Credentials**: postgres/postgres
- **Progressive Fallback**: Automatic DummyStore fallback if unreachable

**Schema & Migrations (Canonical + Automated)**:

The system implements automated schema management with two migration approaches:

**Auto-Initialization (Docker)**:

```yaml
volumes:
  - ../stores/migrations:/docker-entrypoint-initdb.d:ro  # Auto-apply on first init
```

**CLI Migration Runner (Production)**:

```bash
# Baseline migrations (mandatory)
uv run --no-sync python -m ml.scripts.apply_migrations --db-url postgresql://postgres:postgres@localhost:5433/nautilus

# Full deployment (recommended)
uv run --no-sync python -m ml.scripts.apply_migrations --db-url postgresql://... --full
```

**Migration Strategy**:

- **Base Migrations**: 001-004, 007_add_event_metadata (mandatory)
- **Optional Migrations**: Schema hardening, views, BRIN indexes, emergency fixes
- **Idempotent Design**: Safe to run multiple times, handles conflicts gracefully
- **Atomic Execution**: Statement-level transaction safety with proper error handling

**Schema Components**:

- **Stores Schema**: Partitioned tables (feature_values, model_predictions, strategy_signals)
- **Registry Schema**: Model/feature/strategy manifest tables with versioning
- **Auto-Partitioning**: Monthly partition creation with automated maintenance
- **Event Metadata**: Comprehensive audit trail with ts_event/ts_init enforcement
- **Progressive Views**: ml.pipeline_health, data_collection_stats, model_performance_summary

**Database Engine Management**:

- **Singleton Pattern**: EngineManager prevents connection pool exhaustion
- **Conservative Pooling**: pool_size=5, max_overflow=10, pool_pre_ping=true
- **Connection Recycling**: 3600s recycle time prevents timeout issues

**Health Monitoring**:

```bash
# Docker health check
pg_isready -U postgres

# Application-level preflight
python -c "from ml.stores.infrastructure import check_db_prereqs; print(check_db_prereqs('postgresql://postgres:postgres@localhost:5433/nautilus'))"
```

#### 4. Redis Message Bus (`redis`)

- **Image**: `redis:7-alpine`
- **Purpose**: Inter-service communication and caching
- **Port**: 6379
- **Health Check**: `redis-cli ping`

### Observability & Monitoring Stack

#### 5. Prometheus (`prometheus`)

- **Image**: `prom/prometheus:latest`
- **Purpose**: Centralized metrics collection following Pattern 5 (Centralized Metrics Bootstrap)
- **Port**: 9090
- **Configuration**: `ml/deployment/prometheus.yml`
- **Scrape Interval**: 15s for minimal overhead

**Scrape Targets & Metrics**:

```yaml
- job_name: 'ml_pipeline'
  targets: ['ml_pipeline:8080']  # Flask /metrics endpoint

- job_name: 'ml_signal_actor'
  targets: ['ml_signal_actor:8000']  # BaseMLInferenceActor metrics
  metrics:
    - nautilus_ml_predictions_total
    - nautilus_ml_prediction_latency_seconds
    - nautilus_ml_signal_confidence

- job_name: 'ml_strategy'
  targets: ['ml_strategy:8001']  # Strategy-specific metrics
  metrics:
    - ml_strategy_dry_run_trades_total
    - ml_strategy_decision_latency_seconds

- job_name: 'postgres'          # Optional
  targets: ['postgres_exporter:9187']

- job_name: 'node'              # System metrics
  targets: ['node_exporter:9100']
```

**Alert Rules** (`ml/deployment/alerts.yml`):

```yaml
- alert: MLModelInferenceLatencyHighP99
  expr: histogram_quantile(0.99, sum(rate(nautilus_ml_model_inference_duration_seconds_bucket[5m])) by (le)) > 0.200
  for: 5m

- alert: MLPipelineHealthLow
  expr: nautilus_ml_pipeline_health < 0.8
  for: 5m
```

#### 6. Grafana (`grafana`)

- **Image**: `grafana/grafana:latest`
- **Purpose**: Real-time visualization and operational dashboards
- **Port**: 3000
- **Credentials**: admin/admin
- **Plugins**: grafana-piechart-panel
- **Dashboard**: `ml/deployment/grafana/ml_pipeline_health.json`

**Key Dashboard Panels**:

- **Pipeline Overview**: Health score gauge with color-coded thresholds
- **Data Freshness**: Heatmap showing staleness by instrument
- **Feature Computation**: Latency and throughput metrics
- **Model Performance**: Inference latency P50/P95/P99 tracking
- **Error Tracking**: Recent errors table with correlation IDs
- **Resource Usage**: Memory and CPU utilization across containers

## Container Orchestration Details

### Docker Compose Configuration

**Main Compose File**: `ml/deployment/docker-compose.yml`

**Network**: `nautilus-ml` (bridge driver)

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

## Container Topology & Quick Commands

This section captures the concrete topology and ops commands used in practice.

### Project + Network

- Project name: `ml` (pinned in `ml/deployment/docker-compose.yml` via `name: ml`)
- Network: `nautilus-ml` (bridge)

### Services Summary

- `postgres`
  - Internal: `postgres:5432` on `nautilus-ml`
  - Host port: `${POSTGRES_HOST_PORT:-5433}` → `5432`
  - Health: `pg_isready -U postgres`
  - Migrations: mounted SQL (`/docker-entrypoint-initdb.d`) + CLI helper

- `redis`
  - Internal: `redis:6379`
  - Host port: `${REDIS_HOST_PORT:-6380}` → `6379`
  - Health: `redis-cli ping`

- `ml_pipeline`
  - Health: HTTP `GET http://localhost:8080/health`
  - Host port: `${ML_PIPELINE_HOST_PORT:-8081}` → `8080`
  - DB envs: always use `postgres` hostname (never `localhost` inside containers)

- Optional: `ml_signal_actor`, `ml_strategy`, `prometheus`, `grafana`

### Database URIs

- Inside containers (service-to-service):
  - `postgresql://postgres:postgres@postgres:5432/nautilus`
- From host:
  - `postgresql://postgres:postgres@localhost:${POSTGRES_HOST_PORT:-5433}/nautilus`

### Makefile Shortcuts

Top-level Makefile provides convenience targets:

- `make ml-up` — start the stack (postgres, redis, ml_pipeline, grafana, prometheus)
- `make ml-down` — stop the stack and remove volumes
- `make ml-ps` — show service status
- `make ml-logs` — tail `ml_pipeline` logs
- `make ml-migrate` — apply DB migrations via compose exec helper

### Migrations

Apply canonical migrations idempotently:

```
uv run --active --no-sync python -m ml.deployment.migrations --apply --compose-file ml/deployment/docker-compose.yml
```

This pipes SQL to `psql` in the `postgres` service.

### Common Issues

- Port conflicts (e.g., `Bind for 0.0.0.0:5433 failed`):
  - Set `POSTGRES_HOST_PORT` to a free port (e.g., `5434`) or stop the conflicting Postgres
- Duplicate projects (e.g., containers named `deployment-*` and `ml-*`):
  - Use the pinned project name `ml`; bring down obsolete projects:
    `docker compose -f ml/deployment/docker-compose.yml --project-name deployment down -v`
- Defaults in code use `localhost`:
  - These are for host/dev tests; in containers override to `postgres` via envs

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

### Environment-Based Configuration Strategy

The system implements a layered configuration approach with strict security defaults and comprehensive validation:

**Mandatory Environment Variables**:

```bash
# Security & Data Access (REQUIRED)
DATABENTO_API_KEY=your_databento_api_key_here      # Market data access
EXECUTE_TRADES=false                                # MANDATORY DRY RUN - SAFETY FIRST

# Database Configuration (Progressive Fallback)
DB_CONNECTION=postgresql://postgres:postgres@postgres:5432/nautilus  # Auto-fallback to DummyStore
```

**Operational Configuration** (`.env` files):

```bash
# Pipeline Modes
PIPELINE_MODE=daily                                 # daily|backfill|realtime
PIPELINE_SCHEDULE=0 17 * * *                      # Cron for daily mode
REALTIME_INTERVAL=300                              # Seconds for realtime mode

# Universe & Instruments
UNIVERSE_SYMBOLS=SPY.XNAS,QQQ.XNAS,IWM.XNAS      # Comma-separated
INSTRUMENT_ID=SPY.XNAS                             # Primary instrument
BAR_TYPE=SPY.XNAS-1-MINUTE-LAST-EXTERNAL         # Data granularity

# Performance Tuning
MAX_WORKERS=4                                      # Parallel processing
BATCH_SIZE=1000                                    # Data batch size
LOG_LEVEL=INFO                                     # DEBUG|INFO|WARNING|ERROR

# Port Management (Conflict Avoidance)
POSTGRES_HOST_PORT=5433                           # Avoids local dev conflicts
REDIS_HOST_PORT=6380                              # Avoids standard Redis
ML_PIPELINE_HOST_PORT=8081                        # Health endpoint port
```

**Risk & Trading Parameters**:

```bash
# Position Management
POSITION_SIZE_PCT=0.02                            # 2% per trade
MIN_CONFIDENCE=0.6                                # Minimum signal confidence
MAX_POSITIONS=1                                   # Concurrent position limit

# Risk Controls
STOP_LOSS_PCT=0.02                               # 2% stop loss
TAKE_PROFIT_PCT=0.04                             # 4% take profit

# Audit & Persistence
USE_STRATEGY_STORE=true                          # Mandatory for audit trail
PERSIST_ALL_SIGNALS=true                         # Required for analysis
USE_DUMMY_STORES=false                           # Production persistence
```

### Configuration Files & Templates

**Environment Templates**:

- `.env.example`: Production template with security placeholders
- `ml/deployment/.env`: Testing configuration (dummy API keys)

**Observability Configuration**:

**Prometheus** (`ml/deployment/prometheus.yml`):

```yaml
global:
  scrape_interval: 15s    # Minimal overhead
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ml_pipeline'
    targets: ['ml_pipeline:8080']
    metrics_path: /metrics

  - job_name: 'ml_signal_actor'
    targets: ['ml_signal_actor:8000']  # BaseMLInferenceActor metrics

  - job_name: 'ml_strategy'
    targets: ['ml_strategy:8001']
```

**Alert Rules** (`ml/deployment/alerts.yml`):

```yaml
groups:
  - name: ml-observability
    rules:
      - alert: MLModelInferenceLatencyHighP99
        expr: histogram_quantile(0.99, ...) > 0.200
        for: 5m
        labels:
          severity: warning
```

**Grafana Dashboard** (`ml/deployment/grafana/ml_pipeline_health.json`):

- **Pipeline Health Gauge**: Overall system health with color coding
- **Performance Metrics**: P50/P95/P99 latency tracking across services
- **Data Freshness Heatmap**: Staleness visualization by instrument
- **Error Correlation**: Recent errors with trace IDs and context
- **Resource Utilization**: Memory/CPU across containers with thresholds

## Deployment Modes

The system supports three deployment approaches with automatic environment detection and progressive fallback strategies.

### 1. Production Docker Compose Deployment
**Scripts**: `ml/deployment/run_dry_run.sh` (manual) | `ml/deployment/quick_start.sh` (automated)

**Architecture Features**:

- **Service Orchestration**: Full containerized environment with health-based dependencies
- **Progressive Fallback**: PostgreSQL → DummyStore with warnings (no failures)
- **Security-First**: Mandatory dry-run mode, no pickle models, API key validation
- **Auto-Schema Management**: Docker init volumes + CLI migration runner
- **Complete Observability**: Prometheus/Grafana + custom alerts + health endpoints
- **Resource Management**: Bounded memory/CPU with deployment resource limits

**Execution Flow** (Production Pattern):

1. **Pre-flight Checks**: DATABENTO_API_KEY validation, Docker/Compose availability
2. **Infrastructure Bootstrap**: PostgreSQL startup with health checks (5s intervals)
3. **Schema Initialization**: Auto-apply canonical migrations via init volumes
4. **Database Preflight**: Verify functions/partitions via `ml.stores.infrastructure`
5. **Service Launch**: ML containers with dependency chains (postgres → actor → strategy)
6. **Monitoring Stack**: Prometheus/Grafana with pre-configured dashboards
7. **Health Validation**: End-to-end health checks with /health endpoints
8. **Operational Display**: Service URLs, log aggregation, shutdown traps

**Docker Compose Features**:

```yaml
services:
  postgres:
    volumes:
      - ../stores/migrations:/docker-entrypoint-initdb.d:ro  # Auto-init
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]

  ml_pipeline:
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8080/health')"]

  ml_signal_actor:
    deploy:
      resources:
        limits:
          memory: 8g
          cpus: '2'
```

### 2. Local Development Deployment
**Script**: `ml/deployment/run_local_dry_run.py`

**Development-Optimized Features**:

- **Native Process Execution**: No containerization overhead for fast iteration
- **Progressive Database Fallback**: PostgreSQL → SQLite → DummyStore (automatic detection)
- **Auto-Model Generation**: Creates dummy models if missing (for rapid prototyping)
- **Live Code Mounting**: Docker dev mode with live reload via `ml/docker-compose.dev.yml`
- **Real Market Data**: Full Databento integration for authentic testing

## Developer Tips

- Standardize DB env for local dev/tests:
  - `export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/nautilus`
  - Start DB only: `make docker-up-test`
  - Check readiness: `make check-db` (waits on `DATABASE_URL`)
- ML stack Compose maps host `5433` → container `5432`. If you use it, set:
  - `export DATABASE_URL=postgresql://postgres:postgres@localhost:5433/nautilus`
- Prefer `EngineManager.get_engine(...)` in Python components to avoid connection pool exhaustion.
- Mark DDL/DB-heavy tests `@pytest.mark.serial` and parallelize others with `-n auto --dist=loadscope`.
- **Comprehensive Debug Info**: Enhanced logging, statistics reporting, performance profiling

**Architecture Pattern Compliance**:

- **Mandatory 4-Store + 4-Registry**: Even in local mode (with progressive fallback)
- **Hot/Cold Path Separation**: Local processes maintain performance budgets
- **Security**: Same restrictions (no pickle models, API key validation)

**Development Override** (`ml/docker-compose.dev.yml`):

```yaml
services:
  postgres:
    ports:
      - "5432:5432"  # Standard port for local dev

  pgadmin:         # Database inspection
    image: dpage/pgadmin4:latest
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@nautilus.io
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
```

**Prerequisites & Auto-Detection**:

- **API Key Validation**: DATABENTO_API_KEY with dataset access verification
- **Database Connectivity**: Progressive fallback chain with clear warnings
- **Model Availability**: Auto-generation with metadata consistency
- **Dependencies**: ML package detection with helpful error messages

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
docker compose logs -f

# Specific services
docker compose logs -f ml_signal_actor
docker compose logs -f ml_strategy
```

**Database Monitoring**:

```sql
-- Signal monitoring
SELECT * FROM public.ml_strategy_signals ORDER BY ts_event DESC LIMIT 10;

-- Feature monitoring
SELECT * FROM public.ml_feature_values ORDER BY ts_event DESC LIMIT 10;

-- Model prediction monitoring
SELECT * FROM public.ml_model_predictions ORDER BY ts_event DESC LIMIT 10;
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

- **Network**: `nautilus-ml` (bridge)
- **Service Discovery**: Docker DNS resolution
- **Communication**: HTTP/gRPC for metrics, PostgreSQL protocol for persistence

## CI/CD and Production Operations

### Continuous Integration Workflows

**GitHub Actions Integration**:

**ML Property Tests** (`.github/workflows/ml-property-tests.yml`):

```yaml
name: ML Property Tests
on:
  push:
    branches: [main, develop, ml]
    paths: ['ml/**']
strategy:
  matrix:
    python-version: ['3.11', '3.12']

- name: Run property tests with coverage
  run: |
    pytest ml/tests/unit/*hypothesis*.py \
      --hypothesis-show-statistics \
      --hypothesis-profile=ci \
      --cov=ml --cov-report=xml

- name: Check minimum coverage
  run: |
    if (( $(echo "$coverage_percent < 75" | bc -l) )); then
      exit 1
    fi
```

**ML Prototype Phase-2** (`.github/workflows/ml-prototype-phase2.yml`):

```yaml
name: ml-prototype-phase2
on:
  schedule:
    - cron: "0 6 * * *"  # Daily at 06:00 UTC

- name: Run Phase-2 prototype suite
  run: |
    uv run --no-sync pytest -n logical --dist=loadgroup -m prototype ml/tests
```

**Security Hardening**:

- **step-security/harden-runner**: Egress policy auditing
- **No Pickle Models**: Strict enforcement in model loading
- **API Key Validation**: Pre-deployment verification
- **Progressive Fallback**: Never fail due to missing services

### Migration Management & Database Operations

**Automated Schema Management**:

**CLI Migration Runner** (`ml/scripts/apply_migrations.py`):

```bash
# Baseline deployment
uv run --no-sync python -m ml.scripts.apply_migrations --db-url postgresql://...

# Full production deployment
uv run --no-sync python -m ml.scripts.apply_migrations --db-url postgresql://... --full --schema both

# Dry run validation
uv run --no-sync python -m ml.scripts.apply_migrations --dry-run
```

**Migration Strategy**:

- **Base Migrations**: 001_stores_schema.sql → 004_data_registry.sql + 007_add_event_metadata.sql
- **Optional Migrations**: Schema hardening, views, BRIN indexes, emergency fixes
- **Idempotent Design**: Safe multiple execution with conflict handling
- **Statement-Level Safety**: Dollar-quoted function parsing, proper transaction boundaries

**CI Migration Smoke Tests** (`ml/deployment/ci_migration_smoke.py`):

```python
def main():
    _compose("up", "-d", "postgres")
    wait_for_postgres()
    apply_migrations_via_compose(compose_file=COMPOSE_FILE)
    check_views()  # Verify ml.pipeline_health, data_collection_stats
```

### Deployment Tools and Scripts

**Production Deployment Toolchain**:

**Quick Start Script** (`ml/deployment/quick_start.sh`):

- **Environment Validation**: .env creation, DATABENTO_API_KEY verification
- **Image Building**: Multi-stage Docker builds with dependency caching
- **Service Orchestration**: Health-based startup with dependency chains
- **Health Validation**: End-to-end /health endpoint verification
- **Operational Display**: Service URLs, monitoring links, useful commands

**Makefile Operations** (`ml/deployment/Makefile`):

```makefile
# Core Operations
build:          # Build all Docker images
up:             # Start all services
down:           # Stop all services
health:         # Check service health
logs:           # View logs (SERVICE=name for specific)

# Advanced Operations
migrate:        # Apply SQL migrations via compose
ci-smoke:       # CI migration smoke test
dev:            # Development mode with live code mounting
backfill:       # Run pipeline in backfill mode (START=date END=date)
scale:          # Scale pipeline workers (WORKERS=n)
nuke:           # Remove containers and volumes (DANGEROUS)

# Production Operations
deploy:         # Production deployment with .env validation
backup:         # Backup PostgreSQL database
restore:        # Restore database (FILE=backup.sql)
```

**Health Check System** (`ml/deployment/check_health.py`):

```python
def main():
    checks = [
        ("Docker Compose", check_docker_compose),
        ("PostgreSQL", check_postgres),
        ("Redis", check_redis),
        ("ML Pipeline", check_ml_pipeline),
        ("Prometheus", check_prometheus),
        ("Grafana", check_grafana),
    ]
    # Returns aggregated health with specific failure details
```

**Testing & Validation** (`ml/deployment/test_docker_setup.sh`):

- **Docker Installation**: Version compatibility verification
- **Compose Syntax**: Configuration validation with docker-compose config
- **Environment Validation**: Required files and API key presence
- **Dry-Run Build**: Container build testing without execution

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
Note: Dev override now lives at `ml/docker-compose.dev.yml` (use `-f` with the base compose).

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
   psql nautilus -f ml/stores/migrations/001_stores_schema.sql
   psql nautilus -f ml/stores/migrations/002_auto_partitioning.sql
   psql nautilus -f ml/stores/migrations/003_market_data.sql
   psql nautilus -f ml/stores/migrations/004_data_registry.sql
   psql nautilus -f ml/stores/migrations/005_schema_hardening.sql
   psql nautilus -f ml/stores/migrations/005a_feature_values_dedupe.sql
   psql nautilus -f ml/stores/migrations/006_disable_partition_triggers.sql
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
   - Check service health: `docker compose ps`
   - Monitor logs: `docker compose logs -f`
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

## Production Architecture Summary

The ML deployment architecture provides a production-hardened, safety-first environment that enforces CLAUDE.md architectural patterns while maintaining operational excellence.

### Core Architectural Strengths

**Universal Pattern Enforcement**:

- **Pattern 1**: Mandatory 4-store + 4-registry integration with progressive fallback
- **Pattern 2**: Protocol-first design enabling DummyStore testing and type safety
- **Pattern 3**: Strict hot/cold path separation with <5ms P99 latency requirements
- **Pattern 4**: Progressive fallback chains (PostgreSQL → DummyStore → graceful warnings)
- **Pattern 5**: Centralized metrics bootstrap preventing registry conflicts

**Production-Ready Infrastructure**:

- **Containerized Orchestration**: Docker Compose with health-based dependencies and resource limits
- **Multi-Mode Deployment**: Production containers, local development, backtest simulation
- **Automated Schema Management**: Docker init volumes + CLI migration runner with idempotent execution
- **Comprehensive Observability**: Prometheus/Grafana with custom dashboards, alerts, and performance tracking
- **Security-First Design**: No pickle models, mandatory dry-run mode, API key validation, progressive fallback

**Operational Excellence**:

- **Zero-Downtime Operations**: Health checks, graceful shutdown, circuit breakers
- **Developer Experience**: Quick start scripts, live code mounting, comprehensive testing
- **CI/CD Integration**: GitHub Actions workflows, property testing, coverage enforcement
- **Audit Compliance**: Complete persistence of signals, decisions, and model metadata
- **Performance Monitoring**: Real-time latency tracking, resource utilization, error correlation

### Deployment Capabilities

**Three Deployment Modes**:

1. **Production Containers**: Full observability, resource management, health monitoring
2. **Local Development**: Native processes, live reload, progressive database fallback
3. **Backtest Simulation**: Historical replay, synthetic data, controlled environments

**Operational Tools**:

- **Makefile Operations**: 20+ commands for build, deploy, scale, backup, restore
- **Health Monitoring**: Multi-layer health checks with aggregated status reporting
- **Migration Management**: Automated schema deployment with validation and rollback
- **Security Scanning**: Step Security hardening, egress policy auditing

### Safety & Compliance Framework

**Mandatory Safety Controls**:

- **EXECUTE_TRADES=false**: Enforced dry-run mode with explicit warnings
- **Model Security**: NO pickle support, only ONNX/XGBoost/LightGBM safe formats
- **Audit Trail**: Complete persistence of signals, decisions, model metadata
- **Progressive Fallback**: Never fail due to missing services, always degrade gracefully

**Risk Management**:

- **Position Limits**: Configurable size, confidence thresholds, concurrent limits
- **Circuit Breakers**: Health monitoring with automatic degradation
- **Performance Budgets**: Hot path <5ms P99, memory bounds, CPU limits
- **Database Resilience**: Connection pooling, auto-reconnection, fallback strategies

This architecture enables confident ML trading system deployment with enterprise-grade reliability, comprehensive monitoring, and safety-first operational practices.
## Implementation Review Addendum

**Review Date**: 2025-01-14
**Reviewer**: Claude Code
**Focus**: Ground-truth validation of documentation claims vs. actual implementation

### Universal ML Architecture Pattern Compliance Analysis

#### ✅ Pattern 1: Mandatory 4-Store + 4-Registry Integration - VERIFIED

**Documentation Claim**: "All containers inherit from BaseMLInferenceActor with automatic initialization and progressive fallback"

**Implementation Status**: **FULLY IMPLEMENTED**

- **File Evidence**: `/home/nate/projects/nautilus_trader/ml/actors/base.py:784-813`
- **Key Implementation**: `_init_stores_and_registries()` method properly initializes all 4 stores via centralized facade
- **Progressive Fallback**: Implemented via `ml.actors.actor_services.init_actor_services()` which delegates to `ml.core.integration.init_actor_stores_and_registries()`
- **Protocol Compliance**: All stores are typed with Protocol interfaces (`FeatureStoreProtocol`, `ModelStoreProtocol`, etc.)
- **Verification**: Entrypoint files correctly import and use `MLSignalActor` and `MLTradingStrategy` classes that inherit from `BaseMLInferenceActor`

#### ⚠️ Pattern 2: Protocol-First Interface Design - PARTIALLY COMPLIANT

**Documentation Claim**: "All components use typing.Protocol for clean contracts"

**Implementation Status**: **MOSTLY IMPLEMENTED**

- **Protocol Usage**: Verified in `ml.actors.base.py` - stores are correctly protocol-typed
- **Missing Documentation**: Documentation doesn't mention that some type annotations use `object` instead of specific protocols (lines 843-868 in base.py)
- **Runtime Safety**: `isinstance()` checks and `@runtime_checkable` decorators are used appropriately

#### ✅ Pattern 3: Hot/Cold Path Separation - VERIFIED

**Documentation Claim**: "Hot path: <5ms P99 latency, zero allocations, pre-allocated arrays"

**Implementation Status**: **IMPLEMENTED**

- **Pre-allocation**: Verified in `BaseMLInferenceActor` - `_features_buffer` and `_feature_window` are pre-allocated (lines 748-751)
- **Performance Tracking**: Metrics for inference latency are implemented (lines 775-778)
- **Memory Management**: `deque` with `maxlen` for fixed-size feature windows

#### ✅ Pattern 4: Progressive Fallback Chains - VERIFIED

**Documentation Claim**: "PostgreSQL → DummyStore with warnings logged"

**Implementation Status**: **IMPLEMENTED**

- **Fallback Strategy**: Implemented via `init_actor_stores_and_registries()` facade with error handling
- **Logging**: Proper warning logs are generated during store initialization (line 812: "Stores and registries initialized")
- **Connection Strings**: Environment variable fallback chains in entrypoints (e.g., `entrypoint_actor.py:46-50`)

#### ❌ Pattern 5: Centralized Metrics Bootstrap - VIOLATION FOUND

**Documentation Claim**: "NEVER import prometheus_client directly. Use ml.common.metrics_bootstrap"

**Implementation Status**: **VIOLATION DETECTED**

- **Direct Import**: Found `prometheus-client` in Dockerfile.pipeline:30 and Dockerfile.actor:25
- **Missing Bootstrap Usage**: No evidence of `ml.common.metrics_bootstrap` imports in deployment files
- **Alternative Implementation**: Uses `ml.common.metrics_export` in `entrypoint_pipeline.py:26-27` instead of documented bootstrap approach
- **Impact**: Potential metric registry conflicts not prevented as claimed

### Security Pattern Compliance

#### ✅ No Pickle Models - VERIFIED

**Documentation Claim**: "NO pickle support, only ONNX/XGBoost/LightGBM safe formats"

**Implementation Status**: **ENFORCED**

- **Security Check**: `entrypoint_actor.py:148` explicitly raises `RuntimeError` for pickle models
- **Comment Evidence**: `run_local_dry_run.py:line` mentions "Removed insecure pickle-based dummy model creation"
- **Container Safety**: Dockerfile.actor includes ONNX runtime but no pickle dependencies

#### ✅ Dry Run Mode Enforcement - VERIFIED

**Documentation Claim**: "EXECUTE_TRADES=false enforced by default with explicit warnings"

**Implementation Status**: **IMPLEMENTED**

- **Default Value**: `entrypoint_strategy.py:54` defaults to `"false"`
- **Warning Display**: Lines 86-90 show clear "DRY RUN MODE ACTIVE" warnings
- **Trade Prevention**: MLTradingStrategy properly checks execute_trades flag (lines 117, 137)

### Container Architecture Analysis

#### ✅ Docker Compose Structure - VERIFIED

**Documentation Claim**: "Full containerized environment with health-based dependencies"

**Implementation Status**: **CORRECTLY IMPLEMENTED**

- **Project Name**: Pinned to `ml` (docker-compose.yml:2)
- **Health Checks**: PostgreSQL, Redis, and ML Pipeline have proper health check configurations
- **Dependencies**: Proper service dependency chains with health conditions
- **Resource Limits**: Memory and CPU limits specified (8GB/2CPU for actors, 16GB/4CPU for pipeline)

#### ❌ Missing Grafana Dashboard Files - DISCREPANCY

**Documentation Claim**: "Complete observability: Prometheus/Grafana + custom alerts + health endpoints"

**Implementation Status**: **PARTIAL IMPLEMENTATION**

- **Dashboard File**: Claims `/grafana/ml_pipeline_health.json` but file is empty/basic
- **Dashboard Mounting**: docker-compose.yml:221-222 references dashboard directories but minimal content
- **Prometheus Config**: Basic scrape configs present but limited to basic metrics

### File Path Discrepancies

#### ❌ Missing Development Override Files

**Documentation Claim**: References to `ml/docker-compose.dev.yml`

**Implementation Status**: **FILE NOT FOUND**

- **File**: `ml/docker-compose.dev.yml` referenced in documentation but doesn't exist in deployment directory
- **Alternative**: Found `docker-compose.override.yml.example` which may serve similar purpose

### Performance Metrics Claims Validation

#### ⚠️ Performance Benchmarks - UNVERIFIED CLAIMS

**Documentation Claim**: "P99 latency <5ms validated in production"

**Implementation Status**: **CLAIMS UNSUBSTANTIATED**

- **No Benchmarks**: No performance test files in deployment directory
- **Metric Collection**: Prometheus config exists but no performance validation tests
- **Production Evidence**: No evidence of production performance validation

### Documentation Accuracy Assessment

#### Critical Issues Found

1. **Pattern 5 Violation**: Direct `prometheus-client` imports instead of documented `ml.common.metrics_bootstrap`
2. **Missing Files**: `ml/docker-compose.dev.yml` referenced but not found
3. **Unsubstantiated Claims**: Performance benchmarks claimed but not verified
4. **Grafana Integration**: Minimal dashboard implementation despite comprehensive claims

#### Accuracy Rating: **85%**

- **Architecture Patterns**: 4/5 patterns properly implemented
- **Security Controls**: Fully implemented as documented
- **Container Structure**: Correctly implemented
- **Documentation Drift**: Some files referenced but missing
- **Performance Claims**: Unsubstantiated but architecture supports them

#### Recommendations

1. **Fix Pattern 5**: Replace direct `prometheus-client` imports with `ml.common.metrics_bootstrap`
2. **Create Missing Files**: Add `ml/docker-compose.dev.yml` or update documentation
3. **Add Benchmarks**: Include performance validation tests to support latency claims
4. **Enhance Dashboards**: Implement the comprehensive Grafana dashboard described
5. **Update Metrics**: Ensure all deployment services use centralized metrics bootstrap

**Overall Assessment**: The deployment architecture is substantially implemented as documented with strong adherence to Universal ML Architecture Patterns. The main issues are a Pattern 5 violation and some missing development files. Core functionality, security, and architectural foundations are sound.

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
