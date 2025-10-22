# ML Deployment Architecture Context

**Last Updated**: 2025-01-19
**Implementation Status**: 98% Complete – Production-ready containerized deployment with comprehensive observability

---

## Executive Summary

The ML deployment architecture provides a production-hardened, containerized environment for running Nautilus Trader ML strategies with real-time market data integration. Built on Docker Compose v2 with comprehensive monitoring, this system enforces the mandatory 4-store + 4-registry pattern, implements progressive fallback strategies, maintains strict security controls (ONNX-only models, dry-run-by-default), and provides multiple deployment modes (production containers, local development, backtest simulation).

**Core Capabilities:**
- **Multi-service orchestration** with health-based dependencies and graceful degradation
- **Three deployment modes** supporting different use cases (production/dev/backtest)
- **Comprehensive observability** via Prometheus/Grafana with custom dashboards and alerts
- **Security-first design** with ONNX-only model enforcement and mandatory dry-run mode
- **Progressive fallback** chains ensuring system resilience (PostgreSQL → DummyStore)
- **Automated schema management** via Docker init volumes and CLI migration runner

---

## Architecture Overview

### Service Topology

```
                    ┌─────────────────────────────────────────┐
                    │       Observability Layer               │
                    │  Prometheus + Grafana + Alerts          │
                    └────────┬──────────────────┬─────────────┘
                             │                  │
           ┌─────────────────┼──────────────────┼─────────────────┐
           │                 │                  │                 │
           ▼                 ▼                  ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
    │ Pipeline │      │  Signal  │      │ Strategy │      │Dashboard │
    │ Service  │      │  Actor   │      │ Service  │      │ Control  │
    │          │◄────►│          │◄────►│          │◄────►│  Plane   │
    │ (cold)   │      │  (hot)   │      │  (hot)   │      │ (cold)   │
    └────┬─────┘      └────┬─────┘      └────┬─────┘      └────┬─────┘
         │                 │                  │                 │
         └─────────────────┼──────────────────┼─────────────────┘
                           │                  │
              ┌────────────┴──────────────────┴────────────┐
              │     4-Store + 4-Registry Persistence       │
              └────────────┬───────────────┬────────────────┘
                           │               │
                  ┌────────┴────┐    ┌────┴────┐
                  │  PostgreSQL │    │  Redis  │
                  │  Database   │    │  Bus    │
                  └─────────────┘    └─────────┘
```

### Project Structure

```
ml/deployment/
├── docker-compose.yml              # Production stack (project: ml)
├── docker-compose.test.yml         # Test stack (project: ml-test)
├── docker-compose.override.yml     # Local overrides (external network bridge)
├── .env.example                    # Environment variable template
├── README.md                       # Quick start and troubleshooting guide
│
├── entrypoint_pipeline.py          # Pipeline service entrypoint (410 lines)
├── entrypoint_actor.py             # Signal actor entrypoint (639 lines)
├── entrypoint_strategy.py          # Trading strategy entrypoint (449 lines)
├── entrypoint_mock.py              # Mock data testing entrypoint (388 lines)
│
├── Dockerfile.pipeline             # Pipeline container (50 lines)
├── Dockerfile.actor                # Actor container (47 lines)
├── Dockerfile.strategy             # Strategy container (similar to actor)
│
├── check_health.py                 # Multi-service health checker (198 lines)
├── migrations.py                   # DB migration utility (166 lines)
├── ci_migration_smoke.py           # CI migration smoke tests (99 lines)
│
├── security.py                     # Model artifact validation (28 lines)
├── scheduling_utils.py             # Schedule parsing utilities (105 lines)
├── metrics_http.py                 # Flask health/metrics server (44 lines)
├── mock_databento.py               # Synthetic data generator
│
├── prometheus.yml                  # Prometheus scrape configuration
├── alerts.yml                      # Alert rules (165 lines)
├── grafana/
│   └── ml_pipeline_health.json     # Dashboard definition (1500+ lines)
│
└── Makefile                        # Deployment convenience targets (144 lines)
```

---

## Docker Compose Configuration

### Production Stack (`docker-compose.yml`)

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/docker-compose.yml` (331 lines)

**Project Name**: `ml` (line 2) – Ensures consistent naming across environments

#### Services Breakdown

| Service | Purpose | Hot/Cold | Resources | Ports |
|---------|---------|----------|-----------|-------|
| `postgres` | Primary data persistence | Cold | Default | 5433:5432 |
| `postgres_playground` | Sandbox database | Cold | Default | 5435:5432 |
| `redis` | Message bus and caching | Cold | Default | 6380:6379 |
| `streaming_persistence_worker` | Event stream persistence | Cold | Default | None |
| `ml_signal_actor` | Real-time ML inference | **Hot** | 8GB/2CPU | 8000:8000 |
| `ml_strategy` | Trading decision execution | **Hot** | 8GB/2CPU | 8001:8001 |
| `ml_pipeline` | Data collection + features | Cold | 16GB/4CPU | 8081:8080 |
| `prometheus` | Metrics collection | Cold | Default | 9090:9090 |
| `grafana` | Visualization dashboard | Cold | Default | 3000:3000 |
| `ml_dashboard` | Control plane UI | Cold | Default | 8010:8010 |

#### Service Dependencies

```yaml
ml_pipeline:
  depends_on:
    postgres:
      condition: service_healthy  # Waits for pg_isready success

ml_signal_actor:
  depends_on: [postgres, redis]

ml_strategy:
  depends_on: [postgres, redis, ml_signal_actor]

grafana:
  depends_on: [prometheus]

ml_dashboard:
  depends_on: [prometheus, grafana, streaming_persistence_worker]
```

**Health Checks** (lines 22-26, 42-46, 57-61, 240-246):
- PostgreSQL: `pg_isready -U postgres` every 5s
- Redis: `redis-cli ping` every 5s
- Pipeline: HTTP GET `/health` every 30s with 60s start period

#### Named Volumes

```yaml
volumes:
  postgres_data:                    # Persistent database storage
  postgres_playground_data:         # Sandbox database storage
  prometheus_data:                  # Metrics time series
  grafana_data:                     # Dashboard configurations
  streaming_state_data:             # Streaming training state snapshots
```

#### Network Configuration

```yaml
networks:
  nautilus-ml:
    driver: bridge
    name: nautilus-ml              # Explicit network name for service discovery
```

**Container-to-Container Communication**:
- Services communicate via Docker DNS: `postgres:5432`, `redis:6379`
- No localhost references inside containers (handled via service names)

---

## Service Entrypoints

### 1. Pipeline Service (`entrypoint_pipeline.py`)

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/entrypoint_pipeline.py` (413 lines)

**Purpose**: Automated data collection and feature computation (COLD PATH)

**Key Components**:

```python
class PipelineRunner:
    def __init__(self) -> None:
        self.scheduler: DataScheduler | None = None
        self.running: bool = False
        self._shutdown_event = threading.Event()

        # Graceful shutdown handlers (lines 115-130)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
```

**Operational Modes** (lines 219, 244-252):

1. **Daily Mode** (`PIPELINE_MODE=daily`):
   - Immediate execution on startup (line 289)
   - Scheduled runs via `PIPELINE_SCHEDULE` (cron or HH:MM format)
   - Falls back to `REALTIME_INTERVAL` if schedule invalid
   - Cooperative shutdown via threading.Event (lines 314-327)

2. **Backfill Mode** (`PIPELINE_MODE=backfill`):
   - Single `run_daily_update()` invocation (line 275)
   - Maps to daily update for simplicity (full loop TBD)

3. **Realtime Mode** (`PIPELINE_MODE=realtime`):
   - Continuous updates with retry logic (lines 338-376)
   - Exponential backoff via `ml.common.retry_utils` (lines 343-363)
   - Configurable interval via `REALTIME_INTERVAL` (default 300s)

**Configuration Loading** (lines 132-165):
```python
def _create_config(self) -> SchedulerConfig:
    # Parse comma-separated symbols (line 138)
    raw_symbols = os.environ.get("UNIVERSE_SYMBOLS", "SPY.XNAS")
    symbols = [s.strip() for s in raw_symbols.split(",") if s.strip()]

    # Databento configuration (lines 141-145)
    databento_config = DatabentoConfig(
        dataset=os.environ.get("DATABENTO_DATASET", "EQUS.MINI"),
        schema=os.environ.get("DATABENTO_SCHEMA", "ohlcv-1m"),
        stype_in=os.environ.get("DATABENTO_STYPE_IN", "raw_symbol"),
    )

    # Universe expansion (lines 148-152)
    universe = UniverseConfig(expansion_mode=os.environ.get("UNIVERSE_MODE", "moderate"))
    full_universe = list(dict.fromkeys(symbols + universe.get_full_universe()))
```

**Unified Ingestion Flags** (lines 231-238):
```python
use_orchestrator = parse_bool_env(os.environ.get("USE_ORCHESTRATOR"))
dual_write = parse_bool_env(os.environ.get("DUAL_WRITE"))
self.scheduler = DataScheduler(
    catalog=catalog,
    config=config,
    use_orchestrator=use_orchestrator,  # Unified control path
    dual_write=dual_write,              # SQL + Parquet in one pass
)
```

**Health Endpoints** (lines 84-98):
- `/health`: Returns pipeline status with `last_run` timestamp
- `/metrics`: Prometheus metrics via `ml.common.metrics_export.generate_latest()`

**Flask Server** (lines 384-394):
```python
health_host = os.environ.get("HEALTH_CHECK_HOST", "127.0.0.1")
health_thread = threading.Thread(
    target=lambda: app.run(
        host=health_host,
        port=int(os.environ.get("HEALTH_CHECK_PORT", "8080")),
        debug=False,
    ),
)
health_thread.daemon = True
health_thread.start()
```

**Observability Bootstrap** (lines 396-404):
```python
# Auto-start observability flushing if configured
try:
    mgr: MLIntegrationManager = MLIntegrationManager.__new__(MLIntegrationManager)
    auto_start_if_configured(mgr)  # ml.observability.bootstrap
except Exception:
    logger.debug("Observability auto-start skipped", exc_info=True)
```

### 2. Signal Actor Service (`entrypoint_actor.py`)

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/entrypoint_actor.py` (639 lines)

**Purpose**: Real-time ML inference for signal generation (HOT PATH)

**Key Components**:

```python
class MLSignalActorNode:
    def __init__(self) -> None:
        self.node: TradingNode | None = None
        self.running: bool = False
        self._tasks: list[asyncio.Task[Any]] = []
        self._healthy: bool = False
```

**Security Enforcement** (lines 118-127):
```python
# Enforce ONNX-only and existence
try:
    assert_allowed_model_path(model_path)  # ml.deployment.security
except ValueError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)
if not Path(model_path).exists():
    print(f"ERROR: Model not found at {model_path}")
    sys.exit(1)
```

**Multi-Instrument Configuration** (lines 221-244):
```python
universe: list[str] | None = _get_list("ACTOR_UNIVERSE") or _get_list("UNIVERSE_SYMBOLS")
if not universe:
    # Default multi-instrument universe (lines 223-231)
    universe = [
        "SPY.EQUS",   # ETFs use EQUS aggregated venue
        "QQQ.EQUS",
        "AAPL.XNAS",  # Equities use XNAS symbols
        "MSFT.XNAS",
        "NVDA.XNAS",
    ]

actor_config = MLSignalActorConfig(
    **actor_kwargs,
    max_batch_size=max_batch_size,            # Default 128
    feature_dim=feature_dim,                  # Default 64
    initial_universe=universe,
    flush_max_latency_ms=flush_max_latency_ms,  # Default 0 (disabled)
)
```

**Live Data Recording** (lines 313-469):
```python
live_record_enable = os.getenv("ML_LIVE_RECORD_ENABLE", "1") in {"1", "true", "yes"}
if live_record_enable:
    datasets_csv = os.getenv("ML_LIVE_RECORD_DATASETS", "bars")
    dataset_tokens = {t.strip().lower() for t in datasets_csv.split(",") if t.strip()}

    # Create RecorderActor with LiveDataRecorder (lines 333-347)
    recorder = LiveDataRecorder(
        data_store=mgr.data_store,
        data_registry=mgr.data_registry,
        buffer_size=int(os.getenv("ML_LIVE_RECORD_BUFFER", "1000")),
        flush_interval_ms=int(os.getenv("ML_LIVE_RECORD_FLUSH_MS", "1000")),
    )
    rec_actor = RecorderActor(
        recorder=recorder,
        record_bars=record_bars,
        record_quotes=record_quotes,
        record_trades=record_trades,
    )
    self.node.trader.add_actor(rec_actor)
```

**Progressive Fallback to Catalog** (lines 360-469):
```python
except Exception as exc:
    # JSON/catalog fallback when PostgreSQL unavailable (lines 362-469)
    try:
        catalog_path = os.getenv("CATALOG_PATH", "").strip()
        if not catalog_path:
            raise RuntimeError("CATALOG_PATH required for fallback")

        from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
        catalog = ParquetDataCatalog(catalog_path)
        writer = ParquetCatalogMarketDataWriter(catalog)
        data_store_fallback = CatalogWriteFacade(writer)

        # JSON DataRegistry for events/watermarks (lines 374-391)
        registry_path = Path.home() / ".nautilus" / "ml" / "registry"
        persistence = PersistenceConfig(
            backend=BackendType.JSON,
            json_path=registry_path,
        )
        data_registry = DataRegistry(
            registry_path=registry_path,
            persistence_config=persistence,
        )
```

**Databento Configuration** (lines 261-280):
```python
venue_dataset_map = {"EQUS": "EQUS.MINI"}  # Anonymized EQUS for EQUS.MINI

data_config: DatabentoDataClientConfig = DatabentoDataClientConfig(
    api_key=databento_api_key or "",
    http_gateway="https://hist.databento.com",
    live_gateway="wss://stream.databento.com",
    use_exchange_as_venue=False,
    venue_dataset_map=venue_dataset_map,
)

node_config = TradingNodeConfig(
    trader_id=TraderId("ML-ACTOR-001"),
    data_engine=data_engine_cfg,
    data_clients={"DATABENTO": data_config},
    exec_clients={},  # No execution for signal actor
)
```

**Metrics Server** (lines 590-600):
```python
port = int(os.getenv("METRICS_PORT", "8000"))
host = os.getenv("METRICS_HOST", "127.0.0.1")
app = build_app(lambda: actor_node._healthy)  # ml.deployment.metrics_http
http_thread = threading.Thread(
    target=lambda: app.run(host=host, port=port, debug=False),
    daemon=True,
)
http_thread.start()
```

**Event Loop Management** (lines 624-638):
```python
def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    """
    Return a running event loop for the current thread.

    Python 3.12 no longer auto-creates a loop when calling get_event_loop.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
```

### 3. Trading Strategy Service (`entrypoint_strategy.py`)

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/entrypoint_strategy.py` (449 lines)

**Purpose**: Trading decision execution based on ML signals (HOT PATH decision making)

**Key Components**:

```python
class MLStrategyNode:
    def __init__(self) -> None:
        self.node: TradingNode | None = None
        self.running: bool = False
        self._tasks: list[asyncio.Task[Any]] = []
        self._healthy: bool = False
```

**Mandatory Dry-Run Mode** (lines 79-116):
```python
# DRY RUN MODE (default to false)
execute_trades = os.getenv("EXECUTE_TRADES", "false").lower() == "true"

print("=" * 80)
print("ML TRADING STRATEGY - CONTAINER MODE")
print("=" * 80)
print(f"Execute Trades: {execute_trades} {'(DRY RUN MODE)' if not execute_trades else '(LIVE MODE)'}")
print("=" * 80)

if not execute_trades:
    print("\n⚠️  DRY RUN MODE ACTIVE ⚠️")
    print("Strategy will process signals and make decisions")
    print("but will NOT submit actual orders to the exchange")
    print("=" * 80)
```

**Strategy Configuration** (lines 119-140):
```python
strategy_config = MLStrategyConfig(
    strategy_id=strategy_id,
    instrument_id=instrument_id,
    ml_signal_source=ml_signal_source,
    position_size_pct=position_size_pct,      # Default 2%
    min_confidence=min_confidence,            # Default 0.6
    max_positions=max_positions,              # Default 1
    stop_loss_pct=stop_loss_pct,              # Default 2%
    take_profit_pct=take_profit_pct,          # Default 4%
    use_strategy_store=use_strategy_store,    # Default true
    strategy_store_config=(
        {
            "connection_string": db_connection,
            "batch_size": 100,
            "flush_interval_ms": 1000,
        }
        if use_strategy_store
        else None
    ),
    persist_all_signals=persist_all_signals,  # Default true
    execute_trades=execute_trades,            # DRY RUN CONTROL
)
```

**Graceful Shutdown with Statistics** (lines 274-318):
```python
async def shutdown(self) -> None:
    print("\nShutting down...")
    self.running = False
    self._healthy = False

    if self.node:
        # Print final statistics (lines 285-307)
        print("\n" + "=" * 80)
        print("FINAL STATISTICS")
        print("=" * 80)
        try:
            node_any = cast(Any, self.node)
            trader = getattr(node_any, "trader", None)
            strategies = trader.strategies() if trader and hasattr(trader, "strategies") else {}

            if strategies:
                strategy = next(iter(strategies.values()))
                print(f"Signals Received: {getattr(strategy, '_signals_received', 0)}")
                print(f"Dry Run Trades: {getattr(strategy, '_dry_run_trades', 0)}")
                print(f"Execute Trades Setting: {getattr(strategy._config, 'execute_trades', False)}")
        except Exception:
            print("Signals Received: 0")
            print("Dry Run Trades: 0")
```

**Health Heartbeat Window** (lines 361-400):
```python
def _run_health_heartbeat(self, *, reason: str) -> bool:
    """
    Optional heartbeat window for tests to verify strategy lifecycle.

    Enabled via ML_STRATEGY_HEARTBEAT_ENABLED (default: enabled).
    """
    if os.getenv("ML_STRATEGY_HEARTBEAT_ENABLED", "1").strip().lower() in {"0", "false", "off"}:
        return False

    duration = self._get_heartbeat_float("ML_STRATEGY_HEARTBEAT_DURATION_SECONDS", 120.0)
    if duration <= 0.0:
        return False

    interval = max(0.5, self._get_heartbeat_float("ML_STRATEGY_HEARTBEAT_INTERVAL_SECONDS", 5.0))
    deadline = time.monotonic() + duration

    logger.info("Entering strategy heartbeat window", extra={"reason": reason, "duration_seconds": duration})
    self._healthy = True
    try:
        while time.monotonic() < deadline:
            time.sleep(interval)
    finally:
        self._healthy = False
        logger.info("Strategy heartbeat window expired", extra={"reason": reason})
    return True
```

---

## Health Checking System

### Multi-Service Health Checker (`check_health.py`)

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/check_health.py` (198 lines)

**Purpose**: Comprehensive health validation for all deployment services

**Health Check Functions**:

```python
def check_postgres() -> bool:
    """Check PostgreSQL via docker compose exec (lines 59-70)"""
    command = _compose_command("exec", "-T", "postgres", "pg_isready", "-U", "postgres")
    result = run_command(command, capture_output=True, text=True, check=False, timeout=10)
    return result.returncode == 0

def check_redis() -> bool:
    """Check Redis via docker compose exec (lines 73-83)"""
    command = _compose_command("exec", "-T", "redis", "redis-cli", "ping")
    result = run_command(command, capture_output=True, text=True, check=False, timeout=10)
    return "PONG" in (result.stdout or "")

def check_ml_pipeline() -> bool:
    """Check ML Pipeline via HTTP endpoint (lines 86-96)"""
    port = os.environ.get("ML_PIPELINE_HOST_PORT", "8080")
    response = requests.get(f"http://localhost:{port}/health", timeout=5)
    return response.status_code == 200

def check_prometheus() -> bool:
    """Check Prometheus health endpoint (lines 99-107)"""
    response = requests.get("http://localhost:9090/-/healthy", timeout=5)
    return response.status_code == 200

def check_grafana() -> bool:
    """Check Grafana API health (lines 110-118)"""
    response = requests.get("http://localhost:3000/api/health", timeout=5)
    return response.status_code == 200

def check_docker_compose() -> bool:
    """Check Docker Compose services via JSON output (lines 121-154)"""
    command = _compose_command("ps", "--format", "json")
    result = run_command(command, capture_output=True, text=True, check=False, timeout=10)

    if result.returncode != 0:
        return False

    stdout = result.stdout.strip()
    if stdout:
        try:
            services = json.loads(stdout)
            required = {"postgres", "ml_pipeline"}
            running = {s.get("Service") for s in services if s.get("State") == "running"}
            return required.issubset(running)
        except Exception:
            return False
    return False
```

**Docker Compose Command Resolution** (lines 27-40):
```python
def _compose_base() -> list[str]:
    """Resolve docker-compose command invocation"""
    override = os.environ.get("DOCKER_COMPOSE_BIN")
    if override:
        return [override]
    docker_compose = shutil.which("docker-compose")
    if docker_compose:
        return [docker_compose]
    docker = shutil.which("docker")
    if docker:
        return [docker, "compose"]
    raise FileNotFoundError("docker-compose binary not found in PATH")
```

**Health Check Aggregator** (lines 157-194):
```python
def main() -> None:
    checks = [
        ("Docker Compose", check_docker_compose),
        ("PostgreSQL", check_postgres),
        ("Redis", check_redis),
        ("ML Pipeline", check_ml_pipeline),
        ("Prometheus", check_prometheus),
        ("Grafana", check_grafana),
    ]

    all_healthy = True
    results = []

    for name, check_func in checks:
        print(f"Checking {name}...", end=" ")
        healthy, message = check_service_health(name, check_func)
        print(f"[{'✓' if healthy else '✗'}] {message}")
        results.append((name, healthy, message))
        if not healthy:
            all_healthy = False

    if all_healthy:
        print("✓ All services are healthy!")
        sys.exit(0)
    else:
        print("✗ Some services are unhealthy. Please check logs:")
        print("  make logs SERVICE=<service_name>")
        sys.exit(1)
```

### Metrics HTTP Server (`metrics_http.py`)

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/metrics_http.py` (44 lines)

**Purpose**: Lightweight Flask server for /health and /metrics endpoints

```python
def build_app(is_healthy: Callable[[], bool]) -> Flask:
    """
    Create Flask app with health and metrics endpoints.

    Uses ml.common.metrics_export facade to avoid direct prometheus_client imports.
    """
    app = Flask(__name__)

    @app.get("/health")
    def health() -> tuple[Response, int]:
        status = {"healthy": bool(is_healthy())}
        return jsonify(status), 200 if status["healthy"] else 503

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    return app
```

**Usage in Entrypoints**:
```python
# Actor entrypoint (lines 590-600)
app = build_app(lambda: actor_node._healthy)
http_thread = threading.Thread(
    target=lambda: app.run(host="127.0.0.1", port=8000, debug=False),
    daemon=True,
)
http_thread.start()

# Strategy entrypoint (lines 414-425)
app = build_app(lambda: strategy_node._healthy)
http_thread = threading.Thread(
    target=lambda: app.run(host="127.0.0.1", port=8001, debug=False),
    daemon=True,
)
http_thread.start()
```

---

## Database Migration System

### Migration Utility (`migrations.py`)

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/migrations.py` (166 lines)

**Purpose**: Typed API for listing and applying canonical migrations via Docker Compose

**Key Functions**:

```python
MIGRATIONS_DIR: Path = Path("ml/stores/migrations").resolve()  # line 34

def list_migration_files() -> list[Path]:
    """
    Return migration files sorted by filename (lexicographic).

    Only .sql files included. Never raises for missing directory.
    """
    if not MIGRATIONS_DIR.exists():
        return []
    files = sorted(p for p in MIGRATIONS_DIR.iterdir() if p.suffix == ".sql")
    return list(files)

def apply_migrations_via_compose(
    *,
    compose_file: Path | None,
    database: str = "nautilus",
    user: str = "postgres",
    migrations: Iterable[Path] | None = None,
) -> None:
    """
    Apply migrations to running Postgres container via compose exec.

    Reads SQL from host and pipes to psql stdin to avoid bind-mount confusion.
    """
    files = list(migrations) if migrations is not None else list_migration_files()
    for file in files:
        sql = file.read_text(encoding="utf-8")
        cmd = _compose_cmd(compose_file) + [
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            user,
            database,
        ]
        try:
            COMMAND_RUNNER(cmd, input=sql, text=True, timeout=60, log=logger)
        except SubprocessExecutionError as exc:
            logger.error("migration_failed file=%s returncode=%s", file, exc.returncode, exc_info=True)
            raise
```

**CLI Interface** (lines 142-162):
```python
def main() -> None:
    parser = argparse.ArgumentParser(description="ML DB migrations helper")
    parser.add_argument("--apply", action="store_true", help="Apply migrations using docker compose exec")
    parser.add_argument("--compose-file", type=Path, default=Path("ml/deployment/docker-compose.yml"))
    args = parser.parse_args()

    if args.apply:
        apply_migrations_via_compose(compose_file=args.compose_file)
    else:
        for p in list_migration_files():
            print(p)
```

**Usage**:
```bash
# Apply all canonical migrations
python -m ml.deployment.migrations --apply --compose-file ml/deployment/docker-compose.yml

# List migration files
python -m ml.deployment.migrations
```

### CI Migration Smoke Test (`ci_migration_smoke.py`)

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/ci_migration_smoke.py` (99 lines)

**Purpose**: Automated validation of migration application and view creation

**Workflow** (lines 81-94):
```python
def main() -> None:
    # Start postgres only
    _compose("up", "-d", "postgres", timeout=30)
    wait_for_postgres()

    # Apply migrations
    from ml.deployment.migrations import apply_migrations_via_compose
    apply_migrations_via_compose(compose_file=COMPOSE_FILE)

    # Validate core views exist
    check_views()

    print("Migration smoke OK")
```

**PostgreSQL Readiness Wait** (lines 21-46):
```python
def wait_for_postgres(timeout: int = 30) -> None:
    start = time.time()
    while time.time() - start < timeout:
        result = run_command(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "exec", "-T", "postgres", "pg_isready", "-U", "postgres"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            log=logger,
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError("PostgreSQL did not become ready in time")
```

**View Validation** (lines 49-78):
```python
def check_views() -> None:
    sql = """
    SELECT 1 FROM pg_views WHERE viewname IN (
        'pipeline_health',
        'data_collection_stats',
        'model_performance_summary',
        'strategy_signal_summary'
    ) AND schemaname = 'ml';
    """
    cmd = [
        "docker", "compose", "-f", str(COMPOSE_FILE),
        "exec", "-T", "postgres", "psql", "-U", "postgres", "nautilus",
        "-v", "ON_ERROR_STOP=1", "-c", sql,
    ]
    try:
        run_command(cmd, timeout=30, log=logger)
    except SubprocessExecutionError as exc:
        raise RuntimeError(f"View validation failed: {exc}") from exc
```

---

## Security and Utilities

### Model Security (`security.py`)

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/security.py` (28 lines)

**Purpose**: Enforce ONNX-only model artifact usage in production

```python
def assert_allowed_model_path(path: str) -> None:
    """
    Validate model artifact path for deployment.

    Only ONNX models are allowed. Raises ValueError with actionable message
    if the path is not permitted.
    """
    suffix = Path(path).suffix.lower()
    if suffix != ".onnx":
        raise ValueError(
            "Only ONNX model artifacts are permitted in production (.*.onnx). "
            f"Refused: {path}",
        )
```

**Usage in Actor Entrypoint** (lines 118-127 of entrypoint_actor.py):
```python
try:
    assert_allowed_model_path(model_path)
except ValueError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)
if not Path(model_path).exists():
    print(f"ERROR: Model not found at {model_path}")
    sys.exit(1)
```

### Scheduling Utilities (`scheduling_utils.py`)

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/scheduling_utils.py` (105 lines)

**Purpose**: Deterministic UTC schedule computation for daily pipeline runs

**Key Components**:

```python
@dataclass(slots=True, frozen=True)
class DailyTime:
    """A daily time specification (UTC) with hour and minute components."""
    hour: int
    minute: int

def parse_bool_env(value: str | None) -> bool:
    """
    Parse boolean environment value.

    Accepts: "1", "true", "yes", "on" (case-insensitive).
    """
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}

def parse_daily_spec(spec: str) -> DailyTime:
    """
    Parse daily time specification.

    Supported formats (UTC):
    - "HH:MM" (e.g., "17:00")
    - Crontab-like: "M H * * *" (e.g., "0 17 * * *")
    """
    s = spec.strip()

    # Try HH:MM (lines 54-63)
    if ":" in s and " " not in s:
        parts = s.split(":", 1)
        hour_str, minute_str = parts[0].strip(), parts[1].strip()
        if not (hour_str.isdigit() and minute_str.isdigit()):
            raise ValueError(f"Invalid daily time spec: {spec!r}")
        hour, minute = int(hour_str), int(minute_str)
        _validate_hm(hour, minute, spec)
        return DailyTime(hour=hour, minute=minute)

    # Try crontab-like (lines 66-70)
    fields = s.split()
    if len(fields) == 5 and fields[0].isdigit() and fields[1].isdigit():
        minute, hour = int(fields[0]), int(fields[1])
        _validate_hm(hour, minute, spec)
        return DailyTime(hour=hour, minute=minute)

    raise ValueError(f"Invalid daily schedule format: {spec!r}")

def compute_next_utc_run(now: datetime, daily: DailyTime) -> datetime:
    """
    Compute next run datetime in UTC from daily time.

    Assumes `now` is timezone-aware UTC or naive treated as UTC.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    else:
        now = now.astimezone(UTC)

    run_today = now.replace(hour=daily.hour, minute=daily.minute, second=0, microsecond=0)
    if now < run_today:
        return run_today
    return (run_today + timedelta(days=1)).replace(tzinfo=UTC)
```

**Usage in Pipeline Entrypoint** (lines 296-311):
```python
schedule_raw = os.environ.get("PIPELINE_SCHEDULE")
if schedule_raw:
    try:
        daily: DailyTime = parse_daily_spec(schedule_raw)
        now = datetime.now(UTC)
        next_run = compute_next_utc_run(now, daily)
        sleep_seconds = max(0.0, (next_run - now).total_seconds())
    except ValueError:
        # Bad spec; fall back to interval mode
        sleep_seconds = float(interval_seconds)
else:
    sleep_seconds = float(interval_seconds)
```

---

## Observability Configuration

### Prometheus Scrape Configuration

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/prometheus.yml` (26 lines)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - /etc/prometheus/alerts.yml

scrape_configs:
  # ML Pipeline (exposes /metrics on port 8080)
  - job_name: 'ml_pipeline'
    static_configs:
      - targets: ['ml_pipeline:8080']
    metrics_path: /metrics

  # ML Signal Actor metrics
  - job_name: 'ml_signal_actor'
    static_configs:
      - targets: ['ml_signal_actor:8000']
    metrics_path: /metrics

  # ML Strategy metrics
  - job_name: 'ml_strategy'
    static_configs:
      - targets: ['ml_strategy:8001']
    metrics_path: /metrics
```

### Alert Rules Configuration

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/alerts.yml` (165 lines)

**Alert Groups**:

1. **Model Performance Alerts** (lines 5-16):
```yaml
- alert: MLModelInferenceLatencyHighP99
  expr: |
    histogram_quantile(0.99,
      sum(rate(nautilus_ml_model_inference_duration_seconds_bucket[5m])) by (le)
    ) > 0.200
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High model inference latency (P99 > 200ms)"
```

2. **Pipeline Health Alerts** (lines 18-26):
```yaml
- alert: MLPipelineHealthLow
  expr: nautilus_ml_pipeline_health < 0.8
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Pipeline health below threshold"
```

3. **Ingestion Monitoring** (lines 28-56):
```yaml
- alert: MLIngestErrorsHigh
  expr: increase(nautilus_ml_data_collection_errors_total[5m]) > 0
  for: 10m

- alert: MLIngestWatermarkLagHigh
  expr: max_over_time(nautilus_ml_watermark_lag_seconds[10m]) > 300
  for: 10m

- alert: MLIngestRateDrop
  expr: sum by (dataset_type) (rate(nautilus_ml_data_events_total{stage="INGESTED"}[10m])) < 0.01
  for: 15m
```

4. **Observability Worker Alerts** (lines 58-101):
```yaml
- alert: MLObsAsyncBackpressureDrops
  expr: increase(nautilus_ml_backpressure_drops_total[5m]) > 0
  for: 10m

- alert: MLObsAsyncBackpressureSustained
  expr: rate(nautilus_ml_backpressure_drops_total[5m]) > 0.5
  for: 10m
  labels:
    severity: critical

- alert: MLObsAsyncQueueDepthHigh
  expr: nautilus_ml_observability_queue_depth{component="obs_async_worker"} > 3072
  for: 10m

- alert: MLObsAsyncFlushLatencyHighP99
  expr: |
    histogram_quantile(0.99,
      sum(rate(nautilus_ml_observability_async_flush_duration_seconds_bucket[5m])) by (le)
    ) > 0.5
  for: 10m
```

5. **Aggregator Monitoring** (lines 103-131):
```yaml
- alert: MLAggregatorDuplicatesHigh
  expr: rate(nautilus_ml_aggregator_duplicates_total[5m]) > 1
  for: 10m

- alert: MLAggregatorBufferHigh
  expr: max_over_time(nautilus_ml_aggregator_buffer_size[10m]) > 5000
  for: 10m

- alert: MLAggregatorWatermarkLagHigh
  expr: max_over_time(nautilus_ml_aggregator_watermark_lag_seconds[10m]) > 300
  for: 10m
```

6. **Streaming Training Backlog** (lines 133-165):
```yaml
- alert: MLStreamingBacklogWarning
  expr: max(ml_tft_streaming_training_backlog) > 4
  for: 2m
  labels:
    severity: warning

- alert: MLStreamingBacklogCritical
  expr: max(ml_tft_streaming_training_backlog) > 8
  for: 2m
  labels:
    severity: critical

- alert: MLStreamingWorkersMissing
  expr: |
    (sum by (dataset_id) (ml_tft_streaming_training_backlog) > 0)
      and on (dataset_id)
    (sum by (dataset_id) (ml_tft_streaming_workers_active) <= 0)
  for: 2m
  labels:
    severity: critical
```

---

## Container Specifications

### Pipeline Dockerfile

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/Dockerfile.pipeline` (50 lines)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (lines 8-13)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Nautilus Trader from PyPI (line 16)
RUN pip install --no-cache-dir nautilus-trader

# Copy ML sources (line 17)
COPY ml ./ml

# Install runtime dependencies (lines 20-34)
RUN pip install --no-cache-dir \
    flask \
    requests \
    structlog \
    sqlalchemy \
    psycopg2-binary \
    pandas \
    numpy \
    polars \
    pyarrow \
    msgspec \
    prometheus-client \
    databento \
    scikit-learn \
    rich

# Create necessary directories (line 37)
RUN mkdir -p /app/data/catalog /app/models /app/logs /app/configs

# Copy entrypoint script (line 42)
COPY ml/deployment/entrypoint_pipeline.py /app/entrypoint.py

# Health check (lines 45-46)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import ml.data.scheduler; import psycopg2; import sys; sys.exit(0)" || exit 1

# Default command (line 49)
CMD ["python", "/app/entrypoint.py"]
```

### Actor Dockerfile

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/Dockerfile.actor` (47 lines)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (lines 6-11)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (line 14)
COPY ml ./ml

# Install runtime dependencies including ONNX runtime (lines 17-30)
RUN pip install --no-cache-dir \
    nautilus-trader \
    numpy \
    pandas \
    msgspec \
    structlog \
    sqlalchemy \
    psycopg2-binary \
    prometheus-client \
    requests \
    flask \
    onnxruntime \
    onnx

# Create models directory (line 33)
RUN mkdir -p /app/models

# Copy entrypoint scripts (lines 36-39)
COPY ml/deployment/entrypoint_actor.py /app/entrypoint.py
COPY ml/deployment/entrypoint_actor.py /app/entrypoint_actor.py
COPY ml/deployment/entrypoint_mock.py /app/entrypoint_mock.py
COPY ml/deployment/mock_databento.py /app/mock_databento.py

# Health check (lines 42-43)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run the actor (line 46)
CMD ["python", "/app/entrypoint.py"]
```

---

## Environment Configuration

### Environment Variable Template (`.env.example`)

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/.env.example` (29 lines)

```bash
# External integrations
DATABENTO_API_KEY=replace-me
DATABENTO_DATASET=EQUS.MINI

# Schedule and pipeline behaviour
PIPELINE_MODE=daily
PIPELINE_SCHEDULE=0 17 * * *
UNIVERSE_SYMBOLS=SPY.XNAS
LOG_LEVEL=INFO

# Host port overrides (each maps host_port:container_port)
POSTGRES_HOST_PORT=5433
REDIS_HOST_PORT=6380
ML_ACTOR_HOST_PORT=8000
ML_STRATEGY_HOST_PORT=8001
ML_PIPELINE_HOST_PORT=8081
ML_DASHBOARD_HOST_PORT=8010

# Streaming persistence worker configuration
ML_BUS_REDIS_STREAM=ml-events
ML_STREAM_PERSIST_ENABLE=1
ML_STREAM_PERSIST_BATCH_SIZE=256
ML_STREAM_PERSIST_BLOCK_MS=1000
ML_STREAM_PERSIST_POLL_INTERVAL_SECONDS=0.5

# Optional: direct host connection for tooling outside stack
# ML_DB_CONNECTION=postgresql://postgres:postgres@localhost:5433/nautilus
```

### Common Environment Variables

| Variable | Default | Purpose | Defined In |
|----------|---------|---------|------------|
| `DATABENTO_API_KEY` | *(required)* | Live market data access | .env.example, entrypoints |
| `DATABENTO_DATASET` | `EQUS.MINI` | Dataset identifier | .env.example, compose |
| `PIPELINE_MODE` | `daily` | Schedule mode (daily/backfill/realtime) | .env.example, entrypoint_pipeline.py:219 |
| `PIPELINE_SCHEDULE` | `0 17 * * *` | Cron or HH:MM UTC schedule | .env.example, entrypoint_pipeline.py:296 |
| `UNIVERSE_SYMBOLS` | `SPY.XNAS` | Comma-separated instrument list | .env.example, entrypoint_pipeline.py:137 |
| `LOG_LEVEL` | `INFO` | Logging verbosity | .env.example, multiple services |
| `POSTGRES_HOST_PORT` | `5433` | Host → container port mapping | .env.example, docker-compose.yml:18 |
| `REDIS_HOST_PORT` | `6380` | Host → container port mapping | .env.example, docker-compose.yml:54 |
| `ML_PIPELINE_HOST_PORT` | `8081` | Pipeline health endpoint port | .env.example, docker-compose.yml:248 |
| `ML_ACTOR_HOST_PORT` | `8000` | Actor metrics endpoint port | .env.example, docker-compose.yml:131 |
| `ML_STRATEGY_HOST_PORT` | `8001` | Strategy metrics endpoint port | .env.example, docker-compose.yml:178 |
| `EXECUTE_TRADES` | `false` | **Dry run control** (SAFETY) | entrypoint_strategy.py:80 |
| `USE_ORCHESTRATOR` | `false` | Enable unified ingestion | entrypoint_pipeline.py:231 |
| `DUAL_WRITE` | `false` | SQL + Parquet dual write | entrypoint_pipeline.py:232 |
| `ML_LIVE_RECORD_ENABLE` | `1` | Enable live data recording | entrypoint_actor.py:313 |
| `ML_LIVE_RECORD_DATASETS` | `bars` | Dataset types to record | entrypoint_actor.py:319 |

---

## Makefile Targets

**File**: `/home/nate/projects/nautilus_trader/ml/deployment/Makefile` (144 lines)

### Core Operations

```makefile
build:          # Build all Docker images
up:             # Start all services
down:           # Stop all services
health:         # Check service health
logs:           # View logs (use SERVICE=name for specific service)
ps:             # Show container status
```

### Database Operations

```makefile
migrate:        # Apply SQL migrations via compose exec
                # Invokes: python -m ml.deployment.migrations --apply
ci-smoke:       # CI migration smoke test (start DB, apply, verify views)
                # Invokes: python -m ml.deployment.ci_migration_smoke
```

### Deployment Modes

```makefile
pipeline:       # Start only the ML pipeline service
                # docker compose up -d postgres && docker compose up ml_pipeline

backfill:       # Run pipeline in backfill mode
                # docker compose run --rm -e PIPELINE_MODE=backfill ml_pipeline
                # Requires: START=YYYY-MM-DD END=YYYY-MM-DD

realtime:       # Run pipeline in realtime mode
                # docker compose run --rm -e PIPELINE_MODE=realtime ml_pipeline

dev:            # Development mode with live code mounting
                # docker compose -f docker-compose.yml -f ../docker-compose.dev.yml up
```

### Operational Commands

```makefile
restart:        # Restart a service (SERVICE=name)
                # docker compose restart $(SERVICE)

scale:          # Scale pipeline workers (WORKERS=n)
                # docker compose up -d --scale ml_pipeline=$(WORKERS)

deploy:         # Production deployment
                # Validates .env presence, starts with --env-file

backup:         # Backup PostgreSQL database
                # docker compose exec postgres pg_dump -U postgres nautilus > backup_<timestamp>.sql

restore:        # Restore database (FILE=backup.sql)
                # docker compose exec -T postgres psql -U postgres nautilus < $(FILE)

nuke:           # Remove containers and volumes (DANGEROUS)
                # docker compose down -v

clean:          # Clean up volumes and prune system
                # docker compose down -v && docker system prune -f
```

### Dashboard Integration

```makefile
grafana-import: # Import default dashboard via HTTP API
                # Requires: GRAFANA_API_TOKEN
                # curl -H "Authorization: Bearer $TOKEN" -X POST http://localhost:3000/api/dashboards/db
```

---

## Deployment Patterns and Best Practices

### Universal ML Architecture Pattern Compliance

All deployment services enforce the **5 Universal ML Patterns** from CLAUDE.md:

#### Pattern 1: Mandatory 4-Store + 4-Registry Integration

**Implementation**: All services extend `BaseMLInferenceActor` which auto-initializes stores/registries

```python
# entrypoint_actor.py lines 326-332
mgr = MLIntegrationManager(
    db_connection=db_connection,
    auto_start_postgres=False,
    auto_migrate=False,
    ensure_healthy=False,
    strict_protocol_validation=False,
)
recorder = LiveDataRecorder(
    data_store=mgr.data_store,      # Auto-initialized
    data_registry=mgr.data_registry, # Auto-initialized
    buffer_size=1000,
    flush_interval_ms=1000,
)
```

**Progressive Fallback** (entrypoint_actor.py lines 360-469):
- PRIMARY: PostgreSQL via `MLIntegrationManager`
- FALLBACK: ParquetDataCatalog + JSON DataRegistry
- FINAL: DummyStore with warnings

#### Pattern 2: Protocol-First Interface Design

**Store Typing** (all entrypoints use protocol-typed stores):
```python
from ml.stores.protocols import DataStoreProtocol, DataRegistryProtocol

data_store: DataStoreProtocol = mgr.data_store
data_registry: DataRegistryProtocol = mgr.data_registry
```

**Benefits**:
- Structural typing without implementation coupling
- Duck typing support for testing (DummyStore conforms)
- Type safety without circular dependencies

#### Pattern 3: Hot/Cold Path Separation

**Hot Path Services** (< 5ms P99 latency requirement):
- `ml_signal_actor`: Real-time inference, pre-allocated arrays, ONNX runtime
- `ml_strategy`: Signal processing, decision making

**Cold Path Services**:
- `ml_pipeline`: Data collection, feature computation
- `ml_dashboard`: Control plane, monitoring

**Hot Path Enforcement** (entrypoint_actor.py lines 143-152):
```python
feature_config = MLFeatureConfig(
    lookback_window=20,
    indicators={
        "sma": {"period": 10},
        "rsi": {"period": 14},
        "bbands": {"period": 20, "std": 2},
    },
    normalize_features=True,
    fill_missing_with=0.0,  # Pre-allocated defaults
)
```

#### Pattern 4: Progressive Fallback Chains

**Database Connectivity** (all entrypoints):
```
PostgreSQL → DummyStore → warnings logged
```

**Registry Loading** (entrypoint_actor.py lines 374-391):
```
PostgreSQL backend → JSON file backend → local registry path
```

**Configuration** (entrypoint_pipeline.py lines 159-163):
```python
feature_store_connection=os.environ.get(
    "DB_CONNECTION",
    os.environ.get("FEATURE_STORE_CONNECTION",
    os.environ.get("DATABASE_URL")),
)
```

#### Pattern 5: Centralized Metrics Bootstrap

**Implementation** (all entrypoints use facade):
```python
from ml.common.metrics_export import CONTENT_TYPE_LATEST, generate_latest

@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
```

**Never import `prometheus_client` directly** – always use `ml.common.metrics_export`

### Security Controls

#### ONNX-Only Model Enforcement

**Security Module** (`security.py` lines 12-24):
```python
def assert_allowed_model_path(path: str) -> None:
    suffix = Path(path).suffix.lower()
    if suffix != ".onnx":
        raise ValueError(
            "Only ONNX model artifacts are permitted in production (.*.onnx). "
            f"Refused: {path}",
        )
```

**Enforcement in Entrypoints** (entrypoint_actor.py lines 118-127):
```python
try:
    assert_allowed_model_path(model_path)
except ValueError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)
```

**No Pickle Support** (entrypoint_actor.py line 481):
```python
def _create_dummy_model(self, model_path: str) -> None:
    raise RuntimeError("Dummy pickle models are no longer supported.")
```

#### Mandatory Dry-Run Mode

**Default Configuration** (entrypoint_strategy.py line 80):
```python
execute_trades = os.getenv("EXECUTE_TRADES", "false").lower() == "true"
```

**Explicit Warning Display** (entrypoint_strategy.py lines 112-116):
```python
if not execute_trades:
    print("\n⚠️  DRY RUN MODE ACTIVE ⚠️")
    print("Strategy will process signals and make decisions")
    print("but will NOT submit actual orders to the exchange")
```

**Docker Compose Enforcement** (docker-compose.yml line 153):
```yaml
environment:
  EXECUTE_TRADES: "false"  # DRY RUN - NO REAL TRADES!
```

---

## Testing and CI/CD Integration

### Test Infrastructure

**Unit Tests** (`ml/tests/unit/deployment/`):
- `test_check_health.py` – Health checker validation
- `test_migrations.py` – Migration utility testing
- `test_metrics_http.py` – Flask endpoint testing
- `test_model_security.py` – ONNX enforcement validation
- `test_scheduling_utils.py` – Schedule parsing tests

**Integration Tests** (`ml/tests/integration/deployment/`):
- `test_deployment_integration.py` – End-to-end deployment validation

### CI Smoke Test Workflow

```python
# ci_migration_smoke.py (lines 81-94)
def main() -> None:
    # 1. Start postgres container
    _compose("up", "-d", "postgres", timeout=30)

    # 2. Wait for readiness
    wait_for_postgres()

    # 3. Apply migrations
    from ml.deployment.migrations import apply_migrations_via_compose
    apply_migrations_via_compose(compose_file=COMPOSE_FILE)

    # 4. Validate core views exist
    check_views()

    print("Migration smoke OK")
```

**Usage in CI**:
```bash
make ci-smoke
# or
python -m ml.deployment.ci_migration_smoke
```

---

## Operational Procedures

### Production Deployment Checklist

1. **Environment Setup**:
   ```bash
   cd ml/deployment
   cp .env.example .env
   # Edit .env: set DATABENTO_API_KEY, adjust ports if needed
   ```

2. **Validate Configuration**:
   ```bash
   # Check compose syntax
   docker compose config

   # Verify .env variables
   grep -v '^#' .env | grep -v '^$'
   ```

3. **Database Initialization**:
   ```bash
   # Start PostgreSQL only
   docker compose up -d postgres

   # Wait for health
   docker compose ps postgres

   # Apply migrations
   make migrate
   # or
   python -m ml.deployment.migrations --apply --compose-file docker-compose.yml
   ```

4. **Service Startup**:
   ```bash
   # Start full stack
   make up
   # or
   docker compose up -d

   # Check status
   make ps
   docker compose ps
   ```

5. **Health Validation**:
   ```bash
   # Automated health check
   python -m ml.deployment.check_health

   # Manual endpoint checks
   curl http://localhost:8081/health  # Pipeline
   curl http://localhost:8000/health  # Actor
   curl http://localhost:8001/health  # Strategy
   curl http://localhost:9090/-/healthy  # Prometheus
   curl http://localhost:3000/api/health  # Grafana
   ```

6. **Monitor Logs**:
   ```bash
   # All services
   docker compose logs -f

   # Specific service
   make logs SERVICE=ml_pipeline
   docker compose logs -f ml_signal_actor
   ```

### Troubleshooting Guide

#### Port Conflicts

**Symptom**: `Bind for 0.0.0.0:5433 failed: port is already allocated`

**Solution**:
```bash
# Edit .env to use different ports
POSTGRES_HOST_PORT=5543
REDIS_HOST_PORT=6390
ML_PIPELINE_HOST_PORT=8091

# Restart stack
make down
make up
```

#### Database Connection Failures

**Symptom**: Pipeline exits immediately with `could not connect to server`

**Solution**:
```bash
# Check PostgreSQL health
docker compose ps postgres
docker compose logs postgres

# Verify migrations applied
docker compose exec postgres psql -U postgres -d nautilus -c "\dt"

# Check connection string in .env
echo $DB_CONNECTION
```

#### Missing Models

**Symptom**: Actor exits with `ERROR: Model not found at /app/models/model.onnx`

**Solution**:
```bash
# Verify model mount
docker compose config | grep -A5 volumes

# Check model exists on host
ls -lh ml/models/

# Restart actor
make restart SERVICE=ml_signal_actor
```

#### Grafana Dashboard Not Loading

**Symptom**: Grafana shows no datasource or dashboards

**Solution**:
```bash
# Check Prometheus health
curl http://localhost:9090/-/healthy

# Verify datasource provisioning
docker compose exec grafana ls /etc/grafana/provisioning/datasources/

# Import dashboard manually
make grafana-import GRAFANA_API_TOKEN=<token>
```

#### Stale Volumes

**Symptom**: Disk space issues, old anonymous volumes

**Solution**:
```bash
# List anonymous volumes
docker volume ls --filter label=com.docker.volume.anonymous=

# Remove unused volumes
docker volume prune

# Nuclear option (removes ALL volumes)
make nuke  # DANGEROUS - loses all data
```

---

## Integration with Other Modules

### Cross-Module References

**Data Pipeline Integration**:
- Uses `ml.data.scheduler.DataScheduler` (entrypoint_pipeline.py:233)
- Integrates `ml.data.ingest.orchestrator.IngestionOrchestrator`
- Wraps `ml.data.providers` for SQL coverage/writers

**Store Integration**:
- Initializes via `ml.core.integration.MLIntegrationManager` (entrypoint_pipeline.py:398)
- Uses `ml.stores.feature_store.FeatureStore` (entrypoint_pipeline.py:181)
- Uses `ml.stores.model_store.ModelStore` (entrypoint_pipeline.py:188)
- Uses `ml.stores.writers.LiveDataRecorder` (entrypoint_actor.py:333)

**Actor Integration**:
- Extends `ml.actors.multi_signal.MultiInstrumentSignalActor` (entrypoint_actor.py:25)
- Uses `ml.actors.recorder.RecorderActor` for live data (entrypoint_actor.py:339)
- Leverages `ml.config.base.MLFeatureConfig` (entrypoint_actor.py:143)

**Strategy Integration**:
- Uses `ml.strategies.ml_strategy.MLTradingStrategy` (entrypoint_strategy.py:28)
- Integrates `ml.config.base.MLStrategyConfig` (entrypoint_strategy.py:119)

**Registry Integration**:
- Uses `ml.registry.data_registry.DataRegistry` (entrypoint_actor.py:388)
- Integrates `ml.registry.persistence.PersistenceConfig` (entrypoint_actor.py:377)

**Observability Integration**:
- Uses `ml.observability.bootstrap.auto_start_if_configured` (all entrypoints)
- Leverages `ml.common.metrics_export` for Prometheus endpoints
- Integrates `ml.common.logging_config` for structured logging

---

## Known Limitations and Future Work

### Current Limitations

⚠️ **Kubernetes Support**: Docker Compose only, no Helm charts or K8s manifests
⚠️ **Execution Clients**: Dry-run mode only, broker integrations require manual configuration
⚠️ **Auto-Scaling**: No horizontal pod autoscaling based on load metrics
⚠️ **Multi-Region**: Single region deployment, no geographic redundancy
⚠️ **Secrets Management**: Environment variables only, no external secret stores (Vault, AWS Secrets Manager)

### Future Enhancements

**Planned Features**:
1. Kubernetes deployment manifests with StatefulSets for PostgreSQL
2. Helm charts for multi-environment deployments
3. Horizontal pod autoscaling based on queue depth and latency
4. Multi-region active-active architecture
5. External secret store integration (HashiCorp Vault, AWS Secrets Manager)
6. Enhanced live trading execution client integrations (Interactive Brokers, Alpaca)
7. Advanced canary deployment support for model A/B testing
8. Distributed tracing integration (OpenTelemetry, Jaeger)

---

## Quick Reference

### Port Map

| Service | Container | Host Default | Override Variable |
|---------|-----------|--------------|-------------------|
| Postgres | 5432 | 5433 | `POSTGRES_HOST_PORT` |
| Postgres Playground | 5432 | 5435 | `PLAYGROUND_POSTGRES_HOST_PORT` |
| Redis | 6379 | 6380 | `REDIS_HOST_PORT` |
| ML Signal Actor | 8000 | 8000 | `ML_ACTOR_HOST_PORT` |
| ML Strategy | 8001 | 8001 | `ML_STRATEGY_HOST_PORT` |
| ML Pipeline | 8080 | 8081 | `ML_PIPELINE_HOST_PORT` |
| ML Dashboard | 8010 | 8010 | `ML_DASHBOARD_HOST_PORT` |
| Prometheus | 9090 | 9090 | *(edit compose file)* |
| Grafana | 3000 | 3000 | *(edit compose file)* |

### Connection Strings

**Inside Containers** (service-to-service):
```
postgresql://postgres:postgres@postgres:5432/nautilus
redis://redis:6379/0
```

**From Host** (local development):
```
postgresql://postgres:postgres@localhost:5433/nautilus
redis://localhost:6380/0
```

### Essential Commands

```bash
# Start production stack
make ml-up

# Check service status
make ml-ps

# View logs
make ml-logs                    # Pipeline only
make ml-logs SERVICE=ml_signal_actor

# Health check
python -m ml.deployment.check_health

# Apply migrations
make ml-migrate

# Stop stack
make ml-down

# Nuclear reset (removes volumes)
make ml-down-v
```

### File Locations

| Purpose | Path |
|---------|------|
| Main compose file | `ml/deployment/docker-compose.yml` |
| Test stack compose | `ml/deployment/docker-compose.test.yml` |
| Environment template | `ml/deployment/.env.example` |
| Pipeline entrypoint | `ml/deployment/entrypoint_pipeline.py` |
| Actor entrypoint | `ml/deployment/entrypoint_actor.py` |
| Strategy entrypoint | `ml/deployment/entrypoint_strategy.py` |
| Health checker | `ml/deployment/check_health.py` |
| Migration utility | `ml/deployment/migrations.py` |
| Prometheus config | `ml/deployment/prometheus.yml` |
| Alert rules | `ml/deployment/alerts.yml` |
| Grafana dashboard | `ml/deployment/grafana/ml_pipeline_health.json` |
| Deployment README | `ml/deployment/README.md` |

---

## Summary

The ML deployment architecture provides a **production-ready, containerized environment** for running Nautilus Trader ML strategies with:

✅ **Comprehensive service orchestration** via Docker Compose v2 with health-based dependencies
✅ **Security-first design** with ONNX-only models and mandatory dry-run mode
✅ **Progressive fallback chains** ensuring system resilience (PostgreSQL → DummyStore)
✅ **Complete observability** via Prometheus/Grafana with custom dashboards and alerts
✅ **Multiple deployment modes** (production containers, local dev, backtest)
✅ **Automated schema management** via Docker init volumes and CLI migration runner
✅ **Universal ML pattern compliance** enforcing 4-store + 4-registry architecture

The system is **98% complete** and ready for production use with comprehensive monitoring, health checking, and operational tooling. All deployment services adhere to CLAUDE.md architectural patterns while maintaining operational excellence through graceful degradation, comprehensive logging, and safety-first controls.

---

## Cross-Module Documentation

- **Data Pipeline**: See `context_data.md` for data ingestion and collection
- **Stores**: See `context_stores.md` for persistence layer details
- **Actors**: See `context_actors.md` for ML inference actors
- **Strategies**: See `context_strategies.md` for trading strategy framework
- **Registry**: See `context_registry.md` for lifecycle management
- **Orchestration**: See `context_orchestration.md` for pipeline orchestration
- **Monitoring**: See `ops/dashboard_runbook.md` for operational procedures
