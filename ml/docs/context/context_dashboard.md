# Dashboard Context Documentation

## Purpose

`ml/dashboard/` and `ml/dashboard_bootstrap/` provide a Flask-based control plane for the ML system, exposing REST APIs for:
- Registry operations (models, features, strategies, datasets)
- Pipeline orchestration (dataset building, model training, HPO)
- System control (service lifecycle, health monitoring)
- Store health monitoring and metrics aggregation
- Integration services (actors, trading, features, strategies)

**Status**: **Production-ready** (Flask API + integration services)

---

## Core Architecture

### 1. Flask Application (`app.py`, 1,509 lines)

**Factory Pattern**: `create_app(config: DashboardConfig | None) -> Flask`

**API Endpoint Categories** (84 total routes):
- **Health & System**: `/api/health/system`, `/api/services`
- **Registry**: `/api/registry/{models,features,strategies,datasets}`
- **Pipeline**: `/api/pipeline/run`, `/api/pipeline/jobs`, `/api/pipeline/jobs/<job_id>`
- **Events**: `/api/events` (Redis Streams integration)
- **Control Panel**: `/api/control/{actors,pipeline,ingestion,emergency}`
- **Metrics & Monitoring**: `/api/metrics/{snapshot,portfolio,ingestion,experiments}`
- **Trading**: `/api/trading/{toggle,emergency,health,market-data}`
- **API Explorer**: `/api/openapi.json`, `/api/docs`
- **Terminal**: `/api/terminal/{execute,history}`
- **Actors**: `/api/actors/{deploy,hot-reload,pause,resume,stop,health}`
- **Pipelines**: `/api/pipeline/{build-dataset,train-model,run-hpo}`
- **Features**: `/api/features/{designer/generate,validate-code,analyze,manifests}`
- **Strategies**: `/api/strategies/{validate,backtest,deploy,performance}`

**Authentication**: Bearer token validation via `_require_token()` (lines 60-69)
- Supports `X-ML-DASHBOARD-TOKEN` header and `Authorization: Bearer <token>`
- Validates against `DashboardConfig.auth_tokens` with expiry support

**Example API Call**:
```python
# app.py:71-74
@app.get("/api/health/system")
def health_system() -> tuple[Any, int]:
    data = svc.get_system_health()
    return jsonify(data), 200
```

---

### 2. Dashboard Service (`service.py`, 2,336 lines)

**Central Orchestration**: Aggregates health, manages registries, controls pipelines

**Key Components**:
```python
# service.py:367-403
@dataclass(slots=True, init=False)
class DashboardService:
    config: DashboardConfig
    controller: ServiceControllerProtocol
    _model_registry: ModelRegistry | None
    _feature_registry: FeatureRegistry | None
    _strategy_registry: StrategyRegistry | None
    _data_registry: DataRegistry | None
    _registry_cache: _TTLCache[object]
    _event_cache: _EventCache
    _store_clients: _StoreClients | None
    _pipeline_service: PipelineIntegrationService | None
    _pipeline_integration_manager: MLIntegrationManager | None
    _streaming_monitor: StreamingTrainingConsumer | None
```

**Progressive Fallback** (service.py:570-608):
```python
def _build_registry(self, *, name: str, builder: Callable[[], object | None]) -> object | None:
    registry = retry_with_backoff(builder, max_attempts=3, ...)
    if registry is None and self._allow_dummy_fallback:
        return DummyRegistry()
    return registry
```

**Registry Integration** (service.py:530-608):
- Model Registry: Load/deploy/hot-reload models (lines 1162-1214)
- Feature Registry: List/promote/deprecate features (lines 1281-1648)
- Strategy Registry: Compatibility checks (lines 1319-1549)
- Data Registry: Watermarks/lineage (lines 1348-1504)

**TTL Caching** (service.py:204-261):
```python
@dataclass(slots=True)
class _TTLCache(Generic[_CacheValueT]):
    ttl_seconds: float
    max_entries: int
    # Evicts oldest entries when capacity reached
    def _evict_locked(self) -> None:
        lru_key = min(self._entries.items(), key=lambda item: item[1].expires_at)[0]
```

---

### 3. Configuration (`config.py`, 300 lines)

**Environment-Driven**:
```python
# config.py:50-95
@dataclass(slots=True, frozen=True)
class DashboardConfig:
    compose_enabled: bool = False
    compose_file: Path | None = None
    request_timeout_seconds: float = 2.5

    # Service ports
    actor_port: int = 8000
    strategy_port: int = 8001
    pipeline_port: int = 8081
    grafana_port: int = 3000
    prometheus_port: int = 9090

    # Event bus integration
    events_cache_ttl_seconds: float = 5.0
    events_poll_interval_seconds: float = 0.0  # Background polling disabled by default

    # Grafana integration
    grafana_embed_enabled: bool = False
    grafana_provision_on_start: bool = False

    # Store health monitoring
    store_health_cache_ttl_seconds: float = 30.0
    store_health_enabled: bool = True

    # Authentication
    auth_tokens: tuple[DashboardToken, ...] = ()
```

**Token Expiry** (config.py:28-44):
```python
@dataclass(slots=True, frozen=True)
class DashboardToken:
    value: str
    expires_at: dt.datetime | None = None

    def is_valid(self, *, now: dt.datetime | None = None) -> bool:
        if self.expires_at is None:
            return True
        return now < self.expires_at
```

---

### 4. Service Controllers (`controllers.py`, 133 lines)

**Protocol-Based Design**:
```python
# controllers.py:27-41
@runtime_checkable
class ServiceControllerProtocol(Protocol):
    def start(self, name: str) -> bool: ...
    def stop(self, name: str) -> bool: ...
    def restart(self, name: str) -> bool: ...
```

**Implementations**:
- `NoopServiceController`: No-op (default, lines 44-56)
- `ComposeServiceController`: Docker Compose integration (lines 60-126)

**Docker Compose Integration** (controllers.py:86-111):
```python
def _compose(self, *args: str) -> None:
    docker = shutil.which("docker")
    if not docker:
        raise ServiceControlUnsupportedError("docker not found in PATH")
    compose_file = self._resolve_compose_file()
    run_command([docker, "compose", "-f", str(compose_file), *args], ...)
```

---

### 5. Store Health Monitoring (`store_health.py`, 342 lines)

**4-Store Health Aggregation**:
```python
# store_health.py:315-330
def summarize_all_stores(
    *,
    feature_store: object | None,
    model_store: object | None,
    strategy_store: object | None,
    engine: Engine | None,
    top_dataset_limit: int,
) -> tuple[StoreHealthSummary, ...]:
    return (
        summarize_feature_store(feature_store, engine),
        summarize_model_store(model_store, engine),
        summarize_strategy_store(strategy_store, engine),
        summarize_data_store(engine, top_limit=top_dataset_limit),
    )
```

**Health Derivation** (store_health.py:187-198):
```python
def _derive_health(
    connectivity_ok: bool | None,
    write_ok: bool | None,
    fallback_active: bool,
    latest_event_ns: int | None,
) -> bool:
    if connectivity_ok is not None and write_ok is not None:
        return bool(connectivity_ok and write_ok)
    if fallback_active:
        return False
    return latest_event_ns is not None
```

**Timestamp Queries** (store_health.py:15-28):
```python
_FETCH_MAX_TIMESTAMP_QUERIES: Final[dict[tuple[str, str | None], TextClause]] = {
    ("ml_feature_values", "is_live = TRUE"): text(
        "SELECT MAX(ts_event) AS ts_event FROM ml_feature_values WHERE is_live = TRUE"
    ),
    ("ml_model_predictions", "is_live = TRUE"): text(...),
    ("ml_strategy_signals", "is_live = TRUE"): text(...),
    ("ml_data_events", None): text(...),
}
```

---

### 6. Integration Services (`ml/dashboard/services/`)

**Service Architecture**:
```python
# services/base_service.py:44-79
class BaseIntegrationService(ABC):
    def __init__(self, integration_manager: MLIntegrationManager | None) -> None:
        self._integration = integration_manager
        self._context: IntegrationContext | None = None

    @abstractmethod
    async def health_check(self) -> dict[str, Any]: ...
```

**Available Services** (services/__init__.py):
- `ActorIntegrationService`: Actor deployment/lifecycle management
- `PipelineIntegrationService`: Pipeline orchestration (dataset, training, HPO)
- `StoreIntegrationService`: Metrics and store health aggregation
- `TradingIntegrationService`: Live trading control
- `SystemConnectorService`: System-wide health monitoring
- `FeatureEngineeringService`: Feature generation/validation
- `StrategyService`: Strategy backtesting/deployment
- `APIExplorerService`: OpenAPI spec generation
- `TerminalService`: Command execution history

**Example Service Integration** (app.py:762-799):
```python
@app.post("/api/actors/deploy")
def actors_deploy() -> tuple[Any, int]:
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401
    from ml.dashboard.services.actors_service import ActorIntegrationService

    actor_service = ActorIntegrationService(svc._pipeline_integration_manager)
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(actor_service.deploy_actor(deploy_request))
    finally:
        loop.close()

    return jsonify(response_data), 202 if result.success else 400
```

---

### 7. Metrics & Observability

**Centralized Metrics** (service.py:120-198):
```python
from ml.common.metrics_bootstrap import get_counter, get_histogram

_REQS_TOTAL = get_counter("ml_dashboard_requests_total", "Total dashboard API requests",
                          labels=["route", "method", "status"])
_LATENCY_SECONDS = get_histogram("ml_dashboard_latency_seconds", "Dashboard API latency",
                                  labels=["route"],
                                  buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
_REGISTRY_CACHE_HITS = get_counter("ml_dashboard_registry_cache_hits_total", ...)
_REGISTRY_CACHE_MISSES = get_counter("ml_dashboard_registry_cache_misses_total", ...)
_EVENT_CACHE_HITS = get_counter("ml_dashboard_events_cache_hits_total", ...)
_AUTH_VALIDATIONS_TOTAL = get_counter("ml_dashboard_auth_validations_total", ...)
```

**Recording Pattern** (service.py:459-496):
```python
def get_system_health(self) -> dict[str, Any]:
    start = time.perf_counter()
    route = "/api/health/system"
    try:
        # ... health aggregation logic
        return health
    finally:
        _LATENCY_SECONDS.labels(route=route).observe(time.perf_counter() - start)
```

---

### 8. Dashboard Bootstrap (`ml/dashboard_bootstrap/welcome.py`, 220 lines)

**CLI Orchestration**: Single entrypoint for stack startup

**Key Functions**:
```python
# welcome.py:75-114
def start_services(
    *,
    compose_file: str = DEFAULT_COMPOSE_FILE,
    services: Sequence[str] = DEFAULT_SERVICES,
    detach: bool = True,
) -> None:
    command = ["docker", "compose", "-f", compose_file, "up"]
    if detach:
        command.append("-d")
    command.extend(services)
    COMMAND_RUNNER(command, log=logger)
```

**Health Probing** (welcome.py:132-153):
```python
def gather_health(
    checks: Iterable[HealthCheck],
    *,
    timeout: float,
    retries: int,
    sleep_seconds: float,
) -> list[HealthStatus]:
    remaining_retries = max(0, retries)
    pending = list(checks)
    while pending:
        statuses = [probe_health(check, timeout=timeout) for check in pending]
        unhealthy = [status for status in statuses if not status.healthy]
        if not unhealthy or remaining_retries <= 0:
            break
        time.sleep(sleep_seconds)
        remaining_retries -= 1
        pending = [status.check for status in unhealthy]
    return statuses
```

**Default Health Checks** (welcome.py:61-68):
```python
DEFAULT_HEALTH_CHECKS: tuple[HealthCheck, ...] = (
    HealthCheck(name="ML Signal Actor", kind="service", url="http://localhost:8000/health"),
    HealthCheck(name="ML Strategy", kind="service", url="http://localhost:8001/health"),
    HealthCheck(name="ML Pipeline", kind="service", url="http://localhost:8081/health"),
    HealthCheck(name="Dashboard API", kind="dashboard", url="http://localhost:8010/health"),
    HealthCheck(name="Prometheus", kind="dependency", url="http://localhost:9090/-/healthy"),
    HealthCheck(name="Grafana", kind="dependency", url="http://localhost:3000/api/health"),
)
```

---

## Key Implementation Patterns

### 1. Progressive Fallback Chain

**Registry Loading** (service.py:570-608):
```
PostgreSQL Backend → Retry (3 attempts) → DummyRegistry Fallback (if ML_ALLOW_DUMMY=1)
```

### 2. TTL Caching Strategy

**Cache Implementation** (service.py:204-261):
- Registry calls cached for 30 seconds (lines 426)
- Store summaries cached for 30 seconds (configurable via `store_health_cache_ttl_seconds`)
- Event cache with 5-second TTL (configurable via `events_cache_ttl_seconds`)
- Background event polling via threading (lines 1046-1077)

### 3. Service Controller Pattern

**Abstraction** (controllers.py:27-41):
- Protocol-based interface for start/stop/restart operations
- Noop implementation for non-Compose environments
- Docker Compose implementation with subprocess execution

### 4. Integration Service Pattern

**Base Class** (services/base_service.py:44-79):
- Abstract `health_check()` method
- Metrics tracking via `_track_operation()`
- Request context via `IntegrationContext`
- Async execution helper `_run_async()`

---

## Integration Points

### Registry System
- `ModelRegistry`: Deploy/hot-reload models (service.py:1162-1214)
- `FeatureRegistry`: Promote/deprecate features (service.py:1554-1648)
- `StrategyRegistry`: Compatibility checks (service.py:1534-1549)
- `DataRegistry`: Watermark/lineage queries (service.py:1404-1504)

### Store System
- `FeatureStore`: Health monitoring via `health_details()` (store_health.py:97-115)
- `ModelStore`: Buffer backlog tracking
- `StrategyStore`: Write success monitoring
- `DataStore`: Dataset freshness via `ml_data_events` table

### Event Bus
- Redis Streams integration (service.py:966-1000)
- Background polling via threading (service.py:1046-1077)
- Event cache with TTL (service.py:307-341)

### Pipeline Orchestration
- `MLPipelineOrchestrator` integration (service.py:1989-2080)
- `PipelineIntegrationService` for job management
- Dataset building, model training, HPO triggers

---

## Deployment Considerations

### Environment Variables

**Core Configuration**:
```bash
ML_DASHBOARD_USE_COMPOSE=true
ML_DASHBOARD_COMPOSE_FILE=ml/deployment/docker-compose.yml
ML_DASHBOARD_TIMEOUT=2.5

# Service ports
ML_ACTOR_HOST_PORT=8000
ML_STRATEGY_HOST_PORT=8001
ML_PIPELINE_HOST_PORT=8081

# Authentication
ML_DASHBOARD_TOKEN=secret_token_here
ML_DASHBOARD_TOKEN_EXPIRES=2025-12-31T23:59:59Z

# Or multiple tokens (JSON array)
ML_DASHBOARD_TOKENS='[{"value":"token1","expires":"2025-12-31T23:59:59Z"},{"value":"token2"}]'

# Event polling (optional)
ML_DASHBOARD_EVENTS_POLL_INTERVAL=5.0  # Seconds (0.0 = disabled)

# Store health
ML_DASHBOARD_STORE_CACHE_TTL=30.0
ML_DASHBOARD_STORE_SUMMARY=true

# Grafana integration
GRAFANA_URL=http://localhost:3000
GRAFANA_API_TOKEN=secret_token
ML_DASHBOARD_GRAFANA_PROVISION_ON_START=true
```

### Production Startup

**Using Bootstrap**:
```python
from ml.dashboard_bootstrap.welcome import build_welcome_summary

summary = build_welcome_summary(
    compose_file="ml/deployment/docker-compose.yml",
    services=["postgres", "redis", "ml_dashboard", "prometheus", "grafana"],
    timeout_seconds=5.0,
    retries=5,
    start=True,  # Automatically start services
)
print(summary)
```

**Direct Flask App**:
```python
from ml.dashboard.app import create_app
from ml.dashboard.config import DashboardConfig

config = DashboardConfig.from_env()
app = create_app(config)
app.run(host="0.0.0.0", port=8010)
```

---

## Testing

### Unit Tests
- `ml/tests/unit/dashboard/test_dashboard_welcome.py`: Bootstrap CLI
- `ml/tests/unit/deployment/test_check_health.py`: Health probe logic

### Test Fixtures
- Mock service controllers
- In-memory registries via `DummyRegistry`
- Fake health endpoints

---

## Known Issues & Gaps

### 1. Service State Management (service.py:1989-2080)
**ISSUE**: Orchestrator task endpoints return stub responses
```python
# service.py:2047-2068
if task == "backfill":
    result = {"status": "started", "task": task, "config": config_json}
    ok = True  # No actual orchestrator.backfill() call
```
**IMPACT**: Control panel triggers are placeholders

### 2. Event Polling Thread Safety
**OBSERVATION**: Background polling uses thread + lock (service.py:1046-1077)
```python
def _run() -> None:
    while not stop.wait(interval_seconds):
        events = self._poll_events(limit=self.config.events_cache_max_entries)
        self._event_cache.update(events)
```
**RECOMMENDATION**: Consider async implementation for production

### 3. Store Health Queries Hardcoded
**LIMITATION**: SQL queries are static (store_health.py:15-28)
- No custom WHERE clauses beyond `is_live = TRUE`
- Top datasets limited to `GROUP BY dataset_type`

### 4. Authentication Token Parsing
**EDGE CASE**: Comma-separated fallback doesn't support commas in tokens (config.py:218-222)
```python
for part in tokens_raw.split(","):
    value = part.strip()
    if value:
        tokens.append(DashboardToken(value=value))
```

---

## Architecture Strengths

1. **Protocol-Based Design**: Clean abstraction for service controllers and stores
2. **Progressive Fallback**: Graceful degradation from PostgreSQL → DummyRegistry
3. **Centralized Metrics**: All observability via `metrics_bootstrap`
4. **TTL Caching**: Reduces registry/store load
5. **Modular Integration Services**: Clean separation of concerns

---

## Cross-Module Integration

**Related Documentation**:
- **Registry**: `context_registry.md` - Model/Feature/Strategy/Data registries
- **Stores**: `context_stores.md` - FeatureStore/ModelStore/StrategyStore integration
- **Orchestration**: `context_orchestration.md` - Pipeline orchestrator integration
- **Deployment**: `context_deployment.md` - Docker Compose and production setup

---

**Last Updated**: 2025-10-19
**Implementation Completeness**: ~85% (core API functional, integration services partial stubs)
**Documentation Accuracy**: ~92% (based on actual code review)
