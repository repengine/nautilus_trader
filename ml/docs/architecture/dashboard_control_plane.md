# Dashboard Control Plane (Cold Path)

## Overview

The Dashboard Control Plane provides a small, typed HTTP API to observe and control
the Nautilus Trader ML system. It reuses existing health, metrics, and event
infrastructure and strictly follows the Universal ML Architecture Patterns.

See `ml/docs/ops/dashboard_runbook.md` for operational procedures, token
rotation guidance, and troubleshooting flows.

- Cold-path only: no changes to hot paths; no allocations in tight loops.
- Metrics via `ml.common.metrics_bootstrap` (never import `prometheus_client` directly).
- Events/topics via `ml.config.events` and `ml.common.message_topics`.
- Structured logging via `ml.common.logging_config` (structlog + stdlib interop).

## Package

- Package: `ml.dashboard`
- Public API: `ml/dashboard/__init__.py`
- Components:
  - `config.DashboardConfig` — env-parsed configuration
  - `service.DashboardService` — health aggregation, control actions, pipeline trigger
  - `app.create_app()` — Flask app factory exposing the HTTP API

## HTTP API

- `GET /api/health/system` → Aggregated liveness for services and observability
- `GET /api/services` → List of known services + endpoint links
- `POST /api/services/<name>:action` → Control service (start|stop|restart)
  - Body: `{ "action": "start|stop|restart" }`
  - Requires: `ML_DASHBOARD_USE_COMPOSE=1` and a compose file
- `POST /api/pipeline/run` → Notify orchestrator of a pipeline run
  - Body: `{ "mode": "daily|backfill|realtime", ... }`
  - Emits an event via bus (noop by default)
- `POST /api/observability/grafana/provision` → Idempotent Grafana dashboard provisioning hook
  - Body: `{ "title": "Custom Title" }` (optional)
- `GET /api/observability/status` → Last-known Grafana provisioning status and embed metadata
- `GET /api/observability/summary` → Cold-path Prometheus snapshot (request rate, latency, failures)
- `GET /api/observability/stores` → Store health summaries (feature/model/strategy/data freshness)
- `GET /health` → Control-plane health (always 200 once bootstrapped)
- `GET /metrics` → Prometheus metrics for dashboard process

Notes

- The dashboard performs only best‑effort operations; errors in optional dependencies
  (e.g., message bus) do not alter control flow.

## Environment and Configuration

Environment variables parsed by `DashboardConfig.from_env()`:

- `ML_DASHBOARD_USE_COMPOSE`: enable docker compose control (default: false)
- `ML_DASHBOARD_COMPOSE_FILE`: path to compose file; fallback discovery:
  `ml/deployment/docker-compose.yml`, `docker-compose.yml`
- `ML_DASHBOARD_TIMEOUT`: HTTP timeouts for health probes (seconds, default 2.5)
- Service ports (host-side mappings used by compose by default):
  - `ML_ACTOR_HOST_PORT` (default: 8000)
- `ML_STRATEGY_HOST_PORT` (default: 8001)
- `ML_PIPELINE_HOST_PORT` (default: 8081)
- `GRAFANA_HOST_PORT` (default: 3000)
- `PROMETHEUS_HOST_PORT` (default: 9090)
- `REDIS_HOST_PORT` (default: 6380)
- `GRAFANA_URL`: base URL for Grafana API calls and embeds (defaults to `http://localhost:<GRAFANA_HOST_PORT>`)
- `GRAFANA_API_TOKEN` / (`GF_ADMIN_USER`, `GF_SECURITY_ADMIN_PASSWORD`): credentials for provisioning
- `GRAFANA_FOLDER_UID`: optional folder where dashboards are stored
- `ML_DASHBOARD_GRAFANA_UID`: stable dashboard UID used for idempotent provisioning (default: `ml-control-plane`)
- `ML_DASHBOARD_GRAFANA_TITLE`: dashboard title override
- `ML_DASHBOARD_GRAFANA_REFRESH`: dashboard auto-refresh interval (default: `30s`)
- `ML_DASHBOARD_GRAFANA_DATASOURCE_UID`: Prometheus datasource UID injected into the dashboard template
- `ML_DASHBOARD_GRAFANA_EMBED`: enable iframe embeds in the HTML template (truthy/falsey)
- `ML_DASHBOARD_GRAFANA_PANELS`: comma-separated panel IDs to embed via `d-solo`
- `ML_DASHBOARD_GRAFANA_THEME`: embed theme (`light` default)
- `ML_DASHBOARD_GRAFANA_ORG_ID`: Grafana organisation for embeds (default: 1)
- `ML_DASHBOARD_GRAFANA_EMBED_URL`: optional override for embed base URL (defaults to `GRAFANA_URL`)
- `ML_DASHBOARD_GRAFANA_PROVISION_ON_START`: provision Grafana during service startup when enabled
- `PROMETHEUS_URL`: base URL for the Prometheus API (defaults to `http://localhost:<PROMETHEUS_HOST_PORT>`)
- `ML_DASHBOARD_PROM_TIMEOUT`: timeout for Prometheus summary queries (seconds, default: 2.5)
- `ML_DB_CONNECTION`: PostgreSQL connection string for store summaries (optional; when unset summaries fall back to disabled state)
- `ML_DASHBOARD_STORE_CACHE_TTL`: TTL for cached store summaries (seconds, default: 30)
- `ML_DASHBOARD_STORE_CACHE_MAX`: Maximum cached store summary entries (default: 8)
- `ML_DASHBOARD_STORE_TOP_DATASETS`: Number of datasets to include in data-store freshness list (default: 5)
- `ML_DASHBOARD_STORE_SUMMARY`: Enable/disable store summary endpoint (`1`/`0`, default: enabled)
- `ML_DASHBOARD_TOKENS`: JSON array of bearer tokens (`[{"value":"token","expires":"2025-01-01T00:00:00Z"}]`)
- `ML_DASHBOARD_TOKEN`: Legacy single token fallback (no expiry) with optional `ML_DASHBOARD_TOKEN_EXPIRES`
- Bus control (optional): `ML_BUS_*` from `ml/config/bus.py`
- Logging: `ML_LOG_LEVEL`, `ML_LOG_FORMAT`, optional `LOG_FILE`
- `ML_DASHBOARD_EVENTS_CACHE_TTL`: TTL (seconds) for cached event payloads (default: 5)
- `ML_DASHBOARD_EVENTS_CACHE_MAX`: Maximum recent events retained in cache (default: 200)
- `ML_DASHBOARD_EVENTS_POLL_INTERVAL`: Interval (seconds) for optional background polling (default: 0 = disabled)

## Metrics

- `ml_dashboard_requests_total{route, method, status}` — control-plane request counter
- `ml_dashboard_latency_seconds{route}` — request latency histogram (seconds)
- `ml_dashboard_registry_cache_hits_total{entry}` / `ml_dashboard_registry_cache_misses_total{entry}` — cache behaviour for registry-backed endpoints
- `ml_dashboard_registry_fallback_total{registry,reason}` — fallback activations when registries are unavailable or error
- `ml_dashboard_registry_retry_total{registry}` — retry attempts while constructing registry clients
- `ml_dashboard_events_cache_hits_total` / `ml_dashboard_events_cache_misses_total` — cache behaviour for event history
- `ml_dashboard_events_poll_total` — number of Redis poll attempts (background + on-demand)
- `ml_dashboard_events_failure_total{reason}` — event polling failures (disabled/error)

Use Grafana to visualize these metrics; Prometheus scrapes `/metrics` from the
dashboard service.

## Registry Access and Fallbacks

Registry-backed endpoints (`/api/registry/*`) use a shared 30-second TTL cache keyed
by endpoint and request filters. Mutating actions such as deploy, hot reload,
rollback, promote, and deprecate invalidate matching cache entries so subsequent
reads reflect the latest state.

- Cache hit/miss counters expose whether repeated API calls are served from memory or
  require fresh registry reads.
- `_safe_get` and registry constructors call `ml.common.retry_utils.retry_with_backoff`
  to absorb transient failures before incrementing fallback counters.
- When registry reads fail, the dashboard emits `ml_dashboard_registry_fallback_total`
  and returns an empty payload rather than raising, preserving cold-path resilience.
- Dummy registry fallbacks are enabled only when `ML_ALLOW_DUMMY=1` (or equivalent truthy
  value) is present in the environment; otherwise failures surface via the metrics while
  the API returns empty payloads.
- Event history is cached for a configurable TTL (default 5 seconds); optional background
  polling can be enabled via `ML_DASHBOARD_EVENTS_POLL_INTERVAL` to warm the cache.

These safeguards keep dashboard responses stable while surfacing observability
signals that can be consumed by the broader monitoring stack.

## UI Refresh Behaviour

- The HTML template refreshes core sections (health, services, models) every 15 seconds
  and polls recent events every 5 seconds using `fetch` timers—no hot-path dependencies
  are introduced.
- Prometheus snapshot cards (request rate, latency P95, failure counts) are populated via
  the cold-path `/api/observability/summary` endpoint and fall back gracefully when
  Prometheus is unreachable.
- Optional Grafana embeds are rendered from environment-driven panel lists; if embeds are
  disabled the UI hides the card while still exposing the provisioning status link.
- Store health tables render via `/api/observability/stores`, displaying connectivity/write status,
  freshness for feature/model/strategy stores, and top dataset freshness for the data store.
  - Manual controls remain available for immediate refresh via the existing buttons.

## Deployment (Docker Compose)

Add a `ml_dashboard` service to the stack (see `ml/deployment/docker-compose.yml`).

Example excerpt:

```yaml
  ml_dashboard:
    build:
      context: ../..
      dockerfile: ml/deployment/Dockerfile.pipeline
    environment:
      ML_DASHBOARD_USE_COMPOSE: "${ML_DASHBOARD_USE_COMPOSE:-0}"
      ML_DASHBOARD_TIMEOUT: "${ML_DASHBOARD_TIMEOUT:-2.5}"
      ML_ACTOR_HOST_PORT: "${ML_ACTOR_HOST_PORT:-8000}"
      ML_STRATEGY_HOST_PORT: "${ML_STRATEGY_HOST_PORT:-8001}"
      ML_PIPELINE_HOST_PORT: "${ML_PIPELINE_HOST_PORT:-8081}"
      PROMETHEUS_HOST_PORT: "${PROMETHEUS_HOST_PORT:-9090}"
      GRAFANA_HOST_PORT: "${GRAFANA_HOST_PORT:-3000}"
    command: ["python", "-m", "ml.dashboard.serve"]
    ports:
      - "${ML_DASHBOARD_HOST_PORT:-8010}:8010"
    depends_on:
      - prometheus
      - grafana
    networks:
      - nautilus-ml
```

## Usage

Python (local):

```python
from ml.dashboard import create_app, DashboardConfig

app = create_app(DashboardConfig.from_env())
app.run(host="0.0.0.0", port=8010)
```

HTTP:

```sh
curl -s localhost:8010/api/health/system | jq
curl -s localhost:8010/api/services | jq
curl -s -X POST localhost:8010/api/services/ml_pipeline:action -d '{"action":"restart"}' -H 'content-type: application/json'
curl -s -X POST localhost:8010/api/pipeline/run -d '{"mode":"backfill","instrument":"SPY.EQUS"}' -H 'content-type: application/json'
curl -s localhost:8010/api/observability/status | jq
curl -s localhost:8010/api/observability/summary | jq
```

## Constraints and Guardrails

- No direct `prometheus_client` imports.
- Use enums (`Stage`, `Source`, `EventStatus`) and topic builders.
- Keep publish/observability off hot path; best-effort try/except.
- Strict typing and `mypy --strict` clean for all new code.
