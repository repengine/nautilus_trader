# Dashboard Control Plane (Cold Path)

## Overview

The Dashboard Control Plane provides a small, typed HTTP API to observe and control
the Nautilus Trader ML system. It reuses existing health, metrics, and event
infrastructure and strictly follows the Universal ML Architecture Patterns.

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
- Bus control (optional): `ML_BUS_*` from `ml/config/bus.py`
- Logging: `ML_LOG_LEVEL`, `ML_LOG_FORMAT`, optional `LOG_FILE`

## Metrics

- `ml_dashboard_requests_total{route, method, status}` — control-plane request counter
- `ml_dashboard_latency_seconds{route}` — request latency histogram (seconds)

Use Grafana to visualize these metrics; Prometheus scrapes `/metrics` from the
dashboard service.

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
```

## Constraints and Guardrails

- No direct `prometheus_client` imports.
- Use enums (`Stage`, `Source`, `EventStatus`) and topic builders.
- Keep publish/observability off hot path; best-effort try/except.
- Strict typing and `mypy --strict` clean for all new code.

