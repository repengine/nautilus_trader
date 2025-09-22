# ML Dashboard Runbook (Cold Path)

## Overview

This runbook describes how to operate the ML dashboard control plane. The
service is cold-path only and exposes cached health, observability, and control
endpoints. All workflows rely on the existing infra components described in
`ml/docs/architecture/dashboard_control_plane.md`.

## Quick Start

- Launch with `python -m ml.dashboard.serve` or via the Docker Compose service
  defined in `ml/deployment/docker-compose.yml`.
- Ensure the following dependencies are reachable before exposing the UI:
  - Grafana (`GRAFANA_URL`) for provisioning and optional embeds.
  - Prometheus (`PROMETHEUS_URL`) for snapshot metrics.
  - Postgres (`ML_DB_CONNECTION`) when store summaries are enabled.
- Export `ML_DASHBOARD_USE_COMPOSE=0` when running outside Docker Compose. The
  service controllers fall back to `NoopServiceController` in this mode.

## Authentication Tokens

- Preferred configuration: set `ML_DASHBOARD_TOKENS` to a JSON array of
  `{ "value": str, "expires": iso8601 }` objects. Tokens are validated using
  `hmac.compare_digest` and expiry timestamps are honoured.
- Legacy fallback: `ML_DASHBOARD_TOKEN` (+ optional
  `ML_DASHBOARD_TOKEN_EXPIRES`) is still supported. Mixing both formats merges
  the token list.
- Rotate tokens by updating the environment variable and restarting the
  service. All validation outcomes are tracked via the
  `ml_dashboard_auth_validations_total{result="*"}` counter; monitor spikes in
  `invalid` to detect misuse.

## Observability Endpoints

| Endpoint | Purpose | Notes |
| --- | --- | --- |
| `/api/health/system` | Aggregated component health | Uses cached registry results when available. |
| `/api/observability/status` | Grafana provisioning status | Includes cached dashboard URL and embed metadata. |
| `/api/observability/summary` | Prometheus snapshot (request rate, latency P95, error counts) | Returns `{ "ok": false }` when Prometheus is unavailable; metrics recorded in `ml_dashboard_requests_total`. |
| `/api/observability/stores` | Store health summaries | Requires `ML_DB_CONNECTION` and enabled store health feature. |

### Store Health Triage

- The summary is sourced via `summarize_all_stores` (see
  `ml/dashboard/store_health.py`) and cached with TTL to avoid repeated DB
  hits.
- `healthy=False` with `fallback_active=True` indicates the store client or the
  database engine was unavailable. Inspect `_STORE_FALLBACK_TOTAL` metrics for
  detailed reasons.
- When `latest_event_ns` is null, investigate upstream pipelines; freshness is
  calculated per-store and for the top N datasets (`ML_DASHBOARD_STORE_TOP_DATASETS`).
- Disable the feature temporarily by exporting `ML_DASHBOARD_STORE_SUMMARY=0`.

### Grafana Provisioning

- Trigger manually via `POST /api/observability/grafana/provision`. Supply an
  optional `{"title": "custom name"}` payload to override the dashboard name.
- Provisioning is idempotent; cached success responses include `cached: true`.
- Metrics:
  - `ml_dashboard_requests_total{route="/api/observability/grafana/provision"}`
    for request outcomes.
  - `ml_dashboard_latency_seconds{route="/api/observability/grafana/provision"}`
    to monitor latency.
- For automated provisioning on boot set
  `ML_DASHBOARD_GRAFANA_PROVISION_ON_START=1`. Failures are captured in
  `_grafana_status` and surfaced via `/api/observability/status`.

### Prometheus Snapshots

- Configure `PROMETHEUS_URL` to point at the scrape API. Queries are executed
  via `PrometheusQueryHelper` with retry/backoff; timeouts are controlled by
  `ML_DASHBOARD_PROM_TIMEOUT`.
- Dashboard cards display the last snapshot value and fall back to a placeholder
  when the helper is disabled or returns `{ "ok": false }`.

## UI Operations

- Access the HTML dashboard root (`/`) for a lightweight overview. Cached
  sections update on load; click “Refresh” on the Events card to trigger a new
  poll.
- Grafana embeds are shown only when `ML_DASHBOARD_GRAFANA_EMBED=1` **and** at
  least one panel ID is provided via `ML_DASHBOARD_GRAFANA_PANELS`.
- Store health tables include freshness timestamps (ISO) and age (seconds) for
  quick triage.

## Validations and Diagnostics

- Static checks: `uv run --active --no-sync mypy ml/dashboard --strict` and
  `ruff check ml/dashboard ml/tests/unit/dashboard`.
- Targeted tests: `uv run --active --no-sync pytest ml/tests/unit/dashboard/`
  and integration coverage via `uv run --active --no-sync pytest \
  ml/tests/integration/dashboard/`.
- Metrics validators: `make validate-metrics`; advisory suite `make
  validate-nautilus-patterns` before releases.
- Inspect Prometheus metrics locally with `curl -s localhost:8010/metrics | rg
  ml_dashboard_`.
- Runtime metrics snapshot: import `DashboardService` and call
  `get_metrics_snapshot()` to inspect cache hit ratios, store summary latency
  P95, Grafana provisioning success rate, and registry latency P95. Use
  `evaluate_success_criteria()` to receive a pass/fail summary against the
  roadmap thresholds.

## Troubleshooting

- **Token failures**: Check `ml_dashboard_auth_validations_total` breakdown and
  ensure `ML_DASHBOARD_TOKENS` is correctly formatted JSON. Fall back to a
  single token by setting `ML_DASHBOARD_TOKENS` empty and using
  `ML_DASHBOARD_TOKEN`.
- **Grafana provisioning errors**: Fetch `/api/observability/status` and review
  the `error` field. Verify the API token and folder UID; rerun provisioning
  with `force=true` using the service method if needed.
- **Store summary errors**: Inspect
  `ml_dashboard_store_summary_failures_total{reason}` and confirm the Postgres
  connection string. Temporarily disable the feature while databases are
  offline.
- **Event polling lag**: Validate Redis connectivity and review
  `ml_dashboard_events_failure_total`. Cached results may persist up to
  `ML_DASHBOARD_EVENTS_CACHE_TTL` seconds.
