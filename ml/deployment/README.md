# Nautilus Trader ML Deployment Stack

This directory packages the Docker Compose configurations for running the Nautilus Trader ML stack in production-like and test modes. It consolidates the previous deployment notes into a single guide so environment variables, host port overrides, and volume mappings live in one place.

## Stack Overview

| Stack | Compose Files | Purpose |
| ----- | ------------- | ------- |
| **Production stack (`ml`)** | `docker-compose.yml` (+ optional `docker-compose.override.yml`) | Persistent Postgres, Redis, actors, strategy, pipeline, observability, dashboard. |
| **Test stack (`ml-test`)** | `docker-compose.test.yml` | Ephemeral stack with mock data, alternate ports, and separate volumes for CI/local verification. |

Compose assigns the project name from the first file (`name: ml` / `name: ml-test`). Make-based helpers (`make ml-up`, `make ml-down`) wrap the production stack.

## Requirements

- Docker Engine with Compose v2 (`docker compose`).
- Python tooling (`uv`, `make`) for migrations and helper scripts.
- Databento credentials when running against live market data.

## Quick Start (Production Stack)

```bash
cd ml/deployment
cp .env.example .env  # Edit defaults to match your environment
# Optional: customise volume mounts by editing docker-compose.override.yml or adding your own override
make -C ../.. ml-up
make -C ../.. ml-ps     # Check container status
make -C ../.. ml-logs   # Follow pipeline logs
# When finished
make -C ../.. ml-down
```

> Compose automatically loads the `.env` file that lives next to `docker-compose.yml`. The sample file documents the supported overrides; adjust values before `make ml-up`.

For the test stack, use the provided `ml/deployment/.env.test` when launching:

```bash
docker compose --env-file ml/deployment/.env.test -f ml/deployment/docker-compose.test.yml up -d
```

## Environment Variables

| Variable | Default | Description | Defined In |
| -------- | ------- | ----------- | ---------- |
| `DATABENTO_API_KEY` | *(none)* | Databento API key for live data ingestion. | `.env`, `ml/deployment/.env.example`, pipeline/actor env blocks |
| `DATABENTO_DATASET` | `EQUS.MINI` | Databento dataset identifier. | `.env.example`, services |
| `PIPELINE_MODE` | `daily` | Pipeline schedule (`daily`, `backfill`, `realtime`). | `.env.example`, `ml_pipeline` env |
| `PIPELINE_SCHEDULE` | `0 17 * * *` | Cron expression used when `PIPELINE_MODE=daily`. | `.env.example`, `ml_pipeline` env |
| `UNIVERSE_SYMBOLS` | `SPY.XNAS` | Comma-separated symbol universe. | `.env.example`, `ml_pipeline` env |
| `LOG_LEVEL` | `INFO` | Default log level for services. | `.env.example`, multiple services |
| `POSTGRES_HOST_PORT` | `5433` | Host port mapped to Postgres (`5433:5432`). | `.env.example`, compose `postgres.ports` |
| `REDIS_HOST_PORT` | `6380` | Host port mapped to Redis (`6380:6379`). | `.env.example`, compose `redis.ports` |
| `ML_ACTOR_HOST_PORT` | `8000` | Host port for `ml_signal_actor` HTTP interface. | `.env.example`, compose `ml_signal_actor.ports` |
| `ML_STRATEGY_HOST_PORT` | `8001` | Host port for `ml_strategy`. | `.env.example`, compose `ml_strategy.ports` |
| `ML_PIPELINE_HOST_PORT` | `8081` | Host port for pipeline service health API (container 8080). | `.env.example`, compose `ml_pipeline.ports` |
| `ML_DASHBOARD_HOST_PORT` | `8010` | Host port for dashboard UI (container 8010). | `.env.example`, compose `ml_dashboard.ports` |
| `ML_DB_CONNECTION` | *(optional)* | Host-facing Postgres URI used by local tooling/scripts. | `.env.example`, consumed by `MLIntegrationManager` |
| `TEST_POSTGRES_HOST_PORT` | `5434` | Host port for test Postgres (`ml-test` stack). | `.env.test`, `docker-compose.test.yml` |
| `TEST_REDIS_HOST_PORT` | `6381` | Host port for test Redis. | `.env.test`, `docker-compose.test.yml` |
| `TEST_ACTOR_HOST_PORT` | `8002` | Host port for test signal actor. | `.env.test`, `docker-compose.test.yml` |
| `TEST_PROMETHEUS_HOST_PORT` | `9091` | Host port for test Prometheus. | `.env.test`, `docker-compose.test.yml` |

Set these in `.env` (one per line `KEY=value`). The Compose CLI will substitute values and the Make tasks will pick them up automatically. When orchestration requires different ports, create stack-specific `.env` files and pass them via `COMPOSE_PROJECT_NAME`/`--env-file` as needed.

## Port Map

| Service | Container Port(s) | Default Host Port | Override Mechanism |
| ------- | ----------------- | ----------------- | ------------------ |
| Postgres (`postgres`) | 5432 | `${POSTGRES_HOST_PORT:-5433}` | Set `POSTGRES_HOST_PORT` in `.env` prior to `ml-up`. |
| Redis (`redis`) | 6379 | `${REDIS_HOST_PORT:-6380}` | Set `REDIS_HOST_PORT`. |
| ML Signal Actor (`ml_signal_actor`) | 8000 | `${ML_ACTOR_HOST_PORT:-8000}` | Set `ML_ACTOR_HOST_PORT`. |
| ML Strategy (`ml_strategy`) | 8001 | `${ML_STRATEGY_HOST_PORT:-8001}` | Set `ML_STRATEGY_HOST_PORT`. |
| Pipeline API (`ml_pipeline`) | 8080 | `${ML_PIPELINE_HOST_PORT:-8081}` | Set `ML_PIPELINE_HOST_PORT`. |
| Dashboard (`ml_dashboard`) | 8010 | `${ML_DASHBOARD_HOST_PORT:-8010}` | Set `ML_DASHBOARD_HOST_PORT`. |
| Prometheus (`prometheus`) | 9090 | `9090` | Edit compose file or add an override to change. |
| Grafana (`grafana`) | 3000 | `3000` | Edit compose file or add an override to change. |

The test stack reserves its own ports via environment overrides (`TEST_POSTGRES_HOST_PORT`, `TEST_REDIS_HOST_PORT`, `TEST_ACTOR_HOST_PORT`, `TEST_PROMETHEUS_HOST_PORT`) defined in `.env.test`; adjust those values when running integration stacks alongside production.

## Volume Map

| Purpose | Container Path | Default Backing | Customisation |
| ------- | -------------- | ---------------- | ------------- |
| Postgres data | `/var/lib/postgresql/data` | Named volume `postgres_data` | `docker-compose.override.yml` binds to `/home/nate/nautilus_data/postgres`; adjust `device` or provide your own override. |
| Prometheus TSDB | `/prometheus` | Named volume `prometheus_data` | Override binds to `/home/nate/nautilus_data/prometheus`. |
| Grafana data | `/var/lib/grafana` | Named volume `grafana_data` | Override binds to `/home/nate/nautilus_data/grafana`. |
| Model artefacts | `/app/models` | Read-only bind `../models` | Ensure models checked out locally. |
| Pipeline data catalog | `/app/data` | Bind `../data` (read/write) | Provide host directory for dataset outputs. |
| Pipeline logs | `/app/logs` | Bind `../logs` (read/write) | Rotate or mount elsewhere if desired. |

Test stack volumes (`ml-test_logs-test`, `ml-test_prometheus_test_data`) are created automatically and can be removed with `docker compose -f docker-compose.test.yml down -v` after use.

### Cleaning Up Old Volumes

Anonymous volumes (long hex names under `~/docker-data/volumes`) are safe to prune once the containers that created them are gone:

```bash
docker volume ls --filter label=com.docker.volume.anonymous=
docker volume rm <name>  # remove specific ones
# or
docker volume prune      # remove all unused volumes
```

## Managing the Stack

Common Make targets (run from repository root):

- `make ml-up` – start the production stack using `docker-compose.yml` (+ override if present).
- `make ml-up-core` – start core services without rebuilding images.
- `make ml-down` – stop the stack and remove named volumes.
- `make ml-logs` – tail `ml_pipeline` logs.
- `make ml-ps` – show container status.
- `make ml-migrate` – apply database migrations through the running Postgres container.

For the test stack:

```bash
cd ml/deployment
docker compose --env-file ./.env.test -f docker-compose.test.yml up -d
# ... run tests ...
docker compose --env-file ./.env.test -f docker-compose.test.yml down -v
```

## Database Operations

- **Apply migrations:** `make ml-migrate` or run `uv run --active --no-sync python -m ml.deployment.migrations --apply --compose-file ml/deployment/docker-compose.yml`.
- **Validate prerequisites:**

  ```bash
  uv run --active --no-sync python -c "from ml.stores.infrastructure import check_db_prereqs; import os; uri=os.getenv('ML_DB_CONNECTION','postgresql://postgres:postgres@localhost:' + os.getenv('POSTGRES_HOST_PORT','5433') + '/nautilus'); print(check_db_prereqs(uri))"
  ```

- **Backups:** Use `make backup` / `make restore` or `docker exec postgres pg_dump ...` as needed.

## Observability & Interfaces

- Pipeline health: `http://localhost:${ML_PIPELINE_HOST_PORT:-8081}/health`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (default credentials `admin/admin`)
- Dashboard control plane: `http://localhost:${ML_DASHBOARD_HOST_PORT:-8010}`

Ensure `ml/docs/ops/dashboard_runbook.md` is consulted for dashboard-specific configuration and authentication details.

## Local Dry Run (Optional)

For bare-metal experiments without Docker, follow `python run_local_dry_run.py` as documented in the repository root README. Ensure Postgres is available locally and migrations have been applied using the scripts above.

## Troubleshooting

- **Port already in use:** Set the relevant host override in `.env` (for example `POSTGRES_HOST_PORT=5543`) before launching the stack.
- **Pipeline container exits immediately:** Check `make ml-logs` for Databento or database connection errors. Validate that migrations have run and that credentials are present.
- **Dashboard cannot reach services:** Confirm that `ML_DB_CONNECTION` (for host tooling) and message bus settings are aligned with Compose defaults.
- **Stale volumes consuming space:** Identify anonymous volumes with `docker volume ls --filter label=com.docker.volume.anonymous=` and remove those no longer referenced.
- **Test stack interfering with production:** The test compose file deliberately uses separate ports (`5434`, `6381`, `8002`, `9091`). Do not run both stacks without ensuring ports differ.

## Further Reading

- `ml/docs/context/context_deployment.md` – architectural patterns and deployment conventions.
- `ml/docs/ops/dashboard_runbook.md` – dashboard operations and health procedures.
- `ml/docs/development/CODING_STANDARDS.md` – coding and deployment standards.

For issues, start with `make ml-logs`, check container health (`make ml-ps`), and consult the documentation above.
