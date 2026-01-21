# Nautilus Trader ML Deployment Stack

This directory packages the Docker Compose configurations for running the Nautilus Trader ML stack in production-like and test modes. It consolidates the previous deployment notes into a single guide so environment variables, host port overrides, and volume mappings live in one place.

## Stack Overview

| Stack | Compose Files | Purpose |
| ----- | ------------- | ------- |
| **Production stack (`ml`)** | `docker-compose.yml` (+ optional `docker-compose.override.yml`) | Persistent Postgres, Redis, streaming training runner + persistence worker, actors, strategy, pipeline, observability, dashboard. |
| **Test stack (`ml-test`)** | `docker-compose.test.yml` | Ephemeral stack with mock data, alternate ports, and separate volumes for CI/local verification. |

Compose assigns the project name from the first file (`name: ml` / `name: ml-test`). Make-based helpers (`make ml-up`, `make ml-down`) wrap the production stack.

## Requirements

- Docker Engine with Compose v2 (`docker compose`).
- Python tooling (`uv`, `make`) for migrations and helper scripts.
- Databento credentials when running against live market data.
- Rebuild the ML images after pulling this branch so the updated `ml_signal_actor` and `ml_strategy` containers pick up the new `databento` dependency (`docker compose build ml_signal_actor ml_strategy`).

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

The repository ships a `docker-compose.override.yml` that bridges the stack to an
external `docker_nautilus-network` (and expects a `nautilus-database` container).
If you are running the ML stack standalone, either create the external network or
temporarily move/rename the override file so Compose falls back to the defaults.

For the test stack, use the provided `ml/deployment/.env.test` when launching:

```bash
docker compose --env-file ml/deployment/.env.test -f ml/deployment/docker-compose.test.yml up -d
```

Once the stack is up, confirm the streaming persistence worker is healthy—the container keeps the Redis stream snapshots in sync for the dashboard state API. Tail its logs with:

```bash
docker compose logs -f streaming_persistence_worker
```

The `streaming_training_runner` container attaches the dataset planner and Lightning worker to the Redis bus so cohorts run without manual CLI invocations. It respects the `ML_STREAMING_*` settings from `.env` to choose the dataset slice, training limits, epoch budget, accelerator, and promotion threshold. Artifacts are copied into the registry (`/root/.nautilus/ml/models` by default) and per-run manifests are written under `ML_STREAMING_OUTPUT_DIR`. Set `ML_STREAMING_MAX_PLANS=0` for continuous operation; use `1` for a single cohort. When GPUs are unavailable either override `ML_STREAMING_ACCELERATOR=cpu` or merge the CPU override (`docker compose -f ml/deployment/docker-compose.yml -f ml/deployment/docker-compose.cpu.yml up`) to drop GPU reservations entirely.

Monitor the runner with:

```bash
docker compose logs -f streaming_training_runner
```

## Azure Spot Checkpointing

- Cloud-init and Terraform snippets live in `ml/deployment/azure/` for mounting Azure Blob/Files storage with managed identities. Render the templates with the appropriate storage account, container, and identity IDs before attaching them to spot VMs.
- Export `ML_STREAMING_CHECKPOINT_DIR` (the blobfuse mount point) and enable `ML_STREAMING_AZURE_EVENTS_ENABLED=1` so the runner’s scheduled-event watcher triggers `save_checkpoint_now` ahead of spot evictions.
- Additional operational guidance (VS Code Remote, dashboards, metrics) is tracked in `ml/docs/implementation/azure_spot_checkpointing_plan.md` and the dashboard runbook.

To review recent cohorts, run:

```bash
poetry run python -m ml.scripts.summarize_streaming_manifests \
    --manifest-dir ml_out/tft_streaming_artifacts/full_tft_95 \
    --limit 10
```

Paste the Markdown summary into `ml/docs/ops/streaming_scaling_experiments.md` to keep the runbook current.

## Environment Variables

| Variable | Default | Description | Defined In |
| -------- | ------- | ----------- | ---------- |
| `DATABENTO_API_KEY` | *(none)* | Databento API key for live data ingestion. | `.env`, `ml/deployment/.env.example`, pipeline/actor env blocks |
| `DATABENTO_DATASET` | `EQUS.MINI` | Databento dataset identifier. | `.env.example`, services |
| `SEC_IDENTITY` | *(none)* | Full SEC EDGAR User-Agent identity string (alternative to `SEC_USER_AGENT_*`). | `.env`, earnings ingestion |
| `SEC_USER_AGENT_NAME` | *(none)* | SEC EDGAR User-Agent contact name. | `.env`, earnings ingestion |
| `SEC_USER_AGENT_EMAIL` | *(none)* | SEC EDGAR User-Agent contact email. | `.env`, earnings ingestion |
| `SEC_USER_AGENT_PHONE` | *(none)* | SEC EDGAR User-Agent contact phone. | `.env`, earnings ingestion |
| `PIPELINE_MODE` | `daily` | Pipeline schedule (`daily`, `backfill`, `realtime`). | `.env.example`, `ml_pipeline` env |
| `PIPELINE_SCHEDULE` | `0 17 * * *` | Cron expression used when `PIPELINE_MODE=daily`. | `.env.example`, `ml_pipeline` env |
| `UNIVERSE_SYMBOLS` | `SPY.XNAS` | Comma-separated symbol universe. | `.env.example`, `ml_pipeline` env |
| `MARKET_DATASET_INPUTS` | *(empty)* | Optional JSON array or comma-separated descriptor IDs for supplemental feeds (defaults wire `EQUS.MINI_TBBO`, `EQUS.MINI_MBP1`, and `XNAS.ITCH_MBP10`). Entries are validated against `ml/config/market_feed_descriptors.json`; unsupported dataset/schema pairs now raise during pipeline bootstrap. | `.env`, `ml_pipeline` env |
| `MARKET_BACKFILL_LOOKBACK_DAYS` | `365` | Gap-detection window (days) applied to configured market datasets during orchestrator backfills (clamped to dataset license bounds). | `.env`, `ml_pipeline` env |
| `LOG_LEVEL` | `INFO` | Default log level for services. | `.env.example`, multiple services |
| `POSTGRES_HOST_PORT` | `5433` | Host port mapped to Postgres (`5433:5432`). | `.env.example`, compose `postgres.ports` |
| `REDIS_HOST_PORT` | `6380` | Host port mapped to Redis (`6380:6379`). | `.env.example`, compose `redis.ports` |
| `ML_ACTOR_HOST_PORT` | `8000` | Host port for `ml_signal_actor` HTTP interface. | `.env.example`, compose `ml_signal_actor.ports` |
| `ML_STRATEGY_HOST_PORT` | `8001` | Host port for `ml_strategy`. | `.env.example`, compose `ml_strategy.ports` |
| `ML_PIPELINE_HOST_PORT` | `8081` | Host port for pipeline service health API (container 8080). | `.env.example`, compose `ml_pipeline.ports` |
| `ML_DASHBOARD_HOST_PORT` | `8010` | Host port for dashboard UI (container 8010). | `.env.example`, compose `ml_dashboard.ports` |
| `ML_BUS_REDIS_STREAM` | `ml-events` | Redis Streams channel consumed by streaming persistence worker and dashboard. | `.env.example`, `streaming_persistence_worker` env |
| `ML_STREAM_PERSIST_*` | see defaults | Worker polling overrides (`ENABLE`, `BATCH_SIZE`, `BLOCK_MS`, `POLL_INTERVAL_SECONDS`). | `.env.example`, `streaming_persistence_worker` env |
| `ML_STREAMING_DATASET_DIR` | `ml_out/full_tft_95` | Dataset directory mounted in the runner container. | `.env`, `.env.example`, `streaming_training_runner` command |
| `ML_STREAMING_OUTPUT_DIR` | `ml_out/tft_streaming_artifacts/full_tft_95` | Directory for logits, telemetry, and manifests. | `.env`, `.env.example`, `streaming_training_runner` command |
| `ML_STREAMING_STATE_PATH` | `ml_out/streaming_training_state.json` | Local snapshot path consumed by the dashboard. | `.env`, `.env.example`, `streaming_training_runner` command |
| `ML_STREAMING_MAX_TOTAL_ROWS` | `120000` | Planner and worker row cap (<=0 disables). | `.env`, `.env.example`, `streaming_training_runner` command |
| `ML_STREAMING_MAX_TOTAL_SEQUENCES` | `90000` | Planner and worker sequence cap (<=0 disables). | `.env`, `.env.example`, `streaming_training_runner` command |
| `ML_STREAMING_MAX_SHARDS` | `32` | Planner and worker shard cap (<=0 disables). | `.env`, `.env.example`, `streaming_training_runner` command |
| `ML_STREAMING_MAX_EPOCHS` | `2` | Maximum training epochs per cohort. | `.env`, `.env.example`, `streaming_training_runner` command |
| `ML_STREAMING_PLAN_INTERVAL_SECONDS` | `900` | Delay between cohorts (0 runs once). | `.env`, `.env.example`, `streaming_training_runner` command |
| `ML_STREAMING_PROMOTION_THRESHOLD` | `0.55` | Metric gate used to flag promotable runs. | `.env`, `.env.example`, `streaming_training_runner` command |
| `ML_STREAMING_PROMOTION_COMMAND` | `true` | Optional command executed when a cohort clears promotion gates (placeholders: `{logits}`, `{manifest}`, `{model_id}`, `{plan_id}`, `{dataset_id}`); default `true` is a no-op. | `.env`, `.env.example`, `streaming_training_runner` env/command |
| `ML_STREAMING_ACCELERATOR` | `auto` | Lightning accelerator (`cpu`, `gpu`, or `auto`). | `.env`, `.env.example`, `streaming_training_runner` command |
| `ML_STREAMING_DEVICES` | `1` | Number of accelerator devices to use. | `.env`, `.env.example`, `streaming_training_runner` command |
| `ML_STREAMING_MAX_PLANS` | `0` | Number of cohorts to run (0 = infinite loop). | `.env`, `.env.example`, `streaming_training_runner` command |
| `ML_STREAMING_GPU_MONITOR_INTERVAL` | `30` | GPU sampling interval in seconds (<=0 disables). | `.env`, `.env.example`, `streaming_training_runner` command |
| `ML_STREAMING_BUS_RETRY_ATTEMPTS` | `3` | Number of attempts made when publishing events to the message bus. | `.env`, `.env.example`, `streaming_training_runner` command |
| `ML_STREAMING_BUS_RETRY_DELAY_SECONDS` | `0.5` | Delay between message-bus publish retries in seconds. | `.env`, `.env.example`, `streaming_training_runner` command |
| `ML_DB_CONNECTION` | *(optional)* | Host-facing Postgres URI used by local tooling/scripts. | `.env.example`, consumed by `MLIntegrationManager` |
| `TEST_POSTGRES_HOST_PORT` | `5434` | Host port for test Postgres (`ml-test` stack). | `.env.test`, `docker-compose.test.yml` |
| `TEST_REDIS_HOST_PORT` | `6381` | Host port for test Redis. | `.env.test`, `docker-compose.test.yml` |
| `TEST_ACTOR_HOST_PORT` | `8002` | Host port for test signal actor. | `.env.test`, `docker-compose.test.yml` |
| `TEST_PROMETHEUS_HOST_PORT` | `9091` | Host port for test Prometheus. | `.env.test`, `docker-compose.test.yml` |
| `PROMETHEUS_HOST_PORT` | `9090` | Host port surfaced in dashboard links; edit compose to remap the container port if needed. | `docker-compose.yml` (`ml_dashboard` env) |
| `GRAFANA_HOST_PORT` | `3000` | Host port surfaced in dashboard links; edit compose to remap the container port if needed. | `docker-compose.yml` (`ml_dashboard` env) |

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
| Postgres data | `/var/lib/postgresql/data` | Named volume `postgres_data` | Add a compose override to bind-mount a host path if you need long-term persistence outside Docker. |
| Prometheus TSDB | `/prometheus` | Named volume `prometheus_data` | Supply an override with a host bind to retain metrics beyond container lifecycle. |
| Grafana data | `/var/lib/grafana` | Named volume `grafana_data` | Use an override to capture dashboards locally when required. |
| Model artefacts | `/app/models` | `../models` (RO for actors, RW for pipeline) | Keep model artefacts under version control or adjust the bind target via override. |
| Pipeline data catalog | `/app/data` | Bind `../../data` (read/write) | Provide host directory for dataset outputs. |
| Pipeline logs | `/app/logs` | Bind `../logs` (read/write) | Rotate or mount elsewhere if desired. |

Test stack volumes (`ml-test_logs-test`, `ml-test_prometheus_test_data`) are created automatically and can be removed with `docker compose -f docker-compose.test.yml down -v` after use.

### Catalog Rehydration

The `ml_pipeline` container can restore the canonical `market_data` table from the on-disk Parquet catalog before contacting external data feeds. Configure this behaviour via environment variables (all optional):

| Variable | Default | Description |
| --- | --- | --- |
| `CATALOG_REHYDRATE_ENABLED` | `0` | Set to `1` to enable catalog → Postgres replay on startup. |
| `CATALOG_REHYDRATE_LOOKBACK_DAYS` | `5` | Rolling window (in days) to scan for missing coverage. |
| `CATALOG_REHYDRATE_BATCH_SIZE` | `1000` | Maximum rows inserted per batch write. |
| `CATALOG_REHYDRATE_IDENTIFIER_TEMPLATE` | `{instrument_id}-1-MINUTE-LAST-EXTERNAL` | Template used to resolve catalog identifiers for each instrument. |
| `CATALOG_REHYDRATE_IDENTIFIER_TEMPLATE_MAP` | _empty_ | Optional schema→template map (JSON or comma-delimited `schema=template`) to override identifiers per schema. |
| `CATALOG_REHYDRATE_DATASET_TYPE_TEMPLATES` | _defaults_ | Optional dataset_type→template map; defaults: bars=`{instrument_id}-1-MINUTE-LAST-EXTERNAL`, tbbo/trades/mbp1=`{instrument_id}`. |
| `CATALOG_REHYDRATE_TABLE` | `market_data` | Target SQL table for restored rows. |
| `CATALOG_REHYDRATE_RESCAN` | `0` | When `1`, re-run the rehydration pass before each scheduled ingestion cycle. |

When enabled, the pipeline compares catalog coverage with SQL coverage and only replays missing day buckets, preventing redundant Databento downloads during recovery.

### Coverage Restoration Gate

Set `COVERAGE_RESTORE_ENABLED=1` to require successful coverage classification/restoration before ingestion proceeds. The entrypoint treats failures as fatal unless `COVERAGE_RESTORE_ALLOW_FAILURE=1` is set (not recommended for Tier‑1). Catalog coverage and the rehydrator share identifier templates: bars use bar-type identifiers by default, TBBO/Trades/MBP use raw `instrument_id`, and you can override schemas via `CATALOG_REHYDRATE_IDENTIFIER_TEMPLATE_MAP` when needed. This keeps parquet scans fast and avoids redundant Databento downloads when the catalog already holds the data.

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

### Schema Bootstrap & External Databases

- Run the schema audit before applying migrations to legacy databases:

  ```bash
  poetry run python -m ml.stores.schema_audit inspect --db-url postgresql://ml:ml@postgres.nautilus-ml:5432/nautilus_ml
  ```

  The CLI reports whether `ml_feature_values`, `ml_model_predictions`, `ml_strategy_signals`, and `market_data` are partitioned, have default partitions, and keep the generated `spread`/`mid_price` columns. It also ensures helper functions (e.g., `create_monthly_partitions`) exist.
- Once the audit is green, run the migrations runner (automatically executed inside the pipeline container) from the same network boundary:

  ```bash
  poetry run python -m ml.stores.migrations_runner apply \
    --db-url postgresql://ml:ml@postgres.nautilus-ml:5432/nautilus_ml
  ```

- The runner records checksums in `ml_schema_migrations` and the pipeline entrypoint refuses to start ingestion if migrations or the schema health check fail. Use `plan` mode for dry runs:
  `poetry run python -m ml.stores.migrations_runner plan --db-url …`
- Instrumentation guardrails: the pipeline now verifies that `ml_data_events` and `ml_data_watermarks` exist immediately after migrations. If either table is missing, the container exits with a `SchemaHealthCheckError` so Databento ingestion never runs against an uninstrumented database. Re-run the migrations runner (or load the missing tables) before restarting the stack.

  ```bash
  uv run --active --no-sync python -c "from ml.stores.infrastructure import check_db_prereqs; import os; uri=os.getenv('ML_DB_CONNECTION','postgresql://postgres:postgres@localhost:' + os.getenv('POSTGRES_HOST_PORT','5433') + '/nautilus'); print(check_db_prereqs(uri))"
  ```

- **Backups:** Use `make backup` / `make restore` or `docker exec postgres pg_dump ...` as needed.

### Coverage Restoration

- Enable `COVERAGE_RESTORE_ENABLED=1` so the pipeline automatically inspects SQL vs. parquet coverage before ingestion resumes. Optional tuning knobs:
  - `COVERAGE_RESTORE_LOOKBACK_DAYS` (default `5`)
  - `COVERAGE_MAX_BUCKETS_PER_RUN` (default `500`) caps how many buckets are restored or re-ingested per run to prevent runaway recovery jobs.
- Startup sequence:
  1. `CoverageManager` runs a schema audit (via `ml.stores.schema_audit`) and classifies gaps per dataset/instrument.
  2. Missing buckets that exist in the parquet catalog are restored via `ParquetCatalogRehydrator` (the bucket list is passed directly so only the uncovered windows replay).
  3. Remaining buckets are forwarded to `DataScheduler.run_targeted_update`, which ingests just those windows from Databento.
- Manual CLI (mirrors the pipeline automation):

  ```bash
  poetry run python -m ml.data.coverage.manager \
    --db-url postgresql://ml:ml@postgres.nautilus-ml:5432/nautilus_ml \
    --catalog-path /data/catalog \
    --dataset EQUS.MINI:ohlcv-1m:AAPL.XNAS,MSFT.XNAS \
    --lookback-days 3 --json
  ```

- Provide multiple `--dataset dataset_id:schema:symbol1,symbol2` arguments to cover TBBO/MBP datasets; the CLI prints bucket classifications (JSON or logs) so you can validate the plan before restarting services.
  For operator tooling that reuses the deployment environment variables (universe symbols, dataset overrides, etc.), run:

  ```bash
  poetry run python -m ml.cli.coverage_restore --json
  ```

  The CLI boots the `PipelineRunner`, executes a single coverage restoration pass (including targeted scheduler updates), and emits the same summary exposed on the pipeline `/health` endpoint.

## Observability & Interfaces

- Pipeline health: `http://localhost:${ML_PIPELINE_HOST_PORT:-8081}/health`
  - The JSON payload now includes a `coverage` section with the last run timestamps, bucket counts, and any recent restoration errors.
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
