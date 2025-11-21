# ML Signal Actor Deployment Guide

## Current Configuration

The ML Signal Actor is configured to receive live market data from Databento's EQUS.MINI dataset with **EXTERNAL bar aggregation** as the default.

### Key Settings
- **Instrument**: `SPY.EQUS` (aggregated US equities)
- **Bar Type**: `SPY.EQUS-1-MINUTE-LAST-EXTERNAL` (Databento OHLCV-1m bars)
- **Dataset**: `EQUS.MINI` (Databento aggregated equities feed)
- **Model**: `dummy_bullish_model.onnx` (replace with your trained model)

## Deployment Commands

### Quick Start

```bash
# Set your Databento API key
export DATABENTO_API_KEY="your-api-key-here"

# Start the ML Signal Actor only
docker compose -f ml/deployment/docker-compose.yml up -d ml_signal_actor

# Start the full stack (actor + strategy + pipeline)
docker compose -f ml/deployment/docker-compose.yml up -d

# Monitor logs
docker compose -f ml/deployment/docker-compose.yml logs -f ml_signal_actor

# Check health
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

### Direct Docker Run (without compose)

```bash
# Build the image
docker build -f ml/deployment/Dockerfile.actor -t ml-signal-actor .

# Run the container
docker run --rm \
  --name ml-signal-actor \
  -e DATABENTO_API_KEY="your-api-key-here" \
  -e USE_DUMMY_STORES="false" \
  -e INSTRUMENT_ID="SPY.EQUS" \
  -e BAR_TYPE="SPY.EQUS-1-MINUTE-LAST-EXTERNAL" \
  -e DB_CONNECTION="postgresql://postgres:postgres@localhost:5432/nautilus" \
  -v $(pwd)/ml/models:/app/models:ro \
  -p 8000:8000 \
  ml-signal-actor
```

## Important Notes

### Schema migrations & health checks
- Run the schema audit to confirm the database already matches the partitioned layout expected by the pipeline:

  ```bash
  poetry run python -m ml.stores.schema_audit inspect \
    --db-url postgresql://USER:PASS@postgres.nautilus-ml:5432/nautilus_ml
  ```

  The audit reports whether the feature/model/strategy tables are partitioned by `ts_event`, whether `market_data` retains the generated `spread`/`mid_price` columns, and whether helper functions (e.g., `create_monthly_partitions`) exist.
- After the audit is clean, run the migrations runner (also executed automatically by the pipeline entrypoint) to apply SQL files and record checksums:

  ```bash
  poetry run python -m ml.stores.migrations_runner apply \
    --db-url postgresql://USER:PASS@postgres.nautilus-ml:5432/nautilus_ml
  ```

- Leave the `ml_schema_migrations` table intact; it prevents redundant DDL on each restart and surfaces checksum mismatches immediately. If any audit/migration check fails the pipeline container exits before Databento ingestion resumes.
- The entrypoint now also refuses to start unless the instrumentation tables (`ml_data_events`, `ml_data_watermarks`) exist. Run the migrations runner (or seed the tables) before restarting so coverage metrics and scheduler telemetry stay intact.

### Coverage Restoration
- Set `COVERAGE_RESTORE_ENABLED=1` so the pipeline checks for SQL/parquet gaps before it resumes ingestion. Optional tuning: `COVERAGE_RESTORE_LOOKBACK_DAYS` (default `5`).
  - Use `COVERAGE_MAX_BUCKETS_PER_RUN` (default `500`) to cap each restoration pass; the pipeline logs skipped counts when residual gaps remain.
- At startup the coverage manager:
  1. Runs the schema audit.
  2. Restores missing buckets from the parquet catalog when possible.
  3. Invokes `run_targeted_update` so Databento only replays the residual buckets.
- Manual CLI (same logic as the pipeline):

  ```bash
  poetry run python -m ml.data.coverage.manager \
    --db-url postgresql://USER:PASS@postgres.nautilus-ml:5432/nautilus_ml \
    --catalog-path /data/catalog \
    --dataset EQUS.MINI:ohlcv-1m:AAPL.XNAS,MSFT.XNAS
  ```

- Supply multiple `--dataset` arguments for TBBO/MBP datasets and add `--json` to capture machine-readable summaries for runbooks.
  To run the exact pipeline workflow (build scheduler config from env vars, execute catalog restoration + targeted Databento ingestion) outside the container, use:

  ```bash
  poetry run python -m ml.cli.coverage_restore --json
  ```

  The CLI emits the same summary that now appears in the pipeline health payload.

### Databento dependency in runtime images
- The `ml_signal_actor` and `ml_strategy` Dockerfiles now install the `databento` package so targeted Databento calls never fail at import time. Rebuild the images (`docker compose build ml_signal_actor ml_strategy`) after pulling these changes if you publish custom artifacts.

### MARKET_DATASET_INPUTS validation
- Supplemental dataset configs (`MARKET_DATASET_INPUTS` env var) are now validated during pipeline bootstrap. Each entry must reference a descriptor listed in `ml/config/market_feed_descriptors.json`, and the resolved dataset/schema pair must match that allowlist. Invalid descriptors (e.g., `DBEQ.MINI`) or mismatched schemas (such as `mbp-10` on `EQUS.MINI`) cause the pipeline and coverage CLI to exit early so ingest jobs cannot run with unsupported Databento settings.

### Market Hours
- **EXTERNAL bars** (OHLCV-1m from Databento) are only available during market hours:
  - Monday-Friday: 9:30 AM - 4:00 PM ET
  - No data outside these hours or on weekends/holidays

### Bar Aggregation Types
- **EXTERNAL** (default): Uses Databento's pre-aggregated OHLCV bars
  - Pros: Lower latency, less CPU usage, official aggregation
  - Cons: Only available during market hours

- **INTERNAL**: Aggregates bars from trades locally
  - Pros: Can aggregate from any data source
  - Cons: Requires instrument definitions, higher CPU usage
  - Note: Currently not configured for EQUS venue

### Port Configuration
- ML Signal Actor: `8000` (metrics/health)
- ML Strategy: `8001` (metrics/health)
- ML Pipeline: `8081` (health)
- PostgreSQL: `5433` (host) → `5432` (container)
- Redis: `6380` (host) → `6379` (container)
- Prometheus: `9090`
- Grafana: `3000` (user: admin, pass: admin)

### Environment Variables

Override defaults using these environment variables:

```bash
# Databento
DATABENTO_API_KEY=your-key-here
DATABENTO_DATASET=EQUS.MINI

# Trading Configuration
INSTRUMENT_ID=SPY.EQUS
BAR_TYPE=SPY.EQUS-1-MINUTE-LAST-EXTERNAL
ACTOR_ID=MLSignalActor-001
UNIVERSE_SYMBOLS=SPY.XNAS,AAPL.XNAS,MSFT.XNAS  # Optional multi‑instrument universe

# Database
DB_CONNECTION=postgresql://postgres:postgres@postgres:5432/nautilus
USE_DUMMY_STORES=false

# Model
MODEL_PATH=/app/models/dummy_bullish_model.onnx

# Ports (to avoid conflicts)
ML_ACTOR_HOST_PORT=8000
ML_STRATEGY_HOST_PORT=8001
ML_PIPELINE_HOST_PORT=8081
POSTGRES_HOST_PORT=5433
REDIS_HOST_PORT=6380

# Multi‑instrument batching (optional)
MAX_BATCH_SIZE=128
FEATURE_DIM=64
FLUSH_MAX_LATENCY_MS=0
```

## Monitoring

### Health Checks
```bash
# Actor health
curl http://localhost:8000/health

# Strategy health
curl http://localhost:8001/health

# Pipeline health
curl http://localhost:8081/health
```
The `/health` response includes a `coverage` block with the last restoration timestamps, bucket counts, and the most recent error (if any).

### Prometheus Metrics
Access metrics at http://localhost:9090

### Grafana Dashboards
Access dashboards at http://localhost:3000
- Username: admin
- Password: admin

## Troubleshooting

### No bars arriving outside market hours
This is expected behavior. EXTERNAL bars from Databento are only available during market hours (9:30 AM - 4:00 PM ET).

### "No instrument found" error with INTERNAL bars
INTERNAL bar aggregation requires instrument definitions. Currently not configured for EQUS venue. Use EXTERNAL bars instead.

### Port conflicts
If ports are already in use, override with environment variables:
```bash
export ML_ACTOR_HOST_PORT=8100
export POSTGRES_HOST_PORT=5432
```

### Container keeps restarting
Check logs for specific errors:
```bash
docker compose -f ml/deployment/docker-compose.yml logs ml_signal_actor
```

Common issues:
- Invalid API key (must be 32 characters)
- Database connection issues
- Model file not found

## Next Steps

1. **Replace the dummy model** with your trained ONNX model in `ml/models/`
2. **Configure the ML Strategy** to trade based on signals
3. **Set up the ML Pipeline** for data collection and feature computation
4. **Monitor performance** through Prometheus/Grafana dashboards
5. **Test during market hours** to see live data flow

## Testing

To verify the setup during market hours:

```bash
# Check if bars are arriving
docker logs ml-ml_signal_actor-1 --tail 50 | grep -E "Bar received|Processing bar"

# Check signal generation
docker logs ml-ml_signal_actor-1 --tail 50 | grep -E "Signal|Prediction"

# Check metrics
curl -s http://localhost:8000/metrics | grep ml_
```

### Off‑Hours / Mock Data Mode

If you need to validate the end‑to‑end stack outside market hours, use the mock data profile. It spins up a separate project (`ml-test`) with synthetic bars and a dedicated metrics port.

```bash
# Bring up the mock stack (postgres-test, redis-test, actor in mock mode)
docker compose -f ml/deployment/docker-compose.test.yml up -d --build

# Tail the actor logs
docker compose -f ml/deployment/docker-compose.test.yml logs -f ml_signal_actor_test

# Check metrics (exposed on 8002)
curl -s http://localhost:8002/metrics | grep ml_

# Tear down when finished
docker compose -f ml/deployment/docker-compose.test.yml down -v
```

Environment knobs (defaults shown):

- `USE_MOCK_DATA=true` enables synthetic bar stream
- `MOCK_DATA_RATE=10` bars/sec, `MOCK_INITIAL_PRICE=650.0`, `MOCK_VOLATILITY=0.002`
- Test DB: `postgres-test` on `5434`; actor metrics on `8002`

Default universe behavior

- If `UNIVERSE_SYMBOLS` is not set, the actor enables a default US‑centric universe:
  - `SPY.EQUS, QQQ.EQUS, AAPL.XNAS, MSFT.XNAS, NVDA.XNAS`
- Set `UNIVERSE_SYMBOLS` (comma‑separated) to override.

## Orchestrator Smoke Test (Docker)

Run a one‑shot MLPipelineOrchestrator inside a container to validate end‑to‑end dataset build → HPO → train → stage‑2 promotion gates (returns engine):

```bash
# Optional: export DATABENTO_API_KEY for ingestion (otherwise builder uses existing catalog)
export DATABENTO_API_KEY=...  # optional

# Build and run the smoke (starts Postgres/Redis via compose if needed)
make docker-orchestrator-smoke

# Inspect outputs under ./data/out_smoke and ml_registry (mounted into the container)
ls -la data/out_smoke
```

Flags used in the smoke run can be adjusted by editing the `docker-orchestrator-smoke` target in the Makefile.

Notes
- Stage 2 `--stage2_engine backtest` attempts to use Nautilus Trader BacktestEngine. If unavailable or parity is incomplete in the container, the orchestrator automatically falls back to the returns‑based engine for stable gating metrics.

## Support

For issues or questions:
- Check the logs first: `docker compose logs ml_signal_actor`
- Verify market hours and trading calendar
- Ensure your Databento API key is valid and has appropriate permissions
- Review the Nautilus Trader documentation for additional configuration options
