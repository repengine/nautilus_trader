# Observability Quickstart

This guide shows how to enable Unified Observability for the ML layer using the CLI or environment-driven bootstrapping. All observability work stays off the hot path; only cheap metric observations run inside tight loops.

## CLI usage

Flush current observability tables to disk or a database, or start a background flusher:

- Flush to JSONL under a base path:
  - `uv run -m ml.cli.observability flush-jsonl --base-path ./observability --format jsonl --seed-sample`
- Flush to a SQL DB (SQLite/Postgres):
  - `uv run -m ml.cli.observability flush-db --db-url sqlite:///./observability.db --seed-sample`
- Start background flushing (off hot path):
  - `uv run -m ml.cli.observability start --sink db --db-url sqlite:///./observability.db --interval 10 --duration 30 --seed-sample`

See `ml/cli/observability.py` for all options.

## Auto-start in apps (env-driven)

Apps can automatically start background flushing at startup by setting environment variables and calling the bootstrap helper. This keeps code changes minimal and avoids hot-path coupling.

Environment variables (subset):
- `ML_OBS_SINK`: `file` or `db` (default `file`)
- `ML_OBS_BASE_PATH`: base directory for file sinks (default `./observability`)
- `ML_OBS_FILE_FORMAT`: `jsonl` or `csv` (default `jsonl`)
- `ML_OBS_DB_URL`: SQLAlchemy connection string (e.g., `sqlite:///./observability.db`)
- `ML_OBS_INTERVAL_SECONDS`: flush interval seconds (default `60`)

Code snippet (already wired in container entrypoints):

```python
from ml.observability.bootstrap import auto_start_if_configured
from ml.core.integration import MLIntegrationManager

# Create a lightweight integration manager instance (no heavy init)
mgr: MLIntegrationManager = MLIntegrationManager.__new__(MLIntegrationManager)
auto_start_if_configured(mgr)
```

Where this is called
- `ml/deployment/entrypoint_actor.py`
- `ml/deployment/entrypoint_strategy.py`
- `ml/deployment/entrypoint_pipeline.py`

This call is safe and a no-op when observability env variables are not set or misconfigured.

## ObservabilityConfig

`ml/config/observability.py` defines `ObservabilityConfig` and `from_env()` so you can compose configuration explicitly in tests or custom runners. The integration manager exposes convenience methods:
- `start_observability_from_config(cfg)`
- `start_observability_from_env()`
- `start_observability_flush(..., sink=\"file|db\", ...)`

## Notes
- Keep hot-path code free from heavy logic; the service, persistence, and scheduler run off-path.
- All timestamps are in ns; schemas validated by Pandera in contract tests.
- See README “Unified Observability” section for additional examples.
- Example Grafana dashboards: import `ml/deployment/grafana/ml_pipeline_health.json` and point the datasource to your Prometheus instance.
