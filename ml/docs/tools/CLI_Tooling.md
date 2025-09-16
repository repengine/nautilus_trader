# ML CLI Tooling: Build, Report, Promote

This guide documents the lightweight ML dataset tooling added under `ml/scripts/` and `ml/pipelines/`.

Contents

- Build per-symbol datasets with `build_tft_dataset.py`
- Orchestrate builds with `ml.pipelines.build_runner`
- Generate dataset quality reports with `dataset_report.py`
- Promote feature sets based on metrics with `promote_features.py`
- Run end-to-end pipeline with `ml.cli.pipeline_orchestrator`
- Schedule pipeline runs with `ml.cli.pipeline_scheduler`

## Databento Guards (US Equities — Standard)

To prevent accidental API requests outside your subscription coverage, this repo enables guardrails by default when you activate the venv. They enforce dataset/schema allowlists, clamp time ranges, and cap symbols.

Defaults (can be overridden via env):

- Dataset: `EQUS.MINI`
- Schemas: `ohlcv-1m,tbbo,bbo,trades,mbp-1,mbp-10,mbo,imbalance`
- Max days by schema: `ohlcv-1m:36500,tbbo:365,bbo:365,trades:365,mbp-1:365,mbp-10:31,mbo:31,imbalance:31`
- Max symbols: `120`; Strict mode: `1`

Usage:

- Activate: `source .venv/bin/activate` (auto-loads `env/databento_standard.sh`) or `source scripts/activate_ml.sh`.
- Adjust: edit `env/databento_standard.sh` or export overrides before running CLIs.

CLIs with built-in guard enforcement:

- `ml/cli/backfill_ohlcv_recent.py` (minute bars)
- `ml/data/collector.py` (L2 MBP-1, L1 trades/TBBO, minute bars)
- `ml/data/ingest/databento_adapter.py` (core adapter)

## 1) Build a Dataset (Single-Run)

Create a TFT-style training dataset per symbol with optional macro joins and lookback/target params.

Example:

```bash
python -m ml.scripts.build_tft_dataset \
  --data_dir data/tier1 \
  --symbols SPY,QQQ \
  --out_dir /tmp/tft_ds \
  --include_macro --macro_lag_days 1 \
  --horizon_minutes 15 --threshold 0.001 --lookback_periods 30
```

Outputs in `out_dir`:

- `dataset.parquet` and `dataset.csv`
- `features_npz.npz` (X_train/X_val/features)
- Optional sidecar `feature_set.json` if registration is enabled in other flows

Notes:

- If ParquetDataCatalog lookup doesn’t find bars, the builder falls back to reading `data/tier1/<SYMBOL>/ohlcv-1m_{historical,recent}.parquet` and normalizes timestamps.
- Macro features are joined via as-of semantics with a publication lag to prevent leakage.

## 2) Build Runner (Orchestrate Multiple Symbols)

Use the runner to execute per-symbol builds from a JSON/TOML config. Supports a subprocess mode for clean env isolation and a simple parallel path.

Config example: `ml/config/build_runner_example.json`

```json
{
  "data_dir": "data/tier1",
  "out_dir": "/tmp/tft_ds_small",
  "symbols": ["SPY", "QQQ"],
  "include_macro": true,
  "macro_lag_days": 1,
  "include_micro": false,
  "include_l2": false,
  "horizon_minutes": 15,
  "threshold": 0.001,
  "lookback_periods": 30,
  "workers": 1,
  "use_subprocess": true
}
```

Run:

```bash
python -m ml.pipelines.build_runner --config ml/config/build_runner_example.json
# Or via Make:
make ml-build-runner CONFIG=ml/config/build_runner_example.json
```

Results:

- Per-symbol outputs under `out_dir/<SYMBOL>/…`
- Progress log: `out_dir/progress.jsonl`
- Prometheus metrics (if enabled):
  - `nautilus_ml_build_runner_runs_total{status}`
  - `nautilus_ml_build_runner_task_duration_seconds{symbol}`

## 3) Dataset Quality Report

Summarize macro null-rates, feature coverage, and target distribution.

```bash
python -m ml.scripts.dataset_report \
  --dataset /tmp/tft_ds_small/SPY/dataset.parquet \
  --out_json /tmp/tft_ds_small/SPY/report.json \
  --out_md /tmp/tft_ds_small/SPY/report.md
# Or via Make:
make ml-dataset-report DATASET=/tmp/tft_ds_small/SPY/dataset.parquet \
  OUT_JSON=/tmp/tft_ds_small/SPY/report.json OUT_MD=/tmp/tft_ds_small/SPY/report.md
```

Outputs:

- JSON report for automation, plus optional Markdown summary.

## 4) Promote Feature Sets via Quality Gates

Use `promote_features.py` to update `perf_digest` and validate against gates. Promotion moves the feature set to `PROD` when all required gates pass.

Inline gates example:

```bash
python -m ml.scripts.promote_features \
  --feature_registry_dir ~/.nautilus/ml/features \
  --feature_set_id feature_set_123 \
  --metrics_json /tmp/metrics.json \
  --gate pr_auc gte 0.70 required \
  --gate logloss lte 0.60 required
```

Gates from file (`ml/config/promotion_gates_example.json`):

```bash
python -m ml.scripts.promote_features \
  --feature_registry_dir ~/.nautilus/ml/features \
  --feature_set_id feature_set_123 \
  --metrics_json /tmp/metrics.json \
  --gates_json ml/config/promotion_gates_example.json

# Or via Make:
make ml-promote-features FEATURE_REGISTRY_DIR=~/.nautilus/ml/features \
  FEATURE_SET_ID=feature_set_123 METRICS_JSON=/tmp/metrics.json \
  GATES_JSON=ml/config/promotion_gates_example.json
```

Exit code is nonzero on gate failure; printed JSON includes `{ "promoted": bool, "stage": "prod|…" }`.

## 5) Quickstart: Build + Report (SPY, QQQ)

```bash
python -m ml.pipelines.build_runner --config ml/config/build_runner_example.json
python -m ml.scripts.dataset_report --dataset /tmp/tft_ds_small/SPY/dataset.parquet --out_json /tmp/tft_ds_small/SPY/report.json --out_md /tmp/tft_ds_small/SPY/report.md
python -m ml.scripts.dataset_report --dataset /tmp/tft_ds_small/QQQ/dataset.parquet --out_json /tmp/tft_ds_small/QQQ/report.json --out_md /tmp/tft_ds_small/QQQ/report.md
```

You should see macro columns (e.g., `DGS10`, `VIXCLS`) in the datasets and a target distribution summary in the markdown files.

## 6) End-to-End Pipeline Orchestrator (Cold Path)

Run ingestion + dataset build + optional HPO + teacher training in a single command. All heavy work remains off hot paths.

```bash
python -m ml.cli.pipeline_orchestrator \
  --ingest \
  --dataset_id EQUS.MINI --schema bars --instruments SPY.NYSE --lookback_days 7 \
  --coverage_mode catalog --catalog_path ./data/catalog \
  --write_mode parquet \
  --data_dir data/tier1 --symbols SPY.NYSE --out_dir ml_out \
  --include_micro --horizon_minutes 15 --threshold 0.001 --lookback_periods 30 \
  --hpo --hpo_epochs 2 --hpo_batch_size 32 --hpo_tail_rows 5000 --hpo_limit_groups 50 \
  --train --teacher_model_id teacher_model --max_epochs 5
```

Notes:

- Ingestion:
  - `--coverage_mode catalog` uses Parquet catalog coverage (requires `--catalog_path`).
  - `--coverage_mode sql` uses SQL coverage (requires `--db`).
  - `--write_mode parquet` writes raw data to Parquet; `--write_mode datastore` uses `DataStore` (with adapters when `CATALOG_PATH` is set via `MLIntegrationManager`).
  - Set `DATABENTO_API_KEY` to enable Databento API; when missing, ingestion is skipped.

- Dataset build, HPO, and training reuse in-process CLIs and respect all repo standards (typed, strict, and off hot paths).

Bootstrap and Partitions (recommended)

- For production-like runs, initialize the ML stack via MLIntegrationManager so migrations and partitions are handled off the hot path. Example:

```python
from ml.core.integration import MLIntegrationManager

mgr = MLIntegrationManager(
    auto_start_postgres=False,
    auto_migrate=True,
    ensure_healthy=True,
)

# Optional scheduled maintenance
if mgr.partition_manager is not None:
    mgr.partition_manager.run_maintenance()
```

- One-time conversion (legacy DBs): if store tables were created without partitioning, convert them once:

```bash
make db-convert-stores-to-partitioned DATABASE_URL=postgresql://postgres:postgres@localhost:5432/nautilus AHEAD=3
```

This creates partitioned parents and current/future-month partitions for `ml_feature_values`, `ml_model_predictions`, and `ml_strategy_signals`.

## 7) Pipeline Scheduler (Cold Path)

Run the orchestrator on a fixed UTC schedule or interval. Uses a file lock to avoid overlapping runs and emits SUCCESS/FAILED events via the DataRegistry.

Flags and env (env is used as fallback):

- `--schedule-time HH:MMZ` or `--interval-min N` (env: `ORCH_SCHEDULE_TIME`, `ORCH_INTERVAL_MIN`)
- `--config <path>` to a JSON/TOML orchestrator config (env: `ORCH_CONFIG`)
- `--dry-run` to log actions only (env: `ORCH_DRY_RUN=1`)
- `--force` to ignore existing outputs (env: `ORCH_FORCE=1`)
- Lock path/TTL via env: `ORCH_LOCK_PATH`, `ORCH_LOCK_TTL_HOURS` (default 12)

Examples:

```bash
# Run daily at 02:30Z from env
ORCH_SCHEDULE_TIME=02:30Z ORCH_CONFIG=ml/config/pipeline.toml \
  make ml-pipeline-scheduler

# Or every 1440 minutes (24h) with flags and a dry run
make ml-pipeline-scheduler INTERVAL_MIN=1440 ORCH_CONFIG=ml/config/pipeline.toml DRY_RUN=1
```

See the detailed Orchestration Runbook for environment variables, lock behavior, and example invocations: `ml/docs/tools/ORCHESTRATION_RUNBOOK.md`.
