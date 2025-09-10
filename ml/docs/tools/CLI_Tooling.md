# ML CLI Tooling: Build, Report, Promote

This guide documents the lightweight ML dataset tooling added under `ml/scripts/` and `ml/pipelines/`.

Contents

- Build per-symbol datasets with `build_tft_dataset.py`
- Orchestrate builds with `ml.pipelines.build_runner`
- Generate dataset quality reports with `dataset_report.py`
- Promote feature sets based on metrics with `promote_features.py`

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
