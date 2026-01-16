# Chronos Training Runbook (Ops)

This runbook documents operational procedures for training time series forecasting models using AutoGluon's Chronos foundation models.

## Overview

Chronos is a family of pretrained transformer models for time series forecasting. This integration provides:

- **Chronos-2**: Best accuracy, 120M params (teacher model)
- **Chronos-Bolt**: 250x faster inference (student model)
- **Forward return regression**: More granular than binary classification
- **Native covariate support**: Calendar, macro, technical indicators

### Why Chronos over PyTorch Forecasting TFT?

| Metric | PyTorch Forecasting TFT | Chronos-2 | Chronos-Bolt |
|--------|-------------------------|-----------|--------------|
| Training (4M rows) | 6+ hours (OOM killed) | ~30 min | ~15 min |
| Memory Usage | 60GB+ (killed) | ~8GB | ~4GB |
| Inference Latency | ~50ms | ~20ms | ~5ms |
| Covariate Support | Manual setup | Native | Native |

## Goal and Plan

### Broad Goal

Train Chronos models that produce reliable, risk-adjusted profits after costs, with reproducible, leakage-free evaluation. We prioritize correctness, out-of-sample validity, and trading viability over raw fit.

### Overall Plan (Phased)

**Phase 0: Data/Label Integrity (complete, small scope)**
- Validate monotonic timestamps, no lookahead, and correct target shift.
- Enforce feature manifest hygiene (numeric-only; no `forward_return`, `y`, or meta fields).
- Contract tests for dataset schema and coverage; run dataset validation gates.

**Phase 1: Baseline Pipeline Fidelity (small scope, SPY.EQUS)**
- Add a time-split evaluation harness (train/val/test by timestamp).
- Train Chronos baseline plus a naive baseline on the same splits.
- Log split boundaries, row counts, and metrics to a compact report.
- Gate: Chronos metrics beat naive baseline and train/val/test gaps are stable.

**Phase 2: Trading Viability Check**
- Map predictions to signals with explicit thresholds and risk limits.
- Run a costed backtest (fees + slippage) on holdout windows.
- Track Sharpe, max drawdown, turnover, and sensitivity to threshold changes.

**Phase 3: Robustness and Scale**
- Walk-forward splits, regime segmentation, and permutation checks to detect noise fitting.
- Expand symbols/time windows once Phase 0–2 criteria hold.
- Re-run evaluation and backtests on the expanded scope.

### Immediate Next Tasks (Phase 1)

- Implement time-split evaluation utilities in `ml/training/autogluon`.
- Produce a baseline evaluation report for SPY.EQUS using parquet fallback.
- Record metrics in `reports/experiments/chronos_v1/evaluation/summary.json`.

## Prerequisites

### Dependencies

```bash
# Install via uv (recommended)
uv pip install "autogluon-timeseries>=1.2.0"

# Or via poetry (may have wheel issues)
poetry add "autogluon-timeseries>=1.2.0"
```

### Environment Variables

```bash
# Required for parquet-only data loading (no Databento subscription)
export ML_TFT_ALLOW_PARQUET_FALLBACK=1

# Optional: GPU configuration
export CUDA_VISIBLE_DEVICES=0  # GPU device ID
export AG_LIGHTNING_DEVICES=1  # Number of GPUs for AutoGluon

# Optional: capture structured logs to a file
export LOG_FILE=reports/experiments/chronos_v1/run.log
```

### Data Requirements

Ensure data directory contains parquet files with:
- `instrument_id`: Symbol identifier
- `ts_event`: Nanosecond timestamp
- `close`: Price for forward return computation
- Optional: `hour`, `dow` for calendar covariates

Note: The AutoGluon adapter canonicalizes `timestamp` to `ts_event` on load for
TFT dataset builder outputs. Treat `ts_event` as the canonical column.

## Quick Start

### Single Model Training

```bash
# Train Chronos-2 teacher (best accuracy)
python -m ml.cli.train_chronos \
    --symbols SPY,AAPL,MSFT \
    --preset chronos2 \
    --time-limit 1800 \
    --horizon 15

# Train Chronos-Bolt (fast inference)
python -m ml.cli.train_chronos \
    --symbols SPY \
    --preset bolt_small \
    --time-limit 600 \
    --cpu-only

# Hyperparameter tuning (AutoGluon HPO)
python -m ml.cli.train_chronos \
    --symbols SPY \
    --preset chronos2 \
    --time-limit 1800 \
    --fine-tune \
    --tune-num-trials 20 \
    --tune-searcher bayes \
    --tune-scheduler local
```

### Time-Split Evaluation (Phase 1)

```bash
ML_TFT_ALLOW_PARQUET_FALLBACK=1 python -m ml.experiments.chronos_training_experiment \
    --symbols SPY \
    --preset bolt_small \
    --time_limit 600 \
    --time_split_eval
```

Outputs an evaluation summary at:
`reports/experiments/chronos_v1/evaluation/summary.json`.

Notes:
- Use `--eval_min_series_rows` to enforce per-series coverage across train/val/test (default: 1).
- Series without full split coverage are dropped from the evaluation report.

### Recent Evaluation Runs

- 2026-01-08: Symbols `TSLA,NVDA,QQQ,SPY,META,AVGO,AMD,MSTR,COIN,MSFT,AAPL,AMZN,NFLX,LLY,PLTR,GOOGL,UNH,IWM,ADBE,TSM,CRWD,MU,COST,GOOG,CRM,GS,TMO,INTC,BA,ORCL,AMAT,CAT,MA,HD,TLT,MRVL,JPM,UBER,ACN` using `data/catalog`, preset `chronos2`, `--time_limit 1800`, and `--eval_min_series_rows 5000`. Market-hours filtered rows 11,511,968 → 3,358,995. Chronos2 failed with CUDA OOM during prediction; no model fit, so Chronos metrics were empty. Baseline RMSE val/test 0.00542 / 0.00444.
- 2026-01-08: Symbols `TSLA,NVDA,QQQ,SPY,META,AVGO,AMD,MSTR,COIN,MSFT,AAPL,AMZN,NFLX,LLY,PLTR,GOOGL,UNH,IWM,ADBE,TSM,CRWD,MU,COST,GOOG,CRM,GS,TMO,INTC,BA,ORCL,AMAT,CAT,MA,HD,TLT,MRVL,BRK.B,JPM,UBER,ACN` using `data/catalog` and `--eval_min_series_rows 5000`. Market-hours filtered rows 11,511,968 → 3,358,995. `BRK.B` was parsed as `BRK` and had no data, so 39 series met split coverage. Baseline RMSE val/test 0.00542 / 0.00444; Chronos RMSE val/test 0.00228 / 0.00210.
- 2026-01-08: Symbols `TSLA,NVDA,QQQ,SPY,META,AVGO,AMD,MSTR,COIN,MSFT,AAPL,AMZN,NFLX,LLY,PLTR,GOOGL,UNH,IWM,ADBE,TSM,CRWD,MU,COST,GOOG,CRM,GS,TMO,INTC,BA,ORCL` using `data/catalog` and `--eval_min_series_rows 5000`. Market-hours filtered rows 9,107,495 → 2,657,451. All 30 series met split coverage. Baseline RMSE val/test 0.00554 / 0.00465; Chronos RMSE val/test 0.00255 / 0.00236.
- 2026-01-08: Symbols `TSLA,NVDA,QQQ,SPY,META,AVGO,AMD,MSTR,COIN,MSFT,AAPL,AMZN,NFLX,LLY,PLTR,GOOGL,UNH,IWM,ADBE,TSM` using `data/catalog` and `--eval_min_series_rows 5000`. Market-hours filtered rows 6,376,140 → 1,860,294. All 20 series met split coverage. Baseline RMSE val/test 0.00549 / 0.00431; Chronos RMSE val/test 0.00226 / 0.00188.
- 2026-01-08: Symbols `SPY,AAPL,MSFT,NVDA,AMZN,GOOG,META,TSLA,JPM,XOM`. Market-hours filtered rows 7,413,076 → 2,162,160. Series coverage filter dropped `AAPL` (GOOG had no data). Baseline RMSE val/test 0.00326 / 0.00495; Chronos RMSE val/test 0.00178 / 0.00233.
- 2026-01-08: Symbols `SPY,AAPL,MSFT,NVDA,AMZN`, market-hours filtered rows 4,291,348 → 1,251,600. Baseline RMSE val/test 0.00269 / 0.00535; Chronos RMSE val/test 0.00134 / 0.00288. AutoGluon no longer flags macro `__value_*` columns as non-informative (adapter de-duplicates redundant macro columns).
- Report: `reports/experiments/chronos_v1/evaluation/summary.json`

### Option 2 E2E Validation Runs

- 2026-01-14: `run_2025-11-28_5sym_2h_v5` (SPY,AAPL,MSFT,AMZN,NVDA), TestClock fast path
  on `data/catalog` with `ML_STRICT_FEATURE_PARITY=1`. ONNX model
  `chronos_option2_distilled_lgbm_v1` produced 485 predictions and 165 strategy
  signals, persisted to `ml_out/pipeline_validation_option2/run_2025-11-28_5sym_2h_v5`.
- 2026-01-14: `run_2025-11-28_5sym_2h_v6` (same scope/window) confirmed model_id
  propagation (`chronos_option2_distilled_lgbm_v1`) in predictions/signals with
  identical counts.
- 2026-01-14: `run_2025-11-28_5sym_2h_v12` (same scope/window) produced 505
  predictions, 175 strategy signals, and 5 order intents at
  `ml_out/pipeline_validation_option2/run_2025-11-28_5sym_2h_v12/orders/order_intents.jsonl`.
  OrderExecutor logged "Invalid market prices" before falling back to market
  orders; actor persistence worker stop still times out (non-fatal).

### Next Steps

- Retry the `chronos2` time-split evaluation on the same 39-series subset with `--cpu_only` (or reduce the subset size) to avoid CUDA OOM during prediction.
- Expand to 50-60 symbols from `data/catalog`, keep coverage gating on, and confirm no series drop from split coverage once chronos2 is stable.
- Review per-series metrics for outliers and confirm market-hours filtering plus macro column de-duplication remain stable at scale.
- If metrics are stable, proceed to teacher-student distillation on the same subset and record teacher vs student deltas in the report.

### Teacher-Student Distillation

```bash
python -m ml.cli.train_chronos \
    --symbols SPY,QQQ,AAPL,MSFT,NVDA \
    --distill \
    --teacher-preset chronos2 \
    --student-preset bolt_small \
    --teacher-time-limit 3600 \
    --student-time-limit 1800
```

#### Distillation Notes

- Distillation uses rolling forecasts to align teacher predictions to the
  forecasted `ts_event` timestamps (horizon step configurable).
- Defaults assume `prediction_length=15`; adjust rolling parameters via
  `ChronosDistillationConfig` when changing horizons.
- When known covariates are enabled (calendar features), the distillation
  pipeline builds future covariate frames for the full forecast horizon.
  If coverage drops due to market gaps, use `--distill-window-strategy contiguous`
  and target `--distill-min-coverage` around 0.3-0.5 for fidelity.

## Fidelity Checklist

- Use `--tune-num-trials` (>=10) with `--num-val-windows` >= 2 and avoid `--skip-model-selection`.
- Ensure `--fine-tune` is enabled for tuning runs (auto-enabled when `--tune-num-trials` is set).
- Enable `--refit-full` for teacher training to reduce overfitting to early windows.
- Capture `run.log` via `LOG_FILE` and keep `models/**/logs/predictor_log.txt` for audit.
- Compare teacher vs. student metrics on a holdout slice; ensure student tracks teacher.

### Using the Experiment Script Directly

```bash
ML_TFT_ALLOW_PARQUET_FALLBACK=1 python -m ml.experiments.chronos_training_experiment \
    --symbols SPY,AAPL \
    --preset chronos2 \
    --time_limit 1800 \
    --out_dir reports/experiments/chronos_v1
```

## CLI Reference

### train_chronos Options

| Flag | Default | Description |
|------|---------|-------------|
| `--symbols` | (required) | Comma-separated symbol list |
| `--preset` | `chronos2` | Model preset: chronos2, bolt_small, bolt_tiny |
| `--time-limit` | 1800 | Training time budget (seconds) |
| `--horizon` | 15 | Forecast horizon (minutes) |
| `--data-dir` | `data/tier1` | Parquet data directory |
| `--output-dir` | auto | Output directory for models |
| `--distill` | false | Enable teacher-student distillation |
| `--cpu-only` | false | Disable GPU acceleration |
| `--no-soft-labels` | false | Skip soft label export |
| `--no-ensemble` | false | Disable ensembling during tuning |
| `--num-val-windows` | 1 | Validation windows for tuning |
| `--refit-every-n-windows` | 1 | Refit cadence for rolling windows |
| `--refit-full` | false | Refit best model on full dataset |
| `--skip-model-selection` | false | Skip model selection/tuning |
| `--fine-tune` | false | Enable Chronos fine-tuning during training |
| `--tune-num-trials` | (unset) | Enable AutoGluon HPO with N trials |
| `--tune-scheduler` | (unset) | AutoGluon HPO scheduler (local, ray) |
| `--tune-searcher` | (unset) | AutoGluon HPO searcher (random, bayes) |
| `--distill-forecast-step` | 1 | Forecast step for soft label alignment |
| `--distill-min-history` | 75 | Minimum history before generating soft labels |
| `--distill-stride` | 15 | Stride between rolling forecast cutoffs |
| `--distill-max-windows` | (unset) | Cap rolling windows per series |
| `--distill-max-series` | (unset) | Cap number of series used for distillation |
| `--distill-sample-fraction` | (unset) | Sample fraction of windows per series |
| `--distill-window-strategy` | (unset) | Window sampling strategy (uniform, contiguous) |
| `--distill-min-coverage` | 0.05 | Minimum soft label coverage threshold |

### Presets

| Preset | Parameters | Use Case | Inference Speed |
|--------|------------|----------|-----------------|
| `chronos2` | 120M | Teacher training | ~20ms |
| `chronos_large` | 710M | Maximum accuracy | ~100ms |
| `bolt_small` | ~20M | Production inference | ~5ms |
| `bolt_tiny` | ~5M | Edge deployment | ~2ms |

## Output Structure

```
reports/experiments/chronos_v1/
├── dataset/
│   ├── dataset.parquet       # Training data
│   └── dataset_metadata.json
├── evaluation/
│   └── summary.json          # Time-split evaluation report
├── models/
│   ├── chronos2/             # Teacher model
│   │   └── predictor.pkl
│   └── bolt_small/           # Student model
│       └── predictor.pkl
└── soft_labels/
    └── soft_labels.parquet  # Teacher soft labels (rolling forecasts)
```

## Programmatic Usage

### Basic Training

```python
from ml.config.autogluon import ChronosTrainingConfig, AutoGluonDataConfig
from ml.training.autogluon.chronos_trainer import ChronosTrainer
from ml.data.autogluon_adapter import compute_forward_return
import polars as pl

# Load and prepare data
df = pl.read_parquet("data/tier1/SPY.parquet")
df = compute_forward_return(df, horizon=15)

# Configure
config = ChronosTrainingConfig(
    prediction_length=15,
    preset="bolt_small",
    time_limit=600,
    data_config=AutoGluonDataConfig(
        known_covariates=("hour", "dow"),
    ),
)

# Train
trainer = ChronosTrainer(config)
result = trainer.train(df)

# Generate predictions
predictions = trainer.predict(df)
```

### Teacher-Student Distillation

```python
from ml.config.autogluon import ChronosDistillationConfig, ChronosTrainingConfig
from ml.training.autogluon.chronos_trainer import train_teacher_student

# Configure teacher and student
teacher_config = ChronosTrainingConfig(preset="chronos2", time_limit=3600)
student_config = ChronosTrainingConfig(preset="bolt_small", time_limit=1800)

distill_config = ChronosDistillationConfig(
    teacher_config=teacher_config,
    student_config=student_config,
    export_soft_labels=True,
)

# Run distillation pipeline
result = train_teacher_student(df, distill_config)

# Access trained models
teacher = result["teacher"]
student = result["student"]
soft_labels = result["soft_labels"]

# Persist student for fast inference
student.persist()
```

### Using ChronosTeacher with Existing Distillation Pipeline

```python
from ml.training.teacher.chronos_teacher import ChronosTeacher, ChronosTeacherConfig

config = ChronosTeacherConfig(
    preset="chronos2",
    prediction_length=15,
    time_limit=1800,
)

teacher = ChronosTeacher(config)
teacher.fit(dataset)

# Generate soft labels for LightGBM student
soft_labels = teacher.get_soft_labels(dataset, temperature=1.0)
```

## Troubleshooting

### Common Issues

#### 1. OOM (Out of Memory)

**Symptoms**: Process killed, CUDA out of memory errors

**Solutions**:
- Use smaller preset: `--preset bolt_tiny`
- Reduce time limit: `--time-limit 300`
- Enable CPU-only: `--cpu-only`
- Subsample data before training

#### 2. Dataset Build Fails

**Symptoms**: "No parquet files found", vintage validation errors

**Solutions**:
```bash
# Ensure parquet fallback is enabled
export ML_TFT_ALLOW_PARQUET_FALLBACK=1

# Check data directory has parquet files
ls data/tier1/*.parquet
```

#### 3. AutoGluon Import Errors

**Symptoms**: `ModuleNotFoundError: No module named 'autogluon'`

**Solutions**:
```bash
# Reinstall with uv
uv pip install --force-reinstall "autogluon-timeseries>=1.2.0"

# Verify installation
python -c "from autogluon.timeseries import TimeSeriesPredictor; print('OK')"
```

#### 4. Slow Training

**Symptoms**: Training takes longer than expected

**Solutions**:
- Enable GPU: Remove `--cpu-only` flag
- Use faster preset: `--preset bolt_small`
- Reduce time limit for initial testing
- Check GPU utilization: `nvidia-smi`

#### 5. Training Appears Idle or Exits When Backgrounded

**Symptoms**: Log stops after "Using LoRA..." or process disappears when run with `&`/`nohup`

**Solutions**:
- Run in the foreground or inside `tmux`/`screen` for long jobs
- Avoid piping through `tee` unless you also capture stderr and handle `SIGPIPE`
- Keep `PYTHONUNBUFFERED=1` so logs flush promptly

### Performance Tuning

#### GPU Optimization

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python -m ml.cli.train_chronos --symbols SPY --preset chronos2

# Multi-GPU (experimental)
python -m ml.cli.train_chronos --symbols SPY --num-gpus 2
```

#### Memory Optimization

```python
# Persist models in memory for faster inference
trainer.persist()

# For large datasets, subsample during development
df_sample = df.sample(fraction=0.1, seed=42)
```

## Monitoring

### Metrics

Key metrics emitted during training:

| Metric | Description |
|--------|-------------|
| `ml_chronos_training_time_seconds` | Total training duration |
| `ml_chronos_rows_processed` | Number of training rows |
| `ml_chronos_eval_rmse` | Validation RMSE |
| `ml_chronos_inference_latency_ms` | Prediction latency |

### Logs

Training logs are written to:
- Console (stdout)
- AutoGluon logs: `AutogluonModels/` directory

### Health Checks

```python
# Check trainer status
info = trainer.get_model_info()
print(info["status"])  # "fitted" or "not_fitted"
print(info["training_metrics"])
```

## Model Registry Integration

### Register Trained Model

```python
from ml.registry import ModelRegistry, ModelManifest, ModelRole

manifest = ModelManifest(
    model_id="chronos_bolt_v1",
    role=ModelRole.STUDENT,
    architecture="Chronos-Bolt",
    performance_metrics=result["metrics"],
    deployment_constraints={
        "max_inference_latency_ms": 10.0,
        "memory_limit_mb": 2048.0,
    },
)

registry = ModelRegistry()
registry.register_model(trainer.save_path, manifest)
```

## Migration from TFT

If migrating from PyTorch Forecasting TFT:

1. **Same dataset format**: Chronos uses the same parquet input
2. **Similar config structure**: Horizon, lookback periods map directly
3. **Target change**: Binary `y` → continuous `forward_return`
4. **Faster iteration**: 30 min vs 6+ hours for 4M rows

```python
# TFT-style config (old)
tft_config = TFTDatasetTaskConfig(horizon_minutes=15, threshold=0.002)

# Chronos config (new)
chronos_config = ChronosTrainingConfig(
    prediction_length=15,
    target_column="forward_return",  # Regression target
)
```

## Appendix

### Supported Eval Metrics

- `RMSE`: Root Mean Squared Error (default)
- `MAE`: Mean Absolute Error
- `MAPE`: Mean Absolute Percentage Error
- `MASE`: Mean Absolute Scaled Error
- `SMAPE`/`sMAPE`: Symmetric MAPE
- `WAPE`: Weighted Absolute Percentage Error

### File Locations

| Component | Path |
|-----------|------|
| Config | `ml/config/autogluon.py` |
| Trainer | `ml/training/autogluon/chronos_trainer.py` |
| Teacher | `ml/training/teacher/chronos_teacher.py` |
| Data Adapter | `ml/data/autogluon_adapter.py` |
| CLI | `ml/cli/train_chronos.py` |
| Experiment | `ml/experiments/chronos_training_experiment.py` |
| Tests | `ml/tests/unit/training/autogluon/` |
