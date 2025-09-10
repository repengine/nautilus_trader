# TFT Teacher Iteration Plan (Current State + Next Steps)

## Summary

We validated the full cold-path pipeline (dataset → TFT teacher → calibrated probabilities) and trained two teachers. The second model uses L2 features across 5 tickers and shows clear improvements in discrimination and calibration. The distillation path enforces strict feature parity and shape alignment. Next, we scale chunked builds and caching toward universe-level training.

## Current State (2025‑09‑09)

- Dataset: SPY, minute OHLCV with micro + macro joins
  - Horizon: 15 minutes; Threshold: 0.1%
  - Output artifacts: `/tmp/tft_run/dataset.{parquet,csv}`, `features_npz.npz`
  - FeatureRegistry: `feature_set_1757445061178541` (schema hash recorded)
- TFT Teacher v1 (GPU, 1 epoch, SPY-only)
  - Model ID: `tft_teacher_spy_v1`
  - Artifacts: `/tmp/tft_run/teacher_preds.npz` (q_train, q_val, y_val_true)
  - Validation metrics (SPY, 15m horizon):
    - AUC: 0.5444
    - PR‑AUC: 0.1273 (baseline unknown)
    - LogLoss: 0.3489
    - Brier: 0.0990

- TFT Teacher v2 (GPU, 3 epochs, 5 tickers with L2, 30d chunked)
  - Model ID: `tft_teacher_5x30d_v1`
  - Dataset: SPY, QQQ, AAPL, MSFT, NVDA; last 30d; L2 aggregated in 7‑day chunks and merged (rows ≈ 16,260; val ≈ 3,252)
  - Validation metrics (15m horizon):
    - AUC: 0.5756
    - PR‑AUC: 0.2521 (prevalence 0.094 → ~2.68× baseline)
    - LogLoss: 0.3044
    - Brier: 0.0832
  - Notes: L2 depth features and multi‑instrument training improved discrimination and calibration vs v1.
- Distillation Pipeline (ready)
  - Parity: strict name + order check vs FeatureRegistry manifest
  - Shapes: training split uses `X_train` with `q_train`; option to use validation via `--use_val_for_distill`
  - Metrics: AUC/PR‑AUC/Brier/LogLoss computed for validation and written to the student manifest

## Changes Implemented

- Teacher CLI
  - Emits both `q_train` and `q_val` (aligned to `X_train`/`X_val`) plus `y_val_true`
  - Logistic fallback imputes NaNs (train means) to avoid failures
- TFT Trainer
  - GPU aware, numeric‑only unknown reals, NaN filling, stable single‑output loss
- Distillation CLI
  - Strict FeatureRegistry parity checks; flexible split selection; records validation metrics
- Dataset Builder
  - New date range handling (start/end) and optional filtering in direct path
  - Chunked build mode (`--chunk_days N`) to bound memory
  - L2 scan uses Polars lazy + column projection; added verbose logging of rows/sizes
- Build CLI
  - New `--start/--end` ISO date arguments

## Next Dataset (Target)

- Symbols: SPY, QQQ, AAPL, MSFT, NVDA
- Window: last 30 calendar days
- Features:
  - L2 per‑minute (MBP‑10 aggregation): depth imbalance top‑k, depth‑weighted price deviation, bid/ask slopes
  - L1 microstructure, macro (as‑of with lag), event features (known‑future)
- CLI example:
  - `python -m ml.scripts.build_tft_dataset \\
     --data_dir data/tier1 \\
     --symbols SPY,QQQ,AAPL,MSFT,NVDA \\
     --out_dir /tmp/tft_l2_5x30d \\
     --start YYYY-MM-DD --end YYYY-MM-DD \\
     --horizon_minutes 15 --threshold 0.001 --lookback_periods 60 \\
     --include_micro --include_l2 --include_macro \\
     --register_features --feature_registry_dir ~/.nautilus/ml/features`

## Next Training Run

- TFT Teacher (GPU)
  - Epochs: 3–5 with early stopping
  - Known‑future: TOD/DOW/session flags
  - Static: exchange, asset_class, tick_size
  - Objective: upgrade to BCE-with-logits wrapper (classification‑aligned); current baseline uses stable single‑output PF loss + Platt calibration

## Acceptance Criteria

- Teacher validation improvement over baseline:
  - AUC: ≥ 0.56 (per symbol aggregate)
  - PR‑AUC: ≥ 1.5× prevalence baseline
  - Probability calibration reasonable (LogLoss/Brier improve vs baseline)
- Distilled student achieves similar validation metrics with ONNX export
- End‑to‑end paper‑trading: actor loads student ONNX, metrics scrape functional

## Risks & Mitigations

- Data availability: L2 parquet completeness varies by symbol → best‑effort joins with logging; validate final coverage report
- Training time: 5 symbols × 30 days → short datasets; expected 20–40 minutes/epoch on GTX 1660 Ti; scale workers/mixed precision to improve
- Label sparsity: tune threshold/horizon; consider regression + rank‑based evaluation if classification plateaus

## Owner Actions

1) Build 5×30d dataset with L2 (chunked weekly), register feature set
2) Train TFT teacher (3–5 epochs), record metrics; distill and validate student
3) Implement BCE loss wrapper; re‑train on expanded window (60–90d)
4) Persist per‑minute L2 cache per day; switch builder to read cache preferentially
5) Prepare actor config for paper‑trading (model_id + feature_set_id), run walk‑forward gates
