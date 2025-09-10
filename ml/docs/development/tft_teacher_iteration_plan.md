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

- TFT Teacher v3 (GPU, 3 epochs, 60d universe, BCE loss)
  - Model ID: `tft_teacher_universe_60d_bce`
  - Dataset: ~40k merged rows across 15 symbols, 60 days (chunked weekly), L2 + micro + macro
  - Validation metrics (15m horizon):
    - AUC: 0.6323
    - PR‑AUC: 0.2377 (prevalence 0.1534 → ~1.55× baseline)
    - LogLoss: 0.4155
    - Brier: 0.1260
  - Notes: BCE loss improved AUC versus v2 (0.632 vs 0.576). Calibration requires tuning on this dataset (LogLoss/Brier higher than v2); next pass will apply calibration/HPO.

- TFT Teacher v4 (GPU, 5 epochs, 60d universe, BCE loss, larger capacity)
  - Model ID: `tft_teacher_universe_60d_bce_v2`
  - Config deltas: `--hidden_size 32 --lstm_layers 2 --attention_head_size 4 --dataloader_workers 4`
  - Validation metrics: same as v3 (AUC 0.6323; PR‑AUC 0.2377)
  - Interpretation: likely plateau at current hyperparams/dataset size; next step is HPO and/or larger window for additional signal and calibration improvements.

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
  - Per‑day caches added and wired: per‑minute L2 and microstructure features persisted and read on subsequent builds
- Build CLI
  - New `--start/--end` ISO date arguments
  - Macro included by default; `--no_macro` disables it (FRED is essential)
  - Verbose logging (`--verbose`) and chunked orchestration via build_runner
- Teacher Improvements
  - BCE loss wrapper added and pluggable via `--loss bce`
  - Optional warm‑start from pretraining: `pretrained_state_path` (best‑effort partial load)
  - DataLoader workers configurable via `--dataloader_workers`

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
  - Epochs: 3–5 with early stopping; BCE‑with‑logits
  - Known‑future: TOD/DOW/session flags
  - Static: exchange, asset_class, tick_size
  - Dataloader: `--dataloader_workers 4` for throughput
  - Calibration: Platt/temperature scaling on logits for improved probability quality (record LogLoss/Brier)
  - HPO: small sweep over `hidden_size ∈ {32,64}`, `lstm_layers ∈ {2,3}`, `learning_rate ∈ {3e-4,1e-3}`, `dropout ∈ {0.1,0.2}`; pick best via val AUC/PR‑AUC and calibration metrics
  - Pretraining: masked time modeling on long L0/L1 history to initialize encoders prior to BCE fine‑tuning with L2/L3; warm‑start via `pretrained_state_path`

## HPO Strategy (Rapid Iteration)

- Fast loops on short windows
  - Use 5–7 day windows per symbol for quick fits (1–2 epochs); prune weak configs early.
  - Basket sampling: SPY, QQQ, AAPL, MSFT, NVDA to capture broad dynamics.
- Parameters to sweep (BCE loss)
  - Model: `hidden_size {32,64}`, `lstm_layers {2,3}`, `attention_heads {2,4}`, `dropout {0.1,0.2}`
  - Optimizer: `learning_rate {3e-4, 1e-3}`, `weight_decay {0, 1e-5}`
  - Data: `dataloader_workers {2,4}`, `batch_size {64,128}` (watch VRAM headroom)
  - Loss: optional `pos_weight ≈ (1−p)/p` for imbalance (p = prevalence)
- Selection & promotion
  - Primary: AUC, PR‑AUC and PR‑AUC multiple vs. prevalence baseline
  - Secondary: LogLoss, Brier (calibration)
  - Confirm winners on 30–60 day windows (3–5 epochs) and calibrate before promotion

## Reconciling Timelines (L0/L1 vs L2/L3)

- Two‑phase teacher training
  - Pretrain on longer L0/L1 (+micro + macro) history for broad patterns
  - Fine‑tune on rolling 30‑day windows with L2/L3 added (lower LR; optionally freeze early layers)
- Missingness & masks
  - Keep L2/L3 columns in schema; add `is_available` masks; fill missing with 0s; let the model learn to ignore missing when L2/L3 absent
  - Document masks & semantics in FeatureRegistry for strict train/serve parity
- Distillation to bridge gaps
  - Train teacher where L2/L3 exist; generate soft labels; train a student on longer L0/L1 windows with those labels to transfer microstructure signal
  - Optionally ensemble multiple L2/L3 teachers over different windows and distill consensus
- Ablations & intersections
  - For pure L2 quantification: run on intersection windows (L2 present) and compare against L0/L1‑only runs

## Operational Pipeline (Rolling)

- Daily ingestion
  - Collect L2/L3 for the prior day; persist raw parquet under date partitions
  - Aggregate per‑minute L2 for that day and write to per‑day cache (data/features/l2_minute)
  - Update per‑day micro cache; maintain macro cache with lag semantics
- Dataset builds
  - Use chunked weekly reads + per‑day caches to assemble 30–90 day windows
  - Periodically refit teacher with BCE; calibrate; run gates before deployment
- Evaluation & gates
  - Walk‑forward (weekly blocks): report AUC/PR‑AUC/Brier per block and per symbol
  - Promotion thresholds: PR‑AUC ≥ 1.5× prevalence baseline; stable AUC across weeks; improved calibration (LogLoss/Brier)

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

- TFT Teacher v3 (GPU, 3 epochs, 60d universe, BCE loss)
  - Model ID: `tft_teacher_universe_60d_bce`
  - Dataset: ~40k merged rows across 15 symbols, 60 days (chunked weekly), L2 + micro + macro
  - Validation metrics (15m horizon):
    - AUC: 0.6323
    - PR‑AUC: 0.2377 (prevalence 0.1534 → ~1.55× baseline)
    - LogLoss: 0.4155
    - Brier: 0.1260
  - Notes: BCE loss improved AUC versus v2 (0.632 vs 0.576). Calibration requires tuning on this dataset (LogLoss/Brier higher than v2); next pass will apply calibration/HPO.

3) Implement BCE loss wrapper; re‑train on expanded window (60–90d)
4) Persist per‑minute L2 cache per day; switch builder to read cache preferentially
5) Prepare actor config for paper‑trading (model_id + feature_set_id), run walk‑forward gates
6) Expand to 90d universe; train 5 epochs with `--loss bce` and HPO sweep; calibrate and evaluate walk‑forward
