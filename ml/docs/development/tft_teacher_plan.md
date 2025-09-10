# TFT Teacher Plan — Results, Metrics, Gates, and Roadmap

This is the single source of truth for the TFT teacher: current state, performance targets, commands to reproduce, and the promotion gates we must clear before distillation.

## 1) Overview

Goal: Train a high‑fidelity Temporal Fusion Transformer (TFT) teacher on Databento L0/L1 + rolling L2/L3 with FRED macro. Preserve train/serve parity, avoid leakage, and support daily incremental updates. Distillation is deferred until the teacher clears gates.

Completed

- Macro (FRED) joins enabled by default with publication lag.
- Per‑day caches for L2 and micro (per‑minute); chunked weekly builds.
- Build runner orchestration from JSON/TOML; stable on 60‑day windows.
- BCE loss option for TFT; dataloader workers; gradient clipping; calibration in CLI.
- MTM pretraining scaffold to warm‑start encoders.

## 2) Data & Availability

- L0/L1: long coverage for pretraining and long windows (60–90d+).
- L2/L3: rolling ~30 days; present on recent rows; per‑day per‑minute caches.
- Macro: enabled by default, joined as‑of with lag; no leakage.
- Known‑future time features and static covariates included.
- Action: add `is_l2_available` and `is_macro_available` masks; keep depth columns with 0‑fill so the model learns availability explicitly.

## 3) Current Results (15‑min horizon)

- v1 (SPY‑only, 1 epoch, Poisson): AUC 0.5444; PR‑AUC 0.1273; LogLoss 0.3489; Brier 0.0990.
- v2 (5 tickers, 30d L2, 3 epochs, Poisson): AUC 0.5756; PR‑AUC 0.2521; LogLoss 0.3044; Brier 0.0832.
- v3 (15 symbols, 60d, L2+micro+macro, 3 epochs, BCE): AUC 0.6323; PR‑AUC 0.2377 (prev 0.1534 → ~1.55×); LogLoss 0.4155; Brier 0.1260.
- v4 (same data, 5 epochs, BCE): plateaued near v3. Conclusion: need masks, calibration, longer windows, and LR HPO.

## 4) Pipeline Architecture

- Builder: ISO `--start/--end`; `--chunk_days`; reads per‑day caches under `data/features/{l2_minute|micro_minute}/<SYMBOL>/year=YYYY/month=MM/day=DD.parquet`.
- Build runner: `ml/pipelines/build_runner.py` orchestrates per‑symbol, weekly chunked builds.
- Teacher CLI: `--loss bce`, workers, warm‑start option; post‑training Platt calibration.
- Pretraining: `ml/training/teacher/pretrain_mtm.py` (GRU autoencoder for MTM) produces state dict for warm‑start.

## 5) Training Protocol

Two‑phase

- Pretrain encoders (MTM) on long L0/L1 (+micro + macro); warm‑start TFT.
- Fine‑tune with BCE on rolling L2/L3 intersection (~30d) at lower LR (optionally freeze early layers).

Mixed availability

- Keep L2/L3 columns present; add masks; 0‑fill numerics; let the model learn when to use depth/macro.

Evaluation & calibration

- Time‑based splits only; weekly walk‑forward; apply Platt/temperature scaling; require LogLoss/Brier improvements.

## 6) HPO (fast → deep)

- Short windows (5–7d) on SPY/QQQ/AAPL/MSFT/NVDA, 1–2 epochs: prune `{hidden_size: 32,64} × {lstm_layers: 2,3} × {attention: 2,4} × {dropout: 0.1,0.2}`.
- Promote winners to 60–90d with 3–5 epochs; add LR `{3e‑4, 1e‑3}`; calibrate and apply gates.
- Consider `pos_weight≈(1−p)/p` if prevalence is small.

## 7) Commands (reference)

Build 60d universe (L2+micro+macro; weekly chunks)

```bash
python -m ml.pipelines.build_runner --config ml/config/build_universe_60d.json
```

Merge per‑symbol datasets

```bash
python - <<'PY'
import polars as pl; from pathlib import Path
root=Path('/tmp/tft_universe_60d'); frames=[pl.read_parquet(str(p/'dataset.parquet')) for p in root.iterdir() if (p/'dataset.parquet').exists()]
df=pl.concat(frames, how='vertical');
if 'instrument_id' in df.columns and 'time_index' in df.columns:
    df=df.sort(['instrument_id','time_index'])
out=root/'merged'; out.mkdir(parents=True, exist_ok=True)
df.write_parquet(str(out/'dataset.parquet')); df.write_csv(str(out/'dataset.csv'))
print('Merged rows:', len(df))
PY
```

Train BCE teacher (3–5 epochs)

```bash
FID=$(python -c 'import json,os;d=json.load(open(os.path.expanduser("~/.nautilus/ml/features/feature_registry.json")));print(max(d["features"].items(), key=lambda kv: kv[1]["manifest"]["created_at"])[0])')
python -m ml.training.teacher.tft_cli \
  --train_data_csv /tmp/tft_universe_60d/merged/dataset.csv \
  --out_dir /tmp/tft_universe_60d/merged \
  --model_id tft_teacher_universe_60d_bce \
  --feature_registry_dir "$HOME/.nautilus/ml/features" \
  --feature_set_id "$FID" \
  --max_epochs 3 \
  --loss bce \
  --dataloader_workers 4 \
  --static_categoricals asset_class,exchange \
  --static_reals tick_size \
  --known_future_reals tod_sin,tod_cos,dow_sin,dow_cos,is_market_open,is_premarket,is_aftermarket,hour,minute,dow
```

HPO (BCE, small grid)

```bash
python -m ml.scripts.hpo_tft \
  --dataset_csv /tmp/tft_universe_60d/merged/dataset.csv \
  --out_dir /tmp/tft_universe_60d/hpo \
  --feature_registry_dir ~/.nautilus/ml/features \
  --feature_set_id "$FID" \
  --epochs 2 \
  --workers 4
```

Promotion gate (example)

```bash
python -m ml.scripts.promote_model_if_metrics_pass \
  --teacher_npz /tmp/tft_universe_60d/merged/teacher_preds.npz \
  --min_auc 0.60 \
  --min_pr_auc_multiple 1.5
```

Pretraining (MTM) warm‑start (programmatic)

```python
from ml.training.teacher.pretrain_mtm import PretrainConfig, MTMPretrainer
import numpy as np
X = np.random.randn(10000, 30, 64).astype(np.float32)
state_path = MTMPretrainer(PretrainConfig(input_dim=64, hidden_dim=64, seq_len=30, epochs=2)).fit_and_save(X, out_dir="/tmp/mtm")
# Pass with --pretrained_state_path /tmp/mtm/pretrained_state.pt
```

## 8) Performance Metrics & Promotion Gates

Predictive quality (offline)

- AUC ≥ 0.62 and PR‑AUC ≥ 1.5× prevalence baseline (per week and overall).
- Calibration improves (LogLoss/Brier down after Platt/temperature scaling); ECE ≤ 0.02.
- Precision@k ≥ 1.5× baseline at the daily trade budget; stability within ±10% across weeks/instruments.

Conversion to trades

- Thresholds chosen by expected net value E[p·μ+ − (1−p)·μ− − cost].
- Markout positively monotone at 1/5/15/30 minutes (15‑min primary) for predicted positives.
- Realized spread positive at 15‑min; hit rate ≥ 55% after costs on positives.

Strategy performance (cost‑aware backtest)

- Net Sharpe ≥ 1.0; t‑stat ≥ 2.0; max drawdown ≤ 10% of annualized return; Calmar ≥ 0.7.
- Realized slippage/fees ≤ 1.2× model; ≥ 70% edge retained at target notional/ADV.

Risk/capacity diagnostics

- Regime robustness (VIX, earnings) with no catastrophic flips.
- Balanced symbol contributions (top‑3 < 50% PnL), drift monitoring, exposure control within policy bands.

Promotion gates (teacher‑first)

- Weekly walk‑forward over 90d passes offline gates above.
- Purged/embargoed, cost‑aware backtest passes strategy gates above.
- Only then lock the teacher and consider distillation.

Online monitoring (post‑deploy)

- PnL, Sharpe(rolling), drawdown, hit rate; realized slippage vs model; fill/cancel ratios.
- Online calibration (ECE/Brier) per instrument/time‑of‑day; drift and mask coverage.
- Safeguards: kill‑switch thresholds and auto‑scale down on impact breaches.

## 9) Roadmap (90 days → production)

1) Masks & calibration
   - Add `is_l2_available`, `is_macro_available`; retrain BCE 3–5 epochs; calibrate; re‑score LogLoss/Brier and gates.
2) 90‑day universe + HPO
   - `days_back=90`, `chunk_days=7`; HPO including LR; select winner via PRx → AUC.
3) Pretrain → fine‑tune
   - MTM on long L0/L1 (+micro + macro); warm‑start BCE on 30d L2/L3; calibrate.
4) Validation hardening
   - Weekly walk‑forward over 90d; per‑symbol and aggregate tables; raise gates if baselines improve.
5) Freeze teacher
   - Lock config/artifacts and only then consider student distillation.

## 10) Dependency Notes

- Verified stack for training: `pytorch-forecasting==1.4.0`, `pytorch-lightning==2.5.4`, `lightning==2.5.4`, `torchmetrics==1.8.x`.
- NumPy 2.x works with the above; callbacks/checkpointing minimized in our usage to avoid legacy `np.Inf` issues. If using mlflow, pin `pyarrow<20`.
