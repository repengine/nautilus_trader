# Full Dataset Build Plan (Databento + FRED)

This document outlines the plan to build, validate, and register a full ML training dataset from Databento (L1/L2) and FRED macro series, with clear deliverables, phased tasks, and acceptance criteria.

## Summary

- Build minute‑granularity features from L1 trades/quotes and L2 MBP‑10.
- Join FRED macro data with point‑in‑time (as‑of) semantics and publication lag.
- Emit targets (e.g., forward returns 15m with configurable threshold).
- Export feature manifest to the FeatureRegistry and link to teacher/student models.
- Provide reproducible pipeline (CLI + sidecars + manifests) and quality gates.

## Deliverables

- [x] Per‑minute L1 microstructure features (midprice, spread_bps, imbalances, realized_vol).
- [x] Per‑minute L2 features (depth imbalances top‑k, depth‑weighted price deviation, bid/ask slopes).
- [x] FRED as‑of join utility (Polars/Pandas) with configurable lag.
- [x] Dataset builder integration flags: `include_macro`, `include_micro`, `include_l2`, `include_events`.
- [x] Dataset sidecar: `feature_set.json` (feature_set_id, registry dir, names, flags).
- [x] Feature manifest export (FeatureRegistry) with schema hash + pipeline signature.
- [x] End‑to‑end pipeline CLI (build → teacher → distill) reading sidecar.
- [x] Unit, integration, property, and metamorphic tests per `ml/tests/docs/TESTING_STRATEGY.md`.
- [ ] Full symbol/time build job (bars + L1 + macro + events; L2 on prioritized tickers/time windows).
- [ ] Quality gates + promotion CLI for features based on evaluation metrics.
- [ ] Runtime/coverage report: macro null‑rates, feature coverage per symbol, target stats (via `dataset_report` CLI).

## Phases & Tasks

### Phase 1 — Baseline Dataset (Bars + L1 Micro + Macro + Events)

- [x] Scan OHLCV minute Parquet via `ParquetDataCatalog` per symbol.
- [x] Compute L1 micro features via Polars `group_by_dynamic('1m')`.
- [x] Join macro via FRED as‑of with publication lag; forward‑fill to next release.
- [x] Generate targets (forward return, horizon/threshold). Prevent leakage.
- [x] Add known‑future time features (TOD, DOW, sessions) explicitly.
- [x] Integrate optional Event features (known‑future) via provider.
- [x] Persist dataset (partitioned) + dataset‐level sidecar.

### Phase 2 — L2 Feature Integration

- [x] Implement L2 per‑minute aggregation (MBP‑10): depth imbalance top‑k, depth‑weighted price deviation, bid/ask slopes.
- [x] Integrate L2 features guarded by `include_l2` (best‑effort join).
- [ ] Prioritize symbols/time windows (e.g., SPY/QQQ/AAPL last 30–90 days) to validate I/O and runtime.
- [ ] Roll out to remaining symbols after runtime validation.

### Phase 3 — Validation & Quality Gates

- [x] Property/metamorphic tests:
  - FRED lag monotonicity (nulls do not decrease with added lag).
  - L2 invariance to size scaling.
  - Builder time index monotonic (0..n‑1).
- [x] Contracts: timestamp monotonicity; price/size sanity; macro join null‑rates measured.
- [ ] Quality gates & promotion CLI:
  - [ ] Compute PR‑AUC/logloss; write to FeatureRegistry `perf_digest`.
  - [ ] Validate against gates (e.g., PR‑AUC ≥ 0.7, logloss ≤ 0.6). Promote to PROD.
  - [ ] Generate dataset quality report and attach to registry entry (as artifact) for audit.

### Phase 4 — Manifests, Sidecars, and Registry Integration

- [x] Export feature manifest: names, dtypes, schema hash, pipeline signature, capability flags.
- [x] Dataset sidecar feature_set.json for pipeline handoff (feature_set_id + registry dir).
- [x] Pipeline auto‑reads sidecar to pass registry args to teacher/student CLIs.
- [ ] Add dataset manifest (columns, dtypes, partitions, retention) to DataRegistry (optional).

### Phase 5 — Pipeline & Orchestration

- [x] Dataset build CLI (`ml/scripts/build_tft_dataset.py`).
- [x] End‑to‑end pipeline CLI (`ml/pipelines/tft_train_distill.py`).
- [ ] Add job runner for distributed symbol‑level parallelism (optional: Ray/Dask) under config.
- [ ] Daily incremental scheduler (baseline implemented) to extend the dataset and refresh macro/events.

### Phase 6 — Performance & Infra

- [ ] NVMe storage, ensure >300 GB free for intermediates + outputs.
- [ ] Configure Polars threading; symbol‑level parallelism within memory budget.
- [ ] Partition outputs by symbol and time (e.g., year/month/day) for efficient scans.

### Phase 7 — Monitoring & Observability

- [ ] Prometheus counters and timers for collection, feature calc, writes; error counters.
- [ ] Write minimal JSONL logs for per‑symbol progress and quality stats.

### Phase 8 — CI/CD Gates for ML Subset

- [x] mypy (`mypy ml --strict`) must be clean.
- [x] ruff must be clean for new/changed code (legacy file warnings excluded).
- [x] pytest ML suite with unit/integration/property/metamorphic tests must pass.
- [ ] Add CI job that scopes to `ml/` and enforces coverage ≥ 90% (targeted).

### Phase 9 — Documentation & Runbooks

- [ ] Update user docs with run commands, flags, and manifests.
- [ ] Troubleshooting guide for lag joins, L2 schema mismatches, and performance tuning.

## Testing Matrix

- [x] Unit: L1/L2 aggregators, FRED join, builder integration, manifest export, event provider.
- [x] Integration: builder with events; pipeline sidecar + feature registration.
- [x] Property: FRED lag monotonicity; builder time index monotonicity (Hypothesis).
- [x] Metamorphic: L2 invariance under size scaling; metrics invariants (ROC AUC transform invariance; logloss complement symmetry).

## Acceptance Criteria

- [ ] Dataset builds successfully for full Tier‑1 over target period with acceptable runtime/storage.
- [ ] Feature manifest exported and registered; pipeline signatures reproducible.
- [ ] Macro joins obey publication lag; null‑rates documented per series.
- [ ] Contracts satisfied (timestamp monotonicity; no leakage; price/size sanity; L2 ordering).
- [ ] Quality gates pass and feature set promoted to PROD.

## Resource Estimates

- CPU: 8–32 cores; memory: 32–64 GB; NVMe storage.
- Baseline build (bars + L1 + macro + events) for 70–80 symbols over 6–12 months: multi‑hour to ~1 day depending on parallelism and I/O.
- L2 feature aggregation: heavier; validate on top symbols/time windows first.

## Risks & Mitigations

- I/O bottlenecks: use NVMe; partition reads/writes; streaming scans.
- Memory pressure: process per symbol/day; cap workers; avoid wide joins.
- Timezone & PIT issues: enforce ns UTC; hypothesis tests for join properties.
- L2 schema drift: assert column presence; guard joins with best‑effort; log missing.

## KPIs

- Build runtime per symbol/day; total runtime.
- Macro null‑rates per series; coverage.
- Target distribution and stability across periods/symbols.
- Feature parity across teacher/student manifests (schema hash equality).

## Run Commands (Examples)

- Build + Register:
  - `python -m ml.scripts.build_tft_dataset --data_dir data/tier1 --symbols SPY,QQQ,AAPL \\
    --out_dir /tmp/tft_ds --include_macro --macro_lag_days 1 --include_micro --include_l2 \\
    --horizon_minutes 15 --threshold 0.001 --lookback_periods 60 \\
    --register_features --feature_registry_dir ~/.nautilus/ml/features --feature_role teacher`
- Pipeline (reads sidecar):
  - `python -m ml.pipelines.tft_train_distill --data_dir data/tier1 --symbols SPY,QQQ,AAPL \\
    --out_dir /tmp/tft_ds --include_macro --include_micro --include_l2 \\
    --horizon_minutes 15 --threshold 0.001 --lookback_periods 60 \\
    --train_teacher --teacher_model_id tft_teacher_v1 \\
    --model_registry_dir ~/.nautilus/ml/models --student_model_id lgb_student_v1`
- Evaluate predictions:
  - `python -m ml.scripts.evaluate_predictions --preds /path/to/preds.npz`

- Dataset report:
  - `python -m ml.scripts.dataset_report --dataset /tmp/tft_ds/dataset.parquet --out_json /tmp/tft_ds/report.json --out_md /tmp/tft_ds/report.md`

- Orchestrate per‑symbol builds from config:
  - `python -m ml.pipelines.build_runner --config ml/config/pipeline_config_example.json`

---

Maintainers: please update the checklists as tasks complete. This plan is designed to be executed incrementally: start with the baseline (Bars + L1 + Macro + Events), validate quality and runtime, then expand L2 coverage.
