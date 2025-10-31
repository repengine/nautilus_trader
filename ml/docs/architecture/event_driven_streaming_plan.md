## Event-Driven Streaming Training Architecture

### Overview

- Purpose: transition TFT streaming pipeline from single-process CLI to event-driven services.
- Scope: dataset service, bounded training workers, orchestration layer, shared message contracts, telemetry.
- Constraints: hot-path isolation, strict typing (see AGENTS.md), centralized imports, config-driven parameters.

### Project Context & Goals

- Phase 1 roadmap centers on delivering the event-driven TFT streaming pipeline while keeping training/inference parity across macro, calendar, events, earnings, micro, and L2 feature families as Phase 1c begins (leakage validation, artefact refresh, microbench performance, promotion hygiene).
- Objectives:
  1. Finish Phase 1b wiring: capability flags, environment/config parity, publication lag exposure, documentation alignment.
  2. Execute Phase 1c validation: leakage guards, refreshed parity artefacts, microbench telemetry, promotion gate readiness.

### Progress to Date

- Capability toggles propagate end-to-end (dataset metadata, plan payloads, registry parity digest) with the targeted suites green.
- `DatasetServiceConfig.from_env` mirrors CLI toggle surfaces and the `ML_STREAMING_MICRO_BASE_DIR` / `ML_STREAMING_L2_BASE_DIR` overrides are documented.
- Publication lags are validated (`ml/features/validation.validate_known_future_effective_times`) and surfaced through builder macros/earnings, `StreamingPlanMessage.publication_lags`, and dataset metadata; docs remain aligned.
- Micro/L2 performance guard reran on 2025-10-22T22:51:17Z UTC (suite completed in 0.49 s with teardown dominating at 0.10 s) while holding persistence backlog processing below 0.25 s and keeping the <5 ms budget intact.
- Guardrail suites exercised: `poetry run mypy ml --strict`, `poetry ruff check ml`, parity tests, the refreshed performance microbench, planner dataset unit tests, `poetry run pytest ml/tests/integration/training/event_driven/test_plan_to_result.py::test_streaming_pipeline_records_gpu_telemetry`, `make validate-metrics`, and `make validate-events`.
- Streaming runner and persistence Docker images now bake in the `redis` client (`pip install redis` inside `Dockerfile.streaming` / `Dockerfile.pipeline`) so `_imports.HAS_REDIS` remains true and publish paths avoid the fallback. Rebuilt via `docker compose -f ml/deployment/docker-compose.yml build streaming_training_runner streaming_persistence_worker ml_dashboard`, confirmed `python -c "import redis; print(redis.__version__)` returns 7.0.0 inside both containers, and `docker compose exec redis redis-cli XLEN ml-events` exposes the live stream (length 17 after the latest cohort).
- Curriculum / ensemble / AMP canary `full_tft_95-098aefc6abb0` (2025-10-25T12:55Z) exercised the new flags end-to-end with bf16 AMP, curriculum stages `2000:0.55, *:0.75`, and two ensemble members. Manifest: `ml_out/tft_streaming_artifacts/full_tft_95/full_tft_95-098aefc6abb0_manifest.json` (ROC-AUC≈0.495, PR-AUC≈0.542, LogLoss≈0.813, temperature/platt/isotonic deltas −0.098/−0.125/−0.125, Brier≈0.306, GPU peak 1.688 GB). Telemetry captured `worker_curriculum_enabled=True`, `worker_train_fraction=0.75`, `worker_ensemble_enabled=True`, and `ensemble_optional_members_skipped=1.0` (shape-mismatch skip for the optional member). The corresponding dashboard snapshot lives at `ml_out/streaming_training_state_snapshot.json` and the compose-backed state (`/app/ml_out/streaming_training_state.json`) now surfaces the same plan/result via `/api/training/streaming/state`.

#### Detailed Updates

- Component builder now reuses the shared feature augmenters (macro, calendar, events, earnings, micro, L2) with configuration toggles surfaced through `TFTStreamingConfig`, `DatasetServiceConfig`, the streaming runner CLI, and deployment `.env` files. Feature manifests record capability flags for each family to preserve training/inference parity.
- Macro capability plumbing stays aligned via `ml/tests/unit/features/test_feature_config_macro_integration.py`; update the accompanying artefacts in `ml/tests/validation_reports/phase_1a/` (`macro_only_v1_summary.md`, `macro_revisions_core_summary.md`, etc.) whenever series lists or revision defaults change.
- Streaming plan payloads now surface the capability toggles (`include_*`) via `payload.capability_flags` so downstream services can assert parity. A Pandera contract test (`ml/tests/contracts/test_streaming_payloads.py::test_calendar_event_payload_schema`) guards the schema.
- FeatureRegistry now persists capability flag diffs inside each manifest’s `parity_digest`, logging any macro/micro/L2 toggle changes so parity audits can trace rollout history. Coverage: `ml/tests/unit/registry/test_feature_registry.py::test_register_feature_set_records_capability_flag_diff`.
- `DatasetServiceConfig.from_env` mirrors the CLI toggle surface (e.g., `ML_STREAMING_INCLUDE_MACRO`, `ML_STREAMING_INCLUDE_EVENTS`, `ML_STREAMING_MICRO_BASE_DIR`, `ML_STREAMING_L2_BASE_DIR`) so planner deployments inherit capability decisions and parquet roots without code edits.
- Streaming worker restores BCE loss parity: `StreamingWorkerConfig.loss_name`/`loss_pos_weight` feed `TFTTeacherConfig`, defaulting to BCE, and surface in telemetry (`worker_loss_name`, `worker_loss_pos_weight`) plus manifest `training_config` entries. CLI/env expose `--loss`, `--loss-pos-weight`, and `ML_STREAMING_TFT_LOSS(_POS_WEIGHT)` to keep automation in sync.
- Train/validation shard splitter now groups shards per instrument in time order so every instrument appears in both splits (single-shard symbols mirror into validation), preventing the legacy “alphabet remainder” validation-only drift.
- `_collect_streaming_logits` moves batches onto the teacher device before inference, eliminating CUDA device mismatches; GPU-aware unit tests pin the behaviour.
- Promotion telemetry now records calibrated metrics directly: the streaming consumer emits `ml_tft_streaming_validation_metric{dataset_id,plan_id,metric}` for base + calibrated log-loss/ECE/Brier, manifests include calibration deltas via the updated summarizer, and the sweep CLI exposes `--enable-platt-calibration` / `--enable-isotonic-calibration` alongside temperature knobs. Study summaries ship `best_metrics` for dashboards to diff best-trial deltas without parsing each artefact.
- Streaming training runner exposes calibration toggles (`--enable-temperature-calibration`, `--temperature-calibration-min/max/steps`, `--enable-platt-calibration`, `--enable-isotonic-calibration`) so canary cohorts can opt-in without editing configs. Dashboards now surface calibrated metrics per dataset via the new column fed by `ml_tft_streaming_validation_metric`.
- PhaseOne signal enrichment toggles (`include_macro_deltas`, `include_calendar_lags`, `include_clustering_tags`, `include_context_features`) now propagate through `DatasetBuildConfig`, the build runner, orchestration APIs, and `TFTDatasetBuilder`. When enabled, the builder appends the corresponding macro delta, calendar lag, clustering, and context columns, and dataset metadata captures the capability flags (`ml/tests/unit/data/test_tft_dataset_builder_phase_one.py`). The refreshed PhaseOne parity cohort (SPY, 2024-01-01 → 2024-01-08) lives under `ml_out/phase_one_signals_cohort`; `ml/tests/validation_reports/phase_1a/phase_one_signals_summary.md` documents component-only `PAYEMS_delta_1d`, `UNRATE_delta_1d`, and the canonical `close` column while all shared numeric fields remain identical to the legacy builder.
- Streaming dataset planner now runs guardrails before emitting plans via `ml.training.event_driven.guardrails.enforce_dataset_guardrails`. Configurable controls (`ML_STREAMING_MIN_POSITIVE_RATE`, `ML_STREAMING_MAX_POSITIVE_RATE`, `ML_STREAMING_SCHEMA_REFERENCE_COLUMNS`, `ML_STREAMING_KNOWN_FUTURE_PAIRS`, etc.) gate positive-rate thresholds, schema parity, and known-future timestamp checks; outcomes feed Prometheus counter `ml_tft_streaming_dataset_guardrails_total`.
- Optuna sweep harness lives at `ml/scripts/run_streaming_worker_sweep.py`, orchestrating `StreamingWorkerConfig` hyperparameters via `ml.training.event_driven.sweep`. Trial artefacts land under `ml_out/streaming_worker_sweeps/` with per-trial metrics logged by `FileTrialLogger`.
- Streaming worker optionally performs temperature scaling (toggle `ML_STREAMING_TFT_ENABLE_TEMPERATURE_CALIBRATION`) and now supports Platt (`ML_STREAMING_TFT_ENABLE_PLATT_CALIBRATION`) plus isotonic (`ML_STREAMING_TFT_ENABLE_ISOTONIC_CALIBRATION`) calibrators. Each emits calibrated log-loss/ECE/Brier scores and deltas so promotion reviews can compare pre/post calibration performance.
- Streaming metadata and plan payloads now expose `publication_lags` (macro/earnings/events) sourced from `DatasetMetadata`, validated via `ml.features.validation.validate_known_future_effective_times` and covered by `ml/tests/unit/features/test_known_future_transforms.py` plus `ml/tests/unit/features/earnings/test_parity.py`.
- Enabling L2 automatically switches on microstructure features in the builder/service so the L2 path can reuse the minute-level cache.
- Targeted validation is green: `poetry run ruff check ml`, `poetry run mypy ml --strict`, and `poetry run pytest ml/tests/unit/data/test_tft_dataset_builder_store.py ml/tests/unit/training/event_driven/test_dataset_service.py ml/tests/integration/pipeline/test_tft_pipeline_sidecar.py`.
- Latest manifest summary captured via `poetry run python -m ml.scripts.summarize_streaming_manifests --manifest-dir ml_out/tft_streaming_artifacts/full_tft_95 --limit 10` on 2025-10-22 22:43 UTC; see `ml/docs/ops/streaming_scaling_experiments.md` for GPU and metric deltas that inform promotion thresholds.
- Micro/L2 performance guardrails hold: `pytest -q ml/tests/performance -k microbench` (2025-10-22T22:51:17Z UTC) completed in 0.49 s with teardown-bound microbench runs and the persistence check still asserting <0.25 s backlog processing.
- Phase 1a parity artefacts now live under `ml/tests/validation_reports/phase_1a/` (`macro_only_v1_summary.md`, `macro_revisions_core_summary.md`, `student_lightweight_summary.md`), with micro/L2 cohorts refreshed on 2025-10-22T22:51:17Z UTC. The updated micro and L2 runs match on 1,113,763 joined rows, and the component builder continues to emit the additional `close` column. Streaming manifests `ml_out/tft_streaming_artifacts/full_tft_95/full_tft_95-b3b1ed2783ff_manifest.json` (ROC-AUC ≈ 0.671, peak GPU ≈ 1.79 GB) and `ml_out/tft_streaming_artifacts/full_tft_95/full_tft_95-0c15e27ea27e_manifest.json` (ROC-AUC ≈ 0.666, peak GPU ≈ 1.85 GB) anchor the current benchmarking baseline.
- Calendar/event and earnings toggles now have published parity summaries (`calendar_events_summary.md`, `earnings_summary.md`). Micro/L2 parity reports (`micro_summary.md`, `l2_summary.md`) refreshed on 2025-10-22T22:51:17Z compare staged Tier 1 feeds with 1,113,763 joined rows; shared columns match and the component builder emits the additional `close` column. Databento still excludes `BRK.XNAS`; capture parity once an alternate feed is available.
- Round-robin shard limiting landed on 2025-11-02: `_limit_metadata_for_streaming` now shares row/sequence budgets across instruments and records per-instrument totals/skips in `StreamingLimitSummary`. `StreamingLoaderTelemetry` forwards those maps so plan/result payloads, persistence snapshots, and dashboards expose the train/validation instrument mix (see `ml/tests/unit/training/teacher/test_streaming_loader.py::test_streaming_limits_round_robin_instrument_mix`).
- Zero-validation cohorts are no longer treated as success: `LightningStreamingWorker._compute_metrics` raises `ValidationDatasetEmptyError`, the worker emits `EventStatus.DEFERRED`, telemetry caps carry `validation_failure_*` details, and dashboards render the failure badge alongside the loss configuration (covered by `ml/tests/unit/training/event_driven/test_worker.py::test_worker_defers_when_validation_empty`). Plan/result payloads include the new caps so promotion guardrails can refuse empty cohorts.

### Remaining Tasks

- **Phase 1b exit criteria:** maintain the guardrail suite, refresh `ml/tests/validation_reports/phase_1a/` artefacts after changes, and keep docs/runbooks synchronized.
- ✅ (2025-10-23T05:11Z) PhaseOne signal cohort recorded (SPY, 2024-01-01 → 2024-01-08); artefacts stored in `ml_out/phase_one_signals_cohort` with parity captured in `ml/tests/validation_reports/phase_1a/phase_one_signals_summary.md`.
- **Wave-based capacity scaling for teacher training (2025-10-23T05:45Z)**  
  1. Record baseline metrics with the default caps (`max_total_rows=120k`, `max_total_sequences=90k`, `max_shards=32`, `max_epochs=2`, plan interval 900 s). Monitor GPU peaks (`ml_tft_streaming_worker_gpu_peak_mb`) and backlog gauges (`ml_tft_streaming_training_backlog`) via the dashboard (`http://localhost:8010`).  
  2. Raise the caps incrementally (e.g., +30 k rows or +8 shards per wave) while the runner loops continuously (`ML_STREAMING_MAX_PLANS=0`). Capture each cohort’s manifest, GPU usage, and metrics with `poetry run python -m ml.scripts.summarize_streaming_manifests --manifest-dir ml_out/tft_streaming_artifacts/full_tft_95 --limit 10`. Back off if GPU peaks approach 85 % of device RAM or backlog alarms trigger. Track the automated recommendation output with `poetry run python -m ml.scripts.recommend_streaming_wave --manifest-dir ml_out/tft_streaming_artifacts/full_tft_95` to keep the increments aligned with recent telemetry.  
  3. Once a wave yields better ROC-/PR-AUC, swap in the real promotion command (replace the default `true`) so successful cohorts update the registry automatically. Continue iterating until additional capacity no longer improves metrics, then document caps/epochs in `ml/docs/ops/streaming_scaling_experiments.md`.  
  4. After each wave (and before any promotion), re-run the guardrail suite (`make validate-metrics`, `make validate-events`, `poetry run pytest -q ml/tests/performance -k microbench`, `poetry run pytest -q ml/tests/integration/training/event_driven/test_plan_to_result.py::test_streaming_pipeline_records_gpu_telemetry`, `poetry run mypy ml --strict`, `poetry run python -m ruff check ml`) to ensure capability toggles and DTO schemas remain compliant.  
  5. Keep manifests from the best-performing cohorts under version control (or in the registry) so the actor/downstream services can roll back quickly if a later wave underperforms.
- **Phase 1c validation:**
  1. ✅ (2025-10-22T22:51:17Z UTC) Artefact diffs: regenerated `micro_summary.md` / `l2_summary.md` with Tier 1 feeds and updated `parity_summary.md` (1,113,763 joined rows, `close` column only difference).
2. ✅ (2025-10-23T05:34Z UTC) Performance envelope: reran `poetry run pytest -q ml/tests/performance -k microbench` (0.51 s suite; persistence backlog check stayed <0.25 s) and logged the run here and in `ml/docs/ops/streaming_scaling_experiments.md`.
  3. ✅ (2025-10-22 22:43 UTC) Benchmark manifests: executed `poetry run python -m ml.scripts.summarize_streaming_manifests --manifest-dir ml_out/tft_streaming_artifacts/full_tft_95 --limit 10` and pushed GPU/metric deltas to the ops doc.
4. ✅ (2025-10-23T05:34Z UTC) Observability & promotion gates: `make validate-metrics`, `make validate-events`, and `poetry run pytest -q ml/tests/integration/training/event_driven/test_plan_to_result.py::test_streaming_pipeline_records_gpu_telemetry` all passed (integration case completed in 0.24 s, telemetry assertions intact).
5. ✅ (2025-10-23T05:34Z UTC) Guardrail recap: metrics bootstrap and event constant validators returned OK; microbench/perf suite and streaming telemetry integration remained green, confirming the PhaseOne cohort promotion gates after artefact refresh.

#### Wave tracker (2025-10-24T13:15Z refresh)

- Cohorts executed after the 150 k / 112.5 k / 40 cap bump recovered the historical teacher baseline. Latest manifest (`full_tft_95-75a2141432f8`) reports ROC-AUC ≈ 0.670, PR-AUC ≈ 0.681, log loss ≈ 0.732, ECE₂₀ ≈ 0.198, Brier ≈ 0.269, peak GPU ≈ 1.56 GiB, and 111,830 selected rows across five shards.
- Run `sudo /home/nate/projects/nautilus_trader/.venv/bin/python -m ml.scripts.recommend_streaming_wave --manifest-dir /proc/2404911/root/app/ml_out/tft_streaming_artifacts/full_tft_95 --max-total-rows 120000 --max-total-sequences 90000 --max-shards 32 --shard-row-budget 120000` to generate the next wave proposal against the live runner volume. The current snapshot recommends `max_total_rows=150000`, `max_total_sequences≈112500`, `max_shards=40`, and `shard_row_budget=150000` (within the 6 GiB device envelope) while flagging the ROC regression (∆≈0.171 vs. best cohort) and confirming peak GPU ≈ 2.1 GiB. Snapshot payload:

  ```json
  {
    "current": {"shard_row_budget": 120000, "max_total_rows": 120000, "max_total_sequences": 90000, "max_shards": 32},
    "proposed": {"shard_row_budget": 150000, "max_total_rows": 150000, "max_total_sequences": 112500, "max_shards": 40},
    "notes": ["Best ROC-AUC observed: 0.672", "Mean ROC-AUC over last wave: 0.501", "Peak GPU usage: 2133.0 MiB"],
    "warnings": ["Recent ROC-AUC mean 0.501 lags best 0.672 by 0.171"]
  }
  ```
- Action items before Wave N+1: keep `ML_STREAMING_MAX_PLANS=0`, restart the runner (or deployment) with the updated caps, and rerun the guardrail bundle (`make validate-metrics`, `make validate-events`, `poetry run pytest -q ml/tests/performance -k microbench`, `poetry run pytest -q ml/tests/integration/training/event_driven/test_plan_to_result.py::test_streaming_pipeline_records_gpu_telemetry`, `poetry run mypy ml --strict`, `poetry run ruff check ml`). Archive best-performing manifests (`b720150ccd0a`, `b66920c11774`, `d81ccea14354`, `75a2141432f8`) for rollback coverage and promotion audits. Promotion command remains disabled by default (`ML_STREAMING_PROMOTION_COMMAND` empty) until ROC/PR/log-loss/ECE gates in this document are met.
- Deployment overrides updated: `ml/deployment/.env` and `.env.example` now set `ML_STREAMING_MAX_TOTAL_ROWS=150000`, `ML_STREAMING_MAX_TOTAL_SEQUENCES=112500`, `ML_STREAMING_MAX_SHARDS=40`, and `ML_STREAMING_SHARD_ROW_BUDGET=150000` so follow-on restarts automatically adopt the next-wave caps via `StreamingWorkerConfig.from_env()`.

### Beyond Phase 1: Smart Model Roadmap

- **Data & Feature Quality**
  - Keep `full_tft_95` fresh with every macro revision; regenerate validation artefacts under `ml/tests/validation_reports/phase_1a/` and extend Pandera contracts to catch schema drift when toggles change.
  - Enable macro deltas, calendar lags, clustering tags, context features, and L2 exposure incrementally through config/env flags. Each rollout requires parity tests, manifest comparisons, doc updates, and guardrail runs.
  - Leakage and label sparsity guardrails now execute in the planner (`enforce_dataset_guardrails`); next step is extending drift alerts (Pandera/leakage suites, parity artefact refresh) whenever capability toggles change.

- **Model & Training Enhancements**
  - ✅ Optuna/Bayesian sweep harness wraps `StreamingWorkerConfig` (batch size, hidden size, dropout, optimizer, scheduler). Invoke via `poetry run python -m ml.scripts.run_streaming_worker_sweep --dataset-dir ...`; trial artefacts land under `ml_out/streaming_worker_sweeps/` with deterministic seeds available through the CLI.
  - ✅ Temperature scaling calibration toggle (`ML_STREAMING_TFT_ENABLE_TEMPERATURE_CALIBRATION`) surfaces calibrated log-loss/ECE/Brier metrics in worker telemetry and sweep artefacts. Next: add Platt/isotonic options and bubble calibration deltas into promotion dashboards.
  - ✅ Curriculum-aware train splitting now flows through `StreamingWorkerConfig.curriculum`: set capability flags via env (`ML_STREAMING_CURRICULUM_STAGES=50000:0.55;*:0.8`, `ML_STREAMING_CURRICULUM_ENABLED=1`), the runner (`--enable-curriculum --curriculum-stage 50000:0.55 --curriculum-stage *:0.8`), or the sweep CLI. The worker records the effective fraction inside telemetry (`worker_curriculum_enabled`, `worker_train_fraction`) and enforces the plan-dependent split inside `_prepare_context`, with coverage from `ml/tests/unit/config/test_streaming_pipeline_config.py::test_curriculum_parsing_and_resolution` and `ml/tests/unit/training/event_driven/test_worker.py::test_curriculum_schedule_adjusts_train_fraction`.
  - ✅ Streaming ensembles now blend freshly trained logits with archived artefacts before metrics persist. Configure via env (`ML_STREAMING_ENSEMBLE_ENABLED=1`, `ML_STREAMING_ENSEMBLE_MEMBERS=peer_a.npz:0.4:required;peer_b.npz:0.6`) or CLI (`--enable-ensemble --ensemble-member ml_out/best_peer_logits.npz:1.0:required --ensemble-blend-mode weighted`). The worker loads `.npz` payloads from absolute paths or templates referencing `{plan_id}` / `{dataset_id}`, applies weighted or mean fusion, and logs the member count. Tests cover parsing and blending via `ml/tests/unit/config/test_streaming_pipeline_config.py::test_ensemble_parsing_and_validation` and `ml/tests/unit/training/event_driven/test_worker.py::test_ensemble_blending_merges_logits`.
  - ✅ AMP/mixed-precision gating is explicit: set `ML_STREAMING_ENABLE_AMP=1` + `ML_STREAMING_AMP_PRECISION=16-mixed` or pass `--enable-amp --amp-precision 16-mixed` to the runner; the worker now threads the selected precision into `TFTTeacher` so GPU cohorts can flip between bf16/16-mixed/32 without code changes. Sweeps accept the same flags, ensuring curriculum/ensemble/AMP explorations stay in sync.
  - ✅ Streaming dashboard surfaces resolved train fractions, ensemble usage, and peak GPU telemetry per dataset (see `ml/dashboard/service.py` + `ml/dashboard/templates/index_enhanced.html`), so operators can confirm curriculum triggers and peer blending directly from the control plane. Metrics land via `ensemble_members_used`, `ensemble_optional_members_skipped`, and `worker_train_fraction` entries in the persistence snapshot.
  - ✅ Canonical ensemble inventories (weights, required flags, row counts, skip reasons) flow through `StreamingEnsembleTelemetry`; dashboards and manifest summaries render the table so reviewers can audit peer health without unpacking NPZ artefacts.
  - ✅ Economic + stability diagnostics (slippage-adjusted Sharpe, hit rate, turnover, drawdown, KS/calibration drift) are emitted with every `StreamingRunTelemetry` bundle via `ml/training/event_driven/economic_metrics.py` and surface in the dashboard + manifest summaries for promotion gating.
  - ✅ `ml/scripts/validate_wave.py` orchestrates the rollout bundle (mypy, ruff, targeted pytest, validate-metrics/events, coverage, manifest summaries, doc freshness) with unit coverage in `ml/tests/unit/scripts/test_validate_wave.py`.
  - ✅ Adaptive scheduling in `ml/training/event_driven/orchestrator.py` respects backlog/GPU thresholds, exports Prometheus metrics (`ml_tft_streaming_orchestrator_adaptive_deferrals_total`, `ml_tft_streaming_orchestrator_adaptive_cooldown_seconds`), and the CLI runner now catches `AdaptiveSchedulingDeferred` to pause cleanly.
  - Explore curriculum or ensemble variants (sector, liquidity buckets) with downstream blending while preserving typed interfaces and parity coverage.
  - Enable mixed precision and longer sequence horizons when GPU headroom allows; protect P99 with the microbench suite and document any AMP rollout.

- **Evaluation & Economics**
  - Extend manifests with economic metrics (slippage-adjusted Sharpe, hit rate, turnover, drawdown) and enforce economic gates before promotion.
  - Build stability diagnostics (probability distribution KS tests, feature importance drift) to block promotion when deviations exceed guardrails.
  - Automate reliability diagrams and calibration plots, targeting ECE <0.18 ahead of auto-promotion and storing artefacts alongside each manifest.

- **Pipeline & Process**
  - Factor orchestration responsibilities (planner, worker, promotion) into discrete services/modules with typed protocols to simplify testing.
  - Add a `make validate-wave` entrypoint that chains mypy, ruff, targeted pytest suites, manifest summaries, and doc refreshes for every wave.
  - Expand observability: emit Prometheus gauges for ROC/PR trends, calibration drift, data freshness, and fallback counters.

- **Compute & Scaling**
  - Trial larger GPUs or cloud accelerators once local hardware saturates; keep device selection config-driven so `StreamingWorkerConfig.from_env()` remains the single source of truth.
  - Investigate multi-GPU/distributed Lightning training after single-GPU tuning stabilizes, adding configuration knobs and tests for the new strategies.

- **Promotion Gate Tuning**
  - Raise promotion thresholds (e.g., ROC-AUC ≥0.68, PR-AUC ≥0.66, log loss ≤0.73, ECE ≤0.18) and require consecutive successful cohorts before invoking the real promotion command.
  - Use `ml/docs/ops/streaming_scaling_experiments.md` as a scoreboard for cohort metrics, resource usage, economic results, and promotion decisions so reversions stay trivial.

### Standards & Guardrails

- Follow `AGENTS.md`: strict typing via `poetry run mypy ml --strict`, clean lint (`poetry ruff check ml`), targeted pytest, ≥90 % ML coverage, and no heavy operations in hot paths.
- Maintain observability and fallback patterns: structured logging with `exc_info=True`, telemetry counters for fallbacks, message topics through `ml.common.message_topics.build_topic_for_stage`, and bus configuration via `MessageBusConfig.from_env()`.
- Reference companion tests/artefacts whenever documenting new capability toggles, metrics, or configuration changes.
- Avoid bare `try` / `except`; enforce config-driven enums for capabilities and event statuses.

### Required Patterns & Recent Additions

- Capability flag propagation across dataset metadata, plan payloads, and the registry `parity_digest`.
- Publication lag exposure via metadata and plan payloads enforced by `ml.features.validation.validate_known_future_effective_times`.
- `DatasetServiceConfig.from_env` environment parity with `ML_STREAMING_MICRO_BASE_DIR` / `ML_STREAMING_L2_BASE_DIR` overrides.
- Registry logging of capability diffs and the micro/L2 performance guard (<5 ms P99) with documented evidence.

### Key Files & Methods

- Documentation: `ml/docs/architecture/event_driven_streaming_plan.md`, `ml/docs/ops/streaming_scaling_experiments.md`.
- Config & pipeline: `ml/config/streaming_pipeline.py`, `ml/training/teacher/streaming_loader.py`, `ml/cli/streaming_training_runner.py`.
- Metadata & builders: `ml/data/__init__.py`, `ml/data/tft_dataset_builder.py`, `ml/data/fred_join.py`.
- Messaging: `ml/training/event_driven/payloads.py`.
- Tests: `ml/tests/contracts/test_streaming_payloads.py`, `ml/tests/unit/features/test_known_future_transforms.py`, `ml/tests/unit/features/earnings/test_parity.py`, `ml/tests/performance/test_streaming_persistence_microbench.py`.

### Testing & Validation Pipeline

- `poetry run mypy ml --strict`
- `poetry ruff check ml`
- `pytest -q ml/tests/contracts/test_streaming_payloads.py`
- `pytest -q ml/tests/unit/features/test_known_future_transforms.py`
- `pytest -q ml/tests/unit/features/earnings/test_parity.py`
- `pytest -q ml/tests/performance -k microbench`
- `pytest -q ml/tests/unit/cli/test_streaming_training_runner_adaptive.py`
- `pytest -q ml/tests/unit/training/event_driven/test_dataset_service.py`
- `pytest -q ml/tests/unit/training/teacher/test_streaming_loader.py::test_streaming_limits_round_robin_instrument_mix`
- `pytest -q ml/tests/unit/training/event_driven/test_worker.py::test_worker_defers_when_validation_empty`
- Additional guardrails (based on scope): `pytest ml/tests/unit/features/test_feature_config_macro_integration.py`, `pytest ml/tests/unit/data/test_dataset_build_macro.py`, etc.
- Promotion/observability rehearsals: `pytest -q ml/tests/integration/training/event_driven/test_plan_to_result.py::test_streaming_pipeline_records_gpu_telemetry`, `make validate-metrics`, `make validate-events`.
- Wave pre-flight: `poetry run python -m ml.scripts.recommend_streaming_wave --manifest-dir ml_out/tft_streaming_artifacts/full_tft_95 --run-validate-wave --validate-wave-args '--manifest-limit 5'` (or invoke `ml.scripts.validate_wave` directly for bespoke targets).
- Manual telemetry audit before promotion: confirm dashboards surface worker loss config, train/validation instrument coverage (including skipped instruments), and any `validation_failure_*` reasons introduced by the latest cohort; cross-check the manifest caps against the same fields.

### Working Notes

- Maintain structured doc updates (bullet format, inline code for paths/tests).
- Refresh `ml/tests/validation_reports/phase_1a/` artefacts when toggles change and highlight the latest artefacts in references.
- Document performance runs with timestamp, duration, and key metrics.
- Keep the Phase 1 checklist status current and cite associated tests/artefacts when updating this document.
- Run the validation bundle (`poetry run python -m ml.scripts.recommend_streaming_wave --run-validate-wave ...` or `poetry run python -m ml.scripts.validate_wave ...`) before promoting new waves; gate on green mypy/ruff/pytest/validate-metrics-events/coverage/doc freshness.

### Feature Toggle Matrix

| Capability | Streaming runner CLI flags | Environment variables (`ml/deployment/.env*`) | Config surfaces |
|------------|---------------------------|-----------------------------------------------|-----------------|
| Macro features | `--include-macro` / `--no-include-macro`<br>`--include-macro-revisions` / `--no-include-macro-revisions` | `ML_STREAMING_INCLUDE_MACRO`, `ML_STREAMING_INCLUDE_MACRO_REVISIONS` | `DatasetServiceConfig.include_macro`, `TFTStreamingConfig.include_macro`, `DatasetBuildConfig.include_macro` |
| Calendar features | `--include-calendar` / `--no-include-calendar` | `ML_STREAMING_INCLUDE_CALENDAR` | `DatasetServiceConfig.include_calendar`, `TFTStreamingConfig.include_calendar`, `DatasetBuildConfig.include_calendar` |
| Event features | `--include-events` / `--no-include-events` | `ML_STREAMING_INCLUDE_EVENTS` | `DatasetServiceConfig.include_events`, `TFTStreamingConfig.include_events`, `DatasetBuildConfig.include_events` |
| Earnings features | `--include-earnings` / `--no-include-earnings` | `ML_STREAMING_INCLUDE_EARNINGS` | `DatasetServiceConfig.include_earnings`, `TFTStreamingConfig.include_earnings`, `DatasetBuildConfig.include_earnings` |
| Microstructure features | `--include-micro` / `--no-include-micro` | `ML_STREAMING_INCLUDE_MICRO` | `DatasetServiceConfig.include_micro`, `TFTStreamingConfig.include_micro`, `DatasetBuildConfig.include_micro` |
| L2 order book features | `--include-l2` / `--no-include-l2` | `ML_STREAMING_INCLUDE_L2` | `DatasetServiceConfig.include_l2`, `TFTStreamingConfig.include_l2`, `DatasetBuildConfig.include_l2` |

Notes:
- CLI defaults honour the environment; the CLI `--no-*` switches force-disable even if env variables are set.
- `DatasetServiceConfig` enforces service-wide minimums (e.g. enabling macro features for all plans), overriding request-level opt-outs via `_apply_service_caps`.
- Feature manifests produced during dataset builds include capability flags so registry consumers can enforce parity checks during promotion.
- Macro revision defaults: unless the relevant `include_macro_revisions` toggle is enabled, revision joins stay disabled. When enabled, the builder defaults to `revision_mode="core"` with empty `revision_windows`, mirroring legacy behaviour; configure `macro_revision_windows` via config or env before enabling to avoid silent no-ops.
- Student mode (`student_mode=True`) forces macro, events, L2, and earnings joins off regardless of CLI/env switches to keep the lightweight student path consistent. Document overrides explicitly when comparing student vs. teacher datasets.

#### Tier 1 Universes

- The Intelligent TFT pipeline now publishes the full 95-instrument Tier 1 universe under `ml/config/universes.py:TIER1_FULL_95` (aliased as `TIER1_DEFAULT`). The symbols mirror the `ml_out/full_tft_95` manifest (e.g. `AAPL.XNAS`, `SPY.XNAS`, `TSLA.XNAS`).
- Lightweight smoke/test workflows may continue to rely on the historical 12-name basket via `TIER1_CORE_12`.
- Ingestion helpers (`ml/data/ingest/l2_efficient.py:get_tier1_symbols`) honour `ML_TIER1_SYMBOL_SET` (`full`/`core`) so operators can switch between the complete universe and the compact test set without editing code.
- Progress files (`tier1_l1_progress.json`) remain compatible; when populated they take precedence over the configured set.

#### Example: Enabling macro revisions for the streaming runner

```bash
export ML_STREAMING_INCLUDE_MACRO=1
export ML_STREAMING_INCLUDE_MACRO_REVISIONS=1
poetry run python -m ml.cli.streaming_training_runner \
    --dataset-dir ml_out/full_tft_95 \
    --output-dir ml_out/tft_streaming_artifacts/full_tft_95 \
    --include-macro \
    --include-macro-revisions \
    --max-plans 1
```

Ensure `macro_revision_windows` is populated via config (e.g. `[1, 7, 30]`) before enabling revisions to avoid empty feature columns.

#### Example: Curriculum + ensemble run with AMP

```bash
poetry run python -m ml.cli.streaming_training_runner \
    --dataset-dir ml_out/full_tft_95 \
    --output-dir ml_out/tft_streaming_artifacts/full_tft_95 \
    --max-plans 1 \
    --enable-amp --amp-precision 16-mixed \
    --enable-curriculum --curriculum-stage 60000:0.55 --curriculum-stage *:0.75 \
    --enable-ensemble \
    --ensemble-member ml_out/tft_streaming_artifacts/full_tft_95/best_peer_logits.npz:0.5:required \
    --ensemble-member ml_out/tft_streaming_artifacts/full_tft_95/canary_logits.npz:0.5 \
    --ensemble-blend-mode weighted
```

The worker records the effective train fraction in telemetry, uses AMP-aware precision when `--enable-amp` is set, and persists blended logits/metrics that downstream dashboards and promotion gates can audit via the manifest bundle.

2025-10-25 execution (bf16 on CPU, shard row budget 4 000, `--gpu-monitor-interval 30`) produced plan `full_tft_95-098aefc6abb0` with manifests under `ml_out/tft_streaming_artifacts/full_tft_95/` and a state snapshot at `ml_out/streaming_training_state_snapshot.json`. Redis events were written to `ml-events` (current length 17 via `docker compose exec redis redis-cli XLEN ml-events`), the persistence worker mirrored them into `/app/ml_out/streaming_training_state.json` after replaying the backlog inside the `streaming_persistence_worker` container, and `/api/training/streaming/state` now reports `worker_train_fraction=0.75`, `worker_curriculum_enabled=true`, `worker_ensemble_enabled=true`, and `max_gpu_memory_mb=1688.0` for the dataset card. Optional ensemble member skips are captured via `ensemble_optional_members_skipped=1.0` so dashboards explain the weighting behaviour.

#### Peer logits metadata refresh (2025-10-27)

- Added `ml/scripts/ensure_peer_logits_metadata.py` to backfill per-row identifiers (`train_row_ids`, `val_row_ids`, etc.) into historical peer logits artefacts. The CLI copies metadata from a reference teacher artefact after verifying train/validation lengths match.
- Usage: `poetry run python -m ml.scripts.ensure_peer_logits_metadata --reference ml_out/tft_streaming_artifacts/full_tft_95/full_tft_95-098aefc6abb0_logits.npz --peer ml_out/tft_streaming_artifacts/full_tft_95/best_peer_logits.npz --peer ml_out/tft_streaming_artifacts/full_tft_95/canary_logits.npz`. Repeat per peer directory or pass `--peer-dir` to glob every artefact in a cohort drop.
- The script refuses to proceed when the reference is missing metadata or when logits shapes differ, so regenerated peers are guaranteed to stay aligned with the latest guardrails.
- After refresh, update the manifest’s `cohort_run.peer_artifacts` section (or the ops runbook) to note the validated peers are metadata-complete; future ensemble runs can rely on `ensemble_members_misaligned=0.0`.

#### Curriculum guards & AMP gating (2025-10-27)

- `StreamingWorkerConfig` now accepts structured guard rules so curriculum stages only engage when recent telemetry is healthy. Define guards via `--curriculum-guard LABEL:min_rows=...,max_gpu_mb=...,train_fraction=...` or `ML_STREAMING_CURRICULUM_GUARDS="phase1:min_rows=50000,max_gpu_mb=2200,fallback_fraction=0.6"`. Guards evaluate row counts plus optional ROC/backlog/GPU hints supplied in `plan.caps` and fall back to the configured fraction (or baseline) when the predicates fail.
- Stage specs support optional labels (`--curriculum-stage 60000:0.55:phase1`), enabling guards to target specific regimes. Telemetry now records `worker_curriculum_stage` and `worker_curriculum_reason`, which surface in manifests, persistence snapshots, and the dashboard.
- AMP can be auto-disabled per plan by supplying `--amp-guard-threshold-mb <MiB>` (or `ML_STREAMING_AMP_GUARD_THRESHOLD_MB`). When recent peak GPU usage stored in `plan.caps['recent_peak_gpu_mb']` crosses the guard, the worker reverts to the base precision, logs the reason, and exposes `worker_amp_enabled` / `worker_amp_guard_reason` for operators.

### Testing & Validation Checklist

- Static analysis: `poetry run ruff check ml` and `poetry run mypy ml --strict` must pass before and after doc/config updates.
- Unit/integration coverage for parity: `poetry run pytest ml/tests/unit/data/test_tft_dataset_builder_store.py ml/tests/unit/training/event_driven/test_dataset_service.py ml/tests/integration/pipeline/test_tft_pipeline_sidecar.py`.
- When toggles change, extend coverage with scenario-specific suites (`ml/tests/unit/features/`, `ml/tests/integration/training/event_driven/test_plan_to_result.py`) and update `ml/tests/validation_reports/` with parity artefacts.
- Before promotions, keep `make validate-metrics` and `make validate-events` green so dashboards and downstream automation remain consistent with updated capability flags.

#### Parity artefact workflow

1. Generate matched teacher/student datasets with both legacy and component builders (`ml.data.tft_dataset_builder_legacy.TFTDatasetBuilder`, `ml.data.tft_dataset_builder.TFTDatasetBuilder`) using identical configs.
2. Store comparative summaries under `ml/tests/validation_reports/phase_1a/` (e.g. histogram diffs, schema checks, macro revision availability tables). Include metadata describing which toggles were enabled.
3. Link artefacts in commit messages and update the doc (this section) once Phase 1a closure is complete so operators know where to find the latest validation evidence. Latest evidence: `ml/tests/validation_reports/phase_1a/macro_only_v1_summary.md`, `macro_revisions_core_summary.md`, `student_lightweight_summary.md`, `calendar_events_summary.md`, `earnings_summary.md`, `micro_summary.md`, `l2_summary.md`, and the scenario index in `ml/tests/validation_reports/phase_1a/parity_summary.md`. Micro/L2 runs now succeed with all shared columns matching; the component builder adds a `close` column and `BRK.XNAS` remains pending until a data source is available.

##### Generating a parity report

```bash
poetry run python - <<'PY'
from __future__ import annotations

from pathlib import Path

import polars as pl

from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from ml.data.tft_dataset_builder import TFTDatasetBuilder as ComponentBuilder
from ml.data.tft_dataset_builder_legacy import TFTDatasetBuilder as LegacyBuilder

OUTPUT_DIR = Path("ml/tests/validation_reports/phase_1a")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

catalog = ParquetDataCatalog("ml_out/full_tft_95")  # swap in the dataset root under test
symbols = ["SPY.XNAS"]

legacy = LegacyBuilder(
    catalog=catalog,
    symbols=symbols,
    include_macro=True,
    include_macro_revisions=False,
    student_mode=False,
)
component = ComponentBuilder(
    catalog=catalog,
    symbols=symbols,
    include_macro=True,
    include_macro_revisions=False,
    student_mode=False,
)

join_cols = ["timestamp", "instrument_id", "time_index"]
legacy_df = pl.DataFrame(legacy.build_training_dataset(use_polars=True)).sort(join_cols)
component_df = pl.DataFrame(component.build_training_dataset(use_polars=True)).sort(join_cols)

merged = legacy_df.join(component_df, on=join_cols, how="outer", suffix="_component")

schema_drift: list[str] = []
numeric_delta: dict[str, float] = {}
for column in legacy_df.columns:
    if column in join_cols:
        continue
    component_column = f"{column}_component"
    if component_column not in merged.columns:
        schema_drift.append(column)
        continue
    legacy_series = merged[column]
    component_series = merged[component_column]
    if legacy_series.dtype != component_series.dtype or legacy_series.null_count() != component_series.null_count():
        schema_drift.append(column)
        continue
    if legacy_series.dtype.is_numeric() or legacy_series.dtype == pl.Boolean:
        diff = (legacy_series - component_series).abs().max()  # type: ignore[operator]
        if diff and diff > 0:
            numeric_delta[column] = float(diff)
    elif (legacy_series != component_series).any():
        schema_drift.append(column)

lines: list[str] = [
    "# Macro Only Parity",
    "",
    f"Rows compared: {len(merged)}",
    "",
]
if schema_drift:
    lines.append("## Columns with schema/nullability drift")
    lines.append("")
    lines.extend(f"- {name}" for name in sorted(schema_drift))
    lines.append("")
if numeric_delta:
    lines.append("## Non-zero numeric deltas")
    lines.append("")
    lines.extend(f"- {name}: max abs diff {value}" for name, value in sorted(numeric_delta.items()))
    lines.append("")
else:
    lines.append("All numeric columns matched exactly.")

summary = OUTPUT_DIR / "macro_only_v1_summary.md"
summary.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote parity summary to {summary}")
PY
```

- Swap `ml_out/full_tft_95` for the dataset directory under validation; ensure both builders point to the same parquet root.
- Re-run the snippet with additional flag combinations (e.g. enabling macro revisions or student mode) and commit the resulting Markdown summaries alongside `parity_summary.md` for traceability.
- Latest capture (2025-10-23T05:11Z): `ml_out/phase_one_signals_cohort` (SPY, 2024-01-01 → 2024-01-08) produced 2,939 joined rows. `ml/tests/validation_reports/phase_1a/phase_one_signals_summary.md` shows component-only macro delta columns (`PAYEMS_delta_1d`, `UNRATE_delta_1d`) plus the canonical `close` column; all shared numeric metrics remained at parity with the legacy builder.

##### Materializing micro/L2 parity inputs

Microstructure and L2 parity runs require the tier1 parquet feeds that power the legacy builder. Populate them locally (or mount an existing cache) before re-running the parity script:

1. **Backfill recent L0 minute bars** so micro aggregators have a canonical source:
   ```bash
   poetry run python -m ml.cli.backfill_ohlcv_recent \
       --tier 1 \
       --days 30 \
       --data-dir data/tier1
   ```
   Outputs land under `data/tier1/<SYMBOL>/l0/<SYMBOL>_ohlcv.parquet`.
2. **Populate L2 order book shards** with the resume-safe helper:
   ```bash
   export POLARS_MAX_THREADS=1
   export PYARROW_NUM_THREADS=1
   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1

   for offset in 0 5 10 15 20; do
       poetry run python -m ml.cli.populate_l2_efficient \
           --tier 1 \
           --days 30 \
           --check-gaps \
           --max-symbols 5 \
           --symbol-offset "${offset}" \
           --rate-limit 10 \
           --sleep-between-symbols 1
   done
   ```
   L2 artifacts are written to `data/tier1/<SYMBOL>/l2/<SYMBOL>_mbp-10.parquet`, with progress tracked in `data/tier1/.l2_progress.json`.
3. **Verify availability** before re-running parity:
   ```bash
   ls data/tier1/SPY/l0
   ls data/tier1/SPY/l2
   ```
4. **Enable parquet fallback** when generating parity artefacts:
   ```bash
   export ML_TFT_ALLOW_PARQUET_FALLBACK=1
   ```
5. **Re-run parity** once both feeds exist. Use the snippet below with `include_micro=True` / `include_l2=True` to regenerate `micro_summary.md` and `l2_summary.md`, then update `parity_summary.md` with the resulting notes (component-only `close` column, shared-column parity).

> Note: Databento currently omits `BRK.XNAS` from `EQUS.MINI` / `DBEQ.BASIC`. Capture parity for that symbol once an alternate feed is staged and update the summaries accordingly.

> Tip: when running inside containers, mount the tier1 directory read-only into planners/workers and point `micro_base_dir` / `l2_base_dir` at that path via config or environment (`ML_STREAMING_MICRO_BASE_DIR`, `ML_STREAMING_L2_BASE_DIR`).

#### Market Data Persistence

- Ingestion orchestrators continue to dual-write via the datastore: quotes/trades land in PostgreSQL through the configured `RawIngestionWriter`, while parquet artifacts under `data/tier1/` act as cold-path fallbacks for training and parity jobs.
- The parquet training datasets (`ml_out/full_tft_95/...`) remain derived outputs; ingestion CLIs should always pipe new market data through the datastore/SQL path first and only rely on parquet for offline validation.

#### Phase 1 Closeout Checklist

| Task | Owner | Blocking Dependencies | Validation |
|------|-------|-----------------------|------------|
| Seed Tier 1 micro + L2 parquet feeds (`ml.cli.backfill_ohlcv_recent`, `ml.cli.populate_l2_efficient`) | Data platform | Fast storage volume mounted at `data/tier1` | Re-run `poetry run python -m ml.scripts.run_streaming_cohort --dry-run --sample 10 --include-micro --include-l2` to ensure joins succeed |
| Refresh parity reports for micro/L2 (`micro_summary.md`, `l2_summary.md`) | ML infra | Tier 1 feeds present, legacy builder enabled | Run the parity snippet below with `include_micro=True` / `include_l2=True`; inspect diffs and refresh `parity_summary.md`. |
| Update FeatureRegistry capability flags for micro/L2 | Registry team | Updated manifests from parity run | `poetry run pytest ml/tests/unit/features/test_microstructure.py ml/tests/unit/features/test_l2_aggregate.py` |
| Record micro/L2 artefact evidence in ops docs | ML ops | Fresh Markdown summaries | Append to `ml/docs/ops/streaming_scaling_experiments.md` + parity directory commit |
| Execute Phase 1b regression suite | ML infra | Feature wiring complete | `poetry run pytest -q ml/tests/integration/training/event_driven/test_plan_worker_round_trip.py --maxfail=1` |
| Execute Phase 1c validation suite | ML infra | Artefact and performance updates | `pytest -q ml/tests/performance -k microbench && make validate-metrics && make validate-events` |
| Update promotion runbooks with new gates | Strategy ops | Phase 1 metrics captured | Cross-link `ML_STREAMING_PROMOTION_COMMAND` usage in ops docs; confirm staging automation |

### Components

- **DatasetService**  
  - Responsibility: scan parquet repositories, plan Arrow shards, enforce caps (`TFTStreamingConfig`), publish shard assignments.  
  - Inputs: dataset manifest events, shard budget commands.  
  - Outputs: `DatasetPlanEvent` (`Stage.DATASET_PLANNED`, `EventStatus.SUCCESS|DEFERRED`).  
  - Observability: `ml_tft_streaming_metadata_*`, queue lag gauge, fallback counter.  
  - Fallback chain: parquet scan → cached plan (redis) → static file → dummy (returns 0 shards with warning).
  - Implementation stub: `StreamingDatasetPlanner` (`ml/training/event_driven/dataset_service.py`) merges per-job caps with `DatasetServiceConfig` before producing `DatasetPlanEvent`.

- **StreamingTrainingWorker**  
  - Responsibility: consume dataset plans, run bounded Lightning jobs, emit logits + metrics.  
  - Inputs: `DatasetPlanEvent`, worker config (devices, max rows, sequences).  
  - Outputs: `TrainingResultEvent` (contains `StreamingRunTelemetry`, checkpoint URI, metrics).  
  - Backpressure: respects `max_concurrent_jobs`, publishes `EventStatus.BUSY` when saturated.  
  - Failure handling: circuit breaker by data source, fallback to cached checkpoint.  
  - Reference implementation: `LightningStreamingWorker` (`ml/training/event_driven/worker.py`) builds streaming dataloaders, drives the TFT teacher, persists logits/telemetry artifacts, and emits Prometheus metrics (`ml_tft_streaming_training_runs_total`, `ml_tft_streaming_training_duration_seconds`).

- **TrainingOrchestrator**  
  - Responsibility: coordinate dataset service, workers, registries; track job lifecycle.  
  - Inputs: CLI schedule, dataset/worker status events.  
  - Outputs: orchestration commands (`PlanShardCommand`, `StartTrainingCommand`, `FinalizeTrainingEvent`).  
  - Persistence: orchestrator state store (JSON/persistent) with monotonic timestamps and direct backlog gauge updates (`ml_tft_streaming_training_backlog{dataset_id}`) so dashboards reflect saturation even if the persistence worker lags.
  - Reference implementation: `InMemoryStreamingOrchestrator` (`ml/training/event_driven/orchestrator.py`) publishes stage-aware topics via `build_topic_for_stage`, executes plans immediately with a registered worker, and records heartbeats/results on the configured message bus with retry buffering and in-memory fallbacks.
- **StreamingPersistenceWorker**  
  - Responsibility: subscribe to Redis Streams, hydrate the file-backed streaming state store, and surface backlog metrics for the dashboard.  
  - Inputs: bus payloads via `StreamingTrainingPersistenceService.create_stream_consumer`.  
  - Outputs: JSON snapshots at `ML_STREAM_PERSIST_STATE_PATH`, optional observability metrics through `ObservabilityService`.  
  - Reference implementation: `StreamingTrainingPersistenceWorker` (`ml/consumers/streaming_training_worker.py`) with CLI wrapper `ml.cli.streaming_persistence_worker`.  
  - Observability: backlog gauge (`ml_tft_streaming_training_backlog`), per-worker progress/RSS metrics, and active worker counts (`ml_tft_streaming_workers_active`) labeled by dataset for backlog widgets.
  - Lightning worker produces logits per plan (`<plan_id>_logits.npz`) alongside telemetry; artefacts live under the configured worker output directory (e.g., `ml_out/tft_artifacts`).

### Dataset Readiness

- **Build the full-production TFT dataset**
  1. Run the orchestrator with `ml/config/orchestrator/production_full.toml` (ingestion + dataset stages enabled). This pulls the full 95-instrument universe, macro (FRED/ALFRED), calendar, and earnings feeds.
     ```bash
     poetry run python -m ml.cli.pipeline_orchestrator \
         --config ml/config/orchestrator/production_full.toml
     ```
  2. Ensure external dependencies are configured (FRED/ALFRED parquet paths, event feeds, `DB_CONNECTION` for historical bars).
  3. Validate outputs under `ml_out/production_full/` (`dataset.parquet`, `dataset_metadata.json`, `features_npz.npz`, etc.) using `ml.cli.dataset_report` if desired.

- **Convert macro vintages to age features**
  ```bash
  poetry run python -m ml.cli.convert_vintage_age \
      --source ml_out/production_full/dataset.parquet \
      --metadata ml_out/production_full/dataset_metadata.json \
      --overwrite
  ```
  - Produces `dataset_with_vintage_age.parquet` and updates metadata (`vintage_age_columns`). This is mandatory for leakage-free streaming training.
  - Metadata sanity checks: `time_idx_col` (`time_index`), `group_id_col` (`instrument_id`), target `y`, plus the expanded macro/calendar/earnings feature lists.

- **Plan + train a GPU cohort locally** (slow but exercises the full flow)
  ```bash
  poetry run python -m ml.scripts.run_streaming_cohort \
      --dataset-dir ml_out/production_full \
      --output-dir ml_out/tft_streaming_artifacts/production_full \
      --state-path ml_out/streaming_training_state_snapshot.json \
      --max-total-rows 120000 \
      --max-total-sequences 90000 \
      --max-shards 32 \
      --max-epochs 1 \
      --batch-size 48 \
      --accelerator gpu \
      --devices 1
  ```
  - Result (latest run, 2025‑10‑19, GPU cohort on `ml_out/full_tft_95`): plan `full_tft_95-05c32e888427`, 4 shards (90 642 rows) selected, `roc_auc ≈ 0.496`, peak GPU memory ≈1 899 MB, logits persisted at `ml_out/tft_streaming_artifacts/full_tft_95/full_tft_95-05c32e888427_logits.npz`, state snapshot at `ml_out/streaming_training_state_snapshot.json`, and the manifest tying dataset/metrics/artifacts lives at `ml_out/tft_streaming_artifacts/full_tft_95/full_tft_95-05c32e888427_manifest.json`.
  - New config knobs: `--max-epochs` now flows through to `StreamingWorkerConfig.max_epochs` so production runs can train for dozens of epochs, and passing `0` (or negative) to the `--max-total-rows/-sequences/-shards` flags disables those limits when hardware allows. A follow-up multi-epoch cohort (plan `full_tft_95-1763b898c3ad`) ran the full 90 k-row slice for two epochs, reached `roc_auc≈0.519`, and peaked at ~1.88 GB VRAM; the manifest lives at `ml_out/tft_streaming_artifacts/full_tft_95/full_tft_95-1763b898c3ad_manifest.json`.
  - Sequential cohorts should cap at `max_shards=32` / `max_total_rows≈120k` to stay within 6 GB VRAM when orchestrated in production.

- **Prepare for actor inference**
  - Review `dataset_metadata.json` to confirm all feature families (macro, calendar, earnings) are present under `static_reals`, `time_varying_known_reals`, `time_varying_unknown_reals`, `vintage_age_columns`, and that `capability_flags` mirrors the toggles used during the build (e.g., `include_micro`, `include_l2`).
  - Store the logits (`logits.npz`) and training metadata (snapshot, metrics) in your registry so the actor wrapper can load the trained teacher during inference; the manifest above is the canonical hand-off bundle for the latest run.
  - Registry promotion (teacher logits artefact) now lives in `~/.nautilus/ml/models` as `model_1760910442082150` with digest `3964…0f2f`; reference that `model_id` in actor configs or distillation harnesses that expect registry lookups.

- **Runner dataset contract**
  - The compose stack mounts `${ML_STREAMING_DATASET_DIR}` into the runner at `/app/${ML_STREAMING_DATASET_DIR}`; keep it pointed at the production slice (`ml_out/full_tft_95`) so `dataset_with_vintage_age.parquet`, metadata, and reports stay in sync.
  - Confirm `dataset_with_vintage_age.parquet`, `dataset_metadata.json`, and `report.json` exist before launching the loop. Missing artefacts raise `FileNotFoundError` during runner bootstrap—rebuild the dataset rather than letting the container churn.
  - When migrating to a new machine, copy the entire dataset directory (including `report.md/json`) to avoid schema drift; the loader validates feature layout against the report on every startup.

- **Additional tips**
  - The integration test `pytest -q ml/tests/integration/training/event_driven/test_plan_to_result.py::test_streaming_pipeline_records_gpu_telemetry` remains a deterministic smoke path for planner → worker → persistence without touching the full dataset.
  - If you need a lighter smoke run before the full build, rerun the CLI with lower `max_total_rows`/`max_shards` (e.g., `--max-total-rows 20000 --max-shards 4`), verify metrics/logits, then revert to production caps.

#### Vintage-age conversion helper

- Run `python -m ml.cli.convert_vintage_age --source ml_out/full_tft_95/dataset.parquet` to emit `*_vintage_age_minutes` columns and refresh `dataset_metadata.json` in place. The CLI streams batches via PyArrow, preserves schema order, and writes `dataset_with_vintage_age.parquet`.
- Keep `time_index` exclusive to `time_idx_col`; exclude it from `time_varying_known_reals` when constructing `TFTStreamingConfig` so the TFT template dataset retains an integer time index.
- Metadata gains `vintage_age_columns` so downstream tooling can validate the conversion without re-parsing the parquet.
- The dataset build pipeline now exposes `TFTDatasetTaskConfig.convert_vintage_to_age` (CLI flag `--convert-vintage-age`) which automatically performs the conversion and rewrites `dataset_metadata.json` after each nightly refresh.
- Dataset rebuilds remove both `dataset.parquet` and `dataset_with_vintage_age.parquet` before writing new artefacts so vintage-age cohorts do not accumulate duplicate gigabyte-scale files over time.

### Hardware Guidance

- The reference dev machine is an i7‑11700F CPU with a 6 GB GTX 1660 Ti. Full-dataset training is feasible with throttled settings:
  - Reduce `max_total_rows` / `max_shards` per plan (process the dataset in cohorts).
  - Lower `batch_size` (64–128) and use gradient accumulation via `TFTTeacherConfig(accumulate_grad_batches=2)` to simulate larger batches without extra VRAM.
  - Limit `max_in_flight_plans` to 1 and tune `train_fraction` to keep per-plan memory manageable.
  - Keep dataloader workers low (`num_workers=1-2`) and disable shard shuffling while tuning.
- Prefer sequential cohorts of ~120 k rows (`max_total_rows=120_000`, `max_shards≈32`, `batch_size≈48`) when using the vintage-age parquet; the smoke run completed in ~33 min with roc_auc≈0.49 and stayed under the 6 GB VRAM ceiling.
- Enable peak GPU telemetry via `StreamingWorkerConfig.gpu_memory_monitor_interval_seconds` (e.g., 30 s probes); results surface in the worker log and `ml_tft_streaming_worker_gpu_peak_mb` gauge.
- Latest GPU cohorts (2025-10-19/20) ran the vintage-age slice with caps (`max_total_rows=120_000`, `max_shards=32`): the 1-epoch sanity pass (plan `full_tft_95-05c32e888427`) landed at roc_auc≈0.496 with peak GPU 1 899 MB, while the 2-epoch follow-up (plan `full_tft_95-1763b898c3ad`) improved to roc_auc≈0.519 with peak GPU 1.88 GB; manifests for both runs live under `ml_out/tft_streaming_artifacts/full_tft_95/`. The earlier CPU smoke (plan `full_tft_95-b754461da1fd`) remains a reference for sequential fallback expectations.
- To reproduce the cohort locally (slow but deterministic), run:
  ```bash
  poetry run python -m ml.scripts.run_streaming_cohort \
      --dataset-dir ml_out/full_tft_95 \
      --output-dir ml_out/tft_streaming_artifacts/full_tft_95 \
      --state-path ml_out/streaming_training_state_snapshot.json \
      --max-total-rows 120000 \
      --max-total-sequences 90000 \
      --max-shards 32 \
      --batch-size 48 \
      --accelerator cpu
  ```
  The script plans using the production metadata, enforces the sequential cohort limits, trains the TFT teacher (CPU fallback), and prints the metrics/GPU peak for documentation.
- Expect multi-hour runs; monitor `ml_tft_streaming_training_backlog` and `ml_tft_streaming_workers_active` via the dashboard and Prometheus alerts (warning at backlog ≥4, critical at ≥8; worker critical when active count drops to zero).

### From CLI Smoke to Production Stack

Once the CLI smoke paths above are green, migrate to the fully event-driven topology so training runs continuously inside the `ml_training` container:

1. **Provision services**
   - Launch the streaming dataset planner, orchestrator, worker(s), and persistence worker inside `ml_training` (Docker Compose/Kubernetes). Ensure Redis/Kafka (or the configured message bus) is accessible and `MessageBusConfig.from_env()` is populated via container environment variables.
   - Mount the production dataset (`ml_out/full_tft_95` or its remote equivalent) read-only for planners/workers; mount registry roots (`~/.nautilus/ml/models`, feature registry, etc.) read-write so logits/metadata persist.

2. **Configure runtime**
   - Set `StreamingWorkerConfig` values (caps, `max_epochs`, accelerator/devices, GPU monitor interval) through container env or configuration files.
   - Keep dataset service shard budgets aligned with planner expectations; ensure orchestration gates (retry limits, max in-flight plans) reflect desired concurrency.

3. **Start the event loop**
   - Orchestrator publishes `DatasetPlanRequest` messages on the configured bus topics.
   - Dataset planner consumes requests, scans parquet, applies limits, and emits `DatasetPlanEvent` payloads.
   - Streaming workers consume plans, run Lightning-backed TFT training, emit `TrainingResultEvent` (metrics, telemetry, logits URI), and write artifacts under the shared registry path.
   - Persistence worker tails the bus, writes `ml_out/streaming_training_state.json`, and updates Prometheus gauges (`ml_tft_streaming_training_backlog`, `ml_tft_streaming_worker_gpu_peak_mb`, etc.).

4. **Monitor and iterate**
   - Use the dashboard (`/api/training/streaming/state`) and Prometheus metrics to verify backlog stays bounded, GPU usage is within budget, and validation metrics climb toward promotion thresholds.
   - Adjust worker/device count, `max_total_rows`, and `max_epochs` as needed; promote the best-performing teacher via the model registry once gates (e.g., roc_auc ≥ target) are satisfied.
   - The Compose profile sets `--max-plans 0` with a one-hour interval, so the runner loops indefinitely; set `ML_STREAMING_MAX_PLANS=1` (or override the CLI flags) when you only want a single cohort.

5. **Actor integration**
   - Update actor configurations (or auto-promotion rules) to pull logits from the registry model ID produced by the streaming stack.
   - Run `validate-events`, `validate-metrics`, and focused pytest suites before promoting to production.

### Operating Modes & Scheduling

- **Continuous loop:** the compose profile exports `ML_STREAMING_MAX_PLANS=0` and `ML_STREAMING_PLAN_INTERVAL_SECONDS=900`, so the runner submits a new plan roughly every 15 minutes. Leave `max_plans` unset (`0`) for steady-state training and tune the interval higher only if GPU utilisation threatens other jobs.
- **Single cohort:** set `ML_STREAMING_MAX_PLANS=1` (or launch the CLI with `--max-plans 1`) when you need a one-shot validation. Combine with `docker compose ... up --abort-on-container-exit` for deterministic CI-style runs that terminate once the manifest is written.
- **Accelerator overrides:** switch to CPU fallback via `ML_STREAMING_ACCELERATOR=cpu` and `ML_STREAMING_DEVICES=1` when GPUs are absent, or bump device count when orchestrating multiple GPUs. Always verify `nvidia-smi` inside the container to confirm CUDA visibility before trusting the metrics.
- **Interval tuning:** shorten `ML_STREAMING_PLAN_INTERVAL_SECONDS` during iteration to collect faster feedback, but restore the production cadence before promoting so manifests capture representative resource usage. Keep the interval above the worst-case cohort runtime plus a buffer to avoid overlapping plans that would trip backlog alerts (e.g., stay ≥ cohort runtime + 5 min).
- **Adaptive guards:** configure `--adaptive-backlog-threshold` (or `ML_STREAMING_ADAPTIVE_BACKLOG_THRESHOLD`) to defer new plans when outstanding cohorts reach the configured ceiling. Combine with `--adaptive-gpu-threshold-mb`, `--adaptive-cooldown-seconds`, and `--adaptive-interval-multiplier` to scale the plan cadence automatically after GPU-heavy cohorts; the runner guarantees the adjusted interval is at least the cooldown duration.
- **Resource caps:** the environment (.env / .env.example) exposes `ML_STREAMING_MAX_TOTAL_ROWS`, `ML_STREAMING_MAX_TOTAL_SEQUENCES`, `ML_STREAMING_MAX_SHARDS`, and `ML_STREAMING_MAX_EPOCHS`; adjust them instead of editing code when testing alternative budgets. `StreamingWorkerConfig.from_env()` now mirrors these overrides so the runner picks up new caps, retry budgets, and GPU monitor cadences without touching Python.
- **CPU override compose:** merge `docker-compose.cpu.yml` (`docker compose -f ml/deployment/docker-compose.yml -f ml/deployment/docker-compose.cpu.yml up`) when you need a GPU-free single-shot run; the override drops GPU reservations and forces `ML_STREAMING_ACCELERATOR=cpu`.

### Promotion Gates & Metrics

- **Primary metric:** require the streaming teacher to exceed the historical ROC-AUC baseline (for example, `roc_auc ≥ 0.55` if the prior teacher sits near 0.52).
- **Threshold config:** manage gates via `StreamingPromotionConfig` (env overrides `ML_STREAMING_PROMOTE_MIN_ROC_AUC`, `ML_STREAMING_PROMOTE_MIN_PR_AUC_MULTIPLE`, `ML_STREAMING_PROMOTE_MAX_LOG_LOSS`, `ML_STREAMING_PROMOTE_MIN_SLIPPAGE_SHARPE`, `ML_STREAMING_PROMOTE_MIN_HIT_RATE`, `ML_STREAMING_PROMOTE_MAX_TURNOVER`, `ML_STREAMING_PROMOTE_MAX_DRAWDOWN`, `ML_STREAMING_PROMOTE_MAX_KS_STATISTIC`, `ML_STREAMING_PROMOTE_MAX_CALIBRATION_DRIFT`). Default baselines are `roc_auc ≥ 0.55`, `pr_auc_multiple ≥ 1.1`, and `log_loss ≤ 0.75`; tighten them via env or CLI when promotion criteria evolve. The streaming runner automatically converts these settings into promotion checks when CLI flags are absent so automation, dashboards, and manual reviews stay aligned.
- **Forward returns:** economic and stability telemetry rely on the `validation_return_column` (default `forward_return`). Update via `ML_STREAMING_VALIDATION_RETURN_COLUMN` or `--validation-return-column` if the dataset emits a different field so Sharpe/hit-rate/turnover metrics remain populated.
- **Secondary metrics:** track PR-AUC/PR@K, log-loss, calibration error, and instrument-level lift. The streaming worker now persists `pr_auc`, `pr_auc_multiple`, `log_loss`, `brier_score`, and `calibration_ece_*` alongside `roc_auc`, so the manifest exposes a consistent schema for promotion tooling. Reject cohorts that improve headline ROC-AUC but regress on critical slices even if the aggregate metric clears the gate.
- **Resource checks:** place hard ceilings on wall-clock runtime and GPU memory (`resources.max_gpu_memory_mb`) so the loop respects cluster budgets.
- **Regression guardrails:** compare against the previously promoted plan and block cohorts that regress more than an agreed delta even if they clear absolute gates.
- **Business validation:** run shadow actors/backtests to confirm signal lift (Sharpe/PnL) before promotion; encode the checks in promotion tooling (`ml/cli/promote_model_if_metrics_pass.py` now accepts both `--manifest {manifest}` and `--teacher-npz {logits}` so economic/stability metrics gate alongside ROC/PR).
- **Automation hooks:** export `ML_STREAMING_PROMOTION_THRESHOLD` in `.env` (primary ROC-AUC gate) and use `ML_STREAMING_PROMOTION_COMMAND` (or `--promotion-command` CLI flag) to let the runner invoke automation when cohorts clear the gate. Commands accept placeholders like `{manifest}`, `{logits}`, `{model_id}`, `{plan_id}`, and `{dataset_id}`; a typical value is `poetry run python -m ml.cli.promote_model_if_metrics_pass --manifest {manifest} --teacher-npz {logits}`. The default command (`true`) acts as a no-op so the runner can launch without automation; replace it with your real promotion script once guardrails are ready. The command exits non-zero when gates fail so you can block registry promotion, page operators, or enqueue a fallback cohort automatically.
- **Registry hygiene:** surface the promoted `model_id` and hash in the manifest, update `~/.nautilus/ml/models/registry.json`, and prune superseded staging entries once a teacher is promoted to prevent stale logits from bloating the registry volume.
- **Secondary gates:** configure additional checks with `--promotion-metric-check` (repeatable). Example: `--promotion-metric-check pr_auc>=0.60 --promotion-metric-check calibration_ece_20<=0.05` ensures PR lift and calibration pass before automation fires. The runner backfills missing metrics from the logits artifact before evaluating gates, so secondary checks always see a complete metric set.

Promotion automation flow:
1. Allow the loop to emit a manifest per cohort (written under `${ML_STREAMING_OUTPUT_DIR}`).
2. Configure `ML_STREAMING_PROMOTION_COMMAND="poetry run python -m ml.cli.promote_model_if_metrics_pass --manifest {manifest} --teacher-npz {logits}"` (or an equivalent script) so the runner promotes automatically when the gate passes.
3. Feed the manifest into a watcher (Make target, cron, or CI job) for additional business checks; if automation fails, rerun `ml/cli/promote_model_if_metrics_pass.py` manually, update the registry via `ModelRegistry.promote`, and notify actor owners with the manifest excerpt (metrics, resource usage, dataset signature).

Running inside this continuous topology is the final “done” state: no manual CLI invocations, plans flow through the bus, workers train in place, registries/stores stay current, and actors observe steadily refreshed models.
The `ml.cli.streaming_training_runner` (wired into the `streaming_training_runner` compose service) now wraps these steps: it loads the production dataset, schedules plans via `InMemoryStreamingOrchestrator`, runs the Lightning worker, publishes events to Redis, copies logits into the registry, and writes per-plan manifests. Configuration lives under the `ML_STREAMING_*` environment variables so operators can adjust caps/epochs/devices without editing the codebase.

#### Roadmap to the Intelligent TFT Model

- **Phase 1 – Feature enrichment:** expand the dataset planner to emit macro deltas, calendar lag windows, clustering tags, and any additional context features the “Intelligent” teacher requires. Update `TFTStreamingConfig` schemas alongside property/contract tests so new signals stay leakage-free.
- 2025-10-25 update: the planner now attaches a `phase_one_signals` payload to `DatasetPlanEvent` / `StreamingPlanMessage` containing `macro_delta_columns`, `calendar_lag_columns`, `clustering_tag_columns`, and `context_feature_columns`. These lists flow through `TFTStreamingConfig.phase_one_signals` so the runner and downstream actors can assert training/inference parity on the enriched feature families.
- New configuration toggles (`ML_STREAMING_INCLUDE_MACRO_DELTAS`, `ML_STREAMING_INCLUDE_CALENDAR_LAGS`, `ML_STREAMING_INCLUDE_CLUSTERING_TAGS`, `ML_STREAMING_INCLUDE_CONTEXT_FEATURES`) allow operators to enable the Phase 1 feature families explicitly across `DatasetServiceConfig`, CLI flags, and capability telemetry.
- **Phase 1a – Legacy feature parity:** ✅ modern builder now routes through the shared augmenters (macro/calendar/events/earnings/micro/L2) with registry capability flags and streaming configuration toggles (`TFTStreamingConfig`, `DatasetServiceConfig`, CLI/env). Coverage now includes macro revision defaults and student-mode overrides (`test_component_builder_macro_revision_defaults`, `test_component_builder_student_mode_forces_feature_flags`), alongside planner enforcement of service-level toggles. Latest artefacts: `ml/tests/validation_reports/phase_1a/macro_only_v1_summary.md`, `macro_revisions_core_summary.md`, `student_lightweight_summary.md`, `calendar_events_summary.md`, and `earnings_summary.md`. Placeholders (`micro_summary.md`, `l2_summary.md`) capture the outstanding dependency on tier1 micro/L2 parquet feeds. Remaining work: keep the summaries in sync when toggles change and embed the toggle matrix in operator runbooks so capability flags are visible across manifests and dashboards.
- **Phase 1b – Feature pipeline wiring (target: Weeks 2–3 after Phase 1a artefacts land; dependencies: `phase_1a` parity summaries, green CI on builder tests):**
  - **Status (2025-10-22):** Capability flags now propagate end-to-end. The TFT component builder writes per-family toggles into `dataset_metadata.json.capability_flags`, `StreamingPlanMessage` payloads expose the same `include_*` booleans for downstream parity checks, and the planner enforces service-level overrides before publishing. When `include_l2=True`, the builder/service automatically lift `include_micro=True` so microstructure caches stay warm for order-book features; student mode continues to zero out the heavy joins regardless of request flags. Validation is locked behind the guardrail suite: `poetry run mypy ml --strict`, `poetry ruff check ml`, `poetry run pytest ml/tests/unit/features/test_feature_config_macro_integration.py`, `poetry run pytest ml/tests/contracts/test_streaming_payloads.py::test_calendar_event_payload_schema`, `poetry run pytest ml/tests/unit/features/test_known_future_transforms.py`, `poetry run pytest ml/tests/unit/data/test_dataset_build_macro.py`, `poetry run pytest ml/tests/unit/data/test_tft_dataset_builder_store.py`, `poetry run pytest ml/tests/unit/features/test_microstructure.py ml/tests/unit/features/test_l2_aggregate.py`, `poetry run pytest ml/tests/unit/features/test_earnings_features.py ml/tests/unit/features/test_earnings_transforms.py`, `pytest -q ml/tests/performance -k microbench`, and `poetry run pytest ml/tests/integration/training/event_driven/test_plan_to_result.py::test_plan_worker_round_trip ml/tests/integration/training/event_driven/test_plan_to_result.py::test_streaming_pipeline_records_gpu_telemetry`.
  1. Macro regression parity (Week 2 Day 1–2): ✅ keep `poetry run pytest ml/tests/unit/features/test_feature_config_macro_integration.py` and `pytest ml/tests/unit/data/test_tft_dataset_builder_store.py::test_component_builder_respects_include_flags` green so builder and pipeline specs stay aligned. Refresh the parity artefacts under `ml/tests/validation_reports/phase_1a/` (e.g., `macro_only_v1_summary.md`, `macro_revisions_core_summary.md`) whenever macro series, revision defaults, or capability flags shift, and update this section with the new evidence.  
  2. Calendar/events (Week 2 Day 3): reuse `ml/features/engineering.py` helpers and extend Pandera schemas under `ml/tests/contracts/` to catch missing future columns early. Wire CLI/env toggles through `ml/cli/streaming_training_runner.py` and `ml/config/streaming_pipeline.py`. ✅ `DatasetServiceConfig.from_env` now mirrors the CLI toggles (`ML_STREAMING_INCLUDE_*`), and calendar/event breadth is guarded by `poetry run pytest ml/tests/contracts/test_streaming_payloads.py::test_calendar_event_payload_schema` plus `poetry run pytest ml/tests/unit/features/test_known_future_transforms.py`. Document any newly required columns and schema deltas here before rolling out.  
  3. Earnings (Week 3 Day 1): route `ml/features/earnings/*` through FeatureStore, covering publication lag constraints with property tests (`pytest ml/tests/unit/features/earnings/test_earnings_features.py ml/tests/unit/features/earnings/test_earnings_transforms.py`). ✅ Builder enforces non-negative `earnings_lag_days` when earnings joins are enabled (`pytest ml/tests/unit/data/test_tft_dataset_builder_store.py::test_component_builder_enforces_earnings_lag_days`); update `earnings_summary.md` once the latest cohort is staged.  
4. Micro → L2 (Week 3 Day 2–4, sequential): ✅ planner, CLI, and builders now honour `micro_base_dir` / `l2_base_dir` overrides (via `ML_STREAMING_MICRO_BASE_DIR`, `ML_STREAMING_L2_BASE_DIR`, or CLI flags) so L2 runs reuse the warmed micro cache. Guardrails: `pytest ml/tests/unit/data/test_tft_dataset_builder_store.py::test_component_builder_enforces_earnings_lag_days`, `pytest ml/tests/unit/data/test_dataset_build_macro.py::test_tft_dataset_task_config_overrides_base_dirs`, `pytest ml/tests/unit/features/test_microstructure.py`, `pytest ml/tests/unit/features/test_l2_aggregate.py`, and the persistent performance check `pytest -q ml/tests/performance -k microbench` (latest 2025-10-22T22:51:17Z run stayed under 0.25 s backlog processing with a 0.49 s suite). Capture fresh latency numbers here and in `ml/docs/ops/streaming_scaling_experiments.md` after each rerun.  
  5. Registry manifest update (Week 3 Day 5): ✅ `ml/registry/feature_registry.py` now persists capability flags per family and captures diffs under `parity_digest["capability_flags_diff"]`; rerun `poetry run pytest ml/tests/unit/registry/test_feature_registry.py::test_register_feature_set_records_capability_flag_diff` alongside the planner/worker suites (`pytest ml/tests/integration/training/event_driven/test_plan_worker_round_trip.py::test_plan_worker_round_trip` and `pytest ml/tests/integration/training/event_driven/test_plan_to_result.py::test_streaming_pipeline_records_gpu_telemetry`) to confirm registries, planner, and worker agree on feature availability.  
  6. Exit criteria: `poetry run mypy ml --strict`, `poetry ruff check ml`, targeted feature tests, refreshed artefacts under `ml/tests/validation_reports/phase_1a/` for each newly wired family, and updated documentation (this file + ops guides).
- **Phase 1c – Validation & benchmarking (target: Week 4; blocked by Phase 1b wiring and artefact refresh):**  
  1. ✅ Leakage guard (Week 4 Day 1): publication lag validators now live in `ml.features.validation.validate_known_future_effective_times`, exercised within `ml/data/fred_join.py` and the earnings builder. Guardrails cover `ml/tests/unit/features/test_known_future_transforms.py`, `ml/tests/unit/features/earnings/test_parity.py`, and the negative config case in `ml/tests/contracts/test_streaming_payloads.py`.  
  2. Artefact diffs (Week 4 Day 2): re-run the parity script for every newly wired feature family and persist Markdown reports under `ml/tests/validation_reports/phase_1a/` (replace the current `micro_summary.md` / `l2_summary.md` placeholders once tier1 feeds are online). Update `parity_summary.md` with the new scenarios and link the artefacts in commit notes.  
3. Performance envelope (Week 4 Day 3): execute `pytest -q ml/tests/performance -k microbench` and record P99 latency in this doc plus `ml/docs/ops/streaming_scaling_experiments.md`. Latest run (2025-10-22T22:51:17Z UTC) cleared in 0.49 s with teardown-bound microbench tests and the persistence guard still asserting <0.25 s backlog processing; refresh the numbers after each rerun and keep `pytest -q ml/tests/performance/test_streaming_persistence_microbench.py` in the validation rotation to guard hot-path budgets.  
  4. Benchmark manifests (Week 4 Day 4): summarize the latest cohorts (`full_tft_95-458c9417c1d7_manifest.json`, `full_tft_95-2623b7a7c6a5_manifest.json`, plus the earlier baselines) with `poetry run python -m ml.scripts.summarize_streaming_manifests --manifest-dir ml_out/tft_streaming_artifacts/full_tft_95 --limit 10` and paste GPU/metric deltas into `ml/docs/ops/streaming_scaling_experiments.md`. Check the resulting table into the repo.  
  5. Observability & promotion gates (Week 4 Day 5): run `pytest -q ml/tests/integration/training/event_driven/test_plan_to_result.py::test_streaming_pipeline_records_gpu_telemetry` plus `make validate-metrics` and `make validate-events`. Confirm the promotion threshold (`roc_auc ≥ 0.55` placeholder) remains recorded in manifests and runbooks before declaring Phase 1 complete; simulate a failing promotion command via `poetry run python -m ml.cli.promote_model_if_metrics_pass --manifest <bad manifest> --teacher-npz <bad path>` to ensure fallbacks are documented.
- **Phase 2 – Adaptive scheduling:** ✅ (2025-10-22T23:05Z) runner/orchestrator now adapt to backlog and GPU telemetry. `TrainingOrchestratorConfig` exposes `adaptive_backlog_threshold`, `adaptive_gpu_threshold_mb`, `adaptive_cooldown_seconds`, and `adaptive_interval_multiplier`; CLI flags (`--adaptive-backlog-threshold`, `--adaptive-gpu-threshold-mb`, `--adaptive-cooldown-seconds`, `--adaptive-interval-multiplier`) surface the knobs. `InMemoryStreamingOrchestrator` updates `ml_tft_streaming_training_backlog{dataset_id}` directly so dashboards react without the persistence worker, and the runner defers plan launches when backlog saturates or scales plan intervals when GPU peaks breach the guard. Coverage: `pytest -q ml/tests/unit/config/test_streaming_pipeline_config.py`, `pytest -q ml/tests/unit/cli/test_streaming_training_runner_adaptive.py`, `pytest -q ml/tests/integration/training/event_driven/test_plan_to_result.py::test_streaming_pipeline_records_gpu_telemetry`, `pytest -q ml/tests/performance -k microbench`, `poetry run mypy ml --strict`, and `poetry run ruff check ml`.
  - **Phase 3 – Intelligent promotion (prereqs: adaptive scheduler metrics, registry capability flags):** enhance `ml/cli/promote_model_if_metrics_pass.py`, streaming runner promotion hooks, and ModelRegistry (`ml/stores/model_store.py`, `ml/registry/feature_registry.py`) to enforce multi-metric gates. Cover automation with `pytest ml/tests/integration/cli/test_streaming_persistence_worker_cli.py`, `pytest ml/tests/unit/cli/test_streaming_training_runner_metrics.py`, and promotion manifest contract tests. Document promotion thresholds in `.env`, config classes, and this doc so on-call can trace gate failures quickly.
- **Phase 4 – Production validation (prereqs: promotion automation + adaptive scheduling):** formalise shadow actor/backtest loops, extend FeatureStore parity suites, and run `pytest -q ml/tests/performance` (including microbenchmarks) before promoting the “Intelligent TFT Model.” Archive validation artefacts under `ml/tests/validation_reports/phase_1a/` (or a new phase directory), refresh `ml/docs/ops/streaming_scaling_experiments.md`, keep `make validate-metrics` / `make validate-events` green prior to prod enablement, and capture business validation evidence in `ml/docs/ops/streaming_scaling_experiments.md`.

#### Manifest metric interpretation

- `roc_auc`: headline promotion gate; regression of ≥0.01 vs. current teacher triggers manual review even if absolute threshold passes.
- `pr_auc` / `pr_auc_multiple`: capture class-imbalance sensitivity; watch `pr_auc_multiple` for slice stability (values <1.0 indicate regression vs. baseline).
- `log_loss` and `brier_score`: confirm calibration improvements; a decreasing `log_loss` with rising `brier_score` usually means overconfident predictions.
- `calibration_ece_*`: ensure expected calibration error monotonicity; values above 0.05 warrant recalibration or temperature tuning before deployment.
- `resources.max_gpu_memory_mb`: cross-check against GPU quotas; anything within 10% of the hard ceiling should trigger a backlog review or worker concurrency adjustment before promoting.

#### Manifest summarizer CLI

- Generate cohort summaries for ops docs and retros with:
  ```bash
  poetry run python -m ml.scripts.summarize_streaming_manifests \
      --manifest-dir ml_out/tft_streaming_artifacts/full_tft_95 \
      --limit 10
  ```
- The CLI prints Markdown suitable for `ml/docs/ops/streaming_scaling_experiments.md`; commit the table after every cohort batch so runbooks mirror actual GPU usage and promotion metrics.

### Troubleshooting Guide

- **Redis backpressure**
  - Symptoms: rising `ml_tft_streaming_training_backlog`, Redis stream depth >10k, or delayed `TrainingResultEvent` delivery.
  - Actions: scale the persistence worker (increase consumers or bump `max_fetch_batch`), flush stale plans via `ml.cli.streaming_persistence_worker --trim`, and ensure `StreamingWorkerConfig.max_concurrent_jobs` aligns with orchestrator `max_in_flight_plans`. Backpressure persisting >15 min should raise a P1 and pause new plan submissions via `ML_STREAMING_MAX_PLANS=0`.
- **Promotion automation flow stalls**
  - Verify `streaming_training_runner` logs for `promotion_command_placeholder_missing` or non-zero return codes. Re-run the resolved command from container shell (see `_run_promotion_command`) and capture stdout/stderr for the incident doc. If the command fails due to metrics regression, annotate the manifest with the gate that tripped and enqueue the fallback cohort (`PRIMARY → CACHED → FILE → DUMMY`) to keep the loop alive.
  - When automation is intentionally paused, set `ML_STREAMING_PROMOTION_COMMAND=` (empty) and document the reason in the dashboard annotation so on-call knows manual promotion is required.
- **Observer dashboard gaps**
  - If `/api/training/streaming/state` renders stale data, confirm the persistence worker is writing `ml_out/streaming_training_state.json` and that the dashboard pod mounts the same volume. Reload Prometheus rules for `ml_tft_streaming_worker_gpu_peak_mb` and backlog gauges if charts freeze; missing metrics usually indicate exporter scrape failures or renamed labels.
  - Use `make validate-metrics` and `validate-events` before declaring the dashboard healthy; schema drift in payload DTOs is a common cause of missing widgets.

### Operational Playbooks

- **Promotion command failed**
  - Inspect `docker compose logs streaming_training_runner` for the offending plan ID and copy the resolved command from the `promotion_command` log line.
  - Run the command manually with the manifest/logits paths to confirm reproducibility; if failure persists, fall back to `ModelRegistry.promote` with explicit cohort metadata and raise `ml_fallback_activations_total{type="promotion_command"}` for observability.
  - Record the incident in `ml/docs/ops/streaming_scaling_experiments.md` plus the architecture doc with remediation steps, then re-enable automation once green.
- **Message-bus retry counter spikes (`ml_tft_streaming_bus_publish_attempts_total`)**
  - Check metric labels to determine failure mode (`outcome="retry"` vs. `"success"`). Sustained retries imply Redis/Kafka is overloaded or credentials expired.
  - Flush retry buffers by restarting the orchestrator after persisting state, validate `MessageBusConfig.from_env()` values, and run `poetry run python -m ml.cli.streaming_training_runner --dry-run --max-plans 1` to confirm publishing succeeds before resuming the loop.
  - Document the spike, cause, and mitigation in the ops changelog; repeated incidents should trigger a scaling review for the bus cluster.

### Training Pipeline (Summary)

1. **Planner** scans `dataset.parquet`, applies caps, and emits `DatasetPlanEvent`.
2. **Orchestrator** tracks plan lifecycle, publishes events, and handles retries/backlog.
3. **Worker** builds streaming dataloaders, fits the TFT teacher, writes logits/telemetry, and reports Prometheus metrics.
4. **Persistence Worker** mirrors plan/result/heartbeat payloads to the dashboard state file (`ml_out/streaming_training_state.json`) and updates gauges.

#### Redis cursor persistence

- Snapshot schema now includes a `stream_cursor` field. The persistence worker seeds Redis `XREAD` calls with that cursor (defaults to `$` when empty), so restarts resume from the last acknowledged ID instead of tailing only fresh messages.
- The cursor is written after every successful poll and exposed via `/api/training/streaming/state` (`stream_cursor` key) to simplify health checks.
- To replay the entire stream, stop the worker, set `stream_cursor` to `"0-0"` (or remove the file entirely), and restart. To skip ahead, overwrite it with any valid Redis stream ID.
- Telemetry/state snapshots live at both `ml_out/streaming_training_state_snapshot.json` (local runs) and `/app/ml_out/streaming_training_state.json` (Compose volume). Operators can run `jq '.stream_cursor' /app/ml_out/streaming_training_state.json` or `curl -s http://localhost:8010/api/training/streaming/state | jq '.stream_cursor'` to confirm the persisted cursor advances after each batch.

#### Ensemble alignment guardrails

- Every logits artifact now stores per-row identifiers: `train_row_ids`, `train_instrument_ids`, `train_time_indices`, plus the validation equivalents. Identifiers follow the `instrument_id::time_index` format so peers can align rows deterministically without relying on positional lengths.
- The streaming worker validates peer metadata before blending. Optional members with mismatched/missing metadata are skipped (and counted via `ensemble_optional_members_skipped`), while required members raise immediately. Each mismatch increments `ensemble_members_misaligned`, which flows through training metrics and now renders on the dashboard (⚠ flag next to the ensemble summary).
- Operators can confirm alignment by inspecting artifacts, e.g.:

  ```bash
  python - <<'PY'
  import numpy as np

  payload = np.load("ml_out/tft_streaming_artifacts/full_tft_95/full_tft_95-098aefc6abb0_logits.npz", allow_pickle=False)
  print(payload["val_row_ids"][:5])
  PY
  ```

  Matching `row_ids` between the base run and peer artifacts is required before enabling a peer in production.
5. **Dashboard** consumes the snapshot to render backlog/worker widgets; alerts in `ml/deployment/alerts.yml` trigger on high backlog or missing workers.

### Streaming Training Workflow Tips

- Before the first full run, start with a smaller cohort (e.g., 10–20 symbols or a shorter date range) to verify configs and hardware headroom. Use `StreamingWorkerConfig.max_total_rows`/`max_shards` to cap slices.
- Attach `LightningStreamingWorker` to `InMemoryStreamingOrchestrator`, submit a `DatasetPlanRequest`, monitor the plan via `/api/training/streaming/state`, and inspect the resulting logits file and `TrainingResultEvent.metrics`.
- The dashboard state API now mirrors `telemetry.resources.max_gpu_memory_mb` for the latest result per dataset—capture the value in post-run reports so GPU headroom trends stay visible alongside backlog metrics.
- Log actual backlog/worker telemetry into `ml/docs/ops/streaming_scaling_experiments.md` monthly to confirm thresholds.

### Manifest Telemetry & Reporting

- **Fields to watch:** `cohort_run.metrics` carries validation metrics, while `cohort_run.telemetry` captures selected rows/sequences, shard counts, and `resources.max_gpu_memory_mb`. Treat the manifest as the source of truth when deciding whether a cohort is promotable.
- **Advanced metrics:** extend the manifest by piping the logits through `ml.common.metrics_bootstrap.compute_classification_metrics` to append PR-AUC, calibration buckets, and log-loss. Persist the enriched dict under `cohort_run.metrics` so downstream tooling (promotion gate, dashboards) reads a consistent schema.
- **Quick roll-up:** run `poetry run python -m ml.scripts.summarize_streaming_manifests --manifest-dir ml_out/tft_streaming_artifacts/full_tft_95 --limit 10` to emit a Markdown table covering ROC-AUC, PR-AUC, calibration, and GPU peaks. Paste the output into `ml/docs/ops/streaming_scaling_experiments.md` after each cohort batch.
- **Dashboards:** mirror key manifest fields into the persistence snapshot so Grafana panels display current ROC-AUC, PR-AUC, and GPU peaks. A lightweight cronjob can parse manifests and push the aggregates into `ml/tests/validation_reports/` for retention.

### Troubleshooting & Recovery

- **Redis publish backpressure:** the warning `streaming orchestrator failed to publish event after 3 attempts` indicates Redis is unreachable or saturated. Verify `ML_BUS_ENABLE=1`, inspect `docker compose -f ml/deployment/docker-compose.yml logs redis`, and confirm the stream configured via `ML_BUS_REDIS_STREAM` exists (`redis-cli XINFO STREAM <stream>`). Tune `ML_STREAMING_BUS_RETRY_ATTEMPTS` / `ML_STREAMING_BUS_RETRY_DELAY_SECONDS` to widen retry windows, and watch `ml_tft_streaming_bus_publish_attempts_total{outcome}` for spikes. Reduce `max_in_flight_plans`, or bump `ML_STREAMING_PLAN_INTERVAL_SECONDS` if the bus floods during retries.
- **Training restarts or container exits:** if the runner restarts mid-plan, check `docker compose ... logs streaming_training_runner` for CUDA OOM or heartbeat timeouts, then dial down `ML_STREAMING_MAX_TOTAL_ROWS`/`ML_STREAMING_MAX_SHARDS` or increase the `--worker-timeout-seconds` flag (add `ML_STREAMING_WORKER_TIMEOUT_SECONDS` to an override compose file if you need a persistent change). Use `nvidia-smi` to validate that GPU memory drops back to baseline after each retry.
- **Missing manifests/logits:** on failure the runner leaves partial artefacts; clean the corresponding plan directory under `${ML_STREAMING_OUTPUT_DIR}` and rerun the plan (set `ML_STREAMING_MAX_PLANS=1` and `ML_STREAMING_PLAN_INTERVAL_SECONDS=0` temporarily) to avoid promoting stale data.
- **Persistence lag:** if the dashboard snapshot stops updating, ensure `ML_STREAM_PERSIST_ENABLE=1`, restart the persistence worker, and flush Redis backlog (`redis-cli XTRIM <stream> MAXLEN ~ 10000`) before resuming.

### Operational Runbook

- **Grafana dashboards:** track `ml_tft_streaming_training_runs_total`, `ml_tft_streaming_training_duration_seconds`, `ml_tft_streaming_training_backlog`, and `ml_tft_streaming_workers_active`. Alert when backlog ≥4 for two consecutive intervals or when active workers drop to zero.
- **Alert response:** investigate alert spikes by correlating Grafana with manifest timestamps; if ROC-AUC dips or GPU usage spikes, pause the loop (`docker compose ... down`) and rerun a capped cohort to establish baselines before resuming.
- **Failed publishes:** wrap bus publishes in the existing retry helpers and log with `exc_info=True` plus `extra={"plan_id": ..., "topic": ...}`. If Redis remains unavailable, set `ML_BUS_ENABLE=0` to force the runner into local fallback mode while infrastructure recovers, and record the manual intervention in the manifest notes.
- **Registry updates:** after promotion, run `poetry run mypy ml --strict`, `poetry ruff check ml`, and targeted pytest suites. Document the promoted plan, metrics, and resource footprint in `ml/docs/architecture/event_driven_streaming_plan.md` so future operators know the current teacher baseline.

### Message Contracts

- Topic builder: `ml.common.message_topics.build_topic_for_stage(config.stage, ...)`.  
- Events use the schema defined in `ml/training/event_driven/payloads.py` (`schema_version="1.0.0"`) with deterministic `correlation_id`, dataset metadata, and stage/source/status alignment.  
- Message types:
  - `StreamingPlanMessage`: dataset caps/limits, parquet path, streaming config snapshot.  
  - `StreamingResultMessage`: model id, telemetry snapshot, artifact URIs, validation metrics.  
  - `StreamingHeartbeatMessage`: worker id, progress, RSS, shard counters with stage-aware status (`success`/`partial`).  
- Failed publishes are buffered for observability and can be retried, mirroring the broader engine bus semantics.

### Configuration

- `DatasetServiceConfig`: parquet root, shard budgets, retry policy, cache TTL.  
- `StreamingWorkerConfig`: `max_total_rows`, `max_total_sequences`, `max_shards`, `max_epochs`, runtime/accelerator bounds, validation metric selection, artifact key naming, calibration toggles (`enable_temperature_calibration`, `temperature_calibration_*`, `enable_platt_calibration`, `enable_isotonic_calibration`), plus retry controls (`max_retry_attempts`, `retry_backoff_seconds`).  
- `TrainingOrchestratorConfig`: concurrency, command topic, heartbeat interval, fallback policies, dataset retry limits feeding bus retry attempts.

### Milestones

1. **M4 Dataset Planner (Weeks 1–2)**  
   - Deliver dataset service skeleton + config validation.  
   - Contract tests for `DatasetPlanEvent`.  
   - CI target: `pytest -q ml/tests/unit/pipeline/test_dataset_service.py`.

2. **M5 Worker Container & Telemetry (Weeks 3–4)**  
   - Worker skeleton with bounded Lightning loop, streaming telemetry integration.  
   - Add micro-bench hook for shard resume time.  
   - Metrics validation via `make validate-metrics` and Prometheus counters/histograms for streaming runs.

3. **M6 Orchestrator Integration (Weeks 5–6)**  
   - Command handling, state machine, message bus config parity.  
  - Integration tests to replay dataset → worker workflow using in-memory bus.

4. **M7 Observability & Scaling (Weeks 7–8)**  
   - Dashboard cards: shard backlog, worker utilization, fallback counters.  
   - Multi-worker scaling plan: shard queue with explicit checkpoint partitions.  
   - Document recommended cap presets per hardware tier.

### Testing Status

- **Unit coverage delivered**: planner (`ml/tests/unit/training/event_driven/test_dataset_service.py`), orchestrator/backpressure (`test_orchestrator.py`), worker retries (`test_worker.py`), bus serialization (`test_bus.py`), Redis persistence consumers (`ml/tests/unit/consumers/test_streaming_training_worker.py`, `test_streaming_training_service.py`), configuration + CLI validation.
- **Integration progress**:
  - ✅ Contract/schema verification for streaming payloads (`ml/tests/contracts/test_streaming_payloads.py`).
  - ✅ In-memory orchestration flow covering planner → bus → worker (`ml/tests/integration/training/event_driven/test_plan_to_result.py`).
  - ✅ Streaming cohort smoke with GPU telemetry and persistence validation (`ml/tests/integration/training/event_driven/test_plan_to_result.py::test_streaming_pipeline_records_gpu_telemetry`).
  - ✅ Redis Streams persistence worker integration using stubbed Redis (`ml/tests/integration/consumers/test_streaming_persistence_integration.py`).
  - ✅ CLI smoke test asserting streaming summary artefacts and state snapshots (`ml/tests/integration/cli/test_streaming_persistence_worker_cli.py`).
  - ✅ Dashboard state endpoint contract test (`ml/tests/integration/dashboard/test_streaming_state_endpoint.py`).
  - ✅ Microbenchmark covering persistence backlog throughput (`ml/tests/performance/test_streaming_persistence_microbench.py`).
  - ✅ Dashboard streaming endpoint now exposes dataset summaries (plan/result recency, worker counts, backlog) for UI widgets.
  - ✅ Enhanced dashboard renders streaming backlog + worker widgets (`ml/dashboard/templates/index_enhanced.html`).

### Next Actions

### Immediate Checklist (Redis + Curriculum/Ensemble/AMP)

- [x] Bake the `redis` client into `Dockerfile.streaming` / `Dockerfile.pipeline` so `_imports.HAS_REDIS` is set inside containers.
- [x] Rebuild and push the refreshed images: `docker compose -f ml/deployment/docker-compose.yml build streaming_training_runner streaming_persistence_worker ml_dashboard` (verified `python -c "import redis; print(redis.__version__)` returns `7.0.0` inside both streaming containers).
- [x] Run a calibrated cohort with curriculum, ensemble members, and AMP enabled via `poetry run python -m ml.cli.streaming_training_runner ... --enable-curriculum --curriculum-stage 2000:0.55 --curriculum-stage '*:0.75' --enable-ensemble --enable-amp --enable-temperature-calibration --enable-platt-calibration --enable-isotonic-calibration`. Archived plan `full_tft_95-098aefc6abb0` plus manifest/logits under `ml_out/tft_streaming_artifacts/full_tft_95/`.
- [x] Validate persistence + dashboards: inspected `ml_out/streaming_training_state_snapshot.json` and `/app/ml_out/streaming_training_state.json` for `worker_train_fraction`, curriculum/ensemble booleans, GPU peak metrics; `curl http://localhost:8010/api/training/streaming/state | jq` now shows the same telemetry and outstanding-plan summary.
- [x] Backfill telemetry/doc evidence once the cohort lands: `ml/docs/ops/streaming_scaling_experiments.md` manifest table updated, Ops checklist marked complete, and this document references the new artifacts and state snapshots.

### Next Actions

1. Continue the Model & Training Enhancements roadmap: curriculum and ensemble exploration, AMP/mixed-precision gating, economic metrics, stability diagnostics, and the validate-wave target, documenting promotion criteria as each increment lands.
2. Backfill calibrated telemetry into `ml/docs/ops/streaming_scaling_experiments.md` after each sweep/monthly cohort and archive summaries under `ml/tests/validation_reports/` for promotion audits.
3. Evaluate auto-scaling hooks (worker pool / orchestrator knobs) once live backlog metrics are in place and document recommended rollout steps.
4. ✅ Close the Redis backlog gap: `RedisStreamsConsumer` and the persistence worker now persist the Redis stream cursor (`stream_cursor`) inside `ml_out/streaming_training_state*.json`. On restart the worker seeds `XREAD` with that stored ID so backlog replays automatically; see “Redis cursor persistence” below for reset/validation steps.
5. ✅ Fix ensemble member alignment in the streaming worker: logits artifacts now persist per-row identifiers (instrument + time index), ensemble blending cross-checks metadata, skips/raises on mismatches, and emits `ensemble_members_misaligned` telemetry for dashboards. See “Ensemble alignment guardrails” below for operating notes.
6. Publish canonical peer logits per shard so dashboards/manifests can reference a single source of truth; store the peer inventory in `cohort_run.peer_artifacts` with metadata-complete flags.
7. Extend manifest summarizers and dashboard tables to display `ensemble_members_misaligned` so reviewers can spot misaligned peers without digging into raw JSON.
8. Add deterministic alignment tests for sliced cohorts (macro-only, events-only, micro/L2) that load regenerated peer artefacts via the new metadata CLI to guard against regressions before promoting future cohorts.
