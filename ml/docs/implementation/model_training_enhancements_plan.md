## Model & Training Enhancements Execution Plan

This document expands on the roadmap items cited in `ml/docs/architecture/event_driven_streaming_plan.md` (§“Beyond Phase 1: Smart Model Roadmap”, §“Next Actions”) and details how we will implement the remaining model/training improvements while honoring the guardrails in `AGENTS.md`.

### 1. Curriculum Scheduling & AMP Guardrails

**Goal:** Make curriculum splits and AMP usage dynamic so they react to recent cohort telemetry instead of static thresholds.

- ✅ Guards + AMP threshold implemented (2025-10-27): `StreamingWorkerConfig` exposes `curriculum.guards` and `amp_guard_threshold_mb`, new CLI/env flags (`--curriculum-guard`, `ML_STREAMING_CURRICULUM_GUARDS`, `--amp-guard-threshold-mb`, `ML_STREAMING_AMP_GUARD_THRESHOLD_MB`) landed, telemetry records `worker_curriculum_stage`, `worker_curriculum_reason`, `worker_amp_enabled`, and `worker_amp_guard_reason`, and the dashboard renders the new fields. Coverage: `ml/tests/unit/config/test_streaming_pipeline_config.py`, `ml/tests/unit/training/event_driven/test_worker.py::test_curriculum_guard_and_amp_guard_annotations`, `ml/dashboard/tests/test_streaming_monitor.py`.

- **Config Surface:**
  - Extend `StreamingWorkerConfig.curriculum` (`ml/config/streaming_pipeline.py`) with optional guard predicates (`min_rows`, `max_gpu_mb`, `min_roc_auc`, `max_backlog`).
  - Add CLI/env flags (`--curriculum-guard`, `ML_STREAMING_CURRICULUM_GUARDS`) so operators can define stage selection logic without code edits.
- **Worker Logic:**
  - Update `LightningStreamingWorker._prepare_context` (`ml/training/event_driven/worker.py`) to evaluate guards against the latest manifests/persistence snapshots. Persist the chosen stage via telemetry fields (`worker_curriculum_stage`, `worker_curriculum_reason`).
  - Introduce AMP guard thresholds (`StreamingWorkerConfig.amp_guard_threshold_mb`, CLI `--amp-guard-threshold-mb`) and downgrade precision automatically when GPU headroom < threshold.
- **Telemetry & Docs:**
  - Document guard semantics and telemetry fields in `ml/docs/architecture/event_driven_streaming_plan.md` and `ml/docs/ops/streaming_scaling_experiments.md`.
  - Surface guard decisions on the dashboard (`ml/dashboard/service.py`, `ml/dashboard/templates/index_enhanced.html`).
- **Tests:**
  - Config parsing (`ml/tests/unit/config/test_streaming_pipeline_config.py`).
  - Worker unit tests for stage selection/AMP downgrade (`ml/tests/unit/training/event_driven/test_worker.py`).
  - Integration replay (`ml/tests/integration/training/event_driven/test_plan_to_result.py`) to ensure telemetry propagation.

### 2. Ensemble Roadmap & Canonical Peer Logits

**Goal:** Guarantee ensemble peers stay aligned and observable end-to-end.

- ✅ **Peer Catalog:** Workers now emit canonical inventories for the primary run and every peer via `StreamingEnsembleTelemetry`. Each member captures weight, required flag, row counts, and skip reason so manifests/dashboards surface alignment state without ad-hoc scraping.
- ✅ **Dashboard & Summaries:** Manifest summaries (`ml/scripts/summarize_streaming_manifests.py`) and the streaming dashboard (`ml/dashboard/service.py`, `ml/dashboard/templates/index_enhanced.html`) display `ensemble_members_misaligned`, member inventories, and telemetry badges; `ml/dashboard/tests/test_streaming_monitor.py` covers the endpoint rendering.
- ✅ **Sliced Alignment Tests:** Unit coverage in `ml/tests/unit/training/event_driven/test_worker.py::test_worker_emits_economic_and_stability_metrics` and the existing ensemble fixtures assert skip reasons, inventory payloads, and misalignment counters across optional/required blends.
- ✅ **Docs:** This plan and `ml/docs/ops/streaming_scaling_experiments.md` now document the inventory schema and review workflow.

### 3. Economic & Stability Diagnostics

**Goal:** Promote only cohorts that clear economic/stability gates, not just ROC/PR metrics.

- ✅ **Telemetry Schema:** `StreamingRunTelemetry` now carries `StreamingEconomicTelemetry` and `StreamingStabilityTelemetry` payloads (see `ml/training/teacher/streaming_telemetry.py`). The worker persists Sharpe, hit rate, turnover, drawdown, and KS/calibration drift metrics via `ml/training/event_driven/economic_metrics.py`.
- ✅ **Computation:** `LightningStreamingWorker` integrates the new evaluator, emits metrics in the training result payload, and persists validation returns when available for richer diagnostics.
- ✅ **Promotion Gates:** `StreamingPromotionConfig` centralises promotion thresholds (env overrides: `ML_STREAMING_PROMOTE_*`), `ml/cli/promote_model_if_metrics_pass.py` now evaluates manifest + economic/stability metrics, and the streaming runner consumes the same config so promotion commands and scheduler gates stay aligned. Dashboard serializers surface economic/stability blocks for ops review.
- ✅ **Return Telemetry:** `LightningStreamingWorker` now backfills validation returns directly from the parquet dataset using `StreamingWorkerConfig.validation_return_column` (default `forward_return`, override via `ML_STREAMING_VALIDATION_RETURN_COLUMN`), so economic and stability metrics always populate manifests and dashboards.
- ✅ **Default Thresholds:** `StreamingPromotionConfig` ships with sensible baselines (`roc_auc ≥ 0.55`, `pr_auc_multiple ≥ 1.1`, `log_loss ≤ 0.75`); tighten or relax them through the `ML_STREAMING_PROMOTE_MIN_*`/`MAX_*` environment knobs as promotion criteria evolve.
- ✅ **Tests:** `ml/tests/unit/training/event_driven/test_worker.py::test_worker_emits_economic_and_stability_metrics` and dashboard/service contracts assert the new telemetry surfaces; payload regression tests continue to pass with the extended schema.
- ✅ 2025-10-27: Ran `poetry run python -m ml.cli.streaming_training_runner` (plan `full_tft_95-01385f0cb57c`, `--validation-return-column forward_return`, calibration defaults on) against `ml_out/full_tft_95`. Manifest and logits now contain `val_returns` (mean `1.3e-4`), and `poetry run python -m ml.cli.promote_model_if_metrics_pass --manifest ...` correctly blocked promotion (`roc_auc=0.471`, `pr_auc_multiple=0.929`, `log_loss=0.805`).
- 🔍 2025-10-27: Last 30 manifests remain bimodal (16 ≥0.55 ROC-AUC, 14 <0.55). High cohorts such as `full_tft_95-4d1ce987334f` retain rich logits (`std(z_val) ≈ 0.25`), whereas low cohorts like `full_tft_95-648b89e50dfb` collapse (`std(z_val) ≈ 0.002`). Follow-ups: diff curriculum/AMP telemetry in `ml_out/streaming_training_state_snapshot.json`, confirm seeding/loader parity, and inspect validation return distributions before tuning thresholds.
- ⚙️ 2025-10-27: Streaming loader uses a fixed `seed=7` (see `_resolve_dataset_spec`), so collapsed and healthy cohorts share identical shard order. Replays for the low-variance cohort can use `poetry run python -m ml.cli.streaming_training_runner --dataset-dir ml_out/full_tft_95 --output-dir ml_out/tft_streaming_artifacts/full_tft_95 --state-path ml_out/streaming_training_state_snapshot.json --shard-row-budget 150000 --max-total-rows 150000 --max-total-sequences 112500 --max-shards 40 --batch-size 48 --max-epochs 2 --max-plans 1` to reproduce plan `full_tft_95-648b89e50dfb`. Consider exposing a CLI/plan seed override to probe stochastic effects.
- 📊 2025-10-28: Full-cap replay (`full_tft_95-3f2549f2be7e`, same command as above) completed despite CLI timeout (artefacts written). Metrics remain degraded (`roc_auc=0.489`, `pr_auc_multiple=0.916`, `log_loss=0.756`) with collapsed validation logits (`std(z_val)=0.009`). Economic telemetry is still absent because `val_returns` were not emitted for this dataset; investigate parquet coverage and loader alignment before tightening promotion gates.
- 🛠️ 2025-10-28: Adjusted `_collect_streaming_logits` to capture row metadata before Lightning mutates batches and added debug instrumentation. Fresh small-cohort replay (`full_tft_95-380b30fb894b`, 4 k rows) now persists `val_returns` alongside row identifiers, confirming the loader path is restored; rerun a full-cap cohort next to validate economic metrics end-to-end.
- ✅ 2025-10-29: Full-cap cohort (`full_tft_95-33a163eb4533`) now writes `val_returns` and row metadata in the logits artefact (886 776 rows) and populates economic metrics (Sharpe≈−9.87, hit_rate≈0.0012, turnover≈0.0023). Warning `validation_returns_missing_rows` flagged 154 203 absent join rows; we zero-fill the gap but should audit parquet coverage to reduce noise before promoting settings.
- 🔎 2025-10-30: Polars anti-join on `full_tft_95-33a163eb4533` shows the 154 203 missing validation-return rows stem from 6 429 `time_index` values where row metadata labels everything as `VNQI`, but the parquet rows belong to other ETFs (≈48 % `VNQ`, ≈45 % `VTI`, remainder `VWO`/`VZ`/`VEA`/`VIXY` plus a handful of singletons). Each missing index contributes ~24 logits (decoder horizon), so the instrument mismatch inside `_collect_streaming_logits`/`StreamingRowMetadata` is the root cause—not an empty parquet slice.
- ✅ 2025-10-30: Worker `_extract_validation_returns` now aligns on `time_index` (globally unique) and logs dataset/metadata instrument drift, restoring the 154 203 missing rows for `full_tft_95-33a163eb4533` while preserving zero-fill guardrails for any truly absent indices.
- ✅ 2025-10-31: Validation metadata now aligns with parquet instruments—`TFTStreamingDataset` emits `decoder_group_ids`, the teacher consumes them for lossless row metadata, fallback joins are removed, `validation_returns_instrument_mismatch` is silenced, and manifests/dashboard state expose `validation_returns` diagnostics (`fallback_join`, `mismatch_count`, `missing_count`).
- ✅ 2025-10-30: Full-cap replay (`full_tft_95-ae1aa1106661`) completed with `val_returns` (886 776 rows) present and clean telemetry (no `validation_returns_missing_rows`, `mismatch_count=0`). Metrics: ROC-AUC 0.667, PR-AUC 0.680, LogLoss 0.728, Sharpe −9.85, hit rate 0.067, turnover ~0.00.
- ✅ 2025-10-30: Streaming telemetry now emits `validation_returns` diagnostics (fallback flag, mismatch count, missing count) so manifests and the dashboard expose join health immediately.
- ⏭️ **Next Stage Improvements**
  - Track upcoming full-cap cohorts to ensure the fallback join keeps `validation_returns_missing_rows` suppressed; promote instrumentation in dashboards so ops can spot regressions quickly.
  - Add a configurable dataset/worker seed surface (CLI + config) so stochastic effects can be explored without editing code; align planner payloads and manifests to persist the chosen seed.
  - Wire the Optuna-based streaming sweep runner (`ml/training/event_driven/sweep.py`) into the CLI/orchestrator so hyperparameter searches can operate on the same event-driven stack instead of manual loops.
  - Evaluate whether the existing teacher-side HPO script (`ml/cli/hpo_tft.py`) or distillation utilities can be modernised or retired once the streaming sweep is live; document the chosen path here.
  - Compare logits/return distributions for recent high vs. low performers with the restored telemetry (std/mean of `z_val`, forward-return stats) to understand why economic metrics remain negative.
  - ✅ 2025-10-31: Added `ml/scripts/compare_streaming_cohorts.py` to summarize top/bottom cohorts (metric ranking, `z_val` / `val_returns` stats, validation diagnostics) so regressions are easy to diff after each run (`poetry run python -m ml.scripts.compare_streaming_cohorts --manifest-dir ml_out/tft_streaming_artifacts/full_tft_95 --pretty`).
  - Review long-running runner container telemetry (`ml_out/streaming_training_state_snapshot.json`, container logs) to ensure no stale plans or retries accumulate and that the snapshot includes the new cohorts.
  - ✅ 2025-11-01: Streaming loader now emits `decoder_time_idx` as `int64`, preserving parquet `(instrument_id, time_index)` tuples through the teacher so the precise join holds without time-only fallback. Worker realignment no longer cross-attaches returns; residual mismatches emit warnings and zero-fill returns instead of borrowing from peers. Runner and worker configs expose `--dataset-seed` / `--worker-seed`, the seeds flow through planner caps, telemetry, and manifests, and tiny replay `full_tft_95-936ebf484513` (`max_total_rows=2000`) plus full-cap cohort `full_tft_95-80c171548afd` (`max_total_rows=150000`) both report `validation_returns` telemetry of `fallback_join=false`, `mismatch_count=0`, `missing_count=0` with logits artefacts retaining `val_time_indices`/`val_instrument_ids` provenance.
  - Validate the freshly restarted `streaming_training_runner` container by reviewing the latest cohort (`full_tft_95-e331f6189964` from 2025-10-31) for clean telemetry (expected: `fallback_join=false`, `mismatch_count=0`, `missing_count=0`) and correct seed propagation.
  - ✅ 2025-10-31: Streaming persistence worker now gates Redis payloads with streaming-aware idempotency, unblocking `streaming_training_state.json` updates and keeping dashboards in sync after restarts.
  - ✅ 2025-10-31: Deployment `.env` / `.env.example` and `docker-compose.yml` now surface `ML_STREAMING_DATASET_SEED` and `ML_STREAMING_WORKER_SEED`, so container restarts reuse the configured seeds and manifest telemetry stays aligned with manual runs.
- Guard manifests and telemetry against cross-instrument return attachment: consider treating missing joins as holes (or dropping those rows) rather than borrowing returns when the precise join fails; document the expected behaviour once the loader fix lands.

### 4. Loss Function Alignment & Split Hygiene

**Goal:** Restore the legacy BCE training characteristics and fix cohort fragmentation that emerged after the event-driven refactor.

- **Root Causes (identified 2025-10-30):**
  - `LightningStreamingWorker` always constructs `TFTTeacherConfig()` with default Poisson loss (`ml/training/event_driven/worker.py:866-904`). Legacy runners set `--loss bce`; without that override the streaming path optimises binary targets as count data, driving logits toward the global prevalence.
  - `split_metadata_by_row_fraction` orders shards lexicographically by instrument before honouring the train fraction (`ml/training/teacher/streaming_loader.py:1384-1423`). Once the cumulative target is met, remaining instruments fall entirely into validation—models never see those symbols during training.
  - `_collect_streaming_logits` forwards CPU tensors to whatever device Lightning used for fitting (`ml/training/teacher/tft_teacher.py:542-585`). Post-GPU runs immediately trigger device-mismatch exceptions unless operators force `accelerator=cpu`.

- ✅ 2025-11-01: Streaming worker threads `StreamingWorkerConfig.loss_name` / `loss_pos_weight` through to `TFTTeacherConfig`, defaults to BCE, and records the selection in telemetry + manifests. Runner/CLI expose `--loss`, `--loss-pos-weight`, and env overrides (`ML_STREAMING_TFT_LOSS`, `ML_STREAMING_TFT_LOSS_POS_WEIGHT`).
- ✅ 2025-11-01: Train/validation splitting now operates per instrument/time, mirroring single-shard symbols into both cohorts. Updated tests in `ml/tests/unit/training/teacher/test_streaming_loader.py` guard against future starvation regressions.
- ✅ 2025-11-01: `_collect_streaming_logits` promotes batches (and metadata tensors) onto the teacher’s device before inference, restoring CUDA compatibility. GPU-aware unit coverage skips when `torch.cuda` is unavailable.

- **Additional Risk Drivers (identified 2025-10-31):**
  - `_limit_metadata_for_streaming` consumed shards in discovery order (`ml/training/teacher/streaming_loader.py:717-807`). Because `collect_streaming_metadata` built shards instrument-by-instrument (`ml/training/teacher/streaming_loader.py:504-569`), hitting row/sequence/shard caps starved the tail of the instrument universe and telemetry never saw the skew.
  - `_compute_metrics` returned `{}` when the validation loader yielded zero sequences and the worker still emitted `EventStatus.SUCCESS` (`ml/training/event_driven/worker.py:1345-1355`), so empty cohorts slipped past promotion/economic gates unnoticed.
- ✅ 2025-11-02: `_limit_metadata_for_streaming` now walks shards in a round-robin order per instrument, enforces row/sequence/shard caps fairly, and emits per-instrument totals in `StreamingLimitSummary`. `StreamingLoaderTelemetry` carries the selected/skipped maps so dashboards and guardrails can spot skew. Coverage: `ml/tests/unit/training/teacher/test_streaming_loader.py::test_streaming_limits_round_robin_instrument_mix`.
- ✅ 2025-11-02: `LightningStreamingWorker._compute_metrics` raises `ValidationDatasetEmptyError` when either validation logits or labels are empty. The worker returns `EventStatus.DEFERRED`, records the failure reason in telemetry caps, and dashboards expose the details. Coverage: `ml/tests/unit/training/event_driven/test_worker.py::test_worker_defers_when_validation_empty`.
- ✅ 2025-11-02: Streaming plan/result payloads, dashboards, and docs surface loss configuration plus train/validation instrument coverage. Validation failures now render with reason + counts, and manifests inherit the same caps. Manual: verified the dashboard card renders loss name, top instruments, and validation failure badges for synthetic empty-validation runs.

- **Follow-up Work:**
  1. Add coverage skew assertions to `validate-wave` (fail when any instrument exceeds configured share or when skipped rows spike) and gate promotions on the new telemetry fields.
  2. Extend promotion runbooks with the validation failure triage flow, wiring alerts when `validation_failure_reason` persists across consecutive cohorts.

- **Validation Checklist:**
  - `poetry run mypy ml --strict`
  - `poetry ruff check ml`
  - `pytest ml/tests/unit/training/teacher/test_streaming_loader.py::test_streaming_limits_round_robin_instrument_mix`
  - `pytest ml/tests/unit/training/event_driven/test_worker.py::test_worker_defers_when_validation_empty`
  - `poetry run pytest ml/tests/integration/training/event_driven/test_plan_to_result.py -k streaming`
  - Manual telemetry review: confirm dashboards/manifests show loss name/pos weight, train & validation instrument summaries (including skipped rows), and validation failure badges before promoting a cohort.

### 5. Validate-Wave Automation

**Goal:** Provide a reproducible validation bundle before changing dataset caps or promotion gates.

- ✅ **Command:** `ml/scripts/validate_wave.py` chains mypy, ruff, targeted pytest suites, `make validate-metrics`, `make validate-events`, coverage reporting, and manifest summaries. It enforces doc freshness with configurable staleness thresholds.
- ✅ **Tests:** `ml/tests/unit/scripts/test_validate_wave.py` exercises success, subprocess failure, and doc-staleness flows to keep the bundle robust.
- ✅ **CI Integration:** `ml/scripts/recommend_streaming_wave.py` exposes `--run-validate-wave` to invoke the bundle (with optional `--validate-wave-args`) so wave recommendations and CI jobs gate on the shared validation checklist.

### 6. Auto-Scaling Hooks & Adaptive Scheduling

**Goal:** Manage backlog/GPU pressure automatically.

- ✅ **Config:** `TrainingOrchestratorConfig` already carries adaptive knobs; new logic in `ml/training/event_driven/orchestrator.py` honours backlog/GPU thresholds, emits metrics (`ml_tft_streaming_orchestrator_adaptive_deferrals_total`, `ml_tft_streaming_orchestrator_adaptive_cooldown_seconds`), and raises `AdaptiveSchedulingDeferred` when deferrals trigger.
- ✅ **Implementation:** CLI runner (`ml/cli/streaming_training_runner.py`) gracefully handles adaptive skips, reusing the cooldown window before reattempting.
- ✅ **Tests:** `ml/tests/unit/training/event_driven/test_orchestrator.py` verifies backlog/GPU deferrals, and the runner suite reuses the adaptive helpers.

### 7. Documentation & Ops Workflow

- Treat `ml/docs/architecture/event_driven_streaming_plan.md` as the authoritative checklist; update it (plus `ml/docs/ops/streaming_scaling_experiments.md`) whenever curriculum/ensemble/AMP/economic knobs change.
- Record evidence (plan IDs, telemetry snippets, validation commands) alongside every cohort or peer refresh.
- Maintain parity artefacts in `ml/tests/validation_reports/phase_1a/` when toggles shift to keep promotion audits reproducible.

### Validation Matrix

| Area | Tests / Commands |
|------|------------------|
| Typing & lint | `poetry run mypy ml --strict`, `poetry run ruff check ml` |
| Curriculum/AMP guards | `ml/tests/unit/config/test_streaming_pipeline_config.py`, `ml/tests/unit/training/event_driven/test_worker.py` |
| Ensemble alignment & peers | New sliced alignment tests + `ml/tests/integration/training/event_driven/test_plan_to_result.py` |
| Economic metrics | Metric unit tests + `ml/tests/contracts/test_streaming_payloads.py` |
| Adaptive orchestration | `ml/tests/unit/training/event_driven/test_dataset_service.py`, orchestrator/adaptive unit tests, integration plan-to-result flow |
| Persistence & dashboard | `ml/tests/integration/consumers/test_streaming_persistence_integration.py`, `ml/tests/integration/dashboard/test_streaming_state_endpoint.py` |
| Performance budget | `ml/tests/performance/test_streaming_persistence_microbench.py` |

All changes must continue to respect the guardrails enumerated in `AGENTS.md` (typed APIs, hot-path restrictions, logging standards, fallback chains, centralized imports, ≥90 % ML coverage, documented public interfaces).
