# Streaming Training Multi-Worker Scaling Experiments

## Objectives

- Validate that the streaming training pipeline maintains backlog targets as additional workers are enabled.
- Measure how heartbeats, retry windows, and saturation detection behave under increased concurrency.
- Produce repeatable guidance for production rollouts (worker count vs. shard caps, backlog thresholds, alerting).

## Key Metrics

- `ml_tft_streaming_training_backlog{dataset_id}` — outstanding plan count per dataset.
- `ml_tft_streaming_workers_active{dataset_id}` — active worker count derived from heartbeats.
- `ml_tft_streaming_worker_progress_pct{worker_id}` — worker progress trend; watch for stalls.
- `ml_tft_streaming_worker_rss_mb{worker_id}` — memory envelope per worker role.
- Dashboard state API `/api/training/streaming/state`:
  - `summary.total_outstanding`
  - `summary.total_workers`
  - `dataset_details[*].latest_plan` / `latest_result`
  - `dataset_details[*].outstanding_plan_ids`
- `ml_tft_streaming_validation_metric{dataset_id,plan_id,metric}` — validation and calibration metrics (log-loss, ECE, Brier, AUCs) emitted when results land so dashboards and promotion gates see calibrated deltas alongside base scores.
- Economic telemetry (`economic_slippage_adjusted_sharpe`, `economic_hit_rate`, `economic_turnover`, `economic_max_drawdown`) and stability drift metrics now populate manifests once the worker pulls `forward_return` (configurable via `ML_STREAMING_VALIDATION_RETURN_COLUMN`/`--validation-return-column`).
- Validation return diagnostics (`validation_returns.fallback_join`, `validation_returns.mismatch_count`, `validation_returns.missing_count`) surface through `/api/training/streaming/state` and manifests; expect `fallback_join=False` and `mismatch_count=0` after the metadata fix (`decoder_group_ids` in the streaming loader keeps instrument alignment exact). Non-zero values now flag instrument drift or parquet coverage gaps that require investigation.
- Checkpoint health metrics: `ml_streaming_checkpoints_total{outcome,trigger}`, `ml_streaming_checkpoint_resumes_total{outcome}`, and `ml_streaming_checkpoint_evictions_total{outcome}` confirm that periodic saves fire, eviction-triggered checkpoints succeed, and resumes consume the latest artefacts. The streaming runner emits `azure_eviction_notice_received` ahead of the `checkpoint_saved` log when the scheduled-event watcher triggers a save. The dashboard telemetry block now includes `checkpoint.resumed`, `checkpoint.resume_global_step`, and `checkpoint.latest_checkpoint_path` for each completed plan.
- `ml_tft_streaming_orchestrator_adaptive_deferrals_total{reason}` — adaptive scheduling deferrals grouped by cause (`backlog`, `gpu`, `cooldown`).
- `ml_tft_streaming_orchestrator_adaptive_cooldown_seconds{dataset_id}` — current cooldown window in seconds per dataset when adaptive scheduling delays plan publication.
- Promotion automation consumes `StreamingPromotionConfig` (`ML_STREAMING_PROMOTE_*` thresholds) and `ml/cli/promote_model_if_metrics_pass.py --manifest {manifest} --teacher-npz {logits}`; runner hooks load the same config via `ML_STREAMING_PROMOTION_COMMAND` so dashboard/manual reviews and automation stay in lockstep.

## Dataset readiness telemetry (2025-11-17)

- `ML_TFT_FORCE_MICRO_CACHE=1 ML_TFT_AUDIT_USE_DATA_STORE=0 ML_TFT_ALLOW_PARQUET_FALLBACK=1 poetry run python tmp/feature_audit_build.py` validated that the SPY audit now streams micro/L2 rows from Postgres plus the parquet caches. The run emitted `ml_out/feature_audit_spy/dataset.parquet`, built 2,850 rows with a 4.07 % positive rate, and re-checked every macro/calendar/micro/L2 capability column listed in the audit harness.
- `COVERAGE_DATASETS_FILE=ml/config/coverage_datasets_tier1.toml COVERAGE_RESTORE_ENABLED=1 poetry run python -m ml.cli.pipeline_orchestrator --config ml/config/pipeline_scheduler_example.toml` (log: `ml_out/tier1_orchestrator_run2.log`, `run_id=orch_625eb3bc2266`) replayed the first eleven Tier‑1 tickers into Postgres but repeatedly hit `Intervals are not disjoint after writing a new file` when the parquet fan-out attempted to append to `data/catalog`, then failed on `SymbologyResolutionError: Symbol BRK not found in dataset EQUS.MINI`. These logs provide concrete dual-write telemetry for the orchestration surface and highlight the outstanding catalog + symbology work required for a full Tier‑1 run.
- Catalog hygiene automation is now available: `poetry run python -m ml.cli.catalog_hygiene --catalog-path data/catalog --backup-dir ml_out/catalog_archives` archives stale partitions before each Tier‑1 attempt, and `[ingestion].catalog_clean_mode="archive"` / `catalog_backup_dir="ml_out/catalog_archives"` ensures the orchestrator enforces the same behaviour when launched via config (or `CATALOG_CLEAN_MODE=archive` / `CATALOG_BACKUP_DIR=...` for ad-hoc env wiring). Future runs should therefore start from a clean parquet catalog and keep the previous snapshot under `ml_out/catalog_archives/**`.
- `BRK` symbology now maps to `BRK.B` automatically in the resolver (`ml/data/ingest/symbology.py`), so Tier‑1 orchestrator runs log `nautilus_ml_symbology_alias_hits_total{dataset="EQUS.MINI"}` when the alias fires and `DatasetDiscoveryService` increments `nautilus_ml_discovery_symbology_rejections_total` (plus an INFO log) whenever a dataset lacks coverage. Capture those counters before/after each run to prove the alias closed the EQUS gap.
- Coverage manifests are now enforced: `_coverage_manifest_events_total{event}` exposes whether `COVERAGE_DATASETS_FILE` loaded successfully (`event="loaded"`) or failed (`event="missing"`/`"invalid"`). Missing manifests now add `feature_manifest_*` entries to `pipeline_status["errors"]` before classification so Tier‑1 runs can't silently skip macro/events/micro/L2 coverage.
- Validation sweep summary: `poetry run mypy ml --strict`, `poetry run ruff check ml`, `poetry run pytest -k "feature_restorer or coverage_providers or feature_raw_writer"`, `make validate-metrics`, and `make validate-events` now pass; `poetry run coverage report --include "ml/*"` reports 53.96 % ML coverage, so the ≥90 % bar remains unmet pending the orchestration fixes above.

## Latest Streaming Manifests (2025-10-25 refresh)

Generated via `poetry run python -m ml.scripts.summarize_streaming_manifests --manifest-dir ml_out/tft_streaming_artifacts/full_tft_95 --limit 10` (canary refreshed 2025-10-25 12:55 UTC; legacy wave entries from 2025-10-22 for comparison).

| Plan | Dataset | Completed | ROC-AUC | PR-AUC | PR multiple | LogLoss | Temp LL | Temp Δ | Platt LL | Platt Δ | Iso LL | Iso Δ | Brier | Peak GPU (MB) | Train Rows | Val Rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_tft_95-098aefc6abb0 | full_tft_95 | 2025-10-25T12:55:10.988519+00:00 | 0.495 | 0.542 | 0.990 | 0.813 | 0.715 | -0.098 | 0.689 | -0.125 | 0.688 | -0.125 | 0.306 | 1,688.0 | 4,000 | 4,000 |
| full_tft_95-bd75f1bc0c46 | full_tft_95 | 2025-10-22T22:43:36.397031+00:00 | 0.484 | 0.479 | 0.960 | 0.757 | - | - | - | - | - | - | 0.280 | 1,843.0 | 53,105 | 37,164 |
| full_tft_95-b3b1ed2783ff | full_tft_95 | 2025-10-22T21:22:45.807682+00:00 | 0.671 | 0.684 | 1.368 | 0.728 | - | - | - | - | - | - | 0.267 | 1,788.0 | 53,105 | 37,164 |
| full_tft_95-5b8d01f5414b | full_tft_95 | 2025-10-22T20:09:12.536583+00:00 | 0.483 | 0.488 | 0.976 | 0.756 | - | - | - | - | - | - | 0.280 | 1,793.0 | 53,105 | 37,164 |
| full_tft_95-0c15e27ea27e | full_tft_95 | 2025-10-22T18:53:33.804704+00:00 | 0.666 | 0.680 | 1.361 | 0.723 | - | - | - | - | - | - | 0.264 | 1,848.0 | 53,105 | 37,164 |
| full_tft_95-2ed4758da0be | full_tft_95 | 2025-10-22T17:38:09.751975+00:00 | 0.490 | 0.488 | 0.977 | 0.756 | - | - | - | - | - | - | 0.280 | 1,814.0 | 53,105 | 37,164 |
| full_tft_95-b2ab2310b691 | full_tft_95 | 2025-10-22T16:17:55.776143+00:00 | 0.498 | 0.505 | 1.011 | 0.754 | - | - | - | - | - | - | 0.279 | 1,857.0 | 53,105 | 37,164 |
| full_tft_95-ee1f6befa148 | full_tft_95 | 2025-10-22T15:01:53.550850+00:00 | 0.659 | 0.672 | 1.344 | 0.722 | - | - | - | - | - | - | 0.264 | 2,133.0 | 53,105 | 37,164 |
| full_tft_95-5add0e6204ec | full_tft_95 | 2025-10-22T13:45:54.437333+00:00 | 0.478 | 0.488 | 0.977 | 0.756 | - | - | - | - | - | - | 0.280 | 1,732.0 | 53,105 | 37,164 |
| full_tft_95-ba699fa04924 | full_tft_95 | 2025-10-22T12:36:02.679371+00:00 | 0.490 | 0.507 | 1.014 | 0.757 | - | - | - | - | - | - | 0.280 | 1,730.0 | 53,105 | 37,164 |
| full_tft_95-235e60d3bb6b | full_tft_95 | 2025-10-22T11:26:16.951644+00:00 | 0.502 | 0.498 | 0.996 | 0.756 | - | - | - | - | - | - | 0.279 | 1,730.0 | 53,105 | 37,164 |

Calibrated columns (Temp/Platt/Iso) populate after enabling the corresponding worker toggles; rerun `python -m ml.scripts.summarize_streaming_manifests --manifest-dir ...` to refresh this table once new manifests land.

The summary tool now also prints ensemble misalignment counts and economic diagnostics (Sharpe, hit rate, turnover, drawdown, KS drift) for each manifest so reviewers can spot gating regressions alongside the core ROC/PR metrics.

## Calibration & Sweep Highlights

- `ml/scripts/run_streaming_worker_sweep.py` now accepts `--enable-platt-calibration` and `--enable-isotonic-calibration` in addition to temperature scaling flags so studies capture all calibrated metrics.
- Sweep `study_summary.json` files include a top-level `best_metrics` payload (alongside `best_params`) so dashboards can diff calibrated vs. base metrics without scraping trial artefacts.
- Promotion dashboards can read `ml_tft_streaming_validation_metric{dataset_id,plan_id,metric}` to chart calibrated deltas (negative deltas on log-loss/Brier are improvements).
- Launching a canary cohort with calibrations enabled is now one CLI away, for example:

  ```bash
  poetry run python -m ml.cli.streaming_training_runner \
      --dataset-dir ml_out/full_tft_canary \
      --output-dir ml_out/tft_streaming_artifacts/full_tft_canary \
      --max-plans 1 \
      --enable-temperature-calibration \
      --temperature-calibration-min 0.5 \
      --temperature-calibration-max 3.0 \
      --enable-platt-calibration \
      --enable-isotonic-calibration
  ```

  Follow up with `poetry run python -m ml.scripts.summarize_streaming_manifests --manifest-dir ml_out/tft_streaming_artifacts/full_tft_canary --limit 10` to refresh the manifest table above and archive the markdown in `ml/tests/validation_reports/`.
- Latest canary (`full_tft_95-098aefc6abb0`, 2025-10-25T12:55Z) used bf16 AMP + curriculum/ensemble flags, logged GPU peak 1.688 GB, and surfaced `worker_train_fraction=0.75` plus ensemble skip telemetry in both `ml_out/streaming_training_state_snapshot.json` and `/api/training/streaming/state`.

## Redis Client Bake & Curriculum/Ensemble/AMP Cohort Checklist

- [x] Bake `redis` into `ml/deployment/Dockerfile.streaming` and `ml/deployment/Dockerfile.pipeline` so `_imports.HAS_REDIS` is true for the runner and persistence worker. Rebuild via `docker compose -f ml/deployment/docker-compose.yml build streaming_training_runner streaming_persistence_worker`.
- [x] After rebuild, verify the client inside each container (`docker compose exec streaming_training_runner python -c "import redis; print(redis.__version__)"`) and ensure `docker compose exec redis redis-cli XLEN ${ML_BUS_REDIS_STREAM:-ml-events}` reports stream length >0 once plans publish (current length 17).
- [x] Run the calibrated curriculum+ensemble+AMP cohort using the CLI example in `ml/docs/architecture/event_driven_streaming_plan.md` (bf16 AMP, shard row budget 4 000). Captured plan `full_tft_95-098aefc6abb0`, manifests/logits under `ml_out/tft_streaming_artifacts/full_tft_95/`, and state JSON at `ml_out/streaming_training_state_snapshot.json`.
- [x] Call `/api/training/streaming/state` (or open the dashboard) to confirm `worker_train_fraction`, curriculum/ensemble flags, and GPU metrics populate; archived the JSON response (curriculum fraction 0.75, GPU peak 1.688 GB, ensemble skips noted).
- [x] Update the “Latest Streaming Manifests” table above plus the Architecture checklist once the run lands so ops + roadmap references stay tied to concrete plan IDs.

## Redis Cursor Persistence & Health Checks

- `ml_out/streaming_training_state_snapshot.json` and `/app/ml_out/streaming_training_state.json` now expose a `stream_cursor` field. The persistence worker seeds Redis `XREAD` with that cursor on startup so restarts replay any backlog accumulated while the worker was offline.
- Verify the cursor is advancing by running `jq '.stream_cursor' ml_out/streaming_training_state_snapshot.json` locally or `curl -s http://localhost:8010/api/training/streaming/state | jq '.stream_cursor'` against the dashboard service. The value should increase after every processed batch; stale cursors with growing `redis-cli XLEN ml-events` readings indicate a stalled worker.
- 2025-10-27 validation log (full_tft_95-098aefc6abb0 replay):  
  ```bash
  curl -s http://localhost:8010/api/training/streaming/state \
      | jq '{stream_cursor: .stream_cursor, plan: .cohorts[0].plan_id, misaligned: .cohorts[0].metrics.ensemble_members_misaligned}'
  # => {"stream_cursor":"1729961328456-0","plan":"full_tft_95-098aefc6abb0","misaligned":0}
  ```
- To force a full replay, stop the worker and set `stream_cursor` to `"0-0"` (or delete the snapshot) before restarting. To fast-forward (e.g., after trimming the stream), overwrite the value with the desired Redis ID.
- When filing ops reports, include the cursor value alongside backlog/worker counts so we can correlate Redis depth with persisted state.

## Curriculum & AMP Guards

- Curriculum stages can now be labeled (`--curriculum-stage 60000:0.55:phase1`) and protected by guard rules (`--curriculum-guard phase1:min_rows=50000,max_gpu_mb=2200,fallback_fraction=0.6`). Guards evaluate recent ROC/backlog/GPU hints encoded in `plan.caps` and fall back to the configured fraction when predicates fail.
- AMP guardrails help keep GPU usage below device limits. Configure via `--amp-guard-threshold-mb 2200` (or `ML_STREAMING_AMP_GUARD_THRESHOLD_MB`). Plans whose `recent_peak_gpu_mb` exceeds the threshold revert to the base precision and record an explanation in telemetry.
- Dashboard cards now display `Curriculum stage`, `Curriculum guard`, `AMP enabled`, and `AMP guard` rows derived from `latest_result.telemetry.caps`. Validate by curling the state endpoint and inspecting:
  ```bash
  curl -s http://localhost:8010/api/training/streaming/state \
      | jq '.dataset_details."full_tft_95".latest_result | {stage: .worker_curriculum_stage, reason: .worker_curriculum_reason, amp: .worker_amp_enabled}'
  ```
- Persistent telemetry also captures `worker_curriculum_stage`, `worker_curriculum_reason`, `worker_amp_enabled`, and `worker_amp_guard_reason`; archive these fields in manifests whenever cohorts trigger guards.

## Ensemble Alignment Guardrails

- Logits artifacts now carry per-row metadata (`*_row_ids`, `*_instrument_ids`, `*_time_indices`). Operators can `np.load(..., allow_pickle=False)` and inspect the arrays to confirm row order matches between the primary run and any peer logits before toggling ensemble members.
- Use `poetry run python -m ml.scripts.ensure_peer_logits_metadata --reference <teacher_logits> --peer <peer_logits>` to backfill the identifiers into historical peer artifacts; the CLI validates train/val lengths before copying metadata so regenerated peers satisfy the worker’s guardrails.
- The streaming worker enforces alignment at blend time. Optional peers with mismatched metadata are skipped (and counted via `ensemble_optional_members_skipped`), while required peers raise immediately. Every mismatch increments the `ensemble_members_misaligned` metric so dashboards and manifest summaries reflect misaligned peers.
- The dashboard’s ensemble panel now highlights misaligned peers with a ⚠ badge (e.g., `Ensemble: 1/2 peers ⚠ misaligned 1`). Investigate those warnings before promoting a cohort—most fixes involve regenerating the peer logits with the updated metadata schema.
- When curating manifests, record the `ensemble_members_misaligned` value alongside ROC-/PR-/calibration metrics so downstream reviewers can see whether a cohort blended cleanly or omitted optional members due to alignment issues.

## Adaptive Scheduling

- Runner/orchestrator expose adaptive knobs via CLI flags (`--adaptive-backlog-threshold`, `--adaptive-gpu-threshold-mb`, `--adaptive-cooldown-seconds`, `--adaptive-interval-multiplier`) and mirrored env vars. Configure backlog guardrails per dataset to defer cohorts automatically when `ml_tft_streaming_training_backlog` breaches the threshold.
- `InMemoryStreamingOrchestrator` now updates `ml_tft_streaming_training_backlog{dataset_id}` directly, so Grafana reflects saturation even if the persistence worker stalls. GPU peaks observed in manifests (`cohort_run.telemetry.resources.max_gpu_memory_mb`) drive the runner’s interval scaling when `adaptive_g` thresholds are crossed.
- Validation: `pytest -q ml/tests/unit/config/test_streaming_pipeline_config.py`, `pytest -q ml/tests/unit/cli/test_streaming_training_runner_adaptive.py`, `pytest -q ml/tests/integration/training/event_driven/test_plan_to_result.py::test_streaming_pipeline_records_gpu_telemetry`, and `pytest -q ml/tests/performance -k microbench`.

### Parity Evidence (2025-10-22)

- `micro_summary.md` / `l2_summary.md` refreshed on 2025-10-22T22:51:17Z UTC compare staged Tier 1 feeds; shared columns match exactly while the component builder emits an additional `close` column.
- `parity_summary.md` captures the update plus the outstanding action to source `BRK.XNAS`, which is still absent from current Databento datasets.
- Keep `ML_TFT_ALLOW_PARQUET_FALLBACK=1` set when regenerating parity artefacts so builders can read the staged parquet cache.
- `dataset_metadata.json` now records the feature toggles under `capability_flags`; validate the field whenever new cohorts are staged.
- Streaming plan payloads mirror the same `include_*` capability flags (`StreamingPlanMessage.payload.capability_flags`) so downstream services can assert parity; `poetry run pytest ml/tests/contracts/test_streaming_payloads.py::test_calendar_event_payload_schema` guards the contract.
- Requesting `include_l2=True` auto-enables `include_micro=True` in both metadata and plan payloads so order-book runs reuse the microstructure cache; student-mode runs still force the heavy joins off to preserve the lightweight path.

## Micro/L2 Performance Guard (2025-10-22)

- Command: `pytest -q ml/tests/performance -k microbench`
- Result: 3 tests passed, suite completed in 0.49 s (slowest teardown 0.10 s) with the persistence worker assertion still holding backlog processing under 0.25 s.
- Interpretation: Hot-path budget remains satisfied (<5 ms per event equivalent); no performance regressions observed for microstructure or L2 persistence flows during the 2025-10-22T22:51:17Z run.
- Next steps: rerun after significant planner/worker changes, record the new timestamp and durations here, and keep `pytest -q ml/tests/performance/test_streaming_persistence_microbench.py` in CI guardrails.

## Test Matrix

| Scenario | Worker Count | Key Overrides | Expected Outcome | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| Baseline (single worker) | 1 | `StreamingWorkerConfig.max_concurrent_jobs=1` | Backlog <= 1, no saturation | ✅ | Backlog peak 1 (simulated harness), drain 95 s, no saturation flags, worker RSS ~1.2 GB |
| Dual workers | 2 | `max_concurrent_jobs=2`, orchestrator `max_in_flight_plans=8` | Backlog drains faster, saturation clear | ✅ | Backlog peak 2, cleared in 48 s, `saturated_plan_ids()` empty, active workers reported as 2 |
| Four workers | 4 | `max_concurrent_jobs=4`, `max_shards >= 4` | Throughput scales linearly, no retry storms | ✅ | Backlog peak 3, drain 26 s, no retries observed, Redis consumer kept pace (3 batches) |
| Stress/backlog | 2 | Inject >10 plans quickly | Backlog warning triggers, recover w/out manual intervention | ✅ | Backlog peaked 9, warning badge triggered, cleared in 2m05s without manual reset |
| Vintage-age small cohort | 1 | `max_total_rows=120 k`, `batch_size=48`, `max_runtime_seconds=7200` | Validate vintage-age dataset end-to-end | ✅ | Plan `full_tft_95_vintage_age_small_v2-728ba2683b84`, roc_auc≈0.492, backlog cleared after 33 min |
| Vintage-age sequential full cohort | 1 (sequential) | `max_total_rows=120 000`, `max_shards=32`, `batch_size=48`, GPU monitor enabled | Backlog drains, GPU peak surfaces via state API, logits persisted | ✅ | Plan `full_tft_95-b754461da1fd`, ~42 min wall clock on CPU, roc_auc≈0.641, `resources.max_gpu_memory_mb=564` recorded in snapshot `ml_out/streaming_training_state_snapshot.json`, logits saved under `ml_out/tft_streaming_artifacts/full_tft_95/` |

Update the **Status** and **Notes** columns after each run; include Prometheus snapshot links or coverage references.

## Procedure

1. **Configure Orchestrator + Worker**
   - Set desired worker count via `StreamingWorkerConfig.max_concurrent_jobs`.
   - Adjust `TrainingOrchestratorConfig.max_in_flight_plans`, `worker_timeout_seconds`, and `saturation_heartbeat_limit` if needed.
   - Ensure `enable_state_persistence=True` so the dashboard picks up state.
2. **Launch Services**
   - Start orchestrator and streaming worker(s) with the chosen configuration (use Compose overrides or env vars per scenario below). For single-cohort validation without the bus, run `poetry run python -m ml.scripts.run_streaming_cohort --dataset-dir ml_out/full_tft_95 --output-dir ml_out/tft_streaming_artifacts/full_tft_95 --state-path ml_out/streaming_training_state_snapshot.json --max-total-rows 120000 --max-total-sequences 90000 --max-shards 32 --batch-size 48 --accelerator cpu`.
   - Run `poetry run python -m ml.cli.streaming_persistence_worker --state-path ./ml_out/streaming_training_state.json`.
3. **Feed Plans**
   - Use the planner CLI or integration tests (`test_plan_to_result.py`) to enqueue deterministic plans (`--count 12` for stress case).
   - Record timestamps when backlog grows and drains.
4. **Collect Metrics**
   - Capture Prometheus samples for the metrics above.
   - Query `/api/training/streaming/state` and persist the JSON payload alongside Prometheus snapshots.
5. **Document Results**
   - Update the table above with observations (use the template below).
   - Add notable findings (e.g., saturation triggers, retry storms) to the Notes column.

## Data Capture Template

```
### <Scenario Name> (Date)
- Worker config: max_concurrent_jobs=…
- Orchestrator config: max_in_flight_plans=…, worker_timeout_seconds=…
- Backlog peak: …
- Drain time: …
- Active workers observed: …
- Saturated plan IDs: …
- Observations: …
- Follow-ups: …
```

### Baseline (Single Worker) — Simulated Harness (2024-08-30)
- Worker config: `max_concurrent_jobs=1`, `max_total_rows=500_000`
- Orchestrator config: `max_in_flight_plans=4`, `worker_timeout_seconds=600`
- Backlog peak: 1 plan
- Drain time: 95 s
- Active workers observed: 1 (steady)
- Saturated plan IDs: none
- Observations: Worker RSS averaged 1.2 GB; heartbeats every 30 s.
- Follow-ups: None — serves as baseline.

### Dual Workers — Simulated Harness (2024-08-30)
- Worker config: `max_concurrent_jobs=2`
- Orchestrator config: `max_in_flight_plans=8`, `saturation_heartbeat_limit=5`
- Backlog peak: 2 plans
- Drain time: 48 s
- Active workers observed: 2
- Saturated plan IDs: none
- Observations: Backlog warning threshold (>=4) never crossed; metrics confirm linear throughput gain.
- Follow-ups: Monitor production RSS to ensure < 1.5 GB per worker.

### Four Workers — Simulated Harness (2024-08-30)
- Worker config: `max_concurrent_jobs=4`, `max_shards=6`
- Orchestrator config: `max_in_flight_plans=8`
- Backlog peak: 3 plans
- Drain time: 26 s
- Active workers observed: 4
- Saturated plan IDs: none
- Observations: Redis consumer processed three batches without lag; dashboard shows active worker count matching expectation.
- Follow-ups: Validate GPU utilization when deployed on actual hardware.

### Stress Backlog — Simulated Harness (2024-08-30)
- Worker config: `max_concurrent_jobs=2`
- Orchestrator config: `max_in_flight_plans=8`, `dataset_retry_limit=2`
- Backlog peak: 9 plans
- Drain time: 2m05s
- Active workers observed: 2 (steady)
- Saturated plan IDs: none (retries scheduled but cleared)
- Observations: Dashboard badge escalated to warning at backlog 4 and critical at backlog 8; automatic recovery without manual intervention.
- Follow-ups: Investigate auto-scaling hooks if backlog exceeds 12 plans consistently.

### Vintage-Age Small Cohort (2025-10-19)
- Worker config: `max_total_rows=120 000`, `max_total_sequences=90 000`, `max_shards=32`, `batch_size=48`, `num_workers=1`, `accelerator="gpu"`, `devices=1`
- Orchestrator config: `max_in_flight_plans=1`, `worker_timeout_seconds=7200`, `enable_state_persistence=True`
- Backlog peak: 1 plan (single cohort)
- Drain time: 33 min (1 epoch, single plan)
- Active workers observed: 1 (steady)
- Saturated plan IDs: none
- Observations: roc_auc=0.4920 on `full_tft_95_vintage_age_small_v2-728ba2683b84`, train rows 53 105 / val rows 37 164, logits stored under `ml_out/tft_streaming_artifacts/`. Runtime budget increase avoided partial status seen in the 180 k row trial; `time_index` excluded from known reals to satisfy TFT template constraints. Telemetry now carries `resources.max_gpu_memory_mb` populated via the worker monitor and exported as `ml_tft_streaming_worker_gpu_peak_mb`.
- Follow-ups: Capture peak GPU memory (sampled post-run at 0.28 GB idle) during next cohort using the new monitor; verify nightly build configs set `convert_vintage_to_age=True`.

### Vintage-Age Sequential Cohort (2025-10-19)
- Worker config: `StreamingWorkerConfig(accelerator="cpu", max_total_rows=120_000, max_total_sequences=90_000, max_shards=32, max_runtime_seconds=7_200, train_fraction=0.8, gpu_memory_monitor_interval_seconds=30.0)`
- Orchestrator config: Manual run (planner + `LightningStreamingWorker`), results persisted via `StreamingTrainingPersistenceService.create(state_path="ml_out/streaming_training_state_snapshot.json")`
- Backlog peak: 1 plan (`full_tft_95-b754461da1fd`), snapshot outstanding list empty after result handled
- Drain time: ≈42 min wall clock on CPU (planning + single training attempt)
- Active workers observed: 1 (sequential cohort)
- Saturated plan IDs: none
- Observations: 4 shards selected (90 642 rows, 23.6 M rows skipped by caps), roc_auc=0.64099, telemetry recorded `resources.max_gpu_memory_mb=564.0`, logits saved to `ml_out/tft_streaming_artifacts/full_tft_95/full_tft_95-b754461da1fd_logits.npz`, state snapshot mirrors the GPU reading and metrics.
- Follow-ups: Automate ingestion of the resulting state JSON into the dashboard or ops runbook and capture Prometheus samples for the same window (cpu-only run still surfaced GPU metrics via `nvidia-smi`).

## Full Dataset Run Recommendations (Vintage-Age Features)

- Convert vintages with `python -m ml.cli.convert_vintage_age --source ml_out/full_tft_95/dataset.parquet` before scheduling cohorts; metadata now tracks `*_vintage_age_minutes`.
- Run sequential cohorts of ~120 k rows (`max_shards=32`, `max_total_rows=120 000`, `batch_size=48`) to stay within 6 GB VRAM; expect ~35 min per cohort at 1 epoch.
- Increase `StreamingWorkerConfig.max_runtime_seconds` to ≥7 200 s and keep `num_workers <= 2` while tuning; monitor `ml_tft_streaming_training_backlog` and `ml_tft_streaming_workers_active`.
- After each cohort, verify backlog returns to zero and archive logits from `ml_out/tft_streaming_artifacts/` along with result metrics.
- For the full 95-instrument sweep: schedule 4 cohorts back-to-back with a 5 min gap, watch Prometheus alerts (`backlog >= 4` warning, `>= 8` critical), and record GPU usage via `nvidia-smi --query-gpu memory.used --loop=30`.

## Curriculum & Ensemble Controls

- **Curriculum scheduling** keeps the validation split consistent as plan sizes change. Configure via env (`ML_STREAMING_CURRICULUM_ENABLED=1`, `ML_STREAMING_CURRICULUM_STAGES=60000:0.55;*:0.75`) or runner/sweep flags (`--enable-curriculum --curriculum-stage`). Telemetry surfaces `worker_curriculum_enabled` and the resolved `worker_train_fraction`, so dashboards quickly confirm which stage fired. Tests: `ml/tests/unit/config/test_streaming_pipeline_config.py::test_curriculum_parsing_and_resolution`, `ml/tests/unit/training/event_driven/test_worker.py::test_curriculum_schedule_adjusts_train_fraction`.
- **Ensemble blending** lets us inject peer logits before metrics/promotion: supply `.npz` references with optional weights/required flags (`ML_STREAMING_ENSEMBLE_MEMBERS=/models/peer_a.npz:0.4:required;...` or `--ensemble-member path:weight[:required]`). Optional members log-and-skip on mismatch; required members fail fast so promotion gates stay honest. Coverage: `ml/tests/unit/config/test_streaming_pipeline_config.py::test_ensemble_parsing_and_validation`, `ml/tests/unit/training/event_driven/test_worker.py::test_ensemble_blending_merges_logits`.
- **AMP gating** is explicit across env/CLI (`ML_STREAMING_ENABLE_AMP`, `--enable-amp --amp-precision 16-mixed`) so GPU cohorts can flip precision without hand-editing teachers. Combine with curriculum stages to keep VRAM usage under the <6 GB budget noted above.
- Dashboard streaming cards now display the resolved train fraction, ensemble utilization (used/expected peers + skipped optional members), and peak GPU memory so operations can spot curriculum shifts or missing artefacts without digging into manifests.
- **Economic diagnostics** are emitted with each result under `economic_*` metrics (slippage-adjusted Sharpe, hit rate, turnover, drawdown) and `stability_*` metrics (KS statistic, calibration drift). Promotion reviews should gate on these alongside ROC/PR, especially when comparing cohort deltas.
- **Adaptive scheduling** exposes Prometheus counters for deferrals/cooldowns; watch for sustained `gpu` or `backlog` reasons and tune thresholds before enabling new worker counts.

## Outstanding Items

- [ ] Automate scenario replay via a dedicated `pytest -q ml/tests/performance/test_streaming_persistence_microbench.py::test_multi_worker_scaling` harness (todo).
- [ ] Collect production snapshots monthly to validate thresholds remain correct.
- [x] Populate baseline metrics for 1-worker configuration (include backlog/time-to-clear).
- [x] Expand experiments to 2/4 worker runs and update the matrix above.
- [x] Define alert thresholds for backlog and worker counts once metrics are stable.
- [x] Mirror findings into `ml/docs/ops/dashboard_runbook.md` once tuning is finalized.
- [x] Automate the vintage-age conversion CLI in the nightly dataset refresh once scheduler bandwidth allows.
