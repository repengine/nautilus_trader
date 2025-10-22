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

## Latest Streaming Manifests (2025-10-21)

Generated via `poetry run python -m ml.scripts.summarize_streaming_manifests --manifest-dir ml_out/tft_streaming_artifacts/full_tft_95 --limit 10` (2025-10-21 19:43 UTC).

| Plan | Dataset | Completed | ROC-AUC | PR-AUC | PR multiple | LogLoss | Brier | Peak GPU (MB) | Train Rows | Val Rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_tft_95-458c9417c1d7 | full_tft_95 | 2025-10-21T18:49:37.833599+00:00 | 0.669 | 0.681 | 1.363 | 0.725 | 0.266 | 1,866.0 | 53,105 | 37,164 |
| full_tft_95-2623b7a7c6a5 | full_tft_95 | 2025-10-21T17:34:26.018581+00:00 | 0.665 | 0.678 | 1.357 | 0.727 | 0.266 | 1,858.0 | 53,105 | 37,164 |
| full_tft_95-39dea65fd983 | full_tft_95 | 2025-10-21T16:09:56.965504+00:00 | 0.488 | 0.486 | 0.973 | 0.755 | 0.279 | 1,897.0 | 53,105 | 37,164 |
| full_tft_95-610258fdd2da | full_tft_95 | 2025-10-21T14:54:24.140187+00:00 | 0.485 | 0.487 | 0.974 | 0.752 | 0.278 | 1,702.0 | 53,105 | 37,164 |
| full_tft_95-6e9bff431bf9 | full_tft_95 | 2025-10-21T13:44:45.119480+00:00 | 0.483 | 0.489 | 0.979 | 0.756 | 0.279 | 1,708.0 | 53,105 | 37,164 |

### Parity Evidence (2025-10-22)

- `micro_summary.md` / `l2_summary.md` now compare staged Tier 1 feeds; shared columns match exactly while the component builder emits an additional `close` column.
- `parity_summary.md` captures the update plus the outstanding action to source `BRK.XNAS`, which is still absent from current Databento datasets.
- Keep `ML_TFT_ALLOW_PARQUET_FALLBACK=1` set when regenerating parity artefacts so builders can read the staged parquet cache.
- `dataset_metadata.json` now records the feature toggles under `capability_flags`; validate the field whenever new cohorts are staged.
- Streaming plan payloads mirror the same `include_*` capability flags (`StreamingPlanMessage.payload.capability_flags`) so downstream services can assert parity; `poetry run pytest ml/tests/contracts/test_streaming_payloads.py::test_calendar_event_payload_schema` guards the contract.
- Requesting `include_l2=True` auto-enables `include_micro=True` in both metadata and plan payloads so order-book runs reuse the microstructure cache; student-mode runs still force the heavy joins off to preserve the lightweight path.

## Micro/L2 Performance Guard (2025-10-22)

- Command: `pytest -q ml/tests/performance -k microbench`
- Result: 3 tests passed, suite completed in 0.54 s (peak individual microbench 0.10 s) with the persistence worker assertion holding backlog processing under 0.25 s.
- Interpretation: Hot-path budget remains satisfied (<5 ms per event equivalent); no performance regressions observed for microstructure or L2 persistence flows.
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

## Outstanding Items

- [ ] Automate scenario replay via a dedicated `pytest -q ml/tests/performance/test_streaming_persistence_microbench.py::test_multi_worker_scaling` harness (todo).
- [ ] Collect production snapshots monthly to validate thresholds remain correct.
- [x] Populate baseline metrics for 1-worker configuration (include backlog/time-to-clear).
- [x] Expand experiments to 2/4 worker runs and update the matrix above.
- [x] Define alert thresholds for backlog and worker counts once metrics are stable.
- [x] Mirror findings into `ml/docs/ops/dashboard_runbook.md` once tuning is finalized.
- [x] Automate the vintage-age conversion CLI in the nightly dataset refresh once scheduler bandwidth allows.
