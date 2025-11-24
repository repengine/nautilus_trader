# Streaming Pipeline Legacy Fallback & Persistence Replay Plan

## Context

- Recent refactor replaced the monolithic `MLPipelineOrchestrator` with a component façade (`ml/orchestration/pipeline_orchestrator_component.py`) that should keep the legacy implementation available until feature parity is reached.
- Current façade ignores `ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=1` and always runs the incomplete component path. Critical entry points (`run`, `run_training_only`, `train_teacher`, `distill_student`, `run_hpo`) now just warn and return `1`.
- The streaming persistence worker (`ml/consumers/streaming_training_worker.py`) starts polling Redis with `last_id="$"`, so cold starts skip the backlog and fail to rebuild dashboard state.

## Goals

1. Restore the ability to execute the legacy pipeline orchestrator while the component modules mature.
2. Ensure the streaming persistence worker replays historical Redis events on cold start, preserving dashboards and metrics.
3. Maintain typing, lint, and test coverage requirements (`poetry run mypy ml --strict`, `poetry ruff check ml`, ≥90 % coverage for ML modules).

## Findings

- `pipeline_orchestrator_component.MLPipelineOrchestrator.__init__`: logs a warning when the legacy flag is set but forces `_use_legacy = False`. No legacy object is created.
- Training/HPO methods inside the façade are stubbed with warnings and `return 1`, breaking CLI flows such as `ml.cli.pipeline_orchestrator`.
- `StreamingTrainingPersistenceWorker.poll_once` defaults to `last_id="$"` when `_last_stream_id` is `None`, so a fresh worker only sees new events after it starts.

## Work Plan

### 1. Reinstate Legacy Orchestrator Fallback

- ✅ **Conditional construction**
  - When `ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=1`, instantiate `ml.orchestration.pipeline_orchestrator_legacy.MLPipelineOrchestrator` with the original kwargs and set `_use_legacy = True`.
  - Only build component dependencies when `_use_legacy` is `False` to avoid double initialization.
- ✅ **Delegation logic**
  - For config, discovery, binding, ingestion, and dataset methods continue using the component path.
  - For currently stubbed training/HPO/pipeline methods, delegate to the legacy instance if `_use_legacy` is `True`; otherwise keep the warning (or raise `NotImplementedError` with migration guidance).
  - Update `__getattr__` so it proxies to the legacy object only when `_use_legacy` is enabled; otherwise raise `AttributeError` as today.
- ✅ **Compatibility exports**
  - Confirm the helper re-exports (`_run_ingestion_stage`, `parse_args`, etc.) continue to point at the legacy module and do not trigger imports when not required.
- ✅ **Tests & validation**
  - Add unit tests that toggle the env flag and assert `run`/`train_teacher` route to the legacy implementation.
  - Execute `poetry run mypy ml --strict`, `poetry ruff check ml`, and targeted orchestration tests (search for existing `pipeline_orchestrator` suites; add smoke test if missing).

### 2. Fix Streaming Persistence Cold-Start Replay

- ✅ **Cursor initialization**
  - In `StreamingTrainingPersistenceWorker.poll_once`, detect when no cursor has been persisted and default to `"0-0"` instead of `"$"`, ensuring backlog replay. Respect stored cursors loaded via `_ensure_service`.
  - Consider allowing an optional config override (e.g., `start_id`) if operators need to skip history.
- ✅ **Robust persistence**
  - Keep `_update_cursor_from_consumer` writing the new cursor after each batch. Emit telemetry/logging when persistence fails so operators can act.
- ✅ **Tests & validation**
  - Extend `ml/tests/unit/consumers/test_streaming_training_worker.py` to verify a cold worker processes queued events.
  - Run relevant suites:
    `poetry run pytest ml/tests/unit/consumers/test_streaming_training_worker.py`
    `poetry run pytest ml/tests/integration/cli/test_streaming_persistence_worker_cli.py`
    `poetry run pytest -q ml/tests/performance -k streaming_persistence`.

### 3. Documentation & Rollout

- Document flag behaviour and the temporary dual-path approach in the orchestration docs (e.g., `ml/docs/architecture/event_driven_streaming_plan.md` or similar).
- Update the streaming ops runbook (`ml/docs/ops/streaming_scaling_experiments.md` / dashboard runbook) to note backlog replay expectations and cursor reset procedures.
- After implementation, rerun full validation matrix: `make validate-metrics`, `make validate-events`, and any affected integration suites.

## Status Update (2025-11-04)

- Component façade now instantiates the legacy orchestrator when `ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=1`, delegating unsupported methods while preserving component behaviour elsewhere. Added unit coverage in `ml/tests/unit/orchestration/test_pipeline_orchestrator_component.py`.
- Streaming persistence worker defaults its initial Redis cursor to `"0-0"` so cold starts replay historical events; augmented unit coverage to assert both cold-start and persisted-cursor flows.
- Validated with `poetry run mypy ml --strict`, `poetry run ruff check ml`, and targeted pytest runs for the new tests. Remaining rollout tasks tracked above.

## Open Questions

- Do we want the component façade to raise `NotImplementedError` (loud failure) when `_use_legacy` is `False`, or keep the current warning/exit-code contract?
- Should we expose an explicit CLI flag (e.g., `--legacy`) in addition to the environment variable for easier operator control?
- Is there an upper bound on backlog size we need to guard against (e.g., configurable replay window) to prevent long blocking periods on worker restart?

## Next Actions

- ✅ Document flag behaviour and dual-path expectations in the orchestration docs.
- ✅ Update streaming ops runbooks with backlog replay guidance and cursor reset procedures.
- ✅ Run the broader validation matrix (`make validate-metrics`, `make validate-events`, etc.) before promoting downstream.

## Validation Log (2025-11-04)

- `make validate-metrics` → Metrics bootstrap validation OK.
- `make validate-events` → Event constants validation OK.

## Decision Status

- **Component façade behaviour when legacy mode is disabled** – retain current warning/exit-code contract (status quo). Revisit if we need stricter failure semantics after component parity improves.
- **Operator control surface** – keep the environment variable (`ML_USE_LEGACY_PIPELINE_ORCHESTRATOR`) as the toggle; no CLI or dashboard switch for now.

## Backlog Guard Assessment

- Hot-path review suggests very large Redis backlogs remain rare; continue monitoring `redis-cli XLEN` alongside the persisted `stream_cursor`. If backlogs consistently exceed operational thresholds, revisit the idea of a config-driven replay cap (`ML_STREAM_PERSIST_MAX_REPLAY_BATCHES`) and update this plan.
