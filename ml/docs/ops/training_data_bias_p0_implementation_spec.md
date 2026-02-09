# Training Data Bias P0 Implementation Spec

Date: 2026-02-08  
Status: Planning-only (no production code changes yet)

## Objective
Execute P0 with low rollout risk by making sampling behavior deterministic, enforcing coverage guardrails, and enforcing target-contract consistency across event-driven streaming planning and training.

## Active-Path Evidence (Current Code)
1. Planner applies limits and guardrails during plan creation.
   - `ml/training/event_driven/dataset_service.py:51`
   - `ml/training/event_driven/dataset_service.py:89`
2. Global run emits pre-split train/validation metadata.
   - `ml/training/event_driven/global_run.py:323`
   - `ml/training/event_driven/global_run.py:464`
   - `ml/training/event_driven/global_run.py:465`
3. Worker re-applies limits in `_prepare_context`, including on pre-split metadata.
   - `ml/training/event_driven/worker.py:980`
   - `ml/training/event_driven/worker.py:996`
   - `ml/training/event_driven/worker.py:1000`
   - `ml/training/event_driven/worker.py:1023`
   - `ml/training/event_driven/worker.py:1027`
4. Limit behavior is shard-granular and drops whole shards when limits would be exceeded.
   - `ml/training/teacher/streaming_loader.py:885`
   - `ml/training/teacher/streaming_loader.py:887`
5. Existing guardrails do not enforce instrument/time coverage.
   - `ml/training/event_driven/guardrails/dataset.py:53`
   - `ml/training/event_driven/guardrails/dataset.py:59`
   - `ml/training/event_driven/guardrails/dataset.py:66`
6. Target-contract validators already exist in shared data metadata utilities.
   - `ml/data/metadata.py:501`
   - `ml/data/metadata.py:636`
   - `ml/data/metadata.py:808`
7. Event-driven plan request/event contracts currently do not carry target semantics payload.
   - `ml/training/event_driven/services.py:59`
   - `ml/training/event_driven/services.py:101`
8. Short-entry fallback still depends on inferred account mode when policy is omitted.
   - `ml/strategies/base_facade.py:260`
   - `ml/strategies/base_facade.py:278`
   - `ml/strategies/ml_strategy.py:417`

## P0 Scope
1. Single-pass selection semantics for planning/training split paths.
2. Hard-fail dataset coverage guardrails (instrument/time/collapse-share).
3. Target semantics contract propagation and validation at plan time.
4. Telemetry needed to confirm no sampling-collapse regressions.

## Proposed Contract Changes
### Config Changes
| Field | Location | Default | Failure behavior |
|---|---|---:|---|
| `min_distinct_instruments: PositiveInt \| None` | `DatasetServiceConfig` | `None` | Raise `DatasetGuardrailError` when selected distinct instruments below threshold. |
| `max_single_instrument_row_share: float \| None` | `DatasetServiceConfig` | `None` | Raise `DatasetGuardrailError` when one instrument exceeds configured row-share bound. |
| `min_temporal_span_bars: PositiveInt \| None` | `DatasetServiceConfig` | `None` | Raise `DatasetGuardrailError` when selected shard time span is below threshold. |
| `enforce_target_semantics_contract: bool` | `DatasetServiceConfig` | `False` | When `True`, missing/invalid semantics or target mismatch raises `DatasetGuardrailError`. |
| `presplit_limit_mode: Literal[\"reuse_plan\", \"legacy_relimit\"]` | `StreamingWorkerConfig` | `"reuse_plan"` | Invalid value rejected by config validation; mode controls pre-split re-limiting behavior. |

### Environment Variables
1. `ML_STREAMING_MIN_DISTINCT_INSTRUMENTS`
2. `ML_STREAMING_MAX_SINGLE_INSTRUMENT_ROW_SHARE`
3. `ML_STREAMING_MIN_TEMPORAL_SPAN_BARS`
4. `ML_STREAMING_ENFORCE_TARGET_SEMANTICS_CONTRACT`
5. `ML_STREAMING_PRESPLIT_LIMIT_MODE`

### Request/Event Contract Changes
| Field | Location | Default | Failure behavior |
|---|---|---:|---|
| `target_semantics: dict[str, object] \| None` | `DatasetPlanRequest` | `None` | If enforcement on and missing/invalid, plan fails with `DatasetGuardrailError`. |
| `target_contract: dict[str, object] \| None` | `DatasetPlanEvent` | `None` | Populated when semantics provided; omitted only when enforcement is disabled. |

### Guardrail Contract Semantics
1. `instrument_coverage`: checks distinct instruments and dominant share from selected shard rows.
2. `temporal_coverage`: checks selected shard global span as `max(time_end) - min(time_start) + 1`.
3. `target_contract`: validates semantics contract plus `target_col` membership in labels/legacy aliases.
4. All configured checks are hard-fail (`DatasetGuardrailError`), not warning-only.

## Module-by-Module Task List
### `ml/config/streaming_pipeline.py`
- [ ] Add `DatasetServiceConfig` fields for instrument/time/target-contract guardrails.
- [ ] Add `StreamingWorkerConfig.presplit_limit_mode`.
- [ ] Extend `__post_init__` validations for new ranges/enums.
- [ ] Extend `DatasetServiceConfig.from_env` and `StreamingWorkerConfig.from_env` for new env vars.
- [ ] Acceptance: invalid ranges/enums fail fast with `ValidationError`.

### `ml/training/event_driven/services.py`
- [ ] Add `target_semantics` to `DatasetPlanRequest`.
- [ ] Add `target_contract` to `DatasetPlanEvent`.
- [ ] Keep fields optional for backward compatibility in first rollout phase.
- [ ] Acceptance: existing callers compile; new fields default safely.

### `ml/training/event_driven/dataset_service.py`
- [ ] Pass request `target_semantics` into plan event.
- [ ] Ensure guardrails can consume request semantics without re-reading external artifacts.
- [ ] Acceptance: planner returns `target_contract` when semantics supplied.

### `ml/training/event_driven/global_run.py`
- [ ] Mirror dataset-service behavior for `target_semantics`/`target_contract` propagation.
- [ ] Ensure guardrails run on per-plan combined metadata with new coverage checks enabled.
- [ ] Acceptance: global plans fail early on contract/coverage violations.

### `ml/training/event_driven/guardrails/dataset.py`
- [ ] Add `_validate_instrument_coverage(...)`.
- [ ] Add `_validate_temporal_coverage(...)`.
- [ ] Add `_validate_target_contract(...)` using shared semantics validators.
- [ ] Invoke new validators in `enforce_dataset_guardrails(...)`.
- [ ] Extend `_GUARDRAIL_COUNTER` dimensions (`instrument_coverage`, `temporal_coverage`, `target_contract`).
- [ ] Acceptance: each configured threshold generates deterministic pass/fail and increments counter with correct labels.

### `ml/training/event_driven/worker.py`
- [ ] Implement `presplit_limit_mode="reuse_plan"` path to avoid independent re-limiting of pre-split metadata.
- [ ] Keep explicit opt-in compatibility path for legacy behavior (`legacy_relimit`).
- [ ] Keep telemetry parity; add explicit mode indicator in `telemetry.caps`.
- [ ] Acceptance: in `reuse_plan` mode, pre-split shard IDs are unchanged between plan and worker context.

### `ml/training/event_driven/payloads.py`
- [ ] Include `target_contract` in plan payload serialization.
- [ ] Include `presplit_limit_mode` and guardrail outputs in payload caps/metadata as needed.
- [ ] Acceptance: payload schema remains backward-compatible for existing consumers.

### `ml/cli/streaming_training_runner.py`
- [ ] Populate `DatasetPlanRequest.target_semantics` from validated dataset metadata.
- [ ] Acceptance: event-driven plans carry semantics contract in runner path by default.

### `ml/training/event_driven/sweep.py`
- [ ] Preserve `target_semantics` when cloning `DatasetPlanRequest` for trials.
- [ ] Acceptance: sweep runs do not silently drop target contract payload.

## Dependency Order and Rollout Phases
### Phase 1: Non-breaking contract plumbing
- [ ] Add request/event/config fields with backward-compatible defaults.
- [ ] Add payload support and tests.

### Phase 2: Guardrail implementation (disabled by default where needed)
- [ ] Implement instrument/time/target-contract validators.
- [ ] Keep enforcement controlled by config.

### Phase 3: Worker selection semantics
- [ ] Add `presplit_limit_mode`.
- [ ] Make `"reuse_plan"` default.

### Phase 4: Progressive enforcement rollout
- [ ] Enable coverage thresholds in staging.
- [ ] Enable `enforce_target_semantics_contract` in staging after producers pass semantics.
- [ ] Promote to production after telemetry baseline check.

## Test Plan
### Unit Tests
- [ ] `ml/tests/unit/config/test_streaming_pipeline_config.py`: new field validation + env parsing.
- [ ] `ml/tests/unit/training/event_driven/test_dataset_service.py`: instrument/time/target-contract guardrails.
- [ ] `ml/tests/unit/training/event_driven/test_global_run.py`: global-run propagation and failure paths.
- [ ] `ml/tests/unit/training/event_driven/test_worker.py`: `presplit_limit_mode` behavior.
- [ ] `ml/tests/unit/training/event_driven/test_bus.py`: payload schema additions.
- [ ] `ml/tests/unit/training/event_driven/test_sweep.py`: request clone preserves semantics.

### Integration Tests
- [ ] `ml/tests/integration/training/event_driven/test_plan_to_result.py`: end-to-end plan->worker pass/fail guardrail scenarios.
- [ ] Add one staging-like integration asserting no pre-split collapse under `reuse_plan`.

### Property Tests
- [ ] Add/extend property coverage for instrument-share and temporal-span invariants under random shard layouts.
- [ ] Validate guardrail monotonicity: stricter thresholds never convert fail->pass.

## Verification Checklist
1. `poetry run mypy ml --strict`
2. `poetry run ruff check ml`
3. `make validate-fixtures`
4. Focused tests via `poetry run pytest -k <area>`
5. Coverage verification for ML expectations via `poetry run coverage report`
6. Confirm coverage policy targets: ML modules >=90%, general Python >=80%.

## Telemetry and Observability Checks (No Sampling-Collapse Regression)
1. Guardrail counter emits expected dimensions/results:
   - `ml_tft_streaming_dataset_guardrails_total{dimension="instrument_coverage",...}`
   - `ml_tft_streaming_dataset_guardrails_total{dimension="temporal_coverage",...}`
   - `ml_tft_streaming_dataset_guardrails_total{dimension="target_contract",...}`
2. Worker telemetry caps include per-instrument selected/total/skipped maps:
   - `worker_train_instrument_rows_selected`
   - `worker_train_instrument_rows_total`
   - `worker_train_instrument_rows_skipped`
   - `worker_validation_instrument_rows_selected`
   - `worker_validation_instrument_rows_total`
   - `worker_validation_instrument_rows_skipped`
   - Source: `ml/training/event_driven/worker.py:1995`
3. Run-level telemetry remains serializable and stable:
   - `ml/training/teacher/streaming_telemetry.py:253`
4. Plan payload includes target contract when present:
   - `ml/training/event_driven/payloads.py:252`

## Risks and Rollback Plan
### Risks
1. False-positive plan failures from aggressive thresholds.
2. Compatibility breaks if semantics enforcement is enabled before all request producers pass payloads.
3. Behavior drift if worker pre-split limit mode changes without explicit rollout.

### Rollback
1. Set coverage thresholds to disabled (`None` / unset env vars).
2. Set `ML_STREAMING_ENFORCE_TARGET_SEMANTICS_CONTRACT=0`.
3. Switch `ML_STREAMING_PRESPLIT_LIMIT_MODE=legacy_relimit` if needed.
4. Keep payload fields optional; do not remove legacy parsing paths in first release.

## Open Questions (With Recommended Defaults)
1. Coverage metric basis for P0: rows vs sequences?  
Recommendation: rows for P0 (`instrument_row_counts`), sequences in P1.
2. Should target contract enforcement be on by default immediately?  
Recommendation: no; default `False`, enable after producer compatibility check.
3. Should worker cap validation data in global pre-split runs?  
Recommendation: default to planner-owned split (`reuse_plan`), retain legacy mode for rollback only.
4. Should short-entry explicitness be enforced in code or rollout policy first?  
Recommendation: enforce in config/release checklist first, code hard-fail in follow-up.

## Ready for Implementation Gate
Implementation may start only when all are true:

- [ ] This spec is approved without unresolved P0 scope changes.
- [ ] Config field names/defaults and env var names are frozen.
- [ ] Rollout defaults are agreed (`enforce_target_semantics_contract=False`, `presplit_limit_mode="reuse_plan"`).
- [ ] Test file targets and command checklist are accepted.
- [ ] Telemetry acceptance checks are agreed and mapped to dashboards/log queries.
