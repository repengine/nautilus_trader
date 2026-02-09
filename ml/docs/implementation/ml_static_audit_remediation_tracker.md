# ML Static Audit Remediation Tracker

## Scope
This document records the static ML risk audit performed on February 6, 2026 and tracks remediation work across the identified risk areas.

## Baseline Audit Snapshot (2026-02-06)

| # | Area | Audit Status | Residual Risk |
|---|---|---|---|
| 1 | Feature state time-bound causality | Partial | Medium |
| 2 | Label causality/execution contract | Partial | Medium |
| 3 | Inference synchronous/unbounded | Mostly true | High |
| 4 | Output semantics underspecified | Partial | Medium |
| 5 | Artifact compatibility/load gates | Partial | High |
| 6 | Drift/health monitoring at inference | Weak/partial | High |
| 7 | Reproducibility/provenance | Partial | Medium |
| 8 | ML errors treated non-fatal | Mostly true | High |
| 9 | Python/core stale-state boundary | Partial | Medium |
| 10 | ML testing completeness for ML correctness | Partial | Medium |

Highest residual risk areas from static audit: **#3, #5, #6, #8**.

## Static Audit Result (Recorded)

1. Feature state time-bound: **Partial**
- Existing anti-leak checks: `ml/features/validation.py:716`, `ml/training/event_driven/guardrails/dataset.py:276`.
- Gap: real-time macro features ignore event time and use latest cache values: `ml/features/macro_transforms.py:224`, `ml/features/pipeline_stream.py:216`.
- Gap: no monotonic timestamp assertions found at actor ingress: `ml/actors/base.py:1330`.

2. Label causality contract: **Partial**
- Target semantics formalized: `ml/config/targets.py:296`.
- Dataset semantics validated during training: `ml/training/common/training_orchestrator.py:407`.
- Gap: labels mostly derived from forward close shift plus fixed cost adjustment: `ml/training/datasets/target_generator.py:643`.
- Gap: fallback simplistic label generation path exists: `ml/training/common/data_preparation.py:190`.

3. Inference synchronous/unbounded: **Mostly true (high risk)**
- Inference is inline in hot path: `ml/actors/base.py:1330`, `ml/actors/base.py:1375`, `ml/actors/base.py:1442`.
- Deadline overrun only warns/updates health: `ml/actors/base.py:1518`.
- Batch actor can still flush from `on_bar`: `ml/actors/multi_signal.py:224`, `ml/actors/multi_signal.py:255`, `ml/actors/multi_signal.py:301`.

4. Output semantics underspecified: **Partial**
- Strong output normalization: `ml/common/prediction_surface.py:111`.
- Gap: output schema/calibration metadata remains optional and best-effort sidecar ingestion: `ml/common/model_sidecar.py:87`, `ml/registry/model_registry_facade.py:236`.

5. Artifact compatibility/load checks: **Partial**
- Strength: schema hash required during registration and digest verification in registry load: `ml/registry/model_registry_facade.py:1242`, `ml/registry/model_registry_facade.py:471`.
- Gap: strict parity env-gated: `ml/registry/model_registry_facade.py:1247`.
- Gap: hot-reload schema mismatch warns only: `ml/registry/common/deployment_manager.py:317`.
- Gap: direct-path ONNX load disables strict integrity: `ml/actors/base.py:574`, `ml/actors/base.py:620`.
- Gap: lexical semantic version resolution: `ml/registry/common/version_manager.py:196`, `ml/registry/common/version_manager.py:238`.

6. Drift/health monitoring at inference: **Weak/partial**
- Drift collectors exist: `ml/monitoring/collectors/features.py:332`, `ml/monitoring/collectors/performance.py:287`.
- Gap: no runtime call sites found for collector drift recording in actor/training/registry flows.
- Current actor health focus is latency/failure, not live distribution drift: `ml/actors/base.py:1514`, `ml/actors/base.py:1522`, `ml/actors/base.py:1572`.

7. Reproducibility/provenance: **Partial**
- Seed/config capture exists in parts of pipeline: `ml/training/event_driven/worker.py:1061`, `ml/training/common/persistence.py:338`.
- Gap: hardcoded trainer seeds (`42`) in non-distilled trainers: `ml/training/non_distilled/xgboost.py:391`, `ml/training/non_distilled/lightgbm.py:253`.
- Gap: no strict, centralized deterministic/hardware provenance contract in manifest.

8. ML errors treated non-fatal: **Mostly true**
- Prediction exceptions are logged and routed through health/circuit breaker: `ml/actors/base.py:1562`.
- Gap: no mandatory risk action (halt/size reduction/model disable) tied to ML failure state in actor path.
- Shared fallback utilities explicitly swallow and continue by design: `ml/common/error_handlers.py:186`.

9. Python/core stale-state boundary: **Partial**
- Reset hook exists in signal facade: `ml/actors/signal_facade_impl.py:818`.
- Gap: indicator-state restoration called out as simplified: `ml/actors/base.py:2443`.
- Gap: explicit replay/rewind invalidation contract not found for all ML state.

10. Testing completeness for ML correctness: **Partial**
- Strong coverage in known-future checks, target semantics, normalization, and registry integrity/parity:
  - `ml/tests/unit/features/test_known_future_transforms.py:566`
  - `ml/tests/unit/training/test_target_generator_semantics.py:22`
  - `ml/tests/unit/common/test_prediction_surface.py:11`
  - `ml/tests/unit/registry/test_model_registry_strict_parity_unit.py:21`
  - `ml/tests/e2e/test_model_registry_e2e.py:338`
- Gap: limited tests for hard inference deadline/drop behavior, runtime drift-triggered actions, and replay/clock-jump invalidation.

## Remediation Priorities

### P0 (Immediate)
- [x] #3 Enforce hard inference deadlines and bounded isolation path.
- [x] #5 Tighten compatibility gate defaults, strict integrity in direct loads, and deterministic version resolution.
- [x] #6 Wire drift collectors into inference-time data path with action thresholds.
- [x] #8 Convert ML failures into explicit strategy/risk state transitions.

### P1 (Next)
- [x] #1 Add monotonic timestamp assertions at ML ingress and timestamp-bound macro real-time mode.
- [x] #2 Formalize execution-aware label contract (latency/fill semantics).
- [x] #7 Remove hardcoded seeds and enforce deterministic provenance capture.

### P2 (Follow-up)
- [x] #9 Define explicit replay/rewind reset contract for all actor ML state.
- [x] #10 Add CI-grade tests for deadline miss behavior, drift-triggered disable paths, and replay invalidation.
- [x] #4 Require output schema/calibration declarations for production models.

## Work Tracking Board

| Workstream | Risk IDs | Owner | Status | Target Date | Notes |
|---|---|---|---|---|---|
| Inference isolation + budget enforcement | #3 | OWNER_TBD (placeholder) | Completed (validated closeout) | TBD | PR-05/PR-06 baseline enforcement and PR-19 strict rollout hardening are completed locally; PR-20 added deterministic deadline outcome + halted-state persistence operational validation evidence; PR-21.1 reran strict validation matrix with `make pytest-ml-runtime-correctness` (`39 passed`) and `make pytest-ml-strict-policy` runtime leg (`39 passed`). |
| Compatibility/integrity hardening | #5 | OWNER_TBD (placeholder) | Completed (local) | TBD | PR-14 completed locally: `RegistryPolicyConfig.from_env` now defaults to strict compatibility/output semantics with explicit opt-out env controls; strict registry load/hot-reload integrity violations no longer degrade to warn-only migration bypass paths. |
| Drift wiring + action policy | #6 | OWNER_TBD (placeholder) | Completed (local) | TBD | PR-07 completed locally: runtime drift monitor wiring, policy thresholds/actions, replay-safe action downgrade, and observability hooks added. |
| Failure-to-risk-state integration | #8 | OWNER_TBD (placeholder) | Completed (validated closeout) | TBD | PR-05/PR-06 transition hooks and PR-07 drift fail-closed reuse were completed locally; PR-19 hardened risk-transition enforcement defaults + halted-state no-downgrade behavior; PR-20 added deterministic live/replay risk-transition hook observability coverage; PR-21.1 reran focused runtime tests and strict lanes with no regressions. |
| Causality and timestamp contracts | #1, #2 | OWNER_TBD (placeholder) | Completed (local) | TBD | PR-08 replay/backstep runtime reset integration, PR-09 epoch-cut target semantics contract (`epoch-1`) enforcement, PR-10 wall-clock horizon alignment, PR-15 ingress monotonic/backstep contract hardening, PR-17 timestamp-bound macro real-time mode hardening, and PR-18 execution-aware label contract hardening are completed locally. |
| Reproducibility hardening | #7 | OWNER_TBD (placeholder) | Completed (local) | TBD | PR-12 completed locally: canonical reproducibility provenance is now serialized and validated across dataset metadata, export sidecars, and persistence-manifest boundaries. |
| Replay/reset contract | #9 | OWNER_TBD (placeholder) | Completed (local) | TBD | PR-08 completed locally: shared base runtime reset hook wired to facade rewind detection with feature/indicator/drift invalidation and replay integration tests. |
| ML correctness test expansion | #10, #4 | OWNER_TBD (placeholder) | Completed (validated closeout) | TBD | PR-13 completed locally for Risk #10 runtime correctness; PR-14 completed locally for Risk #4 production output-semantics blocking contracts; PR-16 completed locally with strict-policy CI composition (`pytest-ml-strict-policy`); PR-21.1 reran `make pytest-ml-runtime-correctness`, `make pytest-ml-registry-hardening`, and `make pytest-ml-strict-policy` with all commands passing. |

## PR Acceptance Checklist (For This Effort)

- [x] Includes explicit risk ID(s) being addressed.
- [ ] Adds/updates tests for new ML behavior and failure modes.
- [x] Runs: `poetry run mypy ml --strict`.
- [x] Runs: `poetry run ruff check ml`.
- [x] Runs focused tests: `poetry run pytest -k <area>`.
- [ ] Runs coverage check and keeps thresholds (ML >= 90%, general >= 80%).
- [x] Runs `make validate-fixtures`.
- [x] Uses `ml.common.metrics_bootstrap` and validates metrics/events when changed.

PR-21 note:
- `poetry run coverage report` was rerun for evidence; repository-wide totals include non-ML packages and are tracked separately from targeted runtime-closeout scope.

## Decision Log

| Date | Decision | Rationale | Owner |
|---|---|---|---|
| 2026-02-06 | Establish static-audit remediation tracker | Create a single source of truth for ML risk hardening execution and evidence | Codex + Team |
| 2026-02-08 | Close PR-21 for risks #3/#8/#10 after full strict matrix rerun | Validation matrix and strict-policy lanes passed end-to-end with no semantic regressions versus PR-09..PR-20 | Codex + Team |

## Progress Log

| Date | Update | Links |
|---|---|---|
| 2026-02-06 | Initial static audit snapshot captured in tracker | This file |
| 2026-02-06 | PR-01 completed: added actor/registry policy scaffolding, remediation metric names/counters, and focused unit tests for env/config behavior | `ml/config/policy.py`, `ml/config/base.py`, `ml/config/registry.py`, `ml/config/names.py`, `ml/common/metrics.py` |
| 2026-02-06 | PR-02 completed (additive): introduced shared `CausalityGuard`, output semantics validator, and reproducibility helper utilities with focused unit tests and validation runs (`mypy`, `ruff`, fixture validation, focused pytest) | `ml/common/causality_guard.py`, `ml/common/output_semantics.py`, `ml/common/reproducibility.py`, `ml/tests/unit/common/test_causality_guard.py`, `ml/tests/unit/common/test_output_semantics.py`, `ml/tests/unit/common/test_reproducibility.py` |
| 2026-02-06 | PR-03 completed (additive): wired registry policy-driven strict compatibility/output semantics gating, strict digest enforcement on registry load path, migration/unsigned override metrics, and focused registry policy tests | `ml/registry/model_registry_facade.py`, `ml/config/policy.py`, `ml/tests/unit/registry/test_model_registry_policy_gating.py`, `ml/tests/unit/config/test_policy_config.py` |
| 2026-02-06 | PR-04 completed (additive): enforced policy-driven digest integrity + output semantics on actor direct-path ONNX loads and expanded sidecar digest extraction aliases with focused unit coverage | `ml/common/model_load_policy.py`, `ml/common/model_sidecar.py`, `ml/actors/base.py`, `ml/actors/common/model.py`, `ml/actors/common/model_warmup.py`, `ml/tests/unit/common/test_model_load_policy.py`, `ml/tests/unit/common/test_model_sidecar.py`, `ml/tests/unit/actors/test_model_loader.py`, `ml/tests/unit/actors/common/test_model_load_policy_gating.py` |
| 2026-02-06 | PR-05 completed locally (additive): wired policy-gated base/facade inference deadline guard (`drop`/`halt`), ML failure action transitions (`log_only`/`degraded`/`halt`), remediation metrics emission, and risk-halt transition hooks with focused unit coverage | `ml/actors/base.py`, `ml/actors/signal_facade_impl.py`, `ml/actors/common/remediation.py`, `ml/tests/unit/actors/test_inference_deadline_guard.py`, `ml/tests/unit/actors/test_signal_facade_impl_unit.py`, `ml/tests/unit/actors/common/test_remediation.py` |
| 2026-02-06 | PR-06 completed locally (additive): aligned multi-signal batch dispatch with shared deadline/failure policy hooks; `on_bar`/flush now enforce halted-state and timeout outcomes while preserving permissive defaults | `ml/actors/multi_signal.py`, `ml/tests/unit/actors/test_multi_signal_actor.py`, `ml/docs/implementation/ml_static_audit_remediation_tracker.md` |
| 2026-02-06 | PR-07 completed locally (additive): wired runtime drift monitoring into facade inference path, added shared drift outcome remediation hook in base actor, gated actions by warmup baseline/sample windows, and added replay-safe drift-action tests | `ml/actors/common/drift_monitoring.py`, `ml/actors/base.py`, `ml/actors/signal_facade_impl.py`, `ml/actors/common/model_warmup.py`, `ml/tests/unit/actors/test_drift_monitoring_component.py`, `ml/tests/unit/actors/test_signal_facade_impl_unit.py`, `ml/docs/implementation/ml_static_audit_remediation_tracker.md` |
| 2026-02-06 | PR-08 completed locally (additive): integrated shared replay/rewind runtime reset lifecycle in base actor, wired facade backstep detection to pre-inference reset, added feature/indicator/drift invalidation hooks, and added focused replay reset tests | `ml/actors/base.py`, `ml/actors/signal_facade_impl.py`, `ml/actors/common/features.py`, `ml/actors/common/drift_monitoring.py`, `ml/tests/unit/actors/test_base_actor_cold_path.py`, `ml/tests/unit/actors/test_signal_facade_impl_unit.py`, `ml/tests/unit/actors/common/test_features.py`, `ml/tests/integration/replay/test_actor_reset_on_rewind.py` |
| 2026-02-07 | PR-09 completed locally as a pre-alpha epoch cut: canonical target semantics contract (`epoch-1`) is now single-source and enforced in config parsing, metadata emission/validation, and orchestrator training gate; no v1/v2 migration path retained | `ml/config/targets.py`, `ml/training/datasets/target_generator.py`, `ml/data/metadata.py`, `ml/training/common/training_orchestrator.py`, `ml/tests/contracts/test_dataset_target_semantics_contracts.py`, `ml/tests/unit/data/test_target_semantics_metadata.py`, `ml/tests/unit/training/common/test_training_orchestrator.py`, `ml/tests/unit/config/test_target_semantics_config.py` |
| 2026-02-07 | PR-10 completed locally: added canonical horizon resolution mode semantics (`bar_index`/`wall_clock`), shared wall-clock timestamp-aligned target generation, strict metadata alignment validation, and orchestrator fail-fast mode mismatch checks | `ml/config/targets.py`, `ml/training/datasets/target_generator.py`, `ml/data/metadata.py`, `ml/training/common/training_orchestrator.py`, `ml/data/__init__.py`, `ml/tests/unit/training/test_target_generator_horizon_alignment.py`, `ml/tests/unit/data/test_target_semantics_metadata.py`, `ml/tests/unit/training/common/test_training_orchestrator.py`, `ml/tests/contracts/test_dataset_target_semantics_contracts.py`, `ml/tests/unit/config/test_target_semantics_config.py`, `ml/tests/utils/targets.py` |
| 2026-02-07 | PR-11 completed locally: removed hardcoded non-distilled trainer/HPO seeds, added canonical config-driven seed resolution/reuse across trainer + worker + hyperparameter paths, and added focused determinism guard tests | `ml/common/reproducibility.py`, `ml/training/non_distilled/xgboost.py`, `ml/training/non_distilled/lightgbm.py`, `ml/training/common/hyperparameter.py`, `ml/training/optuna_optimizer.py`, `ml/training/event_driven/worker.py`, `ml/tests/unit/common/test_reproducibility.py`, `ml/tests/unit/training/common/test_hyperparameter.py`, `ml/tests/unit/training/event_driven/test_worker.py`, `ml/tests/unit/training/test_non_distilled_seed_config.py` |
| 2026-02-07 | PR-12 completed locally: added one shared config-driven provenance builder/validator, serialized deterministic/runtime provenance into dataset metadata and model/export artifacts, and added focused round-trip/persistence boundary tests | `ml/common/reproducibility.py`, `ml/data/build.py`, `ml/data/metadata.py`, `ml/data/__init__.py`, `ml/training/common/persistence.py`, `ml/training/export.py`, `ml/tests/unit/common/test_reproducibility.py`, `ml/tests/unit/data/test_metadata_defaults.py`, `ml/tests/unit/training/common/test_persistence.py`, `ml/tests/unit/training/test_export_provenance.py` |
| 2026-02-07 | PR-13 completed locally: added dedicated runtime-correctness CI selector/entrypoint, expanded deadline timeout/halt persistence invariants, strengthened warmup-gated drift action assertions, expanded replay rewind recovery/stale-state rejection coverage, and added CI aggregation contracts for runtime subset wiring | `pytest.ini`, `ml/pytest.ini`, `Makefile`, `ml/tests/unit/actors/test_inference_deadline_guard.py`, `ml/tests/unit/actors/test_signal_facade_impl_unit.py`, `ml/tests/integration/replay/test_actor_reset_on_rewind.py`, `ml/tests/contracts/test_runtime_correctness_contracts.py`, `ml/docs/implementation/ml_static_audit_remediation_tracker.md` |
| 2026-02-07 | PR-14 completed locally: hardened strict-by-default registry compatibility/output semantics config derivation, enforced strict output-semantics validation under strict compatibility, blocked strict digest/hot-reload migration warn-bypass paths, added production strict contracts, and added a targeted registry hardening CI lane | `ml/config/registry.py`, `ml/registry/model_registry_facade.py`, `ml/tests/unit/registry/test_model_registry_policy_gating.py`, `ml/tests/contracts/test_registry_behavioral.py`, `Makefile`, `ml/docs/implementation/ml_static_audit_remediation_tracker.md` |
| 2026-02-07 | PR-16 completed locally: added composed strict-policy CI lane (`pytest-ml-strict-policy`) with fail-fast runtime→registry sequencing, expanded runtime contract enforcement for composed lane drift detection, and added strict-default production registration/load fail-fast contracts for compatibility/output/digest gates | `Makefile`, `ml/tests/contracts/test_runtime_correctness_contracts.py`, `ml/tests/contracts/test_registry_behavioral.py`, `ml/docs/implementation/ml_static_audit_remediation_tracker.md` |
| 2026-02-07 | PR-17 completed locally: hardened macro real-time causality to require event-time-bounded eligibility (no future release leakage), added deterministic empty-output behavior when no eligible release exists, enforced macro streaming timestamp requirements, and added focused macro/pipeline stream tests for watermark-safe backstep handling | `ml/features/macro_transforms.py`, `ml/features/pipeline_stream.py`, `ml/tests/unit/features/test_macro_transforms_parity.py`, `ml/tests/unit/features/test_pipeline_stream_executor.py`, `ml/docs/implementation/ml_static_audit_remediation_tracker.md` |
| 2026-02-07 | PR-18 completed locally: added canonical execution-aware label contract fields (execution price columns, latency bars, unresolved-context mode), emitted strict execution metadata in dataset target semantics, enforced metadata/config/orchestrator execution alignment fail-fast checks, and expanded contract/unit tests for deterministic unresolved execution handling | `ml/config/targets.py`, `ml/training/datasets/target_generator.py`, `ml/data/metadata.py`, `ml/training/common/training_orchestrator.py`, `ml/tests/contracts/test_dataset_target_semantics_contracts.py`, `ml/tests/unit/training/test_target_generator_horizon_alignment.py`, `ml/tests/unit/training/common/test_training_orchestrator.py`, `ml/tests/unit/data/test_target_semantics_metadata.py`, `ml/tests/unit/config/test_target_semantics_config.py` |
| 2026-02-07 | PR-19 completed locally: hardened strict rollout for Risks #3/#8 by applying production-like env strict defaults for deadline/failure policy (`deadline_guard=true`, timeout/failure action `halt`) unless explicitly overridden, and tightened actor fail-closed behavior to prevent post-halt policy downgrade while surfacing missing risk-transition hooks deterministically | `ml/config/base.py`, `ml/actors/base.py`, `ml/tests/unit/config/test_env_builders.py`, `ml/tests/unit/actors/test_inference_deadline_guard.py`, `ml/tests/contracts/test_runtime_correctness_contracts.py`, `ml/tests/unit/actors/test_signal_facade_impl_unit.py`, `ml/tests/unit/actors/test_base_actor_cold_path.py`, `ml/tests/unit/actors/test_multi_signal_actor.py` |
| 2026-02-07 | PR-20 completed locally: added deterministic operational validation coverage for strict rollout outcomes (`drop`/`halt`, `log_only`/`degraded`/`halt`, halted-state persistence/non-downgrade), replay/live-like risk-transition payload checks, unavailable/failure hook observability paths, and runtime-correctness lane inclusion for multi-signal strict remediation tests; also hardened risk-transition logging fallback for deterministic fail-closed reason marking under logger keyword incompatibilities | `ml/tests/unit/actors/test_inference_deadline_guard.py`, `ml/tests/unit/actors/test_multi_signal_actor.py`, `ml/tests/contracts/test_runtime_correctness_contracts.py`, `Makefile`, `ml/actors/base.py`, `ml/docs/implementation/ml_static_audit_remediation_tracker.md` |
| 2026-02-08 | PR-21 completed locally (closeout): reran full strict validation matrix (`mypy`, `ruff`, `validate-fixtures`, focused runtime pytest, `coverage report`, `validate-metrics`, `validate-events`, runtime/registry/strict-policy lanes), all commands exited successfully; updated risk DoD closeout evidence and residual-risk decision for #3/#8/#10 with no runtime code changes required | `ml/docs/implementation/ml_static_audit_remediation_tracker.md`, `ml/tests/unit/actors/test_inference_deadline_guard.py`, `ml/tests/unit/actors/test_multi_signal_actor.py`, `ml/tests/contracts/test_runtime_correctness_contracts.py`, `Makefile` |
| 2026-02-08 | Strict-policy alignment final closeout pass: fixed xdist-sensitive symbology metric assertion drift using delta-based counter assertions in tests, reran strict lanes, and confirmed full-suite baseline with zero failures (`8158 passed, 85 skipped, 1 xfailed`). | `ml/tests/unit/data/ingest/test_symbology.py`, `ml/docs/implementation/strict_policy_test_alignment_plan.md` |

## Synthesis Method (2026-02-06)

This remediation plan was synthesized from 5 parallel analysis streams:

| Stream | Risk IDs | Focus |
|---|---|---|
| A | #3, #8 | Bounded inference + failure-to-risk-state behavior |
| B | #5, #4 | Compatibility/integrity gates + output semantics contracts |
| C | #6, #10 | Runtime drift wiring + runtime correctness testing |
| D | #1, #2, #9 | Causality contracts + label realism + reset/rewind invalidation |
| E | #7 | Determinism and provenance |

## Integrated Remediation Program

### Program Principles
- Strict-by-default for production safety, with temporary explicit compatibility flags only where migration is required.
- Keep hot path bounded and allocation-light; do heavy checks on cold path/interval.
- Reuse common/protocol components; no duplicate implementations across actors/stores.
- Every behavior change ships with targeted tests and rollout telemetry.

### Cross-Stream Design Decisions
- Introduce one shared causality guard capability (single implementation) and reuse it in actor + feature compute paths.
- Enforce artifact integrity and output semantics at both registry and direct-path loading boundaries.
- Treat inference deadline misses and ML runtime errors as explicit state transitions, not passive logs.
- Add a runtime drift policy component with config-driven thresholds/actions and replay/backtest-safe behavior.
- Add deterministic execution and provenance capture as first-class metadata in dataset + model artifacts.

## Per-Risk Remediation Spec (Synthesis)

| Risk | Required Fixes | Primary Files | Mandatory Tests |
|---|---|---|---|
| #1 Feature causality | Add `CausalityGuard`; enforce monotonic/no-future checks before state mutation | `ml/actors/common/features.py`, `ml/stores/common/feature_computation.py`, `ml/features/pipeline_stream.py`, `ml/config/actors.py` | `ml/tests/unit/actors/test_causality_guard.py`, `ml/tests/unit/stores/test_feature_computation_causality.py`, `ml/tests/unit/features/test_pipeline_stream_causality.py` |
| #2 Label realism | Extend target semantics with execution-aware fields and timestamp-based horizon option | `ml/config/targets.py`, `ml/training/datasets/target_generator.py`, `ml/training/common/training_orchestrator.py`, `ml/data/metadata.py` | `ml/tests/unit/training/test_target_semantics_contract.py`, `ml/tests/unit/training/test_target_generator_horizon_alignment.py` |
| #3 Bounded inference | Add deadline guard path and bounded execution semantics; drop/halt on timeout | `ml/config/base.py`, `ml/actors/base.py`, `ml/actors/signal_facade_impl.py`, `ml/actors/multi_signal.py` | `ml/tests/unit/actors/test_inference_deadline_guard.py`, `ml/tests/unit/actors/test_signal_facade_deadline.py`, `ml/tests/integration/test_ml_signal_pipeline.py` |
| #4 Output semantics | Require explicit output schema/positive-class semantics for serveable models; validate at load | `ml/registry/model_registry_facade.py`, `ml/common/prediction_surface.py`, `ml/common/model_sidecar.py` | `ml/tests/unit/registry/test_model_registry_facade.py`, `ml/tests/unit/common/test_prediction_surface.py` |
| #5 Compatibility/integrity | Strict digest enforcement in all load paths; tighten parity defaults | `ml/config/registry.py`, `ml/registry/common/model_persistence.py`, `ml/common/security.py`, `ml/actors/common/model.py` | `ml/tests/integration/registry/test_model_registry_security.py`, `ml/tests/unit/actors/test_base_actor_cold_path.py` |
| #6 Drift runtime health | Add inference-time drift monitor, policy thresholds, and action hooks | `ml/config/actors.py`, `ml/actors/common/drift_monitoring.py`, `ml/actors/signal_facade_impl.py`, `ml/actors/common/model_warmup.py` | `ml/tests/unit/actors/test_drift_monitoring_component.py`, `ml/tests/unit/actors/test_mlsignal_actor_facade.py`, `ml/tests/integration/test_replay_drift_policy.py` |
| #7 Reproducibility | Remove hardcoded seeds; deterministic mode helper; capture provenance | `ml/common/reproducibility.py`, `ml/training/common/hyperparameter.py`, `ml/training/non_distilled/xgboost.py`, `ml/training/non_distilled/lightgbm.py`, `ml/data/metadata.py`, `ml/training/export.py` | `ml/tests/unit/training/common/test_hyperparameter_seed.py`, `ml/tests/unit/training/event_driven/test_worker.py`, `ml/tests/unit/data/test_metadata_defaults.py`, `ml/tests/unit/training/test_export_provenance.py` |
| #8 Failure semantics | Convert ML failures into explicit halt/degrade transitions + strategy store risk events | `ml/actors/base.py`, `ml/actors/signal_facade_impl.py`, `ml/stores/strategy_store.py` | `ml/tests/unit/actors/test_inference_deadline_guard.py`, `ml/tests/integration/test_ml_strategy_backtest.py` |
| #9 Replay/reset invalidation | Define and invoke unified actor reset lifecycle on rewind/backstep | `ml/actors/base.py`, `ml/actors/signal_facade_impl.py`, `ml/features/facade.py`, `ml/features/indicators.py` | `ml/tests/integration/replay/test_actor_reset_on_rewind.py` |
| #10 Runtime ML testing gaps | Add runtime contract tests for deadline, drift-action, and replay state correctness | `ml/tests/unit/actors/*`, `ml/tests/integration/*`, `ml/tests/property/*` | New runtime-focused suites listed in this document |

## Dependency Graph

### Foundation Dependencies
- F1: Config scaffolding (`CausalityConfig`, inference deadline config, drift config, strict registry policy flags).
- F2: Shared utilities (`CausalityGuard`, reproducibility helper, output semantics validator).
- F3: Metrics scaffolding for deadline/drift/halt/compatibility migration counters.

### Execution Dependencies
- `#3/#8` depends on F1 + F3.
- `#5/#4` depends on F1 + F2.
- `#6` depends on F1 + F3 and optionally `#3` gating hooks.
- `#1/#9` depends on F1 + F2.
- `#2` depends on `#1` timestamp contract and metadata extensions.
- `#7` independent of trading-runtime behavior but required before hard reproducibility claims.
- `#10` spans all workstreams and should be delivered in each PR slice, not deferred.

## PR Slice Plan (Ordered, Mergeable)

| Slice | Scope | Risks | Dependencies | Acceptance |
|---|---|---|---|---|
| PR-01 | Config + policy scaffolding and metrics names (no behavior change) | #1 #3 #4 #5 #6 #7 #8 #9 | None | Type/lint clean; no runtime behavior regression |
| PR-02 | Shared validators/utilities (`CausalityGuard`, output semantics validator, reproducibility helper) | #1 #4 #7 #9 | PR-01 | Unit tests for utility behavior |
| PR-03 | Registry strict compatibility/output semantics gating + digest strictness in registry path | #4 #5 | PR-01 PR-02 | Registry unit/integration tests pass |
| PR-04 | Direct model-path loader integrity/semantics enforcement + sidecar metadata expansion | #4 #5 | PR-03 | Actor cold-path tests pass |
| PR-05 | Base/facade inference deadline guard + failure state transition hooks (flagged rollout) | #3 #8 | PR-01 PR-02 | Deadline unit tests + integration smoke |
| PR-06 | Multi-signal batch deadline behavior alignment | #3 #8 | PR-05 | Multi-signal behavior tests pass |
| PR-07 | Runtime drift monitor wiring + action policy + observability | #6 | PR-01 PR-02 PR-05 | Drift unit tests + replay safety tests |
| PR-08 | Replay/rewind reset lifecycle integration | #1 #9 | PR-02 PR-05 | Replay integration tests pass |
| PR-09 | Target semantics epoch cut (`epoch-1`) + strict contract validation | #2 | PR-01 PR-08 | Target contract tests pass |
| PR-10 | Timestamp-based horizon label generation mode (`wall_clock`) | #2 | PR-09 | Target generation alignment tests pass |
| PR-11 | Determinism cleanup (remove hardcoded seeds + deterministic worker/dataloader) | #7 | PR-01 PR-02 | Reproducibility unit tests pass |
| PR-12 | Provenance capture in dataset/model artifacts + registry serialization strategy | #7 | PR-11 | Metadata round-trip and export tests pass |
| PR-13 | Runtime correctness test expansion and CI entry points | #10 | PR-05 PR-07 PR-08 PR-10 PR-12 | New CI subset stable and green |
| PR-14 | Strict compatibility/output-semantics hardening + targeted CI lane | #5 #4 | PR-03 PR-04 PR-13 | Strict default + strict-mode blocking tests/contracts pass |
| PR-15 | Ingress causality monotonic/backstep contract hardening | #1 | PR-08 PR-13 | Base ingress guard enforces warn/drop/reset before state mutation with focused unit + runtime contract coverage |
| PR-16 | Strict CI lane composition and production policy contracts | #10 #5 #4 | PR-14 PR-15 | Runtime+registry strict subsets are required and contract-enforced in a single CI profile |
| PR-17 | Timestamp-bound macro real-time mode hardening | #1 | PR-15 PR-16 | Streaming macro path is event-time-bounded (no future macro leakage), requires timestamp in stream execution, and handles no-eligible/backstep cases deterministically |
| PR-18 | Execution-aware label contract hardening | #2 | PR-10 PR-17 | Label metadata/config/orchestrator contracts include execution-aware fields with fail-fast mismatch gates and contract coverage |
| PR-19 | Strict rollout hardening for deadline/failure risk transitions | #3 #8 | PR-05 PR-06 PR-16 PR-18 | Production-like env defaults enforce strict deadline/failure posture unless explicitly overridden; halted state is fail-closed/non-downgradable and missing risk-transition hook is observable/deterministic |
| PR-20 | Operational validation + telemetry acceptance for strict rollout | #3 #8 | PR-19 | Replay/live-like deterministic validation for halt/degraded outcomes, risk-transition hook observability checks, and acceptance evidence in runtime correctness artifacts |
| PR-21 | Program closeout + DoD evidence finalization | #3 #8 #10 | PR-20 | Full strict validation matrix passes with evidence links and remaining risk DoD states marked complete |

## PR Slice Status

| Slice | Owner | Status | Updated | Notes |
|---|---|---|---|---|
| PR-01 | OWNER_TBD (placeholder) | Completed | 2026-02-06 | Additive config/policy scaffolding and remediation metric naming landed with no runtime behavior change. |
| PR-02 | OWNER_TBD (placeholder) | Completed | 2026-02-06 | Added shared common-layer validators/utilities only (no actor/registry enforcement wiring yet) to keep slice backward-compatible. |
| PR-03 | OWNER_TBD (placeholder) | Completed | 2026-02-06 | Registry path now enforces policy-driven compatibility/output semantics/digest gates with strict mode + migration override semantics while preserving permissive defaults. |
| PR-04 | OWNER_TBD (placeholder) | Completed | 2026-02-06 | Direct-path actor ONNX loaders now apply shared policy-gated digest/semantics checks with sidecar digest alias expansion while keeping permissive defaults unchanged. |
| PR-05 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-06 | Base/facade inference paths now share policy-driven deadline guard + ML failure transition hooks and emit remediation metrics; strict behavior remains flag-gated. |
| PR-06 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-06 | Multi-signal batch path now applies shared deadline/failure helpers, enforces halted-state in `on_bar`, and stops further dispatch on timeout-driven halt. |
| PR-07 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-06 | Runtime drift monitor now records inference-time drift, maps policy thresholds to actions, applies replay-safe downgrades, and routes fail-closed outcomes through shared halt remediation hooks. |
| PR-08 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-06 | Shared base replay reset hook now clears inference halt/failure + feature/health/circuit state, facade detects timestamp backsteps before halted gating, and replay integration tests validate post-rewind recovery. |
| PR-09 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-07 | Completed explicit pre-alpha epoch cut: single canonical target semantics contract (`epoch-1`) with strict config parsing, metadata contract/capability validation, and orchestrator fail-fast gate (no v1/v2 migration path). |
| PR-10 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-07 | Added explicit horizon resolution mode semantics (`bar_index`/`wall_clock`) with shared timestamp-aligned label generation, strict metadata horizon alignment validation, and orchestrator fail-fast mode mismatch checks. |
| PR-11 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-07 | Removed hardcoded seed defaults in non-distilled trainer/HPO paths, introduced one shared seed resolver, and propagated resolved seeds deterministically through streaming worker + Optuna sampler flows with focused unit coverage. |
| PR-12 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-07 | Serialized canonical reproducibility provenance (resolved seed, deterministic mode, runtime/version fields) into dataset metadata and model/export artifacts with strict payload validation and focused boundary tests. |
| PR-13 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-07 | Added `runtime_correctness` marker + CI entrypoint, expanded deadline/drift/rewind runtime invariants, and introduced runtime aggregation contracts for targeted suite wiring. |
| PR-14 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-07 | Enforced strict-by-default registry compatibility/output semantics env derivation, tightened strict output-semantics enforcement in facade policy checks, blocked strict digest/hot-reload warn-bypass paths, added strict policy gating tests + production contracts, and added `pytest-ml-registry-hardening` targeted lane. |
| PR-15 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-07 | Added policy-driven ingress monotonic/backstep guard in base actor before state mutation/halt gates, reused one shared monotonic helper across base+facade paths, expanded runtime-correctness lane coverage to include base ingress tests, and added runtime contract ordering assertions. |
| PR-16 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-07 | Added `pytest-ml-strict-policy` composed lane, runtime contract enforcement for composed/fail-fast ordering, and strict-default production registration/load fail-fast contracts for compatibility/output semantics/digest gates. |
| PR-17 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-07 | Hardened `MacroFeatureTransform.compute_realtime` to enforce event-time-bounded release eligibility with deterministic no-eligible/backstep handling, and updated streaming executor to require/pass `timestamp_ns` for macro transforms with focused unit coverage. |
| PR-18 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-07 | Risk #2 execution-aware label contract hardening completed across config/target-generation/metadata/orchestrator paths with strict fail-fast alignment gates and deterministic unresolved-context handling covered by focused contract/unit tests. |
| PR-19 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-07 | Hardened Risks #3/#8 strict rollout by applying production-like env strict remediation defaults in actor config, adding non-downgradable halted-state guardrails, and ensuring missing risk-transition hooks are surfaced deterministically with focused runtime/config contracts. |
| PR-20 | OWNER_TBD (placeholder) | Completed (local) | 2026-02-07 | Added deterministic operational validation coverage for strict rollout outcomes (deadline drop/halt, ML failure log_only/degraded/halt, halted-state non-downgrade), replay/live-like risk-transition payload behavior, unavailable/failure hook observability paths, and runtime-correctness lane inclusion for multi-signal strict remediation tests. |
| PR-21 | OWNER_TBD (placeholder) | Completed (local closeout) | 2026-02-08 | Ran full strict validation matrix with passing outcomes, finalized DoD evidence links for risks #3/#8/#10, and completed residual-risk sweep with no semantic regressions versus PR-09..PR-20. |

## PR-13+ Forward Plan (Concrete First Edits for PR-13)

### PR-05 Outcome (Completed Locally)
- Implemented base/facade deadline guard and ML failure transitions behind policy controls with permissive defaults preserved.
- Added one shared decision helper in `ml/actors/common/remediation.py` to keep timeout/failure policy logic single-sourced.
- Added focused unit coverage for permissive vs strict paths in base + facade actor flows.

### PR-06 Outcome (Completed Locally)
- Multi-signal batched inference now reuses PR-05 remediation hooks for deadline and failure actions (Risks #3 and #8) with additive rollout behavior preserved.

### PR-07 Outcome (Completed Locally)
- Added shared runtime drift policy component (`ml/actors/common/drift_monitoring.py`) with config accessors, threshold mapping helpers, and inference-time drift scoring.
- Wired drift recording + policy evaluation into facade prediction flow (`ml/actors/signal_facade_impl.py`) with replay/backtest-safe action downgrades.
- Added a single shared base hook for drift-to-remediation transitions (`ml/actors/base.py`): `log_only` observability, `degraded` health transition, and fail-closed routing through existing halt path.
- Extended warmup gating (`ml/actors/common/model_warmup.py`) so drift actions require minimum baseline/sample windows.
- Added focused tests for drift thresholds/actions and replay-safe behavior (`ml/tests/unit/actors/test_drift_monitoring_component.py`, `ml/tests/unit/actors/test_signal_facade_impl_unit.py`).

### PR-08 Outcome (Completed Locally)
- Added one shared runtime reset hook in `ml/actors/base.py` that clears halted/failure state, warm-up counters, feature windows, health monitor counters, and circuit breaker state.
- Added pre-bar runtime hook in `ml/actors/base.py` and facade backstep detection in `ml/actors/signal_facade_impl.py` so rewind resets occur before halted-gate and post-rewind inference.
- Extended invalidation points via `ml/actors/common/features.py` (`reset_runtime_state`) and `ml/actors/common/drift_monitoring.py` (`reset_runtime_state`), plus facade indicator/feature-engineer reset handling.
- Added focused replay reset tests: unit coverage in `ml/tests/unit/actors/test_signal_facade_impl_unit.py`, `ml/tests/unit/actors/test_base_actor_cold_path.py`, `ml/tests/unit/actors/common/test_features.py`, and integration coverage in `ml/tests/integration/replay/test_actor_reset_on_rewind.py`.

### PR-09 Outcome (Completed Locally)
- Owner: `OWNER_TBD (placeholder)`.
- Canonical `epoch-1` target semantics contract is now single-source across `ml/config/targets.py`, `ml/training/datasets/target_generator.py`, `ml/data/metadata.py`, and `ml/training/common/training_orchestrator.py`.
- Added strict parse-time guards for `TargetSemanticsConfig.from_dict`/`from_json` contract payloads and focused contract-gating tests.
- Kept default label math unchanged in this slice by design.

### PR-10 Target
- Implemented timestamp-based horizon label generation mode (`wall_clock`) for Risk #2 while preserving the PR-09 canonical contract boundary.

### PR-10 Outcome (Completed Locally)
- Owner: `OWNER_TBD (placeholder)`.
- Extended canonical target semantics config (`ml/config/targets.py`) with explicit horizon resolution mode and wall-clock timestamp-column semantics while keeping `bar_index` default behavior unchanged.
- Added one shared wall-clock forward-return resolver in `ml/training/datasets/target_generator.py` reused by both Polars and Pandas generation paths; unresolved future timestamps are handled deterministically via `zero_return`.
- Emitted and validated required metadata horizon mode/alignment fields in `ml/training/datasets/target_generator.py` and `ml/data/metadata.py`, and wired orchestrator fail-fast config-vs-metadata mode checks in `ml/training/common/training_orchestrator.py`.
- Added focused coverage for horizon alignment, metadata validation, orchestrator mismatch gating, and contract metadata assertions in:
  - `ml/tests/unit/training/test_target_generator_horizon_alignment.py`
  - `ml/tests/unit/data/test_target_semantics_metadata.py`
  - `ml/tests/unit/training/common/test_training_orchestrator.py`
  - `ml/tests/contracts/test_dataset_target_semantics_contracts.py`
  - `ml/tests/unit/config/test_target_semantics_config.py`

### PR-11 Outcome (Completed Locally)
- Owner: `OWNER_TBD (placeholder)`.
- Removed hardcoded trainer seed defaults in `ml/training/non_distilled/xgboost.py` and `ml/training/non_distilled/lightgbm.py`; both now resolve seeds from config via one shared helper.
- Centralized seed validation/resolution in `ml/common/reproducibility.py` and reused it in `ml/training/common/hyperparameter.py`, `ml/training/optuna_optimizer.py`, and `ml/training/event_driven/worker.py`.
- Added focused coverage for resolver guards, sampler seed propagation, worker seed fallback, and trainer-level seed assertions in:
  - `ml/tests/unit/common/test_reproducibility.py`
  - `ml/tests/unit/training/common/test_hyperparameter.py`
  - `ml/tests/unit/training/event_driven/test_worker.py`
  - `ml/tests/unit/training/test_non_distilled_seed_config.py`

### PR-12 Outcome (Completed Locally)
- Owner: `OWNER_TBD (placeholder)`.
- Added one shared config-driven reproducibility builder/validator in `ml/common/reproducibility.py` and reused it across dataset + training export/persistence boundaries.
- Extended dataset artifact metadata serialization in `ml/data/build.py` + `ml/data/metadata.py` to persist canonical reproducibility payloads (`seed`, `deterministic_mode`, runtime/version fields) with strict payload validation.
- Added export/persistence provenance serialization in `ml/training/export.py` and `ml/training/common/persistence.py` so sidecar metadata and manifest `training_config` include consistent reproducibility payloads.
- Added focused coverage for round-trip serialization, export payload validation, and persistence boundary guards in:
  - `ml/tests/unit/data/test_metadata_defaults.py`
  - `ml/tests/unit/training/test_export_provenance.py`
  - `ml/tests/unit/training/common/test_persistence.py`
  - `ml/tests/unit/common/test_reproducibility.py`

### PR-13 Outcome (Completed Locally)
- Owner: `OWNER_TBD (placeholder)`.
- Added one dedicated runtime-correctness selector marker (`runtime_correctness`) in `pytest.ini` and `ml/pytest.ini`.
- Added a dedicated Makefile CI entrypoint (`pytest-ml-runtime-correctness`) wired to deadline/drift/rewind runtime files plus CI aggregation contracts.
- Expanded deadline/runtime guard tests in `ml/tests/unit/actors/test_inference_deadline_guard.py`:
  - no-publish-on-timeout invariant for `DROP` and `HALT`
  - halted-state persistence invariant after timeout-triggered halt
- Expanded drift action assertions in `ml/tests/unit/actors/test_signal_facade_impl_unit.py` for `LOG_ONLY`/`DEGRADED`/`FAIL_CLOSED` under warmup-ready gating and replay-safe downgrade behavior.
- Expanded replay integration assertions in `ml/tests/integration/replay/test_actor_reset_on_rewind.py` for post-rewind inference recovery and stale-state rejection.
- Added CI-focused runtime aggregation contracts in `ml/tests/contracts/test_runtime_correctness_contracts.py` to enforce selector/entrypoint/module wiring consistency.

### PR-14 Outcome (Completed Locally)
- Owner: `OWNER_TBD (placeholder)`.
- Tightened strict-by-default env derivation in `ml/config/registry.py` so `RegistryPolicyConfig.from_env` now defaults to strict compatibility + required output semantics unless explicitly relaxed.
- Hardened strict enforcement in `ml/registry/model_registry_facade.py`:
  - strict compatibility now implies output-semantics-required validation
  - strict output-semantics, missing-digest, and hot-reload schema mismatch violations no longer route through migration warn-bypass handling.
- Added strict-mode blocking coverage in `ml/tests/unit/registry/test_model_registry_policy_gating.py` for strict defaults, strict output semantics, strict digest integrity, and strict hot-reload compatibility.
- Added production strict contract checks in `ml/tests/contracts/test_registry_behavioral.py` and preserved legacy behavioral scenarios via explicit permissive policy wiring.
- Added targeted CI lane entrypoint `pytest-ml-registry-hardening` in `Makefile`.

### PR-15 Outcome (Completed Locally)
- Owner: `OWNER_TBD (placeholder)`.
- Added policy-driven ingress causality enforcement in `ml/actors/base.py` before any pre-bar state mutation and before halted/deadline gates:
  - shared guard hook in `on_bar` with `warn_only`/`drop`/`reset` behavior from `remediation_policy`.
  - causality violation metric emission via `causality_monotonic_violations_total`.
  - ingress timestamp watermark (`_last_ingress_ts_event`) reset-safe with replay/runtime invalidation.
- Added one shared helper in `ml/actors/common/features.py` (`is_monotonic_ingress_timestamp`) and reused it in both `ml/actors/base.py` and `ml/actors/signal_facade_impl.py` to keep backstep detection single-sourced.
- Added focused ingress monotonic unit coverage in `ml/tests/unit/actors/test_base_actor_cold_path.py` for strict mode differences (`warn_only`/`drop`/`reset`) and ordering behavior.
- Extended runtime correctness contracts in `ml/tests/contracts/test_runtime_correctness_contracts.py` with ingress guard ordering assertions and updated runtime lane module composition.
- Extended the existing runtime-correctness CI subset in `Makefile` to include `ml/tests/unit/actors/test_base_actor_cold_path.py` (marker-scoped).

### PR-16 Outcome (Completed Locally)
- Owner: `OWNER_TBD (placeholder)`.
- Added one combined strict-policy CI lane in `Makefile` (`pytest-ml-strict-policy`) that composes `pytest-ml-runtime-correctness` then `pytest-ml-registry-hardening` with explicit fail-fast chaining.
- Expanded runtime contracts in `ml/tests/contracts/test_runtime_correctness_contracts.py` to enforce strict-policy lane composition and fail-fast execution ordering.
- Expanded strict production contracts in `ml/tests/contracts/test_registry_behavioral.py`:
  - strict-default registration blocks feature parity mismatch (`feature_schema_hash`) in addition to missing feature-set/output semantics gates.
  - strict-default load blocks missing output semantics and missing artifact digest with fail-fast `ValueError` expectations.

### PR-17 Outcome (Completed Locally)
- Owner: `OWNER_TBD (placeholder)`.
- Hardened `ml/features/macro_transforms.py` real-time path with timestamp-bounded eligibility:
  - macro series features are only emitted when `release_ts <= ts_event` (no future-release leakage).
  - deterministic empty output is returned when no eligible macro release exists for the event timestamp.
  - local macro watermark guard rejects timestamp backsteps deterministically and preserves forward-only progression semantics.
- Updated `ml/features/pipeline_stream.py` to require/pass `timestamp_ns` for macro transforms (same event-time source used across macro/calendar/event transforms).
- Added focused tests in `ml/tests/unit/features/test_macro_transforms_parity.py` and `ml/tests/unit/features/test_pipeline_stream_executor.py` for no-eligible handling, backstep watermark behavior, macro timestamp pass-through, and missing timestamp fail-fast checks.

### PR-18 Outcome (Completed Locally)
- Owner: `OWNER_TBD (placeholder)`.
- Hardened canonical `epoch-1` target semantics with execution-aware label contract fields:
  - explicit execution price columns (`entry_price_column`/`exit_price_column`),
  - execution latency (`latency_bars` with `latency_unit='bars'`),
  - deterministic unresolved execution handling policy (`zero_return`/`fail`).
- Updated target generation to use one shared execution-aware forward-return implementation across bar-index and wall-clock modes, including deterministic unresolved handling with optional fail-fast escalation.
- Enforced fail-fast metadata and orchestrator contract alignment for execution semantics (contract + horizon mode + execution contract).
- Added focused contract/unit coverage for execution metadata presence, config parsing/validation, generator unresolved-context behavior, and orchestrator mismatch rejection.

### PR-19 Outcome (Completed Locally)
- Owner: `OWNER_TBD (placeholder)`.
- Hardened strict rollout posture for Risks #3/#8 in actor env-derived config (`ml/config/base.py`):
  - production-like env (`ML_ENV`/`NAUTILUS_ENV`/`ENVIRONMENT` in `prod|production|live`) now defaults remediation to strict fail-closed (`enable_inference_deadline_guard=True`, `inference_timeout_action=halt`, `ml_failure_action=halt`) unless explicitly overridden by remediation env keys.
  - non-production and dummy-store paths keep permissive behavior unchanged.
- Hardened base actor fail-closed behavior (`ml/actors/base.py`):
  - `_apply_configured_ml_failure_action` now no-ops when already halted to prevent post-halt policy downgrade.
  - halt transition path now surfaces missing risk-transition hook availability deterministically and marks halt reason accordingly.
  - repeated halt action re-entry is idempotent.
- Added/updated focused tests and contracts:
  - runtime actor deadline/failure semantics: `ml/tests/unit/actors/test_inference_deadline_guard.py`
  - runtime contract guardrail assertion: `ml/tests/contracts/test_runtime_correctness_contracts.py`
  - production-like strict remediation env defaults + explicit permissive override preservation: `ml/tests/unit/config/test_env_builders.py`

### PR-20 Outcome (Completed Locally)
- Owner: `OWNER_TBD (placeholder)`.
- Expanded deterministic strict-rollout validation coverage for Risks #3/#8 in runtime actor tests:
  - deadline timeout outcomes (`drop`/`halt`) and no-publish/drop invariants,
  - ML failure action outcomes (`log_only`/`degraded`/`halt`),
  - halted-state persistence and no post-halt policy downgrade behavior.
- Added explicit risk-transition hook observability coverage for:
  - success path (live vs replay `is_live` payload),
  - unavailable hook path,
  - hook write-failure path.
- Expanded multi-signal runtime correctness coverage for strict remediation outcomes and added that module to the runtime-correctness CI lane composition.
- Closed a determinism/observability gap in the risk-transition hook logging path by using best-effort keyword-safe logging in `ml/actors/base.py` so unavailable/failure hook paths cannot bypass fail-closed reason marking due logger keyword incompatibilities.

### Overall Remaining-Risk Execution Plan (Ordered, Testable)
1. PR-21.1 completed (2026-02-08): Full strict validation matrix rerun and evidence capture.
Acceptance criteria:
  - `poetry run mypy ml --strict`, `poetry run ruff check ml`, `make validate-fixtures`, `poetry run coverage report`, `make validate-metrics`, `make validate-events`, `make pytest-ml-runtime-correctness`, `make pytest-ml-registry-hardening`, and `make pytest-ml-strict-policy` all executed and were recorded as successful.
  - Runtime and strict-policy evidence captured: `make pytest-ml-runtime-correctness` (`39 passed`), `make pytest-ml-registry-hardening` (`19 passed`), `make pytest-ml-strict-policy` (runtime `39 passed` + registry `19 passed`).
Dependencies:
  - PR-20 runtime-correctness lane updates (including multi-signal runtime module) remained intact.
2. PR-21.2 completed (2026-02-08): Risk DoD closeout for #3/#8/#10 with evidence links.
Acceptance criteria:
  - Tracker tables and progress log link to the exact files/tests and command outcomes proving timeout/failure/halt/replay observability behavior and strict lane composition.
Dependencies:
  - PR-21.1 command results.
3. PR-21.3 completed (2026-02-08): Final residual-risk sweep and closeout decision.
Acceptance criteria:
  - Work Tracking Board and PR Slice Status now mark #3/#8/#10 closeout validation complete with no open semantic regressions against PR-09..PR-20 behavior.
Dependencies:
  - PR-21.1 and PR-21.2 complete.
4. Post-closeout residual-risk execution plan (ongoing).
Acceptance criteria:
  - Keep strict-policy runtime/registry subsets as release gates (`make pytest-ml-runtime-correctness`, `make pytest-ml-registry-hardening`, `make pytest-ml-strict-policy`).
  - Record any regressions or policy drifts in this tracker with risk IDs and failing command evidence.
Dependencies:
  - No additional code dependency; operational cadence only.

## Rollout Flags and Migration Controls

| Control | Initial Default | Intended End State | Notes |
|---|---|---|---|
| `enable_inference_deadline_guard` | `False` generally; `True` by default for production-like env-derived actor config | `True` for live actors | Explicit env override remains available for phased rollout |
| `inference_timeout_action` | `drop` generally; `halt` by default for production-like env-derived actor config | `halt` for production-critical strategies | Explicit env override remains available for phased rollout |
| strict model compatibility policy | permissive migration flag available | strict-by-default | Temporary bypass metrics required |
| unsigned artifact allowance | `False` (strict) with temporary override | `False` | Migration override only |
| drift action policy | `LOG_ONLY` | `DEGRADED`/`FAIL_CLOSED` where validated | Keep replay/backtest safe mode |
| causality monotonic enforcement | warn-only | strict drop/reset policy | Enable per actor after telemetry baseline |
| target semantics contract | `epoch-1` strict canonical contract | `epoch-1` until next explicit breaking epoch | Pre-alpha epoch cut: regenerate artifacts instead of migration shims |
| deterministic mode | off by default initially | on in reproducibility-critical pipelines | Warn-only option for unsupported ops |

## PR-21 DoD Closeout Evidence (Risks #3/#8/#10)

| Risk | Evidence | Links | Status |
|---|---|---|---|
| #3 | Deadline/drop/halt bounded inference behavior and replay-safe reset behavior remain enforced under runtime lane + strict-policy composition. | `ml/tests/unit/actors/test_inference_deadline_guard.py`, `ml/tests/unit/actors/test_multi_signal_actor.py`, `ml/tests/integration/replay/test_actor_reset_on_rewind.py`, `ml/tests/contracts/test_runtime_correctness_contracts.py`, `Makefile` | Closed (validated 2026-02-08) |
| #8 | Failure-to-risk-state transitions (including unavailable/failure hook observability paths and halted-state no-downgrade invariants) remain deterministic in strict runtime tests. | `ml/tests/unit/actors/test_inference_deadline_guard.py`, `ml/tests/unit/actors/test_multi_signal_actor.py`, `ml/tests/contracts/test_runtime_correctness_contracts.py`, `ml/actors/base.py`, `Makefile` | Closed (validated 2026-02-08) |
| #10 | Runtime correctness CI coverage is actively enforced through dedicated runtime lane and composed strict-policy lane, both rerun successfully in PR-21.1. | `ml/tests/contracts/test_runtime_correctness_contracts.py`, `ml/tests/unit/actors/test_inference_deadline_guard.py`, `ml/tests/unit/actors/test_signal_facade_impl_unit.py`, `ml/tests/unit/actors/test_base_actor_cold_path.py`, `ml/tests/unit/actors/test_multi_signal_actor.py`, `Makefile` | Closed (validated 2026-02-08) |

## Definition of Done by Risk

| Risk | DoD |
|---|---|
| #1 | All online feature paths reject/handle non-monotonic input before state mutation; causality violation metrics emitted. |
| #2 | Dataset semantics include execution-aware fields; training/orchestrator validates semantics alignment; wall-clock horizon path covered by tests. |
| #3 | Inference path has bounded deadline behavior with explicit timeout outcome and no signal publish on timeout. |
| #4 | Serveable model registration/load fails when output semantics are missing/invalid in strict mode. |
| #5 | Artifact integrity enforced for registry + direct-path loads; missing digest blocked unless explicit migration override. |
| #6 | Drift signals computed at runtime with configured thresholds and deterministic action policy validated in tests. |
| #7 | No hardcoded seeds in primary trainers/HPO; deterministic mode exists; provenance recorded in dataset/model artifacts. |
| #8 | ML runtime failures trigger explicit degrade/halt transitions and risk-store events with observability. |
| #9 | Rewind/backstep causes full inference-state invalidation before further predictions. |
| #10 | CI includes runtime ML correctness tests for deadline/drift/replay behaviors, not just offline schema checks. |

## Validation Matrix (Per PR)

| Validation | Required |
|---|---|
| Types | `poetry run mypy ml --strict` |
| Lint | `poetry run ruff check ml` |
| Fixtures | `make validate-fixtures` |
| Focused tests | `poetry run pytest -k <area>` |
| Coverage | `poetry run coverage report` (ML modules >= 90%) |
| Metrics/events validators (when touched) | `make validate-metrics` and `make validate-events` |

## Execution Cadence

- Weekly planning: update owners, target dates, and blocked dependencies in the work tracking board.
- Per-PR update: append entry in Progress Log with risk IDs, merged slices, and test evidence.
- Exit review: mark each risk DoD as complete with links to merged PRs and CI runs.

## Progress Log (Extended)

| Date | Update | Links |
|---|---|---|
| 2026-02-06 | Synthesized 5-stream remediation plan (A/B/C/D/E) into ordered PR program | This file |
| 2026-02-06 | PR-01 implementation completed locally with additive scaffolding only and validation evidence (`mypy`, focused `pytest`, fixture validation, metrics/events validators) | This file |
| 2026-02-06 | PR-02 implementation completed locally with additive shared utility modules + unit tests and validation evidence (`mypy`, `ruff`, fixture validation, focused pytest, coverage snapshots); enforcement wiring deferred to PR-03/PR-05 per slice plan | `ml/common/causality_guard.py`, `ml/common/output_semantics.py`, `ml/common/reproducibility.py`, `ml/tests/unit/common/test_causality_guard.py`, `ml/tests/unit/common/test_output_semantics.py`, `ml/tests/unit/common/test_reproducibility.py` |
| 2026-02-06 | PR-03 implementation completed locally with policy-driven registry compatibility/output semantics/digest enforcement and focused strict-vs-permissive tests (migration override and unsigned artifact override paths included) | `ml/registry/model_registry_facade.py`, `ml/config/policy.py`, `ml/tests/unit/registry/test_model_registry_policy_gating.py`, `ml/tests/unit/config/test_policy_config.py` |
| 2026-02-06 | PR-04 implementation completed locally with shared direct-path load policy enforcement for digest + output semantics, sidecar digest alias expansion, and focused strict-vs-permissive unit tests across actor loader paths | `ml/common/model_load_policy.py`, `ml/common/model_sidecar.py`, `ml/actors/base.py`, `ml/actors/common/model.py`, `ml/actors/common/model_warmup.py`, `ml/tests/unit/common/test_model_load_policy.py`, `ml/tests/unit/common/test_model_sidecar.py`, `ml/tests/unit/actors/test_model_loader.py`, `ml/tests/unit/actors/common/test_model_load_policy_gating.py` |
| 2026-02-06 | PR-05 implementation completed locally with shared remediation decision helper, base/facade deadline guard enforcement, ML failure transition hooks, and focused strict-vs-permissive unit tests | `ml/actors/base.py`, `ml/actors/signal_facade_impl.py`, `ml/actors/common/remediation.py`, `ml/tests/unit/actors/test_inference_deadline_guard.py`, `ml/tests/unit/actors/test_signal_facade_impl_unit.py`, `ml/tests/unit/actors/common/test_remediation.py` |
| 2026-02-06 | PR-06 implementation completed locally with multi-signal batch-path deadline/failure alignment, halted-state enforcement, and strict-vs-permissive actor tests for disabled/drop/halt modes | `ml/actors/multi_signal.py`, `ml/tests/unit/actors/test_multi_signal_actor.py`, `ml/docs/implementation/ml_static_audit_remediation_tracker.md` |
| 2026-02-06 | PR-08 implementation completed locally with shared replay runtime reset hook, facade backstep detection, feature/drift/indicator invalidation wiring, and replay-focused unit/integration coverage | `ml/actors/base.py`, `ml/actors/signal_facade_impl.py`, `ml/actors/common/features.py`, `ml/actors/common/drift_monitoring.py`, `ml/tests/unit/actors/test_base_actor_cold_path.py`, `ml/tests/unit/actors/test_signal_facade_impl_unit.py`, `ml/tests/unit/actors/common/test_features.py`, `ml/tests/integration/replay/test_actor_reset_on_rewind.py` |
| 2026-02-07 | PR-09 completed locally with explicit pre-alpha epoch cut: canonical `epoch-1` target semantics contract is strictly enforced in config parse, metadata contract validation, and orchestrator gating with focused contract/metadata/orchestrator/config tests | `ml/config/targets.py`, `ml/training/datasets/target_generator.py`, `ml/data/metadata.py`, `ml/training/common/training_orchestrator.py`, `ml/tests/contracts/test_dataset_target_semantics_contracts.py`, `ml/tests/unit/data/test_target_semantics_metadata.py`, `ml/tests/unit/training/common/test_training_orchestrator.py`, `ml/tests/unit/config/test_target_semantics_config.py` |
| 2026-02-07 | PR-10 completed locally with explicit horizon resolution contract metadata, shared wall-clock alignment implementation, strict metadata/orchestrator mode enforcement, and focused wall-clock alignment tests | `ml/config/targets.py`, `ml/training/datasets/target_generator.py`, `ml/data/metadata.py`, `ml/training/common/training_orchestrator.py`, `ml/data/__init__.py`, `ml/tests/unit/training/test_target_generator_horizon_alignment.py`, `ml/tests/unit/data/test_target_semantics_metadata.py`, `ml/tests/unit/training/common/test_training_orchestrator.py`, `ml/tests/contracts/test_dataset_target_semantics_contracts.py`, `ml/tests/unit/config/test_target_semantics_config.py`, `ml/tests/utils/targets.py` |
| 2026-02-07 | PR-11 completed locally with canonical seed resolution, removal of hardcoded non-distilled trainer/HPO seed defaults, deterministic worker seed fallback plumbing, and focused seed-propagation/guard coverage | `ml/common/reproducibility.py`, `ml/training/non_distilled/xgboost.py`, `ml/training/non_distilled/lightgbm.py`, `ml/training/common/hyperparameter.py`, `ml/training/optuna_optimizer.py`, `ml/training/event_driven/worker.py`, `ml/tests/unit/common/test_reproducibility.py`, `ml/tests/unit/training/common/test_hyperparameter.py`, `ml/tests/unit/training/event_driven/test_worker.py`, `ml/tests/unit/training/test_non_distilled_seed_config.py` |
| 2026-02-07 | PR-12 completed locally with strict provenance serialization/validation across dataset metadata and model/export artifacts, including persistence-manifest boundary propagation and focused round-trip guard tests | `ml/common/reproducibility.py`, `ml/data/build.py`, `ml/data/metadata.py`, `ml/training/common/persistence.py`, `ml/training/export.py`, `ml/tests/unit/common/test_reproducibility.py`, `ml/tests/unit/data/test_metadata_defaults.py`, `ml/tests/unit/training/common/test_persistence.py`, `ml/tests/unit/training/test_export_provenance.py` |
| 2026-02-07 | PR-13 completed locally with runtime-correctness CI selector/entrypoint, expanded deadline/drift/rewind runtime invariants, and aggregation contracts for targeted suite wiring | `pytest.ini`, `ml/pytest.ini`, `Makefile`, `ml/tests/unit/actors/test_inference_deadline_guard.py`, `ml/tests/unit/actors/test_signal_facade_impl_unit.py`, `ml/tests/integration/replay/test_actor_reset_on_rewind.py`, `ml/tests/contracts/test_runtime_correctness_contracts.py` |
| 2026-02-07 | PR-14 completed locally with strict-by-default registry compatibility/output semantics derivation, strict output/digest/hot-reload enforcement hardening, expanded strict policy gating tests, production strict contracts, and dedicated registry hardening CI subset | `ml/config/registry.py`, `ml/registry/model_registry_facade.py`, `ml/tests/unit/registry/test_model_registry_policy_gating.py`, `ml/tests/contracts/test_registry_behavioral.py`, `Makefile`, `ml/docs/implementation/ml_static_audit_remediation_tracker.md` |
| 2026-02-07 | PR-15 completed locally with ingress causality monotonic/backstep contract hardening (pre-state-mutation ingress guard, warn/drop/reset policy behavior, causality metric emission, shared base+facade monotonic helper reuse, and runtime lane/contract expansion) | `ml/actors/base.py`, `ml/actors/common/features.py`, `ml/actors/signal_facade_impl.py`, `ml/tests/unit/actors/test_base_actor_cold_path.py`, `ml/tests/contracts/test_runtime_correctness_contracts.py`, `Makefile`, `ml/docs/implementation/ml_static_audit_remediation_tracker.md` |
| 2026-02-07 | PR-16 completed locally with strict-policy CI composition (`pytest-ml-strict-policy`), fail-fast runtime→registry contract enforcement, and strict-default production registration/load fail-fast contracts for compatibility/output semantics/digest gates | `Makefile`, `ml/tests/contracts/test_runtime_correctness_contracts.py`, `ml/tests/contracts/test_registry_behavioral.py`, `ml/docs/implementation/ml_static_audit_remediation_tracker.md` |
| 2026-02-07 | PR-17 completed locally with timestamp-bound macro real-time hardening (release eligibility bounded by event time, deterministic no-eligible/backstep behavior, macro timestamp requirement in stream executor) and focused causality tests | `ml/features/macro_transforms.py`, `ml/features/pipeline_stream.py`, `ml/tests/unit/features/test_macro_transforms_parity.py`, `ml/tests/unit/features/test_pipeline_stream_executor.py`, `ml/docs/implementation/ml_static_audit_remediation_tracker.md` |
| 2026-02-07 | PR-18 completed locally with execution-aware label contract hardening (canonical execution fields in `epoch-1`, strict metadata/orchestrator execution alignment, and deterministic unresolved-context handling in shared target generation) plus focused contract/unit coverage | `ml/config/targets.py`, `ml/training/datasets/target_generator.py`, `ml/data/metadata.py`, `ml/training/common/training_orchestrator.py`, `ml/tests/contracts/test_dataset_target_semantics_contracts.py`, `ml/tests/unit/training/test_target_generator_horizon_alignment.py`, `ml/tests/unit/training/common/test_training_orchestrator.py`, `ml/tests/unit/data/test_target_semantics_metadata.py`, `ml/tests/unit/config/test_target_semantics_config.py` |
| 2026-02-07 | PR-19 completed locally with strict rollout hardening for Risks #3/#8: production-like env strict defaults for deadline/failure remediation, halted-state no-downgrade guardrails, deterministic missing risk-transition handling, and updated runtime/config contracts | `ml/config/base.py`, `ml/actors/base.py`, `ml/tests/unit/actors/test_inference_deadline_guard.py`, `ml/tests/contracts/test_runtime_correctness_contracts.py`, `ml/tests/unit/config/test_env_builders.py`, `ml/docs/implementation/ml_static_audit_remediation_tracker.md` |
| 2026-02-07 | PR-20 completed locally with deterministic operational validation for strict rollout outcomes (`drop`/`halt` timeouts, `log_only`/`degraded`/`halt` ML failures, halted-state persistence/non-downgrade), replay/live-like risk-transition hook payload checks, unavailable/failure hook observability coverage, and runtime-correctness lane expansion to include multi-signal strict-remediation tests; added keyword-safe best-effort hook logging fallback to preserve fail-closed determinism | `ml/tests/unit/actors/test_inference_deadline_guard.py`, `ml/tests/unit/actors/test_multi_signal_actor.py`, `ml/tests/contracts/test_runtime_correctness_contracts.py`, `Makefile`, `ml/actors/base.py`, `ml/docs/implementation/ml_static_audit_remediation_tracker.md` |
| 2026-02-08 | PR-21 completed locally as end-to-end closeout for risks #3/#8/#10: reran strict validation matrix (`mypy`, `ruff`, fixture validation, focused runtime pytest, coverage report, metrics/events validators, runtime lane, registry lane, strict-policy lane), all commands succeeded, and tracker evidence/closeout status were finalized with no runtime code changes required | `ml/docs/implementation/ml_static_audit_remediation_tracker.md`, `ml/tests/unit/actors/test_inference_deadline_guard.py`, `ml/tests/unit/actors/test_multi_signal_actor.py`, `ml/tests/contracts/test_runtime_correctness_contracts.py`, `Makefile` |
