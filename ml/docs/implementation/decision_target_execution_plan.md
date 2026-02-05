# Decision-Target Execution Plan (Best-Practice Implementation)

## Purpose
Execute the Decision-Target Review backlog with best-practice defaults, while
maintaining typing + testing standards and preserving hot-path constraints.
This is an execution plan (not a findings doc). It references
`ml/docs/implementation/decision_target_review_plan.md` for evidence.

Primary evidence:
- Review plan (source of detailed evidence): `ml/docs/implementation/decision_target_review_plan.md`
- Quick link: [Decision-Target Review Plan](decision_target_review_plan.md)

## Scope
- Inference semantics + signal generation + strategy decisioning
- Target generation + training + evaluation + cross-validation
- Registry/manifest metadata + signal/event persistence
- Portfolio aggregation + risk/exit alignment
- Observability + hot-path safety + deployment configuration
- Tests + typing + validation gates

## Assumptions (Best-Practice Defaults)
These are the recommended choices to implement unless superseded by an explicit
product decision:
- Canonical prediction surface = calibrated probability in [0, 1].
- Confidence = calibrated probability (or explicit uncertainty if available).
- Decision mapping uses a neutral band around threshold (default 0.5).
- Purged/embargoed CV is enforced for overlapping horizons.
- Training trading metrics include costs/slippage (same cost model as Stage2).
- Decision metadata is versioned and persisted end-to-end.
- Async persistence is mandatory in production; synchronous fallback is disabled.
- Portfolio allocation uses multi-signal batches when correlation logic is enabled.
- Calibration/output schema is stored in the registry manifest and surfaced to inference.

## Pre-flight Decisions (Confirmed)
- Canonical prediction surface = calibrated probability in [0, 1], with neutral band.
- Confidence = calibrated probability (or explicit uncertainty if available).
- Purged/embargoed CV enforced for overlapping horizons.
- Training trading metrics include costs/slippage (match Stage2 model).
- Decision metadata is versioned and persisted end-to-end.
- Async persistence is mandatory in production; sync fallback is disabled.
- Portfolio allocation uses multi-signal batches when correlation logic is enabled.
- Calibration/output schema is stored in the registry manifest and surfaced to inference.

## Migration Policy (Confirmed)
- No backward-compat/versioning required (pre-alpha). In-place schema changes are allowed.
- Preserve existing market and feature data in Postgres; avoid touching market-data
  tables or feature-store tables unless explicitly approved.
- Changes should be scoped to ML strategy/registry schemas (signals, decision events,
  manifests/contracts) and supporting metadata tables only.

## Implementation Policy (Strict + Controlled Auto-Fill)
To prevent drift, the system is now **strict by default** with **narrow auto-fill**:
- Required fields must be present; missing required metadata is a hard error.
- Auto-fill is allowed **only** when the value is unambiguously derivable from
  model manifest or config (e.g., decision metadata assembled from manifest fields).
- No legacy fallbacks or implicit heuristics (pre-alpha, no backward compat).
- Any auto-fill should be centralized, logged once, and the resolved values must
  be embedded in the persisted payload.

## Guardrails (Non-negotiable)
- No hot-path I/O or blocking calls (publish/persist must be non-blocking).
- All tunables live in `ml/config` dataclasses with validation in `__post_init__`.
- All functions/methods must have complete type annotations.
- Use `ml.common.metrics_bootstrap` for metrics; `exc_info=True` in except logs.
- No duplication: reuse shared utilities and common components.
- Update decision metadata schema in one canonical place and reuse it everywhere.
- Tests must be deterministic; property/contract tests preferred for invariants.
- After each workstream milestone, bring failing tests back to 0 before proceeding.
- Follow fixture plug-in guidance in `ml/tests/fixtures/FIXTURE_GUIDE.md`.
 - Do not modify or truncate market/feature data tables in Postgres.

## Definition of Done (Global)
- Prediction semantics and decision mapping are documented and enforced at runtime.
- Decision metadata is persisted in MLSignal/StrategySignal/DecisionEvent and aligns with registry contract.
- Targets and training evaluation are aligned with trading outcomes (incl. cost model).
- Purged/embargoed CV is enforced where required; exceptions documented and tested.
- Async persistence and publish behavior are enforced in deployment configs.
- Multi-signal portfolio allocation is exercised by production path when enabled.
- All tests pass and coverage thresholds are met; mypy/ruff/validators clean.
- Fixture plug-in exports validated (`make validate-fixtures`) and schema/contract
  updates reflected in registry/bootstrap + store migrations.

## Workstreams and Checklists

### Status Update (2026-02-03)
Strict mode (no backward compat) is now enforced for Tasks 1–3. Key updates:
- Task 1: explicit positive-class mapping is required end-to-end (normalization + manifests);
  classifier manifests now fail fast without `decision_config.positive_class_index` and
  student distillation paths auto-provide index=0 for single-probability outputs.
- Task 2: decision metadata is required across signals/events (schema + DB + store writes),
  and legacy write paths are removed.
- Task 3: `target_semantics` is mandatory for dataset builds and orchestration; all CLIs,
  pipelines, and test fixtures now pass explicit semantics; legacy fallbacks removed.

Remaining work now sits in Tasks 4–8 (CV/exit/hot-path/calibration/portfolio),
see the checklists below.

### Status Update (2026-02-04)
Task 6 (hot-path safety) is complete with shared async bus bridge, explicit
sync-fallback gating, and deployment defaults wired for async persistence.

### Status Update (2026-02-05)
Strict target semantics enforcement now extends beyond TFT/Chronos:
- Base ML training orchestration validates dataset metadata target_semantics and target_col
  when training from local dataset artifacts.
- Streaming training runner validates dataset_metadata target_semantics against the resolved
  streaming target_col.
- Student distillation requires target_semantics metadata and validates target_col alignment.
- HPO orchestration resolves target_col from dataset metadata and enforces target_semantics alignment.
- CLI/config help text now calls out target_col alignment with target_semantics.
Rationale: prevent implicit/legacy target selection and ensure every training entrypoint
fails fast unless target semantics are explicit and aligned.

### 1) Prediction Semantics + Decision Mapping
**Goal:** Canonical prediction surface and consistent mapping.

Evidence anchors (current behavior):
- Mixed prediction surfaces: class index + confidence in sklearn path and ONNX single-output confidence defaults. (`ml/actors/base.py:2242-2275`)
- Facade clamps prediction to [-1, 1], uses argmax + default confidence=0.5 for single-output ONNX/generic. (`ml/actors/signal_facade_impl.py:1363-1480`)
- Strategy decision assumes prediction in [0, 1] with 0.5 threshold. (`ml/strategies/base_facade.py:968-987`, `ml/strategies/ml_strategy.py:265-275`)
- Model exit policy uses a 0.5-centered band. (`ml/strategies/common/model_exit_policy.py:139-160`)
- `prediction_threshold` is explicitly a confidence threshold. (`ml/config/base.py:85-108`)

Checklist:
- [x] Define `PredictionSurface` schema (probability, optional logits, calibration info).
- [x] Update inference adapters to normalize outputs into canonical surface.
- [x] Remove auto-pass confidence defaults; make confidence explicit or derived.
- [x] Align signal strategies to operate on canonical confidence/probability.
- [x] Update strategy mapping to use neutral band and canonical surface.
- [x] Document mapping in manifest + code docstrings.
- [x] Require explicit positive-class mapping (index/label) in model metadata
      for any vector output; remove heuristic class selection.
- [x] Fail fast when vector outputs are observed without an explicit mapping.
- [x] Enforce classifier mapping in manifest creation/export/distillation paths.

Tests to update/add:
- [x] Unit tests for mapping classifier outputs → canonical surface.
- [x] Property tests for signal gating invariants (threshold + neutral band).
- [x] Regression tests for ONNX single-output behavior.
- [x] Regression tests for missing positive-class metadata (must fail).

### 2) Decision Metadata Schema + Persistence
**Goal:** Versioned decision metadata across signals and events.

Evidence anchors (current behavior):
- StrategySignal lacks decision metadata fields. (`ml/stores/base.py:205-246`)
- DecisionEvent payload only includes core signal fields. (`ml/strategies/services/decision_publisher.py:25-113`)
- Registry signals manifest uses `signal` int, not `signal_type`. (`ml/registry/bootstrap_datasets.py:224-249`)
- Trust-layer plan expects richer execution params and metadata. (`ml/strategies/ARBITER_TRUST_LAYER_PLAN.md:42-85`)

Checklist:
- [x] Define `DecisionMetadataV1` schema (policy, horizon, label, calibration, model lineage).
- [x] Add schema to MLSignal metadata and StrategySignal JSONB.
- [x] Update DecisionEvent payloads to include decision metadata.
- [x] Align registry `signals` manifest schema with DB fields.
- [x] Update schema touchpoints together (manifest + DB + audit):
      `ml/registry/bootstrap_datasets.py`, `ml/stores/strategy_store.py`,
      `ml/stores/migrations_bootstrap/001_bootstrap.sql`, `ml/stores/schema_audit.py`.
- [x] Add DataWriter validation (contract enforcement or preflight).
- [x] Remove legacy conversion paths (no backward compatibility).
- [x] Enforce `decision_metadata` as required (contract + DB + writer).

Tests to update/add:
- [x] Contract tests for signals dataset schema.
- [x] Unit tests for DecisionEvent payload composition.
- [x] Tests that missing decision metadata fails fast.

### 3) Targets + Label Semantics
**Goal:** Cost-aware, horizon-aligned target definitions with manifest metadata.

Evidence anchors (current behavior):
- Binary label: `y = forward_return > threshold` with NaN fill. (`ml/training/datasets/target_generator.py:122-223`)
- TFT dataset builder repeats the same binary label logic. (`ml/data/tft_dataset_builder_facade.py`)
- Target generation component documents binary-only targets. (`ml/data/common/target_generation.py:48-165`)
- LightGBM/XGBoost configs allow non-binary objectives without a standardized target meaning. (`ml/config/lightgbm.py:257-320`, `ml/config/xgboost.py:70-125`)
- Non-distilled trainer only validates `target_col` presence. (`ml/training/non_distilled/lightgbm.py:39-73`)

Checklist:
- [x] Implement target variants (binary long-only + multi-class + cost-aware).
- [x] Add multi-horizon target generation (explicit column naming).
- [x] Persist target semantics in dataset manifests/sidecars.
- [x] Update dataset builders to require explicit target semantics.
- [x] Update training configs to require target semantics declaration.
- [x] Remove legacy `TargetSemanticsConfig.from_legacy` fallbacks (explicit config required).
- [x] Enforce explicit target semantics in all CLIs/configs (incl. Chronos/AutoGluon).

Tests to update/add:
- [x] Unit tests for multi-horizon target generation invariants.
- [x] Contract tests for target schema in datasets.
- [x] Metamorphic tests: cost-aware labels respond to fees/slippage.
- [x] Tests that missing target semantics fail fast.

Proposed target naming + semantics (Task 3 design notes)
- Deployment cadence assumption (for horizon alignment):
  - Current deployment defaults `BAR_TYPE=SPY.EQUS-1-MINUTE-LAST-EXTERNAL`
    (documented in `ML_DEPLOYMENT_README.md`, `ml/deployment/docker-compose.yml`,
    and `ml/deployment/entrypoint_actor.py`).
  - Task 3 horizons should be expressed in minutes and validated as integer
    multiples of the bar interval. If production overrides `BAR_TYPE`, adjust
    horizons accordingly and persist both `horizon_minutes` and `horizon_bars`
    in metadata.
- Targets are composable: multi-horizon + cost-aware returns + binary/multi-class/regression
  can all coexist in a single dataset. Training selects one target column per model
  (unless a multi-head trainer is introduced later).
- Recommended column naming (explicit horizon suffix, minutes-based):
  - `forward_return_{horizon}` (e.g., `forward_return_15m`)
  - `cost_return_{horizon}` (e.g., `cost_return_15m`) optional
  - `target_bin_{horizon}` (binary long-only, from a return column + threshold)
  - `target_class_{horizon}` (multiclass: short/neutral/long)
  - `target_reg_{horizon}` (regression target; typically a direct alias of a return column)
- Multiclass encoding (numeric, stable across trainers):
  - `-1` = short, `0` = neutral, `1` = long
  - Neutral band derived from thresholds around zero (or around cost-adjusted zero)
- Threshold defaults and config separation:
  - Default return-space threshold remains 10 bps (0.001) for binary labels.
  - Multiclass uses symmetric ±10 bps by default, but supports asymmetric thresholds.
  - Do not tie label thresholds to `prediction_neutral_band`; use a separate
    target-threshold config and persist thresholds in metadata (bps + decimal).
- Cost-aware label guidance:
  - Reuse Stage2 cost components (`cost_bps`, `commission_bps`, `slippage_bps`).
  - Default cost components to 0.0 (no silent behavior change).
  - Apply a round-trip cost per horizon to cost-aware returns (entry+exit),
    not per-turn, and include the full cost model in target semantics metadata.
- Best-practice semantics metadata (stored in dataset manifest metadata or sidecar):
  - `target_semantics.version = "v1"`
  - `horizons`: list of `{label: "15m", minutes: 15}`
  - `returns`: map of return column → `{horizon_minutes, basis, cost_model}`
  - `labels`: map of target column → `{type, return_col, threshold, neutral_band, classes}`
  - Example:
    ```
    target_semantics:
      version: v1
      horizons:
        - { label: "15m", minutes: 15 }
      returns:
        forward_return_15m: { horizon_minutes: 15, basis: "raw" }
        cost_return_15m: { horizon_minutes: 15, basis: "net", cost_model: { fees_bps: 1.0, slippage_bps: 2.0 } }
      labels:
        target_bin_15m: { type: "binary", return_col: "forward_return_15m", threshold: 0.001 }
        target_class_15m: { type: "multiclass", return_col: "cost_return_15m", thresholds: { short: -0.001, long: 0.001 }, classes: { "-1": "short", "0": "neutral", "1": "long" } }
        target_reg_15m: { type: "regression", return_col: "cost_return_15m" }
    ```
- Alignment with Tasks 1–2:
  - Task 1 neutral-band decision mapping should align with `target_class_*` thresholds.
  - Task 2 DecisionMetadataV1 `label` + `horizon` should be populated from target semantics.
- Dataset defaults:
  - Always emit raw return columns for backward compatibility and analytics.
  - Emit cost-aware return columns only when a cost model is configured (or a flag is set).
  - Require explicit `target_column` when multiple targets are present unless
    `target_semantics` marks a `primary_target`.

Primary codebase touchpoints for Task 3
- Target generation:
  - `ml/training/datasets/target_generator.py` (canonical target generator)
  - `ml/data/common/target_generation.py` (shared component; remove duplicate logic)
  - `ml/data/tft_dataset_builder_facade.py` (call shared generator + emit multiple target cols)
- Training config + validation:
  - `ml/config/base.py` (`MLTrainingConfig.target_column` + new target semantics config)
  - `ml/config/lightgbm.py`, `ml/config/xgboost.py` (require target semantics)
  - `ml/training/datasets/time_series_formatter.py` (ensure target selection aligns)
- Manifests/metadata:
  - `ml/registry/dataclasses.py` (`DatasetManifest.metadata` for target_semantics)
  - Dataset builders write sidecar manifest metadata when datasets are exported.

### 4) Leakage Controls + Evaluation
**Goal:** Enforce purged/embargoed CV and realistic evaluation metrics.

Evidence anchors (current behavior):
- Validation splitter performs time splits only (no purge/embargo). (`ml/training/datasets/validation_splitter.py:36-132`)
- Purged CV is optional and depends on config; embargo_pct defaults to 0. (`ml/training/common/cross_validation.py:187-217`, `ml/training/common/cross_validation.py:348-412`)
- Chronos uses timestamp fraction splits without purge/embargo. (`ml/training/autogluon/chronos_evaluation.py:331-391`)
- HPO TFT only passes `--val_days` (time-window validation). (`ml/cli/hpo_tft.py:167-171`, `ml/cli/hpo_tft.py:305-331`)
- Training trading metrics are costless (signals * returns). (`ml/training/common/evaluation.py:166-213`)

Checklist:
- [x] Add `embargo_pct` to shared training config and validators.
- [x] Wire `purge_gap`/`embargo_pct` through all trainers/CLIs.
- [x] Align HPO and Chronos to declared CV strategy or document exceptions.
- [x] Add cost/slippage model to training evaluation (match Stage2 config).
- [x] Document when time-window validation is acceptable.
- [x] Follow-up: require explicit `validation_strategy` across teacher/orchestrator/Chronos entrypoints and pass `embargo_pct`/`purge_gap` without silent fallback (time-window now an explicit exception).
Note: Chronos evaluation now honors `validation_strategy` and uses the last purged split for train/validation when selected.

Tests to update/add:
- [x] Unit tests for purge/embargo usage (config wiring).
- [x] Regression tests ensuring CV strategy is enforced across trainers.
- [x] Tests for training trading metrics with cost model.

Time-window validation is acceptable when:
- Running rapid HPO sweeps or teacher pretraining where time-budget is tight.
- Chronos foundation-model evaluation where purged CV is infeasible on full panels.
- Producing a quick diagnostic/health signal, not a final model selection gate.
- Datasets are non-overlapping by construction and a single contiguous holdout window is
  explicitly required by the experiment design.

Use purged/embargoed CV for final model selection whenever overlapping horizons or
autocorrelated returns can leak information across folds.

### 5) Exit Policy vs Horizon
**Goal:** Ensure exits align with label horizon or document divergence.

Evidence anchors (current behavior):
- Exit policy and model-exit configs have no horizon linkage fields. (`ml/config/base.py:819-887`)
- Model exit policy uses 0.5 threshold band (assumes [0,1] prediction). (`ml/strategies/common/model_exit_policy.py:139-160`)
- Strategy decision path uses 0.5 threshold. (`ml/strategies/ml_strategy.py:265-275`)

Checklist:
- [x] Define rule for mapping horizon → exit defaults (configurable).
- [x] Persist horizon metadata into decision/execution params.
- [x] Update exit policy logic to use horizon-based defaults (if enabled).
- [x] Document policy in strategy config and in decision metadata.
Notes (handoff):
- Recommendation: default max holding derived from horizon with no slack. Use `max_holding_ms = horizon_ms * 1.0`
  when horizon-mapping is enabled and no explicit max holding override is provided.
- Recommendation: derive model-exit min hold when horizon-mapping is enabled using
  `min_hold_ms = horizon_ms * 0.25`, clamped (suggested clamp: min 5s, max 5m).
- Recommendation: persist both `horizon` (dict, unit-aware) and `horizon_ms` (int) in execution params.
- Plumbing gaps to fix:
  - `training_config` exported for model manifests should include a scalar `target_horizon_minutes`
    (derived from target semantics primary horizon).
  - Registry load should include `training_config` in model metadata so `decision_metadata_from_model_metadata`
    can populate `horizon`.
  - Add a shared helper to compute `horizon_ms` from decision metadata (unit-aware) for reuse in strategy
    execution params and exit logic.
- Likely touch points:
  - `ml/training/common/mlflow_tracking.py` (add `target_horizon_minutes` to training_config export)
  - `ml/actors/base.py` (include manifest.training_config in `_model_metadata` when loading from registry)
  - `ml/common/decision_metadata.py` (helper to resolve horizon + compute ms)
  - `ml/config/base.py` (new horizon→exit mapping config on `MLStrategyConfig`)
  - `ml/strategies/ml_strategy.py` and `ml/strategies/common/model_exit_policy.py` (apply horizon-derived defaults)
  - `ml/strategies/common/decision_persistence.py` (inject horizon + horizon_ms into execution_params)
  - Update tests in `ml/tests/unit/strategies/` and docs accordingly.

Update (2026-02-04):
- Added `ExitHorizonConfig` (env overrides) and wired horizon-derived defaults for max holding and model-exit min hold.
- Training config export now emits `target_horizon_minutes`; registry load includes `training_config` in model metadata.
- Added `resolve_decision_horizon_ms` helper and injected `horizon` + `horizon_ms` into execution params.
- Tests added for horizon-derived exit defaults and execution param enrichment.

Tests to update/add:
- [x] Unit tests for horizon-driven exit default computation.
- [x] Strategy tests verifying horizon-based exit behavior.

### 6) Hot-Path Safety (Persistence + Publish)
**Goal:** No blocking operations on hot paths.

Evidence anchors (current behavior):
- Sync fallback writes to FeatureStore/ModelStore when async worker absent. (`ml/actors/base.py:1459-1511`)
- Signal facade persists predictions synchronously. (`ml/actors/signal_facade_impl.py:470-503`, `ml/actors/signal_facade_impl.py:1109-1122`)
- Actor publish is synchronous and dispatches handlers inline. (`nautilus_trader/common/actor.pyx:2429-2447`, `nautilus_trader/common/component.pyx:2733-2763`)
- ML bus is optional and disabled by default unless env flags are set. (`ml/config/bus.py:1-80`)
- Actor/strategy containers do not enable ML bus by default. (`ml/deployment/docker-compose.yml:157-238`)

Checklist:
- [x] Enforce async persistence in production configs (entrypoints + env).
- [x] Disable synchronous fallback in production or gate behind explicit flag.
- [x] Ensure publish is non-blocking or done off hot path.
- [x] Add telemetry for enqueue drops/backpressure.

Notes (handoff, 2026-02-04):
- Hot-path sync writes exist in `ml/actors/base.py` (`_generate_prediction_protected` sync fallback), `ml/actors/signal_facade_impl.py` (`_persist_prediction` writes to ModelStore), and `ml/actors/common/features.py` (`persist_features_async` sync fallback).
- Async persistence worker already tracks queue depth/drops in `ml/observability/ml_async_persistence.py`. Actor-side bus publishing is non-blocking via `ml/actors/ml_domain_events.py`, but strategy decision publishing still uses synchronous `MessageBusConfig` publishers in `ml/strategies/common/decision_persistence.py`.
- Production compose does not set `ML_ENABLE_ASYNC_PERSISTENCE` / bus flags for `ml_signal_actor` + `ml_strategy` (only the streaming services do); `.env.example` defaults exist but are not wired in compose.
- Proposed config addition: add an explicit gate such as `allow_sync_persistence_fallback` (or `require_async_persistence`) on `MLActorConfig` with env override to drop-and-metric when no worker is available.
- Proposed refactor: centralize feature/prediction persistence helper to avoid duplicate feature-dict construction, and enforce drop-only behavior when sync fallback is disabled.
- Publishing plan: keep bus disabled for strategy until a non-blocking path exists, or move/refactor `DomainEventBridge` into `ml/common` so strategies can use a shared async publisher without cross-domain imports.
- Deployment/entrypoints: wire `ML_ENABLE_ASYNC_PERSISTENCE=1` and the new sync-fallback flag in `ml/deployment/docker-compose.yml` and `ml/deployment/entrypoint_actor.py`; add bus env (`ML_BUS_ENABLE=1`, `ML_BUS_FROM_ACTOR=1`, `ML_BUS_FROM_STORE=0`) only when async publishing is ready.
- Tests to add: unit tests for sync-fallback gating in `BaseMLInferenceActor` and `MLSignalActorFacade`; config env parsing tests; async bus bridge queue drop metrics if introduced.

Update (2026-02-04):
- Added shared `DomainEventBridge` in `ml/common/bus_bridge.py` and refactored actor/strategy publishers to use it with `ML_BUS_FROM_ACTOR`/`ML_BUS_FROM_STRATEGY` gating.
- Added `ML_ALLOW_SYNC_PERSISTENCE_FALLBACK` to `MLActorConfig`, enforced drop-and-metric behavior when async persistence is missing, and centralized feature dict construction.
- Wired async persistence + bus env defaults in deployment compose and actor entrypoint (`ML_ENABLE_ASYNC_PERSISTENCE=1`, `ML_ALLOW_SYNC_PERSISTENCE_FALLBACK=0`, `ML_BUS_ENABLE=1`).
- Added backpressure + queue depth telemetry in the shared bridge; drop metrics for sync fallback.

Tests to update/add:
- [x] Unit tests for persistence fallback gating and feature persistence helper (`ml/tests/unit/actors/common/test_features.py`).
- [x] Unit tests for strategy bus bridge + bus config env parsing (`ml/tests/unit/strategies/test_strategy_bus_bridge.py`, `ml/tests/unit/config/test_actor_bus_config.py`).
- [x] Integration tests validating publish path is disabled or async in prod mode.

### 7) Calibration + Output Schema Persistence
**Goal:** Calibration metadata and output schema are in registry and inference.

Evidence anchors (current behavior):
- Student meta includes output_schema and calibrator params. (`ml/training/student/lightgbm.py:321-343`)
- Student manifest builder does not ingest student meta fields. (`ml/registry/utils.py:90-147`)
- ONNX metadata extraction records only shapes/names, not calibration schema. (`ml/actors/common/model.py:479-521`)

Checklist:
- [x] Extend ModelManifest to include output_schema + calibration params.
- [x] Ingest `student.meta.json` during registration and model load.
- [x] Use manifest calibration in inference (no silent defaults).
- [x] Document calibration lifecycle (train → registry → inference).

Update (2026-02-04):
- Added output schema + calibration fields to ModelManifest and persisted them in registry storage.
- Registration now ingests student sidecar metadata; ModelComponent merges sidecar metadata on load.
- Inference decision metadata now includes calibration when present in model metadata.
- Documented calibration lifecycle in `ml/training/README.md`.

Tests to update/add:
- [x] Unit tests for manifest ingestion of student meta.
- [x] Inference tests verifying calibration parameters are applied.

### 8) Portfolio Aggregation Activation
**Goal:** Multi-signal allocation is exercised when portfolio/correlation logic is enabled.

Evidence anchors (current behavior):
- Strategy path calls `allocate_signals([signal], ...)` (single-signal list). (`ml/strategies/common/position_management.py:829-833`)
- PortfolioManager supports multi-signal allocations and correlation adjustments. (`ml/strategies/portfolio.py:138-182`)
- Signal routing aggregates to a single MLSignal before strategy handling. (`ml/strategies/common/signal_routing.py:271-367`)
- Multi-signal actor batches inference but emits per-instrument signals. (`ml/actors/multi_signal.py:220-323`)

Checklist:
- [x] Define batching window semantics (time/size-based).
- [x] Update signal routing to batch signals across instruments.
- [x] Ensure allocation uses list of signals, not singletons.
- [x] Track aggregated_from metadata consistently.

Update (2026-02-04):
- Added `PortfolioBatchingConfig` (window/min/max) and wired it into strategy
  position management to batch signals across instruments before allocation.
- Introduced shared `PortfolioSignalBatcher` keyed by portfolio identity to
  assemble cross-instrument batches without altering per-instrument strategies.
- Allocation now uses batched signal lists when batching is enabled; existing
  signal metadata (including `aggregated_from`) is preserved.

Tests to update/add:
- [x] Unit tests for multi-signal allocation path.
- [x] Property tests for allocation invariants under batching.

Update (2026-02-04):
- Added property-based coverage for batched allocation invariants and a
  lightweight batching test to confirm shared portfolio grouping across
  strategies without a heavy integration harness.

## Rollout Plan (High-Level)
1) Decision confirmations (if any deviations from best-practice defaults).
2) Schema + manifest changes (metadata + contracts + migrations).
3) Inference + strategy alignment (prediction mapping + gating).
4) Target + training updates (labels + CV + evaluation).
5) Hot-path enforcement (async persistence + publish).
6) Portfolio aggregation activation (batching).
7) End-to-end tests + validation runs.

## Testing / Typing / Validation Gates
Required before merge:
- `poetry run mypy ml --strict`
- `poetry ruff check ml`
- `make validate-fixtures`
- `make validate-metrics`
- `make validate-events`
- `coverage report` (ML ≥90%, general ≥80%)
- Focused pytest: `poetry run pytest -k <area>`

Milestone discipline:
- After each workstream, run focused tests + mypy/ruff and resolve failures
  before starting the next workstream. This keeps failures scoped and prevents
  cross-stream regressions.

Fixture discipline:
- Use fixture plug-ins only; do not import fixtures directly.
- Add fixtures under `ml/tests/fixtures/` and update `__all__` + exports guard.
- Follow `ml/tests/fixtures/FIXTURE_GUIDE.md` for dataset/store/telemetry patterns.

## Operational Validation (Post-Merge)
- Successful training runs using new CV + target semantics.
- Replay/backtest verifying exit policies and portfolio aggregation behavior.
- Confirm inference consumes registry calibration + output schema.

Update (2026-02-04):
- Replay harness run (AAPL.EQUS, bar_spec=1-MINUTE-LAST, warm_up=5, lookback=5,
  model=chronos_option2_distilled_lgbm_v1) completed; signals persisted to
  `public.ml_strategy_signals` (DB 5432). Decision metadata includes calibration
  (platt) for the run window 2025-01-14 14:30–21:00 UTC.
- `ModelComponent` load (poetry env) confirms sidecar-provided `output_schema`
  + `calibration` are present in model metadata.
Update (2026-02-05):
- Replay summary registry emission now re-checks for `replay_summary` after
  auto-registration and skips event/watermark updates when the dataset is still
  missing, emitting a fallback metric to avoid FK warnings.
- Replay harness now resolves a single DB connection (prefers explicit actor
  override; otherwise selects the first reachable candidate) and uses it for
  actor initialization + replay summary persistence to avoid split persistence.
- Training operational validation: minimal TFT run completed on
  `baseline_spy_small_v3` using purged CV (`cv_splits=2`, `test_fraction=0.2`,
  `purge_gap=1`, `embargo_hours=24`) with `max_epochs=1`, `batch_size=32`,
  `hidden_size=8`, `loss=bce`, `seed=7`. Outputs written under
  `ml_out/op_validation_tft_small_20260205` (teacher preds/meta, model metrics,
  validation returns). Dataset metadata still lacks `target_semantics`, so a
  rebuild with explicit target semantics is required for subsequent training
  runs now that strict validation is enforced.
- Strict target semantics enforcement added: training coordinator and TFT
  teacher CLI now require `dataset_metadata.json` to include a full
  `target_semantics` payload (version/horizons/labels/returns), with unit tests
  covering the new validation guard.
- Additional strictness added: training now validates that `target_col` is
  declared in `target_semantics` labels (or legacy aliases), with unit tests
  covering aligned and misaligned configurations.
- Chronos training experiment now resolves `target_col` from target semantics
  (or `--target_col` override) and validates dataset metadata includes
  `target_semantics` plus a matching target declaration; new unit tests cover
  target resolution edge cases.
- Target-semantic rebuild + training validation completed: built
  `tft_baseline_spy_small_v3_target_semantics` from
  `SPY.EQUS-1-MINUTE-LAST-EXTERNAL` catalog bars (2024-01-02 to 2024-01-04) with
  explicit semantics (`target_bin_15m`) and registered a new feature set
  (`feature_set_1770256818342716`). Training run completed with purged CV
  (`cv_splits=2`, `test_fraction=0.2`, `purge_gap=1`, `embargo_hours=24`,
  `max_epochs=1`, `batch_size=32`, `hidden_size=8`, `loss=bce`, `seed=7`);
  outputs written under `ml_out/op_validation_tft_target_semantics_20260205`
  (teacher preds/meta, model metrics).

## Impact Analysis Checklist
Use this checklist after each workstream to catch secondary blast-radius items
that unit tests might miss.
- [x] Registry/bootstrap schemas updated (manifests + contracts + schema_hash if needed). (N/A for Task 8: no schema changes.)
- [x] DB migrations and schema audits updated consistently. (N/A for Task 8: no persisted schema changes.)
- [x] Deployment entrypoints and compose/env defaults updated to reflect new configs. (N/A for Task 8: no env/config surface added.)
- [x] Streaming/evaluation paths reviewed for semantic alignment. (N/A for Task 8: runtime strategy-only change.)
- [x] Metrics/event topics still aligned with new decision metadata. (N/A for Task 8: no new event surfaces.)
- [x] Backward compatibility addressed (conversion or versioning for persisted data). (N/A for Task 8: batching is in-memory only.)
- [x] Documentation updated (plan + relevant ops/docs). (Task 8 notes updated here.)

## Secondary Touchpoints (Likely Downstream Impact)
These modules commonly depend on the workstreams and often require updates even
when not directly edited.
- Deployment/config: `ml/deployment/*.py`, `ml/deployment/docker-compose*.yml`,
  `ml/config/*.py`, `ML_DEPLOYMENT_README.md`.
- Registry/metadata: `ml/registry/*.py`, `ml/stores/schema_audit.py`,
  `ml/stores/migrations_bootstrap/001_bootstrap.sql`.
- Evaluation/streaming: `ml/orchestration/*`, `ml/training/event_driven/*`,
  `ml/training/common/evaluation.py`, `ml/training/autogluon/*`.
- Persistence/events: `ml/stores/common/data_writer.py`,
  `ml/stores/services/strategy_services.py`, `ml/strategies/services/decision_publisher.py`.
- Tests/fixtures: `ml/tests/fixtures/*`, `ml/tests/unit/**`, `ml/tests/contract/**`,
  `ml/tests/property/**`, `ml/tests/e2e/**`.

## Risks
- Backward compatibility: legacy signals schema and downstream consumers.
- Performance regressions if hot-path enforcement is misconfigured.
- Training data leakage if purge/embargo wiring is incomplete.

## References
- `ml/docs/implementation/decision_target_review_plan.md`
- `ml/docs/implementation/trade_execution_risk_portfolio_plan.md`
