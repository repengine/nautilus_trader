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

### 1) Prediction Semantics + Decision Mapping
**Goal:** Canonical prediction surface and consistent mapping.

Evidence anchors (current behavior):
- Mixed prediction surfaces: class index + confidence in sklearn path and ONNX single-output confidence defaults. (`ml/actors/base.py:2242-2275`)
- Facade clamps prediction to [-1, 1], uses argmax + default confidence=0.5 for single-output ONNX/generic. (`ml/actors/signal_facade_impl.py:1363-1480`)
- Strategy decision assumes prediction in [0, 1] with 0.5 threshold. (`ml/strategies/base_facade.py:968-987`, `ml/strategies/ml_strategy.py:265-275`)
- Model exit policy uses a 0.5-centered band. (`ml/strategies/common/model_exit_policy.py:139-160`)
- `prediction_threshold` is explicitly a confidence threshold. (`ml/config/base.py:85-108`)

Checklist:
- [ ] Define `PredictionSurface` schema (probability, optional logits, calibration info).
- [ ] Update inference adapters to normalize outputs into canonical surface.
- [ ] Remove auto-pass confidence defaults; make confidence explicit or derived.
- [ ] Align signal strategies to operate on canonical confidence/probability.
- [ ] Update strategy mapping to use neutral band and canonical surface.
- [ ] Document mapping in manifest + code docstrings.

Tests to update/add:
- [ ] Unit tests for mapping classifier outputs → canonical surface.
- [ ] Property tests for signal gating invariants (threshold + neutral band).
- [ ] Regression tests for ONNX single-output behavior.

### 2) Decision Metadata Schema + Persistence
**Goal:** Versioned decision metadata across signals and events.

Evidence anchors (current behavior):
- StrategySignal lacks decision metadata fields. (`ml/stores/base.py:205-246`)
- DecisionEvent payload only includes core signal fields. (`ml/strategies/services/decision_publisher.py:25-113`)
- Registry signals manifest uses `signal` int, not `signal_type`. (`ml/registry/bootstrap_datasets.py:224-249`)
- Trust-layer plan expects richer execution params and metadata. (`ml/strategies/ARBITER_TRUST_LAYER_PLAN.md:42-85`)

Checklist:
- [ ] Define `DecisionMetadataV1` schema (policy, horizon, label, calibration, model lineage).
- [ ] Add schema to MLSignal metadata and StrategySignal JSONB.
- [ ] Update DecisionEvent payloads to include decision metadata.
- [ ] Align registry `signals` manifest schema with DB fields.
- [ ] Update schema touchpoints together (manifest + DB + audit):
      `ml/registry/bootstrap_datasets.py`, `ml/stores/strategy_store.py`,
      `ml/stores/migrations_bootstrap/001_bootstrap.sql`, `ml/stores/schema_audit.py`.
- [ ] Add DataWriter validation (contract enforcement or preflight).
- [ ] Add backward-compatible conversion for legacy payloads.

Tests to update/add:
- [ ] Contract tests for signals dataset schema.
- [ ] Unit tests for DecisionEvent payload composition.
- [ ] Property tests for schema migration (legacy → new).

### 3) Targets + Label Semantics
**Goal:** Cost-aware, horizon-aligned target definitions with manifest metadata.

Evidence anchors (current behavior):
- Binary label: `y = forward_return > threshold` with NaN fill. (`ml/training/datasets/target_generator.py:122-223`)
- TFT dataset builder repeats the same binary label logic. (`ml/data/tft_dataset_builder.py:2328-2356`)
- Target generation component documents binary-only targets. (`ml/data/common/target_generation.py:48-165`)
- LightGBM/XGBoost configs allow non-binary objectives without a standardized target meaning. (`ml/config/lightgbm.py:257-320`, `ml/config/xgboost.py:70-125`)
- Non-distilled trainer only validates `target_col` presence. (`ml/training/non_distilled/lightgbm.py:39-73`)

Checklist:
- [ ] Implement target variants (binary long-only + multi-class + cost-aware).
- [ ] Add multi-horizon target generation (explicit column naming).
- [ ] Persist target semantics in dataset manifests/sidecars.
- [ ] Update dataset builders to support target variants.
- [ ] Update training configs to require target semantics declaration.

Tests to update/add:
- [ ] Unit tests for multi-horizon target generation invariants.
- [ ] Contract tests for target schema in datasets.
- [ ] Metamorphic tests: cost-aware labels respond to fees/slippage.

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
  - `ml/data/tft_dataset_builder.py` (call shared generator + emit multiple target cols)
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
- [ ] Add `embargo_pct` to shared training config and validators.
- [ ] Wire `purge_gap`/`embargo_pct` through all trainers/CLIs.
- [ ] Align HPO and Chronos to declared CV strategy or document exceptions.
- [ ] Add cost/slippage model to training evaluation (match Stage2 config).
- [ ] Document when time-window validation is acceptable.

Tests to update/add:
- [ ] Unit tests for purge/embargo usage (config wiring).
- [ ] Regression tests ensuring CV strategy is enforced across trainers.
- [ ] Tests for training trading metrics with cost model.

### 5) Exit Policy vs Horizon
**Goal:** Ensure exits align with label horizon or document divergence.

Evidence anchors (current behavior):
- Exit policy and model-exit configs have no horizon linkage fields. (`ml/config/base.py:819-887`)
- Model exit policy uses 0.5 threshold band (assumes [0,1] prediction). (`ml/strategies/common/model_exit_policy.py:139-160`)
- Strategy decision path uses 0.5 threshold. (`ml/strategies/ml_strategy.py:265-275`)

Checklist:
- [ ] Define rule for mapping horizon → exit defaults (configurable).
- [ ] Persist horizon metadata into decision/execution params.
- [ ] Update exit policy logic to use horizon-based defaults (if enabled).
- [ ] Document policy in strategy config and in decision metadata.

Tests to update/add:
- [ ] Unit tests for horizon-driven exit default computation.
- [ ] Strategy tests verifying horizon-based exit behavior.

### 6) Hot-Path Safety (Persistence + Publish)
**Goal:** No blocking operations on hot paths.

Evidence anchors (current behavior):
- Sync fallback writes to FeatureStore/ModelStore when async worker absent. (`ml/actors/base.py:1459-1511`)
- Signal facade persists predictions synchronously. (`ml/actors/signal_facade_impl.py:470-503`, `ml/actors/signal_facade_impl.py:1109-1122`)
- Actor publish is synchronous and dispatches handlers inline. (`nautilus_trader/common/actor.pyx:2429-2447`, `nautilus_trader/common/component.pyx:2733-2763`)
- ML bus is optional and disabled by default unless env flags are set. (`ml/config/bus.py:1-80`)
- Actor/strategy containers do not enable ML bus by default. (`ml/deployment/docker-compose.yml:157-238`)

Checklist:
- [ ] Enforce async persistence in production configs (entrypoints + env).
- [ ] Disable synchronous fallback in production or gate behind explicit flag.
- [ ] Ensure publish is non-blocking or done off hot path.
- [ ] Add telemetry for enqueue drops/backpressure.

Tests to update/add:
- [ ] Unit tests for persistence worker enqueue + fallback behavior.
- [ ] Integration tests validating publish path is disabled or async in prod mode.

### 7) Calibration + Output Schema Persistence
**Goal:** Calibration metadata and output schema are in registry and inference.

Evidence anchors (current behavior):
- Student meta includes output_schema and calibrator params. (`ml/training/student/lightgbm.py:321-343`)
- Student manifest builder does not ingest student meta fields. (`ml/registry/utils.py:90-147`)
- ONNX metadata extraction records only shapes/names, not calibration schema. (`ml/actors/common/model.py:479-521`)

Checklist:
- [ ] Extend ModelManifest to include output_schema + calibration params.
- [ ] Ingest `student.meta.json` during registration and model load.
- [ ] Use manifest calibration in inference (no silent defaults).
- [ ] Document calibration lifecycle (train → registry → inference).

Tests to update/add:
- [ ] Unit tests for manifest ingestion of student meta.
- [ ] Inference tests verifying calibration parameters are applied.

### 8) Portfolio Aggregation Activation
**Goal:** Multi-signal allocation is exercised when portfolio/correlation logic is enabled.

Evidence anchors (current behavior):
- Strategy path calls `allocate_signals([signal], ...)` (single-signal list). (`ml/strategies/common/position_management.py:829-833`)
- PortfolioManager supports multi-signal allocations and correlation adjustments. (`ml/strategies/portfolio.py:138-182`)
- Signal routing aggregates to a single MLSignal before strategy handling. (`ml/strategies/common/signal_routing.py:271-367`)
- Multi-signal actor batches inference but emits per-instrument signals. (`ml/actors/multi_signal.py:220-323`)

Checklist:
- [ ] Define batching window semantics (time/size-based).
- [ ] Update signal routing to batch signals across instruments.
- [ ] Ensure allocation uses list of signals, not singletons.
- [ ] Track aggregated_from metadata consistently.

Tests to update/add:
- [ ] Unit tests for multi-signal allocation path.
- [ ] Property tests for allocation invariants under batching.

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

## Impact Analysis Checklist
Use this checklist after each workstream to catch secondary blast-radius items
that unit tests might miss.
- [ ] Registry/bootstrap schemas updated (manifests + contracts + schema_hash if needed).
- [ ] DB migrations and schema audits updated consistently.
- [ ] Deployment entrypoints and compose/env defaults updated to reflect new configs.
- [ ] Streaming/evaluation paths reviewed for semantic alignment.
- [ ] Metrics/event topics still aligned with new decision metadata.
- [ ] Backward compatibility addressed (conversion or versioning for persisted data).
- [ ] Documentation updated (plan + relevant ops/docs).

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
