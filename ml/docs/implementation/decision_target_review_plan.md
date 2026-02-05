# Decision-Target Review Plan (ML System)

## Purpose
Ensure each ML component is trained backwards from the trading decision it actually improves. This plan audits the end-to-end chain:
features -> prediction -> signal -> strategy -> risk/sizing -> portfolio -> PnL.

## Scope
- ML inference actors and prediction surfaces
- Target/label generation and dataset builders
- Signal strategies and gating policies
- Strategy execution, exits, sizing, and risk controls
- Portfolio allocation and correlation logic
- Pipelines, registry handoffs, and evaluation
- Observability, fallbacks, and hot-path constraints

## Audit Target Inventory (Broad Map)
- Config and decision plumbing:
  - `ml/config/actors.py`, `ml/config/base.py`, `ml/config/runtime.py`
  - `ml/config/events.py`, `ml/config/names.py`, `ml/config/constants.py`
  - `ml/config/registry.py`, `ml/config/streaming_pipeline.py`, `ml/config/orchestrator/`
- Actors and inference:
  - `ml/actors/base.py`, `ml/actors/signal.py`, `ml/actors/signal_facade_impl.py`
  - `ml/actors/enhanced.py`, `ml/actors/multi_signal.py`, `ml/actors/adapters.py`
  - `ml/actors/actor_services.py`, `ml/actors/recorder.py`
  - `ml/actors/common/` (adaptive_threshold, prediction_buffer, model, features, signal_strategy)
- Features and preprocessing:
  - `ml/features/` (pipeline, microstructure, macro, earnings, indicators)
  - `ml/preprocessing/joins.py`, `ml/preprocessing/stationarity.py`, `ml/preprocessing/vintage_age.py`
  - `ml/preprocessing/event_ingestion.py`
- Data ingestion and dataset assembly:
  - `ml/data/tft_dataset_builder_facade.py`
  - `ml/data/feature_computation_manager.py`, `ml/data/feature_manifest_export.py`
  - `ml/data/ingest/`, `ml/data/providers/`, `ml/data/rehydration/`, `ml/data/validation.py`
  - `ml/data/catalog/`, `ml/data/sources/`, `ml/data/coverage/`
- Targets and dataset APIs (labels/horizons):
  - `ml/training/datasets/target_generator.py`
  - `ml/training/datasets/validation_splitter.py`
  - `ml/training/datasets/time_series_formatter.py`
  - `ml/training/datasets/dataset_serializer.py`
- Training and export:
  - `ml/training/base.py`, `ml/training/export.py`, `ml/training/optuna_optimizer.py`
  - `ml/training/non_distilled/`, `ml/training/student/`, `ml/training/teacher/`
  - `ml/training/event_driven/` (orchestrator, services, dataset_service, guardrails)
- Pipelines and orchestration:
  - `ml/pipelines/build_runner.py`, `ml/pipelines/tft_train_distill.py`
  - `ml/orchestration/pipeline_orchestrator*.py`, `ml/orchestration/dataset_builder.py`
  - `ml/orchestration/training_coordinator.py`, `ml/orchestration/scheduler.py`
- Stores and persistence (4-store pattern):
  - `ml/stores/data_store.py`, `ml/stores/feature_store.py`
  - `ml/stores/model_store.py`, `ml/stores/strategy_store.py`
  - `ml/stores/feature_store_facade.py`, `ml/stores/data_store_facade.py`
  - `ml/stores/file_backed.py`, `ml/stores/feature_versioning.py`
- Registry and deployment gates:
  - `ml/registry/model_registry_facade.py`, `ml/registry/model_registry.py`
  - `ml/registry/feature_registry.py`, `ml/registry/strategy_registry.py`
  - `ml/registry/model_quality_validator.py`, `ml/registry/lineage_manager.py`
  - `ml/registry/model_deployment_mgr.py`, `ml/registry/ab_testing_manager.py`
- Strategy layer (signals -> orders):
  - `ml/strategies/ml_strategy.py`, `ml/strategies/base.py`
  - `ml/strategies/sizing.py`, `ml/strategies/risk.py`, `ml/strategies/portfolio.py`
  - `ml/strategies/execution.py`, `ml/strategies/analytics.py`, `ml/strategies/protocols.py`
  - `ml/strategies/common/` (exit policies, correlation, positions)
- Consumers and streaming training:
  - `ml/consumers/streaming_training*.py`, `ml/consumers/redis_streams_consumer.py`
  - `ml/consumers/lineage_writer.py`, `ml/consumers/aggregator.py`
- Observability and metrics:
  - `ml/common/metrics_bootstrap.py`, `ml/common/metrics_manager.py`
  - `ml/common/message_topics.py`, `ml/common/event_emitter.py`
  - `ml/observability/` (ml_async_persistence, service, pipeline, scheduler, tracing)
- Evaluation and validation:
  - `ml/evaluation/metrics.py`
  - `ml/tests/` (property, metamorphic, contracts, validation, e2e)
- Schema and contracts:
  - `ml/schema.py`, `ml/schema/`
  - `ml/stores/schema_audit.py`
- CLI and scripts (thin adapters):
  - `ml/cli/`, `ml/scripts/`

## Decision-to-Target Traceability (Template)
For each model/head/actor:
- Decision improved:
- Target/label definition (formula, horizon, costs):
- Feature set and horizon alignment:
- Prediction type (regression/classification/quantile/vol):
- Signal mapping (thresholds/calibration/logic):
- Strategy usage (entry/exit/sizing/allocation):
- Evaluation metrics (ML + trading):
- Fallbacks + telemetry:

## Review Areas and Checklists

### A) Actor Inference and Prediction
- [ ] Inventory inference actors and model-loading entrypoints
- [ ] Map each actor to a decision layer (alpha timing, allocation, execution)
- [ ] Confirm prediction outputs are typed and documented with semantics
- [ ] Validate confidence/probability calibration path (if any)
- [ ] Confirm hot-path constraints (no I/O, no heavy allocs)
- [ ] Verify publish/metrics are off hot path with try/except + counters

### B) Targets / Labels / Training Datasets
- [ ] Inventory target generators and dataset builders
- [ ] Ensure targets are cost-aware where required (fees/slippage)
- [ ] Confirm horizon alignment and label windows
- [ ] Check for leakage (as-of joins, purging/embargo)
- [ ] Verify multi-horizon or multi-head structure where needed
- [ ] Ensure label metadata is persisted in manifests/sidecars

### C) Signal Generation (Policies)
- [ ] Enumerate signal strategies and selection logic
- [ ] Verify signal policies align with prediction semantics
- [ ] Check for separate policies per decision layer
- [ ] Confirm regime-aware or adaptive thresholds are justified
- [ ] Ensure signal metadata includes model_id, horizon, label info

### D) Strategy Execution (Entry/Exit)
- [ ] Map signal -> order path for each strategy
- [ ] Validate exit logic aligns with training horizon and label family
- [ ] Check time-in-trade and stop/target rules for leakage or mismatch
- [ ] Confirm strategy respects constraints (position limits, cooldowns)

### E) Sizing and Risk Controls
- [ ] Review sizing models (Kelly/volatility/composite) inputs
- [ ] Confirm sizing uses appropriate risk forecasts (vol/cov) when available
- [ ] Validate risk limits vs. target distribution (cost-aware)
- [ ] Ensure staged risk actions and fallback states are observable

### F) Portfolio Allocation
- [ ] Verify allocation logic (equal/risk parity/kelly) matches signals
- [ ] Confirm correlation estimates are aligned with horizon
- [ ] Check rebalance and concentration logic vs. intended cadence

### G) Pipelines and Orchestration
- [ ] Review dataset build pipeline inputs for horizon/threshold
- [ ] Confirm training/export flow preserves label semantics in metadata
- [ ] Validate registry handoffs (feature schema hash, model id)
- [ ] Ensure inference uses the same feature schema as training

### H) Evaluation and Validation
- [ ] Confirm walk-forward splits, purging/embargo for overlap
- [ ] Include costs/slippage and constraints in backtests
- [ ] Track stability across regimes and OOS robustness
- [ ] Report both ML metrics and trading metrics

### I) Observability and Fallbacks
- [ ] Ensure fallback activation metrics exist for critical paths
- [ ] Confirm health/status telemetry from actors and strategies
- [ ] Verify error logs include exc_info=True

## Verified Behavior Summary (Initial Pass)
Only verified code behavior is listed below. Decision points and desired changes are captured in
Vector 1 tasks/open questions and section notes.

### A) Actor Inference and Prediction
- Verified behavior: Prediction semantics vary by path (argmax class index vs prob vs signed score), confidence defaults differ (base vs facade vs batch), and decision_policy metadata is not applied in hot paths; signal metadata is minimal.

### B) Targets / Labels / Training Datasets
- Verified behavior: Labels are primarily binary forward_return > threshold (long-only); no explicit short/neutral, and cost/slippage are not embedded in labels. Label metadata is not consistently persisted.

### C) Signal Generation (Policies)
- Verified behavior: Default strategies gate on confidence; extremes/momentum assume prediction magnitude is meaningful. Decision-layer-specific policies are not explicit and metadata is minimal.

### D) Strategy Execution (Entry/Exit)
- Verified behavior: Strategy uses 0.5 threshold in [0,1] space; decision persistence/events store only model_predictions/risk_metrics/execution_params without decision metadata; decision events use Stage.SIGNAL_EMITTED.

### E) Sizing and Risk Controls
- Verified behavior: Sizing uses realized volatility and historical win/loss; risk limits are static and not keyed to label horizon.

### F) Portfolio Allocation
- Verified behavior: Allocation is typically per-signal with confidence gating; correlation lookback defaults are long and not explicitly tied to signal horizon.

### G) Pipelines and Orchestration
- Verified behavior: Manifest/export defaults omit decision_policy/config; student.meta output_schema is not surfaced to inference metadata; promotion does not ingest student.meta.

### H) Evaluation and Validation
- Verified behavior: Stage2 returns include costs; streaming economic metrics apply slippage; training evaluation is costless; purge/embargo CV is inconsistent and some trainers bypass it.

### I) Observability and Fallbacks
- Verified behavior: Fallback activation metrics exist for stores; some logs omit exc_info; decision events lack decision metadata.

## Decision Backlog (Explicit Decisions Required)
Decisions to make based on verified behavior (no solutions implied):
- [ ] Decide the canonical prediction surface and label-to-action mapping (probability vs signed score vs class index, plus neutral/hold band).
- [ ] Decide the meaning of confidence and gating (confidence threshold vs prediction magnitude, plus portfolio/strategy gating rules).
- [ ] Decide signal routing aggregation and analytics direction semantics (probability vs signed prediction scale).
- [ ] Decide decision metadata schema + persistence across MLSignal, StrategySignal, and DecisionEvent (including ARBITER/Trust Layer fields).
- [ ] Decide the canonical signals schema (`signal_type` vs `signal` int) and whether DataWriter enforces registry contracts.
- [ ] Decide whether decision_policy/decision_config must be applied in hot paths (adapter context + hot reload).
- [ ] Decide FORCE_SIGNAL_MODE policy (test-only vs allowed) and required telemetry/guardrails.
- [ ] Decide where calibration parameters live (manifest vs sidecar) and whether inference should consume them (teacher/streaming/student).
- [ ] Decide whether cost-aware trading metrics are required beyond Stage2 (training eval/streaming), and whether validation returns are mandatory.
- [ ] Decide whether purge/embargo must be enforced across all trainers and where embargo config lives.
- [ ] Decide whether portfolio allocation should be multi-signal by default and how correlation/vol horizons align to signals.
- [ ] Decide whether decision events need a distinct Stage enum from raw signal emission.
- [ ] Decide whether exit_policy_config/model_exit_config should be tied to label horizon (max_holding_ms/min_hold_ms defaults and enforcement).
- [ ] Decide whether non-distilled trainers should standardize target semantics (binary vs regression/multiclass) and if multi-horizon/cost-aware labels are required.
- [ ] Decide whether async persistence is mandatory (to avoid synchronous store writes in actor hot paths) and what guarantees exist for non-blocking publish/metrics.
- [ ] Decide whether `publish_data` should be allowed in ML hot paths given synchronous handlers and optional external (database) publish.

## Vector 1: Actor Inference & Signal Semantics (Detailed Targets)

### Audit targets
- `ml/actors/base.py` (BaseMLInferenceActor, MLSignal, ONNX/Sklearn prediction paths)
- `ml/actors/signal_facade_impl.py` (prediction pipeline, sanitize, signal publish)
- `ml/actors/common/signal_strategy.py` (threshold/extremes/momentum/ensemble/adaptive semantics)
- `ml/actors/common/signal_metadata.py` (metadata payloads)
- `ml/actors/multi_signal.py` (batched inference path)
- `ml/actors/enhanced.py` (test-focused actor expectations)
- `ml/actors/adapters.py` (policy adapter mapping)
- `ml/config/actors.py` (strategy/threshold mapping and aliases)
- `ml/actors/common/prediction_buffer.py` (history/confidence/volatility buffers)

### Checklist
- [ ] Define prediction semantics per actor/model type (regression vs classification vs logits)
- [ ] Define mapping from classification outputs to signed prediction (+1/-1) or continuous score
- [ ] Confirm confidence semantics (probability, calibrated score, or heuristic)
- [ ] Ensure gating threshold applies to the intended quantity (confidence vs prediction magnitude)
- [ ] Ensure signal strategies assume the same prediction scale (thresholds/extremes/momentum)
- [ ] Align signal_type mapping with decision semantics (classification label vs sign)
- [ ] Validate ONNX output conventions (single output vs prediction+confidence)
- [ ] Ensure model manifest metadata (decision policy, horizon, label spec) is surfaced in signal metadata
- [ ] Confirm batched inference path preserves the same semantics and thresholds

### Initial observations to validate (not decisions)
- `signal_facade_impl._predict` clamps predictions to [-1, 1] but `predict_proba` returns class index. Multi-class collapses to ±1; binary maps 0 to “sell” via sign. Need explicit class-to-signal mapping.
- `BaseMLInferenceActor` publishes on confidence only; regression outputs with confidence defaults may emit signals without magnitude gating.
- ONNX paths default confidence=0.95 when only one output exists; may always pass threshold.
- `signal_metadata` currently lacks label/horizon/decision-policy context; only bar close/spec.

### Vector 1 Review Notes (session 2026-01-28)
- Evidence: Base actor publishes signals on confidence only; no prediction magnitude gating. (`ml/actors/base.py:1549`)
- Evidence: Base ONNX path defaults confidence to 0.95 for single-output models. (`ml/actors/base.py:2252`)
- Evidence: `MLActorConfig` defines `prediction_threshold` as a minimum confidence (default 0.5). (`ml/config/base.py:262`, `ml/config/base.py:303`)
- Evidence: Signal facade clamps predictions to [-1, 1] and uses `predict_proba` argmax. (`ml/actors/signal_facade_impl.py:1363`, `ml/actors/signal_facade_impl.py:1468`)
- Evidence: Signal facade defaults confidence to 0.5 when only a prediction output is present (ONNX/generic); combined with threshold `>=`, can auto-pass at 0.5. (`ml/actors/signal_facade_impl.py:1463`, `ml/actors/signal_facade_impl.py:1480`)
- Evidence: `FORCE_SIGNAL_MODE` env var forces prediction/confidence to (1.0, 1.0), bypassing model outputs. (`ml/actors/signal_facade_impl.py:480`)
- Evidence: Strategy context built in `_try_generate_signal` includes history/adaptive threshold/model_id but no decision metadata (label/horizon). (`ml/actors/signal_facade_impl.py:611`)
- Evidence: Signal store and metrics map `signal_type` based on prediction sign; strength uses `abs(prediction)`. (`ml/actors/signal_facade_impl.py:658`)
- Evidence: Strategy thresholds are confidence-based; `ThresholdSignalStrategy` gates only on confidence. (`ml/actors/common/signal_strategy.py:217`)
- Evidence: Built-in strategy factory uses `prediction_threshold` as the confidence threshold for all strategies. (`ml/actors/common/signal_strategy.py:1058`)
- Evidence: `build_signal_metadata` only includes bar close/spec (no label/horizon/decision policy). (`ml/actors/common/signal_metadata.py:13`)
- Evidence: Multi-instrument ONNX batch path uses raw outputs and sets confidence=0.5 when only one output exists; no sanitization. (`ml/actors/multi_signal.py:436`, `ml/actors/multi_signal.py:441`)
- Evidence: Model manifest includes `decision_policy`/`decision_config`, but `MLSignalActorFacade` initializes strategy without passing metadata; policy only applied when `_create_strategy()` is explicitly called. (`ml/actors/signal_facade_impl.py:398`, `ml/actors/signal_facade_impl.py:1240`, `ml/actors/common/signal_strategy.py:1010`)
- Evidence: `SignalStrategyComponent` calls policy adapters with `actor=None`, so adapters requiring actor context will fail and fall back to built-ins. (`ml/actors/common/signal_strategy.py:1019`, `ml/actors/adapters.py:37`)
- Evidence: `_extract_output_scalar` collapses non-scalar outputs to the first element; classification vector outputs are reduced to a single value. (`ml/actors/signal_facade_impl.py:1527`)
- Evidence: `_sanitize_prediction_output` clamps predictions to [-1, 1], collapsing multi-class argmax values >1 to 1. (`ml/actors/signal_facade_impl.py:1363`)
- Evidence: Multi-instrument batch ONNX path bypasses `_sanitize_prediction_output`, so semantics differ by path. (`ml/actors/multi_signal.py:413`)
- Evidence: Strategy layer expects prediction in [0, 1] and applies a 0.5 threshold for BUY/SELL; mismatches with signed or class-index outputs. (`ml/strategies/base_facade.py:968`, `ml/strategies/ml_strategy.py:264`)
- Evidence: Model exit policy uses a 0.5 threshold band for flips/neutral exits, assuming prediction in [0, 1]. (`ml/strategies/common/model_exit_policy.py:139`)
- Evidence: ModelManifest defines `decision_policy` and free-form `decision_config` without schema. (`ml/registry/base.py:121`)
- Evidence: Training export stub populates `decision_policy=None` and `decision_config={}` by default. (`ml/training/export.py:604`, `ml/training/export.py:605`)
- Evidence: Training persistence populates decision_policy/config from trainer config attributes (if present). (`ml/training/common/persistence.py:353`)
- Evidence: Teacher/Student CLIs accept `--decision_policy` and `--decision_config` JSON; student manifest builder does not set these unless CLI updates after creation. (`ml/training/teacher/tft_cli.py:989`, `ml/training/student/lightgbm_cli.py:90`, `ml/registry/utils.py:132`)
- Evidence: Target generation defines binary label `y = (forward_return > threshold)` and fills trailing NaNs as 0; no explicit short/neutral class. (`ml/training/datasets/target_generator.py:215`)
- Evidence: TFTDatasetBuilder repeats binary label `y` generation from forward returns with positive threshold. (`ml/data/tft_dataset_builder_facade.py`)
- Evidence: TFT teacher CLI defaults `target_col` to `y` and computes “signals” from probabilities using a 0.5 threshold (long-only gate). (`ml/training/teacher/tft_cli.py:138`, `ml/training/teacher/tft_cli.py:287`)
- Evidence: LightGBM student export declares ONNX output schema `binary_proba` (shape [N,1]), implying probability output rather than class index. (`ml/training/student/lightgbm.py:321`)
- Evidence: Base ONNX inference path defaults confidence to 0.95 when only a single output is present. (`ml/actors/base.py:2086`)
- Evidence: Base sklearn path defaults confidence to 1.0 when `predict_proba` is absent. (`ml/actors/base.py:2272`)
- Evidence: Signal facade defaults confidence to 0.5 for ONNX/single-output and generic predict paths. (`ml/actors/signal_facade_impl.py:1463`, `ml/actors/signal_facade_impl.py:1477`)
- Evidence: Multi-instrument batch ONNX path defaults confidence to 0.5 on single-output models. (`ml/actors/multi_signal.py:441`)
- Evidence: Prediction dataset schema constrains prediction to [-1,1] and confidence to [0,1], but does not encode meaning/calibration. (`ml/registry/bootstrap_datasets.py:190`, `ml/registry/bootstrap_datasets.py:209`)

### Vector 1 Follow-up Work (post-decision)
- [ ] Document prediction semantics contract in manifest/decision_config (label mapping, prediction scale, confidence meaning).
- [ ] Implement label-to-signal mapping for classifiers (class index -> signed/prob scale).
- [ ] Align confidence defaults across base/facade/batch (avoid auto-pass on single-output ONNX).
- [ ] Add magnitude-aware gating or dedicated strategies for regression outputs if required.
- [ ] Enrich `signal_metadata` and strategy context with horizon/label/decision policy identifiers.
- [ ] Persist decision metadata into StrategySignal/DecisionEvent payloads (schema vs execution_params).
- [ ] Align signals dataset manifest/contract with StrategySignal schema and conversion rules.
- [ ] Add DataWriter signal schema validation if enforcement is required.
- [ ] Normalize batched inference outputs to match single-instrument sanitize/mapping rules.
- [ ] Align signal routing aggregation + analytics direction logic to the chosen prediction scale.
- [ ] Ensure decision_policy is applied on initial load and hot reload (if required).
- [ ] Add FORCE_SIGNAL_MODE telemetry/guardrails (if allowed).
- [ ] Promote student.meta output_schema/calibration fields into manifest/actor metadata if required.

### Vector 1 Semantics Matrix (current behavior)
- BaseMLInferenceActor:
  - `_predict`: implemented by concrete actor (ONNX uses raw outputs; single-output => confidence=0.95).
  - Gating: `confidence >= prediction_threshold` only.
  - Sanitization: none (prediction passed through).
  - Signal metadata: bar close/spec only.
- MLSignalActorFacade:
  - `_predict`: clamps predictions to [-1, 1]; `predict_proba` uses argmax + max prob; single-output confidence defaults to 0.5.
  - ONNX output: `_extract_output_scalar` reduces non-scalar outputs to first element.
  - Gating: strategy-specific confidence threshold; `FORCE_SIGNAL_MODE` forces (1.0, 1.0).
  - Signal_type: derived from sign of prediction when persisting to StrategyStore.
- MultiInstrumentSignalActor:
  - Batch ONNX: outputs flattened with `.reshape(-1)`; no sanitization; confs from output[1] or 0.5 fallback.
  - Per-row fallback: uses facade `_predict` (sanitization).
- Strategy consumption (MLTradingStrategy):
  - `target_side_from_prediction` assumes prediction in [0, 1] and applies threshold 0.5.

### Vector 1 Evidence Questions (facts to verify)
- See “Remaining Evidence Gaps” for the consolidated list of evidence to collect.

## Review Log (Initial Pass)

### A) Actor Inference and Prediction
- Inventory:
  - ml/actors/base.py (MLSignal, health monitor, model/runtime config)
  - ml/actors/signal_facade_impl.py (actor facade)
  - ml/actors/common/signal_strategy.py (signal policies)
  - ml/actors/multi_signal.py (batch inference path)
  - ml/actors/adapters.py (policy adapters)
  - ml/actors/common/signal_metadata.py (signal metadata helper)
  - ml/config/actors.py (strategy/threshold mapping)
- Notes:
  - Pending: map each actor to explicit decision layer and target semantics.
  - Pending: verify prediction metadata includes horizon/label info.
  - Check: `_predict` returns class index when using `predict_proba`; docstrings expect prediction in [-1, 1].
  - Check: signal_type derived from prediction sign; may mis-handle binary class labels.
  - Check: ONNX single-output defaults confidence=0.95; could bypass thresholds.
  - Evidence: Concrete inference actors include `ONNXMLInferenceActor` (ONNX), `EnhancedMLInferenceActor` (ProductionModelLoader), and `MLSignalActorFacade` (signal actor); `PickleMLInferenceActor` is disabled. (ml/actors/base.py, ml/actors/signal_facade_impl.py)
  - Evidence: `ONNXMLInferenceActor` assumes ONNX outputs `[prediction, confidence]` if 2 outputs, else assigns confidence=0.95. (ml/actors/base.py)
  - Evidence: `MLSignalActorFacade` ONNX path uses `_extract_output_scalar` for output[0] as prediction; when single-output, confidence defaults to 0.5. (ml/actors/signal_facade_impl.py)
  - Evidence: `MultiInstrumentSignalActor` batch ONNX path expects outputs[0]=preds and outputs[1]=confs; if only one output, confidence defaults to 0.5. (ml/actors/multi_signal.py)
  - Evidence: `ChronosInferenceAdapter` emits mean forecast values only (no confidence), representing a regression-style output surface outside signal actors. (ml/actors/common/chronos_inference.py)
  - Evidence: Registry model loader includes `decision_policy`/`decision_config` in `_model_metadata`, but facade strategy creation is called without metadata, so policy is not applied. (ml/actors/common/registry.py, ml/actors/signal_facade_impl.py)
  - Evidence: `SignalStrategyComponent.create_strategy` passes `actor=None` to policy adapters, so adapters that require actor context (e.g., dynamic threshold) cannot use actor state. (ml/actors/common/signal_strategy.py, ml/actors/adapters.py)
  - Evidence: `StrategyConfig` legacy threshold_long/short collapses to absolute `prediction_threshold`, losing directional semantics. (ml/config/actors.py)
  - Evidence: Base actor stores manifest metadata (incl. decision_policy/config) in `_model_metadata`, but signal facade initializes strategy with config-only and `_create_strategy()` (metadata-aware) is defined but never invoked. (ml/actors/base.py, ml/actors/signal_facade_impl.py)
  - Evidence: Signal facade strategy context includes history/adaptive_threshold/market_regime/log_predictions/model_id but no decision/horizon/label metadata. (ml/actors/signal_facade_impl.py)
  - Evidence: StrategyStore persistence derives `signal_type` from `prediction > 0` and uses `abs(prediction)` for strength; no neutral/hold path in persistence. (ml/actors/signal_facade_impl.py)
  - Evidence: AdaptiveStrategy adds extra metadata (adaptive_threshold, signal_strength, market_regime) but no label/horizon/policy context. (ml/actors/common/signal_strategy.py)
  - Evidence: Enhanced ML actor sklearn path uses `predict_proba` argmax for prediction and max prob for confidence (class index surface, no mapping to signed/prob semantics). (ml/actors/base.py)
  - Evidence: Signal facade hot reload updates `_model_metadata` but does not recreate the signal strategy, so manifest decision_policy changes are not applied on reload. (ml/actors/signal_facade_impl.py)
  - Evidence: Base ONNX inference path reads `outputs[0][0]` as prediction and `outputs[1][0]` as confidence; vector outputs are truncated to first element. (ml/actors/base.py)
  - Evidence: LightGBM student ONNX export emits a single `probability` output (`binary_proba`); base ONNX path treats it as prediction and assigns default confidence (0.95 in base actor, 0.5 in facade). (ml/training/student/lightgbm.py, ml/actors/base.py, ml/actors/signal_facade_impl.py)
  - Evidence: Model registry load path only returns an ONNX session; it does not ingest `student.meta.json` or `.onnx.meta.json` into manifest/model metadata. (ml/registry/model_registry_facade.py)
  - Evidence: Actor bus signal events include only `model_id`, `prediction`, `confidence`, and timestamps in metadata; no decision_policy/label/horizon context. (ml/actors/signal_facade_impl.py:723-769)
  - Evidence: Signal metadata builder only includes bar close/spec; adaptive strategy adds adaptive_threshold/signal_strength/market_regime, but no decision/label/horizon fields. (ml/actors/common/signal_metadata.py:34-37, ml/actors/common/signal_strategy.py:701-713)
  - Evidence: `MLSignal` metadata is an unstructured dict with no schema or required decision fields. (ml/actors/base.py:653-720)
  - Evidence: `MLSignal` payload is limited to instrument_id/model_id/prediction/confidence/features/metadata/ts_* with no explicit decision-layer fields, reinforcing a generic signal surface. (ml/actors/base.py:653-720)
  - Evidence: Signal routing aggregation metadata only tracks `aggregated_from` (and `action` in voting mode). (ml/strategies/common/signal_routing.py:359-410)
  - Evidence: Facade persists signals to StrategyStore with `signal_type` based on prediction sign and `execution_params={"threshold": adaptive_threshold}`; no decision/label metadata. (ml/actors/signal_facade_impl.py:653-664)
  - Evidence: `MLSignalActor` is a compatibility alias for `MLSignalActorFacade`; public API points to facade semantics. (ml/actors/signal.py:18-33)
  - Evidence: Deployment and dry-run entrypoints instantiate `MLSignalActor` directly for live/backtest runs, making facade semantics the effective production path. (ml/deployment/entrypoint_actor.py:312, ml/deployment/run_backtest_dry_run.py:194)
  - Evidence: `MultiInstrumentSignalActor` extends `MLSignalActor` and uses batched inference; used primarily in tests/fixtures. (ml/actors/multi_signal.py:93, ml/tests/property/test_multi_signal_coordination.py:487)
  - Evidence: `EnhancedMLInferenceActor` is test/perf-focused and returns `(prediction, confidence)=(0.0, 0.0)` with no model load. (ml/actors/enhanced.py:77-86, ml/actors/enhanced.py:104-113)
  - Evidence: Deployment entrypoint aliases `MLSignalActor` to `MultiInstrumentSignalActor` and constructs `MultiInstrumentSignalActorConfig`; container actor path uses batched inference by default. (ml/deployment/entrypoint_actor.py:25-58, ml/deployment/entrypoint_actor.py:259-312)
  - Evidence: Local dry-run/backtest scripts instantiate `MLSignalActor` and `MLTradingStrategy` in-process. (ml/deployment/run_local_dry_run.py:200-223, ml/deployment/run_backtest_dry_run.py:186-204)
  - Evidence: Dry-run example wires `MLSignalActor` + `MLTradingStrategy` into the backtest engine (no other actor layers). (ml/examples/dry_run_example.py:143-145)
  - Evidence: Strategy container entrypoint uses `MLTradingStrategy` with `ml_signal_source` env default `MLSignalActor-001`. (ml/deployment/entrypoint_strategy.py:46-120)
  - Evidence: FORCE_SIGNAL_MODE is supported in deployment entrypoints (custom always-signal strategy) and mock injection writes dummy signals when enabled. (ml/deployment/entrypoint_actor.py:193-230, ml/deployment/entrypoint_mock.py:165-199)
  - Evidence: Base actor hot path persists features/predictions synchronously when persistence worker is absent (feature_store/model_store writes). (ml/actors/base.py:1459-1511)
  - Evidence: Feature validation checks only finite values, dtype, and length against manifest; no schema-hash or name/order validation is enforced. (ml/actors/common/features.py:450-520)
  - Evidence: `assert_features_parity` helper exists but has no call sites outside tests. (ml/actors/model_loader_utils.py:46-82, ml/tests/unit/common/test_model_loader_utils.py:61-77)
  - Evidence: Entry-point config builds `MLSignalActorConfig` and instantiates `MLSignalActor` (multi-instrument) by default; optional `RecorderActor` only records bars. (ml/deployment/entrypoint_actor.py:174-368)
  - Evidence: Dashboard deployment service only registers an `MLSignalActor` factory (no other actor types), enforcing `model_id` at deployment. (ml/dashboard/services/actors_service.py:171-528)
  - Evidence: Parquet live replay harness attaches `MLSignalActor` + `MLTradingStrategy` per instrument with `ml_signal_source` set to the actor ID. (ml/orchestration/parquet_live_replay_harness.py:772-858)

### B) Targets / Labels / Training Datasets
- Inventory:
  - ml/training/datasets/target_generator.py (forward_return + binary y)
  - ml/pipelines/build_runner.py (horizon_minutes, threshold)
- Notes:
  - Threshold is configurable but currently generic; review cost-aware labeling and slippage integration.
  - Confirm label metadata propagates into export/registry sidecars.
  - Evidence: TargetGenerator uses `forward_return = (future_close - current_close) / current_close` and binary label `y = forward_return > threshold`, filling trailing NaNs with 0 (no explicit short/neutral). (ml/training/datasets/target_generator.py)
  - Evidence: TargetGenerator docstring + implementation define binary `y` and `forward_return` only; labels are `(forward_return > threshold)` with NaN fill. (ml/training/datasets/target_generator.py:122-223)
  - Evidence: TFT dataset builder repeats binary `y` generation with `min_return_threshold` default 0.001; no cost/slippage adjustments baked in. (`ml/data/tft_dataset_builder_facade.py`)
  - Evidence: TFTDatasetBuilder `_generate_targets_polars` computes `forward_return` and binary `y` only, filling NaNs; no short/neutral or cost-aware labels. (`ml/data/tft_dataset_builder_facade.py`)
  - Evidence: TargetGenerationComponent in `ml/data/common/target_generation.py` duplicates the same binary label logic with forward_return and NaN fill. (ml/data/common/target_generation.py)
  - Evidence: TargetGenerationComponent documents binary-only targets (`y` in {0,1}) and computes `forward_return` sidecar with NaN fill. (ml/data/common/target_generation.py:46-167)
  - Evidence: Target generation tests iterate different horizons by re-running the same generator; outputs remain single `y`/`forward_return` columns (no multi-horizon target columns). (ml/tests/unit/data/common/test_target_generation.py:231-266)
  - Evidence: AutoGluon/Chronos data config defaults `target_column="forward_return"` for regression targets; adapter can rename to `target`. (ml/config/autogluon.py, ml/data/autogluon_adapter.py)
  - Evidence: Non-distilled LightGBM/XGBoost trainers default `target_col="target"` and do not enforce label semantics. (ml/training/non_distilled/lightgbm.py, ml/training/non_distilled/xgboost.py)
  - Evidence: LightGBM/XGBoost trainers only validate `target_col` presence (default "target"), leaving target semantics external to the trainer. (ml/training/non_distilled/lightgbm.py:39-73, ml/training/non_distilled/xgboost.py:77-90)
  - Evidence: Chronos distillation uses `soft_target`/`distilled_target` columns and optional hard labels for validation alignment. (ml/training/autogluon/chronos_distillation.py)
  - Evidence: Teacher base supports Platt calibration for logits -> probabilities; TFT CLI calibrates on validation logits and emits calibrated probabilities (`q_val`). (ml/training/teacher/base.py, ml/training/teacher/tft_cli.py)
  - Evidence: Chronos soft-label generator produces rolling soft labels from teacher forecasts with temperature scaling and aligns to timestamps as `(item_id, timestamp, soft_target)`. (ml/training/autogluon/soft_label_generator.py)
  - Evidence: Dataset validation can enforce `forward_return` alignment to future prices when `forward_return_horizon` is set (fills NaNs/Infs with 0.0, checks tolerance). (ml/data/validation.py)
  - Evidence: Streaming dataset guardrails check positive-rate thresholds, required schema columns, and known-future effective-time pairs (pyarrow optional). (ml/training/event_driven/guardrails/dataset.py)
  - Evidence: LightGBM non-distilled `predict` returns probabilities by default and only converts to labels/argmax when `return_labels=True` (binary/multiclass); target semantics remain external to the trainer. (ml/training/non_distilled/lightgbm.py:218-233)
  - Evidence: XGBoost non-distilled `predict` returns probabilities by default for binary objectives; labels only when `return_labels=True`, implying target semantics are caller-defined. (ml/training/non_distilled/xgboost.py:300-349)
  - Evidence: LightGBM training config supports objectives {regression, binary, multiclass, lambdarank} with default regression, implying non-binary/regression targets are supported outside TFT. (ml/config/lightgbm.py:257-320, ml/config/lightgbm.py:584-587)
  - Evidence: XGBoost training config supports objectives {binary:logistic, reg:squarederror, multi:softprob, reg:logistic} with default binary:logistic, enabling non-binary/regression targets via config. (ml/config/xgboost.py:70-125, ml/config/xgboost.py:393-396)

### C) Signal Generation (Policies)
- Inventory:
  - ml/actors/common/signal_strategy.py (threshold/extremes/momentum/ensemble/adaptive)
- Notes:
  - Default strategy uses confidence thresholds; confirm calibration source and decision mapping.
  - Signal publish uses `signal_type` based on sign of prediction; confirm mapping for classification labels.
  - Evidence: ThresholdSignalStrategy gates on confidence >= threshold; prediction is passed through without gating. (ml/actors/common/signal_strategy.py:188-264)
  - Evidence: ExtremesStrategy/MomentumStrategy require prediction magnitude/momentum plus confidence threshold, so prediction scale must be meaningful. (ml/actors/common/signal_strategy.py:267-499)
  - Evidence: EnsembleStrategy uses weighted confidences to compute ensemble score; prediction is preserved unchanged. (ml/actors/common/signal_strategy.py:511-603)
  - Evidence: AdaptiveStrategy uses confidence vs adaptive_threshold only (signal_strength = confidence / adaptive_threshold). (ml/actors/common/signal_strategy.py)
  - Evidence: SignalRoutingComponent aggregates multi-model signals using weighted average or voting; voting uses prediction > 0.5 and sets aggregated predictions to 0.8/0.2, assuming probability-like semantics. (ml/strategies/common/signal_routing.py:342-413)
  - Evidence: Strategy analytics labels direction by prediction sign (long if prediction > 0), which conflicts with probability outputs in [0,1]. (ml/strategies/analytics.py:170-175)
  - Evidence: MLSignalActorFacade updates PredictionBuffer with prediction/confidence/volatility and runs AdaptiveThreshold regime detection on the volatility window. (ml/actors/signal_facade_impl.py:489-511, ml/actors/signal_facade_impl.py:1009-1039)
  - Evidence: Volatility for adaptive threshold is computed as absolute price change between consecutive closes (no normalization). (ml/actors/signal_facade_impl.py:1040-1056)
  - Evidence: AdaptiveThreshold regime boundaries are fixed (avg_vol < 0.001, <0.005, else high_volatility) and threshold uses base + avg_volatility * factor with clamping. (ml/actors/common/adaptive_threshold.py:20-70, ml/actors/common/adaptive_threshold.py:200-260)
  - Evidence: PredictionBuffer uses fixed-size ring buffers for prediction/confidence/volatility, with optional history lists for cold-path analysis. (ml/actors/common/prediction_buffer.py:40-160, ml/actors/common/prediction_buffer.py:170-220)

### D) Strategy Execution (Entry/Exit)
- Inventory:
  - ml/strategies/ml_strategy.py (signal handling, exit policy)
- Notes:
  - Exit policies exist (stop-loss/take-profit/timeouts); align with label horizon.
  - Evidence: Strategy helper `target_side_from_prediction` assumes prediction in [0, 1] and uses 0.5 threshold; MLStrategy uses that threshold explicitly. (ml/strategies/base_facade.py, ml/strategies/ml_strategy.py)
  - Evidence: Model exit policy uses a 0.5-centered prediction band to trigger neutral/flip decisions. (ml/strategies/common/model_exit_policy.py)
  - Evidence: ExitPolicyConfig defines stop_loss_pct/take_profit_pct/max_holding_ms with no horizon linkage fields; ModelExitConfig defines exit_prediction_band/min_hold_ms similarly without horizon semantics. (ml/config/base.py:819-887)
  - Evidence: MLStrategyConfig.from_env populates max_holding_ms and model-exit settings solely from ML_MAX_HOLDING_MS/ML_MODEL_EXIT_* env values; no horizon-derived defaults are applied. (ml/config/base.py:1181-1245)
  - Evidence: MLStrategy exit policy uses stop_loss_pct/take_profit_pct/max_holding_ms from config and compares time-in-trade (ts_opened) to max_holding_ms; no horizon linkage is referenced. (ml/strategies/ml_strategy.py:70-237, ml/strategies/common/model_exit_policy.py:60-73)
  - Evidence: Model exit policy enforces `min_hold_ms` before exits and uses `exit_prediction_band` around a fixed 0.5 threshold; no label horizon is referenced. (ml/strategies/common/model_exit_policy.py:124-194)
  - Evidence: MLStrategy execution params include `target_side`, `model_id`, and `action` only; risk_metrics include confidence/prediction/positions, but no decision_policy/label/horizon context. (ml/strategies/ml_strategy.py:302-314)
  - Evidence: Decision persistence risk_metrics contain only confidence/prediction/active_positions/pending_orders plus optional position_size; no decision metadata. (ml/strategies/common/decision_persistence.py:1021-1067)
  - Evidence: Decision persistence execution_params only include stop_loss_pct/take_profit_pct/position_size/max_positions/current_positions and optional positions metadata; no decision metadata. (ml/strategies/common/decision_persistence.py:1069-1124)
  - Evidence: Decision persistence builds `model_predictions` from signal prediction and optional `aggregated_from` metadata; no other signal metadata is propagated. (ml/strategies/common/decision_persistence.py:1150-1161)
  - Evidence: Strategy consumption of `signal.metadata` only extracts `model_id` (and `aggregated_from` in persistence); no decision/horizon fields are referenced. (ml/strategies/base_facade.py:1447-1456, ml/strategies/common/decision_persistence.py:1151-1158)
  - Evidence: Decision event payloads (DecisionPersistenceComponent/StrategyDecisionPublisher) include only signal_type, strength, model_predictions, risk_metrics, execution_params, and ts_event; no decision_policy/label/horizon fields. (ml/strategies/common/decision_persistence.py:976-989, ml/strategies/services/decision_publisher.py:100-113)
  - Evidence: StrategySignal schema stores model_predictions/risk_metrics/execution_params JSON only; no explicit decision metadata fields. (ml/stores/base.py:204-246, ml/stores/services/strategy_services.py:774-783)
  - Evidence: StrategySignalEventService emits dataset events with metadata `{component: strategy_id}` only (no signal/decision metadata). (ml/stores/services/strategy_services.py:881-910)
  - Evidence: StrategyDecisionPublisher defines DecisionEvent payloads limited to dataset_id/stage/status/source/strategy_id/instrument_id/signal_type/strength/model_predictions/risk_metrics/execution_params/ts_event. (ml/strategies/services/decision_publisher.py:25-113)
  - Evidence: Decision publisher emits events under Stage.SIGNAL_EMITTED; Stage enum has no DECISION_* stage. (ml/strategies/services/decision_publisher.py:94-113, ml/config/events.py:14-35)
  - Evidence: StrategySignal dataclass fields exclude decision_policy/label/horizon/calibration metadata. (ml/stores/base.py:204-246)
  - Evidence: StrategySignalEventService auto-registers the 'signals' dataset schema with only signal_type/strength/model_predictions/risk_metrics/execution_params (no decision metadata). (ml/stores/services/strategy_services.py:774-809)
  - Evidence: StrategySignalWriteService writes `signal_type`/`strength`/`model_predictions`/`risk_metrics`/`execution_params` and publishes Stage.SIGNAL_EMITTED for dataset_id="signals". (ml/stores/services/strategy_services.py:44-109)
  - Evidence: DataWriter emits DataEvent for signals with metadata limited to strategy_id; no decision context is included in event metadata. (ml/stores/common/data_writer.py:772-880)
  - Evidence: DataFrame-to-signals conversion expects `signal_type`/`strength` columns; decision metadata is not represented in the converter schema. (ml/stores/common/data_frame_converters.py:213-303)
  - Evidence: Bootstrap registry signals manifest/contract only defines `signal` (int) and `strength` without decision metadata fields, and does not include signal_type/model_predictions/risk_metrics/execution_params. (ml/registry/bootstrap_datasets.py:223-255, ml/registry/bootstrap_datasets.py:1050-1067)
  - Evidence: StrategyStore table definition for `ml_strategy_signals` includes signal_type/strength/model_predictions/risk_metrics/execution_params columns only; no decision metadata columns. (ml/stores/strategy_store.py:226-240)
  - Evidence: StrategyStore SQL schema persists `signal_type`, `model_predictions`, `risk_metrics`, and `execution_params` as JSONB; no decision metadata columns. (ml/stores/migrations_bootstrap/001_bootstrap.sql:225-241)
  - Evidence: DataReader read_signals is a passthrough to StrategyStore and does not enrich decision metadata; returns whatever columns are stored. (ml/stores/common/data_reader.py:600-695)
  - Evidence: DataWriter write_signals does not run schema validation or contract enforcement (no preflight/validator usage); it only ensures dataset registration and writes via StrategyStore. (ml/stores/common/data_writer.py:772-880)
  - Evidence: Schema audit expects `signal_type` on `ml_strategy_signals`, reinforcing DB/schema side uses `signal_type` rather than `signal` int. (ml/stores/schema_audit.py:316-332)
  - Evidence: Arbiter/Trust Layer plan expects `MLSignal.metadata` to include decision context (mu/sigma/p/horizon_ms/etc.) and `execution_params` to include decision policy fields (arm/propensity/action/LCB/m_t/u0/etc.). (ml/strategies/ARBITER_TRUST_LAYER_PLAN.md:24-106)

#### Signals Dataset Schema Gap (Migration Sketch)
- [ ] Pick canonical field names for `signals` (signal_type vs signal int) and document the mapping rule.
- [ ] Align registry bootstrap manifest/contract with StrategySignal storage columns (signal_type, model_predictions, risk_metrics, execution_params, run_id/ingested_at).
- [ ] Document the chosen DataWriter validation approach (preflight/contract enforcement vs bypass) after Decision Backlog is resolved.
- [ ] Define a backward-compatible ingest/conversion path for legacy payloads that only carry `signal` (int).
- [ ] Document the chosen migration tooling approach for signals schema changes (e.g., ContractEnforcerComponent).

### E) Sizing and Risk Controls
- Inventory:
  - ml/strategies/sizing.py (Kelly/vol/composite)
  - ml/strategies/risk.py (risk checks + staged actions)
- Notes:
  - Review how sizing inputs are derived (realized vs predicted volatility).
  - Evidence: Sizing uses realized volatility targeting + Kelly from historical wins/losses; confidence scaling optional, no forecast vol/cov inputs. (ml/strategies/sizing.py)
  - Evidence: Risk limits are static config thresholds (loss per trade, drawdown, exposure) and not keyed to model horizon/label. (ml/strategies/risk.py)
  - Evidence: Staged risk actions emit telemetry via `ml_risk_action_total` when HALT/LIQUIDATE triggers fire. (ml/strategies/risk.py:83-422)

### F) Portfolio Allocation
- Inventory:
  - ml/strategies/portfolio.py (risk parity/equal/kelly allocation)
- Notes:
  - Correlation lookback/horizon needs mapping to decision cadence.
  - Evidence: Risk-parity allocation uses inverse realized volatility and confidence scaling; correlation lookback default 60 days with decay. (ml/strategies/portfolio.py)
  - Evidence: PortfolioManager is invoked per-signal (`allocate_signals([signal], ...)`), so cross-signal allocation/correlation logic may not engage unless signals are batched upstream. (ml/strategies/common/position_management.py:827-832)
  - Evidence: ReturnsUpdater controls returns cadence via ReturnsConfig (bar_spec/annualization); no explicit linkage to prediction horizon. (ml/strategies/common/returns_updater.py, ml/config/base.py)
  - Evidence: PortfolioManager filters viable signals by confidence >= 0.5 and scales allocation weights by confidence (risk parity and kelly paths). (ml/strategies/portfolio.py:210-307)

### G) Pipelines and Orchestration
- Inventory:
  - ml/pipelines/build_runner.py
  - ml/training/README.md (export/registry contracts)
- Notes:
  - Verify training/export metadata captures label definitions and decision context.
  - Evidence: Model manifest includes `decision_policy`/`decision_config`, but export stub defaults to None/{} unless provided; trainer persistence can populate from config. (ml/registry/base.py, ml/training/export.py, ml/training/common/persistence.py)
  - Evidence: Dataset build pipeline config passes `horizon_minutes` and `threshold` defaults (15, 0.001) without cost-aware label integration. (ml/pipelines/build_runner.py)
  - Evidence: LightGBM student distiller writes `student.meta.json` with output_schema/calibration metadata, but the student manifest builder does not surface output_schema in ModelManifest. (ml/training/student/lightgbm.py, ml/registry/utils.py)
  - Evidence: ModelComponent extracts ONNX input/output names and shapes only; no code path loads `student.meta.json` output_schema into actor metadata. (ml/actors/common/model.py, ml/training/student/lightgbm.py)
  - Evidence: Promotion stage resolves model_id from model_metrics.json or teacher_meta.json; no mention of student.meta.json in stage2 promotion. (ml/orchestration/promotions.py)
  - Evidence: Student distillation CLIs register ModelManifest via build_student_manifest without passing student.meta.json fields (output_schema, calibrator_kind/params). (ml/training/distillation/cli.py:121-179, ml/training/student/lightgbm_cli.py:49-101)
  - Evidence: Chronos distillation registers student manifest via build_student_manifest and does not persist student.meta.json metadata into the registry. (ml/training/autogluon/chronos_distillation.py:441-463)
  - Evidence: LightGBM student distiller bakes Platt parameters into ONNX graph and records calibrator_params in student.meta.json, but registry/actor metadata does not ingest these parameters. (ml/training/student/lightgbm.py:279-343, ml/actors/common/model.py:479-521, ml/actors/common/registry.py:535-552)
  - Evidence: TFT teacher CLI writes `model_metrics.json` with metrics (roc_auc/pr_auc/logloss/brier/ece/sharpe) but no calibration parameters; `teacher_meta.json` only stores calibrator flag + logits hint. (ml/training/teacher/tft_cli.py:1326-1377, ml/training/teacher/tft_cli.py:1133-1141)
  - Evidence: BaseTeacher calibration stores Platt coef/intercept only in-memory (`_platt_coef/_platt_intercept`); no persistence path in base interface. (ml/training/teacher/base.py:41-83)
  - Evidence: Stage controller auto-creates `model_metrics.json` with minimal fields (model_id/model_path/architecture/feature_schema_hash/serveable) when missing, without calibration metadata. (ml/orchestration/common/stage_controller.py:717-727)
  - Evidence: Streaming training runner writes manifest `cohort_run.metrics` from `result.metrics` and registers ModelManifest.performance_metrics with those values; no calibration parameters or output_schema are persisted. (ml/cli/streaming_training_runner.py:846-915)
  - Evidence: Student distillation CLIs register manifests directly and do not use `student.meta.json` beyond printing its path (meta_path is unused). (ml/training/student/lightgbm_cli.py:60-103, ml/training/distillation/cli.py:128-180)
  - Evidence: ModelManifest fields include decision_policy/config but no output_schema or calibration parameters. (ml/registry/base.py:100-127)
  - Evidence: `build_student_manifest` returns a ModelManifest without output_schema/calibrator fields; only core manifest fields are set. (ml/registry/utils.py:66-131)
  - Evidence: ONNX model metadata extraction records input/output shapes and names only; no output_schema/calibration metadata is captured. (ml/actors/common/model.py:479-521)
  - Evidence: Registry model load populates `_model_metadata` from manifest fields only (decision_policy/config, feature schema, etc.). (ml/actors/common/registry.py:535-553)

### H) Evaluation and Validation
- Inventory:
  - ml/evaluation/metrics.py (binary metrics, ECE)
  - ml/training/datasets/validation_splitter.py (time-based splits)
- Notes:
  - Need explicit checklist of leakage guards (purge/embargo), walk-forward splits, and trading metrics.
  - Current splitter does time splits but no purge/embargo logic for overlapping horizons.
  - Evidence: Evaluation module provides only ML classification metrics (log loss/ROC/PR AUC/ECE), not trading metrics or cost-aware PnL. (ml/evaluation/metrics.py)
  - Evidence: ValidationSplitter docstring claims purged walk-forward support, but `split_dataset` only performs contiguous ratio splits with no purge/embargo logic. (ml/training/datasets/validation_splitter.py:36-152)
  - Evidence: Stage2 returns engine uses prediction threshold 0.5 to generate long/short signals and applies cost model (cost/commission/slippage bps) when computing backtest metrics. (ml/orchestration/stage2_engine.py)
  - Evidence: Stage2 computes realized returns by looking up bar closes at `ts` and `ts + horizon` (horizon_minutes) and uses that to score strategy returns. (ml/orchestration/stage2_engine.py)
  - Evidence: Stage2 cost model applies bps charges both on entry (`abs(signals) > 0`) and on turns (`diff(signals)`), not just round-trip cost. (ml/orchestration/stage2_engine.py)
  - Evidence: Training evaluation trading metrics convert classifier predictions via 0.5 threshold (long/short) and regression via sign; no cost/slippage adjustments applied. (ml/training/common/evaluation.py:166-254)
  - Evidence: Training evaluation annualizes Sharpe via sqrt(252) and assumes daily returns in `calculate_trading_metrics`. (ml/training/common/evaluation.py:221-231)
  - Evidence: Streaming economic metrics compute slippage-adjusted Sharpe/turnover/drawdown from probabilities with 0.5 threshold; defaults to synthetic returns if validation_returns missing. (ml/training/event_driven/economic_metrics.py)
  - Evidence: Streaming economic metrics default to +/-10 bps returns when `validation_returns` missing; slippage cost is applied per-signal (`slippage_bps/10_000 * |signal|`). (ml/training/event_driven/economic_metrics.py:20-154)
  - Evidence: CrossValidationComponent maps `standard`/`blocked` strategies to time_series CV; purged CV falls back to time_series if PurgedCrossValidator import fails. (ml/training/common/cross_validation.py:151-396)
  - Evidence: Purged CV pulls `purge_gap`/`embargo_pct` from config with defaults of 0 when unset. (ml/training/common/cross_validation.py:405-412)
  - Evidence: CrossValidationComponent reads `embargo_pct` from trainer config but AdvancedTrainingConfig does not define it, so embargo defaults to 0.0 unless a trainer config adds the attribute. (ml/training/common/cross_validation.py, ml/config/shared.py)
  - Evidence: AdvancedTrainingConfig defaults cv_strategy="time_series" and purge_gap=10. (ml/config/shared.py)
  - Evidence: AdvancedTrainingConfig only exposes cv_strategy/cv_folds/purge_gap (no embargo_pct field). (ml/config/shared.py:303-335)
  - Evidence: TFT teacher CLI uses `create_purged_splits` with purge_gap and embargo_hours, deriving embargo_pct from dataset span. (ml/training/teacher/tft_cli.py, ml/tasks/datasets/splits.py)
  - Evidence: TFT teacher CLI falls back to `create_purged_splits(..., purge_gap, embargo_hours)` when time-window validation is not used. (ml/training/teacher/tft_cli.py:576-605)
  - Evidence: `create_purged_splits` derives `embargo_pct = embargo_hours / total_hours` (clamped to [0, 0.5]) from the training span and passes it into PurgedCrossValidator. (ml/tasks/datasets/splits.py:40-83)
  - Evidence: Cross-validation runs only when cv_folds > 1; otherwise training proceeds without CV. (ml/training/common/cross_validation.py)
  - Evidence: LightGBM/XGBoost configs only set cv_strategy/cv_folds/purge_gap via AdvancedTrainingConfig (env-driven); no CLI-level overrides for non-distilled trainers. (ml/config/lightgbm.py, ml/config/xgboost.py, ml/config/shared.py)
  - Evidence: LightGBM/XGBoost `from_env()` only attaches `advanced_config` when ML_TRAIN_* keys are present. (ml/config/lightgbm.py, ml/config/xgboost.py)
  - Evidence: LightGBM/XGBoost config properties default cv_strategy="time_series" and cv_folds=5 even when advanced_config is None, so CV is enabled by default for those trainers. (ml/config/lightgbm.py, ml/config/xgboost.py)
  - Evidence: Deployment `.env.example` does not define ML_TRAIN_* overrides, so CV strategy defaults unless CLI args are used. (ml/deployment/.env.example:1-110)
  - Evidence: docker-compose env for streaming_training_runner sets runtime inputs but no ML_TRAIN_* overrides. (ml/deployment/docker-compose.yml:86-120)
  - Evidence: ChronosTrainer is a standalone AutoGluon trainer (not BaseMLTrainer), so it bypasses CrossValidationComponent and uses its own time-split evaluation utilities. (ml/training/autogluon/chronos_trainer.py, ml/training/autogluon/chronos_evaluation.py)
  - Evidence: ChronosTrainer is defined as a standalone class (no BaseMLTrainer inheritance). (ml/training/autogluon/chronos_trainer.py:92-139)
  - Evidence: Chronos evaluation splits data by timestamp fractions (train_fraction/val_fraction) without purge/embargo, using boundary timestamps for train/val/test. (ml/training/autogluon/chronos_evaluation.py:331-419)
  - Evidence: Chronos evaluation/baselines report regression metrics (mse/rmse/mae) via calculate_regression_metrics only. (ml/training/autogluon/chronos_evaluation.py:520-577)
  - Evidence: Streaming pipeline config enables temperature/Platt/isotonic calibration toggles for evaluation. (ml/config/streaming_pipeline.py:612-617)
  - Evidence: Streaming worker computes temperature/Platt/isotonic calibration metrics on logits/probabilities and reports them as metrics (best temperature chosen by log-loss). (ml/training/event_driven/worker.py:2168-2332)
  - Evidence: Streaming worker calibration includes best temperature in metrics output but does not persist calibration parameters for inference. (ml/training/event_driven/worker.py:2180-2328)
  - Evidence: Streaming promotion config gates on economic metrics (slippage-adjusted Sharpe/hit-rate/turnover/drawdown) and stability_calibration_drift; these depend on metrics emitted by streaming worker rather than persisted calibration parameters. (ml/config/streaming_pipeline.py:1100-1125)
  - Evidence: TFT teacher CLI uses time-window split when `val_days` set; otherwise uses `create_purged_splits` with purge_gap/embargo_hours, and falls back to a fixed 80/20 split if no CV split available. (ml/training/teacher/tft_cli.py:576-619)
  - Evidence: HPO TFT CLI only passes `--val_days` to the teacher CLI (time-window validation), with no purge/embargo flags in the common args. (ml/cli/hpo_tft.py:167-171, ml/cli/hpo_tft.py:306-331)
  - Evidence: Student distillation CLI uses pre-split NPZ inputs and does not expose CV/purge/embargo flags. (ml/training/distillation/cli.py:52-71)
  - Evidence: TFT teacher CLI computes Sharpe using probability-thresholded signals (>=0.5) against validation returns (no explicit cost adjustments). (ml/training/teacher/tft_cli.py:123-140, ml/training/teacher/tft_cli.py:1221-1224)
  - Evidence: TFT teacher CLI writes `teacher_meta.json` with `calibrator` flag and `onnx_output_is_logits` only (no calibration parameters). (ml/training/teacher/tft_cli.py:1133-1141)
  - Evidence: Legacy teacher CLI (compat) writes `teacher_meta.json` with only model_id + calibrator flag (no calibration parameters). (ml/training/teacher/cli.py:106-123)
  - Evidence: Runtime strategy analytics track fees/slippage and compute cost_ratio in execution-quality metrics (not used by training evaluation). (ml/strategies/analytics.py:133-252, ml/strategies/analytics.py:395-408)

### F) Portfolio Allocation
- Inventory:
  - ml/strategies/portfolio.py (PortfolioManager + correlation adjustments)
  - ml/strategies/common/position_management.py (allocation hook)
  - ml/strategies/common/signal_routing.py (multi-model aggregation)
  - ml/strategies/base_facade.py (signal routing + portfolio wiring)
- Notes:
  - Evidence: SignalRoutingComponent aggregates buffered multi-model signals within a time window and emits a single MLSignal with `metadata["aggregated_from"]`. (ml/strategies/common/signal_routing.py:271-367)
  - Evidence: Strategy on_data path routes each MLSignal through SignalRoutingComponent; only the aggregated (single) signal is forwarded to _handle_ml_signal. (ml/strategies/base_facade.py:1107-1162)
  - Evidence: PositionManagementComponent calls PortfolioManager.allocate_signals with a list containing a single signal; no multi-signal batch is passed from the strategy path. (ml/strategies/common/position_management.py:768-845)
  - Evidence: PortfolioManager.allocate_signals is designed to accept a list of signals and applies optional correlation adjustments + position limits. (ml/strategies/portfolio.py:138-207)
  - Evidence: Multi-signal allocation is exercised in tests (allocate_signals called with multiple signals), but the production call site remains single-signal. (ml/tests/unit/strategies/test_portfolio_and_exposure_invariants.py:96-120, ml/tests/unit/strategies/test_portfolio_allocation_risk_parity.py:9-50, ml/strategies/common/position_management.py:829-832)

### I) Observability and Fallbacks
- Inventory:
  - ml/common/metrics_bootstrap.py, ml/common/metrics_manager.py (metric helpers)
  - ml/strategies/risk.py, ml/strategies/sizing.py, ml/strategies/portfolio.py (counters/gauges)
- Notes:
  - Ensure fallback activation metrics are present for stores and actor inference.
  - Evidence: Signal facade hot path calls `_persist_prediction()` and `_try_generate_signal()`; model_store + strategy_store writes occur synchronously on the hot path. (ml/actors/signal_facade_impl.py:450-504, ml/actors/signal_facade_impl.py:655-664, ml/actors/signal_facade_impl.py:1113-1122)
  - Evidence: Base actor hot path uses async persistence worker when configured, but falls back to synchronous feature_store/model_store writes when worker is absent; publish uses `publish_data` directly. (ml/actors/base.py:1459-1511, ml/actors/base.py:1549-1591)
  - Evidence: MLActorConfig defaults `enable_async_persistence=True` and is overridden by ML_ENABLE_ASYNC_PERSISTENCE/ML_PERSISTENCE_* env vars; deployment `.env.example` sets async persistence enabled. (ml/config/base.py:326-467, ml/deployment/.env.example:82-85)
  - Evidence: Deployment `.env` does not set ML_ENABLE_ASYNC_PERSISTENCE or ML_PERSISTENCE_* overrides, so runtime relies on code defaults/env injection. (ml/deployment/.env:1-75)
  - Evidence: ml_signal_actor/ml_strategy docker-compose env blocks omit ML_ENABLE_ASYNC_PERSISTENCE/ML_PERSISTENCE_* overrides (defaults apply unless injected elsewhere). (ml/deployment/docker-compose.yml:158-230)
  - Evidence: FeaturesComponent persists via async queue when worker exists; sync fallback writes directly to FeatureStore. (ml/actors/common/features.py:540-619)
  - Evidence: Multi-signal actor flush path uses best-effort OpenTelemetry + metrics; flush invokes per-instrument `_generate_prediction_protected()`. (ml/actors/multi_signal.py:269-315)
  - Evidence: Signal facade uses best-effort actor-bus publish with try/except + `exc_info=True`. (ml/actors/signal_facade_impl.py:712-781)
  - Evidence: Actor `publish_data` calls `_msgbus.publish_c(...)` directly; no async wrapper. (nautilus_trader/common/actor.pyx:2429-2446)
  - Evidence: Message bus `publish_c` invokes subscription handlers synchronously and may serialize/publish to database when `external_pub` and `_database` are set. (nautilus_trader/common/component.pyx:2712-2763)
  - Evidence: Kernel sets message bus database only when `config.message_bus.database` is provided; otherwise `_msgbus_db` is None (no external backing). (nautilus_trader/system/kernel.py:277-340)
  - Evidence: `MessageBusConfig` defines optional `database` (default None) and external stream settings for backing. (nautilus_trader/common/config.py:338-396)
  - Evidence: ML signal actor container builds `TradingNodeConfig` without a `message_bus` override. (ml/deployment/entrypoint_actor.py:267-305)
  - Evidence: ML strategy container builds `TradingNodeConfig` without a `message_bus` override. (ml/deployment/entrypoint_strategy.py:153-173)
  - Evidence: Local dry-run uses `TradingNodeConfig` without a `message_bus` override. (ml/deployment/run_local_dry_run.py:199-209)
  - Evidence: ML streaming containers explicitly enable the ML bus (Redis) via `ML_BUS_ENABLE=1`, but ml_signal_actor/ml_strategy env blocks omit ML_BUS_* settings. (ml/deployment/docker-compose.yml:63-120, ml/deployment/docker-compose.yml:157-239)
  - Evidence: Deployment override compose file only sets DB/pipeline envs; no message-bus settings are present. (ml/deployment/docker-compose.override.yml:1-35)
  - Evidence: Deployment `.env` contains DB/pipeline/streaming keys but no ML_BUS_* entries; ML bus defaults rely on environment parsing. (ml/deployment/.env:1-75, ml/deployment/.env.example:45-55, ml/config/bus.py:1-112)
  - Evidence: Deployment README documents ML bus configuration solely via ML_BUS_* env vars and describes it as optional. (ML_DEPLOYMENT_README.md:130-156)
  - Evidence: ActorBusConfig defaults `from_actor=False`/`from_store=False` and is driven entirely by ML_BUS_FROM_ACTOR/ML_BUS_FROM_STORE env flags (publishing disabled unless explicitly enabled). (ml/config/actor_bus.py:19-78)
  - Evidence: Actor-side DomainEventBridge is only initialized when ML_BUS_FROM_ACTOR=true and MessageBusConfig.enabled=true; otherwise publishing is disabled and store-level publishing is left intact. (ml/actors/ml_domain_events.py:506-583)
  - Evidence: MLIntegrationManagerFacade/ActorFactory configure_message_bus are explicit no-op stubs (no Nautilus message-bus wiring in ML integration layer). (ml/core/integration_facade.py:1029-1045, ml/core/common/actor_factory.py:181-209)
  - Evidence: DataWriterComponent declares all methods cold-path and emits events/watermarks best-effort (non-blocking). (ml/stores/common/data_writer.py:3-11, ml/stores/common/data_writer.py:82-88, ml/stores/common/data_writer.py:1474-1527)
  - Evidence: LiveDataRecorder flushes asynchronously via asyncio tasks and records metrics best-effort. (ml/stores/writers.py:443-560)
  - Evidence: Signal facade uses best-effort bus publish with `exc_info=True` on failures; base actor prediction failure logging does not include `exc_info`. (ml/actors/signal_facade_impl.py, ml/actors/base.py)
  - Evidence: Inference fallback counters track input/output shape mismatches. (ml/actors/signal_facade_impl.py)
  - Evidence: Sizing/risk/portfolio record metrics via `metrics_bootstrap` counters/gauges. (ml/strategies/sizing.py, ml/strategies/risk.py, ml/strategies/portfolio.py)
  - Evidence: Actor StoreOperationsComponent emits `ml_fallback_activations_total` per store when falling back to DummyStore. (ml/actors/common/store_operations.py)
  - Evidence: Store operations layer defines progressive fallback chains and records `ml_fallback_activations_total` on activation. (ml/stores/common/store_operations.py)
  - Evidence: DataRegistry fallback logs POSTGRES/JSON init failures without exc_info in warning logs. (ml/stores/mixins.py:635-660)

## Remaining Evidence Gaps (Needs Code Evidence)
- [ ] Hot-path constraints: confirm runtime deployments set ML_ENABLE_ASYNC_PERSISTENCE/ML_PERSISTENCE_* (repo `.env`/compose do not) and whether sync fallbacks occur in practice.
- [ ] Decision metadata propagation: check for any downstream consumers that require decision metadata beyond model_predictions/risk_metrics/execution_params.
- [ ] Target variants: confirm any production datasets/models use non-binary, multi-horizon, or cost-aware labels, or non-default LightGBM/XGBoost objectives.
- [ ] Evaluation leakage controls: blocked until a successful training run exists; confirm whether any runs use TFT CLI purge/embargo args or set cv_strategy="purged" (env/CLI), given default configs/time-window splits.

## Open Questions
- Do any downstream consumers rely on decision metadata beyond `model_predictions`/`risk_metrics`/`execution_params`, and where should that metadata live?
- Should `max_holding_ms` / `min_hold_ms` be derived from label horizon (horizon_minutes), or remain purely strategy-configured?
- What is the canonical meaning of `target` for non-distilled trainers (binary vs regression/multiclass), and are multi-horizon/cost-aware labels required?
- Should async persistence be mandatory in production to avoid synchronous store writes in hot paths, or are synchronous fallbacks acceptable?
- No successful training runs yet: once runs succeed, confirm whether any use cv_strategy="purged" or TFT CLI purge/embargo args (embargo_hours → embargo_pct).

## Issue Readiness & Change Map
This section summarizes whether each issue is ready for implementation or blocked by
design decisions / operational validation. It does not replace the Decision Backlog.

### 1) Prediction/Decision Semantics Inconsistency
- Decision required: define canonical prediction surface (probability vs signed score vs class index),
  map classifier outputs explicitly, and define confidence meaning/gating rules.
- Implementation ready once decided: normalize ONNX/single-output handling, align strategy thresholds,
  and enforce signal strategies against the chosen scale.
- Operational validation: confirm live inference output conventions once a real model is deployed.

### 2) Decision Metadata Not Propagated + Signals Schema Drift
- Decision required: define decision metadata schema (policy/horizon/label/calibration) and
  choose canonical signals contract fields (`signal_type` vs `signal`).
- Implementation ready once decided: add metadata propagation into MLSignal/StrategySignal/DecisionEvent,
  update registry contract + DataWriter enforcement, and add backward-compatible conversion.
- Operational validation: confirm downstream consumers and any external schemas (none known yet).

### 3) Target Semantics Narrow / Not Aligned With Trading Outcomes
- Decision required: select label family (binary long-only vs multi-class/short/neutral vs cost-aware)
  and whether multi-horizon labels are mandatory.
- Implementation ready once decided: add target generator variants, update dataset builders,
  and persist target metadata in manifests/sidecars.
- Operational validation: verify training runs produce expected label distributions.

### 4) Leakage Controls Inconsistent
- Decision required: enforce purged/embargoed CV across trainers (or document explicit exceptions)
  and decide whether training eval must include costs/slippage.
- Implementation ready once decided: add embargo_pct to shared config, propagate to trainers/CLIs,
  and align Chronos/HPO/TFT split behavior.
- Operational validation: confirm with successful training runs (blocked today).

### 5) Exit Policy vs Horizon Mismatch
- Decision required: tie exit policy defaults to label horizon or keep fully strategy-configured.
- Implementation ready once decided: link horizon_minutes → max_holding_ms/min_hold_ms or
  explicitly persist horizon metadata for auditing.
- Operational validation: check behavior in replay/backtest once horizon linkage is defined.

### 6) Hot-Path Blocking Risks
- Decision required: mandate async persistence in production or accept sync fallbacks;
  define whether publish_data is allowed on hot paths.
- Implementation ready once decided: enforce configs at deployment entrypoints and/or
  disable synchronous fallbacks.
- Operational validation: observe latency/queue drops under load in live or replay.

### 7) Calibration/Output Schema Not Persisted to Registry
- Decision required: where output_schema + calibration params live (manifest vs sidecar).
- Implementation ready once decided: ingest student.meta.json into ModelManifest or
  extend manifest schema; surface into actor metadata.
- Operational validation: verify inference uses calibration consistently once enabled.

### 8) Portfolio Aggregation Under-used
- Decision required: whether portfolio allocation should be multi-signal by default.
- Implementation ready once decided: batch signals upstream before allocate_signals,
  or document single-signal behavior as intended.
- Operational validation: compare allocation outcomes with multi-signal batching.

## Operational Validation Blockers
- No successful training runs yet, so leakage-control enforcement and target semantics
  are validated only at code level (not at runtime).
- No external deployments; message bus and async persistence settings are inferred
  from repo defaults and compose files only.
