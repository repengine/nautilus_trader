# Full Verification Plan (Claims Ledger)

Scope: manual verification of documented claims without relying on git history.
Sources are limited to:
- `training_status.md`
- `ml/docs/ops/pipeline_validation_option2_plan.md`
- `ml/docs/ops/chronos_training_runbook.md`

## Evidence Types (Needed-for-Done Methods)
- CODE: verify code path exists in the referenced module(s).
- CONFIG: verify config field or env var hook exists.
- ENTRY: verify runnable CLI/pipeline/entrypoint exists.
- TEST: verify test exists (and optionally pass when run later).

Status legend:
- UNVERIFIED: not yet checked
- VERIFIED: evidence collected
- FAILED: evidence missing or contradicted
- BLOCKED: cannot verify yet (missing data/access)

Each claim below includes the needed-for-done methods up front so the audit can
continue even if this chat ends.

---

## Claims Ledger - training_status.md

### Training Surfaces (Capabilities)
- [VERIFIED] TS-001 (CODE): BaseMLTrainer orchestrates data prep, optional Optuna HPO, CV, evaluation, MLflow tracking, and ONNX export; supports FeatureStore parity when DB connection configured. Evidence: `ml/training/base_facade.py`.
- [VERIFIED] TS-002 (CODE): MLflow tracking exists but is deprecated in favor of ModelRegistry. Evidence: `ml/training/common/mlflow_tracking.py`.
- [VERIFIED] TS-003 (CODE): Export utilities provide model type detection, ONNX conversion, and metadata sidecar. Evidence: `ml/training/export.py`.
- [VERIFIED] TS-004 (CODE): LightGBM trainer supports polars ingestion, categorical handling, GPU config, GOSS/DART/EFB options, early stopping, and feature importance. Evidence: `ml/training/non_distilled/lightgbm.py`.
- [VERIFIED] TS-005 (CODE): XGBoost trainer supports polars ingestion, GPU acceleration, and monotonic constraints. Evidence: `ml/training/non_distilled/xgboost.py`.
- [VERIFIED] TS-006 (CODE): tft_cli loads CSV/Parquet into pandas, enforces feature schema, splits train/val, and trains TFT teacher; can fallback to logistic regression if TFT fails. Evidence: `ml/training/teacher/tft_cli.py`.
- [VERIFIED] TS-007 (CODE): TFTTeacher uses PyTorch Forecasting + Lightning, supports warm-start from pretrained state dict, and builds validation using full sorted dataset. Evidence: `ml/training/teacher/tft_teacher.py`.
- [VERIFIED] TS-008 (CODE): `ml/training/teacher/cli.py` forwards to `tft_cli` when registry args present, otherwise runs legacy calibration-only mode. Evidence: `ml/training/teacher/cli.py`.
- [VERIFIED] TS-009 (CODE): `tft_model.py` is a placeholder scaffold distinct from `tft_teacher.py`. Evidence: `ml/training/teacher/tft_model.py`.
- [VERIFIED] TS-010 (CODE): MTM pretrainer exists for optional warm starts. Evidence: `ml/training/teacher/pretrain_mtm.py`.
- [VERIFIED] TS-011 (CODE): LightGBM student distiller trains on soft labels, optionally calibrates, and exports ONNX with metadata. Evidence: `ml/training/student/lightgbm.py`.
- [VERIFIED] TS-012 (ENTRY+CODE): Distillation CLI enforces feature registry parity and registers the student in the local registry. Evidence: `ml/training/distillation/cli.py`.
- [VERIFIED] TS-013 (CODE): Rolling soft labels are generated via helper that walks per-instrument cutoffs, calls predictor predict with future covariates, and aligns forecast step to origin row. Evidence: `ml/training/autogluon/soft_label_generator.py`.
- [VERIFIED] TS-014 (CONFIG+CODE): ChronosDistillationConfig includes rolling-window controls and soft-label targeting options (min_history, stride, forecast_step, label_strategy, soft_target_column, distilled_target_column, coverage threshold). Evidence: `ml/config/autogluon.py`.
- [VERIFIED] TS-015 (CODE): Distillation enforces soft-label coverage thresholds and requires student covariates match teacher; student targets blended by default. Evidence: `ml/training/autogluon/chronos_trainer.py`, `ml/config/autogluon.py`.
- [FAILED] TS-016 (CODE): TODO exists for baseline-improvement gating once naive baseline evaluation is available. Evidence gap: no matching TODO found in `ml/` sources.
- [VERIFIED] TS-017 (CODE): Streaming loader does low-memory metadata scans and shard replay to avoid full dataset materialization. Evidence: `ml/training/teacher/streaming_loader.py`.
- [VERIFIED] TS-018 (CODE): Dataset planner scans parquet metadata, applies caps, enforces guardrails. Evidence: `ml/training/event_driven/dataset_service.py`, `ml/training/event_driven/guardrails/dataset.py`.
- [VERIFIED] TS-019 (CODE): Streaming worker computes validation metrics and calibration diagnostics and persists logits. Evidence: `ml/training/event_driven/worker.py`.
- [VERIFIED] TS-020 (CODE): Streaming sweeps use Optuna to explore worker hyperparameters. Evidence: `ml/training/event_driven/sweep.py`.

### Pipelines / Entry Points
- [VERIFIED] TS-021 (ENTRY+CODE): `tft_train_distill` wraps the orchestrator to wire dataset build, teacher training, and student distillation. Evidence: `ml/pipelines/tft_train_distill.py`.
- [VERIFIED] TS-022 (ENTRY+CODE): `build_runner` orchestrates multi-symbol dataset builds with optional parallelism and progress logging; calls dataset build main by default. Evidence: `ml/pipelines/build_runner.py`.

### Consumers / Observability
- [VERIFIED] TS-023 (CODE): Streaming training state store collects plan/result/heartbeat records and updates gauges for backlog/progress/metrics. Evidence: `ml/consumers/streaming_training.py`.
- [VERIFIED] TS-024 (CODE): Persistence service wires streaming training events to Redis Streams consumption. Evidence: `ml/consumers/streaming_training_service.py`.
- [VERIFIED] TS-025 (CODE): Redis Streams consumer provides gate + handler loop (example-level). Evidence: `ml/consumers/redis_streams_consumer.py`.
- [VERIFIED] TS-026 (CODE): Aggregator, retry, idempotent, lineage consumers provide buffering, DLQ retry, watermark gating, and lineage persistence. Evidence: `ml/consumers/aggregator.py`, `ml/consumers/retry.py`, `ml/consumers/idempotent.py`, `ml/consumers/lineage_writer.py`.

### Recent Updates (CPU Training Assessment) - Feature handling + CPU enforcement
- [VERIFIED] TS-027 (CODE+TEST): tft_cli resolves feature columns by coercing datetime-like and numeric-like strings, auto-detecting static categoricals, encoding dynamic categoricals to numeric codes, filling missing static categoricals, and using numeric-only fallback. Evidence: `ml/training/teacher/tft_cli.py` (`_resolve_tft_feature_columns`).
- [FAILED] TS-028 (CODE): TFTTeacher forces prediction to respect `--accelerator cpu` and mirrors training-time NA handling for numeric features and static categoricals during prediction. Evidence gap: `ml/training/teacher/tft_teacher.py` fills NA only in training path and does not explicitly pin prediction device to CPU beyond training-time accelerator config.
- [VERIFIED] TS-029 (TEST): Unit coverage exists for new feature-resolution behavior. Evidence: `ml/tests/unit/training/teacher/test_tft_cli_feature_handling.py`.

### Fixes applied (Chronos)
- [VERIFIED] TS-043 (CODE): Chronos dataset prep canonicalizes `timestamp` -> `ts_event` and drops redundant column. Evidence: `ml/experiments/chronos_training_experiment.py` (calls `canonicalize_timestamp_column`).
- [VERIFIED] TS-044 (CODE): AutoGluon adapter canonicalizes `timestamp` -> `ts_event` (drops duplicates). Evidence: `ml/data/autogluon_adapter.py` (`canonicalize_timestamp_column`).
- [VERIFIED] TS-045 (CODE): Soft-label merge aligns timezone-aware `ts_event` for Polars/Pandas. Evidence: `ml/training/autogluon/soft_label_generator.py` (`_merge_soft_labels` timezone handling).
- [VERIFIED] TS-046 (CONFIG+ENTRY+CODE): Distillation window controls + coverage thresholds added to experiment config and CLI. Evidence: `ml/config/autogluon.py` (ChronosDistillationConfig), `ml/experiments/chronos_training_experiment.py` (distill_* args + overrides).
- [VERIFIED] TS-047 (CODE): Distillation builds future covariate frames via AutoGluon, synthesizes missing time-based covariates, tracks coverage, and adds window sampling strategy knob. Evidence: `ml/training/autogluon/soft_label_generator.py` (`make_future_data_frame`, `_build_future_covariates`, `_sample_indices`), `ml/training/autogluon/chronos_trainer.py` (coverage logging).
- [VERIFIED] TS-048 (CONFIG+ENTRY+CODE): AutoGluon tuning configurable via ChronosTuningConfig and CLI flags. Evidence: `ml/config/autogluon.py` (ChronosTuningConfig), `ml/experiments/chronos_training_experiment.py` (`--tune-num-trials`, `--tune-searcher`, `--tune-scheduler`).
- [VERIFIED] TS-049 (DOC): Ops runbook captures updated CLI flags and experiment notes. Evidence: `ml/docs/ops/chronos_training_runbook.md` (tuning/distill flags table and notes).
- [VERIFIED] TS-050 (CODE): Logging creates LOG_FILE parent directories to avoid run aborts. Evidence: `ml/common/logging_config.py`.
- [FAILED] TS-051 (CODE): TFT dataset builders import FeatureStore from facade to avoid import errors. Evidence: `ml/data/tft_dataset_builder.py` imports `FeatureStore` from `ml.stores.feature_store`, not a facade module.
- [FAILED] TS-052 (CODE): Chronos tuning supplies fine-tune search spaces and requires fine_tune=True so HPO no longer errors on missing search spaces. Evidence gap: `ml/experiments/chronos_training_experiment.py` parses `--fine_tune` but does not use it; `ml/training/autogluon/chronos_trainer.py` does not define tuning search spaces.

### Key Gaps / Risks
- [VERIFIED] TS-065 (CODE): tft_cli reads full dataset into pandas and sorts/slices for train/val; memory heavy on large datasets. Evidence: `ml/training/teacher/tft_cli.py` (read_parquet/read_csv into pandas, then sort/slice).
- [VERIFIED] TS-066 (CODE): TFTTeacher.fit makes additional copies increasing memory pressure. Evidence: `ml/training/teacher/tft_teacher.py` (pd.DataFrame(...).copy(), sorted slices).
- [VERIFIED] TS-067 (CODE): TFT fallback to logistic regression changes model class and metric characteristics. Evidence: `ml/training/teacher/tft_cli.py` (logistic regression fallback on TFT failure).
- [FAILED] TS-069 (CODE): TFTTeacher.fit_streaming bootstraps small PF dataset and trains on streaming dataloaders to avoid full materialization. Evidence gap: `ml/training/teacher/tft_teacher.py` uses a baseline fallback model when `_tft` is None and does not bootstrap a PF dataset.
- [FAILED] TS-070 (CODE): Streaming worker uses TFTStreamingDataModule for train/val loaders. Evidence gap: `ml/training/event_driven/worker.py` uses `build_streaming_dataloader`, not `TFTStreamingDataModule`.
- [VERIFIED] TS-071 (CONFIG+CODE): StreamingWorkerConfig exposes tuning knobs/caps; optimizer/scheduler knobs accepted but not wired; lr_scheduler stored but unused; onecycle/cosine not mapped. Evidence: `ml/config/streaming_pipeline.py` (optimizer/lr_scheduler config) + `ml/training/teacher/tft_teacher.py` (stores but does not use optimizer/scheduler).
- [VERIFIED] TS-072 (CODE): Streaming worker persists logits only; no model artifact export or registry integration. Evidence: `ml/training/event_driven/worker.py` (`_persist_logits`, no registry writes).
- [VERIFIED] TS-073 (ENTRY+CODE): HPO CLI supports subprocess isolation but orchestrator path does not expose the flag. Evidence: `ml/cli/hpo_tft.py` (`--subprocess`), no corresponding flag in `ml/training/event_driven/` orchestration.
- [VERIFIED] TS-074 (CODE): Multiple TFT surfaces exist: full implementation in tft_teacher.py and placeholder in tft_model.py. Evidence: `ml/training/teacher/tft_teacher.py`, `ml/training/teacher/tft_model.py`.

### Streaming Trainer Goals + Done Criteria
- [FAILED] TS-075 (CODE+TEST): High-fidelity parity goal: streaming training matches offline TFT behavior on fixed sample within tolerance, honoring feature toggles and schema parity. Evidence gap: no streaming parity tests found in `ml/tests/`.
- [FAILED] TS-076 (CODE+TEST): Scale + reliability goal: large-scale streaming shards train deterministically with bounded RSS/VRAM and safe resume. Evidence gap: no determinism/scale tests found for streaming worker.
- [FAILED] TS-078 (CODE): Baseline current: streaming training uses bootstrap PF dataset + streaming dataloaders; fallback baseline if dependencies unavailable. Evidence gap: `ml/training/teacher/tft_teacher.py` uses a baseline fallback when `_tft` is None, but no PF dataset bootstrap in streaming path.
- [VERIFIED] TS-079 (CODE): Baseline current: streaming metadata captures global numeric stats and feeds target_scale; per-instrument target stats not captured. Evidence: `ml/training/teacher/streaming_loader.py` (`TFTStreamingPreprocessor.build_metadata`, `TFTStreamingDataset` uses `numeric_stats` for `target_scale`).
- [VERIFIED] TS-080 (CODE): Baseline current: streaming workers persist logits only; no model export/registry integration. Evidence: `ml/training/event_driven/worker.py` (`_persist_logits` only).
- [VERIFIED] TS-081 (CODE): Baseline current: planner -> worker -> persistence/telemetry pipeline is live with guardrails and capped shard scheduling. Evidence: `ml/training/event_driven/dataset_service.py`, `ml/training/event_driven/guardrails/dataset.py`, `ml/training/event_driven/worker.py` (`apply_streaming_limits`).
- [VERIFIED] TS-082 (CODE): Baseline current: scheduler support not wired; lr_scheduler stored but unused. Evidence: `ml/config/streaming_pipeline.py` (lr_scheduler), `ml/training/teacher/tft_teacher.py` (stores but does not use).
- [VERIFIED] TS-083 (DOC): Baseline reset (2026-01-06) makes prior experiment results stale for model-quality comparisons. Evidence: `training_status.md`.
- [VERIFIED] TS-084 (CODE): Progress: streaming loader emits target_scale from global target stats; per-instrument stats still missing. Evidence: `ml/training/teacher/streaming_loader.py` (global `numeric_stats` + `target_scale`).
- [FAILED] TS-085 (CODE): Progress: streaming training bootstraps PF dataset with categorical vocab coverage and trains/infers logits via streaming dataloaders. Evidence gap: streaming worker uses `build_streaming_dataloader` without PF dataset bootstrap.
- [FAILED] TS-087 (CODE+TEST): Progress: streaming TFT teacher materializes shard data via polars+pandas, trains real TFT, aligns logits/targets/metadata via prediction helper, and restricts dummy fallback to missing-dependency cases. Evidence gap: `ml/training/teacher/tft_teacher.py` uses baseline fallback and does not train real TFT in streaming path.

### Requirements for done
- [FAILED] TS-093 (CODE+TEST): Per-feature scalers + per-instrument target normalization derived from streaming metadata; parity test confirms streaming ~= offline on small sample. Evidence gap: streaming uses global `numeric_stats` for `target_scale` with no per-instrument normalization, and no parity tests found under `ml/tests/` (`ml/training/teacher/streaming_loader.py`).
- [FAILED] TS-096 (CODE): Scheduler parity: onecycle/cosine map to real schedulers in streaming. Evidence gap: scheduler name is passed into `TFTTeacher` but no mapping or usage exists (`ml/training/event_driven/worker.py`, `ml/training/teacher/tft_teacher.py`).
- [FAILED] TS-097 (TEST+DOC): Fidelity + scale validation: parity tests + microbench/scale runs updated in ops docs and validation reports. Evidence gap: no streaming-vs-offline parity tests in `ml/tests/`; only persistence microbench exists (`ml/tests/performance/test_streaming_persistence_microbench.py`).

### Concrete implementation plan (phased)
- [VERIFIED] TS-098 (CODE): Streaming stats + encoders fidelity (global target stats + target_scale emitted; per-instrument stats pending). Evidence: global `numeric_stats` captured and `target_scale` uses global mean/std; no per-instrument target stats in metadata (`ml/training/teacher/streaming_loader.py`).
- [VERIFIED] TS-100 (CODE): Checkpoint resume determinism pending (checkpoint saves trainer state only; no shard cursor/seed captured). Evidence: checkpoint metadata tracks plan_id/dataset_id/epoch/global_step/metrics only; no shard cursor/seed fields (`ml/training/event_driven/worker.py`).
- [FAILED] TS-101 (CODE): Streaming training without full materialization done (bootstrap dataset + streaming dataloaders via TFTStreamingDataModule). Evidence gap: worker uses `build_streaming_dataloader` and does not use `TFTStreamingDataModule` (`ml/training/event_driven/worker.py`, `ml/training/teacher/streaming_loader.py`).
- [FAILED] TS-102 (CODE): Align materialization formatting by routing through TimeSeriesFormatter. Evidence gap: no `TimeSeriesFormatter` implementation under `ml/` (only doc/test mentions).
- [FAILED] TS-104 (CODE+TEST): Scheduler support: implement onecycle/cosine mapping in streaming with unit coverage. Evidence gap: no scheduler mapping in `TFTTeacher`, no unit tests for scheduler wiring (`ml/training/teacher/tft_teacher.py`).
- [FAILED] TS-105 (TEST+DOC): Validation + ops updates: run parity + microbench suites and refresh streaming_scaling_experiments.md with cohorts. Evidence gap: parity/scale validation not represented in tests; doc updates are runtime-only and not verifiable in code (`ml/docs/ops/streaming_scaling_experiments.md`).

### Recommended Next Steps (Prioritized)
- [FAILED] TS-106 (CODE): Align materialization with offline formatting (TimeSeriesFormatter route). Evidence gap: no `TimeSeriesFormatter` implementation under `ml/` (only doc/test mentions).
- [FAILED] TS-108 (CODE+TEST): Extend streaming optimizer/scheduler support (onecycle/cosine mapping). Evidence gap: scheduler names are stored but not mapped to Lightning/PyTorch schedulers; no tests (`ml/training/teacher/tft_teacher.py`).
- [FAILED] TS-109 (TEST): Add fidelity/scale validation tests (streaming vs offline + microbench). Evidence gap: no parity tests; only persistence microbench exists (`ml/tests/performance/test_streaming_persistence_microbench.py`).
- [FAILED] TS-110 (TEST+ENTRY): Add registry/parity coverage for streaming teacher export. Evidence gap: no tests covering registry export/parity for streaming teacher; no entrypoint for streaming teacher export verification.
- [FAILED] TS-111 (CODE+DOC): Clarify TFT implementation paths (remove or deprecate tft_model.py). Evidence gap: `ml/training/teacher/tft_model.py` still exists as a placeholder without a deprecation marker.
- [FAILED] TS-112 (DOC+CONFIG): Document baseline resource guidance (batch size, tail_rows, limit_groups, precision). Evidence gap: guidance is scattered; no dedicated baseline defaults doc or config guidance (see TODO in `training_status.md` and CLI examples in `ml/docs/README.md`, `ml/docs/tools/CLI_Tooling.md`).

---

## Claims Ledger - ml/docs/ops/pipeline_validation_option2_plan.md

### Goal and Scope
- [VERIFIED] OP2-001 (DOC): Goal is end-to-end plumbing validation for 5-10 symbols (not fidelity/profitability). Evidence: `ml/docs/ops/pipeline_validation_option2_plan.md`.
- [VERIFIED] OP2-002 (CONFIG+DOC): Scope uses 5-10 symbols from data/catalog (example list). Evidence: `ml/docs/ops/pipeline_validation_option2_plan.md`.
- [VERIFIED] OP2-003 (CODE): ONNX artifact required for actor inference; Chronos AutoGluon predictors not loadable by MLSignalActor. Evidence: `ml/actors/common/model.py` (ONNX-only enforcement).
- [VERIFIED] OP2-004 (CONFIG): Data source is local parquet catalog with ML_TFT_ALLOW_PARQUET_FALLBACK=1. Evidence: `ml/data/tft_dataset_builder.py` (ML_TFT_ALLOW_PARQUET_FALLBACK).
- [VERIFIED] OP2-005 (DOC): Broker not configured; order intents serialized for manual inspection. Evidence: `ml/docs/ops/pipeline_validation_option2_plan.md`.
- [VERIFIED] OP2-006 (CODE+CONFIG): Message bus Noop/Redis only; rely on JSONL store outputs unless file-backed publisher added. Evidence: `ml/common/message_bus.py`.
- [VERIFIED] OP2-007 (DOC): Replay pacing uses TestClock fast path; live-paced replay deferred. Evidence: `ml/docs/ops/pipeline_validation_option2_plan.md`.
- [VERIFIED] OP2-008 (CODE): Registries are metadata catalogs; feature/model parity checks must pass. Evidence: `ml/registry/feature_registry.py`, `ml/registry/model_registry_facade.py`, `ml/registry/data_registry.py`.

### Registration semantics
- [VERIFIED] OP2-009 (CODE): FeatureRegistry is schema contract (ordered feature names + dtypes + schema hash + pipeline signature). Evidence: `ml/registry/feature_registry.py`.
- [VERIFIED] OP2-010 (CODE): DataRegistry stores dataset manifest + lineage (not data). Evidence: `ml/registry/data_registry.py`.
- [VERIFIED] OP2-011 (CODE): ModelRegistry stores manifest + artifact path; enforces ONNX + feature parity (strict when ML_STRICT_FEATURE_PARITY=1). Evidence: `ml/registry/model_registry_facade.py`.
- [VERIFIED] OP2-012 (CODE): StrategyRegistry stores strategy manifests + model requirements. Evidence: `ml/registry/strategy_registry.py`.

### Strategy and actor readiness
- [VERIFIED] OP2-013 (CODE): BaseMLStrategyFacade composes six components (signal routing, decision persistence, position management, order submission, lifecycle, performance). Evidence: `ml/strategies/base_facade.py`.
- [VERIFIED] OP2-014 (CODE): MLTradingStrategy and SimpleMLStrategy implement decision logic and support execute_trades=False. Evidence: `ml/strategies/ml_strategy.py`, `ml/strategies/base_facade.py`.
- [VERIFIED] OP2-015 (CODE): DecisionPersistenceComponent persists decisions and can publish events via StrategyDecisionPublisher. Evidence: `ml/strategies/common/decision_persistence.py`.
- [VERIFIED] OP2-016 (CODE+CONFIG): StrategyStore supports Postgres and JSONL file fallback via ML_FILE_STORE_PATH. Evidence: `ml/core/common/store_initialization.py`, `ml/stores/file_backed.py`.
- [VERIFIED] OP2-017 (TEST): Strategy logic, store conformance, and integration backtests exist. Evidence: `ml/tests/unit/strategies/`, `ml/tests/integration/test_ml_strategy_backtest.py`.
- [VERIFIED] OP2-018 (CODE): BaseMLInferenceActor integrates 4 stores + 4 registries and enforces hot-path rules. Evidence: `ml/actors/base.py`.
- [VERIFIED] OP2-019 (CODE): MLSignalActorFacade provides signal generation, optional domain event emission, publishes MLSignal to bus. Evidence: `ml/actors/signal_facade_impl.py`.
- [VERIFIED] OP2-020 (CODE+CONFIG): Actor-side event publishing uses enums and build_topic_for_stage when enabled via ActorBusConfig + MessageBusConfig. Evidence: `ml/actors/signal_facade_impl.py`, `ml/actors/ml_domain_events.py`, `ml/config/actor_bus.py`, `ml/config/bus.py`.
- [VERIFIED] OP2-021 (CODE): File-backed stores serialize predictions/decisions to JSONL. Evidence: `ml/stores/file_backed.py`.
- [VERIFIED] OP2-022 (CONFIG): File-backed stores only activate when Postgres unavailable; otherwise persistence goes to Postgres. Evidence: `ml/core/integration_facade.py`, `ml/core/common/store_initialization.py`.
- [FAILED] OP2-023 (CODE): Missing items include order-intent serialization hook, orchestration harness, and ONNX-compatible inference artifact. Evidence: `ml/strategies/base_facade.py` (order intent serialization exists), `ml/actors/common/model.py` (ONNX required for inference).

### Phase 0: Preflight and Environment
- [VERIFIED] OP2-024 (DOC): Pick 5-10 symbols with confirmed coverage in data/catalog. Evidence: `ml/docs/ops/pipeline_validation_option2_plan.md`.
- [VERIFIED] OP2-025 (DOC): Exclude symbols requiring suffix parsing (BRK.B). Evidence: `ml/docs/ops/pipeline_validation_option2_plan.md`.
- [VERIFIED] OP2-026 (CONFIG): Export ML_TFT_ALLOW_PARQUET_FALLBACK=1 and ML_FILE_STORE_PATH. Evidence: `ml/docs/ops/pipeline_validation_option2_plan.md`.
- [VERIFIED] OP2-027 (CONFIG): Decide persistence target; JSONL outputs require Postgres unavailable. Evidence: `ml/docs/ops/pipeline_validation_option2_plan.md`.

### Phase 2: Model Artifact Preparation (ONNX)
- [FAILED] OP2-034 (CODE+ENTRY): Alternative: Chronos inference adapter (deferred unless blocked). Evidence gap: no Chronos inference adapter found under `ml/actors/`.

### Phase 3: Actor + Strategy Harness
- [VERIFIED] OP2-036 (CONFIG+CODE): Configure MLSignalActor to load ONNX model. Evidence: `ml/config/base.py` (model_path/model_id), `ml/actors/base.py` (load path).
- [VERIFIED] OP2-037 (CONFIG): Configure MLStrategyConfig execute_trades=False and use_strategy_store=True with ml_signal_source. Evidence: `ml/config/base.py`.
- [VERIFIED] OP2-039 (DOC): Use TestClock fast path; live-paced replay out of scope. Evidence: `ml/docs/ops/pipeline_validation_option2_plan.md`.

### Phase 4: Order-Intent Serialization
- [FAILED] OP2-043 (CODE): Optional file-backed MessagePublisherProtocol if bus-style events desired. Evidence gap: `ml/common/message_bus.py` supports Noop/Redis only.

### Phase 5: Observability + Health
- [VERIFIED] OP2-044 (ENTRY): validate-metrics and validate-events are run. Evidence: `Makefile` (validate-metrics, validate-events targets).
- [BLOCKED] OP2-045 (CODE+ENTRY): Actor/strategy health checks green; no hot-path violations. Evidence gap: requires executing the pipeline and observing runtime health checks; not verifiable in code-only audit.
- [VERIFIED] OP2-046 (CODE): Domain events use enums and build_topic_for_stage. Evidence: `ml/actors/signal_facade_impl.py`, `ml/actors/ml_domain_events.py`.

### Validation and Review Checklist
- [BLOCKED] OP2-059 (ENTRY): Health checks and metrics show no degraded states or hot-path errors. Evidence gap: requires runtime execution and telemetry inspection.

### Exit Criteria
- [BLOCKED] OP2-061 (ENTRY): End-to-end run completes without exceptions. Evidence gap: requires executing the E2E harness.
- [BLOCKED] OP2-064 (ENTRY): Health checks show stores/registries healthy or expected fallback. Evidence gap: requires runtime checks via integration manager/health endpoint.

### Follow-Ups
- [FAILED] OP2-067 (CODE+ENTRY): Replace broker stub with real broker integration. Evidence gap: order submission remains callback-based with no broker integration path (`ml/strategies/common/order_submission.py`).
- [FAILED] OP2-068 (CODE+ENTRY): Enable full message bus publishing after payload validation. Evidence gap: message bus supports Noop/Redis only and no payload validation gate exists (`ml/common/message_bus.py`, `ml/actors/ml_domain_events.py`).
- [FAILED] OP2-069 (ENTRY): Add live-paced replay with Redis after TestClock validation is stable. Evidence gap: no live-paced replay harness found under `ml/deployment/` or data replay adapters.

---

## Claims Ledger - ml/docs/ops/chronos_training_runbook.md

### Overview / Rationale
- [VERIFIED] RUN-001 (DOC): Chronos-2 is best accuracy (120M params) teacher model. Evidence: `ml/docs/ops/chronos_training_runbook.md`.
- [VERIFIED] RUN-002 (DOC): Chronos-Bolt is 250x faster inference (student model). Evidence: `ml/docs/ops/chronos_training_runbook.md`.
- [VERIFIED] RUN-003 (DOC): Forward return regression used for target. Evidence: `ml/docs/ops/chronos_training_runbook.md`.
- [VERIFIED] RUN-004 (DOC): Chronos supports native covariates (calendar, macro, technical indicators). Evidence: `ml/docs/ops/chronos_training_runbook.md`.

### Goal and Plan
- [VERIFIED] RUN-005 (DOC): Phase 0 tasks include monotonic timestamps, no lookahead, correct target shift. Evidence: `ml/docs/ops/chronos_training_runbook.md`.
- [VERIFIED] RUN-006 (DOC): Enforce feature manifest hygiene (numeric-only; exclude forward_return, y, meta fields). Evidence: `ml/docs/ops/chronos_training_runbook.md`.
- [VERIFIED] RUN-007 (TEST+ENTRY): Contract tests for dataset schema/coverage; run dataset validation gates. Evidence: `ml/tests/unit/data/test_dataset_validation.py`, `ml/docs/ops/chronos_training_runbook.md`.
- [VERIFIED] RUN-008 (CODE+ENTRY): Time-split evaluation harness (train/val/test by timestamp). Evidence: `ml/training/autogluon/chronos_evaluation.py`, `ml/experiments/chronos_training_experiment.py` (`--time_split_eval`).
- [VERIFIED] RUN-012 (CODE+ENTRY): Map predictions to signals with thresholds and risk limits. Evidence: `ml/strategies/ml_strategy.py` (threshold mapping + risk metrics), `ml/strategies/base_facade.py` (position sizing/risk checks).

### Immediate Next Tasks (Phase 1)
- [VERIFIED] RUN-016 (CODE+ENTRY): Implement time-split evaluation utilities in ml/training/autogluon. Evidence: `ml/training/autogluon/chronos_evaluation.py`, `ml/experiments/chronos_training_experiment.py`.

### Prerequisites / Env
- [VERIFIED] RUN-019 (DOC): Install AutoGluon TimeSeries dependencies (uv or poetry). Evidence: `ml/docs/ops/chronos_training_runbook.md`.
- [VERIFIED] RUN-020 (CONFIG): Set ML_TFT_ALLOW_PARQUET_FALLBACK=1 for parquet-only loading. Evidence: `ml/docs/ops/chronos_training_runbook.md`.
- [VERIFIED] RUN-021 (CONFIG): Optional GPU env (CUDA_VISIBLE_DEVICES) and LOG_FILE for logs. Evidence: `ml/docs/ops/chronos_training_runbook.md`.

### Quick Start / CLI
- [VERIFIED] RUN-023 (ENTRY): train_chronos CLI supports Chronos-2 teacher training. Evidence: `ml/cli/train_chronos.py` (`--preset chronos2`).
- [VERIFIED] RUN-024 (ENTRY): train_chronos CLI supports Chronos-Bolt training. Evidence: `ml/cli/train_chronos.py` (`--preset bolt_small`).
- [VERIFIED] RUN-025 (ENTRY): train_chronos supports HPO tuning. Evidence: `ml/cli/train_chronos.py` (`--tune-num-trials`, `--tune-searcher`, `--tune-scheduler`).

### Time-Split Evaluation
- [VERIFIED] RUN-026 (CONFIG+CODE): eval_min_series_rows enforces per-series coverage across splits. Evidence: `ml/config/autogluon.py` (min_rows_per_series_split), `ml/experiments/chronos_training_experiment.py` (`--eval_min_series_rows`).

### Teacher-Student Distillation / Fidelity Checklist
- [VERIFIED] RUN-042 (CODE+ENTRY): Distillation uses rolling forecasts to align teacher predictions to forecast-origin rows. Evidence: `ml/training/autogluon/soft_label_generator.py` (rolling windows + forecast_step).
- [VERIFIED] RUN-043 (CONFIG): Defaults assume prediction_length=15; rolling parameters adjustable. Evidence: `ml/config/autogluon.py` (prediction_length default 15; ChronosDistillationConfig).
- [VERIFIED] RUN-044 (CODE+ENTRY): Known covariates require future covariate values at prediction time. Evidence: `ml/training/autogluon/soft_label_generator.py` (`make_future_data_frame` + `_build_future_covariates`).
- [VERIFIED] RUN-045 (CONFIG+ENTRY): Use --tune-num-trials >=10 and --num-val-windows >=2; avoid --skip-model-selection. Evidence: `ml/experiments/chronos_training_experiment.py` (flags + config).
- [FAILED] RUN-046 (CONFIG+ENTRY): Ensure --fine-tune enabled for tuning runs. Evidence gap: `ml/experiments/chronos_training_experiment.py` parses `--fine_tune` but does not auto-enable or use it in training config.
- [VERIFIED] RUN-047 (CONFIG+ENTRY): Enable --refit-full for teacher training. Evidence: `ml/experiments/chronos_training_experiment.py` (`--refit_full` flag).

### Troubleshooting / Operational Notes
- [VERIFIED] RUN-050 (DOC): OOM mitigations include smaller preset, reduced time limit, CPU-only, subsample. Evidence: `ml/docs/ops/chronos_training_runbook.md`.
- [VERIFIED] RUN-051 (DOC): Dataset build fails if parquet fallback disabled or missing files. Evidence: `ml/docs/ops/chronos_training_runbook.md`.
- [VERIFIED] RUN-052 (DOC): AutoGluon import errors resolved by reinstalling with uv or verifying install. Evidence: `ml/docs/ops/chronos_training_runbook.md`.
- [VERIFIED] RUN-053 (DOC): Slow training mitigations: enable GPU, use bolt_small, reduce time limit. Evidence: `ml/docs/ops/chronos_training_runbook.md`.
- [VERIFIED] RUN-054 (DOC): Long runs should be foregrounded or use tmux; avoid tee unless handling stderr/SIGPIPE. Evidence: `ml/docs/ops/chronos_training_runbook.md`.

### Outputs / Registry Integration
- [VERIFIED] RUN-056 (CODE+ENTRY): Model registry integration available for trained models. Evidence: `ml/training/export.py`, `ml/training/distillation/cli.py`, `ml/training/common/persistence.py`.

---

## Verification Workflow (Order of Checks)
1) Doc claims -> map to expected evidence type(s) above.
2) CODE/CONFIG/ENTRY checks (static inspection).
3) TEST presence checks (and later, targeted runs if needed).
4) Record status + evidence paths inline for each claim.

## Audit Notes Template
Use this section to track open issues or contradictions found during verification.
- Gap:
- Evidence:
- Impact:
- Next action:
