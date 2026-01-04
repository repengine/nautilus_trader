# Training Status - Investigation Notes

Scope: reviewed `ml/training/`, `ml/pipelines/`, `ml/consumers/`, plus the related
orchestrator glue and streaming config to map training capabilities and gaps. All
citations point to the exact locations for follow-up work.

## Training Surfaces (Capabilities)

### Base trainer framework (tabular/classic models)
- `BaseMLTrainer` orchestrates data prep, optional Optuna HPO, CV, evaluation, MLflow
  tracking, and ONNX export; it also supports FeatureStore parity when a DB connection
  is configured. [`ml/training/base.py:50`](ml/training/base.py:50)
- MLflow tracking exists but is explicitly noted as deprecated in favor of the
  ModelRegistry. [`ml/training/common/mlflow_tracking.py:1`](ml/training/common/mlflow_tracking.py:1)
- Export utilities provide model type detection, ONNX conversion, and a metadata
  sidecar. [`ml/training/export.py:3`](ml/training/export.py:3)

### Non-distilled trainers
- LightGBM trainer supports polars ingestion, categorical handling, GPU config,
  GOSS/DART/EFB options, early stopping, and feature importance. 
  [`ml/training/non_distilled/lightgbm.py:30`](ml/training/non_distilled/lightgbm.py:30)
- XGBoost trainer supports polars ingestion, GPU acceleration, and monotonic constraints. 
  [`ml/training/non_distilled/xgboost.py:80`](ml/training/non_distilled/xgboost.py:80)

### TFT teacher training (offline)
- `tft_cli` loads CSV/Parquet into pandas, enforces feature schema, splits train/val,
  and trains a TFT teacher; it can fall back to logistic regression if TFT fails. 
  [`ml/training/teacher/tft_cli.py:444`](ml/training/teacher/tft_cli.py:444), 
  [`ml/training/teacher/tft_cli.py:694`](ml/training/teacher/tft_cli.py:694)
- `TFTTeacher` uses PyTorch Forecasting + Lightning, can warm-start from a pretrained
  state dict via a safe loader, and builds validation using the full sorted dataset. 
  [`ml/training/teacher/tft_teacher.py:168`](ml/training/teacher/tft_teacher.py:168), 
  [`ml/training/teacher/tft_teacher.py:335`](ml/training/teacher/tft_teacher.py:335)
- The compatibility CLI `ml/training/teacher/cli.py` forwards to `tft_cli` when
  registry args are present, or runs legacy calibration-only mode otherwise. 
  [`ml/training/teacher/cli.py:7`](ml/training/teacher/cli.py:7)
- There is a placeholder `tft_model.py` (scaffold) distinct from the real implementation
  in `tft_teacher.py`. [`ml/training/teacher/tft_model.py:21`](ml/training/teacher/tft_model.py:21)
- A small masked-time-modeling (MTM) pretrainer exists for optional warm starts. 
  [`ml/training/teacher/pretrain_mtm.py:1`](ml/training/teacher/pretrain_mtm.py:1)

### Distillation (teacher -> student)
- LightGBM student distiller trains on soft labels, optionally calibrates, and exports
  ONNX with metadata. [`ml/training/student/lightgbm.py:1`](ml/training/student/lightgbm.py:1)
- The distillation CLI enforces feature registry parity and registers the student in
  the local registry. [`ml/training/distillation/cli.py:52`](ml/training/distillation/cli.py:52)

### Event-driven streaming pipeline
- Streaming loader does low-memory metadata scans and shard replay to avoid full dataset
  materialization. [`ml/training/teacher/streaming_loader.py:1`](ml/training/teacher/streaming_loader.py:1)
- Dataset planner scans parquet metadata, applies caps, and enforces guardrails. 
  [`ml/training/event_driven/dataset_service.py:91`](ml/training/event_driven/dataset_service.py:91), 
  [`ml/training/event_driven/guardrails/dataset.py:30`](ml/training/event_driven/guardrails/dataset.py:30)
- Streaming worker computes validation metrics and calibration diagnostics and persists
  logits. [`ml/training/event_driven/worker.py:2039`](ml/training/event_driven/worker.py:2039), 
  [`ml/training/event_driven/worker.py:2009`](ml/training/event_driven/worker.py:2009)
- Streaming sweeps use Optuna to explore worker hyperparameters. 
  [`ml/training/event_driven/sweep.py:1`](ml/training/event_driven/sweep.py:1)

## Pipelines / Entry Points
- `tft_train_distill` is a wrapper around the orchestrator that wires dataset build,
  teacher training, and student distillation in one CLI flow. 
  [`ml/pipelines/tft_train_distill.py:1`](ml/pipelines/tft_train_distill.py:1)
- `build_runner` orchestrates multi-symbol dataset builds with optional parallelism
  and progress logging; it calls the dataset build main by default to avoid subprocess
  overhead. [`ml/pipelines/build_runner.py:1`](ml/pipelines/build_runner.py:1)

## Consumers / Observability
- Streaming training state store collects plan/result/heartbeat records and updates
  gauges for backlog/progress/metrics. [`ml/consumers/streaming_training.py:1`](ml/consumers/streaming_training.py:1)
- Persistence service wires streaming training events to Redis Streams consumption. 
  [`ml/consumers/streaming_training_service.py:1`](ml/consumers/streaming_training_service.py:1)
- Redis Streams consumer provides the basic gate + handler loop (example-level). 
  [`ml/consumers/redis_streams_consumer.py:1`](ml/consumers/redis_streams_consumer.py:1)
- Aggregator, retry, idempotent, and lineage consumers provide buffering, DLQ retry,
  watermark gating, and lineage persistence. 
  [`ml/consumers/aggregator.py:1`](ml/consumers/aggregator.py:1), 
  [`ml/consumers/retry.py:1`](ml/consumers/retry.py:1), 
  [`ml/consumers/idempotent.py:1`](ml/consumers/idempotent.py:1), 
  [`ml/consumers/lineage_writer.py:1`](ml/consumers/lineage_writer.py:1)

## Key Gaps / Risks (Impacting CPU/DRAM/GPU Constraints)

### Orchestrator wiring limitations
- `TeacherTrainConfig` only exposes `max_epochs` (no batch size, workers, accelerator,
  precision, or model-size knobs), so the pipeline cannot tune training to improved
  CPU/DRAM without code changes. [`ml/orchestration/config_types.py:245`](ml/orchestration/config_types.py:245)
- Orchestrator calls the teacher CLI with `--train_data_csv` only, even though parquet
  is built alongside the dataset. This forces pandas CSV reads in large runs. 
  [`ml/orchestration/pipeline_orchestrator.py:2692`](ml/orchestration/pipeline_orchestrator.py:2692)

### Offline TFT training memory profile
- `tft_cli` reads the full dataset into pandas and sorts/slices for train/val, which
  is memory heavy on large datasets. [`ml/training/teacher/tft_cli.py:444`](ml/training/teacher/tft_cli.py:444)
- `TFTTeacher.fit` makes additional copies (sorted/full validation dataset), further
  increasing memory pressure. [`ml/training/teacher/tft_teacher.py:200`](ml/training/teacher/tft_teacher.py:200)
- If TFT dependencies fail, the fallback is logistic regression; while robust, it
  changes model class and metric characteristics. [`ml/training/teacher/tft_cli.py:694`](ml/training/teacher/tft_cli.py:694)

### Streaming training is effectively a stub
- `TFTTeacher.fit_streaming` is a placeholder returning empty arrays, so streaming
  training produces no real logits. [`ml/training/teacher/tft_teacher.py:564`](ml/training/teacher/tft_teacher.py:564)
- The streaming worker calls `fit_streaming` directly, so the placeholder path is
  currently used end-to-end. [`ml/training/event_driven/worker.py:1017`](ml/training/event_driven/worker.py:1017)

### Streaming worker config vs. implementation drift
- `StreamingWorkerConfig` exposes many tuning knobs (accelerator, precision, model
  sizes, AMP, calibration) and caps. [`ml/config/streaming_pipeline.py:581`](ml/config/streaming_pipeline.py:581)
- The worker passes optimizer/lr_scheduler/precision into `TFTTeacher`, but
  `TFTTeacher` stores these fields without using them for actual optimizer/scheduler
  setup. [`ml/training/event_driven/worker.py:1485`](ml/training/event_driven/worker.py:1485),
  [`ml/training/teacher/tft_teacher.py:160`](ml/training/teacher/tft_teacher.py:160)
- Streaming worker persists logits artifacts only; no model artifact export or
  registry integration is present in this path. [`ml/training/event_driven/worker.py:2009`](ml/training/event_driven/worker.py:2009)

### HPO / resource isolation
- The HPO CLI supports subprocess isolation, but the orchestrator path does not
  expose that flag, so memory can accumulate across trials. 
  [`ml/cli/hpo_tft.py:186`](ml/cli/hpo_tft.py:186), 
  [`ml/orchestration/pipeline_orchestrator.py:2621`](ml/orchestration/pipeline_orchestrator.py:2621)

### Multiple TFT surfaces cause ambiguity
- There is a full TFT implementation (`tft_teacher.py`) and a placeholder TFT model
  scaffold (`tft_model.py`), which can confuse downstream imports and expectations. 
  [`ml/training/teacher/tft_teacher.py:168`](ml/training/teacher/tft_teacher.py:168), 
  [`ml/training/teacher/tft_model.py:21`](ml/training/teacher/tft_model.py:21)

## Recommended Next Steps (Prioritized)

1) **Expose training knobs in the orchestrator config.**  
   Extend `TeacherTrainConfig` to include batch size, workers, accelerator, devices,
   precision, hidden size, etc., and wire them into `train_teacher` so CPU/DRAM
   improvements are usable from pipelines. 
   [`ml/orchestration/config_types.py:245`](ml/orchestration/config_types.py:245), 
   [`ml/orchestration/pipeline_orchestrator.py:2663`](ml/orchestration/pipeline_orchestrator.py:2663), 
   [`ml/training/teacher/tft_cli.py:260`](ml/training/teacher/tft_cli.py:260)

2) **Prefer parquet for offline TFT training.**  
   Update the orchestrator (and CLI wiring) to pass `--train_data_parquet` when
   available to reduce CSV memory load and speed up IO. 
   [`ml/orchestration/pipeline_orchestrator.py:2692`](ml/orchestration/pipeline_orchestrator.py:2692), 
   [`ml/training/teacher/tft_cli.py:444`](ml/training/teacher/tft_cli.py:444)

3) **Implement real streaming training or disable the stub.**  
   Either implement `TFTTeacher.fit_streaming` using the streaming loader, or gate
   the streaming pipeline until a real trainer exists; the current placeholder
   yields empty logits. 
   [`ml/training/teacher/tft_teacher.py:564`](ml/training/teacher/tft_teacher.py:564), 
   [`ml/training/event_driven/worker.py:1017`](ml/training/event_driven/worker.py:1017)

4) **Align streaming worker config with actual trainer behavior.**  
   Use the optimizer/lr_scheduler fields from `StreamingWorkerConfig` to build
   real Lightning optimizers and schedulers inside `TFTTeacher` (or the worker), so
   config changes have effect. 
   [`ml/config/streaming_pipeline.py:581`](ml/config/streaming_pipeline.py:581), 
   [`ml/training/event_driven/worker.py:1485`](ml/training/event_driven/worker.py:1485)

5) **Add model artifact export/registration to streaming runs.**  
   Decide how streaming fits into registry lifecycles, and persist a model
   artifact (or a distilled student) in addition to logits. 
   [`ml/training/event_driven/worker.py:2009`](ml/training/event_driven/worker.py:2009), 
   [`ml/training/distillation/cli.py:52`](ml/training/distillation/cli.py:52)

6) **Expose HPO subprocess mode in orchestrator config.**  
   Add a flag to pass `--subprocess` (and timeout) into HPO runs to reduce memory
   accumulation in large sweeps. 
   [`ml/cli/hpo_tft.py:186`](ml/cli/hpo_tft.py:186), 
   [`ml/orchestration/pipeline_orchestrator.py:2621`](ml/orchestration/pipeline_orchestrator.py:2621)

7) **Clarify TFT implementation paths.**  
   Either remove or clearly deprecate `tft_model.py` to avoid confusion with the
   real `tft_teacher.py`. 
   [`ml/training/teacher/tft_model.py:21`](ml/training/teacher/tft_model.py:21)

8) **Document baseline resource guidance (CPU/DRAM/GPU).**  
   Capture recommended defaults (batch size, tail_rows, limit_groups, precision)
   in a config doc or README so operators can tune without code changes. 
   [`ml/training/teacher/tft_cli.py:318`](ml/training/teacher/tft_cli.py:318), 
   [`ml/config/streaming_pipeline.py:581`](ml/config/streaming_pipeline.py:581)

