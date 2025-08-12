# ML Training Layer

Cold‑path (offline) model training that produces self‑contained artifacts for the
hot path to load. Keep all training concerns here — never mix runtime inference logic.

This layer targets:

- Reproducible training and evaluation on Nautilus data.
- Deterministic, production‑ready exports (ONNX/native) with sidecar metadata.
- Teacher→Student distillation flows for sub‑ms inference.

## Directory Structure

- `base.py`: Abstract trainer orchestration.
  - Provides `BaseMLTrainer` with hooks for data prep, CV, Optuna HPO, MLflow tracking,
    evaluation, and model export integration.

- `export.py`: Unified export utilities and contracts.
  - Free funcs: `detect_model_type`, `save_model_with_metadata`, `convert_to_onnx`.
  - Contracts: `ModelExportMixin`, `TrainingActorContract`.
  - Constant: `DEFAULT_ONNX_OPSET` (default ONNX opset used by exporters).
  - Writes small JSON sidecars with technical metadata (not the registry manifest).

- `model_exporter.py`: Compatibility shim re‑exporting from `export.py`.
  - Keep imports stable if downstreams reference `ml.training.model_exporter`.

- `lightgbm.py`: Compatibility shim re‑exporting non‑distilled LightGBM trainer.

- `xgboost.py` (if present): XGBoost trainer. May be relocated to `non_distilled/`.
  - If moved, keep a compatibility shim mirroring the `lightgbm.py` approach.

- `non_distilled/`
  - `lightgbm.py`: Non‑distilled LightGBM trainer (cold‑path classic).
    - Integrates early stopping, GPU/GOSS/DART/EFB options, feature importance,
      and ONNX conversion via `onnxmltools`.
  - `xgboost.py` (preferred location): Non‑distilled XGBoost trainer.
    - GPU, monotonic constraints, SHAP, Optuna HPO, ONNX conversion with f0..fn mapping.

- `optuna_optimizer.py`: Optuna wrapper for XGBoost HPO.
  - Study creation with samplers/pruners, objective factory, stats summaries.

- `student/`
  - `lightgbm.py`: Student distiller trained on teacher soft labels (production‑oriented).
    - Objectives: `logit_mse`, `soft_ce`, `hybrid` (custom grad for hybrid).
    - Optional Platt calibration on raw scores; baked into ONNX with Sigmoid and linear layer.
    - Emits strict metadata (`feature_names`, `feature_schema_hash`, `best_iteration`, opset, etc.).
  - `lightgbm_student.py`: Thin shim re‑exporting `LightGBMStudentDistiller` and `schema_hash` for
    canonical import path (`ml.training.student.lightgbm`).
  - `lightgbm_cli.py`: CLI to train, export, and register a student in the local registry.

- `teacher/`
  - `base.py`: Base teacher interface + Platt calibration helper; TFT placeholder class.
  - `tft_cli.py`: CLI scaffold to calibrate a TFT teacher and emit soft labels for distillation.

- `__init__.py`: Public API re‑exports for trainers (non‑distilled LightGBM/XGBoost, base).

## Integrations

- Console scripts (pyproject):
  - `ml-teacher-tft` → `ml.training.teacher.tft_cli:main`
  - `ml-student-lightgbm` → `ml.training.student.lightgbm_cli:main`

- Registry integration (student CLI):
  - Uses `ml.registry.model_registry.LocalModelRegistry`.
  - Uses `ml.registry.utils.build_feature_schema` and `build_student_manifest`.
  - Students should be registered with role `STUDENT`, `data_requirements=L1_ONLY`, and `parent_id=<teacher_id>`.

- Inference compatibility:
  - `ModelExportMixin.save_for_production` selects format (`onnx`/native) and writes sidecar metadata.
  - `ModelExportMixin.validate_inference_compatibility` smoke‑tests ONNX via onnxruntime.
  - `TrainingActorContract` defines the minimal handoff for actor consumers.

## Flows

- Non‑distilled training (LightGBM/XGBoost):
  1) Prepare data in `prepare_data` (Polars→NumPy),
  2) Optional Optuna HPO,
  3) Train with early stopping/CV,
  4) Evaluate and compute metrics,
  5) Export via `ModelExportMixin` (`onnx`/native + metadata).

- Teacher→Student distillation:
  1) Teacher fits/calibrates on validation slice to produce soft labels (probabilities/logits),
  2) Student (LightGBM) trains on teacher soft labels with selected objective,
  3) Optional Platt calibration on student raw scores (against true labels),
  4) Export student ONNX with Sigmoid and optional linear calibration baked in,
  5) Register artifact and manifest in local registry.

## Dependency Policy

- Heavy deps are guarded via `ml._imports` feature flags (`HAS_*`) and local imports.
- Cold path may use pandas/polars/float64 and heavy frameworks.
- Hot path loads exported artifacts with minimal dependencies (onnxruntime, NumPy float32).

## Export Formats

- ONNX (`.onnx`): Default cross‑platform format; opset defaults to `DEFAULT_ONNX_OPSET` (currently 17).
- Native:
  - LightGBM (`.lgb`) — supports both sklearn wrapper and raw Booster.
  - XGBoost (`.xgb`/`.json`).
- Sidecars:
  - Exporters write a small technical metadata JSON next to the artifact
    (size, modified time, input/output hints, training metadata snapshot).
  - Deployment identity and lineage live in the Model Registry manifest, not in this sidecar.

## Testing Guidance

- Add acceptance tests comparing framework predictions vs ONNX Runtime on a small
  validation slice (float32 parity within a tight tolerance).
- Use property tests (Hypothesis) for shapes/dtypes/monotonicity when appropriate.
- Keep student CLI deterministic (seeded, fixed folds) for reproducible artifacts.

## Naming & Migrations

- Prefer explicit names to avoid import clashes (e.g., `lightgbm_student.py` for students).
- If relocating trainers (e.g., `xgboost` → `non_distilled/`), keep a compatibility shim at the
  old path until all imports are migrated.

## Examples

- Train and export a non‑distilled LightGBM model (pseudo‑code):
  - Build a subclass of `BaseMLTrainer` or use `non_distilled/lightgbm.LightGBMTrainer`.
  - Call `trainer.train(data)` then `trainer.save_for_production("/path/model")`.

- Calibrate a teacher and emit soft labels:
  - `ml-teacher-tft --student_window_npz /tmp/val_slice.npz --out_dir /tmp/teacher --model_id tft_v1`

- Distill a student and register it:
  - `ml-student-lightgbm --features_npz /tmp/features.npz --teacher_npz /tmp/teacher/preds.npz \
     --out_dir /tmp/student --model_id student_lgbm_v1 --parent_id tft_v1 --registry_dir /models`
