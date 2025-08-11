# Teacher–Student Distillation for Live L1 Inference

This guide describes how to train a high‑quality teacher model (with T+1 access
to L1/L2/L3 market data), distill it into a fast LightGBM student (L1-only), ver
sion and validate the artifacts, and deploy the student for live inference via N
autilus Trader’s ML actor and strategy.

The approach ensures train↔serve parity, tight latency budgets (<5 ms p99), and
clean deployment/rollback via the model registries.

## Architecture Overview
- Teacher (offline): Temporal Fusion Transformer (TFT) or other heavy model trained with rich features (L2/L3 allowed). Outputs calibrated probabilities on a st
udent window.
- Student (online): LightGBM distilled on L1-only features to mimic teacher prob
abilities; exported to ONNX with sigmoid (+ optional Platt) baked in for low‑lat
ency inference.
- Registries:
  - Distillation Registry: Training‑side versioning, lineage (teacher→student),
acceptance metrics, purging.
  - Local Model Registry: Production manifest and auto‑deployment gating (L1-only, latency, lineage).
- Inference: `ONNXMLInferenceActor` loads the ONNX model, validates feature sche
ma, computes L1 features online, infers probability, and publishes `MLSignal`.
- Execution: `MLTradingStrategy` consumes signals (prediction=probability, confi
dence), applies thresholds and risk rules, and places orders.

## Data and Feature Parity
- Offline features (teacher): May include L2/L3 microstructure and order‑flow; u
sed to generate high‑quality targets.
- Online features (student): Must be strictly L1‑derivable; defined by `FeatureC
onfig` and computed by `FeatureEngineer` with identical math between batch and o
nline modes.
- Parity tools:
  - `FeatureParityValidator`: Confirms batch vs online computed features match w
ithin tolerance.
  - Student sidecar metadata must include ordered `feature_names` and a `feature
_schema_hash` to assert online parity at load time.

## End‑to‑End Flow
1) Train Teacher (offline)
2) Calibrate Teacher outputs (Platt/Isotonic)
3) Distill Student on L1‑only features using soft labels
4) Export Student to ONNX (+ metadata + acceptance test)
5) Register Teacher and Student (versioned) in Distillation Registry
6) Validate Student acceptance; if pass → create production `ModelManifest`
7) Register & (optionally) auto‑deploy via Local Model Registry
8) Actor loads active model and serves signals; strategy trades

## Pseudocode — Teacher Training + Calibration
```python
# Build rich dataset (omitted): L1/L2/L3 features, target.
from TFT_teacher_model import TFTTeacher, TFTTeacherConfig

# 1) Train TFT teacher (simplified)
teacher = TFTTeacher(TFTTeacherConfig(max_epochs=E, hidden_size=H, ...))
teacher.fit(time_series_dataset)

# 2) Calibrate on student window using true labels available offline
p_raw_val = tft_infer_probs(student_window)  # raw probs from TFT
teacher.calibrate(p_raw_val, y_val_true)     # Platt or Isotonic
q_val = teacher.predict_proba(p_raw_val)     # calibrated probabilities

# 3) Persist teacher preds for distillation
np.savez('teacher_preds.npz', q_train=q_val, y_val_true=y_val_true)
```

## Pseudocode — Student Distillation + Export
```python
from lightgbm_student_model import LightGBMStudentDistiller

# 1) Load L1‑only features (ordered) and teacher probabilities
X_train, X_val, feature_names = load_features()
T = np.load('teacher_preds.npz')
q_train = T['q_train']
y_val_true = T.get('y_val_true')  # optional for post‑calibration

# 2) Train student with soft labels (choose objective)
distiller = LightGBMStudentDistiller(objective='hybrid', kd_lambda=0.5, early_st
opping=200)
distiller.fit(X_train, q_train, X_val, y_val_true)

# 3) Export ONNX with sigmoid (+ Platt if present) and sidecar metadata
onnx_path, meta_path = distiller.export_onnx(
    feature_names=feature_names,
    out_dir='distilled_model',
    model_id='es_l1_student_v1',
    flags={'distilled_from': 'TFT'},
)

# 4) Optional acceptance test (framework vs ORT parity)
acc = acceptance_test(onnx_path, X_val, distiller.predict_proba(X_val))
assert acc['skipped'] or acc['pass']
```

## Pseudocode — Distillation Registry (Training‑Side)
```python
from ml.registry.model_registry import DistillationRegistry, ModelVersion

reg = DistillationRegistry(base_path=Path('registry_store'))

# Register teacher
teacher_ver = ModelVersion(model_type='teacher', architecture='TFT')
reg.register_teacher(
    model=teacher_obj_or_artifact,
    version=teacher_ver,
    training_data={'range': '2024-01..2024-06'},
    performance_metrics={'val_auc': 0.74},
)

# Register student
student_ver = ModelVersion(model_type='student', architecture='ONNX', parent_id=
teacher_ver.version_id)
reg.register_student(
    model=open(onnx_path, 'rb').read(),
    version=student_ver,
    teacher_id=teacher_ver.version_id,
    distillation_metrics={
        'feature_parity_error': 1e-12,
        'accuracy_loss': 0.01,
        'p99_latency_ms': 2.7,
    },
)

# Validate student against acceptance criteria
ok = reg.validate_student(
    student_id=student_ver.version_id,
    acceptance_criteria={'max_feature_error': 1e-10, 'max_accuracy_loss': 0.05,
'max_latency_ms': 5.0},
)
assert ok
```

## Pseudocode — Production Manifest + Local Deployment
```python
from ml.registry.base import ModelManifest, ModelRole, DataRequirements
from ml.registry.local_registry import LocalModelRegistry

# Build manifest from sidecar metadata
meta = json.load(open('distilled_model/student.meta.json'))
feature_schema = {name: 'float32' for name in meta['feature_names']}

manifest = ModelManifest(
    model_id=meta['model_id'],
    role=ModelRole.STUDENT,
    data_requirements=DataRequirements.L1_ONLY,
    architecture='LightGBM+ONNX',
    feature_schema=feature_schema,
    feature_schema_hash=meta['feature_schema_hash'],
    parent_id=teacher_ver.version_id,
    training_config={'opset': meta['opset']},
    performance_metrics={'p99_latency_ms': 2.7},
    deployment_constraints={'max_latency_ms': 5.0},
)

local = LocalModelRegistry(Path('prod_registry'))
model_id = local.register_model(Path(onnx_path), manifest, auto_deploy=True)
print('Deployed model:', model_id)
```

## Pseudocode — Actor Inference (ONNXMLInferenceActor)
```python
# On start:
model, metadata = model_loader.load_model(config.model_path)      # ONNX + merge
d sidecar
expected_names = metadata.get('feature_names', [])
engineer = FeatureEngineer(feature_config)
assert engineer.get_feature_names() == expected_names, 'Feature order mismatch'

# On each bar (hot path):
features = engineer.calculate_features_online(current_bar, indicator_mgr, scaler
=None)  # float32
p = onnx_session.run(output_names, {input_name: features.reshape(1, -1)})[0][0]
confidence = max(p, 1.0 - p)
publish(MLSignal(instrument_id, model_id, prediction=p, confidence=confidence))
```

## Pseudocode — Strategy Consumption (MLTradingStrategy)
```python
# on_data(MLSignal):
if signal.confidence >= cfg.min_confidence:
    side = BUY if signal.prediction > cfg.prediction_threshold else SELL
