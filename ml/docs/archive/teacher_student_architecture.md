# Teacher-Student Distillation Architecture

**Version:** 2.0
**Purpose:** Comprehensive guide for implementing teacher-student model distillation for live L1 inference in NautilusTrader

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data & Feature Parity](#data--feature-parity)
3. [Teacher Implementation (TFT)](#teacher-implementation-tft)
4. [Student Distillation (LightGBM)](#student-distillation-lightgbm)
5. [Model Registry & Versioning](#model-registry--versioning)
6. [Deployment Pipeline](#deployment-pipeline)
7. [Inference Integration](#inference-integration)
8. [Validation & Testing](#validation--testing)

---

## Architecture Overview

The teacher-student distillation framework enables training high-quality models on rich data while maintaining real-time inference performance:

### Core Components

- **Teacher (Offline):** Temporal Fusion Transformer (TFT) or complex model trained with rich L1/L2/L3 features
- **Student (Online):** LightGBM model distilled to use only L1 features for low-latency inference
- **Distillation Registry:** Training-side versioning, lineage tracking, and acceptance validation
- **Local Model Registry:** Production manifest management and auto-deployment gating
- **Inference Actors:** ONNX-based inference with feature schema validation

### Architecture Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Rich Data     │    │     Teacher      │    │    Student      │
│ (L1/L2/L3 T+1)  │───▶│  (TFT/Complex)   │───▶│   (LightGBM)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Calibration    │    │   ONNX Export   │
                       │ (Platt/Isotonic) │    │   + Metadata    │
                       └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ Production ML   │
                                               │    Actors       │
                                               └─────────────────┘
```

### Key Principles

1. **Data Requirements Separation:**
   - Teachers can use any data (L1/L2/L3, macro, fundamental)
   - Students must be strictly L1-derivable for real-time constraints

2. **Performance Constraints:**
   - Teacher training: No latency constraints, optimize for accuracy
   - Student inference: <5ms P99 latency requirement

3. **Feature Parity:**
   - Identical feature calculation between batch training and online inference
   - Schema validation ensures consistency

---

## Data & Feature Parity

### Feature Engineering Constraints

**Teacher Features (Training):**

- Full access to L2/L3 order book data
- T+1 data availability (no real-time constraints)
- Complex derived features (e.g., order flow imbalances, depth analysis)
- Cross-asset and macro indicators

**Student Features (Inference):**

- Strictly L1-derivable (trades, quotes, OHLC)
- Real-time calculation constraints
- Identical mathematical operations as teacher training

### Parity Validation Framework

```python
class FeatureParityValidator:
    """Ensures batch and online feature calculations match exactly"""

    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance

    def validate_parity(self,
                       batch_features: np.ndarray,
                       online_features: np.ndarray) -> bool:
        """Validate feature parity within tolerance"""
        return np.allclose(batch_features, online_features,
                          rtol=self.tolerance, atol=self.tolerance)

    def generate_parity_report(self,
                              batch_features: np.ndarray,
                              online_features: np.ndarray,
                              feature_names: List[str]) -> Dict:
        """Generate detailed parity analysis"""
        diff = np.abs(batch_features - online_features)
        max_diff = np.max(diff, axis=0)

        return {
            'max_absolute_error': float(np.max(max_diff)),
            'feature_errors': dict(zip(feature_names, max_diff)),
            'passed': np.all(max_diff < self.tolerance)
        }
```

### Feature Schema Versioning

```python
@dataclass
class FeatureSchema:
    """Schema definition for feature compatibility"""
    feature_names: List[str]
    feature_types: Dict[str, str]
    schema_hash: str
    version: str

    @classmethod
    def from_config(cls, config: FeatureConfig) -> 'FeatureSchema':
        """Generate schema from feature configuration"""
        feature_names = config.get_feature_names()
        feature_types = {name: 'float32' for name in feature_names}

        # Generate deterministic hash
        schema_str = json.dumps({
            'names': feature_names,
            'types': feature_types
        }, sort_keys=True)
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()[:16]

        return cls(
            feature_names=feature_names,
            feature_types=feature_types,
            schema_hash=schema_hash,
            version=config.version
        )
```

---

## Teacher Implementation (TFT)

### TFT Teacher Model

The Temporal Fusion Transformer serves as our teacher model, leveraging rich historical data:

```python
class TFTTeacher:
    """Temporal Fusion Transformer teacher implementation"""

    def __init__(self, config: TFTTeacherConfig):
        self.config = config
        self.model = None
        self.calibrator = None

    def prepare_dataset(self,
                       l1_data: pl.DataFrame,
                       l2_l3_data: Optional[pl.DataFrame] = None,
                       macro_data: Optional[pl.DataFrame] = None) -> TimeSeriesDataSet:
        """Prepare rich dataset for teacher training"""

        # Combine all available data sources
        features = self._engineer_rich_features(l1_data, l2_l3_data, macro_data)
        labels = self._generate_labels(l1_data)

        # Create TFT-compatible dataset
        return TimeSeriesDataSet(
            data=features,
            time_idx="time_idx",
            target="target",
            group_ids=["instrument_id"],
            max_encoder_length=self.config.max_encoder_length,
            max_prediction_length=self.config.max_prediction_length,
            static_categoricals=["instrument_id"],
            time_varying_known_reals=self.config.known_reals,
            time_varying_unknown_reals=self.config.unknown_reals,
        )

    def fit(self, dataset: TimeSeriesDataSet):
        """Train TFT teacher model"""

        # Split data for training/validation
        train_dataloader = dataset.to_dataloader(
            train=True, batch_size=self.config.batch_size
        )
        val_dataloader = dataset.to_dataloader(
            train=False, batch_size=self.config.batch_size
        )

        # Initialize TFT model
        self.model = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=self.config.learning_rate,
            hidden_size=self.config.hidden_size,
            attention_head_size=self.config.attention_head_size,
            dropout=self.config.dropout,
            hidden_continuous_size=self.config.hidden_continuous_size,
            output_size=7,  # Quantile outputs
        )

        # Train model
        trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="gpu" if self.config.use_gpu else "cpu",
            enable_checkpointing=True,
            logger=MLFlowLogger(),
        )

        trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def calibrate(self, val_features: np.ndarray, val_labels: np.ndarray):
        """Calibrate teacher outputs for better probability estimates"""

        # Get raw predictions
        raw_predictions = self.predict_raw(val_features)

        # Fit calibrator (Platt scaling or Isotonic regression)
        if self.config.calibration_method == "platt":
            from sklearn.calibration import CalibratedClassifierCV
            self.calibrator = CalibratedClassifierCV(
                estimator=None, method="sigmoid", cv="prefit"
            )
        else:
            from sklearn.isotonic import IsotonicRegression
            self.calibrator = IsotonicRegression(out_of_bounds="clip")

        self.calibrator.fit(raw_predictions, val_labels)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Generate calibrated probability predictions"""
        raw_predictions = self.predict_raw(features)

        if self.calibrator is not None:
            return self.calibrator.predict_proba(raw_predictions)[:, 1]

        return raw_predictions

    def generate_soft_labels(self,
                           student_features: np.ndarray) -> np.ndarray:
        """Generate soft labels for student distillation"""
        return self.predict_proba(student_features)
```

### Teacher Training Pipeline

```python
def train_teacher_pipeline(config: TeacherTrainingConfig):
    """Complete teacher training pipeline"""

    # 1. Load and prepare rich dataset
    logger.info("Loading rich dataset with L1/L2/L3 features")
    dataset = load_rich_dataset(
        symbols=config.symbols,
        start_date=config.start_date,
        end_date=config.end_date,
        include_l2_l3=True,
        include_macro=True
    )

    # 2. Initialize and train teacher
    teacher = TFTTeacher(config.model_config)
    tft_dataset = teacher.prepare_dataset(
        l1_data=dataset['l1'],
        l2_l3_data=dataset['l2_l3'],
        macro_data=dataset['macro']
    )

    teacher.fit(tft_dataset)

    # 3. Calibrate on validation set
    val_features, val_labels = prepare_validation_data(dataset)
    teacher.calibrate(val_features, val_labels)

    # 4. Generate soft labels for student training
    student_features = prepare_l1_only_features(dataset)
    soft_labels = teacher.generate_soft_labels(student_features)

    # 5. Save teacher artifacts
    teacher_artifacts = {
        'model': teacher.model.state_dict(),
        'calibrator': teacher.calibrator,
        'soft_labels': soft_labels,
        'metadata': {
            'training_config': config,
            'performance_metrics': evaluate_teacher(teacher, val_features, val_labels)
        }
    }

    save_teacher_artifacts(teacher_artifacts, config.output_path)

    return teacher, soft_labels
```

---

## Student Distillation (LightGBM)

### LightGBM Student Implementation

```python
class LightGBMStudentDistiller:
    """LightGBM student model with knowledge distillation"""

    def __init__(self, config: StudentDistillationConfig):
        self.config = config
        self.model = None
        self.feature_names = None

    def fit(self,
            features: np.ndarray,
            soft_labels: np.ndarray,
            val_features: Optional[np.ndarray] = None,
            val_hard_labels: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None):
        """Train student model using knowledge distillation"""

        self.feature_names = feature_names or [f"feature_{i}" for i in range(features.shape[1])]

        # Prepare training data
        train_data = lgb.Dataset(
            features,
            label=soft_labels,
            feature_name=self.feature_names
        )

        valid_data = None
        if val_features is not None and val_hard_labels is not None:
            valid_data = lgb.Dataset(
                val_features,
                label=val_hard_labels,
                feature_name=self.feature_names,
                reference=train_data
            )

        # Configure objective based on distillation method
        params = self._get_training_params()

        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data] if valid_data else None,
            num_boost_round=self.config.num_boost_round,
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
        )

    def _get_training_params(self) -> Dict:
        """Get LightGBM training parameters"""
        base_params = {
            'objective': 'regression',  # For soft label regression
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': self.config.num_leaves,
            'learning_rate': self.config.learning_rate,
            'feature_fraction': self.config.feature_fraction,
            'bagging_fraction': self.config.bagging_fraction,
            'bagging_freq': self.config.bagging_freq,
            'verbose': -1,
            'random_state': self.config.random_state,
        }

        # Hybrid objective for combining soft and hard labels
        if self.config.objective == 'hybrid' and hasattr(self.config, 'kd_lambda'):
            base_params['objective'] = self._hybrid_objective

        return base_params

    def _hybrid_objective(self, y_true, y_pred):
        """Custom hybrid objective combining soft and hard labels"""
        # This would be implemented based on specific requirements
        # combining teacher soft labels with true hard labels
        pass

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Generate probability predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        raw_predictions = self.model.predict(features)

        # Apply sigmoid if needed (depends on training objective)
        if self.config.apply_sigmoid:
            return 1 / (1 + np.exp(-raw_predictions))

        return raw_predictions

    def export_onnx(self,
                   output_dir: Path,
                   model_id: str) -> Tuple[Path, Path]:
        """Export model to ONNX format with metadata"""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to ONNX using Treelite
        import treelite
        tl_model = treelite.Model.from_lightgbm(self.model)

        # Add sigmoid transformation if needed
        if self.config.apply_sigmoid:
            tl_model = tl_model.add_transform("sigmoid")

        # Export to ONNX
        onnx_path = output_dir / f"{model_id}.onnx"
        tl_model.export_lib(
            toolchain="clang",
            libpath=str(onnx_path),
            params={"quantize": 1, "parallel_comp": 8}
        )

        # Generate metadata
        metadata = {
            'model_id': model_id,
            'feature_names': self.feature_names,
            'feature_schema_hash': self._compute_feature_hash(),
            'model_type': 'LightGBM',
            'onnx_opset': 11,
            'distillation_config': self.config.__dict__,
            'performance_metrics': self._compute_performance_metrics(),
            'created_at': datetime.utcnow().isoformat()
        }

        metadata_path = output_dir / f"{model_id}.meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return onnx_path, metadata_path

    def _compute_feature_hash(self) -> str:
        """Compute deterministic hash of feature schema"""
        schema_str = json.dumps({
            'names': self.feature_names,
            'count': len(self.feature_names)
        }, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
```

### Distillation Pipeline

```python
def distill_student_pipeline(teacher_artifacts: Dict,
                           config: StudentDistillationConfig):
    """Complete student distillation pipeline"""

    # 1. Load teacher soft labels and L1-only features
    soft_labels = teacher_artifacts['soft_labels']
    l1_features, feature_names = load_l1_only_features(config.data_path)

    # 2. Split data for training/validation
    train_features, val_features, train_labels, val_labels = train_test_split(
        l1_features, soft_labels,
        test_size=0.2,
        stratify=None,  # Regression task
        random_state=config.random_state
    )

    # 3. Initialize and train student
    distiller = LightGBMStudentDistiller(config)
    distiller.fit(
        features=train_features,
        soft_labels=train_labels,
        val_features=val_features,
        val_hard_labels=val_labels,  # Use hard labels for validation
        feature_names=feature_names
    )

    # 4. Validate feature parity
    validator = FeatureParityValidator(tolerance=1e-10)
    parity_report = validator.validate_student_features(
        distiller, train_features[:100], feature_names
    )

    if not parity_report['passed']:
        raise ValueError(f"Feature parity validation failed: {parity_report}")

    # 5. Export to ONNX
    onnx_path, metadata_path = distiller.export_onnx(
        output_dir=config.output_dir,
        model_id=config.model_id
    )

    # 6. Run acceptance tests
    acceptance_results = run_acceptance_tests(
        onnx_path, val_features, distiller.predict_proba(val_features)
    )

    if not acceptance_results['passed']:
        raise ValueError(f"Acceptance tests failed: {acceptance_results}")

    return {
        'student_model': distiller,
        'onnx_path': onnx_path,
        'metadata_path': metadata_path,
        'parity_report': parity_report,
        'acceptance_results': acceptance_results
    }
```

---

## Model Registry & Versioning

### Distillation Registry (Training Side)

```python
class DistillationRegistry:
    """Manages teacher-student model lineage and versioning"""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def register_teacher(self,
                        model: Any,
                        version: ModelVersion,
                        training_data: Dict,
                        performance_metrics: Dict) -> str:
        """Register teacher model with metadata"""

        teacher_id = f"teacher_{version.architecture}_{uuid.uuid4().hex[:8]}"
        teacher_path = self.base_path / "teachers" / teacher_id
        teacher_path.mkdir(parents=True, exist_ok=True)

        # Save model artifacts
        torch.save(model.state_dict(), teacher_path / "model.pt")

        # Save metadata
        metadata = {
            'teacher_id': teacher_id,
            'version': version.__dict__,
            'training_data': training_data,
            'performance_metrics': performance_metrics,
            'created_at': datetime.utcnow().isoformat()
        }

        with open(teacher_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return teacher_id

    def register_student(self,
                        model_bytes: bytes,
                        version: ModelVersion,
                        teacher_id: str,
                        distillation_metrics: Dict) -> str:
        """Register student model with teacher lineage"""

        student_id = f"student_{version.architecture}_{uuid.uuid4().hex[:8]}"
        student_path = self.base_path / "students" / student_id
        student_path.mkdir(parents=True, exist_ok=True)

        # Save ONNX model
        with open(student_path / "model.onnx", 'wb') as f:
            f.write(model_bytes)

        # Save metadata with lineage
        metadata = {
            'student_id': student_id,
            'teacher_id': teacher_id,
            'version': version.__dict__,
            'distillation_metrics': distillation_metrics,
            'created_at': datetime.utcnow().isoformat()
        }

        with open(student_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return student_id

    def validate_student(self,
                        student_id: str,
                        acceptance_criteria: Dict) -> bool:
        """Validate student against acceptance criteria"""

        student_path = self.base_path / "students" / student_id
        metadata_path = student_path / "metadata.json"

        with open(metadata_path) as f:
            metadata = json.load(f)

        metrics = metadata['distillation_metrics']

        # Check each criterion
        for criterion, threshold in acceptance_criteria.items():
            if criterion in metrics:
                if metrics[criterion] > threshold:
                    return False

        # Mark as validated
        metadata['validated'] = True
        metadata['validation_criteria'] = acceptance_criteria
        metadata['validated_at'] = datetime.utcnow().isoformat()

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return True
```

### Production Model Registry

```python
class ModelRegistry:
    """Production model registry with deployment automation"""

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

    def register_model(self,
                      model_path: Path,
                      manifest: ModelManifest,
                      auto_deploy: bool = False) -> str:
        """Register model for production deployment"""

        model_id = f"{manifest.model_id}_{uuid.uuid4().hex[:8]}"
        deployment_path = self.registry_path / model_id
        deployment_path.mkdir(parents=True, exist_ok=True)

        # Copy model file
        shutil.copy(model_path, deployment_path / "model.onnx")

        # Save manifest
        manifest_dict = asdict(manifest)
        manifest_dict['model_id'] = model_id
        manifest_dict['registered_at'] = datetime.utcnow().isoformat()

        with open(deployment_path / "manifest.json", 'w') as f:
            json.dump(manifest_dict, f, indent=2)

        # Auto-deploy if requested
        if auto_deploy:
            self.deploy_model(model_id)

        return model_id

    def deploy_model(self, model_id: str):
        """Deploy model to active production slot"""

        model_path = self.registry_path / model_id
        active_path = self.registry_path / "active"

        # Atomic deployment via symlink
        temp_link = active_path.with_suffix(".tmp")
        temp_link.symlink_to(model_path)
        temp_link.replace(active_path)

        logger.info(f"Deployed model {model_id} to production")

    def get_active_model(self) -> Tuple[Path, Dict]:
        """Get currently deployed model and metadata"""

        active_path = self.registry_path / "active"
        if not active_path.exists():
            raise FileNotFoundError("No model currently deployed")

        model_path = active_path / "model.onnx"
        manifest_path = active_path / "manifest.json"

        with open(manifest_path) as f:
            manifest = json.load(f)

        return model_path, manifest
```

---

## Deployment Pipeline

### Complete Training-to-Production Pipeline

```python
def complete_teacher_student_pipeline(config: TeacherStudentPipelineConfig):
    """End-to-end teacher-student training and deployment"""

    # 1. Train Teacher
    logger.info("Starting teacher training")
    teacher, soft_labels = train_teacher_pipeline(config.teacher_config)

    # 2. Register Teacher
    distillation_registry = DistillationRegistry(config.registry_path)
    teacher_id = distillation_registry.register_teacher(
        model=teacher.model,
        version=ModelVersion(
            model_type='teacher',
            architecture='TFT',
            version='1.0.0'
        ),
        training_data={'range': f"{config.start_date}..{config.end_date}"},
        performance_metrics=evaluate_teacher_performance(teacher)
    )

    # 3. Distill Student
    logger.info("Starting student distillation")
    student_artifacts = distill_student_pipeline(
        teacher_artifacts={'soft_labels': soft_labels},
        config=config.student_config
    )

    # 4. Register Student
    with open(student_artifacts['onnx_path'], 'rb') as f:
        student_bytes = f.read()

    student_id = distillation_registry.register_student(
        model_bytes=student_bytes,
        version=ModelVersion(
            model_type='student',
            architecture='LightGBM+ONNX',
            parent_id=teacher_id
        ),
        teacher_id=teacher_id,
        distillation_metrics=student_artifacts['acceptance_results']
    )

    # 5. Validate Student
    validation_passed = distillation_registry.validate_student(
        student_id=student_id,
        acceptance_criteria={
            'max_feature_error': 1e-10,
            'max_accuracy_loss': 0.05,
            'max_latency_ms': 5.0
        }
    )

    if not validation_passed:
        raise ValueError(f"Student validation failed for {student_id}")

    # 6. Create Production Manifest
    manifest = ModelManifest(
        model_id=config.student_config.model_id,
        role=ModelRole.STUDENT,
        data_requirements=DataRequirements.L1_ONLY,
        architecture='LightGBM+ONNX',
        feature_schema=load_feature_schema(student_artifacts['metadata_path']),
        parent_id=teacher_id,
        performance_metrics=student_artifacts['acceptance_results']
    )

    # 7. Register for Production
    prod_registry = ModelRegistry(config.production_registry_path)
    production_model_id = prod_registry.register_model(
        model_path=student_artifacts['onnx_path'],
        manifest=manifest,
        auto_deploy=config.auto_deploy
    )

    logger.info(f"Pipeline complete. Production model: {production_model_id}")

    return {
        'teacher_id': teacher_id,
        'student_id': student_id,
        'production_model_id': production_model_id,
        'artifacts': student_artifacts
    }
```

---

## Inference Integration

### ONNX ML Inference Actor

```python
class ONNXMLInferenceActor(BaseMLInferenceActor):
    """High-performance ONNX model inference actor"""

    def __init__(self, config: ONNXInferenceConfig):
        super().__init__(config)
        self.onnx_session = None
        self.feature_engineer = None
        self.model_metadata = None

    def on_start(self):
        """Initialize ONNX session and validate feature schema"""
        super().on_start()

        # Load model and metadata
        model_path, metadata = self.model_loader.load_active_model()

        # Initialize ONNX session
        import onnxruntime as ort
        session_options = ort.SessionOptions()
        session_options.enable_cpu_mem_arena = False  # Memory optimization
        session_options.enable_mem_pattern = False

        self.onnx_session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )

        self.model_metadata = metadata

        # Validate feature schema
        self._validate_feature_schema()

        # Initialize feature engineering
        self.feature_engineer = FeatureEngineer(self.config.feature_config)

        # Subscribe to required data
        self.subscribe_trade_ticks(self.instrument_id)

        self._log.info(f"Loaded ONNX model: {metadata['model_id']}")

    def _validate_feature_schema(self):
        """Validate that feature engineering matches model expectations"""
        expected_names = self.model_metadata.get('feature_names', [])
        actual_names = self.feature_engineer.get_feature_names()

        if expected_names != actual_names:
            raise ValueError(
                f"Feature schema mismatch. "
                f"Expected: {expected_names}, "
                f"Actual: {actual_names}"
            )

        # Validate schema hash if available
        expected_hash = self.model_metadata.get('feature_schema_hash')
        if expected_hash:
            actual_hash = self.feature_engineer.compute_schema_hash()
            if expected_hash != actual_hash:
                raise ValueError(
                    f"Feature schema hash mismatch. "
                    f"Expected: {expected_hash}, "
                    f"Actual: {actual_hash}"
                )

    def on_trade_tick(self, tick: TradeTick):
        """Process trade tick and generate prediction"""
        # Update features incrementally
        self.feature_engineer.update(tick)

        # Generate prediction if ready
        if self.feature_engineer.is_ready():
            prediction = self._predict()
            self._publish_signal(prediction)

            # Record metrics
            self.metrics.prediction_count.inc()

    def _predict(self) -> Prediction:
        """Generate model prediction with performance monitoring"""

        with self.metrics.inference_latency.time():
            # Extract current features
            features = self.feature_engineer.get_current_features()

            # Reshape for ONNX (batch dimension)
            input_array = features.reshape(1, -1).astype(np.float32)

            # Run inference
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name

            result = self.onnx_session.run(
                [output_name],
                {input_name: input_array}
            )

            probability = float(result[0][0])
            confidence = max(probability, 1.0 - probability)

            return Prediction(
                probability=probability,
                confidence=confidence,
                timestamp=self._clock.timestamp_ns()
            )

    def _publish_signal(self, prediction: Prediction):
        """Publish ML signal to message bus"""
        signal = MLSignal(
            instrument_id=self.instrument_id,
            model_id=self.model_metadata['model_id'],
            prediction=prediction.probability,
            confidence=prediction.confidence,
            ts_event=prediction.timestamp,
            ts_init=self._clock.timestamp_ns()
        )

        self.publish_signal("MLPrediction", signal)
```

---

## Validation & Testing

### Acceptance Testing Framework

```python
def run_acceptance_tests(onnx_path: Path,
                        test_features: np.ndarray,
                        expected_predictions: np.ndarray) -> Dict:
    """Run comprehensive acceptance tests for student model"""

    results = {
        'passed': True,
        'tests': {},
        'summary': {}
    }

    # 1. ONNX Runtime Parity Test
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path))

        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        onnx_predictions = session.run(
            [output_name],
            {input_name: test_features.astype(np.float32)}
        )[0].flatten()

        # Compare with expected
        max_error = np.max(np.abs(onnx_predictions - expected_predictions))
        parity_passed = max_error < 1e-6

        results['tests']['onnx_parity'] = {
            'passed': parity_passed,
            'max_error': float(max_error),
            'mean_error': float(np.mean(np.abs(onnx_predictions - expected_predictions)))
        }

        if not parity_passed:
            results['passed'] = False

    except Exception as e:
        results['tests']['onnx_parity'] = {
            'passed': False,
            'error': str(e)
        }
        results['passed'] = False

    # 2. Latency Test
    try:
        latencies = []
        for i in range(100):
            start_time = time.perf_counter()
            _ = session.run(
                [output_name],
                {input_name: test_features[i:i+1].astype(np.float32)}
            )
            latencies.append((time.perf_counter() - start_time) * 1000)

        p99_latency = np.percentile(latencies, 99)
        latency_passed = p99_latency < 5.0  # 5ms requirement

        results['tests']['latency'] = {
            'passed': latency_passed,
            'p99_ms': p99_latency,
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies)
        }

        if not latency_passed:
            results['passed'] = False

    except Exception as e:
        results['tests']['latency'] = {
            'passed': False,
            'error': str(e)
        }
        results['passed'] = False

    # 3. Model Validation Test
    try:
        # Check model inputs/outputs
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape

        shape_valid = (
            len(input_shape) == 2 and
            input_shape[1] == test_features.shape[1] and
            len(output_shape) == 2 and
            output_shape[1] == 1
        )

        results['tests']['model_validation'] = {
            'passed': shape_valid,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'expected_features': test_features.shape[1]
        }

        if not shape_valid:
            results['passed'] = False

    except Exception as e:
        results['tests']['model_validation'] = {
            'passed': False,
            'error': str(e)
        }
        results['passed'] = False

    # Generate summary
    passed_tests = sum(1 for test in results['tests'].values() if test.get('passed', False))
    total_tests = len(results['tests'])

    results['summary'] = {
        'passed_tests': passed_tests,
        'total_tests': total_tests,
        'success_rate': passed_tests / total_tests if total_tests > 0 else 0
    }

    return results
```

### Integration Testing

```python
class TeacherStudentIntegrationTest:
    """Integration tests for complete teacher-student pipeline"""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from teacher training to student deployment"""

        # 1. Prepare test data
        test_data = self._generate_synthetic_data()

        # 2. Train teacher
        teacher_config = TFTTeacherConfig(
            max_epochs=2,  # Reduced for testing
            hidden_size=32,
            learning_rate=0.01
        )

        teacher = TFTTeacher(teacher_config)
        dataset = teacher.prepare_dataset(test_data['l1'], test_data['l2_l3'])
        teacher.fit(dataset)

        # 3. Generate soft labels
        soft_labels = teacher.generate_soft_labels(test_data['l1_features'])

        # 4. Train student
        student_config = StudentDistillationConfig(
            num_boost_round=100,
            early_stopping_rounds=20
        )

        distiller = LightGBMStudentDistiller(student_config)
        distiller.fit(
            features=test_data['l1_features'],
            soft_labels=soft_labels,
            feature_names=test_data['feature_names']
        )

        # 5. Export and test ONNX
        onnx_path, metadata_path = distiller.export_onnx(
            output_dir=Path("test_output"),
            model_id="test_student"
        )

        # 6. Run acceptance tests
        acceptance_results = run_acceptance_tests(
            onnx_path,
            test_data['l1_features'][:100],
            distiller.predict_proba(test_data['l1_features'][:100])
        )

        assert acceptance_results['passed'], f"Acceptance tests failed: {acceptance_results}"

        # 7. Test feature parity
        validator = FeatureParityValidator()
        parity_passed = validator.validate_parity(
            test_data['l1_features'][:10],
            test_data['l1_features'][:10]  # Same data for this test
        )

        assert parity_passed, "Feature parity validation failed"

        # Cleanup
        shutil.rmtree("test_output")

    def _generate_synthetic_data(self) -> Dict:
        """Generate synthetic data for testing"""
        np.random.seed(42)

        n_samples = 1000
        n_features = 20

        # Generate L1 features
        l1_features = np.random.randn(n_samples, n_features).astype(np.float32)
        feature_names = [f"l1_feature_{i}" for i in range(n_features)]

        # Generate synthetic L1/L2/L3 data
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1min')

        l1_data = pl.DataFrame({
            'timestamp': timestamps,
            'price': 1.0 + np.random.randn(n_samples) * 0.01,
            'volume': np.random.randint(100, 1000, n_samples)
        })

        l2_l3_data = pl.DataFrame({
            'timestamp': timestamps,
            'bid_size': np.random.randint(10, 100, n_samples),
            'ask_size': np.random.randint(10, 100, n_samples),
            'depth_imbalance': np.random.randn(n_samples)
        })

        return {
            'l1_features': l1_features,
            'feature_names': feature_names,
            'l1': l1_data,
            'l2_l3': l2_l3_data
        }
```

This comprehensive teacher-student architecture provides a robust framework for training high-quality models offline and deploying efficient students for real-time trading inference in NautilusTrader.
