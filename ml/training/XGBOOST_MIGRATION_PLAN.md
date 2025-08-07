# XGBoost Trainer Migration Plan (Phase 3.1)

## Executive Summary

This document outlines the comprehensive plan for migrating the UnifiedXGBoostTrainer from the OLD/trade/nautilus_ml system to the new Nautilus ML architecture. The migration will enhance the existing `ml/training/xgboost.py` with advanced features while maintaining strict hot/cold path separation and architectural compliance.

## Current State Analysis

### Existing Components (NEW System)

1. **ml/training/xgboost.py**: Basic XGBoostTrainer with:
   - Single/multi-asset support
   - Feature importance calculation
   - SHAP value computation
   - Integration with FeatureEngineer

2. **ml/data/loader.py**: MLDataLoader with:
   - Efficient Parquet data loading
   - Caching mechanisms
   - Multi-instrument support
   - Integration with ParquetDataCatalog

3. **ml/features/engineering_enhanced.py**: Enhanced features including:
   - Microstructure features
   - Trade flow features
   - Feature quality metrics

4. **ml/monitoring/**: Comprehensive monitoring with:
   - Prometheus metrics collection
   - Grafana dashboards
   - Feature/model/performance collectors

### Components to Migrate (OLD System)

1. **UnifiedXGBoostTrainer** features:
   - Optuna hyperparameter optimization
   - GPU acceleration configuration
   - Monotonic constraints
   - MLflow experiment tracking
   - Feature decay tracking
   - Cross-sectional features for portfolios
   - Advanced validation strategies

## Migration Architecture

### 1. Enhanced XGBoostTrainer Structure

```python
ml/training/
├── xgboost_unified.py       # Enhanced trainer with all features
├── optimization/
│   ├── __init__.py
│   ├── optuna_optimizer.py  # Hyperparameter optimization
│   └── search_spaces.py     # XGBoost-specific search spaces
└── mlflow/
    ├── __init__.py
    ├── tracking.py           # MLflow integration
    └── registry.py           # Model registry management
```

### 2. Key Design Principles

#### Hot/Cold Path Separation

- **COLD PATH (Training)**:
  - Full Optuna optimization
  - MLflow tracking
  - SHAP analysis
  - Cross-validation
  - Feature importance tracking

- **HOT PATH (Inference)**:
  - Pre-loaded ONNX models
  - No MLflow calls
  - No feature recalculation
  - Pre-allocated buffers
  - < 2ms inference time

#### Integration Points

1. **MLDataLoader Integration**:

   ```python
   loader = MLDataLoader(catalog)
   data = loader.load_bars_multi(instruments, start, end)
   trainer.train(data)
   ```

2. **FeatureEngineer Integration**:

   ```python
   enhanced_engineer = EnhancedFeatureEngineer(config)
   features = enhanced_engineer.calculate_features_batch(data)
   ```

3. **Monitoring Integration**:

   ```python
   collector = ModelMetricsCollector()
   collector.record_training_metrics(model_id, metrics)
   ```

## Implementation Plan

### Phase 1: Core Enhancement (Week 1)

#### 1.1 Extend XGBoostTrainer

```python
class UnifiedXGBoostTrainer(XGBoostTrainer):
    """Enhanced XGBoost trainer with full feature set."""

    def __init__(self, config: UnifiedXGBoostConfig):
        super().__init__(config)
        self._optuna_config = config.optuna_config
        self._mlflow_config = config.mlflow_config
        self._gpu_config = config.gpu_config

        # Feature tracking
        self._importance_history = []
        self._feature_decay_threshold = config.feature_decay_threshold

        # Validation metadata for multi-asset
        self._validation_metadata = None
```

#### 1.2 Add GPU Acceleration

```python
def _get_gpu_params(self) -> dict[str, Any]:
    """Configure GPU acceleration."""
    if self._gpu_config.enabled:
        return {
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': self._gpu_config.device_id,
            'max_bin': self._gpu_config.max_bin,
        }
    return {'tree_method': 'hist'}
```

#### 1.3 Implement Monotonic Constraints

```python
def _apply_monotonic_constraints(
    self,
    feature_names: list[str],
    constraints: dict[str, int]
) -> str:
    """Apply monotonic constraints for interpretability."""
    constraint_list = []
    for feature in feature_names:
        constraint_list.append(str(constraints.get(feature, 0)))
    return f"({','.join(constraint_list)})"
```

### Phase 2: Optuna Integration (Week 1-2)

#### 2.1 Create Optimizer Module

```python
# ml/training/optimization/optuna_optimizer.py
class XGBoostOptunaSampler:
    """Optuna sampler for XGBoost hyperparameters."""

    def sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        }
```

#### 2.2 Implement Optimization Loop

```python
def optimize_hyperparameters(
    self,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
    metric: str = 'sharpe_ratio'
) -> dict[str, Any]:
    """Optimize hyperparameters using Optuna."""

    def objective(trial: optuna.Trial) -> float:
        params = self._sampler.sample_params(trial)

        # Train model with sampled params
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )

        # Calculate metric
        predictions = model.predict_proba(X_val)[:, 1]
        return self._calculate_metric(y_val, predictions, metric)

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
```

### Phase 3: MLflow Integration (Week 2)

#### 3.1 Create MLflow Tracker

```python
# ml/training/mlflow/tracking.py
class MLflowXGBoostTracker:
    """MLflow tracking for XGBoost models."""

    def __init__(self, tracking_uri: str, experiment_name: str):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_training_run(
        self,
        model: xgb.XGBModel,
        params: dict[str, Any],
        metrics: dict[str, float],
        features: list[str],
        importance: dict[str, float]
    ) -> str:
        """Log complete training run to MLflow."""
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log feature importance
            for feature, score in importance.items():
                mlflow.log_metric(f"importance_{feature}", score)

            # Log model
            mlflow.xgboost.log_model(
                model,
                "model",
                registered_model_name="xgboost_unified"
            )

            # Log artifacts
            mlflow.log_dict(features, "features.json")

            return run.info.run_id
```

#### 3.2 Model Registry Integration

```python
def register_model(
    self,
    run_id: str,
    model_name: str,
    stage: str = "Staging"
) -> str:
    """Register model in MLflow registry."""
    client = mlflow.tracking.MlflowClient()

    # Register model version
    model_version = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id
    )

    # Transition to stage
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=stage
    )

    return model_version.version
```

### Phase 4: Advanced Features (Week 2-3)

#### 4.1 Cross-Validation Strategy

```python
def cross_validate(
    self,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    strategy: str = 'time_series'
) -> dict[str, list[float]]:
    """Advanced cross-validation with multiple strategies."""

    if strategy == 'time_series':
        cv = TimeSeriesSplit(n_splits=cv_folds)
    elif strategy == 'blocked':
        cv = BlockedTimeSeriesSplit(n_splits=cv_folds)
    elif strategy == 'purged':
        cv = PurgedKFold(n_splits=cv_folds, purge_gap=10)

    scores = defaultdict(list)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        # Train model
        model = self._train_fold(X_fold_train, y_fold_train)

        # Evaluate
        metrics = self._evaluate_fold(model, X_fold_val, y_fold_val)
        for key, value in metrics.items():
            scores[key].append(value)

    return dict(scores)
```

#### 4.2 Feature Decay Tracking

```python
def track_feature_importance_decay(
    self,
    current_importance: dict[str, float]
) -> list[str]:
    """Track feature importance decay over time."""

    if not self._importance_history:
        self._importance_history.append(current_importance)
        return []

    # Compare with historical average
    decayed_features = []
    historical_avg = self._calculate_historical_average()

    for feature, current_score in current_importance.items():
        historical_score = historical_avg.get(feature, 0)
        if historical_score > 0:
            decay_ratio = (historical_score - current_score) / historical_score
            if decay_ratio > self._feature_decay_threshold:
                decayed_features.append(feature)
                print(f"Warning: Feature '{feature}' importance "
                      f"declined by {decay_ratio:.1%}")

    self._importance_history.append(current_importance)
    return decayed_features
```

#### 4.3 Model Serialization

```python
def export_to_onnx(
    self,
    output_path: str,
    initial_types: list[tuple[str, Any]] | None = None
) -> None:
    """Export model to ONNX for inference."""
    import onnxmltools
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    if initial_types is None:
        initial_types = [
            ('float_input', FloatTensorType([None, len(self._feature_names)]))
        ]

    # Convert to ONNX
    onnx_model = onnxmltools.convert_xgboost(
        self._model,
        initial_types=initial_types,
        target_opset=12
    )

    # Save
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Model exported to ONNX: {output_path}")
```

### Phase 5: Integration & Testing (Week 3)

#### 5.1 Integration Tests

```python
# ml/tests/integration/test_xgboost_unified.py
def test_full_pipeline_integration():
    """Test complete training pipeline."""
    # Load data
    loader = MLDataLoader(catalog)
    data = loader.load_bars_multi(["EURUSD", "GBPUSD"])

    # Train with optimization
    config = UnifiedXGBoostConfig(
        enable_optuna=True,
        enable_mlflow=True,
        enable_gpu=True,
        n_trials=10
    )
    trainer = UnifiedXGBoostTrainer(config)

    # Prepare features
    X, y, metadata = trainer.prepare_data(data)

    # Optimize and train
    best_params = trainer.optimize_hyperparameters(X, y)
    results = trainer.train(X, y, params=best_params)

    # Export for inference
    trainer.export_to_onnx("model.onnx")

    # Verify monitoring
    assert trainer._metrics_collector.get_metrics()
```

#### 5.2 Performance Benchmarks

```python
def benchmark_training_performance():
    """Benchmark training performance."""
    benchmarks = {
        'data_loading': [],
        'feature_engineering': [],
        'training': [],
        'inference': []
    }

    # Run benchmarks
    for size in [1000, 10000, 100000]:
        data = generate_test_data(size)

        # Measure each phase
        t0 = time.time()
        loader.load_data(data)
        benchmarks['data_loading'].append(time.time() - t0)

        # ... measure other phases

    # Assert performance requirements
    assert np.mean(benchmarks['inference']) < 0.002  # < 2ms
```

## Configuration Schema

```python
# ml/config/xgboost_unified.py
class UnifiedXGBoostConfig(XGBoostTrainingConfig):
    """Configuration for unified XGBoost trainer."""

    # Optuna configuration
    enable_optuna: bool = False
    optuna_config: OptunaConfig | None = None

    # MLflow configuration
    enable_mlflow: bool = False
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "xgboost_unified"

    # GPU configuration
    enable_gpu: bool = False
    gpu_device_id: int = 0
    gpu_max_bin: int = 256

    # Advanced features
    enable_cross_validation: bool = True
    cv_strategy: str = "time_series"
    cv_folds: int = 5

    # Feature tracking
    track_feature_decay: bool = True
    feature_decay_threshold: float = 0.3

    # Model export
    export_onnx: bool = True
    onnx_output_path: str = "./models/xgboost.onnx"
```

## Monitoring Integration

### Metrics to Track

1. **Training Metrics**:
   - Training time
   - Best iteration
   - Feature importance
   - Cross-validation scores
   - Hyperparameter optimization progress

2. **Model Metrics**:
   - Inference latency
   - Prediction accuracy
   - Feature drift
   - Model drift

3. **Resource Metrics**:
   - GPU utilization
   - Memory usage
   - CPU usage

### Prometheus Metrics

```python
# Training metrics
xgboost_training_duration_seconds = Histogram(
    'xgboost_training_duration_seconds',
    'XGBoost training duration',
    ['model_id', 'dataset']
)

xgboost_best_iteration = Gauge(
    'xgboost_best_iteration',
    'Best training iteration',
    ['model_id']
)

xgboost_feature_importance = Gauge(
    'xgboost_feature_importance',
    'Feature importance scores',
    ['model_id', 'feature']
)
```

## Testing Strategy

### Unit Tests

1. Test each component in isolation
2. Mock external dependencies (MLflow, Optuna)
3. Verify configuration validation
4. Test error handling

### Integration Tests

1. End-to-end training pipeline
2. Data loader integration
3. Feature engineer integration
4. Monitoring integration

### Performance Tests

1. Training time benchmarks
2. Inference latency (< 2ms requirement)
3. Memory usage profiling
4. GPU utilization

### Feature Parity Tests

1. Verify features match between training and inference
2. Test with existing parity validation suite
3. Tolerance: 1e-10

## Migration Timeline

### Week 1

- [ ] Extend base XGBoostTrainer with GPU support
- [ ] Implement monotonic constraints
- [ ] Add feature decay tracking
- [ ] Create Optuna optimizer module

### Week 2

- [ ] Complete Optuna integration
- [ ] Implement MLflow tracking
- [ ] Add model registry support
- [ ] Create cross-validation strategies

### Week 3

- [ ] ONNX export functionality
- [ ] Integration testing
- [ ] Performance benchmarking
- [ ] Documentation

### Week 4

- [ ] Production deployment preparation
- [ ] Monitoring dashboard updates
- [ ] Final testing and validation
- [ ] Migration of existing models

## Risk Mitigation

### Technical Risks

1. **GPU Compatibility**: Test on multiple GPU types
2. **Memory Management**: Implement proper cleanup
3. **MLflow Latency**: Async logging for training
4. **ONNX Compatibility**: Validate inference accuracy

### Operational Risks

1. **Model Registry Migration**: Gradual transition
2. **Performance Regression**: Continuous benchmarking
3. **Feature Drift**: Automated monitoring alerts

## Success Criteria

1. **Performance**:
   - Training time comparable to OLD system
   - Inference < 2ms (P99)
   - Memory usage stable over 24h

2. **Functionality**:
   - All OLD features migrated
   - MLflow tracking operational
   - Optuna optimization working
   - ONNX export functional

3. **Quality**:
   - Test coverage > 90%
   - Zero mypy errors
   - Documentation complete
   - Integration tests passing

## Next Steps

1. Review and approve migration plan
2. Set up development branch
3. Begin Phase 1 implementation
4. Schedule weekly progress reviews
5. Prepare production deployment plan
