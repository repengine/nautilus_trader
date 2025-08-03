# Analysis: Neural Forecast Trainer - Polars/msgspec Handling

## Key Observations

### 1. **Correct Separation of Concerns**

The trainer properly separates training (using Polars) from runtime configuration:

```python
# Training method accepts Polars DataFrame directly
def train(self, train_data: pl.DataFrame, val_data: pl.DataFrame, optimize_hyperparams: bool = True) -> dict[str, Any]:
```

```python
# Configuration uses plain dict, NOT msgspec config
def __init__(self, config: dict[str, Any], settings: Settings | None = None):
```

### 2. **Data Transformation Pipeline**

The trainer handles the Polars → Pandas → Numpy transformation cleanly:

```python
def prepare_data(self, data: pl.DataFrame, target_col: str = "close") -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    # 1. Use Polars for feature engineering
    features_df, scaler = self.feature_engineer.calculate_features_batch(data, fit_scaler=True)
    
    # 2. Convert to Pandas for NeuralForecast
    df = DataFrameConverter.polars_to_pandas(data)
    features_pd = DataFrameConverter.polars_to_pandas(features_df)
    
    # 3. Create NeuralForecast format
    nf_df = pd.DataFrame({
        "unique_id": "series_1",
        "ds": pd.to_datetime(df["timestamp"]),
        "y": df[target_col].values,
    })
```

### 3. **Model Artifacts Saved to Disk**

The trainer saves everything needed for inference:

```python
def _save_artifacts(self, results: dict[str, Any]):
    # Save scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(results["scaler"], f)
    
    # Save feature config
    with open("feature_config.pkl", "wb") as f:
        pickle.dump(self.feature_engineer.config, f)
    
    # Save model config as JSON
    model_config = {
        "model_type": self.model_type,
        "forecast_horizon": self.forecast_horizon,
        "feature_names": results["feature_names"],
        "params": results["params"],
    }
    self.mlflow_manager.log_dict_as_json(model_config, "model_config")
```

### 4. **Production Bundle Pattern**

The `save_model_bundle` method creates a complete package:

```python
model_data = {
    "model": results["model"],
    "scaler": results["scaler"],
    "feature_config": self.feature_engineer.config,
    "feature_names": results["feature_names"],
    "version": version,
    "strategy_name": strategy_name,
    "test_metrics": results["metrics"],
    "model_params": results["params"],
    "model_type": self.model_type,
    "framework": "neural_forecast",
}

with open(model_path, "wb") as f:
    pickle.dump(model_data, f)
```

## How This Avoids Polars/msgspec Conflicts

### 1. **Training Script Pattern**
- This is a standalone training script, not a Nautilus Actor/Strategy
- It freely uses Polars for data processing
- Saves everything to disk as .pkl files

### 2. **For Inference Integration**

To use this in Nautilus, you would create an inference actor that:

```python
class NeuralForecastInferenceActor(Actor):
    def __init__(self, config: ActorConfig):
        super().__init__(config)
        
        # Load from disk - no Polars in config!
        with open(config.model_bundle_path, 'rb') as f:
            self.model_bundle = pickle.load(f)
            
        self.model = self.model_bundle['model']
        self.scaler = self.model_bundle['scaler']
        self.feature_names = self.model_bundle['feature_names']
        
        # No Polars in hot path!
        self.feature_buffer = deque(maxlen=self.model_bundle['input_size'])
        
    def on_bar(self, bar: Bar):
        # Update features without Polars
        features = self._extract_features_numpy(bar)
        prediction = self.model.predict(features)
```

### 3. **Configuration Pattern**

Note how the trainer uses plain dict config, not msgspec:

```python
config = {
    "instrument": instrument,
    "model_type": "TFT",
    "forecast_horizon": 24,
    "input_size": 168,
    "n_trials": 50,
    # Simple types only!
}
```

## Lessons for Our ML Integration

### 1. **Two-Stage Pattern**
- **Stage 1**: Training scripts (use Polars freely)
- **Stage 2**: Inference actors (load from disk, no Polars)

### 2. **Model Bundle Concept**
- Save everything needed for inference in one .pkl file
- Include model, scaler, feature config, metadata
- Load this bundle in inference actor

### 3. **Feature Engineering Bridge**
- Training: Polars → Pandas → Numpy
- Inference: Nautilus Bar → Numpy directly

### 4. **MLflow Integration**
- Use MLflow for experiment tracking
- Save artifacts for reproducibility
- But load from local .pkl for inference

## Migration Strategy

To adapt this for our ml/ folder:

```python
# ml/training/neural_forecast_trainer.py
class NeuralForecastTrainer:
    """Offline trainer - uses Polars freely"""
    def train(self, data: pl.DataFrame) -> None:
        # ... training code ...
        self.save_model_bundle("models/neural_forecast.pkl")

# ml/actors/neural_forecast_actor.py  
class NeuralForecastActor(Actor):
    """Online inference - no Polars!"""
    def __init__(self, config: ActorConfig):
        self.model_bundle = pickle.load(open(config.model_path, 'rb'))
```

## Key Takeaways

1. **This trainer does it right** - Polars for training, disk storage, no Polars in inference
2. **Model bundles** are a great pattern for ML deployment
3. **Plain dict config** avoids msgspec serialization issues
4. **Clear separation** between training and inference environments

This is a good pattern to follow for our ML integration!