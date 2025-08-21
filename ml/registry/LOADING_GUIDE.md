# Registry Loading Guide

This guide provides comprehensive documentation on how to load and query items from the ML Registry system in Nautilus Trader.

## Table of Contents

- [Model Registry](#model-registry)
- [Feature Registry](#feature-registry)
- [Strategy Registry](#strategy-registry)
- [Practical Examples](#practical-examples)
- [Error Handling](#error-handling)

## Model Registry

### Primary Loading Methods

```python
from pathlib import Path
from ml.registry.model_registry import ModelRegistry
from ml.registry.base import ModelRole, DataRequirements, DeploymentStatus

registry = ModelRegistry(Path("registry"))

# 1. Load by model_id (returns ONNX InferenceSession)
model_session = registry.load_model("lgb_student_v1")

# 2. Get model info with manifest (returns ModelInfo)
model_info = registry.get_model("lgb_student_v1")
```

### Secondary Query Methods

```python
# By Role
student_models = registry.get_models_by_role(ModelRole.STUDENT)
teacher_models = registry.get_models_by_role(ModelRole.TEACHER)
inference_models = registry.get_models_by_role(ModelRole.INFERENCE)
ensemble_models = registry.get_models_by_role(ModelRole.ENSEMBLE)

# By Data Requirements
l1_only_models = registry.get_models_by_data_requirements(DataRequirements.L1_ONLY)
l1_l2_models = registry.get_models_by_data_requirements(DataRequirements.L1_L2)
historical_models = registry.get_models_by_data_requirements(DataRequirements.HISTORICAL)

# By Deployment Target
deployed_models = registry.get_deployed_models("ml_signal_actor")
active_deployments = registry.get_active_deployments()

# By Lineage (Teacher-Student relationships)
lineage = registry.get_model_lineage("lgb_student_v2")  # Returns chain from root

# Latest Model Matching Criteria
latest = registry.get_latest_model(
    role=ModelRole.STUDENT,
    data_requirements=DataRequirements.L1_ONLY,
    parent_id="tft_teacher_v1"  # Optional: specific teacher
)
```

### Model Information Structure

```python
model_info = registry.get_model("lgb_student_v1")

# Access manifest fields
model_info.manifest.model_id           # "lgb_student_v1"
model_info.manifest.role               # ModelRole.STUDENT
model_info.manifest.data_requirements  # DataRequirements.L1_ONLY
model_info.manifest.architecture       # "LightGBM"
model_info.manifest.feature_schema     # {"close": "float32", "volume": "float32"}
model_info.manifest.feature_schema_hash # "abc123..."
model_info.manifest.parent_id          # "tft_teacher_v1"
model_info.manifest.performance_metrics # {"accuracy": 0.85, "latency_ms": 1.2}
model_info.manifest.deployment_constraints # {"max_latency_ms": 5}

# Access deployment info
model_info.deployment_status           # DeploymentStatus.ACTIVE
model_info.deployed_to                 # ["ml_signal_actor"]
model_info.file_path                   # Path("models/lgb_student_v1.onnx")
```

## Feature Registry

### Primary Loading Methods

```python
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole, FeatureStage

feature_registry = FeatureRegistry(Path("registry"))

# 1. Load by feature_set_id
manifest = feature_registry.get_feature_set("student_features_v1")

# 2. Check if registered
exists = feature_registry.is_registered("student_features_v1")
```

### Secondary Query Methods

```python
# By Schema Hash (for compatibility checking)
compatible_features = feature_registry.resolve_by_schema_hash("abc123def456...")

# By Role
student_features = feature_registry.list_by_role(FeatureRole.STUDENT)
teacher_features = feature_registry.list_by_role(FeatureRole.TEACHER)
inference_features = feature_registry.list_by_role(FeatureRole.INFERENCE_SUPPORT)

# Get All Features
all_features = feature_registry.list_all()

# By Stage (filter from list_all)
prod_features = [f for f in feature_registry.list_all()
                 if f.stage == FeatureStage.PROD]
staging_features = [f for f in feature_registry.list_all()
                    if f.stage == FeatureStage.STAGING]

# By Lineage (parent-child relationships)
lineage = feature_registry.get_lineage("student_features_v2")
```

### Feature Manifest Structure

```python
manifest = feature_registry.get_feature_set("student_features_v1")

# Identity
manifest.feature_set_id        # "student_features_v1"
manifest.name                  # "L1 Student Features"
manifest.version               # "1.0.0"
manifest.role                  # FeatureRole.STUDENT
manifest.stage                 # FeatureStage.PROD

# Schema
manifest.feature_names         # ["close_ratio", "volume_ma", "rsi"]
manifest.feature_dtypes        # ["float32", "float32", "float32"]
manifest.schema_hash           # "abc123..."

# Pipeline
manifest.pipeline_signature    # "sha256:..."
manifest.pipeline_version      # "1.0.0"
manifest.capabilities          # {"handles_nans": True, "stateful": True}

# Performance
manifest.parity_tolerance      # 1e-10
manifest.parity_digest         # {"max_diff": 1e-11, "mean_diff": 1e-12}
manifest.perf_digest          # {"p50_latency_ms": 0.3, "p99_latency_ms": 0.45}

# Relationships
manifest.parent_feature_set_id # "teacher_features_v1"
manifest.constraints          # {"max_latency_ms": 0.5, "min_bars_warmup": 20}
```

## Strategy Registry

### Primary Loading Methods

```python
from ml.registry.strategy_registry import StrategyRegistry
from ml.registry.strategy_registry import StrategyType, MarketRegime

strategy_registry = StrategyRegistry(Path("registry"))

# 1. Load by strategy_id
strategy_info = strategy_registry.get_strategy("ml_momentum_v1")

# 2. Check if registered
exists = strategy_registry.is_registered("ml_momentum_v1")
```

### Secondary Query Methods

```python
# By Market Regime
volatile_strategies = strategy_registry.get_strategies_for_regime(MarketRegime.VOLATILE)
trending_up = strategy_registry.get_strategies_for_regime(MarketRegime.TRENDING_UP)
ranging = strategy_registry.get_strategies_for_regime(MarketRegime.RANGING)

# By Instrument Type
fx_strategies = strategy_registry.get_strategies_for_instrument_type("FX")
crypto_strategies = strategy_registry.get_strategies_for_instrument_type("CRYPTO")
equity_strategies = strategy_registry.get_strategies_for_instrument_type("EQUITY")

# By Performance (ranked)
top_by_sharpe = strategy_registry.get_strategies_ranked_by_performance(
    metric="sharpe",
    use_live_metrics=False  # Use backtest metrics
)
top_by_win_rate = strategy_registry.get_strategies_ranked_by_performance(
    metric="win_rate",
    use_live_metrics=True   # Use live metrics if available
)

# By Lineage (parent strategies)
lineage = strategy_registry.get_strategy_lineage("ml_momentum_v2")

# Compatibility Checking
is_compatible = strategy_registry.check_compatibility(
    strategy_id="ml_momentum_v1",
    active_strategies=["trend_follower_v1", "market_maker_v2"]
)

# Requirements Validation
requirements_met = strategy_registry.validate_requirements(
    strategy_id="ml_momentum_v1",
    available_models=["lgb_student_v1", "xgb_inference_v2"],
    available_features=["student_features_v1"]
)
```

### Strategy Manifest Structure

```python
strategy_info = strategy_registry.get_strategy("ml_momentum_v1")

# Access manifest
manifest = strategy_info.manifest
file_path = strategy_info.file_path  # Path to strategy implementation

# Identity
manifest.strategy_id           # "ml_momentum_v1"
manifest.strategy_type         # StrategyType.MOMENTUM
manifest.version              # "1.0.0"

# Requirements
manifest.required_models      # ["lgb_student_v1"]
manifest.required_features    # ["student_features_v1"]

# Market Conditions
manifest.suitable_regimes     # [MarketRegime.TRENDING_UP, MarketRegime.VOLATILE]
manifest.instrument_types     # ["FX", "CRYPTO"]
manifest.timeframe_range      # ("1m", "1h")

# Risk Parameters
manifest.max_position_size    # 100000.0
manifest.max_leverage         # 3.0
manifest.max_drawdown         # 0.10
manifest.stop_loss_type       # "trailing"

# Performance Constraints
manifest.min_sharpe_ratio     # 1.5
manifest.min_win_rate         # 0.55
manifest.max_correlation_with_portfolio # 0.7

# Dependencies
manifest.parent_strategy_id   # "base_momentum_v1"
manifest.incompatible_strategies # ["ml_mean_reversion_v1"]

# Configuration
manifest.config_schema        # {"threshold": "float", "lookback": "int"}
manifest.default_config       # {"threshold": 0.7, "lookback": 20}

# Performance Metrics
manifest.backtest_metrics     # {"sharpe": 2.1, "win_rate": 0.62}
manifest.live_metrics         # {"sharpe": 1.9, "win_rate": 0.58}

# Metadata
manifest.created_at           # 1234567890.0
manifest.last_modified        # 1234567900.0
manifest.author              # "ML Team"
manifest.description         # "Momentum strategy using ML signals"
```

## Practical Examples

### Complete Trading Session Setup

```python
from pathlib import Path
from ml.registry import (
    ModelRegistry,
    FeatureRegistry,
    StrategyRegistry,
    FeatureStage,
    DeploymentStatus
)

def setup_trading_session(strategy_id: str, base_path: Path):
    """Load and validate all components for a trading session."""

    # Initialize registries
    model_registry = ModelRegistry(base_path / "models")
    feature_registry = FeatureRegistry(base_path / "features")
    strategy_registry = StrategyRegistry(base_path / "strategies")

    # 1. Load strategy
    strategy = strategy_registry.get_strategy(strategy_id)
    if not strategy:
        raise ValueError(f"Strategy {strategy_id} not found")

    # 2. Validate market regime compatibility
    current_regime = detect_market_regime()  # Your implementation
    if current_regime not in strategy.manifest.suitable_regimes:
        raise ValueError(f"Strategy not suitable for {current_regime}")

    # 3. Load and validate all required models
    models = {}
    for model_id in strategy.manifest.required_models:
        model_session = model_registry.load_model(model_id)
        model_info = model_registry.get_model(model_id)

        if not model_session:
            raise ValueError(f"Model {model_id} not found")

        if model_info.deployment_status != DeploymentStatus.ACTIVE:
            raise ValueError(f"Model {model_id} not active")

        models[model_id] = {
            "session": model_session,
            "info": model_info
        }

    # 4. Load and validate all required features
    features = {}
    for feature_id in strategy.manifest.required_features:
        feature_manifest = feature_registry.get_feature_set(feature_id)

        if not feature_manifest:
            raise ValueError(f"Feature set {feature_id} not found")

        if feature_manifest.stage != FeatureStage.PROD:
            raise ValueError(f"Features {feature_id} not in production")

        features[feature_id] = feature_manifest

    # 5. Cross-validate schema compatibility
    for model_id, model_data in models.items():
        model_schema_hash = model_data["info"].manifest.feature_schema_hash

        # Find matching feature set
        matching_features = [
            f for f in features.values()
            if f.schema_hash == model_schema_hash
        ]

        if not matching_features:
            raise ValueError(f"No features match model {model_id} schema")

    # 6. Check strategy compatibility with active strategies
    active_strategies = get_active_strategies()  # Your implementation
    if not strategy_registry.check_compatibility(strategy_id, active_strategies):
        raise ValueError(f"Strategy {strategy_id} incompatible with active strategies")

    return {
        "strategy": strategy,
        "models": models,
        "features": features
    }
```

### Finding Best Model for Current Conditions

```python
def find_best_model(
    role: ModelRole,
    data_requirements: DataRequirements,
    max_latency_ms: float
) -> str | None:
    """Find the best performing model matching constraints."""

    registry = ModelRegistry(Path("registry"))

    # Get all candidates
    candidates = registry.get_models_by_role(role)

    # Filter by data requirements
    candidates = [
        m for m in candidates
        if m.manifest.data_requirements == data_requirements
    ]

    # Filter by latency constraint
    candidates = [
        m for m in candidates
        if m.manifest.performance_metrics.get("inference_latency_ms", float("inf")) <= max_latency_ms
    ]

    # Sort by accuracy (or other metric)
    candidates.sort(
        key=lambda m: m.manifest.performance_metrics.get("accuracy", 0),
        reverse=True
    )

    if candidates:
        return candidates[0].manifest.model_id
    return None
```

### A/B Testing Setup

```python
def setup_ab_test(
    control_model_id: str,
    treatment_model_id: str,
    target_actor: str,
    traffic_split: float = 0.5
):
    """Setup A/B test between two models."""

    registry = ModelRegistry(Path("registry"))

    # Verify both models exist and are compatible
    control = registry.get_model(control_model_id)
    treatment = registry.get_model(treatment_model_id)

    if not control or not treatment:
        raise ValueError("Models not found")

    # Check schema compatibility
    if control.manifest.feature_schema_hash != treatment.manifest.feature_schema_hash:
        raise ValueError("Models have incompatible feature schemas")

    # Start A/B test
    from ml.registry.dataclasses import ABTestConfig

    test_id = registry.start_ab_test(
        control_id=control_model_id,
        treatment_id=treatment_model_id,
        target=target_actor,
        config=ABTestConfig(
            traffic_split=traffic_split,
            metric="accuracy",
            min_sample_size=1000,
            max_duration_hours=24
        )
    )

    return test_id
```

## Error Handling

### Common Patterns

```python
def safe_model_load(model_id: str) -> Any:
    """Load model with comprehensive error handling."""

    registry = ModelRegistry(Path("registry"))

    try:
        # Check if registered
        model_info = registry.get_model(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not registered")

        # Check deployment status
        if model_info.deployment_status == DeploymentStatus.RETIRED:
            raise ValueError(f"Model {model_id} is retired")

        if model_info.deployment_status == DeploymentStatus.FAILED:
            raise ValueError(f"Model {model_id} failed deployment")

        # Load model
        model_session = registry.load_model(model_id)
        if not model_session:
            raise RuntimeError(f"Failed to load model {model_id}")

        return model_session

    except FileNotFoundError as e:
        raise RuntimeError(f"Model file not found for {model_id}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading {model_id}: {e}")
```

### Validation Helpers

```python
def validate_feature_model_compatibility(
    feature_id: str,
    model_id: str
) -> bool:
    """Validate that features and model are compatible."""

    feature_registry = FeatureRegistry(Path("registry/features"))
    model_registry = ModelRegistry(Path("registry/models"))

    feature = feature_registry.get_feature_set(feature_id)
    model = model_registry.get_model(model_id)

    if not feature or not model:
        return False

    # Check schema hash
    if feature.schema_hash != model.manifest.feature_schema_hash:
        return False

    # Check feature names match
    expected_features = set(model.manifest.feature_schema.keys())
    provided_features = set(feature.feature_names)

    if expected_features != provided_features:
        return False

    # Check data types
    for name, dtype in model.manifest.feature_schema.items():
        idx = feature.feature_names.index(name)
        if feature.feature_dtypes[idx] != dtype:
            return False

    return True
```

## Query Optimization Tips

1. **Use Caching**: Registries have built-in LRU caching for frequently accessed items
2. **Batch Queries**: Load all required items at startup rather than on-demand
3. **Schema Hash Indexing**: Use schema hashes for fast compatibility checking
4. **Lazy Loading**: Models are loaded into memory only when accessed via `load_model()`
5. **Metadata First**: Use `get_model()` for metadata checks before loading with `load_model()`

## Summary

The registry system provides rich querying capabilities:

| Registry | Primary ID | Secondary Lookups | Special Features |
|----------|------------|-------------------|------------------|
| **Model** | `model_id` | Role, Data Requirements, Deployment Target | Lineage, A/B Testing, Hot Reload |
| **Feature** | `feature_set_id` | Schema Hash, Role, Stage | Parity Validation, Pipeline Tracking |
| **Strategy** | `strategy_id` | Market Regime, Instrument Type, Performance | Compatibility Checking, Requirements Validation |

All registries follow consistent patterns:

- Primary lookup by unique ID
- Secondary filtering by attributes
- Relationship tracking (lineage)
- Performance-based ranking
- Validation and compatibility checking
