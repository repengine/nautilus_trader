# ML Component Migration Mapping

## Overview

This document provides a detailed mapping of components from `OLD/trade/nautilus_ml` to the new `ml/` directory, including required modifications for Nautilus compatibility.

## Component Migration Map

### 1. Training Components

#### BaseTrainer

- **OLD**: `OLD/trade/nautilus_ml/training/base_trainer.py`
- **NEW**: `ml/training/base.py`
- **Changes**:

  ```python
  # OLD: Using Settings and dict config
  def __init__(self, config: dict[str, Any], settings: Settings | None = None):
      self.config = config
      self.settings = settings or Settings()

  # NEW: Using msgspec config
  from msgspec import Struct

  class TrainerConfig(Struct, frozen=True):
      model_type: str
      feature_config: dict
      training_params: dict

  def __init__(self, config: TrainerConfig):
      self.config = config
  ```

#### XGBoostTrainer

- **OLD**: `OLD/trade/nautilus_ml/training/train_xgboost.py` (1823 lines!)
- **NEW**: `ml/training/xgboost_trainer.py`
- **Key Changes**:
  - Extract MLflow operations to separate cold-path module
  - Simplify to core training logic
  - Move SHAP analysis to optional post-processing
  - Reduce to ~500 lines of core functionality

#### LightGBMTrainer

- **OLD**: `OLD/trade/nautilus_ml/training/train_lightgbm_unified.py`
- **NEW**: `ml/training/lightgbm_trainer.py`
- **Similar simplification as XGBoost**

### 2. Actor Components

#### GenericInferenceActor

- **OLD**: `OLD/trade/nautilus_ml/actors/generic_inference_actor.py`
- **NEW**: `ml/actors/ml_inference_actor.py`
- **Critical Changes**:

```python
# OLD: Custom MLPrediction class
class MLPrediction(Data):
    def __init__(self, instrument_id, model_name, ...):
        self.instrument_id = instrument_id
        # Complex initialization

# NEW: Simplified Nautilus-compatible
class MLSignal(Data):
    def __init__(self, instrument_id: InstrumentId,
                 prediction: int, probability: float,
                 ts_event: int, ts_init: int):
        self.instrument_id = instrument_id
        self.prediction = prediction
        self.probability = probability
        self._ts_event = ts_event
        self._ts_init = ts_init
```

#### Message Publishing Pattern

```python
# OLD
self.publish_data(type(signal), signal)

# NEW
self.publish_signal(
    signal,
    topic=f"ml.signals.{instrument_id}"
)
```

### 3. Strategy Components

#### EnsembleMLStrategy

- **OLD**: `OLD/trade/nautilus_ml/strategies/ensemble_ml_strategy.py`
- **NEW**: `ml/strategies/ml_ensemble_strategy.py`
- **Major Changes**:

```python
# OLD: Custom message types
from ..messages.portfolio import PortfolioTarget

# NEW: Standard Nautilus patterns
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.enums import OrderSide

# Direct order execution instead of portfolio targets
def _execute_signal(self, signal: MLSignal):
    if signal.prediction == 1 and signal.probability > 0.7:
        order = self.order_factory.market(
            instrument_id=signal.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self._calculate_position_size(signal)
        )
        self.submit_order(order)
```

### 4. Feature Engineering

#### FeatureEngineerV2

- **OLD**: `OLD/trade/nautilus_ml/features/feature_engineering.py`
- **NEW**: `ml/features/feature_engine.py`
- **Critical Requirement**: EXACT feature parity

```python
# NEW: Add parity validation
class FeatureEngine:
    def validate_parity(self, training_features: np.ndarray,
                       inference_features: np.ndarray) -> bool:
        """Validate feature parity within tolerance."""
        return np.allclose(training_features, inference_features,
                          rtol=1e-10, atol=1e-10)
```

### 5. Data Loading

#### UnifiedNautilusDataLoader

- **OLD**: `OLD/trade/nautilus_ml/data/unified_loader.py`
- **NEW**: `ml/data/catalog_loader.py`
- **Simplification**: Remove multi-backend support, focus on ParquetDataCatalog

### 6. Model Registry

#### ModelRegistry & MLflowManager

- **OLD**: `OLD/trade/nautilus_ml/registry/model_registry.py`
- **NEW**: `ml/models/registry.py`
- **Change**: Keep for cold path only, remove from hot path

### 7. Configuration

#### ML Config Classes

- **OLD**: Using Pydantic throughout
- **NEW**: msgspec.Struct everywhere

```python
# Example conversion
# OLD
from pydantic import BaseModel
class MLActorConfig(BaseModel):
    model_name: str
    update_frequency: float = 60.0

# NEW
from msgspec import Struct
class MLActorConfig(Struct, frozen=True):
    model_name: str
    update_frequency: float = 60.0
```

## Dependency Changes

### Remove Dependencies

- `pydantic` → `msgspec`
- `river` (not used in core)
- `mlflow` (from hot path)
- Complex monitoring libraries

### Keep Dependencies

- `numpy` (hot path)
- `polars` (cold path only)
- `xgboost`, `lightgbm` (training)
- `mlflow` (cold path only)

## File Size Reduction Strategy

| Component | OLD Lines | Target Lines | How |
|-----------|-----------|--------------|-----|
| XGBoostTrainer | 1823 | 500 | Extract MLflow, SHAP, visualization |
| GenericInferenceActor | 699 | 300 | Remove offline mode, simplify |
| EnsembleMLStrategy | 513 | 250 | Direct execution, remove portfolio layer |
| FeatureEngineering | 800+ | 400 | Core features only |

## Testing Migration

### Test Structure

```
ml/tests/
├── unit/
│   ├── test_actors.py
│   ├── test_strategies.py
│   └── test_trainers.py
├── integration/
│   ├── test_ml_pipeline.py
│   └── test_message_flow.py
└── performance/
    ├── test_inference_latency.py
    └── test_feature_computation.py
```

### Critical Tests to Port

1. `test_feature_parity.py` - MUST PASS
2. `test_ml_pipeline_simple.py` - Core flow
3. `test_ensemble_strategy.py` - Signal aggregation

## Migration Priority

### Phase 1 (Critical Path)

1. Base classes and configs
2. Feature engineering (exact parity)
3. Basic inference actor
4. Simple ML strategy

### Phase 2 (Core ML)

1. XGBoost trainer (simplified)
2. Model registry (cold path)
3. Ensemble strategy

### Phase 3 (Advanced)

1. LightGBM trainer
2. Neural trainers (if needed)
3. Advanced actors

### Phase 4 (Nice to Have)

1. Monitoring integration
2. Advanced visualizations
3. AutoML features

## Code Quality Requirements

### Every Module Must Have

1. Type hints (mypy clean)
2. Docstrings (Google style)
3. Unit tests (≥90% coverage)
4. Integration test
5. Performance benchmark

### Example Module Structure

```python
"""Module docstring explaining purpose."""

from __future__ import annotations

from typing import TYPE_CHECKING

from msgspec import Struct
import numpy as np

from nautilus_trader.common.actor import Actor
from nautilus_trader.model.data import Bar

if TYPE_CHECKING:
    from nautilus_trader.common.clock import Clock

class MLActor(Actor):
    """
    Actor for ML inference.

    Parameters
    ----------
    config : MLActorConfig
        The actor configuration

    """

    def __init__(self, config: MLActorConfig) -> None:
        super().__init__(config)
        # Implementation
```

## Migration Execution Plan

### Week 1: Foundation

- [ ] Create directory structure
- [ ] Base classes with msgspec
- [ ] Core data types
- [ ] Basic tests

### Week 2: Features

- [ ] Port FeatureEngine
- [ ] Implement parity tests
- [ ] Feature actor

### Week 3: Inference

- [ ] Port inference actor
- [ ] Basic ML strategy
- [ ] Message flow tests

### Week 4: Training

- [ ] Simplified XGBoost trainer
- [ ] Model registry (cold path)
- [ ] End-to-end test

### Week 5: Polish

- [ ] Documentation
- [ ] Examples
- [ ] Performance tuning

## Success Metrics

1. **Code Reduction**: 50% fewer lines
2. **Test Coverage**: ≥90% for all modules
3. **Performance**: <5ms inference
4. **Type Safety**: 0 mypy errors
5. **Feature Parity**: <1e-10 tolerance
