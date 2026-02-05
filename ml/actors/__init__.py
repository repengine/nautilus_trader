"""
ML actors for real-time inference in Nautilus Trader.

This module provides production-ready ML actors that follow the Universal ML Architecture
Patterns and maintain sub-millisecond performance requirements for hot path operations.

## Core Architecture

All ML actors MUST inherit from `BaseMLInferenceActor` which implements:
- Universal Pattern 1: Mandatory 4-Store + 4-Registry Integration
- Universal Pattern 2: Protocol-First Interface Design
- Universal Pattern 3: Hot/Cold Path Separation
- Universal Pattern 4: Progressive Fallback Chains
- Universal Pattern 5: Centralized Metrics Bootstrap

## Hot Path vs Cold Path

**Hot Path** (real-time, <5ms P99):
- `BaseMLInferenceActor` - Core inference base class
- `MLSignalActor` - Production signal generation
- `EnhancedMLInferenceActor` - Advanced inference features
- `MLSignal` - Signal data type

**Cold Path** (training, analytics, I/O):
- Model loading and configuration
- Registry management and strategy adaptation
- Background services and metrics collection

## Essential Classes

The public API focuses on essential components needed for ML inference:

- `BaseMLInferenceActor`: Mandatory base class implementing all Universal Patterns
- `MLSignalActor`: Production-ready signal generation with configurable strategies
- `EnhancedMLInferenceActor`: Advanced actor with hot-reload and optimization features
- `MLSignal`: Core data type for ML signals (replaces legacy types)

## Configuration

- `MLSignalActorConfig`: Primary configuration for signal actors
- `OptimizationConfig`: Performance optimization settings
- `StrategyConfig`: Signal generation parameters
- `SignalPolicy` (alias): Actor-side decision policy type used to map predictions to `MLSignal`

## Signal Policies (Actor-side)

Signal generation follows a policy pattern with built-in implementations:
- Threshold-based signaling
- Extremes detection
- Momentum analysis
- Ensemble methods
- Adaptive thresholds

Custom policies can be provided via model manifests without code changes
(Open/Closed Principle compliance).

## Performance Requirements

All hot path operations maintain strict performance targets:
- P99 feature computation: <500μs
- P99 model inference: <2ms
- P99 end-to-end signaling: <5ms
- Zero allocations in hot path
- Memory stability over 24h operation

## Examples

```python
from ml.actors import BaseMLInferenceActor, MLSignalActor, MLSignalActorConfig

# Basic signal actor configuration
config = MLSignalActorConfig(
    model_path="model.onnx",
    prediction_threshold=0.75,
    signal_strategy="adaptive"
)

# Create and use signal actor
actor = MLSignalActor(config)
# Actor automatically initializes all required stores and registries
```

"""

from ml.actors.base import BaseMLInferenceActor
from ml.actors.base import MLSignal
from ml.actors.base import PickleMLInferenceActor as _PickleMLInferenceActor  # noqa: F401
from ml.actors.enhanced import EnhancedMLInferenceActor
from ml.actors.signal import AdaptiveSignal
from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.actors.signal import OptimizationLevel
from ml.actors.signal import SignalPolicy
from ml.actors.signal import SignalStrategy
from ml.actors.signal import create_signal_actor
from ml.config.actors import OptimizationConfig
from ml.config.actors import StrategyConfig

from . import ml_domain_events as ml_domain_events


__all__ = [
    "AdaptiveSignal",
    "BaseMLInferenceActor",
    "EnhancedMLInferenceActor",
    "MLSignal",
    "MLSignalActor",
    "MLSignalActorConfig",
    "OptimizationConfig",
    "OptimizationLevel",
    "SignalPolicy",
    "SignalStrategy",
    "StrategyConfig",
    "create_signal_actor",
]

# Internal implementation note:
# The following components are intentionally not exported in __all__:
# - _PickleMLInferenceActor: Deprecated security stub
# - ONNXMLInferenceActor: Internal implementation detail
# - Adapter classes: Implementation details for strategy loading
# - Model loading utilities: Cold path operations
# - Actor services: Internal dependency injection
# - Domain events: Background messaging infrastructure
#
# These follow the principle of exposing only essential public APIs
# while keeping implementation details private.
