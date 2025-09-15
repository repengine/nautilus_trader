# ruff: noqa: RUF022
#!/usr/bin/env python3

"""
ML Pipeline Orchestration Module.

This module provides end-to-end pipeline orchestration components for batch ML operations
in Nautilus Trader. All pipelines operate in the cold path and compose existing CLIs
and APIs into typed, testable workflows suitable for long-running services or batch jobs.

## Core Pipeline Components

### End-to-End TFT Pipeline
- `tft_train_distill`: Complete pipeline from dataset building through teacher training to student distillation
- Composes dataset building, teacher training, and knowledge distillation in a single workflow
- Supports feature registry integration and model promotion workflows

### Dataset Build Orchestration
- `build_runner`: Multi-symbol dataset build orchestration with local concurrency
- `BuildConfig`: Configuration for batch dataset building operations
- `BuildTask`: Unit of work for dataset building operations
- `BuildWindow`: Time window configuration for historical data processing

## Universal ML Architecture Patterns Compliance

This module implements the Universal ML Architecture Patterns:

1. **Pattern 1: N/A** - Pipelines don't inherit from BaseMLInferenceActor (cold path only)
2. **Pattern 2: Protocol-First Interface Design** - Uses typed configuration and protocols
3. **Pattern 3: Hot/Cold Path Separation** - Strictly cold path operations only
4. **Pattern 4: Progressive Fallback Chains** - Graceful handling of external dependencies
5. **Pattern 5: Centralized Metrics Bootstrap** - Uses ml.common.metrics_bootstrap for monitoring

## Performance Characteristics

- **Cold Path Only**: No latency constraints, optimized for reliability over speed
- **Batch Processing**: Designed for large-scale dataset building and model training
- **Concurrent Execution**: Support for parallel processing across multiple symbols
- **Resumable Operations**: Progress tracking and resumable pipeline execution
- **Resource Management**: Configurable resource limits and timeout handling

## Usage Examples

### End-to-End TFT Pipeline
```python
from ml.pipelines import run_tft_train_distill_pipeline

# Complete TFT workflow from data to distilled model
result = run_tft_train_distill_pipeline([
    "--data_dir", "data/tier1",
    "--symbols", "SPY,QQQ",
    "--out_dir", "ml_out/tft_pipeline",
    "--train_teacher",
    "--teacher_model_id", "tft_teacher_v1",
    "--student_model_id", "tft_student_v1",
    "--model_registry_dir", "models/registry"
])
```

### Multi-Symbol Dataset Building
```python
from ml.pipelines import execute_build_runner, BuildConfig, BuildWindow

# Configure multi-symbol build with time windows
config = BuildConfig(
    data_dir=Path("data/tier1"),
    out_dir=Path("datasets/multi_symbol"),
    symbols=["SPY", "QQQ", "IWM"],
    window=BuildWindow(days_back=30),
    workers=4,
    include_macro=True,
    include_micro=True
)

# Execute with progress tracking
results = execute_build_runner(config)
print(f"Built {results['succeeded']}/{results['total']} datasets")
```

### Configuration-Driven Building
```python
from ml.pipelines import load_build_config, execute_build_runner

# Load from JSON/TOML configuration file
config = load_build_config(Path("build_config.json"))
results = execute_build_runner(config)
```

## Integration Points

- **ml.api.data**: Cold path dataset building API
- **ml.cli.***: Individual CLI components for orchestration
- **ml.registry.***: Feature and model registry integration
- **ml.training.***: Teacher training and distillation workflows
- **ml.common.metrics_bootstrap**: Centralized metrics collection

## Configuration Management

All pipelines use immutable dataclass configurations with frozen=True:

- Type-safe configuration with full annotations
- Environment variable integration where appropriate
- JSON/TOML configuration file support
- Progressive configuration resolution with fallbacks

## Progress Tracking and Monitoring

- **Progress Logging**: JSONL progress logs for resumable operations
- **Metrics Collection**: Prometheus metrics for pipeline execution
- **Error Handling**: Structured error reporting with retry policies
- **Health Monitoring**: Component health checks and dependency validation

## Non-Goals (Cold Path Only)

- Real-time pipeline execution (use actors for hot path operations)
- Sub-millisecond latency operations (use pre-allocated buffers)
- Hot path feature computation (use ml.features for real-time features)
- Trading signal generation (use ml.actors and ml.strategies)

## Security Considerations

- No pickle/joblib usage in pipeline components
- Safe configuration parsing and validation
- Environment variable sanitization
- Graceful error handling with structured logging

## Dependencies

- **Core**: pathlib, dataclasses, concurrent.futures
- **Configuration**: json, tomllib (Python 3.11+) or tomli
- **ML Components**: ml.api, ml.cli, ml.registry, ml.training
- **Monitoring**: ml.common.metrics_bootstrap

See Also
--------
ml.orchestration : Higher-level orchestration with scheduling
ml.deployment : Container deployment and service orchestration
ml.api : Public API facades for cold path operations
ml.cli : Individual command-line interfaces
"""

from __future__ import annotations

# Configuration classes for pipeline orchestration
from ml.pipelines.build_runner import BuildConfig
from ml.pipelines.build_runner import BuildTask
from ml.pipelines.build_runner import BuildWindow

# Core pipeline execution functions
from ml.pipelines.build_runner import execute as execute_build_runner
from ml.pipelines.build_runner import load_config as load_build_config
from ml.pipelines.build_runner import plan_tasks as plan_build_tasks
from ml.pipelines.tft_train_distill import main as run_tft_train_distill_pipeline


# ============================================================================
# PUBLIC API SURFACE
# ============================================================================

"""
ruff: noqa: RUF022
"""

__all__ = [
    "BuildConfig",
    "BuildTask",
    "BuildWindow",
    "execute_build_runner",
    "load_build_config",
    "plan_build_tasks",
    "run_tft_train_distill_pipeline",
]
