# TDD Strategy for Registry Triad Integration

## Overview

This document outlines a Test-Driven Development approach to fully integrate the Model, Feature, and Strategy registries throughout the ML system. We will refactor aggressively, create proper abstractions, and ensure clean integration without backward compatibility debt.

## Core Principles

1. **No Backward Compatibility**: Clean refactor as we go, fix all integration points
2. **Proper Abstractions**: Create base classes and interfaces where needed
3. **Functional Testing**: Iterate until systems work correctly
4. **Prometheus Metrics**: Expose all registry operations as metrics
5. **Clean Naming**: No versioned files like `unified_final_v2.py` - refactor in place

## Phase 1: Registry Manager and Refactor Existing Base Classes

### Step 1.1: Refactor Existing Base Classes

We'll add registry functionality directly to the existing base classes instead of creating new ones.

#### File: `ml/training/base.py` (REFACTOR IN PLACE)

```python
"""Base trainer class for ML model training with registry integration."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ml._imports import HAS_MLFLOW, HAS_ONNX, HAS_OPTUNA, HAS_POLARS
from ml.config.base import MLFeatureConfig, MLTrainingConfig
from ml.registry.manager import RegistryManager
from ml.registry import ModelManifest, ModelRole, DataRequirements
from ml.monitoring.collectors.registry import REGISTRY_METRICS

logger = logging.getLogger(__name__)


class BaseMLTrainer(ABC):
    """
    Base class for ML model trainers with registry integration.

    Inherits existing functionality and adds registry management.
    """

    def __init__(self, config: MLTrainingConfig) -> None:
        """Initialize with registry support."""
        self._config = config
        self._feature_config = config.feature_config or MLFeatureConfig()

        # Registry integration
        self._registry_manager = None
        if hasattr(config, 'registry_path') and config.registry_path:
            self._registry_manager = RegistryManager(Path(config.registry_path))

        # Training state (existing)
        self._model: Any = None
        self._feature_names: list[str] = []
        self._training_metrics: dict[str, Any] = {}
        self._is_fitted = False

        # Optional components (existing)
        self._mlflow_run_id: str | None = None
        self._optuna_study: Any | None = None
        self._cv_results: list[dict[str, float]] = []

    @property
    def model_registry(self):
        """Access model registry."""
        if self._registry_manager:
            return self._registry_manager.model_registry
        return None

    @property
    def feature_registry(self):
        """Access feature registry."""
        if self._registry_manager:
            return self._registry_manager.feature_registry
        return None

    def train_and_register(self, *args, **kwargs) -> str:
        """Train model and register in registry."""
        if not self._registry_manager:
            raise RuntimeError("Registry not configured. Set registry_path in config.")

        # Train using existing train method
        result = self.train(*args, **kwargs)

        # Save model
        model_path = self.save_model()

        # Create and register manifest
        manifest = self.create_manifest()
        model_id = self.model_registry.register_model(
            model_path=model_path,
            manifest=manifest,
            auto_deploy=getattr(self._config, "auto_deploy", False)
        )

        # Emit metrics
        REGISTRY_METRICS.model_registered.labels(
            model_id=model_id,
            role=manifest.role.value,
            architecture=self.get_architecture()
        ).inc()

        return model_id

    @abstractmethod
    def create_manifest(self) -> ModelManifest:
        """Create model manifest for registry."""
        ...

    @abstractmethod
    def get_architecture(self) -> str:
        """Get model architecture name."""
        ...

    # Keep all existing abstract methods
    @abstractmethod
    def train(self, data: Any, validation_data: Any | None = None, **kwargs) -> dict[str, Any]:
        """Train the ML model (existing method)."""
        ...


#### File: `ml/models/loader.py` (REFACTOR IN PLACE)

```python
"""Production model loader with registry integration."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml._imports import HAS_LIGHTGBM, HAS_ONNX, HAS_XGBOOST
from ml.models.factory import create_model_wrapper
from ml.registry.manager import RegistryManager
from ml.monitoring.collectors.registry import REGISTRY_METRICS


class ModelLoader(ABC):
    """Abstract base class for model loading strategies."""

    def __init__(self, registry_path: Path | None = None):
        """Initialize with optional registry."""
        self._registry_manager = None
        if registry_path:
            self._registry_manager = RegistryManager(registry_path)

    @abstractmethod
    def load_model(self, path: str) -> tuple[Any, dict[str, Any]]:
        """Load model and return model with metadata."""
        ...

    def load_from_registry(self, model_id: str) -> Any:
        """Load model from registry by ID."""
        if not self._registry_manager:
            raise RuntimeError("Registry not configured")

        # Use registry's caching and validation
        model = self._registry_manager.model_registry.load_model(model_id)

        if model is None:
            raise ValueError(f"Model {model_id} not found in registry")

        # Validate deployment status
        model_info = self._registry_manager.model_registry.get_model(model_id)
        if model_info.deployment_status == "retired":
            raise ValueError(f"Cannot load retired model {model_id}")

        # Emit metrics
        REGISTRY_METRICS.model_loaded.labels(model_id=model_id).inc()

        return model


class ProductionModelLoader(ModelLoader):
    """Production-grade model loader with registry support."""

    def __init__(self, registry_path: Path | None = None):
        """Initialize with optional registry."""
        super().__init__(registry_path)

    def load_model(self, path: str) -> tuple[Any, dict[str, Any]]:
        """Load model with format detection (existing functionality)."""
        # Keep existing implementation
        ...

    def load_model_by_id(self, model_id: str) -> Any:
        """Load model from registry."""
        return self.load_from_registry(model_id)


#### File: `ml/strategies/base.py` (REFACTOR IN PLACE)

```python
"""Base ML strategy with registry integration."""

from nautilus_trader.trading.strategy import Strategy
from ml.registry.manager import RegistryManager
from ml.monitoring.collectors.registry import REGISTRY_METRICS


class BaseMLStrategy(Strategy, ABC):
    """
    Base class for ML-driven trading strategies with registry support.
    """

    def __init__(self, config: MLStrategyConfig) -> None:
        """Initialize with registry support."""
        super().__init__(config)
        self._config = config

        # Registry integration
        self._registry_manager = None
        if hasattr(config, 'registry_path') and config.registry_path:
            self._registry_manager = RegistryManager(Path(config.registry_path))

        # Existing state
        self._active_positions = 0
        self._pending_orders = 0
        self._last_signal_time = 0
        # ... rest of existing init

    def on_start(self) -> None:
        """Initialize strategy with registry validation."""
        self.log.info(f"Starting {self.__class__.__name__}")

        # Validate requirements if registry configured
        if self._registry_manager and hasattr(self._config, 'strategy_id'):
            if not self._validate_requirements():
                raise RuntimeError(f"Strategy requirements not met")

            # Load model from registry if configured
            if hasattr(self._config, 'model_id'):
                self._load_model_from_registry(self._config.model_id)

        # Continue with existing on_start logic
        # Subscribe to ML signals, etc.
        ...

    def _validate_requirements(self) -> bool:
        """Validate strategy requirements via registry."""
        if not self._registry_manager:
            return True  # No registry, skip validation

        strategy_id = self._config.strategy_id
        strategy_info = self._registry_manager.strategy_registry.get_strategy(strategy_id)

        if not strategy_info:
            self.log.warning(f"Strategy {strategy_id} not found in registry")
            return True  # Allow unregistered strategies

        # Check model requirements
        if strategy_info.manifest.required_models:
            for model_id in strategy_info.manifest.required_models:
                if not self._registry_manager.model_registry.get_model(model_id):
                    self.log.error(f"Required model {model_id} not found")
                    return False

        # Check feature requirements
        if strategy_info.manifest.required_features:
            for feature_id in strategy_info.manifest.required_features:
                if not self._registry_manager.feature_registry.get_feature_set(feature_id):
                    self.log.error(f"Required feature {feature_id} not found")
                    return False

        return True

    def _load_model_from_registry(self, model_id: str) -> None:
        """Load model from registry."""
        if not self._registry_manager:
            return

        try:
            self._model = self._registry_manager.model_registry.load_model(model_id)
            REGISTRY_METRICS.model_loaded.labels(model_id=model_id).inc()
            self.log.info(f"Loaded model {model_id} from registry")
        except Exception as e:
            self.log.error(f"Failed to load model {model_id}: {e}")
            raise

### Step 1.2: Implement Registry Manager

#### File: `ml/registry/manager.py`

```python
"""Registry manager that coordinates all three registries."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
import logging

from ml.registry import LocalModelRegistry, ModelRole, DeploymentStatus
from ml.registry.feature_registry import LocalFeatureRegistry
from ml.registry.strategy_registry import LocalStrategyRegistry
from ml.monitoring.collectors.registry import REGISTRY_METRICS


logger = logging.getLogger(__name__)


class RegistryManager:
    """
    Coordinates Model, Feature, and Strategy registries.

    Ensures system coherence and provides cross-registry operations.
    """

    def __init__(self, base_path: Path):
        """Initialize all three registries."""
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize registries
        self.model_registry = LocalModelRegistry(
            base_path / "models",
            cache_size=20,
            batch_save_interval=0.1
        )
        self.feature_registry = LocalFeatureRegistry(base_path / "features")
        self.strategy_registry = LocalStrategyRegistry(base_path / "strategies")

        # Thread lock for coherence operations
        self._lock = threading.RLock()

        # Initialize metrics
        self._init_metrics()

        logger.info(f"Initialized RegistryManager at {base_path}")

    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        # Set initial gauge values
        REGISTRY_METRICS.total_models.set(len(self.model_registry.get_all_models()))
        REGISTRY_METRICS.total_features.set(len(self.feature_registry.list_all()))
        REGISTRY_METRICS.total_strategies.set(len(self.strategy_registry._load_registry()))

    def validate_system_coherence(self) -> List[str]:
        """
        Check for system-wide coherence issues.

        Returns list of issue descriptions.
        """
        issues = []

        with self._lock:
            # Check for orphaned student models
            for model in self.model_registry.get_models_by_role(ModelRole.STUDENT):
                if model.manifest.parent_id:
                    parent = self.model_registry.get_model(model.manifest.parent_id)
                    if not parent:
                        issues.append(
                            f"Orphaned student model {model.manifest.model_id}: "
                            f"parent {model.manifest.parent_id} not found"
                        )
                    elif model.manifest.model_id not in parent.manifest.children_ids:
                        issues.append(
                            f"Inconsistent lineage: {model.manifest.model_id} claims "
                            f"parent {model.manifest.parent_id} but parent doesn't list it"
                        )

            # Check for strategies with missing requirements
            registry_data = self.strategy_registry._load_registry()
            for strategy_id in registry_data:
                strategy = self.strategy_registry.get_strategy(strategy_id)
                if strategy:
                    # Check models
                    if strategy.manifest.required_models:
                        for model_id in strategy.manifest.required_models:
                            if not self.model_registry.get_model(model_id):
                                issues.append(
                                    f"Strategy {strategy_id} missing required model {model_id}"
                                )

                    # Check features
                    if strategy.manifest.required_features:
                        for feature_id in strategy.manifest.required_features:
                            if not self.feature_registry.get_feature_set(feature_id):
                                issues.append(
                                    f"Strategy {strategy_id} missing required feature {feature_id}"
                                )

            # Check for feature compatibility
            for model in self.model_registry.get_all_models():
                if model.manifest.feature_schema_hash:
                    # Find features with same hash
                    matching_features = self.feature_registry.resolve_by_schema_hash(
                        model.manifest.feature_schema_hash
                    )
                    if not matching_features and model.deployment_status == DeploymentStatus.ACTIVE:
                        issues.append(
                            f"Active model {model.manifest.model_id} has no matching feature sets "
                            f"for schema hash {model.manifest.feature_schema_hash}"
                        )

        # Emit metrics
        REGISTRY_METRICS.coherence_issues.set(len(issues))

        return issues

    def cascade_deprecation(self, model_id: str) -> Dict[str, List[str]]:
        """
        Handle cascading effects of model deprecation.

        Returns dict of affected components.
        """
        affected = {"strategies": [], "child_models": []}

        with self._lock:
            # Find dependent strategies
            registry_data = self.strategy_registry._load_registry()
            for strategy_id in registry_data:
                strategy = self.strategy_registry.get_strategy(strategy_id)
                if strategy and strategy.manifest.required_models:
                    if model_id in strategy.manifest.required_models:
                        affected["strategies"].append(strategy_id)

            # Find child models (e.g., students of a teacher)
            model = self.model_registry.get_model(model_id)
            if model:
                affected["child_models"] = model.manifest.children_ids.copy()

            # Retire the model
            self.model_registry.retire_model(model_id)

            # Emit metrics
            REGISTRY_METRICS.model_retired.labels(model_id=model_id).inc()
            REGISTRY_METRICS.cascade_affected_strategies.labels(model_id=model_id).set(
                len(affected["strategies"])
            )

        logger.warning(
            f"Deprecated model {model_id}, affected: "
            f"{len(affected['strategies'])} strategies, "
            f"{len(affected['child_models'])} child models"
        )

        return affected

    def cascade_feature_deprecation(self, feature_set_id: str) -> Dict[str, List[str]]:
        """
        Handle cascading effects of feature deprecation.

        Returns dict of affected components.
        """
        affected = {"models": [], "strategies": []}

        with self._lock:
            # Get feature info
            feature = self.feature_registry.get_feature_set(feature_set_id)
            if not feature:
                return affected

            # Find models using this feature schema
            for model in self.model_registry.get_all_models():
                if model.manifest.feature_schema_hash == feature.schema_hash:
                    affected["models"].append(model.manifest.model_id)

            # Find strategies requiring this feature
            registry_data = self.strategy_registry._load_registry()
            for strategy_id in registry_data:
                strategy = self.strategy_registry.get_strategy(strategy_id)
                if strategy and strategy.manifest.required_features:
                    if feature_set_id in strategy.manifest.required_features:
                        affected["strategies"].append(strategy_id)

            # Deprecate the feature
            self.feature_registry.deprecate(feature_set_id, "Cascaded deprecation")

            # Emit metrics
            REGISTRY_METRICS.feature_deprecated.labels(feature_set_id=feature_set_id).inc()

        return affected

    def check_compatibility(self, model_id: str, feature_set_id: str) -> bool:
        """Check if model and feature set are compatible."""
        model = self.model_registry.get_model(model_id)
        feature = self.feature_registry.get_feature_set(feature_set_id)

        if not model or not feature:
            return False

        # Check schema hash
        if model.manifest.feature_schema_hash != feature.schema_hash:
            return False

        # Check data requirements
        if model.manifest.data_requirements != feature.data_requirements:
            return False

        return True

    def get_full_lineage(self, strategy_id: str) -> Dict[str, Any]:
        """Get complete lineage from strategy through models to features."""
        lineage = {
            "strategy": strategy_id,
            "models": [],
            "features": []
        }

        strategy = self.strategy_registry.get_strategy(strategy_id)
        if not strategy:
            return lineage

        # Get models and their lineage
        if strategy.manifest.required_models:
            for model_id in strategy.manifest.required_models:
                model_lineage = self.model_registry.get_model_lineage(model_id)
                lineage["models"].extend([m.manifest.model_id for m in model_lineage])

        # Get features and their lineage
        if strategy.manifest.required_features:
            for feature_id in strategy.manifest.required_features:
                feature_lineage = self.feature_registry.get_lineage(feature_id)
                lineage["features"].extend([f.feature_set_id for f in feature_lineage])

        return lineage

    def check_deployment_readiness(self, strategy_id: str) -> tuple[bool, List[str]]:
        """
        Check if strategy has all requirements for deployment.

        Returns (is_ready, list_of_missing_requirements).
        """
        missing = []

        strategy = self.strategy_registry.get_strategy(strategy_id)
        if not strategy:
            return False, ["Strategy not found"]

        # Check models
        if strategy.manifest.required_models:
            for model_id in strategy.manifest.required_models:
                model = self.model_registry.get_model(model_id)
                if not model:
                    missing.append(f"Model: {model_id}")
                elif model.deployment_status != DeploymentStatus.ACTIVE:
                    missing.append(f"Model not active: {model_id}")

        # Check features
        if strategy.manifest.required_features:
            for feature_id in strategy.manifest.required_features:
                feature = self.feature_registry.get_feature_set(feature_id)
                if not feature:
                    missing.append(f"Feature: {feature_id}")
                elif feature.stage.value != "prod":
                    missing.append(f"Feature not in prod: {feature_id}")

        return len(missing) == 0, missing

    def get_metrics(self) -> Dict[str, Any]:
        """Get registry metrics for monitoring."""
        return {
            "total_models": len(self.model_registry.get_all_models()),
            "active_models": len(self.model_registry.get_active_models()),
            "total_features": len(self.feature_registry.list_all()),
            "total_strategies": len(self.strategy_registry._load_registry()),
            "coherence_issues": len(self.validate_system_coherence()),
            "cache_size": len(self.model_registry._model_cache)
        }
```

### Step 1.3: Prometheus Metrics Collector

#### File: `ml/monitoring/collectors/registry.py`

```python
"""Prometheus metrics for registry operations."""

from prometheus_client import Counter, Gauge, Histogram, Summary
import time


class RegistryMetrics:
    """Registry-specific Prometheus metrics."""

    def __init__(self):
        # Counters
        self.model_registered = Counter(
            'ml_registry_model_registered_total',
            'Total models registered',
            ['model_id', 'role', 'architecture']
        )

        self.model_loaded = Counter(
            'ml_registry_model_loaded_total',
            'Total model load operations',
            ['model_id']
        )

        self.model_retired = Counter(
            'ml_registry_model_retired_total',
            'Total models retired',
            ['model_id']
        )

        self.feature_deprecated = Counter(
            'ml_registry_feature_deprecated_total',
            'Total features deprecated',
            ['feature_set_id']
        )

        self.strategy_started = Counter(
            'ml_registry_strategy_started_total',
            'Total strategy startups',
            ['strategy_id', 'model_id']
        )

        self.hot_reload_success = Counter(
            'ml_registry_hot_reload_success_total',
            'Successful hot reloads',
            ['target', 'old_model', 'new_model']
        )

        self.hot_reload_failure = Counter(
            'ml_registry_hot_reload_failure_total',
            'Failed hot reloads',
            ['target', 'reason']
        )

        # Gauges
        self.total_models = Gauge(
            'ml_registry_models_total',
            'Total models in registry'
        )

        self.total_features = Gauge(
            'ml_registry_features_total',
            'Total feature sets in registry'
        )

        self.total_strategies = Gauge(
            'ml_registry_strategies_total',
            'Total strategies in registry'
        )

        self.coherence_issues = Gauge(
            'ml_registry_coherence_issues',
            'Number of system coherence issues'
        )

        self.model_performance = Gauge(
            'ml_registry_model_performance',
            'Model performance metrics',
            ['model_id', 'metric']
        )

        self.cascade_affected_strategies = Gauge(
            'ml_registry_cascade_affected_strategies',
            'Strategies affected by model deprecation',
            ['model_id']
        )

        # Histograms
        self.registration_duration = Histogram(
            'ml_registry_registration_duration_seconds',
            'Time to register a model',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )

        self.loading_duration = Histogram(
            'ml_registry_loading_duration_seconds',
            'Time to load a model',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )

        self.validation_duration = Histogram(
            'ml_registry_validation_duration_seconds',
            'Time to validate system coherence',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )

        # Summary
        self.cache_hit_rate = Summary(
            'ml_registry_cache_hit_rate',
            'Model cache hit rate'
        )

    def time_registration(self):
        """Context manager for timing registration."""
        return self.registration_duration.time()

    def time_loading(self):
        """Context manager for timing loading."""
        return self.loading_duration.time()

    def time_validation(self):
        """Context manager for timing validation."""
        return self.validation_duration.time()


# Global metrics instance
REGISTRY_METRICS = RegistryMetrics()
```

## Phase 2: Refactor Training Modules

### Step 2.1: Update ModelExportMixin

#### File: `ml/training/model_exporter.py` (REFACTOR IN PLACE)

```python
"""Model export utilities with registry integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml._imports import HAS_XGBOOST
from ml.models import ModelType, detect_model_type
from ml.models.saver import convert_to_onnx, save_model_with_metadata
from ml.registry.manager import RegistryManager
from ml.registry import ModelManifest
from ml.monitoring.collectors.registry import REGISTRY_METRICS


class ModelExportMixin(ABC):
    """
    Mixin for model export with registry integration.

    Add registry support to existing export functionality.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with optional registry."""
        super().__init__(*args, **kwargs)
        self._registry_manager = None

        # Check if registry_path in kwargs or config
        if 'registry_path' in kwargs:
            self._registry_manager = RegistryManager(Path(kwargs['registry_path']))
        elif hasattr(self, '_config') and hasattr(self._config, 'registry_path'):
            if self._config.registry_path:
                self._registry_manager = RegistryManager(Path(self._config.registry_path))


    def save_for_production_with_registry(
        self,
        path: str | Path,
        manifest: ModelManifest | None = None,
        format: str = "auto",
        auto_deploy: bool = False,
    ) -> str:
        """
        Save model and register in registry.

        Extends existing save_for_production with registry support.
        """
        # Use existing save_for_production
        saved_path = self.save_for_production(path, format=format)

        # Register if registry configured
        if self._registry_manager and manifest:
            model_id = self._registry_manager.model_registry.register_model(
                model_path=saved_path,
                manifest=manifest,
                auto_deploy=auto_deploy
            )

            # Emit metrics
            REGISTRY_METRICS.model_registered.labels(
                model_id=model_id,
                role=manifest.role.value,
                architecture=manifest.architecture
            ).inc()

            return model_id

        return str(saved_path)

    # Keep existing abstract methods
    @abstractmethod
    def get_model(self) -> Any:
        """Get the trained model instance (existing)."""
        ...

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Get feature names (existing)."""
        ...

    @abstractmethod
    def get_training_metadata(self) -> dict[str, Any]:
        """Get training metadata (existing)."""
        ...
```

### Step 2.2: Refactor Student Distiller

#### File: `ml/training/student/lightgbm_student.py` (REFACTOR IN PLACE)

```python
"""LightGBM student distillation with registry integration."""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import lightgbm as lgb

from ml.training.base import BaseMLTrainer
from ml.registry import ModelRole, DataRequirements
from ml.monitoring.collectors.registry import REGISTRY_METRICS


class LightGBMStudentDistiller(BaseMLTrainer):
    """Student model distiller using LightGBM."""

    def __init__(
        self,
        config: MLTrainingConfig,
        teacher_id: str,
        feature_names: Optional[list[str]] = None,
        objective: str = "logit_mse",
        **kwargs
    ):
        # Pass config to parent BaseMLTrainer
        super().__init__(config, **kwargs)
        self.teacher_id = teacher_id
        self.feature_names = feature_names or []
        self.objective = objective
        self.model = None

    def get_model_role(self) -> ModelRole:
        return ModelRole.STUDENT

    def get_data_requirements(self) -> DataRequirements:
        return DataRequirements.L1_ONLY

    def get_architecture(self) -> str:
        return "LightGBM"

    def create_manifest(self) -> Any:
        """Override to add parent_id."""
        manifest = super().create_manifest()
        manifest.parent_id = self.teacher_id

        # Verify teacher exists and update its children
        teacher = self.model_registry.get_model(self.teacher_id)
        if teacher:
            if manifest.model_id not in teacher.manifest.children_ids:
                teacher.manifest.children_ids.append(manifest.model_id)
                # This will be saved when student is registered

        return manifest

    def train(self, X: np.ndarray, y_soft: np.ndarray) -> lgb.Booster:
        """Train student on soft labels from teacher."""
        # Validate inputs
        X = X.astype(np.float32)
        y_soft = y_soft.astype(np.float32)

        # Create LightGBM dataset
        train_data = lgb.Dataset(X, label=y_soft)

        # Training parameters
        params = {
            'objective': 'regression',  # Use MSE for distillation
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': 4,
            'force_col_wise': True,
            'seed': 42
        }

        # Train with timing
        with REGISTRY_METRICS.time_registration():
            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )

        # Compute metrics
        y_pred = self.model.predict(X, num_iteration=self.model.best_iteration)
        self.metrics = {
            'rmse': float(np.sqrt(np.mean((y_pred - y_soft) ** 2))),
            'best_iteration': self.model.best_iteration,
            'num_features': X.shape[1]
        }

        return self.model

    def export_onnx(self, model: lgb.Booster, output_path: Path) -> None:
        """Export LightGBM model to ONNX."""
        import onnxmltools
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
        import onnx

        # Convert to ONNX
        onnx_model = onnxmltools.convert_lightgbm(
            model,
            name='LightGBMStudent',
            initial_types=[('input', onnxmltools.FloatTensorType([None, len(self.feature_names)]))],
            target_opset=12
        )

        # Add metadata
        onnx_model.metadata_props.append(
            onnx.StringStringEntryProto(key='teacher_id', value=self.teacher_id)
        )
        onnx_model.metadata_props.append(
            onnx.StringStringEntryProto(key='objective', value=self.objective)
        )

        # Save
        onnx.save(onnx_model, str(output_path))
        self.model_path = output_path

    def compute_metrics(self, model: lgb.Booster, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Compute validation metrics."""
        import time

        # Inference latency
        start = time.perf_counter()
        for _ in range(100):
            _ = model.predict(X_val[:1], num_iteration=model.best_iteration)
        latency_ms = (time.perf_counter() - start) / 100 * 1000

        # Accuracy metrics
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)

        metrics = {
            'val_rmse': float(np.sqrt(np.mean((y_pred - y_val) ** 2))),
            'inference_latency_ms': latency_ms,
            'model_size_kb': self.model_path.stat().st_size / 1024 if self.model_path else 0
        }

        self.metrics.update(metrics)
        return metrics
```

## Phase 3: Functional Testing Iterations

### Test File: `ml/tests/functional/test_registry_integration.py`

```python
"""Functional tests for registry integration - iterate until working."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import Mock, patch

from ml.registry.manager import RegistryManager
from ml.training.student.lightgbm import LightGBMStudentDistiller
from ml.models.loader import ProductionModelLoader
from ml.strategies.ml_strategy import MLTradingStrategy
from ml.config.base import MLStrategyConfig, MLTrainingConfig
from ml.registry import ModelManifest, ModelRole, DataRequirements, DeploymentStatus


class TestRegistryIntegrationFunctional:
    """Functional tests - keep iterating until these pass."""

    def test_complete_workflow(self):
        """Test complete workflow from training to deployment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Initialize registry manager
            manager = RegistryManager(base_path)

            # Step 1: Register a mock teacher
            teacher_manifest = ModelManifest(
                model_id="teacher_1",
                role=ModelRole.TEACHER,
                data_requirements=DataRequirements.L1_L2_L3,
                architecture="TFT",
                feature_schema={"feat1": "float32", "feat2": "float32", "feat3": "float32"},
                feature_schema_hash="test_hash",
                children_ids=[]
            )

            teacher_path = base_path / "teacher.onnx"
            teacher_path.touch()

            teacher_id = manager.model_registry.register_model(
                teacher_path,
                teacher_manifest
            )

            assert teacher_id is not None

            # Step 2: Train and register student
            config = MLTrainingConfig(registry_path=str(base_path))
            distiller = LightGBMStudentDistiller(
                config=config,
                teacher_id=teacher_id,
                feature_names=["feat1", "feat2", "feat3"]
            )

            # Mock training data
            X_train = np.random.randn(100, 3).astype(np.float32)
            y_soft = np.random.randn(100, 1).astype(np.float32)

            # This should train and auto-register
            student_id = distiller.train_and_register(
                X_train, y_soft,
                output_path=base_path / "student.onnx"
            )

            assert student_id is not None

            # Step 3: Verify registration
            student_info = manager.model_registry.get_model(student_id)
            assert student_info is not None
            assert student_info.manifest.parent_id == teacher_id
            assert student_info.manifest.role == ModelRole.STUDENT

            # Step 4: Load model through registry
            loader = ProductionModelLoader(registry_path=base_path)
            model = loader.load_model(student_id)
            assert model is not None

            # Step 5: Validate system coherence
            issues = manager.validate_system_coherence()
            assert len(issues) == 0  # No coherence issues

            # Step 6: Check metrics were emitted
            metrics = manager.get_metrics()
            assert metrics["total_models"] == 2
            assert metrics["coherence_issues"] == 0

    def test_cascade_deprecation(self):
        """Test cascading effects of deprecation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            manager = RegistryManager(base_path)

            # Register teacher-student chain
            teacher_id = self._register_mock_model(manager, "teacher_1", role=ModelRole.TEACHER)
            student_id = self._register_mock_model(
                manager, "student_1", role=ModelRole.STUDENT, parent_id=teacher_id
            )

            # Register strategy using student
            strategy_id = self._register_mock_strategy(
                manager, "strategy_1", required_models=[student_id]
            )

            # Deprecate teacher
            affected = manager.cascade_deprecation(teacher_id)

            # Student should be in affected
            assert student_id in affected["child_models"]

            # Teacher should be retired
            teacher = manager.model_registry.get_model(teacher_id)
            assert teacher.deployment_status == DeploymentStatus.RETIRED

    def test_hot_reload(self):
        """Test hot reload functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            manager = RegistryManager(base_path)

            # Register and deploy initial model
            model_v1 = self._register_mock_model(manager, "model_v1")
            manager.model_registry.deploy_model(model_v1, "ml_signal_actor")

            # Register new version
            model_v2 = self._register_mock_model(manager, "model_v2")

            # Hot reload
            success = manager.model_registry.hot_reload_model(
                "ml_signal_actor", model_v2
            )

            assert success

            # v1 should be retired, v2 should be active
            v1_info = manager.model_registry.get_model(model_v1)
            v2_info = manager.model_registry.get_model(model_v2)

            assert v1_info.deployment_status == DeploymentStatus.RETIRED
            assert v2_info.deployment_status == DeploymentStatus.ACTIVE
            assert "ml_signal_actor" in v2_info.deployed_to

    def _register_mock_model(self, manager, model_id, role=ModelRole.INFERENCE, parent_id=None):
        """Helper to register mock model."""
        manifest = ModelManifest(
            model_id=model_id,
            role=role,
            data_requirements=DataRequirements.L1_ONLY,
            architecture="Mock",
            feature_schema={"feat1": "float32"},
            feature_schema_hash="test_hash",
            parent_id=parent_id,
            children_ids=[]
        )

        path = manager.base_path / f"{model_id}.onnx"
        path.touch()

        return manager.model_registry.register_model(path, manifest)

    def _register_mock_strategy(self, manager, strategy_id, required_models=None):
        """Helper to register mock strategy."""
        from ml.registry.strategy_registry import StrategyManifest, StrategyType, MarketRegime

        manifest = StrategyManifest(
            strategy_id=strategy_id,
            strategy_type=StrategyType.MOMENTUM,
            version="1.0.0",
            required_models=required_models or [],
            required_features=[],
            suitable_regimes=[MarketRegime.TRENDING_UP],
            instrument_types=["FX"],
            timeframe_range=("1m", "1h"),
            max_position_size=100000.0,
            max_leverage=3.0,
            max_drawdown=0.10,
            stop_loss_type="trailing",
            min_sharpe_ratio=1.5,
            min_win_rate=0.55,
            max_correlation_with_portfolio=0.7,
            incompatible_strategies=[],
            config_schema={},
            default_config={},
            backtest_metrics={},
            created_at=0.0,
            last_modified=0.0,
            author="test",
            description="test strategy"
        )

        path = manager.base_path / f"{strategy_id}.py"
        path.touch()

        return manager.strategy_registry.register_strategy(path, manifest)


class TestMetricsExport:
    """Test Prometheus metrics are properly exported."""

    @patch('ml.monitoring.collectors.registry.REGISTRY_METRICS')
    def test_metrics_on_registration(self, mock_metrics):
        """Test metrics are emitted on model registration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RegistryManager(Path(tmpdir))

            # Register a model
            self._register_model(manager)

            # Check counter was incremented
            mock_metrics.model_registered.labels.assert_called()
            mock_metrics.total_models.set.assert_called()

    @patch('ml.monitoring.collectors.registry.REGISTRY_METRICS')
    def test_metrics_on_loading(self, mock_metrics):
        """Test metrics are emitted on model loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RegistryManager(Path(tmpdir))

            # Register and load
            model_id = self._register_model(manager)
            loader = ProductionModelLoader(registry_path=Path(tmpdir))

            with pytest.raises(ValueError):  # Will fail but metrics should be attempted
                loader.load_model(model_id)

            mock_metrics.model_loaded.labels.assert_called()
```

## Phase 4: Refactoring Checklist

### Immediate Refactoring Tasks

1. **Delete Legacy Files**

   ```bash
   # Remove versioned/duplicate files
   rm ml/registry/unified_registry.py
   rm ml/registry/registry_v2.py
   rm ml/models/loader_old.py
   ```

2. **Update All Imports**

   ```python
   # Change all imports to use refactored base classes
   # All base classes now have registry support built-in
   from ml.training.base import BaseMLTrainer
   from ml.models.loader import ProductionModelLoader
   from ml.strategies.base import BaseMLStrategy
   ```

3. **Remove Backward Compatibility Code**

   ```python
   # Remove all fallback code like:
   # if not registry_available:
   #     return load_legacy_model()
   #
   # Just fail fast with clear errors
   ```

4. **Enforce Registry Usage**

   ```python
   # In all model loading code:
   def load_model(self, model_id: str):
       # No fallbacks - registry is required
       if not self._registry_manager:
           raise RuntimeError("Registry not initialized")

       model = self._registry_manager.model_registry.load_model(model_id)
       if not model:
           raise ValueError(f"Model {model_id} not found in registry")

       return model
   ```

## Success Criteria

### Functional Requirements

- [ ] All training modules use `RegistryAwareTrainer` base class
- [ ] All model loading goes through registry
- [ ] All strategies validate requirements via registry
- [ ] System coherence validation reports no issues
- [ ] Hot reload works without downtime
- [ ] Cascade deprecation identifies all affected components

### Metrics Requirements

- [ ] All registry operations emit Prometheus metrics
- [ ] Metrics exposed at `/metrics` endpoint
- [ ] Grafana dashboards show registry health
- [ ] Alert rules configured for coherence issues

### Code Quality Requirements

- [ ] No versioned files (e.g., `*_v2.py`, `*_final.py`)
- [ ] All components use proper base classes
- [ ] No backward compatibility code
- [ ] Clean abstractions throughout
- [ ] 90% test coverage on ml/ directory

## Monitoring Dashboard

### Grafana Dashboard Config

```yaml
# ml/monitoring/dashboards/registry.yaml
apiVersion: 1
providers:
  - name: 'ML Registry'
    folder: 'ML'
    type: file
    options:
      path: /var/lib/grafana/dashboards

dashboards:
  - name: "ML Registry Health"
    panels:
      - title: "Total Models"
        type: graph
        targets:
          - expr: ml_registry_models_total

      - title: "Coherence Issues"
        type: stat
        targets:
          - expr: ml_registry_coherence_issues

      - title: "Model Registration Rate"
        type: graph
        targets:
          - expr: rate(ml_registry_model_registered_total[5m])

      - title: "Cache Hit Rate"
        type: gauge
        targets:
          - expr: ml_registry_cache_hit_rate

      - title: "Hot Reload Success Rate"
        type: stat
        targets:
          - expr: |
            sum(ml_registry_hot_reload_success_total) /
            (sum(ml_registry_hot_reload_success_total) +
             sum(ml_registry_hot_reload_failure_total))
```

## Alert Rules

```yaml
# ml/monitoring/alerts/registry.yaml
groups:
  - name: ml_registry
    rules:
      - alert: RegistryCoherenceIssues
        expr: ml_registry_coherence_issues > 0
        for: 5m
        annotations:
          summary: "Registry has {{ $value }} coherence issues"

      - alert: HighRegistrationLatency
        expr: histogram_quantile(0.99, ml_registry_registration_duration_seconds) > 1
        for: 5m
        annotations:
          summary: "P99 registration latency is {{ $value }}s"

      - alert: LowCacheHitRate
        expr: ml_registry_cache_hit_rate < 0.5
        for: 10m
        annotations:
          summary: "Cache hit rate is {{ $value }}"
```

This approach ensures clean refactoring without technical debt, proper abstractions, and full observability through Prometheus metrics.
