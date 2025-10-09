# Task: [Phase 2.3] ModelRegistry Decomposition

## Context
**Phase:** 2 - Core Store Refactoring (God Classes)
**Task ID:** 2.3
**Depends On:** Phase 2.1 (DataStore Decomposition) and Phase 2.2 (MLPipelineOrchestrator Decomposition)
**Estimated Effort:** 15 hours
**Impact Score:** Critical - Refactoring 2,272-line god class into 5 components

## Scope
Decompose the monolithic `ModelRegistry` class (2,272 lines) into 5 focused components using the Strangler Fig pattern. This refactoring improves testability, maintainability, and reduces cognitive load while maintaining 100% backward compatibility through a facade.

**Current State:** `ml/registry/model_registry.py` (2,272 lines)
**Target State:** 5 specialized components + 1 facade (~400 lines total in facade)

**Components to Extract:**

1. **ModelPersistence** - Model saving/loading, artifact management, file I/O, SHA-256 integrity
2. **ModelDeploymentManager** - Deployment tracking, version management, rollback, hot reload
3. **ModelQualityValidator** - Quality gates, validation results, gate evaluation
4. **ABTestingManager** - A/B test configuration, statistical analysis, metric tracking
5. **CanaryDeploymentManager** - Canary release management, gradual rollout, promotion

## Required Reading

- [x] REFACTORING_PLAN.md (Phase 2.3, lines 210-221)
- [x] AGENT_TASK_FRAMEWORK.md (Task Template Structure section)
- [x] CLAUDE.md (Universal ML Architecture Patterns)
- [x] ml/docs/development/CODING_STANDARDS.md
- [x] ml/docs/architecture/universal_patterns_guide.md
- [x] ml/registry/model_registry.py (entire file - understand current structure)
- [x] tasks/phase_2_1_datastore_decomposition.md (proven pattern - reference)
- [x] tasks/phase_2_2_mlpipeline_orchestrator_decomposition.md (recent pattern - reference)

## Definition of Done

- [ ] All 5 components extracted with clear single responsibilities
- [ ] ModelRegistry facade maintains 100% backward compatibility
- [ ] All public APIs preserved (no breaking changes)
- [ ] Feature flag `ML_USE_LEGACY_MODEL_REGISTRY` implemented and tested
- [ ] All existing tests pass without modification
- [ ] New unit tests for each component (≥90% coverage per component)
- [ ] Integration tests verify facade behavior matches original
- [ ] Zero new circular dependencies introduced
- [ ] Zero new architecture violations
- [ ] Ruff check passes (zero violations)
- [ ] MyPy --strict passes (zero errors)
- [ ] make validate-nautilus-patterns passes
- [ ] Documentation updated with architecture diagrams
- [ ] Rollback plan tested and documented

## Files to Modify

### Create New Components (5 files)

- [ ] `ml/registry/model_persistence.py` - Persistence and artifact management (~400 lines)
- [ ] `ml/registry/model_deployment_mgr.py` - Deployment management (~350 lines)
- [ ] `ml/registry/model_quality_validator.py` - Quality validation (~350 lines)
- [ ] `ml/registry/ab_testing_manager.py` - A/B testing (~350 lines)
- [ ] `ml/registry/canary_deployment_mgr.py` - Canary deployment (~350 lines)

### Create New Tests (5 files)

- [ ] `ml/tests/unit/registry/test_model_persistence.py`
- [ ] `ml/tests/unit/registry/test_model_deployment_mgr.py`
- [ ] `ml/tests/unit/registry/test_model_quality_validator.py`
- [ ] `ml/tests/unit/registry/test_ab_testing_manager.py`
- [ ] `ml/tests/unit/registry/test_canary_deployment_mgr.py`

### Create Integration Tests (1 file)

- [ ] `ml/tests/integration/registry/test_model_registry_facade.py`

### Modify Existing (3 files)

- [ ] `ml/registry/model_registry.py` - Replace with facade or rename to _legacy.py
- [ ] `ml/registry/__init__.py` - Export new components + facade
- [ ] Update existing model registry tests to support feature flag

## Implementation Steps

### Week 1: Extract Persistence and Quality Components

#### Day 1-2: Extract ModelPersistence (lines 144-519, 527-616, 1118-1214)

**Responsibility:** All persistence operations including loading, saving, artifact integrity, caching.

**Line Ranges to Extract:**

- `_load_registry()` (lines 144-199)
- `_save_registry()` (lines 201-235)
- `_do_save()` (lines 237-264)
- `_flush_batch_save()` (lines 266-290)
- `_model_info_to_dict()` (lines 292-328)
- `_dict_to_model_info()` (lines 330-382)
- `_db_to_model_info()` (lines 384-431)
- `_save_model_to_db()` (lines 433-518)
- `_generate_model_id()` (lines 520-525)
- `_calculate_file_sha256()` (lines 527-561)
- `_verify_artifact_integrity()` (lines 563-616)
- `load_model()` (lines 1118-1214)
- `get_artifact_path()` (lines 1042-1056)
- `_validate_model_path()` (lines 1370-1393)
- `flush()` (lines 1345-1359)
- `__del__()` (lines 1361-1368)

**Steps:**

1. Create `ml/registry/model_persistence.py`
2. Define `ModelPersistenceProtocol` with all persistence methods
3. Extract persistence methods into `ModelPersistence` class
4. Handle both JSON and PostgreSQL backends (via PersistenceManager)
5. Implement model caching with LRU eviction
6. Preserve batch save logic and threading
7. Maintain SHA-256 integrity verification for security
8. Add proper type annotations and docstrings
9. Create comprehensive unit tests
10. Test both JSON and PostgreSQL backends
11. Test artifact integrity verification
12. Verify ONNX runtime integration for model loading

**Example Structure:**

```python
"""Model persistence and artifact management."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
import logging
import threading
import time

from ml.registry.base import ModelInfo
from ml.registry.persistence import BackendType, PersistenceManager

logger = logging.getLogger(__name__)


class ModelPersistenceProtocol(Protocol):
    """Protocol for model persistence operations."""

    def load_registry(self) -> tuple[
        dict[str, ModelInfo],
        dict[str, dict[str, Any]],
        dict[str, list[str]],
    ]: ...

    def save_registry(
        self,
        models: dict[str, ModelInfo],
        ab_tests: dict[str, dict[str, Any]],
        deployments: dict[str, list[str]],
        immediate: bool = False,
    ) -> None: ...

    def load_model(self, model_id: str) -> object | None: ...

    def get_artifact_path(self, model_id: str) -> Path | None: ...

    def calculate_file_sha256(self, file_path: Path) -> str: ...

    def verify_artifact_integrity(
        self,
        file_path: Path,
        expected_digest: str | None,
    ) -> None: ...


class ModelPersistence:
    """
    Manages model persistence, artifact loading, and integrity verification.

    Handles both JSON and PostgreSQL backends, implements model caching with
    LRU eviction, and provides SHA-256 integrity verification for security.
    """

    def __init__(
        self,
        registry_path: Path,
        persistence_manager: PersistenceManager,
        cache_size: int = 10,
        batch_save_interval: float = 0.1,
        onnx_runtime_config: Any | None = None,
    ) -> None:
        """
        Initialize model persistence.

        Parameters
        ----------
        registry_path : Path
            Registry directory path
        persistence_manager : PersistenceManager
            Persistence backend manager
        cache_size : int
            Maximum models to cache
        batch_save_interval : float
            Batch save interval in seconds
        onnx_runtime_config : OnnxRuntimeConfig | None
            ONNX runtime configuration
        """
        self.registry_path = registry_path
        self.persistence = persistence_manager
        self.cache_size = cache_size
        self.batch_save_interval = batch_save_interval
        self._onnx_rt = onnx_runtime_config
        self._registry_root = registry_path.resolve()
        self.registry_file = registry_path / "registry.json"

        # Model cache
        self._model_cache: dict[str, Any] = {}
        self._cache_access_times: dict[str, float] = {}

        # Batch save state
        self._lock = threading.RLock()
        self._pending_save = False
        self._save_timer: threading.Timer | None = None

    def load_registry(self) -> tuple[
        dict[str, ModelInfo],
        dict[str, dict[str, Any]],
        dict[str, list[str]],
    ]:
        """
        Load registry from persistence backend.

        Returns
        -------
        tuple[dict[str, ModelInfo], dict[str, dict[str, Any]], dict[str, list[str]]]
            (models, ab_tests, deployments)
        """
        # Implementation extracted from model_registry.py:144-199
        ...

    def save_registry(
        self,
        models: dict[str, ModelInfo],
        ab_tests: dict[str, dict[str, Any]],
        deployments: dict[str, list[str]],
        immediate: bool = False,
    ) -> None:
        """
        Save registry with optional batching.

        Parameters
        ----------
        models : dict[str, ModelInfo]
            Models to save
        ab_tests : dict[str, dict[str, Any]]
            A/B tests to save
        deployments : dict[str, list[str]]
            Deployments to save
        immediate : bool
            If True, save immediately
        """
        # Implementation extracted from model_registry.py:201-290
        ...

    def load_model(self, model_id: str, model_info: ModelInfo) -> object | None:
        """
        Load model from cache or disk with integrity verification.

        SECURITY: Only loads ONNX models, verifies SHA-256 digest.

        Parameters
        ----------
        model_id : str
            Model ID to load
        model_info : ModelInfo
            Model information

        Returns
        -------
        object | None
            Loaded ONNX InferenceSession or None
        """
        # Implementation extracted from model_registry.py:1118-1214
        ...

    # All persistence methods...
```

#### Day 3-4: Extract ModelQualityValidator (lines 1565-1697)

**Responsibility:** Quality gate validation, gate evaluation, validation results.

**Line Ranges to Extract:**

- `_validate_quality_gates()` (lines 1565-1603)
- `_evaluate_gate()` (lines 1605-1659)
- `validate_model_quality()` (lines 1661-1697)
- `_apply_quality_gates()` (lines 846-875) - calls validator

**Steps:**

1. Create `ml/registry/model_quality_validator.py`
2. Define `ModelQualityValidatorProtocol`
3. Extract quality validation methods
4. Use QualityGate and ValidationResult dataclasses
5. Support all comparison operators (gte, lte, gt, lt, eq)
6. Add proper type annotations and docstrings
7. Create comprehensive unit tests
8. Test all gate evaluation scenarios
9. Test required vs optional gates
10. Test margin calculations

**Example Structure:**

```python
"""Model quality validation and gate evaluation."""

from __future__ import annotations

from typing import Any, Protocol
import logging

from ml.registry.dataclasses import QualityGate, ValidationResult

logger = logging.getLogger(__name__)


class ModelQualityValidatorProtocol(Protocol):
    """Protocol for model quality validation operations."""

    def validate_quality_gates(
        self,
        model_id: str,
        metrics: dict[str, float],
        gates: list[QualityGate],
    ) -> ValidationResult: ...

    def evaluate_gate(
        self,
        gate: QualityGate,
        actual_value: float | None,
    ) -> dict[str, Any]: ...


class ModelQualityValidator:
    """
    Validates models against quality gates.

    Performs quality gate evaluation with support for multiple comparison
    operators, required vs optional gates, and detailed result reporting.
    """

    def __init__(self) -> None:
        """Initialize quality validator."""
        pass

    def validate_quality_gates(
        self,
        model_id: str,
        metrics: dict[str, float],
        gates: list[QualityGate],
    ) -> ValidationResult:
        """
        Validate model metrics against quality gates.

        Parameters
        ----------
        model_id : str
            Model identifier
        metrics : dict[str, float]
            Model metrics
        gates : list[QualityGate]
            Quality gates to check

        Returns
        -------
        ValidationResult
            Validation results with pass/fail status
        """
        # Implementation extracted from model_registry.py:1565-1603
        ...

    def evaluate_gate(
        self,
        gate: QualityGate,
        actual_value: float | None,
    ) -> dict[str, Any]:
        """
        Evaluate a single quality gate.

        Parameters
        ----------
        gate : QualityGate
            Gate to evaluate
        actual_value : float | None
            Actual metric value

        Returns
        -------
        dict[str, Any]
            Gate evaluation result
        """
        # Implementation extracted from model_registry.py:1605-1659
        ...

    # All quality validation methods...
```

### Week 2: Extract Deployment and Testing Components

#### Day 1-2: Extract ModelDeploymentManager (lines 926-977, 1295-1343, 1395-1429, 2098-2157)

**Responsibility:** Model deployment tracking, version management, rollback, hot reload, retirement.

**Line Ranges to Extract:**

- `deploy_model()` (lines 926-976)
- `get_active_models()` (lines 978-987)
- `get_all_models()` (lines 989-994)
- `list_compatible()` (lines 997-1017)
- `resolve_latest()` (lines 1019-1040)
- `get_model()` (lines 1058-1063)
- `get_models_by_role()` (lines 1065-1074)
- `get_models_by_data_requirements()` (lines 1076-1088)
- `get_model_lineage()` (lines 1090-1116)
- `track_performance()` (lines 1216-1247)
- `update_metadata()` (lines 1250-1281)
- `get_performance_history()` (lines 1283-1293)
- `rollback()` (lines 1295-1343)
- `retire_model()` (lines 1395-1429)
- `hot_reload_model()` (lines 2098-2157)
- `_maybe_auto_deploy()` (lines 877-924)

**Steps:**

1. Create `ml/registry/model_deployment_mgr.py`
2. Define `ModelDeploymentManagerProtocol`
3. Extract deployment management methods
4. Maintain deployment tracking state
5. Preserve auto-deploy logic
6. Add proper type annotations and docstrings
7. Create comprehensive unit tests
8. Test deployment lifecycle (deploy → active → retire)
9. Test rollback scenarios
10. Test hot reload with schema validation

**Example Structure:**

```python
"""Model deployment management and lifecycle tracking."""

from __future__ import annotations

from typing import Any, Protocol
import logging
import time

from ml.registry.base import DeploymentStatus, ModelInfo, ModelRole, DataRequirements

logger = logging.getLogger(__name__)


class ModelDeploymentManagerProtocol(Protocol):
    """Protocol for model deployment operations."""

    def deploy_model(
        self,
        model_id: str,
        target: str,
        config: dict[str, Any] | None = None,
    ) -> bool: ...

    def rollback(self, target: str, to_model_id: str) -> bool: ...

    def retire_model(self, model_id: str) -> bool: ...

    def hot_reload_model(self, target: str, new_model_id: str) -> bool: ...

    def get_active_models(self) -> list[ModelInfo]: ...


class ModelDeploymentManager:
    """
    Manages model deployment lifecycle and tracking.

    Handles deployment operations, version management, rollback,
    hot reload, and retirement of models.
    """

    def __init__(
        self,
        models: dict[str, ModelInfo],
        deployments: dict[str, list[str]],
        policy_config: Any | None = None,
    ) -> None:
        """
        Initialize deployment manager.

        Parameters
        ----------
        models : dict[str, ModelInfo]
            Models registry (reference)
        deployments : dict[str, list[str]]
            Deployments registry (reference)
        policy_config : RegistryPolicyConfig | None
            Policy configuration
        """
        self._models = models
        self._deployments = deployments
        self._policy = policy_config

    def deploy_model(
        self,
        model_id: str,
        target: str,
        config: dict[str, Any] | None = None,
    ) -> bool:
        """
        Deploy a model to a target.

        Parameters
        ----------
        model_id : str
            Model ID to deploy
        target : str
            Deployment target
        config : dict[str, Any] | None
            Deployment configuration

        Returns
        -------
        bool
            True if deployment successful
        """
        # Implementation extracted from model_registry.py:926-976
        ...

    def rollback(self, target: str, to_model_id: str) -> bool:
        """
        Rollback to a previous model version.

        Parameters
        ----------
        target : str
            Deployment target
        to_model_id : str
            Model ID to rollback to

        Returns
        -------
        bool
            True if rollback successful
        """
        # Implementation extracted from model_registry.py:1295-1343
        ...

    # All deployment management methods...
```

#### Day 3-4: Extract ABTestingManager (lines 1431-1500, 1502-1561, 1909-1966, 1968-2094)

**Responsibility:** A/B test configuration, statistical analysis, metric tracking, result analysis.

**Line Ranges to Extract:**

- `configure_ab_test()` (lines 1431-1500)
- `compare_models()` (lines 1502-1561)
- `compare_models_statistically()` (lines 1909-1966)
- `run_ab_test()` (lines 1968-2018)
- `track_ab_test_metric()` (lines 2020-2035)
- `analyze_ab_test()` (lines 2037-2094)

**Steps:**

1. Create `ml/registry/ab_testing_manager.py`
2. Define `ABTestingManagerProtocol`
3. Extract A/B testing methods
4. Integrate with `welch_t_test` from statistics module
5. Maintain A/B test state and metrics
6. Add proper type annotations and docstrings
7. Create comprehensive unit tests
8. Test statistical comparison (Welch's t-test)
9. Test metric tracking and analysis
10. Test A/B test lifecycle

**Example Structure:**

```python
"""A/B testing configuration and statistical analysis."""

from __future__ import annotations

from typing import Any, Protocol
import logging
import time

from ml.registry.base import DeploymentStatus, ModelInfo
from ml.registry.statistics import welch_t_test

logger = logging.getLogger(__name__)


class ABTestingManagerProtocol(Protocol):
    """Protocol for A/B testing operations."""

    def configure_ab_test(
        self,
        models: list[str],
        split_ratio: float,
        duration_hours: int,
        target: str,
    ) -> dict[str, Any] | None: ...

    def run_ab_test(
        self,
        model_a_id: str,
        model_b_id: str,
        split_ratio: float,
        duration_hours: float,
        target: str,
    ) -> str: ...

    def analyze_ab_test(self, test_id: str) -> dict[str, Any] | None: ...


class ABTestingManager:
    """
    Manages A/B testing between models.

    Configures A/B tests, tracks metrics, performs statistical analysis
    using Welch's t-test, and provides result analysis.
    """

    def __init__(
        self,
        models: dict[str, ModelInfo],
        deployments: dict[str, list[str]],
        policy_config: Any | None = None,
    ) -> None:
        """
        Initialize A/B testing manager.

        Parameters
        ----------
        models : dict[str, ModelInfo]
            Models registry (reference)
        deployments : dict[str, list[str]]
            Deployments registry (reference)
        policy_config : RegistryPolicyConfig | None
            Policy configuration
        """
        self._models = models
        self._deployments = deployments
        self._policy = policy_config
        self._ab_tests: dict[str, dict[str, Any]] = {}
        self._ab_test_metrics: dict[str, dict[str, list[float]]] = {}

    def configure_ab_test(
        self,
        models: list[str],
        split_ratio: float,
        duration_hours: int,
        target: str,
    ) -> dict[str, Any] | None:
        """
        Configure A/B test between models.

        Parameters
        ----------
        models : list[str]
            List of model IDs to test (expects 2)
        split_ratio : float
            Traffic split ratio for first model
        duration_hours : int
            Test duration in hours
        target : str
            Deployment target

        Returns
        -------
        dict[str, Any] | None
            A/B test configuration
        """
        # Implementation extracted from model_registry.py:1431-1500
        ...

    def compare_models_statistically(
        self,
        model_ids: list[str],
        metric: str,
    ) -> dict[str, Any] | None:
        """
        Perform statistical comparison using Welch's t-test.

        Parameters
        ----------
        model_ids : list[str]
            Model IDs to compare (exactly 2)
        metric : str
            Metric to compare

        Returns
        -------
        dict[str, Any] | None
            Statistical comparison results
        """
        # Implementation extracted from model_registry.py:1909-1966
        ...

    # All A/B testing methods...
```

#### Day 5: Extract CanaryDeploymentManager (lines 1701-1905, 2159-2272)

**Responsibility:** Canary deployment management, gradual rollout, promotion, rollback detection.

**Line Ranges to Extract:**

- `start_canary_deployment()` (lines 1701-1777)
- `get_canary_deployment()` (lines 1779-1785)
- `update_canary_metrics()` (lines 1787-1816)
- `evaluate_canary()` (lines 1817-1840)
- `evaluate_canary_for_rollback()` (lines 1842-1865)
- `auto_promote_canary()` (lines 1867-1905)
- `start_gradual_rollout()` (lines 2159-2223)
- `get_rollout_status()` (lines 2225-2243)
- `advance_rollout_stage()` (lines 2245-2272)

**Steps:**

1. Create `ml/registry/canary_deployment_mgr.py`
2. Define `CanaryDeploymentManagerProtocol`
3. Extract canary deployment methods
4. Use CanaryConfig, CanaryDeployment, RolloutPlan dataclasses
5. Maintain canary state and rollout plans
6. Add proper type annotations and docstrings
7. Create comprehensive unit tests
8. Test canary lifecycle (start → evaluate → promote/rollback)
9. Test gradual rollout stages
10. Test automatic promotion and rollback

**Example Structure:**

```python
"""Canary deployment and gradual rollout management."""

from __future__ import annotations

from typing import Any, Protocol
import logging
import time

from ml.registry.base import DeploymentStatus, ModelInfo
from ml.registry.dataclasses import CanaryConfig, CanaryDeployment, RolloutPlan

logger = logging.getLogger(__name__)


class CanaryDeploymentManagerProtocol(Protocol):
    """Protocol for canary deployment operations."""

    def start_canary_deployment(
        self,
        model_id: str,
        target: str,
        config: CanaryConfig,
        baseline_model_id: str | None = None,
    ) -> str: ...

    def evaluate_canary(self, deployment_id: str) -> tuple[bool, str]: ...

    def auto_promote_canary(self, deployment_id: str) -> bool: ...

    def start_gradual_rollout(
        self,
        current_model_id: str,
        new_model_id: str,
        target: str,
        stages: list[float],
        stage_duration_minutes: int,
    ) -> str: ...


class CanaryDeploymentManager:
    """
    Manages canary deployments and gradual rollouts.

    Handles canary deployment lifecycle, metric tracking, automatic
    promotion/rollback, and multi-stage gradual rollouts.
    """

    def __init__(
        self,
        models: dict[str, ModelInfo],
        ab_testing_manager: Any,
    ) -> None:
        """
        Initialize canary deployment manager.

        Parameters
        ----------
        models : dict[str, ModelInfo]
            Models registry (reference)
        ab_testing_manager : ABTestingManager
            A/B testing manager for traffic splitting
        """
        self._models = models
        self._ab_testing_mgr = ab_testing_manager
        self._canary_deployments: dict[str, CanaryDeployment] = {}
        self._rollout_plans: dict[str, RolloutPlan] = {}

    def start_canary_deployment(
        self,
        model_id: str,
        target: str,
        config: CanaryConfig,
        baseline_model_id: str | None = None,
    ) -> str:
        """
        Start a canary deployment for a model.

        Parameters
        ----------
        model_id : str
            Model to deploy as canary
        target : str
            Deployment target
        config : CanaryConfig
            Canary configuration
        baseline_model_id : str | None
            Baseline model for comparison

        Returns
        -------
        str
            Canary deployment ID
        """
        # Implementation extracted from model_registry.py:1701-1777
        ...

    def evaluate_canary(self, deployment_id: str) -> tuple[bool, str]:
        """
        Evaluate if canary should be promoted.

        Parameters
        ----------
        deployment_id : str
            Canary deployment ID

        Returns
        -------
        tuple[bool, str]
            (should_promote, reason)
        """
        # Implementation extracted from model_registry.py:1817-1840
        ...

    # All canary deployment methods...
```

### Week 2: Create Facade and Integration Testing

#### Day 6: Create ModelRegistry Facade

**Responsibility:** Maintain backward-compatible public API by delegating to specialized components.

**Steps:**

1. Create `ml/registry/model_registry_facade.py`
2. Replicate all public methods from original ModelRegistry
3. Initialize all 5 components in constructor
4. Delegate each method to appropriate component(s)
5. Add feature flag check: `ML_USE_LEGACY_MODEL_REGISTRY`
6. If flag is set, use original implementation; otherwise use new components
7. Preserve AbstractRegistry inheritance
8. Add proper type annotations and docstrings

**Example Structure:**

```python
"""
ModelRegistry facade maintaining backward compatibility.

This facade delegates to specialized components while preserving the original
public API. Feature flag ML_USE_LEGACY_MODEL_REGISTRY controls legacy vs new path.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

from ml.registry.abstract_registry import AbstractRegistry
from ml.registry.base import ModelInfo, ModelManifest, ModelRole, DataRequirements
from ml.registry.model_persistence import ModelPersistence
from ml.registry.model_deployment_mgr import ModelDeploymentManager
from ml.registry.model_quality_validator import ModelQualityValidator
from ml.registry.ab_testing_manager import ABTestingManager
from ml.registry.canary_deployment_mgr import CanaryDeploymentManager

logger = logging.getLogger(__name__)


class ModelRegistry(AbstractRegistry):
    """
    Model registry with configurable persistence backend.

    This facade delegates to specialized components while maintaining
    100% backward compatibility with the original ModelRegistry API.
    """

    def __init__(
        self,
        registry_path: Path,
        cache_size: int = 10,
        batch_save_interval: float = 0.1,
        persistence_config: Any | None = None,
        policy_config: Any | None = None,
        onnx_runtime_config: Any | None = None,
    ) -> None:
        """
        Initialize ModelRegistry with configurable backend.

        Parameters match original constructor for compatibility.
        """
        # Check feature flag
        use_legacy = os.getenv("ML_USE_LEGACY_MODEL_REGISTRY", "0") == "1"

        if use_legacy:
            # Use original implementation
            logger.info("Using legacy ModelRegistry implementation")
            from ml.registry.model_registry_legacy import ModelRegistryLegacy
            self._impl = ModelRegistryLegacy(
                registry_path=registry_path,
                cache_size=cache_size,
                batch_save_interval=batch_save_interval,
                persistence_config=persistence_config,
                policy_config=policy_config,
                onnx_runtime_config=onnx_runtime_config,
            )
            self._use_legacy = True
            # Expose persistence for AbstractRegistry
            self.persistence = self._impl.persistence
        else:
            # Use new component-based implementation
            logger.info("Using component-based ModelRegistry implementation")

            # Initialize persistence backend
            from ml.registry.persistence import BackendType, PersistenceConfig, PersistenceManager
            if persistence_config is None:
                persistence_config = PersistenceConfig(
                    backend=BackendType.JSON,
                    json_path=registry_path,
                )
            persistence_manager = PersistenceManager(persistence_config)

            # Call parent init
            super().__init__(persistence_manager)

            # Initialize components
            self._model_persistence = ModelPersistence(
                registry_path=registry_path,
                persistence_manager=persistence_manager,
                cache_size=cache_size,
                batch_save_interval=batch_save_interval,
                onnx_runtime_config=onnx_runtime_config,
            )

            # Load registry
            self._models, self._ab_tests, self._deployments = (
                self._model_persistence.load_registry()
            )

            self._quality_validator = ModelQualityValidator()
            self._deployment_manager = ModelDeploymentManager(
                models=self._models,
                deployments=self._deployments,
                policy_config=policy_config,
            )
            self._ab_testing_manager = ABTestingManager(
                models=self._models,
                deployments=self._deployments,
                policy_config=policy_config,
            )
            self._canary_deployment_manager = CanaryDeploymentManager(
                models=self._models,
                ab_testing_manager=self._ab_testing_manager,
            )

            self._use_legacy = False
            self.registry_path = registry_path
            self._policy = policy_config

    # Delegate all public methods
    def register_model(
        self,
        model_path: Path,
        manifest: ModelManifest,
        auto_deploy: bool = False,
        quality_gates: Any | None = None,
        enforce_quality: bool = False,
    ) -> str:
        """Register a new model."""
        if self._use_legacy:
            return self._impl.register_model(
                model_path,
                manifest,
                auto_deploy,
                quality_gates,
                enforce_quality,
            )
        # Complex registration logic using multiple components
        # Uses: model_persistence, quality_validator, deployment_manager
        ...

    def deploy_model(
        self,
        model_id: str,
        target: str,
        config: dict[str, Any] | None = None,
    ) -> bool:
        """Deploy a model to a target."""
        if self._use_legacy:
            return self._impl.deploy_model(model_id, target, config)
        return self._deployment_manager.deploy_model(model_id, target, config)

    def load_model(self, model_id: str) -> object | None:
        """Load model from cache or disk."""
        if self._use_legacy:
            return self._impl.load_model(model_id)
        if model_id not in self._models:
            return None
        return self._model_persistence.load_model(model_id, self._models[model_id])

    def validate_model_quality(
        self,
        model_id: str,
        gates: Any,
    ) -> Any:
        """Validate model quality."""
        if self._use_legacy:
            return self._impl.validate_model_quality(model_id, gates)
        return self._quality_validator.validate_quality_gates(
            model_id,
            self._models[model_id].manifest.performance_metrics if model_id in self._models else {},
            gates,
        )

    def start_canary_deployment(
        self,
        model_id: str,
        target: str,
        config: Any,
        baseline_model_id: str | None = None,
    ) -> str:
        """Start canary deployment."""
        if self._use_legacy:
            return self._impl.start_canary_deployment(
                model_id,
                target,
                config,
                baseline_model_id,
            )
        return self._canary_deployment_manager.start_canary_deployment(
            model_id,
            target,
            config,
            baseline_model_id,
        )

    # ... all other public methods delegated similarly ...

    def _health_snapshot(self) -> tuple[int, float | None]:
        """Get health snapshot for AbstractRegistry."""
        if self._use_legacy:
            return self._impl._health_snapshot()
        try:
            count = len(self._models)
        except AttributeError:
            return 0, None
        if count == 0:
            return 0, None
        try:
            last_modified = max(mi.manifest.last_modified for mi in self._models.values())
        except ValueError:
            last_modified = None
        return count, last_modified
```

#### Day 7: Integration Testing and Validation

**Steps:**

1. Move original `model_registry.py` to `model_registry_legacy.py`
2. Update `ml/registry/__init__.py` exports
3. Create integration test suite:
   - Test facade with legacy flag ON - verify original behavior
   - Test facade with legacy flag OFF - verify new behavior
   - Compare outputs between legacy and new implementations
   - Test all public APIs for backward compatibility
4. Run full test suite
5. Fix any failures
6. Performance benchmarking (ensure no regression)
7. Update documentation

## Testing Strategy

### Unit Tests (Per Component)

```python
# ml/tests/unit/registry/test_model_persistence.py
def test_model_persistence_load_json():
    """Test loading from JSON backend."""
    ...

def test_model_persistence_save_postgres():
    """Test saving to PostgreSQL backend."""
    ...

def test_model_persistence_artifact_integrity():
    """Test SHA-256 integrity verification."""
    ...

def test_model_persistence_model_caching():
    """Test LRU model cache."""
    ...


# ml/tests/unit/registry/test_model_deployment_mgr.py
def test_deployment_manager_deploy():
    """Test model deployment."""
    ...

def test_deployment_manager_rollback():
    """Test deployment rollback."""
    ...

def test_deployment_manager_hot_reload():
    """Test hot reload with schema validation."""
    ...


# ml/tests/unit/registry/test_model_quality_validator.py
def test_quality_validator_gates():
    """Test quality gate validation."""
    ...

def test_quality_validator_comparisons():
    """Test all comparison operators."""
    ...


# ml/tests/unit/registry/test_ab_testing_manager.py
def test_ab_testing_configure():
    """Test A/B test configuration."""
    ...

def test_ab_testing_statistical_comparison():
    """Test Welch's t-test comparison."""
    ...


# ml/tests/unit/registry/test_canary_deployment_mgr.py
def test_canary_deployment_lifecycle():
    """Test canary deployment lifecycle."""
    ...

def test_canary_gradual_rollout():
    """Test gradual rollout stages."""
    ...
```

### Integration Tests

```python
# ml/tests/integration/registry/test_model_registry_facade.py
def test_facade_backward_compatibility():
    """Verify facade matches original API."""
    legacy = ModelRegistryLegacy(...)
    facade = ModelRegistry(...)  # Feature flag OFF

    # Compare outputs for same operations
    assert facade.register_model(...) == legacy.register_model(...)
    ...

def test_facade_feature_flag_legacy():
    """Test facade with ML_USE_LEGACY_MODEL_REGISTRY=1."""
    os.environ["ML_USE_LEGACY_MODEL_REGISTRY"] = "1"
    registry = ModelRegistry(...)
    # Verify using legacy path
    ...

def test_facade_feature_flag_new():
    """Test facade with ML_USE_LEGACY_MODEL_REGISTRY=0."""
    os.environ["ML_USE_LEGACY_MODEL_REGISTRY"] = "0"
    registry = ModelRegistry(...)
    # Verify using new component path
    ...
```

## Testing Requirements

### Unit Tests (≥90% coverage per component)

- [ ] test_model_persistence.py - All persistence operations
- [ ] test_model_deployment_mgr.py - All deployment operations
- [ ] test_model_quality_validator.py - All quality validation
- [ ] test_ab_testing_manager.py - All A/B testing
- [ ] test_canary_deployment_mgr.py - All canary deployment

### Integration Tests

- [ ] test_model_registry_facade.py - Backward compatibility
- [ ] test_feature_flag_toggle.py - Legacy vs new behavior
- [ ] test_component_integration.py - Components work together

### Performance Tests

- [ ] Benchmark model loading (target: <5ms P99 for cached)
- [ ] Benchmark persistence operations (should not regress)
- [ ] Benchmark deployment operations (should not regress)

### Regression Tests

- [ ] All existing ModelRegistry tests pass unchanged
- [ ] No behavioral changes in public APIs
- [ ] Model lineage tracking unchanged
- [ ] A/B testing functionality unchanged

## Rollback Plan

### Immediate Rollback (Production Issue)

```bash
# Set feature flag to use legacy implementation
export ML_USE_LEGACY_MODEL_REGISTRY=1

# Restart services
kubectl rollout restart deployment/ml-service
```

### Code Rollback (Development Issue)

```bash
# Revert all new files
git checkout ml/registry/model_persistence.py
git checkout ml/registry/model_deployment_mgr.py
git checkout ml/registry/model_quality_validator.py
git checkout ml/registry/ab_testing_manager.py
git checkout ml/registry/canary_deployment_mgr.py
git checkout ml/registry/model_registry_facade.py
git checkout ml/registry/model_registry_legacy.py

# Restore original
git checkout ml/registry/model_registry.py
git checkout ml/registry/__init__.py

# Revert tests
git checkout ml/tests/unit/registry/test_model_persistence.py
git checkout ml/tests/unit/registry/test_model_deployment_mgr.py
git checkout ml/tests/unit/registry/test_model_quality_validator.py
git checkout ml/tests/unit/registry/test_ab_testing_manager.py
git checkout ml/tests/unit/registry/test_canary_deployment_mgr.py
git checkout ml/tests/integration/registry/test_model_registry_facade.py
```

### Verification After Rollback

```bash
# Run tests
pytest ml/tests/unit/registry/ -v
pytest ml/tests/integration/registry/ -v

# Verify imports
python -c "from ml.registry import ModelRegistry; print('OK')"
```

## Success Metrics

### Code Quality Metrics

- Lines reduced: 2,272 → ~2,200 (5 components + facade)
- Average file size: 2,272 → ~370 lines (84% reduction per file)
- Cyclomatic complexity: Reduced by ~70% (smaller focused functions)
- Test coverage: ≥90% per component (up from current coverage)

### Architecture Metrics

- Number of responsibilities: 1 god class → 5 focused components
- Files affected: 1 → 6 (5 components + 1 facade)
- Circular dependencies: 0 (no new cycles introduced)
- Protocol conformance: 100% (all components implement protocols)

### Performance Metrics

- Model loading latency: <5ms P99 cached (no regression)
- Registration latency: <50ms P99 (no regression)
- Deployment latency: <100ms P99 (no regression)
- Memory usage: ≤10% increase (acceptable for better structure)

### Testing Metrics

- Unit tests: +5 test files, ~200 new tests
- Integration tests: +1 test file, ~20 new tests
- Test execution time: <10% increase (parallel execution)
- Coverage: Maintain or improve to ≥90%

### Maintainability Metrics

- Cognitive load: Reduced (smaller focused classes)
- Onboarding time: Faster (clearer separation of concerns)
- Change impact: Localized (changes affect single component)
- Documentation: Improved (clear component boundaries)

## Notes

### Critical Requirements

- **100% Backward Compatibility:** All public APIs must work identically
- **Feature Flag:** `ML_USE_LEGACY_MODEL_REGISTRY` must work for safe rollout
- **No Breaking Changes:** Existing code using ModelRegistry must work unchanged
- **Zero New Cycles:** No circular dependencies introduced
- **Strangler Fig Pattern:** New code alongside old, then switch via flag
- **Security Preserved:** SHA-256 integrity verification must be maintained

### Strangler Fig Pattern Benefits

- Safe incremental migration
- Easy rollback via feature flag
- Low risk to production
- Allows parallel testing of old vs new
- Confidence in correctness before full cutover

### Component Interaction Flow

```
┌─────────────────────────────────────────────────────────────┐
│              ModelRegistry (Public API)                      │
│  - Maintains backward compatibility                          │
│  - Feature flag toggle (legacy vs new)                       │
│  - Inherits from AbstractRegistry                            │
└──────────────┬──────────────────────────────────────────────┘
               │
               ├──> ModelPersistence
               │    - Load/save registry (JSON/PostgreSQL)
               │    - Model artifact loading
               │    - SHA-256 integrity verification
               │    - Model caching (LRU)
               │
               ├──> ModelQualityValidator
               │    - Quality gate validation
               │    - Gate evaluation
               │    - Validation results
               │
               ├──> ModelDeploymentManager
               │    - Deploy/rollback/retire
               │    - Hot reload
               │    - Version management
               │    - Performance tracking
               │
               ├──> ABTestingManager
               │    - A/B test configuration
               │    - Statistical comparison (Welch's t-test)
               │    - Metric tracking
               │    - Result analysis
               │
               └──> CanaryDeploymentManager
                    - Canary deployment
                    - Gradual rollout
                    - Auto-promotion
                    - Rollback detection
                    └──> ABTestingManager (composition)
```

### Architecture Decisions

1. **Protocol-First:** All components implement protocols for testability
2. **Dependency Injection:** Components receive dependencies in constructor
3. **Composition Over Inheritance:** Components compose each other, not inherit
4. **Single Responsibility:** Each component has one clear purpose
5. **Shared State:** Components share references to models/deployments dictionaries
6. **Security First:** Preserve SHA-256 integrity verification in ModelPersistence

### Migration Path

1. **Week 1:** Extract independent components (ModelPersistence, ModelQualityValidator)
2. **Week 2:** Extract dependent components (ModelDeploymentManager, ABTestingManager, CanaryDeploymentManager)
3. **Week 2:** Create facade with feature flag
4. **Week 3:** Integration testing and validation
5. **Week 4:** Deploy with flag ON (legacy mode)
6. **Week 5:** Gradual rollout with flag OFF (new mode)
7. **Week 6:** Remove legacy code if new mode stable

### Known Challenges

- **State Management:** Components share references to _models and _deployments
- **Component Dependencies:** CanaryDeploymentManager depends on ABTestingManager
- **Registration Logic:** Complex registration flow uses multiple components
- **Persistence Backends:** Must support both JSON and PostgreSQL
- **Security:** Must preserve SHA-256 integrity verification
- **Caching:** Model caching with LRU eviction in ModelPersistence
- **Threading:** Batch save threading in ModelPersistence

### Future Considerations

- **Phase 3:** Further decomposition if components still too large
- **Phase 4:** Extract common lineage/compatibility logic
- **Phase 5:** Async model loading for improved performance
- **Phase 6:** Distributed model registry for multi-node deployments

### Dependencies on Other Tasks

- **Blocks:** Phase 3 tasks - cleaner registry patterns
- **Blocked By:** Phase 2.1 and 2.2 - proven decomposition patterns
- **Related:** Feature/Strategy registry decompositions may follow similar pattern

---

## Approval Checklist

Before marking this task complete, verify:

- [ ] All 5 components extracted and tested independently
- [ ] Facade delegates correctly to all components
- [ ] Feature flag tested in both states (legacy ON and OFF)
- [ ] All existing tests pass without modification
- [ ] New unit tests achieve ≥90% coverage per component
- [ ] Integration tests verify backward compatibility
- [ ] No new circular dependencies (verified with import checks)
- [ ] Ruff, MyPy, and pattern validation all pass
- [ ] Performance benchmarks show no regression
- [ ] SHA-256 integrity verification preserved
- [ ] Model caching functionality preserved
- [ ] Both JSON and PostgreSQL backends work
- [ ] Documentation updated with architecture diagrams
- [ ] Rollback plan documented and tested
- [ ] TASK_REPORT.md generated with detailed changes
