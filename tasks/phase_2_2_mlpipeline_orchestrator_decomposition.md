# Task: [Phase 2.2] MLPipelineOrchestrator Decomposition

## Context
**Phase:** 2 - Core Store Refactoring (God Classes)
**Task ID:** 2.2
**Depends On:** Phase 2.1 (DataStore Decomposition Complete)
**Estimated Effort:** 25 hours
**Impact Score:** Critical - Refactoring LARGEST file (4,598 lines) into 5 components

## Scope
Decompose the monolithic `MLPipelineOrchestrator` class (4,598 lines) into 5 focused components using the Strangler Fig pattern. This refactoring improves testability, maintainability, and reduces cognitive load while maintaining 100% backward compatibility through a facade.

**Current State:** `ml/orchestration/pipeline_orchestrator.py` (4,598 lines - LARGEST FILE IN CODEBASE)
**Target State:** 5 specialized components + 1 facade (~600 lines total in facade)

**Components to Extract:**

1. **ConfigResolver** - Configuration resolution, market inputs, window bounds computation
2. **IngestionCoordinator** - Ingestion pipeline coordination, backfill management, auto-fill universe
3. **DatasetBuilder** - Dataset construction, validation, metadata management
4. **BindingResolver** - Market binding resolution, discovery integration, coverage checks
5. **DiscoveryClient** - Service discovery, health checks, dataset availability

## Required Reading

- [x] REFACTORING_PLAN.md (Phase 2.2, lines 197-208)
- [x] AGENT_TASK_FRAMEWORK.md (Task Template Structure section)
- [x] CLAUDE.md (Universal ML Architecture Patterns)
- [x] ml/docs/development/CODING_STANDARDS.md
- [x] ml/docs/architecture/universal_patterns_guide.md
- [x] ml/orchestration/pipeline_orchestrator.py (entire file - understand current structure)
- [x] tasks/phase_2_1_datastore_decomposition.md (reference example - proven pattern)

## Definition of Done

- [ ] All 5 components extracted with clear single responsibilities
- [ ] MLPipelineOrchestrator facade maintains 100% backward compatibility
- [ ] All public APIs preserved (no breaking changes)
- [ ] Feature flag `ML_USE_LEGACY_PIPELINE_ORCHESTRATOR` implemented and tested
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

- [ ] `ml/orchestration/config_resolver.py` - Configuration resolution (~350 lines)
- [ ] `ml/orchestration/ingestion_coordinator.py` - Ingestion orchestration (~800 lines)
- [ ] `ml/orchestration/dataset_builder.py` - Dataset building (~700 lines)
- [ ] `ml/orchestration/binding_resolver.py` - Binding resolution (~500 lines)
- [ ] `ml/orchestration/discovery_client.py` - Discovery and health (~300 lines)

### Create Shared Types Module (1 file)

- [ ] `ml/orchestration/shared_types.py` - Common dataclasses and protocols (~200 lines)

### Create New Tests (6 files)

- [ ] `ml/tests/unit/orchestration/test_config_resolver.py`
- [ ] `ml/tests/unit/orchestration/test_ingestion_coordinator.py`
- [ ] `ml/tests/unit/orchestration/test_dataset_builder.py`
- [ ] `ml/tests/unit/orchestration/test_binding_resolver.py`
- [ ] `ml/tests/unit/orchestration/test_discovery_client.py`
- [ ] `ml/tests/integration/orchestration/test_pipeline_orchestrator_facade.py`

### Modify Existing (3 files)

- [ ] `ml/orchestration/pipeline_orchestrator.py` - Replace with facade or rename to _legacy.py
- [ ] `ml/orchestration/__init__.py` - Export new components + facade
- [ ] Update existing integration tests to support feature flag

## Implementation Steps

### Week 3: Extract Configuration and Discovery Components

#### Day 1-2: Extract ConfigResolver (lines 173-279 + supporting methods)

**Responsibility:** All configuration resolution including market inputs, window bounds, symbol mapping, default values.

**Line Ranges to Extract:**

- `_apply_default_market_inputs()` (lines 173-213)
- `_collect_symbol_map()` (lines 216-278)
- `_compute_window_start_iso()` (lines 281-290)
- `_resolve_window_bounds_ns()` (lines ~450-500)
- `_prepare_dataset_config()` (lines 501-521)
- `_symbol_to_instruments()` (lines 740-765)
- `_collect_instrument_ids()` (lines 767-785)
- `_infer_default_schema()` (lines 726-730)

**Steps:**

1. Create `ml/orchestration/shared_types.py` first for common dataclasses
2. Create `ml/orchestration/config_resolver.py`
3. Define `ConfigResolverProtocol` with all configuration methods
4. Extract configuration methods into `ConfigResolver` class
5. Add proper type annotations and docstrings
6. Remove dependencies on orchestrator instance state (pass as parameters)
7. Create comprehensive unit tests
8. Verify no hard-coded values remain (all from config)

**Example Structure:**

```python
"""Configuration resolution for ML pipeline orchestrator."""

from __future__ import annotations

from typing import Protocol
from collections import OrderedDict
import logging

from ml.config.market_data import MarketDatasetInput
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.config_loader import IngestionStageConfig

logger = logging.getLogger(__name__)


class ConfigResolverProtocol(Protocol):
    """Protocol for configuration resolution operations."""

    def apply_default_market_inputs(
        self,
        cfg: DatasetBuildConfig,
    ) -> DatasetBuildConfig: ...

    def collect_symbol_map(
        self,
        ds_cfg: DatasetBuildConfig | None,
        ingestion_cfg: IngestionStageConfig,
    ) -> dict[str, tuple[str, ...]]: ...

    def prepare_dataset_config(
        self,
        cfg: DatasetBuildConfig,
        resolved_inputs: tuple[MarketDatasetInput, ...] | None,
        bindings: tuple[ResolvedMarketBinding, ...],
    ) -> DatasetBuildConfig: ...


class ConfigResolver:
    """
    Resolves and prepares configuration for ML pipeline operations.

    Handles market input resolution, symbol mapping, window bounds computation,
    and dataset config preparation with proper defaults and validation.
    """

    def __init__(
        self,
        market_feed_descriptors: Any | None = None,
        default_lookback_years: int = 3,
    ) -> None:
        """
        Initialize configuration resolver.

        Parameters
        ----------
        market_feed_descriptors : MarketFeedDescriptors | None
            Market feed descriptor registry
        default_lookback_years : int
            Default lookback period for historical data
        """
        self.market_feed_descriptors = market_feed_descriptors
        self.default_lookback_years = default_lookback_years

    def apply_default_market_inputs(
        self,
        cfg: DatasetBuildConfig,
    ) -> DatasetBuildConfig:
        """
        Seed dataset configs with descriptor-driven market inputs.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration

        Returns
        -------
        DatasetBuildConfig
            Configuration with market inputs populated
        """
        # Implementation extracted from pipeline_orchestrator.py:173-213
        ...

    def collect_symbol_map(
        self,
        ds_cfg: DatasetBuildConfig | None,
        ingestion_cfg: IngestionStageConfig,
    ) -> dict[str, tuple[str, ...]]:
        """
        Collect symbol to instrument ID mappings from configs.

        Parameters
        ----------
        ds_cfg : DatasetBuildConfig | None
            Dataset build configuration
        ingestion_cfg : IngestionStageConfig
            Ingestion stage configuration

        Returns
        -------
        dict[str, tuple[str, ...]]
            Symbol to instrument IDs mapping
        """
        # Implementation extracted from pipeline_orchestrator.py:216-278
        ...

    # All other configuration methods...
```

#### Day 3: Extract DiscoveryClient (lines 678-723 + health check methods)

**Responsibility:** Service discovery, health checks, dataset availability queries, coverage information.

**Line Ranges to Extract:**

- `_discover_market_inputs()` (lines 678-723)
- `_discover_binding_for_symbol()` (lines ~1000-1100)
- Service health check methods
- Coverage query helpers
- `_ns_to_datetime()` (lines 733-738)

**Steps:**

1. Create `ml/orchestration/discovery_client.py`
2. Define `DiscoveryClientProtocol` with discovery methods
3. Extract discovery methods into `DiscoveryClient` class
4. Take DatasetDiscoveryService as dependency (constructor injection)
5. Add proper type annotations and docstrings
6. Create comprehensive unit tests with mocked discovery service
7. Test error handling for discovery failures

**Example Structure:**

```python
"""Discovery client for ML pipeline orchestration."""

from __future__ import annotations

from typing import Protocol
from datetime import datetime
import logging

from ml.config.market_data import MarketDatasetInput
from ml.data.ingest.discovery import (
    DatasetDiscoveryService,
    DiscoveryRequest,
    DatasetDiscoveryError,
)
from ml.data.ingest.market_bindings import ResolvedMarketBinding

logger = logging.getLogger(__name__)


class DiscoveryClientProtocol(Protocol):
    """Protocol for dataset discovery operations."""

    def discover_market_inputs(
        self,
        symbol_map: dict[str, tuple[str, ...]],
        schema: str,
        start_ns: int,
        end_ns: int,
        dataset_hint: str | None = None,
    ) -> tuple[MarketDatasetInput, ...]: ...

    def discover_binding_for_symbol(
        self,
        symbol: str,
        instrument_ids: tuple[str, ...] | None,
        schema: str,
        start_ns: int,
        end_ns: int,
    ) -> ResolvedMarketBinding | None: ...


class DiscoveryClient:
    """
    Client for dataset discovery and service health checks.

    Provides high-level discovery operations with error handling,
    policy enforcement, and coverage validation.
    """

    def __init__(
        self,
        dataset_discovery: DatasetDiscoveryService | None = None,
    ) -> None:
        """
        Initialize discovery client.

        Parameters
        ----------
        dataset_discovery : DatasetDiscoveryService | None
            Dataset discovery service instance
        """
        self.dataset_discovery = dataset_discovery
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize Prometheus metrics using centralized bootstrap."""
        from ml.common.metrics_bootstrap import get_counter

        self.discovery_requests_counter = get_counter(
            "ml_discovery_requests_total",
            "Total discovery requests by status",
        )

    def discover_market_inputs(
        self,
        symbol_map: dict[str, tuple[str, ...]],
        schema: str,
        start_ns: int,
        end_ns: int,
        dataset_hint: str | None = None,
    ) -> tuple[MarketDatasetInput, ...]:
        """
        Discover market inputs for given symbols and time range.

        Parameters
        ----------
        symbol_map : dict[str, tuple[str, ...]]
            Symbol to instrument IDs mapping
        schema : str
            Data schema (e.g. 'ohlcv-1m', 'tbbo')
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds
        dataset_hint : str | None
            Optional dataset ID hint

        Returns
        -------
        tuple[MarketDatasetInput, ...]
            Discovered market inputs
        """
        # Implementation extracted from pipeline_orchestrator.py:678-723
        ...

    # All discovery methods...
```

### Week 4: Extract Binding and Ingestion Components

#### Day 1-3: Extract BindingResolver (lines 787-1109)

**Responsibility:** Market binding resolution, coverage checks, binding validation, priority selection.

**Line Ranges to Extract:**

- `_resolve_market_inputs()` (lines 523-676)
- `_filter_candidate_bindings()` (lines 787-810)
- `_binding_priority_key()` (lines 812-819)
- `_binding_allowed()` (lines 821-929)
- `_select_binding_with_coverage()` (lines 931-1050)
- Coverage query methods

**Steps:**

1. Create `ml/orchestration/binding_resolver.py`
2. Define `BindingResolverProtocol` with binding resolution methods
3. Extract binding methods into `BindingResolver` class
4. Inject CoverageProvider, IngestionService, DiscoveryClient dependencies
5. Preserve binding selection logic and priority ordering
6. Add proper type annotations and docstrings
7. Create comprehensive unit tests
8. Test coverage checks and binding validation

**Example Structure:**

```python
"""Market binding resolution for ML pipeline orchestration."""

from __future__ import annotations

from typing import Protocol
import logging

from ml.config.market_data import MarketDatasetInput
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.data.ingest.orchestrator import IngestionOrchestrator
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.discovery_client import DiscoveryClient
from ml.stores.protocols import CoverageProviderProtocol

logger = logging.getLogger(__name__)


class BindingResolverProtocol(Protocol):
    """Protocol for market binding resolution operations."""

    def resolve_market_inputs(
        self,
        cfg: DatasetBuildConfig,
        symbol_map: dict[str, tuple[str, ...]],
        start_ns: int,
        end_ns: int,
    ) -> tuple[
        tuple[MarketDatasetInput, ...] | None,
        tuple[ResolvedMarketBinding, ...],
    ]: ...

    def filter_candidate_bindings(
        self,
        candidates: tuple[ResolvedMarketBinding, ...],
        start_ns: int,
        end_ns: int,
        symbol: str,
        default_schema: str,
    ) -> tuple[ResolvedMarketBinding, ...]: ...


class BindingResolver:
    """
    Resolves market bindings with coverage validation.

    Handles binding discovery, filtering, priority selection, and
    validation against coverage and cost policies.
    """

    def __init__(
        self,
        coverage_provider: CoverageProviderProtocol | None = None,
        ingestion_service: Any | None = None,
        discovery_client: DiscoveryClient | None = None,
    ) -> None:
        """
        Initialize binding resolver.

        Parameters
        ----------
        coverage_provider : CoverageProviderProtocol | None
            Coverage provider for data availability checks
        ingestion_service : DatabentoIngestionService | None
            Ingestion service for availability and cost checks
        discovery_client : DiscoveryClient | None
            Discovery client for binding discovery
        """
        self.coverage = coverage_provider
        self.service = ingestion_service
        self.discovery_client = discovery_client
        self._initialize_metrics()

    def resolve_market_inputs(
        self,
        cfg: DatasetBuildConfig,
        symbol_map: dict[str, tuple[str, ...]],
        start_ns: int,
        end_ns: int,
    ) -> tuple[
        tuple[MarketDatasetInput, ...] | None,
        tuple[ResolvedMarketBinding, ...],
    ]:
        """
        Resolve market inputs with coverage validation.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset configuration
        symbol_map : dict[str, tuple[str, ...]]
            Symbol to instrument IDs mapping
        start_ns : int
            Start timestamp in nanoseconds
        end_ns : int
            End timestamp in nanoseconds

        Returns
        -------
        tuple[tuple[MarketDatasetInput, ...] | None, tuple[ResolvedMarketBinding, ...]]
            Resolved inputs and bindings
        """
        # Implementation extracted from pipeline_orchestrator.py:523-676
        ...

    # All binding resolution methods...
```

#### Day 4-5: Extract IngestionCoordinator (lines 1130-1770)

**Responsibility:** Ingestion pipeline coordination, backfill management, auto-fill universe, pre-ingestion tasks.

**Line Ranges to Extract:**

- `run_pre_ingestion()` (lines 1730-1770)
- `backfill()` (lines 1772-1786)
- `backfill_binding()` (lines 1788-1798)
- `backfill_coverage()` (lines 1800-1821)
- `_auto_fill_universe()` (lines 1130-1211)
- `_auto_fill_schema()` (lines 1212-1500+)
- Auto-fill L2/L3 methods
- Ingestion planning and execution methods

**Steps:**

1. Create `ml/orchestration/ingestion_coordinator.py`
2. Define `IngestionCoordinatorProtocol`
3. Extract ingestion orchestration methods
4. Take IngestionOrchestrator, DatabentoIngestor as dependencies
5. Preserve backfill logic and auto-fill universe functionality
6. Add proper type annotations and docstrings
7. Create comprehensive unit tests
8. Test backfill and auto-fill scenarios

**Example Structure:**

```python
"""Ingestion coordination for ML pipeline orchestration."""

from __future__ import annotations

from typing import Protocol, Any
from pathlib import Path
import logging

from ml.data.ingest.orchestrator import (
    BackfillWindowList,
    IngestionOrchestrator,
)
from ml.data.ingest.market_bindings import ResolvedMarketBinding
from ml.orchestration.config_types import (
    AutoFillUniverseConfig,
    DatasetBuildConfig,
    PreIngestionOptions,
)

logger = logging.getLogger(__name__)


class IngestionCoordinatorProtocol(Protocol):
    """Protocol for ingestion coordination operations."""

    def run_pre_ingestion(
        self,
        options: PreIngestionOptions,
        ingestion_cfg: Any,
    ) -> dict[str, Any]: ...

    def backfill(
        self,
        dataset_id: str,
        schema: str,
        symbols: tuple[str, ...],
        lookback_days: int,
    ) -> dict[str, BackfillWindowList]: ...

    def backfill_binding(
        self,
        binding: ResolvedMarketBinding,
        lookback_days: int,
    ) -> dict[str, BackfillWindowList]: ...


class IngestionCoordinator:
    """
    Coordinates ingestion pipelines and backfill operations.

    Manages pre-ingestion tasks, backfill scheduling, auto-fill universe
    population, and integration with ingestion services.
    """

    def __init__(
        self,
        ingestion_orchestrator: IngestionOrchestrator | None = None,
        ingestor: Any | None = None,
        service: Any | None = None,
        registry: Any | None = None,
        coverage_provider: Any | None = None,
    ) -> None:
        """
        Initialize ingestion coordinator.

        Parameters
        ----------
        ingestion_orchestrator : IngestionOrchestrator | None
            Orchestrator for ingestion operations
        ingestor : DatabentoIngestor | None
            Direct ingestor for backfill operations
        service : DatabentoIngestionService | None
            Ingestion service for dataset operations
        registry : RegistryProtocol | None
            Registry for dataset registration
        coverage_provider : CoverageProviderProtocol | None
            Coverage provider for gap analysis
        """
        self.ingestion_orchestrator = ingestion_orchestrator
        self.ingestor = ingestor
        self.service = service
        self.registry = registry
        self.coverage = coverage_provider
        self._initialize_metrics()

    def run_pre_ingestion(
        self,
        options: PreIngestionOptions,
        ingestion_cfg: Any,
    ) -> dict[str, Any]:
        """
        Run pre-ingestion tasks (L2/L3 population, macro data, etc.).

        Parameters
        ----------
        options : PreIngestionOptions
            Pre-ingestion configuration
        ingestion_cfg : IngestionStageConfig
            Ingestion stage configuration

        Returns
        -------
        dict[str, Any]
            Pre-ingestion results and metrics
        """
        # Implementation extracted from pipeline_orchestrator.py:1730-1770
        ...

    def backfill(
        self,
        dataset_id: str,
        schema: str,
        symbols: tuple[str, ...],
        lookback_days: int,
    ) -> dict[str, BackfillWindowList]:
        """
        Backfill market data for symbols.

        Parameters
        ----------
        dataset_id : str
            Dataset identifier
        schema : str
            Data schema
        symbols : tuple[str, ...]
            Symbols to backfill
        lookback_days : int
            Days to backfill

        Returns
        -------
        dict[str, BackfillWindowList]
            Backfill results by symbol
        """
        # Implementation extracted from pipeline_orchestrator.py:1772-1786
        ...

    # All ingestion coordination methods...
```

### Week 5: Extract Dataset Builder and Create Facade

#### Day 1-3: Extract DatasetBuilder (lines 1823-2395 + 2564-2880)

**Responsibility:** Dataset construction, validation, metadata management, pipeline execution.

**Line Ranges to Extract:**

- `build_dataset()` (lines 1823-2395)
- Dataset validation methods
- Pipeline signature computation
- Metadata expectations handling
- Dataset registration and storage
- Integration with HPO/training stages from `run()` method (lines 2564-2880)

**Steps:**

1. Create `ml/orchestration/dataset_builder.py`
2. Define `DatasetBuilderProtocol`
3. Extract dataset building methods
4. Take DataStore, Registry, ConfigResolver dependencies
5. Preserve validation and metadata logic
6. Add proper type annotations and docstrings
7. Create comprehensive unit tests
8. Test dataset validation and pipeline execution

**Example Structure:**

```python
"""Dataset building for ML pipeline orchestration."""

from __future__ import annotations

from typing import Protocol, Any
from pathlib import Path
import logging

from ml.data import (
    DatasetMetadata,
    DatasetMetadataExpectations,
    DatasetValidationConfig,
)
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.shared_types import BuildArtifacts
from ml.stores.protocols import DataStoreFacadeProtocol
from ml.registry.protocols import RegistryProtocol

logger = logging.getLogger(__name__)


class DatasetBuilderProtocol(Protocol):
    """Protocol for dataset building operations."""

    def build_dataset(
        self,
        cfg: DatasetBuildConfig,
    ) -> int: ...

    def validate_dataset(
        self,
        dataset_path: Path,
        expectations: DatasetMetadataExpectations,
        validation_config: DatasetValidationConfig,
    ) -> tuple[bool, DatasetMetadata]: ...


class DatasetBuilder:
    """
    Builds and validates ML datasets.

    Handles dataset construction from market data, feature engineering,
    validation against expectations, metadata management, and storage.
    """

    def __init__(
        self,
        data_store: DataStoreFacadeProtocol | None = None,
        registry: RegistryProtocol | None = None,
        config_resolver: Any | None = None,
        binding_resolver: Any | None = None,
        ingestion_coordinator: Any | None = None,
        default_data_dir: Path | None = None,
    ) -> None:
        """
        Initialize dataset builder.

        Parameters
        ----------
        data_store : DataStoreFacadeProtocol | None
            Data store for dataset persistence
        registry : RegistryProtocol | None
            Registry for dataset registration
        config_resolver : ConfigResolver | None
            Configuration resolver
        binding_resolver : BindingResolver | None
            Binding resolver for market data
        ingestion_coordinator : IngestionCoordinator | None
            Ingestion coordinator for data acquisition
        default_data_dir : Path | None
            Default data directory
        """
        self.data_store = data_store
        self.registry = registry
        self.config_resolver = config_resolver
        self.binding_resolver = binding_resolver
        self.ingestion_coordinator = ingestion_coordinator
        self.default_data_dir = default_data_dir or Path.cwd() / "data"
        self._initialize_metrics()

    def build_dataset(
        self,
        cfg: DatasetBuildConfig,
    ) -> int:
        """
        Build ML dataset from configuration.

        Parameters
        ----------
        cfg : DatasetBuildConfig
            Dataset build configuration

        Returns
        -------
        int
            Exit code (0 for success)
        """
        # Implementation extracted from pipeline_orchestrator.py:1823-2395
        ...

    def validate_dataset(
        self,
        dataset_path: Path,
        expectations: DatasetMetadataExpectations,
        validation_config: DatasetValidationConfig,
    ) -> tuple[bool, DatasetMetadata]:
        """
        Validate dataset against expectations.

        Parameters
        ----------
        dataset_path : Path
            Path to dataset file
        expectations : DatasetMetadataExpectations
            Expected dataset metadata
        validation_config : DatasetValidationConfig
            Validation configuration

        Returns
        -------
        tuple[bool, DatasetMetadata]
            (validation_passed, dataset_metadata)
        """
        # Implementation of validation logic
        ...

    # All dataset building methods...
```

#### Day 4-5: Create MLPipelineOrchestrator Facade

**Responsibility:** Maintain backward-compatible public API by delegating to specialized components.

**Steps:**

1. Create `ml/orchestration/pipeline_orchestrator_facade.py`
2. Replicate all public methods from original MLPipelineOrchestrator
3. Initialize all 5 components in constructor
4. Delegate each method to appropriate component(s)
5. Add feature flag check: `ML_USE_LEGACY_PIPELINE_ORCHESTRATOR`
6. If flag is set, use original implementation; otherwise use new components
7. Add proper type annotations and docstrings
8. Preserve all CLI integration points

**Example Structure:**

```python
"""
MLPipelineOrchestrator facade maintaining backward compatibility.

This facade delegates to specialized components while preserving the original
public API. Feature flag ML_USE_LEGACY_PIPELINE_ORCHESTRATOR controls legacy vs new path.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

from ml.orchestration.config_resolver import ConfigResolver
from ml.orchestration.ingestion_coordinator import IngestionCoordinator
from ml.orchestration.dataset_builder import DatasetBuilder
from ml.orchestration.binding_resolver import BindingResolver
from ml.orchestration.discovery_client import DiscoveryClient
from ml.orchestration.config_types import (
    DatasetBuildConfig,
    HPOConfig,
    OrchestratorConfig,
    TeacherTrainConfig,
    StudentDistillConfig,
)
from ml.orchestration.shared_types import BuildArtifacts

logger = logging.getLogger(__name__)


class MLPipelineOrchestrator:
    """
    High-level ML pipeline orchestrator (cold path only).

    This facade delegates to specialized components while maintaining
    100% backward compatibility with the original MLPipelineOrchestrator API.
    """

    def __init__(
        self,
        *,
        connection_string: str | None = None,
        registry: Any | None = None,
        data_store: Any | None = None,
        ingestion_orchestrator: Any | None = None,
        ingestor: Any | None = None,
        service: Any | None = None,
        dataset_discovery: Any | None = None,
        coverage_provider: Any | None = None,
        default_data_dir: Path | None = None,
    ) -> None:
        """
        Initialize MLPipelineOrchestrator.

        Parameters match original constructor for compatibility.
        """
        # Check feature flag
        use_legacy = os.getenv("ML_USE_LEGACY_PIPELINE_ORCHESTRATOR", "0") == "1"

        if use_legacy:
            # Use original implementation
            logger.info("Using legacy MLPipelineOrchestrator implementation")
            from ml.orchestration.pipeline_orchestrator_legacy import (
                MLPipelineOrchestratorLegacy,
            )
            self._impl = MLPipelineOrchestratorLegacy(
                connection_string=connection_string,
                registry=registry,
                data_store=data_store,
                ingestion_orchestrator=ingestion_orchestrator,
                ingestor=ingestor,
                service=service,
                dataset_discovery=dataset_discovery,
                coverage_provider=coverage_provider,
                default_data_dir=default_data_dir,
            )
            self._use_legacy = True
        else:
            # Use new component-based implementation
            logger.info("Using component-based MLPipelineOrchestrator implementation")

            # Initialize components
            self._config_resolver = ConfigResolver()
            self._discovery_client = DiscoveryClient(
                dataset_discovery=dataset_discovery,
            )
            self._binding_resolver = BindingResolver(
                coverage_provider=coverage_provider,
                ingestion_service=service,
                discovery_client=self._discovery_client,
            )
            self._ingestion_coordinator = IngestionCoordinator(
                ingestion_orchestrator=ingestion_orchestrator,
                ingestor=ingestor,
                service=service,
                registry=registry,
                coverage_provider=coverage_provider,
            )
            self._dataset_builder = DatasetBuilder(
                data_store=data_store,
                registry=registry,
                config_resolver=self._config_resolver,
                binding_resolver=self._binding_resolver,
                ingestion_coordinator=self._ingestion_coordinator,
                default_data_dir=default_data_dir,
            )
            self._use_legacy = False

            # Store dependencies for direct access if needed
            self.registry = registry
            self.data_store = data_store
            self.service = service
            self.coverage = coverage_provider

    # Delegate all public methods
    def run_pre_ingestion(self, *args, **kwargs):
        """Run pre-ingestion tasks."""
        if self._use_legacy:
            return self._impl.run_pre_ingestion(*args, **kwargs)
        return self._ingestion_coordinator.run_pre_ingestion(*args, **kwargs)

    def backfill(self, *args, **kwargs):
        """Backfill market data."""
        if self._use_legacy:
            return self._impl.backfill(*args, **kwargs)
        return self._ingestion_coordinator.backfill(*args, **kwargs)

    def backfill_binding(self, *args, **kwargs):
        """Backfill market data for binding."""
        if self._use_legacy:
            return self._impl.backfill_binding(*args, **kwargs)
        return self._ingestion_coordinator.backfill_binding(*args, **kwargs)

    def backfill_coverage(self, *args, **kwargs):
        """Backfill coverage gaps."""
        if self._use_legacy:
            return self._impl.backfill_coverage(*args, **kwargs)
        return self._ingestion_coordinator.backfill_coverage(*args, **kwargs)

    def build_dataset(self, cfg: DatasetBuildConfig) -> int:
        """Build ML dataset."""
        if self._use_legacy:
            return self._impl.build_dataset(cfg)
        return self._dataset_builder.build_dataset(cfg)

    def run_hpo(self, cfg: HPOConfig, dataset_csv: Path, out_dir: Path) -> int:
        """Run hyperparameter optimization."""
        if self._use_legacy:
            return self._impl.run_hpo(cfg, dataset_csv, out_dir)
        # HPO logic remains in facade for now (may extract in future)
        ...

    def train_teacher(
        self,
        cfg: TeacherTrainConfig,
        dataset_csv: Path,
        out_dir: Path,
    ) -> int:
        """Train teacher model."""
        if self._use_legacy:
            return self._impl.train_teacher(cfg, dataset_csv, out_dir)
        # Training logic remains in facade for now (may extract in future)
        ...

    def distill_student(
        self,
        cfg: StudentDistillConfig,
        dataset_csv: Path,
        teacher_dir: Path,
        out_dir: Path,
    ) -> int:
        """Distill student model."""
        if self._use_legacy:
            return self._impl.distill_student(cfg, dataset_csv, teacher_dir, out_dir)
        # Distillation logic remains in facade for now (may extract in future)
        ...

    def run(self, cfg: OrchestratorConfig) -> int:
        """Run full ML pipeline."""
        if self._use_legacy:
            return self._impl.run(cfg)
        # Full pipeline orchestration using components
        ...

    def run_training_only(self, cfg: OrchestratorConfig) -> int:
        """Run training-only pipeline."""
        if self._use_legacy:
            return self._impl.run_training_only(cfg)
        # Training-only pipeline using components
        ...

    def get_health_status(self) -> dict[str, Any]:
        """Get health status from all components."""
        if self._use_legacy:
            return self._impl.get_health_status()

        return {
            "config_resolver": "healthy",
            "binding_resolver": "healthy",
            "ingestion_coordinator": "healthy",
            "dataset_builder": "healthy",
            "discovery_client": "healthy",
        }
```

#### Day 6-7: Integration Testing and Validation

**Steps:**

1. Move original `pipeline_orchestrator.py` to `pipeline_orchestrator_legacy.py`
2. Update `ml/orchestration/__init__.py` exports
3. Create integration test suite:
   - Test facade with legacy flag ON - verify original behavior
   - Test facade with legacy flag OFF - verify new behavior
   - Compare outputs between legacy and new implementations
   - Test all public APIs for backward compatibility
   - Test CLI integration points
4. Run full test suite
5. Fix any failures
6. Performance benchmarking (ensure no regression)
7. Update documentation

## Testing Strategy

### Unit Tests (Per Component)

```python
# ml/tests/unit/orchestration/test_config_resolver.py
def test_config_resolver_apply_default_market_inputs():
    """Test market input defaults applied correctly."""
    ...

def test_config_resolver_collect_symbol_map():
    """Test symbol mapping collection."""
    ...

def test_config_resolver_prepare_dataset_config():
    """Test dataset config preparation."""
    ...


# ml/tests/unit/orchestration/test_discovery_client.py
def test_discovery_client_discover_market_inputs():
    """Test market input discovery."""
    ...

def test_discovery_client_error_handling():
    """Test discovery error handling."""
    ...


# ml/tests/unit/orchestration/test_binding_resolver.py
def test_binding_resolver_filter_candidates():
    """Test binding candidate filtering."""
    ...

def test_binding_resolver_priority_selection():
    """Test binding priority selection."""
    ...

def test_binding_resolver_coverage_validation():
    """Test binding coverage validation."""
    ...


# ml/tests/unit/orchestration/test_ingestion_coordinator.py
def test_ingestion_coordinator_backfill():
    """Test backfill coordination."""
    ...

def test_ingestion_coordinator_auto_fill_universe():
    """Test auto-fill universe."""
    ...


# ml/tests/unit/orchestration/test_dataset_builder.py
def test_dataset_builder_build_success():
    """Test successful dataset build."""
    ...

def test_dataset_builder_validation():
    """Test dataset validation."""
    ...
```

### Integration Tests

```python
# ml/tests/integration/orchestration/test_pipeline_orchestrator_facade.py
def test_facade_backward_compatibility():
    """Verify facade matches original API."""
    legacy = MLPipelineOrchestratorLegacy(...)
    facade = MLPipelineOrchestrator(...)  # Feature flag OFF

    # Compare outputs for same operations
    assert facade.build_dataset(...) == legacy.build_dataset(...)
    ...

def test_facade_feature_flag_legacy():
    """Test facade with ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=1."""
    os.environ["ML_USE_LEGACY_PIPELINE_ORCHESTRATOR"] = "1"
    orchestrator = MLPipelineOrchestrator(...)
    # Verify using legacy path
    ...

def test_facade_feature_flag_new():
    """Test facade with ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=0."""
    os.environ["ML_USE_LEGACY_PIPELINE_ORCHESTRATOR"] = "0"
    orchestrator = MLPipelineOrchestrator(...)
    # Verify using new component path
    ...

def test_cli_integration():
    """Test CLI integration preserved."""
    # Test main() entry point with both legacy and new paths
    ...
```

## Testing Requirements

### Unit Tests (≥90% coverage per component)

- [ ] test_config_resolver.py - All configuration methods
- [ ] test_discovery_client.py - All discovery operations
- [ ] test_binding_resolver.py - All binding resolution
- [ ] test_ingestion_coordinator.py - All ingestion operations
- [ ] test_dataset_builder.py - All dataset building

### Integration Tests

- [ ] test_pipeline_orchestrator_facade.py - Backward compatibility
- [ ] test_feature_flag_toggle.py - Legacy vs new behavior
- [ ] test_component_integration.py - Components work together
- [ ] test_cli_integration.py - CLI entry points work

### Performance Tests

- [ ] Benchmark dataset building (target: no regression)
- [ ] Benchmark backfill operations (should not regress)
- [ ] Benchmark binding resolution (should not regress)

### Regression Tests

- [ ] All existing orchestrator tests pass unchanged
- [ ] No behavioral changes in public APIs
- [ ] CLI commands work identically
- [ ] Integration with HPO/training unchanged

## Rollback Plan

### Immediate Rollback (Production Issue)

```bash
# Set feature flag to use legacy implementation
export ML_USE_LEGACY_PIPELINE_ORCHESTRATOR=1

# Restart services
kubectl rollout restart deployment/ml-orchestrator
```

### Code Rollback (Development Issue)

```bash
# Revert all new files
git checkout ml/orchestration/config_resolver.py
git checkout ml/orchestration/ingestion_coordinator.py
git checkout ml/orchestration/dataset_builder.py
git checkout ml/orchestration/binding_resolver.py
git checkout ml/orchestration/discovery_client.py
git checkout ml/orchestration/shared_types.py
git checkout ml/orchestration/pipeline_orchestrator_facade.py
git checkout ml/orchestration/pipeline_orchestrator_legacy.py

# Restore original
git checkout ml/orchestration/pipeline_orchestrator.py
git checkout ml/orchestration/__init__.py

# Revert tests
git checkout ml/tests/unit/orchestration/test_config_resolver.py
git checkout ml/tests/unit/orchestration/test_ingestion_coordinator.py
git checkout ml/tests/unit/orchestration/test_dataset_builder.py
git checkout ml/tests/unit/orchestration/test_binding_resolver.py
git checkout ml/tests/unit/orchestration/test_discovery_client.py
git checkout ml/tests/integration/orchestration/test_pipeline_orchestrator_facade.py
```

### Verification After Rollback

```bash
# Run tests
pytest ml/tests/unit/orchestration/ -v
pytest ml/tests/integration/orchestration/ -v

# Verify imports
python -c "from ml.orchestration import MLPipelineOrchestrator; print('OK')"

# Verify CLI
python -m ml.orchestration.pipeline_orchestrator --help
```

## Success Metrics

### Code Quality Metrics

- Lines reduced: 4,598 → ~3,450 (6 components + facade)
- Average file size: 4,598 → ~575 lines (87% reduction per file)
- Cyclomatic complexity: Reduced by ~70% (smaller focused functions)
- Test coverage: ≥90% per component (up from ~70% for monolith)

### Architecture Metrics

- Number of responsibilities: 1 god class → 5 focused components
- Files affected: 1 → 7 (5 components + shared types + facade)
- Circular dependencies: 0 (no new cycles introduced)
- Protocol conformance: 100% (all components implement protocols)

### Performance Metrics

- Dataset build latency: No regression
- Backfill operation latency: No regression
- Binding resolution latency: <100ms P99 (no regression)
- Memory usage: ≤10% increase (acceptable for better structure)

### Testing Metrics

- Unit tests: +5 test files, ~150 new tests
- Integration tests: +1 test file, ~15 new tests
- Test execution time: <10% increase (parallel execution)
- Coverage: 70% → 90% (improved testability)

### Maintainability Metrics

- Cognitive load: Reduced (smaller focused classes)
- Onboarding time: Faster (clearer separation of concerns)
- Change impact: Localized (changes affect single component)
- Documentation: Improved (clear component boundaries)

## Notes

### Critical Requirements

- **100% Backward Compatibility:** All public APIs must work identically
- **Feature Flag:** `ML_USE_LEGACY_PIPELINE_ORCHESTRATOR` must work for safe rollout
- **No Breaking Changes:** Existing code using orchestrator must work unchanged
- **Zero New Cycles:** No circular dependencies introduced
- **Strangler Fig Pattern:** New code alongside old, then switch via flag
- **CLI Preservation:** All CLI entry points must continue working

### Strangler Fig Pattern Benefits

- Safe incremental migration
- Easy rollback via feature flag
- Low risk to production
- Allows parallel testing of old vs new
- Confidence in correctness before full cutover

### Component Interaction Flow

```
┌─────────────────────────────────────────────────────────────┐
│         MLPipelineOrchestrator (Public API)                  │
│  - Maintains backward compatibility                          │
│  - Feature flag toggle (legacy vs new)                       │
│  - CLI integration preserved                                 │
└──────────────┬──────────────────────────────────────────────┘
               │
               ├──> ConfigResolver
               │    - Market input defaults
               │    - Symbol mapping
               │    - Window bounds
               │
               ├──> DiscoveryClient
               │    - Dataset discovery
               │    - Service health checks
               │    - Coverage queries
               │
               ├──> BindingResolver
               │    - Market binding resolution
               │    - Coverage validation
               │    - Priority selection
               │    └──> DiscoveryClient (composition)
               │
               ├──> IngestionCoordinator
               │    - Backfill management
               │    - Auto-fill universe
               │    - Pre-ingestion tasks
               │
               └──> DatasetBuilder
                    - Dataset construction
                    - Validation
                    - Metadata management
                    └──> ConfigResolver (composition)
                    └──> BindingResolver (composition)
                    └──> IngestionCoordinator (composition)
```

### Architecture Decisions

1. **Protocol-First:** All components implement protocols for testability
2. **Dependency Injection:** Components receive dependencies in constructor
3. **Composition Over Inheritance:** Components compose each other, not inherit
4. **Single Responsibility:** Each component has one clear purpose
5. **Metrics Bootstrap:** Use `ml.common.metrics_bootstrap` for all metrics
6. **Shared Types Module:** Common dataclasses in `shared_types.py` to avoid duplication

### Migration Path

1. **Week 3:** Extract independent components (ConfigResolver, DiscoveryClient)
2. **Week 4:** Extract dependent components (BindingResolver, IngestionCoordinator)
3. **Week 5:** Extract DatasetBuilder and create facade
4. **Week 6:** Integration testing and validation
5. **Week 7:** Deploy with flag ON (legacy mode)
6. **Week 8:** Gradual rollout with flag OFF (new mode)
7. **Week 9:** Remove legacy code if new mode stable

### Known Challenges

- **CLI Integration:** Must preserve all CLI entry points and argument parsing
- **State Management:** Original orchestrator has minimal state - should remain stateless
- **Component Dependencies:** DatasetBuilder depends on multiple other components
- **Training Integration:** HPO/training methods may remain in facade initially
- **Metrics:** Must maintain identical metric names and labels for dashboard compatibility
- **Error Handling:** Must maintain identical error behavior and logging
- **Performance:** Large file with complex logic - must ensure no performance degradation

### Future Considerations

- **Phase 3:** Further decomposition of HPO/training logic into separate components
- **Phase 4:** Extract CLI logic into separate module
- **Phase 5:** Consider async/concurrent orchestration for parallel operations
- **Phase 6:** Improve observability with distributed tracing

### Dependencies on Other Tasks

- **Blocks:** Phase 2.3 (ModelRegistry decomposition) - uses orchestrator patterns
- **Blocked By:** Phase 2.1 (DataStore decomposition) - uses DataStore facade
- **Related:** Phase 1 DRY violations - cleaner configs improve orchestration

---

## Approval Checklist

Before marking this task complete, verify:

- [ ] All 5 components extracted and tested independently
- [ ] Facade delegates correctly to all components
- [ ] Feature flag tested in both states (legacy ON and OFF)
- [ ] All existing tests pass without modification
- [ ] New unit tests achieve ≥90% coverage per component
- [ ] Integration tests verify backward compatibility
- [ ] CLI commands work identically in both modes
- [ ] No new circular dependencies (verified with import checks)
- [ ] Ruff, MyPy, and pattern validation all pass
- [ ] Performance benchmarks show no regression
- [ ] Documentation updated with architecture diagrams
- [ ] Rollback plan documented and tested
- [ ] TASK_REPORT.md generated with detailed changes
