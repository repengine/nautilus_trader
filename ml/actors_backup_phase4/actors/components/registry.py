"""
Registry Operations Component.

This module implements Universal Pattern #1 (Mandatory 4-Registry Integration) and
Universal Pattern #4 (Progressive Fallback Chains with caching) from CLAUDE.md.

All ML actors MUST use this component to initialize and access the 4 mandatory registries:
- FeatureRegistry
- ModelRegistry
- StrategyRegistry
- DataRegistry

The component provides:
- Automatic registry initialization with PostgreSQL
- Progressive fallback chain: PostgreSQL → File-based loading
- Query result caching with TTL for performance
- Model loading with multiple fallback strategies
- Feature name mapping from manifests
- Metadata extraction and constraint validation
- Centralized metrics for fallback activations
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Protocol

from ml.common.metrics_bootstrap import get_counter
from ml.config.base import MLActorConfig


if TYPE_CHECKING:
    from ml.registry.data import DataRegistry
    from ml.registry.feature import FeatureRegistry
    from ml.registry.model import ModelRegistry
    from ml.registry.strategy import StrategyRegistry


class RegistryProtocol(Protocol):
    """
    Protocol for registry operations component.

    Defines the interface for managing all 4 mandatory ML registries with
    progressive fallback chains and caching.
    """

    @property
    def feature_registry(self) -> FeatureRegistry:
        """Return initialized FeatureRegistry."""
        ...

    @property
    def model_registry(self) -> ModelRegistry:
        """Return initialized ModelRegistry."""
        ...

    @property
    def strategy_registry(self) -> StrategyRegistry:
        """Return initialized StrategyRegistry."""
        ...

    @property
    def data_registry(self) -> DataRegistry:
        """Return initialized DataRegistry."""
        ...


class RegistryComponent:
    """
    Manages initialization and access to all 4 mandatory ML registries.

    Implements Universal Pattern #1 (Mandatory 4-Registry Integration) and
    Universal Pattern #4 (Progressive Fallback Chains with caching).

    The component handles:
    - Registry initialization with PostgreSQL (PRIMARY)
    - Fallback to file-based loading when PostgreSQL unavailable (FALLBACK)
    - Query result caching with TTL (300s default) to reduce database load
    - Model loading with multiple fallback strategies
    - Feature name mapping from manifest during prediction
    - Metadata extraction (schema hashes, deployment constraints)
    - Metrics emission for fallback activations

    Example:
        >>> config = MLActorConfig(
        ...     actor_id="my_actor",
        ...     db_connection="postgresql://localhost/nautilus",
        ...     enable_registry_caching=True,
        ...     registry_cache_ttl=300,
        ... )
        >>> component = RegistryComponent(config)
        >>> # Registries are initialized automatically
        >>> feature_registry = component.feature_registry
        >>> schema = component._query_feature_registry("rsi_14")
    """

    def __init__(
        self,
        config: MLActorConfig,
        logger: logging.Logger | None = None,
        services: Any | None = None,
    ) -> None:
        """
        Initialize registry operations component.

        Args:
            config: ML actor configuration containing database connection
                    and registry settings
            logger: Optional logger (defaults to module logger)
            services: Pre-initialized ActorServices (to avoid duplicate initialization)
                      If None, will call init_actor_services internally

        Raises:
            RuntimeError: If allow_dummy_fallback=False and PostgreSQL connection fails
        """
        self._config = config
        self._logger = logger or logging.getLogger(__name__)

        # Registry references (initialized in _init_registries)
        self._feature_registry: FeatureRegistry | None = None
        self._model_registry: ModelRegistry | None = None
        self._strategy_registry: StrategyRegistry | None = None
        self._data_registry: DataRegistry | None = None

        # Cache for query results (reduces database load)
        self._cache: dict[str, tuple[Any, float]] = {}
        self._cache_ttl = getattr(config, "registry_cache_ttl", 300.0)  # 5 minutes default
        self._enable_caching = getattr(config, "enable_registry_caching", True)

        # Metrics for fallback tracking
        self._fallback_counter = get_counter(
            "ml_registry_fallback_total",
            "Total registry fallback activations by registry and stage",
            labelnames=("registry", "stage"),
        )

        # Cache hit/miss metrics
        self._cache_hit_counter = get_counter(
            "ml_registry_cache_hits_total",
            "Total registry cache hits by registry",
            labelnames=("registry",),
        )
        self._cache_miss_counter = get_counter(
            "ml_registry_cache_misses_total",
            "Total registry cache misses by registry",
            labelnames=("registry",),
        )

        # Initialize registries (pass pre-initialized services if provided)
        self._init_registries(services=services)

    def _init_registries(self, services: Any | None = None) -> None:
        """
        Initialize all 4 registries with progressive fallback.

        Attempts to initialize PostgreSQL-backed registries first.
        Falls back to file-based loading if PostgreSQL unavailable.

        Args:
            services: Pre-initialized ActorServices (to avoid duplicate initialization)
                      If None, will call init_actor_services internally

        Raises:
            RuntimeError: If allow_dummy_fallback=False and PostgreSQL fails
        """
        from ml.actors.actor_services import init_actor_services

        try:
            # Use pre-initialized services if provided, otherwise initialize now
            if services is None:
                services = init_actor_services(self._config)

            self._feature_registry = services.feature_registry
            self._model_registry = services.model_registry
            self._strategy_registry = services.strategy_registry
            self._data_registry = services.data_registry

            self._logger.info("Initialized all 4 registries with PostgreSQL")

        except Exception as e:
            # Handle fallback based on configuration
            allow_fallback = getattr(self._config, "allow_dummy_fallback", True)

            if not allow_fallback:
                self._logger.error(
                    f"Failed to initialize registries: {e}",
                    exc_info=True,
                )
                raise RuntimeError(f"Failed to initialize registries: {e}") from e

            # Fallback to file-based loading
            self._logger.warning(
                f"Failed to initialize registries with PostgreSQL, "
                f"falling back to file-based loading: {e}",
                exc_info=True,
            )

            # Initialize with fallback - init_actor_services handles this internally
            try:
                services = init_actor_services(self._config)
            except Exception as fallback_error:
                self._logger.error(
                    f"Even fallback initialization failed: {fallback_error}",
                    exc_info=True,
                )
                raise

            self._feature_registry = services.feature_registry
            self._model_registry = services.model_registry
            self._strategy_registry = services.strategy_registry
            self._data_registry = services.data_registry

            # Emit fallback metrics for each registry
            self._fallback_counter.labels(registry="feature", stage="file").inc()
            self._fallback_counter.labels(registry="model", stage="file").inc()
            self._fallback_counter.labels(registry="strategy", stage="file").inc()
            self._fallback_counter.labels(registry="data", stage="file").inc()

            self._logger.info(
                "Initialized all 4 registries with file-based fallback"
            )

    @property
    def feature_registry(self) -> FeatureRegistry:
        """
        Return initialized FeatureRegistry.

        Returns:
            FeatureRegistry instance (PostgreSQL or file-based fallback)

        Raises:
            RuntimeError: If registries not initialized
        """
        if self._feature_registry is None:
            raise RuntimeError("FeatureRegistry not initialized")
        return self._feature_registry

    @property
    def model_registry(self) -> ModelRegistry:
        """
        Return initialized ModelRegistry.

        Returns:
            ModelRegistry instance (PostgreSQL or file-based fallback)

        Raises:
            RuntimeError: If registries not initialized
        """
        if self._model_registry is None:
            raise RuntimeError("ModelRegistry not initialized")
        return self._model_registry

    @property
    def strategy_registry(self) -> StrategyRegistry:
        """
        Return initialized StrategyRegistry.

        Returns:
            StrategyRegistry instance (PostgreSQL or file-based fallback)

        Raises:
            RuntimeError: If registries not initialized
        """
        if self._strategy_registry is None:
            raise RuntimeError("StrategyRegistry not initialized")
        return self._strategy_registry

    @property
    def data_registry(self) -> DataRegistry:
        """
        Return initialized DataRegistry.

        Returns:
            DataRegistry instance (PostgreSQL or file-based fallback)

        Raises:
            RuntimeError: If registries not initialized
        """
        if self._data_registry is None:
            raise RuntimeError("DataRegistry not initialized")
        return self._data_registry

    def _get_feature_registry(self) -> FeatureRegistry:
        """Get cached FeatureRegistry reference (hot path <1μs)."""
        if self._feature_registry is None:
            raise RuntimeError("FeatureRegistry not initialized")
        return self._feature_registry

    def _get_model_registry(self) -> ModelRegistry:
        """Get cached ModelRegistry reference (hot path <1μs)."""
        if self._model_registry is None:
            raise RuntimeError("ModelRegistry not initialized")
        return self._model_registry

    def _get_strategy_registry(self) -> StrategyRegistry:
        """Get cached StrategyRegistry reference (hot path <1μs)."""
        if self._strategy_registry is None:
            raise RuntimeError("StrategyRegistry not initialized")
        return self._strategy_registry

    def _get_data_registry(self) -> DataRegistry:
        """Get cached DataRegistry reference (hot path <1μs)."""
        if self._data_registry is None:
            raise RuntimeError("DataRegistry not initialized")
        return self._data_registry

    def _query_feature_registry(self, feature_name: str) -> dict[str, Any] | None:
        """
        Query feature schema from FeatureRegistry with caching.

        Args:
            feature_name: Name of feature to query

        Returns:
            Feature schema dict or None if not found
        """
        cache_key = f"feature:{feature_name}"

        # Check cache first
        if self._enable_caching and cache_key in self._cache:
            cached_value, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                self._cache_hit_counter.labels(registry="feature").inc()
                return cached_value  # type: ignore[no-any-return]

        # Cache miss - query registry
        self._cache_miss_counter.labels(registry="feature").inc()

        try:
            registry = self._get_feature_registry()
            # Query registry for feature schema
            # This is a placeholder - actual query method depends on FeatureRegistry API
            if hasattr(registry, "get_feature_manifest"):
                result: Any = registry.get_feature_manifest(feature_name)
                if result is not None:
                    # Convert to dict if needed
                    schema_dict: dict[str, Any] = result if isinstance(result, dict) else result.__dict__
                    # Cache result
                    if self._enable_caching:
                        self._cache[cache_key] = (schema_dict, time.time())
                    return schema_dict
            return None
        except Exception as e:
            self._logger.error(
                f"Failed to query feature registry for {feature_name}: {e}",
                exc_info=True,
            )
            return None

    def _query_model_registry(self, model_id: str) -> dict[str, Any] | None:
        """
        Query model manifest from ModelRegistry with caching.

        Args:
            model_id: ID of model to query

        Returns:
            Model manifest dict or None if not found
        """
        cache_key = f"model:{model_id}"

        # Check cache first
        if self._enable_caching and cache_key in self._cache:
            cached_value, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                self._cache_hit_counter.labels(registry="model").inc()
                return cached_value  # type: ignore[no-any-return]

        # Cache miss - query registry
        self._cache_miss_counter.labels(registry="model").inc()

        try:
            registry = self._get_model_registry()
            # Query registry for model manifest
            if hasattr(registry, "get_model"):
                result: Any = registry.get_model(model_id)
                if result is not None:
                    # Convert to dict if needed
                    manifest_dict: dict[str, Any] = result if isinstance(result, dict) else result.__dict__
                    # Cache result
                    if self._enable_caching:
                        self._cache[cache_key] = (manifest_dict, time.time())
                    return manifest_dict
            return None
        except Exception as e:
            self._logger.error(
                f"Failed to query model registry for {model_id}: {e}",
                exc_info=True,
            )
            return None

    def _query_strategy_registry(self, strategy_id: str) -> dict[str, Any] | None:
        """
        Query strategy config from StrategyRegistry with caching.

        Args:
            strategy_id: ID of strategy to query

        Returns:
            Strategy config dict or None if not found
        """
        cache_key = f"strategy:{strategy_id}"

        # Check cache first
        if self._enable_caching and cache_key in self._cache:
            cached_value, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                self._cache_hit_counter.labels(registry="strategy").inc()
                return cached_value  # type: ignore[no-any-return]

        # Cache miss - query registry
        self._cache_miss_counter.labels(registry="strategy").inc()

        try:
            registry = self._get_strategy_registry()
            # Query registry for strategy config
            if hasattr(registry, "get_strategy"):
                result: Any = registry.get_strategy(strategy_id)
                if result is not None:
                    # Convert to dict if needed
                    config_dict: dict[str, Any] = result if isinstance(result, dict) else result.__dict__
                    # Cache result
                    if self._enable_caching:
                        self._cache[cache_key] = (config_dict, time.time())
                    return config_dict
            return None
        except Exception as e:
            self._logger.error(
                f"Failed to query strategy registry for {strategy_id}: {e}",
                exc_info=True,
            )
            return None

    def _query_data_registry(self, dataset_id: str) -> dict[str, Any] | None:
        """
        Query dataset metadata from DataRegistry with caching.

        Args:
            dataset_id: ID of dataset to query

        Returns:
            Dataset metadata dict or None if not found
        """
        cache_key = f"data:{dataset_id}"

        # Check cache first
        if self._enable_caching and cache_key in self._cache:
            cached_value, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                self._cache_hit_counter.labels(registry="data").inc()
                return cached_value  # type: ignore[no-any-return]

        # Cache miss - query registry
        self._cache_miss_counter.labels(registry="data").inc()

        try:
            registry = self._get_data_registry()
            # Query registry for dataset metadata
            if hasattr(registry, "get_dataset"):
                result: Any = registry.get_dataset(dataset_id)
                if result is not None:
                    # Convert to dict if needed
                    metadata_dict: dict[str, Any] = result if isinstance(result, dict) else result.__dict__
                    # Cache result
                    if self._enable_caching:
                        self._cache[cache_key] = (metadata_dict, time.time())
                    return metadata_dict
            return None
        except Exception as e:
            self._logger.error(
                f"Failed to query data registry for {dataset_id}: {e}",
                exc_info=True,
            )
            return None

    def _is_cache_expired(self, key: str) -> bool:
        """
        Check if cache entry is expired.

        Args:
            key: Cache key to check

        Returns:
            True if expired or not found, False otherwise
        """
        if key not in self._cache:
            return True

        _, cached_time = self._cache[key]
        return time.time() - cached_time >= self._cache_ttl

    def _clear_cache(self) -> None:
        """Clear all cached query results."""
        self._cache.clear()

    def _try_load_from_registry(self) -> bool:
        """
        Attempt to load model and metadata from registry; return True if loaded.

        This method reads from self._config.model_id (NO PARAMETERS per V2 correction).

        Priority:
        1. Use model_id with shared ModelRegistry (preferred)
        2. Fall back to model_path for testing/development

        Returns:
            True if model loaded from registry, False otherwise

        Raises:
            ValueError: If model_id provided but not found and no fallback path
        """
        # Check if we have a model_id to use with registry
        if not hasattr(self._config, "model_id") or not self._config.model_id:
            return False

        # Use the shared registry instance
        registry = self._model_registry
        if registry is None:
            raise RuntimeError("ModelRegistry not initialized")

        model_info: Any = registry.get_model(self._config.model_id)
        if not model_info:
            # If model_id provided but not found, check for fallback path
            if hasattr(self._config, "model_path") and self._config.model_path:
                self._logger.warning(
                    f"Model {self._config.model_id} not found in registry, "
                    f"falling back to direct path: {self._config.model_path}",
                )
                # Emit fallback metric
                self._fallback_counter.labels(registry="model", stage="file_path").inc()
                return False  # Let the fallback path handle it
            raise ValueError(f"Model {self._config.model_id} not found in registry and no fallback model_path provided")

        # Extract metadata from manifest
        manifest = model_info.manifest

        # Populate _model_metadata (this will be accessed by actor)
        self._model_metadata = {
            "model_id": manifest.model_id,
            "version": manifest.version,
            "type": manifest.architecture,
            "role": manifest.role.value,
            "data_requirements": manifest.data_requirements.value,
            "feature_schema": manifest.feature_schema,
            "feature_schema_hash": manifest.feature_schema_hash,
            "parent_id": manifest.parent_id,
            "performance_metrics": manifest.performance_metrics,
            "deployment_constraints": manifest.deployment_constraints,
            "decision_policy": getattr(manifest, "decision_policy", None),
            "decision_config": getattr(manifest, "decision_config", {}),
            "artifact_sha256_digest": getattr(manifest, "artifact_sha256_digest", None),
        }

        # Stash manifest feature names/dtypes and hash (dual tracking)
        try:
            self._manifest_feature_names = list(manifest.feature_schema.keys())
            self._manifest_feature_schema_hash = manifest.feature_schema_hash
            self._manifest_feature_dtypes = [
                manifest.feature_schema[name] for name in self._manifest_feature_names
            ]
        except Exception:
            self._manifest_feature_names = []
            self._manifest_feature_schema_hash = None
            self._manifest_feature_dtypes = []

        # Use manifest features if configured
        if (
            hasattr(self._config, "use_manifest_features")
            and self._config.use_manifest_features
        ):
            self._feature_names = list(manifest.feature_schema.keys())
            self._logger.info(f"Using {len(self._feature_names)} features from manifest")

        # Check deployment constraints
        if "max_latency_ms" in manifest.deployment_constraints:
            max_latency = manifest.deployment_constraints["max_latency_ms"]
            if hasattr(self._config, "max_inference_latency_ms") and (
                self._config.max_inference_latency_ms > max_latency
            ):
                self._logger.warning(
                    f"Config latency {self._config.max_inference_latency_ms}ms exceeds model constraint {max_latency}ms",
                )

        return True
