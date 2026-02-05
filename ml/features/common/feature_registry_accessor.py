"""
FeatureRegistryAccessor component - encapsulates registry access logic.

Extracted from FeatureEngineer god class (Phase 2.1.2).
Provides safe, defensive access to 4 registry instances (FeatureRegistry, ModelRegistry,
StrategyRegistry, DataRegistry) from an injected stores container.
"""

from __future__ import annotations

import logging
from typing import cast


logger = logging.getLogger(__name__)


class FeatureRegistryAccessor:
    """
    Encapsulates registry access operations.

    Provides a clean interface for accessing the 4 registries (feature, model, strategy, data)
    from the stores container, with graceful degradation when the container or specific
    registries are unavailable.

    Parameters
    ----------
    stores : object | None
        The stores container instance (typically ActorStoresRegistries returned by
        init_ml_stores_and_registries()). If None, all registry properties will return None.

    Examples
    --------
    >>> from ml.common.actor_initialization import init_ml_stores_and_registries
    >>> from ml.config.core import DatabaseConfig
    >>> db_config = DatabaseConfig.from_env()
    >>> stores = init_ml_stores_and_registries(db_config)
    >>> accessor = FeatureRegistryAccessor(stores=stores)
    >>>
    >>> # Access registries
    >>> if accessor.feature_registry is not None:
    ...     feature_manifest = accessor.feature_registry.get("price_sma_20")
    >>>
    >>> if accessor.model_registry is not None:
    ...     model_manifest = accessor.model_registry.get("xgb_classifier_v1")
    >>>
    >>> # Graceful degradation when stores unavailable
    >>> accessor_no_stores = FeatureRegistryAccessor(stores=None)
    >>> assert accessor_no_stores.feature_registry is None
    >>> assert accessor_no_stores.model_registry is None

    """

    def __init__(
        self,
        stores: object | None = None,
    ) -> None:
        """
        Initialize the FeatureRegistryAccessor.

        Parameters
        ----------
        stores : object | None
            The stores container instance. If None, all registry properties will return None.

        """
        self._stores = stores

    @property
    def feature_registry(self) -> object | None:
        """
        Access the feature registry from the injected stores container.

        Returns None if stores were not injected or if the container doesn't have a
        feature_registry attribute.

        Returns
        -------
        object | None
            The FeatureRegistry instance if available, otherwise None.

        Examples
        --------
        >>> accessor = FeatureRegistryAccessor(stores=stores)
        >>> registry = accessor.feature_registry
        >>> if registry is not None:
        ...     manifest = registry.get("price_sma_20")

        """
        if self._stores is not None and hasattr(self._stores, "feature_registry"):
            return cast(object, self._stores.feature_registry)
        return None

    @property
    def model_registry(self) -> object | None:
        """
        Access the model registry from the injected stores container.

        Returns None if stores were not injected or if the container doesn't have a
        model_registry attribute.

        Returns
        -------
        object | None
            The ModelRegistry instance if available, otherwise None.

        Examples
        --------
        >>> accessor = FeatureRegistryAccessor(stores=stores)
        >>> registry = accessor.model_registry
        >>> if registry is not None:
        ...     manifest = registry.get("xgb_classifier_v1")

        """
        if self._stores is not None and hasattr(self._stores, "model_registry"):
            return cast(object, self._stores.model_registry)
        return None

    @property
    def strategy_registry(self) -> object | None:
        """
        Access the strategy registry from the injected stores container.

        Returns None if stores were not injected or if the container doesn't have a
        strategy_registry attribute.

        Returns
        -------
        object | None
            The StrategyRegistry instance if available, otherwise None.

        Examples
        --------
        >>> accessor = FeatureRegistryAccessor(stores=stores)
        >>> registry = accessor.strategy_registry
        >>> if registry is not None:
        ...     manifest = registry.get("momentum_strategy_v1")

        """
        if self._stores is not None and hasattr(self._stores, "strategy_registry"):
            return cast(object, self._stores.strategy_registry)
        return None

    @property
    def data_registry(self) -> object | None:
        """
        Access the data registry from the injected stores container.

        Returns None if stores were not injected or if the container doesn't have a
        data_registry attribute.

        Returns
        -------
        object | None
            The DataRegistry instance if available, otherwise None.

        Examples
        --------
        >>> accessor = FeatureRegistryAccessor(stores=stores)
        >>> registry = accessor.data_registry
        >>> if registry is not None:
        ...     manifest = registry.get("historical_bars_2024")

        """
        if self._stores is not None and hasattr(self._stores, "data_registry"):
            return cast(object, self._stores.data_registry)
        return None
