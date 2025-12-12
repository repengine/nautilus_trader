"""
Actor services facade for wiring stores and registries.

Actors should depend only on Protocol-typed services, not concrete stores. This facade
centralizes initialization via the Integration Manager helpers and returns a simple
container of services for the actor to attach.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ml.stores.base import DummyStore
    from ml.stores.file_backed import FileDataStore
    from ml.stores.protocols import DataStoreFacadeProtocol
    from ml.stores.protocols import FeatureStoreStrictProtocol as _FeatureStoreT
    from ml.stores.protocols import ModelStoreStrictProtocol as _ModelStoreT
    from ml.stores.protocols import StrategyStoreStrictProtocol as _StrategyStoreT
    _DataStoreFacadeT = DataStoreFacadeProtocol | FileDataStore | DummyStore
else:  # pragma: no cover - runtime-only typing shims
    from typing import Any as _DataStoreFacadeT  # type: ignore[no-redef]
    from typing import Any as _FeatureStoreT  # type: ignore[no-redef]
    from typing import Any as _ModelStoreT  # type: ignore[no-redef]
    from typing import Any as _StrategyStoreT  # type: ignore[no-redef]


@dataclass(slots=True)
class ActorServices:
    feature_store: _FeatureStoreT
    model_store: _ModelStoreT
    strategy_store: _StrategyStoreT
    data_store: _DataStoreFacadeT
    feature_registry: object
    model_registry: object
    strategy_registry: object
    data_registry: object


def init_actor_services(config: Any) -> ActorServices:
    """
    Initialize actor services using centralized integration helpers.

    This function delegates to `ml.core.integration.init_ml_stores_and_registries`
    which implements progressive fallback, registry wiring, and shared DataRegistry
    injection. The actor attaches these to instance attributes and proceeds without
    importing any concrete stores directly.

    """
    from ml.core.integration import init_ml_stores_and_registries

    result = init_ml_stores_and_registries(config)

    # Prefer direct attachment if stores already conform to strict protocols; otherwise use adapters
    from ml.stores.adapters import FeatureStoreStrictAdapter
    from ml.stores.adapters import ModelStoreStrictAdapter
    from ml.stores.adapters import StrategyStoreStrictAdapter
    from ml.stores.protocols import FeatureStoreStrictProtocol as _FSP
    from ml.stores.protocols import ModelStoreStrictProtocol as _MSP
    from ml.stores.protocols import StrategyStoreStrictProtocol as _SSP

    feature_store = (
        result.feature_store
        if isinstance(result.feature_store, _FSP)
        else FeatureStoreStrictAdapter(result.feature_store)
    )
    model_store = (
        result.model_store
        if isinstance(result.model_store, _MSP)
        else ModelStoreStrictAdapter(result.model_store)
    )
    strategy_store = (
        result.strategy_store
        if isinstance(result.strategy_store, _SSP)
        else StrategyStoreStrictAdapter(result.strategy_store)
    )

    return ActorServices(
        feature_store=feature_store,
        model_store=model_store,
        strategy_store=strategy_store,
        data_store=result.data_store,
        feature_registry=result.feature_registry,
        model_registry=result.model_registry,
        strategy_registry=result.strategy_registry,
        data_registry=result.data_registry,
    )


__all__ = ["ActorServices", "init_actor_services"]
