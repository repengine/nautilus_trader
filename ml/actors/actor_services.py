"""
Actor services facade for wiring stores and registries.

Actors should depend only on Protocol-typed services, not concrete stores. This facade
centralizes initialization via the Integration Manager helpers and returns a simple
container of services for the actor to attach.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from ml.stores.protocols import FeatureStoreStrictProtocol as _FeatureStoreT
    from ml.stores.protocols import ModelStoreStrictProtocol as _ModelStoreT
    from ml.stores.protocols import StrategyStoreStrictProtocol as _StrategyStoreT
else:  # pragma: no cover - runtime-only typing shims
    from typing import Any as _FeatureStoreT  # type: ignore[no-redef]
    from typing import Any as _ModelStoreT  # type: ignore[no-redef]
    from typing import Any as _StrategyStoreT  # type: ignore[no-redef]


@dataclass(slots=True)
class ActorServices:
    feature_store: _FeatureStoreT
    model_store: _ModelStoreT
    strategy_store: _StrategyStoreT
    data_store: object
    feature_registry: object
    model_registry: object
    strategy_registry: object
    data_registry: object


def init_actor_services(config: Any) -> ActorServices:
    """
    Initialize actor services using centralized integration helpers.

    This function delegates to `ml.core.integration.init_actor_stores_and_registries`
    which implements progressive fallback, registry wiring, and shared DataRegistry
    injection. The actor attaches these to instance attributes and proceeds without
    importing any concrete stores directly.

    """
    from ml.core.integration import init_actor_stores_and_registries

    result = init_actor_stores_and_registries(config)

    # Wrap legacy stores with strict adapters for protocol‑typed actor surfaces
    from ml.stores.adapters import FeatureStoreStrictAdapter
    from ml.stores.adapters import ModelStoreStrictAdapter
    from ml.stores.adapters import StrategyStoreStrictAdapter

    wrapped_feature = FeatureStoreStrictAdapter(result.feature_store)
    wrapped_model = ModelStoreStrictAdapter(result.model_store)
    wrapped_strategy = StrategyStoreStrictAdapter(result.strategy_store)

    return ActorServices(
        feature_store=cast(_FeatureStoreT, wrapped_feature),
        model_store=cast(_ModelStoreT, wrapped_model),
        strategy_store=cast(_StrategyStoreT, wrapped_strategy),
        data_store=result.data_store,
        feature_registry=result.feature_registry,
        model_registry=result.model_registry,
        strategy_registry=result.strategy_registry,
        data_registry=result.data_registry,
    )


__all__ = ["ActorServices", "init_actor_services"]
