from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ml.stores.feature_store import FeatureStore
    from ml.stores.model_store import ModelStore
    from ml.stores.protocols import FeatureStoreStrictProtocol
    from ml.stores.protocols import ModelStoreStrictProtocol
    from ml.stores.protocols import StrategyStoreStrictProtocol
    from ml.stores.strategy_store import StrategyStore

    # Type-level adapter conformance check: direct assignment must type-check
    def _type_conformance(
        f: FeatureStore, m: ModelStore, s: StrategyStore
    ) -> tuple[FeatureStoreStrictProtocol, ModelStoreStrictProtocol, StrategyStoreStrictProtocol]:
        return (f, m, s)
