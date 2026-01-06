from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from ml.common.message_bus import MessagePublisherProtocol
from ml.core.integration import MLIntegrationManager
from ml.tests.utils.stubs import build_integration_manager_stub
from ml.registry.data_registry import DataRegistry
from ml.registry.persistence import BackendType
from ml.registry.persistence import PersistenceConfig
from ml.stores.data_store_facade import DataStore


class CapturePublisher(MessagePublisherProtocol):
    def __init__(self) -> None:
        self.count = 0

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        self.count += 1
        return True


class TestIntegrationPublisher:
    def test_set_message_publisher_applies_to_data_store(self, tmp_path: Path) -> None:
        # Arrange a lightweight manager with a real DataStore (sqlite + JSON registry)
        mgr = build_integration_manager_stub()
        reg_dir = tmp_path / "reg"
        registry = DataRegistry(
            registry_path=reg_dir,
            persistence_config=PersistenceConfig(backend=BackendType.JSON, json_path=reg_dir),
        )
        store = DataStore(
            connection_string="sqlite:///:memory:",
            registry=registry,
            feature_store=cast(Any, object()),
            model_store=cast(Any, object()),
            strategy_store=cast(Any, object()),
        )
        mgr.data_store = store

        pub = CapturePublisher()

        # Act
        mgr.set_message_publisher(pub)

        # Assert
        assert getattr(store, "publisher") is pub
