"""RuntimeAttacher component tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ml.orchestration.config_types import IntegrationConfig
from ml.orchestration.runtime_attacher import RuntimeAttacher


@pytest.fixture
def manager() -> Mock:
    """Provide a mock integration manager with store/registry attributes."""
    mgr = Mock()
    mgr.data_registry = Mock(name="data_registry")
    mgr.feature_registry = Mock(name="feature_registry")
    mgr.model_registry = Mock(name="model_registry")
    mgr.strategy_registry = Mock(name="strategy_registry")
    mgr.data_store = Mock(name="data_store")
    mgr.feature_store = Mock(name="feature_store")
    mgr.model_store = Mock(name="model_store")
    mgr.strategy_store = Mock(name="strategy_store")
    mgr.partition_manager = Mock(name="partition_manager")
    return mgr


def test_attach_runtime_returns_none_when_disabled(tmp_path: Path) -> None:
    """Attachment should short-circuit when integration disabled."""
    attacher = RuntimeAttacher()
    cfg = IntegrationConfig(enabled=False)

    assert attacher.attach_runtime(cfg, dataset_out_dir=tmp_path) is None


def test_attach_runtime_uses_factory(monkeypatch: pytest.MonkeyPatch, manager: Mock, tmp_path: Path) -> None:
    """attach_runtime should build manager from factory and expose components."""
    factory = Mock(return_value=manager)
    attacher = RuntimeAttacher(integration_manager_factory=factory)
    cfg = IntegrationConfig(enabled=True, run_validators=False)

    result = attacher.attach_runtime(cfg, dataset_out_dir=tmp_path)

    assert result is manager
    factory.assert_called_once()
    assert attacher.data_store is manager.data_store
    assert attacher.feature_registry is manager.feature_registry
    assert attacher.partition_manager is manager.partition_manager


def test_run_validators_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_validators should raise when any validator fails."""
    metrics_mod = SimpleNamespace(main=lambda: 1)
    events_mod = SimpleNamespace(main=lambda: 0)
    monkeypatch.setitem(sys.modules, "tools.validate_metrics_bootstrap", metrics_mod)
    monkeypatch.setitem(sys.modules, "tools.validate_event_constants", events_mod)

    attacher = RuntimeAttacher()

    with pytest.raises(RuntimeError):
        attacher.run_validators()
