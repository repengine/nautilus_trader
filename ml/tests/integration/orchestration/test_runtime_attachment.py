"""Integration-like tests for RuntimeAttacher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from ml.orchestration.config_types import IntegrationConfig
from ml.orchestration.runtime_attacher import RuntimeAttacher


@pytest.fixture
def integration_manager() -> Mock:
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


@pytest.mark.integration
def test_attach_runtime_uses_existing_manager(
    integration_manager: Mock,
    tmp_path: Path,
) -> None:
    """attach_runtime should reuse provided integration manager."""
    attacher = RuntimeAttacher(integration_manager=integration_manager)
    cfg = IntegrationConfig(enabled=True, run_validators=False)

    result = attacher.attach_runtime(cfg, dataset_out_dir=tmp_path)

    assert result is integration_manager
    assert attacher.data_store is integration_manager.data_store
    assert attacher.model_registry is integration_manager.model_registry


@pytest.mark.integration
def test_attach_runtime_skips_when_disabled(
    integration_manager: Mock,
    tmp_path: Path,
) -> None:
    """Disabled integration config should short-circuit attachment."""
    attacher = RuntimeAttacher(integration_manager=integration_manager)
    cfg = IntegrationConfig(enabled=False)

    assert attacher.attach_runtime(cfg, dataset_out_dir=tmp_path) is None
