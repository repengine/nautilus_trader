from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ml.dashboard.config import DashboardConfig
from ml.dashboard.controllers import NoopServiceController
from ml.dashboard.service import (
    DashboardService,
    _REGISTRY_CACHE_HITS,
    _REGISTRY_CACHE_MISSES,
    _REGISTRY_FALLBACK_TOTAL,
)
from ml.registry.base import DummyRegistry


def _metric_value(counter: Any, **labels: str) -> float:
    """Return the current value for a labelled Prometheus counter."""

    return counter.labels(**labels)._value.get()


def _reset_metric(counter: Any, **labels: str) -> None:
    counter.labels(**labels)._value.set(0.0)


def _make_model_stub(model_id: str) -> SimpleNamespace:
    manifest = SimpleNamespace(
        model_id=model_id,
        role=SimpleNamespace(value="INFERENCE"),
        version="1",
        architecture="test",
        feature_schema_hash="abc123",
    )
    return SimpleNamespace(
        manifest=manifest,
        deployment_status=SimpleNamespace(value="ready"),
        deployed_to=["ml_signal_actor"],
    )


class StubModelRegistry:
    def __init__(self, model_ids: list[str]) -> None:
        self.calls = 0
        self._model_ids = model_ids

    def get_all_models(self) -> list[SimpleNamespace]:
        self.calls += 1
        return [_make_model_stub(mid) for mid in self._model_ids]

    def get_active_models(self) -> list[SimpleNamespace]:
        return []

    def deploy_model(self, model_id: str, target: str, config: dict[str, Any] | None = None) -> bool:
        self._model_ids = [model_id]
        return True

    def hot_reload_model(self, target: str, new_model_id: str) -> bool:
        self._model_ids = [new_model_id]
        return True

    def rollback(self, target: str, to_model_id: str) -> bool:
        self._model_ids = [to_model_id]
        return True


class FailingModelRegistry:
    def get_all_models(self) -> list[SimpleNamespace]:
        raise RuntimeError("registry unavailable")

    def get_active_models(self) -> list[SimpleNamespace]:
        raise RuntimeError("registry unavailable")

    def deploy_model(self, model_id: str, target: str, config: dict[str, Any] | None = None) -> bool:
        return False

    def hot_reload_model(self, target: str, new_model_id: str) -> bool:
        return False

    def rollback(self, target: str, to_model_id: str) -> bool:
        return False


def _make_service() -> DashboardService:
    cfg = DashboardConfig()
    svc = DashboardService(config=cfg, controller=NoopServiceController())
    svc._registry_cache.clear()
    return svc


def test_list_models_uses_cache() -> None:
    svc = _make_service()
    stub = StubModelRegistry(["m1"])
    svc._model_registry = stub
    key = svc._cache_key("models")
    _reset_metric(_REGISTRY_CACHE_HITS, entry=key)
    _reset_metric(_REGISTRY_CACHE_MISSES, entry=key)

    first = svc.list_models()
    assert [row["model_id"] for row in first] == ["m1"]
    assert stub.calls == 1
    assert _metric_value(_REGISTRY_CACHE_MISSES, entry=key) == 1.0
    assert _metric_value(_REGISTRY_CACHE_HITS, entry=key) == 0.0

    second = svc.list_models()
    assert second == first
    assert stub.calls == 1  # Served from cache
    assert _metric_value(_REGISTRY_CACHE_HITS, entry=key) == 1.0


def test_list_models_records_fallback_on_error() -> None:
    svc = _make_service()
    svc._model_registry = FailingModelRegistry()
    key = svc._cache_key("models")
    fallback_labels = {"registry": "model", "reason": "list_failed"}
    _reset_metric(_REGISTRY_CACHE_MISSES, entry=key)
    _reset_metric(_REGISTRY_CACHE_HITS, entry=key)
    _reset_metric(_REGISTRY_FALLBACK_TOTAL, **fallback_labels)

    result = svc.list_models()
    assert result == []
    assert _metric_value(_REGISTRY_CACHE_MISSES, entry=key) == 1.0
    assert _metric_value(_REGISTRY_CACHE_HITS, entry=key) == 0.0
    assert _metric_value(_REGISTRY_FALLBACK_TOTAL, **fallback_labels) == 1.0


def test_deploy_model_invalidates_cache() -> None:
    svc = _make_service()
    stub = StubModelRegistry(["initial"])
    svc._model_registry = stub
    key = svc._cache_key("models")
    _reset_metric(_REGISTRY_CACHE_MISSES, entry=key)
    _reset_metric(_REGISTRY_CACHE_HITS, entry=key)

    original = svc.list_models()
    assert stub.calls == 1
    assert [row["model_id"] for row in original] == ["initial"]

    deploy_result = svc.deploy_model("updated", "ml_signal_actor")
    assert deploy_result["ok"] is True

    refreshed = svc.list_models()
    assert stub.calls == 2  # Cache invalidated after deploy
    assert [row["model_id"] for row in refreshed] == ["updated"]
    assert _metric_value(_REGISTRY_CACHE_MISSES, entry=key) == 2.0


def test_model_registry_dummy_fallback(monkeypatch: Any) -> None:
    monkeypatch.setenv("ML_ALLOW_DUMMY", "1")

    def _raise(_: DashboardService) -> None:
        raise RuntimeError("init failure")

    monkeypatch.setattr(DashboardService, "_build_model_registry", _raise, raising=True)

    svc = _make_service()
    svc._model_registry = None
    key = svc._cache_key("models")
    fallback_labels = {"registry": "model", "reason": "dummy_registry"}
    _reset_metric(_REGISTRY_CACHE_MISSES, entry=key)
    _reset_metric(_REGISTRY_CACHE_HITS, entry=key)
    _reset_metric(_REGISTRY_FALLBACK_TOTAL, **fallback_labels)

    models = svc.list_models()
    assert models == []
    assert isinstance(svc._model_registry, DummyRegistry)
    assert _metric_value(_REGISTRY_FALLBACK_TOTAL, **fallback_labels) == 1.0
