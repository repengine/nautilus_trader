from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ml.core.integration import init_ml_stores_and_registries


class _LabelsCapture:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def labels(self, **labels: str) -> _LabelsCapture:  # noqa: D401
        self.calls.append(labels)
        return self

    def inc(self, *_args: Any, **_kwargs: Any) -> None:  # noqa: D401
        return None


class _CounterCapture:
    def __init__(self) -> None:
        self.labels_obj = _LabelsCapture()

    def labels(self, **labels: str) -> _LabelsCapture:
        return self.labels_obj.labels(**labels)


@dataclass(slots=True)
class _Cfg:
    db_connection: str | None = "postgresql://invalid:invalid@localhost:5432/nautilus"
    allow_dummy_fallback: bool = True
    use_dummy_stores: bool = False


@pytest.mark.contracts
def test_fallback_activation_emits_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force EngineManager.get_engine to raise and trigger fallback
    monkeypatch.setattr(
        "ml.core.db_engine.EngineManager.get_engine",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(Exception("DB unreachable")),
    )

    # Capture metrics emitted via metrics_bootstrap.get_counter
    counter = _CounterCapture()
    monkeypatch.setattr(
        "ml.common.metrics_bootstrap.get_counter",
        lambda *_args, **_kwargs: counter,
    )

    _ = init_ml_stores_and_registries(_Cfg())

    # One or more fallback metric emissions should have occurred
    assert counter.labels_obj.calls, "Expected fallback activation metric to be emitted"
    # Check labels contain expected fields
    lab = counter.labels_obj.calls[-1]
    assert lab.get("component") == "actor_stores"
    assert lab.get("level") in {"dummy", "file"}
