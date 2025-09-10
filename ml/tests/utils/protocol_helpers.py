from __future__ import annotations

from typing import Any

from ml.common.protocols import MLComponentProtocol


def assert_implements_ml_component(obj: object) -> None:
    """
    Assert that an object implements MLComponentProtocol and its methods work.
    """
    assert isinstance(
        obj,
        MLComponentProtocol,
    ), f"{type(obj)} does not implement MLComponentProtocol"

    health = obj.get_health_status()  # type: ignore[attr-defined]
    assert isinstance(health, dict)

    metrics = obj.get_performance_metrics()  # type: ignore[attr-defined]
    assert isinstance(metrics, dict)
    for k, v in metrics.items():
        assert isinstance(k, str)
        assert isinstance(v, float)

    issues = obj.validate_configuration()  # type: ignore[attr-defined]
    assert isinstance(issues, list)
    for item in issues:
        assert isinstance(item, str)
