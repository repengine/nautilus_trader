from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, cast

import pytest

from ml.tests.utils.db import build_postgres_url


def _load_init_ml_stores_and_registries(
    isolated_prometheus_registry: object,
) -> Callable[[Any], object]:
    """Reload integration + metrics modules and return fresh init function."""
    reload_modules = getattr(isolated_prometheus_registry, "reload_modules", None)
    if callable(reload_modules):
        reload_modules(
            (
                "ml.common.metrics_bootstrap",
                "ml.core.integration_facade",
                "ml.core.integration",
            )
        )

    from ml.core.integration import init_ml_stores_and_registries

    return init_ml_stores_and_registries


@dataclass(slots=True)
class _Cfg:
    db_connection: str | None = build_postgres_url(
        user="invalid",
        password="invalid",
        database="nautilus",
    )
    allow_dummy_fallback: bool = True
    use_dummy_stores: bool = False


@pytest.mark.contracts
def test_fallback_activation_emits_metric(
    isolated_prometheus_registry: object,
) -> None:
    init_ml_stores_and_registries = _load_init_ml_stores_and_registries(
        isolated_prometheus_registry,
    )
    _ = init_ml_stores_and_registries(_Cfg())
    registry = cast(Any, getattr(isolated_prometheus_registry, "registry"))
    dummy_value = registry.get_sample_value(
        "ml_fallback_activations_total",
        labels={"component": "actor_stores", "level": "dummy"},
    )
    file_value = registry.get_sample_value(
        "ml_fallback_activations_total",
        labels={"component": "actor_stores", "level": "file"},
    )
    assert (dummy_value or 0.0) > 0.0 or (file_value or 0.0) > 0.0
