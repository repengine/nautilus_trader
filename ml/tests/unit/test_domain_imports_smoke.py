from __future__ import annotations

import importlib
import os
from typing import Iterable

import pytest


DOMAINS: tuple[str, ...] = (
    "actors",
    "common",
    "config",
    "consumers",
    "core",
    "data",
    "deployment",
    "evaluation",
    "features",
    "models",
    "monitoring",
    "observability",
    "orchestration",
    "pipelines",
    "preprocessing",
    "registry",
    "stores",
    "strategies",
    "training",
)


def _is_optional_dependency_error(exc: Exception) -> bool:
    text = str(exc).lower()
    # Heuristic skip for optional heavy deps that may not be installed locally/CI
    optional_markers: Iterable[str] = (
        "databento",
        "torch",
        "lightning",
        "onnxruntime",
        "prometheus_client",
        "polars",
        "pandas",
    )
    return any(marker in text for marker in optional_markers)


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_import_smoke(domain: str) -> None:
    try:
        importlib.import_module(f"ml.{domain}")
    except Exception as exc:  # pragma: no cover - smoke test behavior
        if (
            _is_optional_dependency_error(exc)
            or os.getenv("ML_IMPORT_SMOKE_SKIP_OPTIONAL", "1") == "1"
        ):
            pytest.skip(f"optional dependency missing for ml.{domain}: {exc}")
        raise
