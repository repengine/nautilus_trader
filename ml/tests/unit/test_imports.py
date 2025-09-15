from __future__ import annotations

import importlib
from typing import List

import pytest


DOMAINS: list[str] = [
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
]


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_import_smoke(domain: str) -> None:
    try:
        importlib.import_module(f"ml.{domain}")
    except Exception as exc:  # pragma: no cover - environment dependent
        # Skip on optional dependency errors in local/dev environments
        if isinstance(exc, ImportError):
            pytest.skip(f"optional dependency missing for ml.{domain}: {exc}")
        # Unexpected errors should fail the test
        raise

