"""
Repository-level pytest configuration helpers.

This stub complements ml/pytest.ini when tests are invoked from the repo root by
registering common markers and relaxing noisy warnings. It avoids changing test
semantics and keeps behavior aligned with ml/pytest.ini.

"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path


pytest_plugins = ("ml.tests.fixtures.pytest_plugins",)

_REPO_ROOT = Path(__file__).resolve().parent
_LEGACY_TESTS_DIR = _REPO_ROOT / "tests"
_THIRD_PARTY_TESTS_DIR = _REPO_ROOT / ".libraries"


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    # Ensure Pandera import warning is disabled consistently
    os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

    # Register common markers to avoid PytestUnknownMarkWarning when running
    # from the repo root without loading ml/pytest.ini.
    markers: Sequence[str] = (
        # Core taxonomy
        "unit: Unit tests",
        "integration: Integration tests",
        "e2e: End-to-end tests",
        "system: System tests",
        "property: Property-based tests",
        "contract: Contract tests",
        "contracts: Contract tests (alias)",
        "metamorphic: Metamorphic relationship tests",
        "combinatorial: Pairwise/Combinatorial tests",
        "stateful: Stateful property-based tests",
        # Perf/control
        "benchmark: Performance benchmarks",
        "slow: Slow running tests",
        "prototype: Prototypes excluded by default",
        # Parallelization and deps
        "serial: Must run serially",
        "parallel_safe: Safe for parallel execution",
        "database: Requires PostgreSQL",
        "redis: Requires Redis",
        "docker: Requires Docker",
        # Misc
        "flaky: Known flaky test",
    )
    for m in markers:
        config.addinivalue_line("markers", m)

    # Reduce noisy warnings in local runs; CI can tighten these.
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning")
    config.addinivalue_line("filterwarnings", "ignore::PendingDeprecationWarning")
    config.addinivalue_line("filterwarnings", "ignore::ResourceWarning")
    # Pandera future import warning
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*pandas-specific classes and functions from the top-level pandera.*:FutureWarning",
    )
    # Silence third-party PyFilesystem2/pkg_resources deprecation noise
    config.addinivalue_line(
        "filterwarnings",
        "ignore:pkg_resources is deprecated as an API.*:UserWarning",
    )
    # Pytest fixtures usage warnings in example tests
    config.addinivalue_line("filterwarnings", "ignore::pytest.PytestReturnNotNoneWarning")
    # Pytest-benchmark plug-in noisy notices
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*Benchmark fixture was not used.*:pytest.PytestWarning",
    )


def pytest_ignore_collect(path, config) -> bool:  # type: ignore[no-untyped-def]
    """
    Skip legacy non-ML test directories when running focused ML shards.
    """
    keyword_expr = (getattr(config.option, "keyword", "") or "").lower()
    if "earnings" not in keyword_expr:
        return False

    path_str = os.fspath(path)
    return path_str.startswith(str(_LEGACY_TESTS_DIR)) or path_str.startswith(
        str(_THIRD_PARTY_TESTS_DIR),
    )
