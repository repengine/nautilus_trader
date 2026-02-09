#!/usr/bin/env python3
"""
Registry policy environment fixtures for strict-policy test alignment.

These fixtures isolate registry policy environment variables so test suites can
declare strict or permissive behavior explicitly at call sites.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

REGISTRY_POLICY_ENV_KEYS: tuple[str, ...] = (
    "ML_STRICT_MODEL_COMPATIBILITY",
    "ML_STRICT_FEATURE_PARITY",
    "ML_ALLOW_COMPATIBILITY_MIGRATION_OVERRIDE",
    "ML_ALLOW_UNSIGNED_ARTIFACTS",
    "ML_REQUIRE_OUTPUT_SEMANTICS",
)

STRICT_REGISTRY_POLICY_ENV: Mapping[str, str] = {
    "ML_STRICT_MODEL_COMPATIBILITY": "true",
    "ML_STRICT_FEATURE_PARITY": "true",
    "ML_ALLOW_COMPATIBILITY_MIGRATION_OVERRIDE": "false",
    "ML_ALLOW_UNSIGNED_ARTIFACTS": "false",
    "ML_REQUIRE_OUTPUT_SEMANTICS": "true",
}

PERMISSIVE_REGISTRY_POLICY_ENV: Mapping[str, str] = {
    "ML_STRICT_MODEL_COMPATIBILITY": "false",
    "ML_STRICT_FEATURE_PARITY": "false",
    "ML_ALLOW_COMPATIBILITY_MIGRATION_OVERRIDE": "true",
    "ML_ALLOW_UNSIGNED_ARTIFACTS": "true",
    "ML_REQUIRE_OUTPUT_SEMANTICS": "false",
}


def _clear_registry_policy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove registry policy environment keys to avoid cross-test leakage."""
    for key in REGISTRY_POLICY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _apply_registry_policy_env(
    monkeypatch: pytest.MonkeyPatch,
    values: Mapping[str, str],
) -> None:
    """Apply registry policy environment values for scoped test behavior."""
    for key, value in values.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def isolated_registry_policy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Clear all registry policy env keys for deterministic policy behavior.

    This fixture should be used by default in registry tests that depend on
    policy semantics so ambient shell state cannot change test intent.
    """
    _clear_registry_policy_env(monkeypatch)


@pytest.fixture
def strict_registry_policy_env(
    monkeypatch: pytest.MonkeyPatch,
    isolated_registry_policy_env: None,
) -> None:
    """
    Set explicit strict registry policy environment defaults for a test.

    Use this fixture when asserting strict compatibility, digest, feature-parity,
    or output-semantics behavior.
    """
    del isolated_registry_policy_env
    _apply_registry_policy_env(monkeypatch, STRICT_REGISTRY_POLICY_ENV)


@pytest.fixture
def permissive_registry_policy_env(
    monkeypatch: pytest.MonkeyPatch,
    isolated_registry_policy_env: None,
) -> None:
    """
    Set explicit permissive registry policy values for legacy migration tests.

    This fixture is intentionally non-autouse and must be requested directly by
    tests that validate permissive compatibility behavior.
    """
    del isolated_registry_policy_env
    _apply_registry_policy_env(monkeypatch, PERMISSIVE_REGISTRY_POLICY_ENV)


__all__ = [
    "PERMISSIVE_REGISTRY_POLICY_ENV",
    "REGISTRY_POLICY_ENV_KEYS",
    "STRICT_REGISTRY_POLICY_ENV",
    "isolated_registry_policy_env",
    "permissive_registry_policy_env",
    "strict_registry_policy_env",
]
