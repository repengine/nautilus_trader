"""
Feature flag tests for MLPipelineOrchestrator.

Phase 2.2.8: Verify feature flag behavior for legacy/facade switching.
Tests verify ML_USE_LEGACY_ORCHESTRATOR environment variable handling.

Test Design: reports/tests/phase_2_2_8_test_design_report.md

"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import pytest


@contextmanager
def env_var(name: str, value: str | None) -> Iterator[None]:
    """
    Context manager for temporarily setting environment variable.
    """
    original = os.environ.get(name)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


# ============================================================================
# FEATURE FLAG TESTS
# ============================================================================


@pytest.mark.unit
class TestFeatureFlagFunction:
    """
    Tests for use_legacy_orchestrator() function.
    """

    def test_feature_flag_legacy_mode_enabled_with_1(self) -> None:
        """
        Verify ML_USE_LEGACY_ORCHESTRATOR=1 uses legacy implementation.

        Given:
        - Environment variable set to "1"

        When:
        - Calling use_legacy_orchestrator()

        Then:
        - Returns True (use legacy)

        """
        from ml.orchestration.feature_flags import use_legacy_orchestrator

        with env_var("ML_USE_LEGACY_ORCHESTRATOR", "1"):
            assert use_legacy_orchestrator() is True

    def test_feature_flag_legacy_mode_enabled_with_true(self) -> None:
        """
        Verify ML_USE_LEGACY_ORCHESTRATOR=true uses legacy.

        Given:
        - Environment variable set to "true" (lowercase)

        When:
        - Calling use_legacy_orchestrator()

        Then:
        - Returns True

        """
        from ml.orchestration.feature_flags import use_legacy_orchestrator

        with env_var("ML_USE_LEGACY_ORCHESTRATOR", "true"):
            assert use_legacy_orchestrator() is True

    def test_feature_flag_legacy_mode_enabled_with_TRUE(self) -> None:
        """
        Verify ML_USE_LEGACY_ORCHESTRATOR=TRUE uses legacy (case insensitive).

        Given:
        - Environment variable set to "TRUE" (uppercase)

        When:
        - Calling use_legacy_orchestrator()

        Then:
        - Returns True

        """
        from ml.orchestration.feature_flags import use_legacy_orchestrator

        with env_var("ML_USE_LEGACY_ORCHESTRATOR", "TRUE"):
            assert use_legacy_orchestrator() is True

    def test_feature_flag_legacy_mode_enabled_with_yes(self) -> None:
        """
        Verify ML_USE_LEGACY_ORCHESTRATOR=yes uses legacy.

        Given:
        - Environment variable set to "yes"

        When:
        - Calling use_legacy_orchestrator()

        Then:
        - Returns True

        """
        from ml.orchestration.feature_flags import use_legacy_orchestrator

        with env_var("ML_USE_LEGACY_ORCHESTRATOR", "yes"):
            assert use_legacy_orchestrator() is True

    def test_feature_flag_facade_mode_with_0(self) -> None:
        """
        Verify ML_USE_LEGACY_ORCHESTRATOR=0 uses facade.

        Given:
        - Environment variable set to "0"

        When:
        - Calling use_legacy_orchestrator()

        Then:
        - Returns False (use facade)

        """
        from ml.orchestration.feature_flags import use_legacy_orchestrator

        with env_var("ML_USE_LEGACY_ORCHESTRATOR", "0"):
            assert use_legacy_orchestrator() is False

    def test_feature_flag_facade_mode_with_false(self) -> None:
        """
        Verify ML_USE_LEGACY_ORCHESTRATOR=false uses facade.

        Given:
        - Environment variable set to "false"

        When:
        - Calling use_legacy_orchestrator()

        Then:
        - Returns False

        """
        from ml.orchestration.feature_flags import use_legacy_orchestrator

        with env_var("ML_USE_LEGACY_ORCHESTRATOR", "false"):
            assert use_legacy_orchestrator() is False

    def test_feature_flag_default_is_facade(self) -> None:
        """
        Verify unset variable defaults to facade.

        Given:
        - Environment variable not set

        When:
        - Calling use_legacy_orchestrator()

        Then:
        - Returns False (default to new facade)

        """
        from ml.orchestration.feature_flags import use_legacy_orchestrator

        with env_var("ML_USE_LEGACY_ORCHESTRATOR", None):
            assert use_legacy_orchestrator() is False

    def test_feature_flag_invalid_value_uses_facade(self) -> None:
        """
        Verify invalid values default to facade.

        Given:
        - Environment variable set to invalid value

        When:
        - Calling use_legacy_orchestrator()

        Then:
        - Returns False (safe default)

        """
        from ml.orchestration.feature_flags import use_legacy_orchestrator

        with env_var("ML_USE_LEGACY_ORCHESTRATOR", "invalid"):
            assert use_legacy_orchestrator() is False

        with env_var("ML_USE_LEGACY_ORCHESTRATOR", "maybe"):
            assert use_legacy_orchestrator() is False

        with env_var("ML_USE_LEGACY_ORCHESTRATOR", ""):
            assert use_legacy_orchestrator() is False


@pytest.mark.unit
class TestFeatureFlagImportBehavior:
    """
    Tests for module-level import behavior based on feature flag.
    """

    def test_import_returns_legacy_when_flag_enabled(self) -> None:
        """
        Verify use_legacy_orchestrator() returns True when flag=1.

        Given:
        - ML_USE_LEGACY_ORCHESTRATOR=1

        When:
        - Checking feature flag

        Then:
        - Returns True (use legacy implementation)

        """
        from ml.orchestration.feature_flags import use_legacy_orchestrator

        with env_var("ML_USE_LEGACY_ORCHESTRATOR", "1"):
            assert use_legacy_orchestrator() is True

    def test_import_returns_facade_when_flag_disabled(self) -> None:
        """
        Verify use_legacy_orchestrator() returns False when flag=0.

        Given:
        - ML_USE_LEGACY_ORCHESTRATOR=0

        When:
        - Checking feature flag

        Then:
        - Returns False (use facade)

        """
        from ml.orchestration.feature_flags import use_legacy_orchestrator

        with env_var("ML_USE_LEGACY_ORCHESTRATOR", "0"):
            assert use_legacy_orchestrator() is False

    def test_import_returns_facade_by_default(self) -> None:
        """
        Verify use_legacy_orchestrator() returns False when flag not set.

        Given:
        - ML_USE_LEGACY_ORCHESTRATOR not set

        When:
        - Checking feature flag

        Then:
        - Returns False (facade is default)

        """
        from ml.orchestration.feature_flags import use_legacy_orchestrator

        with env_var("ML_USE_LEGACY_ORCHESTRATOR", None):
            assert use_legacy_orchestrator() is False
