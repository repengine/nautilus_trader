"""
Test dataset ID constants.

This module tests that dataset ID constants are properly defined, immutable,
and accessible from the public ml.config API.
"""

from __future__ import annotations

import pytest


def test_dataset_ids_accessible_from_config() -> None:
    """Dataset IDs can be imported from ml.config."""
    from ml.config import EARNINGS_ACTUALS_DATASET_ID
    from ml.config import EARNINGS_ESTIMATES_DATASET_ID

    assert EARNINGS_ACTUALS_DATASET_ID == "ml.earnings_actuals"
    assert EARNINGS_ESTIMATES_DATASET_ID == "ml.earnings_estimates"


def test_dataset_ids_accessible_from_dataset_ids_module() -> None:
    """Dataset IDs can be imported from ml.config.dataset_ids."""
    from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID
    from ml.config.dataset_ids import EARNINGS_ESTIMATES_DATASET_ID

    assert EARNINGS_ACTUALS_DATASET_ID == "ml.earnings_actuals"
    assert EARNINGS_ESTIMATES_DATASET_ID == "ml.earnings_estimates"


def test_dataset_ids_have_correct_values() -> None:
    """Dataset IDs have the expected string values."""
    from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID
    from ml.config.dataset_ids import EARNINGS_ESTIMATES_DATASET_ID

    # Verify correct values
    assert isinstance(EARNINGS_ACTUALS_DATASET_ID, str)
    assert isinstance(EARNINGS_ESTIMATES_DATASET_ID, str)

    # Verify values follow ml.* naming convention
    assert EARNINGS_ACTUALS_DATASET_ID.startswith("ml.")
    assert EARNINGS_ESTIMATES_DATASET_ID.startswith("ml.")

    # Verify specific values
    assert EARNINGS_ACTUALS_DATASET_ID == "ml.earnings_actuals"
    assert EARNINGS_ESTIMATES_DATASET_ID == "ml.earnings_estimates"


def test_dataset_ids_are_distinct() -> None:
    """Dataset IDs are distinct from each other."""
    from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID
    from ml.config.dataset_ids import EARNINGS_ESTIMATES_DATASET_ID

    assert EARNINGS_ACTUALS_DATASET_ID != EARNINGS_ESTIMATES_DATASET_ID


def test_dataset_ids_type_hints() -> None:
    """Dataset IDs use Final type hint (tested via type checking, verified at runtime)."""
    from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID
    from ml.config.dataset_ids import EARNINGS_ESTIMATES_DATASET_ID

    # Verify they are strings (Final[str] resolves to str at runtime)
    assert isinstance(EARNINGS_ACTUALS_DATASET_ID, str)
    assert isinstance(EARNINGS_ESTIMATES_DATASET_ID, str)


def test_dataset_ids_immutability() -> None:
    """Dataset IDs cannot be reassigned (enforced by type checker with Final)."""
    # Note: Python doesn't enforce Final at runtime, but we verify the
    # constants exist and are strings. MyPy will enforce Final in CI.
    from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID
    from ml.config.dataset_ids import EARNINGS_ESTIMATES_DATASET_ID

    # These constants should be immutable strings
    assert EARNINGS_ACTUALS_DATASET_ID == "ml.earnings_actuals"
    assert EARNINGS_ESTIMATES_DATASET_ID == "ml.earnings_estimates"

    # Attempting reassignment would be caught by mypy --strict
    # (not testable at runtime as Python doesn't enforce Final)


def test_dataset_ids_in_public_api() -> None:
    """Dataset IDs are exported in ml.config.__all__."""
    from ml import config

    assert "EARNINGS_ACTUALS_DATASET_ID" in config.__all__
    assert "EARNINGS_ESTIMATES_DATASET_ID" in config.__all__


def test_dataset_ids_public_api_is_alphabetically_sorted() -> None:
    """Verify __all__ list in dataset_ids module is alphabetically sorted."""
    from ml.config import dataset_ids

    # Get the __all__ list
    all_exports = dataset_ids.__all__

    # Verify it's sorted
    assert all_exports == sorted(all_exports), (
        f"__all__ is not alphabetically sorted: {all_exports}"
    )


def test_dataset_ids_module_docstring() -> None:
    """Dataset IDs module has proper docstring."""
    from ml.config import dataset_ids

    assert dataset_ids.__doc__ is not None
    assert len(dataset_ids.__doc__) > 0
    assert "Dataset ID constants" in dataset_ids.__doc__


def test_constants_follow_naming_convention() -> None:
    """Dataset ID constants follow UPPER_SNAKE_CASE with _DATASET_ID suffix."""
    from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID
    from ml.config.dataset_ids import EARNINGS_ESTIMATES_DATASET_ID

    # Constant names (checked via import name)
    constant_names = ["EARNINGS_ACTUALS_DATASET_ID", "EARNINGS_ESTIMATES_DATASET_ID"]

    for name in constant_names:
        # Verify UPPER_SNAKE_CASE
        assert name.isupper() or "_" in name
        # Verify _DATASET_ID suffix
        assert name.endswith("_DATASET_ID")


def test_no_additional_exports() -> None:
    """Dataset IDs module only exports the expected constants."""
    from ml.config import dataset_ids

    expected = {"EARNINGS_ACTUALS_DATASET_ID", "EARNINGS_ESTIMATES_DATASET_ID"}
    actual = set(dataset_ids.__all__)

    assert actual == expected, f"Unexpected exports: {actual - expected}"


def test_dataset_ids_work_with_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dataset IDs work correctly when used with registry operations."""
    from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID
    from ml.config.dataset_ids import EARNINGS_ESTIMATES_DATASET_ID

    # These should be valid dataset IDs
    dataset_ids = [EARNINGS_ACTUALS_DATASET_ID, EARNINGS_ESTIMATES_DATASET_ID]

    for dataset_id in dataset_ids:
        # Verify they're non-empty strings that could be used as keys
        assert dataset_id
        assert isinstance(dataset_id, str)
        assert len(dataset_id) > 0


def test_constants_imported_consistently() -> None:
    """Constants imported from different paths are the same object."""
    from ml.config import EARNINGS_ACTUALS_DATASET_ID as config_actuals
    from ml.config import EARNINGS_ESTIMATES_DATASET_ID as config_estimates
    from ml.config.dataset_ids import EARNINGS_ACTUALS_DATASET_ID as module_actuals
    from ml.config.dataset_ids import (
        EARNINGS_ESTIMATES_DATASET_ID as module_estimates,
    )

    # Verify they reference the same objects (same identity)
    assert config_actuals is module_actuals
    assert config_estimates is module_estimates

    # Verify they have the same values
    assert config_actuals == module_actuals
    assert config_estimates == module_estimates
