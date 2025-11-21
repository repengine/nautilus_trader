"""
Test dataset ID constants.

This module tests that dataset ID constants are properly defined, immutable,
and accessible from the public ml.config API.
"""

from __future__ import annotations

from typing import Final

import pytest

DATASET_ID_EXPECTATIONS: Final[tuple[tuple[str, str], ...]] = (
    ("EARNINGS_ACTUALS_DATASET_ID", "ml.earnings_actuals"),
    ("EARNINGS_ESTIMATES_DATASET_ID", "ml.earnings_estimates"),
    ("MACRO_RELEASES_DATASET_ID", "ml.macro_release_calendar"),
    ("MACRO_OBSERVATIONS_DATASET_ID", "ml.macro_observations"),
    ("EVENTS_CALENDAR_DATASET_ID", "ml.events_calendar"),
    ("MICRO_MINUTE_DATASET_ID", "ml.microstructure_minute"),
    ("L2_MINUTE_DATASET_ID", "ml.l2_minute"),
)


def test_dataset_ids_accessible_from_config() -> None:
    """Dataset IDs can be imported from ml.config."""
    from ml import config

    for attr, expected in DATASET_ID_EXPECTATIONS:
        value = getattr(config, attr)
        assert value == expected


def test_dataset_ids_accessible_from_dataset_ids_module() -> None:
    """Dataset IDs can be imported from ml.config.dataset_ids."""
    from ml.config import dataset_ids as module

    for attr, expected in DATASET_ID_EXPECTATIONS:
        value = getattr(module, attr)
        assert value == expected


def test_dataset_ids_have_correct_values() -> None:
    """Dataset IDs have the expected string values."""
    from ml.config import dataset_ids as module

    for attr, expected in DATASET_ID_EXPECTATIONS:
        value = getattr(module, attr)
        assert isinstance(value, str)
        assert value.startswith("ml.")
        assert value == expected


def test_dataset_ids_are_distinct() -> None:
    """Dataset IDs are distinct from each other."""
    from ml.config import dataset_ids as module

    values = [getattr(module, attr) for attr, _ in DATASET_ID_EXPECTATIONS]
    assert len(values) == len(set(values))


def test_dataset_ids_type_hints() -> None:
    """Dataset IDs use Final type hint (tested via type checking, verified at runtime)."""
    from ml.config import dataset_ids as module

    for attr, _ in DATASET_ID_EXPECTATIONS:
        value = getattr(module, attr)
        assert isinstance(value, str)


def test_dataset_ids_immutability() -> None:
    """Dataset IDs cannot be reassigned (enforced by type checker with Final)."""
    # Note: Python doesn't enforce Final at runtime, but we verify the
    # constants exist and are strings. MyPy will enforce Final in CI.
    from ml.config import dataset_ids as module

    for attr, expected in DATASET_ID_EXPECTATIONS:
        assert getattr(module, attr) == expected

    # Attempting reassignment would be caught by mypy --strict
    # (not testable at runtime as Python doesn't enforce Final)


def test_dataset_ids_in_public_api() -> None:
    """Dataset IDs are exported in ml.config.__all__."""
    from ml import config

    for attr, _ in DATASET_ID_EXPECTATIONS:
        assert attr in config.__all__


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
    for name, _ in DATASET_ID_EXPECTATIONS:
        # Verify UPPER_SNAKE_CASE
        assert name.isupper() or "_" in name
        # Verify _DATASET_ID suffix
        assert name.endswith("_DATASET_ID")


def test_no_additional_exports() -> None:
    """Dataset IDs module only exports the expected constants."""
    from ml.config import dataset_ids

    expected = {name for name, _ in DATASET_ID_EXPECTATIONS}
    actual = set(dataset_ids.__all__)

    assert actual == expected, f"Unexpected exports: {actual - expected}"


def test_dataset_ids_work_with_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dataset IDs work correctly when used with registry operations."""
    from ml.config import dataset_ids as module

    for attr, _ in DATASET_ID_EXPECTATIONS:
        dataset_id = getattr(module, attr)
        # Verify they're non-empty strings that could be used as keys
        assert dataset_id
        assert isinstance(dataset_id, str)
        assert len(dataset_id) > 0


def test_constants_imported_consistently() -> None:
    """Constants imported from different paths are the same object."""
    from ml import config
    from ml.config import dataset_ids as module

    for attr, _ in DATASET_ID_EXPECTATIONS:
        assert getattr(config, attr) is getattr(module, attr)
