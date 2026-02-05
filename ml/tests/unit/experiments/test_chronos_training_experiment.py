from __future__ import annotations

import pytest

from ml.config.targets import TargetHorizonSpec
from ml.config.targets import TargetSemanticsConfig
from ml.experiments.chronos_training_experiment import _resolve_target_column


def test_resolve_target_column_when_missing_returns_primary_target() -> None:
    semantics = TargetSemanticsConfig()
    resolved = _resolve_target_column(semantics, None)
    assert resolved == semantics.resolved_primary_target()


def test_resolve_target_column_when_target_in_labels_returns_target() -> None:
    semantics = TargetSemanticsConfig()
    target_col = semantics.label_columns()[0]
    resolved = _resolve_target_column(semantics, target_col)
    assert resolved == target_col


def test_resolve_target_column_when_legacy_aliases_enabled_allows_alias() -> None:
    semantics = TargetSemanticsConfig(legacy_aliases=True)
    resolved = _resolve_target_column(semantics, "y")
    assert resolved == "y"


def test_resolve_target_column_when_multiple_labels_and_missing_target_raises() -> None:
    semantics = TargetSemanticsConfig(
        horizons=(
            TargetHorizonSpec(minutes=15, label="15m"),
            TargetHorizonSpec(minutes=30, label="30m"),
        ),
    )
    with pytest.raises(ValueError, match="target_col must be provided"):
        _resolve_target_column(semantics, None)
