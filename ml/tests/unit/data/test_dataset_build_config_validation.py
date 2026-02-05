from __future__ import annotations

from pathlib import Path

import pytest

from ml.data import DatasetBuildConfig
from ml.data.build import _resolve_target_semantics


@pytest.mark.unit
def test_dataset_build_requires_target_semantics() -> None:
    cfg = DatasetBuildConfig(
        data_dir=Path("data"),
        out_dir=Path("out"),
        symbols=["SPY"],
        target_semantics=None,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="target_semantics must be provided"):
        _resolve_target_semantics(cfg)
