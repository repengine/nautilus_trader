from __future__ import annotations

from pathlib import Path

import pytest

from ml.cli.feature_cli import cli_deprecate
from ml.cli.feature_cli import cli_promote_with_gates
from ml.cli.feature_cli import cli_register_default


@pytest.mark.parallel_safe
@pytest.mark.unit
def test_cli_register_and_promote(tmp_path: Path) -> None:
    fid = cli_register_default(str(tmp_path), name="default", version="1.0.0")
    # Provide simple gates that likely pass for default manifest-less digests
    # Since no digests are present, promoting should fail when required gates are specified.
    ok = cli_promote_with_gates(
        str(tmp_path),
        fid,
        gates=[{"metric_name": "tolerance", "threshold": 0.0, "comparison": "gte"}],
    )
    assert ok is False
    # Deprecate works
    cli_deprecate(str(tmp_path), fid, reason="test")
