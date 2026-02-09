from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from ml.cli import train_tft_quick as cli
from ml.tests.utils.targets import build_default_target_semantics_payload


def test_train_tft_quick_cli_builds_config_from_canonical_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_train(config: object) -> object:
        captured["config"] = config

        class _Result:
            dataset_parquet = tmp_path / "dataset.parquet"
            dataset_csv = tmp_path / "dataset.csv"
            dataset_shape = (3, 5)
            target_distribution_json = '{"0":2,"1":1}'
            trained = False
            sample_predictions = None

        return _Result()

    monkeypatch.setattr(cli, "train_tft_quick", _fake_train)

    target_semantics = build_default_target_semantics_payload()
    rc = cli.main(
        [
            "--target-semantics",
            json.dumps(target_semantics),
        ],
    )

    assert rc == 0
    assert "config" in captured

    config = captured["config"]
    default_fields = cli.QuickTFTTrainConfig.__dataclass_fields__
    default_symbols = cast(tuple[str, ...], default_fields["symbols"].default)
    default_data_dirs = cast(tuple[Path, ...], default_fields["data_dirs"].default)
    default_output_dir = cast(Path, default_fields["output_dir"].default)

    assert tuple(config.symbols) == default_symbols
    assert tuple(config.data_dirs) == default_data_dirs
    assert config.output_dir == default_output_dir
