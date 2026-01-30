from __future__ import annotations

from ml.cli.parquet_live_replay_harness import _build_parser


def test_parser_accepts_max_holding_ms_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--catalog-path",
            "data/catalog",
            "--instrument-id",
            "SPY.EQUS",
            "--model-id",
            "dummy",
            "--model-path",
            "model.onnx",
            "--start-time",
            "2024-11-26T14:30:00Z",
            "--end-time",
            "2024-11-26T15:30:00Z",
            "--max-holding-ms",
            "60000",
        ]
    )

    assert args.max_holding_ms == 60000
