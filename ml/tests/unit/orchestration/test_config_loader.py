#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from ml.orchestration.config_loader import load_orchestrator_config, to_pipeline_args


def test_load_json_and_to_args(tmp_path: Path) -> None:
    cfg_json = tmp_path / "cfg.json"
    cfg_json.write_text(
        """
        {
          "dataset": {
            "data_dir": "data/tier1",
            "symbols": "SPY.NYSE,QQQ.NYSE",
            "out_dir": "out",
            "include_macro": true,
            "macro_lag_days": 2,
            "include_micro": false,
            "include_l2": true,
            "horizon_minutes": 30,
            "threshold": 0.002,
            "lookback_periods": 40
          },
          "hpo": {"enabled": true, "epochs": 3, "batch_size": 16, "tail_rows": 100, "limit_groups": 10},
          "teacher": {"enabled": true, "model_id": "teacher_X", "max_epochs": 7}
        }
        """,
        encoding="utf-8",
    )
    cfg = load_orchestrator_config(str(cfg_json))
    args = to_pipeline_args(cfg)
    # Spot check
    assert "--include_macro" in args
    assert "--include_l2" in args
    assert "--hpo" in args
    assert ["--teacher_model_id", "teacher_X"] == args[
        args.index("--teacher_model_id") : args.index("--teacher_model_id") + 2
    ]


def test_load_toml(tmp_path: Path) -> None:
    cfg_toml = tmp_path / "cfg.toml"
    cfg_toml.write_text(
        """
        [dataset]
        data_dir = "data/tier1"
        symbols = "SPY.NYSE"
        out_dir = "out"
        include_macro = false
        include_l2 = false
        horizon_minutes = 15
        threshold = 0.001
        lookback_periods = 30

        [hpo]
        enabled = false

        [teacher]
        enabled = true
        model_id = "T1"
        max_epochs = 5

        [promotions]
        auto_register_model = true
        auto_promote = true
        gates_json = "ml/config/promotion_gates_example.json"
        deploy_target = "ml_actor"
        auto_register_features = true
        feature_metrics_json = "/tmp/feat.json"
        refresh_features = true
        """,
        encoding="utf-8",
    )
    cfg = load_orchestrator_config(str(cfg_toml))
    args = to_pipeline_args(cfg)
    assert "--hpo" not in args  # hpo disabled
    assert "--train" in args
    # promotions flags present
    assert "--auto_register_model" in args
    assert "--auto_promote" in args
    assert ["--gates_json", "ml/config/promotion_gates_example.json"] == args[
        args.index("--gates_json") : args.index("--gates_json") + 2
    ]
    assert ["--deploy_target", "ml_actor"] == args[
        args.index("--deploy_target") : args.index("--deploy_target") + 2
    ]
    assert "--auto_register_features" in args
    assert ["--feature_metrics_json", "/tmp/feat.json"] == args[
        args.index("--feature_metrics_json") : args.index("--feature_metrics_json") + 2
    ]
    assert "--refresh_features" in args
