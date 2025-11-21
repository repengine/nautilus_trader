#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.orchestration.config_loader import IngestionStageConfig, load_orchestrator_config, to_pipeline_args

pytestmark = pytest.mark.usefixtures(
    "isolated_prometheus_registry",
    "mock_tracing_backend",
    "isolated_orchestrator_env",
)


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
            "lookback_periods": 40,
            "market_dataset_id": "LEGACY.BARS",
            "market_inputs": [
              {"descriptor_id": "EQUS.MINI", "symbols": ["SPY"]}
            ]
          },
          "hpo": {"enabled": true, "epochs": 3, "batch_size": 16, "tail_rows": 100, "limit_groups": 10},
          "teacher": {"enabled": true, "model_id": "teacher_X", "max_epochs": 7},
          "student": {"enabled": true, "model_id": "student_X", "model_registry_dir": "registry"},
          "integration": {
            "enabled": true,
            "db_connection": "postgresql://example",
            "auto_start_postgres": true,
            "auto_migrate": true,
            "ensure_healthy": false,
            "strict_protocol_validation": true,
            "run_validators": false
          }
        }
        """,
        encoding="utf-8",
    )
    cfg = load_orchestrator_config(str(cfg_json))
    assert cfg.dataset.market_inputs is not None
    assert cfg.dataset.market_inputs[0].descriptor_id == "EQUS.MINI"
    assert cfg.dataset.market_inputs[0].symbols == ("SPY",)
    args = to_pipeline_args(cfg)
    # Spot check
    assert "--include_macro" in args
    assert "--include_l2" in args
    assert ["--market_dataset_id", "LEGACY.BARS"] == args[
        args.index("--market_dataset_id") : args.index("--market_dataset_id") + 2
    ]
    market_inputs_json = args[args.index("--market_inputs_json") + 1]
    payload = json.loads(market_inputs_json)
    assert payload[0]["descriptor_id"] == "EQUS.MINI"
    assert "--hpo" in args
    assert ["--teacher_model_id", "teacher_X"] == args[
        args.index("--teacher_model_id") : args.index("--teacher_model_id") + 2
    ]
    assert "--distill_student" in args
    assert ["--student_model_registry_dir", "registry"] == args[
        args.index("--student_model_registry_dir") : args.index("--student_model_registry_dir") + 2
    ]
    assert "--attach-runtime" in args
    assert ["--runtime-db-connection", "postgresql://example"] == args[
        args.index("--runtime-db-connection") : args.index("--runtime-db-connection") + 2
    ]
    assert "--runtime-auto-start-db" in args
    assert "--runtime-auto-migrate" in args
    assert "--runtime-no-ensure-healthy" in args
    assert "--runtime-strict-protocol-validation" in args
    assert "--runtime-skip-validators" in args


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


def test_to_pipeline_args_includes_catalog_cleaning(tmp_path: Path) -> None:
    cfg_toml = tmp_path / "cfg_ingestion.toml"
    cfg_toml.write_text(
        """
        [dataset]
        data_dir = "data/tier1"
        symbols = "SPY.NYSE"
        out_dir = "out"
        """,
        encoding="utf-8",
    )
    cfg = load_orchestrator_config(str(cfg_toml))
    ingestion = IngestionStageConfig(
        enabled=True,
        dataset_id="EQUS.MINI",
        schema="bars",
        instruments=("SPY.NYSE",),
        lookback_days=5,
        catalog_path="data/catalog",
        catalog_clean_mode="archive",
        catalog_backup_dir="ml_out/catalog_archives",
    )
    args = to_pipeline_args(cfg, ingestion=ingestion)
    assert "--ingest" in args
    assert ["--catalog_path", "data/catalog"] == args[
        args.index("--catalog_path") : args.index("--catalog_path") + 2
    ]
    assert ["--catalog_clean_mode", "archive"] == args[
        args.index("--catalog_clean_mode") : args.index("--catalog_clean_mode") + 2
    ]
    assert ["--catalog_backup_dir", "ml_out/catalog_archives"] == args[
        args.index("--catalog_backup_dir") : args.index("--catalog_backup_dir") + 2
    ]
