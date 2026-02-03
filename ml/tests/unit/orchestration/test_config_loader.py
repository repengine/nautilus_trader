#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.orchestration.config_loader import IngestionStageConfig, load_orchestrator_config, to_pipeline_args
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.config_types import HPOConfig
from ml.orchestration.config_types import OrchestratorConfig
from ml.orchestration.config_types import TeacherTrainConfig
from ml.tests.utils.targets import build_default_target_semantics_payload

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
            "target_semantics": {
              "version": "v1",
              "horizons": [{"minutes": 30}],
              "binary": {"enabled": true, "threshold_bps": 20.0, "return_basis": "raw"}
            },
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
    assert "--target_semantics" in args
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
        target_semantics = '{"version":"v1","horizons":[{"minutes":15}],"binary":{"enabled":true,"threshold_bps":10.0,"return_basis":"raw"}}'
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


def test_to_pipeline_args_includes_teacher_overrides(tmp_path: Path) -> None:
    cfg = OrchestratorConfig(
        dataset=DatasetBuildConfig(
            data_dir=str(tmp_path),
            symbols="SPY",
            out_dir=str(tmp_path / "out"),
            target_semantics=build_default_target_semantics_payload(),
        ),
        hpo=HPOConfig(enabled=False),
        teacher=TeacherTrainConfig(
            enabled=True,
            model_id="teacher_X",
            max_epochs=3,
            batch_size=128,
            dataloader_workers=2,
            accelerator="cpu",
            devices=1,
            precision="bf16",
            hidden_size=32,
            lstm_layers=2,
            attention_head_size=4,
            dropout=0.2,
            learning_rate=1e-3,
            loss="bce",
            pos_weight="auto",
            tail_rows=50,
            limit_groups=10,
            val_days=5,
            test_fraction=0.1,
            static_categoricals=("sector",),
            known_future_reals=("day_of_week",),
            save_interpretability=True,
            export_safetensors=True,
            pretrained_state_path="pretrained.safetensors",
            register_teacher=True,
            decision_policy="ml.policy.Policy",
            decision_config={"alpha": 0.5},
            prefer_parquet=False,
        ),
    )
    args = to_pipeline_args(cfg)

    def _arg_value(flag: str) -> str:
        return args[args.index(flag) + 1]

    assert _arg_value("--batch_size") == "128"
    assert _arg_value("--dataloader_workers") == "2"
    assert _arg_value("--accelerator") == "cpu"
    assert _arg_value("--precision") == "bf16"
    assert _arg_value("--hidden_size") == "32"
    assert _arg_value("--loss") == "bce"
    assert _arg_value("--static_categoricals") == "sector"
    assert _arg_value("--known_future_reals") == "day_of_week"
    assert "--save_interpretability" in args
    assert "--export_safetensors" in args
    assert _arg_value("--pretrained_state_path") == "pretrained.safetensors"
    assert "--register_teacher" in args
    assert _arg_value("--decision_policy") == "ml.policy.Policy"
    decision_payload = json.loads(_arg_value("--decision_config"))
    assert decision_payload == {"alpha": 0.5}
    assert "--no-prefer_parquet" in args


def test_to_pipeline_args_includes_catalog_cleaning(tmp_path: Path) -> None:
    cfg_toml = tmp_path / "cfg_ingestion.toml"
    target_payload = json.dumps(build_default_target_semantics_payload())
    cfg_toml.write_text(
        f"""
        [dataset]
        data_dir = "data/tier1"
        symbols = "SPY.NYSE"
        out_dir = "out"
        target_semantics = '{target_payload}'
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
