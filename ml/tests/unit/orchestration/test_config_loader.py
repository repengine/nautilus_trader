#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import ml.orchestration.config_loader as config_loader_module
from ml.config.market_data import MarketDatasetInput
from ml.orchestration.config_loader import IngestionStageConfig, load_orchestrator_config, to_pipeline_args
from ml.orchestration.config_loader import load_orchestrator_run_config
from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.config_types import DatasetValidationConfig
from ml.orchestration.config_types import HPOConfig
from ml.orchestration.config_types import IntegrationConfig
from ml.orchestration.config_types import OrchestratorConfig
from ml.orchestration.config_types import PromotionsConfig
from ml.orchestration.config_types import StudentDistillConfig
from ml.orchestration.config_types import TeacherTrainConfig
from ml.data.vintage import VintagePolicy
from ml.registry.dataclasses import StorageKind
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


def test_internal_loader_helpers_handle_overrides_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    assert config_loader_module._env_truthy(" yes ") is True
    assert config_loader_module._env_truthy("off") is False
    assert config_loader_module._env_truthy(None) is False

    monkeypatch.setenv("ML_TFT_FORCE_MICRO_CACHE", "true")
    assert config_loader_module._force_micro_cache_policy() is True

    overrides = config_loader_module._collect_overrides(
        {
            "ML_ORCH__DATASET__SYMBOLS": "SPY,QQQ",
            "ML_ORCH____": "ignored",
            "OTHER_ENV": "noop",
        },
    )
    assert overrides == {("dataset", "symbols"): "SPY,QQQ"}

    payload: dict[str, object] = {"dataset": {"symbols": "SPY"}}
    config_loader_module._apply_overrides(payload, overrides)
    assert cast(dict[str, str], payload["dataset"])["symbols"] == "SPY,QQQ"

    with pytest.raises(ValueError, match="Cannot override segment"):
        config_loader_module._apply_overrides(
            {"dataset": "not-a-mapping"},
            {("dataset", "symbols"): "SPY"},
        )

    assert config_loader_module._parse_override_value('{"k": 1}') == {"k": 1}
    assert config_loader_module._parse_override_value("true") is True
    assert config_loader_module._parse_override_value("null") is None
    assert config_loader_module._parse_override_value("   ") == ""
    assert config_loader_module._parse_override_value("raw-token") == "raw-token"

    with pytest.raises(ValueError, match="Unsupported stage value"):
        config_loader_module._parse_stage("unknown-stage")

    assert config_loader_module._ensure_sequence(["a", "b"]) == ("a", "b")
    assert config_loader_module._ensure_sequence(None) == ()
    assert config_loader_module._ensure_sequence("x") == ("x",)
    assert config_loader_module._is_tuple_field(tuple[str, ...]) is True
    assert config_loader_module._is_tuple_field(str) is False


def test_symbol_and_market_input_coercion_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    alias = next(iter(config_loader_module._UNIVERSE_ALIAS_MAP.keys()))
    alias_symbols = config_loader_module._tokenize_symbol_entry(f"@{alias}")
    assert alias_symbols == list(config_loader_module._UNIVERSE_ALIAS_MAP[alias])
    assert config_loader_module._tokenize_symbol_entry("SPY,QQQ") == ["SPY", "QQQ"]
    assert config_loader_module._tokenize_symbol_entry("SPY") == ["SPY"]

    with pytest.raises(ValueError, match="Unknown universe alias"):
        config_loader_module._tokenize_symbol_entry("@definitely-missing")

    expanded = config_loader_module._expand_symbol_sequence(
        ["spy.xnas, qqq.xnas", "SPY.XNAS", 123],
        drop_venues=True,
    )
    assert expanded == ("SPY", "QQQ", "123")
    assert config_loader_module._expand_symbol_sequence(None, drop_venues=False) == ()

    monkeypatch.setenv("ML_TFT_FORCE_MICRO_CACHE", "1")
    dataset_payload = {
        "data_dir": str(tmp_path / "data"),
        "symbols": ["SPY.XNAS", "QQQ.XNAS"],
        "out_dir": str(tmp_path / "out"),
        "instrument_ids": ["SPY.XNAS", "QQQ.XNAS"],
        "target_semantics": json.dumps(build_default_target_semantics_payload()),
        "macro_series_ids": ["DGS10"],
        "market_inputs": [
            {
                "descriptor_id": "EQUS.MINI",
                "dataset_id": "EQUS.MINI",
                "symbols": ["SPY.XNAS"],
                "schema_override": "ohlcv-1m",
                "storage_kind": "parquet",
            },
        ],
    }
    dataset_cfg = config_loader_module._coerce_dataset(dataset_payload)
    assert dataset_cfg.symbols == "SPY,QQQ"
    assert dataset_cfg.instrument_ids == ("SPY.XNAS", "QQQ.XNAS")
    assert dataset_cfg.market_inputs is not None
    assert dataset_cfg.market_inputs[0].descriptor_id == "EQUS.MINI"
    assert dataset_cfg.micro_cache_policy == "cache_first"
    assert dataset_cfg.l2_cache_policy == "cache_first"
    assert dataset_cfg.macro_series_ids == ("DGS10",)

    with pytest.raises(ValueError, match=r"dataset\.target_semantics"):
        config_loader_module._coerce_dataset(
            {
                "data_dir": str(tmp_path / "data"),
                "symbols": "SPY",
                "out_dir": str(tmp_path / "out"),
                "target_semantics": "{not-json}",
            },
        )

    ingestion_cfg = config_loader_module._coerce_ingestion(
        {
            "enabled": True,
            "dataset_id": 123,
            "schema": "bars",
            "instruments": ["SPY.XNAS"],
            "instrument_ids": [],
            "symbols": ["SPY.XNAS"],
            "market_inputs": [],
            "catalog_path": tmp_path / "catalog",
            "catalog_clean_mode": "archive",
            "catalog_backup_dir": tmp_path / "backups",
        },
    )
    assert ingestion_cfg is not None
    assert ingestion_cfg.dataset_id == "123"
    assert ingestion_cfg.symbols == ("SPY",)
    assert ingestion_cfg.instrument_ids is None
    assert ingestion_cfg.market_inputs is None

    existing_input = MarketDatasetInput(
        descriptor_id="EQUS.MINI",
        dataset_id="EQUS.MINI",
        symbols=("SPY.XNAS",),
    )
    market_inputs = config_loader_module._coerce_market_inputs(
        [
            existing_input,
            {"dataset_id": "XNAS.ITCH", "symbols": ["AAPL.XNAS"], "storage_kind": "postgres"},
        ],
    )
    assert market_inputs[0] is existing_input
    assert market_inputs[1].storage_kind_override is StorageKind.POSTGRES

    with pytest.raises(ValueError, match="market_inputs entries must be mappings"):
        config_loader_module._coerce_market_inputs(["invalid-entry"])

    with pytest.raises(ValueError, match="TeacherTrainConfig section must be a mapping"):
        config_loader_module._coerce_dataclass("not-a-mapping", TeacherTrainConfig)

    teacher_cfg = config_loader_module._coerce_dataclass(
        {
            "enabled": True,
            "model_id": "teacher",
            "static_categoricals": ["sector"],
            "known_future_reals": ["day_of_week"],
        },
        TeacherTrainConfig,
    )
    assert tuple(teacher_cfg.static_categoricals) == ("sector",)
    assert tuple(teacher_cfg.known_future_reals) == ("day_of_week",)


def test_load_orchestrator_run_config_applies_env_and_auto_fill(tmp_path: Path) -> None:
    cfg_json = tmp_path / "run_cfg.json"
    cfg_json.write_text(
        json.dumps(
            {
                "stage": "dataset",
                "dataset": {
                    "data_dir": str(tmp_path / "data"),
                    "symbols": "SPY.XNAS",
                    "out_dir": str(tmp_path / "out"),
                    "target_semantics": build_default_target_semantics_payload(),
                },
                "training": {"teacher": {"enabled": False}},
                "auto_fill": {
                    "enabled": True,
                    "dataset_id": "EQUS.MINI",
                    "instrument_ids": ["SPY.XNAS,QQQ.XNAS"],
                },
            },
        ),
        encoding="utf-8",
    )

    run_cfg = load_orchestrator_run_config(
        cfg_json,
        env={
            "ML_ORCH__STAGE": "full",
            "ML_ORCH__DATASET__SYMBOLS": "AAPL.XNAS,MSFT.XNAS",
            "ML_ORCH__INGESTION__ENABLED": "true",
        },
    )
    assert run_cfg.stage is config_loader_module.Stage.FULL
    assert run_cfg.dataset is not None
    assert run_cfg.dataset.symbols == "AAPL,MSFT"
    assert run_cfg.ingestion is not None
    assert run_cfg.ingestion.enabled is True
    assert run_cfg.auto_fill is not None
    assert run_cfg.auto_fill.instrument_ids == ("SPY.XNAS", "QQQ.XNAS")

    list_payload = tmp_path / "bad.json"
    list_payload.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_orchestrator_run_config(list_payload)

    with pytest.raises(FileNotFoundError):
        load_orchestrator_run_config(tmp_path / "missing.json")


def test_to_pipeline_args_covers_optional_sections(tmp_path: Path) -> None:
    dataset = DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        symbols="SPY,QQQ",
        out_dir=str(tmp_path / "out"),
        target_semantics=build_default_target_semantics_payload(),
        include_macro=True,
        macro_lag_days=3,
        include_micro=True,
        include_l2=True,
        include_events=True,
        include_calendar=True,
        instrument_ids=("SPY.XNAS", "QQQ.XNAS"),
        market_dataset_id="EQUS.MINI",
        market_inputs=(
            MarketDatasetInput(
                descriptor_id="EQUS.MINI",
                dataset_id="EQUS.MINI",
                symbols=("SPY.XNAS",),
                schema_override="ohlcv-1m",
                storage_kind_override=StorageKind.PARQUET,
                start="2025-01-01",
                end="2025-01-10",
            ),
        ),
        student_mode=True,
        emit_dataset_events=True,
        start_iso="2025-01-01",
        end_iso="2025-01-31",
        chunk_days=5,
        fred_vintage_dir=str(tmp_path / "vintage"),
        events_dir=str(tmp_path / "events"),
        feature_registry_dir=str(tmp_path / "feature_registry"),
        register_features=True,
        auto_refresh_macro=False,
        macro_staleness_hours=8,
        macro_series_ids=("DGS10",),
        macro_fred_path=str(tmp_path / "fred.parquet"),
        validation=DatasetValidationConfig(
            min_rows=10,
            min_positive_rate=0.1,
            max_positive_rate=0.9,
            min_feature_coverage=0.8,
        ),
        vintage_policy=VintagePolicy.FINAL,
        vintage_as_of="2025-01-15",
    )
    cfg = OrchestratorConfig(
        dataset=dataset,
        hpo=HPOConfig(enabled=True, epochs=2, batch_size=8, tail_rows=50, limit_groups=4),
        teacher=TeacherTrainConfig(
            enabled=True,
            model_id="teacher_full",
            feature_registry_dir="teacher_features",
            feature_set_id="teacher_set",
            embargo_pct=0.2,
            pos_weight="auto",
            seed=7,
            static_categoricals=("sector",),
            static_reals=("market_cap",),
            known_future_reals=("day_of_week",),
            export_torchscript=True,
            export_safetensors=True,
            pretrained_state_path="weights.safetensors",
            register_teacher=True,
            decision_policy="ml.policy",
            decision_config={"beta": 0.1},
            prefer_parquet=False,
        ),
        student=StudentDistillConfig(
            enabled=True,
            model_id="student",
            parent_model_id="teacher_full",
            model_registry_dir="student_registry",
            feature_registry_dir="student_features",
            feature_set_id="student_set",
            opset=17,
            use_val_for_distill=True,
        ),
        promotions=PromotionsConfig(
            auto_register_model=True,
            gates_json="gates.json",
            auto_promote=True,
            deploy_target="ml_actor",
            auto_register_features=True,
            feature_metrics_json="feature_metrics.json",
            refresh_features=True,
        ),
        auto_fill=config_loader_module.AutoFillUniverseConfig(
            enabled=True,
            dataset_id="EQUS.MINI",
            instrument_ids=("SPY.XNAS",),
            include_l2=False,
            include_l3=True,
            l2_dataset_id="XNAS.BOOK",
            l2_schema="mbp-1",
            l2_days=2,
            l2_progress_file=str(tmp_path / "l2_progress.json"),
            l3_dataset_id="XNAS.DEPTH",
            l3_schema="mbo",
            l3_days=3,
            disable_dataset_l2_ingest=False,
        ),
        integration=IntegrationConfig(
            enabled=True,
            db_connection="postgresql://localhost/ml",
            auto_start_postgres=True,
            auto_migrate=True,
            ensure_healthy=False,
            strict_protocol_validation=True,
            run_validators=False,
        ),
    )
    ingestion = IngestionStageConfig(
        enabled=False,
        dataset_id="EQUS.MINI",
        schema="bars",
        instruments=("SPY.XNAS",),
        lookback_days=5,
        coverage_mode="sql",
        write_mode="sql",
        catalog_path=str(tmp_path / "catalog"),
        catalog_clean_mode="archive",
        catalog_backup_dir=str(tmp_path / "backup"),
        market_dataset_id="EQUS.MINI",
    )

    args = to_pipeline_args(cfg, ingestion=ingestion)
    assert "--include_events" in args
    assert "--include_calendar" in args
    assert "--student_mode" in args
    assert "--emit_dataset_events" in args
    assert "--skip_macro_refresh" in args
    assert "--macro_freshness_hours" in args
    assert "--hpo" in args
    assert "--distill_student" in args
    assert "--student_parent_model_id" in args
    assert "--student_feature_registry_dir" in args
    assert "--student_feature_set_id" in args
    assert "--student_opset" in args
    assert "--student_use_val_for_distill" in args
    assert "--auto_fill_universe" in args
    assert "--auto_fill_skip_l2" in args
    assert "--auto_fill_include_l3" in args
    assert "--auto_fill_allow_dataset_l2_ingest" in args
    assert "--attach-runtime" in args
    assert "--runtime-skip-validators" in args
    assert "--ingest" not in args

    payload_json = args[args.index("--market_inputs_json") + 1]
    payload = json.loads(payload_json)
    assert payload[0]["descriptor_id"] == "EQUS.MINI"
    assert payload[0]["storage_kind"] == "parquet"
    assert payload[0]["start"] == "2025-01-01"
    assert payload[0]["end"] == "2025-01-10"


def test_to_pipeline_args_validates_target_semantics_type(tmp_path: Path) -> None:
    dataset = DatasetBuildConfig(
        data_dir=str(tmp_path / "data"),
        symbols="SPY",
        out_dir=str(tmp_path / "out"),
        target_semantics=build_default_target_semantics_payload(),
    )
    bad_dataset = replace(dataset, target_semantics=cast(dict[str, object], ["bad"]))  # type: ignore[arg-type]
    cfg = OrchestratorConfig(dataset=bad_dataset, hpo=HPOConfig(enabled=False), teacher=TeacherTrainConfig(enabled=False))

    with pytest.raises(ValueError, match="JSON object payload"):
        to_pipeline_args(cfg)
