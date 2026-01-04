"""Unit tests for TrainingCoordinator component.

This test module verifies the TrainingCoordinator component structure.

Phase 2.2.3 Status: STRUCTURAL PHASE
- All tests are SKIPPED for structural phase
- Tests verify component can be instantiated and has correct method signatures
- Full implementation testing deferred to Phase 2.2.8
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from ml.orchestration.config_types import TeacherTrainConfig
from ml.orchestration.training_coordinator import TrainingCoordinator


# ============================= Fixtures =============================


@pytest.fixture
def model_store() -> Mock:
    """Provides mock ModelStore for testing."""
    store = Mock()
    store.save_model.return_value = "model-id"
    store.get_model.return_value = None
    store.get_model_metrics.return_value = {}
    return store


@pytest.fixture
def model_registry() -> Mock:
    """Provides mock ModelRegistry for testing."""
    registry = Mock()
    registry.register_model.return_value = True
    registry.get_model_metadata.return_value = {}
    registry.list_model_versions.return_value = []
    return registry


@pytest.fixture
def training_coordinator(
    model_store: Mock,
    model_registry: Mock,
) -> TrainingCoordinator:
    """Provides TrainingCoordinator instance for testing."""
    return TrainingCoordinator(
        model_store=model_store,
        model_registry=model_registry,
        hpo_main=None,
        teacher_main=None,
        distill_cli=None,
    )


@pytest.fixture
def sample_hpo_config() -> Mock:
    """Provides sample HPO configuration."""
    config = Mock()
    config.hpo_trials = 10
    config.model_type = "xgboost"
    config.search_space = {"max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1]}
    return config


@pytest.fixture
def sample_training_config() -> Mock:
    """Provides sample training configuration."""
    config = Mock()
    config.model_type = "xgboost"
    config.dataset_id = "spy-ohlcv-1m"
    config.symbols = ["SPY"]
    config.start_date = "2024-01-01"
    config.end_date = "2024-12-31"
    return config


@pytest.fixture
def sample_distillation_config() -> Mock:
    """Provides sample distillation configuration."""
    config = Mock()
    config.teacher_model_id = "teacher-v1.0.0"
    config.student_model_type = "xgboost-small"
    config.distillation_temperature = 3.0
    config.alpha = 0.5  # Weight for distillation loss
    return config


@pytest.fixture
def sample_training_only_config() -> Mock:
    """Provides sample training-only configuration."""
    config = Mock()
    config.stages = ["TRAINING"]
    config.model_type = "xgboost"
    return config


@pytest.fixture
def sample_promotion_config() -> Mock:
    """Provides sample promotion configuration."""
    config = Mock()
    config.promotion_threshold = 0.8  # Accuracy threshold
    config.model_id = "xgboost-spy-v1.0.0"
    return config


@pytest.fixture
def mock_hpo_main() -> Mock:
    """Mock hpo_main CLI for testing."""

    def _hpo_main(argv=None):
        return 0  # Success

    return _hpo_main


@pytest.fixture
def mock_teacher_main() -> Mock:
    """Mock teacher_main CLI for testing."""

    def _teacher_main(argv=None):
        return 0  # Success

    return _teacher_main


@pytest.fixture
def mock_distill_cli() -> Mock:
    """Mock distill CLI for testing."""

    def _distill_cli(argv=None):
        return 0  # Success

    return _distill_cli


# ========================= Structural Tests =========================


@pytest.mark.unit
def test_training_coordinator_initializes_with_stores(
    model_store: Mock,
    model_registry: Mock,
) -> None:
    """Verify TrainingCoordinator can be instantiated with required stores.

    Phase 2.2.3: Verifies component structure
    Phase 2.2.8: Will verify stores are used correctly
    """
    coordinator = TrainingCoordinator(
        model_store=model_store,
        model_registry=model_registry,
        hpo_main=None,
        teacher_main=None,
        distill_cli=None,
    )

    assert coordinator is not None
    assert coordinator.model_store is model_store
    assert coordinator.model_registry is model_registry
    assert coordinator.hpo_main is None
    assert coordinator.teacher_main is None
    assert coordinator.distill_cli is None


@pytest.mark.unit
def test_training_coordinator_accepts_optional_cli_callables(
    model_store: Mock,
    model_registry: Mock,
    mock_hpo_main: Mock,
    mock_teacher_main: Mock,
    mock_distill_cli: Mock,
) -> None:
    """Verify TrainingCoordinator accepts optional CLI callable parameters.

    Phase 2.2.3: Verifies optional parameters accepted
    Phase 2.2.8: Will verify CLIs are invoked correctly
    """
    coordinator = TrainingCoordinator(
        model_store=model_store,
        model_registry=model_registry,
        hpo_main=mock_hpo_main,
        teacher_main=mock_teacher_main,
        distill_cli=mock_distill_cli,
    )

    assert coordinator.hpo_main is not None
    assert coordinator.teacher_main is not None
    assert coordinator.distill_cli is not None
    assert callable(coordinator.hpo_main)
    assert coordinator.hpo_main() == 0


@pytest.mark.unit
def test_training_coordinator_has_correct_method_signatures(
    training_coordinator: TrainingCoordinator,
) -> None:
    """Verify all 6 methods exist with correct type signatures.

    Phase 2.2.3: Verifies methods are callable
    Phase 2.2.8: Will verify methods execute correctly
    """
    assert callable(training_coordinator.run_hpo)
    assert callable(training_coordinator.train_teacher)
    assert callable(training_coordinator.distill_student)
    assert callable(training_coordinator.run_training_only)
    assert callable(training_coordinator._handle_promotions)
    assert callable(training_coordinator._execute_stage)


# ========================== Method Tests ============================


@pytest.mark.unit
def test_run_hpo_returns_success_placeholder(
    training_coordinator: TrainingCoordinator,
    sample_hpo_config: Mock,
    tmp_path: Path,
) -> None:
    """Verify run_hpo() returns 0 (success) when disabled.

    Phase 2.2.3: Returns 0 placeholder
    Phase 2.2.8: Will invoke hpo_main CLI and return actual exit code
    """
    # Create a mock config that returns False for enabled
    sample_hpo_config.enabled = False
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.touch()
    result = training_coordinator.run_hpo(sample_hpo_config, dataset_csv, tmp_path)
    assert result == 0  # Success (disabled skips)
    assert isinstance(result, int)


@pytest.mark.unit
def test_train_teacher_returns_success_placeholder(
    training_coordinator: TrainingCoordinator,
    sample_training_config: Mock,
    tmp_path: Path,
) -> None:
    """Verify train_teacher() returns 0 (success) when disabled.

    Phase 2.2.3: Returns 0 placeholder
    Phase 2.2.8: Will invoke teacher_main CLI and save model to ModelStore
    """
    # Create a mock config that returns False for enabled
    sample_training_config.enabled = False
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.touch()
    result = training_coordinator.train_teacher(sample_training_config, dataset_csv, tmp_path)
    assert result == 0  # Success (disabled skips)
    assert isinstance(result, int)


@pytest.mark.unit
def test_distill_student_returns_success_placeholder(
    training_coordinator: TrainingCoordinator,
    sample_distillation_config: Mock,
    tmp_path: Path,
) -> None:
    """Verify distill_student() returns 0 (success) when disabled.

    Phase 2.2.3: Returns 0 placeholder
    Phase 2.2.8: Will invoke distill CLI and save student model
    """
    # Create a mock config that returns False for enabled
    sample_distillation_config.enabled = False
    result = training_coordinator.distill_student(
        sample_distillation_config,
        dataset_dir=tmp_path,
        teacher_cfg=None,
    )
    assert result == 0  # Success (disabled skips)
    assert isinstance(result, int)


@pytest.mark.unit
def test_run_training_only_returns_success_placeholder(
    training_coordinator: TrainingCoordinator,
    sample_training_only_config: Mock,
) -> None:
    """Verify run_training_only() returns 0 (success) in structural phase.

    Phase 2.2.3: Returns 0 placeholder
    Phase 2.2.8: Will run full training workflow (HPO → Teacher → Student → Promotion)
    """
    result = training_coordinator.run_training_only(sample_training_only_config)
    assert result == 0  # Success placeholder
    assert isinstance(result, int)


@pytest.mark.unit
def test_handle_promotions_returns_none_placeholder(
    training_coordinator: TrainingCoordinator,
    sample_promotion_config: Mock,
) -> None:
    """Verify _handle_promotions() returns None in structural phase.

    Phase 2.2.3: Returns None immediately
    Phase 2.2.8: Will promote model if performance exceeds threshold
    """
    result = training_coordinator._handle_promotions(sample_promotion_config)
    assert result is None


@pytest.mark.unit
def test_execute_stage_returns_success_placeholder(
    training_coordinator: TrainingCoordinator,
    sample_hpo_config: Mock,
) -> None:
    """Verify _execute_stage() returns 0 (success) in structural phase.

    Phase 2.2.3: Returns 0 placeholder
    Phase 2.2.8: Will route to appropriate training method based on stage
    """
    from ml.config.events import Stage

    # Use valid Stage value for placeholder testing
    result = training_coordinator._execute_stage(Stage.DATA_INGESTED, sample_hpo_config)
    assert result == 0  # Success placeholder
    assert isinstance(result, int)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("prefer_parquet", "expected_flag"),
    ((True, "--train_data_parquet"), (False, "--train_data_csv")),
)
def test_train_teacher_builds_cli_args(
    model_store: Mock,
    model_registry: Mock,
    tmp_path: Path,
    prefer_parquet: bool,
    expected_flag: str,
) -> None:
    """Verify teacher training builds expected CLI args for parquet/CSV inputs."""
    captured: dict[str, list[str]] = {}

    def _teacher_main(argv: list[str] | None = None) -> int:
        captured["argv"] = argv or []
        return 0

    coordinator = TrainingCoordinator(
        model_store=model_store,
        model_registry=model_registry,
        hpo_main=None,
        teacher_main=_teacher_main,
        distill_cli=None,
    )

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    dataset_csv = dataset_dir / "dataset.csv"
    dataset_csv.write_text("timestamp,close\n1704067200,100.0\n", encoding="utf-8")
    (dataset_dir / "dataset.parquet").write_bytes(b"PAR1")
    metadata = {
        "dataset_id": "test_dataset",
        "vintage_policy": "real_time",
        "vintage_cutoff": None,
    }
    (dataset_dir / "dataset_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    cfg = TeacherTrainConfig(
        enabled=True,
        model_id="teacher_X",
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
        static_categoricals=("sector",),
        known_future_reals=("day_of_week",),
        save_interpretability=True,
        export_safetensors=True,
        pretrained_state_path="pretrained.safetensors",
        register_teacher=True,
        decision_policy="ml.policy.Policy",
        decision_config={"alpha": 0.5},
        prefer_parquet=prefer_parquet,
    )

    rc = coordinator.train_teacher(cfg, dataset_csv, dataset_dir)
    assert rc == 0

    argv = captured["argv"]
    assert expected_flag in argv

    def _arg_value(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    if prefer_parquet:
        assert "--train_data_csv" not in argv
        assert _arg_value("--train_data_parquet") == str(dataset_dir / "dataset.parquet")
    else:
        assert _arg_value("--train_data_csv") == str(dataset_csv)

    assert _arg_value("--batch_size") == "128"
    assert _arg_value("--dataloader_workers") == "2"
    assert _arg_value("--accelerator") == "cpu"
    assert _arg_value("--precision") == "bf16"
    assert _arg_value("--hidden_size") == "32"
    assert _arg_value("--loss") == "bce"
    assert _arg_value("--static_categoricals") == "sector"
    assert _arg_value("--known_future_reals") == "day_of_week"
    assert "--save_interpretability" in argv
    assert "--export_safetensors" in argv
    assert _arg_value("--pretrained_state_path") == "pretrained.safetensors"
    assert "--register_teacher" in argv
    assert _arg_value("--decision_policy") == "ml.policy.Policy"
    decision_payload = json.loads(_arg_value("--decision_config"))
    assert decision_payload == {"alpha": 0.5}
