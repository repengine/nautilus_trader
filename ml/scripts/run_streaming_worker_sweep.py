"""Command-line entry point for streaming worker hyperparameter sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from msgspec.structs import replace as struct_replace

from ml._imports import HAS_OPTUNA
from ml._imports import check_ml_dependencies
from ml.config.streaming_pipeline import CurriculumScheduleConfig
from ml.config.streaming_pipeline import CurriculumStageConfig
from ml.config.streaming_pipeline import DatasetServiceConfig
from ml.config.streaming_pipeline import EnsembleMemberConfig
from ml.config.streaming_pipeline import StreamingEnsembleConfig
from ml.config.streaming_pipeline import StreamingWorkerConfig
from ml.config.streaming_pipeline import parse_curriculum_stage_spec
from ml.config.streaming_pipeline import parse_ensemble_member_spec
from ml.training.event_driven.services import DatasetPlanRequest
from ml.training.event_driven.sweep import StreamingWorkerStudyRunner
from ml.training.event_driven.sweep import SweepSearchSpace
from ml.training.event_driven.sweep import build_trial_runner
from ml.training.teacher.streaming_loader import PhaseOneFeatureSignals
from ml.training.teacher.streaming_loader import TFTStreamingConfig


@dataclass(slots=True, frozen=True)
class FeatureLayout:
    """Dataset feature metadata required for dataset planning."""

    feature_names: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    phase_one_signals: PhaseOneFeatureSignals


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an Optuna sweep for streaming worker hyperparameters.")
    parser.add_argument("--dataset-dir", required=True, help="Path to the dataset directory (expects dataset.parquet and dataset_metadata.json).")
    parser.add_argument("--output-dir", help="Directory to write sweep artefacts (defaults to <dataset-dir>/streaming_worker_sweeps).")
    parser.add_argument("--max-trials", type=int, default=10, help="Maximum number of Optuna trials to execute.")
    parser.add_argument("--seed", type=int, help="Random seed for the sampler.")
    parser.add_argument("--study-name", default="streaming-worker-sweep", help="Optuna study name.")
    parser.add_argument("--base-batch-size", type=int, default=128, help="Initial batch size used when constructing the base dataset plan.")
    parser.add_argument("--batch-size-options", type=int, nargs="+", default=[64, 96, 128, 192], help="Candidate batch sizes for the sweep.")
    parser.add_argument("--hidden-size-options", type=int, nargs="+", default=[16, 32, 64, 96], help="Candidate hidden sizes for the sweep.")
    parser.add_argument("--lstm-layers-options", type=int, nargs="+", default=[1, 2, 3], help="Candidate LSTM layer counts.")
    parser.add_argument("--attention-head-options", type=int, nargs="+", default=[2, 4, 8], help="Candidate attention head sizes.")
    parser.add_argument("--dropout-options", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3], help="Candidate dropout rates.")
    parser.add_argument("--learning-rate-min", type=float, default=1e-4, help="Minimum learning rate sampled (log-uniform).")
    parser.add_argument("--learning-rate-max", type=float, default=5e-3, help="Maximum learning rate sampled (log-uniform).")
    parser.add_argument("--optimizer-options", nargs="+", default=["adam", "adamw"], help="Optimizers considered during the sweep.")
    parser.add_argument("--lr-scheduler-options", nargs="+", default=["reduce_on_plateau", "onecycle", "cosine"], help="LR schedulers considered during the sweep.")
    parser.add_argument("--max-epochs-options", type=int, nargs="+", default=[1, 2, 3], help="Epoch counts sampled by the sweep.")
    parser.add_argument("--max-encoder-length", type=int, default=30, help="Encoder length used when constructing the dataset plan.")
    parser.add_argument("--max-prediction-length", type=int, default=1, help="Prediction length used when constructing the dataset plan.")
    parser.add_argument("--dataloader-workers", type=int, default=0, help="Number of DataLoader workers to use during sweeps.")
    parser.add_argument("--enable-temperature-calibration", action="store_true", help="Enable temperature scaling so calibrated metrics are captured per trial.")
    parser.add_argument("--temperature-min", type=float, help="Minimum temperature to evaluate during calibration (overrides config/env).")
    parser.add_argument("--temperature-max", type=float, help="Maximum temperature to evaluate during calibration (overrides config/env).")
    parser.add_argument("--temperature-steps", type=int, help="Number of calibration temperature samples (overrides config/env).")
    parser.add_argument(
        "--enable-platt-calibration",
        action="store_true",
        help="Enable Platt scaling calibration per trial.",
    )
    parser.add_argument(
        "--enable-isotonic-calibration",
        action="store_true",
        help="Enable isotonic regression calibration per trial.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="Override worker precision (defaults to config/env).",
    )
    parser.add_argument(
        "--enable-amp",
        action="store_true",
        help="Enable automatic mixed precision for sweep trials.",
    )
    parser.add_argument(
        "--amp-precision",
        type=str,
        help="Precision string used when AMP is enabled (e.g., 16-mixed).",
    )
    parser.add_argument(
        "--enable-curriculum",
        action="store_true",
        help="Enable curriculum-aware train fraction overrides during sweeps.",
    )
    parser.add_argument(
        "--curriculum-stage",
        action="append",
        metavar="MAX_ROWS:TRAIN_FRACTION",
        help="Curriculum stage specification (repeatable).",
    )
    parser.add_argument(
        "--curriculum-default-train-fraction",
        type=float,
        help="Fallback train fraction used when curriculum stages do not match.",
    )
    parser.add_argument(
        "--enable-ensemble",
        action="store_true",
        help="Blend sweep logits with existing artefacts before scoring.",
    )
    parser.add_argument(
        "--ensemble-member",
        action="append",
        metavar="PATH[:WEIGHT[:required|optional]]",
        help="Additional logits artefact to blend for each trial (repeatable).",
    )
    parser.add_argument(
        "--ensemble-blend-mode",
        choices=("weighted", "mean"),
        help="Blend strategy when combining ensemble members (defaults to config/env).",
    )
    parser.add_argument(
        "--no-ensemble-normalize-weights",
        dest="ensemble_normalize_weights",
        action="store_false",
        help="Disable ensemble weight normalisation (defaults to enabled).",
    )
    parser.set_defaults(ensemble_normalize_weights=None)
    return parser.parse_args(argv)


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required metadata file missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected mapping structure in {path}, received {type(data).__name__}")
    return data


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value if item is not None)
    if value is None:
        return ()
    return (str(value),)


def _extract_phase_one_signals(metadata: Mapping[str, Any]) -> PhaseOneFeatureSignals:
    candidate_mappings: list[Mapping[str, Any]] = []
    for key in ("phase_one_signals", "phase_one_features"):
        raw = metadata.get(key)
        if isinstance(raw, Mapping):
            candidate_mappings.append(raw)
    column_info = metadata.get("column_info")
    if isinstance(column_info, Mapping):
        for key in ("phase_one_signals", "phase_one_features"):
            raw = column_info.get(key)
            if isinstance(raw, Mapping):
                candidate_mappings.append(raw)

    def _resolve(key: str, *aliases: str) -> tuple[str, ...]:
        names = (key,) + aliases
        for mapping in candidate_mappings:
            for name in names:
                if name in mapping:
                    return _as_str_tuple(mapping.get(name))
        for name in names:
            if name in metadata:
                return _as_str_tuple(metadata.get(name))
        return ()

    return PhaseOneFeatureSignals(
        macro_delta_columns=_resolve("macro_delta_columns", "macro_deltas"),
        calendar_lag_columns=_resolve("calendar_lag_columns", "calendar_lag_windows"),
        clustering_tag_columns=_resolve("clustering_tag_columns", "clustering_tags"),
        context_feature_columns=_resolve("context_feature_columns", "context_features"),
    )


def _build_feature_layout(metadata: Mapping[str, Any]) -> FeatureLayout:
    columns = metadata.get("column_info")
    if not isinstance(columns, Mapping):
        raise ValueError("metadata column_info must be an object")

    categorical = tuple(str(value) for value in columns.get("categorical_columns", ()))
    static_reals = tuple(str(value) for value in columns.get("static_reals", ()))
    known_reals = tuple(
        str(value) for value in columns.get("time_varying_known_reals", ()) if str(value) != "time_index"
    )
    unknown_reals = tuple(str(value) for value in columns.get("time_varying_unknown_reals", ()))
    vintage_age = tuple(str(value) for value in columns.get("vintage_age_columns", ()))

    numeric = tuple(dict.fromkeys(static_reals + known_reals + unknown_reals + vintage_age + ("y",)))
    feature_names = tuple(
        dict.fromkeys(static_reals + known_reals + unknown_reals + vintage_age + categorical),
    )
    phase_one_signals = _extract_phase_one_signals(metadata)
    return FeatureLayout(
        feature_names=feature_names,
        numeric_columns=numeric,
        categorical_columns=categorical,
        phase_one_signals=phase_one_signals,
    )


def _coerce_limit(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def _capability_flag(metadata: Mapping[str, Any], name: str, default: bool) -> bool:
    capability_flags = metadata.get("capability_flags")
    if isinstance(capability_flags, Mapping) and name in capability_flags:
        return bool(capability_flags[name])
    return default


def _build_streaming_config(
    *,
    metadata: Mapping[str, Any],
    layout: FeatureLayout,
    base_batch_size: int,
    dataloader_workers: int,
    service_config: DatasetServiceConfig,
    max_encoder_length: int,
    max_prediction_length: int,
) -> TFTStreamingConfig:
    columns = metadata.get("column_info")
    if not isinstance(columns, Mapping):
        raise ValueError("metadata column_info must be an object")

    lags_map = metadata.get("publication_lags", {})

    def _lag_value(key: str, default: int) -> int:
        raw = lags_map.get(key, default)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return default

    include_macro = _capability_flag(metadata, "include_macro", service_config.include_macro)
    include_calendar = _capability_flag(metadata, "include_calendar", service_config.include_calendar)
    include_events = _capability_flag(metadata, "include_events", service_config.include_events)
    include_earnings = _capability_flag(metadata, "include_earnings", service_config.include_earnings)
    include_micro = _capability_flag(metadata, "include_micro", service_config.include_micro)
    include_l2 = _capability_flag(metadata, "include_l2", service_config.include_l2)
    include_macro_revisions = _capability_flag(
        metadata,
        "include_macro_revisions",
        service_config.include_macro_revisions,
    )
    include_macro_deltas = _capability_flag(
        metadata,
        "include_macro_deltas",
        service_config.include_macro_deltas,
    )
    include_calendar_lags = _capability_flag(
        metadata,
        "include_calendar_lags",
        service_config.include_calendar_lags,
    )
    include_clustering_tags = _capability_flag(
        metadata,
        "include_clustering_tags",
        service_config.include_clustering_tags,
    )
    include_context_features = _capability_flag(
        metadata,
        "include_context_features",
        service_config.include_context_features,
    )

    macro_lag_days = _lag_value("macro_lag_days", 1 if include_macro else 0)
    earnings_lag_days = _lag_value("earnings_lag_days", 1 if include_earnings else 0)
    events_notice_minutes = _lag_value("events_notice_minutes", 0)

    return TFTStreamingConfig(
        time_idx_col=str(columns.get("time_idx_col", "time_index")),
        group_id_col=str(columns.get("group_id_col", "instrument_id")),
        target_col=str(columns.get("target_col", "y")),
        static_categoricals=layout.categorical_columns,
        static_reals=tuple(str(value) for value in columns.get("static_reals", ())),
        time_varying_known_reals=tuple(
            str(value) for value in columns.get("time_varying_known_reals", ()) if str(value) != "time_index"
        ),
        time_varying_unknown_reals=tuple(
            str(value) for value in columns.get("time_varying_unknown_reals", ())
        ),
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        batch_size=base_batch_size,
        drop_last=False,
        shuffle_shards=False,
        seed=7,
        num_workers=dataloader_workers,
        max_total_rows=_coerce_limit(service_config.max_total_rows),
        max_total_sequences=_coerce_limit(service_config.max_total_sequences),
        max_shards=_coerce_limit(service_config.max_shards),
        include_macro=include_macro,
        include_calendar=include_calendar,
        include_events=include_events,
        include_earnings=include_earnings,
        include_micro=include_micro or include_l2,
        include_l2=include_l2,
        include_macro_revisions=include_macro_revisions,
        include_macro_deltas=include_macro_deltas,
        include_calendar_lags=include_calendar_lags,
        include_clustering_tags=include_clustering_tags,
        include_context_features=include_context_features,
        macro_lag_days=macro_lag_days,
        earnings_lag_days=earnings_lag_days,
        events_notice_minutes=events_notice_minutes,
        phase_one_signals=layout.phase_one_signals,
    )


def _build_plan_request(
    *,
    dataset_dir: Path,
    streaming_config: TFTStreamingConfig,
    layout: FeatureLayout,
) -> DatasetPlanRequest:
    parquet_path = dataset_dir / "dataset.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset parquet not found at {parquet_path}")
    return DatasetPlanRequest(
        dataset_id=dataset_dir.name,
        streaming_config=streaming_config,
        feature_names=layout.feature_names,
        categorical_columns=layout.categorical_columns,
        numeric_columns=layout.numeric_columns,
        phase_one_signals=layout.phase_one_signals,
        parquet_path=parquet_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not HAS_OPTUNA:
        check_ml_dependencies(["optuna"])

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else dataset_dir / "streaming_worker_sweeps"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_json(dataset_dir / "dataset_metadata.json")
    layout = _build_feature_layout(metadata)

    service_config = DatasetServiceConfig.from_env(
        str(dataset_dir.parent),
        env=None,
    )
    worker_config = StreamingWorkerConfig.from_env()

    worker_overrides: dict[str, Any] = {}
    calibration_overrides: dict[str, Any] = {}
    if args.enable_temperature_calibration:
        calibration_overrides["enable_temperature_calibration"] = True
    if args.temperature_min is not None:
        calibration_overrides["temperature_calibration_min"] = float(args.temperature_min)
    if args.temperature_max is not None:
        calibration_overrides["temperature_calibration_max"] = float(args.temperature_max)
    if args.temperature_steps is not None:
        calibration_overrides["temperature_calibration_steps"] = int(args.temperature_steps)
    if args.enable_platt_calibration:
        calibration_overrides["enable_platt_calibration"] = True
    if args.enable_isotonic_calibration:
        calibration_overrides["enable_isotonic_calibration"] = True
    if calibration_overrides:
        worker_overrides.update(calibration_overrides)

    if args.precision:
        worker_overrides["precision"] = str(args.precision)
    if args.amp_precision:
        worker_overrides["amp_precision"] = str(args.amp_precision)
    if args.enable_amp:
        worker_overrides["enable_amp"] = True

    if args.enable_curriculum or args.curriculum_stage or args.curriculum_default_train_fraction is not None:
        base_schedule = worker_config.curriculum
        stages: tuple[CurriculumStageConfig, ...]
        if args.curriculum_stage:
            stages = tuple(parse_curriculum_stage_spec(spec) for spec in args.curriculum_stage)
        else:
            stages = base_schedule.stages
        default_fraction = (
            float(args.curriculum_default_train_fraction)
            if args.curriculum_default_train_fraction is not None
            else base_schedule.default_train_fraction
        )
        enabled = base_schedule.enabled or bool(args.enable_curriculum)
        worker_overrides["curriculum"] = CurriculumScheduleConfig(
            enabled=enabled,
            stages=stages,
            default_train_fraction=default_fraction,
        )

    if (
        args.enable_ensemble
        or args.ensemble_member
        or args.ensemble_blend_mode
        or args.ensemble_normalize_weights is not None
    ):
        base_ensemble = worker_config.ensemble
        members: tuple[EnsembleMemberConfig, ...]
        if args.ensemble_member:
            members = tuple(parse_ensemble_member_spec(spec) for spec in args.ensemble_member)
        else:
            members = base_ensemble.members
        blend_mode = (
            str(args.ensemble_blend_mode)
            if args.ensemble_blend_mode
            else base_ensemble.blend_mode
        )
        if args.ensemble_normalize_weights is None:
            normalize_weights = base_ensemble.normalize_weights
        else:
            normalize_weights = bool(args.ensemble_normalize_weights)
        enabled = base_ensemble.enabled or bool(args.enable_ensemble)
        worker_overrides["ensemble"] = StreamingEnsembleConfig(
            enabled=enabled,
            blend_mode=blend_mode,
            normalize_weights=normalize_weights,
            members=members,
        )

    if worker_overrides:
        worker_config = struct_replace(worker_config, **worker_overrides)

    streaming_config = _build_streaming_config(
        metadata=metadata,
        layout=layout,
        base_batch_size=args.base_batch_size,
        dataloader_workers=args.dataloader_workers,
        service_config=service_config,
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
    )
    plan_request = _build_plan_request(
        dataset_dir=dataset_dir,
        streaming_config=streaming_config,
        layout=layout,
    )

    trial_runner = build_trial_runner(
        dataset_service_config=service_config,
        dataset_request=plan_request,
        worker_config=worker_config,
        output_root=output_dir,
    )

    search_space = SweepSearchSpace(
        batch_sizes=tuple(int(value) for value in args.batch_size_options),
        hidden_sizes=tuple(int(value) for value in args.hidden_size_options),
        lstm_layers=tuple(int(value) for value in args.lstm_layers_options),
        attention_head_sizes=tuple(int(value) for value in args.attention_head_options),
        dropouts=tuple(float(value) for value in args.dropout_options),
        learning_rate_range=(float(args.learning_rate_min), float(args.learning_rate_max)),
        optimizers=tuple(str(value).lower() for value in args.optimizer_options),
        lr_schedulers=tuple(str(value).lower() for value in args.lr_scheduler_options),
        max_epochs=tuple(int(value) for value in args.max_epochs_options),
    )

    study_runner = StreamingWorkerStudyRunner(
        runner=trial_runner,
        search_space=search_space,
        output_dir=output_dir,
        study_name=args.study_name,
    )
    study = study_runner.run(args.max_trials, seed=args.seed)

    best_value = getattr(study, "best_value", None)
    best_params = getattr(study, "best_params", {})
    print("Sweep completed.")
    print(f"Best objective: {best_value!r}")
    print(f"Best parameters: {best_params}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main(sys.argv[1:]))
