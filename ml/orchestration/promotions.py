#!/usr/bin/env python3

from __future__ import annotations


# ruff: noqa: E402  # Allow module docstring preceding imports per project style

"""
Promotion helpers for the pipeline orchestrator (cold path).

These helpers compose existing registries and the centralized event emitter.
They avoid expanding orchestrator complexity and keep all work off hot paths.
"""

import sys
from pathlib import Path
from typing import Any, cast


def _ensure_nautilus_numeric_factories() -> None:
    """
    Provide compatibility shims for older nautilus_trader numeric factories.
    """
    try:
        from nautilus_trader.model.objects import Price as _Price
        from nautilus_trader.model.objects import Quantity as _Quantity
    except Exception:
        return

    objects_mod = sys.modules.get("nautilus_trader.model.objects")
    if objects_mod is None:
        return

    if not hasattr(_Price, "from_double"):
        price_cls = cast(type[Any], _Price)
        price_attrs = {
            "from_double": staticmethod(lambda value: _Price.from_str(f"{value}")),
        }
        price_compat = cast(type[Any], type(_Price.__name__, (price_cls,), price_attrs))
        price_compat.__qualname__ = _Price.__qualname__
        setattr(objects_mod, "Price", price_compat)

    if not hasattr(_Quantity, "from_double"):
        qty_cls = cast(type[Any], _Quantity)
        qty_attrs = {
            "from_double": staticmethod(lambda value: _Quantity.from_str(f"{value}")),
        }
        qty_compat = cast(type[Any], type(_Quantity.__name__, (qty_cls,), qty_attrs))
        qty_compat.__qualname__ = _Quantity.__qualname__
        setattr(objects_mod, "Quantity", qty_compat)


_ensure_nautilus_numeric_factories()

from ml.common.event_emitter import emit_dataset_event
from ml.config.events import EventStatus
from ml.config.events import Source
from ml.config.events import Stage
from ml.registry.base import DataRequirements
from ml.registry.base import ModelManifest
from ml.registry.base import ModelRole
from ml.registry.dataclasses import QualityGate


def _load_json(path: Path) -> dict[str, Any]:
    import json

    return dict(json.loads(path.read_text(encoding="utf-8")))


def _ensure_parent_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "Creating parent directory failed (ignored): %s",
            exc,
            exc_info=True,
        )


def register_and_promote_model(
    model_metrics_path: str,
    out_dir: str,
    registry: Any,
    feature_registry: Any,
    gates: list[QualityGate],
    auto_promote: bool,
    deploy_target: str | None,
) -> str:
    """
    Register a model from metrics/artifacts and optionally promote/deploy.

    Parameters
    ----------
    model_metrics_path : str
        JSON file with minimally: {"model_path", "model_id", "architecture",
        "feature_schema" (mapping) or "feature_schema_hash" (str), and
        optionally "feature_set_id", "version", and numeric metrics to track}.
    out_dir : str
        Output directory containing artifacts (used for attachments if desired).
    registry : ModelRegistry-like
        Model registry instance (use MLIntegrationManager to obtain).
    feature_registry : FeatureRegistry-like
        Feature registry (used to validate/augment feature schema linkage).
    gates : list[QualityGate]
        Quality gates to validate during registration.
    auto_promote : bool
        If True and gates pass, deploy to ``deploy_target`` when provided.
    deploy_target : str | None
        Target identifier for deployment.

    """
    # Load metrics and resolve model/artifacts
    metrics_path = Path(model_metrics_path)
    data = _load_json(metrics_path)

    model_path = Path(str(data.get("model_path", Path(out_dir) / "model.onnx")))
    model_id = str(data.get("model_id", "model_1"))
    architecture = str(data.get("architecture", "unknown"))
    feature_schema: dict[str, str] | None = data.get("feature_schema")
    feature_schema_hash: str | None = data.get("feature_schema_hash")
    feature_set_id = data.get("feature_set_id")
    serveable = bool(data.get("serveable", True))
    version = str(data.get("version", "1.0.0"))

    if feature_schema_hash is None and feature_schema is not None:
        # Stable hash of names/types
        import hashlib

        h = hashlib.sha256()
        for k in sorted(feature_schema):
            h.update(k.encode("utf-8"))
            h.update(b"::")
            h.update(str(feature_schema[k]).encode("utf-8"))
            h.update(b"\n")
        feature_schema_hash = h.hexdigest()

    if feature_schema_hash is None:
        raise ValueError("feature_schema_hash or feature_schema is required")

    # Construct minimal manifest
    manifest = ModelManifest(
        model_id=model_id,
        role=ModelRole.TEACHER if not serveable else ModelRole.INFERENCE,
        data_requirements=(
            DataRequirements.HISTORICAL if not serveable else DataRequirements.L1_ONLY
        ),
        architecture=architecture,
        feature_schema=feature_schema or {},
        feature_schema_hash=feature_schema_hash,
        feature_set_id=feature_set_id,
        version=version,
        serveable=serveable,
        artifact_format="onnx" if serveable else "none",
    )

    # Register with quality gates enforced
    model_id_out = registry.register_model(
        model_path=model_path,
        manifest=manifest,
        auto_deploy=False,
        quality_gates=gates,
        enforce_quality=True,
    )

    # Attach optional metadata fields into registry metadata (best-effort)
    try:
        meta_update: dict[str, object] = {}
        td = data.get("training_dataset_id")
        if isinstance(td, str) and td:
            meta_update["training_dataset_id"] = td
        uids = data.get("universe_instrument_ids")
        if isinstance(uids, list):
            meta_update["universe_instrument_ids"] = [str(x) for x in uids]
        usyms = data.get("universe_symbols")
        if isinstance(usyms, list):
            meta_update["universe_symbols"] = [str(x) for x in usyms]
        if meta_update:
            registry.update_metadata(model_id_out, meta_update)
    except Exception:
        # Non-fatal advisory metadata update
        pass

    # Track numeric metrics in performance history
    perf_metrics = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    if perf_metrics:
        registry.track_performance(model_id_out, perf_metrics)

    # Optionally deploy if serveable and gates passed
    if auto_promote and serveable and deploy_target:
        try:
            registry.deploy_model(model_id_out, deploy_target)
        except Exception as exc:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "Model deploy failed (continuing to emit event): %s",
                exc,
                exc_info=True,
            )

    # Emit SUCCESS event for registration/promotion (best-effort)
    try:
        from ml.core.integration import MLIntegrationManager

        mgr = MLIntegrationManager(
            auto_start_postgres=False,
            auto_migrate=False,
            ensure_healthy=False,
        )
        from typing import Any, cast

        data_registry = mgr.data_registry
        emit_dataset_event(
            cast(Any, data_registry),
            dataset_id="model",
            instrument_id="GLOBAL",
            stage=Stage.PREDICTION_EMITTED,
            source=Source.HISTORICAL,
            run_id=f"register_{model_id_out}",
            ts_min=0,
            ts_max=0,
            count=1,
            status=EventStatus.SUCCESS,
            metadata={
                "model_id": model_id_out,
                "auto_promote": bool(auto_promote),
                "deploy_target": str(deploy_target or ""),
            },
            dataset_type="model",
            component="promotions",
        )
    except Exception as exc:
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "Emit model registration event failed: %s",
            exc,
            exc_info=True,
        )

    return str(model_id_out)


def register_or_refresh_features(
    feature_metrics_path: str,
    feature_registry: Any,
    auto_register: bool,
) -> str | None:
    """
    Create/update a FeatureRegistry manifest and attach metrics/artifacts.

    The metrics JSON must contain at least ``feature_set_id`` and can include
    a mapping of numeric metrics which are persisted into ``perf_digest``.

    """
    data = _load_json(Path(feature_metrics_path))
    feature_set_id = str(data.get("feature_set_id", "")).strip()
    if not feature_set_id:
        return None

    info = feature_registry.get_feature_set(feature_set_id)
    if info is None and auto_register:
        # Minimal manifest for registration; callers can enrich later
        from ml.registry.base import DataRequirements
        from ml.registry.feature_registry import FeatureManifest
        from ml.registry.feature_registry import FeatureRole

        manifest = FeatureManifest(
            feature_set_id=feature_set_id,
            name=feature_set_id,
            version="1.0.0",
            role=FeatureRole.TEACHER,
            data_requirements=DataRequirements.HISTORICAL,
            feature_names=[],
            feature_dtypes=[],
            schema_hash="",
            pipeline_signature="",
            pipeline_version="",
        )
        feature_registry.register_feature_set(manifest)

    # Persist metrics into perf_digest
    numeric_metrics = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    if numeric_metrics:
        feature_registry.update_manifest(feature_set_id, perf_digest=numeric_metrics)

    # Emit SUCCESS event
    try:
        from ml.core.integration import MLIntegrationManager

        mgr = MLIntegrationManager(
            auto_start_postgres=False,
            auto_migrate=False,
            ensure_healthy=False,
        )
        from typing import Any, cast

        data_registry = mgr.data_registry
        emit_dataset_event(
            cast(Any, data_registry),
            dataset_id="features",
            instrument_id="GLOBAL",
            stage=Stage.FEATURE_COMPUTED,
            source=Source.HISTORICAL,
            run_id=f"features_{feature_set_id}",
            ts_min=0,
            ts_max=0,
            count=1,
            status=EventStatus.SUCCESS,
            metadata={"feature_set_id": feature_set_id},
            dataset_type="features",
            component="promotions",
        )
    except Exception as exc:
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "Emit feature registration event failed: %s",
            exc,
            exc_info=True,
        )

    return feature_set_id


from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ml.orchestration.stage2_engine import Stage2Result
from ml.orchestration.stage2_engine import build_engine


@dataclass(slots=True, frozen=True)
class Stage2Config:
    """
    Configuration for promotion stage 2 (walk-forward + backtest, cold path).

    All inputs are cold-path friendly. This stage operates on dataset artifacts and
    training outputs to compute trading metrics with simple cost modeling, then applies
    quality gates and optionally deploys the model to a target.

    """

    out_dir: str
    dataset_csv: str
    data_dir: str
    horizon_minutes: int
    # Engine selection: 'returns' (default) or 'backtest' (advisory)
    engine_mode: Literal["returns", "backtest"] = "returns"
    # Cost model knobs (bps) — applied in returns engine and advisory for backtest
    cost_bps: float = 0.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    model_id_hint: str | None = None
    gates: Sequence[QualityGate] = ()
    auto_promote: bool = False
    deploy_target: str | None = None


def _load_json_safe(path: Path) -> dict[str, object] | None:
    try:
        return _load_json(path)
    except Exception:
        return None


def _resolve_model_id_from_artifacts(cfg: Stage2Config) -> str | None:
    # Priority: explicit hint → model_metrics.json → teacher_meta.json
    if cfg.model_id_hint:
        return cfg.model_id_hint
    mm = _load_json_safe(Path(cfg.out_dir) / "model_metrics.json")
    if mm and isinstance(mm.get("model_id"), str):
        return cast(str, mm["model_id"])
    tm = _load_json_safe(Path(cfg.out_dir) / "teacher_meta.json")
    if tm and isinstance(tm.get("model_id"), str):
        return cast(str, tm["model_id"])
    return None


def _evaluate_gates(
    metrics: Mapping[str, float],
    gates: Sequence[QualityGate],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for g in gates:
        val = metrics.get(g.metric_name)
        if val is None:
            if g.required:
                failures.append(f"missing:{g.metric_name}")
            continue
        ok = True
        cmp = (g.comparison or "gte").lower()
        if cmp == "gte":
            ok = float(val) >= float(g.threshold)
        elif cmp == "lte":
            ok = float(val) <= float(g.threshold)
        elif cmp == "gt":
            ok = float(val) > float(g.threshold)
        elif cmp == "lt":
            ok = float(val) < float(g.threshold)
        else:
            ok = float(val) >= float(g.threshold)
        if not ok and g.required:
            failures.append(f"{g.metric_name}:{val} !{cmp} {g.threshold}")
    return (len(failures) == 0, failures)


def run_promotion_stage2(cfg: Stage2Config) -> dict[str, object]:
    """
    Run walk-forward style validation on the held-out window and compute trading
    metrics, then apply quality gates and optionally promote the model.

    Returns a summary dict and writes stage2_metrics.json in out_dir.

    """
    # Choose engine and run
    engine = build_engine(cfg.engine_mode)
    result: Stage2Result
    if isinstance(engine, object):
        result = engine.run(cfg)
        # If backtest engine is unavailable or skipped, fall back to returns engine
        if result.status == "skipped" and str(cfg.engine_mode) == "backtest":
            result = build_engine("returns").run(cfg)
    else:
        result = Stage2Result(status="skipped", metrics={}, reason="invalid engine")

    # Persist stage2 metrics
    out = Path(cfg.out_dir) / "stage2_metrics.json"
    _ensure_parent_dir(out)
    import json as _json

    out.write_text(_json.dumps(result.metrics, indent=2), encoding="utf-8")

    # Apply gates when the engine produced metrics; otherwise record skip reason
    failures: list[str] = []
    if result.status == "skipped":
        ok = False
        if result.reason:
            failures.append(result.reason)
    else:
        ok, failures = _evaluate_gates(result.metrics, list(cfg.gates))

    model_id = _resolve_model_id_from_artifacts(cfg)

    # Track performance in registry and optionally deploy
    if result.status != "skipped":
        try:
            from typing import Any as _Any

            from ml.core.integration import MLIntegrationManager

            mgr = MLIntegrationManager(
                auto_start_postgres=False,
                auto_migrate=False,
                ensure_healthy=False,
            )
            reg_any = cast(_Any, mgr.model_registry)
            if model_id:
                reg_any.track_performance(model_id, result.metrics)
                if ok and cfg.auto_promote and cfg.deploy_target:
                    reg_any.deploy_model(model_id, cfg.deploy_target)
        except Exception as exc:  # pragma: no cover - best-effort integration
            import logging as _logging

            _logging.getLogger(__name__).debug(
                "Stage2 registry integration failed: %s",
                exc,
                exc_info=True,
            )

    # Emit event (best-effort)
    try:
        from typing import Any as _Any

        from ml.core.integration import MLIntegrationManager

        mgr2 = MLIntegrationManager(
            auto_start_postgres=False,
            auto_migrate=False,
            ensure_healthy=False,
        )
        event_status = EventStatus.SUCCESS if ok else EventStatus.FAILED
        if result.status == "skipped":
            event_status = EventStatus.PARTIAL
        emit_dataset_event(
            cast(_Any, mgr2.data_registry),
            dataset_id="model",
            instrument_id="GLOBAL",
            stage=Stage.SIGNAL_EMITTED,
            source=Source.HISTORICAL,
            run_id=f"stage2_{model_id or 'unknown'}",
            ts_min=0,
            ts_max=0,
            count=1,
            status=event_status,
            metadata={"failures": failures, "model_id": model_id or ""},
            dataset_type="model",
            component="promotions.stage2",
        )
    except Exception:
        pass

    final_status: str
    if result.status == "skipped":
        final_status = "skipped"
    else:
        final_status = "passed" if ok else "failed"

    summary: dict[str, object] = {
        "status": cast(object, final_status),
        "model_id": cast(object, model_id),
        "metrics": cast(object, result.metrics),
        "failures": cast(object, failures),
        "promoted": cast(
            object,
            bool(final_status == "passed" and cfg.auto_promote and cfg.deploy_target and model_id),
        ),
    }

    # Write promotion report consolidating gates and config
    try:
        report = {
            **summary,
            "engine_mode": str(cfg.engine_mode),
            "cost_bps": float(cfg.cost_bps),
            "commission_bps": float(cfg.commission_bps),
            "slippage_bps": float(cfg.slippage_bps),
            "dataset_csv": str(cfg.dataset_csv),
            "data_dir": str(cfg.data_dir),
            "horizon_minutes": int(cfg.horizon_minutes),
            "gates": [
                {
                    "metric": g.metric_name,
                    "threshold": g.threshold,
                    "comparison": g.comparison,
                    "required": g.required,
                }
                for g in cfg.gates
            ],
        }
        (Path(cfg.out_dir) / "promotion_report.json").write_text(
            _json.dumps(report, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    return summary


def run_promotion_stage2_backtest(cfg: Stage2Config) -> dict[str, object]:
    """
    Advisory backtest integration.

    Attempts to use Nautilus Trader BacktestEngine if available; otherwise falls back to
    the returns-based implementation. Cost knobs (commission/slippage) are advisory and
    may be folded into the effective bps when BacktestEngine is unavailable.

    """
    try:
        # Lazy import to avoid hard dependency
        from nautilus_trader.backtest.engine import BacktestEngine
        from nautilus_trader.backtest.engine import BacktestEngineConfig

        _ = BacktestEngine, BacktestEngineConfig
    except Exception:
        # Fall back to returns path
        return run_promotion_stage2(dataclass_replace(cfg, engine_mode="returns"))

    # Engine present; until full parity (instrument/model/features) is wired, reuse
    # the deterministic returns-based computation to produce stable metrics.
    return run_promotion_stage2(dataclass_replace(cfg, engine_mode="returns"))


def dataclass_replace(cfg: Stage2Config, **updates: object) -> Stage2Config:
    """
    Create a new Stage2Config with specified fields updated.
    """
    return Stage2Config(
        out_dir=cast(str, updates.get("out_dir", cfg.out_dir)),
        dataset_csv=cast(str, updates.get("dataset_csv", cfg.dataset_csv)),
        data_dir=cast(str, updates.get("data_dir", cfg.data_dir)),
        horizon_minutes=cast(int, updates.get("horizon_minutes", cfg.horizon_minutes)),
        engine_mode=cast(
            Literal["returns", "backtest"],
            updates.get("engine_mode", cfg.engine_mode),
        ),
        cost_bps=cast(float, updates.get("cost_bps", cfg.cost_bps)),
        commission_bps=cast(float, updates.get("commission_bps", cfg.commission_bps)),
        slippage_bps=cast(float, updates.get("slippage_bps", cfg.slippage_bps)),
        model_id_hint=cast(str | None, updates.get("model_id_hint", cfg.model_id_hint)),
        gates=cast(Sequence[QualityGate], updates.get("gates", cfg.gates)),
        auto_promote=cast(bool, updates.get("auto_promote", cfg.auto_promote)),
        deploy_target=cast(str | None, updates.get("deploy_target", cfg.deploy_target)),
    )


__all__ = [
    "Stage2Config",
    "register_and_promote_model",
    "register_or_refresh_features",
    "run_promotion_stage2",
    "run_promotion_stage2_backtest",
]
