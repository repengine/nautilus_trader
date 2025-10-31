#!/usr/bin/env python3
r"""
Promotion gate for streaming teacher cohorts.

This command evaluates validation metrics computed from a logits NPZ artifact
(``--teacher-npz``) and/or a streaming manifest JSON file (``--manifest``).
Thresholds are sourced from :class:`ml.config.streaming_pipeline.StreamingPromotionConfig`,
which may be configured via environment variables (``ML_STREAMING_PROMOTE_*``) or
overridden directly on the CLI. Exits with code ``0`` when all gates pass and
``2`` otherwise.

Example:
    poetry run python -m ml.cli.promote_model_if_metrics_pass \\
        --manifest /path/to/plan_manifest.json \\
        --teacher-npz /path/to/cohort_logits.npz \\
        --min-auc 0.56 \\
        --min-pr-auc-multiple 1.5 \\
        --min-slippage-adjusted-sharpe 0.1 \\
        --max-calibration-drift 0.02
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from ml.config.streaming_pipeline import StreamingPromotionConfig


logger = logging.getLogger(__name__)


def _load_npz_metrics(path: Path) -> dict[str, float]:
    """Return validation metrics derived from a logits NPZ artifact."""
    if not path.exists():
        raise FileNotFoundError(f"teacher_npz not found at {path}")
    with np.load(path, allow_pickle=False) as payload:
        try:
            logits = np.asarray(payload["q_val"], dtype=np.float64).reshape(-1)
            labels = np.asarray(payload["y_val_true"], dtype=np.int_).reshape(-1)
        except KeyError as exc:
            raise ValueError("teacher_npz missing required arrays 'q_val' or 'y_val_true'") from exc

    if logits.size == 0 or labels.size == 0:
        raise ValueError("teacher_npz contains empty validation arrays")

    probabilities = np.clip(logits, 1e-6, 1.0 - 1e-6)
    auc = float(roc_auc_score(labels, probabilities))
    pr = float(average_precision_score(labels, probabilities))
    logloss = float(log_loss(labels, probabilities))
    brier = float(brier_score_loss(labels, probabilities))
    prevalence = float(labels.mean())
    pr_multiple = pr / prevalence if prevalence > 0.0 else 0.0

    return {
        "roc_auc": auc,
        "pr_auc": pr,
        "pr_auc_multiple": pr_multiple,
        "log_loss": logloss,
        "brier_score": brier,
        "positive_rate": prevalence,
    }


def _load_manifest_metrics(path: Path) -> dict[str, float]:
    """Return metrics extracted from a streaming cohort manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("manifest root must be a JSON object")
    cohort = payload.get("cohort_run")
    if not isinstance(cohort, Mapping):
        raise ValueError("manifest missing 'cohort_run' block")
    metrics_block = cohort.get("metrics")
    if not isinstance(metrics_block, Mapping):
        raise ValueError("manifest missing 'cohort_run.metrics'")

    metrics: dict[str, float] = {}
    for key, value in metrics_block.items():
        try:
            metrics[str(key).lower()] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def _augment_metrics(metrics: dict[str, float]) -> None:
    """Insert derived metrics used for absolute-value promotion gates."""
    drift = metrics.get("stability_calibration_drift")
    if drift is not None and "stability_calibration_drift_abs" not in metrics:
        try:
            metrics["stability_calibration_drift_abs"] = abs(float(drift))
        except (TypeError, ValueError):
            logger.debug(
                "failed_to_compute_calibration_drift_abs",
                extra={"value": drift},
                exc_info=True,
            )


def _build_promotion_config(args: argparse.Namespace) -> StreamingPromotionConfig:
    """Return promotion thresholds resolved from environment and CLI overrides."""
    base = StreamingPromotionConfig.from_env()
    return StreamingPromotionConfig(
        min_roc_auc=args.min_auc if args.min_auc is not None else base.min_roc_auc,
        min_pr_auc_multiple=(
            args.min_pr_auc_multiple if args.min_pr_auc_multiple is not None else base.min_pr_auc_multiple
        ),
        max_log_loss=args.max_log_loss if args.max_log_loss is not None else base.max_log_loss,
        min_slippage_adjusted_sharpe=(
            args.min_slippage_adjusted_sharpe
            if args.min_slippage_adjusted_sharpe is not None
            else base.min_slippage_adjusted_sharpe
        ),
        min_hit_rate=args.min_hit_rate if args.min_hit_rate is not None else base.min_hit_rate,
        max_turnover=args.max_turnover if args.max_turnover is not None else base.max_turnover,
        max_drawdown=args.max_drawdown if args.max_drawdown is not None else base.max_drawdown,
        max_ks_statistic=args.max_ks_statistic if args.max_ks_statistic is not None else base.max_ks_statistic,
        max_calibration_drift=(
            args.max_calibration_drift if args.max_calibration_drift is not None else base.max_calibration_drift
        ),
    )


def _extract_metric(metrics: Mapping[str, float], name: str, *, absolute: bool = False) -> float | None:
    """Return the metric value if available and convertible to float."""
    value = metrics.get(name)
    if value is None:
        return None
    try:
        observed = float(value)
    except (TypeError, ValueError):
        return None
    return abs(observed) if absolute else observed


def _evaluate_promotion(
    metrics: Mapping[str, float],
    config: StreamingPromotionConfig,
) -> tuple[bool, list[str]]:
    """Evaluate configured promotion thresholds against the provided metrics."""
    failures: list[str] = []

    if config.min_roc_auc is not None:
        observed_auc = _extract_metric(metrics, "roc_auc")
        if observed_auc is None:
            failures.append("missing metric 'roc_auc'")
        elif observed_auc < float(config.min_roc_auc):
            failures.append(
                f"roc_auc {observed_auc:.4f} below threshold {float(config.min_roc_auc):.4f}",
            )

    for metric, comparator, threshold, absolute in config.metric_rules():
        observed = _extract_metric(metrics, metric, absolute=absolute)
        if observed is None:
            failures.append(f"missing metric '{metric}'")
            continue
        if comparator == "ge" and observed < threshold:
            failures.append(f"{metric} {observed:.4f} below threshold {threshold:.4f}")
        elif comparator == "le" and observed > threshold:
            failures.append(f"{metric} {observed:.4f} above threshold {threshold:.4f}")

    return len(failures) == 0, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promotion gate for TFT streaming cohorts.")
    parser.add_argument(
        "--teacher-npz",
        type=Path,
        default=None,
        help="Optional logits artifact (teacher_preds.npz) used to compute core metrics.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional streaming manifest JSON containing persisted metrics.",
    )
    parser.add_argument(
        "--min-auc",
        type=float,
        default=None,
        help="Minimum ROC-AUC required for promotion (overrides ML_STREAMING_PROMOTE_MIN_ROC_AUC).",
    )
    parser.add_argument(
        "--min-pr-auc-multiple",
        type=float,
        default=None,
        help="Minimum PR-AUC multiple relative to prevalence (overrides ML_STREAMING_PROMOTE_MIN_PR_AUC_MULTIPLE).",
    )
    parser.add_argument(
        "--max-log-loss",
        type=float,
        default=None,
        help="Maximum acceptable log loss (overrides ML_STREAMING_PROMOTE_MAX_LOG_LOSS).",
    )
    parser.add_argument(
        "--min-slippage-adjusted-sharpe",
        type=float,
        default=None,
        help="Minimum slippage-adjusted Sharpe ratio (overrides ML_STREAMING_PROMOTE_MIN_SLIPPAGE_SHARPE).",
    )
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=None,
        help="Minimum hit rate required for promotion (overrides ML_STREAMING_PROMOTE_MIN_HIT_RATE).",
    )
    parser.add_argument(
        "--max-turnover",
        type=float,
        default=None,
        help="Maximum allowable ensemble turnover (overrides ML_STREAMING_PROMOTE_MAX_TURNOVER).",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=None,
        help="Maximum allowable drawdown magnitude (overrides ML_STREAMING_PROMOTE_MAX_DRAWDOWN).",
    )
    parser.add_argument(
        "--max-ks-statistic",
        type=float,
        default=None,
        help="Maximum allowable KS statistic for stability (overrides ML_STREAMING_PROMOTE_MAX_KS_STATISTIC).",
    )
    parser.add_argument(
        "--max-calibration-drift",
        type=float,
        default=None,
        help="Maximum absolute calibration drift (overrides ML_STREAMING_PROMOTE_MAX_CALIBRATION_DRIFT).",
    )
    args = parser.parse_args(argv)

    if args.teacher_npz is None and args.manifest is None:
        parser.error("at least one of --teacher-npz or --manifest must be provided")

    metrics: dict[str, float] = {}
    if args.teacher_npz is not None:
        try:
            metrics.update(_load_npz_metrics(args.teacher_npz))
        except Exception as exc:  # pragma: no cover - fails fast in tests
            logger.error("failed to load teacher_npz metrics", extra={"path": str(args.teacher_npz)}, exc_info=True)
            print(str(exc), file=sys.stderr)
            return 2

    if args.manifest is not None:
        try:
            manifest_metrics = _load_manifest_metrics(args.manifest)
        except Exception as exc:  # pragma: no cover - fails fast in tests
            logger.error("failed to load manifest metrics", extra={"path": str(args.manifest)}, exc_info=True)
            print(str(exc), file=sys.stderr)
            return 2
        metrics.update(manifest_metrics)

    _augment_metrics(metrics)
    if not metrics:
        print("no metrics available for promotion evaluation", file=sys.stderr)
        return 2

    ordered_metrics = {key: metrics[key] for key in sorted(metrics)}
    print(ordered_metrics)

    config = _build_promotion_config(args)
    success, failures = _evaluate_promotion(ordered_metrics, config)
    if success:
        print("PROMOTE: pass gates")
        return 0

    for failure in failures:
        print(f"PROMOTE: {failure}", file=sys.stderr)
    print("PROMOTE: failed gates", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
