from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.cli.streaming_training_runner import REQUIRED_MANIFEST_METRICS
from ml.cli.streaming_training_runner import PromotionMetricCheck
from ml.cli.streaming_training_runner import _normalize_metrics
from ml.cli.streaming_training_runner import _parse_metric_check


def test_normalize_metrics_backfills_missing_values(tmp_path: Path) -> None:
    logits = np.array([0.1, -0.2, 0.9, 1.5], dtype=np.float64)
    targets = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    artifact = tmp_path / "cohort_logits.npz"
    np.savez_compressed(artifact, z_val=logits, y_val=targets)

    metrics = _normalize_metrics({}, artifact)

    for metric_name in REQUIRED_MANIFEST_METRICS:
        assert metric_name in metrics
        assert isinstance(metrics[metric_name], float)


def test_promotion_metric_check_parsing_and_evaluation() -> None:
    check_min = _parse_metric_check("pr_auc>=0.55")
    assert check_min.metric == "pr_auc"
    assert check_min.comparator == "ge"
    assert check_min.threshold == 0.55
    assert check_min.evaluate({"pr_auc": 0.60})
    assert not check_min.evaluate({"pr_auc": 0.50})

    check_max = _parse_metric_check("calibration_ece_20<=0.05")
    assert check_max.metric == "calibration_ece_20"
    assert check_max.comparator == "le"
    assert check_max.threshold == 0.05
    assert check_max.evaluate({"calibration_ece_20": 0.04})
    assert not check_max.evaluate({"calibration_ece_20": 0.06})
