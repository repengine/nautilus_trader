#!/usr/bin/env python3

"""
Statistical utilities for model comparison and A/B testing.

This module provides statistical tests for model performance comparison, extracted from
the legacy registry for use in the new architecture.

"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.config.base import StatsConfig


def welch_t_test(
    sample_a: np.ndarray[Any, np.dtype[np.float64]],
    sample_b: np.ndarray[Any, np.dtype[np.float64]],
    significance_level: float | None = None,
) -> dict[str, Any]:
    """
    Perform Welch's t-test for comparing two samples with unequal variances.

    Parameters
    ----------
    sample_a : np.ndarray
        First sample (control group)
    sample_b : np.ndarray
        Second sample (treatment group)
    significance_level : float, default 0.05
        Statistical significance level

    Returns
    -------
    dict[str, Any]
        Test results including t-statistic, p-value approximation, and significance

    """
    if len(sample_a) < 2 or len(sample_b) < 2:
        return {
            "t_statistic": 0.0,
            "p_value_approx": 1.0,
            "statistically_significant": False,
            "error": "Insufficient samples for test",
        }

    mean_a = np.mean(sample_a)
    mean_b = np.mean(sample_b)
    var_a = np.var(sample_a, ddof=1)
    var_b = np.var(sample_b, ddof=1)

    if var_a <= 0 or var_b <= 0:
        return {
            "t_statistic": 0.0,
            "p_value_approx": 1.0,
            "statistically_significant": False,
            "error": "Zero variance in samples",
        }

    # Calculate pooled standard error
    pooled_se = np.sqrt(var_a / len(sample_a) + var_b / len(sample_b))
    t_stat = (mean_b - mean_a) / pooled_se

    # Calculate degrees of freedom (Welch's formula)
    df = (var_a / len(sample_a) + var_b / len(sample_b)) ** 2 / (
        (var_a / len(sample_a)) ** 2 / (len(sample_a) - 1)
        + (var_b / len(sample_b)) ** 2 / (len(sample_b) - 1)
    )

    # Determine critical value
    stats = StatsConfig()
    alpha = (
        significance_level if significance_level is not None else float(stats.significance_level)
    )
    critical_value = float(stats.z_alpha_default) if alpha == 0.05 else float(stats.z_alpha_default)
    if df < stats.small_sample_df_threshold:
        critical_value = float(stats.conservative_critical_value)

    # Approximate p-value
    p_value_approx = 2 * (1 - 0.5 * (1 + np.tanh(abs(t_stat) / np.sqrt(2))))

    return {
        "t_statistic": float(t_stat),
        "degrees_of_freedom": float(df),
        "p_value_approx": float(p_value_approx),
        "critical_value": float(critical_value),
        "statistically_significant": bool(abs(t_stat) > critical_value),
        "mean_a": float(mean_a),
        "mean_b": float(mean_b),
        "difference": float(mean_b - mean_a),
        "relative_improvement": float((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0.0,
    }


def compare_models(
    models: list[dict[str, Any]],
    metric_name: str,
    baseline_index: int = 0,
) -> dict[str, Any]:
    """
    Compare multiple models on a specific metric.

    Parameters
    ----------
    models : list[dict[str, Any]]
        List of model info dicts with 'model_id' and 'metrics' keys
    metric_name : str
        Name of the metric to compare
    baseline_index : int, default 0
        Index of the baseline model to compare against

    Returns
    -------
    dict[str, Any]
        Comparison results with rankings and relative improvements

    """
    if not models:
        return {"error": "No models provided"}

    if baseline_index >= len(models):
        return {"error": f"Invalid baseline index {baseline_index}"}

    # Extract metrics
    model_metrics = []
    for model in models:
        metrics = model.get("metrics", {})
        value = metrics.get(metric_name)
        model_metrics.append(
            {
                "model_id": model.get("model_id", "unknown"),
                "value": value,
                "metrics": metrics,
            },
        )

    # Sort by metric value (descending)
    model_metrics.sort(
        key=lambda x: x["value"] if x["value"] is not None else -float("inf"),
        reverse=True,
    )

    # Calculate relative improvements
    baseline_value = model_metrics[baseline_index]["value"]

    comparison_results = {
        "metric_name": metric_name,
        "baseline_model": model_metrics[baseline_index]["model_id"],
        "baseline_value": baseline_value,
        "models": [],
    }

    for i, model_metric in enumerate(model_metrics):
        result = {
            "rank": i + 1,
            "model_id": model_metric["model_id"],
            "value": model_metric["value"],
        }

        if model_metric["value"] is not None and baseline_value is not None and baseline_value != 0:
            result["relative_improvement"] = (
                (model_metric["value"] - baseline_value) / baseline_value * 100
            )
            result["improvement_from_baseline"] = model_metric["value"] - baseline_value

        comparison_results["models"].append(result)

    # Find winner
    if model_metrics and model_metrics[0]["value"] is not None:
        comparison_results["winner"] = model_metrics[0]["model_id"]
        comparison_results["winner_value"] = model_metrics[0]["value"]

    return comparison_results


def calculate_sample_size(
    effect_size: float,
    power: float | None = None,
    significance_level: float | None = None,
) -> int:
    """
    Calculate required sample size for A/B test.

    Parameters
    ----------
    effect_size : float
        Expected effect size (Cohen's d)
    power : float, default 0.8
        Statistical power (1 - Type II error rate)
    significance_level : float, default 0.05
        Type I error rate

    Returns
    -------
    int
        Required sample size per group

    """
    if effect_size == 0:
        return 100000  # Very large number for zero effect

    # Approximations for z-scores with config defaults
    stats = StatsConfig()
    alpha = (
        significance_level if significance_level is not None else float(stats.significance_level)
    )
    desired_power = power if power is not None else float(stats.power)
    z_alpha_map = {0.01: 2.576, 0.05: 1.96, 0.10: 1.645}
    z_beta_map = {0.80: 0.84, 0.85: 1.04, 0.90: 1.28, 0.95: 1.645, 0.99: 2.33}

    z_alpha = z_alpha_map.get(alpha, float(stats.z_alpha_default))
    z_beta = z_beta_map.get(desired_power, 0.84 + (desired_power - 0.8) * 4)  # Linear interpolation

    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return max(int(np.ceil(n)), 30)  # Minimum 30 samples
