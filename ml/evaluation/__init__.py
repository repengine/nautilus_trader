"""
ML Evaluation module for model performance assessment and validation.

This module provides lightweight evaluation metrics and utilities for assessing model
performance in the ML pipeline. All evaluation operations are cold path only and designed
for model validation, reporting, and analysis workflows.

The module follows Universal ML Architecture Patterns:
- Pattern 3: Cold path operations only (no hot path evaluation)
- Pattern 2: Protocol-first design for evaluation interfaces
- Pattern 5: Uses centralized metrics bootstrap for monitoring

Core Capabilities:
- Binary classification metrics (ROC AUC, PR AUC, log loss)
- Lightweight NumPy-only implementations for minimal dependencies
- Designed for integration with model training and validation pipelines
- Compatible with prediction evaluation CLIs and reporting workflows

Performance Characteristics:
- NumPy-only implementations avoid heavy ML library dependencies
- Optimized for batch evaluation of predictions vs ground truth
- Suitable for cross-validation, hyperparameter tuning, and model selection
- Memory efficient for large-scale evaluation tasks

Usage Examples:
    # Basic binary classification evaluation
    from ml.evaluation import binary_logloss, roc_auc, pr_auc

    # Evaluate model predictions
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.8, 0.9, 0.2])

    auc_score = roc_auc(y_true, y_pred)
    pr_score = pr_auc(y_true, y_pred)
    logloss = binary_logloss(y_true, y_pred)

Integration Points:
- ml.cli.evaluate_predictions: CLI for batch prediction evaluation
- ml.api.evaluation: Cold path evaluation API facade (planned)
- ml.training.*: Integration with training pipelines for validation
- ml.registry.*: Model promotion workflows based on evaluation metrics

Non-Goals (Cold Path Only):
- Real-time evaluation in trading actors (use pre-computed metrics)
- Hot path inference monitoring (use ml.common.metrics_bootstrap)
- Heavy statistical analysis (use dedicated analysis tools)
- Multi-class or regression metrics (binary classification focus)

Dependencies:
- numpy: Core numerical operations
- typing: Type annotations and protocols
"""

# ============================================================================
# CORE EVALUATION METRICS
# ============================================================================
# Lightweight binary classification metrics with NumPy-only implementations

from ml.evaluation.metrics import binary_logloss
from ml.evaluation.metrics import pr_auc
from ml.evaluation.metrics import roc_auc


# ============================================================================
# PUBLIC API SURFACE
# ============================================================================

__all__ = [
    # Binary classification metrics (primary API)
    "binary_logloss",
    "pr_auc",
    "roc_auc",
]
