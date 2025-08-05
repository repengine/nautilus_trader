# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Configuration for XGBoost model training.

This module provides msgspec-based configuration classes for XGBoost training, extending
the base ML configuration with XGBoost-specific parameters.

"""

from __future__ import annotations

from typing import Any

from ml.config.base import MLTrainingConfig
from nautilus_trader.common.config import NonNegativeFloat
from nautilus_trader.common.config import NonNegativeInt
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt


class XGBoostTrainingConfig(MLTrainingConfig, kw_only=True, frozen=True):
    """
    Configuration for XGBoost model training.

    This configuration extends the base MLTrainingConfig with XGBoost-specific
    parameters for tree-based gradient boosting.

    Parameters
    ----------
    n_estimators : PositiveInt, default 100
        Number of gradient boosted trees. Equivalent to number of boosting rounds.
    max_depth : PositiveInt, default 6
        Maximum depth of a tree. Increasing this value makes the model more complex
        and more likely to overfit.
    learning_rate : PositiveFloat, default 0.3
        Boosting learning rate (xgb's "eta"). Step size shrinkage used in update
        to prevent overfitting.
    min_child_weight : NonNegativeFloat, default 1.0
        Minimum sum of instance weight (hessian) needed in a child.
    subsample : PositiveFloat, default 1.0
        Subsample ratio of the training instances. Setting it to 0.5 means that
        XGBoost would randomly sample half of the training data prior to growing trees.
    colsample_bytree : PositiveFloat, default 1.0
        Subsample ratio of columns when constructing each tree.
    colsample_bylevel : PositiveFloat, default 1.0
        Subsample ratio of columns for each level.
    gamma : NonNegativeFloat, default 0.0
        Minimum loss reduction required to make a further partition on a leaf node.
    reg_alpha : NonNegativeFloat, default 0.0
        L1 regularization term on weights.
    reg_lambda : NonNegativeFloat, default 1.0
        L2 regularization term on weights.
    tree_method : str, default "hist"
        Tree construction algorithm. Options: "hist", "gpu_hist", "exact", "approx".
    gpu_id : NonNegativeInt, default 0
        GPU device ID (only relevant when tree_method="gpu_hist").
    objective : str, default "binary:logistic"
        Learning objective. Options: "binary:logistic", "reg:squarederror", "multi:softprob".
    eval_metric : str, default "auc"
        Evaluation metric for validation data.
    enable_shap : bool, default False
        Whether to compute SHAP values for feature importance analysis.
    monotonic_constraints : dict[str, int] | None, optional
        Monotonic constraints for features. Keys are feature names, values are
        -1 (decreasing), 0 (no constraint), or 1 (increasing).
    multi_asset : bool, default False
        Whether to train on multiple assets with cross-sectional features.
    sector_map : dict[str, str] | None, optional
        Mapping from asset symbols to sector names for multi-asset training.
    cross_sectional_features : bool, default True
        Whether to include cross-sectional ranking features for multi-asset models.
    optimize_hyperparams : bool, default False
        Whether to optimize hyperparameters using Optuna.
    n_trials : PositiveInt, default 100
        Number of optimization trials when optimize_hyperparams is True.
    optimization_metric : str, default "sharpe_ratio"
        Metric to optimize during hyperparameter tuning.

    """

    # Core XGBoost parameters
    n_estimators: PositiveInt = 100
    max_depth: PositiveInt = 6
    learning_rate: PositiveFloat = 0.3
    min_child_weight: NonNegativeFloat = 1.0
    subsample: PositiveFloat = 1.0
    colsample_bytree: PositiveFloat = 1.0
    colsample_bylevel: PositiveFloat = 1.0
    gamma: NonNegativeFloat = 0.0
    reg_alpha: NonNegativeFloat = 0.0
    reg_lambda: NonNegativeFloat = 1.0

    # Hardware settings
    tree_method: str = "hist"  # "hist" for CPU, "gpu_hist" for GPU
    gpu_id: NonNegativeInt = 0

    # Training objective
    objective: str = "binary:logistic"
    eval_metric: str = "auc"

    # Advanced features
    enable_shap: bool = False
    monotonic_constraints: dict[str, int] | None = None

    # Multi-asset configuration
    multi_asset: bool = False
    sector_map: dict[str, str] | None = None
    cross_sectional_features: bool = True

    # Hyperparameter optimization
    optimize_hyperparams: bool = False
    n_trials: PositiveInt = 100
    optimization_metric: str = "sharpe_ratio"

    def __post_init__(self) -> None:
        """
        Post-initialization validation.
        """
        # Validate subsample and colsample ratios
        if not (0.0 < self.subsample <= 1.0):
            msg = f"subsample must be in (0.0, 1.0], got {self.subsample}"
            raise ValueError(msg)

        if not (0.0 < self.colsample_bytree <= 1.0):
            msg = f"colsample_bytree must be in (0.0, 1.0], got {self.colsample_bytree}"
            raise ValueError(msg)

        if not (0.0 < self.colsample_bylevel <= 1.0):
            msg = f"colsample_bylevel must be in (0.0, 1.0], got {self.colsample_bylevel}"
            raise ValueError(msg)

        # Validate tree method
        valid_tree_methods = ["hist", "gpu_hist", "exact", "approx"]
        if self.tree_method not in valid_tree_methods:
            msg = f"tree_method must be one of {valid_tree_methods}, got {self.tree_method}"
            raise ValueError(msg)

        # Validate objective
        valid_objectives = ["binary:logistic", "reg:squarederror", "multi:softprob", "reg:logistic"]
        if self.objective not in valid_objectives:
            msg = f"objective must be one of {valid_objectives}, got {self.objective}"
            raise ValueError(msg)

        # Validate multi-asset settings
        if self.multi_asset and self.sector_map is None:
            msg = "sector_map is required when multi_asset=True"
            raise ValueError(msg)

        # Validate monotonic constraints
        if self.monotonic_constraints is not None:
            for feature, constraint in self.monotonic_constraints.items():
                if constraint not in [-1, 0, 1]:
                    msg = (
                        f"monotonic constraint for {feature} must be -1, 0, or 1, got {constraint}"
                    )
                    raise ValueError(msg)

    def get_xgb_params(self) -> dict[str, Any]:
        """
        Get XGBoost parameters as a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary of XGBoost parameters suitable for model initialization.

        """
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "tree_method": self.tree_method,
            "random_state": self.random_seed,
            "n_jobs": -1,
            "verbosity": 0,
        }

        # Add GPU parameters if using GPU
        if self.tree_method == "gpu_hist":
            params["gpu_id"] = self.gpu_id
            params["predictor"] = "gpu_predictor"

        return params
