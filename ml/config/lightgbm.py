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
Configuration for LightGBM model training.

This module provides msgspec-based configuration classes for LightGBM training,
extending the base ML configuration with LightGBM-specific parameters.

"""

from __future__ import annotations

from typing import Any

from ml.config.base import MLTrainingConfig
from nautilus_trader.common.config import NonNegativeFloat
from nautilus_trader.common.config import NonNegativeInt
from nautilus_trader.common.config import PositiveFloat
from nautilus_trader.common.config import PositiveInt


class LightGBMTrainingConfig(MLTrainingConfig, kw_only=True, frozen=True):
    """
    Configuration for LightGBM model training.

    This configuration extends the base MLTrainingConfig with LightGBM-specific
    parameters for gradient boosting with tree-based learners.

    Parameters
    ----------
    n_estimators : PositiveInt, default 100
        Number of boosting iterations. Equivalent to number of trees to build.
    max_depth : PositiveInt, default 6
        Maximum depth of a tree. A deeper tree might increase accuracy but also
        lead to overfitting. -1 means no limit.
    learning_rate : PositiveFloat, default 0.1
        Boosting learning rate. Shrinkage rate for the contribution of each tree.
    num_leaves : PositiveInt, default 31
        Maximum number of leaves in one tree. Should be less than 2^max_depth.
    min_child_samples : PositiveInt, default 20
        Minimum number of data points in a leaf. This is a crucial parameter
        to control overfitting.
    min_child_weight : NonNegativeFloat, default 1e-3
        Minimum sum of instance weight (hessian) needed in a child (leaf).
    min_split_gain : NonNegativeFloat, default 0.0
        Minimum loss reduction required to make a further partition on a leaf node.
    subsample : PositiveFloat, default 1.0
        Subsample ratio of the training instances. 0.5 means half of training data.
        Ranges from (0.0, 1.0]. Used to prevent overfitting.
    subsample_freq : NonNegativeInt, default 0
        Frequency of subsample. <=0 means no enable. If > 0, will perform subsample
        at every k iteration.
    colsample_bytree : PositiveFloat, default 1.0
        Subsample ratio of columns when constructing each tree. Ranges from (0.0, 1.0].
    reg_alpha : NonNegativeFloat, default 0.0
        L1 regularization term on weights. Helps prevent overfitting.
    reg_lambda : NonNegativeFloat, default 0.0
        L2 regularization term on weights. Helps prevent overfitting.
    scale_pos_weight : PositiveFloat, default 1.0
        Balance of positive and negative weights. Used for imbalanced datasets.
    objective : str, default "regression"
        Learning objective. Options: "regression", "binary", "multiclass", "lambdarank".
    metric : str, default "rmse"
        Metric for evaluation. Auto-selected based on objective if None.
    boosting_type : str, default "gbdt"
        Boosting type. Options: "gbdt", "dart", "goss", "rf".
    early_stopping_rounds : NonNegativeInt, default 10
        Activates early stopping. Training will stop if metric doesn't improve
        for this many rounds. 0 disables early stopping.
    n_jobs : int, default -1
        Number of parallel threads. -1 means use all available cores.
    random_state : NonNegativeInt, default 42
        Random seed for reproducible results.
    verbosity : int, default -1
        Controls the level of LightGBM's verbosity. -1 = silent.
    force_col_wise : bool, default False
        Force column-wise construction of histograms.
    force_row_wise : bool, default False
        Force row-wise construction of histograms.

    """

    # Core boosting parameters
    n_estimators: PositiveInt = 100
    max_depth: PositiveInt = 6
    learning_rate: PositiveFloat = 0.1
    num_leaves: PositiveInt = 31
    min_child_samples: PositiveInt = 20
    min_child_weight: NonNegativeFloat = 1e-3
    min_split_gain: NonNegativeFloat = 0.0

    # Sampling parameters
    subsample: PositiveFloat = 1.0
    subsample_freq: NonNegativeInt = 0
    colsample_bytree: PositiveFloat = 1.0

    # Regularization parameters
    reg_alpha: NonNegativeFloat = 0.0
    reg_lambda: NonNegativeFloat = 0.0
    scale_pos_weight: PositiveFloat = 1.0

    # Learning objective
    objective: str = "regression"
    metric: str = "rmse"
    boosting_type: str = "gbdt"

    # Training control
    early_stopping_rounds: NonNegativeInt = 10
    n_jobs: int = -1
    random_state: NonNegativeInt = 42
    verbosity: int = -1

    # Memory optimization
    force_col_wise: bool = False
    force_row_wise: bool = False

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.
        """
        # Validate LightGBM-specific parameters
        if not (0.0 < self.learning_rate <= 1.0):
            msg = f"learning_rate must be in (0.0, 1.0], got {self.learning_rate}"
            raise ValueError(msg)

        if not (0.0 < self.subsample <= 1.0):
            msg = f"subsample must be in (0.0, 1.0], got {self.subsample}"
            raise ValueError(msg)

        if not (0.0 < self.colsample_bytree <= 1.0):
            msg = f"colsample_bytree must be in (0.0, 1.0], got {self.colsample_bytree}"
            raise ValueError(msg)

        # Check num_leaves vs max_depth relationship
        if self.max_depth > 0 and self.num_leaves >= 2**self.max_depth:
            msg = (
                f"num_leaves ({self.num_leaves}) should be less than 2^max_depth "
                f"({2**self.max_depth}) to avoid overfitting"
            )
            # This is a warning rather than an error since LightGBM allows it
            import warnings

            warnings.warn(msg, UserWarning, stacklevel=2)

        # Validate objective
        valid_objectives = ["regression", "binary", "multiclass", "lambdarank"]
        if self.objective not in valid_objectives:
            msg = f"objective must be one of {valid_objectives}, got {self.objective}"
            raise ValueError(msg)

        # Validate boosting_type
        valid_boosting = ["gbdt", "dart", "goss", "rf"]
        if self.boosting_type not in valid_boosting:
            msg = f"boosting_type must be one of {valid_boosting}, got {self.boosting_type}"
            raise ValueError(msg)

        # Validate metric for objective
        if self.objective == "binary" and self.metric not in [
            "auc",
            "binary_logloss",
            "binary_error",
        ]:
            import warnings

            warnings.warn(
                f"For binary classification, consider using metric='auc' or 'binary_logloss' "
                f"instead of '{self.metric}'",
                UserWarning,
                stacklevel=2,
            )

        # Validate mutual exclusivity of memory optimization flags
        if self.force_col_wise and self.force_row_wise:
            msg = "force_col_wise and force_row_wise cannot both be True"
            raise ValueError(msg)

    def get_lgb_params(self) -> dict[str, Any]:
        """
        Convert configuration to LightGBM parameter dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary of LightGBM parameters.

        """
        return {
            # Core boosting parameters
            "num_iterations": self.n_estimators,
            "max_depth": self.max_depth if self.max_depth > 0 else -1,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "min_child_weight": self.min_child_weight,
            "min_split_gain": self.min_split_gain,
            # Sampling parameters
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "colsample_bytree": self.colsample_bytree,
            # Regularization parameters
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "scale_pos_weight": self.scale_pos_weight,
            # Learning objective
            "objective": self.objective,
            "metric": self.metric,
            "boosting_type": self.boosting_type,
            # Training control
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbosity": self.verbosity,
            # Memory optimization
            "force_col_wise": self.force_col_wise,
            "force_row_wise": self.force_row_wise,
        }

    def validate_for_objective(self, objective: str) -> None:
        """
        Validate configuration parameters for specific objective.

        Parameters
        ----------
        objective : str
            The objective function to validate against.

        Raises
        ------
        ValueError
            If configuration is invalid for the given objective.

        """
        if objective == "binary" and self.scale_pos_weight == 1.0:
            import warnings

            warnings.warn(
                "For imbalanced binary classification, consider setting scale_pos_weight "
                "to sum(negative_instances) / sum(positive_instances)",
                UserWarning,
                stacklevel=2,
            )

        if objective in ["multiclass", "lambdarank"] and self.scale_pos_weight != 1.0:
            import warnings

            warnings.warn(
                f"scale_pos_weight is typically not used with {objective} objective",
                UserWarning,
                stacklevel=2,
            )


# Explicit exports
__all__ = [
    "LightGBMTrainingConfig",
]
