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
ML configuration classes for Nautilus Trader.
"""

from ml.config.base import MLActorConfig
from ml.config.base import MLFeatureConfig
from ml.config.base import MLInferenceConfig
from ml.config.base import MLStrategyConfig
from ml.config.base import MLTrainingConfig
from ml.config.constants import FeatureColumns
from ml.config.constants import IndicatorNames
from ml.config.constants import MLConstants
from ml.config.constants import SystemConstants
from ml.config.constants import TechnicalIndicatorPeriods
from ml.config.constants import TimeConstants
from ml.config.lightgbm import LightGBMTrainingConfig
from ml.config.lightgbm import UnifiedLightGBMConfig  # Backward compatibility
from ml.config.shared import AdvancedTrainingConfig
from ml.config.shared import BaseGPUConfig
from ml.config.shared import LightGBMGPUConfig
from ml.config.shared import MLflowConfig
from ml.config.shared import OptunaConfig
from ml.config.shared import XGBoostGPUConfig
from ml.config.xgboost import UnifiedXGBoostConfig  # Backward compatibility
from ml.config.xgboost import XGBoostTrainingConfig


__all__ = [
    "AdvancedTrainingConfig",
    "BaseGPUConfig",
    "FeatureColumns",
    "IndicatorNames",
    "LightGBMGPUConfig",
    "LightGBMTrainingConfig",
    "MLActorConfig",
    "MLConstants",
    "MLFeatureConfig",
    "MLInferenceConfig",
    "MLStrategyConfig",
    "MLTrainingConfig",
    "MLflowConfig",
    "OptunaConfig",
    "SystemConstants",
    "TechnicalIndicatorPeriods",
    "TimeConstants",
    "UnifiedLightGBMConfig",
    "UnifiedXGBoostConfig",
    "XGBoostGPUConfig",
    "XGBoostTrainingConfig",
]
