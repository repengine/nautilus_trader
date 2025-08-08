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
ML actors for real-time inference in Nautilus Trader.
"""

from ml.actors.base import BaseMLInferenceActor
from ml.actors.base import MLSignal
from ml.actors.base import PickleMLInferenceActor
from ml.actors.signal import AdaptiveSignal
from ml.actors.signal import AdaptiveStrategy
from ml.actors.signal import EnsembleStrategy
from ml.actors.signal import ExtremesStrategy
from ml.actors.signal import MLSignalActor
from ml.actors.signal import MLSignalActorConfig
from ml.actors.signal import ModelSwapper
from ml.actors.signal import MomentumStrategy
from ml.actors.signal import OptimizationConfig
from ml.actors.signal import OptimizationLevel
from ml.actors.signal import PerformanceMonitor
from ml.actors.signal import SignalGenerationStrategy
from ml.actors.signal import SignalStrategy
from ml.actors.signal import StrategyConfig
from ml.actors.signal import ThresholdStrategy


__all__ = [
    "AdaptiveSignal",
    "AdaptiveStrategy",
    "BaseMLInferenceActor",
    "EnsembleStrategy",
    "ExtremesStrategy",
    "MLSignal",
    "MLSignalActor",
    "MLSignalActorConfig",
    "ModelSwapper",
    "MomentumStrategy",
    "OptimizationConfig",
    "OptimizationLevel",
    "PerformanceMonitor",
    "PickleMLInferenceActor",
    "SignalGenerationStrategy",
    "SignalStrategy",
    "StrategyConfig",
    "ThresholdStrategy",
]
