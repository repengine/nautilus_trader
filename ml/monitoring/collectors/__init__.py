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
Extended metrics collectors for ML system observability.

This package provides specialized Prometheus metrics collectors for different aspects of
ML operations, following Nautilus Trader's architectural patterns.

"""

from __future__ import annotations

from ml.monitoring.collectors.base import BaseMetricsCollector
from ml.monitoring.collectors.data import DataQualityCollector
from ml.monitoring.collectors.features import FeatureEngineeringCollector
from ml.monitoring.collectors.model import ModelLifecycleCollector
from ml.monitoring.collectors.performance import PerformanceDegradationMonitor
from ml.monitoring.collectors.registry import MLMetricsRegistry
from ml.monitoring.collectors.resources import ResourceUtilizationCollector


__all__ = [
    "BaseMetricsCollector",
    "DataQualityCollector",
    "FeatureEngineeringCollector",
    "MLMetricsRegistry",
    "ModelLifecycleCollector",
    "PerformanceDegradationMonitor",
    "ResourceUtilizationCollector",
]
