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
MLflow tracking and model registry management for Nautilus ML.

This package provides centralized MLflow utilities for experiment tracking, model
registry management, and monitoring integration specifically designed for financial
machine learning workflows.

"""

from __future__ import annotations

from ml.tracking.mlflow_manager import MLflowManager
from ml.tracking.mlflow_manager import ModelStage
from ml.tracking.model_registry import ModelRegistry
from ml.tracking.monitoring_bridge import MLflowMonitoringBridge


__all__ = [
    "MLflowManager",
    "MLflowMonitoringBridge",
    "ModelRegistry",
    "ModelStage",
]
