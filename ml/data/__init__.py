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
ML data loading utilities for Nautilus Trader.

This package provides high-level data loading utilities specifically designed for ML
workflows in the cold path (training and research). All loaders integrate seamlessly
with Nautilus Trader's data infrastructure and return Polars DataFrames for efficient ML
processing.

"""

from ml.data.loader import MLDataLoader
from ml.data.loader import load_ml_data


__all__ = [
    "MLDataLoader",
    "load_ml_data",
]
