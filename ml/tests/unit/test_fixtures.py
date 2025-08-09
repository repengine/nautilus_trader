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
Common test fixtures and mocks for ML tests.
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt


# Mock sklearn module for tests

# Random generator for numpy 2.0 compatibility
rng = np.random.default_rng(42)


class MockSklearnModule:
    """
    Mock sklearn module.
    """

    class preprocessing:
        """
        Mock preprocessing module.
        """

        class StandardScaler:
            """
            Mock StandardScaler.
            """

            def __init__(self) -> None:
                self.mean_ = None
                self.scale_ = None
                self.n_features_in_: int | None = None

            def fit(self, X: npt.NDArray[np.float64]) -> "MockSklearnModule.preprocessing.StandardScaler":
                """
                Mock fit method.
                """
                self.mean_ = np.mean(X, axis=0)
                self.scale_ = np.std(X, axis=0)
                # Fix indexed assignment error by ensuring scale_ is not None
                if self.scale_ is not None:
                    self.scale_[self.scale_ == 0] = 1.0  # Avoid division by zero
                self.n_features_in_ = X.shape[1]
                return self

            def transform(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                """
                Mock transform method.
                """
                if self.mean_ is None:
                    raise ValueError("Scaler not fitted")
                return (X - self.mean_) / self.scale_

            def fit_transform(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
                """
                Mock fit_transform method.
                """
                self.fit(X)
                return self.transform(X)

    class metrics:
        """
        Mock metrics module.
        """

        @staticmethod
        def accuracy_score(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
            """
            Mock accuracy score.
            """
            if len(y_true) == 0:
                return 0.0
            return float(np.mean(y_true == y_pred))

        @staticmethod
        def precision_score(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64], **kwargs: Any) -> float:
            """
            Mock precision score.
            """
            if len(y_true) == 0:
                return 0.0
            return 0.85

        @staticmethod
        def recall_score(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64], **kwargs: Any) -> float:
            """
            Mock recall score.
            """
            if len(y_true) == 0:
                return 0.0
            return 0.80

        @staticmethod
        def f1_score(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64], **kwargs: Any) -> float:
            """
            Mock f1 score.
            """
            if len(y_true) == 0:
                return 0.0
            return 0.82


# Mock polars module
class MockPolarsModule:
    """
    Mock Polars module.
    """

    class DataFrame:
        """
        Mock DataFrame class.
        """

        def __init__(self, data: Any, schema: list[str] | None = None) -> None:
            self.data = data
            if schema is not None:
                # When schema is provided, data is usually a numpy array
                self._columns = schema
                if hasattr(data, "shape"):
                    self._len = data.shape[0] if data.shape else 0
                else:
                    self._len = len(data) if data else 0
            elif isinstance(data, list):
                # Handle list of dicts (rows)
                if data and isinstance(data[0], dict):
                    self._columns = list(data[0].keys())
                    self._len = len(data)
                else:
                    self._columns = []
                    self._len = 0
            elif isinstance(data, dict):
                # Handle dict of lists (columns)
                self._columns = list(data.keys())
                self._len = len(next(iter(data.values()))) if data else 0
            else:
                self._columns = []
                self._len = 0

        @property
        def columns(self) -> list[str]:
            return self._columns

        def __len__(self) -> int:
            return self._len

        def drop(
            self,
            column: str | list[str],
            strict: bool = False,
        ) -> "MockPolarsModule.DataFrame":
            """
            Mock drop method.
            """
            if isinstance(self.data, dict):
                new_data = self.data.copy()
                columns_to_drop = [column] if isinstance(column, str) else column
                for col in columns_to_drop:
                    if col in new_data:
                        del new_data[col]
                return MockPolarsModule.DataFrame(new_data)
            else:
                # When data is not a dict, keep same schema but remove dropped columns
                new_columns = (
                    [col for col in self._columns if col != column]
                    if isinstance(column, str)
                    else [col for col in self._columns if col not in column]
                )
                new_df = MockPolarsModule.DataFrame({}, schema=new_columns)
                new_df._len = self._len
                # Copy cached data if it exists
                if hasattr(self, "_cached_data"):
                    new_df._cached_data = {
                        k: v
                        for k, v in self._cached_data.items()
                        if k != column and (isinstance(column, str) or k not in column)
                    }
                return new_df

        def with_columns(self, exprs: Any) -> "MockPolarsModule.DataFrame":
            """
            Mock with_columns method.
            """
            return self

        def select(self, columns: list[str]) -> "MockPolarsModule.DataFrame":
            """
            Mock select method.
            """
            return self

        def to_numpy(self) -> npt.NDArray[np.float64]:
            """
            Mock to_numpy method.
            """
            if hasattr(self.data, "shape"):  # Already a numpy array
                return np.asarray(self.data)
            if self._columns and self._len > 0:
                # Generate appropriate data for the DataFrame
                return rng.standard_normal((self._len, len(self._columns)))
            elif self._len > 0:
                # If no columns but has length, return 2D array with 1 column
                return rng.standard_normal((self._len, 1))
            return np.array([]).reshape(0, len(self._columns) if self._columns else 1)

        def shift(self, n: int) -> "MockPolarsModule.DataFrame":
            """
            Mock shift method.
            """
            return self

        def cast(self, dtype: Any) -> "MockPolarsModule.DataFrame":
            """
            Mock cast method.
            """
            return self

        def with_row_count(self, name: str) -> "MockPolarsModule.DataFrame":
            """
            Mock with_row_count method.
            """
            return self

        def __getitem__(
            self,
            key: str | int | slice,
        ) -> "MockPolarsModule.Series | MockPolarsModule.DataFrame":
            """
            Mock getitem method.
            """
            if isinstance(key, str):
                # Return a Series with the actual column data if it exists
                if isinstance(self.data, dict) and key in self.data:
                    return MockPolarsModule.Series(self.data[key])

                # Generate consistent OHLC data by creating it once and caching
                if not hasattr(self, "_cached_data"):
                    self._cached_data = {}
                    # Generate consistent OHLC bars
                    base_prices = np.full(self._len, 100.0) + rng.uniform(-5, 5, self._len)
                    high_offset = rng.uniform(0, 10, self._len)
                    low_offset = rng.uniform(0, 10, self._len)

                    self._cached_data["open"] = base_prices
                    self._cached_data["close"] = base_prices + rng.uniform(-2, 2, self._len)
                    self._cached_data["high"] = np.maximum(
                        np.maximum(self._cached_data["open"], self._cached_data["close"]),
                        base_prices + high_offset,
                    )
                    self._cached_data["low"] = np.minimum(
                        np.minimum(self._cached_data["open"], self._cached_data["close"]),
                        base_prices - low_offset,
                    )
                    self._cached_data["volume"] = rng.uniform(1000, 10000, self._len)

                if key in self._cached_data:
                    return MockPolarsModule.Series(self._cached_data[key])
                elif key == "volume":
                    # Volume should always be positive
                    return MockPolarsModule.Series(rng.uniform(1000, 10000, self._len))
                else:
                    # Other columns can use normal distribution
                    return MockPolarsModule.Series(rng.standard_normal(self._len))
            elif isinstance(key, slice):
                # Handle slicing - return a new DataFrame with sliced data
                start = key.start or 0
                stop = key.stop or self._len
                new_len = max(0, stop - start)
                # Create a new DataFrame with the sliced length
                new_df = MockPolarsModule.DataFrame({}, schema=self._columns)
                new_df._len = new_len
                return new_df
            elif isinstance(key, int):
                # Handle single row access
                return MockPolarsModule.DataFrame({}, schema=self._columns)
            return self

    class Series:
        """
        Mock Series class.
        """

        def __init__(self, data: Any) -> None:
            self.data = data

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, key: int | slice) -> Any:
            """
            Mock getitem for slicing.
            """
            if isinstance(key, slice):
                return MockPolarsModule.Series(self.data[key])
            return self.data[key]

        def to_numpy(self) -> npt.NDArray[np.float64]:
            """
            Mock to_numpy method.
            """
            return np.array(self.data)

        def shift(self, n: int) -> "MockPolarsModule.Series":
            """
            Mock shift method.
            """
            # Create a new Series with shifted data that supports further operations
            shifted_data = self.data
            shifted_series = MockPolarsModule.Series(shifted_data)
            # Add support for division operations that return chainable series
            setattr(shifted_series, "__truediv__", lambda other: self.__truediv__(other))
            setattr(shifted_series, "__gt__", lambda other: self.__gt__(other))
            return shifted_series

        def cast(self, dtype: Any) -> "MockPolarsModule.Series":
            """
            Mock cast method.
            """
            return self

        def __gt__(self, other: Any) -> "MockPolarsModule.Series":
            """
            Mock greater than comparison.
            """
            # Return a Series with boolean-like data that can be cast
            result_data = (
                rng.integers(0, 2, len(self.data))
                if hasattr(self.data, "__len__")
                else np.array([1])
            )
            comparison_series = MockPolarsModule.Series(result_data)
            # Add cast method to the comparison result
            setattr(
                comparison_series,
                "cast",
                lambda dtype: MockPolarsModule.Series(result_data.astype(np.int32)),
            )
            return comparison_series

        def __truediv__(self, other: Any) -> "MockPolarsModule.Series":
            """
            Mock division.
            """
            result_data = self.data if hasattr(self.data, "__len__") else np.array([1.0])
            division_series = MockPolarsModule.Series(result_data)
            # Support further operations on division result
            setattr(division_series, "__sub__", lambda x: MockPolarsModule.Series(result_data))
            return division_series

        def __sub__(self, other: Any) -> "MockPolarsModule.Series":
            """
            Mock subtraction.
            """
            result_data = self.data if hasattr(self.data, "__len__") else np.array([1.0])
            subtraction_series = MockPolarsModule.Series(result_data)
            # Support further operations like comparison
            setattr(subtraction_series, "__gt__", lambda x: self.__gt__(x))
            return subtraction_series

        def rank(self) -> "MockPolarsModule.Series":
            """
            Mock rank method.
            """
            return self

        def over(self, by: str) -> "MockPolarsModule.Series":
            """
            Mock over method.
            """
            return self

        def sum(self) -> float:
            """
            Mock sum method.
            """
            return float(np.sum(self.data))

        def mean(self) -> float:
            """
            Mock mean method.
            """
            return float(np.mean(self.data))

        def std(self, ddof: int = 1) -> float:
            """
            Mock std method.
            """
            return float(np.std(self.data, ddof=ddof))

    class Int32:
        """
        Mock Int32 type.
        """

    @staticmethod
    def concat(dfs: list[Any]) -> "MockPolarsModule.DataFrame":
        """
        Mock concat function.
        """
        if not dfs:
            return MockPolarsModule.DataFrame({})
        # Simple concatenation - just return first df for testing
        first_df = dfs[0]
        if isinstance(first_df, MockPolarsModule.DataFrame):
            return first_df
        else:
            # If not a MockPolarsModule.DataFrame, wrap it
            return MockPolarsModule.DataFrame(first_df)

    @staticmethod
    def lit(value: Any) -> Any:
        """
        Mock lit function.
        """
        return MagicMock(alias=lambda x: value)

    @staticmethod
    def col(name: str) -> Any:
        """
        Mock col function.
        """
        mock_col = MagicMock()
        mock_col.rank.return_value = mock_col
        mock_col.over.return_value = mock_col
        mock_col.alias.return_value = mock_col
        mock_col.mean.return_value = mock_col
        mock_col.__sub__.return_value = mock_col
        return mock_col


# Create singleton instances
mock_sklearn = MockSklearnModule()
mock_polars = MockPolarsModule()
