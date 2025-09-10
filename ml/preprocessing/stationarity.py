"""
Advanced preprocessing for time series stationarity and cross-validation.

Implements fractional differencing, purged cross-validation, and other techniques from
"Advances in Financial Machine Learning" by López de Prado.

"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import numpy.typing as npt


try:
    from numba import jit as _numba_jit
except Exception:  # pragma: no cover - numba optional
    _numba_jit = None

F = TypeVar("F", bound=Callable[..., Any])


def jit_typed(*jit_args: Any, **jit_kwargs: Any) -> Callable[[F], F]:
    """
    Typed wrapper around numba.jit that degrades to identity when unavailable.

    This preserves function type for type checkers while applying JIT at runtime.

    """

    def decorator(func: F) -> F:
        if _numba_jit is None:
            return func
        # Apply numba.jit with the provided args/kwargs
        compiled = _numba_jit(*jit_args, **jit_kwargs)(func)
        return cast(F, compiled)

    return decorator


if TYPE_CHECKING:
    pass


class StationarityTransformer:
    """
    Advanced stationarity transformations for financial time series.

    Implements fractional differencing to achieve stationarity while preserving memory,
    as described in López de Prado (2018).

    """

    def __init__(
        self,
        method: str = "fractional",
        d: float = 0.5,
        threshold: float = 1e-3,
        max_lags: int | None = None,
    ):
        """
        Initialize stationarity transformer.

        Parameters
        ----------
        method : str
            Transformation method ('fractional', 'standard', 'auto')
        d : float
            Differencing order for fractional method
        threshold : float
            Minimum weight magnitude to retain
        max_lags : int | None
            Maximum number of lags for weights

        """
        self.method = method
        self.d = d
        self.threshold = threshold
        self.max_lags = max_lags

        # Store transformation parameters for inverse
        self._weights: npt.NDArray[np.float64] | None = None
        self._mean: float = 0.0
        self._std: float = 1.0
        self._optimal_d: float | None = None

    @staticmethod
    @jit_typed(nopython=True)
    def _compute_weights_numba(d: float, size: int) -> npt.NDArray[np.float64]:
        """
        Compute fractional differencing weights (JIT compiled).
        """
        w = np.ones(size, dtype=np.float64)
        for k in range(1, size):
            w[k] = -w[k - 1] * (d - k + 1) / k
        return w[::-1]  # reverse for convolution

    def fractional_weights(self, d: float, size: int) -> npt.NDArray[np.float64]:
        """
        Compute weights for fractional differencing.

        Parameters
        ----------
        d : float
            Differencing order
        size : int
            Number of weights to compute

        Returns
        -------
        np.ndarray
            Fractional differencing weights

        """
        return self._compute_weights_numba(d, size)

    def fractional_difference(
        self,
        series: npt.NDArray[np.float64],
        d: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Apply fractional differencing to time series.

        Parameters
        ----------
        series : np.ndarray
            Input time series
        d : float | None
            Differencing order (uses self.d if None)

        Returns
        -------
        np.ndarray
            Fractionally differenced series

        """
        if d is None:
            d = self.d

        series = np.asarray(series, dtype=np.float64)

        # Determine weight size
        if self.max_lags:
            weight_size = min(self.max_lags, len(series))
        else:
            weight_size = len(series)

        # Compute weights
        weights = self.fractional_weights(d, weight_size)

        # Drop small weights for efficiency
        mask = np.abs(weights) > self.threshold
        weights = weights[mask]

        # Store for potential inverse transform
        self._weights = weights

        # Apply fractional differencing
        out = np.zeros_like(series)
        for i in range(len(weights), len(series)):
            out[i] = np.dot(weights, series[i - len(weights) + 1 : i + 1])

        return out

    def find_optimal_d(
        self,
        series: npt.NDArray[np.float64],
        adf_threshold: float = 0.05,
        min_d: float = 0.0,
        max_d: float = 1.0,
        step: float = 0.01,
    ) -> float:
        """
        Find optimal differencing order for stationarity.

        Uses ADF test to find minimum d that achieves stationarity
        while preserving maximum memory.

        Parameters
        ----------
        series : np.ndarray
            Input time series
        adf_threshold : float
            P-value threshold for ADF test
        min_d : float
            Minimum d to test
        max_d : float
            Maximum d to test
        step : float
            Step size for d search

        Returns
        -------
        float
            Optimal differencing order

        """
        from statsmodels.tsa.stattools import adfuller

        # Test original series
        adf_result = adfuller(series, autolag="AIC")
        if adf_result[1] < adf_threshold:
            return 0.0  # Already stationary

        # Search for optimal d
        for d in np.arange(min_d, max_d + step, step):
            d_value: float = float(d)
            diff_series = self.fractional_difference(series, d_value)

            # Remove initial zeros
            weights = self._weights if self._weights is not None else np.array([], dtype=np.float64)
            diff_series = diff_series[len(weights) :]

            if len(diff_series) > 10:  # Need enough data for test
                adf_result = adfuller(diff_series, autolag="AIC")
                if adf_result[1] < adf_threshold:
                    self._optimal_d = d_value
                    return d_value

        # If no d achieves stationarity, return max_d
        self._optimal_d = float(max_d)
        return float(max_d)

    def fit_transform(
        self,
        series: npt.NDArray[np.float64],
        auto_d: bool = False,
    ) -> npt.NDArray[np.float64]:
        """
        Fit transformer and apply transformation.

        Parameters
        ----------
        series : np.ndarray
            Input time series
        auto_d : bool
            Automatically find optimal d

        Returns
        -------
        np.ndarray
            Transformed series

        """
        # Store original statistics for inverse transform
        self._mean = float(np.mean(series))
        self._std = float(np.std(series) + 1e-8)

        if auto_d or self.method == "auto":
            self.d = self.find_optimal_d(series)

        if self.method in ["fractional", "auto"]:
            return self.fractional_difference(series)
        elif self.method == "standard":
            return np.diff(series, prepend=series[0])
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def inverse_transform(
        self,
        series: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Inverse transformation (approximate).

        Note: Perfect inverse is not always possible for fractional
        differencing, this provides an approximation.

        Parameters
        ----------
        series : np.ndarray
            Transformed series

        Returns
        -------
        np.ndarray
            Approximately reconstructed original series

        """
        if self.method == "standard":
            # Cumulative sum for standard differencing
            return np.cumsum(series)
        elif self.method in ["fractional", "auto"]:
            # Approximate inverse using integration
            # This is not exact but provides reasonable approximation
            inverse_d = -self.d
            return self.fractional_difference(series, inverse_d)
        else:
            return series


class MarketMicrostructureFeatures:
    """
    Extract market microstructure features for ML models.

    Implements various microstructure metrics including:
    - Roll's spread estimator
    - Kyle's lambda
    - Amihud illiquidity
    - VPIN (Volume-synchronized Probability of Informed Trading)

    """

    @staticmethod
    def roll_spread(prices: npt.NDArray[np.float64]) -> float:
        """
        Calculate Roll's spread estimator.

        Parameters
        ----------
        prices : np.ndarray
            Price series

        Returns
        -------
        float
            Estimated bid-ask spread

        """
        returns = np.diff(prices) / prices[:-1]
        cov = np.cov(returns[:-1], returns[1:])[0, 1]

        if cov < 0:
            spread = 2 * np.sqrt(-cov)
        else:
            spread = 0.0

        return float(spread)

    @staticmethod
    def kyle_lambda(
        prices: npt.NDArray[np.float64],
        volumes: npt.NDArray[np.float64],
    ) -> float:
        """
        Calculate Kyle's lambda (price impact).

        Parameters
        ----------
        prices : np.ndarray
            Price series
        volumes : np.ndarray
            Volume series

        Returns
        -------
        float
            Kyle's lambda estimate

        """
        returns = np.diff(prices) / prices[:-1]
        signed_volumes = volumes[1:] * np.sign(returns)

        # Regression of returns on signed volumes
        if len(returns) > 1 and np.std(signed_volumes) > 0:
            coef = np.polyfit(signed_volumes, np.abs(returns), 1)[0]
            return float(coef)
        return 0.0

    @staticmethod
    def amihud_illiquidity(
        returns: npt.NDArray[np.float64],
        volumes: npt.NDArray[np.float64],
    ) -> float:
        """
        Calculate Amihud illiquidity measure.

        Parameters
        ----------
        returns : np.ndarray
            Return series
        volumes : np.ndarray
            Volume series (in currency units)

        Returns
        -------
        float
            Amihud illiquidity ratio

        """
        # Avoid division by zero
        volumes = np.where(volumes > 0, volumes, 1.0)

        illiquidity = np.mean(np.abs(returns) / volumes)
        return float(illiquidity)

    @staticmethod
    def vpin(
        prices: npt.NDArray[np.float64],
        volumes: npt.NDArray[np.float64],
        bucket_size: int = 50,
    ) -> float:
        """
        Calculate Volume-synchronized Probability of Informed Trading.

        Simplified VPIN calculation based on Easley et al. (2012).

        Parameters
        ----------
        prices : np.ndarray
            Price series
        volumes : np.ndarray
            Volume series
        bucket_size : int
            Size of volume buckets

        Returns
        -------
        float
            VPIN estimate

        """
        returns = np.diff(prices) / prices[:-1]

        # Classify volumes as buy or sell
        buy_volumes = volumes[1:] * (returns > 0)
        sell_volumes = volumes[1:] * (returns <= 0)

        # Create volume buckets
        n_buckets = len(volumes) // bucket_size
        vpin_values = []

        for i in range(n_buckets):
            start = i * bucket_size
            end = (i + 1) * bucket_size

            buy_vol = np.sum(buy_volumes[start:end])
            sell_vol = np.sum(sell_volumes[start:end])
            total_vol = buy_vol + sell_vol

            if total_vol > 0:
                vpin = abs(buy_vol - sell_vol) / total_vol
                vpin_values.append(vpin)

        return float(np.mean(vpin_values)) if vpin_values else 0.0


class FeatureLagGenerator:
    """
    Generate lagged features for time series models.

    Creates various lag-based features including:
    - Simple lags
    - Rolling statistics
    - Exponentially weighted features

    """

    def __init__(
        self,
        lag_periods: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        ewm_spans: list[int] | None = None,
    ):
        """
        Initialize lag generator.

        Parameters
        ----------
        lag_periods : list[int] | None
            Lag periods to generate
        rolling_windows : list[int] | None
            Rolling window sizes
        ewm_spans : list[int] | None
            Exponential weighted spans

        """
        self.lag_periods = lag_periods or [1, 2, 3, 5, 10, 20]
        self.rolling_windows = rolling_windows or [5, 10, 20, 50]
        self.ewm_spans = ewm_spans or [5, 10, 20]

    def create_lagged_features(
        self,
        series: npt.NDArray[np.float64],
        include_rolling: bool = True,
        include_ewm: bool = True,
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        Create comprehensive lagged features.

        Parameters
        ----------
        series : np.ndarray
            Input time series
        include_rolling : bool
            Include rolling statistics
        include_ewm : bool
            Include exponentially weighted features

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of feature arrays

        """
        features = {}

        # Simple lags
        for lag in self.lag_periods:
            if lag < len(series):
                lagged = np.roll(series, lag)
                lagged[:lag] = np.nan
                features[f"lag_{lag}"] = lagged

        # Rolling statistics
        if include_rolling:
            for window in self.rolling_windows:
                if window < len(series):
                    # Rolling mean
                    rolling_mean = np.convolve(
                        series,
                        np.ones(window) / window,
                        mode="same",
                    ).astype(np.float64)
                    features[f"rolling_mean_{window}"] = rolling_mean

                    # Rolling std
                    rolling_std = np.array(
                        [
                            np.std(series[max(0, i - window + 1) : i + 1])
                            for i in range(len(series))
                        ],
                    )
                    features[f"rolling_std_{window}"] = rolling_std

        # Exponentially weighted features
        if include_ewm:
            for span in self.ewm_spans:
                alpha = 2 / (span + 1)
                ewm = np.zeros_like(series)
                ewm[0] = series[0]

                for i in range(1, len(series)):
                    ewm[i] = alpha * series[i] + (1 - alpha) * ewm[i - 1]

                features[f"ewm_{span}"] = ewm

        return features


class DataNormalizer:
    """
    Advanced normalization techniques for financial data.

    Includes:
    - Robust scaling (resistant to outliers)
    - Rank transformation
    - Box-Cox transformation

    """

    def __init__(self, method: str = "robust"):
        """
        Initialize normalizer.

        Parameters
        ----------
        method : str
            Normalization method ('robust', 'rank', 'boxcox')

        """
        self.method = method
        self._params: dict[str, Any] = {}

    def fit_transform(
        self,
        data: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Fit normalizer and transform data.

        Parameters
        ----------
        data : np.ndarray
            Input data

        Returns
        -------
        np.ndarray
            Normalized data

        """
        if self.method == "robust":
            # Use median and MAD for robust scaling
            median = np.median(data)
            mad = np.median(np.abs(data - median))

            self._params["median"] = median
            self._params["mad"] = mad if mad > 0 else 1.0

            return cast(npt.NDArray[np.float64], (data - median) / self._params["mad"])

        elif self.method == "rank":
            # Rank transformation
            from scipy.stats import rankdata

            ranks = rankdata(data)
            n = len(data)

            # Map to uniform distribution
            uniform = (ranks - 0.5) / n

            # Map to normal distribution
            from scipy.stats import norm

            return cast(npt.NDArray[np.float64], norm.ppf(uniform))

        elif self.method == "boxcox":
            # Box-Cox transformation
            from scipy.stats import boxcox

            # Ensure positive values
            min_val = np.min(data)
            if min_val <= 0:
                shift = abs(min_val) + 1
                data = data + shift
                self._params["shift"] = shift
            else:
                self._params["shift"] = 0

            transformed, lambda_param = boxcox(data)
            self._params["lambda"] = lambda_param

            return cast(npt.NDArray[np.float64], transformed)

        else:
            # Standard normalization
            mean = np.mean(data)
            std = np.std(data) + 1e-8

            self._params["mean"] = mean
            self._params["std"] = std

            return cast(npt.NDArray[np.float64], (data - mean) / std)

    def inverse_transform(
        self,
        data: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Inverse transformation.

        Parameters
        ----------
        data : np.ndarray
            Normalized data

        Returns
        -------
        np.ndarray
            Original scale data

        """
        if self.method == "robust":
            median = self._params.get("median", 0)
            mad = self._params.get("mad", 1)
            return cast(npt.NDArray[np.float64], data * mad + median)

        elif self.method == "rank":
            # Approximate inverse using percentiles
            from scipy.stats import norm

            # Map from normal to uniform
            uniform = cast(npt.NDArray[np.float64], norm.cdf(data))
            # This is approximate - exact inverse requires original data
            return uniform

        elif self.method == "boxcox":
            from scipy.special import inv_boxcox

            lambda_param = self._params.get("lambda", 1)
            shift = self._params.get("shift", 0)

            inverse = inv_boxcox(data, lambda_param)
            return cast(npt.NDArray[np.float64], inverse - shift)

        else:
            mean = self._params.get("mean", 0)
            std = self._params.get("std", 1)
            return cast(npt.NDArray[np.float64], data * std + mean)


class PurgedCrossValidator:
    """
    Purged walk-forward cross-validation for financial time series.

    Implements purged and embargoed cross-validation to prevent information leakage
    in financial ML models, as described in López de Prado (2018).

    Parameters
    ----------
    n_splits : int, default 5
        Number of cross-validation splits
    purge_gap : int, default 0
        Number of samples to exclude between train and test sets to prevent leakage
    embargo_pct : float, default 0.0
        Percentage of total samples to embargo after each test set

    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 0,
        embargo_pct: float = 0.0,
    ) -> None:
        """
        Initialize purged cross-validator.

        Parameters
        ----------
        n_splits : int
            Number of splits for cross-validation
        purge_gap : int
            Gap between train and test to prevent leakage
        embargo_pct : float
            Percentage of data to embargo after test set

        Raises
        ------
        ValueError
            If parameters are invalid

        """
        if n_splits < 2:
            msg = f"n_splits must be at least 2, got {n_splits}"
            raise ValueError(msg)
        if purge_gap < 0:
            msg = f"purge_gap must be non-negative, got {purge_gap}"
            raise ValueError(msg)
        if not 0 <= embargo_pct < 1:
            msg = f"embargo_pct must be in [0, 1), got {embargo_pct}"
            raise ValueError(msg)

        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: npt.NDArray[np.float64] | Any,
        y: npt.NDArray[np.float64] | None = None,
        _groups: npt.NDArray[np.int64] | None = None,
    ) -> list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        """
        Generate indices for train/test splits with purging and embargo.

        Parameters
        ----------
        X : array-like
            Features array of shape (n_samples, n_features)
        y : array-like, optional
            Target array (not used, for sklearn compatibility)
        groups : array-like, optional
            Group labels for samples (not used)

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            List of (train_indices, test_indices) tuples

        """
        n_samples = len(X) if hasattr(X, "__len__") else X.shape[0]
        indices = np.arange(n_samples)

        # Calculate embargo size
        embargo_size = int(n_samples * self.embargo_pct)

        # Calculate test size for each fold
        test_size = n_samples // self.n_splits

        splits: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]] = []

        for i in range(self.n_splits):
            # Define test set boundaries
            test_start = i * test_size
            test_end = (i + 1) * test_size if i < self.n_splits - 1 else n_samples

            # Test indices
            test_indices = indices[test_start:test_end]

            # Train indices with purging
            train_indices_before = indices[: max(0, test_start - self.purge_gap)]
            train_indices_after = indices[min(n_samples, test_end + self.purge_gap) :]

            # Apply embargo - remove samples after test set
            if embargo_size > 0 and i < self.n_splits - 1:
                embargo_end = min(n_samples, test_end + embargo_size)
                train_indices_after = train_indices_after[train_indices_after >= embargo_end]

            # Combine train indices
            train_indices = np.concatenate([train_indices_before, train_indices_after])

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits

    def get_n_splits(
        self,
        X: Any | None = None,
        y: Any | None = None,
        _groups: Any | None = None,
    ) -> int:
        """
        Get number of splits.

        Parameters
        ----------
        X : array-like, optional
            Features (not used)
        y : array-like, optional
            Targets (not used)
        groups : array-like, optional
            Groups (not used)

        Returns
        -------
        int
            Number of splits

        """
        return self.n_splits
