"""
Cross-Asset State Management.

Provides serializable dataclasses for managing cross-asset relationship state with
guaranteed parity between hot and cold computation paths.

Performance Targets:
- Hot path: O(1) state updates, zero allocations
- Cold path: Arbitrary complexity allowed
- Serialization: <1ms per state object
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# ===== Constants =====
DEFAULT_ALPHA: Final[float] = 0.94  # EWMA decay factor (common in finance)
MIN_SAMPLES: Final[int] = 30  # Minimum samples before statistics are valid

# ===== Public API =====
__all__ = [
    "CorrelationState",
    "EWMABetaState",
    "ZScoreSpreadState",
]


@dataclass(slots=True)
class EWMABetaState:
    """
    State for incremental EWMA beta computation.

    Tracks exponentially weighted covariance and variance for computing
    rolling beta between an asset and a benchmark/market.

    Attributes
    ----------
    alpha : float
        EWMA decay factor in (0, 1). Common values: 0.94 (RiskMetrics), 0.97
    ewma_cov : float
        Exponentially weighted covariance between asset and market
    ewma_var_market : float
        Exponentially weighted variance of market returns
    n : int
        Number of observations processed
    last_beta : float
        Most recently computed beta value

    Notes
    -----
    - State is serializable for persistence and recovery
    - All fields use float64 precision for numerical stability
    - Hot path updates use O(1) operations only
    """

    alpha: float = DEFAULT_ALPHA
    ewma_cov: float = 0.0
    ewma_var_market: float = 0.0
    n: int = 0
    last_beta: float = 0.0

    def __post_init__(self) -> None:
        """Validate state parameters."""
        if not 0 < self.alpha < 1:
            msg = f"alpha must be in (0, 1), got {self.alpha}"
            raise ValueError(msg)

    def is_valid(self) -> bool:
        """
        Check if state has enough samples for valid statistics.

        Returns
        -------
        bool
            True if n >= MIN_SAMPLES, False otherwise
        """
        return self.n >= MIN_SAMPLES

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self.ewma_cov = 0.0
        self.ewma_var_market = 0.0
        self.n = 0
        self.last_beta = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """
        Serialize state to dictionary for persistence.

        Returns
        -------
        dict[str, float | int]
            Dictionary containing all state fields
        """
        return {
            "alpha": self.alpha,
            "ewma_cov": self.ewma_cov,
            "ewma_var_market": self.ewma_var_market,
            "n": self.n,
            "last_beta": self.last_beta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float | int]) -> EWMABetaState:
        """
        Deserialize state from dictionary.

        Parameters
        ----------
        data : dict[str, float | int]
            Dictionary containing state fields

        Returns
        -------
        EWMABetaState
            Restored state object
        """
        return cls(
            alpha=float(data["alpha"]),
            ewma_cov=float(data["ewma_cov"]),
            ewma_var_market=float(data["ewma_var_market"]),
            n=int(data["n"]),
            last_beta=float(data["last_beta"]),
        )


@dataclass(slots=True)
class ZScoreSpreadState:
    """
    State for incremental Z-scored spread computation.

    Uses Welford's algorithm for numerically stable online mean and variance
    computation of price spreads between asset pairs.

    Attributes
    ----------
    mean : float
        Running mean of the spread
    m2 : float
        Sum of squared deviations (for variance calculation)
    n : int
        Number of observations processed
    last_zscore : float
        Most recently computed z-score

    Notes
    -----
    - Welford's algorithm provides numerical stability for online variance
    - State is serializable for persistence across sessions
    - Hot path updates are O(1) with zero allocations
    - Variance = m2 / (n - 1) for sample variance
    """

    mean: float = 0.0
    m2: float = 0.0
    n: int = 0
    last_zscore: float = 0.0

    def is_valid(self) -> bool:
        """
        Check if state has enough samples for valid statistics.

        Returns
        -------
        bool
            True if n >= MIN_SAMPLES, False otherwise
        """
        return self.n >= MIN_SAMPLES

    def get_std(self) -> float:
        """
        Compute standard deviation from current state.

        Returns
        -------
        float
            Standard deviation, or 0.0 if n < 2

        Notes
        -----
        Uses sample variance (n-1 denominator)
        """
        if self.n < 2:
            return 0.0
        variance: float = self.m2 / (self.n - 1)
        return float(variance**0.5)

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self.mean = 0.0
        self.m2 = 0.0
        self.n = 0
        self.last_zscore = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """
        Serialize state to dictionary for persistence.

        Returns
        -------
        dict[str, float | int]
            Dictionary containing all state fields
        """
        return {
            "mean": self.mean,
            "m2": self.m2,
            "n": self.n,
            "last_zscore": self.last_zscore,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float | int]) -> ZScoreSpreadState:
        """
        Deserialize state from dictionary.

        Parameters
        ----------
        data : dict[str, float | int]
            Dictionary containing state fields

        Returns
        -------
        ZScoreSpreadState
            Restored state object
        """
        return cls(
            mean=float(data["mean"]),
            m2=float(data["m2"]),
            n=int(data["n"]),
            last_zscore=float(data["last_zscore"]),
        )


@dataclass(slots=True)
class CorrelationState:
    """
    State for incremental correlation calculation using Welford's algorithm.

    Maintains running statistics for two time series to compute correlation
    without storing full windows.

    Attributes
    ----------
    n : int
        Number of observations in window
    mean_x : float
        Running mean of first series
    mean_y : float
        Running mean of second series
    m2_x : float
        Sum of squared deviations for first series
    m2_y : float
        Sum of squared deviations for second series
    m2_xy : float
        Sum of cross-products of deviations (covariance accumulator)
    window_size : int
        Maximum window size
    last_correlation : float
        Most recently computed correlation value

    Notes
    -----
    - Welford's algorithm provides numerical stability for online covariance
    - State is serializable for persistence across sessions
    - Hot path updates are O(1) with zero allocations
    - Correlation = cov(X,Y) / (std(X) * std(Y))
    """

    n: int = 0
    mean_x: float = 0.0
    mean_y: float = 0.0
    m2_x: float = 0.0
    m2_y: float = 0.0
    m2_xy: float = 0.0
    window_size: int = 60
    last_correlation: float = 0.0

    def __post_init__(self) -> None:
        """Validate state parameters."""
        if self.window_size < 2:
            msg = f"window_size must be >= 2, got {self.window_size}"
            raise ValueError(msg)

    def is_valid(self) -> bool:
        """
        Check if state has enough samples for valid statistics.

        Returns
        -------
        bool
            True if n >= MIN_SAMPLES, False otherwise
        """
        return self.n >= MIN_SAMPLES

    def get_correlation(self) -> float:
        """
        Compute correlation from current state.

        Returns
        -------
        float
            Correlation coefficient, or 0.0 if insufficient data

        Notes
        -----
        Correlation = cov(X,Y) / (std(X) * std(Y))
        Returns 0.0 if either variance is near zero
        """
        if self.n < 2:
            return 0.0

        # Compute variances and covariance
        var_x = self.m2_x / (self.n - 1)
        var_y = self.m2_y / (self.n - 1)
        cov_xy = self.m2_xy / (self.n - 1)

        # Check for zero variance
        if var_x < 1e-12 or var_y < 1e-12:
            return 0.0

        # Compute correlation
        std_x = float(var_x**0.5)
        std_y = float(var_y**0.5)
        correlation = cov_xy / (std_x * std_y)

        return float(correlation)

    def reset(self) -> None:
        """Reset state to initial conditions."""
        self.n = 0
        self.mean_x = 0.0
        self.mean_y = 0.0
        self.m2_x = 0.0
        self.m2_y = 0.0
        self.m2_xy = 0.0
        self.last_correlation = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """
        Serialize state to dictionary for persistence.

        Returns
        -------
        dict[str, float | int]
            Dictionary containing all state fields
        """
        return {
            "n": self.n,
            "mean_x": self.mean_x,
            "mean_y": self.mean_y,
            "m2_x": self.m2_x,
            "m2_y": self.m2_y,
            "m2_xy": self.m2_xy,
            "window_size": self.window_size,
            "last_correlation": self.last_correlation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float | int]) -> CorrelationState:
        """
        Deserialize state from dictionary.

        Parameters
        ----------
        data : dict[str, float | int]
            Dictionary containing state fields

        Returns
        -------
        CorrelationState
            Restored state object
        """
        return cls(
            n=int(data["n"]),
            mean_x=float(data["mean_x"]),
            mean_y=float(data["mean_y"]),
            m2_x=float(data["m2_x"]),
            m2_y=float(data["m2_y"]),
            m2_xy=float(data["m2_xy"]),
            window_size=int(data["window_size"]),
            last_correlation=float(data["last_correlation"]),
        )
