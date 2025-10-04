"""
Earnings feature transforms for pipeline integration.

Provides TransformSpec classes for declarative earnings feature specification
following the Universal ML Architecture Patterns.

## Transform Specifications

### EarningsSurpriseTransformSpec
Computes earnings surprise features (actual vs. estimate):
- eps_surprise_q0: Dollar surprise (actual - estimate)
- eps_surprise_pct_q0: Percentage surprise
- revenue_surprise_pct_q0: Revenue percentage surprise

### EarningsGrowthTransformSpec
Computes year-over-year and quarter-over-quarter growth:
- eps_growth_yoy: YoY EPS growth percentage
- eps_growth_qoq: QoQ EPS growth percentage

### EarningsMomentumTransformSpec
Computes earnings momentum indicators:
- earnings_beat_streak: Consecutive quarters beating estimates
- eps_volatility_4q: 4-quarter EPS volatility

### EarningsCalendarTransformSpec
Computes earnings calendar features:
- days_to_next_earnings: Days until next earnings announcement

## Usage Example

```python
from ml.features.pipeline import PipelineSpec
from ml.features.earnings import (
    EarningsSurpriseTransformSpec,
    EarningsGrowthTransformSpec,
    EarningsMomentumTransformSpec,
    EarningsCalendarTransformSpec,
)

# Define pipeline with earnings features
pipeline = PipelineSpec(transforms=[
    EarningsSurpriseTransformSpec(ticker="AAPL"),
    EarningsGrowthTransformSpec(ticker="AAPL", lookback_quarters=5),
    EarningsMomentumTransformSpec(ticker="AAPL", lookback_quarters=4),
    EarningsCalendarTransformSpec(ticker="AAPL"),
])
```

## Pattern Compliance

- **Pattern 2**: TransformSpec dataclasses are frozen and picklable
- **Pattern 3**: Transforms define feature names (computation in cold path)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EarningsSurpriseTransformSpec:
    """
    Transform specification for earnings surprise features.

    Computes surprise metrics by comparing actual earnings to consensus estimates:
    - EPS surprise (absolute and percentage)
    - Revenue surprise (percentage)

    Parameters
    ----------
    name : str, default "earnings_surprise"
        Transform name for pipeline identification.
    ticker : str, default ""
        Stock ticker symbol for multi-instrument support.
        Required for proper feature naming in cross-asset pipelines.
    lookback_quarters : int, default 1
        Number of historical quarters to include.
        Currently only supports 1 (most recent quarter).

    Examples
    --------
    >>> spec = EarningsSurpriseTransformSpec(ticker="AAPL")
    >>> spec.compute_feature_names()
    ['eps_surprise_q0_AAPL', 'eps_surprise_pct_q0_AAPL', 'revenue_surprise_pct_q0_AAPL']
    """

    name: str = "earnings_surprise"
    ticker: str = ""
    lookback_quarters: int = 1

    def compute_feature_names(self) -> list[str]:
        """
        Return feature names produced by this transform.

        Returns
        -------
        list[str]
            Ordered list of feature names with ticker suffix.

        Notes
        -----
        Feature names follow the convention: {metric}_{quarter}_{ticker}
        where quarter is 'q0' for the most recent quarter.
        """
        return [
            f"eps_surprise_q0_{self.ticker}",
            f"eps_surprise_pct_q0_{self.ticker}",
            f"revenue_surprise_pct_q0_{self.ticker}",
        ]


@dataclass(frozen=True)
class EarningsGrowthTransformSpec:
    """
    Transform specification for earnings growth features.

    Computes year-over-year (YoY) and quarter-over-quarter (QoQ) EPS growth rates.

    Parameters
    ----------
    name : str, default "earnings_growth"
        Transform name for pipeline identification.
    ticker : str, default ""
        Stock ticker symbol for multi-instrument support.
        Required for proper feature naming in cross-asset pipelines.
    lookback_quarters : int, default 5
        Number of historical quarters needed for growth calculation.
        Requires at least 5 quarters (current + 4 historical).

    Examples
    --------
    >>> spec = EarningsGrowthTransformSpec(ticker="MSFT")
    >>> spec.compute_feature_names()
    ['eps_growth_yoy_MSFT', 'eps_growth_qoq_MSFT']

    Notes
    -----
    - YoY growth compares Q0 vs Q-4 (same quarter last year)
    - QoQ growth compares Q0 vs Q-1 (previous quarter)
    - Growth rates are expressed as percentages
    """

    name: str = "earnings_growth"
    ticker: str = ""
    lookback_quarters: int = 5

    def compute_feature_names(self) -> list[str]:
        """
        Return feature names produced by this transform.

        Returns
        -------
        list[str]
            Ordered list of feature names with ticker suffix.

        Notes
        -----
        Feature names include both YoY and QoQ growth metrics.
        """
        return [
            f"eps_growth_yoy_{self.ticker}",
            f"eps_growth_qoq_{self.ticker}",
        ]


@dataclass(frozen=True)
class EarningsMomentumTransformSpec:
    """
    Transform specification for earnings momentum features.

    Computes momentum indicators based on earnings beat/miss patterns and volatility.

    Parameters
    ----------
    name : str, default "earnings_momentum"
        Transform name for pipeline identification.
    ticker : str, default ""
        Stock ticker symbol for multi-instrument support.
        Required for proper feature naming in cross-asset pipelines.
    lookback_quarters : int, default 4
        Number of historical quarters for momentum calculation.
        Used for beat streak counting and volatility computation.

    Examples
    --------
    >>> spec = EarningsMomentumTransformSpec(ticker="GOOGL")
    >>> spec.compute_feature_names()
    ['earnings_beat_streak_GOOGL', 'eps_volatility_4q_GOOGL']

    Notes
    -----
    - Beat streak: Consecutive quarters with positive earnings surprise
    - Volatility: Coefficient of variation of EPS over lookback period
    """

    name: str = "earnings_momentum"
    ticker: str = ""
    lookback_quarters: int = 4

    def compute_feature_names(self) -> list[str]:
        """
        Return feature names produced by this transform.

        Returns
        -------
        list[str]
            Ordered list of feature names with ticker suffix.

        Notes
        -----
        Includes beat streak (integer count) and volatility (ratio).
        """
        return [
            f"earnings_beat_streak_{self.ticker}",
            f"eps_volatility_4q_{self.ticker}",
        ]


@dataclass(frozen=True)
class EarningsCalendarTransformSpec:
    """
    Transform specification for earnings calendar features.

    Computes time-based features related to upcoming earnings announcements.

    Parameters
    ----------
    name : str, default "earnings_calendar"
        Transform name for pipeline identification.
    ticker : str, default ""
        Stock ticker symbol for multi-instrument support.
        Required for proper feature naming in cross-asset pipelines.

    Examples
    --------
    >>> spec = EarningsCalendarTransformSpec(ticker="TSLA")
    >>> spec.compute_feature_names()
    ['days_to_next_earnings_TSLA']

    Notes
    -----
    - Days to earnings: Calendar days until next scheduled announcement
    - Useful for modeling pre-earnings volatility and positioning
    - Known-future feature suitable for TFT models
    """

    name: str = "earnings_calendar"
    ticker: str = ""

    def compute_feature_names(self) -> list[str]:
        """
        Return feature names produced by this transform.

        Returns
        -------
        list[str]
            Ordered list of feature names with ticker suffix.

        Notes
        -----
        Returns a single feature indicating days to next earnings event.
        """
        return [
            f"days_to_next_earnings_{self.ticker}",
        ]
