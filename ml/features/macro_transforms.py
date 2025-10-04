"""
Macro feature transforms for ALFRED vintage and FRED data integration.

Provides training/inference parity for macro features by implementing both batch
(historical) and real-time computation paths.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from ml._imports import check_ml_dependencies
from ml._imports import pl
from ml.common.metrics_bootstrap import get_counter
from ml.data.validation import MacroCoverageValidator
from ml.data.vintage import VintagePolicy
from ml.features.macro_cache import MacroDataCache
from ml.features.macro_cache import MacroSeriesSnapshot
from ml.features.macro_composites import get_composite_feature_names
from ml.features.macro_composites import get_composite_series_requirements


if pl is None:
    check_ml_dependencies(["polars"])  # pragma: no cover

if TYPE_CHECKING:
    import polars as _pl
    from nautilus_trader.model.data import Bar

    PolarsDataFrame: TypeAlias = _pl.DataFrame
else:  # pragma: no cover - runtime aliasing
    PolarsDataFrame = Any  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class MacroFeatureTransform:
    """
    Transform that adds ALFRED/FRED macro features with training/inference parity.

    This transform ensures that macro features are computed identically in both
    training (batch) and inference (real-time) modes, maintaining feature parity.

    Parameters
    ----------
    macro_series_ids : list[str]
        FRED series identifiers to include (e.g., ["PAYEMS", "UNRATE", "CPIAUCSL"]).
    vintage_base_dir : Path | str
        Directory containing ALFRED vintage data (data/fred/vintages/).
    fred_path : Path | str | None
        Path to FRED indicators parquet file (for series without vintages).
    include_revisions : bool, default False
        Whether to include revision-aware features.
    revision_mode : {"minimal", "core", "full"}, default "core"
        Revision feature mode:
        - minimal: current, prior_1m, revision_1m
        - core: + mom_1m, pct_1m, net_signal_1m
        - full: + prior_3m/12m, revision_3m, mom_3m/12m, pct_12m
    lag_days : int, default 1
        Publication lag for non-vintage series.
    vintage_policy : VintagePolicy, default REAL_TIME
        Policy for vintage selection.
    include_composites : bool, default False
        Whether to compute macro composite features (credit, duration, liquidity, FX).
    composite_history_window : int, default 400
        Number of trailing observations retained in the cache for composite calculations.

    """

    def __init__(
        self,
        macro_series_ids: list[str],
        vintage_base_dir: Path | str,
        fred_path: Path | str | None = None,
        include_revisions: bool = False,
        revision_mode: Literal["minimal", "core", "full"] = "core",
        lag_days: int = 1,
        vintage_policy: VintagePolicy = VintagePolicy.REAL_TIME,
        include_composites: bool = False,
        composite_history_window: int = 400,
        min_coverage: float | None = None,
    ) -> None:
        self.macro_series_ids = macro_series_ids
        self.vintage_base_dir = Path(vintage_base_dir).expanduser()
        self.fred_path = Path(fred_path).expanduser() if fred_path else None
        self.include_revisions = include_revisions
        self.revision_mode = revision_mode
        self.lag_days = lag_days
        self.vintage_policy = vintage_policy
        self.include_composites = include_composites
        if min_coverage is not None and not 0.0 < float(min_coverage) <= 1.0:
            msg = f"min_coverage must be within (0, 1], received {min_coverage}"
            raise ValueError(msg)
        self._composite_history_window = max(0, composite_history_window)

        # Real-time cache (lazy-loaded)
        self._cache: MacroDataCache | None = None
        composite_requirements: tuple[str, ...] = tuple()
        if include_composites:
            composite_requirements = get_composite_series_requirements()
        self._composite_series_requirements: tuple[str, ...] = composite_requirements
        self._series_ids_for_batch: list[str] = list(dict.fromkeys(macro_series_ids))
        self._composite_aux_series: list[str] = []
        if include_composites and composite_requirements:
            aux_series = [
                series
                for series in composite_requirements
                if series not in self._series_ids_for_batch
            ]
            if aux_series:
                self._series_ids_for_batch.extend(aux_series)
                self._composite_aux_series = aux_series
        self._composite_issue_metric: Any | None = None
        self._composite_issue_logged: set[tuple[str, str]] = set()
        self._coverage_validator: MacroCoverageValidator | None = (
            MacroCoverageValidator(min_coverage=float(min_coverage))
            if min_coverage is not None
            else None
        )
        self._last_coverage: dict[str, float] | None = None
        self._last_realtime_coverage: dict[str, bool] | None = None

    def _get_cache(self) -> MacroDataCache:
        """Get or create real-time cache."""
        if self._cache is None:
            aux_series: list[str] = []
            if self.include_composites and self._composite_series_requirements:
                aux_series = [
                    series
                    for series in self._composite_series_requirements
                    if series not in self.macro_series_ids
                ]
                self._composite_aux_series = aux_series
            self._cache = MacroDataCache(
                vintage_base_dir=self.vintage_base_dir,
                series_ids=self.macro_series_ids,
                enable_revisions=self.include_revisions,
                aux_series_ids=aux_series,
                history_window=self._composite_history_window,
            )
        return self._cache

    def compute_batch(
        self,
        df: PolarsDataFrame,
        timestamp_col: str = "timestamp",
        vintage_cutoff: Any = None,
    ) -> PolarsDataFrame:
        """
        Compute macro features for batch (historical) data.

        Uses join_fred_asof to apply point-in-time vintage logic.

        Parameters
        ----------
        df : pl.DataFrame
            Market data with timestamp column.
        timestamp_col : str, default "timestamp"
            Name of timestamp column.
        vintage_cutoff : datetime | None
            Cutoff date for vintage selection (for backtesting).

        Returns
        -------
        pl.DataFrame
            DataFrame with macro features added as columns.

        """
        from ml.data.fred_join import join_fred_asof

        result = join_fred_asof(
            df,
            timestamp_col=timestamp_col,
            lag_days=self.lag_days,
            fred_path=self.fred_path,
            vintage_base_dir=self.vintage_base_dir,
            series_filter=set(self._series_ids_for_batch),
            vintage_policy=self.vintage_policy,
            vintage_cutoff=vintage_cutoff,
            include_revisions=self.include_revisions,
            revision_mode=self.revision_mode,
            revision_windows=None,  # Use defaults
        )
        if pl is None:
            check_ml_dependencies(["polars"])  # pragma: no cover
        assert pl is not None
        if not isinstance(result, pl.DataFrame):
            raise TypeError("join_fred_asof must return a polars DataFrame for polars input")
        output = cast(PolarsDataFrame, result)

        if self._coverage_validator is not None:
            coverage_map = self._coverage_validator.validate_macro_coverage(
                output,
                self._series_ids_for_batch if self.include_composites else tuple(self.macro_series_ids),
            )
            self._last_coverage = coverage_map
            logger.debug(
                "Validated macro series coverage (min=%.3f): %s",
                self._coverage_validator.min_coverage,
                coverage_map,
            )

        if self.include_composites:
            from ml.features.macro_composites import compute_macro_composites_pl

            output = compute_macro_composites_pl(output)
            if self._composite_aux_series:
                columns_to_drop = {
                    column
                    for column in output.columns
                    for series_id in self._composite_aux_series
                    if column.startswith(series_id)
                }
                if columns_to_drop:
                    output = output.drop(sorted(columns_to_drop))

        return output

    def compute_realtime(
        self,
        bar: Bar | None = None,
        ts_event: int | None = None,
    ) -> dict[str, float]:
        """
        Compute macro features for real-time inference.

        Uses cached latest values - no point-in-time filtering needed since we're
        always at "now".

        Parameters
        ----------
        bar : Bar | None
            Current bar (unused, kept for signature compatibility).
        ts_event : int | None
            Event timestamp (unused in real-time - always uses latest).

        Returns
        -------
        dict[str, float]
            Feature name → value mapping for all macro features.

        """
        cache = self._get_cache()

        if not cache.is_loaded():
            logger.warning("Macro cache not loaded, returning empty features")
            return {}

        if self._coverage_validator is not None:
            coverage_flags = cache.get_coverage()
            self._last_realtime_coverage = coverage_flags
            missing_series = [series for series, available in coverage_flags.items() if not available]
            if missing_series:
                logger.warning(
                    "Missing macro series in real-time cache: %s",
                    sorted(missing_series),
                )

        # Get all features from cache (uses latest released values)
        features = cache.get_all_features(mode=self.revision_mode)

        if not self.include_composites:
            return features

        composites, issues = _compute_realtime_composites(cache)

        self._record_composite_issues(issues)

        features.update(composites)
        return features

    def get_feature_names(self) -> list[str]:
        """
        Get all feature names that will be produced.

        Returns
        -------
        list[str]
            Ordered list of feature names.

        """
        # Build feature names based on configuration
        feature_names: list[str] = []

        for series_id in self.macro_series_ids:
            # Base value (always included)
            feature_names.append(f"{series_id}__value_real_time")

            # Minimal mode features
            if self.revision_mode in ["minimal", "core", "full"]:
                feature_names.append(f"{series_id}_prior_1m")
                if self.include_revisions:
                    feature_names.append(f"{series_id}_revision_1m")

            # Core mode features
            if self.revision_mode in ["core", "full"]:
                feature_names.extend([
                    f"{series_id}_mom_1m",
                    f"{series_id}_pct_1m",
                ])
                if self.include_revisions:
                    feature_names.append(f"{series_id}_net_signal_1m")

            # Full mode features
            if self.revision_mode == "full":
                feature_names.extend([
                    f"{series_id}_prior_3m",
                    f"{series_id}_prior_12m",
                    f"{series_id}_mom_3m",
                    f"{series_id}_mom_12m",
                    f"{series_id}_pct_12m",
                ])
                if self.include_revisions:
                    feature_names.append(f"{series_id}_revision_3m")

        if self.include_composites:
            feature_names.extend(get_composite_feature_names())

        return feature_names

    def _record_composite_issues(self, issues: Iterable[tuple[str, str]]) -> None:
        """Record composite computation issues via metrics/logging."""
        if not issues:
            return

        if self._composite_issue_metric is None:
            self._composite_issue_metric = get_counter(
                "ml_macro_composite_missing_total",
                "Count of macro composite computations missing prerequisites",
                labelnames=("series_id", "reason"),
            )

        for series_id, reason in issues:
            metric = self._composite_issue_metric
            metric.labels(series_id=series_id, reason=reason).inc()

            key = (series_id, reason)
            if key in self._composite_issue_logged:
                continue
            logger.warning(
                "Macro composite prerequisite missing",
                extra={
                    "series_id": series_id,
                    "reason": reason,
                },
            )
            self._composite_issue_logged.add(key)

    def get_feature_dtypes(self) -> list[str]:
        """
        Get dtypes for all features.

        Returns
        -------
        list[str]
            List of dtype strings (all float64 for macro features).

        """
        return ["float64"] * len(self.get_feature_names())

    def refresh_cache(self) -> None:
        """
        Refresh the real-time cache with latest data.

        Call this periodically (e.g., daily) to pick up new FRED/ALFRED releases.
        """
        if self._cache is not None:
            logger.info("Refreshing macro feature cache")
            self._cache.refresh()

    def get_cache_coverage(self) -> dict[str, bool]:
        """
        Get cache coverage showing which series are available.

        Returns
        -------
        dict[str, bool]
            Series ID → available mapping.

        """
        cache = self._get_cache()
        return cache.get_coverage()

    def get_transform_config(self) -> dict[str, Any]:
        """
        Get transform configuration for serialization.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary.

        """
        return {
            "transform_type": "macro_features",
            "macro_series_ids": self.macro_series_ids,
            "vintage_base_dir": str(self.vintage_base_dir),
            "fred_path": str(self.fred_path) if self.fred_path else None,
            "include_revisions": self.include_revisions,
            "revision_mode": self.revision_mode,
            "lag_days": self.lag_days,
            "vintage_policy": self.vintage_policy.value,
            "include_composites": self.include_composites,
            "composite_history_window": self._composite_history_window,
            "min_coverage": (
                self._coverage_validator.min_coverage
                if self._coverage_validator is not None
                else None
            ),
        }


def create_macro_transform_from_config(
    macro_series_ids: list[str] | tuple[str, ...] | None,
    vintage_base_dir: Path | str | None,
    fred_path: Path | str | None = None,
    include_revisions: bool = False,
    revision_mode: str = "core",
    lag_days: int = 1,
    include_composites: bool = False,
    min_coverage: float | None = None,
) -> MacroFeatureTransform | None:
    """
    Factory function to create MacroFeatureTransform from config.

    Parameters
    ----------
    macro_series_ids : list[str] | tuple[str, ...] | None
        Series to include (None or empty list disables macro features).
    vintage_base_dir : Path | str | None
        Vintage directory path.
    fred_path : Path | str | None
        FRED data path.
    include_revisions : bool
        Enable revision features.
    revision_mode : str
        Revision mode.
    lag_days : int
        Publication lag.
    include_composites : bool
        Enable macro composite outputs.

    Returns
    -------
    MacroFeatureTransform | None
        Transform instance, or None if macro features disabled.

    """
    if not macro_series_ids or vintage_base_dir is None:
        return None

    series_list = list(macro_series_ids) if isinstance(macro_series_ids, tuple) else macro_series_ids

    return MacroFeatureTransform(
        macro_series_ids=series_list,
        vintage_base_dir=vintage_base_dir,
        fred_path=fred_path,
        include_revisions=include_revisions,
        revision_mode=revision_mode,  # type: ignore[arg-type]
        lag_days=lag_days,
        include_composites=include_composites,
        min_coverage=min_coverage,
    )


def _compute_realtime_composites(
    cache: MacroDataCache,
) -> tuple[dict[str, float], set[tuple[str, str]]]:
    """Compute macro composite features using cached snapshots."""
    feature_names = get_composite_feature_names()
    composites: dict[str, float] = dict.fromkeys(feature_names, math.nan)
    issues: set[tuple[str, str]] = set()
    snapshots: dict[str, MacroSeriesSnapshot | None] = {}

    def _get_snapshot(series_id: str) -> MacroSeriesSnapshot | None:
        if series_id in snapshots:
            return snapshots[series_id]
        snapshot = cache.get_snapshot(series_id)
        snapshots[series_id] = snapshot
        if snapshot is None:
            issues.add((series_id, "missing_series"))
        return snapshot

    def _current(series_id: str) -> float | None:
        snapshot = _get_snapshot(series_id)
        if snapshot is None:
            return None
        return snapshot.current_value

    def _prior_value(series_id: str, months: int) -> float | None:
        snapshot = _get_snapshot(series_id)
        if snapshot is None:
            return None
        attr_map = {1: "prior_1m_value", 3: "prior_3m_value", 12: "prior_12m_value"}
        attr = attr_map.get(months)
        if attr is None:
            return None
        value = cast(float | None, getattr(snapshot, attr))
        if value is None:
            issues.add((series_id, f"missing_prior_{months}m"))
        return value

    def _safe_divide(numerator: float, denominator: float, series_id: str, reason: str) -> float | None:
        if denominator == 0.0:
            issues.add((series_id, reason))
            return None
        return numerator / denominator

    def _history(series_id: str) -> tuple[float, ...]:
        snapshot = _get_snapshot(series_id)
        if snapshot is None:
            return tuple()
        return snapshot.history

    def _rolling_std(series_id: str, window: int) -> float | None:
        history = _history(series_id)
        if len(history) < window or window <= 1:
            issues.add((series_id, f"insufficient_history_{window}"))
            return None
        window_slice = history[-window:]
        mean = sum(window_slice) / float(window)
        variance = sum((value - mean) ** 2 for value in window_slice) / float(window)
        return math.sqrt(variance)

    def _returns_std(series_id: str, window: int) -> float | None:
        history = _history(series_id)
        if len(history) < window + 1:
            issues.add((series_id, f"insufficient_history_{window}"))
            return None

        returns: list[float] = []
        start_index = len(history) - window
        for idx in range(start_index, len(history)):
            prev = history[idx - 1]
            curr = history[idx]
            if prev == 0.0:
                issues.add((series_id, "zero_division"))
                continue
            returns.append((curr - prev) / prev)

        if not returns:
            return None

        mean = sum(returns) / float(len(returns))
        variance = sum((value - mean) ** 2 for value in returns) / float(len(returns))
        return math.sqrt(variance)

    def _average(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / float(len(values))

    ig = _current("BAMLC0A0CM")
    if ig is not None:
        composites["credit_spread_ig"] = ig

    hy = _current("BAMLH0A0HYM2")
    if hy is not None:
        composites["credit_spread_hy"] = hy

    if ig is not None and hy is not None:
        composites["credit_spread_hy_ig"] = hy - ig

    if ig is not None:
        composites["credit_spread_bbb_a"] = ig * 0.4

    ig_prior_3m = _prior_value("BAMLC0A0CM", 3)
    if ig is not None and ig_prior_3m is not None:
        composites["credit_spread_ig_momentum"] = ig - ig_prior_3m

    hy_prior_3m = _prior_value("BAMLH0A0HYM2", 3)
    if hy is not None and hy_prior_3m is not None:
        composites["credit_spread_hy_momentum"] = hy - hy_prior_3m

    ted = _current("TEDRATE")
    if ted is not None:
        composites["ted_spread"] = ted

    credit_components: list[float] = []
    if ig is not None:
        credit_components.append(ig / 100.0)
    if hy is not None:
        credit_components.append(hy / 500.0)
    if ted is not None:
        credit_components.append(ted / 50.0)
    vix = _current("VIXCLS")
    if vix is not None:
        credit_components.append(vix / 50.0)

    credit_risk_index = _average(credit_components)
    if credit_risk_index is not None:
        composites["credit_risk_index"] = credit_risk_index

    distress_components: list[float] = []
    if hy is not None:
        distress_components.append(hy / 1000.0)
    if vix is not None:
        distress_components.append(vix / 80.0)
    if ted is not None:
        distress_components.append(ted / 100.0)

    if len(distress_components) >= 2:
        distress = _average(distress_components)
        if distress is not None:
            composites["credit_distress_index"] = distress

    term_spread = _current("T10Y2Y")
    dgs10 = _current("DGS10")
    dgs2 = _current("DGS2")
    dgs30 = _current("DGS30")
    dgs5 = _current("DGS5")

    if term_spread is not None:
        composites["term_spread"] = term_spread
    elif dgs10 is not None and dgs2 is not None:
        composites["term_spread"] = dgs10 - dgs2

    if dgs30 is not None and dgs5 is not None:
        composites["term_spread_5s30s"] = dgs30 - dgs5

    if dgs30 is not None and dgs2 is not None:
        composites["term_spread_2s30s"] = dgs30 - dgs2

    if dgs10 is not None and dgs2 is not None and dgs30 is not None:
        composites["curve_curvature"] = 2.0 * dgs10 - dgs2 - dgs30

    dfii10 = _current("DFII10")
    if dfii10 is not None:
        composites["real_yield_10y"] = dfii10

    if dgs10 is not None and dfii10 is not None:
        composites["real_term_premium"] = dgs10 - dfii10

    if dgs10 is not None and dgs2 is not None:
        slope = _safe_divide(dgs10 - dgs2, dgs2, "DGS2", "zero_division")
        if slope is not None:
            composites["yield_curve_slope"] = slope

    fedfunds = _current("FEDFUNDS")
    if fedfunds is not None and dgs10 is not None:
        composites["fed_policy_stance"] = fedfunds - dgs10

    walcl = _current("WALCL")
    if walcl is not None:
        composites["fed_balance_sheet"] = walcl
        composites["qe_intensity"] = walcl / 1_000_000.0

    totbkcr = _current("TOTBKCR")
    if totbkcr is not None:
        prior = _prior_value("TOTBKCR", 3)
        if prior is not None:
            composites["bank_credit_growth_3m"] = totbkcr - prior

    liquidity_components: list[float] = []
    if walcl is not None:
        liquidity_components.append(walcl / 10_000_000.0)
    if totbkcr is not None:
        liquidity_components.append(totbkcr / 20_000_000.0)
    if ted is not None:
        liquidity_components.append(-ted / 50.0)
    liquidity_index = _average(liquidity_components)
    if liquidity_index is not None:
        composites["liquidity_index"] = liquidity_index

    fedfunds_std = _rolling_std("FEDFUNDS", 30)
    if fedfunds_std is None:
        fedfunds_std = 0.0  # Fill-null behavior
    if ted is not None:
        composites["sofr_obfr_spread"] = ted + fedfunds_std

    stress_components: list[float] = []
    if vix is not None:
        stress_components.append(vix / 80.0)
    if ted is not None:
        stress_components.append(ted / 100.0)
    if hy is not None:
        stress_components.append(hy / 1000.0)

    spread_for_stress: float | None = None
    if term_spread is not None:
        spread_for_stress = term_spread
    elif dgs10 is not None and dgs2 is not None:
        spread_for_stress = dgs10 - dgs2

    if spread_for_stress is not None:
        stress_components.append(max(-(spread_for_stress), 0.0) / 100.0)

    if len(stress_components) >= 2:
        stress_value = _average(stress_components)
        if stress_value is not None:
            composites["financial_stress_composite"] = stress_value

    payems_snapshot = _get_snapshot("PAYEMS")
    payems_mom = None
    if payems_snapshot is not None:
        prior_payems = _prior_value("PAYEMS", 1)
        if prior_payems is not None and prior_payems != 0.0:
            payems_mom = (payems_snapshot.current_value - prior_payems) / prior_payems
            composites["payems_mom"] = payems_mom
        else:
            issues.add(("PAYEMS", "zero_division"))

    indpro_snapshot = _get_snapshot("INDPRO")
    indpro_mom = None
    if indpro_snapshot is not None:
        prior_indpro = _prior_value("INDPRO", 1)
        if prior_indpro is not None and prior_indpro != 0.0:
            indpro_mom = (indpro_snapshot.current_value - prior_indpro) / prior_indpro
            composites["indpro_mom"] = indpro_mom
        else:
            issues.add(("INDPRO", "zero_division"))

    growth_components: list[float] = []
    if payems_mom is not None:
        growth_components.append(payems_mom * 100.0)
    if indpro_mom is not None:
        growth_components.append(indpro_mom * 100.0)
    cfnai = _current("CFNAI")
    if cfnai is not None:
        growth_components.append(cfnai)

    growth_momentum = _average(growth_components)
    if growth_momentum is not None:
        composites["growth_momentum"] = growth_momentum

    def _yoy(series_id: str) -> float | None:
        current_value = _current(series_id)
        prior_value = _prior_value(series_id, 12)
        if current_value is None or prior_value is None:
            return None
        if prior_value == 0.0:
            issues.add((series_id, "zero_division"))
            return None
        return (current_value - prior_value) / prior_value

    cpi_yoy = _yoy("CPIAUCSL")
    if cpi_yoy is not None:
        composites["cpi_yoy"] = cpi_yoy

    pce_yoy = _yoy("PCEPI")
    if pce_yoy is not None:
        composites["pce_yoy"] = pce_yoy

    ppi_yoy = _yoy("PPIACO")
    if ppi_yoy is not None:
        composites["ppi_yoy"] = ppi_yoy

    inflation_components: list[float] = []
    if cpi_yoy is not None:
        inflation_components.append(cpi_yoy * 100.0)
    if pce_yoy is not None:
        inflation_components.append(pce_yoy * 100.0)
    if ppi_yoy is not None:
        inflation_components.append(ppi_yoy * 100.0)

    inflation_momentum = _average(inflation_components)
    if inflation_momentum is not None:
        composites["inflation_momentum"] = inflation_momentum

    if growth_momentum is not None and inflation_momentum is not None:
        stagflation = 1.0 if (inflation_momentum > 3.0 and growth_momentum < 0.0) else 0.0
        goldilocks = 1.0 if (1.0 <= growth_momentum <= 3.0 and inflation_momentum < 2.5) else 0.0
        composites["stagflation_risk"] = stagflation
        composites["goldilocks_score"] = goldilocks

    dollar_strength = _current("DTWEXBGS")
    if dollar_strength is not None:
        composites["dollar_strength"] = dollar_strength
        prior_dollar = _prior_value("DTWEXBGS", 3)
        if prior_dollar is not None and prior_dollar != 0.0:
            composites["dollar_momentum_3m"] = (dollar_strength - prior_dollar) / prior_dollar
        else:
            issues.add(("DTWEXBGS", "zero_division"))

    fx_pairs = ["DEXUSAL", "DEXUSEU", "DEXJPUS"]
    fx_vols: list[float] = []
    for pair in fx_pairs:
        std = _returns_std(pair, 30)
        if std is not None:
            fx_vols.append(std)

    fx_volatility = _average(fx_vols)
    if fx_volatility is not None:
        composites["fx_volatility_composite"] = fx_volatility
        composites["fx_stress"] = fx_volatility

    return composites, issues
