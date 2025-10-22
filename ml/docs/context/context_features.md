# Feature Engineering Context Document

**Last Updated**: 2025-10-19
**Module Size**: ~8,500 lines across 12 primary files + 4 subdirectories
**Status**: Production-ready core with advanced macro/micro extensions

---

## Executive Summary

The `ml/features/` directory implements a multi-layered feature engineering system that provides batch/online parity for L1 (OHLCV), L2 (order book), and L3 (trade) market microstructure features, plus macroeconomic features with vintage-aware point-in-time semantics. The system is designed for low-latency real-time inference (<5ms P99) while maintaining mathematical identity with batch training computations.

**Core Architecture**:
1. **FeatureEngineer** (engineering.py, 3,296 lines): Technical indicators and price-based features with hot/cold path separation
2. **MacroFeatureTransform** (macro_transforms.py, 795 lines): ALFRED vintage-aware macro features with real-time caching
3. **MacroDataCache** (macro_cache.py, 446 lines): Fast O(1) lookup cache for latest macro releases and revisions
4. **MacroComposites** (macro_composites.py, 523 lines): 37 composite features across credit, duration, liquidity, growth, and FX dimensions
5. **Microstructure Aggregators** (micro_aggregate.py, l2_aggregate.py): Per-minute L1/L2 feature aggregation
6. **Validation System** (validation.py, 693 lines): Parity validation with <1e-10 tolerance
7. **Pipeline Framework** (pipeline.py, 859 lines): Declarative transform catalog with data requirements gating

**Key Differentiators**:
- **Vintage-Aware Macro Features**: Point-in-time macro data using ALFRED vintages prevents look-ahead bias
- **Hot/Cold Path Parity**: Identical computation paths guarantee training/inference consistency
- **Composite Features**: Pre-computed multi-series composites (credit spreads, term structure, regime indicators)
- **Zero-Allocation Hot Path**: Pre-allocated buffers with numpy views for real-time performance

---

## Module Structure

```
ml/features/
├── engineering.py              # Core feature engineering (3,296 lines)
├── validation.py              # Parity validation system (693 lines)
├── pipeline.py                # Declarative pipeline framework (859 lines)
├── macro_cache.py             # Real-time macro data cache (446 lines)
├── macro_transforms.py        # Macro feature transforms (795 lines)
├── macro_composites.py        # Composite macro features (523 lines)
├── micro_aggregate.py         # L1 microstructure aggregation (162 lines)
├── l2_aggregate.py            # L2 order book aggregation (242 lines)
├── microstructure.py          # Advanced L2/L3 features (966 lines)
├── materialize_cli.py         # Feature materialization CLI (122 lines)
├── feature_export.py          # Registry integration (52 lines)
├── __init__.py                # Public API exports (339 lines)
├── cross_asset/               # Cross-asset relationship features
│   ├── beta.py                # EWMA beta computation
│   ├── spreads.py             # Z-score spread features
│   ├── correlation.py         # Rolling correlation
│   └── state.py               # Serializable state dataclasses
├── earnings/                  # Earnings-based fundamental features
│   ├── earnings_features.py   # EPS surprise, growth, momentum
│   └── earnings_transforms.py # Transform specs for earnings
└── README_*.md                # Documentation (macro features, earnings)
```

---

## 1. Core Feature Engineering (`engineering.py`)

### FeatureEngineer Class

**Location**: `/home/nate/projects/nautilus_trader/ml/features/engineering.py:617-2900`

The primary feature computation engine implementing hot/cold path separation for L1 OHLCV features.

#### Architecture

**NOT** integrated with Universal ML Architecture Pattern. The class:
- Does NOT inherit from `BaseMLInferenceActor`
- Does NOT automatically initialize 4 stores or 4 registries
- Takes optional `feature_store: FeatureStoreProtocol | None = None` parameter (line 717)
- Operates as a standalone feature computation library

```python
class FeatureEngineer:
    """Core feature engineering with batch/online parity."""

    def __init__(
        self,
        config: FeatureConfig | None = None,
        metrics_collector: FeatureEngineeringCollector | None = None,
        feature_store: FeatureStoreProtocol | None = None,
    ) -> None:
        self.config = config or FeatureConfig()
        self._metrics = metrics_collector  # Optional metrics
        self.feature_store = feature_store  # Optional store integration

        # Pre-allocate feature buffer for hot path
        spec = self.build_pipeline_spec_from_config()
        n_features = len(PipelineRunner(spec, allowable=...).compute_feature_names())
        buffer_size = n_features + SystemConstants.FEATURE_BUFFER_PAD
        self.feature_buffer = np.zeros(buffer_size, dtype=np.float32)
```

#### Key Methods

**Batch Processing** (lines 900-1100):
```python
def calculate_features(
    self,
    df: DataFrameLike,
    mode: Literal["batch", "online"] = "batch",
    fit_scaler: bool = False,
    scaler: StandardScalerT | None = None,
) -> tuple[PolarsDF | PandasDF, StandardScalerT | None]:
    """
    Compute features for batch (training) or online (inference) mode.

    Batch mode processes entire DataFrame sequentially using same online
    computation path for each row to guarantee parity.
    """
```

**Online Processing** (lines 1300-1450):
```python
def calculate_features_online(
    self,
    current_bar: Bar | None = None,
    indicator_manager: IndicatorManager | None = None,
    scaler: StandardScalerT | None = None,
    *,
    close_price: float | None = None,
    high_price: float | None = None,
    low_price: float | None = None,
    volume: float | None = None,
) -> npt.NDArray[np.float32]:
    """
    Hot path feature computation with zero allocations.

    Returns view of pre-allocated buffer (caller must copy if persisting).
    """
```

#### Feature Categories

**Price-Based Features** (L1_ONLY):
- **Returns**: Configurable periods (default: [1, 5, 10, 20]), safe division
- **Momentum**: Price change over lookback periods ([5, 10, 20])
- **Volatility**: Rolling std dev (5-bar and 20-bar windows)
- **HL Spread**: Normalized (high - low) / close

**Volume Features** (L1_ONLY):
- **Volume Ratios**: Current volume vs SMA([5, 10, 20]), safe division

**Technical Indicators** (L1_ONLY - lines 1900-2500):
- **RSI**: Normalized to [-1, 1] from Nautilus [0, 1], bounds-checked
- **RSI Overbought/Oversold**: Binary signals at 70/30 thresholds
- **Bollinger Bands**: Width (normalized) and position [0, 1]
- **ATR**: Normalized by price with 1e-6 floor to prevent extreme ratios
- **EMA Fast/Slow**: Distance and cross features
- **MACD**: Line, signal, difference (all price-normalized)
- **Price Position 20**: Location within 20-day high-low range

**Microstructure Features** (L1_L2 - when enabled):
- Simplified OHLCV approximations in hot path
- Full L2 order book features in batch path
- See microstructure.py for advanced L2/L3 features

### FeatureConfig Class

**Location**: `/home/nate/projects/nautilus_trader/ml/features/engineering.py:98-280`

Configuration with comprehensive validation in `__post_init__`:

```python
@dataclass(kw_only=True, frozen=True)
class FeatureConfig(MLFeatureConfig):
    # Price features
    return_periods: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    momentum_periods: list[int] = field(default_factory=lambda: [5, 10, 20])

    # Technical indicators (validated ranges in __post_init__)
    rsi_period: int = 14        # [2, 100]
    bb_period: int = 20         # [2, 100]
    bb_std: float = 2.0         # [0.5, 5.0]
    atr_period: int = 20        # [2, 100]
    ema_fast: int = 12          # [2, 50]
    ema_slow: int = 26          # [10, 200], must be > ema_fast
    macd_signal: int = 9        # [2, 50]

    # Volume features
    volume_ma_periods: list[int] = field(default_factory=lambda: [5, 10, 20])

    # Advanced features
    include_microstructure: bool = False
    include_trade_flow: bool = False
    validate_quality: bool = False

    # Data requirements gating
    data_requirements: DataRequirements = DataRequirements.L1_ONLY
```

**Validation** (lines 180-250):
- Range checking for all indicator periods
- Dependency validation (ema_slow > ema_fast)
- Raises `ValueError` on invalid configuration

### IndicatorManager Class

**Location**: `/home/nate/projects/nautilus_trader/ml/features/engineering.py:428-615`

Manages stateful Nautilus indicators for consistent hot/cold path calculations:

```python
class IndicatorManager:
    """Manages Nautilus indicators with bounded state."""

    def __init__(self, config: FeatureConfig) -> None:
        # Initialize indicators (RSI, BB, ATR, EMA, MACD)
        self._price_history: deque[float] = deque(maxlen=PRICE_HISTORY_MAXLEN)

    def update_from_bar(self, bar: Bar) -> None:
        """Update indicators from Nautilus Bar object."""

    def update_from_values(
        self, close: float, high: float, low: float, volume: float
    ) -> None:
        """Update indicators from raw OHLCV values (hot path)."""
```

**Memory Management**:
- Bounded history: `PRICE_HISTORY_MAXLEN = 1000` (line 35)
- Automatic trimming prevents OOM in long-running processes
- Indicators maintain internal state across updates

---

## 2. Macro Feature System

### MacroFeatureTransform

**Location**: `/home/nate/projects/nautilus_trader/ml/features/macro_transforms.py:42-419`

Provides vintage-aware macro features with training/inference parity using ALFRED vintages and FRED data.

```python
class MacroFeatureTransform:
    """
    Transform adding ALFRED/FRED macro features with point-in-time semantics.

    Ensures macro features are computed identically in batch (historical) and
    real-time (inference) modes to prevent look-ahead bias.
    """

    def __init__(
        self,
        macro_series_ids: list[str],          # e.g., ["PAYEMS", "UNRATE", "CPIAUCSL"]
        vintage_base_dir: Path | str,         # data/fred/vintages/
        fred_path: Path | str | None = None,
        include_revisions: bool = False,
        revision_mode: Literal["minimal", "core", "full"] = "core",
        lag_days: int = 1,
        vintage_policy: VintagePolicy = VintagePolicy.REAL_TIME,
        include_composites: bool = False,
        composite_history_window: int = 400,
    ) -> None:
        # Real-time cache (lazy-loaded)
        self._cache: MacroDataCache | None = None
```

**Key Methods**:

**Batch Computation** (lines 148-225):
```python
def compute_batch(
    self,
    df: PolarsDataFrame,
    timestamp_col: str = "timestamp",
    vintage_cutoff: datetime | None = None,
) -> PolarsDataFrame:
    """
    Compute macro features for batch (historical) data.

    Uses join_fred_asof to apply point-in-time vintage logic, ensuring
    features use only data available at each timestamp (no look-ahead bias).
    """
    from ml.data.fred_join import join_fred_asof

    result = join_fred_asof(
        df,
        timestamp_col=timestamp_col,
        lag_days=self.lag_days,
        vintage_base_dir=self.vintage_base_dir,
        series_filter=set(self._series_ids_for_batch),
        vintage_policy=self.vintage_policy,
        include_revisions=self.include_revisions,
        revision_mode=self.revision_mode,
    )
```

**Real-Time Computation** (lines 227-278):
```python
def compute_realtime(
    self,
    bar: Bar | None = None,
    ts_event: int | None = None,
) -> dict[str, float]:
    """
    Compute macro features for real-time inference.

    Uses cached latest values - no point-in-time filtering needed since
    we're always at "now".
    """
    cache = self._get_cache()
    features = cache.get_all_features(mode=self.revision_mode)

    if self.include_composites:
        composites, issues = _compute_realtime_composites(cache)
        features.update(composites)

    return features
```

**Feature Modes**:
- **minimal**: current, prior_1m, revision_1m
- **core**: + mom_1m, pct_1m, net_signal_1m
- **full**: + prior_3m/12m, revision_3m, mom_3m/12m, pct_12m

**Metrics Integration** (lines 329-355):
```python
def _record_composite_issues(self, issues: Iterable[tuple[str, str]]) -> None:
    """Record composite computation issues via metrics/logging."""
    if self._composite_issue_metric is None:
        self._composite_issue_metric = get_counter(
            "ml_macro_composite_missing_total",
            "Count of macro composite computations missing prerequisites",
            labelnames=("series_id", "reason"),
        )
```

**IMPORTANT**: This is the ONLY metrics usage in the entire features module. The transform uses `get_counter` from `ml.common.metrics_bootstrap` (line 18), demonstrating proper metrics bootstrap pattern.

### MacroDataCache

**Location**: `/home/nate/projects/nautilus_trader/ml/features/macro_cache.py:102-446`

Fast O(1) lookup cache for real-time macro feature access with <1ms P99 latency.

```python
@dataclass(slots=True)
class MacroDataCache:
    """
    Fast cache for real-time macro feature access.

    Pre-loads ALFRED vintages on initialization and provides O(1) lookups
    for latest values, prior periods, and revisions.
    """

    vintage_base_dir: Path
    series_ids: list[str]
    enable_revisions: bool = True
    aux_series_ids: list[str] = field(default_factory=list)
    history_window: int = 400  # For composite calculations
    _snapshots: dict[str, MacroSeriesSnapshot] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Load all vintages on initialization."""
        self.refresh()
```

**Data Structures**:

```python
@dataclass(slots=True)
class MacroSeriesSnapshot:
    """Snapshot of macro series current state for real-time inference."""
    series_id: str
    current_value: float
    observation_ts: datetime
    release_ts: datetime
    prior_1m_value: float | None = None
    prior_3m_value: float | None = None
    prior_12m_value: float | None = None
    revision_1m: float | None = None
    revision_3m: float | None = None
    initial_value: float | None = None
    history: tuple[float, ...] = ()  # For composite rolling calculations
```

**Key Methods** (lines 160-322):
- `_load_series_snapshot()`: Load release calendar from parquet, compute revisions
- `_get_prior_value()`: Get value from N months ago (latest release for that observation)
- `_compute_cumulative_revisions()`: Sum revisions over trailing months
- `get_snapshot()`: O(1) lookup of cached snapshot
- `get_features()`: Generate all features for a series (minimal/core/full mode)

**Revision Computation** (lines 214-229):
```python
if self.enable_revisions and prior_1m is not None:
    # Revision = (current value for prior obs) - (initial value for prior obs)
    relativedelta = _load_relativedelta()
    prior_obs_ts = observation_ts - relativedelta(months=1)
    prior_initial = self._get_initial_value(df, prior_obs_ts, _pl=_pl)
    if prior_initial is not None:
        revision_1m = prior_1m - prior_initial
```

### MacroComposites

**Location**: `/home/nate/projects/nautilus_trader/ml/features/macro_composites.py`

Provides 37 composite features across economic dimensions:

**Credit/Risk Spreads (9 features)**:
- `credit_spread_ig`: BAMLC0A0CM (investment grade)
- `credit_spread_hy`: BAMLH0A0HYM2 (high yield)
- `credit_spread_hy_ig`: HY - IG (quality premium)
- `credit_spread_bbb_a`: BBB-A quality spread
- `credit_spread_ig_momentum`: 3-month IG change
- `credit_spread_hy_momentum`: 3-month HY change
- `credit_distress_index`: Composite distress indicator
- `ted_spread`: TEDRATE (bank funding stress)
- `credit_risk_index`: Composite of spreads + VIX

**Duration/Term Structure (8 features)**:
- `term_spread`: T10Y2Y or (DGS10 - DGS2)
- `term_spread_5s30s`: Long-end slope
- `term_spread_2s30s`: Full curve slope
- `curve_curvature`: Butterfly (2*DGS10 - DGS2 - DGS30)
- `real_yield_10y`: DFII10 (TIPS)
- `real_term_premium`: DGS10 - DFII10
- `yield_curve_slope`: (10y - 2y) / 2y
- `fed_policy_stance`: FEDFUNDS relative to 10y

**Liquidity/Funding (6 features)**:
- `fed_balance_sheet`: WALCL
- `qe_intensity`: WALCL / 1M (normalized)
- `bank_credit_growth_3m`: TOTBKCR change
- `sofr_obfr_spread`: Repo market stress
- `financial_stress_composite`: Multi-indicator stress
- `liquidity_index`: Composite of WALCL, TOTBKCR, TEDRATE

**Growth/Inflation Regime (9 features)**:
- `payems_mom`, `indpro_mom`: Employment and production growth
- `growth_momentum`: Composite of PAYEMS, INDPRO, CFNAI
- `cpi_yoy`, `pce_yoy`, `ppi_yoy`: Inflation measures
- `inflation_momentum`: Composite inflation
- `stagflation_risk`: High inflation + weak growth (binary)
- `goldilocks_score`: Moderate growth + low inflation (binary)

**FX Positioning (4 features)**:
- `dollar_strength`: DTWEXBGS (broad dollar index)
- `dollar_momentum_3m`: 3-month USD change
- `fx_volatility_composite`: Cross-pair volatility
- `fx_stress`: Volatility across major pairs

**Batch Implementation** (lines 26-432):
```python
def compute_macro_composites_pl(df: _pl.DataFrame) -> _pl.DataFrame:
    """
    Compute composite macro features from base series.

    Uses Polars expressions for high-performance batch computation.
    """
    composites = df.clone()

    # Credit risk index: Average of normalized spreads + VIX
    credit_components = []
    if "BAMLC0A0CM" in df.columns:
        credit_components.append(pl.col("BAMLC0A0CM") / 100.0)
    if "BAMLH0A0HYM2" in df.columns:
        credit_components.append(pl.col("BAMLH0A0HYM2") / 500.0)
    # ... more components

    if credit_components:
        composites = composites.with_columns(
            pl.mean_horizontal(*credit_components).alias("credit_risk_index"),
        )
```

**Real-Time Implementation** (lines 474-795):
```python
def _compute_realtime_composites(
    cache: MacroDataCache,
) -> tuple[dict[str, float], set[tuple[str, str]]]:
    """Compute macro composite features using cached snapshots."""
    composites: dict[str, float] = dict.fromkeys(get_composite_feature_names(), math.nan)
    issues: set[tuple[str, str]] = set()

    # Helper functions for safe computation
    def _current(series_id: str) -> float | None:
        snapshot = cache.get_snapshot(series_id)
        return snapshot.current_value if snapshot else None

    def _safe_divide(num: float, denom: float, series_id: str, reason: str) -> float | None:
        if denom == 0.0:
            issues.add((series_id, reason))
            return None
        return num / denom

    # Compute composites with error tracking
    ig = _current("BAMLC0A0CM")
    if ig is not None:
        composites["credit_spread_ig"] = ig
```

**Required Series** (lines 490-523):
```python
def get_composite_series_requirements() -> tuple[str, ...]:
    """Return base macro series required for composite features."""
    return (
        "BAMLC0A0CM", "BAMLH0A0HYM2", "TEDRATE", "VIXCLS",
        "T10Y2Y", "DGS10", "DGS2", "DGS5", "DGS30", "DFII10",
        "FEDFUNDS", "WALCL", "TOTBKCR",
        "PAYEMS", "INDPRO", "CFNAI",
        "CPIAUCSL", "PCEPI", "PPIACO",
        "DTWEXBGS", "DEXUSAL", "DEXUSEU", "DEXJPUS",
    )
```

---

## 3. Microstructure Features

### Micro Aggregation (L1)

**Location**: `/home/nate/projects/nautilus_trader/ml/features/micro_aggregate.py:42-162`

Per-minute aggregation from L1 quotes and trades with Polars optimization:

```python
def aggregate_microstructure_minute_pl(
    quotes: PolarsDF | None,
    trades: PolarsDF | None,
    *,
    timestamp_col: str = "ts_event",
    bid_col: str = "bid_px_00",
    ask_col: str = "ask_px_00",
    bid_sz_col: str = "bid_sz_00",
    ask_sz_col: str = "ask_sz_00",
) -> PolarsDF:
    """
    Aggregate L1 microstructure features to per-minute bars.

    Returns DataFrame with columns:
    - midprice: (bid + ask) / 2 averaged per minute
    - spread_bps: ((ask - bid) / midprice) * 10000 in basis points
    - quote_imbalance: (bid_size - ask_size) / (bid_size + ask_size)
    - trade_imbalance: Buy/sell volume imbalance from trade signs
    - realized_vol: High-frequency volatility from trade price movements
    """
```

**Features Computed** (lines 21-27):
```python
MICRO_COLUMNS = [
    "midprice",
    "spread_bps",
    "quote_imbalance",
    "trade_imbalance",
    "realized_vol",
]
```

**Safe Division Pattern** (lines 63-65):
```python
denom = (pl.col(bid_sz_col) + pl.col(ask_sz_col)).cast(pl.Float64)
denom_safe = pl.when(denom > 0).then(denom).otherwise(1.0)
quote_imbalance = (pl.col(bid_sz_col) - pl.col(ask_sz_col)) / denom_safe
```

**Aggregator Class** (lines 118-162):
```python
@dataclass(slots=True)
class MicrostructureAggregator:
    base_dir: Path

    def compute_for_symbol(self, symbol: str) -> PolarsDF:
        """Load quotes/trades and compute per-minute features."""
        q = self._load_l1_quotes(symbol)
        t = self._load_l1_trades(symbol)
        return aggregate_microstructure_minute_pl(q, t)
```

### L2 Aggregation (Order Book Depth)

**Location**: `/home/nate/projects/nautilus_trader/ml/features/l2_aggregate.py:32-170`

Multi-level depth features from MBP-10 snapshots:

```python
def aggregate_l2_minute_pl(
    l2: PolarsDF,
    *,
    timestamp_col: str = "ts_event",
) -> PolarsDF:
    """
    Aggregate L2 order book depth to per-minute features.

    Computes depth_imbalance, dwp_bps, bid_slope, ask_slope for top K levels
    where K ∈ {1, 3, 5, 10}.
    """
```

**Features per Level** (lines 40-60):
- `depth_imbalance_topK`: (bid_qty - ask_qty) / (bid_qty + ask_qty) across top K
- `dwp_bps_topK`: Depth-weighted price deviation from mid in bps
- `bid_slope_topK`: (p_{K-1} - p_0) / (K-1) price slope approximation
- `ask_slope_topK`: (p_{K-1} - p_0) / (K-1) price slope approximation

**Safe Division** (lines 80-85):
```python
def _safe_div(numer: pl.Expr, denom: pl.Expr) -> pl.Expr:
    return numer / pl.when(denom > 0).then(denom).otherwise(1.0)

def _slope_approx(p0: pl.Expr, pk: pl.Expr, k: int) -> pl.Expr:
    return (pk - p0) / max(k - 1, 1)  # Prevent division by zero
```

**Computation** (lines 90-150):
```python
for k in TOPKS:  # [1, 3, 5, 10]
    bid_qty_top_k = sum(pl.col(f"bid_sz_{i:02d}") for i in range(k))
    ask_qty_top_k = sum(pl.col(f"ask_sz_{i:02d}") for i in range(k))

    depth_imb = _safe_div(
        bid_qty_top_k - ask_qty_top_k,
        bid_qty_top_k + ask_qty_top_k,
    )

    # Depth-weighted price
    bid_dwp = sum(
        pl.col(f"bid_px_{i:02d}") * pl.col(f"bid_sz_{i:02d}")
        for i in range(k)
    ) / bid_qty_top_k
```

---

## 4. Validation System

### FeatureParityValidator

**Location**: `/home/nate/projects/nautilus_trader/ml/features/validation.py:70-690`

Comprehensive validation ensuring mathematical identity between batch and online computation:

```python
class FeatureParityValidator:
    """
    Validates feature parity between batch and real-time computation.

    Critical for ML model performance - even small discrepancies can cause
    model failure in production.
    """

    def __init__(
        self,
        config: FeatureConfig | None = None,
        tolerance: float | None = None,
    ) -> None:
        self.config = config or FeatureConfig()
        self.tolerance = tolerance or MLConstants.FEATURE_PARITY_TOLERANCE  # 1e-10
        self.feature_engineer = FeatureEngineer(self.config)
```

**Validation Process** (lines 200-450):

1. **Prepare Data**: Extract OHLCV arrays from DataFrame
2. **Batch Computation**: Process entire DataFrame sequentially
3. **Online Computation**: Process same data row-by-row with indicator warmup
4. **Comparison**: Validate all features within tolerance
5. **Reporting**: Generate detailed parity report with failing features

**Key Validation Method**:
```python
def validate_parity(
    self,
    df: DataFrameLike,
    start_idx: int = 50,  # Skip initial warmup
    end_idx: int | None = None,
) -> dict[str, Any]:
    """
    Validate feature parity between batch and online computation.

    Returns
    -------
    dict[str, Any]
        {
            "parity_passed": bool,
            "max_difference": float,
            "tolerance": float,
            "failing_features": list[str],
            "feature_differences": dict[str, float],
            "validation_time": float,
            "n_samples_validated": int,
        }
    """
```

**Parity Report Structure** (lines 400-420):
```python
{
    "parity_passed": True,
    "max_difference": 1e-12,
    "tolerance": 1e-10,
    "failing_features": [],
    "feature_differences": {
        "rsi": 5e-13,
        "bb_width": 3e-12,
        # ... per-feature max differences
    },
    "validation_time": 1.234,
    "n_samples_validated": 950,
    "parity_details": {
        # Extended diagnostics
    }
}
```

---

## 5. Pipeline Framework

### Declarative Transform System

**Location**: `/home/nate/projects/nautilus_trader/ml/features/pipeline.py`

Provides declarative feature definition with automatic schema computation and data requirements gating.

**Core Protocol** (lines 25-38):
```python
class FeatureTransform(Protocol):
    """Protocol for feature transform plugins."""

    name: str

    def feature_names(self, params: Mapping[str, Any]) -> list[str]: ...

    def requires(self) -> DataRequirements:
        """Return required data level for this transform (used for gating)."""
        ...
```

**Data Requirements Hierarchy** (from `ml/registry/base.py`):
```python
class DataRequirements(Enum):
    L1_ONLY = "L1_ONLY"        # OHLCV bars only
    L1_L2 = "L1_L2"            # + Order book snapshots (MBP-10)
    L1_L2_L3 = "L1_L2_L3"      # + Individual trade records
```

**Core Transforms** (lines 41-112):
```python
_CATALOG: dict[str, FeatureTransform] = {
    "returns": _ReturnsTransform(),           # Configurable periods
    "momentum": _MomentumTransform(),         # Momentum indicators
    "volatility": _VolatilityTransform(),     # Fixed: vol_5, vol_20
    "volume_ratio": _VolumeRatioTransform(),  # Volume vs MA ratios
    "core_indicators": _CoreIndicatorsTransform(),  # RSI, BB, ATR, EMA, MACD
}
```

**Core Indicators Output** (lines 88-108):
```python
class _CoreIndicatorsTransform:
    def feature_names(self, params: Mapping[str, Any]) -> list[str]:
        return [
            "rsi", "rsi_overbought", "rsi_oversold",  # RSI features (3)
            "bb_width", "bb_position",                 # Bollinger Bands (2)
            "atr_normalized",                          # ATR (1)
            "ema_fast_dist", "ema_slow_dist", "ema_cross",  # EMA (3)
            "macd_line", "macd_signal", "macd_difference",  # MACD (3)
            "price_position_20", "hl_spread",          # Price position (2)
        ]  # Total: 14 features
```

**Advanced Transforms** (lines 128-200):
```python
register_transform(_KeltnerTransform())      # Keltner channels (L1_L2)
register_transform(_OBVTransform())          # On-Balance Volume (L1_L2)
register_transform(_MicrostructureTransform())  # L2 order book features
register_transform(_TradeFlowTransform())    # L3 trade flow features
```

**TFT Transforms** (lines 220-400):
```python
register_transform(_CalendarTransform())     # Time-based cyclical features
register_transform(_EventScheduleTransform())  # Earnings, Fed, expiry
register_transform(_MacroIndicatorsTransform())  # VIX, DXY, yields
register_transform(_StaticCovariatesTransform())  # Instrument metadata
```

**Pipeline Spec** (lines 450-550):
```python
@dataclass(frozen=True)
class PipelineSpec:
    transforms: list[TransformSpec]

@dataclass(frozen=True)
class TransformSpec:
    name: str
    params: dict[str, Any]

class PipelineRunner:
    def __init__(self, spec: PipelineSpec, allowable: DataRequirements):
        # Filter transforms based on data requirements
        self.transforms = self._filter_transforms(spec.transforms, allowable)

    def compute_feature_names(self) -> list[str]:
        """Generate ordered feature names from transforms."""
        names = []
        for transform_spec in self.transforms:
            transform = _CATALOG[transform_spec.name]
            names.extend(transform.feature_names(transform_spec.params))
        return names

    def compute_signature(self) -> str:
        """Generate SHA-256 signature for schema versioning."""
        content = msgspec.json.encode(self.transforms).decode('utf-8')
        return hashlib.sha256(content.encode()).hexdigest()
```

**Builder Function** (lines 600-700):
```python
def build_pipeline_spec_from_feature_config(cfg: FeatureConfig) -> PipelineSpec:
    """Build pipeline specification from feature configuration."""
    transforms = [
        # Core L1_ONLY transforms (always available)
        TransformSpec(name="returns", params={"periods": cfg.return_periods}),
        TransformSpec(name="momentum", params={"periods": cfg.momentum_periods}),
        TransformSpec(name="volatility", params={}),
        TransformSpec(name="volume_ratio", params={"periods": cfg.volume_ma_periods}),
        TransformSpec(name="core_indicators", params={}),
    ]

    # Advanced features gated by configuration
    if cfg.include_microstructure:
        transforms.append(TransformSpec(name="microstructure", params={}))
    if cfg.include_trade_flow:
        transforms.append(TransformSpec(name="trade_flow", params={}))

    return PipelineSpec(transforms=transforms)
```

---

## 6. Cross-Asset & Earnings Features

### Cross-Asset Relationship Features

**Location**: `/home/nate/projects/nautilus_trader/ml/features/cross_asset/`

Provides beta, correlation, and spread features with hot/cold path parity.

**Modules**:
- `beta.py`: EWMA beta computation (incremental and batch)
- `correlation.py`: Rolling correlation (incremental and batch)
- `spreads.py`: Z-score spread features (incremental and batch)
- `state.py`: Serializable state dataclasses

**Example - EWMA Beta**:
```python
# Hot path (O(1) incremental)
from ml.features.cross_asset import EWMABetaState, compute_ewma_beta_incremental

state = EWMABetaState(alpha=0.94)
for asset_return, market_return in zip(asset_returns, market_returns):
    beta = compute_ewma_beta_incremental(state, asset_return, market_return)

# Cold path (vectorized batch)
from ml.features.cross_asset import compute_ewma_beta_batch

betas = compute_ewma_beta_batch(asset_returns, market_returns, alpha=0.94)
```

**Features**:
- **EWMA Beta**: Exponentially weighted beta with Welford's algorithm
- **Rolling Correlation**: Incremental correlation updates
- **Z-Score Spreads**: Statistical spread analysis with rolling mean/std

**Performance**:
- P99 latency < 5ms (hot path)
- O(1) computational complexity
- Zero allocations after warmup

### Earnings Features

**Location**: `/home/nate/projects/nautilus_trader/ml/features/earnings/`

Corporate fundamentals integration with hot/cold path parity.

**Modules**:
- `earnings_features.py`: Core earnings computations
- `earnings_transforms.py`: Transform specs for pipeline integration

**Features**:
- **EPS Surprise**: Dollar and percentage surprise
- **YoY/QoQ Growth**: Year-over-year and quarter-over-quarter EPS growth
- **Beat Streak**: Consecutive quarters beating consensus
- **EPS Volatility**: 4-quarter coefficient of variation
- **Days to Earnings**: Calendar feature for earnings dates

**Example**:
```python
# Hot path
from ml.features.earnings import compute_earnings_surprise_incremental

surprise = compute_earnings_surprise_incremental(actual=2.52, estimate=2.45)

# Cold path
from ml.features.earnings import compute_earnings_surprise_batch

surprises = compute_earnings_surprise_batch(actuals, estimates)
```

---

## 7. Integration Patterns

### Actor Integration

**Location**: `/home/nate/projects/nautilus_trader/ml/actors/signal.py:400-600`

ML actors use FeatureEngineer for real-time feature computation:

```python
class MLSignalActor(BaseMLInferenceActor):
    def __init__(self, config: MLSignalActorConfig) -> None:
        super().__init__(config)

        # Initialize feature engineer with config
        self._feature_engineer = FeatureEngineer(
            config=config.feature_config,
            metrics_collector=None,  # Actor handles metrics separately
            feature_store=self.feature_store,  # From BaseMLInferenceActor
        )

    def _on_bar(self, bar: Bar) -> None:
        """Hot path: process bar and generate signal."""
        # Update indicator state
        self._indicator_manager.update_from_bar(bar)

        # Compute features (hot path - zero allocations)
        features = self._feature_engineer.calculate_features_online(
            current_bar=bar,
            indicator_manager=self._indicator_manager,
            scaler=self._scaler,
        )

        # Run model inference
        prediction = self._model_session.run(None, {"input": features})[0]

        # Generate signal
        signal = self._signal_strategy.generate_signal(prediction, features, bar)
```

### TFT Dataset Builder Integration

**Location**: Inferred from macro_transforms.py usage

The TFTDatasetBuilder integrates macro features via `MacroFeatureTransform`:

```python
# In TFTDatasetBuilder
if self.macro_series_ids:
    self._macro_transform = MacroFeatureTransform(
        macro_series_ids=self.macro_series_ids,
        vintage_base_dir=self.vintage_base_dir,
        fred_path=self.fred_path,
        include_revisions=True,
        revision_mode="core",
        include_composites=True,
    )

# During dataset building
df_with_macro = self._macro_transform.compute_batch(
    df,
    timestamp_col="timestamp",
    vintage_cutoff=None,  # Use all available vintages
)
```

### Feature Store Integration

**Pattern** (from engineering.py:717):
```python
class FeatureEngineer:
    def __init__(self, ..., feature_store: FeatureStoreProtocol | None = None):
        self.feature_store = feature_store  # Optional integration

    # Feature store is NOT used within FeatureEngineer
    # Integration happens at actor/orchestrator level
```

**Reality**: FeatureEngineer does NOT automatically persist to FeatureStore. Integration is manual:

```python
# Actors must explicitly persist if desired
features = feature_engineer.calculate_features_online(...)
if self.feature_store:
    self.feature_store.write_features(
        instrument_id=bar.instrument_id,
        ts_event=bar.ts_event,
        features=features.copy(),  # Must copy - features is a view
    )
```

---

## 8. Performance Characteristics

### Hot Path Optimizations

**Pre-Allocation Strategy** (engineering.py:750-800):
```python
# Dynamic buffer sizing based on pipeline requirements
spec = self.build_pipeline_spec_from_config()
allowable = DataRequirements.L1_L2 if (
    self.config.include_microstructure or self.config.include_trade_flow
) else DataRequirements.L1_ONLY
runner = PipelineRunner(spec, allowable=allowable)
n_features = len(runner.compute_feature_names())
buffer_size = n_features + SystemConstants.FEATURE_BUFFER_PAD
self.feature_buffer = np.zeros(buffer_size, dtype=np.float32)
```

**Zero-Copy Returns** (engineering.py:1400):
```python
def calculate_features_online(...) -> npt.NDArray[np.float32]:
    # Compute into pre-allocated buffer
    feature_idx = self._calculate_return_features(...)
    feature_idx = self._calculate_technical_indicator_features(...)
    # ... more features

    # Return view (zero allocation in hot path)
    return self.feature_buffer[:feature_idx]
```

**CRITICAL**: View requires explicit copying for persistence:
```python
# Safe persistence pattern
features = engineer.calculate_features_online(...)
if need_persistence:
    features_copy = features.copy()  # Explicit copy when needed
```

**Bounded Memory** (engineering.py:428-450):
```python
class IndicatorManager:
    def __init__(self, config: FeatureConfig) -> None:
        # Bounded history prevents OOM
        self._price_history: deque[float] = deque(maxlen=PRICE_HISTORY_MAXLEN)
        # PRICE_HISTORY_MAXLEN = 1000 (line 35)
```

### Numerical Stability

**Safe Division** (engineering.py:500-520):
```python
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Production-grade safe division with None/zero checking."""
    if denominator == 0 or denominator is None:
        return default
    return numerator / denominator
```

**ATR Normalization** (engineering.py:2200-2210):
```python
def _normalize_atr(atr: float, close: float) -> float:
    """Normalize ATR with floor to prevent extreme ratios."""
    ratio = safe_divide(float(atr), float(close), default=0.0)
    return 0.0 if ratio < 1e-6 else ratio  # Floor prevents extreme ratios
```

**RSI Normalization** (engineering.py:1950-1960):
```python
def _normalize_rsi(rsi_raw: float) -> float:
    """Convert RSI from [0,1] to [-1,1] for ML compatibility."""
    normalized = (rsi_raw - 0.5) * 2.0
    assert -1 <= normalized <= 1, f"RSI out of bounds: {normalized}"
    return normalized
```

### Macro Cache Performance

**O(1) Lookup** (macro_cache.py:324-334):
```python
def get_snapshot(self, series_id: str) -> MacroSeriesSnapshot | None:
    """O(1) lookup of cached snapshot."""
    return self._snapshots.get(series_id)

def get_features(self, series_id: str, mode: str = "core") -> dict[str, float]:
    """O(1) feature generation from cached snapshot."""
    snapshot = self._snapshots.get(series_id)
    if snapshot is None:
        return {}
    # Direct attribute access - no computation
```

**Lazy Loading** (macro_transforms.py:128-146):
```python
def _get_cache(self) -> MacroDataCache:
    """Get or create real-time cache (lazy initialization)."""
    if self._cache is None:
        self._cache = MacroDataCache(
            vintage_base_dir=self.vintage_base_dir,
            series_ids=self.macro_series_ids,
            enable_revisions=self.include_revisions,
            aux_series_ids=aux_series,
            history_window=self._composite_history_window,
        )
    return self._cache
```

---

## 9. Known Gaps and TODOs

### Incomplete Implementations

**From CODEX_RECOMMENDATIONS_STATUS.md**:

1. **Pipeline Integration** (TODO):
   - Macro transforms not yet integrated into PipelineSpec
   - No macro transform in build_pipeline_spec_from_feature_config
   - Manual addition required for TFT datasets

2. **Dataset Validation** (TODO):
   - Macro coverage validation exists but not enforced
   - No automated checks for minimum coverage thresholds
   - Manual validation required

3. **Contract-Level Covariates** (TODO):
   - Earnings date features not yet implemented
   - Expiry date features not yet implemented
   - Fed meeting dates not yet implemented

### Missing Metrics Integration

**Critical Finding**: Despite extensive documentation claiming metrics integration:
- **ZERO** metrics in `engineering.py` (3,296 lines)
- **ZERO** calls to `get_counter`, `get_histogram`, or `get_gauge`
- **ONLY** metrics usage: `macro_transforms.py:335` (composite issue counter)

**Missing Metrics**:
- Feature computation timers
- Parity validation success rates
- Hot path latency measurements
- Cache hit rates
- Buffer utilization

### Documentation vs Reality

**Overclaimed Features**:
- Universal Architecture compliance (NOT implemented in FeatureEngineer)
- 4-store + 4-registry integration (optional `feature_store` parameter only)
- Centralized metrics bootstrap (1 usage in entire features module)
- Progressive fallback chains (NOT implemented)

**Actual Status**:
- FeatureEngineer is standalone library, NOT integrated with Universal Architecture
- Store integration is manual at actor level
- Metrics are NOT collected automatically
- No progressive fallback - fails immediately on missing dependencies

---

## 10. Testing Strategy

### Test Coverage

**Unit Tests**:
- `ml/tests/unit/features/`: Core feature engineering tests
- `ml/tests/unit/macro/`: Macro feature transform tests
- `ml/tests/unit/validation/`: Parity validation tests

**Integration Tests**:
- `ml/tests/integration/features/`: Feature store integration
- `ml/tests/e2e/test_feature_store_e2e.py`: End-to-end feature persistence

**Property Tests**:
- Hypothesis-based parity validation
- Numerical stability tests
- Bounds checking for normalized features

**Contract Tests**:
- Feature schema validation
- Pipeline signature consistency
- Manifest compatibility

### Validation Reports

**Location**: `/home/nate/projects/nautilus_trader/ml/tests/validation_reports/`

Test artifacts include:
- `test_feature_parity.log`: Parity validation results
- `test_feature_store_integration.log`: Store integration logs
- `features.log`: General feature engineering test logs

---

## 11. CLI Tools

### Feature Materialization CLI

**Location**: `/home/nate/projects/nautilus_trader/ml/features/materialize_cli.py`

**Usage**:
```bash
# Reorder mode (default): reorder existing features to manifest order
python -m ml.features.materialize_cli \
    --feature_registry_dir ml/registry \
    --feature_set_id production_features_v3 \
    --input_csv data/features_raw.csv \
    --output_csv data/features_materialized.csv \
    --target_col target_15m

# From-OHLCV mode: compute features from OHLCV bars
python -m ml.features.materialize_cli \
    --feature_registry_dir ml/registry \
    --feature_set_id production_features_v3 \
    --input_csv data/market_data.csv \
    --output_csv data/features_materialized.csv \
    --from_ohlcv
```

**Modes**:
- **Reorder**: Reads CSV with existing features, outputs in manifest order
- **From-OHLCV**: Computes features from OHLCV using FeatureEngineer (best-effort)

**Output**:
- Features in exact manifest order
- Prepends `time_index` and `instrument_id` if present
- Appends target column if requested

---

## 12. Dependencies

### Required (Core)

- **numpy**: Array operations and pre-allocation
- **msgspec**: Configuration serialization
- **nautilus_trader**: Technical indicators (RSI, BB, ATR, EMA, MACD)

### Optional (Enhanced)

- **polars**: High-performance DataFrame operations (macro/micro aggregation)
- **pandas**: Fallback DataFrame operations
- **scikit-learn**: Feature scaling (StandardScaler)
- **python-dateutil**: Date arithmetic for macro revisions

### Metrics (Minimal Usage)

- **prometheus_client**: Metrics (via `ml.common.metrics_bootstrap`)
  - Used ONLY in `macro_transforms.py:335` for composite issue counter
  - NOT used in core FeatureEngineer despite documentation claims

---

## 13. Key Design Patterns

### 1. Hot/Cold Path Separation

**Implementation**:
- Batch: Sequential processing using same online computation path
- Online: Pre-allocated buffers with numpy views
- Parity: Guaranteed by shared computation core

**NOT** a Universal Architecture Pattern as claimed - just standard batch/online optimization.

### 2. Safe Numerical Operations

**Pattern**:
```python
def safe_divide(num: float, denom: float, default: float = 0.0) -> float:
    if denom == 0 or denom is None:
        return default
    return num / denom

def _normalize_atr(atr: float, close: float) -> float:
    ratio = safe_divide(float(atr), float(close), default=0.0)
    return 0.0 if ratio < 1e-6 else ratio  # Floor extreme ratios
```

**Applied to**:
- Volume ratios
- Price normalization
- ATR normalization
- Bollinger Band position
- All division operations

### 3. Lazy Initialization

**MacroDataCache** (macro_cache.py:130):
```python
def __post_init__(self) -> None:
    """Load all vintages on initialization."""
    self.refresh()
```

**MacroFeatureTransform** (macro_transforms.py:128):
```python
def _get_cache(self) -> MacroDataCache:
    """Get or create real-time cache (lazy initialization)."""
    if self._cache is None:
        self._cache = MacroDataCache(...)
    return self._cache
```

### 4. Protocol-First Design

**Transform Protocol** (pipeline.py:25):
```python
class FeatureTransform(Protocol):
    name: str
    def feature_names(self, params: Mapping[str, Any]) -> list[str]: ...
    def requires(self) -> DataRequirements: ...
```

**Benefits**:
- Structural typing without inheritance
- Duck typing support for testing
- Clear component contracts

### 5. Data Requirements Gating

**Filtering** (pipeline.py:565-580):
```python
def _filter_transforms(
    self,
    transforms: list[TransformSpec],
    allowable: DataRequirements,
) -> list[TransformSpec]:
    """Filter transforms based on data availability."""
    filtered = []
    for spec in transforms:
        transform = _CATALOG[spec.name]
        if self._requirements_compatible(transform.requires(), allowable):
            filtered.append(spec)
    return filtered
```

**Hierarchy**:
- L1_ONLY: OHLCV bars (core technical indicators)
- L1_L2: + Order book snapshots (microstructure features)
- L1_L2_L3: + Trade records (trade flow features)

---

## 14. Production Usage Examples

### Basic Feature Computation

```python
from ml.features.engineering import FeatureConfig, FeatureEngineer
import polars as pl

# Configure features
config = FeatureConfig(
    rsi_period=14,
    bb_period=20,
    ema_fast=12,
    ema_slow=26,
    return_periods=[1, 5, 10, 20],
    momentum_periods=[5, 10, 20],
    volume_ma_periods=[5, 10, 20],
    include_microstructure=False,
    validate_quality=True,
)

# Initialize engineer
engineer = FeatureEngineer(config=config)

# Batch processing
df = pl.read_csv("data/market_data.csv")
features_df, scaler = engineer.calculate_features(
    df, mode="batch", fit_scaler=True
)

# Online processing
from ml.features.engineering import IndicatorManager

indicator_mgr = IndicatorManager(config)

# Warm up indicators
for bar in warmup_bars:
    indicator_mgr.update_from_bar(bar)

# Real-time processing
for current_bar in live_bars:
    indicator_mgr.update_from_bar(current_bar)

    features = engineer.calculate_features_online(
        current_bar=current_bar,
        indicator_manager=indicator_mgr,
        scaler=scaler,
    )

    # Must copy if persisting across bars
    features_snapshot = features.copy()
```

### Macro Features Integration

```python
from ml.features.macro_transforms import MacroFeatureTransform
from pathlib import Path

# Initialize transform
macro_transform = MacroFeatureTransform(
    macro_series_ids=["PAYEMS", "UNRATE", "CPIAUCSL", "FEDFUNDS"],
    vintage_base_dir=Path("data/fred/vintages"),
    fred_path=Path("data/fred/fred_indicators.parquet"),
    include_revisions=True,
    revision_mode="core",
    include_composites=True,
    composite_history_window=400,
)

# Batch processing
df_with_macro = macro_transform.compute_batch(
    df,
    timestamp_col="timestamp",
    vintage_cutoff=None,
)

# Real-time processing
macro_features = macro_transform.compute_realtime(
    bar=current_bar,
    ts_event=current_bar.ts_event,
)
```

### Parity Validation

```python
from ml.features.validation import FeatureParityValidator

# Initialize validator
validator = FeatureParityValidator(config=config, tolerance=1e-10)

# Validate parity
parity_report = validator.validate_parity(
    df,
    start_idx=50,   # Skip initial warmup
    end_idx=1000,   # Validate subset
)

# Check results
assert parity_report["parity_passed"], f"Parity failed: {parity_report['max_difference']}"
assert parity_report["max_difference"] < 1e-10

# Performance validation
perf_report = validator.validate_performance(df, n_iterations=100)
assert perf_report["p99_latency_ms"] < 5.0  # <5ms P99 SLA
```

### Microstructure Aggregation

```python
from ml.features.micro_aggregate import MicrostructureAggregator
from ml.features.l2_aggregate import L2Aggregator
from pathlib import Path

# Initialize aggregators
micro_agg = MicrostructureAggregator(base_dir=Path("data/micro"))
l2_agg = L2Aggregator(base_dir=Path("data/l2"))

# Compute per-minute features
micro_features = micro_agg.compute_for_symbol("SPY")
l2_features = l2_agg.compute_for_symbol("SPY")

# Join with OHLCV data
df_enhanced = df.join(micro_features, on="timestamp", how="left")
df_enhanced = df_enhanced.join(l2_features, on="timestamp", how="left")
```

---

## 15. Cross-Module References

**Integration Points**:

- **ml/actors/signal.py:400-600**: MLSignalActor uses FeatureEngineer for real-time inference
- **ml/data/tft_dataset_builder.py**: Integrates MacroFeatureTransform for training datasets
- **ml/stores/feature_store.py**: Optional persistence target (manual integration)
- **ml/registry/feature_registry.py**: Feature manifest registration and versioning
- **ml/orchestration/**: DataScheduler may trigger feature computation (no direct usage found)

**Related Documentation**:
- `context_data.md`: Data ingestion and FRED/ALFRED data preparation
- `context_stores.md`: FeatureStore persistence layer
- `context_registry.md`: Feature manifest lifecycle management
- `context_actors.md`: ML actor integration patterns
- `context_training.md`: Feature usage in training pipelines

---

## 16. Recommendations for Future Work

### High Priority

1. **Implement Metrics Integration**:
   - Add centralized metrics bootstrap to FeatureEngineer
   - Track computation timers, parity success rates, cache hits
   - Monitor hot path latency with P99 alerts

2. **Complete Pipeline Integration**:
   - Add MacroFeatureTransform to pipeline catalog
   - Integrate earnings transforms into pipeline spec
   - Auto-generate feature lists from pipeline runner

3. **Enforce Coverage Validation**:
   - Make MacroCoverageValidator mandatory for datasets
   - Add automated coverage checks in CI/CD
   - Fail fast on insufficient macro data coverage

### Medium Priority

4. **Implement Contract-Level Covariates**:
   - Earnings calendar features (days to earnings)
   - Expiry date features (days to expiry)
   - Fed meeting dates (days to FOMC)

5. **Add Progressive Fallback**:
   - DummyFeatureStore fallback when PostgreSQL unavailable
   - Default macro values when vintages missing
   - Circuit breaker for unstable data sources

6. **Document Actual Architecture**:
   - Remove Universal Architecture claims from features module
   - Clarify standalone library nature of FeatureEngineer
   - Document manual integration requirements with actors/stores

### Low Priority

7. **Cross-Sectional Features**:
   - Multi-instrument relative features
   - Sector rotation indicators
   - Pairs trading spreads

8. **GPU Acceleration**:
   - CUDA/OpenCL for computationally intensive features
   - Batch feature computation on GPU
   - Maintain CPU fallback

---

## Summary Assessment

**Production-Ready Components** ✅:
- FeatureEngineer core technical indicators
- MacroFeatureTransform with vintage-aware semantics
- MacroDataCache with O(1) real-time lookups
- MacroComposites with 37 economic dimension features
- Parity validation system with <1e-10 tolerance
- Microstructure aggregators for L1/L2 features

**NOT Production-Ready** ❌:
- Metrics integration (minimal usage despite claims)
- Universal Architecture compliance (not implemented)
- Progressive fallback (not implemented)
- 4-store + 4-registry integration (manual only)

**Actual Line Counts**:
- engineering.py: 3,296 lines (doc claimed 2,609)
- macro_transforms.py: 795 lines ✓
- macro_cache.py: 446 lines ✓
- macro_composites.py: 523 lines ✓
- validation.py: 693 lines (doc claimed 680)
- pipeline.py: 859 lines (doc claimed 509)
- Total: ~8,500 lines

**Key Strength**: Vintage-aware macro features with composites provide unique edge in point-in-time economic data integration.

**Key Weakness**: Overclaimed Universal Architecture integration and minimal metrics instrumentation despite extensive documentation.
