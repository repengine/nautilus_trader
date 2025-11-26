"""
Feature registry catalog utilities.

Provides summary statistics and consistency checks across all registered feature
manifests. Designed for offline inspection and CI validation; no hot-path usage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)


class FeatureFamily(Enum):
    """Categorisation for feature columns in manifests."""

    TECHNICAL = "technical"
    MACRO = "macro"
    EVENT = "event"
    CALENDAR = "calendar"
    MICRO = "micro"
    METADATA = "metadata"
    OTHER = "other"


CAPABILITY_FAMILY_MAP: Mapping[str, tuple[FeatureFamily, ...]] = {
    "include_macro": (FeatureFamily.MACRO,),
    "include_macro_revisions": (FeatureFamily.MACRO,),
    "include_macro_composites": (FeatureFamily.MACRO,),
    "include_macro_deltas": (FeatureFamily.MACRO,),
    "include_events": (FeatureFamily.EVENT,),
    "include_earnings": (FeatureFamily.EVENT,),
    "include_calendar": (FeatureFamily.CALENDAR,),
    "include_calendar_lags": (FeatureFamily.CALENDAR,),
    "include_context_features": (FeatureFamily.CALENDAR,),
    "include_micro": (FeatureFamily.MICRO,),
    "include_microstructure": (FeatureFamily.MICRO,),
    "include_trade_flow": (FeatureFamily.MICRO,),
    "include_l2": (FeatureFamily.MICRO,),
}

FAMILY_TO_CAPABILITIES: Mapping[FeatureFamily, tuple[str, ...]] = {
    family: tuple(sorted(key for key, families in CAPABILITY_FAMILY_MAP.items() if family in families))
    for family in FeatureFamily
}

METADATA_FEATURES = {
    "asset_class",
    "exchange",
    "tick_size",
    "currency",
    "fee_class",
    "market_segment",
    "contract_size",
    "lot_size",
    "min_price_increment",
    "margin_initial",
    "margin_maintenance",
}

CALENDAR_PREFIXES = (
    "tod_",
    "hour",
    "minute",
    "dow",
    "month",
)
CALENDAR_EXACT = {
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "days_to_month_end",
    "days_from_month_start",
    "week_of_year",
}

MACRO_KEYWORDS = (
    "macro",
    "credit_",
    "term_spread",
    "fed_policy",
    "yield_curve",
    "real_term_premium",
    "real_yield",
    "liquidity_",
    "stagflation",
    "goldilocks",
    "dollar_",
    "growth_momentum",
    "inflation_momentum",
    "fx_",
    "ted_spread",
    "qe_",
    "bank_credit",
    "sofr",
    "financial_stress",
    "vix",
)

EVENT_PREFIXES = (
    "hours_to_earnings",
    "hours_to_fed_meeting",
    "hours_to_economic_release",
    "hours_to_options_expiry",
    "has_earnings",
    "has_fed_meeting",
    "has_economic_release",
    "has_options_expiry",
    "earnings_within",
    "fed_meeting_within",
    "economic_release_within",
    "options_expiry_within",
    "has_fed_event",
    "has_cpi_event",
)
EVENT_KEYWORDS = (
    "earnings_today",
    "fed_event",
    "cpi_event",
    "event_importance_score",
    "event_clustering_score",
    "total_events",
    "event_density",
    "days_to_next_fed",
    "days_to_next_cpi",
    "days_to_next_earnings",
    "days_to_next_holiday",
    "days_since_last_fed",
    "days_since_last_cpi",
    "days_since_last_earnings",
    "is_triple_witching",
    "is_fomc_week",
    "is_earnings_season",
    "is_holiday_week",
)

MICRO_KEYWORDS = (
    "spread",
    "imbalance",
    "microstructure",
    "mid_return",
    "trade_flow",
    "price_impact",
    "vwap",
    "depth",
    "l2_",
    "orderbook",
    "queue",
)

TECHNICAL_PREFIXES = (
    "return_",
    "momentum_",
    "volatility_",
    "volume_ratio",
    "sma_",
    "ema_",
    "macd_",
    "price_position",
    "forward_return",
    "bb_",
    "rsi",
    "ema_fast",
    "ema_slow",
    "ema_cross",
    "atr_",
)
TECHNICAL_KEYWORDS = (
    "bollinger",
    "ema_fast_dist",
    "ema_slow_dist",
    "ema_cross",
    "hl_spread",
)


def _initialize_family_counter() -> dict[str, int]:
    return {family.value: 0 for family in FeatureFamily}


def _classify_feature_name(name: str) -> FeatureFamily:
    lower = name.lower()

    if lower in METADATA_FEATURES:
        return FeatureFamily.METADATA

    if lower.startswith("is_macro_available"):
        return FeatureFamily.MACRO

    if name.isupper() and any(ch.isalpha() for ch in name):
        return FeatureFamily.MACRO

    if "__value" in lower or "_prior_" in lower or "_revision_" in lower or "_mom_" in lower or "_pct_" in lower or "_net_signal_" in lower:
        return FeatureFamily.MACRO

    if any(keyword in lower for keyword in MACRO_KEYWORDS):
        return FeatureFamily.MACRO

    if lower.startswith(EVENT_PREFIXES) or any(keyword in lower for keyword in EVENT_KEYWORDS):
        return FeatureFamily.EVENT

    if lower.startswith(CALENDAR_PREFIXES) or lower in CALENDAR_EXACT or lower.startswith(("days_to_month", "days_from_month", "is_quarter_", "is_month_", "week_of_year")):
        return FeatureFamily.CALENDAR

    if any(keyword in lower for keyword in MICRO_KEYWORDS):
        return FeatureFamily.MICRO

    if lower.startswith(TECHNICAL_PREFIXES) or any(keyword in lower for keyword in TECHNICAL_KEYWORDS):
        return FeatureFamily.TECHNICAL

    return FeatureFamily.OTHER


def _flags_missing_families(
    capability_flags: Mapping[str, bool],
    family_counts: Mapping[str, int],
) -> list[str]:
    missing: set[str] = set()
    for capability, families in CAPABILITY_FAMILY_MAP.items():
        if not capability_flags.get(capability, False):
            continue
        total = sum(family_counts.get(family.value, 0) for family in families)
        if total == 0:
            missing.add(capability)
    return sorted(missing)


def _families_without_enabled_flag(
    capability_flags: Mapping[str, bool],
    family_counts: Mapping[str, int],
) -> list[str]:
    families_missing: set[str] = set()
    for family in FeatureFamily:
        if family in {FeatureFamily.OTHER, FeatureFamily.METADATA, FeatureFamily.TECHNICAL}:
            continue
        count = family_counts.get(family.value, 0)
        if count == 0:
            continue
        expected_flags = FAMILY_TO_CAPABILITIES.get(family, ())
        if not expected_flags:
            continue
        if not any(capability_flags.get(flag, False) for flag in expected_flags):
            families_missing.add(family.value)
    return sorted(families_missing)


@dataclass(slots=True)
class FeatureSetSummary:
    """Summary statistics for a single feature manifest."""

    feature_set_id: str
    name: str
    version: str
    role: str
    stage: str
    total_features: int
    family_counts: dict[str, int]
    capability_flags: dict[str, bool]
    flags_missing_families: list[str] = field(default_factory=list)
    families_without_flag: list[str] = field(default_factory=list)
    schema_hash: str | None = None
    pipeline_signature: str | None = None
    created_at: float | None = None
    last_modified: float | None = None

    @property
    def has_inconsistencies(self) -> bool:
        return bool(self.flags_missing_families or self.families_without_flag)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_set_id": self.feature_set_id,
            "name": self.name,
            "version": self.version,
            "role": self.role,
            "stage": self.stage,
            "total_features": self.total_features,
            "family_counts": dict(self.family_counts),
            "capability_flags": dict(self.capability_flags),
            "flags_missing_families": list(self.flags_missing_families),
            "families_without_flag": list(self.families_without_flag),
            "schema_hash": self.schema_hash,
            "pipeline_signature": self.pipeline_signature,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
        }


@dataclass(slots=True)
class FeatureCatalogReport:
    """Aggregate report across every feature manifest in the registry."""

    feature_sets: list[FeatureSetSummary]
    totals_by_family: dict[str, int]
    total_feature_sets: int
    total_features: int

    @property
    def has_inconsistencies(self) -> bool:
        return any(summary.has_inconsistencies for summary in self.feature_sets)

    def to_dict(self) -> dict[str, object]:
        return {
            "total_feature_sets": self.total_feature_sets,
            "total_features": self.total_features,
            "totals_by_family": dict(self.totals_by_family),
            "feature_sets": [summary.to_dict() for summary in self.feature_sets],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def render_text(self) -> str:
        lines: list[str] = []
        lines.append(f"Feature sets: {self.total_feature_sets}")
        lines.append(f"Total features: {self.total_features}")
        lines.append("Totals by family:")
        for family, count in sorted(self.totals_by_family.items()):
            lines.append(f"  - {family}: {count}")
        for summary in self.feature_sets:
            lines.append("")
            lines.append(
                f"{summary.name} ({summary.feature_set_id}) "
                f"[version={summary.version}, role={summary.role}, stage={summary.stage}]",
            )
            lines.append(f"  total_features: {summary.total_features}")
            lines.append(
                "  family_counts: "
                + ", ".join(f"{family}={count}" for family, count in sorted(summary.family_counts.items())),
            )
            if summary.flags_missing_families:
                lines.append("  flags_missing_families: " + ", ".join(summary.flags_missing_families))
            if summary.families_without_flag:
                lines.append("  families_without_flag: " + ", ".join(summary.families_without_flag))
        return "\n".join(lines)


def _summarise_manifest(manifest: FeatureManifest) -> FeatureSetSummary:
    family_counts = _initialize_family_counter()
    for feature_name in manifest.feature_names:
        family = _classify_feature_name(feature_name)
        family_counts[family.value] += 1

    capability_flags = {str(key): bool(value) for key, value in (manifest.capability_flags or {}).items()}
    flags_missing = _flags_missing_families(capability_flags, family_counts)
    families_without_flag = _families_without_enabled_flag(capability_flags, family_counts)

    summary = FeatureSetSummary(
        feature_set_id=manifest.feature_set_id,
        name=manifest.name,
        version=manifest.version,
        role=manifest.role.value,
        stage=manifest.stage.value,
        total_features=len(manifest.feature_names),
        family_counts=dict(family_counts),
        capability_flags=dict(sorted(capability_flags.items())),
        flags_missing_families=flags_missing,
        families_without_flag=families_without_flag,
        schema_hash=manifest.schema_hash,
        pipeline_signature=manifest.pipeline_signature,
        created_at=manifest.created_at or None,
        last_modified=manifest.last_modified or None,
    )
    if summary.has_inconsistencies:
        logger.debug(
            "feature_manifest_inconsistency_detected",
            extra={
                "feature_set_id": summary.feature_set_id,
                "flags_missing_families": summary.flags_missing_families,
                "families_without_flag": summary.families_without_flag,
            },
        )
    return summary


def build_feature_catalog(registry: FeatureRegistry) -> FeatureCatalogReport:
    """
    Build an aggregate catalog report for the provided feature registry.

    Parameters
    ----------
    registry :
        Registry instance to inspect.

    Returns
    -------
    FeatureCatalogReport
        Report containing per-manifest summaries and overall totals.
    """
    feature_infos = registry.list_all()
    summaries: list[FeatureSetSummary] = []
    totals_counter = _initialize_family_counter()

    for info in sorted(feature_infos, key=lambda item: (item.manifest.name, item.manifest.version, item.manifest.feature_set_id)):
        summary = _summarise_manifest(info.manifest)
        summaries.append(summary)
        for family, count in summary.family_counts.items():
            totals_counter[family] += count

    totals_by_family = dict(sorted(totals_counter.items()))
    total_features = sum(totals_counter.values())

    return FeatureCatalogReport(
        feature_sets=summaries,
        totals_by_family=totals_by_family,
        total_feature_sets=len(summaries),
        total_features=total_features,
    )
