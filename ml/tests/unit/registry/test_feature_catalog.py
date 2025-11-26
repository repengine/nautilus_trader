"""
Tests for FeatureRegistry feature catalog utilities.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.registry.base import DataRequirements
from ml.registry.feature_registry import FeatureManifest
from ml.registry.feature_registry import FeatureRegistry
from ml.registry.feature_registry import FeatureRole
from ml.registry.feature_registry import compute_schema_hash
from ml.registry.tools.feature_catalog import FeatureFamily
from ml.registry.tools.feature_catalog import build_feature_catalog


def _register_manifest(
    registry_path: Path,
    *,
    feature_names: list[str],
    capability_flags: dict[str, bool] | None = None,
    version: str = "1.0.0",
    name: str = "test_features",
) -> FeatureRegistry:
    registry = FeatureRegistry(registry_path)
    dtypes = ["float32"] * len(feature_names)
    manifest = FeatureManifest(
        feature_set_id="",
        name=name,
        version=version,
        role=FeatureRole.TEACHER,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=feature_names,
        feature_dtypes=dtypes,
        schema_hash=compute_schema_hash(feature_names, dtypes, "sig"),
        pipeline_signature="sig",
        pipeline_version="1.0.0",
        capability_flags=capability_flags or {},
    )
    registry.register_feature_set(manifest)
    return registry


@pytest.mark.unit
def test_feature_catalog_counts_and_mismatches(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    feature_names = [
        "return_1",
        "FEDFUNDS",
        "has_fed_meeting_in_24h",
        "spread_mean",
        "asset_class",
    ]
    capability_flags = {
        "include_macro": True,
        "include_events": True,
        "include_calendar": True,  # Missing calendar features should trigger a mismatch.
        "include_micro": False,  # Micro features exist but flag is false.
    }
    registry = _register_manifest(
        registry_dir,
        feature_names=feature_names,
        capability_flags=capability_flags,
    )

    report = build_feature_catalog(registry)
    assert report.total_feature_sets == 1

    summary = report.feature_sets[0]
    assert summary.total_features == len(feature_names)
    assert summary.family_counts[FeatureFamily.TECHNICAL.value] == 1
    assert summary.family_counts[FeatureFamily.MACRO.value] == 1
    assert summary.family_counts[FeatureFamily.EVENT.value] == 1
    assert summary.family_counts[FeatureFamily.MICRO.value] == 1
    assert summary.family_counts[FeatureFamily.METADATA.value] == 1
    assert summary.family_counts[FeatureFamily.CALENDAR.value] == 0

    # Capability flag enabled but no calendar features detected.
    assert summary.flags_missing_families == ["include_calendar"]
    # Micro features exist while no corresponding capability flag is enabled.
    assert FeatureFamily.MICRO.value in summary.families_without_flag

    # Aggregated totals should match per-manifest counts.
    assert report.totals_by_family[FeatureFamily.TECHNICAL.value] == 1
    assert report.totals_by_family[FeatureFamily.MACRO.value] == 1
    assert report.totals_by_family[FeatureFamily.EVENT.value] == 1
    assert report.totals_by_family[FeatureFamily.MICRO.value] == 1
    assert report.totals_by_family[FeatureFamily.METADATA.value] == 1


@pytest.mark.unit
def test_feature_catalog_multiple_manifests(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    registry = _register_manifest(
        registry_dir,
        feature_names=["return_5", "return_20"],
        capability_flags={"include_macro": False},
        name="returns",
        version="1.0.0",
    )
    # Register a second manifest in the same registry instance.
    dtypes = ["float32", "float32"]
    macro_features = ["GDP", "GDP__value_real_time"]
    manifest = FeatureManifest(
        feature_set_id="",
        name="macro_features",
        version="2.0.0",
        role=FeatureRole.TEACHER,
        data_requirements=DataRequirements.L1_ONLY,
        feature_names=macro_features,
        feature_dtypes=dtypes,
        schema_hash=compute_schema_hash(macro_features, dtypes, "macro_sig"),
        pipeline_signature="macro_sig",
        pipeline_version="1.0.0",
        capability_flags={"include_macro": True},
    )
    registry.register_feature_set(manifest)

    report = build_feature_catalog(registry)
    assert report.total_feature_sets == 2

    totals = report.totals_by_family
    assert totals[FeatureFamily.TECHNICAL.value] == 2
    assert totals[FeatureFamily.MACRO.value] == 2
    # Ensure other families default to zero when absent.
    assert totals[FeatureFamily.EVENT.value] == 0
    assert totals[FeatureFamily.CALENDAR.value] == 0
