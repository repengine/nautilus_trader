"""
Helpers for wiring Phase 3 monitoring snapshots into observability integrations.

These utilities remain cold-path only. They operate on the JSON payload emitted by
``export_phase3_monitoring_snapshot`` and persist derived payloads tailored for
Grafana dashboards and PagerDuty alert automation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from playground.backtest.runner import MonitoringSnapshotResult


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _as_sequence_of_mappings(value: object) -> list[Mapping[str, object]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items: list[Mapping[str, object]] = []
        for entry in value:
            if isinstance(entry, Mapping):
                items.append(entry)
        return items
    return []


@dataclass(frozen=True)
class MonitoringIntegrationArtifacts:
    """
    Paths to integration artefacts derived from a monitoring snapshot.
    """

    grafana_payload_path: Path
    pagerduty_payload_path: Path


def build_grafana_payload(snapshot_payload: Mapping[str, object]) -> dict[str, object]:
    """
    Construct a Grafana-ready payload summarising Phase 3 monitoring outputs.
    """
    sections = _as_mapping(snapshot_payload.get("sections"))
    monte_carlo_section = _as_mapping(sections.get("monte_carlo"))
    heatmap_section = _as_mapping(sections.get("parameter_heatmaps"))
    diagnostics_section = _as_mapping(sections.get("extended_diagnostics"))
    proxy_section = _as_mapping(sections.get("proxy_datasets"))
    vintage_section = _as_mapping(sections.get("vintage_simulations"))

    monte_carlo_metadata = _as_mapping(snapshot_payload.get("monte_carlo_metadata"))
    heatmap_metadata = _as_sequence_of_mappings(snapshot_payload.get("parameter_heatmap_metadata"))

    dashboard_targets = _as_mapping(snapshot_payload.get("dashboard_targets"))
    alert_rules = _as_mapping(snapshot_payload.get("alert_rules"))
    automation_targets = _as_mapping(snapshot_payload.get("automation_targets"))

    grafana_payload = {
        "dashboard_target": dashboard_targets.get("grafana"),
        "alert_rule": alert_rules.get("grafana"),
        "automation_targets": dict(automation_targets),
        "monte_carlo": {
            "summary_path": monte_carlo_section.get("summary_path"),
            "paths_path": monte_carlo_section.get("paths_path"),
            "overlay_summary_path": monte_carlo_section.get("overlay_summary_path"),
            "overlay_category_summary_path": monte_carlo_section.get("overlay_category_summary_path"),
            "baseline_metrics_path": monte_carlo_section.get("baseline_metrics_path"),
            "overlay_category_stats": monte_carlo_metadata.get("overlay_category_stats", []),
            "baseline_metrics": monte_carlo_metadata.get("baseline_metrics"),
            "report_metrics": monte_carlo_metadata.get("report_metrics", []),
            "report_quantiles": monte_carlo_metadata.get("report_quantiles", []),
        },
        "parameter_heatmaps": {
            "summary_path": heatmap_section.get("summary_path"),
            "specs": heatmap_section.get("config_specs", []),
            "metadata": heatmap_metadata,
        },
        "diagnostics": {
            "tail_metrics_path": diagnostics_section.get("tail_metrics_path"),
            "turnover_distribution_path": diagnostics_section.get("turnover_distribution_path"),
            "benchmark_deltas_path": diagnostics_section.get("benchmark_deltas_path"),
            "summary": diagnostics_section.get("summary"),
        },
        "proxies": {
            "summary_path": proxy_section.get("summary_path"),
            "datasets": proxy_section.get("datasets", []),
            "dataset_status": proxy_section.get("dataset_status"),
        },
        "vintage": {
            "summary_path": vintage_section.get("summary_path"),
            "windows": vintage_section.get("windows", []),
        },
    }
    return grafana_payload


def build_pagerduty_payload(snapshot_payload: Mapping[str, object]) -> dict[str, object]:
    """
    Construct a PagerDuty-friendly payload capturing alert and automation context.
    """
    sections = _as_mapping(snapshot_payload.get("sections"))
    proxy_section = _as_mapping(sections.get("proxy_datasets"))
    vintage_section = _as_mapping(sections.get("vintage_simulations"))

    alert_rules = _as_mapping(snapshot_payload.get("alert_rules"))
    automation_targets = _as_mapping(snapshot_payload.get("automation_targets"))
    diagnostics_metadata = _as_mapping(snapshot_payload.get("diagnostics_metadata"))
    proxy_metadata = _as_mapping(snapshot_payload.get("proxy_dataset_metadata"))
    vintage_metadata = snapshot_payload.get("vintage_metadata")
    if not isinstance(vintage_metadata, list):
        vintage_metadata = []

    pagerduty_payload = {
        "service_rule": alert_rules.get("pagerduty"),
        "automation_targets": dict(automation_targets),
        "diagnostics_metadata": diagnostics_metadata,
        "proxy_dataset_metadata": proxy_metadata,
        "proxy_artifacts": {
            "summary_path": proxy_section.get("summary_path"),
            "datasets": proxy_section.get("datasets", []),
            "dataset_status": proxy_section.get("dataset_status"),
        },
        "vintage_artifacts": {
            "summary_path": vintage_section.get("summary_path"),
            "windows": vintage_section.get("windows", []),
        },
        "vintage_metadata": vintage_metadata,
    }
    return pagerduty_payload


def persist_monitoring_integrations(
    *,
    snapshot: MonitoringSnapshotResult,
    output_dir: Path,
    grafana_filename: str = "grafana_dashboard_payload.json",
    pagerduty_filename: str = "pagerduty_alert_payload.json",
) -> MonitoringIntegrationArtifacts:
    """
    Persist Grafana and PagerDuty integration payloads derived from a snapshot.
    """
    grafana_payload = build_grafana_payload(snapshot.payload)
    pagerduty_payload = build_pagerduty_payload(snapshot.payload)

    integration_dir = output_dir / "monitoring"
    integration_dir.mkdir(parents=True, exist_ok=True)

    grafana_path = integration_dir / grafana_filename
    pagerduty_path = integration_dir / pagerduty_filename

    grafana_path.write_text(json.dumps(grafana_payload, indent=2, sort_keys=True), encoding="utf-8")
    pagerduty_path.write_text(json.dumps(pagerduty_payload, indent=2, sort_keys=True), encoding="utf-8")

    return MonitoringIntegrationArtifacts(
        grafana_payload_path=grafana_path,
        pagerduty_payload_path=pagerduty_path,
    )


__all__ = [
    "MonitoringIntegrationArtifacts",
    "build_grafana_payload",
    "build_pagerduty_payload",
    "persist_monitoring_integrations",
]
