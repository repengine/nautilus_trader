from __future__ import annotations

import json
from pathlib import Path

import pytest

from playground.backtest.runner import MonitoringSnapshotResult
from playground.monitoring.integrations import MonitoringIntegrationArtifacts
from playground.monitoring.integrations import build_grafana_payload
from playground.monitoring.integrations import build_pagerduty_payload
from playground.monitoring.integrations import persist_monitoring_integrations


@pytest.fixture()
def sample_snapshot_payload() -> dict[str, object]:
    return {
        "sections": {
            "monte_carlo": {
                "summary_path": "stress/monte_carlo/summary.csv",
                "paths_path": "stress/monte_carlo/paths.csv",
                "overlay_summary_path": "stress/monte_carlo/overlay_summary.csv",
                "overlay_category_summary_path": "stress/monte_carlo/overlay_category_summary.csv",
                "baseline_metrics_path": "stress/monte_carlo/baseline_metrics.csv",
            },
            "parameter_heatmaps": {
                "summary_path": "heatmaps/summary.csv",
                "config_specs": ["turnover-vs-liquidity"],
            },
            "extended_diagnostics": {
                "tail_metrics_path": "diagnostics/tail_metrics.csv",
                "turnover_distribution_path": "diagnostics/turnover_distribution.csv",
                "benchmark_deltas_path": "diagnostics/benchmark_deltas.csv",
                "summary": {"Equal Weight": {"var_p05": -0.03}},
            },
            "proxy_datasets": {
                "summary_path": "proxy_datasets/summary.csv",
                "datasets": ["fixture-proxy"],
                "dataset_status": {"fixture-proxy": "success"},
            },
            "vintage_simulations": {
                "summary_path": "vintage/summary.csv",
                "windows": [{"slug": "fixture", "status": "success"}],
            },
        },
        "dashboard_targets": {"grafana": "dashboards/phase3_risk_model"},
        "alert_rules": {
            "grafana": "alerts/phase3_risk_ruleset.yml",
            "pagerduty": "alerts/phase3-risk-critical",
        },
        "automation_targets": {
            "airflow": "dags/phase3_monitoring_refresh.py",
            "github_actions": ".github/workflows/phase3_monitoring.yml",
        },
        "monte_carlo_metadata": {
            "overlay_category_stats": [{"category": "rates", "activation_count": 3}],
            "baseline_metrics": {"sharpe_ratio": 0.91},
            "report_metrics": ["sharpe_ratio"],
            "report_quantiles": [0.5],
        },
        "parameter_heatmap_metadata": [
            {"slug": "turnover-vs-liquidity", "best_metric": 1.02},
        ],
        "diagnostics_metadata": {"tail": {"Equal Weight": {"var_p05": -0.03}}},
        "proxy_dataset_metadata": {"fixture-proxy": {"status": "success"}},
        "vintage_metadata": [{"slug": "fixture", "status": "success"}],
    }


def test_build_grafana_payload_extracts_overlay_stats(sample_snapshot_payload: dict[str, object]) -> None:
    payload = build_grafana_payload(sample_snapshot_payload)
    assert payload["dashboard_target"] == "dashboards/phase3_risk_model"
    assert payload["alert_rule"] == "alerts/phase3_risk_ruleset.yml"
    monte_carlo = payload["monte_carlo"]
    assert monte_carlo["overlay_category_summary_path"] == "stress/monte_carlo/overlay_category_summary.csv"
    assert monte_carlo["baseline_metrics"]["sharpe_ratio"] == pytest.approx(0.91)
    assert monte_carlo["overlay_category_stats"][0]["category"] == "rates"
    assert payload["parameter_heatmaps"]["metadata"][0]["slug"] == "turnover-vs-liquidity"


def test_build_pagerduty_payload_includes_metadata(sample_snapshot_payload: dict[str, object]) -> None:
    payload = build_pagerduty_payload(sample_snapshot_payload)
    assert payload["service_rule"] == "alerts/phase3-risk-critical"
    assert payload["automation_targets"]["github_actions"] == ".github/workflows/phase3_monitoring.yml"
    assert payload["proxy_dataset_metadata"]["fixture-proxy"]["status"] == "success"
    assert payload["vintage_metadata"][0]["slug"] == "fixture"


def test_persist_monitoring_integrations_writes_files(
    tmp_path: Path,
    sample_snapshot_payload: dict[str, object],
) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(sample_snapshot_payload), encoding="utf-8")
    snapshot = MonitoringSnapshotResult(path=snapshot_path, payload=sample_snapshot_payload)

    artefacts: MonitoringIntegrationArtifacts = persist_monitoring_integrations(
        snapshot=snapshot,
        output_dir=tmp_path,
    )

    grafana_payload = json.loads(artefacts.grafana_payload_path.read_text(encoding="utf-8"))
    pagerduty_payload = json.loads(artefacts.pagerduty_payload_path.read_text(encoding="utf-8"))

    assert grafana_payload["diagnostics"]["tail_metrics_path"] == "diagnostics/tail_metrics.csv"
    assert pagerduty_payload["proxy_dataset_metadata"]["fixture-proxy"]["status"] == "success"
