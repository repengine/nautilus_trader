from __future__ import annotations

import json
from pathlib import Path

import pytest

from playground.backtest.runner import MonitoringSnapshotResult
from playground.monitoring.integrations import MonitoringIntegrationArtifacts
from playground.monitoring.integrations import build_grafana_payload
from playground.monitoring.integrations import build_pagerduty_payload
from playground.monitoring.integrations import persist_monitoring_integrations
from playground.scripts.publish_phase3_monitoring_integrations import main as publish_main


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
            "benchmarks": {
                "root": "backtesting/benchmarks",
                "latest_slug": "train_2010-01-01_2018-12-31__test_2019-01-01_2024-12-31",
                "summary_path": "backtesting/benchmarks/latest/benchmark_summary.csv",
                "baseline_metrics_path": "backtesting/benchmarks/latest/baseline_metrics.csv",
                "comparison_path": "backtesting/benchmarks/latest/performance_comparison_table.csv",
                "audit_path": "backtesting/benchmarks/latest/benchmark_audit.csv",
                "metadata_path": "backtesting/benchmarks/latest/metadata.json",
            },
            "phase4_sensitivity": {
                "summary_path": "sensitivity/summary.csv",
                "report_path": "sensitivity/sensitivity_analysis.pdf",
            },
            "phase4_data_quality": {
                "audit_path": "data_quality/missing_data_audit.json",
            },
            "phase4_outliers": {
                "report_path": "outliers/factor_outlier_report.json",
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
        "phase4_sensitivity_metadata": [
            {"slug": "rolling-window-sensitivity", "metric_value": 0.94},
        ],
        "phase4_data_quality": {
            "missing_ratio": 0.004,
            "missing_by_column": {"factor_duration": 0.0},
        },
        "phase4_outlier_summary": {
            "recommended_treatment": "winsorize",
            "outlier_ratio": 0.01,
        },
    }


def test_build_grafana_payload_extracts_overlay_stats(
    sample_snapshot_payload: dict[str, object],
) -> None:
    payload = build_grafana_payload(sample_snapshot_payload)
    assert payload["dashboard_target"] == "dashboards/phase3_risk_model"
    assert payload["alert_rule"] == "alerts/phase3_risk_ruleset.yml"
    monte_carlo = payload["monte_carlo"]
    assert (
        monte_carlo["overlay_category_summary_path"]
        == "stress/monte_carlo/overlay_category_summary.csv"
    )
    assert monte_carlo["baseline_metrics"]["sharpe_ratio"] == pytest.approx(0.91)
    assert monte_carlo["overlay_category_stats"][0]["category"] == "rates"
    assert payload["parameter_heatmaps"]["metadata"][0]["slug"] == "turnover-vs-liquidity"
    assert payload["benchmarks"]["audit_path"] == "backtesting/benchmarks/latest/benchmark_audit.csv"
    phase4 = payload["phase4"]
    assert phase4["sensitivity"]["report_path"] == "sensitivity/sensitivity_analysis.pdf"
    assert phase4["sensitivity"]["metadata"][0]["slug"] == "rolling-window-sensitivity"
    assert phase4["data_quality"]["summary"]["missing_ratio"] == pytest.approx(0.004)
    assert phase4["outliers"]["summary"]["recommended_treatment"] == "winsorize"


def test_build_pagerduty_payload_includes_metadata(
    sample_snapshot_payload: dict[str, object],
) -> None:
    payload = build_pagerduty_payload(sample_snapshot_payload)
    assert payload["service_rule"] == "alerts/phase3-risk-critical"
    assert (
        payload["automation_targets"]["github_actions"] == ".github/workflows/phase3_monitoring.yml"
    )
    assert payload["proxy_dataset_metadata"]["fixture-proxy"]["status"] == "success"
    assert payload["vintage_metadata"][0]["slug"] == "fixture"
    benchmark_artifacts = payload["benchmark_artifacts"]
    assert benchmark_artifacts["audit_path"] == "backtesting/benchmarks/latest/benchmark_audit.csv"
    phase4 = payload["phase4"]
    assert phase4["sensitivity"]["metadata"][0]["metric_value"] == pytest.approx(0.94)
    assert phase4["data_quality"]["summary"]["missing_by_column"]["factor_duration"] == 0.0
    assert phase4["outliers"]["summary"]["outlier_ratio"] == pytest.approx(0.01)


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


def test_publish_integrations_script_writes_manifest(
    tmp_path: Path,
    sample_snapshot_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path = tmp_path / "phase3_monitoring_snapshot.json"
    snapshot_path.write_text(json.dumps(sample_snapshot_payload), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "publish_phase3_monitoring_integrations.py",
            "--snapshot-path",
            str(snapshot_path),
            "--output-dir",
            str(tmp_path),
        ],
    )

    publish_main()

    manifest_path = tmp_path / "monitoring" / "automation_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["grafana_dashboard_target"] == "dashboards/phase3_risk_model"
    assert manifest["benchmarks"]["audit_path"] == "backtesting/benchmarks/latest/benchmark_audit.csv"
    phase4_manifest = manifest["phase4"]
    assert phase4_manifest["sensitivity_summary_path"] == "sensitivity/summary.csv"
    assert phase4_manifest["sensitivity_report_path"] == "sensitivity/sensitivity_analysis.pdf"
    assert phase4_manifest["data_quality_audit_path"] == "data_quality/missing_data_audit.json"
    assert phase4_manifest["outlier_report_path"] == "outliers/factor_outlier_report.json"
    assert (tmp_path / "monitoring" / "grafana_dashboard_payload.json").exists()
    assert (tmp_path / "monitoring" / "pagerduty_alert_payload.json").exists()


def test_publish_integrations_script_supports_escalation_dry_run(
    tmp_path: Path,
    sample_snapshot_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path = tmp_path / "phase3_monitoring_snapshot.json"
    snapshot_path.write_text(json.dumps(sample_snapshot_payload), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "publish_phase3_monitoring_integrations.py",
            "--snapshot-path",
            str(snapshot_path),
            "--output-dir",
            str(tmp_path),
            "--simulate-escalation",
        ],
    )

    publish_main()

    rehearsal_path = tmp_path / "monitoring" / "pagerduty_escalation_dry_run.json"
    assert rehearsal_path.exists()
    rehearsal = json.loads(rehearsal_path.read_text(encoding="utf-8"))
    assert rehearsal["service_rule"] == "alerts/phase3-risk-critical"
    assert rehearsal["benchmarks"]["audit_path"] == "backtesting/benchmarks/latest/benchmark_audit.csv"
    phase4_rehearsal = rehearsal["phase4"]
    assert phase4_rehearsal["sensitivity_metadata"][0]["slug"] == "rolling-window-sensitivity"
    assert phase4_rehearsal["data_quality_summary"]["missing_ratio"] == pytest.approx(0.004)
    assert phase4_rehearsal["outlier_summary"]["outlier_ratio"] == pytest.approx(0.01)
