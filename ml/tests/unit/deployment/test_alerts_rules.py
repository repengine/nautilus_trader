from __future__ import annotations

from pathlib import Path


def test_alerts_file_contains_expected_rules() -> None:
    # Locate repo-rooted alerts file from this test path: ml/tests/unit/deployment -> ml/deployment
    alerts_path = Path(__file__).resolve().parents[3] / "deployment" / "alerts.yml"
    text = alerts_path.read_text(encoding="utf-8")

    # Basic presence checks for new async observability alerts
    expected = [
        "alert: MLObsAsyncBackpressureDrops",
        "alert: MLObsAsyncBackpressureSustained",
        "alert: MLObsAsyncQueueDepthHigh",
        "alert: MLObsAsyncFlushLatencyHighP99",
    ]
    for marker in expected:
        assert marker in text, f"Missing alert rule: {marker}"
