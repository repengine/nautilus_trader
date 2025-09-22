from __future__ import annotations

from pathlib import Path


def test_ui_template_contains_ts_check_and_typed_helpers() -> None:
    tpl = Path("ml/dashboard/templates/index.html").read_text(encoding="utf-8")
    assert "// @ts-check" in tpl
    assert "/** @typedef" in tpl
    assert "function getHeaders()" in tpl
    assert "function renderGrafanaEmbeds" in tpl
    assert "function formatMetric" in tpl
    assert "async function load()" in tpl
    assert "async function act(" in tpl
    assert "async function featurePromote(" in tpl


def test_ui_template_has_required_elements() -> None:
    tpl = Path("ml/dashboard/templates/index.html").read_text(encoding="utf-8")
    assert 'id="token"' in tpl
    assert 'id="services"' in tpl
    assert 'id="models"' in tpl
    assert 'id="features"' in tpl
    assert 'id="events"' in tpl
    assert 'id="summary"' in tpl
    assert 'id="grafanaStatus"' in tpl
    assert 'id="grafanaEmbeds"' in tpl
