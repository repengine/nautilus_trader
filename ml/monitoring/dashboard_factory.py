
"""
Dashboard factory for generating Grafana dashboards programmatically.

This module provides utilities for creating consistent, reusable Grafana dashboard
components following Nautilus Trader's ML monitoring patterns.

"""

from __future__ import annotations

import json
import logging
from typing import Any


# Configure module logger
logger = logging.getLogger(__name__)


class GrafanaPanelFactory:
    """
    Factory for creating standardized Grafana panel components.
    """

    @staticmethod
    def create_stat_panel(
        title: str,
        expr: str,
        panel_id: int,
        grid_pos: dict[str, int],
        unit: str = "short",
        thresholds: list[dict[str, Any]] | None = None,
        alert_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a standardized stat panel.

        Parameters
        ----------
        title : str
            Panel title
        expr : str
            PromQL expression
        panel_id : int
            Unique panel ID
        grid_pos : dict[str, int]
            Panel grid position with keys: h, w, x, y
        unit : str, optional
            Unit for the value display
        thresholds : list[dict[str, Any]], optional
            Custom threshold configuration
        alert_config : dict[str, Any], optional
            Alert configuration

        Returns
        -------
        dict[str, Any]
            Grafana stat panel configuration

        """
        if thresholds is None:
            thresholds = [
                {"color": "green", "value": None},
                {"color": "yellow", "value": 80},
                {"color": "red", "value": 90},
            ]

        panel = {
            "datasource": {"type": "prometheus", "uid": "${datasource}"},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "mappings": [],
                    "thresholds": {"mode": "absolute", "steps": thresholds},
                    "unit": unit,
                },
            },
            "gridPos": grid_pos,
            "id": panel_id,
            "options": {
                "colorMode": "background",
                "graphMode": "area",
                "justifyMode": "center",
                "orientation": "auto",
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                "textMode": "auto",
            },
            "pluginVersion": "10.2.3",
            "targets": [
                {
                    "datasource": {"type": "prometheus", "uid": "${datasource}"},
                    "expr": expr,
                    "refId": "A",
                },
            ],
            "title": title,
            "type": "stat",
        }

        if alert_config:
            panel["alert"] = alert_config

        return panel

    @staticmethod
    def create_timeseries_panel(
        title: str,
        targets: list[dict[str, Any]],
        panel_id: int,
        grid_pos: dict[str, int],
        unit: str = "short",
        legend_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a standardized time series panel.

        Parameters
        ----------
        title : str
            Panel title
        targets : list[dict[str, Any]]
            List of PromQL targets
        panel_id : int
            Unique panel ID
        grid_pos : dict[str, int]
            Panel grid position
        unit : str, optional
            Unit for the Y axis
        legend_config : dict[str, Any], optional
            Legend configuration

        Returns
        -------
        dict[str, Any]
            Grafana time series panel configuration

        """
        if legend_config is None:
            legend_config = {
                "calcs": ["mean", "lastNotNull"],
                "displayMode": "table",
                "placement": "bottom",
                "showLegend": True,
            }

        return {
            "datasource": {"type": "prometheus", "uid": "${datasource}"},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "custom": {
                        "axisBorderShow": False,
                        "axisCenteredZero": False,
                        "axisColorMode": "text",
                        "axisLabel": "",
                        "axisPlacement": "auto",
                        "barAlignment": 0,
                        "drawStyle": "line",
                        "fillOpacity": 10,
                        "gradientMode": "none",
                        "hideFrom": {"tooltip": False, "viz": False, "legend": False},
                        "insertNulls": False,
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "pointSize": 5,
                        "scaleDistribution": {"type": "linear"},
                        "showPoints": "never",
                        "spanNulls": False,
                        "stacking": {"group": "A", "mode": "none"},
                    },
                    "mappings": [],
                    "unit": unit,
                },
            },
            "gridPos": grid_pos,
            "id": panel_id,
            "options": {"legend": legend_config, "tooltip": {"mode": "multi", "sort": "none"}},
            "targets": targets,
            "title": title,
            "type": "timeseries",
        }

    @staticmethod
    def create_table_panel(
        title: str,
        expr: str,
        panel_id: int,
        grid_pos: dict[str, int],
        transformations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Create a standardized table panel.

        Parameters
        ----------
        title : str
            Panel title
        expr : str
            PromQL expression
        panel_id : int
            Unique panel ID
        grid_pos : dict[str, int]
            Panel grid position
        transformations : list[dict[str, Any]], optional
            Data transformations

        Returns
        -------
        dict[str, Any]
            Grafana table panel configuration

        """
        panel = {
            "datasource": {"type": "prometheus", "uid": "${datasource}"},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "thresholds"},
                    "custom": {"align": "auto", "cellOptions": {"type": "auto"}, "inspect": False},
                    "mappings": [],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [{"color": "green", "value": None}, {"color": "red", "value": 80}],
                    },
                },
            },
            "gridPos": grid_pos,
            "id": panel_id,
            "options": {
                "cellHeight": "sm",
                "footer": {"countRows": False, "fields": "", "reducer": ["sum"], "show": False},
                "showHeader": True,
            },
            "pluginVersion": "10.2.3",
            "targets": [
                {
                    "datasource": {"type": "prometheus", "uid": "${datasource}"},
                    "expr": expr,
                    "format": "table",
                    "instant": True,
                    "refId": "A",
                },
            ],
            "title": title,
            "type": "table",
        }

        if transformations:
            panel["transformations"] = transformations

        return panel

    @staticmethod
    def create_heatmap_panel(
        title: str,
        expr: str,
        panel_id: int,
        grid_pos: dict[str, int],
        color_scheme: str = "RdYlGn",
    ) -> dict[str, Any]:
        """
        Create a standardized heatmap panel.

        Parameters
        ----------
        title : str
            Panel title
        expr : str
            PromQL expression
        panel_id : int
            Unique panel ID
        grid_pos : dict[str, int]
            Panel grid position
        color_scheme : str, optional
            Color scheme for the heatmap

        Returns
        -------
        dict[str, Any]
            Grafana heatmap panel configuration

        """
        return {
            "datasource": {"type": "prometheus", "uid": "${datasource}"},
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "continuous-GrYlRd"},
                    "custom": {"hideFrom": {"legend": False, "tooltip": False, "viz": False}},
                    "mappings": [],
                    "max": 1,
                    "min": 0,
                    "unit": "short",
                },
            },
            "gridPos": grid_pos,
            "id": panel_id,
            "options": {
                "calculate": False,
                "calculation": {},
                "cellGap": 2,
                "cellValues": {"decimals": 2},
                "color": {
                    "exponent": 0.5,
                    "fill": "dark-orange",
                    "mode": "scheme",
                    "reverse": False,
                    "scale": "exponential",
                    "scheme": color_scheme,
                    "steps": 128,
                },
                "exemplars": {"color": "rgba(255,0,255,0.7)"},
                "filterValues": {"le": 1e-9},
                "legend": {"show": False},
                "rowsFrame": {"layout": "auto"},
                "tooltip": {"mode": "single", "showColorScale": False, "yHistogram": False},
                "yAxis": {"axisPlacement": "left", "reverse": False, "unit": "short"},
            },
            "pluginVersion": "10.2.3",
            "targets": [
                {
                    "datasource": {"type": "prometheus", "uid": "${datasource}"},
                    "expr": expr,
                    "format": "time_series",
                    "refId": "A",
                },
            ],
            "title": title,
            "type": "heatmap",
        }

    @staticmethod
    def create_row_panel(
        title: str,
        panel_id: int,
        y_pos: int,
        collapsed: bool = False,
    ) -> dict[str, Any]:
        """
        Create a row panel for organizing dashboard sections.

        Parameters
        ----------
        title : str
            Row title
        panel_id : int
            Unique panel ID
        y_pos : int
            Y position of the row
        collapsed : bool, optional
            Whether the row should be collapsed

        Returns
        -------
        dict[str, Any]
            Grafana row panel configuration

        """
        return {
            "collapsed": collapsed,
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y_pos},
            "id": panel_id,
            "panels": [],
            "title": title,
            "type": "row",
        }


class GrafanaDashboardFactory:
    """
    Factory for creating complete Grafana dashboards.
    """

    def __init__(self) -> None:
        """
        Initialize the dashboard factory.
        """
        self.panel_factory = GrafanaPanelFactory()

    def create_base_dashboard(
        self,
        title: str,
        uid: str,
        tags: list[str],
        refresh: str = "30s",
        time_from: str = "now-6h",
        time_to: str = "now",
    ) -> dict[str, Any]:
        """
        Create a base dashboard structure.

        Parameters
        ----------
        title : str
            Dashboard title
        uid : str
            Unique dashboard identifier
        tags : list[str]
            Dashboard tags
        refresh : str, optional
            Auto-refresh interval
        time_from : str, optional
            Default time range start
        time_to : str, optional
            Default time range end

        Returns
        -------
        dict[str, Any]
            Base dashboard configuration

        """
        return {
            "annotations": {
                "list": [
                    {
                        "builtIn": 1,
                        "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                        "enable": True,
                        "hide": True,
                        "iconColor": "rgba(0, 211, 255, 1)",
                        "name": "Annotations & Alerts",
                        "type": "dashboard",
                    },
                ],
            },
            "editable": True,
            "fiscalYearStartMonth": 0,
            "graphTooltip": 1,
            "id": None,
            "links": [
                {
                    "asDropdown": True,
                    "icon": "external link",
                    "includeVars": True,
                    "keepTime": True,
                    "tags": ["ml-monitoring"],
                    "targetBlank": False,
                    "title": "ML Dashboards",
                    "tooltip": "Navigate to other ML dashboards",
                    "type": "dashboards",
                },
            ],
            "panels": [],
            "refresh": refresh,
            "schemaVersion": 39,
            "tags": tags,
            "templating": {"list": self._create_default_variables()},
            "time": {"from": time_from, "to": time_to},
            "timepicker": {},
            "timezone": "",
            "title": title,
            "uid": uid,
            "version": 1,
            "weekStart": "",
        }

    def _create_default_variables(self) -> list[dict[str, Any]]:
        """
        Create default template variables for ML dashboards.

        Returns
        -------
        list[dict[str, Any]]
            Template variables configuration

        """
        return [
            {
                "current": {"selected": False, "text": "Prometheus", "value": "prometheus"},
                "hide": 0,
                "includeAll": False,
                "label": "Data Source",
                "multi": False,
                "name": "datasource",
                "options": [],
                "query": "prometheus",
                "queryValue": "",
                "refresh": 1,
                "regex": "",
                "skipUrlSync": False,
                "type": "datasource",
            },
            {
                "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
                "datasource": {"type": "prometheus", "uid": "${datasource}"},
                "definition": "label_values(ml_predictions_total, model)",
                "hide": 0,
                "includeAll": True,
                "label": "Model",
                "multi": True,
                "name": "model",
                "options": [],
                "query": {
                    "qryType": 1,
                    "query": "label_values(ml_predictions_total, model)",
                    "refId": "PrometheusVariableQueryEditor-VariableQuery",
                },
                "refresh": 2,
                "regex": "",
                "skipUrlSync": False,
                "sort": 1,
                "type": "query",
            },
            {
                "current": {"selected": False, "text": "5m", "value": "5m"},
                "hide": 0,
                "includeAll": False,
                "label": "Interval",
                "multi": False,
                "name": "interval",
                "options": [
                    {"selected": False, "text": "1m", "value": "1m"},
                    {"selected": True, "text": "5m", "value": "5m"},
                    {"selected": False, "text": "15m", "value": "15m"},
                    {"selected": False, "text": "30m", "value": "30m"},
                    {"selected": False, "text": "1h", "value": "1h"},
                ],
                "query": "1m,5m,15m,30m,1h",
                "queryValue": "",
                "skipUrlSync": False,
                "type": "custom",
            },
        ]

    def create_alert_config(
        self,
        alert_name: str,
        condition_value: float,
        condition_type: str = "gt",
        duration: str = "5m",
        frequency: str = "10s",
        severity: str = "warning",
    ) -> dict[str, Any]:
        """
        Create alert configuration for panels.

        Parameters
        ----------
        alert_name : str
            Name of the alert
        condition_value : float
            Threshold value for the alert
        condition_type : str, optional
            Condition type ('gt', 'lt', 'eq', etc.)
        duration : str, optional
            Duration before alert fires
        frequency : str, optional
            Alert evaluation frequency
        severity : str, optional
            Alert severity level

        Returns
        -------
        dict[str, Any]
            Alert configuration

        """
        return {
            "alertRuleTags": {"severity": severity},
            "conditions": [
                {
                    "evaluator": {"params": [condition_value], "type": condition_type},
                    "operator": {"type": "and"},
                    "query": {"params": ["A", "5m", "now"]},
                    "reducer": {"params": [], "type": "last"},
                    "type": "query",
                },
            ],
            "executionErrorState": "alerting",
            "for": duration,
            "frequency": frequency,
            "handler": 1,
            "name": alert_name,
            "noDataState": "no_data",
            "notifications": [],
        }

    def save_dashboard(self, dashboard: dict[str, Any], filepath: str) -> None:
        """
        Save dashboard to JSON file.

        Parameters
        ----------
        dashboard : dict[str, Any]
            Dashboard configuration
        filepath : str
            Output file path

        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dashboard, f, indent=2, ensure_ascii=False)


def main() -> None:
    """
    Demonstrate usage of the dashboard factory.
    """
    factory = GrafanaDashboardFactory()

    # Create a sample dashboard
    dashboard = factory.create_base_dashboard(
        title="Sample ML Dashboard",
        uid="sample-ml-dashboard",
        tags=["ml-monitoring", "sample"],
    )

    # Add a row
    dashboard["panels"].append(factory.panel_factory.create_row_panel("Metrics Overview", 100, 0))

    # Add a stat panel with alert
    alert_config = factory.create_alert_config("High CPU Usage", 0.8, "gt", "5m")
    cpu_panel = factory.panel_factory.create_stat_panel(
        title="CPU Usage",
        expr="avg(ml_cpu_usage_percent)",
        panel_id=1,
        grid_pos={"h": 4, "w": 6, "x": 0, "y": 1},
        unit="percentunit",
        alert_config=alert_config,
    )
    dashboard["panels"].append(cpu_panel)

    # Add a time series panel
    memory_targets = [
        {
            "datasource": {"type": "prometheus", "uid": "${datasource}"},
            "expr": "ml_memory_usage_percent",
            "legendFormat": "Memory Usage",
            "refId": "A",
        },
    ]
    memory_panel = factory.panel_factory.create_timeseries_panel(
        title="Memory Usage Over Time",
        targets=memory_targets,
        panel_id=2,
        grid_pos={"h": 8, "w": 12, "x": 6, "y": 1},
        unit="percentunit",
    )
    dashboard["panels"].append(memory_panel)

    # Save the dashboard
    factory.save_dashboard(dashboard, "/tmp/sample-dashboard.json")
    logger.info("Dashboard saved to /tmp/sample-dashboard.json")


if __name__ == "__main__":
    main()
