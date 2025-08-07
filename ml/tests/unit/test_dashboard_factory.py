# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Tests for Grafana dashboard factory.

Tests the GrafanaPanelFactory and GrafanaDashboardFactory classes with comprehensive
coverage of all methods and functionality.

"""

import json
import os
import tempfile
from unittest.mock import mock_open
from unittest.mock import patch

from ml.monitoring.dashboard_factory import GrafanaDashboardFactory
from ml.monitoring.dashboard_factory import GrafanaPanelFactory


class TestGrafanaPanelFactory:
    """
    Test cases for GrafanaPanelFactory.
    """

    def test_create_stat_panel_basic(self):
        """
        Test basic stat panel creation.
        """
        panel = GrafanaPanelFactory.create_stat_panel(
            title="Test Stat",
            expr="test_metric",
            panel_id=1,
            grid_pos={"h": 4, "w": 6, "x": 0, "y": 1},
        )

        assert panel["title"] == "Test Stat"
        assert panel["id"] == 1
        assert panel["type"] == "stat"
        assert panel["gridPos"] == {"h": 4, "w": 6, "x": 0, "y": 1}
        assert panel["targets"][0]["expr"] == "test_metric"
        assert panel["fieldConfig"]["defaults"]["unit"] == "short"

    def test_create_stat_panel_with_custom_unit(self):
        """
        Test stat panel creation with custom unit.
        """
        panel = GrafanaPanelFactory.create_stat_panel(
            title="CPU Usage",
            expr="cpu_percent",
            panel_id=2,
            grid_pos={"h": 4, "w": 6, "x": 0, "y": 1},
            unit="percentunit",
        )

        assert panel["fieldConfig"]["defaults"]["unit"] == "percentunit"

    def test_create_stat_panel_with_custom_thresholds(self):
        """
        Test stat panel creation with custom thresholds.
        """
        custom_thresholds = [
            {"color": "blue", "value": None},
            {"color": "orange", "value": 50},
            {"color": "red", "value": 90},
        ]

        panel = GrafanaPanelFactory.create_stat_panel(
            title="Test Stat",
            expr="test_metric",
            panel_id=3,
            grid_pos={"h": 4, "w": 6, "x": 0, "y": 1},
            thresholds=custom_thresholds,
        )

        assert panel["fieldConfig"]["defaults"]["thresholds"]["steps"] == custom_thresholds

    def test_create_stat_panel_with_alert_config(self):
        """
        Test stat panel creation with alert configuration.
        """
        alert_config = {
            "alertRuleTags": {"severity": "critical"},
            "name": "Test Alert",
            "for": "2m",
        }

        panel = GrafanaPanelFactory.create_stat_panel(
            title="Test Stat",
            expr="test_metric",
            panel_id=4,
            grid_pos={"h": 4, "w": 6, "x": 0, "y": 1},
            alert_config=alert_config,
        )

        assert panel["alert"] == alert_config

    def test_create_stat_panel_datasource_config(self):
        """
        Test stat panel has correct datasource configuration.
        """
        panel = GrafanaPanelFactory.create_stat_panel(
            title="Test Stat",
            expr="test_metric",
            panel_id=5,
            grid_pos={"h": 4, "w": 6, "x": 0, "y": 1},
        )

        assert panel["datasource"] == {"type": "prometheus", "uid": "${datasource}"}
        assert panel["targets"][0]["datasource"] == {"type": "prometheus", "uid": "${datasource}"}

    def test_create_timeseries_panel_basic(self):
        """
        Test basic time series panel creation.
        """
        targets = [
            {
                "datasource": {"type": "prometheus", "uid": "${datasource}"},
                "expr": "test_metric",
                "legendFormat": "Test Metric",
                "refId": "A",
            },
        ]

        panel = GrafanaPanelFactory.create_timeseries_panel(
            title="Test Timeseries",
            targets=targets,
            panel_id=10,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 1},
        )

        assert panel["title"] == "Test Timeseries"
        assert panel["id"] == 10
        assert panel["type"] == "timeseries"
        assert panel["gridPos"] == {"h": 8, "w": 12, "x": 0, "y": 1}
        assert panel["targets"] == targets
        assert panel["fieldConfig"]["defaults"]["unit"] == "short"

    def test_create_timeseries_panel_with_custom_unit(self):
        """
        Test time series panel creation with custom unit.
        """
        targets = [{"expr": "metric_ms", "refId": "A"}]

        panel = GrafanaPanelFactory.create_timeseries_panel(
            title="Latency",
            targets=targets,
            panel_id=11,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 1},
            unit="ms",
        )

        assert panel["fieldConfig"]["defaults"]["unit"] == "ms"

    def test_create_timeseries_panel_with_custom_legend(self):
        """
        Test time series panel creation with custom legend configuration.
        """
        targets = [{"expr": "test_metric", "refId": "A"}]
        legend_config = {
            "calcs": ["min", "max"],
            "displayMode": "list",
            "placement": "right",
            "showLegend": False,
        }

        panel = GrafanaPanelFactory.create_timeseries_panel(
            title="Test Timeseries",
            targets=targets,
            panel_id=12,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 1},
            legend_config=legend_config,
        )

        assert panel["options"]["legend"] == legend_config

    def test_create_timeseries_panel_default_legend(self):
        """
        Test time series panel creation uses default legend configuration.
        """
        targets = [{"expr": "test_metric", "refId": "A"}]

        panel = GrafanaPanelFactory.create_timeseries_panel(
            title="Test Timeseries",
            targets=targets,
            panel_id=13,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 1},
        )

        expected_legend = {
            "calcs": ["mean", "lastNotNull"],
            "displayMode": "table",
            "placement": "bottom",
            "showLegend": True,
        }
        assert panel["options"]["legend"] == expected_legend

    def test_create_table_panel_basic(self):
        """
        Test basic table panel creation.
        """
        panel = GrafanaPanelFactory.create_table_panel(
            title="Test Table",
            expr="test_metric",
            panel_id=20,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 1},
        )

        assert panel["title"] == "Test Table"
        assert panel["id"] == 20
        assert panel["type"] == "table"
        assert panel["gridPos"] == {"h": 8, "w": 12, "x": 0, "y": 1}
        assert panel["targets"][0]["expr"] == "test_metric"
        assert panel["targets"][0]["format"] == "table"
        assert panel["targets"][0]["instant"] is True

    def test_create_table_panel_with_transformations(self):
        """
        Test table panel creation with transformations.
        """
        transformations = [
            {
                "id": "organize",
                "options": {"excludeByName": {}, "indexByName": {}, "renameByName": {}},
            },
        ]

        panel = GrafanaPanelFactory.create_table_panel(
            title="Test Table",
            expr="test_metric",
            panel_id=21,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 1},
            transformations=transformations,
        )

        assert panel["transformations"] == transformations

    def test_create_table_panel_without_transformations(self):
        """
        Test table panel creation without transformations.
        """
        panel = GrafanaPanelFactory.create_table_panel(
            title="Test Table",
            expr="test_metric",
            panel_id=22,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 1},
        )

        assert "transformations" not in panel

    def test_create_heatmap_panel_basic(self):
        """
        Test basic heatmap panel creation.
        """
        panel = GrafanaPanelFactory.create_heatmap_panel(
            title="Test Heatmap",
            expr="test_metric",
            panel_id=30,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 1},
        )

        assert panel["title"] == "Test Heatmap"
        assert panel["id"] == 30
        assert panel["type"] == "heatmap"
        assert panel["gridPos"] == {"h": 8, "w": 12, "x": 0, "y": 1}
        assert panel["targets"][0]["expr"] == "test_metric"
        assert panel["targets"][0]["format"] == "time_series"

    def test_create_heatmap_panel_with_custom_color_scheme(self):
        """
        Test heatmap panel creation with custom color scheme.
        """
        panel = GrafanaPanelFactory.create_heatmap_panel(
            title="Test Heatmap",
            expr="test_metric",
            panel_id=31,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 1},
            color_scheme="viridis",
        )

        assert panel["options"]["color"]["scheme"] == "viridis"

    def test_create_heatmap_panel_default_color_scheme(self):
        """
        Test heatmap panel creation uses default color scheme.
        """
        panel = GrafanaPanelFactory.create_heatmap_panel(
            title="Test Heatmap",
            expr="test_metric",
            panel_id=32,
            grid_pos={"h": 8, "w": 12, "x": 0, "y": 1},
        )

        assert panel["options"]["color"]["scheme"] == "RdYlGn"

    def test_create_row_panel_basic(self):
        """
        Test basic row panel creation.
        """
        panel = GrafanaPanelFactory.create_row_panel(
            title="Test Row",
            panel_id=40,
            y_pos=5,
        )

        assert panel["title"] == "Test Row"
        assert panel["id"] == 40
        assert panel["type"] == "row"
        assert panel["gridPos"] == {"h": 1, "w": 24, "x": 0, "y": 5}
        assert panel["collapsed"] is False
        assert panel["panels"] == []

    def test_create_row_panel_collapsed(self):
        """
        Test collapsed row panel creation.
        """
        panel = GrafanaPanelFactory.create_row_panel(
            title="Collapsed Row",
            panel_id=41,
            y_pos=10,
            collapsed=True,
        )

        assert panel["collapsed"] is True


class TestGrafanaDashboardFactory:
    """
    Test cases for GrafanaDashboardFactory.
    """

    def test_init(self):
        """
        Test dashboard factory initialization.
        """
        factory = GrafanaDashboardFactory()

        assert isinstance(factory.panel_factory, GrafanaPanelFactory)

    def test_create_base_dashboard_basic(self):
        """
        Test basic dashboard creation.
        """
        factory = GrafanaDashboardFactory()

        dashboard = factory.create_base_dashboard(
            title="Test Dashboard",
            uid="test-dashboard",
            tags=["ml-monitoring", "test"],
        )

        assert dashboard["title"] == "Test Dashboard"
        assert dashboard["uid"] == "test-dashboard"
        assert dashboard["tags"] == ["ml-monitoring", "test"]
        assert dashboard["refresh"] == "30s"
        assert dashboard["time"]["from"] == "now-6h"
        assert dashboard["time"]["to"] == "now"
        assert dashboard["editable"] is True
        assert dashboard["panels"] == []

    def test_create_base_dashboard_with_custom_settings(self):
        """
        Test dashboard creation with custom settings.
        """
        factory = GrafanaDashboardFactory()

        dashboard = factory.create_base_dashboard(
            title="Custom Dashboard",
            uid="custom-dashboard",
            tags=["custom"],
            refresh="10s",
            time_from="now-1h",
            time_to="now+1m",
        )

        assert dashboard["refresh"] == "10s"
        assert dashboard["time"]["from"] == "now-1h"
        assert dashboard["time"]["to"] == "now+1m"

    def test_create_base_dashboard_annotations(self):
        """
        Test dashboard has proper annotations configuration.
        """
        factory = GrafanaDashboardFactory()

        dashboard = factory.create_base_dashboard(
            title="Test Dashboard",
            uid="test-dashboard",
            tags=["test"],
        )

        annotations = dashboard["annotations"]["list"]
        assert len(annotations) == 1
        assert annotations[0]["name"] == "Annotations & Alerts"
        assert annotations[0]["datasource"]["type"] == "grafana"

    def test_create_base_dashboard_links(self):
        """
        Test dashboard has proper navigation links.
        """
        factory = GrafanaDashboardFactory()

        dashboard = factory.create_base_dashboard(
            title="Test Dashboard",
            uid="test-dashboard",
            tags=["test"],
        )

        links = dashboard["links"]
        assert len(links) == 1
        assert links[0]["title"] == "ML Dashboards"
        assert links[0]["tags"] == ["ml-monitoring"]
        assert links[0]["type"] == "dashboards"

    def test_create_default_variables(self):
        """
        Test creation of default template variables.
        """
        factory = GrafanaDashboardFactory()

        variables = factory._create_default_variables()

        assert len(variables) == 3

        # Check datasource variable
        datasource_var = variables[0]
        assert datasource_var["name"] == "datasource"
        assert datasource_var["type"] == "datasource"
        assert datasource_var["query"] == "prometheus"

        # Check model variable
        model_var = variables[1]
        assert model_var["name"] == "model"
        assert model_var["type"] == "query"
        assert model_var["includeAll"] is True
        assert model_var["multi"] is True
        assert "ml_predictions_total" in model_var["definition"]

        # Check interval variable
        interval_var = variables[2]
        assert interval_var["name"] == "interval"
        assert interval_var["type"] == "custom"
        assert len(interval_var["options"]) == 5

    def test_create_alert_config_basic(self):
        """
        Test basic alert configuration creation.
        """
        factory = GrafanaDashboardFactory()

        alert_config = factory.create_alert_config(
            alert_name="Test Alert",
            condition_value=0.8,
        )

        assert alert_config["name"] == "Test Alert"
        assert alert_config["for"] == "5m"
        assert alert_config["frequency"] == "10s"
        assert alert_config["alertRuleTags"]["severity"] == "warning"
        assert alert_config["conditions"][0]["evaluator"]["params"][0] == 0.8
        assert alert_config["conditions"][0]["evaluator"]["type"] == "gt"

    def test_create_alert_config_with_custom_parameters(self):
        """
        Test alert configuration creation with custom parameters.
        """
        factory = GrafanaDashboardFactory()

        alert_config = factory.create_alert_config(
            alert_name="Critical Alert",
            condition_value=0.95,
            condition_type="lt",
            duration="2m",
            frequency="30s",
            severity="critical",
        )

        assert alert_config["name"] == "Critical Alert"
        assert alert_config["for"] == "2m"
        assert alert_config["frequency"] == "30s"
        assert alert_config["alertRuleTags"]["severity"] == "critical"
        assert alert_config["conditions"][0]["evaluator"]["params"][0] == 0.95
        assert alert_config["conditions"][0]["evaluator"]["type"] == "lt"

    def test_create_alert_config_structure(self):
        """
        Test alert configuration has proper structure.
        """
        factory = GrafanaDashboardFactory()

        alert_config = factory.create_alert_config("Test", 1.0)

        assert "conditions" in alert_config
        assert len(alert_config["conditions"]) == 1

        condition = alert_config["conditions"][0]
        assert "evaluator" in condition
        assert "operator" in condition
        assert "query" in condition
        assert "reducer" in condition
        assert "type" in condition

        assert condition["operator"]["type"] == "and"
        assert condition["reducer"]["type"] == "last"
        assert condition["type"] == "query"

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_dashboard(self, mock_json_dump, mock_file):
        """
        Test saving dashboard to file.
        """
        factory = GrafanaDashboardFactory()

        dashboard = {"title": "Test Dashboard", "uid": "test"}
        filepath = "/tmp/test-dashboard.json"

        factory.save_dashboard(dashboard, filepath)

        mock_file.assert_called_once_with(filepath, "w", encoding="utf-8")
        mock_json_dump.assert_called_once_with(
            dashboard,
            mock_file.return_value.__enter__.return_value,
            indent=2,
            ensure_ascii=False,
        )

    def test_dashboard_integration_example(self):
        """
        Test complete dashboard creation with panels.
        """
        factory = GrafanaDashboardFactory()

        # Create base dashboard
        dashboard = factory.create_base_dashboard(
            title="Integration Test Dashboard",
            uid="integration-test",
            tags=["ml-monitoring", "test"],
        )

        # Add a row panel
        row_panel = factory.panel_factory.create_row_panel("Metrics", 100, 0)
        dashboard["panels"].append(row_panel)

        # Add a stat panel with alert
        alert_config = factory.create_alert_config("High CPU", 0.8)
        stat_panel = factory.panel_factory.create_stat_panel(
            title="CPU Usage",
            expr="avg(cpu_percent)",
            panel_id=1,
            grid_pos={"h": 4, "w": 6, "x": 0, "y": 1},
            unit="percentunit",
            alert_config=alert_config,
        )
        dashboard["panels"].append(stat_panel)

        # Add a timeseries panel
        targets = [
            {
                "datasource": {"type": "prometheus", "uid": "${datasource}"},
                "expr": "memory_usage",
                "legendFormat": "Memory",
                "refId": "A",
            },
        ]
        ts_panel = factory.panel_factory.create_timeseries_panel(
            title="Memory Usage",
            targets=targets,
            panel_id=2,
            grid_pos={"h": 8, "w": 12, "x": 6, "y": 1},
        )
        dashboard["panels"].append(ts_panel)

        # Verify dashboard structure
        assert len(dashboard["panels"]) == 3
        assert dashboard["panels"][0]["type"] == "row"
        assert dashboard["panels"][1]["type"] == "stat"
        assert dashboard["panels"][2]["type"] == "timeseries"

        # Verify stat panel has alert
        assert "alert" in dashboard["panels"][1]
        assert dashboard["panels"][1]["alert"]["name"] == "High CPU"

    def test_templating_variables_in_dashboard(self):
        """
        Test that dashboard includes templating variables.
        """
        factory = GrafanaDashboardFactory()

        dashboard = factory.create_base_dashboard(
            title="Test Dashboard",
            uid="test-dashboard",
            tags=["test"],
        )

        templating = dashboard["templating"]["list"]

        # Should have datasource, model, and interval variables
        variable_names = [var["name"] for var in templating]
        assert "datasource" in variable_names
        assert "model" in variable_names
        assert "interval" in variable_names


class TestDashboardFactoryIntegration:
    """
    Integration tests for the dashboard factory.
    """

    def test_real_file_operations(self):
        """
        Test actual file saving and loading.
        """
        factory = GrafanaDashboardFactory()

        dashboard = factory.create_base_dashboard(
            title="File Test Dashboard",
            uid="file-test",
            tags=["test"],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            filepath = tmp_file.name

        try:
            # Save the dashboard
            factory.save_dashboard(dashboard, filepath)

            # Verify file was created and has content
            assert os.path.exists(filepath)

            # Load and verify content
            with open(filepath, encoding="utf-8") as f:
                loaded_dashboard = json.load(f)

            assert loaded_dashboard["title"] == "File Test Dashboard"
            assert loaded_dashboard["uid"] == "file-test"
            assert loaded_dashboard["tags"] == ["test"]

        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_dashboard_json_serialization(self):
        """
        Test that created dashboards are properly JSON serializable.
        """
        factory = GrafanaDashboardFactory()

        # Create a complex dashboard
        dashboard = factory.create_base_dashboard(
            title="JSON Test Dashboard",
            uid="json-test",
            tags=["test", "serialization"],
        )

        # Add various panel types
        dashboard["panels"].extend(
            [
                factory.panel_factory.create_row_panel("Test Row", 100, 0),
                factory.panel_factory.create_stat_panel(
                    "Test Stat",
                    "test_metric",
                    1,
                    {"h": 4, "w": 6, "x": 0, "y": 1},
                ),
                factory.panel_factory.create_timeseries_panel(
                    "Test TS",
                    [{"expr": "test", "refId": "A"}],
                    2,
                    {"h": 8, "w": 12, "x": 6, "y": 1},
                ),
                factory.panel_factory.create_table_panel(
                    "Test Table",
                    "test_table",
                    3,
                    {"h": 8, "w": 12, "x": 0, "y": 9},
                ),
                factory.panel_factory.create_heatmap_panel(
                    "Test Heatmap",
                    "test_heatmap",
                    4,
                    {"h": 8, "w": 12, "x": 12, "y": 9},
                ),
            ],
        )

        # This should not raise an exception
        json_str = json.dumps(dashboard, indent=2)

        # Verify it can be parsed back
        parsed_dashboard = json.loads(json_str)
        assert parsed_dashboard["title"] == "JSON Test Dashboard"
        assert len(parsed_dashboard["panels"]) == 5


def test_main_function():
    """
    Test the main function runs without error in dry-run mode.
    """
    # Test that main function exists and can be imported
    from ml.monitoring.dashboard_factory import main

    assert callable(main)

    # Test with mocked file operations to avoid creating actual files
    with patch(
        "ml.monitoring.dashboard_factory.GrafanaDashboardFactory.save_dashboard",
    ) as mock_save:
        with patch("builtins.print") as mock_print:
            main()

            mock_save.assert_called_once()
            mock_print.assert_called_once()

            # Verify the dashboard was created with expected properties
            call_args = mock_save.call_args
            dashboard = call_args[0][0]  # First argument is the dashboard
            filepath = call_args[0][1]  # Second argument is the filepath

            assert dashboard["title"] == "Sample ML Dashboard"
            assert dashboard["uid"] == "sample-ml-dashboard"
            assert "ml-monitoring" in dashboard["tags"]
            assert filepath == "/tmp/sample-dashboard.json"
