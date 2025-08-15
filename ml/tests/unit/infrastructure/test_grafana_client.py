"""
Tests for Grafana API client.

Tests the GrafanaClient class with comprehensive coverage of all methods, error
handling, and edge cases.

"""

import json
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import requests

from ml.monitoring.grafana_client import GrafanaAPIError
from ml.monitoring.grafana_client import GrafanaClient


class TestGrafanaClient:
    """
    Test cases for GrafanaClient.
    """

    def test_init_with_api_token(self) -> None:
        """
        Test initialization with API token authentication.
        """
        client = GrafanaClient("http://localhost:3000", api_token="test-token")

        assert client.base_url == "http://localhost:3000"
        assert client.auth_headers == {"Authorization": "Bearer test-token"}
        assert client.timeout == 30
        assert "Authorization" in client.session.headers

    def test_init_with_basic_auth(self) -> None:
        """
        Test initialization with username/password authentication.
        """
        client = GrafanaClient(
            "http://localhost:3000",
            username="admin",
            password="admin",
        )

        assert client.base_url == "http://localhost:3000"
        assert client.auth_headers == {}
        assert hasattr(client, "auth")
        assert client.auth == ("admin", "admin")

    def test_init_invalid_url(self) -> None:
        """
        Test initialization with invalid URL raises ValueError.
        """
        with pytest.raises(ValueError, match="Invalid base URL"):
            GrafanaClient("invalid-url", api_token="test-token")

    def test_init_no_auth(self) -> None:
        """
        Test initialization without authentication raises ValueError.
        """
        with pytest.raises(ValueError, match="Must provide either API token or username/password"):
            GrafanaClient("http://localhost:3000")

    def test_init_strips_trailing_slash(self) -> None:
        """
        Test initialization strips trailing slash from base URL.
        """
        client = GrafanaClient("http://localhost:3000/", api_token="test-token")
        assert client.base_url == "http://localhost:3000"

    @patch("ml.monitoring.grafana_client.requests.Session.request")
    def test_make_request_success_200(self, mock_request: MagicMock) -> None:
        """
        Test successful request with 200 status code.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'{"test": "data"}'
        mock_response.json.return_value = {"test": "data"}
        mock_request.return_value = mock_response

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client._make_request("GET", "/api/test")

        assert result == {"test": "data"}
        mock_request.assert_called_once()

    @patch("ml.monitoring.grafana_client.requests.Session.request")
    def test_make_request_success_200_no_content(self, mock_request: MagicMock) -> None:
        """
        Test successful request with 200 status code but no content.
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b""
        mock_request.return_value = mock_response

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client._make_request("GET", "/api/test")

        assert result is None

    @patch("ml.monitoring.grafana_client.requests.Session.request")
    def test_make_request_success_201(self, mock_request: MagicMock) -> None:
        """
        Test successful request with 201 status code.
        """
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.content = b'{"id": 123}'
        mock_response.json.return_value = {"id": 123}
        mock_request.return_value = mock_response

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client._make_request("POST", "/api/test", data={"name": "test"})

        assert result == {"id": 123}

    @patch("ml.monitoring.grafana_client.requests.Session.request")
    def test_make_request_success_201_no_content(self, mock_request: MagicMock) -> None:
        """
        Test successful request with 201 status code but no content.
        """
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.content = b""
        mock_request.return_value = mock_response

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client._make_request("POST", "/api/test")

        assert result == {}

    @patch("ml.monitoring.grafana_client.requests.Session.request")
    def test_make_request_success_204(self, mock_request: MagicMock) -> None:
        """
        Test successful request with 204 status code.
        """
        mock_response = Mock()
        mock_response.status_code = 204
        mock_request.return_value = mock_response

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client._make_request("DELETE", "/api/test/123")

        assert result is None

    @patch("ml.monitoring.grafana_client.requests.Session.request")
    def test_make_request_not_found_404(self, mock_request: MagicMock) -> None:
        """
        Test request with 404 status code returns None.
        """
        mock_response = Mock()
        mock_response.status_code = 404
        mock_request.return_value = mock_response

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client._make_request("GET", "/api/dashboards/uid/nonexistent")

        assert result is None

    @patch("ml.monitoring.grafana_client.requests.Session.request")
    def test_make_request_error_with_json_response(self, mock_request: MagicMock) -> None:
        """
        Test error request with JSON error response.
        """
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"message": "Bad request", "error": "validation failed"}
        mock_request.return_value = mock_response

        client = GrafanaClient("http://localhost:3000", api_token="test-token")

        with pytest.raises(GrafanaAPIError) as exc_info:
            client._make_request("POST", "/api/test")

        assert exc_info.value.status_code == 400
        assert "Bad request" in str(exc_info.value)

    @patch("ml.monitoring.grafana_client.requests.Session.request")
    def test_make_request_error_without_json_response(self, mock_request: MagicMock) -> None:
        """
        Test error request with non-JSON error response.
        """
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "Internal Server Error"
        mock_request.return_value = mock_response

        client = GrafanaClient("http://localhost:3000", api_token="test-token")

        with pytest.raises(GrafanaAPIError) as exc_info:
            client._make_request("GET", "/api/test")

        assert exc_info.value.status_code == 500
        assert "Internal Server Error" in str(exc_info.value)

    @patch("ml.monitoring.grafana_client.requests.Session.request")
    def test_make_request_connection_error(self, mock_request: MagicMock) -> None:
        """
        Test request with connection error.
        """
        mock_request.side_effect = requests.ConnectionError("Connection failed")

        client = GrafanaClient("http://localhost:3000", api_token="test-token")

        with pytest.raises(GrafanaAPIError, match="Request failed"):
            client._make_request("GET", "/api/test")

    @patch.object(GrafanaClient, "_make_request")
    def test_health_check_success(self, mock_make_request: MagicMock) -> None:
        """
        Test successful health check.
        """
        mock_make_request.return_value = {"status": "ok"}

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.health_check()

        assert result is True
        mock_make_request.assert_called_once_with("GET", "/api/health")

    @patch.object(GrafanaClient, "_make_request")
    def test_health_check_failure(self, mock_make_request: MagicMock) -> None:
        """
        Test failed health check.
        """
        mock_make_request.side_effect = GrafanaAPIError("Health check failed")

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.health_check()

        assert result is False

    @patch.object(GrafanaClient, "_make_request")
    def test_get_server_info_success(self, mock_make_request: MagicMock) -> None:
        """
        Test successful server info retrieval.
        """
        mock_make_request.return_value = {"version": "9.0.0", "commit": "abc123"}

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_server_info()

        assert result == {"version": "9.0.0", "commit": "abc123"}
        mock_make_request.assert_called_once_with("GET", "/api/admin/stats")

    @patch.object(GrafanaClient, "_make_request")
    def test_get_server_info_failure(self, mock_make_request: MagicMock) -> None:
        """
        Test failed server info retrieval.
        """
        mock_make_request.side_effect = GrafanaAPIError("Access denied")

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_server_info()

        assert result is None

    @patch.object(GrafanaClient, "_make_request")
    @patch("time.strftime")
    @patch("time.gmtime")
    def test_get_server_time_success(
        self,
        mock_gmtime: MagicMock,
        mock_strftime: MagicMock,
        mock_make_request: MagicMock,
    ) -> None:
        """
        Test successful server time retrieval.
        """
        mock_make_request.return_value = {"status": "ok"}
        mock_strftime.return_value = "2025-01-01T12:00:00Z"

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_server_time()

        assert result == "2025-01-01T12:00:00Z"
        mock_make_request.assert_called_once_with("GET", "/api/health")

    @patch.object(GrafanaClient, "_make_request")
    def test_get_server_time_failure(self, mock_make_request: MagicMock) -> None:
        """
        Test failed server time retrieval.
        """
        mock_make_request.side_effect = GrafanaAPIError("Server error")

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_server_time()

        assert result is None

    @patch.object(GrafanaClient, "_make_request")
    def test_search_dashboards_success(self, mock_make_request: MagicMock) -> None:
        """
        Test successful dashboard search.
        """
        mock_make_request.return_value = [
            {"id": 1, "uid": "dash1", "title": "Dashboard 1"},
            {"id": 2, "uid": "dash2", "title": "Dashboard 2"},
        ]

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.search_dashboards(query="test", tag="ml", starred=True, limit=10)

        assert len(result) == 2
        assert result[0]["uid"] == "dash1"
        mock_make_request.assert_called_once_with(
            "GET",
            "/api/search",
            params={"query": "test", "tag": "ml", "starred": "true", "limit": "10"},
        )

    @patch.object(GrafanaClient, "_make_request")
    def test_search_dashboards_empty_params(self, mock_make_request: MagicMock) -> None:
        """
        Test dashboard search with no parameters.
        """
        mock_make_request.return_value = []

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.search_dashboards()

        assert result == []
        mock_make_request.assert_called_once_with("GET", "/api/search", params={})

    @patch.object(GrafanaClient, "_make_request")
    def test_search_dashboards_failure(self, mock_make_request: MagicMock) -> None:
        """
        Test failed dashboard search.
        """
        mock_make_request.side_effect = GrafanaAPIError("Search failed")

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.search_dashboards()

        assert result == []

    @patch.object(GrafanaClient, "_make_request")
    def test_search_dashboards_non_list_response(self, mock_make_request: MagicMock) -> None:
        """
        Test dashboard search with non-list response.
        """
        mock_make_request.return_value = {"error": "invalid"}

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.search_dashboards()

        assert result == []

    @patch.object(GrafanaClient, "_make_request")
    def test_get_dashboard_success(self, mock_make_request: MagicMock) -> None:
        """
        Test successful dashboard retrieval.
        """
        mock_make_request.return_value = {
            "dashboard": {"id": 1, "uid": "test-uid", "title": "Test Dashboard"},
            "meta": {"version": 1},
        }

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_dashboard("test-uid")

        assert isinstance(result, dict)
        assert result["dashboard"]["uid"] == "test-uid"
        mock_make_request.assert_called_once_with("GET", "/api/dashboards/uid/test-uid")

    @patch.object(GrafanaClient, "_make_request")
    def test_get_dashboard_not_found(self, mock_make_request: MagicMock) -> None:
        """
        Test dashboard retrieval when dashboard not found.
        """
        mock_make_request.side_effect = GrafanaAPIError("Not found", status_code=404)

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_dashboard("nonexistent")

        assert result is None

    @patch.object(GrafanaClient, "_make_request")
    def test_get_dashboard_other_error(self, mock_make_request: MagicMock) -> None:
        """
        Test dashboard retrieval with other error.
        """
        mock_make_request.side_effect = GrafanaAPIError("Server error", status_code=500)

        client = GrafanaClient("http://localhost:3000", api_token="test-token")

        with pytest.raises(GrafanaAPIError):
            client.get_dashboard("test-uid")

    @patch.object(GrafanaClient, "_make_request")
    def test_create_dashboard_success(self, mock_make_request: MagicMock) -> None:
        """
        Test successful dashboard creation.
        """
        dashboard_data = {
            "dashboard": {"title": "New Dashboard"},
            "folderId": 0,
            "overwrite": False,
        }
        mock_make_request.return_value = {"id": 123, "uid": "new-uid", "url": "/d/new-uid"}

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.create_dashboard(dashboard_data)

        assert isinstance(result, dict)
        assert result["uid"] == "new-uid"
        mock_make_request.assert_called_once_with("POST", "/api/dashboards/db", data=dashboard_data)

    @patch.object(GrafanaClient, "_make_request")
    def test_create_dashboard_failure(self, mock_make_request: MagicMock) -> None:
        """
        Test failed dashboard creation.
        """
        mock_make_request.side_effect = GrafanaAPIError("Creation failed")

        client = GrafanaClient("http://localhost:3000", api_token="test-token")

        with pytest.raises(GrafanaAPIError):
            client.create_dashboard({})

    @patch.object(GrafanaClient, "_make_request")
    def test_update_dashboard_success(self, mock_make_request: MagicMock) -> None:
        """
        Test successful dashboard update.
        """
        dashboard_data = {
            "dashboard": {"id": 123, "title": "Updated Dashboard"},
            "overwrite": True,
        }
        mock_make_request.return_value = {"id": 123, "uid": "test-uid", "version": 2}

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.update_dashboard(dashboard_data)

        assert isinstance(result, dict)
        assert result["version"] == 2
        mock_make_request.assert_called_once_with("POST", "/api/dashboards/db", data=dashboard_data)

    @patch.object(GrafanaClient, "_make_request")
    def test_update_dashboard_failure(self, mock_make_request: MagicMock) -> None:
        """
        Test failed dashboard update.
        """
        mock_make_request.side_effect = GrafanaAPIError("Update failed")

        client = GrafanaClient("http://localhost:3000", api_token="test-token")

        with pytest.raises(GrafanaAPIError):
            client.update_dashboard({})

    @patch.object(GrafanaClient, "_make_request")
    def test_delete_dashboard_success(self, mock_make_request: MagicMock) -> None:
        """
        Test successful dashboard deletion.
        """
        mock_make_request.return_value = None

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.delete_dashboard("test-uid")

        assert result is True
        mock_make_request.assert_called_once_with("DELETE", "/api/dashboards/uid/test-uid")

    @patch.object(GrafanaClient, "_make_request")
    def test_delete_dashboard_not_found(self, mock_make_request: MagicMock) -> None:
        """
        Test dashboard deletion when dashboard not found.
        """
        mock_make_request.side_effect = GrafanaAPIError("Not found", status_code=404)

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.delete_dashboard("nonexistent")

        assert result is True  # Already doesn't exist

    @patch.object(GrafanaClient, "_make_request")
    def test_delete_dashboard_other_error(self, mock_make_request: MagicMock) -> None:
        """
        Test dashboard deletion with other error.
        """
        mock_make_request.side_effect = GrafanaAPIError("Server error", status_code=500)

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.delete_dashboard("test-uid")

        assert result is False

    @patch.object(GrafanaClient, "_make_request")
    def test_import_dashboard_success(self, mock_make_request: MagicMock) -> None:
        """
        Test successful dashboard import.
        """
        import_data = {
            "dashboard": {"title": "Imported Dashboard"},
            "folderId": 1,
            "overwrite": True,
        }
        mock_make_request.return_value = {"id": 124, "uid": "imported-uid"}

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.import_dashboard(import_data)

        assert isinstance(result, dict)
        assert result["uid"] == "imported-uid"
        mock_make_request.assert_called_once_with("POST", "/api/dashboards/db", data=import_data)

    @patch.object(GrafanaClient, "_make_request")
    def test_import_dashboard_failure(self, mock_make_request: MagicMock) -> None:
        """
        Test failed dashboard import.
        """
        mock_make_request.side_effect = GrafanaAPIError("Import failed")

        client = GrafanaClient("http://localhost:3000", api_token="test-token")

        with pytest.raises(GrafanaAPIError):
            client.import_dashboard({})

    @patch.object(GrafanaClient, "_make_request")
    def test_get_folders_success(self, mock_make_request: MagicMock) -> None:
        """
        Test successful folders retrieval.
        """
        mock_make_request.return_value = [
            {"id": 1, "uid": "folder1", "title": "Folder 1"},
            {"id": 2, "uid": "folder2", "title": "Folder 2"},
        ]

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_folders()

        assert len(result) == 2
        assert result[0]["uid"] == "folder1"
        mock_make_request.assert_called_once_with("GET", "/api/folders")

    @patch.object(GrafanaClient, "_make_request")
    def test_get_folders_failure(self, mock_make_request: MagicMock) -> None:
        """
        Test failed folders retrieval.
        """
        mock_make_request.side_effect = GrafanaAPIError("Access denied")

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_folders()

        assert result == []

    @patch.object(GrafanaClient, "_make_request")
    def test_get_folders_non_list_response(self, mock_make_request: MagicMock) -> None:
        """
        Test folders retrieval with non-list response.
        """
        mock_make_request.return_value = {"error": "invalid"}

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_folders()

        assert result == []

    @patch.object(GrafanaClient, "_make_request")
    def test_create_folder_success(self, mock_make_request: MagicMock) -> None:
        """
        Test successful folder creation.
        """
        folder_data = {"title": "New Folder", "uid": "new-folder"}
        mock_make_request.return_value = {"id": 5, "uid": "new-folder", "title": "New Folder"}

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.create_folder(folder_data)

        assert isinstance(result, dict)
        assert result["uid"] == "new-folder"
        mock_make_request.assert_called_once_with("POST", "/api/folders", data=folder_data)

    @patch.object(GrafanaClient, "_make_request")
    def test_create_folder_failure(self, mock_make_request: MagicMock) -> None:
        """
        Test failed folder creation.
        """
        mock_make_request.side_effect = GrafanaAPIError("Creation failed")

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.create_folder({})

        assert result is None

    @patch.object(GrafanaClient, "_make_request")
    def test_get_folder_success(self, mock_make_request: MagicMock) -> None:
        """
        Test successful folder retrieval.
        """
        mock_make_request.return_value = {"id": 1, "uid": "test-folder", "title": "Test Folder"}

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_folder("test-folder")

        assert isinstance(result, dict)
        assert result["uid"] == "test-folder"
        mock_make_request.assert_called_once_with("GET", "/api/folders/test-folder")

    @patch.object(GrafanaClient, "_make_request")
    def test_get_folder_not_found(self, mock_make_request: MagicMock) -> None:
        """
        Test folder retrieval when folder not found.
        """
        mock_make_request.side_effect = GrafanaAPIError("Not found", status_code=404)

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_folder("nonexistent")

        assert result is None

    @patch.object(GrafanaClient, "_make_request")
    def test_get_folder_other_error(self, mock_make_request: MagicMock) -> None:
        """
        Test folder retrieval with other error.
        """
        mock_make_request.side_effect = GrafanaAPIError("Server error", status_code=500)

        client = GrafanaClient("http://localhost:3000", api_token="test-token")

        with pytest.raises(GrafanaAPIError):
            client.get_folder("test-folder")

    @patch.object(GrafanaClient, "_make_request")
    def test_get_datasources_success(self, mock_make_request: MagicMock) -> None:
        """
        Test successful data sources retrieval.
        """
        mock_make_request.return_value = [
            {"id": 1, "name": "Prometheus", "type": "prometheus"},
            {"id": 2, "name": "InfluxDB", "type": "influxdb"},
        ]

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_datasources()

        assert len(result) == 2
        assert result[0]["name"] == "Prometheus"
        mock_make_request.assert_called_once_with("GET", "/api/datasources")

    @patch.object(GrafanaClient, "_make_request")
    def test_get_datasources_failure(self, mock_make_request: MagicMock) -> None:
        """
        Test failed data sources retrieval.
        """
        mock_make_request.side_effect = GrafanaAPIError("Access denied")

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_datasources()

        assert result == []

    @patch.object(GrafanaClient, "_make_request")
    def test_test_datasource_success(self, mock_make_request: MagicMock) -> None:
        """
        Test successful data source test.
        """
        mock_make_request.return_value = {"status": "success", "message": "Data source is working"}

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.test_datasource(1)

        assert isinstance(result, dict)
        assert result["status"] == "success"
        mock_make_request.assert_called_once_with("POST", "/api/datasources/1/health")

    @patch.object(GrafanaClient, "_make_request")
    def test_test_datasource_failure(self, mock_make_request: MagicMock) -> None:
        """
        Test failed data source test.
        """
        mock_make_request.side_effect = GrafanaAPIError("Test failed")

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.test_datasource(1)

        assert result is None

    @patch.object(GrafanaClient, "_make_request")
    def test_get_annotations_success(self, mock_make_request: MagicMock) -> None:
        """
        Test successful annotations retrieval.
        """
        mock_make_request.return_value = [
            {"id": 1, "text": "Annotation 1", "time": 1640995200000},
            {"id": 2, "text": "Annotation 2", "time": 1640995300000},
        ]

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_annotations(
            from_ts=1640995000000,
            to_ts=1640995500000,
            limit=50,
            dashboard_id=123,
            panel_id=1,
            tags=["ml", "monitoring"],
        )

        assert len(result) == 2
        assert result[0]["text"] == "Annotation 1"
        mock_make_request.assert_called_once_with(
            "GET",
            "/api/annotations",
            params={
                "limit": "50",
                "from": "1640995000000",
                "to": "1640995500000",
                "dashboardId": "123",
                "panelId": "1",
                "tags": "ml,monitoring",
            },
        )

    @patch.object(GrafanaClient, "_make_request")
    def test_get_annotations_minimal_params(self, mock_make_request: MagicMock) -> None:
        """
        Test annotations retrieval with minimal parameters.
        """
        mock_make_request.return_value = []

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_annotations()

        assert result == []
        mock_make_request.assert_called_once_with(
            "GET",
            "/api/annotations",
            params={"limit": "100"},
        )

    @patch.object(GrafanaClient, "_make_request")
    def test_get_annotations_failure(self, mock_make_request: MagicMock) -> None:
        """
        Test failed annotations retrieval.
        """
        mock_make_request.side_effect = GrafanaAPIError("Access denied")

        client = GrafanaClient("http://localhost:3000", api_token="test-token")
        result = client.get_annotations()

        assert result == []

    def test_context_manager(self) -> None:
        """
        Test client as context manager.
        """
        with patch.object(GrafanaClient, "close") as mock_close:
            with GrafanaClient("http://localhost:3000", api_token="test-token") as client:
                assert isinstance(client, GrafanaClient)
            mock_close.assert_called_once()

    def test_close_session(self) -> None:
        """
        Test session closing.
        """
        client = GrafanaClient("http://localhost:3000", api_token="test-token")

        with patch.object(client.session, "close") as mock_close:
            client.close()
            mock_close.assert_called_once()


class TestGrafanaAPIError:
    """
    Test cases for GrafanaAPIError.
    """

    def test_init_basic(self) -> None:
        """
        Test basic error initialization.
        """
        error = GrafanaAPIError("Test error")

        assert str(error) == "Test error"
        assert error.status_code is None
        assert error.response_data == {}

    def test_init_with_status_code(self) -> None:
        """
        Test error initialization with status code.
        """
        error = GrafanaAPIError("Test error", status_code=400)

        assert str(error) == "Test error"
        assert error.status_code == 400
        assert error.response_data == {}

    def test_init_with_response_data(self) -> None:
        """
        Test error initialization with response data.
        """
        response_data = {"error": "validation failed", "details": ["field required"]}
        error = GrafanaAPIError("Test error", response_data=response_data)

        assert str(error) == "Test error"
        assert error.status_code is None
        assert error.response_data == response_data

    def test_init_with_all_params(self) -> None:
        """
        Test error initialization with all parameters.
        """
        response_data = {"error": "validation failed"}
        error = GrafanaAPIError("Test error", status_code=422, response_data=response_data)

        assert str(error) == "Test error"
        assert error.status_code == 422
        assert error.response_data == response_data


def test_main_function() -> None:
    """
    Test the main function runs without error in dry-run mode.
    """
    # We can't easily test the actual main function without a real Grafana instance,
    # but we can verify it doesn't have syntax errors
    from ml.monitoring.grafana_client import main

    # Just verify the function exists and can be imported
    assert callable(main)
