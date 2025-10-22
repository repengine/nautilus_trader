"""
Grafana API client for dashboard management.

This module provides a comprehensive client for interacting with Grafana's HTTP API,
specifically optimized for ML monitoring dashboard management operations.

"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Self, cast
from urllib.parse import urljoin
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)


class GrafanaAPIError(Exception):
    """
    Exception raised for Grafana API errors.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_data: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the exception.

        Parameters
        ----------
        message : str
            Error message
        status_code : int, optional
            HTTP status code
        response_data : dict[str, Any], optional
            Response data from Grafana API

        """
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class GrafanaClient:
    """
    Client for interacting with Grafana HTTP API.
    """

    def __init__(
        self,
        base_url: str,
        api_token: str | None = None,
        username: str | None = None,
        password: str | None = None,
        verify_ssl: bool = True,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
    ) -> None:
        """
        Initialize Grafana client.

        Parameters
        ----------
        base_url : str
            Grafana server base URL (e.g., 'http://localhost:3000')
        api_token : str, optional
            API token for authentication (preferred)
        username : str, optional
            Username for basic authentication
        password : str, optional
            Password for basic authentication
        verify_ssl : bool, optional
            Whether to verify SSL certificates
        timeout : int, optional
            Request timeout in seconds
        max_retries : int, optional
            Maximum number of retries for failed requests
        backoff_factor : float, optional
            Backoff factor for retries

        Raises
        ------
        ValueError
            If neither API token nor username/password provided

        """
        # Validate URL
        parsed = urlparse(base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid base URL: {base_url}")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Set up authentication
        if api_token:
            self.auth_headers = {"Authorization": f"Bearer {api_token}"}
        elif username and password:
            self.auth_headers = {}
            self.auth = (username, password)
        else:
            raise ValueError("Must provide either API token or username/password")

        # Initialize session with retry strategy
        self.session = requests.Session()

        # Configure retries
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=backoff_factor,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                **self.auth_headers,
            },
        )

        # Configure SSL verification
        self.session.verify = verify_ssl

        # Set authentication for session if using basic auth
        if hasattr(self, "auth"):
            self.session.auth = self.auth

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Make HTTP request to Grafana API.

        Parameters
        ----------
        method : str
            HTTP method (GET, POST, PUT, DELETE)
        endpoint : str
            API endpoint path
        data : dict[str, Any], optional
            Request payload data
        params : dict[str, Any], optional
            Query parameters

        Returns
        -------
        dict[str, Any] | list[dict[str, Any]] | None
            Response data or None for successful requests without content

        Raises
        ------
        GrafanaAPIError
            If the API request fails

        """
        url = urljoin(self.base_url, endpoint)

        try:
            logger.debug(f"Making {method} request to {url}")

            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=self.timeout,
            )

            logger.debug(f"Response status: {response.status_code}")

            # Handle different response codes
            if response.status_code == 200:
                # Success with content
                if response.content:
                    return cast(dict[str, Any] | list[dict[str, Any]], response.json())
                return None

            elif response.status_code == 201:
                # Created
                return cast(dict[str, Any], response.json()) if response.content else {}

            elif response.status_code == 204:
                # No content (successful deletion, etc.)
                return None

            elif response.status_code == 404:
                # Not found
                return None

            else:
                # Error response
                try:
                    error_data = response.json()
                    error_message = error_data.get("message", f"HTTP {response.status_code}")
                except (json.JSONDecodeError, ValueError):
                    error_message = f"HTTP {response.status_code}: {response.text}"

                raise GrafanaAPIError(
                    message=error_message,
                    status_code=response.status_code,
                    response_data=error_data if "error_data" in locals() else {},
                )

        except requests.RequestException as e:
            raise GrafanaAPIError(f"Request failed: {e}") from e

    def health_check(self) -> bool:
        """
        Check if Grafana server is healthy.

        Returns
        -------
        bool
            True if server is healthy, False otherwise

        """
        try:
            response = self._make_request("GET", "/api/health")
            return response is not None
        except GrafanaAPIError:
            return False

    def get_server_info(self) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Get Grafana server information.

        Returns
        -------
        dict[str, Any] | None
            Server information or None if request fails

        """
        try:
            return self._make_request("GET", "/api/admin/stats")
        except GrafanaAPIError as e:
            logger.warning("Failed to get server info: %s", e, exc_info=True)
            return None

    def get_server_time(self) -> str | None:
        """
        Get server timestamp.

        Returns
        -------
        str | None
            ISO timestamp or None if request fails

        """
        try:
            # Use health endpoint which includes timestamp
            response = self._make_request("GET", "/api/health")
            if response:
                return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return None
        except GrafanaAPIError:
            return None

    def search_dashboards(
        self,
        query: str | None = None,
        tag: str | None = None,
        starred: bool | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for dashboards.

        Parameters
        ----------
        query : str, optional
            Search query string
        tag : str, optional
            Tag to filter by
        starred : bool, optional
            Filter by starred status
        limit : int, optional
            Maximum number of results

        Returns
        -------
        list[dict[str, Any]]
            List of dashboard search results

        """
        params = {}

        if query:
            params["query"] = query
        if tag:
            params["tag"] = tag
        if starred is not None:
            params["starred"] = "true" if starred else "false"
        if limit:
            params["limit"] = str(limit)

        try:
            result = self._make_request("GET", "/api/search", params=params)
            return result if isinstance(result, list) else []
        except GrafanaAPIError as e:
            logger.error("Dashboard search failed: %s", e, exc_info=True)
            return []

    def get_dashboard(self, uid: str) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Get dashboard by UID.

        Parameters
        ----------
        uid : str
            Dashboard UID

        Returns
        -------
        dict[str, Any] | None
            Dashboard data or None if not found

        """
        try:
            return self._make_request("GET", f"/api/dashboards/uid/{uid}")
        except GrafanaAPIError as e:
            if e.status_code == 404:
                logger.debug(f"Dashboard not found: {uid}")
                return None
            logger.error("Failed to get dashboard %s: %s", uid, e, exc_info=True)
            raise

    def create_dashboard(
        self,
        dashboard_data: dict[str, Any],
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Create a new dashboard.

        Parameters
        ----------
        dashboard_data : dict[str, Any]
            Dashboard creation payload

        Returns
        -------
        dict[str, Any] | None
            Created dashboard metadata or None if failed

        """
        try:
            return self._make_request("POST", "/api/dashboards/db", data=dashboard_data)
        except GrafanaAPIError as e:
            logger.error("Failed to create dashboard: %s", e, exc_info=True)
            raise

    def update_dashboard(
        self,
        dashboard_data: dict[str, Any],
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Update an existing dashboard.

        Parameters
        ----------
        dashboard_data : dict[str, Any]
            Dashboard update payload

        Returns
        -------
        dict[str, Any] | None
            Updated dashboard metadata or None if failed

        """
        try:
            return self._make_request("POST", "/api/dashboards/db", data=dashboard_data)
        except GrafanaAPIError as e:
            logger.error("Failed to update dashboard: %s", e, exc_info=True)
            raise

    def delete_dashboard(self, uid: str) -> bool:
        """
        Delete dashboard by UID.

        Parameters
        ----------
        uid : str
            Dashboard UID

        Returns
        -------
        bool
            True if deleted successfully, False otherwise

        """
        try:
            self._make_request("DELETE", f"/api/dashboards/uid/{uid}")
            return True
        except GrafanaAPIError as e:
            if e.status_code == 404:
                logger.debug(f"Dashboard not found for deletion: {uid}")
                return True  # Already doesn't exist
            logger.error("Failed to delete dashboard %s: %s", uid, e, exc_info=True)
            return False

    def import_dashboard(
        self,
        import_data: dict[str, Any],
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Import a dashboard.

        Parameters
        ----------
        import_data : dict[str, Any]
            Import payload with dashboard, folderId, and overwrite settings

        Returns
        -------
        dict[str, Any] | None
            Import result or None if failed

        """
        try:
            return self._make_request("POST", "/api/dashboards/db", data=import_data)
        except GrafanaAPIError as e:
            logger.error("Failed to import dashboard: %s", e, exc_info=True)
            raise

    def get_folders(self) -> list[dict[str, Any]]:
        """
        Get list of folders.

        Returns
        -------
        list[dict[str, Any]]
            List of folder information

        """
        try:
            result = self._make_request("GET", "/api/folders")
            return result if isinstance(result, list) else []
        except GrafanaAPIError as e:
            logger.error("Failed to get folders: %s", e, exc_info=True)
            return []

    def create_folder(
        self,
        folder_data: dict[str, Any],
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Create a new folder.

        Parameters
        ----------
        folder_data : dict[str, Any]
            Folder data (title, uid)

        Returns
        -------
        dict[str, Any] | None
            Created folder information or None if failed

        """
        try:
            return self._make_request("POST", "/api/folders", data=folder_data)
        except GrafanaAPIError as e:
            logger.error("Failed to create folder: %s", e, exc_info=True)
            return None

    def get_folder(self, uid: str) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Get folder by UID.

        Parameters
        ----------
        uid : str
            Folder UID

        Returns
        -------
        dict[str, Any] | None
            Folder information or None if not found

        """
        try:
            return self._make_request("GET", f"/api/folders/{uid}")
        except GrafanaAPIError as e:
            if e.status_code == 404:
                return None
            logger.error("Failed to get folder %s: %s", uid, e, exc_info=True)
            raise

    def get_datasources(self) -> list[dict[str, Any]]:
        """
        Get list of data sources.

        Returns
        -------
        list[dict[str, Any]]
            List of data source information

        """
        try:
            result = self._make_request("GET", "/api/datasources")
            return result if isinstance(result, list) else []
        except GrafanaAPIError as e:
            logger.error("Failed to get data sources: %s", e, exc_info=True)
            return []

    def test_datasource(self, datasource_id: int) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Test a data source connection.

        Parameters
        ----------
        datasource_id : int
            Data source ID

        Returns
        -------
        dict[str, Any] | None
            Test result or None if failed

        """
        try:
            return self._make_request("POST", f"/api/datasources/{datasource_id}/health")
        except GrafanaAPIError as e:
            logger.error("Data source test failed: %s", e, exc_info=True)
            return None

    def get_annotations(
        self,
        from_ts: int | None = None,
        to_ts: int | None = None,
        limit: int = 100,
        alert_id: int | None = None,
        dashboard_id: int | None = None,
        panel_id: int | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get annotations.

        Parameters
        ----------
        from_ts : int, optional
            Start timestamp (milliseconds)
        to_ts : int, optional
            End timestamp (milliseconds)
        limit : int, optional
            Maximum number of annotations to return
        alert_id : int, optional
            Filter by alert ID
        dashboard_id : int, optional
            Filter by dashboard ID
        panel_id : int, optional
            Filter by panel ID
        tags : list[str], optional
            Filter by tags

        Returns
        -------
        list[dict[str, Any]]
            List of annotations

        """
        params = {"limit": str(limit)}

        if from_ts:
            params["from"] = str(from_ts)
        if to_ts:
            params["to"] = str(to_ts)
        if alert_id:
            params["alertId"] = str(alert_id)
        if dashboard_id:
            params["dashboardId"] = str(dashboard_id)
        if panel_id:
            params["panelId"] = str(panel_id)
        if tags:
            params["tags"] = ",".join(tags)

        try:
            result = self._make_request("GET", "/api/annotations", params=params)
            return result if isinstance(result, list) else []
        except GrafanaAPIError as e:
            logger.error("Failed to get annotations: %s", e, exc_info=True)
            return []

    def close(self) -> None:
        """
        Close the HTTP session.
        """
        if self.session:
            self.session.close()

    def __enter__(self) -> Self:
        """
        Context manager entry.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        """
        Context manager exit.
        """
        self.close()


def main() -> None:
    """
    Demonstrate usage of the Grafana client.
    """
    grafana_url = os.environ.get("GRAFANA_URL", "http://localhost:3000")
    api_token = os.environ.get("GRAFANA_API_TOKEN")

    if not api_token:
        logger.warning("GRAFANA_API_TOKEN environment variable is required for Grafana client demo")
        return

    try:
        with GrafanaClient(grafana_url, api_token=api_token) as client:
            # Health check
            if not client.health_check():
                logger.info("Grafana server is not healthy")
                return

            logger.info("Dashboard created successfully")

            # Search for ML dashboards
            ml_dashboards = client.search_dashboards(tag="ml-monitoring")
            logger.info(f"Found {len(ml_dashboards)} ML dashboards")

            for dashboard in ml_dashboards:
                logger.info(f"  - {dashboard.get('title')} (UID: {dashboard.get('uid')})")

            # Get server info
            server_info = client.get_server_info()
            if server_info and isinstance(server_info, dict):
                logger.info(f"Grafana version: {server_info.get('version', 'unknown')}")

    except GrafanaAPIError as e:
        logger.warning("Could not get panels: %s", e, exc_info=True)
    except Exception as e:
        logger.error("Grafana API error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
