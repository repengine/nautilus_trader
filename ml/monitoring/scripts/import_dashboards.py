#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
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
Import Grafana dashboards to running instance.

This script imports ML monitoring dashboards from local JSON files to a running
Grafana instance, with support for overwrite control and validation.

Usage:
    python import_dashboards.py [options]

Example:
    # Import all dashboards from directory
    python import_dashboards.py --url http://localhost:3000 --token <api_token> --input ./dashboards/

    # Import specific dashboard
    python import_dashboards.py --url http://localhost:3000 --token <api_token> --file ml-overview.json

    # Import with overwrite protection
    python import_dashboards.py --url http://localhost:3000 --token <api_token> --input ./dashboards/ --no-overwrite

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from ml.common.logging_config import configure_logging
from ml.monitoring.grafana_client import GrafanaClient


# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


def validate_dashboard_json(dashboard_data: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate dashboard JSON structure.

    Parameters
    ----------
    dashboard_data : dict[str, Any]
        Dashboard configuration data

    Returns
    -------
    tuple[bool, str]
        Tuple of (is_valid, error_message)

    """
    # Check required fields
    required_fields = ["title", "panels"]
    for field in required_fields:
        if field not in dashboard_data:
            return False, f"Missing required field: {field}"

    # Validate UID format if present
    if "uid" in dashboard_data:
        uid = dashboard_data["uid"]
        if not isinstance(uid, str) or len(uid) == 0:
            return False, "Invalid UID: must be non-empty string"

        # Check UID doesn't contain invalid characters
        invalid_chars = [" ", "/", "\\", "?", "#", "[", "]"]
        if any(char in uid for char in invalid_chars):
            return False, f"Invalid UID: contains invalid characters: {uid}"

    # Validate panels structure
    panels = dashboard_data.get("panels", [])
    if not isinstance(panels, list):
        return False, "Panels must be a list"

    # Check for duplicate panel IDs
    panel_ids = []
    for panel in panels:
        if isinstance(panel, dict) and "id" in panel:
            panel_id = panel["id"]
            if panel_id in panel_ids:
                return False, f"Duplicate panel ID found: {panel_id}"
            panel_ids.append(panel_id)

    return True, ""


def import_dashboard_file(
    client: GrafanaClient,
    file_path: Path,
    folder_id: int = 0,
    overwrite: bool = True,
    validate: bool = True,
) -> tuple[bool, str]:
    """
    Import a single dashboard file.

    Parameters
    ----------
    client : GrafanaClient
        Grafana API client
    file_path : Path
        Path to dashboard JSON file
    folder_id : int, optional
        Folder ID to import into
    overwrite : bool, optional
        Whether to overwrite existing dashboards
    validate : bool, optional
        Whether to validate dashboard before import

    Returns
    -------
    tuple[bool, str]
        Tuple of (success, message)

    """
    try:
        logger.info(f"Importing dashboard: {file_path}")

        # Load dashboard JSON
        with open(file_path, encoding="utf-8") as f:
            dashboard_data = json.load(f)

        # Validate dashboard structure
        if validate:
            is_valid, error_msg = validate_dashboard_json(dashboard_data)
            if not is_valid:
                return False, f"Validation failed: {error_msg}"

        # Check if dashboard already exists
        uid = dashboard_data.get("uid")
        if uid and not overwrite:
            existing = client.get_dashboard(uid)
            if existing:
                return False, f"Dashboard already exists (UID: {uid}) and overwrite is disabled"

        # Prepare import payload
        import_data = {
            "dashboard": dashboard_data,
            "folderId": folder_id,
            "overwrite": overwrite,
        }

        # Import dashboard
        result = client.import_dashboard(import_data)

        if result:
            return True, f"Successfully imported: {dashboard_data.get('title', file_path.name)}"
        else:
            return False, "Import failed: unknown error"

    except FileNotFoundError:
        return False, f"File not found: {file_path}"

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"

    except Exception as e:
        return False, f"Import error: {e}"


def import_directory(
    client: GrafanaClient,
    input_dir: Path,
    folder_id: int = 0,
    overwrite: bool = True,
    validate: bool = True,
) -> dict[str, tuple[bool, str]]:
    """
    Import all dashboard files from a directory.

    Parameters
    ----------
    client : GrafanaClient
        Grafana API client
    input_dir : Path
        Input directory containing dashboard JSON files
    folder_id : int, optional
        Folder ID to import into
    overwrite : bool, optional
        Whether to overwrite existing dashboards
    validate : bool, optional
        Whether to validate dashboards before import

    Returns
    -------
    dict[str, tuple[bool, str]]
        Dictionary mapping file names to (success, message) tuples

    """
    results: dict[str, tuple[bool, str]] = {}

    # Find all JSON files
    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found in: {input_dir}")
        return results

    logger.info(f"Found {len(json_files)} dashboard files")

    # Import each file
    for json_file in sorted(json_files):
        success, message = import_dashboard_file(
            client=client,
            file_path=json_file,
            folder_id=folder_id,
            overwrite=overwrite,
            validate=validate,
        )

        results[json_file.name] = (success, message)

        if success:
            logger.info(f" {json_file.name}: {message}")
        else:
            logger.error(f" {json_file.name}: {message}")

    return results


def create_import_manifest(
    results: dict[str, tuple[bool, str]],
    output_dir: Path,
    client: GrafanaClient | None = None,
) -> None:
    """
    Create import manifest file with results.

    Parameters
    ----------
    results : dict[str, tuple[bool, str]]
        Import results mapping filename to (success, message)
    output_dir : Path
        Output directory for manifest
    client : GrafanaClient | None, optional
        Grafana client for timestamp

    """
    manifest = {
        "import_timestamp": client.get_server_time() if client else None,
        "total_files": len(results),
        "successful_imports": sum(1 for success, _ in results.values() if success),
        "failed_imports": sum(1 for success, _ in results.values() if not success),
        "results": {
            filename: {"success": success, "message": message}
            for filename, (success, message) in results.items()
        },
    }

    manifest_file = output_dir / "import_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Import manifest saved: {manifest_file}")


def setup_ml_folder(client: GrafanaClient) -> int:
    """
    Set up ML Monitoring folder in Grafana.

    Parameters
    ----------
    client : GrafanaClient
        Grafana API client

    Returns
    -------
    int
        Folder ID

    """
    folder_title = "ML Monitoring"

    try:
        # Check if folder already exists
        folders = client.get_folders()
        for folder in folders:
            if folder.get("title") == folder_title:
                folder_id = folder.get("id")
                if folder_id is None or not isinstance(folder_id, int):
                    logger.warning(f"Folder {folder_title} missing or invalid ID, using default")
                    return 0
                logger.info(f"Using existing folder: {folder_title} (ID: {folder_id})")
                return int(folder_id)

        # Create new folder
        folder_data = {
            "title": folder_title,
            "uid": "ml-monitoring-folder",
        }

        result = client.create_folder(folder_data)
        if result and isinstance(result, dict):
            folder_id = result.get("id")
            if folder_id is None or not isinstance(folder_id, int):
                logger.warning(
                    f"Created folder {folder_title} missing or invalid ID, using default",
                )
                return 0
            logger.info(f"Created folder: {folder_title} (ID: {folder_id})")
            return int(folder_id)
        else:
            logger.warning(f"Failed to create folder: {folder_title}, using default folder")
            return 0

    except Exception as e:
        logger.warning(f"Error setting up folder: {e}, using default folder")
        return 0


def main() -> int:
    """
    Import Grafana dashboards to running instance.
    """
    parser = argparse.ArgumentParser(
        description="Import Grafana dashboards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--url",
        required=True,
        help="Grafana server URL (e.g., http://localhost:3000)",
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Grafana API token",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input directory containing dashboard JSON files",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Import specific dashboard file",
    )
    parser.add_argument(
        "--folder-id",
        type=int,
        help="Grafana folder ID to import into (use --setup-folder for ML folder)",
    )
    parser.add_argument(
        "--setup-folder",
        action="store_true",
        help="Set up ML Monitoring folder automatically",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing dashboards",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip dashboard validation",
    )
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        help="Verify SSL certificates",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.input and not args.file:
        logger.error("Must specify either --input directory or --file")
        return 1

    if args.input and args.file:
        logger.error("Cannot specify both --input and --file")
        return 1

    # Initialize Grafana client
    try:
        client = GrafanaClient(
            base_url=args.url,
            api_token=args.token,
            verify_ssl=args.verify_ssl,
            timeout=args.timeout,
        )

        # Test connection
        if not client.health_check():
            logger.error("Failed to connect to Grafana server")
            return 1

        logger.info("Connected to Grafana server")

    except Exception as e:
        logger.error(f"Failed to initialize Grafana client: {e}")
        return 1

    # Set up folder
    folder_id = args.folder_id or 0

    if args.setup_folder:
        folder_id = setup_ml_folder(client)

    # Import dashboards
    results: dict[str, tuple[bool, str]] = {}

    try:
        if args.file:
            # Import single file
            success, message = import_dashboard_file(
                client=client,
                file_path=args.file,
                folder_id=folder_id,
                overwrite=not args.no_overwrite,
                validate=not args.no_validate,
            )
            results[args.file.name] = (success, message)

        else:
            # Import directory
            results = import_directory(
                client=client,
                input_dir=args.input,
                folder_id=folder_id,
                overwrite=not args.no_overwrite,
                validate=not args.no_validate,
            )

        # Create import manifest
        output_dir = args.input if args.input else args.file.parent
        create_import_manifest(results, output_dir, client)

        # Print summary
        successful = sum(1 for success, _ in results.values() if success)
        total = len(results)

        logger.info(f"Import completed: {successful}/{total} dashboards successful")

        if successful < total:
            logger.warning(f"{total - successful} dashboards failed to import")
            return 1

        return 0

    except KeyboardInterrupt:
        logger.info("Import cancelled by user")
        return 1

    except Exception as e:
        logger.error(f"Import failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
